"""Critical loss threshold (Section 8.1).

With the bias constraint kept, define

    C_bar(b*, alpha) = inf { F(theta)/F(0) : b >= b*||w||, zeta(theta) < 1 },

the largest threshold for which the theorem's conclusion holds, so the theorem
gives C_bar >= 4/5. Rounding cannot decide the sign of a much smaller zeta - 1, so a
point counts as a counterexample only when zeta <= 1 - tau, confirmed by 40-digit
integration. The least loss on a ray lies at an end of its violation interval, where
zeta - 1 = -tau exactly; the search therefore uses intervals for tau + boundary_rtol,
so that every counterexample satisfies zeta <= 1 - tau with room for rounding.
Every reported value is attained by an explicit, independently rechecked
counterexample, so it is an upper bound on C_bar; a confirmed value below 4/5 would
contradict the theorem.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .gaussian import Geometry, point_values, target_second
from .records import RECHECK_KEYS, agreement, load_config, metadata, recheck, write_csv, write_json
from .search import beta_grid, minimize, theta_of


def violation_interval(co, tau, scale_min=0.0):
    """Scales s >= scale_min where <theta-theta*,grad F> <= (1-tau) Lambda ||theta-theta*||^2."""
    c2, c1, c0 = co.margin(1 - tau)
    disc = c1 * c1 - 4 * c2 * c0
    root = np.sqrt(np.maximum(disc, 0))
    stable = -0.5 * (c1 + np.copysign(root, c1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r1, r2 = stable / c2, c0 / stable
    low, high = np.maximum(np.minimum(r1, r2), scale_min), np.maximum(r1, r2)
    # The leading coefficient A - (1-tau)Lambda is positive on rays with the bias condition.
    valid = (c2 > 0) & (disc > 0) & (high > low)
    return np.where(valid, low, np.nan), np.where(valid, high, np.nan)


def objective_for(tau, margin, samples, scale_min):
    """Least loss ratio on each ray's violation interval, confirmed by stable evaluation.

    Intervals are those of tau + margin; a point counts only if its stably evaluated gap is
    at most -tau. The zero predictor theta=0 has no gradient; intervals reaching it start at
    the actual point scale_min * u instead.
    """

    def objective(co):
        low, high = violation_interval(co, tau + margin, scale_min)
        M = target_second(co.b_star, co.alpha)
        scale = np.clip(co.B / co.A, low, high)
        loss, gap, _ = co.evaluate(np.nan_to_num(scale)[:, None])
        value = np.where(gap[:, 0] <= -tau, 2 * loss[:, 0] / M, np.nan)
        value[np.isnan(low)] = np.nan
        # Near the target the quadratic coefficients cancel; re-sample unconfirmed intervals.
        retry = np.flatnonzero(np.isfinite(low) & np.isnan(value))
        if retry.size:
            grid = low[retry, None] + (high - low)[retry, None] * np.linspace(0, 1, samples)
            sub_loss, sub_gap, _ = co.take(retry).evaluate(grid)
            sub = np.where(sub_gap <= -tau, 2 * sub_loss / M, np.inf)
            j = np.argmin(sub, axis=1)
            found = np.isfinite(sub[np.arange(retry.size), j])
            value[retry[found]] = sub[found, j[found]]
            scale[retry[found]] = grid[found, j[found]]
        return value, scale

    return objective


def threshold_bias(task):
    b_star, config = task
    beta0 = np.arctan(b_star)
    psi = np.linspace(0, np.pi, config["psi_points"])
    beta = beta_grid(
        beta0,
        np.pi / 2,
        config["beta_linear_points"],
        config["beta_geometric_points"],
        config["beta_geometric_min"],
    )
    P, B = (x.ravel() for x in np.meshgrid(psi, beta, indexing="ij"))
    base = Geometry.build(P, B, b_star)
    tau = config["tau"]
    rows = []
    for alpha in config["alpha"]:
        best = minimize(
            objective_for(
                tau, config["boundary_rtol"], config["interval_samples"], config["scale_min"]
            ),
            b_star,
            alpha,
            psi,
            beta,
            base,
            config["refine_levels"],
            config["refine_points"],
        )
        row = {"b_star": b_star, "alpha": alpha, "tau": tau, "found": best is not None}
        if best is None:
            row.update(critical_threshold=np.inf)
        else:
            theta = theta_of(best["psi"], best["beta"], best["scale"])
            values = point_values(theta, b_star, alpha)
            row.update(
                critical_threshold=float(values["loss_ratio"][0]),
                psi=best["psi"],
                beta_minus_face=best["beta"] - beta0,
                scale=best["scale"],
                w_parallel=theta[0],
                w_perp=theta[1],
                bias=theta[2],
                weight_norm=float(np.hypot(theta[0], theta[1])),
                loss_ratio=float(values["loss_ratio"][0]),
                gap=float(values["gap"][0]),
                bias_slack=float(values["bias_slack"][0]),
                scaled=bool(values["scaled"][0]),
            )
        rows.append(row)
    return rows


def run(config_path):
    started = time.perf_counter()
    config, output = load_config(config_path)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = [
            row
            for part in pool.map(threshold_bias, [(b, config) for b in config["b_star"]])
            for row in part
        ]
    counterexamples = recheck([r for r in rows if r["found"]], config["precise_digits"])
    for row in rows:
        for key in (
            *RECHECK_KEYS,
            "psi",
            "beta_minus_face",
            "scale",
            "w_parallel",
            "w_perp",
            "bias",
            "weight_norm",
            "loss_ratio",
            "gap",
            "bias_slack",
            "scaled",
        ):
            row.setdefault(key, np.nan)
    write_csv(output / "thresholds.csv", rows)
    # The paper's criterion, applied literally to the 40-digit value.
    confirmed = [r["precise_gap"] <= -r["tau"] for r in counterexamples]
    lowest = min(counterexamples, key=lambda r: r["critical_threshold"])
    audit = {
        "pairs": len(rows),
        "counterexamples": len(counterexamples),
        "counterexamples_confirmed": int(sum(confirmed)),
        **agreement(counterexamples, config["gap_rtol"], config["gap_atol"], config["loss_rtol"]),
        "lowest_threshold": lowest["critical_threshold"],
        "lowest_threshold_pair": {k: lowest[k] for k in ("b_star", "alpha", "tau")},
        "all_above_four_fifths": all(r["critical_threshold"] > 0.8 for r in counterexamples),
    }
    audit["passed"] = bool(
        audit["counterexamples_confirmed"] == len(counterexamples) and audit["agreement"]
    )
    write_json(output / "audit.json", audit)
    write_json(output / "metadata.json", metadata(config, time.perf_counter() - started))
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/critical_threshold.toml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
