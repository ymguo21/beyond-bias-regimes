"""Convergence rate of GD and oracle projected GD on the population loss (Section 8.2).

Both methods run at the step size eta of the configuration from a common grid of raw starts
theta_init = (r cos psi, r sin psi, beta r). GD starts at theta_init. Oracle projected GD
first applies the start map

    theta_0 = theta* + min{1, R / ||P_H(theta_init) - theta*||} (P_H(theta_init) - theta*)

and then iterates theta_{t+1} = P_H(theta_t - eta grad F(theta_t)), where
H = {(w, b): b >= b* <w, w*>} and R = sqrt(2 C F(0)). Every step uses the closed-form
population gradient of `gradient.py`. With Gaussian inputs both methods stay in the plane of
w* and w_init, so three coordinates describe every iterate.

The speed of a run is the number of iterations per decade of ||theta_t - theta*||^2 in its
slowest phase (`phase_costs`). Along every run we also record the largest excess of a step
taken from A_{b*} over the contraction factor 1 - eta (2 - eta) Lambda that the
Local-to-Regional Signal Identity and the Gradient Norm Control lemma give.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise

import numpy as np
from scipy.special import ndtr

from .gaussian import reference_moments, signal, target_second
from .gradient import population
from .records import load_config, metadata, write_csv, write_json

METHODS = ("gd", "projected_gd")


def raw_starts(config):
    """Raw starts (angle psi to w* in degrees, bias ratio beta, norm r, coordinates)."""
    starts = []
    for angle in config["start_angle_deg"]:
        psi = np.radians(angle)
        for ratio in config["start_bias_ratio"]:
            for norm in config["start_norm"]:
                theta = np.array([norm * np.cos(psi), norm * np.sin(psi), ratio * norm])
                starts.append((angle, ratio, norm, theta))
    return starts


def halfspace_projection(theta, b_star):
    """Euclidean projection of the rows of theta onto H = {b >= b* w_1}."""
    shift = np.maximum(b_star * theta[:, 0] - theta[:, 2], 0) / (1 + b_star**2)
    projected = theta.copy()
    projected[:, 0] -= shift * b_star
    projected[:, 2] += shift
    return projected


def start_map(theta, b_star, radius):
    """Start map of oracle projected GD: project onto H, then pull into B(theta*, R)."""
    target = np.array([1.0, 0.0, b_star])
    offset = halfspace_projection(theta[None], b_star)[0] - target
    return target + min(1.0, radius / np.linalg.norm(offset)) * offset


def build_runs(config):
    runs = []
    for b_star in config["b_star"]:
        for alpha in config["alpha"]:
            radius = np.sqrt(config["C"] * target_second(b_star, alpha))
            for method in METHODS:
                for angle, ratio, norm, raw in raw_starts(config):
                    start = raw if method == "gd" else start_map(raw, b_star, radius)
                    runs.append(
                        {
                            "b_star": b_star,
                            "alpha": alpha,
                            "method": method,
                            "start_angle_deg": angle,
                            "start_bias_ratio": ratio,
                            "start_norm": norm,
                            "start": start,
                        }
                    )
    return runs


def advance(task):
    """Advance a batch of runs, of any pairs and methods, until each one stops."""
    runs, config = task
    n = len(runs)
    eta, C = config["eta"], config["C"]
    b_star = np.array([r["b_star"] for r in runs])
    alpha = np.array([r["alpha"] for r in runs])
    projected = np.array([r["method"] == "projected_gd" for r in runs])
    theta = np.array([r["start"] for r in runs])
    target = np.stack((np.ones(n), np.zeros(n), b_star), axis=1)
    norm2 = 1 + b_star**2
    zero_loss = target_second(b_star, alpha) / 2
    log_factor = np.log1p(-eta * (2 - eta) * signal(b_star, alpha))
    levels = np.array(config["accuracy"])
    window = config["max_steps"] // 10

    # Phase bookkeeping: the first iterate below each level, and the last iterate above each
    # level (but the last) before the first iterate below the next one, with their distances.
    hit = np.full((n, len(levels)), -1, dtype=np.int64)
    at_hit = np.full((n, len(levels)), np.nan)
    last_above = np.full((n, len(levels) - 1), -1, dtype=np.int64)
    at_last_above = np.full((n, len(levels) - 1), np.nan)
    at_window = np.full(n, np.nan)
    steps = np.zeros(n, dtype=np.int64)
    final = np.full((n, 3), np.nan)
    final_distance2 = np.full(n, np.nan)
    start_in_region = np.zeros(n, bool)
    stayed_in_region = np.ones(n, bool)
    step_excess = np.full(n, -np.inf)
    previous_inside = np.zeros(n, bool)
    previous_log_d = np.zeros(n)

    pop = population(theta, b_star, alpha)
    grad, loss = pop["gradient"], pop["loss"]
    active = np.arange(n)
    for t in range(config["max_steps"] + 1):
        th = theta[active]
        error = th - target[active]
        d = np.sum(error**2, axis=1)
        log_d = np.log(d)
        slack = th[:, 2] - b_star[active] * np.hypot(th[:, 0], th[:, 1])
        inside = (slack >= -config["boundary_rtol"]) & (loss[active] <= C * zero_loss[active])
        if t == 0:
            start_in_region[active] = inside
        stayed_in_region[active] &= inside
        if t > 0:
            j = previous_inside[active]
            idx = active[j]
            step_excess[idx] = np.maximum(
                step_excess[idx], log_d[j] - previous_log_d[idx] - log_factor[idx]
            )
        previous_inside[active] = inside
        previous_log_d[active] = log_d

        relative = d / norm2[active]
        if t == window:
            at_window[active] = relative
        above = (relative[:, None] > levels[:-1]) & (hit[active, 1:] < 0)
        last_above[active] = np.where(above, t, last_above[active])
        at_last_above[active] = np.where(above, relative[:, None], at_last_above[active])
        reached = (relative[:, None] <= levels) & (hit[active] < 0)
        hit[active] = np.where(reached, t, hit[active])
        at_hit[active] = np.where(reached, relative[:, None], at_hit[active])

        done = (relative <= levels[-1]) | (t == config["max_steps"])
        steps[active[done]] = t
        final[active[done]] = th[done]
        final_distance2[active[done]] = relative[done]
        active = active[~done]
        if active.size == 0:
            break
        nxt = theta[active] - eta * grad[active]
        nxt[projected[active]] = halfspace_projection(
            nxt[projected[active]], b_star[active][projected[active]]
        )
        theta[active] = nxt
        pop = population(nxt, b_star[active], alpha[active])
        grad[active], loss[active] = pop["gradient"], pop["loss"]

    rows = []
    for i, r in enumerate(runs):
        w = final[i]
        row = {
            "b_star": r["b_star"],
            "alpha": r["alpha"],
            "method": r["method"],
            "start_angle_deg": r["start_angle_deg"],
            "start_bias_ratio": r["start_bias_ratio"],
            "start_norm": r["start_norm"],
            "start_w_parallel": r["start"][0],
            "start_w_perp": r["start"][1],
            "start_bias": r["start"][2],
            "start_distance2": np.sum((r["start"] - target[i]) ** 2) / norm2[i],
            "start_in_region": bool(start_in_region[i]),
            "stayed_in_region": bool(stayed_in_region[i]),
            "converged": bool(final_distance2[i] <= levels[-1]),
            "steps": int(steps[i]),
            "final_w_parallel": w[0],
            "final_w_perp": w[1],
            "final_bias": w[2],
            "final_distance2": final_distance2[i],
            # Firing probability Phi(b / ||w||) of the final iterate relative to the target's.
            "final_firing_ratio": ndtr(w[2] / np.hypot(w[0], w[1])) / ndtr(r["b_star"]),
            "step_excess": step_excess[i],
        }
        row.update(phase_costs(hit[i], at_hit[i], last_above[i], at_last_above[i], levels))
        if not row["converged"]:
            decades = np.log10(at_window[i] / final_distance2[i])
            window_cost = (config["max_steps"] - window) / decades if decades > 0 else np.inf
            row["phase_window"] = window_cost
        else:
            row["phase_window"] = np.nan
        costs = [v for k, v in row.items() if k.startswith("phase_") and not np.isnan(v)]
        row["per_decade"] = max(costs) if costs else np.nan
        rows.append(row)
    return rows


def phase_costs(hit, at_hit, last_above, at_last_above, levels):
    """Iterations per decade of ||theta_t - theta*||^2 in each completed phase of a run.

    The phase between consecutive levels runs from the last iterate above the upper level to
    the first iterate below the lower one, and its cost is its number of steps divided by the
    decades of distance it covers.
    """
    costs = {}
    for k, (high, low) in enumerate(pairwise(levels)):
        name = f"phase_{high:.0e}_{low:.0e}"
        complete = last_above[k] >= 0 and hit[k + 1] >= 0
        costs[name] = (
            (hit[k + 1] - last_above[k]) / np.log10(at_last_above[k] / at_hit[k + 1])
            if complete
            else np.nan
        )
    return costs


def recheck(task):
    """Largest gradient error over its tolerance and largest loss error, in units of
    max(F(0), F), of the closed form against the adaptive integrator, at the start and final
    iterate of each given run."""
    rows, config = task
    gradient_error = loss_error = 0.0
    for row in rows:
        b_star, alpha = row["b_star"], row["alpha"]
        for point in ("start", "final"):
            theta = np.array([row[f"{point}_{k}"] for k in ("w_parallel", "w_perp", "bias")])
            pop = population(theta, b_star, alpha)
            loss, gradient, _ = reference_moments(theta, b_star, alpha)
            allowed = config["gradient_rtol"] * np.linalg.norm(gradient) + config["gradient_atol"]
            error = np.linalg.norm(pop["gradient"][0] - gradient)
            gradient_error = max(gradient_error, error / allowed)
            unit = max(target_second(b_star, alpha) / 2, loss)
            loss_error = max(loss_error, abs(pop["loss"][0] - loss) / unit)
    return gradient_error, loss_error


def summarize(rows, config):
    """Per pair and method: convergence counts and iterations per decade over the runs with a
    finite cost (median, smallest, largest)."""
    summary = []
    for b_star in config["b_star"]:
        for alpha in config["alpha"]:
            for method in METHODS:
                pair = [
                    r
                    for r in rows
                    if r["b_star"] == b_star and r["alpha"] == alpha and r["method"] == method
                ]
                converged = [r for r in pair if r["converged"]]
                cost = np.array([r["per_decade"] for r in converged])
                summary.append(
                    {
                        "b_star": b_star,
                        "alpha": alpha,
                        "method": method,
                        "runs": len(pair),
                        "converged": len(converged),
                        "certified_per_decade": np.log(10)
                        / -np.log1p(-config["eta"] * (2 - config["eta"]) * signal(b_star, alpha)),
                        "median_per_decade_converged": np.median(cost) if converged else np.nan,
                        "median_per_decade_all": np.median([r["per_decade"] for r in pair]),
                        "max_per_decade_converged": cost.max() if converged else np.nan,
                    }
                )
    return summary


def run(config_path):
    started = time.perf_counter()
    config, output = load_config(config_path)
    runs = build_runs(config)
    workers = os.cpu_count()
    # Interleave runs so that every worker gets a share of the slow pairs.
    chunks = [(runs[i::workers], config) for i in range(workers)]
    print(f"Advancing {len(runs)} runs on {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for part in pool.map(advance, chunks) for row in part]
    key = lambda r: (
        -r["b_star"],
        r["alpha"],
        METHODS.index(r["method"]),
        r["start_angle_deg"],
        r["start_bias_ratio"],
        r["start_norm"],
    )
    rows.sort(key=key)
    write_csv(output / "runs.csv.gz", rows)
    write_csv(output / "summary.csv", summarize(rows, config))

    print("Rechecking the closed-form gradient with the adaptive integrator...", flush=True)
    picks = rows[:: config["recheck_stride"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(recheck, [(picks[i::workers], config) for i in range(workers)]))
    projected = [r for r in rows if r["method"] == "projected_gd"]
    audit = {
        "runs": len(rows),
        "converged": sum(r["converged"] for r in rows),
        "projected_runs_outside_region": sum(not r["stayed_in_region"] for r in projected),
        "max_step_excess": max(r["step_excess"] for r in rows),
        "rechecked_runs": len(picks),
        "max_gradient_error_over_tolerance": max(e[0] for e in errors),
        "max_scaled_loss_error": max(e[1] for e in errors),
    }
    audit["passed"] = bool(
        audit["projected_runs_outside_region"] == 0
        and audit["max_step_excess"] <= config["step_tolerance"]
        and audit["max_gradient_error_over_tolerance"] <= 1
        and audit["max_scaled_loss_error"] <= config["loss_atol"]
    )
    write_json(output / "audit.json", audit)
    write_json(output / "metadata.json", metadata(config, time.perf_counter() - started))
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/convergence_rate.toml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
