"""Necessity of the region conditions (Section 8.1): remove one condition of A_{b*} at a time.

Bias condition removed (region F <= C F(0)): the reverse weakest ray gives
zeta < 1 at arbitrarily small loss.

Loss condition relaxed (region b >= b*||w||): the loss threshold is raised
level by level. On each level set F = c F(0) the smallest zeta - 1 is found
exactly along every ray (the level set meets a ray at the two ends of its
loss interval) and searched over directions. The resulting curve m(c) is
nonnegative for c <= 4/5 by the theorem; where it turns negative shows when
the loss condition becomes necessary.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .gaussian import Geometry, kappa, point_values
from .rays import weakest_ray
from .records import agreement, load_config, metadata, recheck, write_csv, write_json
from .search import beta_grid, minimize, theta_of


def least_gap(co, candidates):
    """Smallest gap zeta - 1 over candidate scales on each ray (NaN where none is valid)."""
    _, gap, distance2 = co.evaluate(np.nan_to_num(candidates))
    gap = np.where(np.isfinite(candidates) & (distance2 > 0), gap, np.inf)
    j = np.argmin(gap, axis=1)
    value = gap[np.arange(len(j)), j]
    return np.where(np.isfinite(value), value, np.nan), candidates[np.arange(len(j)), j]


def on_loss_level(level):
    """The two points of each ray on the level set F = level * F(0)."""

    def objective(co):
        low, high = co.loss_interval(level)
        return least_gap(co, np.column_stack((low, high)))

    return objective


def minimizer_row(b_star, alpha, best):
    theta = theta_of(best["psi"], best["beta"], best["scale"])
    values = point_values(theta, b_star, alpha)
    return {
        "b_star": b_star,
        "alpha": alpha,
        "psi": best["psi"],
        "beta": best["beta"],
        "scale": best["scale"],
        "w_parallel": theta[0],
        "w_perp": theta[1],
        "bias": theta[2],
        "loss_ratio": float(values["loss_ratio"][0]),
        "gap": float(values["gap"][0]),
        "bias_slack": float(values["bias_slack"][0]),
        "scaled": bool(values["scaled"][0]),
    }


def relax_loss(task):
    """Smallest zeta - 1 on each loss level set of the region of the bias condition."""
    b_star, config = task
    psi = np.linspace(0, np.pi, config["psi_points"])
    beta = beta_grid(
        np.arctan(b_star),
        np.pi / 2,
        config["beta_points"],
        config["beta_geometric_points"],
        config["beta_geometric_min"],
    )
    P, B = (x.ravel() for x in np.meshgrid(psi, beta, indexing="ij"))
    base = Geometry.build(P, B, b_star)
    search = (config["refine_levels"], config["refine_points"])
    levels = []
    for alpha in config["alpha"]:
        for level in config["loss_levels"]:
            best = minimize(on_loss_level(level), b_star, alpha, psi, beta, base, *search)
            levels.append({"level": level, **minimizer_row(b_star, alpha, best)})
    return levels


def reverse_rays(config):
    deltas = np.geomspace(config["ray_delta_min"], config["ray_delta_max"], config["ray_points"])
    rows = []
    for b_star in config["b_star"]:
        for alpha in config["alpha"]:
            ray = weakest_ray(b_star, alpha, deltas, sign=-1)
            for i, delta in enumerate(deltas):
                rows.append(
                    {
                        "b_star": b_star,
                        "alpha": alpha,
                        "delta": delta,
                        "loss_ratio": float(ray["loss_ratio"][i]),
                        "gap": float(ray["ratio_gap"][i]),
                        "asymptotic_gap": float(ray["asymptotic_gap"][i]),
                        "bias_slack": float(ray["bias_slack"][i]),
                        "within_loss_threshold": bool(ray["loss_ratio"][i] <= config["C"]),
                    }
                )
    return rows


def fixed_example(alpha):
    """b*=-1, theta=((1+kappa/4)w*, -5/4): one step t=1/4 along the reverse ray.

    The point does not depend on alpha, because the weakest direction does not.
    """
    b_star, t = -1.0, 0.25
    k = kappa(b_star)
    n = np.sqrt(1 + k * k)
    ray = weakest_ray(b_star, alpha, np.array([t * n]), sign=-1)
    return {
        "b_star": b_star,
        "alpha": alpha,
        "t": t,
        # Exact point theta* + delta * v with v = -v_min.
        "delta": t * n,
        "v0": k / n,
        "v1": 0.0,
        "v2": -1 / n,
        "w_parallel": float(ray["w"][0]),
        "w_perp": 0.0,
        "bias": float(ray["b"][0]),
        "bias_slack": float(ray["bias_slack"][0]),
        "loss_ratio": float(ray["loss_ratio"][0]),
        "gap": float(ray["ratio_gap"][0]),
        "scaled": True,
    }


def run(config_path):
    started = time.perf_counter()
    config, output = load_config(config_path)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(relax_loss, [(b, config) for b in config["b_star"]]))
    levels = recheck([row for part in results for row in part], config["precise_digits"])
    write_csv(output / "loss_levels.csv", levels)
    rays = reverse_rays(config)
    write_csv(output / "reverse_rays.csv", rays)
    (example,) = recheck([fixed_example(config["example_alpha"])], config["precise_digits"])
    write_json(output / "counterexample.json", example)
    violations = [r for r in levels if r["precise_gap"] < 0]
    audit = {
        "loss_level_minimizers": len(levels),
        **agreement(
            [*levels, example], config["gap_rtol"], config["gap_atol"], config["loss_rtol"]
        ),
        "max_relative_level_mismatch": max(abs(r["loss_ratio"] / r["level"] - 1) for r in levels),
        "loss_levels_with_violation": len(violations),
        # Consistency with the theorem: every violation must leave the full region.
        "violations_outside_region": all(
            r["bias_slack"] < -config["boundary_rtol"] or r["loss_ratio"] > config["C"]
            for r in violations
        ),
        "reverse_rays_below_one": all(r["gap"] < 0 for r in rays),
        "example_gap": example["precise_gap"],
        "example_loss_ratio": example["precise_loss_ratio"],
    }
    audit["passed"] = bool(
        audit["agreement"]
        and audit["max_relative_level_mismatch"] < config["loss_rtol"]
        and audit["violations_outside_region"]
        and audit["reverse_rays_below_one"]
        and example["precise_gap"] < 0
    )
    write_json(output / "audit.json", audit)
    write_json(output / "metadata.json", metadata(config, time.perf_counter() - started))
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/region_conditions.toml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
