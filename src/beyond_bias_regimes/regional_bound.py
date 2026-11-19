"""Regional lower bound (Section 8.1): zeta >= 1 on spheres centred at the target.

Every point theta=theta*+delta*v with ||v||=1 lies on exactly one sphere.
On each sphere the direction v is gridded around the weakest Hessian
direction v_min, the minimum over grid points in the region is refined locally,
and every evaluated point in the region counts toward the lower-bound check.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .gaussian import displacement_values, hessian_frame, kappa
from .rays import weakest_ray
from .records import RECHECK_KEYS, agreement, load_config, metadata, recheck, write_csv, write_json
from .search import zoom


def sphere_distances(config):
    """Relative distances: geometric up to the switch, then linear to the outer radius."""
    decades = np.log10(config["distance_switch"] / config["distance_min"])
    inner = np.geomspace(
        config["distance_min"],
        config["distance_switch"],
        round(decades * config["spheres_per_decade"]) + 1,
    )
    outer = np.arange(
        config["distance_switch"] + config["distance_step"],
        config["distance_max"] + config["distance_step"] / 2,
        config["distance_step"],
    )
    return np.r_[inner, outer]


def angle_grid(config):
    chi = np.r_[
        0.0,
        np.geomspace(config["chi_min"], 0.1, config["chi_log_points"]),
        np.linspace(0.1, np.pi, config["chi_linear_points"] + 1)[1:],
    ]
    return chi, np.linspace(0, np.pi, config["omega_points"])


def sphere_values(b_star, alpha, delta, chi, omega, C):
    """Evaluate a (chi, omega) grid on one sphere; omega in [0, pi] keeps w_perp >= 0."""
    weakest, in_plane, perpendicular = hessian_frame(b_star)
    X, W = np.meshgrid(chi, omega, indexing="ij")
    v = (
        np.cos(X)[..., None] * weakest
        + np.sin(X)[..., None]
        * (np.cos(W)[..., None] * in_plane + np.sin(W)[..., None] * perpendicular)
    ).reshape(-1, 3)
    v[:, 1] = np.abs(v[:, 1])
    values = displacement_values(np.full(len(v), delta), v, b_star, alpha)
    values["in_region"] = (values["bias_slack"] >= 0) & (values["loss_ratio"] <= C)
    values.update(delta=delta, v=v, chi=X.ravel(), omega=W.ravel())
    return values


def scan_sphere(b_star, alpha, delta, config):
    """All evaluated grids on one sphere: the base grid, then zooms around its minimum."""
    chi, omega = angle_grid(config)
    parts = [sphere_values(b_star, alpha, delta, chi, omega, config["C"])]
    for _ in range(config["refine_levels"]):
        current = parts[-1]
        ok = np.flatnonzero(current["in_region"])
        if ok.size == 0:
            break
        i = ok[np.argmin(current["gap"][ok])]
        chi = zoom(chi, current["chi"][i], config["refine_points"])
        omega = zoom(omega, current["omega"][i], config["refine_points"])
        parts.append(sphere_values(b_star, alpha, delta, chi, omega, config["C"]))
    return parts


def scan_pair(task):
    b_star, alpha, config = task
    norm = np.sqrt(1 + b_star**2)
    rows, negatives = [], []
    for index, d in enumerate(sphere_distances(config)):
        parts = scan_sphere(b_star, alpha, d * norm, config)
        row = {
            "b_star": b_star,
            "alpha": alpha,
            "C": config["C"],
            "sphere": index,
            "sphere_distance": d,
            "evaluated_points": sum(part["gap"].size for part in parts),
            "in_region_points": sum(int(part["in_region"].sum()) for part in parts),
        }
        gaps = [np.where(part["in_region"], part["gap"], np.inf) for part in parts]
        k = int(np.argmin([g.min() for g in gaps]))
        if np.isfinite(gaps[k].min()):
            row.update(point_row(parts[k], int(np.argmin(gaps[k])), b_star))
        else:
            row.update({key: np.nan for key in POINT_KEYS})
        row.update(ray_values(b_star, alpha, d * norm, config["C"]))
        rows.append(row)
        for part, gap in zip(parts, gaps):
            for i in np.flatnonzero(gap < 0):
                negatives.append(
                    {
                        **{key: row[key] for key in ("b_star", "alpha", "C", "sphere")},
                        **point_row(part, i, b_star),
                    }
                )
    return rows, negatives


POINT_KEYS = (
    "distance",
    "delta",
    "v0",
    "v1",
    "v2",
    "chi",
    "omega",
    "w_parallel",
    "w_perp",
    "bias",
    "loss_ratio",
    "bias_slack",
    "gap",
    "scaled",
)


def point_row(values, i, b_star):
    v = values["v"][i]
    theta = np.array([1.0, 0.0, b_star]) + values["delta"] * v
    return {
        "distance": float(values["distance"][i]),
        "delta": float(values["delta"]),
        "v0": float(v[0]),
        "v1": float(v[1]),
        "v2": float(v[2]),
        "chi": float(values["chi"][i]),
        "omega": float(values["omega"][i]),
        # Rounded coordinates for reading; the exact point is theta* + delta * v.
        "w_parallel": float(theta[0]),
        "w_perp": float(theta[1]),
        "bias": float(theta[2]),
        "loss_ratio": float(values["loss_ratio"][i]),
        "bias_slack": float(values["bias_slack"][i]),
        "gap": float(values["gap"][i]),
        "scaled": bool(values["scaled"][i]),
    }


def ray_values(b_star, alpha, delta, C):
    """The weakest ray at the same displacement, where its weight is positive."""
    k = kappa(b_star)
    if delta >= np.sqrt(1 + k * k) / k:
        return {
            "ray_gap": np.nan,
            "ray_loss_ratio": np.nan,
            "ray_in_region": False,
            "asymptotic_gap": np.nan,
        }
    ray = weakest_ray(b_star, alpha, np.array([delta]))
    return {
        "ray_gap": float(ray["ratio_gap"][0]),
        "ray_loss_ratio": float(ray["loss_ratio"][0]),
        "ray_in_region": bool(ray["loss_ratio"][0] <= C),
        "asymptotic_gap": float(ray["asymptotic_gap"][0]),
    }


def summarize(rows, config):
    summary = []
    for b_star in config["b_star"]:
        for alpha in config["alpha"]:
            pair = [r for r in rows if r["b_star"] == b_star and r["alpha"] == alpha]
            found = [r for r in pair if r["in_region_points"] > 0]
            lowest = min(found, key=lambda r: r["gap"])
            ray = [r for r in found if r["ray_in_region"]]
            summary.append(
                {
                    "b_star": b_star,
                    "alpha": alpha,
                    "C": config["C"],
                    "evaluated_points": sum(r["evaluated_points"] for r in pair),
                    "in_region_points": sum(r["in_region_points"] for r in pair),
                    "nonempty_spheres": len(found),
                    "outer_sphere_empty": pair[-1]["in_region_points"] == 0,
                    "min_gap": lowest["gap"],
                    "min_gap_distance": lowest["distance"],
                    "innermost_distance": found[0]["distance"],
                    "innermost_gap": found[0]["gap"],
                    "innermost_gap_over_asymptotic": found[0]["gap"] / found[0]["asymptotic_gap"],
                    "max_gap_over_ray": max(r["gap"] / r["ray_gap"] for r in ray),
                    "outermost_distance": found[-1]["distance"],
                }
            )
    return summary


def run(config_path):
    started = time.perf_counter()
    config, output = load_config(config_path)
    tasks = [(b, a, config) for b in config["b_star"] for a in config["alpha"]]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(scan_pair, tasks))
    rows = [row for pair, _ in results for row in pair]
    negatives = [row for _, pair in results for row in pair]
    print(f"Scanned {len(rows)} spheres; rechecking sphere minima with two oracles...", flush=True)
    minima = recheck([r for r in rows if r["in_region_points"] > 0], config["precise_digits"])
    for row in rows:
        for key in RECHECK_KEYS:
            row.setdefault(key, np.nan)
    if negatives:
        recheck(negatives, config["precise_digits"])
        write_csv(output / "negative_points.csv", negatives)
    write_csv(output / "spheres.csv", rows)
    summary = summarize(rows, config)
    write_csv(output / "summary.csv", summary)
    audit = {
        "spheres": len(rows),
        "evaluated_points": sum(r["evaluated_points"] for r in rows),
        "in_region_points": sum(r["in_region_points"] for r in rows),
        "raw_negative_gaps": len(negatives),
        "confirmed_negative_gaps": sum(r["precise_gap"] < 0 for r in negatives),
        "min_gap": min(r["gap"] for r in minima),
        "min_precise_gap": min(r["precise_gap"] for r in minima),
        "all_outer_spheres_empty": all(s["outer_sphere_empty"] for s in summary),
        "max_gap_over_ray": max(s["max_gap_over_ray"] for s in summary),
        **agreement(minima, config["gap_rtol"], config["gap_atol"], config["loss_rtol"]),
    }
    audit["passed"] = bool(
        audit["confirmed_negative_gaps"] == 0
        and audit["min_precise_gap"] > 0
        and audit["all_outer_spheres_empty"]
        and audit["agreement"]
    )
    write_json(output / "audit.json", audit)
    write_json(output / "metadata.json", metadata(config, time.perf_counter() - started))
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regional_bound.toml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
