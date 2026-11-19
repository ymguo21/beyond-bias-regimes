"""Agnostic noise tolerance (Section 8.3) and gains from leakage at matched bias (Section 8.4):
cheapest traps, the least label noise that can stop oracle projected GD.

For each (b*, alpha), accuracy eps = eps_rel F(0) (F(0) of the pair itself, or of ReLU at the
same b* with eps_reference = "relu") and C, we search the traps theta^ in
H cap B(theta*, R), R = sqrt(2 C F(0)), with ||theta^ - theta*||^2 >= eps and
E[f_theta^^2] <= E[f_theta*^2] (so that L(theta^) >= L(theta*) under any stationary labels),
for the least OPT of labels that make theta^ a fixed point of oracle projected GD while theta*
stays stationary (`labels.cheapest_labels`). Adding independent zero-mean noise to these labels
raises OPT and changes no gradient, so every OPT at or above the minimum found has trapping
labels: the minimum U is an upper bound on the least trapping noise, and the Closed-Form Noise
Budget theorem implies U >= B_alpha(b*; eps) for every trap, which the audit checks.

The search covers three families of traps (`FAMILIES`): a coarse grid in (log r, chi, omega)
(`labels.displacement`) seeds Nelder-Mead on log OPT in each family, and sweeps over b*
restart every pair from its neighbours' cheapest traps. The tie surface is solved for
explicitly because the cheapest traps often lie on it, in a valley of OPT far narrower than
any grid. Each cheapest trap is then recomputed by the independent integrator
`labels.reference`, its gradient compared with `gradient.population`, and the optimality of
theta* under its labels audited over the plane (`labels.excess_loss`).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import brentq, minimize

from .budget import budget
from .gaussian import signal, target_second
from .gradient import population
from .labels import (
    cheapest_labels,
    displacement,
    excess_loss,
    hat_dual,
    outward_normal,
    reference,
    solve_labels,
    target,
)
from .records import load_config, metadata, write_csv, write_json

# The admissible traps form a stratified set: the interior of H cap B(theta*, R) minus the
# eps-ball, the face of H, and the tie surface E[f^^2] = E[f*^2] where the optimality
# constraint is active. Each stratum is searched as its own family; the radius bounds are
# handled by clipping. The tie family reaches the face as chi -> 0, so the curve where the
# face meets the tie surface needs no family of its own.
FAMILIES = ("interior", "face", "tie")


class Pair:
    """Constants of one (b*, alpha, eps_rel).

    The accuracy is eps = eps_rel F(0), with F(0) the zero-predictor loss of this pair
    (eps_reference "own") or of ReLU at the same b* (eps_reference "relu"); the latter gives
    every alpha the same eps, as the matched-bias comparison of the noise budgets requires.
    """

    def __init__(self, b_star, alpha, eps_rel, C, tie_tolerance, eps_reference):
        self.b_star, self.alpha, self.eps_rel = b_star, alpha, eps_rel
        self.eps_reference = eps_reference
        self.lam = signal(b_star, alpha)
        self.F0 = 0.5 * target_second(b_star, alpha)
        self.R = np.sqrt(2 * C * self.F0)
        scale = {"own": self.F0, "relu": 0.5 * target_second(b_star, 0.0)}[eps_reference]
        self.eps = eps_rel * scale
        self.B, self.active = budget(b_star, alpha, self.eps)
        self.log_r = (0.5 * np.log(self.eps), np.log(self.R))
        self.tolerance = tie_tolerance * self.F0

    def evaluate(self, log_r, chi, omega, face):
        """OPT of the cheapest labels (inf where the trap would beat theta* by more than the
        rounding tolerance) and the trap gap L(theta^) - L(theta*)."""
        out = cheapest_labels(
            displacement(self.b_star, log_r, chi, omega).reshape(-1, 3),
            self.b_star,
            self.alpha,
            np.broadcast_to(face, np.shape(log_r)).reshape(-1),
        )
        feasible = out["trap_gap"] >= -self.tolerance
        return np.where(feasible, out["opt"], np.inf), out["trap_gap"]

    def tie_points(self, log_r, omega, values, gaps=None):
        """Points of the tie surface at radius r, along chi at this omega.

        Each sign change of the trap gap between consecutive `values` of chi (gaps computed
        unless given) is solved by Brent's method; returns [(opt, (log r, chi, omega))].
        """

        def at(x):
            return (log_r, x, omega)

        if gaps is None:
            gaps = self.evaluate(*at(values), False)[1]
        found = []
        for a, b, ga, gb in zip(values[:-1], values[1:], gaps[:-1], gaps[1:]):
            if ga * gb < 0:
                root = brentq(
                    lambda x: self.evaluate(*at(x), False)[1][0],
                    a,
                    b,
                    xtol=1e-15,
                    rtol=4 * np.finfo(float).eps,
                )
                found.append((self.evaluate(*at(root), False)[0][0], at(root)))
        return found


def place(pair, family, p, grid):
    """(opt, (log r, chi, omega)) of the optimizer coordinates p of a family."""
    log_r = np.clip(p[0], *pair.log_r)
    if family in ("interior", "face"):
        point = (
            (log_r, 0.0, np.clip(p[1], 0, np.pi))
            if family == "face"
            else (log_r, np.clip(p[1], 1e-12, np.pi / 2), np.clip(p[2], 0, np.pi))
        )
        return pair.evaluate(*point, family == "face")[0][0], point
    omega = np.clip(p[1], 0, np.pi)
    found = pair.tie_points(log_r, omega, np.linspace(0, np.pi / 2, grid))
    return min(found, key=lambda f: f[0]) if found else (np.inf, (log_r, 0.0, omega))


def coordinates(family, point):
    log_r, chi, omega = point
    return {
        "interior": [log_r, chi, omega],
        "face": [log_r, omega],
        "tie": [log_r, omega],
    }[family]


def refine(pair, family, point, config):
    """Nelder-Mead on log OPT within a family from a point; returns (opt, point, family)."""

    def objective(p):
        value = place(pair, family, p, config["tie_grid"])[0]
        return np.log(value) if np.isfinite(value) and value > 0 else np.inf

    x0 = coordinates(family, point)
    if family == "interior":
        x0[1] = max(x0[1], 1e-3)
    if not np.isfinite(objective(x0)):
        return np.inf, point, family
    # Infeasible vertices have value inf, which the simplex tests subtract.
    with np.errstate(invalid="ignore"):
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "xatol": config["xatol"],
                "fatol": config["fatol"],
                "maxiter": config["maxiter"],
            },
        )
    opt, found = place(pair, family, result.x, config["tie_grid"])
    return float(opt), found, family


def grid_search(pair, config):
    """Cheapest refined trap from the coarse grid, refining the best seeds of every family."""
    radii = np.linspace(*pair.log_r, config["grid_radii"])
    chis = np.linspace(0, np.pi / 2, config["grid_chi"])
    omegas = np.linspace(0, np.pi, config["grid_omega"])
    L, X, O = np.meshgrid(radii, chis, omegas, indexing="ij")
    opt, gap = pair.evaluate(L, X, O, X == 0)
    opt, gap = opt.reshape(L.shape), gap.reshape(L.shape)
    seeds = {
        "interior": [
            (opt[i, j, k], (radii[i], chis[j], omegas[k]))
            for i, j, k in zip(*np.nonzero(np.isfinite(opt)))
            if j > 0
        ],
        "face": [
            (opt[i, 0, k], (radii[i], 0.0, omegas[k]))
            for i, k in zip(*np.nonzero(np.isfinite(opt[:, 0, :])))
        ],
        "tie": [],
    }
    # Tie seeds: the grid brackets of the tie surface along chi.
    for i, k in zip(*np.nonzero(np.any(gap[:, :-1, :] * gap[:, 1:, :] < 0, axis=1))):
        seeds["tie"] += pair.tie_points(radii[i], omegas[k], chis, gap[i, :, k])
    found = []
    for family, candidates in seeds.items():
        candidates = sorted((c for c in candidates if np.isfinite(c[0])), key=lambda c: c[0])
        for _, point in candidates[: config["seeds"]]:
            found.append(refine(pair, family, point, config))
    return min(found, key=lambda f: f[0])


def search_alpha(task):
    """Cheapest traps for all b* at one alpha, with the neighbour sweeps."""
    alpha, config = task
    pairs = [
        Pair(
            b,
            alpha,
            config["eps_rel"],
            config["C"],
            config["search_tie_tolerance"],
            config["eps_reference"],
        )
        for b in config["b_star"]
    ]
    best = [grid_search(pair, config) for pair in pairs]
    # Sweep at which each cheapest trap last changed (-1: the grid); a neighbour seeds a pair,
    # in the neighbour's family, only if its trap changed since the previous sweep.
    changed = [-1] * len(pairs)
    order = list(range(len(pairs)))
    for sweep in range(config["sweeps"]):
        for i in order if sweep % 2 == 0 else order[::-1]:
            low, high = (
                max(0, i - config["neighbours"]),
                min(len(pairs), i + config["neighbours"] + 1),
            )
            for j in range(low, high):
                if j == i or changed[j] < sweep - 1:
                    continue
                _, (log_r, chi, omega), family = best[j]
                # Neighbours' traps are rescaled to this pair's radius range.
                shifted = log_r - pairs[j].log_r[1] + pairs[i].log_r[1]
                candidate = refine(pairs[i], family, (shifted, chi, omega), config)
                if candidate[0] < best[i][0] * (1 - config["boundary_rtol"]):
                    best[i], changed[i] = candidate, sweep
    return [trap_row(pair, *found, config) for pair, found in zip(pairs, best)]


def trap_row(pair, opt, point, family, config):
    """Record and audit one cheapest trap."""
    b_star, alpha = pair.b_star, pair.alpha
    face = family == "face"
    error = displacement(b_star, *point).reshape(3)
    out = cheapest_labels(error[None], b_star, alpha, face)
    lam1, lam2, mu = out["lam1"][0], out["lam2"][0], out["mu"][0]
    theta = target(b_star) + error
    # Independent integrator: OPT, and both stationarity conditions in its arithmetic.
    ref = reference(error, b_star, alpha)
    normal_hat = hat_dual(outward_normal(b_star), b_star)
    ref_opt = solve_labels(
        ref["G11"][None],
        ref["G12"][None],
        ref["G22"][None],
        ref["gradient_hat"][None],
        normal_hat[None],
        face,
    )[0][0]
    star_side = ref["G11"] @ lam1 + ref["G12"] @ lam2
    trap_side = ref["G12"] @ lam1 + ref["G22"] @ lam2
    trap_rhs = ref["gradient_hat"] + mu * normal_hat
    # Both relative to |grad^ F + mu n^|, the size of the label correlations the trap needs.
    residual_star = np.linalg.norm(star_side) / np.linalg.norm(trap_rhs)
    residual_trap = np.linalg.norm(trap_side - trap_rhs) / np.linalg.norm(trap_rhs)
    # grad F against the closed form of gradient.py, in the basis X~.
    closed = population(theta, b_star, alpha, error=error)["gradient"][0]
    ours = out["gradient_hat"][0].copy()
    ours[0] -= b_star * ours[2]
    gradient_difference = np.linalg.norm(ours - closed) / np.linalg.norm(closed)
    return {
        "b_star": b_star,
        "alpha": alpha,
        "eps_rel": pair.eps_rel,
        # eps = eps_rel F(0), with F(0) of this pair ("own") or of ReLU at the same b* ("relu").
        "eps_reference": pair.eps_reference,
        "Lambda": pair.lam,
        "F0": pair.F0,
        "R": pair.R,
        "eps": pair.eps,
        "budget": pair.B,
        "active_entry": pair.active,
        "opt": float(out["opt"][0]),
        "opt_over_budget": float(out["opt"][0] / pair.B),
        "opt_over_F0": float(out["opt"][0] / pair.F0),
        "budget_over_F0": pair.B / pair.F0,
        "w_parallel": theta[0],
        "w_perp": theta[1],
        "bias": theta[2],
        "e0": error[0],
        "e1": error[1],
        "e2": error[2],
        "distance2_over_eps": float(error @ error / pair.eps),
        "distance_over_R": float(np.linalg.norm(error) / pair.R),
        "family": family,
        "face": bool(face),
        "mu": float(mu),
        **{f"lam1_{i}": float(lam1[i]) for i in range(3)},
        **{f"lam2_{i}": float(lam2[i]) for i in range(3)},
        "condition": float(out["cond"][0]),
        "trap_gap_over_F0": float(out["trap_gap"][0] / pair.F0),
        "reference_trap_gap_over_F0": float(ref["trap_gap"] / pair.F0),
        "reference_opt_difference": float(abs(out["opt"][0] / ref_opt - 1)),
        "residual_star": float(residual_star),
        "residual_trap": float(residual_trap),
        "gradient_difference": float(gradient_difference),
        "min_excess_loss_over_F0": optimality(pair, error, lam1, lam2, config) / pair.F0,
    }


def optimality(pair, error, lam1, lam2, config):
    """Smallest L(theta) - L(theta*) found over the plane (w_1, w_2, b)."""
    b_star, alpha = pair.b_star, pair.alpha
    n = config["optimality_directions"]
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    angle = np.pi * (1 + 5**0.5) * k
    directions = np.stack(
        (np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z), 1
    )
    radii = np.geomspace(
        config["optimality_inner"] * pair.R,
        config["optimality_outer"] * np.sqrt(1 + b_star**2),
        config["optimality_shells"],
    )
    points = (target(b_star) + radii[:, None, None] * directions[None]).reshape(-1, 3)
    trap = target(b_star) + error
    values = excess_loss(points, b_star, alpha, error, lam1, lam2)[0]

    def objective(x):
        return excess_loss(x[None], b_star, alpha, error, lam1, lam2)[0][0]

    starts = [points[i] for i in np.argsort(values)[: config["optimality_seeds"]]] + [trap]
    found = [
        minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-300, "maxiter": 600},
        ).fun
        for x0 in starts
    ]
    return float(min(values.min(), *found, objective(trap)))


def run(config_path):
    started = time.perf_counter()
    config, output = load_config(config_path)
    tasks = [(a, config) for a in config["alpha"]]
    print(
        f"Searching {len(config['b_star']) * len(tasks)} pairs in {len(tasks)} sweeps over b*...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count())) as pool:
        parts = list(pool.map(search_alpha, tasks))
    rows = [row for part in parts for row in part]
    write_csv(output / "traps.csv", rows)
    audit = {
        "pairs": len(rows),
        "min_opt_over_budget": min(r["opt_over_budget"] for r in rows),
        "min_trap_gap_over_F0": min(r["trap_gap_over_F0"] for r in rows),
        "min_reference_trap_gap_over_F0": min(r["reference_trap_gap_over_F0"] for r in rows),
        "max_reference_trap_gap_difference": max(
            abs(r["trap_gap_over_F0"] - r["reference_trap_gap_over_F0"]) for r in rows
        ),
        "max_reference_opt_difference": max(r["reference_opt_difference"] for r in rows),
        "max_gradient_difference": max(r["gradient_difference"] for r in rows),
        "max_residual_star": max(r["residual_star"] for r in rows),
        "max_residual_trap": max(r["residual_trap"] for r in rows),
        "min_excess_loss_over_F0": min(r["min_excess_loss_over_F0"] for r in rows),
        **{f"{family}_traps": sum(r["family"] == family for r in rows) for family in FAMILIES},
        # Traps clipped to a radius bound lie on its sphere up to rounding.
        "traps_on_eps_sphere": sum(
            abs(r["distance2_over_eps"] - 1) <= config["boundary_rtol"] for r in rows
        ),
        "traps_on_R_sphere": sum(
            abs(r["distance_over_R"] - 1) <= config["boundary_rtol"] for r in rows
        ),
    }
    audit["passed"] = bool(
        audit["min_opt_over_budget"] >= 1
        and audit["min_trap_gap_over_F0"] >= -config["tie_tolerance"]
        and audit["min_reference_trap_gap_over_F0"] >= -config["tie_tolerance"]
        and audit["max_reference_opt_difference"] <= config["reference_rtol"]
        and audit["max_gradient_difference"] <= config["gradient_rtol"]
        and audit["max_residual_star"] <= config["residual_tolerance"]
        and audit["max_residual_trap"] <= config["residual_tolerance"]
        and audit["min_excess_loss_over_F0"] >= -config["optimality_tolerance"]
    )
    write_json(output / "audit.json", audit)
    write_json(output / "metadata.json", metadata(config, time.perf_counter() - started))
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/noise_tolerance.toml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
