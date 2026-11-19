"""The numbers quoted in Section 8, computed from the saved results, and the bias thresholds of
the ReLU noise budget quoted in Sections 6 and 7, computed from the closed form.

Writes `results/numbers.md`, which lists them in the order of the paper.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .budget import chi
from .figures import certified_cost, config_of
from .gaussian import signal
from .records import read_csv


def slope(x, y):
    """Least-squares slope of log y against log x, with a free intercept."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def audit(results, name):
    return json.loads((results / name / "audit.json").read_text())


def budget_thresholds():
    """Sections 6 and 7: the target bias below which the middle entry chi(b*) eps attains the
    maximum in the ReLU noise budget. Above it, both budgets are proportional to the square of
    the signal, so the budget gain attains its upper bound varrho^2 / (1 - alpha)^2."""

    def middle_over_first(b):
        return np.log(signal(b, 0.0) / chi(b))

    def middle_over_rest(b, eps):
        lam = signal(b, 0.0)
        rest = max(lam * eps, 2 * np.sqrt(1 + b * b) * lam / np.pi * np.sqrt(eps))
        return np.log(rest / (chi(b) * eps))

    lines = [
        "## 6 and 7 Noise Budget of ReLU",
        "",
        (
            "- middle entry above the first entry exactly for b* below "
            f"{brentq(middle_over_first, -6.0, 0.0, xtol=1e-12):.4f}"
        ),
        "- middle entry attaining the maximum exactly for b* below:",
    ]
    for eps in (1.0, 1e-1, 1e-2):
        b = brentq(middle_over_rest, -6.0, 0.0, args=(eps,), xtol=1e-12)
        lines.append(f"  - eps = {eps:g}: {b:.4f}")
    return [*lines, ""]


def regional_identity(results):
    """Section 8.1: regional lower bound, region conditions and critical loss threshold."""
    bound = config_of(results, "regional_bound")
    bound_audit = audit(results, "regional_bound")
    shown = [0.0, 0.1, 0.9]
    summary = [
        r for r in read_csv(results / "regional_bound" / "summary.csv") if r["alpha"] in shown
    ]
    spheres = [
        r for r in read_csv(results / "regional_bound" / "spheres.csv") if r["alpha"] in shown
    ]
    smallest = min(
        (r for r in spheres if r["in_region_points"] > 0), key=lambda r: r["precise_gap"]
    )

    conditions = config_of(results, "region_conditions")
    alpha = conditions["example_alpha"]
    rays = [
        r
        for r in read_csv(results / "region_conditions" / "reverse_rays.csv")
        if r["alpha"] == alpha
    ]
    example = json.loads((results / "region_conditions" / "counterexample.json").read_text())
    levels = [
        r
        for r in read_csv(results / "region_conditions" / "loss_levels.csv")
        if r["alpha"] == alpha
    ]
    first_negative = {
        b: min(r["level"] for r in levels if r["b_star"] == b and r["precise_gap"] < 0)
        for b in conditions["b_star"]
    }
    top = max(conditions["loss_levels"])

    critical = config_of(results, "critical_threshold")
    rows = read_csv(results / "critical_threshold" / "thresholds.csv")
    found = [r for r in rows if r["found"] and r["precise_loss_ratio"] <= 1]
    lowest = min(found, key=lambda r: r["precise_loss_ratio"])
    alphas = critical["alpha"]
    return [
        "## 8.1 Local-to-Regional Signal Identity",
        "",
        "Regional lower bound, for the leak coefficients 0, 0.1 and 0.9 of the figure:",
        (
            f"- relative distances of the spheres: {bound['distance_min']:g} to "
            f"{bound['distance_max']:g}"
        ),
        (
            f"- evaluated parameters: {sum(r['evaluated_points'] for r in summary):,.0f}, "
            f"of which {sum(r['in_region_points'] for r in summary):,.0f} in the region"
        ),
        (
            f"- points with zeta < 1: {bound_audit['confirmed_negative_gaps']} "
            f"(all six leak coefficients: {bound_audit['evaluated_points']:,} parameters)"
        ),
        (
            f"- smallest zeta - 1: {smallest['precise_gap']:.2g}, at relative distance "
            f"{smallest['distance']:.2g} (innermost sphere: {bound['distance_min']:g})"
        ),
        "",
        f"Necessity of the region conditions, alpha = {alpha:g}:",
        (
            f"- reverse ray: {len(rays) // len(conditions['b_star'])} displacements per b* in "
            f"[{conditions['ray_delta_min']:g}, {conditions['ray_delta_max']:g}]; "
            f"zeta < 1 at {sum(r['gap'] < 0 for r in rays)} of {len(rays)}"
        ),
        (
            f"- example on the reverse ray (b* = {example['b_star']:g}): theta = "
            f"({example['w_parallel']:.6f} w*, {example['bias']:g}), "
            f"F / F(0) = {example['precise_loss_ratio']:.4f}, zeta - 1 = {example['precise_gap']:.4f}"
        ),
        (
            f"- loss levels: {len(conditions['loss_levels'])}, from "
            f"{min(conditions['loss_levels']):g} to {top:g}"
        ),
        (
            f"- smallest minimum over the levels c <= 4/5: "
            f"{min(r['precise_gap'] for r in levels if r['level'] <= 0.8):.2g}"
        ),
        "- first level with a negative minimum: "
        + ", ".join(f"b* = {b:g}: {c:g}" for b, c in first_negative.items()),
        (
            f"- minimum at c = {top:g}: "
            f"{min(r['precise_gap'] for r in levels if r['level'] == top):.2f}"
        ),
        "",
        "Critical loss threshold:",
        (
            f"- grid: {len(critical['b_star'])} target biases, {len(alphas)} leak coefficients, "
            f"{sum(1e-5 <= a <= 0.05 for a in alphas)} of them in [1e-5, 0.05]"
        ),
        (
            f"- pairs without a counterexample with loss ratio <= 1: {len(rows) - len(found)} of "
            f"{len(rows)}; every pair with alpha >= 0.05 among them: "
            f"{all(not r['found'] or r['precise_loss_ratio'] > 1 for r in rows if r['alpha'] >= 0.05)}"
        ),
        (
            f"- smallest upper bound: {lowest['precise_loss_ratio']:.4f}, at b* = "
            f"{lowest['b_star']:g}, alpha = {lowest['alpha']:g}"
        ),
        "",
    ]


def convergence_rate(results):
    """Section 8.2: GD and oracle projected GD."""
    config = config_of(results, "convergence_rate")
    runs = read_csv(results / "convergence_rate" / "runs.csv.gz")
    eta = config["eta"]
    starts = (
        len(config["start_angle_deg"]) * len(config["start_bias_ratio"]) * len(config["start_norm"])
    )
    pairs = [(b, a) for b in config["b_star"] for a in config["alpha"]]

    def runs_of(method, b, a):
        return [r for r in runs if (r["method"], r["b_star"], r["alpha"]) == (method, b, a)]

    deviation = {}
    for method in ("projected_gd", "gd"):
        values = []
        for b, a in pairs:
            rows = runs_of(method, b, a)
            if method == "gd":
                rows = [r for r in rows if r["converged"]]
            if rows:
                half = certified_cost(b, a, eta) / 2
                values.append(abs(np.median([r["per_decade"] for r in rows]) / half - 1))
        deviation[method] = max(values)
    costs = [certified_cost(b, a, eta) for b, a in pairs]
    relu_gd = {b: runs_of("gd", b, 0.0) for b in config["b_star"]}
    # A run ends with the neuron almost never active when its final firing probability is
    # below 1% of the target's.
    dead = {b: [r for r in rows if r["final_firing_ratio"] < 1e-2] for b, rows in relu_gd.items()}
    leaky_stuck = sum(not r["converged"] for r in runs if r["method"] == "gd" and r["alpha"] > 0)
    # Squared radius of the ball of the Region-Free Gradient Descent theorem with rho = 1/2,
    # relative to ||theta*||^2, and the median cost of the first GD phase, for ReLU at b* = -2.
    lam = signal(-2.0, 0.0)
    ball = min(0.5, lam / np.pi) ** 2
    first = np.median(
        [r["phase_1e-02_1e-04"] for r in relu_gd[-2.0] if np.isfinite(r["phase_1e-02_1e-04"])]
    )
    rated = [r for r in runs if r["method"] == "projected_gd" or r["converged"]]
    ratios = [r["per_decade"] / certified_cost(r["b_star"], r["alpha"], eta) for r in rated]
    convergence = audit(results, "convergence_rate")
    return [
        "## 8.2 Convergence Rate of Gradient Descent",
        "",
        f"- raw starts per pair: {starts}; pairs: {len(pairs)}",
        f"- certified cost per decade at eta = {eta:g}: {min(costs):.3g} to {max(costs):.3g}",
        (
            "- largest deviation of the median cost from half the certified cost: "
            f"projected GD {100 * deviation['projected_gd']:.2f}%, "
            f"GD (converged runs) {100 * deviation['gd']:.2f}%"
        ),
        f"- GD runs with alpha > 0 that do not converge: {leaky_stuck}",
        "- ReLU GD runs that do not converge, per b*: "
        + ", ".join(
            f"{b:g}: {sum(not r['converged'] for r in rows)}/{len(rows)}"
            for b, rows in relu_gd.items()
        ),
        "- ReLU GD runs that end with a firing probability below 1% of the target's, per b*: "
        + ", ".join(f"{b:g}: {len(rows)}/{len(relu_gd[b])}" for b, rows in dead.items()),
        (
            "- largest final firing probability of these runs relative to the target's: "
            f"{max(r['final_firing_ratio'] for rows in dead.values() for r in rows):.2g}"
        ),
        (
            f"- ReLU, b* = -2: relative squared radius of the ball with rho = 1/2: {ball:.2g}; "
            "median cost of the GD phase from 1e-2 to 1e-4 over the certified cost: "
            f"{first / certified_cost(-2.0, 0.0, eta):.2f}"
        ),
        (
            "- largest cost over the certified cost (projected GD runs and converged GD runs): "
            f"{max(ratios):.2f}"
        ),
        (
            f"- audits: projected GD runs outside the region: "
            f"{convergence['projected_runs_outside_region']}; largest excess of a step from the "
            f"region over the certified contraction, in log: {convergence['max_step_excess']:.2g}"
        ),
        "",
    ]


def noise_tolerance(results):
    """Section 8.3: cheapest traps against the noise budget."""
    config = config_of(results, "noise_tolerance")
    rows = read_csv(results / "noise_tolerance" / "traps.csv")
    lines = [
        "## 8.3 Agnostic Noise Tolerance",
        "",
        f"- pairs: {len(rows)}; eps = {config['eps_rel']:g} F_alpha(0), C = {config['C']:g}",
        (
            f"- pairs with the cheapest trap above the budget: "
            f"{sum(r['opt'] > r['budget'] for r in rows)}"
        ),
        "- U / B by leak coefficient (smallest, largest):",
    ]
    for alpha in config["alpha"]:
        pair = [r for r in rows if r["alpha"] == alpha]
        low = min(pair, key=lambda r: r["opt_over_budget"])
        high = max(pair, key=lambda r: r["opt_over_budget"])
        lines.append(
            f"  - alpha = {alpha:g}: {low['opt_over_budget']:.3g} at b* = {low['b_star']:g}, "
            f"{high['opt_over_budget']:.3g} at b* = {high['b_star']:g}"
        )
    relu = [r for r in rows if r["alpha"] == 0 and r["b_star"] <= -3]
    lines += [
        f"- largest U / F_0(0) for ReLU at b* <= -3: {max(r['opt_over_F0'] for r in relu):.2g}",
        (
            f"- smallest excess loss over theta* under the trap labels, in units of F(0): "
            f"{audit(results, 'noise_tolerance')['min_excess_loss_over_F0']:.2g}"
        ),
        "",
    ]
    return lines


def matched_bias(results):
    """Section 8.4: gains from leakage at matched bias."""
    convergence = config_of(results, "convergence_rate")
    summary = {
        (r["b_star"], r["alpha"]): r
        for r in read_csv(results / "convergence_rate" / "summary.csv")
        if r["method"] == "projected_gd"
    }
    lines = ["## 8.4 Gains from Leakage at Matched Bias", "", "Iterations (projected GD):"]
    gains = []
    for alpha in [a for a in convergence["alpha"] if a > 0]:
        gain = np.array([signal(b, alpha) / signal(b, 0.0) for b in convergence["b_star"]])
        ratio = np.array(
            [
                summary[(b, 0.0)]["median_per_decade_all"]
                / summary[(b, alpha)]["median_per_decade_all"]
                for b in convergence["b_star"]
            ]
        )
        gains += list(gain)
        excess = ratio / gain
        lines.append(
            f"- alpha = {alpha:g}: slope {slope(gain, ratio):.4f}; ratio over varrho "
            f"{excess.min():.3f} to {excess.max():.3f}"
        )
    lines += [f"- largest varrho: {max(gains):.3g}", "", "Cheapest traps at a common accuracy:"]

    matched = config_of(results, "matched_bias")
    rows = {(r["b_star"], r["alpha"]): r for r in read_csv(results / "matched_bias" / "traps.csv")}
    b_stars = matched["b_star"]
    excess, varrho = [], []
    for alpha in [a for a in matched["alpha"] if a > 0]:
        gain = np.array([rows[(b, alpha)]["Lambda"] / rows[(b, 0.0)]["Lambda"] for b in b_stars])
        trap = np.array([rows[(b, alpha)]["opt"] / rows[(b, 0.0)]["opt"] for b in b_stars])
        scaled = (1 - alpha) ** 2 * trap
        varrho += list(gain)
        excess += list(scaled / gain**2 - 1)
        # b* decreases along the grid, so the trap gain increases strictly when its successive
        # ratios exceed one.
        increasing = all(t1 > t0 for t0, t1 in pairwise(trap))
        keep = gain >= 1.05
        beta = np.log(scaled[keep]) / np.log(gain[keep])
        below = [b for b, x in zip(np.array(b_stars)[keep], beta) if x < 1]
        k = int(np.argmin(beta))
        lines.append(
            f"- alpha = {alpha:g}: trap gain strictly increasing as b* decreases: {increasing}; "
            f"smallest beta {beta[k]:.2f} at b* = {np.array(b_stars)[keep][k]:g}; "
            + (f"beta < 1 for b* from {max(below):g} to {min(below):g}" if below else "beta >= 1")
        )
    example = -3.0
    lines += [
        f"- varrho over the grid: {min(varrho):.3g} to {max(varrho):.3g}",
        f"- largest excess of the scaled trap gain over varrho^2: {100 * max(excess):.3f}%",
        (
            f"- U / B at b* = {example:g}: ReLU {rows[(example, 0.0)]['opt_over_budget']:.2e}, "
            f"alpha = 0.1 {rows[(example, 0.1)]['opt_over_budget']:.3g}"
        ),
        "",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    lines = [
        "# Numbers quoted in Sections 6 to 8",
        "",
        "Generated by `uv run report` from the saved results and the closed-form noise budget.",
        "",
        *budget_thresholds(),
        *regional_identity(args.results),
        *convergence_rate(args.results),
        *noise_tolerance(args.results),
        *matched_bias(args.results),
    ]
    (args.results / "numbers.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
