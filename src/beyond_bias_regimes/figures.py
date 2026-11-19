"""Figures of Section 8, drawn from the saved results.

The six figures of the paper are written to `figures/`. Where a paper figure shows only some
leak coefficients, the full grid is written to `figures/supplementary/`. Presentation
settings are in `configs/figures.toml`.
"""

import argparse
import json
import tomllib
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullLocator

from .gaussian import signal
from .records import read_csv

WIDTH = 6.0  # inches, the text width of the JMLR style
DASHED = (0, (2.5, 1.5))


def style():
    """Computer Modern at 9 pt, typeset by LaTeX: the paper's typeface and JMLR's minimum size."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def config_of(results, name):
    return json.loads((results / name / "metadata.json").read_text())["config"]


def save(fig, path):
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def margins(fig, left, right, bottom, top, **spacing):
    """subplots_adjust with margins in inches."""
    width, height = fig.get_size_inches()
    fig.subplots_adjust(
        left=left / width,
        right=1 - right / width,
        bottom=bottom / height,
        top=1 - top / height,
        **spacing,
    )


def top_legend(fig, handles, top=0.995, ncol=None, compact=False):
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, top),
        ncol=ncol or len(handles),
        frameon=False,
        handlelength=1.5 if compact else 1.8,
        handletextpad=0.5,
        columnspacing=1.0 if compact else 1.4,
    )


def b_star_handles(figures, b_stars, markers=False):
    """One entry per target bias, naming b* on the first entry only."""
    return [
        Line2D(
            [],
            [],
            color=figures["b_star_colors"][f"{b:g}"],
            marker=figures["b_star_markers"][f"{b:g}"] if markers else None,
            markersize=3.5,
            label=rf"${b:g}$" if i else rf"$b^{{\star}}={b:g}$",
        )
        for i, b in enumerate(b_stars)
    ]


def panels(alphas, height, ncols=3):
    """One panel per leak coefficient, sharing both axes."""
    nrows = -(-len(alphas) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(WIDTH, height), sharex=True, sharey=True, squeeze=False
    )
    for i, (ax, alpha) in enumerate(zip(axes.ravel(), alphas)):
        ax.set_title(rf"({chr(97 + i)}) $\alpha={alpha:g}$", loc="left")
    return fig, axes


def display_steps(distance, gap, edges):
    """Display intervals with edges snapped to sampled distances, and the smallest value in
    each closed interval. Each curve starts and stops at its own first and last sample."""
    nearest = distance[np.argmin(np.abs(np.log(distance)[:, None] - np.log(edges)), axis=0)]
    inner = nearest[(nearest > distance[0]) & (nearest < distance[-1])]
    snapped = np.unique(np.r_[distance[0], inner, distance[-1]])
    heights = [gap[(distance >= lo) & (distance <= hi)].min() for lo, hi in pairwise(snapped)]
    return snapped, np.array(heights)


def staircase(ax, edges, heights, color, linestyle="solid"):
    """Heavy treads carry the interval minima; light risers show their order."""
    ax.stairs(heights, edges, baseline=None, color=color, linewidth=0.5, alpha=0.45)
    ax.hlines(
        heights,
        edges[:-1],
        edges[1:],
        color=color,
        linewidth=1.6,
        linestyle=linestyle,
        capstyle="butt",
    )


def violation_band(ax, figures):
    """Shade zeta < 1 below the zero line."""
    ax.axhspan(ax.get_ylim()[0], 0, color=figures["violation_color"], zorder=0, linewidth=0)
    ax.axhline(0, color="0.25", linewidth=0.7, zorder=1)


def symlog_gap_axis(ax, values):
    """Symmetric logarithmic axis for zeta - 1, with ticks on both sides of zero."""
    magnitude = np.abs(values[values != 0])
    low = int(np.floor(np.log10(magnitude.min())))
    high = int(np.ceil(np.log10(magnitude.max())))
    ax.set_yscale("symlog", linthresh=10.0**low, linscale=0.6)
    ax.set_ylim(-(10.0**high), 10.0**high)
    exponents = list(range(high, low, -max(1, int(np.ceil((high - low) / 3)))))
    ticks = [-(10.0**e) for e in exponents] + [0] + [10.0**e for e in reversed(exponents)]
    labels = [rf"$-10^{{{e}}}$" for e in exponents] + ["0"]
    labels += [rf"$10^{{{e}}}$" for e in reversed(exponents)]
    ax.set_yticks(ticks, labels)
    ax.yaxis.set_minor_locator(NullLocator())


DISTANCE_LABEL = r"Relative distance $\|\theta-\theta^{\star}\|/\|\theta^{\star}\|$"
LEVEL_LABEL = r"Loss level $c=F_\alpha(\theta)/F_\alpha(0)$"
BIAS_LABEL = r"Target bias $b^{\star}$"


# Section 8.1 ---------------------------------------------------------------------------------


def regional_bound_figure(results, figures, alphas, path, height):
    """Smallest observed zeta - 1 over the points of A_{b*} in each display interval of the
    relative distance to the target."""
    config = config_of(results, "regional_bound")
    rows = [
        r for r in read_csv(results / "regional_bound" / "spheres.csv") if r["in_region_points"] > 0
    ]
    fig, axes = panels(alphas, height)
    for ax, alpha in zip(axes.ravel(), alphas):
        for b_star in config["b_star"]:
            pair = [r for r in rows if r["b_star"] == b_star and r["alpha"] == alpha]
            distance = np.array([r["distance"] for r in pair])
            gap = np.array([r["gap"] for r in pair])
            color = figures["b_star_colors"][f"{b_star:g}"]
            staircase(ax, *display_steps(distance, gap, figures["distance_edges"]), color)
        ax.set(xscale="log", yscale="log")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
    gaps = np.array([r["gap"] for r in rows if r["alpha"] in alphas])
    low = int(np.floor(np.log10(gaps.min())))
    ax.set_ylim(10.0**low, 10.0 ** int(np.ceil(np.log10(gaps.max()))))
    ax.set_yticks([10.0**e for e in range(0, low - 1, -4)])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks([1e-10, 1e-7, 1e-4, 1e-1])
    fig.supxlabel(DISTANCE_LABEL, y=0.01)
    fig.supylabel(r"Smallest observed $\zeta-1$", x=0.01)
    top_legend(fig, b_star_handles(figures, config["b_star"]))
    margins(fig, 0.6, 0.08, 0.42, 0.45, wspace=0.08, hspace=0.34)
    save(fig, path)


def draw_bias_removed(ax, alpha, results, figures, b_stars):
    """Sphere minima in A_{b*} (solid) and the reverse weakest ray, which violates only the
    bias condition (dashed), as staircases of interval minima. Returns the plotted gaps."""
    spheres = [
        r for r in read_csv(results / "regional_bound" / "spheres.csv") if r["in_region_points"] > 0
    ]
    rays = [
        r
        for r in read_csv(results / "region_conditions" / "reverse_rays.csv")
        if r["within_loss_threshold"]
    ]
    values = []
    for b_star in b_stars:
        color = figures["b_star_colors"][f"{b_star:g}"]
        kept = sorted(
            (r for r in spheres if r["b_star"] == b_star and r["alpha"] == alpha),
            key=lambda r: r["distance"],
        )
        removed = sorted(
            (r for r in rays if r["b_star"] == b_star and r["alpha"] == alpha),
            key=lambda r: r["delta"],
        )
        norm = np.sqrt(1 + b_star**2)
        for points, x, linestyle in ((kept, "distance", "solid"), (removed, "delta", DASHED)):
            distance = np.array([r[x] for r in points]) / (norm if x == "delta" else 1)
            gap = np.array([r["gap"] for r in points])
            staircase(
                ax, *display_steps(distance, gap, figures["distance_edges"]), color, linestyle
            )
            values += list(gap)
    ax.set_xscale("log")
    return values


def draw_loss_relaxed(ax, alpha, results, figures, b_stars, C):
    """Smallest zeta - 1 on each loss level set of the bias condition. Returns the gaps."""
    levels = read_csv(results / "region_conditions" / "loss_levels.csv")
    values = []
    for b_star in b_stars:
        pair = [r for r in levels if r["b_star"] == b_star and r["alpha"] == alpha]
        ax.plot(
            [r["level"] for r in pair],
            [r["precise_gap"] for r in pair],
            "o-",
            color=figures["b_star_colors"][f"{b_star:g}"],
            markersize=1.8,
            linewidth=1.0,
        )
        values += [r["precise_gap"] for r in pair]
    ax.axvline(C, color="0.35", linestyle="--", linewidth=0.7, zorder=1)
    return values


def loss_level_axis(ax, labels):
    ax.set_xscale("logit")
    ax.set_xlim(0.04, 0.99995)
    ax.set_xticks([float(x) for x in labels], labels)
    ax.xaxis.set_minor_locator(NullLocator())


def region_keys(ax):
    """Keys for the solid (in A_{b*}) and dashed (bias violated) curves, on either side of
    the zero line at half height of the symmetric axis."""
    keys = {
        "frameon": False,
        "handlelength": 1.5,
        "handletextpad": 0.4,
        "borderaxespad": 0.15,
        "borderpad": 0.2,
    }
    inside = Line2D([], [], color="0.3", label=r"In $\mathcal{A}_{b^{\star}}$")
    violated = Line2D([], [], color="0.3", linestyle=DASHED, label="Bias violated")
    ax.add_artist(ax.legend(handles=[inside], loc="lower right", bbox_to_anchor=(1, 0.5), **keys))
    ax.legend(handles=[violated], loc="upper right", bbox_to_anchor=(1, 0.5), **keys)


def region_conditions_figure(results, figures, path):
    """Both region-condition ablations at one leak coefficient."""
    config = config_of(results, "region_conditions")
    alpha, b_stars = figures["paper_region_conditions_alpha"], config["b_star"]
    fig, (left, right) = plt.subplots(1, 2, figsize=figures["region_conditions_size"], sharey=True)
    values = draw_bias_removed(left, alpha, results, figures, b_stars)
    values += draw_loss_relaxed(right, alpha, results, figures, b_stars, config["C"])
    symlog_gap_axis(left, np.array(values))
    for ax in (left, right):
        violation_band(ax, figures)
    left.set_xticks([1e-9, 1e-5, 1e-1])
    loss_level_axis(right, ("0.1", "0.8", "0.9999"))
    left.set_title("(a) No bias condition", loc="left")
    right.set_title("(b) No loss condition", loc="left")
    region_keys(left)
    left.set_xlabel(DISTANCE_LABEL)
    right.set_xlabel(LEVEL_LABEL)
    left.set_ylabel(r"$\zeta-1$")
    top_legend(fig, b_star_handles(figures, b_stars))
    margins(fig, 0.72, 0.15, 0.45, 0.5, wspace=0.14)
    save(fig, path)


def region_conditions_bias_all(results, figures, path):
    config = config_of(results, "region_conditions")
    fig, axes = panels(config["alpha"], 3.2)
    values = []
    for ax, alpha in zip(axes.ravel(), config["alpha"]):
        values += draw_bias_removed(ax, alpha, results, figures, config["b_star"])
    symlog_gap_axis(ax, np.array(values))
    for ax in axes.ravel():
        violation_band(ax, figures)
    region_keys(axes[0, 0])
    ax.set_xticks([1e-10, 1e-7, 1e-4, 1e-1])
    fig.supxlabel(DISTANCE_LABEL, y=0.01)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\zeta-1$")
    top_legend(fig, b_star_handles(figures, config["b_star"]))
    margins(fig, 0.68, 0.08, 0.42, 0.45, wspace=0.08, hspace=0.34)
    save(fig, path)


def region_conditions_loss_all(results, figures, path):
    config = config_of(results, "region_conditions")
    fig, axes = panels(config["alpha"], 3.0)
    values = []
    for ax, alpha in zip(axes.ravel(), config["alpha"]):
        values += draw_loss_relaxed(ax, alpha, results, figures, config["b_star"], config["C"])
    symlog_gap_axis(ax, np.array(values))
    for ax in axes.ravel():
        violation_band(ax, figures)
    loss_level_axis(ax, ("0.1", "0.8", "0.99", "0.9999"))
    fig.supxlabel(LEVEL_LABEL, y=0.01)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\zeta-1$")
    top_legend(fig, b_star_handles(figures, config["b_star"]))
    margins(fig, 0.68, 0.14, 0.42, 0.45, wspace=0.2, hspace=0.34)
    save(fig, path)


def critical_threshold_figure(results, figures, path):
    """Least loss ratio of a confirmed counterexample, an upper bound on the critical
    threshold; counterexamples with loss above F(0) are not drawn."""
    config = config_of(results, "critical_threshold")
    rows = read_csv(results / "critical_threshold" / "thresholds.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.axhspan(0, 4 / 5, color="0.93", zorder=0)
    ax.axhline(4 / 5, color="0.25", linestyle="--", linewidth=0.8)
    ax.text(1.5e-5, 0.79, r"Theorem: $C\leq4/5$", va="top", color="0.25")
    for b_star in config["b_star"]:
        pair = [
            r for r in rows if r["b_star"] == b_star and r["found"] and r["precise_loss_ratio"] <= 1
        ]
        ax.plot(
            [r["alpha"] for r in pair],
            [r["precise_loss_ratio"] for r in pair],
            color=figures["b_star_colors"][f"{b_star:g}"],
            marker=figures["b_star_markers"][f"{b_star:g}"],
            markersize=3.5,
        )
    ax.set_xscale("symlog", linthresh=1e-5, linscale=0.3)
    ax.set_xticks(
        [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
        ["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"],
    )
    ax.set_xlim(-2e-6, 1)
    ax.set_ylim(0.75, 1.02)
    ax.set(xlabel=r"Leak coefficient $\alpha$", ylabel=r"Upper bound on $\bar C_\alpha(b^\star)$")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    top_legend(fig, b_star_handles(figures, config["b_star"], markers=True))
    margins(fig, 0.6, 0.09, 0.42, 0.3)
    save(fig, path)


# Section 8.2 ---------------------------------------------------------------------------------


def certified_cost(b_star, alpha, eta):
    """Iterations per decade of ||theta_t - theta*||^2 certified by the geometric bound
    (1 - eta (2 - eta) Lambda)^t of the Regional Gradient Descent theorem."""
    return np.log(10) / -np.log1p(-eta * (2 - eta) * signal(b_star, alpha))


def convergence_rate_figure(results, figures, path):
    """Median measured cost against the certified cost, one marker per method and pair.

    Projected GD: median over all runs, open where most runs have not converged after
    max_steps, so that the median is measured over their final window. GD: median over the
    converged runs, and no marker where none converges.
    """
    config = config_of(results, "convergence_rate")
    runs = read_csv(results / "convergence_rate" / "runs.csv.gz")
    pairs = [(b, a) for a in config["alpha"] for b in config["b_star"]]
    certified = np.array([certified_cost(b, a, config["eta"]) for b, a in pairs])
    fig, ax = plt.subplots(figsize=figures["convergence_rate_size"])
    grid = np.geomspace(certified.min() / 2, certified.max() * 2, 10)
    ax.plot(grid, grid, color="k", linewidth=0.9, zorder=1)
    ax.plot(grid, grid / 2, color="k", linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
    marker_size = {"projected_gd": 4.6, "gd": 2.2}
    for method, z in (("projected_gd", 2), ("gd", 3)):
        color, marker = figures["method_colors"][method], figures["method_markers"][method]
        for (b, a), cert in zip(pairs, certified):
            rows = [r for r in runs if (r["method"], r["b_star"], r["alpha"]) == (method, b, a)]
            if method == "gd":
                rows = [r for r in rows if r["converged"]]
            if not rows:
                continue
            estimated = sum(not r["converged"] for r in rows) > len(rows) / 2
            ax.plot(
                cert,
                np.median([r["per_decade"] for r in rows]),
                linestyle="none",
                marker=marker,
                markersize=marker_size[method],
                mfc="white" if estimated else color,
                mec=color,
                markeredgewidth=0.8,
                zorder=z,
            )
    ax.set(xscale="log", yscale="log", xlim=(grid[0], grid[-1]), ylim=(grid[0] / 2, grid[-1]))
    ticks = [1, 1e2, 1e4, 1e6, 1e8]
    labels = ["1", r"$10^2$", r"$10^4$", r"$10^6$", r"$10^8$"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Certified cost per decade")
    ax.set_ylabel("Measured cost per decade")
    methods = [
        Line2D(
            [],
            [],
            linestyle="none",
            color=figures["method_colors"][m],
            marker=figures["method_markers"][m],
            markersize=3.5,
            label=figures["method_labels"][m],
        )
        for m in ("gd", "projected_gd")
    ]
    lines = [
        Line2D([], [], color="k", linewidth=0.9, label="Certified"),
        Line2D([], [], color="k", linewidth=0.9, linestyle=(0, (4, 2)), label="Half"),
    ]
    keys = {
        "frameon": False,
        "handlelength": 1.2,
        "handletextpad": 0.4,
        "borderaxespad": 0.1,
        "labelspacing": 0.3,
    }
    ax.add_artist(ax.legend(handles=methods, loc="upper left", **keys))
    ax.legend(handles=lines, loc="lower right", **keys)
    margins(fig, 0.5, 0.08, 0.42, 0.08)
    save(fig, path)


# Section 8.3 ---------------------------------------------------------------------------------


def noise_tolerance_figure(results, figures, alphas, path):
    """Noise budget and cheapest trap found against b*, one panel per leak coefficient. Each
    panel keeps its own vertical range."""
    shown = figures["noise_tolerance_display"]
    rows = read_csv(results / "noise_tolerance" / "traps.csv")
    header = 0.24
    nrows = -(-len(alphas) // 3)
    height = header + 1.95 * nrows
    fig, axes = plt.subplots(nrows, 3, figsize=(WIDTH, height), sharex=True, squeeze=False)
    for i, (ax, alpha) in enumerate(zip(axes.ravel(), alphas)):
        pair = sorted((r for r in rows if r["alpha"] == alpha), key=lambda r: r["b_star"])
        b_star = np.array([r["b_star"] for r in pair])
        budget = np.array([r["budget_over_F0"] for r in pair])
        trap = np.array([r["opt_over_F0"] for r in pair])
        low = 10 ** np.floor(np.log10(min(budget.min(), trap.min())) - 0.5)
        high = 10 ** np.ceil(np.log10(max(budget.max(), trap.max())) + 0.5)
        ax.fill_between(b_star, low, budget, color=shown["certified_color"], linewidth=0)
        ax.fill_between(b_star, budget, trap, color=shown["undecided_color"], linewidth=0)
        ax.fill_between(b_star, trap, high, color=figures["violation_color"], linewidth=0)
        ax.plot(b_star, budget, color="k", linewidth=0.7)
        ax.plot(b_star, trap, color="k", linewidth=0.7)
        ax.set(yscale="log", ylim=(low, high), xlim=(b_star.min(), b_star.max()))
        if i // 3 == nrows - 1:
            ax.set_xlabel(BIAS_LABEL)
        ax.set_title(rf"({chr(97 + i)}) $\alpha={alpha:g}$", loc="left")
    for row in axes:
        row[0].set_ylabel(r"$\mathrm{OPT}/F_\alpha(0)$")
    regions = [
        Patch(color=shown["certified_color"], label="GD certified"),
        Patch(color=shown["undecided_color"], label="Undecided"),
        Patch(color=figures["violation_color"], label="A trap exists"),
    ]
    fig.tight_layout(rect=(0, 0, 1, 1 - header / height), w_pad=0.8)
    top_legend(fig, regions, top=1.0)
    save(fig, path)


# Section 8.4 ---------------------------------------------------------------------------------


def iteration_ratios(results, alphas):
    """Per alpha and over the b* of the convergence-rate grid: the signal gain varrho, the ratio
    of the median costs of projected GD, ReLU over Leaky ReLU, and whether either median is an
    estimate (most of its runs have not converged after max_steps)."""
    config = config_of(results, "convergence_rate")
    summary = {
        (r["b_star"], r["alpha"]): r
        for r in read_csv(results / "convergence_rate" / "summary.csv")
        if r["method"] == "projected_gd"
    }
    out = {}
    for alpha in alphas:
        gain, ratio, estimated = [], [], []
        for b_star in config["b_star"]:
            relu, leaky = summary[(b_star, 0.0)], summary[(b_star, alpha)]
            gain.append(signal(b_star, alpha) / signal(b_star, 0.0))
            ratio.append(relu["median_per_decade_all"] / leaky["median_per_decade_all"])
            estimated.append(any(2 * r["converged"] < r["runs"] for r in (relu, leaky)))
        out[alpha] = np.array(gain), np.array(ratio), np.array(estimated)
    return out


def trap_exponents(results, alphas):
    """Per alpha and over the b* of the matched-bias grid: beta with
    (1 - alpha)^2 U_alpha / U_0 = varrho^beta at the common accuracy, left out (NaN) where
    varrho < 1.05, since beta amplifies small deviations of the trap gain there."""
    rows = {(r["b_star"], r["alpha"]): r for r in read_csv(results / "matched_bias" / "traps.csv")}
    b_stars = np.array(config_of(results, "matched_bias")["b_star"])
    out = {}
    for alpha in alphas:
        beta = []
        for b_star in b_stars:
            relu, leaky = rows[(b_star, 0.0)], rows[(b_star, alpha)]
            gain = leaky["Lambda"] / relu["Lambda"]
            trap = (1 - alpha) ** 2 * leaky["opt"] / relu["opt"]
            beta.append(np.log(trap) / np.log(gain) if gain >= 1.05 else np.nan)
        out[alpha] = np.array(beta)
    return b_stars, out


def matched_bias_figure(results, figures, path):
    shown = figures["matched_bias_display"]
    colors, markers = shown["alpha_colors"], shown["alpha_markers"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.4))

    # (a) Ratio of iterations per decade against the signal gain.
    iterations = iteration_ratios(results, shown["alpha"])
    top = max(max(g.max(), r.max()) for g, r, _ in iterations.values())
    line = np.geomspace(1, 10 ** np.ceil(np.log10(top)), 50)
    left.plot(line, line, color="k", linewidth=0.8, zorder=0)
    for alpha, (gain, ratio, estimated) in iterations.items():
        color, marker = colors[f"{alpha:g}"], markers[f"{alpha:g}"]
        for filled in (True, False):
            keep = estimated != filled
            left.plot(
                gain[keep],
                ratio[keep],
                linestyle="none",
                marker=marker,
                markersize=4.5,
                color=color,
                markerfacecolor=color if filled else "white",
            )
    left.set(xscale="log", yscale="log")
    left.yaxis.set_major_locator(LogLocator(base=100, numticks=8))
    left.set_xlabel(r"$\varrho=\Lambda_\alpha(b^{\star})/\Lambda_0(b^{\star})$")
    left.set_ylabel("Ratio of iterations per decade")
    left.set_title("(a) Projected GD", loc="left")

    # (b) Trap exponent beta against b*; lines join every b*, markers at integer b*.
    b_stars, exponents = trap_exponents(results, shown["alpha"])
    integer = np.isclose(b_stars, np.round(b_stars))
    right.axhspan(1, 2, color=shown["band_color"], linewidth=0, zorder=0)
    for alpha, beta in exponents.items():
        color, marker = colors[f"{alpha:g}"], markers[f"{alpha:g}"]
        right.plot(b_stars, beta, color=color, linewidth=1.0, zorder=2)
        right.plot(
            b_stars[integer],
            beta[integer],
            linestyle="none",
            marker=marker,
            markersize=3.5,
            color=color,
            zorder=3,
        )
    right.set(xlim=(b_stars.min(), b_stars.max()), ylim=(0.6, 2.2))
    right.set_xlabel(BIAS_LABEL)
    right.set_ylabel(r"$\beta$")
    exponent = round(np.log10(config_of(results, "matched_bias")["eps_rel"]))
    right.set_title(rf"(b) Cheapest traps, $\varepsilon=10^{{{exponent}}}F_0(0)$", loc="left")

    handles = [
        Line2D(
            [],
            [],
            color=colors[f"{a:g}"],
            marker=markers[f"{a:g}"],
            linestyle="none",
            markersize=4.5,
            label=rf"$\alpha={a:g}$",
        )
        for a in shown["alpha"]
    ]
    handles.append(Line2D([], [], color="k", linewidth=0.8, label=r"$\varrho$"))
    handles.append(Patch(color=shown["band_color"], label=r"$1\leq\beta\leq2$"))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.28 / 2.4), w_pad=1.5)
    top_legend(fig, handles, top=1.0, compact=True)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--config", type=Path, default=Path("configs/figures.toml"))
    args = parser.parse_args()
    figures = tomllib.loads(args.config.read_text())
    results, output = args.results, args.output
    supplementary = output / "supplementary"
    supplementary.mkdir(parents=True, exist_ok=True)
    style()
    alphas = config_of(results, "regional_bound")["alpha"]
    regional_bound_figure(
        results, figures, figures["paper_regional_bound_alpha"], output / "regional_bound.pdf", 2.1
    )
    regional_bound_figure(results, figures, alphas, supplementary / "regional_bound_all.pdf", 3.0)
    region_conditions_figure(results, figures, output / "region_conditions.pdf")
    region_conditions_bias_all(results, figures, supplementary / "region_conditions_bias_all.pdf")
    region_conditions_loss_all(results, figures, supplementary / "region_conditions_loss_all.pdf")
    critical_threshold_figure(results, figures, output / "critical_threshold.pdf")
    convergence_rate_figure(results, figures, output / "convergence_rate.pdf")
    noise_alphas = config_of(results, "noise_tolerance")["alpha"]
    noise_tolerance_figure(
        results, figures, figures["paper_noise_tolerance_alpha"], output / "noise_tolerance.pdf"
    )
    noise_tolerance_figure(
        results, figures, noise_alphas, supplementary / "noise_tolerance_all.pdf"
    )
    matched_bias_figure(results, figures, output / "matched_bias.pdf")
    print(f"Saved the figures to {output}")


if __name__ == "__main__":
    main()
