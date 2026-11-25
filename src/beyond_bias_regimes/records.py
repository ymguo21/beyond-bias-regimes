"""Result files, reproducibility metadata, and independent rechecks shared by experiments."""

import csv
import gzip
import importlib.metadata
import json
import os
import platform
import shutil
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath as mp
import numpy as np

from .gaussian import reference_moments, signal, target_second
from .precise import precise_values


def load_config(path):
    path = Path(path)
    config = tomllib.loads(path.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, output / "config.toml")
    return config, output


def _open_text(path, mode):
    # Large tables are stored gzip-compressed, marked by the suffix .gz.
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="")
    return path.open(mode, newline="")


def write_csv(path, rows):
    with _open_text(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse(value):
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path):
    """Rows with booleans and numbers parsed; other text (e.g. labels) kept as strings."""
    with _open_text(path, "r") as stream:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def metadata(config, elapsed):
    """Configuration, running time and software versions of a run."""
    return {
        "config": config,
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib", "mpmath")
        },
    }


RECHECK_KEYS = (
    "reference_loss_ratio",
    "reference_gap",
    "reference_gap_error",
    "precise_loss_ratio",
    "precise_gap",
    "precise_gap_error",
)


def _recheck(task):
    theta, b_star, alpha, digits, displacement, gap_limit = task
    loss, gradient, quad_error = reference_moments(theta, b_star, alpha, displacement)
    if displacement is None:
        error = theta - np.array([1.0, 0.0, b_star])
    else:
        error = displacement[0] * displacement[1]
    lam_distance2 = signal(b_star, alpha) * (error @ error)
    precise_loss, precise_gap, precise_error = precise_values(
        theta, b_star, alpha, digits, displacement, gap_as_float=gap_limit is None
    )
    result = {
        "reference_loss_ratio": float(2 * loss / target_second(b_star, alpha)),
        "reference_gap": float((error @ gradient - lam_distance2) / lam_distance2),
        "reference_gap_error": float(quad_error * np.sqrt(error @ error) / lam_distance2),
        "precise_loss_ratio": precise_loss,
        "precise_gap": float(precise_gap),
        "precise_gap_error": precise_error,
    }
    if gap_limit is not None:
        with mp.workdps(digits):
            result["precise_gap_confirmed"] = bool(precise_gap <= mp.mpf(gap_limit))
    return result


def recheck(rows, digits, *, gap_limit=None):
    """Recompute rows with the adaptive full-gradient oracle and the mpmath oracle.

    Rows carrying `delta` and `v0..v2` are exact displacements theta*+delta*v.
    If `gap_limit` is set, record whether each gap meets it before rounding to float.
    """
    tasks = [
        (
            np.array([row["w_parallel"], row["w_perp"], row["bias"]]),
            row["b_star"],
            row["alpha"],
            digits,
            (row["delta"], np.array([row["v0"], row["v1"], row["v2"]])) if "delta" in row else None,
            gap_limit,
        )
        for row in rows
    ]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_recheck, tasks, chunksize=4))
    for row, result in zip(rows, results):
        row.update(result)
    return rows


def reference_agreement(rows, loss_rtol):
    """Discrepancies of the adaptive integrator from the mpmath oracle.

    The adaptive integrator resolves the gap zeta - 1 only to its own error estimate (about
    1e-13 near the target), so its gap must agree with the 40-digit value within that estimate,
    and its loss ratio to `loss_rtol`. Gaps smaller than the estimate are counted: their sign
    is decided by the mpmath oracle alone.
    """
    excess = max(
        abs(r["reference_gap"] - r["precise_gap"]) / r["reference_gap_error"] for r in rows
    )
    loss = max(abs(r["reference_loss_ratio"] / r["precise_loss_ratio"] - 1) for r in rows)
    return {
        "max_reference_gap_difference_over_error": excess,
        "max_reference_relative_loss_difference": loss,
        "gaps_below_reference_resolution": sum(
            abs(r["precise_gap"]) < r["reference_gap_error"] for r in rows
        ),
        "reference_agreement": bool(excess <= 1 and loss <= loss_rtol),
    }


def agreement(rows, gap_rtol, gap_atol, loss_rtol):
    """Discrepancies from the mpmath oracle, of the primary evaluation and of the adaptive
    integrator. Scaled-residual gaps must agree to `gap_rtol`; moment-polynomial gaps may add
    an absolute rounding `gap_atol` in zeta; the adaptive integrator must agree within its
    error estimate (`reference_agreement`)."""
    scaled = [r for r in rows if r["scaled"]]
    polynomial = [r for r in rows if not r["scaled"]]
    relative = max((abs(r["gap"] / r["precise_gap"] - 1) for r in scaled), default=0.0)
    absolute = max((abs(r["gap"] - r["precise_gap"]) for r in polynomial), default=0.0)
    polynomial_ok = all(
        abs(r["gap"] - r["precise_gap"]) <= gap_atol + gap_rtol * abs(r["precise_gap"])
        for r in polynomial
    )
    loss = max(abs(r["loss_ratio"] / r["precise_loss_ratio"] - 1) for r in rows)
    reference = reference_agreement(rows, loss_rtol)
    return {
        "scaled_rows": len(scaled),
        "polynomial_rows": len(polynomial),
        "max_relative_gap_difference_scaled": relative,
        "max_absolute_gap_difference_polynomial": absolute,
        "max_relative_loss_difference": loss,
        **reference,
        "agreement": bool(
            relative <= gap_rtol
            and polynomial_ok
            and loss <= loss_rtol
            and reference["reference_agreement"]
        ),
    }
