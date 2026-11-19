"""Grid search over affine directions with exact handling of the scale.

Along a ray theta=s*u, the loss and the correction margin are quadratics in s,
so each experiment reduces the scale analytically and searches only the two
direction angles (psi, beta). A coarse grid is followed by finer grids that
span the neighbours of the current minimizer.
"""

import numpy as np

from .gaussian import Geometry


def zoom(grid, value, points):
    """A finer grid spanning the two neighbours of `value` in `grid`."""
    i = np.searchsorted(grid, value)
    return np.linspace(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], points)


def beta_grid(low, high, linear_points, geometric_points, geometric_min):
    """Linear coverage plus geometric offsets that resolve the face beta=low."""
    fractions = np.r_[
        np.linspace(0, 1, linear_points), np.geomspace(geometric_min, 1, geometric_points)
    ]
    # Clip so that rounding never pushes beta past +-pi/2 (a negative weight cosine).
    return np.unique(np.clip(low + (high - low) * fractions, low, high))


def minimize(objective, b_star, alpha, psi, beta, base, levels, points):
    """Minimize objective(coefficients) -> (value, scale) over directions.

    `base` holds the Geometry of the coarse meshgrid, shared across leak
    coefficients. Returns the best value with its direction and scale.
    """
    P, B = (x.ravel() for x in np.meshgrid(psi, beta, indexing="ij"))
    value, scale = objective(base.coefficients(alpha))
    best = None
    for level in range(levels + 1):
        if np.all(np.isnan(value)):
            break
        i = int(np.nanargmin(value))
        if best is None or value[i] < best["value"]:
            best = {"value": float(value[i]), "psi": P[i], "beta": B[i], "scale": scale[i]}
        if level == levels:
            break
        psi, beta = zoom(psi, P[i], points), zoom(beta, B[i], points)
        P, B = (x.ravel() for x in np.meshgrid(psi, beta, indexing="ij"))
        value, scale = objective(Geometry.build(P, B, b_star).coefficients(alpha))
    return best


def theta_of(psi, beta, scale):
    """Candidate coordinates (w_parallel, w_perp >= 0, b) of the point scale*u(psi, beta)."""
    weight = 0.0 if abs(beta) >= np.pi / 2 else np.cos(beta)
    return scale * np.array([weight * np.cos(psi), weight * np.sin(psi), np.sin(beta)])
