"""Closed-form population loss and gradient at arbitrary parameters, for gradient descent.

Coordinates are those of `gaussian.py`: theta=(w_1,w_2,b) with w*=e_1 and
theta*=(1,0,b_star); GD and its projected variant stay in this plane. Every
function is vectorized over runs, and
b_star and alpha may differ between runs. With z=<theta,X~> and
z*=<theta*,X~>, sigma(z)s(z)=s(z)^2 z and sigma(z*)s(z)=s(z)s(z*)z*, so

    grad F(theta) = E[s(z)^2 X~ X~^T] theta - E[s(z)s(z*) X~ X~^T] theta*.

Both terms reduce to Gaussian moments of one gate, E[1{z>0} X~ X~^T], and of
two gates, E[1{z>0}(z*)_+ X~]; the latter needs the bivariate normal orthant
probability, evaluated through Owen's T function. Nothing is sampled or
integrated numerically. See docs/theory.md for the derivation.
"""

from itertools import pairwise

import numpy as np
from scipy.special import ndtr, owens_t

from .gaussian import phi, rule, target_second


def orthant(h, k, rho, s=None, k_minus=None, h_minus=None):
    """P(Z_1 < h, Z_2 < k) for standard normals with correlation |rho| < 1 (Owen, 1956).

    Near-collinear callers pass s=sqrt(1-rho^2), k_minus=k-rho*h and h_minus=h-rho*k,
    formed without cancellation.
    """
    h, k, rho = np.broadcast_arrays(*(np.asarray(x, float) for x in (h, k, rho)))
    if s is None:
        s = np.sqrt((1 - rho) * (1 + rho))
    if k_minus is None:
        k_minus, h_minus = k - rho * h, h - rho * k
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.where(h != 0, k_minus / (h * s), np.copysign(np.inf, k_minus))
        tk = np.where(k != 0, h_minus / (k * s), np.copysign(np.inf, h_minus))

    # Owen's T(h, +-inf) = (1 - Phi(|h|))/2 * sign; at h=0 it is +-1/4.
    def t_of(x, slope):
        infinite = np.isinf(slope)
        limit = np.sign(slope) * 0.5 * ndtr(-np.abs(x))
        return np.where(infinite, limit, owens_t(x, np.where(infinite, 0.0, slope)))

    straddle = (h * k < 0) | ((h * k == 0) & (h + k < 0))
    value = 0.5 * (ndtr(h) + ndtr(k)) - t_of(h, th) - t_of(k, tk) - 0.5 * straddle
    # Owen's formula is indeterminate at the origin, where the orthant has a closed form.
    origin = 0.25 + np.arctan2(rho, s) / (2 * np.pi)
    return np.where((h == 0) & (k == 0), origin, value)


def _gates(theta, b_star):
    """Moments of the candidate gate g=1{z>0} and the target ramp (z*)_+.

    Returns the scalars needed by `population`; theta has shape (n, 3).
    """
    p, q, b = theta.T
    r = np.hypot(p, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(r > 0, b / r, np.copysign(np.inf, b))
        c = np.where(r > 0, p / r, 0.0)
        s = np.where(r > 0, np.abs(q) / r, 1.0)
    big_a = np.where(np.isfinite(a), a, 0.0)
    gate = ndtr(a)  # P(z>0)
    density = np.where(np.isfinite(a), phi(big_a), 0.0)  # phi(b/r)
    # Joint terms with z*=X_1+b_star. U=(z-b)/r has correlation c with X_1 and
    # s=sqrt(1-c^2)=|q|/r; 1-c is formed as q^2/(r(r+p)) when p>0.
    collinear = s == 0
    s_safe = np.where(collinear, 1.0, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        one_minus_c = np.where(p > 0, q * q / (r * (r + p)), 1 - c)
    m = (b_star - big_a) + one_minus_c * big_a  # b_star - c a
    n = (big_a - b_star) + one_minus_c * b_star  # a - c b_star
    joint = np.where(
        collinear,
        np.where(
            c > 0, ndtr(np.minimum(big_a, b_star)), np.maximum(ndtr(big_a) - ndtr(-b_star), 0)
        ),
        orthant(big_a, b_star, c, s_safe, m, n),
    )
    joint = np.where(np.isfinite(a), joint, np.where(b > 0, ndtr(b_star), 0.0))
    # On the collinear line the conditional probabilities are limits as |q| -> 0.
    upper = np.where(collinear, np.heaviside(m, 0.5), ndtr(m / s_safe))
    across = np.where(collinear, np.heaviside(n, 0.5), ndtr(n / s_safe))
    # E[delta(z)(z*)_+] = phi(a)/r * E[(X_1+b_star)_+ | U=-a]
    ramp_at_gate = np.where(
        collinear,
        np.maximum(m, 0.0),
        m * ndtr(m / s_safe) + s_safe * phi(m / s_safe),
    )
    return {
        "r": r,
        "a": big_a,
        "gate": gate,
        "density": density,
        "joint": joint,  # P(z>0, z*>0)
        "upper": upper,  # P(z*>0 | z=0)
        "across": across,  # P(z>0 | z*=0)
        "ramp_at_gate": ramp_at_gate,
        "finite": np.isfinite(a) & (r > 0),
    }


# Points closer to the target than this fraction of ||theta*|| use the wedge integral.
NEAR_TARGET = 0.1


def wedge(delta, v, b_star, alpha, order=96):
    """Omega = E[s(z)(s(z)-s(z*)) z* X~] at theta = theta* + delta v, without cancellation.

    The integrand vanishes unless the candidate and target gates disagree, which
    confines it to a wedge of width O(delta) around the target gate; there z* is
    O(delta), so Omega = O(delta^2). Conditioning on Z = X_1 = -b* + delta u, the candidate
    preactivation is N(delta (u + v_0 Z + v_2), (delta v_1)^2), and only the scaled
    variable u is integrated, as in `gaussian.local_moments`.
    """
    delta, v = np.asarray(delta, float), np.asarray(v, float)
    b_star, alpha = (np.broadcast_to(np.asarray(x, float), delta.shape) for x in (b_star, alpha))
    v0, v1, v2 = v[:, 0], np.abs(v[:, 1]), v[:, 2]
    p = 1 + delta * v0
    nodes, weights = rule(order)
    result = np.zeros((len(delta), 3))
    center = (b_star * v0 - v2) / p
    width = 12 * v1 / p
    cuts = np.sort(
        np.stack((np.zeros_like(center), center - width, center, center + width)), axis=0
    )
    for low, high in pairwise(cuts):
        half = (high - low) / 2
        u = (high + low)[:, None] / 2 + half[:, None] * nodes
        z = -b_star[:, None] + delta[:, None] * u
        candidate = u + v0[:, None] * z + v2[:, None]
        spread = v1[:, None]
        x = np.divide(candidate, spread, out=np.zeros_like(u), where=spread != 0)
        above = np.where(spread != 0, ndtr(x), candidate > 0)
        below = np.where(spread != 0, ndtr(-x), candidate <= 0)
        density = np.where(spread != 0, phi(x), 0.0)
        a = alpha[:, None]
        # z* <= 0 (u <= 0): weight (1-alpha) on {z > 0}; z* > 0: -alpha (1-alpha) on {z <= 0}.
        gate = np.where(u <= 0, above, -a * below)
        perpendicular = np.where(u <= 0, density, a * density)
        common = (1 - a) * u * phi(z) * half[:, None]
        result[:, 0] += (common * gate * z) @ weights
        result[:, 1] += (common * perpendicular) @ weights
        result[:, 2] += (common * gate) @ weights
    # W_2 is odd in w_perp, the only coordinate entering through its sign.
    result[:, 1] *= np.sign(v[:, 1])
    return delta[:, None] ** 2 * result


def population(theta, b_star, alpha, error=None):
    """Loss F, gradient, and the pieces behind them at points theta (n, 3).

    With e = theta - theta*, the gradient is assembled as

        grad F = alpha^2 e + (1 - alpha^2) Sigma_+(theta) e + Omega(theta),

    where Sigma_+(theta) = E[1{z>0} X~ X~^T] and Omega(theta) = E[s(z)(s(z)-s(z*)) z* X~]. Both
    depend on theta only through its gate. Near the target Omega comes from the
    cancellation-free `wedge`, elsewhere from closed forms.
    Returns `loss`, `gradient`, `fy` = E[f_theta Y], `ff` = E[f_theta^2], `gate_matrix`
    Sigma_+ (n, 3, 3) and `wedge` Omega (n, 3). Rows with w = 0 have the gate 1{b > 0}.
    An optional exact displacement `error` = theta - theta* replaces the rounded
    difference of coordinates in Omega and in the assembly.
    """
    theta = np.atleast_2d(np.asarray(theta, float))
    n = len(theta)
    b_star = np.broadcast_to(np.asarray(b_star, float), (n,))
    alpha = np.broadcast_to(np.asarray(alpha, float), (n,))
    # Nearly collinear weights send some standardized arguments to +-inf; their limits are exact.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        g = _gates(theta, b_star)
    p, q, b = theta.T
    r = g["r"]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r > 0, 1 / r, 0.0)
    gate, dens = g["gate"], g["density"] * g["finite"]
    lin = dens * inv_r  # E[delta(z)]
    curv = -b * dens * inv_r**3  # coefficient of w w^T in E[1{z>0} X X^T]
    gate_matrix = np.zeros((n, 3, 3))
    gate_matrix[:, 0, 0] = gate + curv * p * p
    gate_matrix[:, 1, 1] = gate + curv * q * q
    gate_matrix[:, 0, 1] = gate_matrix[:, 1, 0] = curv * p * q
    gate_matrix[:, 0, 2] = gate_matrix[:, 2, 0] = lin * p
    gate_matrix[:, 1, 2] = gate_matrix[:, 2, 1] = lin * q
    gate_matrix[:, 2, 2] = gate
    target = np.stack((np.ones(n), np.zeros(n), b_star), axis=1)
    gated_target = np.einsum("nij,nj->ni", gate_matrix, target)  # E[1{z>0} z* X~]
    target_ramp = np.stack((ndtr(b_star), np.zeros(n), phi(b_star) + b_star * ndtr(b_star)), axis=1)
    # E[1{z>0}(z*)_+ X~] by Gaussian integration by parts in X.
    at_gate = lin * g["ramp_at_gate"]  # E[delta(z)(z*)_+]
    two_gate_const = b_star * g["joint"] + p * inv_r * dens * g["upper"] + phi(b_star) * g["across"]
    two_gate = np.stack((at_gate * p + g["joint"], at_gate * q, two_gate_const), axis=1)
    a2, mixed, both = (
        (alpha**2)[:, None],
        (alpha * (1 - alpha))[:, None],
        ((1 - alpha) ** 2)[:, None],
    )
    k_target = a2 * target + mixed * (gated_target + target_ramp) + both * two_gate
    # Omega = E[s(z)^2 z* X~] - E[s(z) s(z*) z* X~].
    wedge_term = a2 * target + (1 - a2) * gated_target - k_target
    error = theta - target if error is None else np.atleast_2d(np.asarray(error, float))
    distance = np.linalg.norm(error, axis=1)
    near = (distance > 0) & (distance < NEAR_TARGET * np.sqrt(1 + b_star**2)) & (p > 0)
    if near.any():
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            wedge_term[near] = wedge(
                distance[near], error[near] / distance[near, None], b_star[near], alpha[near]
            )
    # E[s(z)^2 z^2] and E[s(z) s(z*) z z*].
    ramp_second = (r * r + b * b) * gate + b * r * dens
    ff = alpha**2 * np.sum(theta**2, axis=1) + (1 - alpha**2) * ramp_second
    fy = np.sum(theta * k_target, axis=1)
    yy = target_second(b_star, alpha)
    return {
        "loss": 0.5 * (ff - 2 * fy + yy),
        "gradient": gradient_at(target + error, target, alpha, gate_matrix, wedge_term, error),
        "fy": fy,
        "ff": ff,
        "gate_matrix": gate_matrix,
        "wedge": wedge_term,
    }


def gradient_at(theta, target, alpha, gate_matrix, wedge_term, error=None):
    """grad F at points theta sharing the gates of `gate_matrix` and `wedge_term`."""
    error = theta - target if error is None else error
    a2 = (np.asarray(alpha) ** 2)[..., None]
    return a2 * error + (1 - a2) * np.einsum("nij,nj->ni", gate_matrix, error) + wedge_term
