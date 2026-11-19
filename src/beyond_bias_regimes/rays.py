"""Stable exact one-dimensional formulas for the two weakest-direction rays.

Integrate only the gate-switching interval, and add its signed correction
to the target Hessian value. In particular, tiny zeta-1 values are not formed
by subtracting nearly equal loss moments or by finite differences.
"""

import numpy as np

from .gaussian import kappa, phi, rule, signal, target_second


def weakest_ray(b_star, alpha, delta, sign=1, order=64):
    """theta=theta*+sign*delta*v_min, ||v_min||=1; the positive sign satisfies the bias
    condition b >= b*||w||.

    The positive ray has a positive weight only for delta < sqrt(1+kappa^2)/kappa; the
    negative ray is defined for every delta > 0.
    """
    delta = np.asarray(delta, float)
    k = kappa(b_star)
    n = np.sqrt(1 + k * k)
    t = delta / n
    p = 1 - sign * k * t
    width = t * k * k / p
    nodes, weights = rule(order)
    x, weights = (nodes + 1) / 2, weights / 2
    if sign == 1:
        offset = -width[..., None] * (1 - x)
        h = k * k - k * offset
        qchange = (1 - alpha) * h * (k * k * x + alpha * h)
        fchange = 0.5 * (1 - alpha) * k * k * x * (2 * alpha * h + (1 - alpha) * k * k * x)
    else:
        offset = width[..., None] * x
        h = k * k - k * offset
        qchange = -(1 - alpha) * h * (h + alpha * k * k * (1 - x))
        fchange = -0.5 * (1 - alpha) * k * k * (1 - x) * (2 * h - (1 - alpha) * k * k * (1 - x))
    density = phi(-b_star + offset)
    qgap = width * ((qchange * density) @ weights) / n**2
    loss_scaled = signal(b_star, alpha) / 2 + width * ((fchange * density) @ weights) / n**2
    # First-order coefficient for zeta-1 as a function of the unit displacement delta.
    factor = (1 - alpha) * (alpha + 0.5 if sign == 1 else -(1 + alpha / 2))
    slope = factor * phi(b_star) * k**6 / (n**3 * signal(b_star, alpha))
    return {
        "delta": delta,
        "w": p,
        "b": b_star + sign * t,
        "bias_slack": sign * t * k * k,
        "ratio_gap": qgap / signal(b_star, alpha),
        "ratio": 1 + qgap / signal(b_star, alpha),
        "loss": delta**2 * loss_scaled,
        "loss_ratio": 2 * delta**2 * loss_scaled / target_second(b_star, alpha),
        "asymptotic_gap": slope * delta,
    }
