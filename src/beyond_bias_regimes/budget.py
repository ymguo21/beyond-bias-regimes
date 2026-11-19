"""The closed-form noise budget of agnostic GD (Section 6 of the paper).

For an accuracy eps > 0,

    B_alpha(b*; eps) = Lambda / (2 (1-alpha)^2) max{Lambda eps, chi(b*) eps,
                                                    (2 ||theta*|| Lambda / pi) sqrt(eps)},

with chi(b*) = 1 / (25 ((b*)^2 + 7)^2).
"""

import math

import numpy as np

from .gaussian import signal

# Entries of the maximum, in the order of the display.
ENTRIES = ("linear", "signal", "angular")


def chi(b_star):
    return 1 / (25 * (b_star * b_star + 7) ** 2)


def budget(b_star, alpha, eps):
    """(B_alpha(b*; eps), name of the largest entry of the maximum)."""
    lam = signal(b_star, alpha)
    norm = math.sqrt(1 + b_star * b_star)
    entries = (lam * eps, chi(b_star) * eps, 2 * norm * lam / math.pi * math.sqrt(eps))
    k = int(np.argmax(entries))
    return float(lam / (2 * (1 - alpha) ** 2) * entries[k]), ENTRIES[k]
