"""Cheapest trap labels for agnostic GD.

Labels are Y = f_theta*(X) + xi(X) with E[xi^2] = 2 OPT. A trap is a fixed point theta^ of
oracle projected GD in H cap B(theta*, R) with ||theta^ - theta*||^2 >= eps: either
grad L(theta^) = 0, or theta^ lies on the face b = b* <w, w*> of H with grad L(theta^) =
-mu n, mu >= 0, for the outward unit normal n. Stationarity of theta* and the trap condition
are linear in xi and see xi only through E[s(z*) X~ xi] and E[s(z^) X~ xi], so the cheapest xi
lies in the span of the six features s(z*) X^ and (s(z^) - s(z*)) X^,

    xi = <lam1, s(z*) X^> + <lam2, (s(z^) - s(z*)) X^>,
    G11 lam1 + G12 lam2 = 0,        S lam2 = grad^ F(theta^) + mu n^,
    OPT = (1/2) (grad^ F(theta^) + mu n^)^T S^{-1} (grad^ F(theta^) + mu n^),

with S = G22 - G12 G11^{-1} G12 and mu >= 0 minimizing OPT (mu = 0 off the face). Hats
denote the shifted basis X^ = (z*, X_2, 1) = (X_1 + b*, X_2, 1), in which the first feature
coordinate is the target preactivation. The difference features live on the thin wedges where
the two slopes differ; there X_1 is close to -b* and, at strongly negative b*, the features
X_1 and 1 of the basis X~ = (X_1, X_2, 1) would be nearly collinear, whereas z* and 1 are not.

All moments are sums of "atoms" A_p = E[1{p} X^ X^^T] over the on/off patterns p of a few
gates (`atoms`). For fixed X_2 every gate is a half-line in X_1, so a pattern is an interval
whose truncated Gaussian moments are closed form, or Gauss-Legendre in z* on short intervals;
the X_2 integral is Gauss-Legendre on segments split where interval ends cross. Wedge atoms
are computed directly, never as differences. `reference` recomputes the trap quantities by
adaptive integration in the opposite order, as an independent check.

Optimality of theta* needs no constraint on xi beyond the two stationarity conditions: by
homogeneity of f_theta, L(theta^) - L(theta*) = (E[f_theta*^2] - E[f_theta^^2]) / 2 for every
such xi (`trap_gap`), so the search keeps traps with E[f_theta^^2] <= E[f_theta*^2], and
`excess_loss` audits global optimality. See "Trap labels" in docs/theory.md.
"""

from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import ndtr

from .gaussian import phi

X2_MAX = 12.0
X2_NODES, X2_WEIGHTS = leggauss(12)
# Values of z* at which X_2-segments are split, so that interval ends crossing the bulk of
# the Gaussian are resolved.
Z_LEVELS = np.array([-10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0])
# Intervals in X_1 no longer than this use Gauss-Legendre in z* instead of closed forms.
SHORT, SHORT_NODES, SHORT_WEIGHTS = 1.0, *leggauss(20)


# ---- geometry -----------------------------------------------------------------------------


def target(b_star):
    return np.array([1.0, 0.0, b_star])


def outward_normal(b_star):
    """Outward unit normal n of H = {b >= b* w_1}; theta* lies on its face."""
    return np.array([b_star, 0.0, -1.0]) / np.sqrt(1 + b_star**2)


def frame(b_star):
    """Inward normal m = -n and the in-face directions e2 = (0, 1, 0), e3 = m x e2."""
    m = -outward_normal(b_star)
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.cross(m, e2)
    return m, e2, e3 / np.linalg.norm(e3)


def displacement(b_star, log_r, chi, omega):
    """theta - theta* = r (sin chi m + cos chi (cos omega e3 + sin omega e2)).

    chi in [0, pi/2] keeps theta in H (chi = 0 is the face); omega in [0, pi] keeps w_2 >= 0,
    which loses nothing because the problem is symmetric under w_2 -> -w_2.
    """
    m, e2, e3 = frame(b_star)
    log_r, chi, omega = np.broadcast_arrays(*(np.asarray(x, float) for x in (log_r, chi, omega)))
    v = (
        np.sin(chi)[..., None] * m
        + (np.cos(chi) * np.cos(omega))[..., None] * e3
        + (np.cos(chi) * np.sin(omega))[..., None] * e2
    )
    return np.exp(log_r)[..., None] * v


def hat(v, b_star):
    """Coefficients in X^ of the linear form <v, X~>: (v_1, v_2, v_3 - b* v_1).

    For a parameter, the last entry is its bias slack b - b* w_1 in H.
    """
    out = np.array(v, float, copy=True)
    out[..., 2] = out[..., 2] - b_star * out[..., 0]
    return out


def hat_dual(g, b_star):
    """E[g X^] from E[g X~]."""
    out = np.array(g, float, copy=True)
    out[..., 0] = out[..., 0] + b_star * out[..., 2]
    return out


# ---- atoms --------------------------------------------------------------------------------


def _interval_moments(low, high, b_star):
    """Integrals of 1, y, y^2 against the density of y = z* = X_1 + b* on [low, high] in y."""
    high = np.maximum(high, low)
    lo, hi = low - b_star, high - b_star  # in X_1
    upper = lo >= 0
    I0 = np.where(upper, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    plo, phi_hi = phi(lo), phi(hi)
    lplo = np.where(np.isfinite(lo), np.nan_to_num(lo) * plo, 0.0)
    hphi = np.where(np.isfinite(hi), np.nan_to_num(hi) * phi_hi, 0.0)
    I1, I2 = plo - phi_hi, I0 + lplo - hphi
    J0, J1, J2 = I0, I1 + b_star * I0, I2 + 2 * b_star * I1 + b_star**2 * I0
    short = np.isfinite(low) & np.isfinite(high) & (high - low <= SHORT)
    if short.any():
        half = 0.5 * (high[short] - low[short])
        y = (low[short] + half)[:, None] + half[:, None] * SHORT_NODES
        w = half[:, None] * SHORT_WEIGHTS * phi(y - b_star)
        J0, J1, J2 = J0.copy(), J1.copy(), J2.copy()
        J0[short], J1[short], J2[short] = w.sum(1), (w * y).sum(1), (w * y * y).sum(1)
    return J0, J1, J2


def _segments(gates, b_star):
    """Sorted X_2-breakpoints (n, m) for gates (n, k, 3) in the hat basis."""
    a1, a2, a3 = gates[..., 0], gates[..., 1], gates[..., 2]
    points = [np.broadcast_to(np.arange(-X2_MAX, X2_MAX + 0.5), gates.shape[:1] + (25,))]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Gate j switches at y = -(a2 X_2 + a3) / a1, which equals level L at this X_2.
        for level in Z_LEVELS + b_star:
            points.append(np.where(a2 != 0, -(level * a1 + a3) / a2, np.nan))
        for i in range(gates.shape[1]):
            for j in range(i + 1, gates.shape[1]):
                num = a3[:, j] * a1[:, i] - a3[:, i] * a1[:, j]
                den = a2[:, i] * a1[:, j] - a2[:, j] * a1[:, i]
                points.append(np.where(den != 0, num / den, np.nan)[:, None])
    points = np.concatenate(points, axis=1)
    points = np.where(np.isfinite(points), np.clip(points, -X2_MAX, X2_MAX), X2_MAX)
    return np.sort(points, axis=1)


def atoms(gates, b_star, chunk=512):
    """A_p = E[1{pattern p} X^ X^^T] for gates (n, k, 3) in the hat basis; shape (n, 2^k, 3, 3).

    Bit j of p is set when gate j is on (<gate_j, X^> > 0).
    """
    gates = np.asarray(gates, float)
    return np.concatenate(
        [_atoms(gates[i : i + chunk], b_star) for i in range(0, len(gates), chunk)]
    )


def _atoms(gates, b_star):
    n, k, _ = gates.shape
    edges = _segments(gates, b_star)
    half = 0.5 * np.diff(edges, axis=1)
    x2 = (0.5 * (edges[:, 1:] + edges[:, :-1]))[..., None] + half[..., None] * X2_NODES
    weight = ((half[..., None] * X2_WEIGHTS) * phi(x2)).reshape(n, -1)
    x2 = x2.reshape(n, -1)
    a1 = gates[:, :, 0][:, :, None]
    level = gates[:, :, 1][:, :, None] * x2[:, None, :] + gates[:, :, 2][:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        switch = -level / a1
    pos, neg, on_flat = a1 > 0, a1 < 0, level > 0
    inf = np.inf
    # Each gate is on for y in (switch, inf) if a1 > 0, (-inf, switch) if a1 < 0, and on
    # everywhere or nowhere if a1 = 0.
    on_lo = np.where(pos, switch, np.where(neg, -inf, np.where(on_flat, -inf, inf)))
    on_hi = np.where(pos, inf, np.where(neg, switch, np.where(on_flat, inf, -inf)))
    off_lo = np.where(pos, -inf, np.where(neg, switch, np.where(on_flat, inf, -inf)))
    off_hi = np.where(pos, switch, np.where(neg, inf, np.where(on_flat, -inf, inf)))
    out = np.zeros((n, 2**k, 3, 3))
    for p in range(2**k):
        lo, hi = np.full(x2.shape, -inf), np.full(x2.shape, inf)
        for j in range(k):
            on = (p >> j) & 1
            lo = np.maximum(lo, on_lo[:, j] if on else off_lo[:, j])
            hi = np.minimum(hi, on_hi[:, j] if on else off_hi[:, j])
        m0, m1, m2 = (
            m.reshape(lo.shape) * weight
            for m in _interval_moments(lo.ravel(), np.maximum(hi, lo).ravel(), b_star)
        )
        out[:, p, 0, 0] = m2.sum(1)
        out[:, p, 0, 1] = out[:, p, 1, 0] = (x2 * m1).sum(1)
        out[:, p, 0, 2] = out[:, p, 2, 0] = m1.sum(1)
        out[:, p, 1, 1] = (x2 * x2 * m0).sum(1)
        out[:, p, 1, 2] = out[:, p, 2, 1] = (x2 * m0).sum(1)
        out[:, p, 2, 2] = m0.sum(1)
    return out


def slopes(alpha, k):
    """(2^k, k): the slope of gate j under pattern p."""
    on = (np.arange(2**k)[:, None] >> np.arange(k)[None, :]) & 1
    return alpha + (1 - alpha) * on


# ---- cheapest trap labels -----------------------------------------------------------------


def trap_pieces(error, b_star, alpha):
    """Gram blocks, grad^ F and the trap gap at theta^ = theta* + error (n, 3).

    With e^ = hat(error), t^ = hat(theta*) + e^ and per-pattern slopes s*, s^:
    f^ - f* = <s* e^ + (s^ - s*) t^, X^>, so grad^ F = E[s^ (f^ - f*) X^] and the trap gap
    -(E[f^^2] - E[f*^2]) / 2 = -E[(f^ - f*)(f^ + f*)] / 2 are sums over atoms with no
    difference of large terms.
    """
    error = np.atleast_2d(error)
    n = len(error)
    e_h = hat(error, b_star)
    # hat(theta*) = (1, 0, 0), so t^ keeps the full relative accuracy of the bias slack e^_3,
    # which forming theta* + error first would lose at strongly negative b*.
    star_h = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (n, 3))
    t_h = star_h + e_h
    A = atoms(np.stack((star_h, t_h), axis=1), b_star)  # bit 0: theta*, bit 1: theta^
    S = slopes(alpha, 2)
    ss, sh = S[:, 0], S[:, 1]
    diff = ss[None, :, None] * e_h[:, None, :] + (sh - ss)[None, :, None] * t_h[:, None, :]
    total = diff + 2 * ss[None, :, None] * star_h[:, None, :]
    return {
        "G11": np.einsum("p,npij->nij", ss * ss, A),
        "G12": np.einsum("p,npij->nij", ss * (sh - ss), A),
        "G22": np.einsum("p,npij->nij", (sh - ss) ** 2, A),
        "gradient_hat": np.einsum("p,npij,npj->ni", sh, A, diff),
        "trap_gap": -0.5 * np.einsum("npi,npij,npj->n", diff, A, total),
    }


def solve_labels(G11, G12, G22, gradient_hat, normal_hat, face):
    """Cheapest labels from the Gram blocks: (opt, mu, lam1, lam2, condition number).

    The Schur complement is scaled to unit diagonal before solving; rows whose scaled
    complement is singular to working precision get OPT = inf.
    """
    X = np.linalg.solve(G11, G12)
    S = G22 - np.einsum("nij,njk->nik", G12, X)
    S = 0.5 * (S + np.swapaxes(S, 1, 2))
    scale = np.sqrt(np.clip(np.einsum("nii->ni", S), 1e-300, None))
    Ss = S / scale[:, :, None] / scale[:, None, :]
    eig = np.linalg.eigvalsh(Ss)
    ok = eig[:, 0] > 1e-12 * eig[:, -1]
    Ss = np.where(ok[:, None, None], Ss, np.eye(3))
    g, n = gradient_hat / scale, normal_hat / scale
    u = np.linalg.solve(Ss, g[..., None])[..., 0]
    v = np.linalg.solve(Ss, n[..., None])[..., 0]
    face = np.broadcast_to(face, len(g))
    mu = np.where(face, np.maximum(0.0, -np.sum(n * u, axis=1) / np.sum(n * v, axis=1)), 0.0)
    lam2 = u + mu[:, None] * v
    opt = np.where(ok, 0.5 * np.sum((g + mu[:, None] * n) * lam2, axis=1), np.inf)
    lam2 = lam2 / scale
    lam1 = -np.einsum("nij,nj->ni", X, lam2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return opt, mu, lam1, lam2, eig[:, -1] / np.abs(eig[:, 0])


def cheapest_labels(error, b_star, alpha, face):
    """Cheapest trap labels at theta^ = theta* + error (n, 3); face marks points on the face.

    Returns opt, mu, lam1, lam2 (hat basis), the condition number of the scaled Schur
    complement, grad^ F and the trap gap L(theta^) - L(theta*).
    """
    pieces = trap_pieces(error, b_star, alpha)
    normal_hat = np.broadcast_to(hat_dual(outward_normal(b_star), b_star), (len(pieces["G11"]), 3))
    opt, mu, lam1, lam2, cond = solve_labels(
        pieces["G11"], pieces["G12"], pieces["G22"], pieces["gradient_hat"], normal_hat, face
    )
    return {
        "opt": opt,
        "mu": mu,
        "lam1": lam1,
        "lam2": lam2,
        "cond": cond,
        "gradient_hat": pieces["gradient_hat"],
        "trap_gap": pieces["trap_gap"],
    }


def excess_loss(theta, b_star, alpha, trap_error, lam1, lam2):
    """L(theta) - L(theta*) and grad L(theta) (basis X~) at points theta (n, 3) under the
    labels (lam1, lam2) of the trap theta* + trap_error.

    L(theta) - L(theta*) = E[(f - f*)^2] / 2 - E[xi (f - f*)] and grad^ L = E[s (f - f* - xi) X^],
    with f - f* = <s* e^ + (s - s*) t^, X^> and xi = <s* lam1 + (s^ - s*) lam2, X^>, summed
    over the atoms of the gates of theta*, the trap and theta.
    """
    theta = np.atleast_2d(theta)
    n = len(theta)
    e_h = hat(theta - target(b_star), b_star)
    star_h = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (n, 3))
    t_h = star_h + e_h
    trap_h = star_h + hat(trap_error, b_star)
    A = atoms(np.stack((star_h, trap_h, t_h), axis=1), b_star)  # bits: theta*, trap, theta
    S = slopes(alpha, 3)
    ss, sh, st = S[:, 0], S[:, 1], S[:, 2]
    diff = ss[None, :, None] * e_h[:, None, :] + (st - ss)[None, :, None] * t_h[:, None, :]
    noise = ss[None, :, None] * lam1[None, None, :] + (sh - ss)[None, :, None] * lam2[None, None, :]
    excess = np.einsum("npi,npij,npj->n", diff, A, 0.5 * diff - noise)
    gradient = np.einsum("p,npij,npj->ni", st, A, diff - noise)
    gradient[:, 0] -= b_star * gradient[:, 2]
    return excess, gradient


def reference(error, b_star, alpha):
    """Independent adaptive integration of G11, G12, G22, grad^ F and the trap gap at one trap.

    Integrates over y = z* by scipy's adaptive quad, with the moments of X_2 on the half-line of
    the trap gate in closed form: the opposite order to `atoms`, and adaptive rather than fixed.
    """
    e_h = hat(error, b_star)
    p, q, r = np.array([1.0, 0.0, 0.0]) + e_h  # trap gate: p y + q X_2 + r > 0

    def x2_moments(y):
        """Moments of 1, X_2, X_2^2 with the trap gate on and off at this y."""
        whole = np.array([1.0, 0.0, 1.0])
        level = p * y + r
        if q == 0:
            return (whole, 0 * whole) if level > 0 else (0 * whole, whole)
        c = -level / abs(q)
        # sign(q) X_2 > c and < c, each as a Gaussian tail, never as a difference.
        above = np.array([ndtr(-c), phi(c), c * phi(c) + ndtr(-c)])
        below = np.array([ndtr(c), -phi(c), ndtr(c) - c * phi(c)])
        if q < 0:
            above, below = above * np.array([1.0, -1.0, 1.0]), below * np.array([1.0, -1.0, 1.0])
        return above, below

    def second(y, m):
        m0, m1, m2 = m
        return np.array([[y * y * m0, y * m1, y * m0], [y * m1, m2, m1], [y * m0, m1, m0]])

    def integrand(y):
        on, off = x2_moments(y)
        star = 1.0 if y > 0 else alpha
        out = []
        for m, s_hat in ((on, 1.0), (off, alpha)):
            M = second(y, m)
            diff = star * e_h + (s_hat - star) * np.array([p, q, r])
            total = diff + 2 * star * np.array([1.0, 0.0, 0.0])
            out.append(
                np.concatenate(
                    (
                        (star * star * M).ravel(),
                        (star * (s_hat - star) * M).ravel(),
                        ((s_hat - star) ** 2 * M).ravel(),
                        s_hat * M @ diff,
                        [-0.5 * diff @ M @ total],
                    )
                )
            )
        return (out[0] + out[1]) * phi(y - b_star)

    cuts = [b_star - 14.0, 0.0, b_star + 14.0]
    if p != 0:
        center = -r / p
        width = 8 * abs(q) / abs(p)
        cuts.extend((center - width, center, center + width) if q != 0 else (center,))
    cuts = [-np.inf, *sorted({c for c in cuts if b_star - 14 < c < b_star + 14} | {0.0}), np.inf]
    # Each component separately, to relative accuracy: the wedge moments are many orders of
    # magnitude below those of the target gate, and below one another.
    unique = [9 * k + 3 * i + j for k in range(3) for i in range(3) for j in range(i, 3)]
    total = np.zeros(31)
    for c in unique + [27, 28, 29, 30]:
        for low, high in pairwise(cuts):
            total[c] += quad(
                lambda y, c=c: integrand(y)[c],
                low,
                high,
                epsabs=0,
                epsrel=1e-13,
                limit=200,
                full_output=1,
            )[0]
    blocks = total[:27].reshape(3, 3, 3)
    blocks = np.triu(blocks) + np.swapaxes(np.triu(blocks, 1), 1, 2)
    return {
        "G11": blocks[0],
        "G12": blocks[1],
        "G22": blocks[2],
        "gradient_hat": total[27:30],
        "trap_gap": total[30],
    }
