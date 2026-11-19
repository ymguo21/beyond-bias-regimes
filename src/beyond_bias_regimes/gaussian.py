"""Population moments for theta=(w_1,w_2,b), theta*=(1,0,b_star).

Gaussian rotational invariance reduces every candidate/target pair to these
three coordinates. Experiments use analytic truncated moments and two
one-dimensional integrals; near the target they use scaled residuals. The
correction gap zeta-1 is formed without subtracting 1 from a rounded ratio.
The reference oracle integrates conditional residuals and the full gradient
separately. No oracle uses finite differences or sampled training data.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec
from scipy.special import ndtr


def phi(x):
    return np.exp(-(np.asarray(x) ** 2) / 2) / np.sqrt(2 * np.pi)


def kappa(b_star):
    return 2 / (np.sqrt(b_star * b_star + 4) - b_star)


def signal(b_star, alpha):
    return alpha**2 + (1 - alpha**2) * (ndtr(b_star) - phi(b_star) * kappa(b_star))


def target_second(b_star, alpha):
    return alpha**2 * (1 + b_star * b_star) + (1 - alpha**2) * positive_second(-b_star)


def positive_first(threshold):
    return phi(threshold) - threshold * ndtr(-threshold)


def positive_second(threshold):
    return (1 + threshold * threshold) * ndtr(-threshold) - threshold * phi(threshold)


@lru_cache(maxsize=8)
def rule(order):
    return leggauss(order)


def tail_moments(x):
    """Integrals of 1, Z, Z^2 on (x,infinity)."""
    density = phi(x)
    finite_x = np.where(np.isfinite(x), x, 0)
    return np.array([ndtr(-x), density, ndtr(-x) + finite_x * density])


def joint_positive(psi, beta, b_star, order=96):
    """E[U_+ T_+], E[T^2 1{U>0,T>0}], U=cos(beta)V+sin(beta).

    V and Z have correlation cos(psi); T=Z+b_star. Integrate over Z>-b_star.
    Split at the conditional gate and its transition shoulders. The omitted
    Z>14 tail is below 1e-40, at least 30 orders of magnitude below the
    retained moments for b_star in [-6,0].
    Collinear weights and zero weights use exact univariate moments.
    """
    psi, beta = np.broadcast_arrays(np.asarray(psi), np.asarray(beta))
    shape = psi.shape
    psi, beta = psi.ravel(), beta.ravel()
    s, b = np.cos(beta), np.sin(beta)
    s = np.where(np.abs(beta) >= np.pi / 2, 0.0, s)
    p, q = s * np.cos(psi), s * np.sin(psi)
    q = np.where((psi == 0) | (psi == np.pi), 0.0, q)
    product, second = np.zeros_like(p), np.zeros_like(p)
    threshold = -b_star
    line = q == 0
    for index in np.flatnonzero(line):
        if p[index] == 0:
            if b[index] > 0:
                product[index] = b[index] * positive_first(threshold)
                second[index] = positive_second(threshold)
            continue
        root = -b[index] / p[index]
        low = max(threshold, root) if p[index] > 0 else threshold
        high = np.inf if p[index] > 0 else root
        if high <= low:
            continue
        m0, m1, m2 = tail_moments(low) - tail_moments(high)
        product[index] = (
            p[index] * m2 + (b[index] + b_star * p[index]) * m1 + b_star * b[index] * m0
        )
        second[index] = m2 + 2 * b_star * m1 + b_star * b_star * m0

    nodes, weights = rule(order)
    indices = np.flatnonzero(~line)
    for start in range(0, indices.size, 1024):
        ix = indices[start : start + 1024]
        pp, qq, bb = p[ix], q[ix], b[ix]
        root = np.divide(-bb, pp, out=np.full_like(pp, threshold), where=pp != 0)
        width = np.divide(8 * qq, np.abs(pp), out=np.full_like(pp, 28), where=pp != 0)
        cuts = np.sort(
            np.clip(
                np.array(
                    [
                        np.full_like(pp, threshold),
                        root - width,
                        root,
                        root + width,
                        np.full_like(pp, 14.0),
                    ]
                ),
                threshold,
                14,
            ),
            axis=0,
        )
        for low, high in pairwise(cuts):
            half = (high - low) / 2
            z = (high + low)[:, None] / 2 + half[:, None] * nodes
            mean = pp[:, None] * z + bb[:, None]
            x = mean / qq[:, None]
            probability = ndtr(x)
            first = mean * probability + qq[:, None] * phi(x)
            density = phi(z)
            product[ix] += half * (((z + b_star) * first * density) @ weights)
            second[ix] += half * (((z + b_star) ** 2 * probability * density) @ weights)
    return product.reshape(shape), second.reshape(shape)


# Squared distances below this multiple of ||theta*||^2 use the scaled-residual formula.
NEAR_TARGET = 0.01
# Point evaluations also use it whenever the first candidate weight is at least this.
SCALED_MIN_WEIGHT = 0.5


@dataclass
class Geometry:
    """Unit affine directions u=(cos(beta)cos(psi),cos(beta)sin(psi),sin(beta))."""

    psi: np.ndarray
    beta: np.ndarray
    b_star: float
    product: np.ndarray
    gated_second: np.ndarray

    @classmethod
    def build(cls, psi, beta, b_star, order=96):
        psi, beta = np.broadcast_arrays(np.asarray(psi, float), np.asarray(beta, float))
        return cls(psi, beta, b_star, *joint_positive(psi, beta, b_star, order))

    @property
    def directions(self):
        s = np.where(np.abs(self.beta) >= np.pi / 2, 0.0, np.cos(self.beta))
        q = np.where((self.psi == 0) | (self.psi == np.pi), 0.0, s * np.sin(self.psi))
        return np.stack((s * np.cos(self.psi), q, np.sin(self.beta)), axis=-1)

    def coefficients(self, alpha):
        b_star = self.b_star
        s = np.where(np.abs(self.beta) >= np.pi / 2, 0.0, np.cos(self.beta))
        b, rho = np.sin(self.beta), np.cos(self.psi)
        nu = np.divide(-b, s, out=np.where(b > 0, -np.inf, np.inf), where=s != 0)
        probability, density = ndtr(-nu), phi(nu)
        first = s * density + b * probability
        second = probability + s * b * density  # s^2+b^2=1
        gated_target = (1 + b_star * b_star) * probability + (
            rho**2 * np.where(np.isfinite(nu), nu, 0) + 2 * b_star * rho
        ) * density
        inner = s * rho + b_star * b
        cross1 = s * rho * ndtr(b_star) + b * positive_first(-b_star)
        cross2 = s * rho * probability + b_star * first
        A = alpha**2 + (1 - alpha**2) * second
        B = (
            alpha**2 * inner
            + alpha * (1 - alpha) * (cross1 + cross2)
            + (1 - alpha) ** 2 * self.product
        )
        H = alpha**2 * inner + (1 - alpha**2) * cross2
        J = (
            alpha**2 * (1 + b_star * b_star)
            + alpha * (1 - alpha) * (gated_target + positive_second(-b_star))
            + (1 - alpha) ** 2 * self.gated_second
        )
        return Coefficients(self.directions, b_star, alpha, A, B, H, J, inner)


@dataclass
class Coefficients:
    """Along theta=t*u: 2F=At^2-2Bt+M, <theta-theta*,grad F>=At^2-(B+H)t+J."""

    directions: np.ndarray
    b_star: float
    alpha: float
    A: np.ndarray
    B: np.ndarray
    H: np.ndarray
    J: np.ndarray
    inner: np.ndarray

    def take(self, index):
        """The coefficients of a subset of directions."""
        arrays = (self.A, self.B, self.H, self.J, self.inner)
        return Coefficients(
            self.directions[index], self.b_star, self.alpha, *(x[index] for x in arrays)
        )

    def loss_interval(self, C):
        """Scales s on each ray with F(s u) <= C F(0), as (s_-, s_+); NaN where there are none."""
        M = target_second(self.b_star, self.alpha)
        discriminant = self.B**2 - (1 - C) * self.A * M
        valid = (self.B > 0) & (discriminant >= 0)
        root = np.sqrt(np.maximum(discriminant, 0))
        high = np.where(valid, (self.B + root) / self.A, np.nan)
        low = np.divide((1 - C) * M, self.B + root, out=np.full_like(high, np.nan), where=valid)
        return low, high

    def margin(self, level=1.0):
        """Coefficients of <theta-theta*,grad F> - level*Lambda*||theta-theta*||^2 in t."""
        lam = level * signal(self.b_star, self.alpha)
        return (
            self.A - lam,
            -(self.B + self.H) + 2 * lam * self.inner,
            self.J - lam * (1 + self.b_star**2),
        )

    def evaluate(self, scales, order=96):
        """Arrays are (direction, candidate scale); return loss, gap zeta-1, squared distance."""
        t = np.asarray(scales, float)
        M = target_second(self.b_star, self.alpha)
        loss = 0.5 * ((self.A[:, None] * t - 2 * self.B[:, None]) * t + M)
        c2, c1, c0 = (x[:, None] for x in self.margin())
        theta = t[..., None] * self.directions[:, None, :]
        distance2 = np.sum((theta - np.array([1.0, 0.0, self.b_star])) ** 2, axis=-1)
        denominator = signal(self.b_star, self.alpha) * distance2
        gap = np.divide(
            (c2 * t + c1) * t + c0, denominator, out=np.full_like(t, np.nan), where=distance2 > 0
        )
        # Change the evaluation formula near the target; never remove these points.
        near = (distance2 > 0) & (distance2 < NEAR_TARGET * (1 + self.b_star**2))
        if np.any(near):
            error = theta[near] - np.array([1.0, 0.0, self.b_star])
            delta = np.sqrt(distance2[near])
            loss[near], gap[near] = local_moments(
                delta, error / delta[:, None], self.b_star, self.alpha, order
            )
        return loss, gap, distance2

    def stationary_scales(self):
        """All stationary points of the quadratic-over-quadratic correction quotient."""
        n1 = -(self.B + self.H)
        c2 = -2 * self.A * self.inner - n1
        c1 = 2 * (self.A * (1 + self.b_star**2) - self.J)
        c0 = n1 * (1 + self.b_star**2) + 2 * self.inner * self.J
        disc = c1**2 - 4 * c2 * c0
        root = np.sqrt(np.maximum(disc, 0))
        stable = -0.5 * (c1 + np.copysign(root, c1))
        x1 = np.divide(stable, c2, out=np.full_like(c2, np.nan), where=c2 != 0)
        x2 = np.divide(c0, stable, out=np.full_like(c2, np.nan), where=stable != 0)
        linear = (c2 == 0) & (c1 != 0)
        x1[linear] = -c0[linear] / c1[linear]
        return np.stack((np.where(disc >= 0, x1, np.nan), np.where(disc >= 0, x2, np.nan)), axis=-1)


def hessian_frame(b_star):
    """Orthonormal target-Hessian eigenvectors: weakest v_min, in-plane, perpendicular."""
    k = kappa(b_star)
    n = np.sqrt(1 + k * k)
    return np.array([-k, 0.0, 1.0]) / n, np.array([1.0, 0.0, k]) / n, np.array([0.0, 1.0, 0.0])


def hessian_gap(v, b_star, alpha):
    """v^T(Hessian - Lambda)v for unit v, as a sum of nonnegative eigen-components.

    Eigenvalue excesses: phi(b*)(kappa+1/kappa) in-plane and phi(b*)kappa
    perpendicular, both times (1-alpha^2). See docs/theory.md.
    """
    k = kappa(b_star)
    _, in_plane, perpendicular = hessian_frame(b_star)
    scale = (1 - alpha**2) * phi(b_star)
    return scale * ((k + 1 / k) * (v @ in_plane) ** 2 + k * (v @ perpendicular) ** 2)


def local_moments(delta, v, b_star, alpha, order=96):
    """Stable loss and gap zeta-1 at theta=theta*+delta*v, delta>0, ||v||=1.

    The point is defined by (delta, v), not by rounded coordinates, so tiny
    displacements keep their exact direction. With Z=-b_star+delta*u, the
    target Hessian contributes analytically; only the gate-switching
    correction is integrated. All residuals are divided by delta before
    integration, avoiding subtraction of O(1) moments to recover O(delta^2)
    quantities. See docs/theory.md.
    """
    delta, v = np.asarray(delta, float), np.asarray(v, float)
    first_weight = 1 + delta * v[:, 0]
    e0, e1, e2 = v.T
    e1 = abs(e1)
    norm2 = np.sum(v**2, axis=-1)
    base = alpha**2 * norm2 + (1 - alpha**2) * (
        ndtr(b_star) * norm2 + phi(b_star) * (-b_star * e0**2 + 2 * e0 * e2)
    )
    q_gap, loss_scaled = hessian_gap(v, b_star, alpha), base / 2
    nodes, weights = rule(order)
    for start in range(0, len(delta), 512):
        sl = slice(start, start + 512)
        d, v0, v1, v2 = delta[sl], e0[sl], e1[sl], e2[sl]
        p = first_weight[sl]
        center = (b_star * v0 - v2) / p
        width = 12 * v1 / p
        cuts = np.sort(
            np.stack((np.zeros_like(center), center - width, center, center + width)), axis=0
        )
        for low, high in pairwise(cuts):
            half = (high - low) / 2
            u = (high + low)[:, None] / 2 + half[:, None] * nodes
            z = -b_star + d[:, None] * u
            mean = v0[:, None] * z + v2[:, None]
            candidate = u + mean
            x = np.divide(candidate, v1[:, None], out=np.zeros_like(u), where=v1[:, None] != 0)
            density = np.where(v1[:, None] != 0, phi(x), 0.0)
            positive = np.where(v1[:, None] != 0, ndtr(x), candidate > 0)
            negative = np.where(v1[:, None] != 0, ndtr(-x), candidate <= 0)
            first_pos = mean * positive + v1[:, None] * density
            first_neg = mean * negative - v1[:, None] * density
            second = mean**2 + v1[:, None] ** 2
            extra = v1[:, None] * (mean - u) * density
            second_pos, second_neg = second * positive + extra, second * negative - extra
            qchange = np.where(
                u < 0,
                (1 - alpha**2) * second_pos + (1 - alpha) * u * first_pos,
                -(1 - alpha**2) * second_neg - alpha * (1 - alpha) * u * first_neg,
            )
            fchange = 0.5 * np.where(
                u < 0,
                (1 - alpha**2) * second_pos
                + 2 * (1 - alpha) * u * first_pos
                + (1 - alpha) ** 2 * u**2 * positive,
                -(1 - alpha**2) * second_neg
                - 2 * alpha * (1 - alpha) * u * first_neg
                + (1 - alpha) ** 2 * u**2 * negative,
            )
            measure = d * half
            q_gap[sl] += measure * ((qchange * phi(z)) @ weights)
            loss_scaled[sl] += measure * ((fchange * phi(z)) @ weights)
    return delta**2 * loss_scaled, q_gap / signal(b_star, alpha)


def point_values(theta, b_star, alpha, order=96):
    """Loss ratio F/F(0), gap zeta-1, relative distance, and bias slack at arbitrary points.

    Rows are (w_parallel, w_perp, b); w_perp enters only through its magnitude.
    """
    theta = np.atleast_2d(np.asarray(theta, float)).copy()
    theta[:, 1] = np.abs(theta[:, 1])
    error = theta - np.array([1.0, 0.0, b_star])
    delta = np.linalg.norm(error, axis=1)
    return displacement_values(delta, error / delta[:, None], b_star, alpha, order, theta)


def displacement_values(delta, v, b_star, alpha, order=96, theta=None):
    """As `point_values`, for exact points theta*+delta*v (v with v[:, 1] >= 0).

    `scaled` marks the scaled-residual formula; other points use the moment
    polynomials, whose gap carries an absolute rounding error near 1e-14.
    """
    delta, v = np.asarray(delta, float), np.asarray(v, float)
    if theta is None:
        theta = np.array([1.0, 0.0, b_star]) + delta[:, None] * v
    norm2 = 1 + b_star**2
    near = (delta**2 < NEAR_TARGET * norm2) | (1 + delta * v[:, 0] >= SCALED_MIN_WEIGHT)
    loss, gap = np.empty(len(delta)), np.empty(len(delta))
    if near.any():
        loss[near], gap[near] = local_moments(delta[near], v[near], b_star, alpha, order)
    if (~near).any():
        far = theta[~near]
        weight = np.hypot(far[:, 0], far[:, 1])
        geometry = Geometry.build(
            np.arctan2(far[:, 1], far[:, 0]), np.arctan2(far[:, 2], weight), b_star, order
        )
        far_loss, far_gap, _ = geometry.coefficients(alpha).evaluate(
            np.linalg.norm(far, axis=1)[:, None], order
        )
        loss[~near], gap[~near] = far_loss[:, 0], far_gap[:, 0]
    # b-b*||w|| = delta*v2 - b*(||w||-1), with ||w||-1 formed without cancellation.
    weight = np.hypot(1 + delta * v[:, 0], delta * v[:, 1])
    weight_change = delta * (2 * v[:, 0] + delta * (v[:, 0] ** 2 + v[:, 1] ** 2)) / (weight + 1)
    return {
        "loss_ratio": 2 * loss / target_second(b_star, alpha),
        "gap": gap,
        "distance": delta / np.sqrt(norm2),
        "bias_slack": delta * v[:, 2] - b_star * weight_change,
        "scaled": near,
    }


def reference_moments(theta, b_star, alpha, displacement=None):
    """Independent adaptive conditional integration of loss and the full gradient.

    Returns (loss, gradient, quadrature error estimate). The integration domain
    is the full real line, including both Gaussian tails. An optional exact
    displacement (delta, v) defines theta=theta*+delta*v without rounding.
    """
    theta = np.asarray(theta, float)
    p, q, b = theta
    if displacement is None:
        error_direction = theta - np.array([1.0, 0.0, b_star])
        delta = np.linalg.norm(error_direction)
        if delta == 0:
            return 0.0, np.zeros(3), 0.0
        e0, e1, e2 = error_direction / delta
    else:
        delta, (e0, e1, e2) = displacement
    qsign = 1 if q >= 0 else -1
    q = abs(q)

    def integrand(z):
        T = z + b_star
        target_scaled = T / delta
        difference = e0 * z + e2
        S = p * z + b
        if q == 0:
            st, sc = (1 if T > 0 else alpha), (1 if S > 0 else alpha)
            residual = sc * difference + (sc - st) * target_scaled
            return np.array([0.5 * residual**2, residual * sc * z, 0, residual * sc]) * phi(z)
        positive, negative, density = ndtr(S / q), ndtr(-S / q), phi(S / q)
        perpendicular_error = abs(e1)
        dpos = difference * positive + perpendicular_error * density
        dneg = difference * negative - perpendicular_error * density
        dsecond = difference**2 + perpendicular_error**2
        extra = perpendicular_error * (difference - target_scaled) * density
        d2pos, d2neg = dsecond * positive + extra, dsecond * negative - extra
        common = d2pos + alpha**2 * d2neg
        mean_gradient = alpha**2 * difference + (1 - alpha**2) * dpos
        if T > 0:
            loss2 = (
                common
                - 2 * alpha * (1 - alpha) * target_scaled * dneg
                + (1 - alpha) ** 2 * target_scaled**2 * negative
            )
            mean_gradient -= alpha * (1 - alpha) * target_scaled * negative
            target = target_scaled
        else:
            loss2 = (
                common
                + 2 * (1 - alpha) * target_scaled * dpos
                + (1 - alpha) ** 2 * target_scaled**2 * positive
            )
            mean_gradient += (1 - alpha) * target_scaled * positive
            target = alpha * target_scaled
        perpendicular = (
            perpendicular_error * (alpha**2 + (1 - alpha**2) * positive)
            - target * (1 - alpha) * density
        )
        return np.array(
            [0.5 * loss2, z * mean_gradient, qsign * perpendicular, mean_gradient]
        ) * phi(z)

    cuts = [-14.0, -b_star, 14.0]
    if p != 0:
        center, width = -b / p, 8 * q / abs(p)
        cuts.extend(x for x in (center - width, center, center + width) if -14 < x < 14)
    cuts = [-np.inf, *sorted(set(cuts)), np.inf]
    value, error = np.zeros(4), 0.0
    for low, high in pairwise(cuts):
        part, estimate = quad_vec(integrand, low, high, epsabs=2e-16, epsrel=2e-12)
        value += part
        error += estimate
    return delta**2 * value[0], delta * value[1:], delta * error
