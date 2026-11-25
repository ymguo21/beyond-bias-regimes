"""High-precision population values by direct one-dimensional integration.

This oracle shares no formulas with `gaussian.py`. Condition on the target
coordinate Z=<w*,X>; the candidate preactivation is then Gaussian, and its
truncated moments are exact. The remaining integral over Z uses mpmath
tanh-sinh quadrature with breakpoints at both gates. The working precision
grows with the cancellation expected near the target.
"""

import mpmath as mp


def truncated(mean, sd):
    """E[S^k 1{S>0}] and E[S^k 1{S<=0}] for S~N(mean, sd^2), k=0,1,2."""
    if sd == 0:
        positive = mean > 0
        total = (1, mean, mean * mean)
        return (total, (0, 0, 0)) if positive else ((0, 0, 0), total)
    x = mean / sd
    probability, density = mp.ncdf(x), mp.npdf(x)
    first = mean * probability + sd * density
    second = (mean * mean + sd * sd) * probability + mean * sd * density
    total = (1, mean, mean * mean + sd * sd)
    upper = (probability, first, second)
    return upper, tuple(t - u for t, u in zip(total, upper))


def precise_values(theta, b_star, alpha, digits=40, displacement=None, *, gap_as_float=True):
    """Loss ratio F/F(0), gap zeta-1, and quadrature error estimate of the gap.

    Returns floats by default. Set `gap_as_float=False` to retain the working-precision gap.

    An optional exact displacement (delta, v) defines theta=theta*+delta*v in
    working precision instead of the rounded coordinates `theta`.
    """
    theta = [float(x) for x in theta]
    if displacement is None:
        delta = ((theta[0] - 1) ** 2 + theta[1] ** 2 + (theta[2] - b_star) ** 2) ** 0.5
    else:
        delta = float(displacement[0])
    # Near the target, O(1) conditional moments cancel to O(delta^2).
    lost = max(0, -2 * int(mp.log10(delta)))
    with mp.workdps(digits + lost):
        bs, al = mp.mpf(b_star), mp.mpf(alpha)
        if displacement is None:
            p, q, b = (mp.mpf(x) for x in theta)
        else:
            d, v = mp.mpf(displacement[0]), [mp.mpf(float(x)) for x in displacement[1]]
            p, q, b = 1 + d * v[0], d * v[1], bs + d * v[2]
        q = abs(q)

        def integrand(z):
            t = z + bs
            target = t if t > 0 else al * t
            (u0, u1, u2), (l0, l1, l2) = truncated(p * z + b, q)
            # S>0: residual S-target, slope 1; S<=0: residual alpha*S-target, slope alpha.
            correction = (u2 - (target + t) * u1 + target * t * u0) + al * (
                al * l2 - (al * t + target) * l1 + target * t * l0
            )
            loss2 = (u2 - 2 * target * u1 + target**2 * u0) + (
                al * al * l2 - 2 * al * target * l1 + target**2 * l0
            )
            return mp.mpc(correction, loss2) * mp.npdf(z)

        cuts = {-bs}
        if p != 0:
            center, width = -b / p, q / abs(p)
            cuts |= {center} | {center + s * k * width for s in (-1, 1) for k in (1, 8)}
        points = [-mp.inf, *sorted(cuts), mp.inf]
        value, error = mp.quad(integrand, points, error=True)
        a = -bs
        upper_second = (1 + a * a) * mp.ncdf(-a) - a * mp.npdf(a)
        zero_loss2 = upper_second + al * al * (1 + a * a - upper_second)
        k = 2 / (mp.sqrt(bs * bs + 4) - bs)
        lam = al * al + (1 - al * al) * (mp.ncdf(bs) - mp.npdf(bs) * k)
        distance2 = (p - 1) ** 2 + q * q + (b - bs) ** 2
        gap = (value.real - lam * distance2) / (lam * distance2)
        return (
            float(value.imag / zero_loss2),
            float(gap) if gap_as_float else gap,
            float(error / (lam * distance2)),
        )
