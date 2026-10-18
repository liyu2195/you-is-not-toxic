"""Confidence intervals used by THREADSAFE.

Unit shares: Wald, switching to Korn-Graubard (effective-n Clopper-Pearson) when the effective number of positive or
negative units is below 10. Contrasts and pair metrics: Wald or Newcombe score intervals built from Wilson limits;
the risk ratio uses MOVER-R. All intervals are two-sided at 95%.
"""
import math
import numpy as np
from scipy.stats import beta as _beta

Z95 = 1.96
MIN_COUNT = 10


def effective_n(est, se, n):
    """Clipped estimate and the sample size at which a binomial proportion would have this standard error (at most n)."""
    p = min(max(float(est), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return p, float(n)
    return p, min(float(n), p * (1 - p) / max(float(se) ** 2, 1e-15))


def korn_graubard(est, se, n):
    """Clopper-Pearson interval at the effective sample size (Korn and Graubard, 1998)."""
    p, ne = effective_n(est, se, n)
    x = ne * p
    lo = float(_beta.ppf(0.025, max(x, 1e-12), ne - x + 1)) if x > 0 else 0.0
    hi = float(_beta.ppf(0.975, x + 1, max(ne - x, 1e-12))) if x < ne else 1.0
    return lo, hi


def unit_share_interval(est, se, n, z=Z95, min_count=MIN_COUNT, exact=False):
    """Interval for a share of units. Returns (lo, hi, kind) with kind in {"wald", "kg", "census"}."""
    if exact:
        return float(est), float(est), "census"
    p, ne = effective_n(est, se, n)
    if min(ne * p, ne * (1 - p)) < min_count:
        lo, hi = korn_graubard(est, se, n)
        return lo, hi, "kg"
    return float(est) - z * float(se), float(est) + z * float(se), "wald"


def wilson(p, n, z=Z95):
    """Wilson score limits for a proportion p observed on n trials."""
    if n <= 0:
        return 0.0, 1.0
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return c - h, c + h


def newcombe(p1, n1, p0, n0, z=Z95, clip=False):
    """Newcombe's hybrid score interval for p1 - p0 from the Wilson limits of each proportion."""
    l1, u1 = wilson(p1, n1, z)
    l0, u0 = wilson(p0, n0, z)
    if clip:
        l1, u1, l0, u0 = max(0.0, l1), min(1.0, u1), max(0.0, l0), min(1.0, u0)
    d = p1 - p0
    return d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def mover_rr(p1, l1, u1, p0, l0, u0):
    """MOVER-R interval for the ratio p1 / p0 (Donner and Zou, 2012), from each proportion's limits."""
    try:
        lo = (p1 * p0 - math.sqrt(max((p1 * p0) ** 2 - l1 * u0 * (2 * p1 - l1) * (2 * p0 - u0), 0))) / (u0 * (2 * p0 - u0))
        hi = (p1 * p0 + math.sqrt(max((p1 * p0) ** 2 - u1 * l0 * (2 * p1 - u1) * (2 * p0 - l0), 0))) / (l0 * (2 * p0 - l0)) if l0 > 0 else math.inf
    except ZeroDivisionError:
        lo, hi = 0.0, math.inf
    return lo, hi


def contrast_interval(e1, v1, n1, e0, v0, n0, z=Z95, min_count=MIN_COUNT):
    """Interval for a difference of two stratum means with variances v1, v0 from samples of n1, n0.
    Wald, or Newcombe on effective-n Wilson limits when an effective count is below min_count. Returns (lo, hi, kind)."""
    eff = []
    for e, v, n in [(e1, v1, n1), (e0, v0, n0)]:
        p = min(max(e, 0.0), 1.0)
        ne = n if p in (0.0, 1.0) or v <= 0 else min(n, p * (1 - p) / v)
        eff.append((p, ne))
    (p1, m1), (p0, m0) = eff
    if min(m1 * p1, m1 * (1 - p1), m0 * p0, m0 * (1 - p0)) >= min_count:
        d, se = e1 - e0, math.sqrt(v1 + v0)
        return d - z * se, d + z * se, "wald"
    lo, hi = newcombe(p1, m1, p0, m0, z)
    return lo, hi, "newcombe"
