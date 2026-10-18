"""THREADSAFE, Algorithm 2: a structural contrast, such as the positive share at depth >= 2 minus the share at thread starts.

Validation is stratified by structure: half the labels go to each stratum, drawn uniformly without replacement, and each
stratum's share is a PPI++ mean with the LLM flag as predictor. Calibrating the LLM on a pooled sample instead leaves
the bias E[P(Z = 1 | s) | stratum] - P(Z = 1 | stratum), which is how a nonexistent escalation gets announced.
"""
import numpy as np
from .intervals import contrast_interval, Z95


def stratum_mean(items, flag, oracle, n, rng):
    """PPI++ share of gold positives in one stratum from n uniformly sampled items. Returns (estimate, variance)."""
    ix = rng.choice(items, n, replace=False)
    z = np.array([oracle(i) for i in ix], dtype=float)
    f = flag[ix].astype(float)
    Np = len(items)
    lam = np.cov(z, f)[0, 1] / (np.var(f, ddof=1) * (1 + n / Np)) if np.var(f) > 0 else 0.0
    return float(lam * flag[items].mean() + (z - lam * f).mean()), float(np.var(z - lam * f, ddof=1) / n)


def run(stratum1, stratum0, flag, oracle, budget, rng=None, z=Z95):
    """Contrast mean(stratum1) - mean(stratum0) with `budget` gold labels, half in each stratum.
    stratum1, stratum0: arrays of item indices; flag: LLM flag for every item. Returns (estimate, lo, hi, kind)."""
    rng = np.random.default_rng() if rng is None else rng
    n1, n0 = budget // 2, budget - budget // 2
    e1, v1 = stratum_mean(stratum1, flag, oracle, n1, rng)
    e0, v0 = stratum_mean(stratum0, flag, oracle, n0, rng)
    lo, hi, kind = contrast_interval(e1, v1, n1, e0, v0, n0, z)
    return e1 - e0, lo, hi, kind
