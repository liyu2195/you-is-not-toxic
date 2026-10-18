"""Addendum 80: shared helpers for B80a-B80d (Civil Comments reply threads and the 5% sample).
Laws and placebo statistics with article-cluster Poisson bootstrap weights, computed from per-cluster sufficient statistics so that
every instrument and benchmark is evaluated on the same resamples (paired):
  depth law       OLS slope of the labels on x = min(depth, 10)
  placebo depth   OLS slope of the flags on x among gold negatives, divided by the FPR (their mean flag rate)
  log RR          log reply risk ratio over reply edges (parent -> child)
  placebo log RR  the same over edges whose two ends are gold-negative
"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import numpy as np, pandas as pd
W = _ROOT


def top_share(score, k, seed):
    """Flag the k comments with the highest score, ties broken at random (seeded)."""
    N = len(score); rng = np.random.default_rng(seed); s = np.where(np.isnan(score), -np.inf, score)
    order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0
    return f


class Pop:
    def __init__(self, name, nb=500, seed=8001, extra_cols=()):
        P = pd.read_parquet(f"{W}/results/x14_pops/{name}.parquet")
        self.P = P; self.N = len(P); self.z = P.z.values.astype(float); self.tox = P.tox.values.astype(float)
        self.cl = pd.factorize(P.cluster.values)[0]; self.nC = int(self.cl.max() + 1)
        self.x = np.minimum(P.depth.values, 10).astype(float)
        ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch
        self.ch, self.pa = ch[keep], pa[keep]; self.ecl = self.cl[self.ch]
        self.Wb = np.vstack([np.ones(self.nC), np.random.default_rng(seed).poisson(1.0, size=(nb, self.nC))])
        self.neg = self.z == 0; self.negboth = (self.z[self.pa] == 0) & (self.z[self.ch] == 0)

    def bc(self, w, mask=None):
        if mask is None:
            return self.Wb @ np.bincount(self.cl, w, self.nC)
        return self.Wb @ np.bincount(self.cl[mask], w[mask], self.nC)

    def ols(self, v, mask=None):
        m = np.ones(self.N, bool) if mask is None else mask
        n, sx, sy, sxx, sxy = (self.bc(a, m) for a in (np.ones(self.N), self.x, v, self.x * self.x, self.x * v))
        return (n * sxy - sx * sy) / (n * sxx - sx ** 2), sy / n

    def depth(self, v):
        return self.ols(v)[0]

    def placebo_depth_rel(self, v):
        s, m = self.ols(v, self.neg)
        with np.errstate(divide="ignore", invalid="ignore"):
            return s / m

    def lrr(self, v, emask=None, w_edge=None):
        vp, vc = v[self.pa], v[self.ch]; mk = np.ones(len(self.ch), bool) if emask is None else emask
        we = np.ones(len(self.ch)) if w_edge is None else w_edge
        def s(a):
            return self.Wb @ np.bincount(self.ecl[mk], (a * we)[mk], self.nC)
        p1, p1c1, p0, p0c1 = s(vp), s(vp * vc), s(1 - vp), s((1 - vp) * vc)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((p1c1 / p1) / (p0c1 / p0))

    def placebo_lrr(self, v):
        return self.lrr(v, self.negboth)


def ci(a):
    a = np.asarray(a, float)
    return [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]


def instruments(pop, P, with_fresh=True):
    """The audited instruments of the evidence section on a population: every y: column, plus the fresh classifiers at 0.5."""
    L = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    if with_fresh:
        for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
            if f"s:{m}@m" in P:
                L[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
    return L
