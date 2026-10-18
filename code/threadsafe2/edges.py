"""THREADSAFE upgrade (Addendum 57), edge regime: the reply risk ratio of gold labels and the labeler's placebo risk ratio.

Per reply edge e = (parent p, child c), with gold z and flags y:
  gold RR     A/P / (C/Q),   A = sum z_p z_c, P = sum z_p, C = sum (1 - z_p) z_c, Q = sum (1 - z_p)
  placebo RR  a/b / (c/d) on edges with both ends gold-negative, w = (1 - z_p)(1 - z_c):
              a = sum w y_p y_c, b = sum w y_p, c = sum w (1 - y_p) y_c, d = sum w (1 - y_p)
Both are estimated on the log scale from Horvitz-Thompson totals, with linearized variance. An edge costs the gold labels of
its ends not yet revealed in the audit.

Designs: 'srs' (uniform edges, THREADSAFE's current pair regime), 'ppi' (uniform edges, PPI++ with the flags' version of each
contribution as the control variate), 'flagprop' (strata (y_p, y_c), proportional), 'upgrade' (strata (y_p, y_c), a 20% pilot
split equally, then a top-up toward the compromise allocation over the two log risk ratios).
"""
import os
import numpy as np
# Sensitivity switch (after the confirmation): TS_PPI_FPC=1 tunes PPI++ for a finite population with exactly known totals,
# lambda = Cov / Var, instead of the iid-unlabeled formula Cov / ((1 + n/N) Var) used in the registered runs.
PPI_TUNE = 0.0 if os.environ.get("TS_PPI_FPC") == "1" else 1.0
from .laws import equal_split, round_alloc, topup, prop_alloc, Z95

SIGN = np.array([1.0, -1.0, -1.0, 1.0])
METRICS = ("log_rr", "log_placebo_rr")


def contributions(zp, zc, yp, yc):
    """Per-edge contributions to the 8 totals: gold A, P, C, Q, then placebo a, b, c, d."""
    zp, zc, yp, yc = (np.asarray(v, float) for v in (zp, zc, yp, yc))
    w = (1 - zp) * (1 - zc)
    return np.column_stack([zp * zc, zp, (1 - zp) * zc, 1 - zp, w * yp * yc, w * yp, w * (1 - yp) * yc, w * (1 - yp)])


def log_metrics(T):
    """(log gold RR, log placebo RR) from the 8 totals; nan when a total is not positive."""
    g, q = T[:4], T[4:]
    lg = float(SIGN @ np.log(g)) if (g > 0).all() else np.nan
    lq = float(SIGN @ np.log(q)) if (q > 0).all() else np.nan
    return lg, lq


def prop_parts(Hs, Ns, T):
    """Estimates and stratified variances of the four conditional proportions behind the two risk ratios:
    gold p1 = A/P, p0 = C/Q; placebo p1 = a/b, p0 = c/d. Hs: per-stratum contribution matrices, Ns: stratum sizes."""
    out = []
    for num, den in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        p = T[num] / T[den] if T[den] > 0 else np.nan; v = 0.0; nd = 0
        for H, Nh in zip(Hs, Ns):
            n = len(H); nd += int((H[:, den] > 0).sum())
            if n >= 2 and T[den] > 0:
                u = (H[:, num] - p * H[:, den]) / T[den]; v += Nh ** 2 * (1 - n / Nh) * np.var(u, ddof=1) / n
        out.append((p, v, nd))
    return out


def influence(H, T):
    """Linearized per-edge variables of the two log risk ratios, rows of H = contributions."""
    g, q = np.maximum(T[:4], 1e-300), np.maximum(T[4:], 1e-300)
    return H[:, :4] @ (SIGN / g), H[:, 4:] @ (SIGN / q)


class EdgePop:
    def __init__(self, parent, child, y):
        self.p, self.c = np.asarray(parent, np.int64), np.asarray(child, np.int64)
        y = np.nan_to_num(np.asarray(y, float))
        self.yp, self.yc = y[self.p], y[self.c]
        self.E = len(self.p)
        sid = (2 * self.yp + self.yc).astype(int)
        self.ids, inv = np.unique(sid, return_inverse=True)
        order = np.argsort(inv, kind="stable"); self.members = np.split(order, np.cumsum(np.bincount(inv, minlength=len(self.ids)))[:-1])
        self.N = np.array([len(m) for m in self.members], float)
        self.Hy = contributions(self.yp, self.yc, self.yp, self.yc)      # the flags' own version: known control variates
        self.Hy_tot = self.Hy.sum(0)

    def truth(self, z):
        return log_metrics(contributions(z[self.p], z[self.c], self.yp, self.yc).sum(0))

    def labeler_log_rr(self):
        return log_metrics(self.Hy_tot)[0]


def _cost_take(pop, order, budget, known):
    """Walk edges in the given order, buying the unrevealed ends, until the label budget would be exceeded."""
    taken = []; used = 0
    for e in order:
        need = (pop.p[e] not in known) + (pop.c[e] not in known)
        if used + need > budget:
            break
        used += need; known.add(pop.p[e]); known.add(pop.c[e]); taken.append(e)
    return np.array(taken, dtype=np.int64), used


def strat_estimate(pop, samples, z):
    H = [contributions(z[pop.p[s]], z[pop.c[s]], pop.yp[s], pop.yc[s]) for s in samples]
    T = sum(pop.N[h] / len(s) * H[h].sum(0) for h, s in enumerate(samples) if len(s))
    lg, lq = log_metrics(T)
    out = {"_props": prop_parts([H[h] for h, s in enumerate(samples) if len(s)], [pop.N[h] for h, s in enumerate(samples) if len(s)], T)}
    for j, (name, th) in enumerate(zip(METRICS, (lg, lq))):
        v = 0.0
        for h, s in enumerate(samples):
            n = len(s)
            if n >= 2:
                u = influence(H[h], T)[j]; v += pop.N[h] ** 2 * (1 - n / pop.N[h]) * np.var(u, ddof=1) / n
        out[name] = (th, v)
    return out


def _labels_used(pop, samples):
    ends = [np.concatenate([pop.p[s], pop.c[s]]) for s in samples if len(s)]
    return int(len(np.unique(np.concatenate(ends)))) if ends else 0


def fill_budget(pop, perms, alloc, budget):
    """Addendum 60: scale a stratified edge allocation by a common factor (capped by the drawn prefixes) until the gold labels it
    needs (unrevealed ends) reach the budget, so that stratified designs spend the budget as uniform sampling does."""
    cap = np.array([len(p) for p in perms]); alloc = np.asarray(alloc, float)
    def used(f):
        a = np.minimum(np.round(alloc * f).astype(int), cap); return _labels_used(pop, [p[:k] for p, k in zip(perms, a)]), a
    best = np.minimum(alloc.astype(int), cap); lo, hi = 1.0, 2.0
    while used(hi)[0] <= budget and (np.minimum(np.round(alloc * hi), cap) < cap).any() and hi < 1024:
        hi *= 2
    for _ in range(16):
        mid = (lo + hi) / 2; u, a = used(mid)
        if u <= budget:
            lo, best = mid, a
        else:
            hi = mid
    return best


def run(pop, z, budget, rng, design="upgrade", pilot_share=0.2, mult=None, fill=False):
    """One audit with `budget` gold labels. Returns {metric: (log estimate, variance)} plus labels used and edges."""
    if design in ("srs", "ppi"):
        order = rng.permutation(pop.E)[: budget + 10]
        s, used = _cost_take(pop, order, budget, set())
        H = contributions(z[pop.p[s]], z[pop.c[s]], pop.yp[s], pop.yc[s]); n = len(s); E = pop.E
        if design == "srs":
            T = E / n * H.sum(0); out = {"_props": prop_parts([H], [E], T)}
            for j, (name, th) in enumerate(zip(METRICS, log_metrics(T))):
                u = influence(H, T)[j]; out[name] = (th, E ** 2 * (1 - n / E) * np.var(u, ddof=1) / n)
        else:
            T0 = E / n * H.sum(0); Hy = pop.Hy[s]; out = {"_props": prop_parts([H], [E], T0)}
            for j, name in enumerate(METRICS):
                uz, uy = influence(H, T0)[j], influence(Hy, T0)[j]
                vy = np.var(uy, ddof=1); lam = float(np.cov(uz, uy)[0, 1] / (vy * (1 + PPI_TUNE * n / E))) if vy > 0 else 0.0
                T = lam * pop.Hy_tot + E / n * (H - lam * Hy).sum(0)
                th = log_metrics(T)[j]; r = influence(H, T)[j] - lam * influence(Hy, T)[j]
                out[name] = (th, E ** 2 * (1 - n / E) * np.var(r, ddof=1) / n)
        out["_labels"], out["_edges"] = int(used), int(n)
        return out
    n_edges = budget // 2
    psize = budget if fill else n_edges                                    # fill=False keeps the frozen confirmation's draws
    perms = [m[rng.choice(len(m), min(len(m), psize), replace=False)] if len(m) else m for m in pop.members]
    N = pop.N
    if design == "flagprop":
        alloc = prop_alloc(N, n_edges)
    else:
        n_p = int(min(max(round(pilot_share * n_edges), 2 * len(N)), n_edges))
        npil = equal_split(N, n_p); pil = [p[:k] for p, k in zip(perms, npil)]
        Hp = [contributions(z[pop.p[s]], z[pop.c[s]], pop.yp[s], pop.yc[s]) for s in pil]
        Tp = sum(N[h] / max(len(s), 1) * Hp[h].sum(0) for h, s in enumerate(pil))
        Tp = np.maximum(Tp, 0.5 * N.sum() / max(n_p, 1))                  # floor: half a pilot observation's weight
        S2 = np.zeros((2, len(N)))
        for h, s in enumerate(pil):
            # model variance per stratum: smoothed joint law of (z_p, z_c) from the pilot, exact over its 4 cells
            cnt = np.full(4, 0.5)
            if len(s):
                np.add.at(cnt, (2 * z[pop.p[s]] + z[pop.c[s]]).astype(int), 1)
            pr = cnt / cnt.sum(); cells = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
            yp = pop.yp[pop.members[h][0]] if len(pop.members[h]) else 0.0; yc = pop.yc[pop.members[h][0]] if len(pop.members[h]) else 0.0
            Hc = contributions(cells[:, 0], cells[:, 1], np.full(4, yp), np.full(4, yc))
            for j in range(2):
                u = influence(Hc, Tp)[j]; S2[j, h] = max(pr @ u ** 2 - (pr @ u) ** 2, 0.0)
        w = 1.0 / np.maximum((np.sqrt(S2) * N).sum(1) ** 2, 1e-300)
        if mult is not None:
            w = w * np.asarray(mult, float)
        sig = np.sqrt((w[:, None] * S2).sum(0)); target = n_edges * N * sig / max((N * sig).sum(), 1e-300)
        alloc = npil + topup(target, npil.astype(float), n_edges, N)
    if fill:
        alloc = fill_budget(pop, perms, alloc, budget)
    samples = [p[:k] for p, k in zip(perms, alloc)]
    out = strat_estimate(pop, samples, z)
    ends = np.concatenate([np.concatenate([pop.p[s], pop.c[s]]) for s in samples]) if samples else np.empty(0)
    out["_labels"], out["_edges"] = int(len(np.unique(ends))), int(sum(len(s) for s in samples))
    return out


def log_wald(th, v, z=Z95):
    """Interval for the risk ratio from a log-scale estimate; (0, inf) when the estimate is undefined."""
    if not np.isfinite(th) or not np.isfinite(v) or v < 0:
        return 0.0, np.inf
    se = np.sqrt(v); return float(np.exp(th - z * se)), float(np.exp(th + z * se))


def wilson(p, n, z=Z95):
    if not (n > 0) or not np.isfinite(p):
        return 0.0, 1.0
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return max(c - h, 0.0), min(c + h, 1.0)


def mover_rr(p1, l1, u1, p0, l0, u0):
    """MOVER-R interval for p1 / p0 (Donner and Zou 2012) from each proportion's limits."""
    try:
        lo = (p1 * p0 - np.sqrt(max((p1 * p0) ** 2 - l1 * u0 * (2 * p1 - l1) * (2 * p0 - u0), 0))) / (u0 * (2 * p0 - u0))
        hi = (p1 * p0 + np.sqrt(max((p1 * p0) ** 2 - u1 * l0 * (2 * p1 - u1) * (2 * p0 - l0), 0))) / (l0 * (2 * p0 - l0)) if l0 > 0 else np.inf
    except ZeroDivisionError:
        lo, hi = 0.0, np.inf
    return float(max(lo, 0.0)), float(hi)


def rr_interval(th, v, parts, min_count=10, z=Z95):
    """Development change (Addendum 58): log-Wald, or MOVER-R on effective-n Wilson limits when an effective count is below
    min_count (THREADSAFE's rule for pair metrics, with effective sizes n* = p(1 - p) / var from the design)."""
    (p1, v1, d1), (p0, v0, d0) = parts
    # round E1: proportions are kept in [0, 1] (stratified designs always are; difference estimators need not be)
    p1, p0 = (float(np.clip(p, 0.0, 1.0)) if np.isfinite(p) else p for p in (p1, p0))
    eff = []
    for p, v_, nd in ((p1, v1, d1), (p0, v0, d0)):
        if not np.isfinite(p) or nd == 0:
            return 0.0, np.inf
        ne = min(p * (1 - p) / v_, nd) if v_ > 0 and 0 < p < 1 else nd   # THREADSAFE: n* = n when p is 0 or 1; at most n
        eff.append(max(ne, 1.0))
    (n1, n0) = eff
    if min(n1 * p1, n1 * (1 - p1), n0 * p0, n0 * (1 - p0)) >= min_count and np.isfinite(th):
        return log_wald(th, v, z)
    (l1, u1), (l0, u0) = wilson(p1, n1, z), wilson(p0, n0, z)
    if p0 <= 0:
        return (mover_rr(p1, l1, u1, max(p0, 1e-12), l0, u0)[0], np.inf)
    return mover_rr(p1, l1, u1, p0, l0, u0)
