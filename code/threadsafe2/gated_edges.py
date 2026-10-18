"""THREADSAFE edge regime, round B (Addendum 63): parent-first gated cluster sampling for the reply risk ratios.

The sampling unit is a parent comment with all its reply edges. A sampled parent's gold label is bought first (THREADSAFE's
gate idea): if it is positive, every reply is labeled; if it is negative, each reply is labeled independently with probability
q, and weighted by 1/q. P and Q of the gold risk ratio need only the parent's label.
Phase 1, pilot: about 15% of the budget, parents split equally over the parent's score levels, every reply labeled (q = 1).
Working model from the pilot: P(z_parent = 1 | parent level), P(z_child = 1 | z_parent, child level) (Jeffreys).
Phase 2, main: systematic PPS over the remaining parents with pi_u ~ sigma_u / sqrt(c_u) (model sd of the cluster's linearized
contribution, compromise over the two log risk ratios with laws-first weights; c_u its expected label cost), a defensive
uniform share delta = 0.1, scaled so that the expected cost fills the budget.
Estimator: difference estimator over clusters, T = sum_pilot t_u + sum_{u not in pilot} t~_u + sum_main (t^_u - t~_u) / pi_u,
t^_u = sum over labeled replies of h_e (x 1/q under a negative parent); conditionally unbiased given the pilot.
Variance: Poisson approximation for parents plus the Bernoulli(q) reply stage.
"""
import numpy as np
from . import edges as E
from .pps import score_levels, systematic, _waterfill, SortedWaterfill
from .laws import equal_split

DELTA = 0.1
CELLS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)


class ClusterPop:
    def __init__(self, parent_row, z_len, y, s):
        """parent_row: per item, its parent's row or -1. y: flags; s: scores (levels computed here)."""
        pr = np.asarray(parent_row); child = np.flatnonzero(pr >= 0); par = pr[child]
        o = np.argsort(par, kind="stable"); self.child, self.par = child[o], par[o]
        self.U, start = np.unique(self.par, return_index=True); self.ptr = np.r_[start, len(self.par)]
        self.m = np.diff(self.ptr).astype(float); self.eu = np.repeat(np.arange(len(self.U)), np.diff(self.ptr))   # edge -> cluster
        y = np.nan_to_num(np.asarray(y, float)); self.y = y; lev = score_levels(s); self.lev = lev
        self.yp, self.yc = y[self.par], y[self.child]
        self.lu = lev[self.U]; self.lc = lev[self.child]
        self.Hy_tot = E.contributions(self.yp, self.yc, self.yp, self.yc).sum(0)

    def truth(self, z):
        return E.log_metrics(E.contributions(z[self.par], z[self.child], self.yp, self.yc).sum(0))

    def labeler_log_rr(self):
        return E.log_metrics(self.Hy_tot)[0]


def _edge_moments(cp, pu, r1, r0, T, q):
    """Per-edge model quantities: expected contributions H~ (8 totals) and, per metric j, E[v | z_p] and E[v^2 | z_p] for z_p = 0, 1."""
    Ed = len(cp.par); pz = pu[cp.eu]
    Ht = np.zeros((Ed, 8)); Ev = np.zeros((2, 2, Ed)); Ev2 = np.zeros((2, 2, Ed))
    for zp in (0, 1):
        rc = r1[cp.lc] if zp else r0[cp.lc]
        for zc, w in ((0, 1 - rc), (1, rc)):
            H = E.contributions(np.full(Ed, zp, float), np.full(Ed, zc, float), cp.yp, cp.yc)
            pw = (pz if zp else 1 - pz) * w
            Ht += pw[:, None] * H
            g, qv = E.influence(H, T)
            for j, v in enumerate((g, qv)):
                Ev[j, zp] += w * v; Ev2[j, zp] += w * v * v
    return Ht, Ev, Ev2


def run(cp, z, budget, rng, q=0.5, mult=(1.0, 0.25), pilot_share=0.15):
    NU = len(cp.U); levels_u = cp.lu; ids, inv = np.unique(levels_u, return_inverse=True); L = len(ids)
    members = [np.flatnonzero(inv == l) for l in range(L)]
    # ---- pilot: parents by level, full clusters, until the pilot's cost (unique labels) reaches pilot_share * budget
    target = pilot_share * budget; perms = [mm[rng.permutation(len(mm))] for mm in members]; take = np.zeros(L, int)
    cost = 0.0; mean_c = np.array([1 + cp.m[mm].mean() if len(mm) else np.inf for mm in members])
    while cost < target:
        progressed = False
        for l in range(L):
            if take[l] < len(perms[l]) and cost < target:
                cost += 1 + cp.m[perms[l][take[l]]]; take[l] += 1; progressed = True
        if not progressed:
            break
    pil = np.concatenate([perms[l][:take[l]] for l in range(L)]) if L else np.empty(0, int)
    inp = np.zeros(NU, bool); inp[pil] = True
    # working model
    pu_l = np.array([(z[cp.U[perms[l][:take[l]]]].sum() + 0.5) / (take[l] + 1.0) for l in range(L)]); pu = pu_l[inv]
    pe = np.flatnonzero(inp[cp.eu]); zpe, zce = z[cp.par[pe]], z[cp.child[pe]]; nlev = int(cp.lc.max()) + 1 if len(cp.lc) else 1
    def rates(mask):
        x = np.bincount(cp.lc[pe][mask], weights=zce[mask], minlength=nlev); n = np.bincount(cp.lc[pe][mask], minlength=nlev)
        pool = (zce[mask].sum() + 0.5) / (mask.sum() + 1.0); return (x + 2 * pool) / (n + 2)
    r1, r0 = rates(zpe == 1), rates(zpe == 0)
    # plug-in totals from the pilot (Horvitz-Thompson over parent levels)
    wl = np.array([len(members[l]) / max(take[l], 1) for l in range(L)])[inv]
    Hp = E.contributions(zpe, zce, cp.yp[pe], cp.yc[pe]); T0 = (wl[cp.eu[pe]][:, None] * Hp).sum(0)
    T0 = np.maximum(T0, 0.5 * len(cp.par) / max(len(pe), 1))
    Ht, Ev, Ev2 = _edge_moments(cp, pu, r1, r0, T0, q)
    # cluster model variances and costs
    sig2 = []
    for j in range(2):
        m1 = np.bincount(cp.eu, weights=Ev[j, 1], minlength=NU); m0 = np.bincount(cp.eu, weights=Ev[j, 0], minlength=NU)
        v1 = np.bincount(cp.eu, weights=Ev2[j, 1] - Ev[j, 1] ** 2, minlength=NU); v0 = np.bincount(cp.eu, weights=Ev2[j, 0] - Ev[j, 0] ** 2, minlength=NU)
        s2v0 = np.bincount(cp.eu, weights=Ev2[j, 0], minlength=NU)
        var = pu * (1 - pu) * (m1 - m0) ** 2 + pu * v1 + (1 - pu) * v0 + (1 - pu) * (1 - q) / q * s2v0
        sig2.append(np.maximum(var, 0))
    own = [np.sqrt(s * (1 + pu * cp.m + (1 - pu) * q * cp.m)).sum() ** 2 for s in sig2]
    w = np.asarray(mult, float) / np.maximum(np.array(own), 1e-300)
    sig = np.sqrt(w[0] * sig2[0] + w[1] * sig2[1]); cu = 1 + pu * cp.m + (1 - pu) * q * cp.m
    rest = np.flatnonzero(~inp); left = budget - cost
    # expected-cost calibration: sum pi_u c_u = left, pi ~ sig/sqrt(c) (+ defensive share), capped at 1
    score = sig[rest] / np.sqrt(cu[rest]); lo, hi = 0.0, 1.0
    def exp_cost(n):
        pi = _waterfill((1 - DELTA) * score, n, DELTA * n / max(len(rest), 1)); return (pi * cu[rest]).sum(), pi
    n_hi = float(len(rest))
    lo_n, hi_n = 0.0, n_hi
    for _ in range(40):
        mid = (lo_n + hi_n) / 2
        if exp_cost(mid)[0] < left:
            lo_n = mid
        else:
            hi_n = mid
    pi_r = exp_cost(lo_n)[1]
    sel = rest[systematic(pi_r, rng)]; pi = np.zeros(NU); pi[rest] = pi_r
    # ---- reveal: pilot clusters fully; main parents, then replies (all if the parent is positive, Bernoulli(q) otherwise)
    tt = np.zeros((NU, 8)); revealed = set(cp.U[pil].tolist()) | set(cp.child[pe].tolist())
    np.add.at(tt, cp.eu[pe], Hp)
    se = np.flatnonzero(np.isin(cp.eu, sel)); zp_s = z[cp.par[se]]
    keep = (zp_s == 1) | (rng.random(len(se)) < q); se = se[keep]; wgt = np.where(z[cp.par[se]] == 1, 1.0, 1.0 / q)
    Hs = E.contributions(z[cp.par[se]], z[cp.child[se]], cp.yp[se], cp.yc[se])
    np.add.at(tt, cp.eu[se], wgt[:, None] * Hs)
    # P and Q come from the parent label alone: replace the sampled parents' gated estimates of P (col 1) and Q (col 3)
    zU = z[cp.U]; tt[sel, 1] = zU[sel] * cp.m[sel]; tt[sel, 3] = (1 - zU[sel]) * cp.m[sel]
    revealed |= set(cp.U[sel].tolist()) | set(cp.child[se].tolist())
    tilde = np.zeros((NU, 8)); np.add.at(tilde, cp.eu, Ht)
    T = tt[pil].sum(0) + tilde[rest].sum(0) + ((tt[sel] - tilde[sel]) / pi[sel][:, None]).sum(0)
    lg, lq = E.log_metrics(T)
    out = {}
    for j, (name, th) in enumerate(zip(E.METRICS, (lg, lq))):
        # linearized cluster residuals and the reply-stage variance
        vs = E.influence(Hs, T)[j]; vt = E.influence(Ht, T)[j]
        tv = np.bincount(cp.eu[se], weights=wgt * vs, minlength=NU); tvt = np.bincount(cp.eu, weights=vt, minlength=NU)
        gsign = E.SIGN / np.maximum(T[:4], 1e-300)
        if j == 0:   # P and Q from the parent label exactly
            tv[sel] = tv[sel] - np.bincount(cp.eu[se], weights=wgt * (Hs[:, 1] * gsign[1] + Hs[:, 3] * gsign[3]), minlength=NU)[sel] \
                + zU[sel] * cp.m[sel] * gsign[1] + (1 - zU[sel]) * cp.m[sel] * gsign[3]
        r = tv[sel] - tvt[sel]
        v1 = ((1 - pi[sel]) / pi[sel] ** 2 * r * r).sum()
        neg = z[cp.par[se]] == 0
        vg = vs - (Hs[:, 1] * gsign[1] + Hs[:, 3] * gsign[3]) if j == 0 else vs     # P and Q are exact: not part of the reply stage
        stage2 = np.bincount(cp.eu[se][neg], weights=((1 - q) / q ** 2) * vg[neg] ** 2, minlength=NU)
        v2 = (stage2[sel] / pi[sel] ** 2).sum() if q < 1 else 0.0
        out[name] = (th, float(v1 + v2))
    out["_props"] = None
    out["_labels"] = int(len(revealed))
    return out


def run_edge_pps(cp, z, budget, rng, mult=(1.0, 0.25), pilot_share=0.15):
    """Round B2 (Addendum 64): edges sampled with probability proportional to their model influence.
    Pilot: about 15% of the budget, edges split equally over parent score levels, both ends labeled. Working model as in run():
    P(z_p = 1 | parent level), P(z_c = 1 | z_p, child level). Main phase: systematic PPS over the remaining edges with
    pi_e ~ sigma_e (compromise over the two log risk ratios, laws-first), a defensive share delta, and the expected number of
    distinct labels (Poisson calculation over each comment's edges) set to the remaining budget. Difference estimator over
    edges; Poisson variance."""
    Ed = len(cp.par); lev_p = cp.lu[cp.eu]; ids, inv = np.unique(lev_p, return_inverse=True); L = len(ids)
    members = [np.flatnonzero(inv == l) for l in range(L)]; NL = np.array([len(m) for m in members], float)
    n1 = int(min(max(round(pilot_share * budget / 2), 2 * L), Ed)); alloc = equal_split(NL, n1)
    pe = np.concatenate([m[rng.choice(len(m), int(a), replace=False)] for m, a in zip(members, alloc) if a > 0])
    inp = np.zeros(Ed, bool); inp[pe] = True
    zpe, zce = z[cp.par[pe]], z[cp.child[pe]]
    pu_l = np.array([(zpe[inv[pe] == l].sum() + 0.5) / ((inv[pe] == l).sum() + 1.0) for l in range(L)])
    pu = np.zeros(len(cp.U)); pu[:] = pu_l[np.searchsorted(ids, cp.lu)]
    nlev = int(cp.lc.max()) + 1
    def rates(mask):
        x = np.bincount(cp.lc[pe][mask], weights=zce[mask], minlength=nlev); n = np.bincount(cp.lc[pe][mask], minlength=nlev)
        pool = (zce[mask].sum() + 0.5) / (mask.sum() + 1.0); return (x + 2 * pool) / (n + 2)
    r1, r0 = rates(zpe == 1), rates(zpe == 0)
    wl = (NL / np.maximum(alloc, 1))[inv[pe]]
    Hp = E.contributions(zpe, zce, cp.yp[pe], cp.yc[pe]); T0 = np.maximum((wl[:, None] * Hp).sum(0), 0.5 * Ed / max(len(pe), 1))
    Ht, Ev, Ev2 = _edge_moments(cp, pu, r1, r0, T0, 1.0)
    pz = pu[cp.eu]; sig2 = []
    for j in range(2):
        m_ = pz * Ev[j, 1] + (1 - pz) * Ev[j, 0]; s2 = pz * Ev2[j, 1] + (1 - pz) * Ev2[j, 0] - m_ ** 2; sig2.append(np.maximum(s2, 0))
    own = [np.sqrt(s).sum() ** 2 for s in sig2]; w = np.asarray(mult, float) / np.maximum(np.array(own), 1e-300)
    sig = np.sqrt(w[0] * sig2[0] + w[1] * sig2[1])
    rest = np.flatnonzero(~inp); pilot_labels = len(np.unique(np.r_[cp.par[pe], cp.child[pe]])); left = max(budget - pilot_labels, 2)
    known = np.zeros(int(max(cp.par.max(), cp.child.max())) + 1, bool); known[cp.par[pe]] = True; known[cp.child[pe]] = True
    wf = SortedWaterfill((1 - DELTA) * sig[rest]); pr_, ch_ = cp.par[rest], cp.child[rest]; M = len(known)
    def exp_labels(n):
        pi = wf(n, DELTA * n / max(len(rest), 1)); l1 = np.log1p(-np.minimum(pi, 1 - 1e-12))
        # P(a comment is labeled) = 1 - prod over its edges of (1 - pi_e), for comments not already labeled in the pilot
        lq = np.bincount(pr_, weights=l1, minlength=M) + np.bincount(ch_, weights=l1, minlength=M)
        return float(((1 - np.exp(lq)) * (~known)).sum()), pi
    lo, hi = 0.0, float(len(rest))
    for _ in range(40):
        mid = (lo + hi) / 2
        if exp_labels(mid)[0] < left:
            lo = mid
        else:
            hi = mid
    pi_r = exp_labels(lo)[1]; sel = rest[systematic(pi_r, rng)]; pi = np.zeros(Ed); pi[rest] = pi_r
    Hs = E.contributions(z[cp.par[sel]], z[cp.child[sel]], cp.yp[sel], cp.yc[sel])
    T = Hp.sum(0) + Ht[rest].sum(0) + ((Hs - Ht[sel]) / pi[sel][:, None]).sum(0)
    lg, lq_ = E.log_metrics(T); out = {}
    for j, (name, th) in enumerate(zip(E.METRICS, (lg, lq_))):
        r = E.influence(Hs, T)[j] - E.influence(Ht[sel], T)[j]
        out[name] = (th, float(((1 - pi[sel]) / pi[sel] ** 2 * r * r).sum()))
    out["_labels"] = int(len(np.unique(np.r_[cp.par[pe], cp.child[pe], cp.par[sel], cp.child[sel]])))
    out["_props"] = None
    return out
