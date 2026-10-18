"""Addendum 74, B1: comparators from the literature and the practitioner's estimator, for the item and edge regimes.

Items (metrics as in laws.py: totals T = sum_i z_i B_i, theta = theta(T)):
  run_ai     active inference (Zrnic and Candes 2024). A uniform burn-in of 10% of the budget Platt-calibrates the labeler's score;
             one-shot systematic PPS over the other items with sigma_i^2 = sum_m w_m b_mi^2 s_i (1 - s_i) (the metric-set compromise
             of pps.compromise_pi, without THREADSAFE's guard), mixed with uniform at tau = 0.5; the difference estimator with the
             calibrated score as prediction, power-tuned per metric (lambda in [0, 1]); Poisson variance.
  run_dsl    design-based supervised learning (Egami et al. 2023). A uniform sample of the whole budget; every item is assigned to
             one of 5 folds and predicted by a ridge logistic model (THREADSAFE's features) fitted on the labeled items outside its
             fold; pseudo-outcomes Yhat + R (Z - Yhat) / pi enter each metric; variance of a simple random sample of residuals.
  run_calag  calibrate and aggregate: a uniform sample; the Jeffreys gold rate of each calibration cell (falling back to its
             coarse cell, then to the overall rate, when empty) is plugged into every item; percentile bootstrap interval over
             the labeled sample (200 resamples). Cells: score level ('calag') or level x depth bucket x size bucket ('calag_s').
Edges (contributions as in edges.py; 8 totals):
  run_edge_ai     uniform burn-in edges (10% of the budget's labels); the flag-cell law P(z_p, z_c | y_p, y_c) (Jeffreys) as the
                  working law; one-shot PPS with the compromise over the two log risk ratios, mixed with uniform at tau = 0.5,
                  calibrated to the remaining labels; difference estimator (not post-stratified); THREADSAFE's interval.
  run_edge_dsl    uniform edges filling the budget; the edge law of edges_multi.edge_law cross-fitted over 5 folds of all edges;
                  pseudo-outcome totals; simple-random-sample variance; THREADSAFE's interval.
  run_edge_calag  uniform edges filling the budget; the flag-cell law (Jeffreys) plugged into every edge; percentile bootstrap.
"""
import numpy as np
from . import edges as E
from .pps import compromise_pi, systematic, logit_fit
from .edges_multi import edge_law, edge_moments, calibrated_pi, pps_props

NBOOT = 200


def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _logit(s):
    s = np.clip(np.asarray(s, float), 1e-3, 1 - 1e-3)
    return np.log(s / (1 - s))


def platt(s_lab, z_lab, s_all):
    """Platt scaling of a score on labeled items: logistic regression of z on (1, logit s)."""
    X = np.column_stack([np.ones(len(s_lab)), _logit(s_lab)])
    beta = logit_fit(X, np.asarray(z_lab, float), lam=1.0)
    return np.clip(_sig(beta[0] + beta[1] * _logit(s_all)), 1e-3, 1 - 1e-3)


# ------------------------------------------------------------------------------------------------------------------- items
def run_ai(metrics, score, budget, rng, z, mult=None, tau=0.5, burn=0.10):
    N = len(z); s = np.nan_to_num(np.asarray(score, float), nan=float(np.nanmedian(score)))
    n0 = int(min(max(round(burn * budget), 20), budget - 2)); s0 = rng.choice(N, n0, replace=False)
    inb = np.zeros(N, bool); inb[s0] = True; R = np.flatnonzero(~inb); n2 = budget - n0
    p = platt(s[s0], z[s0], s)
    mult = np.ones(len(metrics)) if mult is None else np.asarray(mult, float)
    sig2 = []
    for m in metrics:
        th0, c0 = m.theta_c(N / n0 * (z[s0] @ m.B[s0]))
        if not np.isfinite(th0):
            c0 = np.zeros(m.B.shape[1]); c0[0] = 1.0
        phi = m.B[R] @ c0; sig2.append(p[R] * (1 - p[R]) * phi * phi)
    piR = compromise_pi(sig2, mult, n2, delta=tau, guard=None)[0]
    k = systematic(piR, rng); S = R[k]; piS = piR[k]
    out = {}
    for m in metrics:
        Tb = z[s0] @ m.B[s0]; TpR = p[R] @ m.B[R]
        T1 = Tb + TpR + ((z[S] - p[S]) / piS) @ m.B[S]
        th1, c1 = m.theta_c(T1)
        if not np.isfinite(th1):
            out[m.name] = (np.nan, np.nan); continue
        u = m.B[S] @ c1; w = (1 - piS) / piS ** 2
        den = float((w * (p[S] * u) ** 2).sum())
        lam = float(np.clip((w * (z[S] * u) * (p[S] * u)).sum() / den, 0.0, 1.0)) if den > 0 else 0.0
        T = Tb + lam * TpR + ((z[S] - lam * p[S]) / piS) @ m.B[S]
        th, c = m.theta_c(T)
        e = (z[S] - lam * p[S]) * (m.B[S] @ c)
        out[m.name] = (th, float((w * e * e).sum()))
    out["_labels"] = int(n0 + len(S))
    return out


def run_dsl(metrics, X, budget, rng, z, K=5):
    N = len(z); s = rng.choice(N, budget, replace=False); fold = rng.integers(0, K, N)
    p = np.empty(N)
    for k in range(K):
        tr = s[fold[s] != k]; idx = np.flatnonzero(fold == k)
        beta = logit_fit(X[tr], z[tr], lam=1.0); p[idx] = np.clip(_sig(X[idx] @ beta), 1e-4, 1 - 1e-4)
    n = len(s); out = {}
    for m in metrics:
        T = p @ m.B + N / n * ((z[s] - p[s]) @ m.B[s])
        th, c = m.theta_c(T)
        if not np.isfinite(th):
            out[m.name] = (np.nan, np.nan); continue
        e = (z[s] - p[s]) * (m.B[s] @ c)
        out[m.name] = (th, N ** 2 * (1 - n / N) * float(np.var(e, ddof=1)) / n)
    out["_labels"] = int(n)
    return out


class CalagCells:
    """Per-metric totals of B within each fine calibration cell (computed once per population and cell definition)."""
    def __init__(self, metrics, fine, coarse):
        self.fine = np.asarray(fine, np.int64); self.coarse = np.asarray(coarse, np.int64)
        self.nf = int(self.fine.max()) + 1; self.nc = int(self.coarse.max()) + 1
        f2c = np.zeros(self.nf, np.int64); f2c[self.fine] = self.coarse; self.f2c = f2c
        self.S = [np.column_stack([np.bincount(self.fine, weights=m.B[:, j], minlength=self.nf) for j in range(m.B.shape[1])]) for m in metrics]

    def rates(self, fs, zs):
        nf = np.bincount(fs, minlength=self.nf).astype(float); kf = np.bincount(fs, weights=zs, minlength=self.nf)
        cs = self.f2c[fs]; nc = np.bincount(cs, minlength=self.nc).astype(float); kc = np.bincount(cs, weights=zs, minlength=self.nc)
        overall = (zs.sum() + 0.5) / (len(zs) + 1.0)
        rc = np.where(nc > 0, (kc + 0.5) / (nc + 1.0), overall)
        return np.where(nf > 0, (kf + 0.5) / (nf + 1.0), rc[self.f2c])


def run_calag(metrics, cells, budget, rng, z, nboot=NBOOT):
    N = len(z); s = rng.choice(N, budget, replace=False); fs = cells.fine[s]; zs = z[s].astype(float)
    r = cells.rates(fs, zs); out = {}; ci = {}
    ths = [m.theta_c(r @ S_)[0] for m, S_ in zip(metrics, cells.S)]
    boots = np.full((nboot, len(metrics)), np.nan)
    for b in range(nboot):
        ix = rng.integers(0, len(s), len(s)); rb = cells.rates(fs[ix], zs[ix])
        boots[b] = [m.theta_c(rb @ S_)[0] for m, S_ in zip(metrics, cells.S)]
    for j, m in enumerate(metrics):
        bj = boots[:, j]; bj = bj[np.isfinite(bj)]
        lo, hi = (float(np.percentile(bj, 2.5)), float(np.percentile(bj, 97.5))) if len(bj) >= 20 else (np.nan, np.nan)
        out[m.name] = (ths[j], np.nan); ci[m.name] = (lo, hi)
    out["_ci"] = ci; out["_labels"] = int(budget)
    return out


# ------------------------------------------------------------------------------------------------------------------- edges
def _uniform_edges(cp, budget, rng):
    """Uniform edges in random order, each buying its unrevealed ends, until the label budget would be exceeded."""
    Ed = len(cp.par); order = rng.permutation(Ed); known = set(); taken = []; used = 0
    for e in order:
        need = (cp.par[e] not in known) + (cp.child[e] not in known)
        if used + need > budget:
            if used >= budget - 1:
                break
            continue
        used += need; known.add(cp.par[e]); known.add(cp.child[e]); taken.append(e)
        if used >= budget:
            break
    return np.array(taken, np.int64), used


def _flag_cell_law(cp, z, S):
    """Jeffreys estimate of P(z_p, z_c | y_p, y_c) per flag cell from labeled edges S; per-edge a, r1, r0."""
    fc = (2 * cp.yp + cp.yc).astype(int); gc = (2 * z[cp.par[S]] + z[cp.child[S]]).astype(int)
    cnt = np.full((4, 4), 0.5); np.add.at(cnt, (fc[S], gc), 1.0); q = cnt / cnt.sum(1, keepdims=True)
    qe = q[fc]; a = qe[:, 2] + qe[:, 3]; r1 = qe[:, 3] / np.maximum(a, 1e-12); r0 = qe[:, 1] / np.maximum(1 - a, 1e-12)
    return np.clip(a, 1e-4, 1 - 1e-4), np.clip(r1, 1e-4, 1 - 1e-4), np.clip(r0, 1e-4, 1 - 1e-4), q


def _contrib(cp, z, S):
    return E.contributions(z[cp.par[S]], z[cp.child[S]], cp.yp[S], cp.yc[S])


def run_edge_ai(cp, z, budget, rng, mult=(1.0, 0.25), tau=0.5, burn=0.10):
    Ed = len(cp.par); M = int(max(cp.par.max(), cp.child.max())) + 1
    n0 = int(min(max(round(burn * budget / 2), 20), Ed)); pe = rng.choice(Ed, n0, replace=False)
    inp = np.zeros(Ed, bool); inp[pe] = True
    known = np.zeros(M, bool); known[cp.par[pe]] = True; known[cp.child[pe]] = True
    a, r1, r0, _ = _flag_cell_law(cp, z, pe)
    Hp = _contrib(cp, z, pe); T0 = np.maximum(Ed / n0 * Hp.sum(0), 0.5 * Ed / n0)
    Ht, S2 = edge_moments(a, r1, r0, cp.yp, cp.yc, T0)
    rest = np.flatnonzero(~inp); left = max(budget - int(known.sum()), 2)
    pi_r = calibrated_pi([S2[0][rest], S2[1][rest]], mult, cp.par[rest], cp.child[rest], known, M, left, delta=tau, norm="capped", guard=None)
    k = systematic(pi_r, rng); sel = rest[k]; piS = pi_r[k]
    Hs = _contrib(cp, z, sel)
    T = Hp.sum(0) + Ht[rest].sum(0) + ((Hs - Ht[sel]) / piS[:, None]).sum(0)
    out = {}
    for j, (name, th) in enumerate(zip(E.METRICS, E.log_metrics(T))):
        r = E.influence(Hs, T)[j] - E.influence(Ht[sel], T)[j]
        out[name] = (th, float(((1 - piS) / piS ** 2 * r * r).sum()))
    known[cp.par[sel]] = True; known[cp.child[sel]] = True
    out["_labels"] = int(known.sum()); out["_props"] = pps_props(T, [(Hs, Ht[sel], piS)], np.r_[Hp, Hs])
    return out


def _srs_props(H, R, T, Ed):
    n = len(H); out = []
    for num, den in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        if T[den] <= 0:
            out.append((np.nan, 0.0, 0)); continue
        p = T[num] / T[den]; u = (R[:, num] - p * R[:, den]) / T[den]
        v = Ed ** 2 * (1 - n / Ed) * float(np.var(u, ddof=1)) / n if n > 1 else np.nan
        out.append((p, v, int((H[:, den] > 0).sum())))
    return out


def run_edge_dsl(cp, z, budget, rng, Xi, K=5):
    Ed = len(cp.par); S, used = _uniform_edges(cp, budget, rng); n = len(S)
    fold = rng.integers(0, K, Ed); a = np.empty(Ed); r1 = np.empty(Ed); r0 = np.empty(Ed)
    for k in range(K):
        tr = S[fold[S] != k]; idx = np.flatnonzero(fold == k)
        ak, r1k, r0k = edge_law(Xi, cp.par, cp.child, tr, z, lam=1.0, parent_fit="parents", bayes=False)
        a[idx], r1[idx], r0[idx] = ak[idx], r1k[idx], r0k[idx]
    Ht, _ = edge_moments(a, r1, r0, cp.yp, cp.yc, np.ones(8))
    Hs = _contrib(cp, z, S); R = Hs - Ht[S]
    T = Ht.sum(0) + Ed / n * R.sum(0)
    out = {}
    for j, (name, th) in enumerate(zip(E.METRICS, E.log_metrics(T))):
        r = E.influence(Hs, T)[j] - E.influence(Ht[S], T)[j]
        out[name] = (th, Ed ** 2 * (1 - n / Ed) * float(np.var(r, ddof=1)) / n if n > 1 else np.nan)
    out["_labels"] = int(used); out["_props"] = _srs_props(Hs, R, T, Ed)
    return out


def run_edge_calag(cp, z, budget, rng, nboot=NBOOT):
    Ed = len(cp.par); S, used = _uniform_edges(cp, budget, rng)
    fc = (2 * cp.yp + cp.yc).astype(int); Nf = np.bincount(fc, minlength=4).astype(float)
    gold_cells = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); fy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    # contributions of gold cell g under flag cell f: Hc[f, g, :]
    Hc = np.stack([E.contributions(gold_cells[:, 0], gold_cells[:, 1], np.full(4, fy[f, 0]), np.full(4, fy[f, 1])) for f in range(4)])
    fS = fc[S]; gS = (2 * z[cp.par[S]] + z[cp.child[S]]).astype(int)

    def est(fs, gs):
        cnt = np.full((4, 4), 0.5); np.add.at(cnt, (fs, gs), 1.0); q = cnt / cnt.sum(1, keepdims=True)
        T = (Nf[:, None] * np.einsum("fg,fgk->fk", q, Hc)).sum(0)
        return E.log_metrics(T)
    th = est(fS, gS); B_ = np.full((nboot, 2), np.nan)
    for b in range(nboot):
        ix = rng.integers(0, len(S), len(S)); B_[b] = est(fS[ix], gS[ix])
    out = {}; ci = {}
    for j, name in enumerate(E.METRICS):
        bj = B_[:, j]; bj = bj[np.isfinite(bj)]
        ci[name] = (float(np.exp(np.percentile(bj, 2.5))), float(np.exp(np.percentile(bj, 97.5)))) if len(bj) >= 20 else (0.0, np.inf)
        out[name] = (th[j], np.nan)
    out["_ci"] = ci; out["_labels"] = int(used)
    return out
