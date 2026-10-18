"""THREADSAFE edge regime, round E1: edge PPS with a richer working model (the labeler's score, structure and auxiliary
instruments), and its stratified counterpart.

Working model, fitted on the pilot edges (both ends labeled):
  parent:  logit P(z = 1 | x)            ridge logistic on the items at both ends of the pilot edges;
  child:   logit P(z_c = 1 | x_c, z_p, x_p)  ridge logistic on the pilot edges (child features, the parent's gold label and the
           parent's score and auxiliary features).
Per edge e this gives a_e = P(z_p = 1), r1_e = P(z_c = 1 | z_p = 1), r0_e = P(z_c = 1 | z_p = 0), hence the model law of
(z_p, z_c), the expected contributions H~_e (8 totals) and the model sd sigma_e of each log risk ratio's linearized variable.
Design 'pps': systematic PPS over the non-pilot edges with pi_e ~ sigma_e (compromise over the two log risk ratios with
laws-first weights), a defensive share delta, scaled so that the expected number of distinct labels fills the budget.
Estimator: difference estimator over edges; Poisson variance. Conditionally unbiased given the pilot.
"""
import numpy as np
from . import edges as E
from .pps import systematic, SortedWaterfill, logit_fit, own_var, predictive_p
from .laws import equal_split

DELTA = 0.1
CELLS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)


def _sig(eta):
    return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))


def _temper(eta, s2):
    return 1 / (1 + np.exp(-np.clip(eta / np.sqrt(1 + np.pi * np.maximum(s2, 0) / 8), -30, 30)))


def edge_law(Xi, par, child, pe, z, lam=1.0, Xp_extra=None, parent_fit="items", bayes=False):
    """Fit the parent model and the child model on labeled edges pe; return per-edge a, r1, r0 for all edges.
    parent_fit 'items': the parent model is fitted on every labeled comment (rounds B2-E1 first version); 'parents': on the
    parents of the labeled edges, one row per edge (round E1), so that it estimates P(z_p = 1 | x) among parents."""
    rows = par[pe] if parent_fit == "parents" else np.unique(np.r_[par[pe], child[pe]])
    if bayes:                                                                 # posterior-predictive tempering (round E2)
        bi, Hi_ = logit_fit(Xi[rows], z[rows], lam=lam, return_hessian=True); a = np.clip(predictive_p(Xi, bi, Hi_)[par], 1e-4, 1 - 1e-4)
    else:
        bi = logit_fit(Xi[rows], z[rows], lam=lam); a = np.clip(_sig((Xi @ bi)[par]), 1e-4, 1 - 1e-4)
    d = Xi.shape[1]
    Xp_all = Xi if Xp_extra is None else None
    Xc = np.column_stack([Xi[child[pe]], z[par[pe]].astype(float), (Xi[par[pe]] if Xp_extra is None else Xp_extra[pe])])
    if bayes:
        bc, Hc = logit_fit(Xc, z[child[pe]], lam=lam, return_hessian=True)
    else:
        bc = logit_fit(Xc, z[child[pe]], lam=lam)
    base = (Xi @ bc[:d])[child] + ((Xi @ bc[d + 1:])[par] if Xp_extra is None else Xp_extra @ bc[d + 1:])
    if bayes and Xp_extra is None:
        A = np.linalg.inv(Hc); A11, a12, A13, a22, a23, A33 = A[:d, :d], A[:d, d], A[:d, d + 1:], A[d, d], A[d, d + 1:], A[d + 1:, d + 1:]
        qc = ((Xi @ A11) * Xi).sum(1)[child]; qp = ((Xi @ A33) * Xi).sum(1)[par]
        cr_ = ((Xi @ A13)[child] * Xi[par]).sum(1); lc = (Xi @ a12)[child] + (Xi @ a23)[par]
        s0 = qc + qp + 2 * cr_; s1 = s0 + a22 + 2 * lc
        r1 = np.clip(_temper(base + bc[d], s1), 1e-4, 1 - 1e-4); r0 = np.clip(_temper(base, s0), 1e-4, 1 - 1e-4)
    else:
        r1 = np.clip(_sig(base + bc[d]), 1e-4, 1 - 1e-4); r0 = np.clip(_sig(base), 1e-4, 1 - 1e-4)
    return a, r1, r0


def edge_moments_slow(a, r1, r0, yp, yc, T):
    """Reference implementation of edge_moments (loops over the four cells)."""
    Ed = len(a); q = np.column_stack([(1 - a) * (1 - r0), (1 - a) * r0, a * (1 - r1), a * r1])   # cells 00, 01, 10, 11
    Ht = np.zeros((Ed, 8)); m1 = np.zeros((2, Ed)); m2 = np.zeros((2, Ed))
    for k, (zp, zc) in enumerate(CELLS):
        H = E.contributions(np.full(Ed, zp), np.full(Ed, zc), yp, yc); Ht += q[:, k:k + 1] * H
        g, qv = E.influence(H, T)
        for j, v in enumerate((g, qv)):
            m1[j] += q[:, k] * v; m2[j] += q[:, k] * v * v
    return Ht, np.maximum(m2 - m1 ** 2, 0)


def edge_moments(a, r1, r0, yp, yc, T):
    """Per-edge expected contributions (8 totals) and per-metric model variance of the linearized variables, in closed form:
    under cell (z_p, z_c) the gold variable is 1/Q (00), 1/Q - 1/C (01), -1/P (10), 1/A - 1/P (11), and the placebo variable
    is kappa(y_p, y_c) = y_p y_c / a - y_p / b - (1 - y_p) y_c / c + (1 - y_p) / d in cell 00 and 0 otherwise."""
    g, qq = np.maximum(T[:4], 1e-300), np.maximum(T[4:], 1e-300)
    q00, q01, q10, q11 = (1 - a) * (1 - r0), (1 - a) * r0, a * (1 - r1), a * r1
    yp = np.asarray(yp, float); yc = np.asarray(yc, float)
    Ht = np.column_stack([q11, a, q01, 1 - a, q00 * yp * yc, q00 * yp, q00 * (1 - yp) * yc, q00 * (1 - yp)])
    v00, v01, v10, v11 = 1 / g[3], 1 / g[3] - 1 / g[2], -1 / g[1], 1 / g[0] - 1 / g[1]
    mg = q00 * v00 + q01 * v01 + q10 * v10 + q11 * v11; mg2 = q00 * v00 ** 2 + q01 * v01 ** 2 + q10 * v10 ** 2 + q11 * v11 ** 2
    kap = yp * yc / qq[0] - yp / qq[1] - (1 - yp) * yc / qq[2] + (1 - yp) / qq[3]
    S2 = np.vstack([np.maximum(mg2 - mg ** 2, 0), np.maximum(q00 * (1 - q00) * kap ** 2, 0)])
    return Ht, S2


def pps_props(T, parts, H_lab):
    """The four conditional proportions behind the two risk ratios (gold A/P, C/Q; placebo a/b, c/d) with linearized PPS
    variances, for the hybrid interval (edges.rr_interval). parts: list of (Hs, Ht_s, pi_s) per sampling phase; H_lab: the
    contributions of every labeled edge (denominator counts)."""
    out = []
    for num, den in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        if T[den] <= 0:
            out.append((np.nan, 0.0, 0)); continue
        p = T[num] / T[den]; v = 0.0
        for Hs, Hts, piS in parts:
            r = Hs - Hts; u = (r[:, num] - p * r[:, den]) / T[den]; v += float(((1 - piS) / piS ** 2 * u * u).sum())
        out.append((p, v, int((H_lab[:, den] > 0).sum())))
    return out


def _solve_labels(exp_labels, left, n_max, tol=0.25, iters=60):
    """The pi whose expected number of new labels is `left` (largest n with exp_labels(n) <= left, to within tol labels):
    Illinois regula falsi on the increasing function exp_labels(n) - left over [0, n_max]."""
    lo, flo = 0.0, -left
    fhi_full, pi_hi = exp_labels(n_max)
    if fhi_full <= left:
        return pi_hi
    hi, fhi = n_max, fhi_full - left; side = 0; best = exp_labels(0.0)[1]
    for _ in range(iters):
        x = (lo * fhi - hi * flo) / (fhi - flo) if fhi != flo else (lo + hi) / 2
        x = min(max(x, lo + 1e-9 * (hi - lo)), hi - 1e-9 * (hi - lo))
        fx, pix = exp_labels(x); fx -= left
        if fx <= 0:
            lo, flo, best = x, fx, pix
            if side == -1:
                fhi /= 2
            side = -1
        else:
            hi, fhi = x, fx
            if side == 1:
                flo /= 2
            side = 1
        if -tol <= flo <= 0 or hi - lo < 1e-6:
            break
    return best


def cell_sigma(sig, cells):
    """Each edge's (or item's) cell RMS of sigma: the per-unit weight of a Neyman allocation over the cells."""
    cells = np.asarray(cells); nc = int(cells.max()) + 1
    m2 = np.bincount(cells, weights=sig * sig, minlength=nc) / np.maximum(np.bincount(cells, minlength=nc), 1)
    return np.sqrt(m2)[cells]


def calibrated_pi(sig2, mult, rest_par, rest_ch, known, M, left, delta=DELTA, norm="capped", guard=4.0, iters=6, floor_cells=None, cost=None):
    """Edge inclusion probabilities whose expected number of new labels is `left`: pi = waterfill((1 - delta) sigma) with
    sigma^2 = sum_m w_m s2_m. norm 'capped' (round E1): w_m = mult_m / V*_m (capped own optimum at left / 2 edges), with the
    guard of pps.compromise_pi; 'uncapped': w_m = mult_m / (sum sigma_m)^2 (rounds B2 and earlier).
    floor_cells (round E3): the defensive share is a stratified design instead of a uniform one,
    pi = (1 - delta) pi_PPS + delta pi_strata, where pi_strata is the Neyman allocation over the given cells (each unit's
    weight is its cell's RMS sigma), both water-filled to the same expected size.
    cost (round E5): each edge's number of ends not yet labeled (0, 1 or 2). An edge with both ends labeled is free and is
    taken with pi = 1; the others get pi ~ sigma / sqrt(cost) in both shares (PPS and Neyman allocation with unequal costs)."""
    nE = len(sig2[0]); mult = np.asarray(mult, float)
    if norm == "capped":
        own = np.array([own_var(s2, max(left / 2.0, 1.0)) for s2 in sig2])
    else:
        own = np.array([np.sqrt(s2).sum() ** 2 for s2 in sig2])
    w = mult / np.maximum(own, 1e-300)
    free = np.zeros(nE, bool) if cost is None else (np.asarray(cost) == 0)
    rc = None if cost is None else 1.0 / np.sqrt(np.maximum(np.asarray(cost, float), 1.0))
    nf = int((~free).sum())
    for _ in range(iters + 1):
        sig = np.sqrt(sum(wi * s2 for wi, s2 in zip(w, sig2)))
        if floor_cells is not None:
            cs_ = np.maximum(cell_sigma(sig, floor_cells), 1e-12)
            if cost is None:
                wfP = SortedWaterfill(sig); wfS = SortedWaterfill(cs_)
                def mix(n):
                    return (1 - delta) * wfP(n, 0.0) + delta * wfS(n, 0.0)
            else:
                wfP = SortedWaterfill((sig * rc)[~free]); wfS = SortedWaterfill((cs_ * rc)[~free])
                def mix(n):
                    out = np.ones(nE); out[~free] = (1 - delta) * wfP(n, 0.0) + delta * wfS(n, 0.0); return out
        elif cost is None:
            wf = SortedWaterfill((1 - delta) * sig)
            def mix(n):
                return wf(n, delta * n / max(nE, 1))
        else:
            wf = SortedWaterfill(((1 - delta) * sig * rc)[~free])
            def mix(n):
                out = np.ones(nE); out[~free] = wf(n, delta * n / max(nf, 1)); return out
        def exp_labels(n):
            pi = mix(n); l1 = np.log1p(-np.minimum(pi, 1 - 1e-12))
            lq = np.bincount(rest_par, weights=l1, minlength=M) + np.bincount(rest_ch, weights=l1, minlength=M)
            return float(((1 - np.exp(lq)) * (~known)).sum()), pi
        pi = _solve_labels(exp_labels, left, float(nE if cost is None else max(nf, 1)))
        if norm != "capped" or not guard:
            break
        ok = pi > 0
        R = np.array([float((s2[ok] * (1 / pi[ok] - 1)).sum()) / max(own_var(s2, pi.sum()), 1e-300) for s2 in sig2])
        bad = R > guard
        if not bad.any():
            break
        w = np.where(bad, 2 * w, w)
    return pi


def run_pps(cp, z, budget, rng, Xi, mult=(1.0, 0.25), pilot_share=0.15, lam=1.0, Xp_extra=None, delta=DELTA, norm="capped", guard=4.0):
    """One edge audit. cp: gated_edges.ClusterPop (edges, flags, parent levels); Xi: item feature matrix."""
    Ed = len(cp.par); lev_p = cp.lu[cp.eu]; ids, inv = np.unique(lev_p, return_inverse=True); L = len(ids)
    members = [np.flatnonzero(inv == l) for l in range(L)]; NL = np.array([len(m) for m in members], float)
    n1 = int(min(max(round(pilot_share * budget / 2), 2 * L), Ed)); alloc = equal_split(NL, n1)
    pe = np.concatenate([m[rng.choice(len(m), int(k), replace=False)] for m, k in zip(members, alloc) if k > 0])
    inp = np.zeros(Ed, bool); inp[pe] = True
    zpe, zce = z[cp.par[pe]], z[cp.child[pe]]
    a, r1, r0 = edge_law(Xi, cp.par, cp.child, pe, z, lam=lam, Xp_extra=Xp_extra)
    wl = (NL / np.maximum(alloc, 1))[inv[pe]]
    Hp = E.contributions(zpe, zce, cp.yp[pe], cp.yc[pe]); T0 = np.maximum((wl[:, None] * Hp).sum(0), 0.5 * Ed / max(len(pe), 1))
    Ht, S2 = edge_moments(a, r1, r0, cp.yp, cp.yc, T0)
    rest = np.flatnonzero(~inp); pilot_labels = len(np.unique(np.r_[cp.par[pe], cp.child[pe]])); left = max(budget - pilot_labels, 2)
    M = int(max(cp.par.max(), cp.child.max())) + 1
    known = np.zeros(M, bool); known[cp.par[pe]] = True; known[cp.child[pe]] = True
    pi_r = calibrated_pi([S2[0][rest], S2[1][rest]], mult, cp.par[rest], cp.child[rest], known, M, left, delta=delta, norm=norm, guard=guard)
    sel = rest[systematic(pi_r, rng)]; pi = np.zeros(Ed); pi[rest] = pi_r
    Hs = E.contributions(z[cp.par[sel]], z[cp.child[sel]], cp.yp[sel], cp.yc[sel])
    T = Hp.sum(0) + Ht[rest].sum(0) + ((Hs - Ht[sel]) / pi[sel][:, None]).sum(0)
    lg, lq_ = E.log_metrics(T); out = {}
    for j, (name, th) in enumerate(zip(E.METRICS, (lg, lq_))):
        r = E.influence(Hs, T)[j] - E.influence(Ht[sel], T)[j]
        out[name] = (th, float(((1 - pi[sel]) / pi[sel] ** 2 * r * r).sum()))
    out["_labels"] = int(len(np.unique(np.r_[cp.par[pe], cp.child[pe], cp.par[sel], cp.child[sel]])))
    out["_props"] = pps_props(T, [(Hs, Ht[sel], pi[sel])], np.r_[Hp, Hs])
    return out


def run_pps_seq(cp, z, budget, rng, Xi, mult=(1.0, 0.25), pilot_share=0.15, K=3, lam=1.0, delta=DELTA, norm="capped", guard=4.0):
    """Round E1 (sequential): pilot edges as in run_pps; the non-pilot edges are split at random into K folds; fold j is sampled
    by PPS with the working model refitted on every labeled edge so far, its expected new labels set to an equal share of the
    remaining budget. Estimator: pilot totals + sum over folds of the fold's difference estimator (each conditionally unbiased
    given the labels before it); variance: the sum of the folds' Poisson variances."""
    Ed = len(cp.par); lev_p = cp.lu[cp.eu]; ids, inv = np.unique(lev_p, return_inverse=True); L = len(ids)
    members = [np.flatnonzero(inv == l) for l in range(L)]; NL = np.array([len(m) for m in members], float)
    n1 = int(min(max(round(pilot_share * budget / 2), 2 * L), Ed)); alloc = equal_split(NL, n1)
    pe = np.concatenate([m[rng.choice(len(m), int(k), replace=False)] for m, k in zip(members, alloc) if k > 0])
    inp = np.zeros(Ed, bool); inp[pe] = True
    M = int(max(cp.par.max(), cp.child.max())) + 1
    known = np.zeros(M, bool); known[cp.par[pe]] = True; known[cp.child[pe]] = True
    left_total = max(budget - int(known.sum()), 2 * K)
    folds = np.array_split(rng.permutation(np.flatnonzero(~inp)), K)
    Hp = E.contributions(z[cp.par[pe]], z[cp.child[pe]], cp.yp[pe], cp.yc[pe])
    wl = (NL / np.maximum(alloc, 1))[inv[pe]]; T_pil = np.maximum((wl[:, None] * Hp).sum(0), 0.5 * Ed / max(len(pe), 1))
    T_exact = Hp.sum(0); fold_T = np.zeros(8); labeled_e = pe.copy(); store = []
    for j, F in enumerate(folds):
        a, r1, r0 = edge_law(Xi, cp.par, cp.child, labeled_e, z, lam=lam)
        later = np.concatenate(folds[j:])
        Ht_all, _ = edge_moments(a[later], r1[later], r0[later], cp.yp[later], cp.yc[later], T_pil)
        Trun = np.maximum(T_exact + fold_T + Ht_all.sum(0), 0.5)
        Ht, S2 = edge_moments(a[F], r1[F], r0[F], cp.yp[F], cp.yc[F], Trun)
        left = max((budget - int(known.sum())) / (K - j), 1.0)
        piF = calibrated_pi([S2[0], S2[1]], mult, cp.par[F], cp.child[F], known, M, left, delta=delta, norm=norm, guard=guard)
        sk = systematic(piF, rng); S = F[sk]; piS = piF[sk]
        Hs = E.contributions(z[cp.par[S]], z[cp.child[S]], cp.yp[S], cp.yc[S])
        fold_T = fold_T + Ht.sum(0) + ((Hs - Ht[sk]) / piS[:, None]).sum(0)
        store.append((Hs, Ht[sk], piS)); known[cp.par[S]] = True; known[cp.child[S]] = True; labeled_e = np.r_[labeled_e, S]
    T = T_exact + fold_T
    lg, lq_ = E.log_metrics(T); out = {}
    for jm, (name, th) in enumerate(zip(E.METRICS, (lg, lq_))):
        v = 0.0
        for Hs, Hts, piS in store:
            r = E.influence(Hs, T)[jm] - E.influence(Hts, T)[jm]; v += float(((1 - piS) / piS ** 2 * r * r).sum())
        out[name] = (th, v)
    out["_labels"] = int(known.sum()); out["_props"] = pps_props(T, store, np.concatenate([Hp] + [h for h, _, _ in store]))
    return out


def run_edge(cp, z, budget, rng, Xi, K=1, mult=(1.0, 0.25), pilot_share=0.10, pilot_by="cell", post=True, lam=1.0, delta=DELTA,
             norm="capped", guard=4.0, parent_fit="parents", bayes=False, floor="uniform", law_fixed=None, cost=False, tune=False, tune_min=30):
    """Round E1, general form. Pilot: pilot_share * budget / 2 edges, split equally over parent score level ('level') or over
    parent level x child flag ('cell'); both ends labeled. Then K phases over a random partition of the other edges into K folds
    (K = 1: one main phase); phase j: working model refitted on every labeled edge so far, PPS with pi ~ sigma (capped
    compromise, guard, defensive share delta), expected new labels = an equal share of the remaining budget.
    Estimator per phase: sum_F H~ + correction. post=False: sum_S (H - H~) / pi (difference estimator, conditionally unbiased);
    post=True: the correction is post-stratified by the edge's flag cell (y_p, y_c), sum_h (N_h / N^_h) sum_{S_h} (H - H~) / pi
    with N^_h = sum_{S_h} 1 / pi (Hajek within cells), so a model that is miscalibrated within a flag cell costs nothing beyond
    the stratified design. Variance: Poisson, with residuals centered within cells when post=True.
    Round E5. cost: label sharing; an edge's cost is its number of ends not yet labeled before its fold (calibrated_pi).
    tune: power tuning; fold j uses the predictions lambda_m H~ for metric m (lambda_g on the four gold totals, lambda_q on the
    four placebo totals), with lambda_m in [0, 1] the within-cell weighted slope of IF_m(H) on IF_m(H~) over the edges of the
    earlier folds, each with the prediction made before it was drawn (honest; at least tune_min edges, else lambda = 1).
    lambda is fixed before fold j's draw, so every fold's term stays conditionally unbiased; lambda = 0 is the post-stratified
    HT estimator. The design's sd for metric m becomes sqrt(s2_m + (1 - lambda_m)^2 (IF_m(H~) - its cell mean)^2)."""
    Ed = len(cp.par); lev_p = cp.lu[cp.eu]
    pcell = lev_p * 2 + cp.yc.astype(int) if pilot_by == "cell" else lev_p
    ids, inv = np.unique(pcell, return_inverse=True); L = len(ids)
    members = [np.flatnonzero(inv == l) for l in range(L)]; NL = np.array([len(m) for m in members], float)
    n1 = int(min(max(round(pilot_share * budget / 2), 2 * L), Ed)); alloc = equal_split(NL, n1)
    pe = np.concatenate([m[rng.choice(len(m), int(k), replace=False)] for m, k in zip(members, alloc) if k > 0])
    inp = np.zeros(Ed, bool); inp[pe] = True
    M = int(max(cp.par.max(), cp.child.max())) + 1
    known = np.zeros(M, bool); known[cp.par[pe]] = True; known[cp.child[pe]] = True
    folds = np.array_split(rng.permutation(np.flatnonzero(~inp)), K)
    Hp = E.contributions(z[cp.par[pe]], z[cp.child[pe]], cp.yp[pe], cp.yc[pe])
    wl = (NL / np.maximum(alloc, 1))[inv[pe]]; T_pil = np.maximum((wl[:, None] * Hp).sum(0), 0.5 * Ed / max(len(pe), 1))
    fcell = (2 * cp.yp + cp.yc).astype(int)
    if delta == "auto":                                                       # round E2: delta = max(0.1, share of the comments in edges)
        n_in = int(np.unique(np.r_[cp.par, cp.child]).size); delta = float(min(0.9, max(DELTA, budget / max(n_in, 1))))
    T_exact = Hp.sum(0); fold_T = np.zeros(8); labeled_e = pe.copy(); store = []; lams = []; honest = []
    for j, F in enumerate(folds):
        if law_fixed is not None:                                             # diagnostic: an oracle edge law, no learning
            a, r1, r0 = (np.clip(np.asarray(v, float), 1e-4, 1 - 1e-4) for v in law_fixed)
        else:
            a, r1, r0 = edge_law(Xi, cp.par, cp.child, labeled_e, z, lam=lam, parent_fit=parent_fit, bayes=bayes)
        later = np.concatenate(folds[j:])
        Ht_all, _ = edge_moments(a[later], r1[later], r0[later], cp.yp[later], cp.yc[later], T_pil)
        Trun = np.maximum(T_exact + fold_T + Ht_all.sum(0), 0.5)
        Ht, S2 = edge_moments(a[F], r1[F], r0[F], cp.yp[F], cp.yc[F], Trun)
        left = max((budget - int(known.sum())) / (K - j), 1.0)
        lam_j = np.ones(2)
        if tune and sum(len(hh[0]) for hh in honest) >= tune_min:
            for jm in range(2):
                num = den = 0.0
                for Hs_, Hu_, piS_, c_ in honest:                             # honest: the prediction made before the edge was drawn
                    y_ = E.influence(Hs_, Trun)[jm]; x_ = E.influence(Hu_, Trun)[jm]; w_ = 1 / piS_
                    for h in np.unique(c_):
                        mk = c_ == h; ww = w_[mk]; xc = x_[mk] - np.average(x_[mk], weights=ww); yc_ = y_[mk] - np.average(y_[mk], weights=ww)
                        num += float((ww * xc * yc_).sum()); den += float((ww * xc * xc).sum())
                lam_j[jm] = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0
            if (lam_j < 1).any():
                cF = fcell[F]
                for jm in range(2):
                    v_ = E.influence(Ht, Trun)[jm]; mh = np.bincount(cF, weights=v_, minlength=4) / np.maximum(np.bincount(cF, minlength=4), 1)
                    S2[jm] = S2[jm] + (1 - lam_j[jm]) ** 2 * (v_ - mh[cF]) ** 2
        lams.append(lam_j.copy()); LAM = np.r_[np.full(4, lam_j[0]), np.full(4, lam_j[1])]
        costF = ((~known[cp.par[F]]).astype(int) + (~known[cp.child[F]]).astype(int)) if cost else None
        piF = calibrated_pi([S2[0], S2[1]], mult, cp.par[F], cp.child[F], known, M, left, delta=delta, norm=norm, guard=guard,
                            floor_cells=fcell[F] if floor == "cells" else None, cost=costF)
        sk = systematic(piF, rng); S = F[sk]; piS = piF[sk]
        Hs = E.contributions(z[cp.par[S]], z[cp.child[S]], cp.yp[S], cp.yc[S])
        if tune:
            honest.append((Hs, Ht[sk].copy(), piS, fcell[S]))
        Ht = LAM * Ht; R = Hs - Ht[sk]
        if post:
            cS = fcell[S]; NF = np.bincount(fcell[F], minlength=4).astype(float); Nhat = np.bincount(cS, weights=1 / piS, minlength=4)
            f = np.where(Nhat > 0, NF / np.maximum(Nhat, 1e-300), 0.0)
            fold_T = fold_T + Ht.sum(0) + ((f[cS] / piS)[:, None] * R).sum(0)
        else:
            cS = None; f = None; fold_T = fold_T + Ht.sum(0) + (R / piS[:, None]).sum(0)
        store.append((Hs, Ht[sk], piS, cS, f)); known[cp.par[S]] = True; known[cp.child[S]] = True; labeled_e = np.r_[labeled_e, S]
    T = T_exact + fold_T

    def resid_var(u, piS, cS, f):
        if cS is None:
            return float(((1 - piS) / piS ** 2 * u * u).sum())
        num = np.bincount(cS, weights=u / piS, minlength=4); den = np.bincount(cS, weights=1 / piS, minlength=4)
        ub = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0); uc = (u - ub[cS]) * f[cS]
        return float(((1 - piS) / piS ** 2 * uc * uc).sum())
    lg, lq_ = E.log_metrics(T); out = {}
    for jm, (name, th) in enumerate(zip(E.METRICS, (lg, lq_))):
        v = 0.0
        for Hs, Hts, piS, cS, f in store:
            v += resid_var(E.influence(Hs, T)[jm] - E.influence(Hts, T)[jm], piS, cS, f)
        out[name] = (th, v)
    props = []
    for num, den in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        if T[den] <= 0:
            props.append((np.nan, 0.0, 0)); continue
        p_ = T[num] / T[den]; v = 0.0
        for Hs, Hts, piS, cS, f in store:
            r = Hs - Hts; v += resid_var((r[:, num] - p_ * r[:, den]) / T[den], piS, cS, f)
        H_lab = np.concatenate([Hp] + [h for h, *_ in store]); props.append((p_, v, int((H_lab[:, den] > 0).sum())))
    out["_props"] = props; out["_labels"] = int(known.sum()); out["_labeled"] = np.flatnonzero(known); out["_lam"] = np.array(lams)
    return out
