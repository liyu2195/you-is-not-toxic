"""THREADSAFE item regime, round A (Addendum 62): two-phase sampling with probability proportional to influence.

Phase 1, pilot: n1 = 15% of the budget, split equally over the labeler's score levels (capped at level sizes). Working model:
p_l = the level's Jeffreys-smoothed pilot gold rate. Plug-ins for nonlinear metrics come from the pilot (Horvitz-Thompson).
Phase 2, main: systematic PPS in random order over the items not in the pilot, n2 = B - n1, with inclusion probabilities
    pi_i = (1 - delta) * min(1, c * sigma_i) + delta * n2 / N2,   sigma_i^2 = p_i (1 - p_i) * sum_m w_m phi_im^2,
water-filled to sum to n2 (delta = 0.1 defensive uniform share). w_m = mult_m / (sum_i sigma_im)^2 (each metric against its own
optimum; laws-first mult as in Addendum 58).
Estimator (difference estimator, conditionally unbiased given the pilot for every linear metric):
    T = sum_pilot z_i B_i + sum_{i not in pilot} p_i B_i + sum_main (z_i - p_i) B_i / pi_i,   theta = theta(T)
Variance: Poisson approximation sum_main (1 - pi_i) / pi_i^2 e_i^2 with e_i = (z_i - p_i) phi_i at the final plug-ins.
"""
import numpy as np
from .laws import equal_split, Z95

DELTA = 0.1


def score_levels(s, max_levels=8):
    """The labeler's score as at most 8 ordered levels: its own levels if few, else quantile bins refined at the top."""
    s = np.asarray(s, float); s = np.where(np.isnan(s), np.nanmin(s), s); u = np.unique(s)
    if len(u) <= max_levels:
        return np.searchsorted(u, s)
    q = np.unique(np.quantile(s, [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995]))
    return np.searchsorted(q, s, side="right")


def _waterfill(sig, n, floor):
    """pi = min(1, a sigma + floor) summing to n (a >= 0), by sorting: with the k largest capped,
    a_k = (n - k - (N - k) floor) / sum of the rest; take the smallest k whose next item stays below the cap."""
    sig = np.asarray(sig, float); N = len(sig)
    if N == 0:
        return sig
    if sig.sum() <= 0 or n >= N:
        return np.full(N, min(1.0, n / max(N, 1)))
    order = np.argsort(-sig, kind="stable"); ss = sig[order]
    rest = np.r_[np.cumsum(ss[::-1])[::-1], 0.0]                          # rest[k] = sum of ss[k:]
    k = np.arange(N + 1); with_room = rest[:N] > 0
    a = np.where(with_room, (n - k[:N] - (N - k[:N]) * floor) / np.where(with_room, rest[:N], 1.0), np.inf)
    ok = with_room & (a >= 0) & (a * ss + floor < 1.0)                  # item k (0-based) is the first uncapped one
    kk = int(np.argmax(ok)) if ok.any() else N
    pi_sorted = np.ones(N)
    if kk < N:
        pi_sorted[kk:] = np.minimum(1.0, a[kk] * ss[kk:] + floor)
    pi = np.empty(N); pi[order] = pi_sorted
    return pi


def own_var(s2, n):
    """A metric's own optimum on these items: V* = sum s2 (1/pi - 1) with pi ~ sqrt(s2) capped at 1, sum pi = n (round A4:
    the compromise is normalized by V*, as in the ceiling, instead of the uncapped (sum sigma)^2)."""
    sg = np.sqrt(np.maximum(s2, 0)); tot = sg.sum()
    if tot <= 0 or n <= 0:
        return 0.0
    if n * sg.max() / tot < 1.0:                                              # no cap binds: closed form
        return float(tot * tot / n - s2.sum())
    pi = _waterfill(sg, n, 0.0); ok = pi > 0
    return float((s2[ok] * (1 / pi[ok] - 1)).sum())


def cell_sigma(sig, cells):
    """Each unit's cell RMS of sigma: the per-unit weight of a Neyman allocation over the cells."""
    cells = np.asarray(cells); nc = int(cells.max()) + 1
    m2 = np.bincount(cells, weights=sig * sig, minlength=nc) / np.maximum(np.bincount(cells, minlength=nc), 1)
    return np.sqrt(m2)[cells]


def compromise_pi(sig2, mult, n, delta=0.1, guard=4.0, iters=6, floor_cells=None):
    """Inclusion probabilities for a metric set (round A4): pi = waterfill((1 - delta) * sqrt(sum_m w_m s2_m), n, delta * n / N)
    with w_m = mult_m / V*_m (capped own optimum). Guard: if a metric's model variance exceeds `guard` times its own optimum,
    its weight is doubled (at most `iters` times). Returns (pi, weights, ratios).
    floor_cells (round A6): the defensive share is a stratified design instead of a uniform one,
    pi = (1 - delta) waterfill(sigma) + delta waterfill(cell RMS sigma), both of size n."""
    N = len(sig2[0]); own = np.array([own_var(s2, n) for s2 in sig2]); mult = np.asarray(mult, float)
    w = mult / np.maximum(own, 1e-300); floor = delta * n / max(N, 1)
    for _ in range(iters + 1):
        sig = np.sqrt(sum(wi * s2 for wi, s2 in zip(w, sig2)))
        if floor_cells is not None:
            pi = (1 - delta) * _waterfill(sig, n, 0.0) + delta * _waterfill(np.maximum(cell_sigma(sig, floor_cells), 1e-12), n, 0.0)
        else:
            pi = _waterfill((1 - delta) * sig, n, floor)
        ok = pi > 0
        R = np.array([float((s2[ok] * (1 / pi[ok] - 1)).sum()) / max(o, 1e-300) if o > 0 else 1.0 for s2, o in zip(sig2, own)])
        bad = R > guard if guard else np.zeros(len(R), bool)
        if not bad.any():
            break
        w = np.where(bad, 2 * w, w)
    return pi, w, R


def systematic(pi, rng):
    """Systematic PPS in random order: a fixed-size sample (sum pi, rounded) with inclusion probabilities pi."""
    perm = rng.permutation(len(pi)); cs = np.cumsum(pi[perm]); u = rng.random()
    k = np.floor(cs - u); prev = np.r_[np.floor(-u), k[:-1]]
    return perm[k > prev]


def aux_feature(sc):
    """An auxiliary instrument's score as a model feature: logit for probabilities, standardized otherwise."""
    sc = np.asarray(sc, float); sc = np.where(np.isnan(sc), np.nanmedian(sc), sc)
    if sc.min() >= 0 and sc.max() <= 1:
        x = np.log(np.clip(sc, 1e-4, 1 - 1e-4) / (1 - np.clip(sc, 1e-4, 1 - 1e-4)))
    else:
        x = sc
    return (x - x.mean()) / max(x.std(), 1e-9)


def design_matrix(levels, depth, logk, art=None, aux=(), replies=None):
    """Working-model features: score-level indicators, depth-bucket indicators (0 as baseline), centered log size and log article
    size, (round A3) auxiliary instruments' scores, committed before the audit like the labeler's own, and (round A4) the
    comment's number of replies (a leaf indicator and standardized log(1 + replies))."""
    lv = np.asarray(levels); L_ = int(lv.max()) + 1; cols = [(lv == l).astype(float) for l in range(L_)]
    db = np.searchsorted([1, 2, 3, 5], np.asarray(depth), side="right"); cols += [(db == b).astype(float) for b in range(1, 5)]
    lk = np.asarray(logk, float); cols.append((lk - lk.mean()) / max(lk.std(), 1e-9))
    if art is not None:
        la = np.log2(np.maximum(np.asarray(art, float), 1)); cols.append((la - la.mean()) / max(la.std(), 1e-9))
    for sc in aux:
        cols.append(aux_feature(sc))
    if replies is not None:
        r = np.asarray(replies, float); lr = np.log1p(r)
        cols += [(r == 0).astype(float), (lr - lr.mean()) / max(lr.std(), 1e-9)]
    return np.column_stack(cols)


def logit_fit(X, y, lam=1.0, iters=30, return_hessian=False):
    """Ridge-penalized logistic regression by IRLS (columns of X are the features; no separate intercept)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta; p = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); w = np.maximum(p * (1 - p), 1e-6)
        H = X.T @ (w[:, None] * X) + lam * np.eye(X.shape[1]); g = X.T @ (y - p) - lam * beta
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-6:
            break
    if return_hessian:
        eta = X @ beta; p = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); w = np.maximum(p * (1 - p), 1e-6)
        return beta, X.T @ (w[:, None] * X) + lam * np.eye(X.shape[1])
    return beta


def predictive_p(X, beta, H):
    """Posterior-predictive probabilities of the ridge logistic model (round A4b): with eta ~ N(x'beta, x'H^-1 x) under the
    Laplace approximation, P(z = 1) ~ sigmoid(x'beta / sqrt(1 + pi s^2 / 8)). Extreme predictions far from the labeled data
    are tempered toward 1/2, so the design does not treat them as certain."""
    eta = X @ beta; Hi = np.linalg.inv(H); s2 = ((X @ Hi) * X).sum(1)
    return 1 / (1 + np.exp(-np.clip(eta / np.sqrt(1 + np.pi * np.maximum(s2, 0) / 8), -30, 30)))


def run(metrics, levels, budget, rng, z, mult=None, pilot_share=0.15, X=None, lam=1.0, pilot_cells=None, norm="uncapped", guard=None, delta=DELTA):
    """One audit. levels: the labeler's score level per item (the working model's cells). pilot_cells: optional finer cells
    (e.g. level x depth bucket) over which the pilot is split equally (round A2). Returns {metric: (theta, var)}."""
    N = len(z); levels = np.asarray(levels); ids, inv = np.unique(levels, return_inverse=True); L = len(ids)
    if pilot_cells is not None:
        _, pinv = np.unique(np.asarray(pilot_cells), return_inverse=True)
        pmembers = [np.flatnonzero(pinv == c) for c in range(pinv.max() + 1)]; PN = np.array([len(m) for m in pmembers], float)
    members = [np.flatnonzero(inv == l) for l in range(L)]; NL = np.array([len(m) for m in members], float)
    n1 = int(min(max(round(pilot_share * budget), 2 * L), budget - 1))
    if pilot_cells is None:
        alloc = equal_split(NL, n1)
        pil = [m[rng.choice(len(m), int(a), replace=False)] if a > 0 else m[:0] for m, a in zip(members, alloc)]
        pilot = np.concatenate(pil)
    else:
        palloc = equal_split(PN, n1)
        pilot = np.concatenate([m[rng.choice(len(m), int(a), replace=False)] for m, a in zip(pmembers, palloc) if a > 0])
        pil = [pilot[inv[pilot] == l] for l in range(L)]
    in_pilot = np.zeros(N, bool); in_pilot[pilot] = True
    pl = np.array([(z[s].sum() + 0.5) / (len(s) + 1.0) for s in pil])            # working model per level
    p = pl[inv]
    if X is not None:                                                         # round A2: logistic working model on the pilot
        # pilot items are weighted back to the population by level (N_l / n_l), so the model targets population rates
        # sampling depends only on the level, which is a regressor, so the unweighted fit targets P(z | features)
        Xp = X[pilot]; yp_ = z[pilot]; wts = np.ones(len(pilot))
        beta = np.zeros(X.shape[1])
        for _ in range(30):
            eta = Xp @ beta; pr_ = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); ww = wts * np.maximum(pr_ * (1 - pr_), 1e-6)
            H = Xp.T @ (ww[:, None] * Xp) + lam * np.eye(X.shape[1]); g = Xp.T @ (wts * (yp_ - pr_)) - lam * beta
            st = np.linalg.solve(H, g); beta += st
            if np.abs(st).max() < 1e-6:
                break
        p = np.clip(1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))), 1e-3, 1 - 1e-3)
    # plug-ins and per-metric sigma at the pilot (HT over levels)
    sig2 = []; own = []
    if pilot_cells is None:
        pw = np.zeros(N); [pw.__setitem__(s_, NL[l] / max(len(s_), 1)) for l, s_ in enumerate(pil) if len(s_)]
    else:
        pw = np.zeros(N); cnt = np.bincount(pinv[pilot], minlength=len(PN)); pw[pilot] = PN[pinv[pilot]] / np.maximum(cnt[pinv[pilot]], 1)
    for m in metrics:
        T0 = ((pw[pilot] * z[pilot])[:, None] * m.B[pilot]).sum(0)
        th0, c0 = m.theta_c(T0)
        if not np.isfinite(th0):
            c0 = np.zeros(m.B.shape[1]); c0[0] = 1.0
        phi = m.B @ c0; s2 = p * (1 - p) * phi * phi; sig2.append(s2); own.append(np.sqrt(s2).sum() ** 2)
    mult = np.ones(len(metrics)) if mult is None else np.asarray(mult, float)
    rest = np.flatnonzero(~in_pilot); n2 = budget - len(pilot)
    if norm == "capped":                                                      # round A4: capped normalization and guard
        pi_r = compromise_pi([s2[rest] for s2 in sig2], mult, n2, delta=delta, guard=guard)[0]
    else:
        w = mult / np.maximum(np.array(own), 1e-300)
        sig = np.sqrt(sum(wi * s2 for wi, s2 in zip(w, sig2)))
        pi_r = _waterfill((1 - delta) * sig[rest], n2, delta * n2 / max(len(rest), 1))
    sel = rest[systematic(pi_r, rng)]; pi = np.zeros(N); pi[rest] = pi_r
    out = {}; df = {}
    for m in metrics:
        T = (z[pilot, None] * m.B[pilot]).sum(0) + (p[rest, None] * m.B[rest]).sum(0) + (((z[sel] - p[sel]) / pi[sel])[:, None] * m.B[sel]).sum(0)
        th, c = m.theta_c(T)
        e = (z[sel] - p[sel]) * (m.B[sel] @ c); a_ = (1 - pi[sel]) / pi[sel] ** 2 * e * e; v = float(a_.sum()); vv = float(((1 - pi[sel]) * a_ * a_).sum())
        out[m.name] = (th, v); df[m.name] = 2 * v * v / vv if vv > 0 else np.inf
    out["_labels"] = int(len(pilot) + len(sel)); out["_df"] = df
    return out


class SortedWaterfill:
    """_waterfill for repeated calls on the same sigma (different n): sorts once."""
    def __init__(self, sig):
        self.sig = np.asarray(sig, float); self.N = len(self.sig); self.order = np.argsort(-self.sig, kind="stable")
        self.ss = self.sig[self.order]; self.rest = np.r_[np.cumsum(self.ss[::-1])[::-1], 0.0]; self.k = np.arange(self.N)

    def __call__(self, n, floor):
        N = self.N
        if self.sig.sum() <= 0 or n >= N:
            return np.full(N, min(1.0, n / max(N, 1)))
        with_room = self.rest[:N] > 0
        a = np.where(with_room, (n - self.k - (N - self.k) * floor) / np.where(with_room, self.rest[:N], 1.0), np.inf)
        ok = with_room & (a >= 0) & (a * self.ss + floor < 1.0)
        kk = int(np.argmax(ok)) if ok.any() else N
        ps = np.ones(N)
        if kk < N:
            ps[kk:] = np.minimum(1.0, a[kk] * self.ss[kk:] + floor)
        pi = np.empty(N); pi[self.order] = ps
        return pi


def run_seq(metrics, levels, budget, rng, z, X, mult=None, pilot_share=0.15, K=3, lam=1.0, pilot_cells=None, delta=DELTA, norm="capped", guard=4.0,
            pfloor=0.0, bayes=False, prelabeled=None, p_fixed=None, floor_cells=None):
    """Round A4: sequential folds. The pilot is drawn as in run(). The non-pilot items are split at random into K folds of
    equal size; fold j is sampled by systematic PPS (pi ~ sigma, defensive share delta) with the working model refitted on every
    label so far (pilot and folds < j), with n_j = (B - n1) / K labels. Estimator:
        T = sum_pilot z B + sum_j [ sum_{F_j} p_j B + sum_{S_j} (z - p_j) B / pi_j ],
    exactly unbiased for every linear metric (each fold's term is conditionally unbiased given the labels before it).
    Variance: the sum of the folds' Poisson variances at the final plug-ins.
    prelabeled (round U1): comments whose gold labels another phase of the audit already bought (the edge regime). They are
    exact, like the pilot, train the working model, and are excluded from the pilot and the folds; `budget` counts new labels."""
    N = len(z); levels = np.asarray(levels); ids, inv = np.unique(levels, return_inverse=True); L = len(ids)
    pre = np.zeros(N, bool)
    if prelabeled is not None and len(prelabeled):
        pre[np.asarray(prelabeled)] = True
    members = [np.flatnonzero((inv == l) & ~pre) for l in range(L)]
    n1 = int(min(max(round(pilot_share * budget), 2 * L), budget - K))
    if pilot_cells is None:
        alloc = equal_split(np.array([len(m) for m in members], float), n1)
        pilot = np.concatenate([m[rng.choice(len(m), int(a), replace=False)] for m, a in zip(members, alloc) if a > 0])
    else:
        _, pinv = np.unique(np.asarray(pilot_cells), return_inverse=True)
        pm = [np.flatnonzero((pinv == c) & ~pre) for c in range(pinv.max() + 1)]
        palloc = equal_split(np.array([len(m) for m in pm], float), n1)
        pilot = np.concatenate([m[rng.choice(len(m), int(a), replace=False)] for m, a in zip(pm, palloc) if a > 0])
    n_new_pilot = len(pilot)
    if delta == "auto":                                                       # round A5: delta = max(0.1, sampling fraction)
        delta = float(min(0.9, max(DELTA, budget / max(N - pre.sum(), 1))))
    if pre.any():
        pilot = np.r_[pilot, np.flatnonzero(pre)]                            # exact contributions, like the pilot
    in_pilot = np.zeros(N, bool); in_pilot[pilot] = True
    rest = rng.permutation(np.flatnonzero(~in_pilot)); folds = np.array_split(rest, K)
    n2 = budget - n_new_pilot; nj = np.full(K, n2 // K); nj[: n2 - nj.sum()] += 1
    mult = np.ones(len(metrics)) if mult is None else np.asarray(mult, float)
    Tp = [(z[pilot, None] * m.B[pilot]).sum(0) for m in metrics]
    fold_T = [np.zeros_like(t) for t in Tp]; labeled = pilot.copy(); store = []
    later_mask = ~in_pilot
    for j, F in enumerate(folds):
        if p_fixed is not None:                                               # diagnostic: an oracle working model, no learning
            p = np.clip(np.asarray(p_fixed, float), 1e-3, 1 - 1e-3)
        elif bayes:
            beta, Hb = logit_fit(X[labeled], z[labeled], lam=lam, return_hessian=True); p = np.clip(predictive_p(X, beta, Hb), 1e-3, 1 - 1e-3)
        else:
            beta = logit_fit(X[labeled], z[labeled], lam=lam)
            p = np.clip(1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))), 1e-3, 1 - 1e-3)
        pl = p * later_mask                                                   # model predictions on folds j, j+1, ...
        sig2 = []; own = []
        for mi, m in enumerate(metrics):
            Trun = Tp[mi] + fold_T[mi] + pl @ m.B
            th0, c0 = m.theta_c(Trun)
            if not np.isfinite(th0):
                c0 = np.zeros(m.B.shape[1]); c0[0] = 1.0
            phi = m.B[F] @ c0; s2 = np.maximum(p[F] * (1 - p[F]), pfloor) * phi * phi; sig2.append(s2); own.append(np.sqrt(s2).sum() ** 2)
        if norm == "capped":
            piF = compromise_pi(sig2, mult, nj[j], delta=delta, guard=guard, floor_cells=None if floor_cells is None else floor_cells[F])[0]
        else:
            w = mult / np.maximum(np.array(own), 1e-300); sig = np.sqrt(sum(wi * s2 for wi, s2 in zip(w, sig2)))
            piF = _waterfill((1 - delta) * sig, nj[j], delta * nj[j] / max(len(F), 1))
        sk = systematic(piF, rng); S = F[sk]; piS = piF[sk]
        for mi, m in enumerate(metrics):
            fold_T[mi] = fold_T[mi] + p[F] @ m.B[F] + ((z[S] - p[S]) / piS) @ m.B[S]
        store.append((S, piS, p[S], F, piF, p[F])); labeled = np.r_[labeled, S]; later_mask[F] = False
    out = {}; df = {}; skew = {}
    for mi, m in enumerate(metrics):
        T = Tp[mi] + fold_T[mi]; th, c = m.theta_c(T); v = 0.0; vv = 0.0; k2 = k3 = cc = 0.0
        for S, piS, pS, F, piF, pF in store:
            e = (z[S] - pS) * (m.B[S] @ c); a_ = (1 - piS) / piS ** 2 * e * e; v += float(a_.sum()); vv += float(((1 - piS) * a_ * a_).sum())
            # model-based cumulants of the fold's error sum_F (I/pi - 1) e, e = (z - p) phi, z ~ Bernoulli(p)
            ph = m.B[F] @ c; m2 = pF * (1 - pF) * ph ** 2; m3 = pF * (1 - pF) * (1 - 2 * pF) * ph ** 3
            k2 += float(((1 - piF) / piF * m2).sum()); k3 += float(((1 - piF) * (1 - 2 * piF) / piF ** 2 * m3).sum())
            cc += float(((1 - piF) ** 2 / piF ** 2 * m3).sum())
        out[m.name] = (th, v); df[m.name] = 2 * v * v / vv if vv > 0 else np.inf      # Satterthwaite effective df of v
        if k2 > 0:
            s3, c3 = k3 / k2 ** 1.5, cc / k2 ** 1.5; zz = Z95 * Z95
            skew[m.name] = float(np.clip(-(s3 / 6) * (zz - 1) + (c3 / 2) * zz, -1.5, 1.5))   # Edgeworth shift h of the studentized quantiles
        else:
            skew[m.name] = 0.0
    out["_labels"] = int(len(labeled) - pre.sum()); out["_df"] = df; out["_h"] = skew
    return out
