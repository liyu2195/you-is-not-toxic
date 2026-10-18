"""THREADSAFE, Algorithm 1: the share of conversation units with at least one positive comment.

Units are reply threads, article comment sections, the first t comments of a thread (derailment by comment t) or users.
The LLM scores every comment before any gold label is bought. The LLM decides where to look; gold decides the answer.

    For each unit, in random order:
      search  - label its comments in LLM-score order down to the checkpoint, stop at the first gold positive;
      verify  - if none was found, with probability q keep labeling in the same order, stop at the first positive;
    then estimate the share with the design-unbiased unit term and the LLM as a control variate.

q is 1/2 for a first stage of 10-50 units. Afterwards q = 1 if the first stage shows that hidden positives are at least
as common as the residual variance (Proposition 4: A <= 0), based on at least 10 observed positive units.

From 100 sampled units on, the unit term is Theorem 2's output under a working law learned from the audit's own labels:
one hazard P(positive | every earlier comment negative) per (LLM level, search depth) cell, cross-fitted over 5 folds of
the sampled units, so that no unit's term uses a law fitted on its own labels.
"""
from dataclasses import dataclass
import numpy as np
from .intervals import unit_share_interval, Z95


@dataclass(frozen=True)
class Settings:
    checkpoint_share: float = 0.40      # the search covers the comments whose LLM score is in the top 40%
    ordinal_levels: int = 6             # scores with at most this many distinct values are treated as answer levels
    q: float = 0.5                      # verification rate (Proposition 4)
    stage1_share: float = 0.40          # first stage: up to 40% of the budget ...
    stage1_min_units: int = 10          # ... and between 10 ...
    stage1_max_units: int = 50          # ... and 50 units
    switch_min_positive_units: int = 10 # q = 1 only if the first stage observed >= 10 positive units
    vector_cv_min_units: int = 100      # the four-feature control variate needs >= 100 sampled units
    min_units: int = 2
    census_completion: bool = True
    learned_output: bool = True         # Theorem 2's output with the cross-fitted working law (from vector_cv_min_units on)
    crossfit_folds: int = 5
    depth_edges: tuple = (1, 4, 16)     # search-depth buckets 0 | 1-3 | 4-15 | 16+
    law_prior_weight: float = 2.0
    z: float = Z95
    interval_min_count: int = 10


def checkpoint(score, share=0.40, ordinal_levels=6):
    """The lowest score level with at most `share` of all comments at or above it.
    For an ordinal answer (few distinct scores) it is a level; if the top level alone holds more than `share`, it is
    the top level. For a continuous score it is the (1 - share) quantile. Returns (tau, at_or_above)."""
    lv = np.round(score, 4)
    levels = np.unique(lv)
    if len(levels) <= ordinal_levels:
        share_ge = np.array([(lv >= L).mean() for L in levels])
        ok = levels[share_ge <= share]
        tau = float(ok.min()) if len(ok) else float(levels.max())
        return tau, lv >= tau
    tau = float(np.quantile(score, 1.0 - share))
    return tau, score >= tau


class Population:
    """The LLM's view of the population, known before any gold label is bought.

    unit  : unit id of each comment;  score : LLM score;  flag : LLM label (0/1), used for the plug-in;
    comment_id : what the oracle is asked about (default: row index);
    order_key  : optional within-unit ordering key (default: score, ties broken at random).
    """

    def __init__(self, unit, score, flag, comment_id=None, order_key=None, settings=Settings(), rng=None):
        unit = np.asarray(unit)
        score = np.asarray(score, dtype=float)
        flag = np.asarray(flag).astype(int)
        cid = np.arange(len(unit)) if comment_id is None else np.asarray(comment_id)
        if order_key is None:
            tie = (np.random.default_rng() if rng is None else rng).random(len(unit))
            idx = np.lexsort((-tie, -score, unit))
        else:
            idx = np.lexsort((-np.asarray(order_key, dtype=float), unit))
        self.settings = settings
        self.tau, inC = checkpoint(score, settings.checkpoint_share, settings.ordinal_levels)
        u_sorted = unit[idx]
        self.unit_ids, start, k = np.unique(u_sorted, return_index=True, return_counts=True)
        self.N = len(self.unit_ids)
        self.start, self.k = start, k
        self.comment_id = cid[idx]                                   # comments of unit u: comment_id[start[u]:start[u]+k[u]]
        self.n_search = np.add.reduceat(inC[idx].astype(int), start)  # comments at or above the checkpoint
        fl, sc = flag[idx], score[idx]
        plug = np.maximum.reduceat(fl, start).astype(float)           # the LLM's answer for the unit
        nfl = np.minimum(np.add.reduceat(fl, start), 5).astype(float)
        smax = np.maximum.reduceat(sc, start)
        self.X1 = plug[:, None]
        self.X4 = np.column_stack([plug, nfl, np.log(k), smax])
        self.cv = {1: (self.X1.mean(0), np.atleast_2d(np.cov(self.X1.T))), 4: (self.X4.mean(0), np.cov(self.X4.T))}
        self.plugin = float(plug.mean())
        lvr = np.round(sc, 4); uq = np.unique(lvr)                        # LLM levels (0 = highest) for the working law
        if len(uq) <= settings.ordinal_levels: lev = len(uq) - 1 - np.searchsorted(uq, lvr)
        else:
            qs = np.unique(np.quantile(sc, [0.6, 0.8, 0.9, 0.95])); lev = len(qs) - np.searchsorted(qs, sc, side="right")
        self.n_levels = int(lev.max() + 1); nb = len(settings.depth_edges) + 1; self.n_seg = self.n_levels * nb; self.n_depth = nb
        pos = np.arange(len(idx)) - np.repeat(start, k)
        self.seg = lev * nb + np.digitize(pos, settings.depth_edges)    # segment of each comment, in the unit's order

    def comments(self, u):
        s = self.start[u]
        return self.comment_id[s:s + self.k[u]]


def ppi_mean(t, X, mu, Sig, N):
    """PPI++ mean of t with control variates X whose population mean mu and covariance Sig are known."""
    n = len(t)
    if n < X.shape[1] + 3:
        return float(t.mean()), float(t.std(ddof=1) / np.sqrt(n))
    Xc = X - X.mean(0)
    Cxx = Xc.T @ Xc / (n - 1)
    Cxt = Xc.T @ (t - t.mean()) / (n - 1)
    b = np.linalg.lstsq(Cxx * (1 + n / N) + 1e-10 * np.eye(len(mu)), Cxt, rcond=None)[0]
    est = t.mean() - (X.mean(0) - mu) @ b
    res = t - X @ b
    return float(est), float(np.sqrt(np.var(res, ddof=1) / n + b @ Sig @ b / N))


@dataclass
class Result:
    estimate: float
    se: float
    lo: float
    hi: float
    interval: str        # "wald", "kg" or "census"
    n_units: int
    labels: int          # gold labels of the units in the estimate (and of census completion)
    labels_spent: int    # all gold labels bought, including the unit that crossed the budget
    q_stage2: float      # verification rate after the first stage (None if the first stage did not fit the budget)
    census: bool


def _iso_dec(v, w):
    """Weighted isotonic regression, nonincreasing (pool adjacent violators)."""
    bl = []
    for i in range(len(v)):
        bl.append([v[i] * w[i], w[i], 1])
        while len(bl) > 1 and bl[-2][0] / bl[-2][1] < bl[-1][0] / bl[-1][1]:
            a, b = bl.pop(), bl.pop(); bl.append([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
    return np.concatenate([[b[0] / b[1]] * b[2] for b in bl])


def law_from_counts(pop, lab, pc, a=2.0):
    """Hazards per (level, depth) cell from labeled-comment counts: Beta-smoothed toward the level rate (prior weight a),
    isotonic (nonincreasing) in depth within a level and in level within a depth bucket."""
    L_, nb = pop.n_levels, pop.n_depth
    tot = (pc.sum() + 1) / (lab.sum() + 2); labL, pcL = lab.reshape(L_, nb).sum(1), pc.reshape(L_, nb).sum(1)
    lr_ = _iso_dec((pcL + a * tot) / (labL + a), labL + a)
    Hh = ((pc + a * np.repeat(lr_, nb)) / (lab + a)).reshape(L_, nb); W = (lab + a).reshape(L_, nb)
    for l_ in range(L_): Hh[l_] = _iso_dec(Hh[l_], W[l_])
    for b_ in range(nb): Hh[:, b_] = _iso_dec(Hh[:, b_].copy(), W[:, b_].copy())
    return np.clip(Hh.ravel(), 1e-4, 1 - 1e-4)


def _labeled(pop, r):
    """The comments a sampled unit labeled, in its order: positions, gold values and gates (1 in the search, q_u beyond)."""
    u, S, R, H, q_u, c1, c2 = r[0], r[1], r[2], r[3], r[4], r[6], r[7]
    nS = int(pop.n_search[u]); pos = list(range(c1)); z = [0.0] * c1
    if S: z[-1] = 1.0
    if R:
        pos += list(range(nS, nS + c2)); zz = [0.0] * c2
        if H: zz[-1] = 1.0
        z += zz
    pos = np.array(pos, int); return pos, np.array(z), np.where(pos < nS, 1.0, q_u)


def learned_terms(pop, rec, s):
    """Theorem 2's output for each sampled unit, T_u = mu_u + sum over labeled comments of (A_u / (A_ui G_i)) (Z_i - h_i), with
    A_ui = prod_{j <= i} (1 - h_j) along the unit's order, under the working law fitted on the other folds
    (fold = position in the sampling order mod K). E[T_u | gold] = 1{unit has a positive} for any such law."""
    K = s.crossfit_folds; L = [_labeled(pop, r) for r in rec]
    cells = [pop.seg[pop.start[r[0]]:pop.start[r[0]] + pop.k[r[0]]] for r in rec]
    T = np.empty(len(rec))
    for f in range(K):
        lab = np.zeros(pop.n_seg); pc = np.zeros(pop.n_seg)
        for i in range(len(rec)):
            if i % K != f:
                c = cells[i][L[i][0]]; np.add.at(lab, c, 1.0); np.add.at(pc, c, L[i][1])
        p = law_from_counts(pop, lab, pc, s.law_prior_weight)
        for i in range(f, len(rec), K):
            h = p[cells[i]]; cum = np.cumsum(np.log1p(-h)); tot = cum[-1]; ps, z, G = L[i]
            T[i] = 1 - np.exp(tot) + float(np.sum(np.exp(tot - cum[ps]) * (z - h[ps]) / G))
    return T


def _stage1_q(rec, X1, s):
    """Proposition 4's rule from the first stage: q = 1 if A <= 0, with h > 0 and >= 10 observed positive units."""
    q = s.q
    S = np.array([r[1] for r in rec], float); R = np.array([r[2] for r in rec], float); H = np.array([r[3] for r in rec], float)
    t = S + (1 - S) * (R / q) * H
    x = X1[[r[0] for r in rec], 0]
    h = float(((1 - S) * R * H / q).mean())                                  # hidden positives, unbiased
    lam = np.cov(t, x)[0, 1] / np.var(x, ddof=1) if np.var(x) > 0 else 0.0
    A = float(np.var(t - lam * x, ddof=1)) - h * (1 / q - 1) - h             # residual variance minus h
    positives = float(S.sum() + (R * H).sum())
    return 1.0 if (h > 0 and A <= 0 and positives >= s.switch_min_positive_units) else s.q


def run(pop, oracle, budget, rng=None, order=None, coins=None, settings=None):
    """Run the design once. `oracle(comment_id)` returns the gold label (0/1) and is called once per label bought.
    `order` (a permutation of units) and `coins` (one U(0,1) per unit) are drawn from `rng` unless given.
    `settings` overrides the population's settings for the sampling and estimation constants."""
    s, N, M = (pop.settings if settings is None else settings), pop.N, float(budget)
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(N) if order is None else order
    coins = rng.random(N) if coins is None else coins
    spent = 0

    def label(c):
        nonlocal spent
        spent += 1
        return int(oracle(c))

    def verify(ids, nS):
        c2 = 0
        for c in ids[nS:]:
            c2 += 1
            if label(c):
                return 1, c2
        return 0, c2

    rec = []                    # included units: (u, S, R, H, q, cost)
    stage1, q2, open1, cum1, cum = [], None, True, 0.0, 0.0
    for j, u in enumerate(order, start=1):
        ids, nS = pop.comments(u), int(pop.n_search[u])
        S, c1 = 0, 0
        for c in ids[:nS]:                                         # search to the checkpoint
            c1 += 1
            if label(c):
                S = 1
                break
        R, H, c2 = 0, 0, 0
        if not S and coins[u] < s.q:                               # verification at the first-stage rate
            R = 1
            H, c2 = verify(ids, nS)
        q_u = s.q
        if open1:
            if j <= s.stage1_min_units or (j <= s.stage1_max_units and cum1 + c1 + c2 <= s.stage1_share * M):
                cum1 += c1 + c2
                stage1.append((u, S, R, H))
            else:                                                  # the first stage is complete
                open1 = False
                q2 = _stage1_q(stage1, pop.X1, s) if cum1 <= M else None
        if not open1 and q2 is not None:
            q_u = q2
            if not S and not R and coins[u] < q_u:                 # q = 1 after the switch
                R = 1
                H, c2 = verify(ids, nS)
        cum += c1 + c2
        if cum > M and j > s.min_units:                            # this unit crosses the budget: not used
            break
        rec.append((u, S, R, H, q_u, c1 + c2, c1, c2))
        if cum > M and j >= s.min_units:
            break
    if open1 and len(rec) == N and cum1 <= M:                      # every unit fell in the first stage
        q2 = _stage1_q(stage1, pop.X1, s)
    used = sum(r[5] for r in rec)
    n = len(rec)
    if s.census_completion and n == N:                             # census completion
        todo = [r for r in rec if r[1] == 0 and r[2] == 0]
        if used + sum(int(pop.k[r[0]] - pop.n_search[r[0]]) for r in todo) <= M:
            Y = {r[0]: max(r[1], r[3]) for r in rec}
            extra = 0
            for r in todo:
                H, c2 = verify(pop.comments(r[0]), int(pop.n_search[r[0]]))
                Y[r[0]] = H
                extra += c2
            theta = float(np.mean(list(Y.values())))
            return Result(theta, 0.0, theta, theta, "census", n, used + extra, spent, q2, True)
    ix = np.array([r[0] for r in rec])
    S = np.array([r[1] for r in rec], float); R = np.array([r[2] for r in rec], float)
    H = np.array([r[3] for r in rec], float); q = np.array([r[4] for r in rec], float)
    t = S + (1 - S) * (R / q) * H                                  # E[t | unit] = 1{unit has a positive}, for any LLM
    p = 4 if n >= s.vector_cv_min_units else 1
    if s.learned_output and n >= s.vector_cv_min_units:
        t = learned_terms(pop, rec, s)                             # Theorem 2's output; still E[t | unit] = 1{positive}
    X = (pop.X4 if p == 4 else pop.X1)[ix]
    est, se = ppi_mean(t, X, *pop.cv[p], N)
    lo, hi, kind = unit_share_interval(est, se, n, s.z, s.interval_min_count)
    return Result(est, se, lo, hi, kind, n, int(used), spent, q2, False)
