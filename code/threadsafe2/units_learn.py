"""THREADSAFE unit regime, round U2: Algorithm 1 with a search order learned from the audit's own first stage.

Stage 1 (Algorithm 1's first stage: 10-50 units, at most 40% of the budget) searches units in a committed default order
(round U1: the combined percentile-rank score). At its end, a ridge logistic model of the gold label on the instruments'
percentile ranks and the labeler's flag is fitted to every stage-1 labeled comment; its linear predictor is the order (and
the checkpoint, hazard levels and depth cells) for every later unit. Each unit's order is fixed before any of its comments
is labeled, so every unit term keeps E[t_u | gold] = 1{unit has a positive}.
Estimation: stage-1 units use the simple term S + (1 - S)(R/q)H; later units use Theorem 2's output with the hazard law
cross-fitted over later units only (all searched in the same learned order), when there are at least 100 of them. The control
variates are the default population's (committed). PPI++ and the interval as in Algorithm 1.
"""
import numpy as np
from .units import Population, Result, ppi_mean, learned_terms, _stage1_q
from .intervals import unit_share_interval
from .pps import logit_fit


def learned_score(F, labeled_ids, labels, lam=1.0):
    """Linear predictor of a ridge logistic model of z on the feature matrix F (comments x features, with an intercept)."""
    X = np.column_stack([np.ones(len(F)), F])
    beta = logit_fit(X[labeled_ids], np.asarray(labels, float), lam=lam)
    return X @ beta


def run(popA, F, unit, flag, oracle, budget, rng=None, settings=None, tie=None):
    """One audit. popA: the default population (committed order and control variates); F: per-comment features for the
    learned order; unit, flag: per-comment unit ids and labeler flags (as given to popA); tie: per-comment tie breakers."""
    s, N, M = (popA.settings if settings is None else settings), popA.N, float(budget)
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(N); coins = rng.random(N)
    spent = 0; lab_ids, lab_z = [], []

    def label(c):
        nonlocal spent
        spent += 1; v = int(oracle(c)); lab_ids.append(c); lab_z.append(v)
        return v

    def verify(ids, nS):
        c2 = 0
        for c in ids[nS:]:
            c2 += 1
            if label(c):
                return 1, c2
        return 0, c2

    cur = popA; popB = None
    rec = []; which = []          # which population each included unit was searched with
    stage1, q2, open1, cum1, cum = [], None, True, 0.0, 0.0
    for j, u in enumerate(order, start=1):
        ids, nS = cur.comments(u), int(cur.n_search[u])
        S, c1 = 0, 0
        for c in ids[:nS]:
            c1 += 1
            if label(c):
                S = 1
                break
        R, H, c2 = 0, 0, 0
        if not S and coins[u] < s.q:
            R = 1
            H, c2 = verify(ids, nS)
        q_u = s.q
        if open1:
            if j <= s.stage1_min_units or (j <= s.stage1_max_units and cum1 + c1 + c2 <= s.stage1_share * M):
                cum1 += c1 + c2
                stage1.append((u, S, R, H))
            else:
                open1 = False
                q2 = _stage1_q(stage1, popA.X1, s) if cum1 <= M else None
        if not open1 and q2 is not None:
            q_u = q2
            if not S and not R and coins[u] < q_u:
                R = 1
                H, c2 = verify(ids, nS)
        cum += c1 + c2
        if cum > M and j > s.min_units:
            break
        rec.append((u, S, R, H, q_u, c1 + c2, c1, c2)); which.append(0 if cur is popA else 1)
        if cum > M and j >= s.min_units:
            break
        if not open1 and popB is None:                          # stage 1 just ended: learn the order for every later unit
            sc = learned_score(F, np.array(lab_ids), lab_z)
            popB = Population(unit, sc, flag, order_key=sc + 1e-9 * (tie if tie is not None else 0.0), settings=s)
            cur = popB
    n = len(rec)
    used = sum(r[5] for r in rec)
    if s.census_completion and n == N:                              # census completion, as in Algorithm 1
        pops = [popA if w == 0 else popB for w in which]
        todo = [(r, pp) for r, pp in zip(rec, pops) if r[1] == 0 and r[2] == 0]
        if used + sum(int(pp.k[r[0]] - pp.n_search[r[0]]) for r, pp in todo) <= M:
            Y = {r[0]: max(r[1], r[3]) for r in rec}; extra = 0
            for r, pp in todo:
                H, c2 = verify(pp.comments(r[0]), int(pp.n_search[r[0]])); Y[r[0]] = H; extra += c2
            theta = float(np.mean(list(Y.values())))
            return Result(theta, 0.0, theta, theta, "census", n, used + extra, spent, q2, True)
    ix = np.array([r[0] for r in rec])
    S_ = np.array([r[1] for r in rec], float); R_ = np.array([r[2] for r in rec], float)
    H_ = np.array([r[3] for r in rec], float); q = np.array([r[4] for r in rec], float)
    t = S_ + (1 - S_) * (R_ / q) * H_
    wh = np.array(which)
    later = np.flatnonzero(wh == 1)
    if s.learned_output and len(later) >= s.vector_cv_min_units:
        t[later] = learned_terms(popB, [rec[i] for i in later], s)
    p = 4 if n >= s.vector_cv_min_units else 1
    X = (popA.X4 if p == 4 else popA.X1)[ix]
    est, se = ppi_mean(t, X, *popA.cv[p], N)
    lo, hi, kind = unit_share_interval(est, se, n, s.z, s.interval_min_count)
    return Result(est, se, lo, hi, kind, n, int(used), spent, q2, False)
