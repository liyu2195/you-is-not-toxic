"""THREADSAFE upgrade (Addendum 57), item regime: conversation laws and their placebo versions from a few gold labels.

Every metric is a smooth function of totals T_g = sum_i z_i B_ig of the unknown gold labels z against known item features B
(the labeler's flag y is one of the features, committed before the audit). Its first-order influence is z_i * phi_i with
phi_i = B_i . c(T), so one design-based machinery serves every metric:

  linear metrics (gold laws)  theta = T  for a single known weight column phi (depth slope, f96 size slope)
  placebo slopes              weighted OLS slope of y on x among gold negatives (z = 0), from the totals of a*{1, x, y, x^2, xy}

Designs: 'srs' (uniform items, Horvitz-Thompson), 'ppi' (uniform items, PPI++ with the labeler's version of each metric as the
control variate), 'dstrat' (THREADSAFE's contrast regime generalized: equal allocation over depth buckets, PPI++ inside each),
'flagprop' (flag x structure strata, proportional allocation), and 'upgrade' (flag x structure strata, a 20% pilot split equally,
then a top-up toward the compromise allocation n_h ~ N_h sqrt(sum_m w_m S2_hm), w_m = 1 / (sum_h N_h S_hm)^2).
"""
import os
import numpy as np
# Sensitivity switch (after the confirmation): TS_PPI_FPC=1 tunes PPI++ for a finite population with exactly known totals,
# lambda = Cov / Var, instead of the iid-unlabeled formula Cov / ((1 + n/N) Var) used in the registered runs.
PPI_TUNE = 0.0 if os.environ.get("TS_PPI_FPC") == "1" else 1.0

Z95 = 1.96


# ----------------------------------------------------------------------------------------------------------------- metrics
class Linear:
    """theta = sum_i z_i phi_i with phi known for every item."""
    kind = "linear"

    def __init__(self, name, phi):
        self.name, self.B = name, np.asarray(phi, float)[:, None]
        self.K = self.B.sum(0)

    def theta_c(self, T):
        return float(T[0]), np.array([1.0])


class PlaceboSlope:
    """Weighted OLS slope of the flag y on x among gold negatives, item weights a (a = 0 excludes an item)."""
    kind = "placebo"

    def __init__(self, name, x, y, a):
        x, y, a = (np.asarray(v, float) for v in (x, y, a))
        self.name = name
        self.B = np.column_stack([a, a * x, a * y, a * x * x, a * x * y])
        self.K = self.B.sum(0)

    def theta_c(self, T):
        S0, Sx, Sy, Sxx, Sxy = self.K - T
        if S0 <= 0:
            return np.nan, np.zeros(5)
        xb, yb = Sx / S0, Sy / S0; SS0 = Sxx - Sx * Sx / S0
        if SS0 <= 0:
            return np.nan, np.zeros(5)
        pi = (Sxy - Sx * Sy / S0) / SS0
        c = -np.array([xb * yb - pi * xb * xb, -yb + 2 * pi * xb, -xb, -pi, 1.0]) / SS0
        return float(pi), c


def value(metric, v):
    """The metric with labels v (gold z, or the flags y for the labeler's own version of a law)."""
    th, _ = metric.theta_c(np.asarray(v, float) @ metric.B)
    return th


# ----------------------------------------------------------------------------------------------------------------- strata
def depth_bucket(d, edges=(1, 2, 3, 5)):
    """0, 1, 2, 3-4, 5+ by default."""
    return np.searchsorted(np.asarray(edges), np.asarray(d), side="right")


def upgrade_strata(y, depth, logk, budget, small_article=None):
    """Addendum 57's budget rule: B < 500 flag x depth {0, 1, 2+}; < 2,000 flag x depth(5) x size {<2, 2-4, >=4};
    else flag x depth(5) x size {<2, 2-3, 3-4, 4-6, >=6}. Returns integer stratum ids.
    Development change (Addendum 58): where articles exist, items in articles of <= 4 comments form their own strata, because
    they carry about 90% of the article-level size law's weight variance on Civil Comments."""
    y = np.nan_to_num(np.asarray(y, float)).astype(int)
    if budget < 500:
        sid = y * 3 + depth_bucket(depth, (1, 2))
    else:
        db = depth_bucket(depth)
        sb = np.searchsorted([2, 4], logk, side="right") if budget < 2000 else np.searchsorted([2, 3, 4, 6], logk, side="right")
        ns = 3 if budget < 2000 else 5
        sid = (y * 5 + db) * ns + sb
    if small_article is not None:
        sid = sid * 2 + np.asarray(small_article, int)
    return sid


def score_strata(levels, depth, logk, budget, small_article=None):
    """Round A2: strata on the labeler's score level (<= 8) x structure, coarser at small budgets:
    B < 1,000: level x depth {0, 1, 2+}; < 4,000: level x depth(5); else level x depth(5) x size {<2, 2-4, >=4}."""
    lv = np.asarray(levels, int)
    if budget < 1000:
        sid = lv * 3 + depth_bucket(depth, (1, 2))
    elif budget < 4000:
        sid = lv * 5 + depth_bucket(depth)
    else:
        sid = (lv * 5 + depth_bucket(depth)) * 3 + np.searchsorted([2, 4], logk, side="right")
    if small_article is not None:
        sid = sid * 2 + np.asarray(small_article, int)
    return sid


class Strata:
    """Stratum membership and the per-stratum moments of every metric's feature matrix."""

    def __init__(self, sid, metrics, y):
        sid = np.asarray(sid)
        self.ids, inv = np.unique(sid, return_inverse=True)
        self.H = len(self.ids); self.inv = inv
        order = np.argsort(inv, kind="stable"); self.members = np.split(order, np.cumsum(np.bincount(inv, minlength=self.H))[:-1])
        self.N = np.array([len(m) for m in self.members], float)
        y = np.nan_to_num(np.asarray(y, float))
        self.mom = []
        for m in metrics:
            S1 = np.zeros((self.H, m.B.shape[1])); S2 = np.zeros((self.H, m.B.shape[1], m.B.shape[1])); SY = np.zeros_like(S1)
            for h, mem in enumerate(self.members):
                Bh = m.B[mem]; S1[h] = Bh.sum(0); S2[h] = Bh.T @ Bh; SY[h] = (y[mem, None] * Bh).sum(0)
            self.mom.append((S1, S2, SY))


# ----------------------------------------------------------------------------------------------------------------- sampling
def prefix_draws(members, cap, rng):
    """For each stratum, a uniformly random ordered sample without replacement of min(N_h, cap) members: any prefix of it is a
    simple random sample, so a pilot and a top-up drawn as consecutive prefixes form one simple random sample per stratum."""
    return [mem[rng.choice(len(mem), min(len(mem), int(cap)), replace=False)] if len(mem) else mem for mem in members]


def take(perms, n, start=None):
    start = np.zeros(len(perms), int) if start is None else start
    return [p[a:a + int(k)] for p, a, k in zip(perms, start, n)]


def prop_alloc(N, n):
    """Proportional allocation with at least 2 per stratum of size >= 2."""
    floor = np.minimum(np.where(N >= 2, 2, N), N); left = n - floor.sum()
    if left <= 0:
        return round_alloc(floor * n / max(floor.sum(), 1), n, N)
    room = N - floor; a = floor + np.minimum(room, left * N / N.sum())
    return round_alloc(a, n, N)


def equal_split(N, n):
    """Split n as equally as possible over strata, capped at stratum sizes (water-filling)."""
    alloc = np.zeros(len(N)); left = float(n); open_ = N > 0
    while left > 1e-9 and open_.any():
        share = left / open_.sum(); add = np.minimum(share, N - alloc); add[~open_] = 0
        alloc += add; left -= add.sum(); open_ = open_ & (alloc < N - 1e-9)
    return round_alloc(alloc, n, N)


def round_alloc(a, n, N):
    """Integer allocation summing to min(n, sum N), close to a, within [0, N]."""
    n = int(min(n, N.sum())); base = np.floor(a).astype(int); base = np.minimum(base, N.astype(int))
    rem = n - base.sum(); frac = a - np.floor(a)
    order = np.argsort(-frac, kind="stable")
    i = 0
    while rem > 0 and i < 10 * len(a) + n:
        h = order[i % len(a)]
        if base[h] < N[h]:
            base[h] += 1; rem -= 1
        i += 1
    while rem < 0:
        h = int(np.argmax(base)); base[h] -= 1; rem += 1
    return base


def topup(target, have, n_total, N):
    """Main-phase sizes so that have + main is as close to target as possible, summing to n_total, within N."""
    need = np.maximum(target - have, 0.0); room = N - have; left = n_total - have.sum()
    main = np.zeros(len(N))
    if left <= 0:
        return main.astype(int)
    if need.sum() > 0:
        main = np.minimum(need * left / need.sum(), room)
    rest = left - main.sum()
    while rest > 1e-9:                                                # leftover from caps: spread proportionally to target
        open_ = room - main > 1e-9
        if not open_.any():
            break
        w = np.where(open_, np.maximum(target, 1e-12), 0); add = np.minimum(rest * w / w.sum(), room - main)
        main += add; rest -= add.sum()
    return round_alloc(main, int(left), room)


# ----------------------------------------------------------------------------------------------------------------- estimation
def strat_totals(samples, members, N, B, z):
    """Stratified Horvitz-Thompson total of z * B."""
    T = np.zeros(B.shape[1])
    for h, s in enumerate(samples):
        if len(s):
            T += N[h] / len(s) * (z[s, None] * B[s]).sum(0)
    return T


def strat_var(samples, N, u):
    """Stratified variance of a Horvitz-Thompson total of u (u given per sampled item, keyed by position in samples)."""
    v = 0.0
    for h, s in enumerate(samples):
        n = len(s)
        if n >= 2:
            v += N[h] ** 2 * (1 - n / N[h]) * np.var(u[h], ddof=1) / n
    return v


def estimate_strat(metric, samples, members, N, z, S1=None):
    """Stratified estimate of the metric and its linearized variance.
    S1=None: Horvitz-Thompson totals. S1 given (per-stratum known totals of B): the regression estimator inside each stratum,
    T_h = (N_h/n_h) sum_s z B + p_h (S1_h - (N_h/n_h) sum_s B) with p_h the stratum's sample gold rate (development change,
    Addendum 58); its residual (z - p_h) phi removes the within-stratum spread of the known weights phi."""
    if S1 is None:
        T = strat_totals(samples, members, N, metric.B, z)
        th, c = metric.theta_c(T)
        u = [z[s] * (metric.B[s] @ c) for s in samples]
        return th, strat_var(samples, N, u)
    # leave-one-out coefficients (cross-fitting) keep the estimator unbiased to first order:
    # T_h = (N_h/n_h) sum_i [z_i B_i + p_(-i) (S1_h/N_h - B_i)],  p_(-i) = stratum gold rate without item i
    T = np.zeros(metric.B.shape[1]); terms = []
    for h, s in enumerate(samples):
        n = len(s)
        if n == 0:
            terms.append(None); continue
        zs, Bs = z[s], metric.B[s]
        if n == 1 or N[h] == n:
            t = zs[:, None] * Bs + (S1[h] / N[h] - Bs) * (0.0 if N[h] == n else 0.0)
        else:
            pl = (zs.sum() - zs) / (n - 1); t = zs[:, None] * Bs + pl[:, None] * (S1[h] / N[h] - Bs)
        T += N[h] / n * t.sum(0); terms.append(t)
    th, c = metric.theta_c(T)
    u = [np.zeros(0) if t is None else t @ c for t in terms]
    return th, strat_var(samples, N, u)


def estimate_ppi(metric, s, Ntot, z, y, Y_tot=None, ystrat=None):
    """PPI++ on one uniform sample s from a population of Ntot items: T = lam * sum_all y B + (N/n) sum_s (z - lam y) B,
    lam power-tuned on the metric's linearized variable. Y_tot = sum over the population of y * B."""
    n = len(s); Bs = metric.B[s]; zs, ys = z[s], np.nan_to_num(y[s])
    T0 = Ntot / n * (zs[:, None] * Bs).sum(0)
    th0, c = metric.theta_c(T0)
    if not np.isfinite(th0):
        return np.nan, np.nan
    uz, uy = zs * (Bs @ c), ys * (Bs @ c)
    vy = np.var(uy, ddof=1) if n > 1 else 0.0
    lam = float(np.cov(uz, uy)[0, 1] / (vy * (1 + PPI_TUNE * n / Ntot))) if vy > 0 else 0.0
    T = lam * Y_tot + Ntot / n * ((zs - lam * ys)[:, None] * Bs).sum(0)
    th, c = metric.theta_c(T)
    r = zs * (Bs @ c) - lam * ys * (Bs @ c)
    return th, Ntot ** 2 * (1 - n / Ntot) * np.var(r, ddof=1) / n if n > 1 else np.nan


def model_S2(st, metrics, samples, z, T_pilot, pool=None):
    """Per-stratum model variance of each metric's influence: p_h (1 - p_h) E_h[phi^2], phi at pilot plug-ins.
    pool=None: p_h is the stratum's Jeffreys-smoothed pilot rate. pool = array of group ids per stratum (development change,
    Addendum 58: the labeler's flag level): p_h is the group's Horvitz-Thompson pilot rate, so a stratum's own pilot labels barely
    move its allocation, which removes the bias that a stratum-level adaptive allocation causes. Returns (S2 [m, h], w [m])."""
    if pool is None:
        p = np.array([(z[s].sum() + 0.5) / (len(s) + 1.0) for s in samples])
    else:
        pool = np.asarray(pool); p = np.zeros(len(samples))
        for g in np.unique(pool):
            hs = np.flatnonzero(pool == g)
            num = sum(st.N[h] * z[samples[h]].mean() for h in hs if len(samples[h])); den = sum(st.N[h] for h in hs if len(samples[h]))
            ng = sum(len(samples[h]) for h in hs)
            pg = (num / den * ng + 0.5) / (ng + 1.0) if den > 0 else 0.5
            p[hs] = pg
    S2 = np.zeros((len(metrics), st.H))
    for j, m in enumerate(metrics):
        th, c = m.theta_c(T_pilot[j])
        if not np.isfinite(th):
            c = np.zeros(m.B.shape[1]); c[0] = 1.0
        S1, Sq, _ = st.mom[j]
        Ephi = (S1 @ c) / np.maximum(st.N, 1); Ephi2 = np.einsum("hij,i,j->h", Sq, c, c) / np.maximum(st.N, 1)
        S2[j] = np.maximum(p * (1 - p) * Ephi2, 0.0)                 # the regression residual's model variance
    w = 1.0 / np.maximum((np.sqrt(S2) * st.N).sum(1) ** 2, 1e-300)
    return S2, w


def run_strat(metrics, st, budget, rng, z, mode="upgrade", pilot_share=0.2, mult=None, pool=None):
    """One stratified audit. mode 'flagprop': proportional allocation; 'upgrade': pilot + compromise Neyman top-up."""
    N = st.N
    perms = prefix_draws(st.members, budget, rng)
    if mode == "flagprop":
        samples = take(perms, prop_alloc(N, budget))
    else:
        n_p = int(min(max(round(pilot_share * budget), 2 * st.H), budget))
        npil = equal_split(N, n_p)
        pil = take(perms, npil)
        Tp = [strat_totals(pil, st.members, N, m.B, z) for m in metrics]
        S2, w = model_S2(st, metrics, pil, z, Tp, pool=pool)
        if mult is not None:                                          # analyst weights on top of the per-metric normalization
            w = w * np.asarray(mult, float)
        sig = np.sqrt((w[:, None] * S2).sum(0))
        target = budget * N * sig / max((N * sig).sum(), 1e-300)
        main = topup(target, npil.astype(float), budget, N)
        samples = take(perms, npil + main)
    out = {}
    for j, m in enumerate(metrics):
        out[m.name] = estimate_strat(m, samples, st.members, N, z, S1=st.mom[j][0])
    out["_labels"] = int(sum(len(s) for s in samples))
    return out


def run_dstrat(metrics, dmembers, Nd, budget, rng, z, y, Yd):
    """Depth-stratified PPI++: equal allocation over depth buckets, PPI++ inside each (Yd[j][h] = sum over bucket h of y * B)."""
    n = equal_split(Nd, budget); samples = take(prefix_draws(dmembers, budget, rng), n)
    out = {}
    for j, m in enumerate(metrics):
        # combine bucket-level PPI++ totals, then plug in; variance by linearization at the combined estimate
        T = np.zeros(m.B.shape[1]); parts = []
        T0 = strat_totals(samples, dmembers, Nd, m.B, z); th0, c0 = m.theta_c(T0)
        if not np.isfinite(th0):
            out[m.name] = (np.nan, np.nan); continue
        for h, s in enumerate(samples):
            if len(s) < 2:
                T += Nd[h] / max(len(s), 1) * (z[s, None] * m.B[s]).sum(0); parts.append((s, 0.0)); continue
            Bs = m.B[s]; zs, ys = z[s], np.nan_to_num(y[s]); uz, uy = zs * (Bs @ c0), ys * (Bs @ c0)
            vy = np.var(uy, ddof=1); lam = float(np.cov(uz, uy)[0, 1] / (vy * (1 + PPI_TUNE * len(s) / Nd[h]))) if vy > 0 else 0.0
            T += lam * Yd[j][h] + Nd[h] / len(s) * ((zs - lam * ys)[:, None] * Bs).sum(0); parts.append((s, lam))
        th, c = m.theta_c(T)
        u = [z[s] * (m.B[s] @ c) - lam * np.nan_to_num(y[s]) * (m.B[s] @ c) for s, lam in parts]
        out[m.name] = (th, strat_var([s for s, _ in parts], Nd, u))
    out["_labels"] = int(n.sum())
    return out


def wald(th, v, z=Z95):
    if not np.isfinite(th) or not np.isfinite(v) or v < 0:
        return np.nan, np.nan
    se = np.sqrt(v); return th - z * se, th + z * se
