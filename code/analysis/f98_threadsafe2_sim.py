"""Monte Carlo check of the THREADSAFE-2 unit design (Addendum 10) against the current design and cluster PPI.
Each sampled unit's comments are ordered by LLM score (ties at random, seed 98).
  cluster_ppi  : SRS of units, every comment labeled, PPI++ with the unit plug-in (CV1).
  cluster_ppi4 : the same with the vector control variate (CV4: plug-in, #flagged capped at 5, log size, max score).
  search_cur   : current THREADSAFE. Search the LLM-flagged comments, verify the rest of an unsettled unit with q = .5
                 (all remaining comments), CV1.
  ts2          : THREADSAFE-2. Search comments at or above the checkpoint (lowest score level with <= 40% of comments at or
                 above it), stop at the first positive; unsettled -> coin q = .5 -> continue in score order to the first
                 positive (sequential verification); CV4.
  ts2_cv1      : ts2 with CV1 (ablation).   ts2_noseq : ts2 with all-remaining verification (ablation).
  ts2_flag     : sequential verification and CV4 at the LLM-flag checkpoint (ablation).
  ts2_aT[c]    : ts2 with the vector CV only when >= T units are sampled (Addendum 17); suffix c adds census completion (Addendum 25):
                 once every unit is searched, verify all unverified unsettled units if the unspent budget covers it in the worst case;
                 suffix q adds the plug-in verification rate (Addendum 26): q = .5 for the first 50 units, then Prop 4's q* estimated from them;
                 suffix s adds the switch (Addendum 27): stage 1 (40% of the budget, 10-50 units) at q = .5, then q = 1 if A r < 2 h c (estimated), else .5;
                 suffix z (Addendum 29): the same stage 1, then q = 1 only if the estimated A = s2_cv - h is <= 0 (and h > 0);
                 suffix y (Addendum 32): z, and only if stage 1 observed >= 10 positive units.
                 suffix x (Addendum 39): from T sampled units on, the unit term is Theorem 2's output under a working law learned
                 from the audit's own labels: hazards per (LLM level, search depth) cell, 5-fold cross-fitted over the sampled units
                 (each unit's anchor comes from a law fitted without its fold); T_u = mu_u + sum over labeled segments of B_g / G_g.
Unit term t = S + (1 - S) (R/q) H. PPI++ with a vector CV: beta = (1 + n/N)^-1 Cov_s(X)^-1 Cov_s(X, t),
est = mean(t) - (mean_s(X) - mean_N(X))' beta, se^2 = Var_s(t - X beta)/n + beta' Cov_N(X) beta / N.
Stopping: units are taken in random order while the cumulative cost stays within the budget (at least 2 units).
usage: f98_threadsafe2_sim.py UNITS_PARQUET OUT_JSON [--reps 300] [--budgets 250,500,1000] [--q 0.5]"""
import os, re, sys, json, numpy as np, pandas as pd
args = sys.argv[1:]; UP, OUT = args[0], args[1]
def opt(k, d): return args[args.index(k) + 1] if k in args else d
REPS = int(opt("--reps", 300)); BUD = [int(x) for x in opt("--budgets", "250,500,1000").split(",")]; Q = float(opt("--q", 0.5))
rng = np.random.default_rng(98 + int(os.environ.get('SEED_OFFSET', 0)))
U = pd.read_parquet(UP, columns=["unit", "z", "s", "y"])
U["s2"] = U.s.values + 1e-7 * np.random.default_rng(90).random(len(U))
U = U.sort_values(["unit", "s2"], ascending=[True, False]).reset_index(drop=True)
U["u"] = pd.factorize(U.unit)[0]; U["rk"] = U.groupby("u").cumcount()
# checkpoint: the lowest score level with at most 40% of comments at or above it
sv = U.s.values; lv = np.round(sv, 4); uq = np.unique(lv)
if len(uq) <= 6:
    share_ge = np.array([(lv >= L).mean() for L in uq]); ok = uq[share_ge <= 0.40]
    TAU = float(ok.min()) if len(ok) else float(uq.max()); inC = lv >= TAU     # fallback (Addendum 17): top level when it alone holds > 40%
else:
    TAU = float(np.quantile(sv, 0.60)); inC = sv >= TAU
# segments for the learned output (suffix x): (LLM level, search depth) cells; levels as f170 (ordinal answer, or score quantiles)
if len(uq) <= 6: LEV = (len(uq) - 1 - np.searchsorted(uq, lv)).astype(int)
else:
    _qs = np.unique(np.quantile(sv, [0.6, 0.8, 0.9, 0.95])); LEV = (len(_qs) - np.searchsorted(_qs, sv, side="right")).astype(int)
NLV = int(LEV.max() + 1); NBK = 4; SG = NLV * NBK; GVX = np.arange(SG)
SEG = LEV * NBK + np.digitize(U.rk.values, [1, 4, 16])
CHKC = np.zeros(SG, bool)
for _l in range(NLV):
    if (LEV == _l).any(): CHKC[_l * NBK:(_l + 1) * NBK] = bool(inC[LEV == _l].all())
CELL = SEG                                                           # the working law's cell of each comment: (LLM level, search depth)
UKC = np.bincount(U.u.values); USTART = np.r_[0, np.cumsum(UKC)[:-1]]; RKC = U.rk.values; ZC = U.z.values.astype(float)
NFC = np.bincount(U.u.values, weights=inC).astype(int)
FPOSC = np.full(len(UKC), np.inf); _ix1 = np.flatnonzero(U.z.values == 1)
FPOSC[pd.Series(U.u.values[_ix1]).drop_duplicates().values] = RKC[_ix1][~pd.Series(U.u.values[_ix1]).duplicated().values]
def _iso_dec(v, w):
    bl = []
    for i in range(len(v)):
        bl.append([v[i] * w[i], w[i], 1])
        while len(bl) > 1 and bl[-2][0] / bl[-2][1] < bl[-1][0] / bl[-1][1]:
            a_, b_ = bl.pop(), bl.pop(); bl.append([a_[0] + b_[0], a_[1] + b_[1], a_[2] + b_[2]])
    return np.concatenate([[b_[0] / b_[1]] * b_[2] for b_ in bl])
def law_from_counts(lab, pc, a=2.0):
    """Hazards per (level, depth) cell from labeled-comment counts: Beta-smoothed toward the level rate (prior weight 2),
    isotonic (nonincreasing) in depth within a level and in level within a depth bucket."""
    tot = (pc.sum() + 1) / (lab.sum() + 2); labL, pcL = lab.reshape(NLV, NBK).sum(1), pc.reshape(NLV, NBK).sum(1)
    lr_ = _iso_dec((pcL + a * tot) / (labL + a), labL + a)
    H = ((pc + a * np.repeat(lr_, NBK)) / (lab + a)).reshape(NLV, NBK); W = (lab + a).reshape(NLV, NBK)
    for _l in range(NLV): H[_l] = _iso_dec(H[_l], W[_l])
    for _b in range(NBK): H[:, _b] = _iso_dec(H[:, _b].copy(), W[:, _b].copy())
    return np.clip(H.ravel(), 1e-4, 1 - 1e-4)
def learned_terms(ix, Rv, qvv, K=5):
    """Theorem 2's output for the sampled units ix, comment by comment in the design's order:
    T_u = mu_u + sum over labeled comments of (A_u / (A_ui G_i)) (Z_i - h_i), with G_i = 1 in the search and q_u in the verification,
    under the working law fitted on the other folds (fold = position in the sampling order mod K)."""
    kk = UKC[ix]; rows = np.repeat(USTART[ix] - np.r_[0, np.cumsum(kk)[:-1]], kk) + np.arange(kk.sum()); uj = np.repeat(np.arange(len(ix)), kk)
    rk, cell, zz = RKC[rows], CELL[rows], ZC[rows]; nf = NFC[ix][uj]
    lab = (rk <= FPOSC[ix][uj]) & ((rk < nf) | (Rv[uj] == 1)); G = np.where(rk < nf, 1.0, qvv[uj])
    fo = np.arange(len(ix)) % K; foc = fo[uj]; T = np.empty(len(ix))
    for f in range(K):
        tr = (foc != f) & lab
        p = law_from_counts(np.bincount(cell[tr], minlength=SG).astype(float), np.bincount(cell[tr], weights=zz[tr], minlength=SG))
        te = foc == f; ut = uj[te]; h = p[cell[te]]; l1 = np.log1p(-h)
        st = np.flatnonzero(np.r_[True, ut[1:] != ut[:-1]]); blk = np.cumsum(np.r_[True, ut[1:] != ut[:-1]]) - 1
        tot = np.add.reduceat(l1, st); prior = np.cumsum(tot) - tot; incl = np.cumsum(l1) - prior[blk]
        contrib = np.where(lab[te], np.exp(tot[blk] - incl) * (zz[te] - h) / G[te], 0.0)
        T[fo == f] = 1 - np.exp(tot) + np.add.reduceat(contrib, st)
    return T
def stats(inF):
    g = U.assign(F=inF).groupby("u", sort=True)
    k = g.size().values; nF = g.F.sum().values.astype(int); Y = g.z.max().values.astype(float)
    fpos = U.rk.where(U.z == 1).groupby(U.u).min().reindex(np.arange(len(k))).fillna(np.inf).values
    S = fpos < nF; H = ((Y == 1) & ~S).astype(float)
    c1 = np.where(S, fpos + 1, nF).astype(float); r = (k - nF).astype(float)
    c2all = np.where(S, 0, r); c2seq = np.where(H == 1, fpos - nF + 1, np.where(S, 0, r))
    return dict(k=k, Y=Y, S=S.astype(float), H=H, c1=c1, c2all=c2all, c2seq=c2seq)
cur, chk = stats(U.y.values == 1), stats(inC)
N = len(cur["k"]); k = cur["k"]; gold = cur["Y"]; theta = gold.mean()
Yt = U.groupby("u").y.max().values.astype(float); nfl = np.minimum(U.groupby("u").y.sum().values, 5).astype(float)
smax = U.groupby("u").s.max().values; X1 = Yt[:, None]; X4 = np.column_stack([Yt, nfl, np.log(k), smax]); plug_all = Yt.mean()
POP = {1: (X1.mean(0), np.atleast_2d(np.cov(X1.T))), 4: (X4.mean(0), np.cov(X4.T))}
def ppi(t, X, p):
    n = len(t); mu, Sig = POP[p]
    if n < X.shape[1] + 3: return t.mean(), t.std(ddof=1) / np.sqrt(n)
    Xc = X - X.mean(0); Cxx = Xc.T @ Xc / (n - 1); Cxt = Xc.T @ (t - t.mean()) / (n - 1)
    beta = np.linalg.lstsq(Cxx * (1 + n / N) + 1e-10 * np.eye(len(mu)), Cxt, rcond=None)[0]
    est = t.mean() - (X.mean(0) - mu) @ beta; res = t - X @ beta
    se = np.sqrt(np.var(res, ddof=1) / n + beta @ Sig @ beta / N); return est, se
from scipy.stats import beta as _beta
def kg(est, se, n):                                    # Korn-Graubard (Addendum 19): effective-n Clopper-Pearson, alpha = .05
    p = np.clip(est, 0, 1); v = np.maximum(se ** 2, 1e-15)
    ne = np.where((p <= 0) | (p >= 1), n, np.minimum(n, p * (1 - p) / v)); x = ne * p
    lo = np.where(x > 0, _beta.ppf(0.025, np.maximum(x, 1e-12), ne - x + 1), 0.0); hi = np.where(x < ne, _beta.ppf(0.975, x + 1, np.maximum(ne - x, 1e-12)), 1.0)
    return lo, hi
def take_n(cost):
    n = int(np.searchsorted(np.cumsum(cost), M_, side="right")); return max(n, 2)
def run(method, m):
    global M_; M_ = m; order = rng.permutation(N)
    if method.startswith("cluster"):
        n = take_n(k[order].astype(float)); ix = order[:n]; t = gold[ix]
        est, se = ppi(t, (X4 if method == "cluster_ppi4" else X1)[ix], 4 if method == "cluster_ppi4" else 1); return est, se, float(k[ix].sum()), n, 0.0
    st = cur if method in ("search_cur", "ts2_flag") else chk
    c2 = st["c2all"] if method in ("search_cur", "ts2_noseq") else st["c2seq"]
    mm = re.match(r"ts2_a(\d+)([cqszyx]*)$", method); T_ = int(mm.group(1)) if mm else None; fl = mm.group(2) if mm else ""
    U_ = rng.random(N); qv = np.full(N, Q)                                 # one uniform per unit: R = U_ < q (the same stream for every method)
    if "q" in fl or "s" in fl or "z" in fl or "y" in fl:                  # Addendum 26 (q: plug-in q), 27 (s: switch if q* > .71), 29 (z: switch if A <= 0), 32 (y: z + count guard)
        if "s" in fl or "z" in fl or "y" in fl:                                                     # stage 1: up to 40% of the budget, between 10 and 50 units
            cum = np.cumsum((st["c1"] + (1 - st["S"]) * (U_ < Q) * c2)[order]); n1 = min(50, max(10, int(np.searchsorted(cum, 0.4 * M_, side="right"))))
        else: n1 = 50
        f1 = order[:n1]; R1 = (U_[f1] < Q).astype(float); S1, H1 = st["S"][f1], st["H"][f1]
        if float(np.sum(st["c1"][f1] + (1 - S1) * R1 * c2[f1])) <= M_:
            t1 = S1 + (1 - S1) * (R1 / Q) * H1; x1 = Yt[f1]
            ch = float(st["c1"][f1].mean()); hh = float(((1 - S1) * R1 * H1 / Q).mean()); rh = float(((1 - S1) * R1 * c2[f1] / Q).mean())
            lam = np.cov(t1, x1)[0, 1] / np.var(x1, ddof=1) if np.var(x1) > 0 else 0.0
            A_ = float(np.var(t1 - lam * x1, ddof=1)) - hh * (1 / Q - 1) - hh
            if "s" in fl: qh = 1.0 if (hh > 0 and A_ * rh < 2 * hh * ch) else Q
            elif "z" in fl: qh = 1.0 if (hh > 0 and A_ <= 0) else Q
            elif "y" in fl: qh = 1.0 if (hh > 0 and A_ <= 0 and float(S1.sum() + (R1 * H1).sum()) >= 10) else Q      # >= 10 observed positive units (np >= 10)
            else: qh = 1.0 if (A_ <= 0 or rh <= 0) else float(min(1.0, max(0.25, np.sqrt(hh * ch / (A_ * rh)))))
            qv[order[n1:]] = qh; SW_.append(qh)
    R = (U_ < qv).astype(float)
    cost = (st["c1"] + (1 - st["S"]) * R * c2)[order]; n = take_n(cost); ix = order[:n]
    t = st["S"][ix] + (1 - st["S"][ix]) * (R[ix] / qv[ix]) * st["H"][ix]
    p = 1 if method in ("search_cur", "ts2_cv1") else 4
    if mm: p = 4 if n >= T_ else 1                                        # Addendum 17: vector CV only with >= T sampled units
    if mm and "x" in fl and n >= T_: t = learned_terms(ix, R[ix], qv[ix])      # Addendum 39: Theorem 2's output, law learned from the audit
    if "c" in fl and n == N:                                              # Addendum 25: census completion once every unit is searched
        used = float(cost[:n].sum()); todo = (st["S"][ix] == 0) & (R[ix] == 0)
        if used + float(st["c2all"][ix][todo].sum()) <= M_:               # worst-case cost known from labeled comments and unit sizes
            return theta, 0.0, used + float(st["c2seq"][ix][todo].sum()), n, 1.0
    est, se = ppi(t, (X1 if p == 1 else X4)[ix], p); return est, se, float(cost[:n].sum()), n, 0.0

res = {"N_units": int(N), "theta": float(theta), "plugin": float(plug_all), "checkpoint": TAU, "checkpoint_share_items": float(inC.mean()),
       "share_settled_cur": float(cur["S"].mean()), "share_settled_ts2": float(chk["S"].mean()), "results": {}}
print(json.dumps({kk: v for kk, v in res.items() if kk != "results"}), flush=True)
for m in BUD:
    for name in opt("--methods", "cluster_ppi,cluster_ppi4,search_cur,ts2_flag,ts2_noseq,ts2_cv1,ts2").split(","):
        SW_ = []; E = np.array([run(name, m) for _ in range(REPS)]); est, se, used, nu, cens = E[:, 0], E[:, 1], E[:, 2], E[:, 3], E[:, 4] == 1
        lo, hi = est - 1.96 * se, est + 1.96 * se; cov = float(np.mean((lo <= theta) & (theta <= hi)))
        klo, khi = kg(est, se, nu); cov_kg = float(np.mean((klo <= theta) & (theta <= khi)))
        pc = np.clip(est, 0, 1); ne = np.where((pc <= 0) | (pc >= 1), nu, np.minimum(nu, pc * (1 - pc) / np.maximum(se ** 2, 1e-15)))
        small = np.minimum(ne * pc, ne * (1 - pc)) < 10                     # Addendum 20: hybrid = KG when the effective count < 10, else Wald
        hlo, hhi = np.where(small & ~cens, klo, lo), np.where(small & ~cens, khi, hi)        # a census answer is exact: zero-width interval
        z2 = 1.96 ** 2; cen = (pc + z2 / (2 * ne)) / (1 + z2 / ne); hw = 1.96 * np.sqrt(pc * (1 - pc) / ne + z2 / (4 * ne ** 2)) / (1 + z2 / ne)   # effective-n Wilson (reported)
        wlo, whi = cen - hw, cen + hw
        r = dict(bias=float(est.mean() - theta), rmse=float(np.sqrt(np.mean((est - theta) ** 2))), true_sd=float(est.std(ddof=1)), coverage=cov,
                 width=float(np.mean(hi - lo)), labels=float(used.mean()), excl_plugin=float(np.mean((hi < plug_all) | (lo > plug_all))),
                 coverage_kg=cov_kg, width_kg=float(np.mean(khi - klo)), excl_plugin_kg=float(np.mean((khi < plug_all) | (klo > plug_all))), n_units=float(nu.mean()),
                 coverage_hyb=float(np.mean((hlo <= theta) & (theta <= hhi))), width_hyb=float(np.mean(hhi - hlo)), share_kg_used=float(small.mean()),
                 excl_plugin_hyb=float(np.mean((hhi < plug_all) | (hlo > plug_all))), coverage_wilson=float(np.mean((wlo <= theta) & (theta <= whi))), width_wilson=float(np.mean(whi - wlo)),
                 census_rate=float(cens.mean()), **({"share_q1": float(np.mean(np.array(SW_) == 1.0)) if SW_ else 0.0, "mean_q_stage2": float(np.mean(SW_)) if SW_ else float('nan')} if re.match(r"ts2_a\d+c?[qszy]x?$", name) else {}))
        res["results"][f"{name}@{m}"] = r
        print(f"m={m:5d} {name:12s} bias {r['bias']:+.4f} sd {r['true_sd']:.4f} cov {cov:.3f} width {r['width']:.4f} labels {r['labels']:.0f} excl {r['excl_plugin']:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
