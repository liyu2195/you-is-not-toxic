"""Addendum 12: does the LLM score beat the binary flag as the PPI predictor in THREADSAFE's depth-stratified estimator?
Contrast as f54: Delta = P(Z=1 | depth >= 2) - P(Z=1 | depth = 0) over items of fully labeled threads.
  strat_ppi   : current. m/2 labels per stratum, PPI++ with the LLM flag y.
  strat_ppi_s : the same with the LLM score s (Haiku letter score, Qwen probability) as predictor.
usage: f54b_depth_upgrade.py ITEMS_PARQUET OUT_JSON [--reps 500] [--budgets 500,2000]"""
import os, sys, json, numpy as np, pandas as pd
args = sys.argv[1:]; IP, OUT = args[0], args[1]
def opt(k, d): return args[args.index(k) + 1] if k in args else d
REPS = int(opt("--reps", 500)); BUD = [int(x) for x in opt("--budgets", "500,2000").split(",")]
rng = np.random.default_rng(541 + int(os.environ.get('SEED_OFFSET', 0)))
D = pd.read_parquet(IP); z, y, s, dep = D.z.values.astype(float), D.y.values.astype(float), D.s.values.astype(float), D.depth.values
deep, root = np.where(dep >= 2)[0], np.where(dep == 0)[0]; gold = z[deep].mean() - z[root].mean(); plug = y[deep].mean() - y[root].mean()
def ppi_mean(pop, n, f):
    ix = rng.choice(pop, n, replace=False); yy, ff = z[ix], f[ix]; Np = len(pop)
    lam = np.cov(yy, ff)[0, 1] / (np.var(ff, ddof=1) * (1 + n / Np)) if np.var(ff) > 0 else 0.0
    return lam * f[pop].mean() + (yy - lam * ff).mean(), np.var(yy - lam * ff, ddof=1) / n
def wilson(p, n):
    z2 = 1.96 ** 2; c = (p + z2 / (2 * n)) / (1 + z2 / n); h = 1.96 * np.sqrt(p * (1 - p) / n + z2 / (4 * n ** 2)) / (1 + z2 / n); return c - h, c + h
def hyb(ed, vd, nd, er, vr, nr):                      # Addendum 22: Newcombe on effective-n Wilson limits when an effective count < 10
    out = []
    for e_, v_, n_ in [(ed, vd, nd), (er, vr, nr)]:
        p = min(max(e_, 0.0), 1.0); ne = n_ if p in (0.0, 1.0) or v_ <= 0 else min(n_, p * (1 - p) / v_); out.append((p, ne))
    (p1, n1), (p0, n0) = out
    if min(n1 * p1, n1 * (1 - p1), n0 * p0, n0 * (1 - p0)) >= 10: return None
    l1, u1 = wilson(p1, n1); l0, u0 = wilson(p0, n0); d = p1 - p0
    return d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
res = dict(gold=float(gold), plugin=float(plug), corr_flag=float(np.corrcoef(z, y)[0, 1]), corr_score=float(np.corrcoef(z, s)[0, 1]), results={})
for m in BUD:
    for name, f in [("strat_ppi", y), ("strat_ppi_s", s), ("strat_ppi_hyb", y)]:
        R = []
        for _ in range(REPS):
            ed, vd = ppi_mean(deep, m // 2, f); er, vr = ppi_mean(root, m - m // 2, f); e = ed - er; se = np.sqrt(vd + vr)
            hb = hyb(ed, vd, m // 2, er, vr, m - m // 2) if name == "strat_ppi_hyb" else None
            R.append((e, hb[0], hb[1]) if hb else (e, e - 1.96 * se, e + 1.96 * se))
        est, lo, hi = np.array(R).T
        r = dict(bias=float(est.mean() - gold), coverage=float(np.mean((lo <= gold) & (gold <= hi))), width=float(np.mean(hi - lo)), claims_escalation=float(np.mean(lo > 0)), claims_decline=float(np.mean(hi < 0)))
        res["results"][f"{name}@{m}"] = r; print(f"m={m:5d} {name:12s} " + json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("corr flag %.3f score %.3f" % (res["corr_flag"], res["corr_score"]))
