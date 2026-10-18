"""THREADSAFE for pair metrics (final): uniform reply-edge sampling with score-based intervals, plus ablations.
Final design: sample m/2 reply edges uniformly, label parent and reply (a parent label already known is reused).
  Delta = P(Zc=1 | Zp=1) - P(Zc=1 | Zp=0): Newcombe hybrid score interval (Wilson intervals of the two proportions).
  RR: MOVER-R interval from the same Wilson limits (Donner & Zou 2012).
Also reports the Wald interval for the same design (ablation). The LLM-guided label-sharing ablations (parent sampling with
60/40 by LLM flag, or 45/20/35 by LLM score group with reply-count bins) are in f24e_edges_v2.py.
usage: f24h_edges_final.py EDGES_PARQUET ITEMS_PARQUET OUT_JSON [--reps 300] [--budgets 500,1000,2000]"""
import os, sys, json, numpy as np, pandas as pd
args = sys.argv[1:]; EP, IP, OUT = args[:3]
def opt(k, d): return args[args.index(k) + 1] if k in args else d
REPS = int(opt("--reps", 300)); BUD = [int(x) for x in opt("--budgets", "500,1000,2000").split(",")]
rng = np.random.default_rng(2525 + int(os.environ.get('SEED_OFFSET', 0))); Z = 1.959964
E = pd.read_parquet(EP); zp, zc = E.Zp.values.astype(int), E.Zc.values.astype(int); NE = len(E); par = E.parent.values
D_true = zc[zp == 1].mean() - zc[zp == 0].mean(); RR_true = zc[zp == 1].mean() / zc[zp == 0].mean()
def wilson(x, n):
    if n == 0: return 0.0, 0.0, 1.0
    p = x / n; den = 1 + Z ** 2 / n; c = (p + Z ** 2 / (2 * n)) / den; h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / den
    return p, max(0.0, c - h), min(1.0, c + h)
def final(m):
    # sample edges until the label budget is used: 2 labels per edge, 1 when the parent label is already known
    o = rng.permutation(NE); seen = set(); used = 0; take = []
    for e in o:
        c = 1 + (par[e] not in seen)
        if used + c > m: break
        used += c; seen.add(par[e]); take.append(e)
    t = np.array(take); a, b = zp[t], zc[t]; n1, x1, n0, x0 = int(a.sum()), int((a * b).sum()), int((1 - a).sum()), int(((1 - a) * b).sum())
    p1, l1, u1 = wilson(x1, n1); p0, l0, u0 = wilson(x0, n0); d = p1 - p0
    lo = d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2); hi = d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    se = np.sqrt(p1 * (1 - p1) / max(n1, 1) + p0 * (1 - p0) / max(n0, 1))                 # Wald, for the ablation
    rr = p1 / p0 if p0 > 0 else np.nan
    try:
        rl = (p1 * p0 - np.sqrt(max((p1 * p0) ** 2 - l1 * u0 * (2 * p1 - l1) * (2 * p0 - u0), 0))) / (u0 * (2 * p0 - u0))
        ru = (p1 * p0 + np.sqrt(max((p1 * p0) ** 2 - u1 * l0 * (2 * p1 - u1) * (2 * p0 - l0), 0))) / (l0 * (2 * p0 - l0)) if l0 > 0 else np.inf
    except ZeroDivisionError: rl, ru = 0.0, np.inf
    return dict(d=d, lo=lo, hi=hi, wlo=d - Z * se, whi=d + Z * se, rr=rr, rlo=rl, rhi=ru, used=used, n1=n1)
res = {"D_true": float(D_true), "RR_true": float(RR_true), "N_edges": int(NE), "results": {}}
for m in BUD:
    R = [final(m) for _ in range(REPS)]; g = lambda k: np.array([r[k] for r in R], float)
    d, lo, hi, wlo, whi, rr, rlo, rhi = (g(k) for k in ["d", "lo", "hi", "wlo", "whi", "rr", "rlo", "rhi"])
    out = dict(bias=float(d.mean() - D_true), true_sd=float(d.std(ddof=1)), labels=float(g("used").mean()), toxic_parent_edges=float(g("n1").mean()),
               cov_score=float(np.mean((lo <= D_true) & (D_true <= hi))), width_score=float(np.mean(hi - lo)),
               cov_wald=float(np.mean((wlo <= D_true) & (D_true <= whi))), width_wald=float(np.mean(whi - wlo)),
               rr_cov_mover=float(np.mean((rlo <= RR_true) & (RR_true <= rhi))), rr_median_width_mover=float(np.nanmedian(rhi - rlo)))
    res["results"][f"final@{m}"] = out; print(m, json.dumps({k: round(v, 4) for k, v in out.items()}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
