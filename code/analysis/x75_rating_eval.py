"""Addendum 75, B75a analysis: real operating points for Haiku 5.5 and Sonnet 5.5 from 0-100 ratings on the 5% Civil Comments
sample (results/x14_pops/cc_s5.parquet; ratings from x75_rating.py). With the parent, root comments (no parent) keep their
no-context rating. A missing rating counts as 0 (never flagged); the counts are reported.
Flags: the round(r N) comments with the highest rating, ties broken at random (seed 7500), at r = gold prevalence ('matched') and
over the grid 2%, 5%, 7.5%, 10%, 15%, 20%, 30%. Statistics: the depth law, its rate-free version (slope / flag rate), the placebo
depth slope relative to the FPR, the thread size law and the log reply RR, with 500 article-cluster Poisson bootstrap resamples
(seed 7501), paired with gold.
H75a: at matched prevalence, Haiku's depth law exceeds gold's (CI of the paired difference above 0), without and with the parent.
H75b: the same for Sonnet.
H75c: at matched prevalence, Haiku's (no parent) placebo depth slope relative to its FPR exceeds the single rater's of B75b, both
against B75b's leave-one-out gold of draw 0 (seed 7510, reproduced on the full reply threads and restricted to the sample);
paired bootstrap CI of the difference above 0.
Also writes the paper macros (results/x75_rating/x75_rating_eval.json, 'macros').
usage: python3 x75_rating_eval.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x75_rating"; NB = 500; PP = 100.0
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); z = P.z.values.astype(float); N = len(z); prev = float(z.mean())
ids = P.id.astype(np.int64).values; has_par = P.parent_row.values >= 0
def ratings(name):
    R = {}
    for l in open(f"{D}/{name}.jsonl"):
        r = json.loads(l)
        if r.get("rating") is not None:
            R[int(r["id"])] = float(r["rating"])
    return R
S = {}; miss = {}
for m in ("haiku", "sonnet"):
    r0 = ratings(f"{m}_ctx0"); r1 = ratings(f"{m}_ctx1")
    a0 = np.array([r0.get(i, np.nan) for i in ids]); a1 = np.array([r1.get(i, np.nan) if hp else r0.get(i, np.nan) for i, hp in zip(ids, has_par)])
    miss[m] = int(np.isnan(a0).sum()); miss[f"{m}_ctx"] = int(np.isnan(a1).sum())
    S[f"{m}_r"] = np.nan_to_num(a0, nan=0.0); S[f"{m}_r_ctx"] = np.nan_to_num(a1, nan=0.0)
print("missing ratings", miss, "N", N, "prev", round(prev, 4), flush=True)
def flags(s, rate, seed=7500):
    k = int(round(rate * N)); rng = np.random.default_rng(seed); order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0
    return f
# structure and per-cluster sufficient statistics
cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1; x = np.minimum(P.depth.values, 10).astype(float)
un = pd.factorize(P.unit.values)[0]; ku = np.bincount(un).astype(float); ucl = np.zeros(ku.size, np.int64); ucl[un] = cl; lxu = np.log2(np.maximum(ku, 1)); mU = ku >= 2
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
Wb = np.vstack([np.ones(nC), np.random.default_rng(7501).poisson(1.0, size=(NB, nC))])
bc = lambda idx, w: np.bincount(idx, weights=w, minlength=nC)
def ols(cidx, xv, yv, mask):
    c = cidx[mask]; xx = xv[mask]; yy = yv[mask]
    n_, sx, sy, sxx, sxy = (Wb @ bc(c, v) for v in (np.ones(mask.sum()), xx, yy, xx * xx, xx * yy))
    return (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2), sy / n_
def stats(v, g):
    law, rate = ols(cl, x, v, np.ones(N, bool)); pdl, fpr = ols(cl, x, v, g == 0)
    fu = np.bincount(un, weights=v) / ku; sz, _ = ols(ucl, lxu, fu, mU)
    vp, vc = v[pa], v[ch]; p1, p1c1, p0, p0c1 = (Wb @ bc(ecl, w) for w in (vp, vp * vc, 1 - vp, (1 - vp) * vc))
    with np.errstate(divide="ignore", invalid="ignore"):
        lrr = np.log((p1c1 / p1) / (p0c1 / p0))
    return dict(depth=law, depth_rate_free=law / rate, placebo_depth_rel=pdl / fpr, size=sz, log_rr=lrr, flag_rate=rate)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
G = stats(z, z); res = {"N": N, "prev": prev, "missing": miss, "gold": {k: ci(v) for k, v in G.items() if k != "placebo_depth_rel"}, "instruments": {}}
GRID = [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
for name, s in S.items():
    out = {}
    for tag, rate in [("matched", prev)] + [(f"{r:g}", r) for r in GRID]:
        f = flags(s, rate); st = stats(f, z)
        o = {k: ci(v) for k, v in st.items()}
        for law in ("depth", "size", "log_rr"):
            o[f"{law}_minus_gold"] = ci(st[law] - G[law])
        o["depth_rate_free_minus_gold"] = ci(st["depth_rate_free"] - G["depth_rate_free"])
        out[tag] = o
    res["instruments"][name] = out
    m_ = out["matched"]
    print(f"{name:14s} matched: depth {m_['depth'][0]*PP:+.2f} (minus gold {m_['depth_minus_gold'][0]*PP:+.2f} [{m_['depth_minus_gold'][1]*PP:+.2f}, {m_['depth_minus_gold'][2]*PP:+.2f}])"
          f" placebo depth {m_['placebo_depth_rel'][0]*100:+.2f}% size {m_['size_minus_gold'][0]*PP:+.2f} lrr {m_['log_rr_minus_gold'][0]:+.3f} | grid depth-gold:",
          {t: round(out[t]["depth_minus_gold"][0] * PP, 2) for t in out if t != "matched"}, flush=True)
# H75c: the single rater of B75b, draw 0, restricted to the sample
F = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "tox"])
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(F.id.values).values.astype(np.int64); kv = np.clip(np.rint(F.tox.values * n).astype(np.int64), 0, n)
rng = np.random.default_rng(7510); nA = np.ones_like(n); kA = rng.hypergeometric(kv, n - kv, nA); kB = kv - kA
A_full = (kA / np.maximum(nA, 1) >= 0.5 - 1e-12).astype(float); B_full = (kB / np.maximum(n - nA, 1) >= 0.5 - 1e-12).astype(float)
pos = pd.Series(np.arange(len(F)), index=F.id.astype(np.int64).values).reindex(ids).values; assert not np.isnan(pos).any()
A, B = A_full[pos.astype(int)], B_full[pos.astype(int)]
hA = stats(flags(S["haiku_r"], prev), B)["placebo_depth_rel"]; sA = stats(A, B)["placebo_depth_rel"]; d = hA - sA
res["H75c_detail"] = dict(haiku_vs_looB=ci(hA), single_rater=ci(sA), diff=ci(d))
res["H75a"] = all(res["instruments"][k]["matched"]["depth_minus_gold"][1] > 0 for k in ("haiku_r", "haiku_r_ctx"))
res["H75b"] = all(res["instruments"][k]["matched"]["depth_minus_gold"][1] > 0 for k in ("sonnet_r", "sonnet_r_ctx"))
res["H75c"] = bool(ci(d)[1] > 0)
print("H75a", res["H75a"], "H75b", res["H75b"], "H75c", res["H75c"], res["H75c_detail"], flush=True)
# macros for the paper
sg = lambda v, dd: (f"+{v:.{dd}f}" if v >= 0 else f"$-${abs(v):.{dd}f}")
I = res["instruments"]; mm = {}
for k, nm in (("haiku_r", "Haiku"), ("haiku_r_ctx", "HaikuCtx"), ("sonnet_r", "Sonnet"), ("sonnet_r_ctx", "SonnetCtx")):
    mm[f"qRate{nm}DepthPP"] = sg(I[k]["matched"]["depth"][0] * PP, 2); mm[f"qRate{nm}DepthLo"] = sg(I[k]["matched"]["depth_minus_gold"][1] * PP, 2)
mm["qRateGoldDepthPP"] = sg(G["depth"][0] * PP, 2)
mm["qRateHaikuPDR"] = sg(100 * I["haiku_r"]["matched"]["placebo_depth_rel"][0], 1) + "\\%"; mm["qRateSrPDR"] = sg(100 * float(sA[0]), 1) + "\\%"
gv = [I[k][t]["depth_minus_gold"][0] for k in I for t in I[k] if t != "matched"]; gl = [I[k][t]["depth_minus_gold"][1] for k in I for t in I[k] if t != "matched"]
mm["qRateGridMin"] = sg(min(gv) * PP, 2); mm["qRateGridMax"] = sg(max(gv) * PP, 2); mm["qRateNGridPos"] = str(sum(v > 0 for v in gl)); mm["qRateNGrid"] = str(len(gl))
res["macros"] = mm
json.dump(res, open(f"{D}/x75_rating_eval.json", "w"), indent=1); print("saved", f"{D}/x75_rating_eval.json", mm)
