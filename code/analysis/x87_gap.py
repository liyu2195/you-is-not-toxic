"""Addendum 87, B87b and B87c (descriptive).
B87b: over B82a's 100 article-cluster resamples (seed 8201 weights, as x85_converged.py --part rrinf), log RR_inf from the
converged edge NPMLE at m = 20 (SQP, warm-started from the full-data fit) minus gold's log RR on the same weights: point value,
95% CI and the share of resamples with RR_inf <= gold's RR.
B87c: Haiku's (and Haiku shown the parent's) depth slope among matched gold negatives with no vote of any kind that the platform
approved (rating != 'rejected'), 500 resamples (seed 8501), as x85_construct_frac.py.
usage: python3 x87_gap.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code/analysis"))
OUT = sys.argv[1]
_argv = sys.argv; sys.argv = [sys.argv[0], "x", "--part", "none"]
import x85_converged as C
sys.argv = _argv
X = C.X; W = X.W
res = {}
# ---------------------------------------------------------------- B87b
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); K = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
cl = pd.factorize(P.cluster.values)[0]; ecl = cl[ch]; nC = int(cl.max() + 1); z = P.z.values.astype(float)
pat = pd.DataFrame({"np": n[pa], "kp": K[pa], "nc": n[ch], "kc": K[ch]}); codes, uniq = pd.factorize(pd.MultiIndex.from_frame(pat)); U = np.array(uniq.tolist()); S = len(U)
w0 = np.bincount(codes, minlength=S).astype(float)
m = 20; g = (np.arange(m) + 0.5) / m
Ep = X.norm_rows(X.logbinom(U[:, 1][:, None], U[:, 0][:, None], g[None, :])); Ec = X.norm_rows(X.logbinom(U[:, 3][:, None], U[:, 2][:, None], g[None, :]))
hi = g >= 0.5
def rr(G):
    return float(np.log((G[np.ix_(hi, hi)].sum() / G[hi].sum()) / (G[np.ix_(~hi, hi)].sum() / G[~hi].sum())))
G0 = C.em_G_sq(w0, Ep, Ec)
NB = 100; Wb = np.vstack([np.ones(nC), np.random.default_rng(8201).poisson(1.0, size=(NB, nC))]).astype(float)
vp, vc = z[pa], z[ch]
a, b, c, d = (np.bincount(ecl, x_, nC) for x_ in (vp * vc, vp, (1 - vp) * vc, 1 - vp))
lg = np.log(((Wb @ a) / (Wb @ b)) / ((Wb @ c) / (Wb @ d)))
li = [rr(G0)]
for bb in range(1, NB + 1):
    wb = np.bincount(codes, weights=Wb[bb][ecl], minlength=S); li.append(rr(C.em_G_sq(wb, Ep, Ec, G=G0)))
    if bb % 25 == 0:
        print("resample", bb, flush=True)
li = np.array(li); gap = li - lg
ci = lambda a_: [float(a_[0]), float(np.percentile(a_[1:], 2.5)), float(np.percentile(a_[1:], 97.5))]
res["B87b"] = dict(log_rr_inf=ci(li), log_rr_gold=ci(lg), gap=ci(gap), share_inf_le_gold=float(np.mean(gap[1:] <= 0)),
                   rr_inf=float(np.exp(li[0])), rr_gold=float(np.exp(lg[0])))
print("B87b", res["B87b"], flush=True)
# ---------------------------------------------------------------- B87c
F = pd.read_parquet(f"{W}/data/cc_attr_frac.parquet").set_index("id")
M = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "rating"]).set_index("id").reindex(P.id.values)
A = F.reindex(P.id.values); matched = P.id.isin(F.index).values
tox = P.tox.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float)
none = matched & (tox == 0) & np.all([np.nan_to_num(A[k].values.astype(float), nan=1.0) == 0 for k in ("insult", "identity_attack", "obscene", "threat", "severe_toxicity", "sexual_explicit")], axis=0)
approved = (M.rating.values != "rejected")
Wc = np.vstack([np.ones(nC), np.random.default_rng(8501).poisson(1.0, size=(500, nC))])
def slope(v, mask):
    m_ = mask.astype(float); v = np.nan_to_num(v)
    n_, sx, sy, sxx, sxy = (Wc @ np.bincount(cl, m_ * a_, nC) for a_ in (np.ones_like(x), x, v, x * x, x * v))
    return (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2), sy / n_
neg = (z == 0)
for nm, col in (("haiku", "y:haiku"), ("haiku_ctx", "y:haiku_ctx")):
    v = np.nan_to_num(P[col].values.astype(float))
    s1, mu1 = slope(v, neg & none & approved); s2, mu2 = slope(v, neg & none & ~approved)
    res.setdefault("B87c", {})[nm] = dict(slope_approved=ci(100 * s1), level_approved=ci(100 * mu1), slope_rejected=ci(100 * s2), level_rejected=ci(100 * mu2))
res["B87c"]["n_none_approved"] = int((neg & none & approved).sum()); res["B87c"]["n_none_rejected"] = int((neg & none & ~approved).sum())
print("B87c", res["B87c"], flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
