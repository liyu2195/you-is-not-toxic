"""Addendum 88, B88d: noise-free escalation and rater counts by depth (descriptive).
At each depth (0-9, >= 10) of Civil Comments reply threads: the univariate NPMLE of the rater probability pi on the m = 40 grid
(B85a's SQP, scripts/x85_sqp.py) from each comment's votes (k of n), with the Binomial kernel (pi's likelihood up to a factor free of
pi when the protocol adds raters on the strength of votes already cast); each comment's posterior probability that pi >= 0.5; its
OLS depth slope (points per level) = the depth law of a majority of infinitely many raters. 100 article-cluster Poisson resamples
(seed 8805), warm-started. Also the mean number of raters by depth and its OLS slope on depth.
usage: python3 x88_noisefree_depth.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
from scipy.special import gammaln
sys.path.insert(0, (_ROOT + "/code/analysis"))
from x85_sqp import npmle_sqp
W = _ROOT; OUT = sys.argv[1]; NB = 100; m = 40
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "depth", "z", "tox", "cluster"])
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); k = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); db = x.astype(int)
cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
g = (np.arange(m) + 0.5) / m; hi = g >= 0.5
Wb = np.vstack([np.ones(nC), np.random.default_rng(8805).poisson(1.0, size=(NB, nC))])
pat = pd.factorize(pd.MultiIndex.from_arrays([db, n, k]))[0]; S = int(pat.max() + 1)
first = np.zeros(S, np.int64); first[pat] = np.arange(len(pat))
pd_, pn, pk = db[first], n[first], k[first]
logL = (gammaln(pn + 1) - gammaln(pk + 1) - gammaln(pn - pk + 1))[:, None] + pk[:, None] * np.log(g)[None, :] + (pn - pk)[:, None] * np.log1p(-g)[None, :]
L = np.exp(logL - logL.max(1, keepdims=True))                                          # row scaling leaves the NPMLE unchanged


def ols(xv, yv, wv):
    sw = wv.sum(); xb = (wv * xv).sum() / sw; yb = (wv * yv).sum() / sw
    return float((wv * (xv - xb) * (yv - yb)).sum() / (wv * (xv - xb) ** 2).sum())


res = {"grid": m, "draws": []}; G0 = {}
for b in range(NB + 1):
    wc = Wb[b][cl]                                                                     # comment weights
    wp = np.bincount(pat, weights=wc, minlength=S)
    post = np.zeros(S)
    for d in range(11):
        sel = np.flatnonzero((pd_ == d) & (wp > 0))
        xg = npmle_sqp(L[sel], wp[sel], x0=G0.get(d)); xg = xg[0] if isinstance(xg, tuple) else xg
        if b == 0:
            G0[d] = xg
        Lx = L[sel] @ xg; post[sel] = (L[sel][:, hi] @ xg[hi]) / Lx
    q = post[pat]
    draw = dict(noise_free=100 * ols(x, q, wc), gold=100 * ols(x, z, wc), raters=ols(x, n.astype(float), wc))
    res["draws"].append(draw)
    if b == 0:
        res["point"] = draw
        res["by_depth"] = {str(d): dict(gold=float(z[db == d].mean()), noise_free=float(q[db == d].mean()), mean_raters=float(n[db == d].mean()), n=int((db == d).sum())) for d in range(11)}
        print("point", draw, flush=True); print(json.dumps(res["by_depth"]), flush=True)
    if b % 20 == 0:
        print("resample", b, {kk: round(v, 3) for kk, v in draw.items()}, flush=True)
D = res["draws"]
for kk in ("noise_free", "gold", "raters"):
    a = np.array([d[kk] for d in D]); res[kk] = [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
a = np.array([d["noise_free"] - d["gold"] for d in D]); res["noise_free_minus_gold"] = [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
del res["draws"]
print({kk: res[kk] for kk in ("noise_free", "gold", "raters", "noise_free_minus_gold")}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
