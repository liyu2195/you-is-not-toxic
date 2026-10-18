"""v6 (readability round): the exact decomposition of the depth law among replies, for the LLMs shown the parent (no new labels).
Figure 2's earlier "+ parent" rows pooled roots labeled without the parent with replies labeled with it, so a level shift at the
root/reply boundary entered the depth slope. Here every "+ parent" labeler is decomposed among replies only (depth >= 1), the
population of Table 2's "shown the parent" row: Haiku and Qwen (Qwen at its own gold-prevalence threshold, as x76_rethreshold.py
draws it, seed 7650) on Civil Comments reply threads; Sonnet and Gemma (Gemma at its own threshold) on the 5% sample.
For each: TPR and FPR among replies, the three terms of Proposition 1 for the depth law (points per level), the distortion and its
95% interval from 500 Poisson cluster-bootstrap resamples (seed 7651, clusters as in x76_rethreshold.py).
Writes results/x97/x97_decomp_replies.json.
usage: python3 x97_decomp_replies.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os
import numpy as np, pandas as pd
W = _ROOT; OUT = f"{W}/results/x97/x97_decomp_replies.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True); NB = 500
res = {}
for pop, labs in (("cc_threads", ("haiku_ctx", "qwen7b_m_ctx@own")), ("cc_s5", ("sonnet_ctx", "gemma_m_ctx@own"))):
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(float); N = len(z); prev = z.mean()
    flags = {}
    rng = np.random.default_rng(7650)
    for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:                    # as x76_rethreshold.py
        s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(prev * N))
        order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0; flags[f"{m}@own"] = f
    for m in ("haiku_ctx", "sonnet_ctx"):
        if f"y:{m}" in P:
            flags[m] = np.nan_to_num(P[f"y:{m}"].values.astype(float))
    x = np.minimum(P.depth.values, 10).astype(float); r = x >= 1
    cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
    Wb = np.vstack([np.ones(nC), np.random.default_rng(7651).poisson(1.0, size=(NB, nC))])
    def sums(v):
        return np.vstack([np.bincount(cl[r], weights=w, minlength=nC) for w in (np.ones(r.sum()), x[r], x[r] ** 2, v[r], x[r] * v[r])])
    def slope(S):                                                                          # S: 5 x B weighted sums
        n, sx, sxx, sv, sxv = S
        return 100 * (sxv - sx * sv / n) / (sxx - sx * sx / n)
    Sz = Wb @ sums(z).T
    gold = slope(Sz.T)
    xr = x[r]; w = (xr - xr.mean()) / ((xr - xr.mean()) ** 2).sum(); zr = z[r]
    res[pop] = {"gold_depth_replies": [float(gold[0]), *map(float, np.percentile(gold[1:], [2.5, 97.5]))], "labelers": {}}
    for m in labs:
        y = flags[m]; yr = y[r]
        tpr = yr[zr == 1].mean(); fpr = yr[zr == 0].mean()
        LZ = 100 * (w * zr).sum(); LY = 100 * (w * yr).sum()
        uni = (tpr - fpr - 1) * LZ; sfp = 100 * (w[zr == 0] * (yr[zr == 0] - fpr)).sum(); stp = 100 * (w[zr == 1] * (yr[zr == 1] - tpr)).sum()
        assert abs(uni + sfp + stp - (LY - LZ)) < 1e-9
        d = slope((Wb @ sums(y).T).T) - gold
        res[pop]["labelers"][m] = dict(tpr=float(tpr), fpr=float(fpr), law=float(LY), uniform=float(uni), fp=float(sfp), tp=float(stp),
                                       distortion=[float(d[0]), *map(float, np.percentile(d[1:], [2.5, 97.5]))], fp_share=float(sfp / (LY - LZ)))
        print(pop, m, {k: (round(v, 3) if isinstance(v, float) else [round(t, 3) for t in v]) for k, v in res[pop]["labelers"][m].items()})
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
