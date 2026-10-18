"""Addendum 80, B80b: gold-threshold curves. Gold_t = crowd vote fraction >= t, t in {0.3, ..., 0.8}. Every instrument with a
continuous score is thresholded to flag gold_t's share (top-k by score, ties at random, seed 8010 + t index): the seven classifiers
and Qwen on Civil Comments reply threads; Gemma and the Haiku and Sonnet 0-100 ratings (B75a, without the parent) on the 5% sample.
Against gold_t, on the same 500 article-cluster Poisson resamples (seed 8011): the log-RR distortion and the depth-law distortion.
H80c: at each t in {0.4, 0.5, 0.6, 0.7}, the median log-RR distortion over the seven classifiers is > 0 and >= 4 of 7 have paired CIs > 0.
usage: python3 x80_threshold.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code/analysis"))
from x80_common import Pop, ci, top_share
W = _ROOT; OUT = sys.argv[1]; PP = 100.0
TS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CLS7 = ["detoxify_original", "detoxify_unbiased", "imsypp", "sn_rob", "mh_dist", "fb_hate", "cn_off"]
def ratings(name, ids):
    R = {}
    for l in open(f"{W}/results/x75_rating/{name}.jsonl"):
        r = json.loads(l)
        if r.get("rating") is not None:
            R[int(r["id"])] = float(r["rating"])
    return np.array([R.get(int(i), np.nan) for i in ids])
res = {"t": TS, "pops": {}}
for pop in ("cc_threads", "cc_s5"):
    S = Pop(pop, seed=8011); P = S.P
    if pop == "cc_threads":
        SC = {c: P[f"s:{c}@m"].values.astype(float) for c in CLS7} | {"qwen7b_m": P["s:qwen7b_m"].values.astype(float)}
    else:
        ids = P.id.astype(np.int64).values
        SC = {"gemma_m": P["s:gemma_m"].values.astype(float), "haiku_rating": ratings("haiku_ctx0", ids), "sonnet_rating": ratings("sonnet_ctx0", ids)}
    out = {"N": S.N, "by_t": {}}
    for j, t in enumerate(TS):
        g = (S.tox >= t - 1e-12).astype(float); k = int(g.sum()); gl, gd = S.lrr(g), S.depth(g)
        row = {"prev": float(g.mean()), "gold_rr": float(np.exp(gl[0])), "gold_depth": float(gd[0]), "instruments": {}}
        for name, s in SC.items():
            y = top_share(s, k, 8010 + j); il, idp = S.lrr(y), S.depth(y)
            row["instruments"][name] = dict(rr=float(np.exp(il[0])), lrr_minus_gold=ci(il - gl), depth=float(idp[0]), depth_minus_gold=ci(idp - gd))
        out["by_t"][f"{t:g}"] = row
        print(pop, f"t={t:.1f} prev={g.mean():.3f} gold RR {np.exp(gl[0]):.2f} depth {gd[0]*PP:+.2f} |",
              " ".join(f"{n}:{v['rr']:.2f}[{v['lrr_minus_gold'][1]:+.2f},{v['lrr_minus_gold'][2]:+.2f}]" for n, v in row["instruments"].items()), flush=True)
    res["pops"][pop] = out
H = {}
for t in ("0.4", "0.5", "0.6", "0.7"):
    I = res["pops"]["cc_threads"]["by_t"][t]["instruments"]; d = [I[c]["lrr_minus_gold"] for c in CLS7]
    med = float(np.median([x[0] for x in d])); nup = int(sum(x[1] > 0 for x in d)); H[t] = dict(median=med, n_ci_above=nup, ok=bool(med > 0 and nup >= 4))
res["H80c"] = dict(by_t=H, pass_=all(v["ok"] for v in H.values()))
print("H80c", res["H80c"])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
