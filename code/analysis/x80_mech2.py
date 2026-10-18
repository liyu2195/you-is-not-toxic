"""Addendum 80, follow-up to B80d: the reply risk ratio within articles (Civil Comments reply threads; descriptive).
Mantel-Haenszel risk ratio over reply edges, stratified by article, for gold and for every instrument (and the crude RR for
reference). 200 article-cluster Poisson resamples (seed 8052): the CI of each instrument's MH log RR minus gold's.
usage: python3 x80_mech2.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
art = pd.factorize(P.article.values)[0][ch]; nA = art.max() + 1
cl = pd.factorize(P.cluster.values)[0][ch]; nC = cl.max() + 1
# articles nest in clusters (clusters are articles in x14_pops); resample clusters and carry their articles
a2c = np.zeros(nA, np.int64); a2c[art] = cl
Wb = np.vstack([np.ones(nC), np.random.default_rng(8052).poisson(1.0, size=(200, nC))])[:, a2c]      # weights per article
def mh(v):
    yp, yc = v[pa], v[ch]
    a_ = np.bincount(art, yp * yc, nA); n1 = np.bincount(art, yp, nA); c_ = np.bincount(art, (1 - yp) * yc, nA); n0 = np.bincount(art, 1 - yp, nA); nn = n1 + n0
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.where(nn > 0, a_ * n0 / nn, 0.0); den = np.where(nn > 0, c_ * n1 / nn, 0.0)
    return np.log((Wb @ num) / (Wb @ den))
def crude(v):
    yp, yc = v[pa], v[ch]
    return float(np.log((np.sum(yp * yc) / np.sum(yp)) / (np.sum((1 - yp) * yc) / np.sum(1 - yp))))
z = P.z.values.astype(float); g = mh(z)
ins = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    ins[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
res = {"gold": dict(crude_rr=float(np.exp(crude(z))), mh_rr=float(np.exp(g[0])), mh_ci=[float(np.exp(np.percentile(g[1:], 2.5))), float(np.exp(np.percentile(g[1:], 97.5)))]), "instruments": {}}
print("gold crude", round(res["gold"]["crude_rr"], 2), "MH within articles", round(res["gold"]["mh_rr"], 2), flush=True)
for k, v in ins.items():
    m = mh(v); d = m - g
    res["instruments"][k] = dict(crude_rr=float(np.exp(crude(v))), mh_rr=float(np.exp(m[0])), mh_lrr_minus_gold=[float(d[0]), float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))])
    r = res["instruments"][k]; print(f"{k:24s} crude {r['crude_rr']:.2f} MH {r['mh_rr']:.2f} diff {d[0]:+.3f} [{r['mh_lrr_minus_gold'][1]:+.3f}, {r['mh_lrr_minus_gold'][2]:+.3f}]", flush=True)
up = [k for k in CLS12 if res["instruments"][k]["mh_lrr_minus_gold"][1] > 0]; res["n_cls_above_within"] = len(up); res["cls_above_within"] = up
print("classifier settings above gold within articles:", len(up), "of 12")
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
