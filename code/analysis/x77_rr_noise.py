"""Addendum 77, B77c: the contagion of a less noisy gold (descriptive; Civil Comments reply threads, results/x14_pops/cc_threads).
Edges whose two ends both have >= 10 raters (rule fixed from the rater counts alone). On these edges: the reply RR under
k-rater majorities (k = 1, 3, 5, 7, 9 raters drawn without replacement per comment, toxic when at least half of the k are; 20 draws
each, seed 7720), and under the full majority (gold). Fit log RR(k) = a + b / k by least squares over the five k and the full
majority (at k = its mean rater count on these comments); RR_inf = exp(a). Every instrument's RR on the same edges.
usage: python3 x77_rr_noise.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; K = 10
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = (pa != ch) & (n[ch] >= K) & (n[pa] >= K); ch, pa = ch[keep], pa[keep]
rows = np.unique(np.r_[ch, pa]); print("edges", len(ch), "comments", len(rows), "mean raters", n[rows].mean(), flush=True)
def rr(v):
    a, b = v[pa], v[ch]; return float((b[a == 1].mean()) / (b[a == 0].mean()))
z = P.z.values.astype(float); res = {"n_edges": int(len(ch)), "n_comments": int(len(rows)), "mean_raters": float(n[rows].mean()), "gold_rr": rr(z), "k": {}}
rng = np.random.default_rng(7720)
for k in (1, 3, 5, 7, 9):
    vals = []
    for _ in range(20):
        t = rng.hypergeometric(kv, n - kv, np.minimum(k, n)); y = (t / np.minimum(k, n) >= 0.5 - 1e-12).astype(float); vals.append(rr(y))
    res["k"][k] = dict(median=float(np.median(vals)), sd=float(np.std(vals)))
kk = np.array([1, 3, 5, 7, 9, n[rows].mean()]); ll = np.log(np.array([res["k"][k]["median"] for k in (1, 3, 5, 7, 9)] + [res["gold_rr"]]))
A = np.column_stack([np.ones(len(kk)), 1 / kk]); coef = np.linalg.lstsq(A, ll, rcond=None)[0]; res["rr_inf"] = float(np.exp(coef[0])); res["fit_b"] = float(coef[1])
res["instruments"] = {c[2:]: rr(np.nan_to_num(P[c].values.astype(float))) for c in P.columns if c.startswith("y:")}
for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    res["instruments"][f"{m}@0.5"] = rr((P[f"s:{m}@m"].values >= 0.5).astype(float))
print(json.dumps({k: v for k, v in res.items() if k != "instruments"}, indent=1)); print({k: round(v, 2) for k, v in res["instruments"].items()})
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
