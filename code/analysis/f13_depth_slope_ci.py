"""H3: article-cluster bootstrap CI for (annotator depth slope - gold depth slope) on Civil Comments threads (>= 2 comments),
slope = Cov(min(depth, 10), label) / Var(min(depth, 10)), as in f10. Poisson(1) weights over articles, 1,000 replicates.
usage: f13_depth_slope_ci.py THREAD_ITEMS_PARQUET OUT_JSON"""
import sys, json, numpy as np, pandas as pd
T = pd.read_parquet(sys.argv[1]); OUT = sys.argv[2]; rng = np.random.default_rng(13); B = 1000
LAB = [c for c in T.columns if c not in ("root", "article_id", "depth", "tk")]
T["x"] = T.depth.clip(upper=10).astype(float); T["x2"] = T.x ** 2
agg = {"n": ("x", "size"), "sx": ("x", "sum"), "sxx": ("x2", "sum")}
for c in LAB: T[f"y_{c}"] = T[c].astype(float); T[f"xy_{c}"] = T.x * T[c]; agg[f"sy_{c}"] = (f"y_{c}", "sum"); agg[f"sxy_{c}"] = (f"xy_{c}", "sum")
A = T.groupby("article_id").agg(**agg); M = A.values; cols = list(A.columns); ix = {c: i for i, c in enumerate(cols)}
def slopes(S):                      # S: (..., n_stats) weighted totals
    n, sx, sxx = S[..., ix["n"]], S[..., ix["sx"]], S[..., ix["sxx"]]; vx = sxx / n - (sx / n) ** 2
    return {c: (S[..., ix[f"sxy_{c}"]] / n - sx * S[..., ix[f"sy_{c}"]] / n ** 2) / vx for c in LAB}
base = slopes(M.sum(0)); bs = {c: [] for c in LAB}
for _ in range(B // 50):
    Wt = rng.poisson(1.0, size=(50, len(A))).astype(float); S = Wt @ M; sl = slopes(S)
    for c in LAB: bs[c].append(sl[c])
bs = {c: np.concatenate(v) for c, v in bs.items()}
res = {"n_articles": int(len(A)), "n_comments": int(len(T)), "slope": {c: float(base[c]) for c in LAB}, "minus_gold": {}}
for c in LAB:
    if c == "Z": continue
    d = bs[c] - bs["Z"]; res["minus_gold"][c] = [float(base[c] - base["Z"]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
res["gold_ci"] = [float(np.percentile(bs["Z"], 2.5)), float(np.percentile(bs["Z"], 97.5))]
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res))
