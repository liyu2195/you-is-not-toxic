"""Population identification regions given perfectly calibrated item probabilities (isotonic of gold on the LLM score, all
labeled comments): 'at least one' shares for threads and articles, and the reply-association difference (Frechet bounds),
with the independence point. usage: f14_regions.py SCORE(haiku|qwen7b) OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
name, OUT = sys.argv[1], sys.argv[2]; W = _ROOT
import os
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "Z", "parent_id", "article_id", "split"]].set_index("id")
if os.environ.get("CC_SPLIT", "train") != "all": meta = meta[meta.split == os.environ.get("CC_SPLIT", "train")]   # test splits: replies released without parents
meta = meta.drop(columns="split")
if name == "haiku":
    R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{W}/results/full/haiku_tox_ctx0/shard_*.jsonl")) for l in open(f)])
    if "error" in R: R = R[R["error"].isna()]                         # errored requests stay missing; no letter = not toxic (as f10)
    R = R.drop_duplicates("id", keep="last").set_index("id"); s = R.ans.map({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}).fillna(0.0)
else:
    ids, v = [], []
    for f in sorted(glob.glob(f"{W}/results/full/qwen7b/*/shard_*_ctx0.jsonl")):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            ids.append(r["id"]); v.append(r["p"][0] + r["p"][1])
    s = pd.Series(v, index=ids).groupby(level=0).last()
D = meta.join(s.rename("s"), how="inner")
D["h"] = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip").fit(D.s, D.Z).predict(D.s)
par = meta.parent_id.where(meta.parent_id.isin(meta.index)).dropna().astype("int64"); cur = pd.Series(meta.index.values, index=meta.index)
for _ in range(200):                                         # roots from the full reply structure (all comments)
    has = cur.isin(par.index)
    if not has.any(): break
    cur[has] = par.reindex(cur[has].values).values
meta["root"] = cur.values; D["root"] = meta.root.reindex(D.index).values
cD = D.groupby("root").size(); okr = cD == meta.groupby("root").size().reindex(cD.index)                 # keep complete threads
cA = D.groupby("article_id").size(); oka = cA == meta.groupby("article_id").size().reindex(cA.index)     # and complete articles
out = {"n_comments": int(len(D))}
for unit, ok in [("root", okr), ("article_id", oka)]:
    X = D[D[unit].map(ok)]; X = X[X.groupby(unit).Z.transform("size") >= 2]; g = X.groupby(unit)
    lo = g.h.max().mean(); hi = g.h.sum().clip(upper=1).mean(); ind = (1 - np.exp(X.assign(l=np.log1p(-X.h)).groupby(unit).l.sum())).mean()
    out["threads" if unit == "root" else "articles"] = dict(n=int(g.ngroups), gold=float(g.Z.max().mean()), frechet=[float(lo), float(hi)], independence=float(ind))
E = D[D.parent_id.isin(D.index)]; hp = D.h.reindex(E.parent_id.astype("int64")).values; hc = E.h.values
zp = D.Z.reindex(E.parent_id.astype("int64")).values; zc = E.Z.values
def a(j11): return np.sum(j11) / np.sum(hp) - np.sum(hc - j11) / np.sum(1 - hp)
g1, g0 = zc[zp == 1].mean(), zc[zp == 0].mean()
out["association"] = dict(n=int(len(E)), gold_diff=float(g1 - g0), gold_rr=float(g1 / g0), independence_diff=float(a(hp * hc)),
                          independence_rr=float((np.sum(hp * hc) / np.sum(hp)) / (np.sum(hc - hp * hc) / np.sum(1 - hp))),
                          frechet_diff=[float(a(np.maximum(0, hp + hc - 1))), float(a(np.minimum(hp, hc)))])
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out, indent=1))
