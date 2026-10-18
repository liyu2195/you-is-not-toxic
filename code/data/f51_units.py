"""Build population unit tables for the THREADSAFE evaluation from the full labels.
Outputs (results/f51_*.parquet): units for threads (k>=2), articles, derailment prefixes (t=5), and reply edges, each with gold z,
LLM score s and flag y, for Haiku 5.5 (ordinal score; flag = Toxic/Very toxic) and Qwen2.5-7B (prob; flag = matched threshold).
usage: f51_units.py [haiku|qwen7b ...]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, numpy as np, pandas as pd
W = _ROOT; FULL = f"{W}/results/full"
import os
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "Z", "parent_id", "article_id", "created_date", "split"]].set_index("id")
if os.environ.get("CC_SPLIT", "train") != "all": meta = meta[meta.split == os.environ.get("CC_SPLIT", "train")]   # test splits: replies released without parents
meta = meta.drop(columns="split")
def scores(name):
    if name == "haiku":
        R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{FULL}/haiku_tox_ctx0/shard_*.jsonl")) for l in open(f)])
        if "error" in R: R = R[R["error"].isna()]                     # errored requests stay missing; no letter = not toxic (as f10)
        R = R.drop_duplicates("id", keep="last").set_index("id"); s = R.ans.map({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}).fillna(0.0)
        return s, (s >= 2 / 3).astype(int)
    ids, v = [], []
    for f in sorted(glob.glob(f"{FULL}/qwen7b/*/shard_*_ctx0.jsonl")):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            ids.append(r["id"]); v.append(r["p"][0] + r["p"][1])
    s = pd.Series(v, index=ids).groupby(level=0).last(); s = s[s.index.isin(meta.index)]      # same population as f10
    t = np.quantile(s.values, 1 - meta.Z.reindex(s.index).mean()); return s, (s > t).astype(int)
par = meta.parent_id.where(meta.parent_id.isin(meta.index)).dropna().astype("int64")
cur = pd.Series(meta.index.values, index=meta.index)
for _ in range(200):
    has = cur.isin(par.index)
    if not has.any(): break
    cur[has] = par.reindex(cur[has].values).values
meta["root"] = cur.values
for name in sys.argv[1:] or ["haiku"]:
    s, y = scores(name); D = meta.join(pd.DataFrame({"s": s, "y": y}), how="inner")
    rng = np.random.default_rng(0); D["s"] = D.s + 1e-6 * rng.random(len(D))          # random tie-breaking
    cD = D.groupby("root").size(); ok_root = cD == meta.groupby("root").size().reindex(cD.index)          # every comment labeled
    cA = D.groupby("article_id").size(); ok_art = cA == meta.groupby("article_id").size().reindex(cA.index)
    T = D[D.root.map(ok_root)]; T = T[T.groupby("root").Z.transform("size") >= 2]
    T[["root", "Z", "s", "y"]].rename(columns={"root": "unit", "Z": "z"}).to_parquet(f"{W}/results/f51_threads_{name}.parquet")
    A = D[D.article_id.map(ok_art)]; A = A[A.groupby("article_id").Z.transform("size") >= 2]
    A[["article_id", "Z", "s", "y"]].rename(columns={"article_id": "unit", "Z": "z"}).to_parquet(f"{W}/results/f51_articles_{name}.parquet")
    Tp = T.sort_values("created_date").copy(); Tp["pos"] = Tp.groupby("root").cumcount() + 1; Tp = Tp[(Tp.groupby("root").Z.transform("size") >= 5) & (Tp.pos <= 5)]
    Tp[["root", "Z", "s", "y"]].rename(columns={"root": "unit", "Z": "z"}).to_parquet(f"{W}/results/f51_prefix5_{name}.parquet")
    E = D[D.parent_id.isin(D.index)].copy(); E["parent"] = E.parent_id.astype("int64")
    E = pd.DataFrame({"parent": E.parent.values, "child": E.index.values, "Zp": D.Z.reindex(E.parent).values, "Zc": E.Z.values,
                      "Ap": D.y.reindex(E.parent).values, "Ac": E.y.values})
    E.to_parquet(f"{W}/results/f51_edges_{name}.parquet")
    # items for the verification price list (f27): depth via parent links, publication
    dep = pd.Series(0, index=D.index); cur2 = pd.Series(D.index.values, index=D.index); pm = D.parent_id.where(D.parent_id.isin(meta.index)).dropna().astype("int64")
    for _ in range(200):
        has = cur2.isin(pm.index)
        if not has.any(): break
        cur2[has] = pm.reindex(cur2[has].values).values; dep[has] += 1
    pub = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "publication_id"]].set_index("id").publication_id
    Iv = D[D.root.map(ok_root)]
    pd.DataFrame({"id": Iv.index.values, "article": Iv.article_id.values, "pub": pub.reindex(Iv.index).values, "root": Iv.root.values,
                  "depth": dep.reindex(Iv.index).values, "parent": Iv.parent_id.fillna(-1).astype("int64").values,
                  "z": Iv.Z.values, "s": Iv.s.values, "y": Iv.y.values}).to_parquet(f"{W}/results/f51_items_{name}.parquet")
    print(name, "threads", T.root.nunique(), "articles", A.article_id.nunique(), "prefix5", Tp.root.nunique(), "edges", len(E), flush=True)
