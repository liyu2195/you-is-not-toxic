"""Addendum 74, A3 (descriptive): depth slopes on CCC against in-context gold. Loading as f93_ccc_depth.py (H12): in-context
gold = mean in-context code >= .5, out-of-context gold = the original crowd's; Haiku flags = Toxic or Very toxic; Qwen at the
threshold matched to the out-of-context gold prevalence; depth from the train-split reply structure, depth >= 1.
Statistic: OLS slope of each label on min(depth, 6), per reply level, with 2,000 bootstrap resamples of posts; differences from
the in-context humans' slope are paired. Also the slope among in-context-negative posts (the placebo depth slope, absolute and
relative to the FPR).
usage: python3 x74_ccc.py CCC_CSV OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, ast, numpy as np, pandas as pd
CCC, OUT = sys.argv[1], sys.argv[2]; W = _ROOT; rng = np.random.default_rng(7430)
d = pd.read_csv(CCC); mean = lambda s: float(np.mean(ast.literal_eval(s))) if isinstance(s, str) else np.nan
d["h_oc"] = d.tox_codes_oc.map(mean); d["h_ic"] = d.tox_codes_ic.map(mean)
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id", "split"]); meta = meta[meta.split == "train"].set_index("id")
par = meta.parent_id.where(meta.parent_id.isin(meta.index)).dropna().astype("int64"); cur = pd.Series(meta.index.values, index=meta.index); dep = pd.Series(0, index=meta.index)
for _ in range(200):
    has = cur.isin(par.index)
    if not has.any(): break
    cur[has] = par.reindex(cur[has].values).values; dep[has.values] += 1
d["depth"] = dep.reindex(d.id).values
def rd(f):
    rows = []
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: continue
    R = pd.DataFrame(rows)
    if "p" in R: R = R.drop_duplicates("id", keep="last").set_index("id"); return pd.Series([p[0] + p[1] for p in R.p], index=R.index)
    R = R[R.ans.notna()].drop_duplicates("id", keep="last").set_index("id"); return R.ans.map({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0})
d["ha_oc"] = rd(f"{W}/results/ccc/haiku_tox_ctx0.jsonl").reindex(d.id).values; d["ha_ic"] = rd(f"{W}/results/ccc/haiku_tox_ctx1.jsonl").reindex(d.id).values
d["qw_oc"] = rd(f"{W}/results/ccc/qwen7b/ccc_ctx0.jsonl").reindex(d.id).values; d["qw_ic"] = rd(f"{W}/results/ccc/qwen7b/ccc_ctx1.jsonl").reindex(d.id).values
d = d.dropna(subset=["h_oc", "h_ic", "ha_oc", "ha_ic", "qw_oc", "qw_ic", "depth"]); d = d[d.depth >= 1].copy()
d["Y_hic"] = (d.h_ic >= .5).astype(int); d["Y_hoc"] = (d.h_oc >= .5).astype(int)
d["Y_haiku_oc"] = (d.ha_oc >= 2 / 3 - 1e-9).astype(int); d["Y_haiku_ic"] = (d.ha_ic >= 2 / 3 - 1e-9).astype(int)
t = float(np.quantile(d.qw_oc, 1 - d.Y_hoc.mean())); d["Y_qwen_oc"] = (d.qw_oc > t).astype(int); d["Y_qwen_ic"] = (d.qw_ic > t).astype(int)
SRC = ["Y_hic", "Y_hoc", "Y_haiku_ic", "Y_haiku_oc", "Y_qwen_ic", "Y_qwen_oc"]
x = np.minimum(d.depth.values, 6).astype(float); V = {c: d[c].values.astype(float) for c in SRC}; neg = d.Y_hic.values == 0
def slope(xv, yv):
    xc = xv - xv.mean(); return float((xc * (yv - yv.mean())).sum() / (xc * xc).sum())
def stats(ix):
    o = {f"slope_{c}": slope(x[ix], V[c][ix]) for c in SRC}
    for c in SRC[1:]:
        o[f"diff_{c}_vs_Y_hic"] = o[f"slope_{c}"] - o["slope_Y_hic"]
    m = ix[neg[ix]]
    for c in SRC[2:]:
        s_ = slope(x[m], V[c][m]); f_ = float(V[c][m].mean()); o[f"placebo_{c}"] = s_; o[f"placebo_rel_{c}"] = s_ / f_ if f_ > 0 else np.nan
    return o
base = stats(np.arange(len(d))); B = [stats(rng.integers(0, len(d), len(d))) for _ in range(2000)]
ci = {k: [base[k], float(np.nanpercentile([b[k] for b in B], 2.5)), float(np.nanpercentile([b[k] for b in B], 97.5))] for k in base}
res = {"n": int(len(d)), "n_by_depth": {int(k): int(v) for k, v in pd.Series(x).value_counts().sort_index().items()}, "qwen_threshold": t, "estimates": ci,
       "rates_by_depth": {c: {int(k): float(v) for k, v in pd.Series(V[c]).groupby(x).mean().items()} for c in SRC}}
json.dump(res, open(OUT, "w"), indent=1)
for k, v in ci.items(): print(f"{k:34s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
