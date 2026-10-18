"""Addendum 78, B77g: comparisons between publications on Civil Comments reply threads (results/x14_pops/cc_threads.parquet).
Publications with >= 1,000 comments, every pair; metrics: the toxic-comment rate, the share of reply threads with at least one
toxic comment, the share of article sections with at least one. A pair is decided when gold's difference has a 95% article-cluster
bootstrap CI excluding 0 (500 Poisson resamples, seed 7760). Per instrument: the share of decided pairs whose order its point
difference reverses.
usage: python3 x78_pubs.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, itertools, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
pub = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "publication_id"]).set_index("id").publication_id.reindex(P.id.values).values
vc = pd.Series(pub).value_counts(); keep = vc[vc >= 1000].index.values; pubs = sorted(keep.tolist())
labs = {"gold": P.z.values.astype(float)} | {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    labs[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
Wb = np.vstack([np.ones(nC), np.random.default_rng(7760).poisson(1.0, size=(500, nC))])
pi = {p: i for i, p in enumerate(pubs)}; pidx = np.array([pi.get(p, -1) for p in pub]); ok = pidx >= 0
th = pd.factorize(P.unit.values)[0]; ar = pd.factorize(P.article.values)[0]
def unit_tab(u, v):
    """per-cluster x publication: number of units and number with at least one positive (units nest in clusters and publications)."""
    any_ = np.bincount(u, v) > 0; uc = np.zeros(u.max() + 1, np.int64); uc[u] = cl; up = np.full(u.max() + 1, -1); up[u] = pidx
    m = up >= 0; K = len(pubs)
    n = np.zeros((nC, K)); k = np.zeros((nC, K)); np.add.at(n, (uc[m], up[m]), 1.0); np.add.at(k, (uc[m], up[m]), any_[m].astype(float)); return n, k
def item_tab(v):
    K = len(pubs); n = np.zeros((nC, K)); k = np.zeros((nC, K)); np.add.at(n, (cl[ok], pidx[ok]), 1.0); np.add.at(k, (cl[ok], pidx[ok]), v[ok]); return n, k
rates = {}
for L, v in labs.items():
    rates[L] = {}
    for met, (n, k) in (("item", item_tab(v)), ("thread", unit_tab(th, v)), ("article", unit_tab(ar, v))):
        rates[L][met] = (Wb @ k) / np.maximum(Wb @ n, 1e-12)       # resamples x publications
res = {"n_pubs": len(pubs), "n_pairs": len(pubs) * (len(pubs) - 1) // 2, "metrics": {}}
for met in ("item", "thread", "article"):
    G = rates["gold"][met]; dec = []
    for a, b in itertools.combinations(range(len(pubs)), 2):
        d = G[:, a] - G[:, b]; lo, hi = np.percentile(d[1:], [2.5, 97.5])
        if lo > 0 or hi < 0:
            dec.append((a, b, np.sign(d[0])))
    out = {"n_decided": len(dec), "reversed": {}}
    for L in labs:
        if L == "gold":
            continue
        R = rates[L][met][0]; out["reversed"][L] = int(sum(np.sign(R[a] - R[b]) != s for a, b, s in dec))
    res["metrics"][met] = out
    print(met, "decided", len(dec), {L: v for L, v in out["reversed"].items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
