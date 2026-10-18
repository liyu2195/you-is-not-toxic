"""Addendum 78, B77h: what calibrated item labels allow for unit shares (Civil Comments reply threads and article sections).
Each comment's gold probability is the gold rate of its score level (Haiku, Haiku shown the parent: their four levels) or of its flag
(every other instrument), on the whole population. Over all joint laws with these marginals, the share of units with at least one
positive ranges over [mean_u max_i p_i, mean_u min(1, sum_i p_i)] (Frechet); the independence value is 1 - prod_i (1 - p_i).
usage: python3 x78_frechet.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet"); z = P.z.values.astype(float)
res = {}
for unit in ("unit", "article"):
    u = pd.factorize(P[unit].values)[0]; gold = float((np.bincount(u, z) > 0).mean()); res[unit] = {"gold": gold, "instruments": {}}
    for c in [c for c in P.columns if c.startswith("y:")]:
        L = c[2:]
        lev = P[f"s:{L}"].values if L in ("haiku", "haiku_ctx") else P[c].values
        lev = np.nan_to_num(np.asarray(lev, float), nan=-1.0); keys, inv = np.unique(lev, return_inverse=True)
        p = (np.bincount(inv, z) / np.bincount(inv))[inv]
        lo = float(pd.Series(p).groupby(u).max().mean()); hi = float(np.minimum(1.0, np.bincount(u, p)).mean())
        ind = float((1 - np.exp(np.bincount(u, np.log(np.maximum(1 - p, 1e-12))))).mean())
        res[unit]["instruments"][L] = dict(lo=lo, hi=hi, independence=ind, width=hi - lo)
    print(unit, "gold", round(gold, 3), {L: (round(v["lo"], 3), round(v["hi"], 3), round(v["independence"], 3)) for L, v in res[unit]["instruments"].items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
