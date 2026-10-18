"""Addendum 77, B77e: unit shares as a law (descriptive). For every instrument of results/x14_pops/<pop>.parquet (plus the fresh
classifiers at 0.5; the lenient human excluded): the share of units with at least one flag, against gold's share of units with at
least one positive, over all units and over units with >= 2 comments; the distortion's z from 500 Poisson cluster-bootstrap
resamples (seed 7740; units nest in clusters).
usage: python3 x77_units.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; FRESH = ["sn_rob", "mh_dist", "fb_hate", "cn_off"]
res = {}
for pop in ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]:
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(float)
    labs = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    for m in FRESH:
        if f"s:{m}@m" in P:
            labs[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
    uu = pd.factorize(P.unit.values)[0]; ku = np.bincount(uu); cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
    ucl = np.zeros(ku.size, np.int64); ucl[uu] = cl
    Wb = np.vstack([np.ones(nC), np.random.default_rng(7740).poisson(1.0, size=(500, nC))])
    def share(v, mask):
        a = np.bincount(uu, v, ku.size) > 0
        return (Wb @ np.bincount(ucl[mask], a[mask].astype(float), nC)) / (Wb @ np.bincount(ucl[mask], np.ones(mask.sum()), nC))
    out = {}
    for nm, mask in (("all", np.ones(ku.size, bool)), ("ge2", ku >= 2)):
        g = share(z, mask); out[nm] = {"n_units": int(mask.sum()), "gold": float(g[0]), "labelers": {}}
        for L, y in labs.items():
            s = share(y, mask); d = s - g; out[nm]["labelers"][L] = dict(share=float(s[0]), distortion=float(d[0]), ratio=float(s[0] / g[0]), z=float(d[0] / np.std(d[1:], ddof=1)))
    res[pop] = out
    print(pop, "gold", round(out["all"]["gold"], 3), {L: round(v["share"], 3) for L, v in out["all"]["labelers"].items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
