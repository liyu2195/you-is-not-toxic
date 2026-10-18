"""Addendum 61: ceiling efficiency V*_I / V_design per development cell, metric and design (V_design = empirical MSE).
usage: python3 x10_gap.py OUT_JSON SIMDIR [SIMDIR ...] [--designs a,b] [--ceil results/x10_ceiling]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, os
import numpy as np
argv = sys.argv[1:]
def opt(name, default):
    global argv
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; argv = argv[:i] + argv[i + 2:]; return v
    return default
DES = opt("--designs", "upgrade_lw").split(","); CEIL = opt("--ceil", (_ROOT + "/results/x10_ceiling"))
OUT, dirs = argv[0], argv[1:]
rows = []
for f in sorted(glob.glob(f"{CEIL}/*__*.json")):
    C = json.load(open(f)); tag = os.path.basename(f)[:-5]
    sims = {}
    for d in dirs:
        for g in [f"{d}/{tag}.json"] + sorted(glob.glob(f"{d}/{tag}__g*.json")):
            if not os.path.exists(g):
                continue
            R = json.load(open(g))["results"][0]
            te = R.get("truth_edge") or {}
            scale = {"log_rr": te.get("rr"), "log_placebo_rr": te.get("placebo_rr")}      # edge estimates are stored on the RR scale
            for reg in ("item", "edge"):
                for B, dd in (R.get(reg) or {}).items():
                    for des, mm in dd.items():
                        sims.setdefault((reg, int(B), des), {}).update({m: (dict(c, rmse=c["rmse"] / scale[m]) if reg == "edge" and c.get("rmse") and scale.get(m) else c)
                                                                         for m, c in mm.items() if m != "_labels"})
    for reg in ("item", "edge"):
        for I, byB in C.get(reg, {}).items():
            for B, mm in byB.items():
                for m, v in mm.items():
                    vstar = v["compromise"] if reg == "item" else v
                    for des in DES:
                        c = sims.get((reg, int(B), des), {}).get(m)
                        if c and c.get("rmse"):
                            rows.append(dict(cell=tag, reg=reg, I=I, B=int(B), metric=m, design=des, eff=vstar / c["rmse"] ** 2, coverage=c["coverage"]))
json.dump(rows, open(OUT, "w"))
import collections
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["reg"], r["design"], r["I"], r["metric"])].append(r["eff"])
for k in sorted(agg):
    v = np.array(agg[k]); print(f"{k[0]:5s} {k[1]:11s} {k[2]} {k[3]:18s} median eff {np.median(v):.2f}  [{np.min(v):.2f}, {np.max(v):.2f}]  n={len(v)}")
