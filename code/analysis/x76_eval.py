"""Addendum 76, B76d evaluation: laws nobody planned for (the within-thread depth law 'depth_within' and prevalence), added to the
item regime with weight 0 in every allocation, on the 55 confirmation cells (results/x76_b4/*.json, from jobs/make_x76_jobs.py).
H76c: at B >= 1,000, over both targets, the frozen design (pps_seq4_ba) has mean coverage >= 0.935, >= 97% of (cell, budget,
target) entries >= 0.90 and none < 0.85. H76d: on the within-thread law at B >= 1,000, its median width ratio to PPI++ is <= 1.00.
Descriptive: width ratios to PPI++ and uniform sampling for both targets, coverage of every design, detection of the labeler's
within-thread distortion, and the planned metrics' coverage in the same runs.
usage: python3 x76_eval.py IN_DIR OUT_JSON"""
import sys, glob, json, numpy as np
IN, OUT = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"{IN}/*.json")):
    J = json.load(open(f))
    for r in J["results"]:
        for B, des in r["item"].items():
            for d, mets in des.items():
                for mn, v in mets.items():
                    if mn == "_labels":
                        continue
                    rows.append(dict(pop=J["pop"], lab=r["labeler"], B=int(B), design=d, metric=mn, cov=v["coverage"], width=v["width"], detect=v.get("detect"),
                                     truth=r["truth_item"].get(mn), labv=(r.get("labeler_item") or {}).get(mn)))
cells = sorted({(x["pop"], x["lab"]) for x in rows}); print(len(rows), "rows,", len(cells), "cells", flush=True)
T = ("depth_within", "prevalence")
sel = lambda **kw: [x for x in rows if all(x[k] == v for k, v in kw.items())]
res = {"n_cells": len(cells), "coverage": {}, "ratios": {}}
for d in ("pps_seq4_ba", "ppi", "srs"):
    for m in T + ("depth_law", "size_law", "placebo_depth"):
        e = [x["cov"] for x in rows if x["design"] == d and x["metric"] == m and x["B"] >= 1000]
        if e:
            res["coverage"][f"{d}/{m}"] = dict(n=len(e), mean=float(np.mean(e)), min=float(np.min(e)), share90=float(np.mean(np.array(e) >= 0.90)))
E = [x["cov"] for x in rows if x["design"] == "pps_seq4_ba" and x["metric"] in T and x["B"] >= 1000]
res["H76c"] = dict(n=len(E), mean=float(np.mean(E)), share90=float(np.mean(np.array(E) >= 0.90)), min=float(np.min(E)),
                   pass_=bool(np.mean(E) >= 0.935 and np.mean(np.array(E) >= 0.90) >= 0.97 and np.min(E) >= 0.85))
def ratio(m, ref, bmin=1000):
    out = []
    for (p, l) in cells:
        for B in sorted({x["B"] for x in rows if x["pop"] == p and x["lab"] == l}):
            if B < bmin:
                continue
            a = [x["width"] for x in rows if (x["pop"], x["lab"], x["B"], x["design"], x["metric"]) == (p, l, B, "pps_seq4_ba", m)]
            b = [x["width"] for x in rows if (x["pop"], x["lab"], x["B"], x["design"], x["metric"]) == (p, l, B, ref, m)]
            if a and b and a[0] and b[0]:
                out.append(a[0] / b[0])
    return out
for m in T:
    for ref in ("ppi", "srs"):
        v = ratio(m, ref); res["ratios"][f"{m}/{ref}"] = dict(n=len(v), median=float(np.median(v)), q10=float(np.percentile(v, 10)), q90=float(np.percentile(v, 90)))
res["H76d"] = dict(median=res["ratios"]["depth_within/ppi"]["median"], pass_=bool(res["ratios"]["depth_within/ppi"]["median"] <= 1.00))
for d in ("pps_seq4_ba", "ppi"):
    e = [x["detect"] for x in rows if x["design"] == d and x["metric"] == "depth_within" and x["B"] >= 1000 and x["detect"] is not None]
    res[f"detect_within/{d}"] = float(np.mean(e)) if e else None
print(json.dumps(res, indent=1))
mm = dict(qUnCovMean=f"{res['H76c']['mean']:.3f}", qUnCovMin=f"{res['H76c']['min']:.3f}", qUnShareNinety=f"{100 * res['H76c']['share90']:.0f}\\%",
          qUnWithinPPI=f"{res['ratios']['depth_within/ppi']['median']:.2f}", qUnWithinUnif=f"{res['ratios']['depth_within/srs']['median']:.2f}",
          qUnPrevPPI=f"{res['ratios']['prevalence/ppi']['median']:.2f}", qUnPrevUnif=f"{res['ratios']['prevalence/srs']['median']:.2f}", qUnNEntries=str(res["H76c"]["n"]),
          qUnWithinDetect=f"{res['detect_within/pps_seq4_ba']:.2f}" if res["detect_within/pps_seq4_ba"] is not None else "??",
          qUnWithinDetectPPI=f"{res['detect_within/ppi']:.2f}" if res["detect_within/ppi"] is not None else "??")
res["macros"] = mm
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT, mm)
