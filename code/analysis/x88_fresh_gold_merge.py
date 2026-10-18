"""Addendum 88, B88f': merges the parts written by scripts/x88_fresh_gold_part.py into the summary that scripts/x88_fresh_gold.py
writes (same fields: per labeler the truths, per budget and process the coverage of the expected law with and without the rater
variance, the coverage of the realized law, widths, width relative to fixed gold, bias/sd, exposure, raters per label; then
B88f_prime and H88f over all entries). Audits are taken in index order; every (labeler, budget) must have audits 0..REPS-1 exactly.
usage: python3 x88_fresh_gold_merge.py OUT_JSON PART_JSON ..."""
import sys, json, numpy as np
OUT, parts = sys.argv[1], sys.argv[2:]
PROCS = ["fixed", "one", "three", "adaptive"]
G = {}
for f in parts:
    d = json.load(open(f)); G.setdefault((d["lab"], d["B"]), []).append(d)
labs = list(dict.fromkeys(k[0] for k in sorted(G, key=lambda k: (k[0], k[1]))))
ORDER = ["cn_off@m", "fb_hate@m", "gemma_m", "mh_dist@m", "sn_rob@m", "sonnet", "sonnet_ctx"]
labs = [l for l in ORDER if l in labs] + [l for l in labs if l not in ORDER]
BUD = sorted({k[1] for k in G})
res = {"processes": PROCS, "cells": {}, "parts": len(parts)}
for lab in labs:
    cell = None
    for B in BUD:
        ds = sorted(G[(lab, B)], key=lambda d: d["r0"])
        cover = [r for d in ds for r in range(d["r0"], d["r1"])]
        assert cover == list(range(len(cover))), (lab, B, "audits missing or duplicated")
        if cell is None:
            cell = {"truth": ds[0]["truth"], "reps": len(cover)}
        assert all(d["truth"] == cell["truth"] for d in ds)
        names = ds[0]["metrics"]; out = {}
        for pr in PROCS:
            out[pr] = {}
            for m in names:
                A = np.array([row for d in ds for row in d["acc"][pr][m]], float); th, lo, hi, real, lo2, hi2 = A.T
                ok = np.isfinite(lo) & np.isfinite(hi); ok2 = np.isfinite(lo2) & np.isfinite(hi2); tE = cell["truth"][pr][m]
                out[pr][m] = dict(cov_expected=float(np.mean(~ok | ((lo <= tE) & (tE <= hi)))), cov_realized=float(np.mean(~ok | ((lo <= real) & (real <= hi)))),
                                  width=float(np.mean((hi - lo)[ok])), bias_sd=float((np.nanmean(th) - tE) / np.nanstd(th)) if np.nanstd(th) > 0 else None,
                                  cov_expected_rv=float(np.mean(~ok2 | ((lo2 <= tE) & (tE <= hi2)))), width_rv=float(np.mean((hi2 - lo2)[ok2])))
            ex = [e for d in ds for e in d["expo"][pr]]
            out[pr]["_exposure"] = float(np.mean(ex)) if ex else None
        ra = [x for d in ds for x in d["raters"]]
        out["_raters_adaptive"] = float(np.mean(ra)) if ra else None
        for pr in ("one", "three", "adaptive"):
            for m in names:
                out[pr][m]["width_vs_fixed"] = out[pr][m]["width"] / out["fixed"][m]["width"]
                out[pr][m]["width_rv_vs_fixed"] = out[pr][m]["width_rv"] / out["fixed"][m]["width"]
        cell[str(B)] = out
        print(lab, B, {pr: [round(out[pr][m]["cov_expected_rv"], 3) for m in names] for pr in PROCS},
              "width_rv/fixed", {pr: [round(out[pr][m]["width_rv_vs_fixed"], 2) for m in names] for pr in ("one", "three", "adaptive")},
              "raters", out["_raters_adaptive"], flush=True)
    res["cells"][lab] = cell
ent = [(l, B, pr, m) for l in res["cells"] for B in BUD for pr in ("one", "three", "adaptive") for m in res["cells"][l]["truth"]["one"]]
covs = [res["cells"][l][str(B)][pr][m]["cov_expected"] for l, B, pr, m in ent]
covs2 = [res["cells"][l][str(B)][pr][m]["cov_expected_rv"] for l, B, pr, m in ent]
covr = [res["cells"][l][str(B)][pr][m]["cov_realized"] for l, B, pr, m in ent]
res["B88f_prime"] = dict(n=len(covs2), mean_rv=float(np.mean(covs2)), min_rv=float(np.min(covs2)), n_below_090_rv=int(np.sum(np.array(covs2) < 0.90)),
                         mean_realized=float(np.mean(covr)), min_realized=float(np.min(covr)))
res["H88f"] = dict(n=len(covs), mean=float(np.mean(covs)), min=float(np.min(covs)), pass_=bool(np.mean(covs) >= 0.94 and np.min(covs) >= 0.90))
print("B88f'", res["B88f_prime"]); print("H88f", res["H88f"])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
