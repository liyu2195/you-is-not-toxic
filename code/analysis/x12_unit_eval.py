"""Round U1 evaluation (Addendum 67 rule): at B >= 500, ts_rank's mean coverage >= 0.935, no cell below 0.90, and median
width <= 0.97x ts_own. usage: python3 x12_unit_eval.py OUT_JSON DIR [--cand ts_rank] [--ref ts_own] [--bmin 500]"""
import sys, json, glob, os
import numpy as np
argv = sys.argv[1:]
def opt(n, d):
    global argv
    if n in argv:
        i = argv.index(n); v = argv[i + 1]; argv = argv[:i] + argv[i + 2:]; return v
    return d
CAND, REF, BMIN = opt("--cand", "ts_rank"), opt("--ref", "ts_own"), int(opt("--bmin", "500"))
OUT, D = argv[0], argv[1]
rows = []
for f in sorted(glob.glob(f"{D}/*__*.json")):
    if os.path.basename(f).startswith("eval"):
        continue
    R = json.load(open(f))
    for B, dd in R["unit"].items():
        if int(B) >= BMIN and CAND in dd and REF in dd:
            rows.append(dict(cell=os.path.basename(f)[:-5], B=int(B), cov=dd[CAND]["coverage"], ref_cov=dd[REF]["coverage"],
                             ratio=dd[CAND]["width"] / dd[REF]["width"], rmse_ratio=dd[CAND]["rmse"] / dd[REF]["rmse"]))
cov = np.array([r["cov"] for r in rows]); rat = np.array([r["ratio"] for r in rows]); lo = min(rows, key=lambda r: r["cov"])
s = dict(cand=CAND, ref=REF, n_cells=len({r["cell"] for r in rows}), mean_cov=float(cov.mean()), min_cov=float(cov.min()), min_at=f"{lo['cell']} B={lo['B']}",
         ref_mean_cov=float(np.mean([r["ref_cov"] for r in rows])), ref_min_cov=float(min(r["ref_cov"] for r in rows)),
         median_ratio=float(np.median(rat)), range_ratio=[float(rat.min()), float(rat.max())],
         by_cell={c: float(np.median([r["ratio"] for r in rows if r["cell"] == c])) for c in sorted({r["cell"] for r in rows})})
s["adopt"] = bool(s["mean_cov"] >= 0.935 and s["min_cov"] >= 0.90 and s["median_ratio"] <= 0.97)
json.dump(dict(summary=s, rows=rows), open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in s.items() if k != "by_cell"}, indent=0)); print("by cell:", {c: round(v, 2) for c, v in s["by_cell"].items()})
