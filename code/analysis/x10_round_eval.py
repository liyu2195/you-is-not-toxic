"""Development-round evaluation with the registered adoption rule (Addenda 61-64).

For each candidate design, over (cell, budget >= BMIN, metric) entries:
  coverage: mean and minimum (the rule: mean >= 0.935, no entry below 0.90);
  width: ratio to the reference design in the same cell, budget and metric; medians over the law entries and over the
  placebo entries (the rule: <= 0.97 and <= 1.00); the weighted median over all entries (laws x1, placebo x0.25) ranks
  the designs that qualify.
usage: python3 x10_round_eval.py OUT_JSON SIMDIR [SIMDIR ...] [--ref upgrade_lw] [--designs a,b] [--regime item] [--bmin 1000]
       [--exclude-pop cga]   (Addendum 72: drop a population's cells)
       later SIMDIRs override earlier ones for the same cell."""
import sys, json, glob, os, collections
import numpy as np
argv = sys.argv[1:]
def opt(name, default):
    global argv
    if name in argv:
        i = argv.index(name); v = argv[i + 1]; argv = argv[:i] + argv[i + 2:]; return v
    return default
REF = opt("--ref", "upgrade_lw"); DES = opt("--designs", ""); REG = opt("--regime", "item"); BMIN = int(opt("--bmin", "1000"))
EXCL = opt("--exclude-pop", "")
OUT, dirs = argv[0], argv[1:]
LAWS = {"depth_law", "size_law", "article_size_law", "log_rr"}; PLAC = {"placebo_depth", "placebo_size", "log_placebo_rr"}
import re
cells = {}
for d in dirs:
    for f in sorted(glob.glob(f"{d}/*__*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        R = json.load(open(f))["results"][0]
        if R.get(REG):
            tag = re.sub(r"__g\d+$", "", os.path.basename(f)[:-5])          # a cell split over processes: merge its designs
            if EXCL and tag.split("__")[0] == EXCL:
                continue
            C = cells.setdefault(tag, {})
            for B, dd in R[REG].items():
                C.setdefault(B, {}).update(dd)
designs = sorted({des for C in cells.values() for dd in C.values() for des in dd})
cand = [d for d in (DES.split(",") if DES else designs) if d in designs]
rows = []
for tag, C in sorted(cells.items()):
    for B, dd in C.items():
        if int(B) < BMIN or REF not in dd:
            continue
        for des in cand:
            for m, c in dd.get(des, {}).items():
                if m == "_labels":
                    continue
                r = dd[REF].get(m, {})
                ratio = c["width"] / r["width"] if c.get("width") and r.get("width") else None
                rows.append(dict(cell=tag, B=int(B), design=des, metric=m, fam="laws" if m in LAWS else "placebo",
                                 coverage=c["coverage"], ref_coverage=r.get("coverage"), ratio=ratio, rmse=c.get("rmse"), ref_rmse=r.get("rmse")))


def wmedian(x, w):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]; cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


summ = {}
for des in cand:
    R = [r for r in rows if r["design"] == des]
    if not R:
        continue
    cov = np.array([r["coverage"] for r in R]); lo = min(R, key=lambda r: r["coverage"])
    rl = [r["ratio"] for r in R if r["fam"] == "laws" and r["ratio"]]; rp = [r["ratio"] for r in R if r["fam"] == "placebo" and r["ratio"]]
    allr = [(r["ratio"], 1.0 if r["fam"] == "laws" else 0.25) for r in R if r["ratio"]]
    s = dict(n_cells=len({r["cell"] for r in R}), n_entries=len(R), mean_cov=float(cov.mean()), min_cov=float(cov.min()),
             min_at=f'{lo["cell"]} B={lo["B"]} {lo["metric"]}', med_laws=float(np.median(rl)) if rl else None,
             med_placebo=float(np.median(rp)) if rp else None,
             wmed=wmedian([a for a, _ in allr], [b for _, b in allr]) if allr else None,
             n_below_090=int((cov < 0.90).sum()))
    s["qualifies"] = bool(s["mean_cov"] >= 0.935 and s["min_cov"] >= 0.90 and (s["med_laws"] or 9) <= 0.97 and (s["med_placebo"] or 9) <= 1.00)
    # per budget and per cell medians (descriptive)
    s["by_B"] = {B: dict(med_laws=float(np.median([r["ratio"] for r in R if r["B"] == B and r["fam"] == "laws" and r["ratio"]] or [np.nan])),
                         med_placebo=float(np.median([r["ratio"] for r in R if r["B"] == B and r["fam"] == "placebo" and r["ratio"]] or [np.nan])),
                         mean_cov=float(np.mean([r["coverage"] for r in R if r["B"] == B])))
                 for B in sorted({r["B"] for r in R})}
    s["by_cell"] = {c: dict(med_laws=float(np.median([r["ratio"] for r in R if r["cell"] == c and r["fam"] == "laws" and r["ratio"]] or [np.nan])),
                            med_placebo=float(np.median([r["ratio"] for r in R if r["cell"] == c and r["fam"] == "placebo" and r["ratio"]] or [np.nan])),
                            min_cov=float(min(r["coverage"] for r in R if r["cell"] == c)))
                    for c in sorted({r["cell"] for r in R})}
    summ[des] = s
ok = [d for d in summ if d != REF and summ[d]["qualifies"]]
best = min(ok, key=lambda d: summ[d]["wmed"]) if ok else REF
json.dump(dict(ref=REF, regime=REG, bmin=BMIN, dirs=dirs, summary=summ, adopted=best, rows=rows), open(OUT, "w"), indent=1)
for des, s in summ.items():
    print(f"{des:14s} cells {s['n_cells']:2d}  cov mean {s['mean_cov']:.3f} min {s['min_cov']:.3f} ({s['min_at']}) n<0.90 {s['n_below_090']}  "
          f"width laws {s['med_laws']:.3f} placebo {s['med_placebo']:.3f} wmed {s['wmed']:.3f}  qualifies {s['qualifies']}")
    for B, b in s["by_B"].items():
        print(f"    B={B:5d} laws {b['med_laws']:.3f} placebo {b['med_placebo']:.3f} cov {b['mean_cov']:.3f}")
print("adopted:", best)
