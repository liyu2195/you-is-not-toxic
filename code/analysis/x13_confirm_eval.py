"""Final confirmation (Addendum 71): hypotheses H71a-H71d for the item and edge regimes, plus descriptive comparisons.
  H71a/c validity of the final design: mean coverage >= 0.935, at least 97% of entries >= 0.90, no entry below 0.85
          (items at B >= 1,000; edges at B >= 2,000);
  H71b    item efficiency: median width ratio to upgrade_lw <= 0.85 on the laws and <= 0.90 on the placebo slopes;
  H71d    edge efficiency: median log-width ratio to upgrade_lw <= 0.97 (gold risk ratio) and <= 0.95 (placebo risk ratio).
Descriptive: the same ratios against PPI++ and uniform sampling, by labeler group (fresh / Addendum 58) and by budget.
Addendum 72 options: --exclude-pop POP drops POP's cells from the edge regime (its gold risk ratio is undefined);
--item-override DIR replaces the final item design's entries with those in DIR's result files (any number of labelers per file).
usage: python3 x13_confirm_eval.py OUT_JSON ITEM_DIR EDGE_DIR --final-item NAME --final-edge NAME [--exclude-pop cga] [--item-override DIR]"""
import sys, json, glob, os, re
import numpy as np
argv = sys.argv[1:]
def opt(n, d=None):
    global argv
    if n in argv:
        i = argv.index(n); v = argv[i + 1]; argv = argv[:i] + argv[i + 2:]; return v
    return d
FI, FE = opt("--final-item"), opt("--final-edge")
EXCL = opt("--exclude-pop"); IOVR = opt("--item-override")
OUT, IDIR, EDIR = argv[0], argv[1], argv[2]
LAWS = {"depth_law", "size_law", "article_size_law", "log_rr"}
FRESH = ("sn_rob", "mh_dist", "fb_hate", "cn_off")


def load(d, reg):
    cells = {}
    for f in sorted(glob.glob(f"{d}/*__*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        R = json.load(open(f))["results"][0]
        tag = re.sub(r"__g\d+$", "", os.path.basename(f)[:-5]); C = cells.setdefault(tag, {})
        for B, dd in (R.get(reg) or {}).items():
            for des, mm in dd.items():
                C.setdefault(int(B), {}).setdefault(des, {}).update(mm)
    if reg == "edge" and EXCL:
        cells = {t: c for t, c in cells.items() if t.split("__")[0] != EXCL}
    if reg == "item" and IOVR:
        n = 0
        for f in sorted(glob.glob(f"{IOVR}/*.json")):
            J = json.load(open(f))
            for R in J["results"]:
                tag = f"{J['pop']}__{R['labeler'].replace('@', '_at_')}"
                for B, dd in (R.get("item") or {}).items():
                    for des, mm in dd.items():
                        if tag in cells and int(B) in cells[tag] and des in cells[tag][int(B)]:
                            cells[tag][int(B)][des] = dict(mm); n += 1
        print("item override:", n, "cell-budget-design entries replaced")
    return cells


def entries(cells, final, ref, bmin):
    rows = []
    for tag, C in cells.items():
        group = "fresh" if tag.split("__")[1].split("_at_")[0] in FRESH else "a58"
        for B, dd in C.items():
            if B < bmin or final not in dd:
                continue
            for m, c in dd[final].items():
                if m == "_labels":
                    continue
                r = dd.get(ref, {}).get(m, {})
                ratio = c["width"] / r["width"] if c.get("width") and r.get("width") else None
                rows.append(dict(cell=tag, group=group, B=B, metric=m, fam="laws" if m in LAWS else "placebo", coverage=c["coverage"], ratio=ratio))
    return rows


def validity(rows):
    if not rows:
        return None
    cov = np.array([r["coverage"] for r in rows]); lo = min(rows, key=lambda r: r["coverage"])
    return dict(n=len(rows), mean=float(cov.mean()), share_ge_090=float(np.mean(cov >= 0.90)), min=float(cov.min()), min_at=f"{lo['cell']} B={lo['B']} {lo['metric']}",
                passes=bool(cov.mean() >= 0.935 and np.mean(cov >= 0.90) >= 0.97 and cov.min() >= 0.85))


def med(rows, fam=None, group=None, B=None):
    v = [r["ratio"] for r in rows if r["ratio"] and (fam is None or r["fam"] == fam) and (group is None or r["group"] == group) and (B is None or r["B"] == B)]
    return float(np.median(v)) if v else None


out = {}
for reg, d, final, bmin, bars in (("item", IDIR, FI, 1000, (0.85, 0.90)), ("edge", EDIR, FE, 2000, (0.97, 0.95))):
    cells = load(d, reg); res = {"final": final, "n_cells": len(cells)}
    rows_u = entries(cells, final, "upgrade_lw", bmin)
    res["validity"] = validity(rows_u)
    res["validity_by_group"] = {g: validity([r for r in rows_u if r["group"] == g]) for g in ("fresh", "a58")}
    res["vs_upgrade_lw"] = dict(laws=med(rows_u, "laws"), placebo=med(rows_u, "placebo"))
    res["efficiency_passes"] = bool(res["vs_upgrade_lw"]["laws"] <= bars[0] and res["vs_upgrade_lw"]["placebo"] <= bars[1])
    for ref in ("upgrade_lw", "ppi", "srs"):
        rr = entries(cells, final, ref, bmin)
        res[f"desc_vs_{ref}"] = {"all": dict(laws=med(rr, "laws"), placebo=med(rr, "placebo")),
                                 **{g: dict(laws=med(rr, "laws", g), placebo=med(rr, "placebo", g)) for g in ("fresh", "a58")},
                                 "by_B": {B: dict(laws=med(rr, "laws", B=B), placebo=med(rr, "placebo", B=B)) for B in sorted({r["B"] for r in rr})},
                                 "by_metric": {m: float(np.median([r["ratio"] for r in rr if r["metric"] == m and r["ratio"]])) for m in sorted({r["metric"] for r in rr})}}
    # the references' own validity, for the record
    for ref in ("upgrade_lw", "ppi", "srs"):
        rr = entries(cells, ref, final, bmin)
        if rr:
            res[f"validity_{ref}"] = validity(rr)
    out[reg] = res
    print(reg, final, json.dumps({k: v for k, v in res.items() if k in ("n_cells", "validity", "vs_upgrade_lw", "efficiency_passes")}, indent=0))
json.dump(out, open(OUT, "w"), indent=1)
