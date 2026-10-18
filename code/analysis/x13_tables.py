"""Final confirmation (Addendum 71): markdown tables for the v3 results doc.
Validity and efficiency of the frozen designs against upgrade_lw (v2), PPI++ and uniform sampling, and n80, the smallest
budget at which at least 80% of audits detect the labeler's distortion of a law (the interval excludes the labeler's value).
Addendum 72: --exclude-pop POP drops POP from the edge regime; later DIRs override earlier ones (pass the item rerun last).
usage: python3 x13_tables.py OUT_MD --final-item NAME --final-edge NAME [--exclude-pop cga] DIR [DIR ...]"""
import sys, json, glob, os
import numpy as np
argv = sys.argv[1:]
def opt(n):
    global argv
    i = argv.index(n); v = argv[i + 1]; argv = argv[:i] + argv[i + 2:]; return v
FI, FE = opt("--final-item"), opt("--final-edge")
EXCL = opt("--exclude-pop") if "--exclude-pop" in argv else None
OUT, dirs = argv[0], argv[1:]
FRESH = ("sn_rob", "mh_dist", "fb_hate", "cn_off")
C = {}
for d in dirs:
    for f in sorted(glob.glob(f"{d}/*__*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        R = json.load(open(f))
        for r in R["results"]:
            te = r.get("truth_edge") or {}; ti = r.get("truth_item") or {}; li = r.get("labeler_item") or {}; le = r.get("labeler_edge") or {}
            for reg in ("item", "edge"):
                if reg == "edge" and R["pop"] == EXCL:
                    continue
                for B, dd in (r.get(reg) or {}).items():
                    for des, mm in dd.items():
                        for m, c in mm.items():
                            if m == "_labels":
                                continue
                            C[(R["pop"], r["labeler"], reg, m, int(B), des)] = dict(c, truth=(ti.get(m) if reg == "item" else te.get("rr" if m == "log_rr" else "placebo_rr")),
                                                                                  lab_value=(li.get(m) if reg == "item" else (le.get("rr") if m == "log_rr" else None)))
LAWS = {"depth_law", "size_law", "article_size_law", "log_rr"}
MN = {"depth_law": "Depth law", "size_law": "Thread size law", "article_size_law": "Article size law", "placebo_depth": "Placebo depth slope",
      "placebo_size": "Placebo size slope", "log_rr": "Reply risk ratio", "log_placebo_rr": "Placebo risk ratio"}
NAME = {FI: f"v3 items (`{FI}`)", FE: f"v3 edges (`{FE}`)", "upgrade_lw": "v2 (`upgrade_lw`)", "ppi": "PPI++", "srs": "Uniform"}
L = []; w = L.append
for reg, fin, bmin in (("item", FI, 1000), ("edge", FE, 2000)):
    w(f"### {'Items' if reg == 'item' else 'Edges'}: validity at B ≥ {bmin:,} (coverage of the full-gold value)\n")
    w("| Design | Laws: mean | Laws: min | Placebo: mean | Placebo: min | Share of entries ≥ 0.90 | Entries |")
    w("|---|---|---|---|---|---|---|")
    for des in (fin, "upgrade_lw", "ppi", "srs"):
        sel = [c for k, c in C.items() if k[2] == reg and k[5] == des and k[4] >= bmin]
        if not sel:
            continue
        la = [c["coverage"] for k, c in C.items() if k[2] == reg and k[5] == des and k[4] >= bmin and k[3] in LAWS]
        pl = [c["coverage"] for k, c in C.items() if k[2] == reg and k[5] == des and k[4] >= bmin and k[3] not in LAWS]
        allc = np.array(la + pl)
        w(f"| {NAME[des]} | {np.mean(la):.3f} | {np.min(la):.3f} | {np.mean(pl):.3f} | {np.min(pl):.3f} | {np.mean(allc >= 0.90):.3f} | {len(allc)} |")
    w(f"\n### {'Items' if reg == 'item' else 'Edges'}: median width ratio of v3 at B ≥ {bmin:,}\n")
    w("| Metric | vs v2 | vs PPI++ | vs uniform | Entries |")
    w("|---|---|---|---|---|")
    for m in [x for x in MN if (x in ("log_rr", "log_placebo_rr")) == (reg == "edge")]:
        row = []
        for ref in ("upgrade_lw", "ppi", "srs"):
            rs = [C[k]["width"] / C[k[:5] + (ref,)]["width"] for k in C if k[2] == reg and k[3] == m and k[5] == fin and k[4] >= bmin
                  and k[:5] + (ref,) in C and C[k].get("width") and C[k[:5] + (ref,)].get("width")]
            row.append((np.median(rs) if rs else np.nan, len(rs)))
        if row[0][1]:
            w(f"| {MN[m]} | {row[0][0]:.2f} | {row[1][0]:.2f} | {row[2][0]:.2f} | {row[0][1]} |")
    w("")
# n80
w("### The price of the diagnostic: n80, the smallest budget at which ≥ 80% of audits detect the labeler's distortion\n")
w("Rows where some design reaches 80% within the budget grid; '>' means not reached by the largest budget.\n")
w("| Population | Labeler | Law | Labeler's value | Gold value | v3 | v2 | PPI++ | Uniform |")
w("|---|---|---|---|---|---|---|---|---|")
keys = sorted({(k[0], k[1], k[2], k[3]) for k in C if k[3] in LAWS})
for pop, lab, reg, m in keys:
    fin = FI if reg == "item" else FE
    Bs = sorted({k[4] for k in C if k[:4] == (pop, lab, reg, m)})
    def n80(des):
        for B in Bs:
            c = C.get((pop, lab, reg, m, B, des))
            if c and c.get("detect") is not None and c["detect"] >= 0.8:
                return B
        return None
    vals = {des: n80(des) for des in (fin, "upgrade_lw", "ppi", "srs")}
    if all(v is None for v in vals.values()):
        continue
    c0 = C.get((pop, lab, reg, m, Bs[0], fin)) or next(C[k] for k in C if k[:4] == (pop, lab, reg, m))
    fmt = lambda v: f"{v:,}" if v else f">{Bs[-1]:,}"
    lv, gv = c0.get("lab_value"), c0.get("truth")
    lvs = "" if lv is None else (f"{lv:+.4f}" if reg == "item" else f"{lv:.2f}"); gvs = "" if gv is None else (f"{gv:+.4f}" if reg == "item" else f"{gv:.2f}")
    w(f"| {pop} | {lab} | {MN[m]} | {lvs} | {gvs} | {fmt(vals[fin])} | {fmt(vals['upgrade_lw'])} | {fmt(vals['ppi'])} | {fmt(vals['srs'])} |")
open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT, len(L), "lines")
