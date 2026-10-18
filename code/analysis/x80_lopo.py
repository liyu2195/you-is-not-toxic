"""Addendum 80, B80f: leave-one-population-out re-adoption of the frozen designs, from the stored development rounds.
Items: every design run in rounds A1-A7 (results/x10_dev_rA*, plus round 0) on the 17 development cells, pooled, against the
reference upgrade_lw with the registered rule (Addenda 62-70): mean coverage >= 0.935, no entry < 0.90, median width ratio <= 0.97 on
laws and <= 1.00 on placebo statistics at B >= 1,000; among qualifiers the lowest weighted median (laws x1, placebo x0.25) wins.
Edges: rounds E1-E4 (results/x10_dev_rE*), the same rule at B >= 2,000. Units (U1): ts_rank is adopted over ts_own if, at B >= 500,
mean coverage >= 0.935, no cell < 0.90 and median width <= 0.97 x ts_own. Each rule is re-applied seven times, with one population's
development cells removed, and once with none (which must reproduce the frozen choice up to the Addendum 72 CGA-Wiki correction).
Correction registered in Addendum 80's results (09:16 MDT): with --edge-registered the edge rule pools E1-E4 and round E5
(results/x14_dev_rE5; later directories override earlier ones in the same cell), always without CGA-Wiki, as in the decision of
Addendum 73, and holds out each of the other six populations; the item and unit runs are unchanged.
usage: python3 x80_lopo.py OUT_JSON [--edge-registered]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, os, re, numpy as np
W = _ROOT; OUT = sys.argv[1]; EREG = "--edge-registered" in sys.argv
POPS = ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]
LAWS = {"depth_law", "size_law", "article_size_law", "log_rr"}
def load(dirs, reg):
    cells = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*__*.json")):
            if os.path.basename(f).startswith("eval"):
                continue
            try:
                R = json.load(open(f))["results"][0]
            except Exception:
                continue
            if R.get(reg):
                tag = re.sub(r"__g\d+$", "", os.path.basename(f)[:-5]); C = cells.setdefault(tag, {})
                for B, dd in R[reg].items():
                    C.setdefault(B, {}).update(dd)
    return cells
def wmedian(x, w):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]; cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])
def adopt(cells, ref, bmin, excl, always=()):
    rows = []
    for tag, C in cells.items():
        if (excl and tag.split("__")[0] == excl) or tag.split("__")[0] in always:
            continue
        for B, dd in C.items():
            if int(B) < bmin or ref not in dd:
                continue
            for des, mm in dd.items():
                for m, c in mm.items():
                    if m == "_labels":
                        continue
                    r = dd[ref].get(m, {})
                    rows.append((des, tag, m, c["coverage"], c["width"] / r["width"] if c.get("width") and r.get("width") else None))
    summ = {}
    for des in sorted({r[0] for r in rows}):
        R = [r for r in rows if r[0] == des]; cov = np.array([r[3] for r in R])
        rl = [r[4] for r in R if r[2] in LAWS and r[4]]; rp = [r[4] for r in R if r[2] not in LAWS and r[4]]
        allr = [(r[4], 1.0 if r[2] in LAWS else 0.25) for r in R if r[4]]
        s = dict(n_cells=len({r[1] for r in R}), mean_cov=float(cov.mean()), min_cov=float(cov.min()), med_laws=float(np.median(rl)) if rl else None,
                 med_placebo=float(np.median(rp)) if rp else None, wmed=wmedian([a for a, _ in allr], [b for _, b in allr]) if allr else None)
        s["qualifies"] = bool(s["mean_cov"] >= 0.935 and s["min_cov"] >= 0.90 and (s["med_laws"] or 9) <= 0.97 and (s["med_placebo"] or 9) <= 1.00)
        summ[des] = s
    full_cells = max(s["n_cells"] for s in summ.values())
    ok = [d for d, s in summ.items() if d != ref and not d.endswith("_or") and s["qualifies"] and s["n_cells"] == full_cells]   # adoptable designs run on every remaining cell (oracle twins are diagnostics)
    best = min(ok, key=lambda d: summ[d]["wmed"]) if ok else ref
    return best, summ
res = {}
EDIRS = sorted(glob.glob(f"{W}/results/x10_dev_rE*")) + ([f"{W}/results/x14_dev_rE5"] if EREG else [])
for reg, dirs, bmin, frozen in (("item", sorted(glob.glob(f"{W}/results/x10_dev_r0")) + sorted(glob.glob(f"{W}/results/x10_dev_rA*")), 1000, "pps_seq4_ba"),
                                ("edge", EDIRS, 2000, "edge_m3_c15")):
    cells = load(dirs, reg); always = ("cga",) if (EREG and reg == "edge") else ()
    out = {"frozen": frozen, "n_cells": len([t for t in cells if t.split("__")[0] not in always]), "runs": {}, "excluded_always": list(always)}
    for excl in [""] + [p_ for p_ in POPS if p_ not in always]:
        best, summ = adopt(cells, "upgrade_lw", bmin, excl, always)
        top = sorted([(d, s["wmed"]) for d, s in summ.items() if s["qualifies"] and d != "upgrade_lw" and not d.endswith("_or")], key=lambda t: t[1])[:3]
        out["runs"][excl or "none"] = dict(adopted=best, frozen_wmed=summ.get(frozen, {}).get("wmed"), frozen_qualifies=summ.get(frozen, {}).get("qualifies"),
                                           adopted_wmed=summ.get(best, {}).get("wmed"), top3=top,
                                           frozen_over_adopted=(summ[frozen]["wmed"] / summ[best]["wmed"]) if frozen in summ and best in summ else None)
        print(reg, f"exclude {excl or 'none':12s} adopted {best:22s} wmed {summ.get(best, {}).get('wmed')}  frozen {frozen} wmed {summ.get(frozen, {}).get('wmed')} qual {summ.get(frozen, {}).get('qualifies')}", flush=True)
    out["n_frozen_adopted"] = int(sum(v["adopted"] == frozen for k, v in out["runs"].items() if k != "none")); out["n_runs"] = len(out["runs"]) - 1
    out["n_frozen_qualifies"] = int(sum(bool(v["frozen_qualifies"]) for k, v in out["runs"].items() if k != "none"))
    out["max_frozen_over_adopted"] = float(max(v["frozen_over_adopted"] for k, v in out["runs"].items() if k != "none"))
    res[reg] = out
# units: round U1
U = json.load(open(f"{W}/results/x12_unit_U1/eval_U1.json")); rows = U["rows"]
uo = {"runs": {}}
for excl in [""] + POPS:
    R = [r for r in rows if not (excl and r["cell"].split("__")[0] == excl) and int(r["B"]) >= 500]
    cand = R; cov = np.array([r["cov"] for r in cand])
    ratio = [r["ratio"] for r in cand if r.get("ratio")]
    bycell = {}
    for r in cand:
        bycell.setdefault(r["cell"], []).append(r["cov"])
    ok = bool(cov.mean() >= 0.935 and min(min(v) for v in bycell.values()) >= 0.90 and np.median(ratio) <= 0.97)
    uo["runs"][excl or "none"] = dict(adopted="ts_rank" if ok else "ts_own", median_ratio=float(np.median(ratio)), mean_cov=float(cov.mean()))
    print("unit", f"exclude {excl or 'none':12s} ts_rank median ratio {np.median(ratio):.3f} mean cov {cov.mean():.3f} -> {'ts_rank' if ok else 'ts_own'}", flush=True)
uo["n_frozen_adopted"] = int(sum(v["adopted"] == "ts_rank" for k, v in uo["runs"].items() if k != "none")); res["unit"] = uo
res["H80g"] = dict(item=res["item"]["n_frozen_adopted"], edge=res["edge"]["n_frozen_adopted"], edge_runs=res["edge"]["n_runs"], edge_registered=EREG,
                  pass_=bool(res["item"]["n_frozen_adopted"] >= 6 and res["edge"]["n_frozen_adopted"] >= (6 if res["edge"]["n_runs"] == 7 else res["edge"]["n_runs"] - 1)))
print("H80g", res["H80g"]); json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
