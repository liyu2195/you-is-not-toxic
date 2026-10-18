"""Addendum 74, Part B evaluation: new comparators (B1), adversarial labelers (B2) and ablations (B3); hypotheses H74h-H74j.
Reference runs (Addendum 71, corrected by Addendum 72): items results/x11_conf_item_all with CGA-Wiki's entries from
results/x72_item_cga; edges results/x11_conf_edge_all without CGA-Wiki. New runs: results/x74_b1b3 (seed 7400), results/x74_b2 (7450).
Width ratios are v3's mean width over the comparator's in the same cell, budget and metric (log scale for risk ratios); medians
over entries at B >= 1,000 (items) and B >= 2,000 (edges). Writes results/x74_eval/eval_B.json, numbers_B.json and the paper's
tables (paper_v4/tables/confirm_v4.tex, ablation.tex).
usage: python3 x74_eval_B.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, glob, os, re
import numpy as np
W = _ROOT; OUT = f"{W}/results/x74_eval"; os.makedirs(OUT, exist_ok=True)
LAWS = {"depth_law", "size_law", "article_size_law"}; PLAC = {"placebo_depth", "placebo_size"}
ITEM_M = ["depth_law", "size_law", "article_size_law", "placebo_depth", "placebo_size"]; EDGE_M = ["log_rr", "log_placebo_rr"]
V3I, V3E = "pps_seq4_ba", "edge_m3_c15"


def load_dir(d, reg, cells=None, exclude=None):
    cells = {} if cells is None else cells
    for f in sorted(glob.glob(f"{d}/*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        J = json.load(open(f))
        for R in J["results"]:
            tag = f"{J['pop']}__{R['labeler'].replace('@', '_at_')}"
            if exclude and J["pop"] == exclude:
                continue
            for B, dd in (R.get(reg) or {}).items():
                for des, mm in dd.items():
                    cells.setdefault(tag, {}).setdefault(int(B), {})[des] = mm
            if reg == "item" and R.get("truth_item"):
                cells.setdefault(tag, {}).setdefault("_truth", R["truth_item"])
            if reg == "edge" and R.get("truth_edge"):                     # revision: keep edge truths, to skip undefined estimands
                cells.setdefault(tag, {}).setdefault("_truth", {k: R["truth_edge"][k] for k in ("log_rr", "log_placebo_rr")})
    return cells


def entries(cells, a, b, metrics, bmin):
    """(tag, B, metric, width ratio a/b, coverage a, coverage b)"""
    out = []
    for tag, C in cells.items():
        for B, dd in C.items():
            if B == "_truth" or B < bmin or a not in dd or b not in dd:
                continue
            for m in metrics:
                if not defined(C, m):
                    continue
                x, y = dd[a].get(m), dd[b].get(m)
                if not x or not y or x.get("width") is None or y.get("width") is None or y["width"] <= 0:
                    continue
                out.append((tag, B, m, x["width"] / y["width"], x["coverage"], y["coverage"]))
    return out


def med(rows, group):
    v = [r[3] for r in rows if r[2] in group]
    return float(np.median(v)) if v else None


def defined(C, m):
    """Revision (Addendum 78 run log): an entry whose estimand is undefined (e.g. the placebo RR of a labeler that flags no comment,
    stored truth NaN) is not a coverage entry."""
    t = C.get("_truth", {}).get(m)
    return t is None or bool(np.isfinite(t))


def coverage(cells, d, metrics, bmin, bmax=10 ** 9):
    v = [dd[d][m]["coverage"] for C in cells.values() for B, dd in C.items() if B != "_truth" and bmin <= B <= bmax and d in dd for m in metrics if m in dd[d] and defined(C, m)]
    return (float(np.mean(v)), float(np.min(v)), len(v), float(np.mean(np.array(v) < 0.90))) if v else (None, None, 0, None)


# ---------------------------------------------------------------- items
I = load_dir(f"{W}/results/x11_conf_item_all", "item")
for f in sorted(glob.glob(f"{W}/results/x72_item_cga/*.json")):                 # Addendum 72: CGA-Wiki rerun (same seeds)
    J = json.load(open(f))
    for R in J["results"]:
        tag = f"{J['pop']}__{R['labeler'].replace('@', '_at_')}"
        for B, dd in R["item"].items():
            for des, mm in dd.items():
                if tag in I and int(B) in I[tag]:
                    I[tag][int(B)][des] = mm
I = load_dir(f"{W}/results/x74_b1b3", "item", I)
E = load_dir(f"{W}/results/x11_conf_edge_all", "edge", exclude="cga")
E = load_dir(f"{W}/results/x74_b1b3", "edge", E)
res = {"items": {}, "edges": {}, "n_item_cells": len(I), "n_edge_cells": len(E)}
NB = {}
COMP_I = ["ppi", "dsl", "ai", "upgrade_lw", "srs", "calag", "calag_s"]; COMP_E = ["ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs", "edge_calag"]
for c in COMP_I:
    rows = entries(I, V3I, c, ITEM_M, 1000)
    res["items"][c] = dict(n=len(rows), laws=med(rows, LAWS), placebo=med(rows, PLAC), by_metric={m: med(rows, {m}) for m in ITEM_M},
                           cov=coverage(I, c, ITEM_M, 1000), cov_laws_2000=coverage(I, c, ["depth_law", "size_law"], 2000))
for c in COMP_E:
    rows = entries(E, V3E, c, EDGE_M, 2000)
    res["edges"][c] = dict(n=len(rows), gold=med(rows, {"log_rr"}), placebo=med(rows, {"log_placebo_rr"}), cov=coverage(E, c, EDGE_M, 2000))
res["items"]["v3_cov"] = coverage(I, V3I, ITEM_M, 1000); res["edges"]["v3_cov"] = coverage(E, V3E, EDGE_M, 2000)
# bias of v3 in SD units (sd^2 = rmse^2 - bias^2), every entry at every budget
res["v3_bias_sd"] = {}
for nm, C, d, ms, bmin in (("items", I, V3I, ITEM_M, 0), ("edges", E, V3E, EDGE_M, 2000)):
    bs = []
    for tag, CC in C.items():
        for B, dd in CC.items():
            if B == "_truth" or d not in dd or B < bmin:
                continue
            for m in ms:
                x = dd[d].get(m)
                if x and x.get("bias") is not None and x.get("rmse"):
                    s2 = x["rmse"] ** 2 - x["bias"] ** 2
                    if s2 > 0:
                        bs.append(abs(x["bias"]) / np.sqrt(s2))
    mc = float(np.median(np.abs(np.random.default_rng(7475).normal(0, 1 / np.sqrt(500), (2000, len(bs)))).max(1)))   # pure Monte Carlo noise
    res["v3_bias_sd"][nm] = dict(n=len(bs), median=float(np.median(bs)), p99=float(np.percentile(bs, 99)), max=float(np.max(bs)), mc_expected_max=mc)
# ---------------------------------------------------------------- ablations
ABL_I = [("pps_seq4_ba_unif", "Sampling by influence (uniform)"), ("pps_seq4_ba_noaux", "Auxiliary instruments (labeler only)"),
         ("pps_seq1_ba", "Sequential learning (one fold)"), ("pps_seq4_a", "Tempered predictions (plain)")]
ABL_E = {"pps_seq4_ba_unif": "edge_m3_c15_unif", "pps_seq4_ba_noaux": "edge_m3_c15_noaux", "pps_seq1_ba": "edge_m1_c15", "pps_seq4_a": None}
res["ablations"] = {}
for a, lab in ABL_I:
    ri = entries(I, a, V3I, ITEM_M, 1000); ea = ABL_E[a]; re_ = entries(E, ea, V3E, EDGE_M, 2000) if ea else []
    res["ablations"][a] = dict(label=lab, laws=med(ri, LAWS), placebo=med(ri, PLAC), gold=med(re_, {"log_rr"}), plac_rr=med(re_, {"log_placebo_rr"}),
                               cov_items=coverage(I, a, ITEM_M, 1000), cov_edges=coverage(E, ea, EDGE_M, 2000) if ea else None, n_items=len(ri), n_edges=len(re_))
# ---------------------------------------------------------------- B2: adversarial labelers
A2i = load_dir(f"{W}/results/x74_b2", "item"); A2e = load_dir(f"{W}/results/x74_b2", "edge")
res["adversarial"] = {"items": {}, "edges": {}}
for d in [V3I, "upgrade_lw", "ppi", "srs"]:
    res["adversarial"]["items"][d] = coverage(A2i, d, ITEM_M, 1000)
for d in [V3E, "upgrade_lw", "ppi", "srs"]:
    res["adversarial"]["edges"][d] = coverage(A2e, d, EDGE_M, 2000)
adv_unif = {}
for lab in ["adv_rev", "adv_rand", "adv_const", "adv_depth", "adv_trap"]:
    sub = {t: c for t, c in A2i.items() if t.split("__")[1] == lab}; rows = entries(sub, V3I, "srs", ITEM_M, 1000)
    sube = {t: c for t, c in A2e.items() if t.split("__")[1] == lab}; rowse = entries(sube, V3E, "srs", EDGE_M, 2000)
    adv_unif[lab] = dict(laws=med(rows, LAWS), placebo=med(rows, PLAC), gold=med(rowse, {"log_rr"}), plac_rr=med(rowse, {"log_placebo_rr"}),
                         cov_v3=coverage(sub, V3I, ITEM_M, 1000), cov_v2=coverage(sub, "upgrade_lw", ITEM_M, 1000), cov_v3_edge=coverage(sube, V3E, EDGE_M, 2000))
res["adversarial"]["by_labeler"] = adv_unif
# ---------------------------------------------------------------- hypotheses
it, ed = res["items"], res["edges"]
def le(x, b):
    return x is not None and x <= b
res["H74h"] = dict(
    dsl_items=bool(le(it["dsl"]["laws"], 0.95) and le(it["dsl"]["placebo"], 0.95)), dsl_edges=bool(le(ed["edge_dsl"]["gold"], 0.97) and le(ed["edge_dsl"]["placebo"], 0.95)),
    ai_items=bool(le(it["ai"]["laws"], 0.90) and le(it["ai"]["placebo"], 0.90)), ai_edges=bool(le(ed["edge_ai"]["gold"], 0.95) and le(ed["edge_ai"]["placebo"], 0.95)))
res["H74h"]["pass"] = all(res["H74h"].values())
share = lambda d: coverage(I, d, ["depth_law", "size_law"], 2000)[3]
res["H74i"] = dict(calag=share("calag"), calag_s=share("calag_s"), pass_=bool(max(share("calag") or 0, share("calag_s") or 0) >= 0.25))
ai_, ae_ = res["adversarial"]["items"][V3I], res["adversarial"]["edges"][V3E]
res["H74j"] = dict(items=ai_, edges=ae_, pass_=bool(ai_[0] is not None and ai_[0] >= 0.935 and ai_[1] >= 0.90 and ae_[0] is not None and ae_[0] >= 0.935 and ae_[1] >= 0.90))
json.dump(res, open(f"{OUT}/eval_B.json", "w"), indent=1)
# ---------------------------------------------------------------- numbers and tables for the paper
f2 = lambda x: "??" if x is None else f"{x:.2f}"; f3 = lambda x: "??" if x is None else f"{x:.3f}"
NB.update(wPpiCovMean=f3(it["ppi"]["cov"][0]), wPpiCovMin=f2(it["ppi"]["cov"][1]), wCompCovMinAll=f2(min(it[c]["cov"][1] for c in ("ppi", "dsl", "ai", "srs") if it[c]["cov"][1] is not None)),
          wCalagSCovLaws=f2(it["calag_s"]["cov_laws_2000"][0]), wCalagSShareBad=("??" if share("calag_s") is None else f"{100 * share('calag_s'):.0f}\\%"),
          wAblUnifLaws=f2(res["ablations"]["pps_seq4_ba_unif"]["laws"]), wAblUnifPlac=f2(res["ablations"]["pps_seq4_ba_unif"]["placebo"]),
          wAblOneLaws=f2(res["ablations"]["pps_seq1_ba"]["laws"]), wAblPlainMin=f2(res["ablations"]["pps_seq4_a"]["cov_items"][1]))
NB.update(wItemLawsDSL=f2(it["dsl"]["laws"]), wItemPlacDSL=f2(it["dsl"]["placebo"]), wItemLawsAI=f2(it["ai"]["laws"]), wItemPlacAI=f2(it["ai"]["placebo"]),
          wEdgeLawsDSL=f2(ed["edge_dsl"]["gold"]), wEdgePlacDSL=f2(ed["edge_dsl"]["placebo"]), wBiasItemsMax=f2(res["v3_bias_sd"]["items"]["max"]),
          wBiasItemsN=f"{res['v3_bias_sd']['items']['n']:,}".replace(",", "{,}"), wBiasMCMax=f2(res["v3_bias_sd"]["items"]["mc_expected_max"]), wBiasEdgeMax=f2(res["v3_bias_sd"]["edges"]["max"]),
          wCalagCovLaws=f2(it["calag"]["cov_laws_2000"][0]), wCalagShareBad=("??" if share("calag") is None else f"{100 * share('calag'):.0f}\\%"))
if ai_[0] is not None:
    NB.update(wAdvItemCov=f3(ai_[0]), wAdvItemMin=f3(ai_[1]))
if ae_[0] is not None:
    NB.update(wAdvEdgeCov=f3(ae_[0]), wAdvEdgeMin=f3(ae_[1]))
v2adv = res["adversarial"]["items"]["upgrade_lw"]
if v2adv[1] is not None:
    NB["wAdvStratMin"] = f2(v2adv[1])
wu = [x for v in adv_unif.values() for x in (v["laws"], v["placebo"]) if x is not None]
if wu:
    NB["wAdvWorstUnif"] = f2(max(wu))
covs = [v["cov_items"][0] for v in res["ablations"].values() if v["cov_items"][0] is not None] + [v["cov_edges"][0] for v in res["ablations"].values() if v["cov_edges"] and v["cov_edges"][0] is not None]
if covs:
    NB["wAblMinCov"] = f3(min(covs))
json.dump(NB, open(f"{OUT}/numbers_B.json", "w"), indent=1)
# Table 2: rows by metric
NAMES = {"depth_law": "Escalation", "size_law": "Thread size law", "article_size_law": "Article size law", "placebo_depth": "Placebo depth slope",
         "placebo_size": "Placebo size slope", "log_rr": "Reply risk ratio", "log_placebo_rr": "Placebo risk ratio"}
lines = []
for m in ITEM_M:
    cv = coverage(I, V3I, [m], 1000)
    rat = [med(entries(I, V3I, c, [m], 1000), {m}) for c in ("ppi", "dsl", "ai", "upgrade_lw", "srs")]
    lines.append(f"{NAMES[m]} & {f3(cv[0])} & {f3(cv[1])} & " + " & ".join(f2(x) for x in rat) + " \\\\")
for m in EDGE_M:
    cv = coverage(E, V3E, [m], 2000)
    rat = [med(entries(E, V3E, c, [m], 2000), {m}) for c in ("ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs")]
    lines.append(f"{NAMES[m]} & {f3(cv[0])} & {f3(cv[1])} & " + " & ".join(f2(x) for x in rat) + " \\\\")
cvc = coverage(I, "calag", ["depth_law", "size_law"], 2000); cvs = coverage(I, "calag_s", ["depth_law", "size_law"], 2000)
lines.append("\\midrule")
fc = lambda c: "--" if c[0] is None else f"{c[0]:.3f} ({c[1]:.2f})"
lines.append("Comparator coverage, laws and placebo slopes & & & " + " & ".join(fc(coverage(I, c, ITEM_M, 1000)) for c in ("ppi", "dsl", "ai", "upgrade_lw", "srs")) + " \\\\")
lines.append("Comparator coverage, risk ratios & & & " + " & ".join(fc(coverage(E, c, EDGE_M, 2000)) for c in ("ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs")) + " \\\\")
lines.append("\\midrule")
lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{Calibrate and aggregate, laws at ${{\\ge}}$2{{,}}000 labels: coverage {f3(cvc[0])} (min {f2(cvc[1])}); by level, depth and size {f3(cvs[0])} (min {f2(cvs[1])})}} \\\\")
open(f"{W}/paper_v4/tables/confirm_v4.tex", "w").write("\n".join(lines) + "\n")
al = []
for a, lab in ABL_I:
    v = res["ablations"][a]; al.append(f"{lab} & {f2(v['laws'])} & {f2(v['placebo'])} & {f2(v['gold'])} & {f2(v['plac_rr'])} \\\\")
open(f"{W}/paper_v4/tables/ablation.tex", "w").write("\n".join(al).replace("& ?? & ?? \\\\", "& -- & -- \\\\") + "\n")
print(json.dumps({k: res[k] for k in ("H74h", "H74i", "H74j", "v3_bias_sd")}, indent=1))
print(json.dumps({c: {k: v for k, v in res["items"][c].items() if k != "by_metric"} for c in COMP_I}, indent=0)[:2500])
print(json.dumps(res["edges"], indent=0)[:1500])
print(json.dumps(res["ablations"], indent=0)[:2000])
print(NB)
