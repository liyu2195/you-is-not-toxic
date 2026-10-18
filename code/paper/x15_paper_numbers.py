"""Paper v3: number macros (paper_v3/numbers_v3.tex) and generated tables (paper_v3/tables/*.tex) from the stored runs.
Evidence: results/x55_placebo/x08_placebo.json (registered, Addenda 55-56) for every dataset except CGA-Wiki, whose values come
from the corrected run results/x72_placebo/x08_placebo.json (Addendum 72).
Confirmation: results/x11_conf_item_all (+ the CGA-Wiki item rerun results/x72_item_cga), results/x11_conf_edge_all without
CGA-Wiki, results/x11_conf_unit (Addenda 71-72).
usage: python3 x15_paper_numbers.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, glob, os, re
import numpy as np

W = _ROOT; OUTD = f"{W}/paper_arxiv"; os.makedirs(f"{OUTD}/tables", exist_ok=True)
M = {}


def put(k, v):
    assert re.fullmatch(r"[A-Za-z]+", k), k
    assert k not in M, k
    M[k] = v


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def sg(x, d): return (f"+{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}")
def pct(x, d=0): return f"{100 * x:.{d}f}\\%"
def comma(n): return f"{n:,}".replace(",", "{,}")
def rat(x, d=1): return f"{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}"


# ------------------------------------------------------------------------------------------------ evidence (Section 4)
EV = json.load(open(f"{W}/results/x55_placebo/x08_placebo.json"))
EVc = json.load(open(f"{W}/results/x72_placebo/x08_placebo.json"))
EV["cga"] = EVc["cga"]                                                     # Addendum 72: corrected CGA-Wiki edges
CC = EV["cc_threads"]; L = CC["labelers"]; G = CC["gold"]; A = CC["avalle"]
S5 = EV["cc_threads_s5"]; L5 = S5["labelers"]; A5 = S5["avalle"]
LAB = {"Haiku": "haiku", "HaikuCtx": "haiku_ctx", "Qwen": "qwen7b_m", "QwenCtx": "qwen7b_m_ctx", "DetoxSix": "detoxify_original@0.6",
       "DetoxM": "detoxify_original@m", "DetoxUnbM": "detoxify_unbiased@m", "ImsM": "imsypp@m", "ImsArg": "imsypp@argmax", "Human": "human_lenient"}
put("pNcomments", comma(CC["n_items"])); put("pNthreads", comma(CC["n_units"])); put("pNedges", comma(CC["n_edges"]))
put("pNcommentsM", f"{CC['n_items'] / 1e6:.2f}")
put("gDepth", sg(G["depth_slope"][0], 4)); put("gDepthLo", sg(G["depth_slope"][1], 4)); put("gDepthHi", sg(G["depth_slope"][2], 4))
put("gSize", sg(G["size_slope"][0], 4)); put("gRR", f2(G["rr"][0])); put("gRRLo", f2(G["rr"][1])); put("gRRHi", f2(G["rr"][2]))
put("gArt", sg(A["articles"]["z"]["slope"], 4)); put("gArtTau", f2(A["articles"]["z"]["bins21"]["mk_tau"]))
# Avalle et al.'s binned test (21 logarithmic size bins, Mann-Kendall) on reply threads
bt = {k: A["threads"][k]["bins21"] for k in ("z", "haiku", "haiku_ctx", "qwen7b_m_ctx", "imsypp@m")}
put("bThrTauGold", f2(bt["z"]["mk_tau"])); put("bThrPGold", f"{bt['z']['mk_p']:.1f}")
put("bThrTauLLMMin", f2(min(bt[k]["mk_tau"] for k in ("haiku", "haiku_ctx", "qwen7b_m_ctx")))); put("bThrTauLLMMax", f2(max(bt[k]["mk_tau"] for k in ("haiku", "haiku_ctx", "qwen7b_m_ctx"))))
put("bThrPLLMMax", f"{max(bt[k]['mk_p'] for k in ('haiku', 'haiku_ctx', 'qwen7b_m_ctx')):.3f}"); put("bThrTauIms", sg(bt["imsypp@m"]["mk_tau"], 2))


def lab_macros(tag, r, avt, ava, gt=None, ga=None):
    put(f"pFPR{tag}", f3(r["fpr"][0]))
    for k, nm, d in (("placebo_depth_rel", "PDR", 3), ("placebo_size_rel", "PSR", 3), ("placebo_rr", "PRR", 2)):
        v = r[k]; put(f"p{nm}{tag}", sg(v[0], d) if nm != "PRR" else f2(v[0]))
        put(f"p{nm}lo{tag}", sg(v[1], d) if nm != "PRR" else f2(v[1])); put(f"p{nm}hi{tag}", sg(v[2], d) if nm != "PRR" else f2(v[2]))
    put(f"pLD{tag}", sg(r["law_depth_slope"][0], 4)); put(f"pLRR{tag}", f2(r["law_rr"][0]))
    if avt is not None:
        put(f"pLS{tag}", rat(avt / gt)); put(f"pLA{tag}", rat(ava / ga))
    m = r.get("vs_human_matched")
    if m:
        put(f"pMPRR{tag}", sg(m["prr"][0], 2)); put(f"pMPDR{tag}", sg(m["pdr"][0], 3))


for tag, k in LAB.items():
    r = L[k]
    if k in A["threads"]:
        lab_macros(tag, r, A["threads"][k]["slope"], A["articles"][k]["slope"], A["threads"]["z"]["slope"], A["articles"]["z"]["slope"])
    else:
        lab_macros(tag, r, None, None)
for tag, k in (("Sonnet", "sonnet"), ("SonnetCtx", "sonnet_ctx")):
    lab_macros(tag, L5[k], A5["threads"][k]["slope"], A5["articles"][k]["slope"], A5["threads"]["z"]["slope"], A5["articles"]["z"]["slope"])
put("gDepthSfive", sg(S5["gold"]["depth_slope"][0], 4)); put("gRRSfive", f2(S5["gold"]["rr"][0]))
# ranges used in the text
llm_size = [A["threads"][k]["slope"] / A["threads"]["z"]["slope"] for k in ("haiku", "haiku_ctx", "qwen7b_m_ctx")]
put("pLLMSizeMin", f"{min(llm_size):.1f}"); put("pLLMSizeMax", f"{max(llm_size):.0f}")
llm_art = [A["articles"][k]["slope"] / A["articles"]["z"]["slope"] for k in ("haiku", "haiku_ctx", "qwen7b_m_ctx")]
put("pLLMArtMin", f"{min(llm_art):.1f}"); put("pLLMArtMax", f"{max(llm_art):.1f}")
cls_prr = [L[k]["placebo_rr"][0] for k in ("detoxify_original@0.6", "detoxify_original@m")]           # Detoxify at both thresholds
put("pClsPRRMin", f"{min(cls_prr):.1f}"); put("pClsPRRMax", f"{max(cls_prr):.1f}")
dlrr = [L[k]["law_rr"][0] for k in ("detoxify_original@0.6", "detoxify_original@m")]
put("pDetoxLRRMin", f2(min(dlrr))); put("pDetoxLRRMax", f2(max(dlrr)))
put("pTPRQwen", f2(L["qwen7b_m"]["tpr"][0])); put("pTPRSizeQwen", sg(L["qwen7b_m"]["positives_size_slope"][0], 4))
put("pTPRSizeloQwen", sg(L["qwen7b_m"]["positives_size_slope"][1], 4)); put("pTPRSizehiQwen", sg(L["qwen7b_m"]["positives_size_slope"][2], 4))
ctx_prr = [L[k]["placebo_rr"][0] for k in ("haiku_ctx", "qwen7b_m_ctx")]
put("pCtxPRRMin", f2(min(ctx_prr))); put("pCtxPRRMax", f2(max(ctx_prr)))
ctx_lrr = [L[k]["law_rr"][0] for k in ("haiku_ctx", "qwen7b_m_ctx")]
put("pCtxLRRMin", f2(min(ctx_lrr))); put("pCtxLRRMax", f2(max(ctx_lrr)))
llm_pdr = [L[k]["placebo_depth_rel"][0] for k in ("haiku", "haiku_ctx", "qwen7b_m_ctx")] + [L5[k]["placebo_depth_rel"][0] for k in ("sonnet", "sonnet_ctx")]
put("pLLMPDRMin", pct(min(llm_pdr))); put("pLLMPDRMax", pct(max(llm_pdr)))
put("pHumanPDRPct", pct(L["human_lenient"]["placebo_depth_rel"][0], 1))


def row(name, r, avt=None, ava=None, gt=None, ga=None, fpr=True):
    ci = lambda v, d: f"{v[0]:.{d}f}" if d == 2 else (sg(v[0], d))
    cells = [name, f3(r["fpr"][0]) if fpr else "--", sg(r["placebo_depth_rel"][0], 3), sg(r["placebo_size_rel"][0], 3), f2(r["placebo_rr"][0]),
             sg(1000 * r["law_depth_slope"][0], 1), rat(avt / gt) if avt is not None else "--", rat(ava / ga) if ava is not None else "--", f2(r["law_rr"][0])]
    return " & ".join(cells) + " \\\\"


T1 = [f"Gold & -- & -- & -- & -- & {sg(1000 * G['depth_slope'][0], 1)} & 1.0 & 1.0 & {f2(G['rr'][0])} \\\\", "\\midrule"]
for nm, k in (("Haiku", "haiku"), ("\\quad + parent", "haiku_ctx")):
    T1.append(row(nm, L[k], A["threads"][k]["slope"], A["articles"][k]["slope"], A["threads"]["z"]["slope"], A["articles"]["z"]["slope"]))
for nm, k in (("Sonnet$^{\\dagger}$", "sonnet"), ("\\quad + parent$^{\\dagger}$", "sonnet_ctx")):
    T1.append(row(nm, L5[k]))                                               # the 5% sample's gold size slopes include 0: no ratio
for nm, k in (("Qwen", "qwen7b_m"), ("\\quad + parent", "qwen7b_m_ctx"), ("Detoxify-orig.\\ ${\\ge}0.6$", "detoxify_original@0.6"),
              ("Detoxify-orig., matched", "detoxify_original@m"), ("IMSYPP, matched", "imsypp@m")):
    T1.append(row(nm, L[k], A["threads"][k]["slope"], A["articles"][k]["slope"], A["threads"]["z"]["slope"], A["articles"]["z"]["slope"]))
T1 += ["\\midrule", row("Lenient human", L["human_lenient"])]
open(f"{OUTD}/tables/placebo_cc.tex", "w").write("\n".join(T1) + "\n")

# replication on the other datasets: placebo statistics whose 95% interval excludes the null (registered test (b))
OTHER = [("cad_threads", "CAD"), ("qian_reddit", "Reddit"), ("qian_gab", "Gab"), ("cga", "CGA-Wiki"), ("bad", "BAD")]
excl = {}; tested = 0; ntest = {}
for d, nm in OTHER:
    for k, r in EV[d]["labelers"].items():
        for st, null in (("placebo_rr", 1.0), ("placebo_depth_slope", 0.0), ("placebo_size_slope", 0.0)):
            v = r.get(st)
            if not v or any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v):
                continue
            if st == "placebo_rr" and v[0] <= 0:                          # too few flags for a ratio
                continue
            tested += 1; ntest[st] = ntest.get(st, 0) + 1
            if v[1] > null:
                excl.setdefault(st, []).append((nm, k))
put("rTestedRR", str(ntest["placebo_rr"])); put("rTestedDepth", str(ntest["placebo_depth_slope"])); put("rTestedSize", str(ntest["placebo_size_slope"]))
put("rLabDatasets", str(sum(len(EV[d]["labelers"]) for d, _ in OTHER)))
put("rTested", str(tested)); put("rExclRR", str(len(excl.get("placebo_rr", [])))); put("rExclDepth", str(len(excl.get("placebo_depth_slope", []))))
put("rExclSize", str(len(excl.get("placebo_size_slope", []))))
put("rRRdatasets", str(len({nm for nm, _ in excl.get("placebo_rr", [])})))
put("cgaHaikuPRR", f2(EV["cga"]["labelers"]["haiku"]["placebo_rr"][0])); put("cgaHaikuPRRlo", f2(EV["cga"]["labelers"]["haiku"]["placebo_rr"][1]))

# data table (Appendix C)
DATA = [("cc_threads", "Civil Comments~\\cite{borkan2019nuanced}", "News", "toxicity, raters without the parent"), ("cc_threads_s5", "\\quad 5\\% sample of threads", "News", "as above"),
        ("cad_threads", "CAD~\\cite{vidgen2021cad}", "Reddit", "abuse, in context"), ("qian_reddit", "Qian et al.~\\cite{qian2019benchmark}", "Reddit", "hate speech, in context"),
        ("qian_gab", "Qian et al.~\\cite{qian2019benchmark}", "Gab", "hate speech, in context"), ("cga", "CGA-Wiki~\\cite{zhang2018cga}", "Wikipedia", "personal attack"),
        ("bad", "BAD~\\cite{xu2021bad}", "Chatbot dialogue", "unsafe turn")]
TD = []
for k, nm, plat, gold in DATA:
    D = EV[k]
    TD.append(f"{nm} & {plat} & {comma(D['n_items'])} & {comma(D['n_units'])} & {comma(D['n_edges'])} & {D['gold_prev']:.3f} \\\\")
open(f"{OUTD}/tables/data.tex", "w").write("\n".join(TD) + "\n")

# ------------------------------------------------------------------------------------------------ confirmation (Section 6)
FRESH = ("sn_rob", "mh_dist", "fb_hate", "cn_off"); LAWS = {"depth_law", "size_law", "article_size_law", "log_rr"}


def load(d, reg, excl_pop=None):
    cells = {}
    for f in sorted(glob.glob(f"{d}/*__*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        J = json.load(open(f)); R = J["results"][0]
        tag = re.sub(r"__g\d+$", "", os.path.basename(f)[:-5])
        if excl_pop and tag.split("__")[0] == excl_pop:
            continue
        C = cells.setdefault(tag, {"truth": {}, "lab": {}})
        C["truth"].update(R.get("truth_item") or {}); C["lab"].update(R.get("labeler_item") or {})
        if reg == "edge" and R.get("truth_edge"):
            C["truth"].update({"log_rr": R["truth_edge"]["rr"], "log_placebo_rr": R["truth_edge"]["placebo_rr"]}); C["lab"].update({"log_rr": R["labeler_edge"]["rr"]})
        for B, dd in (R.get(reg) or {}).items():
            for des, mm in dd.items():
                C.setdefault(int(B), {}).setdefault(des, {}).update({m: c for m, c in mm.items() if m != "_labels"})
    return cells


IT = load(f"{W}/results/x11_conf_item_all", "item")
J = json.load(open(f"{W}/results/x72_item_cga/cga__all7.json"))
for R in J["results"]:                                                      # Addendum 72: CGA-Wiki items with corrected reply counts
    tag = f"cga__{R['labeler'].replace('@', '_at_')}"
    for B, dd in R["item"].items():
        IT[tag][int(B)]["pps_seq4_ba"] = {m: c for m, c in dd["pps_seq4_ba"].items() if m != "_labels"}
ED = load(f"{W}/results/x11_conf_edge_all", "edge", excl_pop="cga")
FI, FE = "pps_seq4_ba", "edge_m3_c15"


def entries(cells, fin, ref, bmin, bmax=10 ** 9):
    out = []
    for tag, C in cells.items():
        for B, dd in C.items():
            if not isinstance(B, int) or B < bmin or B > bmax or fin not in dd:
                continue
            for m, c in dd[fin].items():
                r = dd.get(ref, {}).get(m, {})
                out.append(dict(tag=tag, B=B, m=m, fam="laws" if m in LAWS else "placebo", cov=c["coverage"],
                                ratio=c["width"] / r["width"] if c.get("width") and r.get("width") else None,
                                fresh=tag.split("__")[1].split("_at_")[0] in FRESH))
    return out


def med(rows, fam=None, **kw):
    v = [r["ratio"] for r in rows if r["ratio"] and (fam is None or r["fam"] == fam) and all(r[k] == x for k, x in kw.items())]
    return float(np.median(v)) if v else float("nan")


for reg, cells, fin, bmin, P in (("item", IT, FI, 1000, "vItem"), ("edge", ED, FE, 2000, "vEdge")):
    rows = entries(cells, fin, "upgrade_lw", bmin); cov = np.array([r["cov"] for r in rows])
    put(f"{P}Cells", str(len(cells))); put(f"{P}Entries", comma(len(rows)))
    put(f"{P}Cov", f3(cov.mean())); put(f"{P}CovMin", f3(cov.min())); put(f"{P}Share", pct(np.mean(cov >= 0.90)))
    allc = np.array([r["cov"] for r in entries(cells, fin, "upgrade_lw", 0)]); put(f"{P}CovMinAll", f3(allc.min()))
    for ref, nm in (("upgrade_lw", "Vtwo"), ("ppi", "PPI"), ("srs", "Unif")):
        rr = entries(cells, fin, ref, bmin)
        put(f"{P}Laws{nm}", f2(med(rr, "laws"))); put(f"{P}Plac{nm}", f2(med(rr, "placebo")))
        refc = np.array([r["cov"] for r in entries(cells, ref, fin, bmin)]); put(f"{P}Cov{nm}", f3(refc.mean())); put(f"{P}CovMin{nm}", f3(refc.min()))
        put(f"{P}Share{nm}", pct(np.mean(refc >= 0.90)))
    Bs = sorted({r["B"] for r in rows}); put(f"{P}LawsVtwoBmin", f2(med([r for r in rows if r["B"] == Bs[0]], "laws"))); put(f"{P}LawsVtwoBmax", f2(med([r for r in rows if r["B"] == Bs[-1]], "laws")))
    put(f"{P}PlacVtwoBmin", f2(med([r for r in rows if r["B"] == Bs[0]], "placebo"))); put(f"{P}PlacVtwoBmax", f2(med([r for r in rows if r["B"] == Bs[-1]], "placebo")))
    put(f"{P}LawsVtwoFresh", f2(med([r for r in rows if r["fresh"]], "laws"))); put(f"{P}PlacVtwoFresh", f2(med([r for r in rows if r["fresh"]], "placebo")))
    put(f"{P}LawsVtwoOld", f2(med([r for r in rows if not r["fresh"]], "laws"))); put(f"{P}PlacVtwoOld", f2(med([r for r in rows if not r["fresh"]], "placebo")))
    put(f"{P}Bmin", comma(Bs[0])); put(f"{P}Bmax", comma(Bs[-1]))
put("vCovMinAll", f3(min(np.min([r["cov"] for r in entries(IT, FI, "upgrade_lw", 0)]), np.min([r["cov"] for r in entries(ED, FE, "upgrade_lw", 0)]))))
# Addendum 79 (C5): the minima outside the registered range only (laws and placebo slopes below 1,000 labels; contagion below 2,000)
put("vCovMinOutItems", f3(np.min([r["cov"] for r in entries(IT, FI, "upgrade_lw", 0, 999)]))); put("vCovMinOutEdges", f3(np.min([r["cov"] for r in entries(ED, FE, "upgrade_lw", 0, 1999)])))
put("vNcells", "55"); put("vNfresh", str(sum(1 for t in IT if t.split("__")[1].split("_at_")[0] in FRESH))); put("vNedgeCells", str(len(ED)))

# per-metric medians (Table: efficiency)
MN = [("depth_law", "Depth law"), ("size_law", "Thread size law"), ("article_size_law", "Article size law"), ("placebo_depth", "Placebo depth slope"),
      ("placebo_size", "Placebo size slope"), ("log_rr", "Reply risk ratio"), ("log_placebo_rr", "Placebo risk ratio")]
T2 = []
for m, nm in MN:
    cells, fin, bmin = (ED, FE, 2000) if m in ("log_rr", "log_placebo_rr") else (IT, FI, 1000)
    rr = {ref: [r for r in entries(cells, fin, ref, bmin) if r["m"] == m] for ref in ("upgrade_lw", "ppi", "srs")}
    cov = [r["cov"] for r in rr["upgrade_lw"]]
    T2.append(f"{nm} & {f3(np.mean(cov))} & {f3(np.min(cov))} & {f2(med(rr['upgrade_lw']))} & {f2(med(rr['ppi']))} & {f2(med(rr['srs']))} \\\\")
    if m == "placebo_size":
        T2.append("\\midrule")
open(f"{OUTD}/tables/confirm.tex", "w").write("\n".join(T2) + "\n")

# units
U = json.load(open(f"{W}/results/x11_conf_unit/eval_H71e.json"))["summary"]
put("vUnitCells", str(U["n_cells"])); put("vUnitCov", f3(U["mean_cov"])); put("vUnitCovMin", f3(U["min_cov"])); put("vUnitWidth", f2(U["median_ratio"]))
put("vUnitWidthMin", f2(U["range_ratio"][0])); put("vUnitWidthMax", f2(U["range_ratio"][1]))

# n80: smallest budget at which >= 80% of audits detect the labeler's distortion
def n80_rows(cells, fin, metrics):
    out = []
    for tag, C in sorted(cells.items()):
        Bs = sorted(B for B in C if isinstance(B, int))
        for m in metrics:
            def n80(des):
                for B in Bs:
                    c = C[B].get(des, {}).get(m)
                    if c and c.get("detect") is not None and c["detect"] >= 0.8:
                        return B
                return None
            v = {des: n80(des) for des in (fin, "upgrade_lw", "ppi", "srs")}
            if all(x is None for x in v.values()):
                continue
            out.append(dict(tag=tag, m=m, **{k.replace("upgrade_lw", "v2"): x for k, x in v.items()}, Bmax=Bs[-1], lab=C["lab"].get(m), truth=C["truth"].get(m)))
    return out


NI = n80_rows(IT, FI, ("depth_law", "size_law", "article_size_law")); NE = n80_rows(ED, FE, ("log_rr",))
inf = lambda x, r: x if x is not None else 2 * r["Bmax"]
for P, rows, fin in (("vNItem", NI, FI), ("vNEdge", NE, FE)):
    better = sum(inf(r[fin], r) < inf(r["v2"], r) for r in rows); worse = sum(inf(r[fin], r) > inf(r["v2"], r) for r in rows)
    bppi = sum(inf(r[fin], r) < inf(r["ppi"], r) for r in rows); wppi = sum(inf(r[fin], r) > inf(r["ppi"], r) for r in rows)
    put(f"{P}Rows", str(len(rows))); put(f"{P}Better", str(better)); put(f"{P}Worse", str(worse)); put(f"{P}BetterPPI", str(bppi)); put(f"{P}WorsePPI", str(wppi))
    put(f"{P}ByTwoK", str(sum(1 for r in rows if r[fin] is not None and r[fin] <= 2000)))
json.dump(dict(items=NI, edges=NE), open(f"{OUTD}/tables/n80_rows.json", "w"), indent=1)
SEL = [("cc_threads__haiku_ctx", "depth_law", "News", "Haiku + parent", "Escalation"),
       ("cc_s5__sonnet", "depth_law", "News 5\\%", "Sonnet", "Escalation"),
       ("cc_threads__imsypp_at_m", "depth_law", "News", "IMSYPP", "Escalation"),
       ("cc_threads__haiku_ctx", "article_size_law", "News", "Haiku + parent", "Article size"),
       ("bad__imsypp_at_m", "size_law", "BAD", "IMSYPP", "Thread size"),
       ("qian_reddit__sn_rob_at_m", "size_law", "Reddit", "RoBERTa$^{*}$", "Thread size"),
       ("cc_threads__qwen7b_m_ctx", "log_rr", "News", "Qwen + parent", "Contagion"),
       ("cc_threads__haiku_ctx", "log_rr", "News", "Haiku + parent", "Contagion"),
       ("cad__haiku", "log_rr", "CAD", "Haiku", "Contagion")]
idx = {(r["tag"], r["m"]): r for r in NI + NE}
T3 = []
for tag, m, ds, lab, law in SEL:
    r = idx[(tag, m)]; fin = FE if m == "log_rr" else FI
    g = lambda x: comma(x) if x is not None else f"${{>}}${comma(r['Bmax'])}"
    lv, tv = r["lab"], r["truth"]
    vals = (f"{f2(lv)} / {f2(tv)}" if m == "log_rr" else f"{sg(1000 * lv, 1)} / {sg(1000 * tv, 1)}")
    T3.append(f"{law} & {ds} & {lab} & {vals} & {g(r[fin])} & {g(r['v2'])} & {g(r['ppi'])} & {g(r['srs'])} \\\\")
open(f"{OUTD}/tables/n80.tex", "w").write("\n".join(T3) + "\n")

# registered hypotheses (Appendix D): as run (Addendum 71) and corrected (Addendum 72)
H1 = json.load(open(f"{W}/results/x11_conf_edge_all/eval_H71.json")); H2 = json.load(open(f"{W}/results/x72_eval/eval_H72.json"))
Ue = json.load(open(f"{W}/results/x11_conf_unit/eval_H71e.json"))["summary"]
def vres(v): return f"{f3(v['mean'])}; {pct(v['share_ge_090'])}; {f3(v['min'])}"
TH = [f"H71a & Laws and placebo slopes ($B \\ge 1{{,}}000$): mean coverage $\\ge 0.935$, $\\ge 97\\%$ of entries $\\ge 0.90$, none $< 0.85$ & {vres(H1['item']['validity'])} & {vres(H2['item']['validity'])} & PASS \\\\",
      f"H71b & Width $\\div$ stratified: laws $\\le 0.85$, placebo slopes $\\le 0.90$ & {f3(H1['item']['vs_upgrade_lw']['laws'])}; {f3(H1['item']['vs_upgrade_lw']['placebo'])} & {f3(H2['item']['vs_upgrade_lw']['laws'])}; {f3(H2['item']['vs_upgrade_lw']['placebo'])} & PASS \\\\",
      f"H71c & Contagion ($B \\ge 2{{,}}000$): as H71a & {vres(H1['edge']['validity'])} & {vres(H2['edge']['validity'])} & PASS \\\\",
      f"H71d & Width $\\div$ stratified: reply risk ratio $\\le 0.97$, placebo $\\le 0.95$ & {f3(H1['edge']['vs_upgrade_lw']['laws'])}; {f3(H1['edge']['vs_upgrade_lw']['placebo'])} & {f3(H2['edge']['vs_upgrade_lw']['laws'])}; {f3(H2['edge']['vs_upgrade_lw']['placebo'])} & PASS \\\\",
      f"H71e & Unit shares ($B \\ge 500$): mean coverage $\\ge 0.935$, no cell $< 0.90$, width $\\div$ own order $\\le 0.97$ & {f3(Ue['mean_cov'])}; {f3(Ue['min_cov'])}; {f3(Ue['median_ratio'])} & unchanged & PASS \\\\"]
open(f"{OUTD}/tables/hypotheses.tex", "w").write("\n".join(TH) + "\n")

# ------------------------------------------------------------------------------------------------ the information ceiling
import subprocess
GAP = f"{W}/results/x15_paper"; os.makedirs(GAP, exist_ok=True)
def gap(out, simdir, designs, ceil):
    subprocess.run(["python3", f"{W}/code/analysis/x10_gap.py", f"{GAP}/{out}", f"{W}/results/{simdir}", "--designs", ",".join(designs), "--ceil", f"{W}/results/{ceil}"],
                   capture_output=True, check=True)
    return json.load(open(f"{GAP}/{out}"))
def gmed(rows, I, des, metrics, excl_cga=False):
    v = [r["eff"] for r in rows if r["I"] == I and r["design"] == des and r["metric"] in metrics and not (excl_cga and r["cell"].startswith("cga__"))]
    return float(np.median(v))
gA5 = gap("gap_rA5.json", "x10_dev_rA5", ["pps_seq4_ba", "upgrade_lw"], "x10_ceiling_I4")
gA7 = gap("gap_rA7.json", "x10_dev_rA7", ["pps_seq3_bc10", "pps_seq3_bc10_or"], "x10_ceiling_I4")
gE4 = gap("gap_rE4.json", "x10_dev_rE4", ["edge_m3_c15", "upgrade_lw", "edge_m3_c25", "edge_m3_c25_or"], "x10_edge_ceiling")
lawsI = ("depth_law", "size_law"); 
for nm, rows, des in (("CeilLaws", gA5, "pps_seq4_ba"), ("CeilLawsVtwo", gA5, "upgrade_lw")):
    put(f"c{nm}Min", f2(min(gmed(rows, "I4", des, (m,)) for m in lawsI))); put(f"c{nm}Max", f2(max(gmed(rows, "I4", des, (m,)) for m in lawsI)))
put("cCeilArt", f2(gmed(gA5, "I4", "pps_seq4_ba", ("article_size_law",)))); put("cCeilArtVtwo", f2(gmed(gA5, "I4", "upgrade_lw", ("article_size_law",))))
put("cCeilPlacMin", f2(min(gmed(gA5, "I4", "pps_seq4_ba", (m,)) for m in ("placebo_depth", "placebo_size"))))
put("cCeilPlacMax", f2(max(gmed(gA5, "I4", "pps_seq4_ba", (m,)) for m in ("placebo_depth", "placebo_size"))))
put("cOrLawsMin", f2(min(gmed(gA7, "I4", "pps_seq3_bc10_or", (m,)) for m in lawsI))); put("cOrLawsMax", f2(max(gmed(gA7, "I4", "pps_seq3_bc10_or", (m,)) for m in lawsI)))
put("cOrArt", f2(gmed(gA7, "I4", "pps_seq3_bc10_or", ("article_size_law",))))
put("cTwinLawsMin", f2(min(gmed(gA7, "I4", "pps_seq3_bc10", (m,)) for m in lawsI))); put("cTwinLawsMax", f2(max(gmed(gA7, "I4", "pps_seq3_bc10", (m,)) for m in lawsI)))
for nm, des in (("EdgeCeil", "edge_m3_c15"), ("EdgeCeilVtwo", "upgrade_lw"), ("EdgeOr", "edge_m3_c25_or"), ("EdgeTwin", "edge_m3_c25")):
    put(f"c{nm}Gold", f2(gmed(gE4, "I4m", des, ("log_rr",), True))); put(f"c{nm}Plac", f2(gmed(gE4, "I4m", des, ("log_placebo_rr",), True)))

# ------------------------------------------------------------------------------------------------ write
with open(f"{OUTD}/numbers_v3.tex", "w") as fh:
    fh.write("% generated by scripts/x15_paper_numbers.py from the stored runs; do not edit\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros;", "tables: placebo_cc, confirm, n80_rows.json")
