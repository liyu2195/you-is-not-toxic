"""Paper v4 revision: number macros for the Addendum 75 and 76 results (paper_v4/numbers_v4b.tex), prefix 'q'.
Sources: results/x75_onerater (B75b), x75_decomp (B75c, B75d), x75_within (B75e, B75f), x75_edit (B75g), x76_select (B76a),
x76_rethreshold (B76e), x75_rating (B75a, via x75_rating_eval.json), x76_edit (B76b, B76c), x76_eval (B76d).
Pending sources leave '??' so that the draft compiles.
usage: python3 x17_numbers_v4b.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, re
import numpy as np
W = _ROOT; OUTD = f"{W}/paper_arxiv"
M = {}
def put(k, v):
    assert re.fullmatch(r"q[A-Za-z]+", k), k
    assert k not in M, k
    M[k] = v
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
def sg(x, d): return (f"+{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}")
def f2(x): return f"{x:.2f}"
def pct(x, d=0): return f"{100 * x:.{d}f}\\%"
def pend(*ks):
    for k in ks:
        put(k, "??")
PP = 100.0
LLM_CC = ["haiku", "haiku_ctx", "qwen7b_m", "qwen7b_m_ctx"]
CLS = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
# ------------------------------------------------------------ B75b: a single human rater against the other raters' majority
O = load(f"{W}/results/x75_onerater/x75_onerater.json")
S = O["splits"]; med = lambda f: float(np.median([f(s) for s in S]))
sr = lambda k: med(lambda s: s["instr"]["crowd_A"][k]); ins = lambda L, k: med(lambda s: s["instr"][L][k])
put("qSrKappa", f2(med(lambda s: s["kappa_A_B"]))); put("qSrFPR", f"{sr('fpr'):.3f}")
put("qSrPDR", sg(100 * sr("placebo_depth_rel"), 1) + "\\%"); put("qSrPSR", sg(100 * sr("placebo_size_rel"), 1) + "\\%"); put("qSrPRR", f2(np.exp(sr("log_placebo_rr"))))
put("qSrDepthDist", f"{abs(sr('depth_minus_goldB')) * PP:.2f}"); put("qSrSizeDist", f"{abs(sr('size_minus_goldB')) * PP:.2f}")
rd = {L: med(lambda s: abs(s["instr"][L]["depth_minus_goldB"])) / med(lambda s: abs(s["instr"]["crowd_A"]["depth_minus_goldB"])) for L in ("haiku", "haiku_ctx", "qwen7b_m_ctx")}
rs = {L: med(lambda s: abs(s["instr"][L]["size_minus_goldB"])) / med(lambda s: abs(s["instr"]["crowd_A"]["size_minus_goldB"])) for L in ("haiku", "haiku_ctx", "qwen7b_m_ctx")}
put("qSrDepthRatio", f"{rd['haiku']:.0f}"); put("qSrSizeRatio", f"{rs['haiku']:.0f}")
put("qSrDepthRatioCtx", f"{rd['haiku_ctx']:.0f}")
pdw = {L: [abs(s_["instr"][L]["depth_minus_goldB"]) / abs(s_["instr"]["crowd_A"]["depth_minus_goldB"]) for s_ in S] for L in ("haiku", "haiku_ctx")}
put("qSrDepthRatioDrawMin", f"{min(pdw['haiku']):.0f}"); put("qSrDepthRatioDrawMax", f"{max(pdw['haiku']):.0f}")
put("qSrDepthRatioCtxDrawMin", f"{min(pdw['haiku_ctx']):.0f}"); put("qSrDepthRatioCtxDrawMax", f"{max(pdw['haiku_ctx']):.0f}")
assert all(r_ > 1 for v_ in pdw.values() for r_ in v_)          # smaller than Haiku's in every draw
put("qSrDepthRatioMin", f"{min(rd.values()):.0f}"); put("qSrDepthRatioMax", f"{max(rd.values()):.0f}")
put("qSrLrrDist", sg(sr("log_rr_minus_goldB"), 2)); put("qHaikuLrrDistLoo", sg(ins("haiku", "log_rr_minus_goldB"), 2))
put("qSrRR", f2(np.exp(sr("log_rr")))); put("qGoldLooRR", f2(np.exp(med(lambda s: s["gold_B"]["log_rr"]))))
put("qHaikuPDRLoo", sg(100 * ins("haiku", "placebo_depth_rel"), 1) + "\\%")
put("qNDraws", str(len(S)))
rbd = list(O["raters_by_depth"].values()); put("qRatersMin", f"{min(rbd):.1f}"); put("qRatersMax", f"{max(rbd):.1f}"); put("qShareFourRaters", pct(O["share_4_raters"]))
# ------------------------------------------------------------ B75c: contagion's three terms; B75d: severity
DC = load(f"{W}/results/x75_decomp/x75_decomp.json")
RT = load(f"{W}/results/x76_rethreshold/x76_rethreshold.json")
L5c = DC["cc_s5"]["labelers"]; LLM_S5 = ["sonnet", "sonnet_ctx", "gemma_m", "gemma_m_ctx"]; g0 = DC["cc_threads"]["gold_log_rr"]
# Addendum 79 (C3): Qwen and Gemma shown the parent enter at their own gold-prevalence threshold, the convention of Table 1 and Figure 1
Lc = dict(DC["cc_threads"]["labelers"]); Lc["qwen7b_m_ctx"] = RT["cc_threads"]["labelers"]["qwen7b_m_ctx@own"]["contagion_terms"]
L5c = dict(L5c); L5c["gemma_m_ctx"] = RT["cc_s5"]["labelers"]["gemma_m_ctx@own"]["contagion_terms"]
uni_llm = [Lc[k]["uniform"] for k in LLM_CC] + [L5c[k]["uniform"] for k in LLM_S5]; uni_cls = [Lc[k]["uniform"] for k in CLS]
put("qUniLLMMin", sg(max(uni_llm), 2)); put("qUniLLMMax", sg(min(uni_llm), 2)); put("qUniClsMin", sg(max(uni_cls), 2)); put("qUniClsMax", sg(min(uni_cls), 2))
put("qUniRRMin", f2(np.exp(g0 + min([Lc[k]["uniform"] for k in LLM_CC] + uni_cls)))); put("qUniRRMax", f2(np.exp(g0 + max([Lc[k]["uniform"] for k in LLM_CC] + uni_cls))))
left = [(g0 + Lc[k]["uniform"]) / g0 for k in LLM_CC + CLS]                     # share of gold's log RR that the permutation baseline leaves
put("qUniLeftMin", pct(min(left))); put("qUniLeftMax", pct(max(left)))
fp_cls = [Lc[k]["fp"] for k in CLS]; tp_cls = [Lc[k]["tp"] for k in CLS]
fp_llm = [Lc[k]["fp"] for k in LLM_CC] + [L5c[k]["fp"] for k in ("sonnet", "sonnet_ctx")]; tp_llm = [Lc[k]["tp"] for k in LLM_CC] + [L5c[k]["tp"] for k in LLM_S5]
put("qFPGemma", sg(L5c["gemma_m"]["fp"], 2)); put("qGemmaRR", f2(np.exp(DC["cc_s5"]["gold_log_rr"] + L5c["gemma_m"]["distortion"]))); put("qGoldRRSfive", f2(np.exp(DC["cc_s5"]["gold_log_rr"])))
put("qFPClsMin", sg(min(fp_cls), 2)); put("qFPClsMax", sg(max(fp_cls), 2)); put("qTPClsMin", sg(min(tp_cls), 2)); put("qTPClsMax", sg(max(tp_cls), 2))
put("qFPLLMMin", sg(min(fp_llm), 2)); put("qFPLLMMax", sg(max(fp_llm), 2)); put("qTPLLMMax", sg(max(tp_llm), 2))
put("qFPQwenCtx", sg(Lc["qwen7b_m_ctx"]["fp"], 2)); put("qFPHaikuCtx", sg(Lc["haiku_ctx"]["fp"], 2)); put("qFPHaiku", sg(Lc["haiku"]["fp"], 2))
put("qNClsTPPos", str(sum(t > 0 for t in tp_cls)))
put("qMaxCheck", f"{max(v['check'] for P_ in DC.values() if isinstance(P_, dict) and 'labelers' in P_ for v in P_['labelers'].values()):.0e}".replace("e-", "\\times 10^{-") + "}")
sev = DC["severity"]
for t, nm in (("0.5", "Five"), ("0.6", "Six"), ("0.7", "Seven"), ("0.8", "Eight")):
    put(f"qSevRR{nm}", f2(sev[t]["gold_rr"])); put(f"qSevPrev{nm}", pct(sev[t]["prev"], 1))
MC = ["detoxify_original@m", "sn_rob@m", "mh_dist@m", "fb_hate@m", "cn_off@m", "imsypp@m"]
d6 = [sev["0.6"][k] for k in MC]; put("qSevSixMin", sg(min(d6), 2)); put("qSevSixMax", sg(max(d6), 2)); put("qSevSixDetox", sg(sev["0.6"]["detoxify_original@0.6"], 2))
d5 = [sev["0.5"][k] for k in MC]; put("qSevFiveMin", sg(min(d5), 2)); put("qSevFiveMax", sg(max(d5), 2))
# ------------------------------------------------------------ B75e: within-thread escalation; B75f: agreement within populations
WI = load(f"{W}/results/x75_within/x75_within.json"); Wn = WI["within"]
def wlaw(pop, L):
    return Wn[pop]["gold_within"][0] + Wn[pop]["labelers"][L]["distortion"][0]
put("qGoldWithinPP", sg(Wn["cc_threads"]["gold_within"][0] * PP, 2))
for L, nm in (("haiku", "Haiku"), ("haiku_ctx", "HaikuCtx"), ("qwen7b_m_ctx", "QwenCtx"), ("qwen7b_m", "Qwen")):
    put(f"q{nm}WithinPP", sg(wlaw("cc_threads", L) * PP, 2)); put(f"q{nm}WithinZ", f"{Wn['cc_threads']['labelers'][L]['z']:.0f}")
r_ = Wn["cc_threads"]["labelers"]["haiku"]; put("qHaikuWithinFPShare", pct(r_["fp_term"][0] / r_["distortion"][0]))
put("qSonnetWithinPP", sg(wlaw("cc_s5", "sonnet") * PP, 2)); put("qSonnetCtxWithinPP", sg(wlaw("cc_s5", "sonnet_ctx") * PP, 2)); put("qGoldWithinSfivePP", sg(Wn["cc_s5"]["gold_within"][0] * PP, 2))
for pop, nm in (("qian_gab", "Gab"), ("qian_reddit", "Reddit"), ("cga", "Cga"), ("bad", "Bad"), ("cad", "Cad")):
    put(f"q{nm}WithinPP", sg(Wn[pop]["gold_within"][0] * PP, 2)); put(f"q{nm}PooledPP", sg(Wn[pop]["gold_pooled"] * PP, 2))
# instruments that flatten gold's within-thread escalation where it is strong (Gab, Reddit, CGA-Wiki, BAD)
fl = []; tot = 0
for pop in ("qian_gab", "qian_reddit", "cga", "bad"):
    gw = Wn[pop]["gold_within"][0]
    for L, r in Wn[pop]["labelers"].items():
        if L == "human_lenient":
            continue
        tot += 1; fl.append(abs(gw + r["distortion"][0]) < abs(gw))
put("qNFlatten", str(sum(fl))); put("qNFlattenOf", str(tot))
share = []; share_red = []; nless = 0; nrev = 0
for pop in ("qian_gab", "qian_reddit", "cga", "bad"):
    gw = Wn[pop]["gold_within"][0]
    for L, r in Wn[pop]["labelers"].items():
        if L == "human_lenient":
            continue
        lw_ = gw + r["distortion"][0]; nless += lw_ < gw; nrev += (lw_ * gw) < 0
        if abs(r["distortion"][0]) > 0:
            sh_ = (r["attenuation_term"][0] - gw) / r["distortion"][0]        # the uniform-error part of the distortion, (TPR - FPR - 1) law(Z)
            share.append(sh_)
            if pop == "qian_reddit":
                share_red.append(sh_)
put("qAttShareMed", pct(float(np.median(share)))); put("qAttShareRedditMed", pct(float(np.median(share_red))))
put("qNAttBelowFull", str(int(np.sum(np.array(share) < 1.0)))); put("qNFlattenLess", str(int(nless))); put("qNFlattenRev", str(int(nrev)))
AG = load(f"{W}/results/x79/x79_agree.json")                    # Addendum 79 (C4): B75f without the lenient benchmark
pops6 = [p for p in AG if p != "cad"]
for law, nm in (("depth", "Depth"), ("size", "Size"), ("depth_within", "Within")):
    v = [AG[p][law]["spearman"] for p in pops6 if law in AG[p]]
    put(f"qRho{nm}Min", sg(max(v), 2)); put(f"qRho{nm}Max", sg(min(v), 2))
v = [AG[p]["log_rr"]["spearman"] for p in AG if "log_rr" in AG[p]]; put("qRhoLrrMin", sg(max(v), 2)); put("qRhoLrrMax", sg(min(v), 2)); put("qRhoLrrNPops", str(len(v)))
put("qRhoCadDepth", sg(AG["cad"]["depth"]["spearman"], 2)); put("qRhoCadSize", sg(AG["cad"]["size"]["spearman"], 2)); put("qRhoCadWithin", sg(AG["cad"]["depth_within"]["spearman"], 2))
put("qRhoCadLrr", sg(AG["cad"]["log_rr"]["spearman"], 2))
# ------------------------------------------------------------ B75g: edits along depth
EG = load(f"{W}/results/x75_edit/x75_edit_gaps.json"); h = EG["h11_train"]; rv = EG["rev"]
for v, nm in (("E0", "Zero"), ("E1", "One"), ("E3", "Three"), ("E4", "Four")):
    put(f"qGap{nm}", sg(100 * h["variants"][v]["gap"], 1))
put("qGapZeroLo", sg(100 * h["variants"]["E0"]["ci"][0], 1)); put("qGapZeroHi", sg(100 * h["variants"]["E0"]["ci"][1], 1))
put("qGapOneLo", sg(100 * h["variants"]["E1"]["ci"][0], 1)); put("qGapOneHi", sg(100 * h["variants"]["E1"]["ci"][1], 1))
put("qGapFourLo", sg(100 * h["variants"]["E4"]["ci"][0], 1)); put("qGapFourHi", sg(100 * h["variants"]["E4"]["ci"][1], 1))
c = h["gap_E1_minus_E0"]; put("qGapCloseOne", sg(100 * c["diff"], 1)); put("qGapCloseOneLo", sg(100 * c["ci"][0], 1)); put("qGapCloseOneHi", sg(100 * c["ci"][1], 1))
c = h["gap_E4_minus_E3"]; put("qGapCloseFour", sg(100 * c["diff"], 1)); put("qGapCloseFourLo", sg(100 * c["ci"][0], 1)); put("qGapCloseFourHi", sg(100 * c["ci"][1], 1))
put("qRevGapZero", sg(100 * rv["variants"]["E0"]["gap"], 1)); put("qRevGapTwo", sg(100 * rv["variants"]["E2"]["gap"], 1))
c = rv["gap_E2_minus_E0"]; put("qRevGapChange", sg(100 * c["diff"], 1)); put("qRevGapChangeLo", sg(100 * c["ci"][0], 1)); put("qRevGapChangeHi", sg(100 * c["ci"][1], 1))
EX = load(f"{W}/results/x74_edit/x74_edit.json")
put("qRevRoot", sg(100 * EX["rev"]["neg_root"]["E2_minus_E0"]["diff"], 1)); put("qRevDeep", sg(100 * EX["rev"]["neg_deep"]["E2_minus_E0"]["diff"], 1))
put("qRevPos", sg(100 * EX["rev"]["pos_any"]["E2_minus_E0"]["diff"], 1))
# ------------------------------------------------------------ B76a: the instrument that agreement selects
SE = load(f"{W}/results/x76_select/x76_select.json"); sm = SE["summary"]["all/top_kappa"]
put("qSelNPops", str(sm["n_pops"])); put("qSelNPopsZthree", str(sm["n_pops_any_z3"])); put("qSelNEntries", str(sm["n_law_entries"])); put("qSelNEntriesZthree", str(sm["n_entries_z3"]))
fig_e = [(p_, l_) for p_ in SE if p_ != "summary" for l_, v_ in SE[p_]["all/top_kappa"]["laws"].items() if l_ != "article_size" and v_.get("z") is not None and np.isfinite(v_["z"])]
put("qSelNFig", str(len(fig_e))); put("qSelNFigZthree", str(sum(abs(SE[p_]["all/top_kappa"]["laws"][l_]["z"]) > 3 for p_, l_ in fig_e)))
all_e_ = [(p_, l_) for p_ in SE if p_ != "summary" for l_ in SE[p_]["all/top_kappa"]["laws"]]; put("qSelNAll", str(len(all_e_)))
assert sum(abs(SE[p_]["all/top_kappa"]["laws"][l_].get("z") or 0) > 3 for p_, l_ in all_e_) == sum(abs(SE[p_]["all/top_kappa"]["laws"][l_]["z"]) > 3 for p_, l_ in fig_e)
cg = SE["cga"]["all/top_kappa"]; put("qSelCgaKappa", f2(cg["kappa"])); put("qSelCgaDepthZ", f"{abs(cg['laws']['depth']['z']):.1f}"); put("qSelCgaWithinZ", f"{abs(cg['laws']['depth_within']['z']):.1f}")
put("qSelCgaDepthPP", sg(cg["laws"]["depth"]["distortion"] * PP, 2)); put("qSelCgaWithinPP", sg(cg["laws"]["depth_within"]["distortion"] * PP, 2))
gb = SE["qian_gab"]["all/top_kappa"]; put("qSelGabKappa", f2(gb["kappa"])); put("qSelGabDepthZ", f"{abs(gb['laws']['depth']['z']):.0f}"); put("qSelGabSizeZ", f"{abs(gb['laws']['size']['z']):.0f}")
cc_ = SE["cc_threads"]["no_detox_unbiased/top_kappa"]; put("qSelCcKappa", f2(cc_["kappa"])); put("qSelCcLrrZ", f"{abs(cc_['laws']['log_rr']['z']):.0f}"); put("qSelCcLrr", sg(cc_["laws"]["log_rr"]["distortion"], 2))
put("qSelRankShare", pct(SE["summary"]["all/top_kappa"]["median_rank_share"]))
# ------------------------------------------------------------ B76e: LLMs shown the parent at their own gold-prevalence threshold
RT = load(f"{W}/results/x76_rethreshold/x76_rethreshold.json")
q = RT["cc_threads"]["labelers"]["qwen7b_m_ctx@own"]; gd = RT["cc_threads"]["gold"]["depth"][0]
put("qQwenOwnFlag", pct(q["flag_rate"][0], 1)); put("qQwenOwnDepthPP", sg((gd + q["laws"]["depth"]["distortion"][0]) * PP, 2)); put("qQwenOwnDepthZ", f"{q['laws']['depth']['z']:.0f}")
put("qQwenOwnFPShare", pct(q["laws"]["depth"]["fp_term"][0] / q["laws"]["depth"]["distortion"][0])); put("qQwenOwnPDR", sg(100 * q["placebo_depth_rel"], 0) + "\\%")
put("qQwenOwnWithinPP", sg((RT["cc_threads"]["gold"]["depth_within"][0] + q["laws"]["depth_within"]["distortion"][0]) * PP, 2))
put("qQwenOwnLRR", f2(np.exp(RT["cc_threads"]["gold"]["log_rr"][0] + q["laws"]["log_rr"]["distortion"][0])))
gm = RT["cc_s5"]["labelers"]["gemma_m_ctx@own"]; put("qGemmaOwnDepthZ", f"{abs(gm['laws']['depth']['z']):.1f}")
put("qQwenOldFlag", pct(RT["cc_threads"]["labelers"]["qwen7b_m_ctx"]["flag_rate"][0], 1))
# ------------------------------------------------------------ derived ranges over the LLMs that manufacture escalation (Qwen with the parent at its own threshold)
DEC = load(f"{W}/results/x74_decomp/x74_decomp.json"); PL = load(f"{W}/results/x74_placebo/x74_placebo.json")
MAN = [("cc_threads", "haiku"), ("cc_threads", "haiku_ctx"), ("cc_s5", "sonnet"), ("cc_s5", "sonnet_ctx")]
shr = [DEC[p_]["labelers"][k]["laws"]["depth"]["fp_term"][0] / DEC[p_]["labelers"][k]["laws"]["depth"]["distortion"][0] for p_, k in MAN] + [q["laws"]["depth"]["fp_term"][0] / q["laws"]["depth"]["distortion"][0]]
put("qFPShareDepthMin", pct(min(shr))); put("qFPShareDepthMax", pct(max(shr)))
pdr_ = [PL["cc_threads" if p_ == "cc_threads" else "cc_threads_s5"]["labelers"][k]["placebo_depth_rel"][0] for p_, k in MAN] + [q["placebo_depth_rel"]]
put("qLLMPDRMin", pct(min(pdr_))); put("qLLMPDRMax", pct(max(pdr_)))
gs = RT["cc_threads"]["gold"]["size"][0]; put("qLSQwenOwn", f"{(gs + q['laws']['size']['distortion'][0]) / gs:.1f}")
rr_ctx = [PL["cc_threads"]["labelers"]["haiku_ctx"]["law_rr"][0], PL["cc_threads_s5"]["labelers"]["sonnet_ctx"]["law_rr"][0],
          float(np.exp(RT["cc_threads"]["gold"]["log_rr"][0] + q["laws"]["log_rr"]["distortion"][0])), float(np.exp(RT["cc_s5"]["gold"]["log_rr"][0] + gm["laws"]["log_rr"]["distortion"][0]))]
put("qCtxRRMin", f2(min(rr_ctx))); put("qCtxRRMax", f2(max(rr_ctx)))
cu = SE["cc_threads"]["all/top_kappa"]; put("qSelCcuKappa", f2(cu["kappa"])); put("qSelCcuLrr", sg(cu["laws"]["log_rr"]["distortion"], 2)); put("qSelCcuLrrZ", f"{abs(cu['laws']['log_rr']['z']):.0f}")
put("qEditOneDeep", sg(100 * EX["h11"]["neg_deep"]["E1_minus_E0"]["diff"], 1)); put("qEditOneRoot", sg(100 * EX["h11"]["neg_root"]["E1_minus_E0"]["diff"], 1))
BD = load(f"{W}/results/x74_bounds/x74_bounds.json")["cc_threads"]; k9 = BD["by_kappa"]["0.9"]["depth"]
put("qBoundRatio", f"{(k9[1] - k9[0]) / abs(BD['gold']['depth']):.0f}")
# ------------------------------------------------------------ B75a: real operating points (0-100 ratings), pending
RA = load(f"{W}/results/x75_rating/x75_rating_eval.json")
KEYS_A = ["qRateHaikuDepthPP", "qRateHaikuCtxDepthPP", "qRateSonnetDepthPP", "qRateSonnetCtxDepthPP", "qRateGoldDepthPP", "qRateHaikuPDR", "qRateSrPDR",
          "qRateHaikuDepthLo", "qRateSonnetDepthLo", "qRateHaikuCtxDepthLo", "qRateSonnetCtxDepthLo", "qRateGridMin", "qRateGridMax", "qRateNGridPos", "qRateNGrid"]
if RA:
    for k in KEYS_A:
        put(k, RA["macros"].get(k, "??"))
else:
    pend(*KEYS_A)
# ------------------------------------------------------------ B76b and B76c: Sonnet and the classifiers on the edits, pending
EV = load(f"{W}/results/x76_edit/x76_edit.json")
KEYS_B = ["qSonParaDiff", "qSonParaLo", "qSonParaHi", "qSonRevDiff", "qSonRevLo", "qSonRevHi", "qSonGapZero", "qSonGapOne", "qSonOneBefore", "qSonOneAfter",
          "qClsYouMin", "qClsYouMax", "qYouRho", "qYouRhoN", "qClsRevMin", "qClsRevMax"]
if EV:
    for k in KEYS_B:
        put(k, EV["macros"].get(k, "??"))
else:
    pend(*KEYS_B)
# ------------------------------------------------------------ B76d: unplanned laws, pending
UV = load(f"{W}/results/x76_eval/x76_eval.json")
KEYS_D = ["qUnCovMean", "qUnCovMin", "qUnShareNinety", "qUnWithinPPI", "qUnWithinUnif", "qUnPrevPPI", "qUnPrevUnif", "qUnNEntries", "qUnWithinDetect", "qUnWithinDetectPPI"]
if UV:
    for k in KEYS_D:
        put(k, UV["macros"].get(k, "??"))
else:
    pend(*KEYS_D)
# ------------------------------------------------------------ Addendum 77: B77b validate and correct, B77c less noisy gold, B77e unit shares, B77f checks
OC = load(f"{W}/results/x77_or_check/x77_or_check.json")
mx = max(v["max_rel_V_minus_M"] for k, v in OC.items() if isinstance(v, dict)); e_ = int(np.floor(np.log10(mx))); put("qOrCheckMax", f"${mx / 10 ** e_:.1f}\\times 10^{{{e_}}}$")
CO = load(f"{W}/results/x77_correct/x77_correct.json"); Ec = CO["entries"]; cv = np.array([e["coverage"] for e in Ec])
put("qCorrN", str(len(cv))); put("qCorrShareBelow", pct(float(np.mean(cv < 0.90)))); put("qCorrMedianCov", f"{np.median(cv):.2f}")
ws = []
for p_, P_ in CO["pops"].items():
    for L_, r_ in P_.items():
        for law_, v_ in r_["laws"].items():
            if v_["corrected_pop_minus_gold"] is not None:
                ws.append(abs(v_["corrected_pop_minus_gold"]) > abs(v_["uncorrected_minus_gold"]))
put("qCorrShareWorse", pct(float(np.mean(ws))))
rl, ru = [], []
for p_, P_ in CO["pops"].items():
    for L_, r_ in P_.items():
        for law_, v_ in r_["laws"].items():
            if v_.get("bias") is None or v_["gold"] == 0:
                continue
            rl.append(abs(v_["bias"]) / abs(v_["gold"])); ru.append(abs(v_["uncorrected_minus_gold"]) / abs(v_["gold"]))
put("qCorrMedRelBias", pct(float(np.median(rl)))); put("qUncMedRelBias", pct(float(np.median(ru))))
byp = {p_: float(np.mean(np.array([e["coverage"] for e in Ec if e["pop"] == p_]) >= 0.90)) for p_ in CO["pops"]}
put("qCorrCgaOK", pct(byp["cga"])); put("qCorrOtherOKMax", pct(max(v for k, v in byp.items() if k != "cga")))
hz = CO["pops"]["cc_threads"]["haiku"]["laws"]["depth"]; put("qCorrHaikuDepthPP", sg((hz["gold"] + hz["bias"]) * PP, 2))
cgd = CO["pops"]["cga"]["detoxify_original@m"]["laws"]["depth"]; put("qCorrCgaDetoxCov", f"{cgd['coverage']:.2f}")
gbw = CO["pops"]["qian_gab"]["fb_hate@m"]["laws"]; put("qCorrGabCov", f"{max(v['coverage'] for v in gbw.values()):.2f}")
RN = load(f"{W}/results/x77_rr_noise/x77_rr_noise.json")
put("qNoiseEdges", format(RN["n_edges"], ",").replace(",", "{,}")); put("qNoiseMeanRaters", f"{RN['mean_raters']:.0f}")
put("qNoiseRRone", f2(RN["k"]["1"]["median"])); put("qNoiseRRnine", f2(RN["k"]["9"]["median"])); put("qNoiseRRFull", f2(RN["gold_rr"])); put("qNoiseRRInf", f2(RN["rr_inf"]))
cr = [v for k, v in RN["instruments"].items() if not k.startswith(("haiku", "qwen", "sonnet", "gemma"))]
put("qNoiseClsMin", f2(min(cr))); put("qNoiseClsMax", f2(max(cr)))
UN = load(f"{W}/results/x77_units/x77_units.json")
uc = UN["cc_threads"]["all"]; put("qUnitGold", f2(uc["gold"])); put("qUnitHaiku", f2(uc["labelers"]["haiku"]["share"])); put("qUnitHaikuCtx", f2(uc["labelers"]["haiku_ctx"]["share"]))
put("qUnitSonnet", f2(UN["cc_s5"]["all"]["labelers"]["sonnet"]["share"])); put("qUnitSonnetGold", f2(UN["cc_s5"]["all"]["gold"]))
ur = UN["qian_reddit"]["all"]; put("qUnitRedditGold", f2(ur["gold"])); put("qUnitRedditHaiku", f2(ur["labelers"]["haiku"]["share"]))
um = [uc["labelers"][k]["share"] for k in uc["labelers"] if k.endswith("@m") and not k.startswith(("haiku", "qwen"))]
put("qUnitClsMin", f2(min(um))); put("qUnitClsMax", f2(max(um)))
gab = DEC["qian_gab"]; gk = max((k for k in gab["labelers"] if k != "human_lenient"), key=lambda k: gab["labelers"][k]["laws"]["depth"]["distortion"][0])
put("qGabRevLaw", sg((gab["gold"]["depth"][0] + gab["labelers"][gk]["laws"]["depth"]["distortion"][0]) * PP, 2)); res_gk = gk
# ------------------------------------------------------------ B77a: human instruments at matched flag rates; B77d: the you-free corpus
HR = load(f"{W}/results/x77_humanrate/x77_humanrate.json")
KEYS_HR = ["qHrHaikuMatched", "qHrHaikuCtxMatched", "qHrHumanMatched", "qHrHumanThirty", "qHrHaikuThirty", "qHrSonnetMatched", "qHrSonnetThirty", "qHrHumanPDRThirty"]
if HR:
    R_ = HR["rates"]
    put("qHrHaikuMatched", sg(R_["matched"]["haiku_r"]["depth"][0] * PP, 2)); put("qHrHaikuCtxMatched", sg(R_["matched"]["haiku_r_ctx"]["depth"][0] * PP, 2))
    put("qHrHumanMatched", sg(R_["matched"]["human"]["depth"][0] * PP, 2)); put("qHrHumanThirty", sg(R_["0.3"]["human"]["depth"][0] * PP, 2))
    put("qHrHaikuThirty", sg(R_["0.3"]["haiku_r"]["depth"][0] * PP, 2)); put("qHrHumanPDRThirty", sg(100 * R_["0.3"]["human"]["placebo_depth_rel"][0], 1) + "\\%")
    if "sonnet_r" in R_["matched"]:
        put("qHrSonnetMatched", sg(R_["matched"]["sonnet_r"]["depth"][0] * PP, 2)); put("qHrSonnetThirty", sg(R_["0.3"]["sonnet_r"]["depth"][0] * PP, 2))
    else:
        pend("qHrSonnetMatched", "qHrSonnetThirty")
else:
    pend(*KEYS_HR)
YF = load(f"{W}/results/x77_youfree/x77_youfree.json")
KEYS_YF = ["qYfShare", "qYfLo", "qYfHi", "qYfOrig", "qYfEdit", "qYfWithinShare", "qYfSizeShare", "qYfPosOrig", "qYfPosEdit", "qYfNYou"]
if YF:
    d_ = YF["laws"]["depth"]; m_ = d_["edit_minus_orig"]
    put("qYfShare", pct(d_["share_removed"])); put("qYfLo", sg(m_[1] * PP, 2)); put("qYfHi", sg(m_[2] * PP, 2))
    put("qYfOrig", sg(d_["dist_orig"][0] * PP, 2)); put("qYfEdit", sg(d_["dist_edit"][0] * PP, 2))
    put("qYfWithinShare", pct(YF["laws"]["depth_within"]["share_removed"])); put("qYfSizeShare", pct(YF["laws"]["size"]["share_removed"]))
    put("qYfPosOrig", f2(YF["flag_rate"]["gold_pos_orig"])); put("qYfPosEdit", f2(YF["flag_rate"]["gold_pos_edit"])); put("qYfNYou", format(YF["n_you_labeled"], ",").replace(",", "{,}"))
else:
    pend(*KEYS_YF)
# ------------------------------------------------------------ Addendum 78: publication comparisons (B77g) and Frechet ranges (B77h)
PB = load(f"{W}/results/x78/x78_pubs.json"); Mt = PB["metrics"]
put("qPubN", str(PB["n_pubs"])); put("qPubItemDec", str(Mt["item"]["n_decided"])); put("qPubThrDec", str(Mt["thread"]["n_decided"])); put("qPubArtDec", str(Mt["article"]["n_decided"]))
put("qPubHaikuItem", str(Mt["item"]["reversed"]["haiku"])); put("qPubHaikuThr", str(Mt["thread"]["reversed"]["haiku"])); put("qPubHaikuArt", str(Mt["article"]["reversed"]["haiku"]))
put("qPubHaikuCtxItem", str(Mt["item"]["reversed"]["haiku_ctx"])); put("qPubFbItem", str(Mt["item"]["reversed"]["fb_hate@m"])); put("qPubFbThr", str(Mt["thread"]["reversed"]["fb_hate@m"]))
FR = load(f"{W}/results/x78/x78_frechet.json")
for unit, nm in (("unit", "Thr"), ("article", "Art")):
    h_ = FR[unit]["instruments"]["haiku"]; put(f"qFr{nm}Lo", f2(h_["lo"])); put(f"qFr{nm}Hi", f2(h_["hi"])); put(f"qFr{nm}Gold", f2(FR[unit]["gold"])); put(f"qFr{nm}Ind", f2(h_["independence"]))
sr_ = FR["unit"]["instruments"]["sn_rob@m"]; put("qFrThrSnLo", f2(sr_["lo"])); put("qFrThrSnHi", f2(sr_["hi"]))
# ------------------------------------------------------------ Addendum 79: corrections from the independent verification
YO = load(f"{W}/results/x79/x79_you.json")["macros"]                 # C1: second-person numbers on cc_threads (civil: vote fraction < 0.2)
for k_, v_ in (("qYouShareZero", "nyouShareZero"), ("qYouShareSix", "nyouShareSix"), ("qYouFprZero", "nfprYouZero"), ("qYouFprSix", "nfprYouSix"),
               ("qNoYouFprZero", "nfprNoYouZero"), ("qNoYouFprSix", "nfprNoYouSix")):
    put(k_, YO[v_])
EV55 = load(f"{W}/results/x55_placebo/x08_placebo.json")["cc_threads"]["avalle"]["threads"]   # C3: Avalle et al.'s binned test
bn = {k_: EV55[k_]["bins21"] for k_ in ("z", "haiku", "haiku_ctx", "qwen7b_m")}
assert len({v_["n_bins"] for v_ in bn.values()}) == 1
put("qMKBins", str(bn["z"]["n_bins"])); put("qMKPGold", f"{bn['z']['mk_p']:.2f}"); put("qMKPHaiku", f"{bn['haiku']['mk_p']:.3f}"); put("qMKPQwen", f"{bn['qwen7b_m']['mk_p']:.2f}")
put("qMKPHaikuCtx", "0.001" if bn["haiku_ctx"]["mk_p"] < 0.001 else f"{bn['haiku_ctx']['mk_p']:.3f}"); put("qMKPHaikuCtxRel", "<" if bn["haiku_ctx"]["mk_p"] < 0.001 else "=")
put("qUnitRedditCut", pct(1 - UN["qian_reddit"]["all"]["labelers"]["haiku"]["share"] / UN["qian_reddit"]["all"]["gold"]))
fr_ = [PL["cc_threads"]["labelers"][k_]["flag_rate"][0] / PL["cc_threads"]["gold_prev"] for k_ in ("haiku", "haiku_ctx")] + \
      [PL["cc_threads_s5"]["labelers"][k_]["flag_rate"][0] / PL["cc_threads_s5"]["gold_prev"] for k_ in ("sonnet", "sonnet_ctx")]
put("qFlagRatioMin", f"{min(fr_):.1f}"); put("qFlagRatioMax", f"{max(fr_):.1f}")
put("qFlagRatioHaiku", f"{fr_[0]:.1f}"); put("qFlagRatioSonnet", f"{fr_[2]:.1f}")                         # without the parent
ga_ = PL["cc_threads"]["gold"]["article_size_slope"][0]; av_ = {k_: PL["cc_threads"]["labelers"][k_]["law_article_size_slope"][0] / ga_ for k_ in ("detoxify_original@0.6", "imsypp@argmax")}
put("qAvDetoxArt", f"{av_['detoxify_original@0.6']:.1f}"); put("qAvImsArt", f"{av_['imsypp@argmax']:.1f}"); put("qAvRatio", f"{av_['imsypp@argmax'] / av_['detoxify_original@0.6']:.0f}")
EB_ = load(f"{W}/results/x74_eval/eval_B.json"); put("qBiasEdgeMC", f2(EB_["v3_bias_sd"]["edges"]["mc_expected_max"]))   # C5
put("qBiasEdgeN", str(EB_["v3_bias_sd"]["edges"]["n"]))
CLS12 = CLS; prr_ = [PL["cc_threads"]["labelers"][k_]["placebo_rr"][1] for k_ in CLS12]                                    # lower CI ends
put("qNClsPRRSr", str(sum(v_ > float(np.exp(sr("log_placebo_rr"))) for v_ in prr_))); put("qNClsPRRLen", str(load(f"{W}/results/x74_eval/eval_A.json")["H74a"]["n_pass"]))
CR = load(f"{W}/results/x74_crowd/x74_crowd.json")["splits"]                                                          # C6: the second crowd's asymmetry
put("qCrowdPrevA", pct(float(np.median([s_["prev_A"] for s_ in CR])), 1)); put("qCrowdPrevB", pct(float(np.median([s_["prev_B"] for s_ in CR])), 1))
lr_ = [s_["instr"]["crowd_A"]["log_rr_minus_goldB"] for s_ in CR if "log_rr_minus_goldB" in s_["instr"]["crowd_A"]]
if len(lr_) == len(CR):
    put("qCrowdLrrMin", sg(max(lr_), 2)); put("qCrowdLrrMax", sg(min(lr_), 2))
else:
    pend("qCrowdLrrMin", "qCrowdLrrMax")
SG = load(f"{W}/results/x79/x79_songap.json")["h11_train"]                                                        # C7: Sonnet's depth gaps
for v_, nm_ in (("E0", "Zero"), ("E1", "One"), ("E3", "Three"), ("E4", "Four")):
    put(f"qSonGap{nm_}Lo", sg(100 * SG["variants"][v_]["ci"][0], 1)); put(f"qSonGap{nm_}Hi", sg(100 * SG["variants"][v_]["ci"][1], 1))
put("qSonGapThreeV", sg(100 * SG["variants"]["E3"]["gap"], 1)); put("qSonGapFourV", sg(100 * SG["variants"]["E4"]["gap"], 1))
put("qSonGapCut", pct(-SG["gap_E1_minus_E0"]["diff"] / SG["variants"]["E0"]["gap"]))
assert abs(100 * SG["variants"]["E0"]["gap"] - float(EV["macros"]["qSonGapZero"])) < 0.05 and abs(100 * SG["variants"]["E1"]["gap"] - float(EV["macros"]["qSonGapOne"])) < 0.05
fr5 = [k_ for k_ in PL["cc_threads"]["labelers"] if k_.endswith("@0.5")]                                          # the four new classifiers at 0.5
put("qFiveNClsPRRMin", f2(min(PL["cc_threads"]["labelers"][k_]["placebo_rr"][0] for k_ in fr5))); put("qFiveNClsPRRMax", f2(max(PL["cc_threads"]["labelers"][k_]["placebo_rr"][0] for k_ in fr5)))
put("qFiveNClsRRMin", f2(min(PL["cc_threads"]["labelers"][k_]["law_rr"][0] for k_ in fr5))); put("qFiveNClsRRMax", f2(max(PL["cc_threads"]["labelers"][k_]["law_rr"][0] for k_ in fr5)))
NBm = load(f"{W}/results/x74_eval/numbers_B.json")                                                                  # percentages for the text
put("qCalagCovPct", pct(float(NBm["wCalagCovLaws"]))); put("qCalagSCovPct", pct(float(NBm["wCalagSCovLaws"]))); put("qPpiCovMinPct", pct(float(NBm["wPpiCovMin"])))
N80 = load(f"{W}/paper_arxiv/tables/n80_rows.json")["edges"]; nn_ = [r_["edge_m3_c15"] for r_ in N80 if r_["tag"].startswith("cc") and r_["edge_m3_c15"]]
put("qEdgeNewsMin", format(min(nn_), ",").replace(",", "{,}")); put("qEdgeNewsMax", format(max(nn_), ",").replace(",", "{,}"))
AB_ = EB_["ablations"]                                                                                             # ablations as text (page limit)
put("qAblUnifGold", f2(AB_["pps_seq4_ba_unif"]["gold"])); put("qAblUnifPlacRR", f2(AB_["pps_seq4_ba_unif"]["plac_rr"]))
put("qAblAuxLaws", f2(AB_["pps_seq4_ba_noaux"]["laws"])); put("qAblAuxPlac", f2(AB_["pps_seq4_ba_noaux"]["placebo"])); put("qAblAuxGold", f2(AB_["pps_seq4_ba_noaux"]["gold"])); put("qAblAuxPlacRR", f2(AB_["pps_seq4_ba_noaux"]["plac_rr"]))
put("qAblOneGold", f2(min(AB_["pps_seq1_ba"][k_] for k_ in ("laws", "placebo", "gold", "plac_rr")))); put("qAblOneMax", f2(max(AB_["pps_seq1_ba"][k_] for k_ in ("laws", "placebo", "gold", "plac_rr"))))
put("qAblPlainCost", pct(1 - min(AB_["pps_seq4_a"]["laws"], AB_["pps_seq4_a"]["placebo"])))
u_ = [AB_["pps_seq4_ba_unif"][k_] for k_ in ("laws", "placebo", "gold", "plac_rr")]; put("qAblUnifMin", f2(min(u_))); put("qAblUnifMax", f2(max(u_)))
u_ = [AB_["pps_seq4_ba_noaux"][k_] for k_ in ("laws", "placebo", "gold", "plac_rr")]; put("qAblAuxMin", f2(min(u_))); put("qAblAuxMax", f2(max(u_)))
# ------------------------------------------------------------ Addendum 80: answers to the third review round
X80 = f"{W}/results/x80"
MA = load(f"{X80}/x80_matched.json"); IA = MA["cc_threads"]["instruments"]; IS = MA["cc_s5"]["instruments"]
put("qMatchNCls", str(MA["H80a"]["n_pass"])); put("qMatchNLLM", str(sum(d["above"] for d in MA["H80b"]["detail"])))
put("qMatchDetoxRR", f2(np.exp(IA["detoxify_original@0.6"]["lrr"][0]))); put("qMatchDetoxHum", f2(np.exp(IA["detoxify_original@0.6"]["lrr_human"][0])))
put("qMatchHaikuRR", f2(np.exp(IA["haiku"]["lrr"][0]))); put("qMatchHaikuHum", f2(np.exp(IA["haiku"]["lrr_human"][0])))
put("qMatchSonnetRR", f2(np.exp(IS["sonnet"]["lrr"][0]))); put("qMatchSonnetHum", f2(np.exp(IS["sonnet"]["lrr_human"][0])))
OWN6 = ["detoxify_original@0.6", "imsypp@argmax", "sn_rob@0.5", "mh_dist@0.5", "fb_hate@0.5", "cn_off@0.5"]
put("qMatchNOwn", str(sum(IA[k_]["lrr_minus_human"][1] > 0 for k_ in OWN6))); put("qMatchNOwnOf", str(len(OWN6)))
TH = load(f"{X80}/x80_threshold.json"); CLS7 = ["detoxify_original", "detoxify_unbiased", "imsypp", "sn_rob", "mh_dist", "fb_hate", "cn_off"]
med_t, n_t = [], []
for t_, row_ in TH["pops"]["cc_threads"]["by_t"].items():
    d_ = [row_["instruments"][c_]["lrr_minus_gold"] for c_ in CLS7]; med_t.append(float(np.median([x_[0] for x_ in d_]))); n_t.append(int(sum(x_[1] > 0 for x_ in d_)))
put("qThrNMin", str(min(n_t))); put("qThrNMax", str(max(n_t))); put("qThrMedMin", sg(min(med_t), 2)); put("qThrMedMax", sg(max(med_t), 2))
put("qThrTLo", f"{min(TH['t']):.1f}"); put("qThrTHi", f"{max(TH['t']):.1f}")
bt_ = TH["pops"]["cc_threads"]["by_t"]; put("qThrGoldRRLo", f2(bt_[f"{min(TH['t'])}"]["gold_rr"])); put("qThrGoldRRHi", f2(bt_[f"{max(TH['t'])}"]["gold_rr"]))
rl_ = [r_["depth_minus_gold"] for b_ in TH["pops"]["cc_s5"]["by_t"].values() for k_, r_ in b_["instruments"].items() if k_ in ("haiku_rating", "sonnet_rating")]
assert len(rl_) == 12 and min(x_[1] for x_ in rl_) > 0                       # every rating, every threshold: paired CI above 0
put("qThrLLMMin", sg(100 * min(x_[0] for x_ in rl_), 2)); put("qThrLLMMax", sg(100 * max(x_[0] for x_ in rl_), 2))
sl_ = [b_["instruments"]["sonnet_rating"]["lrr_minus_gold"] for b_ in TH["pops"]["cc_s5"]["by_t"].values()]; hl_ = [b_["instruments"]["haiku_rating"]["lrr_minus_gold"] for b_ in TH["pops"]["cc_s5"]["by_t"].values()]
assert min(x_[1] for x_ in sl_) > 0 and min(x_[2] for x_ in hl_) > 0                        # Sonnet above gold at every t; Haiku never significantly below
put("qThrSonLrrMin", sg(min(x_[0] for x_ in sl_), 2)); put("qThrSonLrrMax", sg(max(x_[0] for x_ in sl_), 2))
M2 = load(f"{X80}/x80_mech2.json"); put("qWithinGoldRR", f2(M2["gold"]["mh_rr"])); put("qWithinNCls", str(M2["n_cls_above_within"]))
put("qWithinNBelow", str(sum(M2["instruments"][k_]["mh_lrr_minus_gold"][2] < 0 for k_ in CLS)))
MC_ = load(f"{X80}/x80_mech.json"); Ic = MC_["instruments"]; sr1 = MC_["single_rater"]
put("qWithinPlacMin", f2(min(Ic[k_]["placebo_rr_mh_article"] for k_ in CLS))); put("qWithinPlacMax", f2(max(Ic[k_]["placebo_rr_mh_article"] for k_ in CLS))); put("qWithinPlacSr", f2(sr1["placebo_rr_mh_article"]))
put("qIdFprMin", f"{min(Ic[k_]['fpr_ratio'] for k_ in CLS):.1f}"); put("qIdFprMax", f"{max(Ic[k_]['fpr_ratio'] for k_ in CLS):.1f}"); put("qIdFprSr", f"{sr1['fpr_ratio']:.1f}")
ch_ = [Ic[k_]["placebo_rr_no_identity"] / Ic[k_]["placebo_rr"] - 1 for k_ in CLS]; put("qIdChgMin", sg(100 * min(ch_), 0) + "\\%"); put("qIdChgMax", sg(100 * max(ch_), 0) + "\\%")
put("qIdShare", pct(MC_["identity_share"]))
NZ = load(f"{X80}/x80_noise.json"); put("qNoiseSelPrev", pct(NZ["balance"]["parent_gold"]["selected"])); put("qNoiseAllPrev", pct(NZ["balance"]["parent_gold"]["all"], 1))
put("qNoiseESS", f"{NZ['weights']['ess']:.0f}"); put("qNoiseRwRR", f"{NZ['reweighted']['gold_rr_reweighted']:.0f}")
kk_ = NZ["full_population"]["k"]; put("qNoiseKOne", f2(kk_["1"])); put("qNoiseKTwo", f2(kk_["2"])); put("qNoiseKThree", f2(kk_["3"]))
MG = load(f"{X80}/x80_magnitude.json"); put("qMagN", str(MG["all"]["n"])); put("qMagShare", pct(MG["all"]["share_rel_gt_25"])); put("qMagFlip", str(MG["all"]["n_sign_flip"]))
put("qMagFlipSig", str(MG["all"]["n_sign_flip_gold_sig"])); put("qMagManuf", str(MG["all"]["n_manufactured_sig"])); put("qMagLost", str(MG["all"]["n_lost_sig"])); put("qMagMedRel", pct(MG["all"]["median_abs_rel"]))
put("qBestRel", str(MG["best_kappa_z3"]["n_rel_gt_25"])); put("qBestFlip", str(MG["best_kappa_z3"]["n_sign_flip"])); put("qBestN", str(MG["best_kappa_z3"]["n"]))
hw_ = MG["half_width_depth_cc_threads"]; put("qHwDepthLo", f2(100 * hw_["250"])); put("qHwDepthHi", f2(100 * hw_["8000"])); put("qHwDepthMid", f2(100 * hw_["1000"]))
put("qMagLLMShare", pct(MG["by_family"]["llm"]["share_rel_gt_25"])); put("qMagClsShare", pct(MG["by_family"]["classifier"]["share_rel_gt_25"]))
UC = load(f"{X80}/x80_units_cluster.json"); bb = UC["summary"]["by_B"]
put("qUcMed", f2(bb["500"]["median"])); put("qUcMin", f2(min(v_["min"] for v_ in bb.values()))); put("qUcMax", f2(max(v_["max"] for v_ in bb.values())))
put("qUcN", str(bb["500"]["n"])); put("qUcCov", f"{UC['summary']['cluster_cov_mean']:.3f}"); put("qUcCovMin", f"{UC['summary']['cluster_cov_min']:.3f}")
rr_ = [c_["unit"][B_]["ratio_ts_to_best_cluster"] for c_ in UC["cells"].values() for B_ in ("500", "1000", "2000")]; put("qUcPooled", f2(float(np.median(rr_))))
cv_ = [c_["unit"][B_]["ts_rank"]["coverage"] for c_ in UC["cells"].values() for B_ in ("500", "1000", "2000")]
with open(f"{OUTD}/tables/confirm_units.tex", "w") as f_:                                  # Table 3's unit-share row (Addendum 80, B80g)
    f_.write(f"Unit share$^{{*}}$ & {np.mean(cv_):.3f} & {min(cv_):.3f} & {np.median(rr_):.2f} & & & & \\\\\n")
LO = load(f"{X80}/x80_lopo_registered.json"); assert LO["H80g"]["edge_registered"]                     # B80f as registered (E1-E5, edges without CGA-Wiki)
put("qLopoItemN", str(LO["item"]["n_frozen_adopted"])); put("qLopoEdgeN", str(LO["edge"]["n_frozen_adopted"])); put("qLopoUnitN", str(LO["unit"]["n_frozen_adopted"]))
put("qLopoItemRuns", str(LO["item"]["n_runs"])); put("qLopoEdgeRuns", str(LO["edge"]["n_runs"])); assert LO["item"]["n_frozen_qualifies"] == LO["item"]["n_runs"] and LO["edge"]["n_frozen_qualifies"] == LO["edge"]["n_runs"]
put("qLopoEdgeNone", pct(LO["edge"]["runs"]["none"]["frozen_over_adopted"] - 1, 1))
put("qLopoItemRepl", str(LO["item"]["n_runs"] - LO["item"]["n_frozen_adopted"])); put("qLopoEdgeRepl", str(LO["edge"]["n_runs"] - LO["edge"]["n_frozen_adopted"]))
LO1 = load(f"{X80}/x80_lopo.json"); put("qLopoFirstItem", str(LO1["item"]["n_frozen_adopted"])); put("qLopoFirstEdge", str(LO1["edge"]["n_frozen_adopted"]))
put("qLopoItemMax", pct(LO["item"]["max_frozen_over_adopted"] - 1, 1)); put("qLopoEdgeMax", pct(LO["edge"]["max_frozen_over_adopted"] - 1, 1))
PRm = load(f"{W}/results/f124_prompts.json"); A_ = PRm["audit"]                                                     # Addendum 30: three Haiku prompts
put("qPrGold", sg(100 * A_["slope_z"][0], 2)); put("qPrMain", sg(100 * A_["slope_main0"][0], 2)); put("qPrBin", sg(100 * A_["slope_bin0"][0], 2)); put("qPrRev", sg(100 * A_["slope_rev0"][0], 2))
PRN = load(f"{W}/results/x81/prn.json"); put("qPrN", format(PRN["n_slope"], ",").replace(",", "{,}")); assert min(A_[k_][1] for k_ in ("slope_main0", "slope_bin0", "slope_rev0")) > 0   # comments the slopes are fit on (E2)
cb_ = A_["c_slope_bin1_minus_bin0"]; put("qPrBinCtxChg", sg(100 * cb_[0], 2)); assert cb_[0] < 0; put("qPrBinCtxChgAbs", f"{abs(100 * cb_[0]):.2f}"); put("qPrBinCtxLo", sg(100 * cb_[1], 2)); put("qPrBinCtxHi", sg(100 * cb_[2], 2))
YP = load(f"{X80}/x80_youprompt_eval.json")
KEYS_YP = ["qYpHaikuDist", "qYpHaikuDistMain", "qYpHaikuChg", "qYpHaikuChgLo", "qYpHaikuChgHi", "qYpSonnetDist", "qYpSonnetDistMain", "qYpSonnetChg", "qYpSonnetChgLo", "qYpSonnetChgHi",
           "qYpHaikuShare", "qYpSonnetShare", "qYpHaikuTPR", "qYpHaikuTPRMain"]
if YP:
    for k_ in KEYS_YP:
        put(k_, YP["macros"].get(k_, "??"))
    cm_, cy_ = YP["models"]["haiku"]["civil_main"], YP["models"]["haiku"]["civil_you"]
    g5_ = DEC["cc_s5"]["gold"]["depth"][0]; put("qYpHaikuLaw", sg(100 * (g5_ + YP["models"]["haiku"]["depth_dist_you"][0]), 2)); put("qYpSonnetLaw", sg(100 * (g5_ + YP["models"]["sonnet"]["depth_dist_you"][0]), 2))
    put("qYpHaikuLawMain", sg(100 * (g5_ + YP["models"]["haiku"]["depth_dist_main"][0]), 2)); put("qYpSonnetLawMain", sg(100 * (g5_ + YP["models"]["sonnet"]["depth_dist_main"][0]), 2))
    put("qYpShallowMain", f"{cm_['d0_you']:.3f}"); put("qYpShallowYou", f"{cy_['d0_you']:.3f}"); put("qYpDeepMain", f"{cm_['d6_you']:.3f}"); put("qYpDeepYou", f"{cy_['d6_you']:.3f}")
else:
    pend(*KEYS_YP)
# ------------------------------------------------------------ escalation table (Table 2 of v4e): per-model values
d5_ = DEC["cc_s5"]["labelers"]; put("qSonnetDepthPP", sg(100 * (DEC["cc_s5"]["gold"]["depth"][0] + d5_["sonnet"]["laws"]["depth"]["distortion"][0]), 2))
put("qMatchHumHaikuDepth", sg(100 * IA["haiku"]["depth_human"][0], 2)); put("qMatchHumSonnetDepth", sg(100 * IS["sonnet"]["depth_human"][0], 2))
put("qMatchHaikuFlag", pct(IA["haiku"]["flag_rate"])); put("qMatchSonnetFlag", pct(IS["sonnet"]["flag_rate"]))
for nm_, key_ in (("Haiku", "haiku_rating"), ("Son", "sonnet_rating")):
    v_ = [b_["instruments"][key_]["depth_minus_gold"][0] for b_ in TH["pops"]["cc_s5"]["by_t"].values()]
    put(f"qThr{nm_}Min", sg(100 * min(v_), 2)); put(f"qThr{nm_}Max", sg(100 * max(v_), 2))
# ------------------------------------------------------------ Addendum 82: the fourth review round
X82 = f"{W}/results/x82"
DS = load(f"{X82}/x82_desc.json")
if DS:
    sb_ = DS["B82b"]["summary"]; put("qOthFPMin", sg(100 * sb_["fp_min"], 2)); put("qOthFPMax", sg(100 * sb_["fp_max"], 2)); put("qOthFPN", str(sb_["n"])); put("qOthFPPos", str(sb_["n_fp_pos"]))
    put("qOthWFPMin", sg(100 * sb_["within_fp_min"], 2)); put("qOthWFPMax", sg(100 * sb_["within_fp_max"], 2))
    yc_ = DS["B82b"]["you_cad"]["by_depth"]; put("qYouCadZero", pct(yc_["0"])); put("qYouCadSix", pct(yc_["6"]))
    yr_ = DS["B82b"]["you_qian_reddit"]["by_depth"]; put("qYouRedZero", pct(yr_["0"])); put("qYouRedSix", pct(yr_["6"]))
    c_ = DS["B82c"]; hk_ = c_["kappa_ge_06"]; lk_ = c_["kappa_lt_06"]
    put("qMagHiN", str(hk_["n"])); put("qMagHiShare", pct(hk_["share_rel_gt_25"])); put("qMagHiMed", pct(hk_["median_rel"])); put("qMagHiFlip", str(hk_["n_flip"])); put("qMagHiManuf", str(hk_["n_manuf"]))
    put("qMagHiSE", f"{hk_['median_gold_se']:.1f}"); put("qMagHiSEShare", pct(hk_["share_gold_se_gt_2"])); put("qMagLoShare", pct(lk_["share_rel_gt_25"])); put("qMagLoFlip", str(lk_["n_flip"]))
    put("qMagSigShare", pct(c_["gold_significant"]["share_rel_gt_25"])); put("qMagOneShare", pct(c_["one_per_instrument"]["share_rel_gt_25"])); put("qMagOneN", str(c_["one_per_instrument"]["n"]))
    d_ = DS["B82d"]; put("qKapHaikuR", f2(d_["haiku_r"]["kappa"])); put("qKapSonnetR", f2(d_["sonnet_r"]["kappa"])); put("qKapHaikuNat", f2(d_["haiku_native"]["kappa"]))
else:
    pend("qOthFPMin", "qOthFPMax", "qOthFPN", "qOthFPPos", "qOthWFPMin", "qOthWFPMax", "qYouCadZero", "qYouCadSix", "qYouRedZero", "qYouRedSix", "qMagHiN", "qMagHiShare", "qMagHiMed",
         "qMagHiFlip", "qMagHiManuf", "qMagHiSE", "qMagHiSEShare", "qMagLoShare", "qMagLoFlip", "qMagSigShare", "qMagOneShare", "qMagOneN", "qKapHaikuR", "qKapSonnetR", "qKapHaikuNat")
LT = load(f"{X82}/x82_latent.json"); PRO = load(f"{X82}/x82_protocol.json")
if LT and PRO:
    f20 = LT["fits"]["20"]; put("qLatRR", f2(LT["rr_inf"]["rr"][0])); put("qLatRRLo", f2(LT["rr_inf"]["rr"][1])); put("qLatRRHi", f2(LT["rr_inf"]["rr"][2]))
    put("qLatRRTen", f2(LT["fits"]["10"]["rr_inf"])); put("qLatRRForty", f2(LT["fits"]["40"]["rr_inf"]))
    put("qLatSimGold", f2(PRO["sim"]["gold_rr"])); put("qLatSimOne", f2(PRO["sim"]["k1_rr"])); put("qLatOneObs", f2(PRO["obs"]["k1_rr"]))
    CLS12_ = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m_}@{t_}" for m_ in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t_ in ("0.5", "m")]
    put("qLatNAbove", str(sum(LT["instruments"][k_]["minus_inf"][1] > 0 for k_ in CLS12_))); put("qLatNBelow", str(sum(LT["instruments"][k_]["minus_inf"][2] < 0 for k_ in CLS12_)))
    pl_ = [LT["placebo_latent"][k_] for k_ in CLS12_]; put("qLatPlacMin", f2(min(pl_))); put("qLatPlacMax", f2(max(pl_)))
    n4_ = PRO["realism"]["4"]; put("qFourShare", pct(n4_["share_obs"])); put("qTenShare", pct(PRO["realism"]["10"]["share_obs"]))
else:
    pend("qLatRR", "qLatRRLo", "qLatRRHi", "qLatRRTen", "qLatRRForty", "qLatSimGold", "qLatSimOne", "qLatOneObs", "qLatNAbove", "qLatNBelow", "qLatPlacMin", "qLatPlacMax", "qFourShare", "qTenShare")
# ------------------------------------------------------------ Addendum 81: the prompt note within batch (B81a) replaces the x80 cross-batch row
Y81 = load(f"{W}/results/x81/x81_eval.json")
if Y81:
    g5_ = DEC["cc_s5"]["gold"]["depth"][0]; h_, s_ = Y81["models"]["haiku"], Y81["models"]["sonnet"]
    for k_, v_ in (("qYpHaikuLaw", h_["depth_dist_you"]), ("qYpSonnetLaw", s_["depth_dist_you"]), ("qYpHaikuLawMain", h_["depth_dist_main"]), ("qYpSonnetLawMain", s_["depth_dist_main"])):
        assert k_ in M; M[k_] = sg(100 * (g5_ + v_[0]), 2)                                   # B81a (within batch) replaces the x80 values
    put("qYrHaikuChg", sg(100 * h_["depth_change"][0], 2)); put("qYrHaikuChgLo", sg(100 * h_["depth_change"][1], 2)); put("qYrHaikuChgHi", sg(100 * h_["depth_change"][2], 2))
    put("qYrSonnetChg", sg(100 * s_["depth_change"][0], 2)); put("qYrSonnetChgLo", sg(100 * s_["depth_change"][1], 2)); put("qYrSonnetChgHi", sg(100 * s_["depth_change"][2], 2))
    put("qYrAgreeMin", pct(min(h_["drift"]["agreement"], s_["drift"]["agreement"]), 1))
    put("qYrShallowMain", pct(h_["civil_main"]["d0_noyou"], 1)); put("qYrShallowYou", pct(h_["civil_you"]["d0_noyou"], 1))
    put("qYrDeepMain", pct(h_["civil_main"]["d6_you"], 1)); put("qYrDeepYou", pct(h_["civil_you"]["d6_you"], 1))
else:
    pend("qYrHaikuChg", "qYrHaikuChgLo", "qYrHaikuChgHi", "qYrSonnetChg", "qYrSonnetChgLo", "qYrSonnetChgHi", "qYrAgreeMin", "qYrShallowMain", "qYrShallowYou", "qYrDeepMain", "qYrDeepYou")
# ------------------------------------------------------------ Addendum 82: gold's raters (B82a, B82e) and the instruments against RR_inf
if LT and PRO:
    put("qLatGoldRR", f2(LT["gold"]["rr"][0])); put("qLatLogGap", f"{abs(PRO['gaps']['gold']):.3f}")
    put("qLatNSame", str(12 - int(M["qLatNAbove"]) - int(M["qLatNBelow"])))
    put("qLatHaikuRR", f2(LT["instruments"]["haiku"]["rr"])); put("qLatQwenRR", f2(LT["instruments"]["qwen7b_m"]["rr"]))
    ab_ = sorted(LT["instruments"][k_]["rr"] for k_ in CLS12_ if LT["instruments"][k_]["minus_inf"][1] > 0); put("qLatAboveMin", f2(ab_[0])); put("qLatAboveMax", f2(ab_[-1]))
    put("qLatQFour", pct(PRO["q"]["q4"])); put("qLatFiveSixOne", pct(np.mean([PRO["realism"]["5"]["K_obs"]["1"], PRO["realism"]["6"]["K_obs"]["1"]])))
    put("qLatFiveSixOneSim", pct(np.mean([PRO["realism"]["5"]["K_sim"]["1"], PRO["realism"]["6"]["K_sim"]["1"]])))
    put("qLatAttLog", f"{abs(np.log(LT['gold']['rr'][0] / LT['rr_inf']['rr'][0])):.2f}")
# ------------------------------------------------------------ Addendum 83, B83c: size laws and unit shares against matched human instruments
MS = load(f"{W}/results/x83/x83_matched_size.json")
if MS:
    I_ = MS["cc_threads"]["instruments"]; I5_ = MS["cc_s5"]["instruments"]; R5_ = MS["cc_s5"]["ratings"]
    for nm_, d_, k_ in (("Haiku", I_, "haiku"), ("HaikuCtx", I_, "haiku_ctx"), ("Qwen", I_, "qwen7b_m_ctx@own"), ("Sonnet", I5_, "sonnet")):
        put(f"qMs{nm_}Thread", sg(100 * d_[k_]["thread_size"][0], 2)); put(f"qMs{nm_}ThreadHum", sg(100 * d_[k_]["thread_size_human"][0], 2))
        put(f"qMs{nm_}ThreadX", f"{d_[k_]['thread_size'][0] / d_[k_]['thread_size_human'][0]:.1f}")
        put(f"qMs{nm_}Art", sg(100 * d_[k_]["article_size"][0], 2)); put(f"qMs{nm_}ArtHum", sg(100 * d_[k_]["article_size_human"][0], 2))
        put(f"qMs{nm_}Unit", f2(d_[k_]["unit_share"][0])); put(f"qMs{nm_}UnitHum", f2(d_[k_]["unit_share_human"][0]))
    put("qMsHumThreadX", f"{I_['haiku']['thread_size_human'][0] / MS['cc_threads']['gold']['thread_size'][0]:.1f}")
    rt_ = [R5_[k_]["thread_size_minus_gold"] for k_ in ("haiku_r", "sonnet_r")]
    put("qMsRateThreadMin", sg(100 * min(r_[0] for r_ in rt_), 2)); put("qMsRateThreadMax", sg(100 * max(r_[0] for r_ in rt_), 2)); assert all(r_[1] > 0 for r_ in rt_)
    put("qMsRateUnitMin", f2(min(R5_[k_]["unit_share"][0] for k_ in R5_))); put("qMsRateUnitMax", f2(max(R5_[k_]["unit_share"][0] for k_ in R5_)))
    put("qMsGoldUnitSfive", f2(MS["cc_s5"]["gold"]["unit_share"][0]))
    gp_ = um + [R5_[k_]["unit_share"][0] for k_ in R5_]; put("qUnitGpMin", f2(min(gp_))); put("qUnitGpMax", f2(max(gp_)))
else:
    pend("qMsHaikuThread", "qMsHaikuThreadHum", "qMsHaikuThreadX", "qMsHumThreadX", "qMsRateThreadMin", "qMsRateThreadMax")
# ------------------------------------------------------------ Addendum 83: synthetic check of the construct-only test (simulated data)
SB = load(f"{W}/results/x83/synth_binary.json"); SC = load(f"{W}/results/x83/synth_cat.json")
if SB and SC:
    e_ = SB["instruments"]["h0"]["placebo_lrr"]["excess"]; put("qSynHz", sg(e_[0], 2)); put("qSynHzLo", sg(e_[1], 2)); put("qSynHzHi", sg(e_[2], 2))
    e_ = SB["instruments"]["clustered"]["placebo_lrr"]["excess"]; put("qSynTop", sg(e_[0], 2)); put("qSynTopLo", sg(e_[1], 2)); put("qSynTopHi", sg(e_[2], 2))
    e_ = SC["instruments"]["h0"]["placebo_lrr"]["excess"]; put("qSynCatHz", sg(e_[0], 2)); put("qSynCatHzLo", sg(e_[1], 2)); put("qSynCatHzHi", sg(e_[2], 2))
    e_ = SC["instruments"]["clustered"]["placebo_lrr"]["excess"]; put("qSynCatTop", sg(e_[0], 2)); put("qSynCatTopLo", sg(e_[1], 2)); put("qSynCatTopHi", sg(e_[2], 2))
    for nm_, S_ in (("", SB), ("Cat", SC)):
        h_ = S_["instruments"]["h0"]["placebo_slope"]["excess"]; d_ = S_["instruments"]["depthy"]["placebo_slope"]["excess"]
        assert h_[1] < 0 < h_[2] and d_[1] > 0
    put("qSynCatRRBias", f"{SC['instruments']['h0']['lrr']['pred'][0] - SC['instruments']['h0']['lrr']['obs'][0]:.1f}")
else:
    pend("qSynHz", "qSynHzLo", "qSynHzHi", "qSynTop", "qSynTopLo", "qSynTopHi", "qSynCatHz", "qSynCatHzLo", "qSynCatHzHi", "qSynCatTop", "qSynCatTopLo", "qSynCatTopHi", "qSynCatRRBias")
# ------------------------------------------------------------ Addendum 83, B83a-b: the construct-only test
CN = load(f"{W}/results/x83/x83_construct.json")
KEYS_CN = ["qConsObsMin", "qConsObsMax", "qConsPredMin", "qConsPredMax", "qConsNAbove", "qConsExcMin", "qConsExcMax", "qConsRRObsMin", "qConsRRObsMax", "qConsRRPredMin",
           "qConsRRPredMax", "qConsFourObsMin", "qConsFourObsMax", "qConsFourPredMin", "qConsFourPredMax", "qConsThrPlac", "qConsThrPlacLo", "qConsThrPlacHi", "qConsThrRR",
           "qConsHaikuObs", "qConsHaikuPred", "qConsHaikuCtxObs", "qConsHaikuCtxPred", "qConsSonnetObs", "qConsSonnetPred", "qConsSonnetCtxObs", "qConsSonnetCtxPred",
           "qConsLLMShareMax", "qConsNB", "qConsHaikuLawPred", "qConsExcTenMin", "qConsExcTenMax", "qConsExcFortyMin", "qConsExcFortyMax", "qConsShareFortyMax",
           "qConsHaikuPlacObs", "qConsHaikuPlacPred", "qConsNAboveTen", "qConsNAboveForty", "qConsFourDetox", "qConsFourDetoxPred", "qConsUnbExc", "qConsHaikuExc",
           "qConsHaikuExcLo", "qConsHaikuExcHi", "qConsHaikuShare"]
if CN:
    R_ = CN["grids"]["20"]["instruments"]; ex_ = lambda k_, st_, w_="obs": R_[k_][st_][w_][0]
    def rng2(vals, nm):
        put(f"{nm}Min", f2(min(vals))); put(f"{nm}Max", f2(max(vals)))
    rng2([np.exp(ex_(k_, "placebo_lrr")) for k_ in CLS], "qConsObs"); rng2([np.exp(ex_(k_, "placebo_lrr", "pred")) for k_ in CLS], "qConsPred")
    put("qConsNAbove", str(CN["H83a"]["n_above"]))
    exc_ = [ex_(k_, "placebo_lrr", "excess") for k_ in CLS]; put("qConsExcMin", sg(min(exc_), 2)); put("qConsExcMax", sg(max(exc_), 2))
    rng2([np.exp(ex_(k_, "lrr")) for k_ in CLS], "qConsRRObs"); rng2([np.exp(ex_(k_, "lrr", "pred")) for k_ in CLS], "qConsRRPred")
    rng2([np.exp(ex_(k_, "lrr_44")) for k_ in CLS], "qConsFourObs"); rng2([np.exp(ex_(k_, "lrr_44", "pred")) for k_ in CLS], "qConsFourPred")
    T_ = CN["grids"]["20"]["thresholds"]["pi>=0.5"]; put("qConsThrPlac", f2(np.exp(T_["placebo_lrr"]["pred"][0]))); put("qConsThrPlacLo", f2(np.exp(T_["placebo_lrr"]["pred"][1])))
    put("qConsThrPlacHi", f2(np.exp(T_["placebo_lrr"]["pred"][2]))); put("qConsThrRR", f2(np.exp(T_["lrr"]["pred"][0])))
    sh_ = []
    for nm_, k_ in (("Haiku", "haiku"), ("HaikuCtx", "haiku_ctx"), ("Sonnet", "sonnet"), ("SonnetCtx", "sonnet_ctx")):
        o_, p_ = ex_(k_, "placebo_slope"), ex_(k_, "placebo_slope", "pred"); put(f"qCons{nm_}Obs", sg(100 * o_, 2)); put(f"qCons{nm_}Pred", sg(100 * p_, 2)); sh_.append(p_ / o_)
    put("qConsLLMShareMax", pct(max(sh_))); put("qConsNB", str(CN["H83b"]["n_above"])); put("qConsHaikuLawPred", sg(100 * ex_("haiku", "depth_law", "pred"), 2))
    put("qConsHaikuPlacObs", f2(np.exp(ex_("haiku", "placebo_lrr")))); put("qConsHaikuPlacPred", f2(np.exp(ex_("haiku", "placebo_lrr", "pred"))))
    put("qConsFourDetox", f"{np.exp(ex_('detoxify_original@0.6', 'lrr_44')):.1f}"); put("qConsFourDetoxPred", f"{np.exp(ex_('detoxify_original@0.6', 'lrr_44', 'pred')):.1f}")
    put("qConsUnbExc", sg(ex_("detoxify_unbiased@m", "placebo_lrr", "excess"), 2)); put("qConsHaikuExc", sg(100 * ex_("haiku", "placebo_slope", "excess"), 2))
    put("qConsHaikuExcLo", sg(100 * R_["haiku"]["placebo_slope"]["excess"][1], 2)); put("qConsHaikuExcHi", sg(100 * R_["haiku"]["placebo_slope"]["excess"][2], 2))
    put("qConsHaikuShare", pct(ex_("haiku", "placebo_slope", "pred") / ex_("haiku", "placebo_slope")))
    for nm_, m_ in (("Ten", "10"), ("Forty", "40")):
        Rm_ = CN["grids"][m_]["instruments"]; e_ = [Rm_[k_]["placebo_lrr"]["obs"][0] - Rm_[k_]["placebo_lrr"]["pred"][0] for k_ in CLS]
        put(f"qConsExc{nm_}Min", sg(min(e_), 2)); put(f"qConsExc{nm_}Max", sg(max(e_), 2)); put(f"qConsNAbove{nm_}", str(sum(v_ > 0 for v_ in e_)))
    R40_ = CN["grids"]["40"]["instruments"]; put("qConsShareFortyMax", pct(max(R40_[k_]["placebo_slope"]["pred"][0] / R40_[k_]["placebo_slope"]["obs"][0] for k_ in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx"))))
else:
    pend(*KEYS_CN)
# ------------------------------------------------------------ Addendum 83, B83d: the construct-only test with a monotone f (grids 10, 20, 40)
MN = load(f"{W}/results/x83/x83d_par.json")
KEYS_MN = ["qMonoPredMin", "qMonoPredMax", "qMonoExcMin", "qMonoExcMax", "qMonoRRPredMin", "qMonoRRPredMax", "qMonoNTwenty", "qMonoNTen", "qMonoNForty",
           "qMonoLLMTwenty", "qMonoLLMTen", "qMonoLLMForty", "qMonoHaikuPred", "qMonoHaikuObs", "qMonoHaikuExcLo", "qMonoHaikuExcHi", "qMonoShareMin", "qMonoShareMax",
           "qMonoFourDetoxPred", "qMonoHaikuLawPred", "qMonoUnbExc", "qMonoPredAllMin", "qMonoPredAllMax", "qMonoFortyHaikuObs", "qMonoFortyHaikuPred",
           "qMonoExcAllMin", "qMonoExcAllMax", "qMonoSonnetPred", "qMonoSonnetObs", "qMonoFortyHaikuPredMin", "qMonoFortyHaikuPredMax", "qMonoRRPredAllMin",
           "qMonoRRPredAllMax", "qMonoUnbLo", "qMonoUnbHi"]
if MN and all(m_ in MN["grids"] for m_ in ("10", "20", "40")):
    G20_ = MN["grids"]["20"]["instruments"]; ex2_ = lambda R__, k_, st_, w_="obs": R__[k_][st_][w_][0]
    pr_ = [np.exp(ex2_(G20_, k_, "placebo_lrr", "pred")) for k_ in CLS]; put("qMonoPredMin", f2(min(pr_))); put("qMonoPredMax", f2(max(pr_)))
    e_ = [ex2_(G20_, k_, "placebo_lrr", "excess") for k_ in CLS]; put("qMonoExcMin", sg(min(e_), 2)); put("qMonoExcMax", sg(max(e_), 2))
    assert CLS[int(np.argmin(e_))].startswith("sn_rob")                                         # the text names s-nlp's RoBERTa as the smallest
    rp_ = [np.exp(ex2_(G20_, k_, "lrr", "pred")) for k_ in CLS]; put("qMonoRRPredMin", f2(min(rp_))); put("qMonoRRPredMax", f2(max(rp_)))
    for nm_, m_ in (("Twenty", "20"), ("Ten", "10"), ("Forty", "40")):
        put(f"qMonoN{nm_}", str(MN["grids"][m_]["contagion"]["n_above"])); put(f"qMonoLLM{nm_}", str(MN["grids"][m_]["escalation"]["n_above"]))
    pa_ = [np.exp(MN["grids"][m_]["instruments"][k_]["placebo_lrr"]["pred"][0]) for m_ in ("10", "20", "40") for k_ in CLS]; put("qMonoPredAllMin", f2(min(pa_))); put("qMonoPredAllMax", f2(max(pa_)))
    ea_ = [MN["grids"][m_]["instruments"][k_]["placebo_lrr"]["excess"][0] for m_ in ("10", "20", "40") for k_ in CLS]; put("qMonoExcAllMin", sg(min(ea_), 2)); put("qMonoExcAllMax", sg(max(ea_), 2))
    put("qMonoHaikuPred", sg(100 * ex2_(G20_, "haiku", "placebo_slope", "pred"), 2)); put("qMonoHaikuObs", sg(100 * ex2_(G20_, "haiku", "placebo_slope"), 2))
    put("qMonoSonnetPred", sg(100 * ex2_(G20_, "sonnet", "placebo_slope", "pred"), 2)); put("qMonoSonnetObs", sg(100 * ex2_(G20_, "sonnet", "placebo_slope"), 2))
    put("qMonoHaikuExcLo", sg(100 * G20_["haiku"]["placebo_slope"]["excess"][1], 2)); put("qMonoHaikuExcHi", sg(100 * G20_["haiku"]["placebo_slope"]["excess"][2], 2))
    sh_ = [MN["grids"][m_]["instruments"][k_]["placebo_slope"]["pred"][0] / MN["grids"][m_]["instruments"][k_]["placebo_slope"]["obs"][0] for m_ in ("10", "20", "40") for k_ in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx")]
    put("qMonoShareMin", pct(max(0.0, min(sh_)))); put("qMonoShareMax", pct(max(sh_)))
    put("qMonoFourDetoxPred", f"{np.exp(ex2_(G20_, 'detoxify_original@0.6', 'lrr_44', 'pred')):.1f}"); put("qMonoHaikuLawPred", sg(100 * ex2_(G20_, "haiku", "depth_law", "pred"), 2))
    put("qMonoUnbExc", sg(ex2_(G20_, "detoxify_unbiased@m", "placebo_lrr", "excess"), 2))
    put("qMonoFortyHaikuObs", sg(100 * ex2_(G20_, "haiku", "slope_40"), 2)); put("qMonoFortyHaikuPred", sg(100 * ex2_(G20_, "haiku", "slope_40", "pred"), 2))
    f4_ = [100 * MN["grids"][m_]["instruments"]["haiku"]["slope_40"]["pred"][0] for m_ in ("10", "20", "40")]; put("qMonoFortyHaikuPredMin", sg(min(f4_), 2)); put("qMonoFortyHaikuPredMax", sg(max(f4_), 2))
    ra_ = [np.exp(MN["grids"][m_]["instruments"][k_]["lrr"]["pred"][0]) for m_ in ("10", "20", "40") for k_ in CLS]; put("qMonoRRPredAllMin", f2(min(ra_))); put("qMonoRRPredAllMax", f2(max(ra_)))
    ub_ = MN["grids"]["20"]["instruments"]["detoxify_unbiased@m"]["placebo_lrr"]["excess"]; put("qMonoUnbLo", sg(ub_[1], 2)); put("qMonoUnbHi", sg(ub_[2], 2))
    assert all(MN["grids"][m_]["instruments"][k_]["placebo_lrr"]["excess"][1] <= 0 for m_ in ("10", "20", "40") for k_ in ("detoxify_unbiased@m", "detoxify_unbiased@0.6"))   # text: not significant on any grid
    ff_ = [MN["grids"][m_]["instruments"][k_]["placebo_slope"]["excess"][1] for m_ in ("10", "20", "40") for k_ in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx")]; assert min(ff_) > 0
else:
    pend(*KEYS_MN)
# ------------------------------------------------------------ Addendum 85, B85a: the NPMLE and the f-MLE solved to convergence (replace the registered EM values;
# the registered EM values stay available as ...EM macros)
def over(k_, v_):
    assert k_ in M, k_
    M[k_] = v_
C85 = load(f"{W}/results/x85/x85_rrinf.json"); CG = {m_: load(f"{W}/results/x85/x85_construct_{m_}.json") for m_ in ("10", "20", "40")}
CG = {m_: v_["grids"][m_] for m_, v_ in CG.items() if v_}
if C85 and LT:
    put("qLatLogGapReg", sg(LT["H82b"]["gaps"]["gold"], 2)); put("qLatGapOne", sg(LT["H82b"]["gaps"]["k1"], 3)); put("qLatGapThree", sg(LT["H82b"]["gaps"]["k3"], 3)); put("qLatGapGold", sg(LT["H82b"]["gaps"]["gold"], 3)); put("qLatRREM", f2(LT["rr_inf"]["rr"][0])); put("qLatNBelowEM", M["qLatNBelow"]); put("qLatRRTenEM", M["qLatRRTen"]); put("qLatRRFortyEM", M["qLatRRForty"])
    over("qLatRR", f2(C85["rr_inf"]["rr"][0])); over("qLatRRLo", f2(C85["rr_inf"]["rr"][1])); over("qLatRRHi", f2(C85["rr_inf"]["rr"][2]))
    over("qLatRRTen", f2(C85["fits"]["10"]["rr_inf"])); over("qLatRRForty", f2(C85["fits"]["40"]["rr_inf"]))
    h_ = C85["H82a_converged"]; over("qLatNAbove", str(h_["n_above"])); over("qLatNBelow", str(h_["n_below"])); over("qLatNSame", str(12 - h_["n_above"] - h_["n_below"]))
    assert sorted(h_["above"]) == sorted(k_ for k_ in CLS12_ if LT["instruments"][k_]["minus_inf"][1] > 0)       # the same two settings lie above
    over("qLatAttLog", f"{abs(np.log(LT['gold']['rr'][0] / C85['rr_inf']['rr'][0])):.2f}")
    kk_ = [C85["fits"][m_]["kkt"] for m_ in ("10", "20", "40")] + [C85["rr_inf"]["max_resample_kkt"]]; assert max(kk_) <= 1e-8
    put("qLatRRGridMin", f2(min(C85["fits"][m_]["rr_inf"] for m_ in ("10", "20", "40")))); put("qLatRRGridMax", f2(max(C85["fits"][m_]["rr_inf"] for m_ in ("10", "20", "40"))))
else:
    pend("qLatLogGapReg", "qLatGapOne", "qLatGapThree", "qLatGapGold", "qLatRREM", "qLatNBelowEM", "qLatRRTenEM", "qLatRRFortyEM", "qLatRRGridMin", "qLatRRGridMax")
if "20" in CG:
    G_ = CG["20"]["instruments"]; ex3_ = lambda k_, st_, w_="obs": G_[k_][st_][w_][0]
    pr_ = [np.exp(ex3_(k_, "placebo_lrr", "pred")) for k_ in CLS]; over("qMonoPredMin", f2(min(pr_))); over("qMonoPredMax", f2(max(pr_)))
    e_ = [ex3_(k_, "placebo_lrr", "excess") for k_ in CLS]; over("qMonoExcMin", sg(min(e_), 2)); over("qMonoExcMax", sg(max(e_), 2))
    assert CLS[int(np.argmin(e_))].startswith("sn_rob")
    rp_ = [np.exp(ex3_(k_, "lrr", "pred")) for k_ in CLS]; over("qMonoRRPredMin", f2(min(rp_))); over("qMonoRRPredMax", f2(max(rp_)))
    over("qMonoHaikuPred", sg(100 * ex3_("haiku", "placebo_slope", "pred"), 2)); over("qMonoHaikuObs", sg(100 * ex3_("haiku", "placebo_slope"), 2))
    over("qMonoSonnetPred", sg(100 * ex3_("sonnet", "placebo_slope", "pred"), 2)); over("qMonoSonnetObs", sg(100 * ex3_("sonnet", "placebo_slope"), 2)); put("qMonoSonnetExc", sg(100 * ex3_("sonnet", "placebo_slope", "excess"), 2))
    over("qMonoHaikuExcLo", sg(100 * G_["haiku"]["placebo_slope"]["excess"][1], 2)); over("qMonoHaikuExcHi", sg(100 * G_["haiku"]["placebo_slope"]["excess"][2], 2))
    over("qMonoFourDetoxPred", f"{np.exp(ex3_('detoxify_original@0.6', 'lrr_44', 'pred')):.1f}"); over("qMonoHaikuLawPred", sg(100 * ex3_("haiku", "depth_law", "pred"), 2))
    over("qMonoUnbExc", sg(ex3_("detoxify_unbiased@m", "placebo_lrr", "excess"), 2))
    ub_ = G_["detoxify_unbiased@m"]["placebo_lrr"]["excess"]; over("qMonoUnbLo", sg(ub_[1], 2)); over("qMonoUnbHi", sg(ub_[2], 2))
    u6_ = G_["detoxify_unbiased@0.6"]["placebo_lrr"]["excess"]; put("qMonoUnbSixExc", sg(u6_[0], 2)); put("qMonoUnbSixLo", sg(u6_[1], 2)); put("qMonoUnbSixHi", sg(u6_[2], 2))
    over("qMonoFortyHaikuObs", sg(100 * ex3_("haiku", "slope_40"), 2)); over("qMonoFortyHaikuPred", sg(100 * ex3_("haiku", "slope_40", "pred"), 2))
    T_ = CG["20"]["thresholds"]["pi>=0.5"]; over("qConsThrPlac", f2(np.exp(T_["placebo_lrr"]["pred"][0]))); over("qConsThrPlacLo", f2(np.exp(T_["placebo_lrr"]["pred"][1])))
    over("qConsThrPlacHi", f2(np.exp(T_["placebo_lrr"]["pred"][2])))
    for nm_, m_ in (("Twenty", "20"), ("Ten", "10"), ("Forty", "40")):
        if m_ in CG:
            over(f"qMonoN{nm_}", str(CG[m_]["contagion"]["n_above"])); over(f"qMonoLLM{nm_}", str(CG[m_]["escalation"]["n_above"]))
    if all(m_ in CG for m_ in ("10", "20", "40")):
        assert all(CG[m_]["contagion"]["n_above"] == 12 and CG[m_]["escalation"]["n_above"] == 4 for m_ in CG)            # text: "12 of 12 ... 4 of 4 ... on each grid"
        assert all(MN["grids"][m_]["contagion"]["n_above"] == 12 and MN["grids"][m_]["escalation"]["n_above"] == 4 for m_ in ("10", "20", "40"))
        pa_ = [np.exp(CG[m_]["instruments"][k_]["placebo_lrr"]["pred"][0]) for m_ in CG for k_ in CLS]; over("qMonoPredAllMin", f2(min(pa_))); over("qMonoPredAllMax", f2(max(pa_)))
        ea_ = [CG[m_]["instruments"][k_]["placebo_lrr"]["excess"][0] for m_ in CG for k_ in CLS]; over("qMonoExcAllMin", sg(min(ea_), 2)); over("qMonoExcAllMax", sg(max(ea_), 2))
        sh_ = [CG[m_]["instruments"][k_]["placebo_slope"]["pred"][0] / CG[m_]["instruments"][k_]["placebo_slope"]["obs"][0] for m_ in CG for k_ in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx")]
        over("qMonoShareMin", pct(max(0.0, min(sh_)))); over("qMonoShareMax", pct(max(sh_)))
        f4_ = [100 * CG[m_]["instruments"]["haiku"]["slope_40"]["pred"][0] for m_ in CG]; over("qMonoFortyHaikuPredMin", sg(min(f4_), 2)); over("qMonoFortyHaikuPredMax", sg(max(f4_), 2))
        ra_ = [np.exp(CG[m_]["instruments"][k_]["lrr"]["pred"][0]) for m_ in CG for k_ in CLS]; over("qMonoRRPredAllMin", f2(min(ra_))); over("qMonoRRPredAllMax", f2(max(ra_)))
        ff_ = [CG[m_]["instruments"][k_]["placebo_slope"]["excess"][1] for m_ in CG for k_ in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx")]; assert min(ff_) > 0
        cc_ = [CG[m_]["instruments"][k_]["placebo_lrr"]["excess"][1] for m_ in CG for k_ in CLS]; assert min(cc_) > 0; put("qMonoExcLoMin", sg(min(cc_), 3))
        un_ = {m_: CG[m_]["instruments"]["detoxify_unbiased@m"]["placebo_lrr"]["excess"][1] > 0 for m_ in CG}; put("qMonoUnbNSig", str(sum(un_.values())))
        put("qMonoUnbTenExc", sg(CG["10"]["instruments"]["detoxify_unbiased@m"]["placebo_lrr"]["excess"][0], 2))
        put("qMonoUnbTenLo", sg(CG["10"]["instruments"]["detoxify_unbiased@m"]["placebo_lrr"]["excess"][1], 2)); put("qMonoUnbTenHi", sg(CG["10"]["instruments"]["detoxify_unbiased@m"]["placebo_lrr"]["excess"][2], 2))
    else:
        pend("qMonoExcLoMin", "qMonoUnbNSig", "qMonoUnbTenExc", "qMonoUnbTenLo", "qMonoUnbTenHi")
else:
    pend("qMonoSonnetExc", "qMonoUnbSixExc", "qMonoUnbSixLo", "qMonoUnbSixHi", "qMonoExcLoMin", "qMonoUnbNSig", "qMonoUnbTenExc", "qMonoUnbTenLo", "qMonoUnbTenHi")
# the protocol's misfit (Addendum 82 realism; verification E6) and the four-rater count (E10), over reply-edge children as the shares
if PRO:
    put("qLatTenSim", pct(PRO["realism"]["10"]["share_sim"])); put("qLatTenObs", pct(PRO["realism"]["10"]["share_obs"]))
    P_ = __import__("pandas").read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "tox", "parent_row"])
    mt_ = __import__("pandas").read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
    n_ = mt_.toxicity_annotator_count.reindex(P_.id.values).values.astype(np.int64); K_ = np.clip(np.rint(P_.tox.values * n_).astype(np.int64), 0, n_)
    ch_ = np.flatnonzero(P_.parent_row.values >= 0); ch_ = ch_[P_.parent_row.values[ch_] != ch_]
    four_ = n_[ch_] == 4; put("qFourNonUnan", str(int(np.sum(four_ & (K_[ch_] > 0) & (K_[ch_] < 4)))))
    assert abs(four_.mean() - PRO["realism"]["4"]["share_obs"]) < 1e-9
    put("qNEdgeChildren", format(len(ch_), ",").replace(",", "{,}"))
else:
    pend("qLatTenSim", "qLatTenObs", "qFourNonUnan", "qNEdgeChildren")
# the synthetic check of the monotone test (x83d_synth.py; verification E7)
SDB = load(f"{W}/results/x83/synthd_binary.json"); SDC = load(f"{W}/results/x83/synthd_cat.json")
if SDB and SDC:
    for nm_, e_ in (("SydHz", SDB["instruments"]["h0"]["placebo_lrr"]["excess"]), ("SydTop", SDB["instruments"]["clustered"]["placebo_lrr"]["excess"]),
                    ("SydCatHz", SDC["instruments"]["h0"]["placebo_lrr"]["excess"]), ("SydCatTop", SDC["instruments"]["clustered"]["placebo_lrr"]["excess"]),
                    ("SydCatDrift", SDC["instruments"]["depthy"]["placebo_lrr"]["excess"])):
        put(f"q{nm_}", sg(e_[0], 2)); put(f"q{nm_}Lo", sg(e_[1], 2)); put(f"q{nm_}Hi", sg(e_[2], 2))
    d_ = SDB["instruments"]["depthy"]["placebo_slope"]["excess"]; put("qSydDepth", sg(100 * d_[0], 2)); put("qSydDepthLo", sg(100 * d_[1], 2))
    s4_ = SDB["instruments_m40"]["sonnet"]["placebo_slope"]["excess"]; put("qSydSfForty", sg(100 * s4_[0], 2)); put("qSydSfFortyLo", sg(100 * s4_[1], 2)); put("qSydSfFortyHi", sg(100 * s4_[2], 2))
    assert SDB["instruments"]["h0"]["placebo_lrr"]["excess"][1] < 0 < SDB["instruments"]["h0"]["placebo_lrr"]["excess"][2] and s4_[1] > 0
else:
    pend("qSydHz")
# the LLMs' S_FP outside news comments on Gab (verification O7)
DSg = load(f"{W}/results/x82/x82_desc.json")
if DSg:
    gb_ = DSg["B82b"]["qian_gab"]; put("qOthGabN", str(len(gb_))); put("qOthGabPos", str(sum(v_["fp_term"] > 0 for v_ in gb_.values())))
    assert M["qOthGabN"] == M["qOthGabPos"]                                               # the text says "all ... on Gab"
else:
    pend("qOthGabN", "qOthGabPos")
# ------------------------------------------------------------ Addendum 85, B85b: a human reference (single rater; decisions at about 20,000 comments)
HR = load(f"{W}/results/x85/x85_human_ref.json")
if HR:
    p1_, p2_ = HR["part1"], HR["part2"]
    put("qHrSrOff", pct(p1_["single_rater"]["share_rel_gt_25"])); put("qHrSrOffMed", pct(p1_["single_rater"]["median_rel"]))
    put("qHrSrUnifDepth", pct(p1_["single_rater_uniform_share"]["depth"]["median_mass"])); put("qHrSrUnifRR", pct(p1_["single_rater_uniform_share"]["log_rr"]["median_mass"]))
    put("qHrLLMUnifDepth", pct(p1_["instrument_uniform_share"]["depth"]["median_mass_llm"])); put("qHrLLMUnifSize", pct(p1_["instrument_uniform_share"]["size"]["median_mass_llm"]))
    put("qHrLLMOffMed", pct(p1_["instruments_vs_goldB"]["llm"]["median_rel"]))
    put("qHrN", str(p2_["n_subsamples"])); put("qHrSize", format(int(round(p2_["size_median"], -3)), ",").replace(",", "{,}"))
    I2_ = p2_["instruments"]; s2_ = p2_["single_rater_vs_goldB"]
    put("qHrSrDepth", pct(s2_["depth"]["differ"])); put("qHrSrSize", pct(s2_["size"]["differ"])); assert s2_["depth"]["reversal"] == 0 and s2_["size"]["reversal"] == 0
    put("qHrHaikuDepth", pct(I2_["haiku"]["depth"]["differ"])); put("qHrHaikuSize", pct(I2_["haiku"]["size"]["differ"])); put("qHrHaikuRev", pct(I2_["haiku"]["depth"]["reversal"]))
    put("qHrHaikuCtxDepth", pct(I2_["haiku_ctx"]["depth"]["differ"], 1 if I2_["haiku_ctx"]["depth"]["differ"] > 0.99 else 0))
    hk_ = p2_["pooled"]["kappa_ge_06"]; put("qHrHiKapDepth", pct(hk_["depth"])); put("qHrHiKapSize", pct(hk_["size"])); put("qHrHiKapN", str(hk_["n"]))
    cd_ = [I2_[k_]["depth"]["differ"] for k_ in CLS]; put("qHrClsDepthMin", pct(min(cd_))); put("qHrClsDepthMax", pct(max(cd_)))
    put("qHrGoldDepthNeg", pct(p2_["gold_verdicts"]["depth"]["-1"])); put("qHrGoldSizePos", pct(p2_["gold_verdicts"]["size"]["1"]))
    put("qHrRRSame", str(sum(I2_[k_]["log_rr"]["differ"] == 0 for k_ in I2_))); put("qHrRRN", str(len(I2_)))
else:
    pend("qHrSrOff")
# ------------------------------------------------------------ Addendum 85, B85c': the raters' subtype votes (public vote fractions)
FR = load(f"{W}/results/x85/x85_construct_frac.json")
if FR:
    s_, l_ = FR["slopes"], FR["levels"]
    put("qFrMatch", pct(FR["match_rate_neg"], 1)); put("qFrNone", pct(FR["share_none_among_neg"]))
    for nm_, k_ in (("Ins", "insult_any"), ("Haiku", "haiku"), ("HaikuNone", "haiku_zero_votes"), ("HaikuCtxNone", "haiku_ctx_zero_votes"), ("Rej", "rejected"), ("Ida", "identity_attack_any")):
        put(f"qFr{nm_}", sg(s_[k_][0], 2)); put(f"qFr{nm_}Lo", sg(s_[k_][1], 2)); put(f"qFr{nm_}Hi", sg(s_[k_][2], 2))
    put("qFrInsLevel", pct(l_["insult_any"][0] / 100)); put("qFrHaikuNoneLevel", pct(l_["haiku_zero_votes"][0] / 100, 1)); put("qFrRejLevel", pct(l_["rejected"][0] / 100, 1))
    put("qFrInsFlag", pct(FR["insult_by_haiku"]["any_flagged"])); put("qFrInsNoFlag", pct(FR["insult_by_haiku"]["any_not_flagged"]))
else:
    pend("qFrMatch")
# ------------------------------------------------------------ Addendum 85, B85d: coverage at small budgets and PPI++'s minimum
SBG = load(f"{W}/results/x85/x85_small_budgets.json")
if SBG:
    it_ = SBG["items"]; a_ = lambda d_, B_: it_[d_][B_]["all"]
    put("qSbTsCov", f"{a_('pps_seq4_ba', '250')['mean']:.3f}"); put("qSbTsMin", f"{min(a_('pps_seq4_ba', B_)['min'] for B_ in ('250', '500')):.3f}")
    assert all(a_("pps_seq4_ba", B_)["share_lt_090"] == 0 for B_ in ("250", "500"))
    for nm_, d_ in (("Ppi", "ppi"), ("Unif", "srs"), ("Dsl", "dsl"), ("Strat", "upgrade_lw"), ("Ai", "ai")):
        put(f"qSb{nm_}Cov", f"{a_(d_, '250')['mean']:.3f}"); put(f"qSb{nm_}Min", f"{a_(d_, '250')['min']:.3f}"); put(f"qSb{nm_}Below", pct(a_(d_, "250")["share_lt_090"]))
    put("qSbN", str(a_("pps_seq4_ba", "250")["n"])); put("qSbPpiMinPct", pct(a_("ppi", "250")["min"]))
    e5_ = SBG["edges"]["edge_m3_c15"]; put("qSbEdgeMin", f"{min(e5_[B_]['all']['min'] for B_ in ('500', '1000')):.3f}")
    u_ = SBG["units"]["ts_rank"]["250"]["share"]; put("qSbUnitCov", f"{u_['mean']:.3f}"); put("qSbUnitMin", f"{u_['min']:.3f}")
    pm_ = SBG["ppi_min"]["items"]; put("qPpiNBelow", str(pm_["n_lt_090"])); put("qPpiN", str(pm_["n"]))
    lo_ = pm_["lowest"][0]; put("qPpiMinTs", f"{lo_['threadsafe']:.3f}"); assert lo_["coverage"] < 0.70
    put("qPpiSizeShare", pct((pm_["by_metric"].get("article_size_law", 0) + pm_["by_metric"].get("placebo_size", 0) + pm_["by_metric"].get("size_law", 0)) / pm_["n_lt_090"]))
else:
    pend("qSbTsCov")
# ------------------------------------------------------------ Addendum 86, B86a: contagion within articles (Mantel-Haenszel) and within threads
WA = load(f"{W}/results/x86/x86_within.json")
if WA:
    A_ = WA["strata"]["article"]; T_ = WA["strata"]["thread"]; ia_ = A_["instruments"]
    put("qWaGold", f2(A_["gold"]["rr"][0])); put("qWaGoldLo", f2(A_["gold"]["rr"][1])); put("qWaGoldHi", f2(A_["gold"]["rr"][2]))
    st_ = [ia_[k_]["fp"] + ia_[k_]["tp"] for k_ in CLS]; un_ = [ia_[k_]["uniform"] for k_ in CLS]
    put("qWaStructMin", sg(min(st_), 2)); put("qWaStructMax", sg(max(st_), 2)); put("qWaUniMin", sg(max(un_), 2)); put("qWaUniMax", sg(min(un_), 2))
    put("qWaNAbove", str(A_["n_cls_above_gold"])); put("qWaNBelow", str(A_["n_cls_below_gold"])); put("qWaNStruct", str(A_["n_cls_structured_pos"]))
    pl_ = [ia_[k_]["placebo_rr"] for k_ in CLS]; put("qWaPlacMin", f2(min(pl_))); put("qWaPlacMax", f2(max(pl_))); put("qWaSrPlac", f2(ia_["single_rater"]["placebo_rr"]))
    rr_ = [ia_[k_]["rr"][0] for k_ in CLS]; put("qWaClsMin", f2(min(rr_))); put("qWaClsMax", f2(max(rr_)))
    put("qWaHaiku", f2(ia_["haiku"]["rr"][0])); put("qWaSr", f2(ia_["single_rater"]["rr"][0]))
    put("qWtGold", f2(T_["gold"]["rr"][0])); put("qWtSrPlac", f2(T_["instruments"]["single_rater"]["placebo_rr"]))
    put("qWaHpass", "PASS" if WA["H86a"]["pass_"] else "FAIL")
else:
    pend("qWaGold", "qWaGoldLo", "qWaGoldHi", "qWaStructMin", "qWaStructMax", "qWaUniMin", "qWaUniMax", "qWaNAbove", "qWaNBelow", "qWaNStruct", "qWaPlacMin", "qWaPlacMax",
         "qWaSrPlac", "qWaClsMin", "qWaClsMax", "qWaHaiku", "qWaSr", "qWtGold", "qWtSrPlac", "qWaHpass")
# ------------------------------------------------------------ Addendum 86, B86b: decisions at 10,000-100,000 comments (with B85b's 20,000)
DCS = load(f"{W}/results/x86/x86_decisions.json")
if DCS and HR:
    allR_ = [DCS["sizes"][k_] for k_ in ("10000", "50000", "100000")] + [HR["part2"]]
    sd_ = [R_["single_rater_vs_goldB"]["depth"]["differ"] for R_ in allR_]; hd_ = [R_["instruments"]["haiku"]["depth"]["differ"] for R_ in allR_]
    assert all(R_["single_rater_vs_goldB"]["depth"]["reversal"] == 0 for R_ in allR_)
    ph_ = lambda x_: f"{int(np.floor(100 * x_ + 0.5 + 1e-9))}\\%"                                     # round half up (36.5% -> 37%)
    put("qDcSrMin", ph_(min(sd_))); put("qDcSrMax", ph_(max(sd_))); put("qDcHaikuMin", ph_(min(hd_))); put("qDcHaikuMax", ph_(max(hd_)))
    put("qDcHaikuRevMax", pct(max(R_["instruments"]["haiku"]["depth"]["reversal"] for R_ in allR_)))
    put("qDcSizeMin", "10{,}000"); put("qDcSizeMax", "100{,}000")
else:
    pend("qDcSrMin", "qDcSrMax", "qDcHaikuMin", "qDcHaikuMax", "qDcHaikuRevMax", "qDcSizeMin", "qDcSizeMax")
# ------------------------------------------------------------ Addendum 87: within-article nulls (B87a), the paired gap (B87b), approved comments (B87c)
WN = load(f"{W}/results/x87/x87_within_null.json"); GP = load(f"{W}/results/x87/x87_gap.json")
if WN and GP:
    g_ = WN["gold"]; I_ = WN["instruments"]
    put("qWnNull", f2(g_["null_rr"][0])); put("qWnExc", sg(g_["excess"][0], 2)); put("qWnBetween", pct(g_["between_share"]))
    put("qWnAbove", str(WN["summary"]["n_above"])); put("qWnBelow", str(WN["summary"]["n_below"]))
    pe_ = [I_[k_]["placebo_excess"][0] for k_ in CLS]; assert all(I_[k_]["placebo_excess"][1] > 0 for k_ in CLS)
    put("qWnPlacMin", sg(min(pe_), 2)); put("qWnPlacMax", sg(max(pe_), 2)); put("qWnSrPlac", sg(I_["single_rater"]["placebo_excess"][0], 2))
    assert min(pe_) > I_["single_rater"]["placebo_excess"][2]
    b_ = GP["B87b"]["gap"]; put("qGapLog", sg(b_[0], 2)); put("qGapLo", sg(b_[1], 2)); put("qGapHi", sg(b_[2], 2))
    a_ = GP["B87c"]["haiku"]["slope_approved"]; put("qApHaiku", sg(a_[0], 2)); put("qApHaikuLo", sg(a_[1], 2)); put("qApHaikuHi", sg(a_[2], 2))
else:
    pend("qWnNull", "qWnExc", "qWnBetween", "qWnAbove", "qWnBelow", "qWnPlacMin", "qWnPlacMax", "qWnSrPlac", "qGapLog", "qGapLo", "qGapHi", "qApHaiku", "qApHaikuLo", "qApHaikuHi")
if "40" in CG:
    put("qMonoSonnetExcForty", sg(100 * CG["40"]["instruments"]["sonnet"]["placebo_slope"]["excess"][0], 2))
else:
    pend("qMonoSonnetExcForty")
# ------------------------------------------------------------ Addendum 86, B86c: fresh-classifier cells against the others
FC = load(f"{W}/results/x86/x86_fresh.json")
if FC:
    put("qFcCov", f"{FC['fresh']['laws']['cov_mean']:.3f}"); put("qFcMin", f"{FC['fresh']['laws']['cov_min']:.3f}")
    put("qFcPPI", f2(FC["fresh"]["laws"]["ratio"]["ppi"])); put("qOcPPI", f2(FC["others"]["laws"]["ratio"]["ppi"]))
    put("qFcPlacPPI", f2(FC["fresh"]["placebo"]["ratio"]["ppi"])); put("qOcPlacPPI", f2(FC["others"]["placebo"]["ratio"]["ppi"]))
    put("qFcAI", f2(FC["fresh"]["laws"]["ratio"]["ai"])); put("qOcAI", f2(FC["others"]["laws"]["ratio"]["ai"]))
else:
    pend("qFcCov", "qFcMin", "qFcPPI", "qOcPPI", "qFcPlacPPI", "qOcPlacPPI", "qFcAI", "qOcAI")
# ------------------------------------------------------------ Table: laws on the other corpora (gold and the range over instruments)
NM = {"cad": "CAD (Reddit, abuse)", "qian_reddit": "Qian et al., Reddit", "qian_gab": "Qian et al., Gab", "cga": "CGA-Wiki", "bad": "BAD (chatbot dialogue)"}
def rng_(v, d=2):
    return f"{sg(min(v), d)} to {sg(max(v), d)}"
rows_ = []; rows_c = []
NMC = {"cad": "CAD", "qian_reddit": "Qian, Reddit", "qian_gab": "Qian, Gab", "cga": "CGA-Wiki", "bad": "BAD"}
for pop in ("cad", "qian_reddit", "qian_gab", "cga", "bad"):
    r_ = DEC[pop]; g_ = r_["gold"]; Ls = {k: v for k, v in r_["labelers"].items() if k != "human_lenient"}
    gw = Wn[pop]["gold_within"][0]; wl = [gw + Wn[pop]["labelers"][k]["distortion"][0] for k in Ls if k in Wn[pop]["labelers"]]
    pl = [g_["depth"][0] + v["laws"]["depth"]["distortion"][0] for v in Ls.values()]; sz = [g_["size"][0] + v["laws"]["size"]["distortion"][0] for v in Ls.values()]
    if "log_rr" in g_:
        rr = [float(np.exp(g_["log_rr"][0] + v["laws"]["log_rr"]["distortion"][0])) for v in Ls.values() if "log_rr" in v["laws"]]
        rrs = f"{np.exp(g_['log_rr'][0]):.2f} & {min(rr):.2f} to {max(rr):.2f}"
    else:
        rrs = "-- & --"
    rows_.append(f"{NM[pop]} & {format(r_['N'], ',').replace(',', '{,}')} & {100 * r_['prev']:.0f}\\% & {sg(gw * PP, 2)} & {rng_([v * PP for v in wl])} & {sg(g_['depth'][0] * PP, 2)} & {rng_([v * PP for v in pl])} & "
                 f"{sg(g_['size'][0] * PP, 1)} & {rng_([v * PP for v in sz], 1)} & {rrs} & {len(Ls)} \\\\")
    # Addendum 79 (page limit): a compact version for the main text (within-thread escalation and contagion), the full one for the appendix
    nab = sum(float(np.exp(g_["log_rr"][0] + v["laws"]["log_rr"]["distortion"][0])) > np.exp(g_["log_rr"][0]) for v in Ls.values() if "log_rr" in v["laws"]) if "log_rr" in g_ else None
    rows_c.append(f"{NMC[pop]} & {format(r_['N'], ',').replace(',', '{,}')} & {sg(gw * PP, 2)} & {rng_([v * PP for v in wl])} & " + (f"{np.exp(g_['log_rr'][0]):.2f} & {min(rr):.2f}--{max(rr):.2f} & {nab}/{len(rr)}" if "log_rr" in g_ else "-- & -- & --") + " \\\\")
open(f"{OUTD}/tables/corpora.tex", "w").write("\n".join(rows_) + "\n"); open(f"{OUTD}/tables/corpora_c.tex", "w").write("\n".join(rows_c) + "\n")
with open(f"{OUTD}/numbers_v4b.tex", "w") as f:
    f.write("% generated by scripts/x17_numbers_v4b.py\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros ->", f"{OUTD}/numbers_v4b.tex")
for k, v in M.items():
    print(f"  {k} = {v}")
