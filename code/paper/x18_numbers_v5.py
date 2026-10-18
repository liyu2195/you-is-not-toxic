"""Numbers of the v5 draft ("After Perspective") that are not in numbers_v2-v4b: Addenda 88-89 and B84 results, computed from the
stored JSON files. Writes paper_v5/numbers_v5.tex. Macros whose results are not in yet are defined empty, so the draft compiles.
usage: python3 x18_numbers_v5.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import os, json, numpy as np, pandas as pd
W = _ROOT; R = f"{W}/results"; OUT = f"{W}/paper_arxiv/numbers_v5.tex"
M = {}
def sg(x, d=2):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"                                                      # no signed zero
    return s.replace("-", "$-$") if s.startswith("-") else s
def f(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s
def pct(x, d=0):
    v = round(100 * x + 1e-9, d)
    return (f"{v:.{d}f}" if d else f"{int(round(100 * x + 1e-9))}") + "\\%"
def load(p):
    p = f"{R}/{p}"
    return json.load(open(p)) if os.path.exists(p) else None

# ---------------------------------------------------------------- B88d noise-free escalation
nf = load("x88/x88_noisefree_depth.json")
M["cNfDepth"], M["cNfDepthLo"], M["cNfDepthHi"] = sg(nf["noise_free"][0]), sg(nf["noise_free"][1]), sg(nf["noise_free"][2])
M["cRatersSlope"] = sg(nf["raters"][0])
# ---------------------------------------------------------------- B88b reruns
rr = load("x88/x88_rerun.json")
M["cRerunHaiku"] = sg(rr["haiku"]["depth_new"][0]); M["cRerunSonnet"] = sg(rr["sonnet"]["depth_new"][0])
M["cRerunFlagChg"] = pct(rr["haiku"]["flag_changed"], 1)
M["cRerunDiff"], M["cRerunDiffLo"], M["cRerunDiffHi"] = (sg(v) for v in rr["haiku"]["depth_new_minus_old"])
M["cRerunKappaH"] = f(rr["haiku"]["kappa_between_runs"]); M["cRerunKappaS"] = f(rr["sonnet"]["kappa_between_runs"])
# ---------------------------------------------------------------- B88a calibrated thresholds
ca = load("x88/x88_calibrated.json")["pops"]
H = ca["cc_threads"]["instruments"]["haiku:level"]["calibrated"]
nat = sum(H[f"kappa@{n}"]["cut_counts"].get("2", 0) for n in (200, 500, 1000)) / 1500
M["cCalNative"] = pct(nat); M["cCalNativeFive"] = pct(H["kappa@500"]["cut_counts"].get("2", 0) / 500)
M["cCalNativeThousand"] = pct(H["kappa@1000"]["cut_counts"].get("2", 0) / 500, 1); M["cCalFOne"] = pct(H["f1@500"]["cut_counts"].get("2", 0) / 500)
assert all(H[f"{r}@{n}"]["share_excess_sig_pos"] == 1.0 for r in ("kappa", "f1") for n in (200, 500, 1000))
S5 = ca["cc_s5"]["instruments"]
M["cRateHaikuEx"] = sg(S5["haiku:rating"]["calibrated"]["kappa@500"]["excess_median"]); M["cRateSonnetEx"] = sg(S5["sonnet:rating"]["calibrated"]["kappa@500"]["excess_median"])
HP = ca["cc_threads"]["instruments"]["haiku:level"]["calibrated"]; SP_ = S5["sonnet:level"]["calibrated"]
M["cCalPrevStrict"] = pct(min(HP[f"prev@{n_}"]["cut_counts"].get("3", 0) / 500 for n_ in (200, 500, 1000)) - 0.0005, 1)
M["cCalPrevHaiku"] = sg(HP["prev@500"]["excess_median"]); M["cCalPrevSonnet"] = sg(SP_["prev@500"]["excess_median"])
assert all(HP[f"prev@{n_}"]["share_excess_sig_pos"] == 1.0 for n_ in (200, 500, 1000)) and max(SP_[f"prev@{n_}"]["share_excess_sig_pos"] for n_ in (200, 500, 1000)) < 0.01
M["cRateSigMin"] = pct(min(S5[f"{m_}:rating"]["calibrated"][f"kappa@{n_}"]["share_excess_sig_pos"] for m_ in ("haiku", "sonnet") for n_ in (200, 500, 1000)) - 0.0005, 1)
# ---------------------------------------------------------------- B88c many raters
mr = load("x88/x88_manyraters.json")["instruments"]
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
CLS14m = CLS12 + ["detoxify_unbiased@0.6", "detoxify_unbiased@m"]  # all 14 classifier settings
v20 = [mr[k]["20"]["rr_parent_flag"] for k in CLS14m]; v10 = [mr[k]["10"]["rr_parent_flag"] for k in CLS14m]
M["cMrClsMin"] = f(min(v[0] for v in v20)); M["cMrClsMax"] = f(max(v[0] for v in v20)); M["cMrClsN"] = str(sum(v[1] > 1 for v in v20))
M["cMrTenMin"] = f(min(v[0] for v in v10)); M["cMrTenMax"] = f(max(v[0] for v in v10)); assert all(v[1] > 1 for v in v10)
sr = mr["single_rater"]["20"]["rr_parent_flag"]; M["cMrSr"], M["cMrSrLo"], M["cMrSrHi"] = f(sr[0]), f(sr[1]), f(sr[2])
M["cMrCtxHaiku"] = f(mr["haiku_ctx"]["20"]["rr_parent_gold"][0]); M["cMrCtxQwen"] = f(mr["qwen7b_m_ctx"]["20"]["rr_parent_gold"][0])
# ---------------------------------------------------------------- decompositions: classifiers' laws, Sonnet shown the parent
dec = load("x74_decomp/x74_decomp.json")
CLS14 = CLS12 + ["detoxify_unbiased@0.6", "detoxify_unbiased@m"]
L = dec["cc_threads"]["labelers"]
dep = [100 * L[k]["laws"]["depth"]["law"][0] for k in CLS14]
M["vClsDepthMin"] = sg(min(dep)); M["vClsDepthMax"] = sg(max(dep))
sz = [L[k]["laws"][law]["law"][0] for k in CLS14 for law in ("size", "article_size")]
M["vClsSizeSign"] = str(sum(v > 0 for v in sz)); M["vClsSizeN"] = str(len(sz))
M["cSonnetCtxDepth"] = sg(100 * dec["cc_s5"]["labelers"]["sonnet_ctx"]["laws"]["depth"]["law"][0])
# ---------------------------------------------------------------- unparseable answers of the stored LLM labels
mx = 0.0
for pop, cols in (("cc_threads", ["s:haiku", "s:haiku_ctx"]), ("cc_s5", ["s:sonnet", "s:sonnet_ctx", "s:haiku", "s:haiku_ctx"])):
    P = pd.read_parquet(f"{R}/x14_pops/{pop}.parquet", columns=cols)
    for c in cols:
        mx = max(mx, float(P[c].isna().mean()))
M["vUnparsedMax"] = "0.1\\%" if mx < 0.001 else pct(mx + 0.0005, 1)
import glob as _g
um = 0.0; ue = 0
for d_ in _g.glob(f"{R}/x89/labels/*_ctx*/"):
    a_ = [json.loads(l).get("ans") for f_ in _g.glob(d_ + "shard_*.jsonl") for l in open(f_)]
    um = max(um, sum(v is None for v in a_) / max(len(a_), 1))
M["cUnparsedOther"] = pct(um, 1)
# ---------------------------------------------------------------- B88h contrast effect, B88i single fold by budget
bd = load("x88/x88_breakdown.json")
Hc = bd["B88h"]
M["cCtrHaikuTT"] = pct(Hc["haiku_ctx"]["child1_parent1"]); M["cCtrHaikuTC"] = pct(Hc["haiku_ctx"]["child1_parent0"])
M["cCtrQwenTT"] = f(Hc["qwen7b_m_ctx@own"]["child1_parent1"]); M["cCtrQwenTC"] = f(Hc["qwen7b_m_ctx@own"]["child1_parent0"])
Bi = bd["B88i"]
M["cOneFoldSmall"] = f(Bi["250"]["laws"]["median_ratio"]); M["cOneFoldLarge"] = f(Bi["8000"]["laws"]["median_ratio"]); M["cOneFoldThousand"] = f(Bi["1000"]["laws"]["median_ratio"])
M["vAblOneSmall"] = (f" (on laws, \\cOneFoldLarge{{}} times at 8{{,}}000 labels and \\cOneFoldSmall{{}} at 250, where learning across folds matters most)")
# ---------------------------------------------------------------- the 'you' gap on Civil Comments (x89_other_tox.py reference rows), B77d size share
cr = load("x89/x89_cc_reference.json")
if cr:
    M["cYouGapHaiku"] = f(100 * cr["cc_threads"]["instruments"]["haiku_tox"]["fpr_you_minus_other"][0], 1)
    M["cYouGapSonnet"] = f(100 * cr["cc_s5"]["instruments"]["sonnet_tox"]["fpr_you_minus_other"][0], 1)
    M["cYouGapDetox"] = f(100 * cr["cc_threads"]["instruments"]["ref:detoxify_original@m"]["fpr_you_minus_other"][0], 1)
    hz = cr["cc_threads"]["instruments"]["haiku_tox"]; M["cPlacHaiku"] = sg(hz["placebo"][0]); M["cPlacHaikuYou"] = sg(hz["placebo_within_you"][0])
    M["cPlacYouShare"] = pct(1 - hz["placebo_within_you"][0] / hz["placebo"][0])
    hc = cr["cc_threads"]["instruments"]["haiku_tox_ctx"]; M["cHaikuCtxReplies"] = sg(hc["depth_replies"][0]); M["cGoldReplies"] = sg(cr["cc_threads"]["gold"]["depth_replies"][0])
    M["cHaikuReplies"] = sg(hz["depth_replies"][0])
    M["cSonnetCtxReplies"] = sg(cr["cc_s5"]["instruments"]["sonnet_tox_ctx"]["depth_replies"][0])
    Pq = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["depth", "z", "s:qwen7b_m_ctx"]); zq = Pq.z.values; Nq = len(zq)
    sq = Pq["s:qwen7b_m_ctx"].values.astype(float); sq = np.where(np.isnan(sq), -np.inf, sq); kq = int(round(zq.mean() * Nq))
    oq = np.lexsort((np.random.default_rng(7650).random(Nq), -sq)); fq = np.zeros(Nq); fq[oq[:kq]] = 1.0     # as x76_rethreshold.py
    xq = np.minimum(Pq.depth.values, 10).astype(float); rq = xq >= 1; xr = xq[rq] - xq[rq].mean()
    M["cQwenCtxReplies"] = sg(100 * float((xr * (fq[rq] - fq[rq].mean())).sum() / (xr * xr).sum()))
yf = load("x77_youfree/x77_youfree.json"); M["cYfSizeShare"] = pct(yf["laws"]["size"]["share_removed"])
# ---------------------------------------------------------------- B89a: the bias on the other platforms
ot = load("x89/x89_other_tox.json")
if ot:
    FOR = ("cad", "qian_reddit", "qian_gab", "cga"); gh = {c: 100 * ot[c]["instruments"]["haiku_tox"]["fpr_you_minus_other"][0] for c in FOR}
    M["cOthGapMin"] = f(min(gh.values()), 0); M["cOthGapMax"] = f(max(gh.values()), 0)
    gd = {c: 100 * ot[c]["instruments"]["ref:detoxify_original@m"]["fpr_you_minus_other"][0] for c in FOR}; assert all(gd[c] < gh[c] for c in FOR)
    H_ = lambda c, k="haiku_tox": ot[c]["instruments"][k]
    M["cQrPlac"] = sg(H_("qian_reddit")["placebo"][0]); M["cQrPlacYou"] = sg(H_("qian_reddit")["placebo_within_you"][0])
    M["cGabPlac"] = sg(H_("qian_gab")["placebo"][0]); M["cGabPlacYou"] = sg(H_("qian_gab")["placebo_within_you"][0])
    M["cCgaPlac"] = sg(H_("cga")["placebo"][0]); M["cCadPlac"] = sg(H_("cad")["placebo"][0])
    M["cGabGold"] = sg(ot["qian_gab"]["gold"]["depth"][0]); M["cGabHaiku"] = sg(H_("qian_gab")["depth"][0]); M["cGabGoldAbs"] = f(abs(ot["qian_gab"]["gold"]["depth"][0]))
    gs_ = {c: 100 * ot[c]["instruments"]["sonnet_tox"]["fpr_you_minus_other"][0] for c in FOR}; M["cOthGapSonMin"] = f(min(gs_.values()), 0); M["cOthGapSonMax"] = f(max(gs_.values()), 0)
    M["cQrRepGold"] = sg(ot["qian_reddit"]["gold"]["depth_replies"][0]); M["cQrRepHaiku"] = sg(H_("qian_reddit", "haiku_tox_ctx")["depth_replies"][0])
    M["cBadYou"] = pct(ot["bad"]["you_share_neg"])
    M["vAbsPlatforms"] = "appears on Reddit, Gab and Wikipedia talk pages too"
    M["vIntroGeneral"] = (" Asked the same question on Reddit, Gab and Wikipedia talk pages, Haiku flags non-toxic comments that address someone "
                          "\\cOthGapMin--\\cOthGapMax{} points more often than others (Sonnet \\cOthGapSonMin--\\cOthGapSonMax), and where address grows with depth, on Qian et al.'s "
                          "Reddit and Gab, so do its false positives. "
                          "On the news comments, every LLM we tested, open models included, flags non-toxic comments more often the more second-person words they contain.")
# ---------------------------------------------------------------- B89c: the rewrite with Sonnet and with the parent
rw = load("x89/x89_rewrite.json")
if rw:
    sn = rw["sonnet_noctx"]; M["cRwSonShare"] = pct(sn["share_removed"]); M["cRwSonSize"] = pct(sn["size_share_removed"])
    M["cRwSonOrig"] = sg(sn["dist_orig"][0]); M["cRwSonEdit"] = sg(sn["dist_edit"][0])
    txt = f"; for Sonnet the same rewrite removes \\cRwSonShare{{}} of the escalation distortion and \\cRwSonSize{{}} of the inflated size law"
    if rw.get("_complete"):
        txt += (", and among replies shown their parent it still lowers both models' false escalation (Haiku's by \\cRwCtxHaiku{} and Sonnet's by "
                "\\cRwCtxSonnet{} points per level, CIs below 0)")
    M["vRewriteMore"] = txt
# ---------------------------------------------------------------- B89d: the rewrite on the other platforms
ro = load("x89/x89_rewrite_other.json")
if ro:
    dd = [-100 * ro[c]["all"]["diff"][0] for c in ("cad", "qian_reddit", "qian_gab", "cga")]; assert all(ro[c]["all"]["diff"][2] < 0 for c in ("cad", "qian_reddit", "qian_gab", "cga"))
    M["cRoMin"] = f(min(dd), 1); M["cRoMax"] = f(max(dd), 1)
# ---------------------------------------------------------------- Addendum 90: the dose of second-person address (results/x90/x90_dose.json)
do = load("x90/x90_dose.json")
if do:
    C = do["corpora"]; ct = C["cc_threads"]; L_ = ct["labelers"]; s5 = C["cc_s5"]; L5 = s5["labelers"]
    hf = [t[0] for t in L_["haiku"]["fpr_by_count"]]
    for nm_, j in (("Zero", 0), ("One", 1), ("Two", 2), ("Three", 3), ("Four", 4)):
        M[f"cDoseHaiku{nm_}"] = pct(hf[j])
    sf = [t[0] for t in L5["sonnet"]["fpr_by_count"]]; M["cDoseSonZero"] = pct(sf[0]); M["cDoseSonFour"] = pct(sf[4])
    df_ = [t[0] for t in L_["detoxify"]["fpr_by_count"]]; M["cDoseDetZero"] = pct(df_[0], 1); M["cDoseDetFour"] = pct(df_[4], 1)
    nv_ = [t[0] for t in ct["no_votes"]["haiku"]]; M["cDoseNoneZero"] = pct(nv_[0]); M["cDoseNoneFour"] = pct(nv_[4])
    sr_ = [t[0] for t in ct["single_rater"]["fpr_by_count"]]; M["cDoseSrZero"] = pct(sr_[0], 1); M["cDoseSrFour"] = pct(sr_[4], 1)
    M["cDoseSrZeroR"] = pct(sr_[0]); M["cDoseSrFourR"] = pct(sr_[4])                     # rounded, for the abstract
    lb_ = ct["length_bands"]["haiku"]; assert all(all(b[k] < b[k + 1] for k in range(4) if b[k] is not None and b[k + 1] is not None) for b in lb_)   # rises in every band
    mc = ct["mean_count_by_depth"]; M["cDoseCountZero"] = f(mc[0], 1); M["cDoseCountEight"] = f(mc[8], 1)
    w_ = L_["haiku"]["within_count"]; M["cDoseWithinHaiku"], M["cDoseWithinHaikuLo"], M["cDoseWithinHaikuHi"] = sg(w_[0]), sg(w_[1]), sg(w_[2])
    assert abs(L_["haiku"]["placebo"][0] - float(M["cPlacHaiku"].replace("$-$", "-"))) < 0.006               # the same placebo slope as Table 3
    M["cDoseLawHaiku"], M["cDoseLawHaikuFix"] = sg(L_["haiku"]["law_distortion"][0]), sg(L_["haiku"]["law_distortion_within_count"][0])
    M["cDoseLawSon"], M["cDoseLawSonFix"] = sg(L5["sonnet"]["law_distortion"][0]), sg(L5["sonnet"]["law_distortion_within_count"][0])
    assert L_["haiku"]["law_distortion_within_count"][1] < 0 < L_["haiku"]["law_distortion_within_count"][2]      # Haiku's excess vanishes (CI covers 0)
    M["cDoseSonPlac"] = sg(L5["sonnet"]["placebo"][0]); M["cDoseWithinSon"] = sg(L5["sonnet"]["within_count"][0])
    M["cDoseWithinSonLo"], M["cDoseWithinSonHi"] = sg(L5["sonnet"]["within_count"][1]), sg(L5["sonnet"]["within_count"][2])
    M["cDoseCtxPlac"] = sg(L_["haiku_ctx"]["placebo"][0]); M["cDoseCtxWithin"] = sg(L_["haiku_ctx"]["within_count2"][0])
    M["cDoseQwenPlac"] = sg(L_["qwen_ctx"]["placebo"][0]); M["cDoseQwenWithin"] = sg(L_["qwen_ctx"]["within_count2"][0])
    rb = s5["rewrite_by_count"]; assert all(rb[t][k]["change"][2] < 0 for t in rb for k in rb[t])
    M["cRwDoseHOne"] = f(-100 * rb["haiku"]["1"]["change"][0], 0); M["cRwDoseHThree"] = f(-100 * rb["haiku"]["3"]["change"][0], 0)
    _h34 = sorted(round(-100 * rb["haiku"][k_]["change"][0]) for k_ in ("3", "4")); M["cRwDoseHThreeFour"] = f"{_h34[0]}--{_h34[1]}" if _h34[0] != _h34[1] else f"{_h34[0]}"
    M["cRwDoseSOne"] = f(-100 * rb["sonnet"]["1"]["change"][0], 0); M["cRwDoseSFour"] = f(-100 * rb["sonnet"]["4"]["change"][0], 0)
    mt = s5["matched"]; rl = {k_: v["ratio_4_0"][0] for k_, v in mt.items() if v["kind"] == "llm"}; rc = {k_: v["ratio_4_0"][0] for k_, v in mt.items() if v["kind"] == "classifier"}
    M["cModRatioMin"], M["cModRatioMax"] = f(min(rl.values()), 1), f(max(rl.values()), 1); M["cClsRatioMin"], M["cClsRatioMax"] = f(min(rc.values()), 1), f(max(rc.values()), 1)
    assert max(rc, key=rc.get) == "detoxify_original@m" and min(rl.values()) > max(rc.values())
    M["cModFlag"] = pct(np.mean([v["flag_rate"] for v in mt.values()]), 1)
    rat = []
    for c_ in ("cc_threads", "cad", "qian_reddit", "qian_gab", "cga", "bad"):
        h_ = C[c_]["labelers"]["haiku"]; fr_ = [t[0] for t in h_["fpr_by_count"]]; n_ = h_["n_by_count"]
        rat.append(((fr_[3] * n_[3] + fr_[4] * n_[4]) / (n_[3] + n_[4])) / fr_[0])
    M["cOthDoseMin"], M["cOthDoseMax"] = f(min(rat), 1), f(max(rat), 0)
    qr_ = C["qian_reddit"]["labelers"]["haiku"]; M["cQrPlacCount"] = sg(qr_["within_count"][0])
    gb_ = C["qian_gab"]["labelers"]["haiku"]; M["cGabPlacCount"] = sg(gb_["within_count"][0])
    M["cCadPlacCount"] = sg(C["cad"]["labelers"]["haiku"]["within_count"][0]); assert C["cad"]["labelers"]["haiku"]["within_count"][2] < 0
    # v5d+: how much of each forum's law distortion the number of second-person words carries; Wikipedia's and BAD's other mechanisms
    cg_ = C["cga"]["labelers"]["haiku"]; M["cCgaPlacCount"] = sg(cg_["within_count"][0]); assert cg_["within_count"][1] > 0 and C["cga"]["you_share_by_depth"][0] > C["cga"]["you_share_by_depth"][-1]
    for c_, nm_ in (("qian_reddit", "Qr"), ("qian_gab", "Gab")):
        l_ = C[c_]["labelers"]["haiku"]; M[f"c{nm_}LawEx"], M[f"c{nm_}LawExFix"] = sg(l_["law_distortion"][0]), sg(l_["law_distortion_within_count"][0])
        M[f"c{nm_}LawShare"] = pct(1 - l_["law_distortion_within_count"][0] / l_["law_distortion"][0])
    bd_ = ot["bad"]["instruments"]["haiku_tox"]; M["cBadGold"], M["cBadHaiku"] = sg(ot["bad"]["gold"]["depth"][0]), sg(bd_["depth"][0])
    assert bd_["placebo"][1] < 0 < bd_["placebo"][2] and bd_["terms"]["uniform"] < 0 and bd_["terms"]["S_TP"] < 0 and abs(bd_["terms"]["S_FP"]) < 0.05
    M["vNModelsDose"] = {5: "five", 6: "six", 7: "seven"}[len(rl)]
    NM = {"haiku": "Haiku", "sonnet": "Sonnet", "qwen": "Qwen", "gemma": "Gemma", "olmo2": "OLMo-2", "granite": "Granite", "llama31": "Llama"}
    nm = [NM[k_] for k_ in ("haiku", "sonnet", "qwen", "gemma", "olmo2", "granite", "llama31") if k_ in rl]
    M["vDoseModels"] = ", ".join(nm[:-1]) + " and " + nm[-1]
    op = [k_ for k_ in ("qwen", "gemma", "olmo2", "granite", "llama31") if k_ in rl]                 # open models: shallower gradients, falling at a fixed number
    assert all(L5[k_]["within_count"][2] < 0 for k_ in op)
    on = [{"qwen": "Qwen without the parent", "gemma": "Gemma", "olmo2": "OLMo-2", "granite": "Granite", "llama31": "Llama"}[k_] for k_ in op]
    M["vLimitLLMs"] = ", ".join(on[:-1]) + " and " + on[-1]
    ex_ = {k_: L5[k_]["law_distortion"][0] for k_ in op}; M["cOpenExMax"] = sg(max(ex_.values()))
    assert all(L5[k_]["law_distortion_within_count"][0] <= 0.005 for k_ in op)                      # none remains at a fixed number
    yb = C["bad"]["you_share_by_depth"]; M["cBadYouMin"], M["cBadYouMax"] = pct(min(yb)), pct(max(yb))
    fam = load("b84/x84_families.json")
    if fam and all(k_ in fam["settings"] for k_ in ("olmo2_ctx", "granite_ctx", "llama31_ctx")):
        kc = ("olmo2_ctx", "granite_ctx", "llama31_ctx"); ex_r = {k_: fam["settings"][k_]["depth_replies_excess"] for k_ in kc}
        rise = [k_ for k_ in kc if ex_r[k_][1] > 0]; none_ = [k_ for k_ in kc if ex_r[k_][1] <= 0 <= ex_r[k_][2]]; assert rise == ["llama31_ctx"] and len(rise) + len(none_) == 3, ex_r
        M["vLimitCtx"] = ""                                                                    # v5c: replaced by the replies-only maximum below
        M["cOpenCtxExMax"] = sg(max(ex_r[k_][0] for k_ in kc))
        M["cLlamaCtxEx"], M["cLlamaCtxExLo"], M["cLlamaCtxExHi"] = (sg(u) for u in ex_r["llama31_ctx"])
        rr_ = [float(np.exp(fam["settings"][k_]["log_rr"][0])) for k_ in kc]; M["cCtxRROpenMin"], M["cCtxRROpenMax"] = f(min(rr_)), f(max(rr_))
        M["cGoldRRSfiveFam"] = f(float(np.exp(fam["gold"]["log_rr"][0]))); M["cGoldRepliesSfive"] = sg(fam["settings"]["olmo2"]["depth_replies_gold"][0])
        rr0 = {k_: [float(np.exp(u)) for u in fam["settings"][k_]["log_rr"]] for k_ in ("olmo2", "granite", "llama31")}
        assert all(v[1] <= float(np.exp(fam["gold"]["log_rr"][0])) <= v[2] for v in rr0.values())   # without the parent, at gold prevalence: within noise of gold's
# ---------------------------------------------------------------- B88f': fresh gold (results/x88/x88_fresh_gold.json, merged from parts)
fg = load("x88/x88_fresh_gold.json")
if fg:
    bp = fg["B88f_prime"]; assert bp["n_below_090_rv"] == 0
    M["cFgN"] = str(bp["n"]); M["cFgMean"] = pct(bp["mean_rv"], 1); M["cFgMin"] = pct(bp["min_rv"], 1)
    M["cFgReal"] = pct(bp["mean_realized"], 1); M["cFgRealMin"] = pct(bp["min_realized"], 1)
    laws_ = ["depth_law", "size_law", "article_size_law"]; cells_ = fg["cells"]
    wmed = lambda pr: float(np.median([cells_[l][B][pr][m]["width_rv_vs_fixed"] for l in cells_ for B in ("1000", "4000") for m in laws_]))
    M["cFgWOne"], M["cFgWThree"], M["cFgWAdapt"] = f(wmed("one")), f(wmed("three")), f(wmed("adaptive"))
    ra_ = [cells_[l][B]["_raters_adaptive"] for l in cells_ for B in ("1000", "4000")]; M["cFgRaters"] = f(float(np.mean(ra_)), 1)
    M["cFgNoRvMin"] = pct(fg["H88f"]["min"], 0)
    M["vFreshGoldSentence"] = ("Gold need not be fixed: when each purchased label is a fresh draw from the comment's raters (one rater, a majority of three, "
                               "or three raters and two more if they disagree), on seven cells of the 5\\% sample at 1{,}000 and 4{,}000 labels, intervals with Theorem~\\ref{thm:valid}(c)'s rater variance "
                               "cover the law of the expected labels in \\cFgMean{} of audits (minimum \\cFgMin; "
                               "without it, as low as \\cFgNoRvMin), and the adaptive panel widens them by a median factor of only \\cFgWAdapt.")
# ---------------------------------------------------------------- B89c (ii): the rewrite shown the parent
if rw and rw.get("_complete"):
    M["cRwCtxHaiku"] = f(-rw["haiku_ctx"]["change_v1_replies"][0]); M["cRwCtxSonnet"] = f(-rw["sonnet_ctx"]["change_v1_replies"][0])
    assert rw["haiku_ctx"]["change_v1_replies"][2] < 0 and rw["sonnet_ctx"]["change_v1_replies"][2] < 0
    M["cRwCtxShareH"] = pct(rw["haiku_ctx"]["share_removed_v2_replies"]); M["cRwCtxShareS"] = pct(rw["sonnet_ctx"]["share_removed_v2_replies"])
# ---------------------------------------------------------------- B89b: a tuned Haiku (results/x89/x89_tuned.json)
tu = load("x89/x89_tuned.json")
if tu and len(tu["candidates"]) == 4 and all(v["eval_missing"] < 0.002 * 60668 for v in tu["candidates"].values()):
    sel = tu["selected"]; v_ = tu["candidates"][sel]
    DESC = {"T0": "zero-shot", "T8": "eight examples", "T16": "sixteen examples", "TR": "extended thinking"}
    M["cTunedSel"] = DESC[sel]; M["cTunedKappa"] = f(v_["dev_kappa"][str(v_["cut"])] if str(v_["cut"]) in v_["dev_kappa"] else v_["dev_kappa"][v_["cut"]])
    M["cTunedDist"], M["cTunedDistLo"], M["cTunedDistHi"] = (sg(t) for t in v_["distortion"])
    gaps = [100 * c["fpr_you_minus_other"][0] for c in tu["candidates"].values()]; M["cTunedGapMin"], M["cTunedGapMax"] = f(min(gaps), 0), f(max(gaps), 0)
    dists = [c["distortion"][1] for c in tu["candidates"].values()]
    M["cTunedThink"] = sg(tu["candidates"]["TR"]["distortion"][0]); assert all(c["distortion"][1] > 0 for c in tu["candidates"].values())
    M["vTunedSentence"] = (" Tuning and instruction make it worse. Of the main prompt with 0, 8 or 16 gold-labeled examples or with extended thinking, the one that agrees best with "
                           "2{,}000 gold labels from other articles (eight examples) exceeds gold's law by \\cTunedDist{} points per level (CI \\cTunedDistLo{} to \\cTunedDistHi), "
                           "\\cTunedChg{} more than the main prompt rerun alongside it.")
    M["cTunedLaw"] = sg(v_["depth"][0]); M["cThinkLaw"] = sg(tu["candidates"]["TR"]["depth"][0]); assert sel == "T8"
    M["cTunedAllPos"] = str(sum(d > 0 for d in dists))
# ---------------------------------------------------------------- the reverse edit (B74, results/x74_edit/x74_edit.json) as percentages
ed = load("x74_edit/x74_edit.json")
if ed:
    M["cRevHaikuBefore"], M["cRevHaikuAfter"] = pct(ed["rev"]["neg_deep"]["flag_E0"]), pct(ed["rev"]["neg_deep"]["flag_E2"])
# ---------------------------------------------------------------- v6: shares of gold negatives that say "you" on Civil Comments threads (Table 2's row)
ccr = load("x89/x89_cc_reference.json")
if ccr:
    M["cYouRootNeg"], M["cYouDeepNeg"] = pct(ccr["cc_threads"]["you_root"]), pct(ccr["cc_threads"]["you_deep"])
# ---------------------------------------------------------------- B91: can a prompt remove the bias? (results/x91/x91_mitigate.json)
mi = load("x91/x91_mitigate.json")
if mi and all(mi["instruments"][t_]["missing"] < 0.002 * mi["N"] for t_ in ("M1", "M2", "T0")):
    I_ = mi["instruments"]
    for t_, nm_ in (("M1", "One"), ("M2", "Two"), ("T0", "Zero")):
        d_ = I_[t_]["distortion"]; M[f"cMit{nm_}Dist"], M[f"cMit{nm_}DistLo"], M[f"cMit{nm_}DistHi"] = (sg(u) for u in d_)
        fr_ = I_[t_]["fpr_by_count"]; M[f"cMit{nm_}FprZero"], M[f"cMit{nm_}FprFour"] = pct(fr_[0][0]), pct(fr_[4][0])
        M[f"cMit{nm_}Ratio"] = f(I_[t_]["ratio_4_0"][0], 1); M[f"cMit{nm_}Fix"] = sg(I_[t_]["distortion_fixed"][0])
        M[f"cMit{nm_}Law"] = sg(I_[t_]["depth"][0])
    for t_, nm_ in (("M1", "One"), ("M2", "Two")):
        c_ = I_[t_]["change_vs_T0"]; M[f"cMit{nm_}Chg"], M[f"cMit{nm_}ChgLo"], M[f"cMit{nm_}ChgHi"] = (sg(u) for u in c_)
    assert mi["H91a"] and mi["H91b"] and I_["M1"]["change_vs_T0"][1] > 0 and I_["M2"]["change_vs_T0"][1] < 0 < I_["M2"]["change_vs_T0"][2]
    M["vMitigSentence"] = (" Told that addressing someone is not by itself toxic, Haiku's excess grows by \\cMitOneChg{} (CI \\cMitOneChgLo{} to \\cMitOneChgHi), and "
                           "third-person instructions with a note that any ``you'' addresses another reader change it by \\cMitTwoChg{} (CI \\cMitTwoChgLo{} to \\cMitTwoChgHi). "
                           "Each of these prompts keeps the over-flagging of ``you'' (\\cAltRatioMin--\\cAltRatioMax{} times the false-positive rate with four or more second-person words "
                           "as with none); those that worsen escalation also flag top-level comments less, relative to replies, and \\cAltFixMin--\\cAltFixMax{} of their excess "
                           "survives at a fixed number of them.")
    alt = ("M1", "M2", "T8", "T16", "TR"); rt_ = [I_[t_]["ratio_4_0"][0] for t_ in alt]; M["cAltRatioMin"], M["cAltRatioMax"] = f(min(rt_), 1), f(max(rt_), 1)
    wo = [t_ for t_ in alt if I_[t_]["change_vs_T0"][1] > 0]; assert sorted(wo) == sorted(["M1", "T8", "T16", "TR"]), wo   # the prompts that worsen escalation
    sh_ = [I_[t_]["distortion_fixed"][0] / I_[t_]["distortion"][0] for t_ in wo]; M["cAltFixMin"], M["cAltFixMax"] = pct(min(sh_)), pct(max(sh_))
    assert all(I_[t_]["change_fpr_root_vs_T0"][0] < I_[t_]["change_fpr_replies_vs_T0"][0] for t_ in wo)                  # fewer flags at the root, relative to replies
    assert all(I_[t_]["distortion_fixed"][1] > 0 for t_ in wo)
    c_ = I_["T8"]["change_vs_T0"]; M["cTunedChg"], M["cTunedChgLo"], M["cTunedChgHi"] = (sg(u) for u in c_); assert tu["selected"] == "T8"
    print("B91", {k_: M[k_] for k_ in M if k_.startswith("cMit")})
# ---------------------------------------------------------------- B92: matched operating points, strict level, agreement, construct controls
mt2 = load("x92/x92_matched.json")
if mt2:
    rows = {(r["model"], r["op"]): r for r in mt2["rows"]}; g5 = mt2["gold"]["depth"][0]
    gp = {m_: rows[(m_, "rating at gold prevalence")] for m_ in ("haiku", "sonnet")} | {m_: rows[(m_, "gold prevalence")] for m_ in ("qwen", "gemma", "olmo2", "granite", "llama") if (m_, "gold prevalence") in rows}
    M["cMatHaiku"], M["cMatSonnet"] = sg(gp["haiku"]["depth_excess"][0]), sg(gp["sonnet"]["depth_excess"][0])
    opn = [m_ for m_ in ("qwen", "olmo2", "granite", "llama") if m_ in gp and gp[m_]["depth_excess"][1] > 0]
    M["cMatOpenMin"], M["cMatOpenMax"] = sg(min(gp[m_]["depth_excess"][0] for m_ in opn)), sg(max(gp[m_]["depth_excess"][0] for m_ in opn))
    NMw = {"qwen": "Qwen", "olmo2": "OLMo-2", "granite": "Granite", "llama": "Llama"}; nl_ = [NMw[m_] for m_ in opn]
    M["vMatOpen"] = ", ".join(nl_[:-1]) + " and " + nl_[-1]
    M["cMatGemma"] = sg(gp["gemma"]["depth_excess"][0]); assert gp["gemma"]["depth_excess"][1] < 0 < gp["gemma"]["depth_excess"][2]
    M["cMatGemmaAbs"] = f(abs(gp["gemma"]["depth_excess"][0])); assert gp["gemma"]["depth_excess"][0] < 0
    # v6, Table 2's threshold rows as laws on the 5% sample (gold's law plus each excess)
    for m_, nm_ in (("haiku", "H"), ("sonnet", "S")):
        M[f"cThrPrevLaw{nm_}"] = sg(g5 + rows[(m_, "rating at gold prevalence")]["depth_excess"][0]); M[f"cThrVtLaw{nm_}"] = sg(g5 + rows[(m_, "very toxic")]["depth_excess"][0])
    M["cThrKmaxLawH"] = sg(g5 + S5["haiku:rating"]["calibrated"]["kappa@500"]["excess_median"]); M["cThrKmaxLawS"] = sg(g5 + S5["sonnet:rating"]["calibrated"]["kappa@500"]["excess_median"])
    km_ = [rows[(m_, "max kappa")]["depth_excess"][0] for m_ in ("qwen", "gemma", "olmo2", "granite", "llama") if (m_, "max kappa") in rows]
    M["cMatOpenKmaxMin"], M["cMatOpenKmaxMax"] = sg(min(km_)), sg(max(km_))
    th_ = [gp[m_]["depth_excess"][0] / rows[(m_, "native")]["depth_excess"][0] for m_ in ("haiku", "sonnet")]; assert all(0.28 < t_ < 0.38 for t_ in th_), th_   # "about a third"
    M["cMatSonnetRR"], M["cMatSonnetRRLo"], M["cMatSonnetRRHi"] = (sg(u) for u in gp["sonnet"]["log_rr_excess"]); assert gp["sonnet"]["log_rr_excess"][1] > 0
    npos = sum(v["depth_excess"][1] > 0 for v in gp.values()); W_ = {4: "four", 5: "five", 6: "six", 7: "seven"}
    M["vMatCount"], M["vMatTotal"] = W_[npos], W_[len(gp)]
    fixed_sig = [m_ for m_, v in gp.items() if v["depth_excess_fixed"][1] > 0]; assert fixed_sig == ["sonnet"], fixed_sig
    M["cMatSonnetFix"] = sg(gp["sonnet"]["depth_excess_fixed"][0])
    M["cKapHaikuSfive"] = f(rows[("haiku", "native")]["kappa"]); M["cKapGemmaSfive"] = f(gp["gemma"]["kappa"])
    M["cLawHaikuSfive"] = sg(g5 + rows[("haiku", "native")]["depth_excess"][0]); M["cLawGemmaSfive"] = sg(g5 + gp["gemma"]["depth_excess"][0])
ex2 = load("x92/x92_extra.json")
if ex2:
    vt = ex2["cc_very_toxic"]; M["cStrictFlag"] = pct(vt["flag_rate"], 1); M["cStrictUni"] = sg(vt["uniform"]); M["cStrictFP"] = "0.00" if abs(vt["S_FP"]) < 0.005 else sg(vt["S_FP"])
    gaps_ = [100 * v["gap"][0] for c_ in ("cad", "qian_reddit", "qian_gab", "cga") for v in ex2[c_].values()]
    assert all(v["gap"][1] > 0 for c_ in ("cad", "qian_reddit", "qian_gab", "cga") for v in ex2[c_].values())
    M["cOwnGapMin"], M["cOwnGapMax"] = f(min(gaps_), 1), f(max(gaps_), 1)
cn2 = load("x92/x92_construct.json")
if cn2:
    r3 = [cn2[c_]["below_median"]["haiku"]["ratio3"][0] for c_ in ("cad", "qian_reddit", "qian_gab", "cga")]
    assert all(cn2[c_]["below_median"]["haiku"]["gap"][1] > 0 for c_ in ("cad", "qian_reddit", "qian_gab", "cga"))
    M["cDetoxCtrlMin"], M["cDetoxCtrlMax"] = f(min(r3), 1), f(max(r3), 0)
    g3 = [100 * cn2[c_]["below_median"]["haiku"]["gap"][0] for c_ in ("cad", "qian_reddit", "qian_gab", "cga")]; M["cDetoxGapMin"], M["cDetoxGapMax"] = f(min(g3), 1), f(max(g3), 0)
# ---------------------------------------------------------------- the other corpora's settings (appendix)
if ot:
    M["vAppendixGeneral"] = ("\\paragraph{Other corpora.} Haiku and Sonnet answer the Civil Comments prompt verbatim on every comment of CAD, Qian et al.'s Reddit and Gab, "
                             "CGA-Wiki and BAD. Shown the parent, a reply takes its label with the parent and roots keep their label without it (as do replies whose "
                             "parent has no text); the depth law among replies uses depth $\\ge 1$. The rewrite labels every gold-negative comment of the four forums "
                             "that the detector matches, original and edited, shuffled together, and the rewrite with the parent shown labels every Civil Comments "
                             "reply of the 5\\% sample whose text or parent the detector matches: as written, with the reply rewritten, and with both rewritten.")
# ---------------------------------------------------------------- pending results (filled when their JSON files exist)
M.setdefault("vAbsPlatforms", "appears on Reddit and Wikipedia talk pages too"); M.setdefault("vLimitLLMs", "Qwen without the parent and Gemma"); M.setdefault("vLimitCtx", ""); M.setdefault("vMitigSentence", "")
for k in ("vModelsSentence", "vTunedSentence", "vAbsGeneral", "vIntroRewrite", "vIntroGeneral", "vGeneralParagraph", "vRewriteMore", "vFreshGoldSentence", "vAblOneSmall", "vAppendixGeneral"):
    M.setdefault(k, "")
_hasL = bool(do and "llama31" in do["corpora"]["cc_s5"]["labelers"])
M["vNLLMWords"] = "seven" if _hasL else ("six" if (do and "granite" in do["corpora"]["cc_s5"]["labelers"]) else "five")
M["vOpenLLMs"] = (("Qwen2.5-7B-Instruct~\\cite{qwen2024}, Gemma~3~27B~\\cite{gemma2025}, OLMo-2-1124-7B-Instruct~\\cite{olmo2024}, "
                   "Granite-3.1-8B-Instruct~\\cite{granite2024} and Llama-3.1-8B-Instruct~\\cite{llama2024}") if _hasL else
                  ("Qwen2.5-7B-Instruct~\\cite{qwen2024}, Gemma~3~27B~\\cite{gemma2025}, OLMo-2-1124-7B-Instruct~\\cite{olmo2024} and "
                   "Granite-3.1-8B-Instruct~\\cite{granite2024} (the last three on the 5\\% sample)"))
M["vOpenLLMSettings"] = ("Qwen is read with Hugging Face Transformers and the other open models with vLLM, each as the probability of answering ``toxic'' or ``very toxic'' "
                         "at the first answer position, renormalized over the four options, and flagged at gold's flag rate; shown the parent, they get their own threshold. "
                         "Gemma~3 runs with 4-bit weights.")
with open(OUT, "w") as fo:
    fo.write("% generated by scripts/x18_numbers_v5.py\n")
    for k, v in M.items():
        fo.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros ->", OUT)
for k, v in M.items():
    print(k, "=", v[:100])
