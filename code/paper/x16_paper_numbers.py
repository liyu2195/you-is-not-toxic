"""Paper v4: the new number macros (paper_v4/numbers_v4.tex) and tables, from the stored Addendum 74 results.
Sources: results/x74_decomp/x74_decomp.json (exact decomposition, kappa), results/x74_crowd/x74_crowd.json (second crowd,
k-rater gold), results/x74_ccc/x74_ccc.json (in-context gold), results/x74_bounds/x74_bounds.json (sharp agreement bounds),
results/x74_placebo/x74_placebo.json (placebo diagnostics with the fresh classifiers), results/x74_edit/x74_edit.json (edits),
results/x74_eval/ (Part B). Macros already defined for v3 (numbers_v3.tex, numbers_v2.tex, numbers_fig.tex) are reused.
Pending sources leave their macros as '??' so that the draft compiles.
usage: python3 x16_paper_numbers.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, re, glob
import numpy as np
W = _ROOT; OUTD = f"{W}/paper_arxiv"; os.makedirs(f"{OUTD}/tables", exist_ok=True)
M = {}


def put(k, v):
    assert re.fullmatch(r"[A-Za-z]+", k), k
    assert k not in M, k
    M[k] = v


def sg(x, d): return (f"+{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}")
def f2(x): return f"{x:.2f}"
def pct(x, d=0): return f"{100 * x:.{d}f}\\%"
def comma(n): return f"{n:,}".replace(",", "{,}")
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


PP = 100.0                                                                  # slopes in percentage points
# ---------------------------------------------------------------- Section 2: sharp agreement bounds (Proposition 2)
B = load(f"{W}/results/x74_bounds/x74_bounds.json")
bc = B["cc_threads"]; k9 = bc["by_kappa"]["0.9"]["depth"]
put("bKnineDepthLo", sg(k9[0] * PP, 2)); put("bKnineDepthHi", sg(k9[1] * PP, 2)); put("gDepthK", sg(bc["gold"]["depth"] * PP, 2))
hb = bc["labelers"]["haiku"]["depth"]; put("bHaikuGoldLo", sg(hb["gold_lo"] * PP, 2)); put("bHaikuGoldHi", sg(hb["gold_hi"] * PP, 2))
k5 = bc["by_kappa"]["0.5"]["depth"]; put("bKfiveDepthLo", sg(k5[0] * PP, 2)); put("bKfiveDepthHi", sg(k5[1] * PP, 2))
# ---------------------------------------------------------------- Section 3: decomposition on Civil Comments
D = load(f"{W}/results/x74_decomp/x74_decomp.json")
cc = D["cc_threads"]; s5 = D["cc_s5"]; Lc = cc["labelers"]; L5 = s5["labelers"]
g = cc["gold"]
put("wGoldDepthPP", sg(g["depth"][0] * PP, 2)); put("wGoldDepthPPLo", sg(g["depth"][1] * PP, 2)); put("wGoldDepthPPHi", sg(g["depth"][2] * PP, 2))
put("wGoldSizePP", sg(g["size"][0] * PP, 2)); put("wGoldArtPP", sg(g["article_size"][0] * PP, 2))
put("wGoldDepthSfivePP", sg(s5["gold"]["depth"][0] * PP, 2)); put("wHaikuDepthPP", sg(Lc["haiku"]["laws"]["depth"]["law"][0] * PP, 2)); put("wHaikuCtxDepthPP", sg(Lc["haiku_ctx"]["laws"]["depth"]["law"][0] * PP, 2))
MANU = [(Lc, "haiku"), (Lc, "haiku_ctx"), (Lc, "qwen7b_m_ctx"), (L5, "sonnet"), (L5, "sonnet_ctx")]
sh = [L_[k]["laws"]["depth"]["fp_term"][0] / L_[k]["laws"]["depth"]["distortion"][0] for L_, k in MANU]
# Addendum 79: wFPShareDepth* and wFPShareSize* used Qwen shown the parent at the no-context threshold; the paper uses
# x17's qFPShareDepth* (own threshold). They are no longer written.
CLS = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
miss = [Lc[k]["laws"]["log_rr"]["miss_part"][0] for k in CLS]; fpp = [Lc[k]["laws"]["log_rr"]["fp_part"][0] for k in CLS]
put("wClsMissMin", sg(min(miss), 2)); put("wClsMissMax", sg(max(miss), 2)); put("wClsFPMax", sg(max(fpp), 2))
put("wNClsMissPos", str(sum(m_ > 0 for m_ in miss))); put("wNClsFPNeg", str(sum(f_ < 0 for f_ in fpp)))
ctx_fp = [Lc[k]["laws"]["log_rr"]["fp_part"][0] for k in ("haiku_ctx", "qwen7b_m_ctx")]
put("wCtxFPMin", sg(min(ctx_fp), 2)); put("wCtxFPMax", sg(max(ctx_fp), 2))
put("wQwenRRFP", sg(Lc["qwen7b_m"]["laws"]["log_rr"]["fp_part"][0], 2)); put("wQwenRRMiss", sg(Lc["qwen7b_m"]["laws"]["log_rr"]["miss_part"][0], 2))
# classifiers' child TPR after a toxic vs a civil parent (Detoxify at 0.6), from the population
import pandas as pd
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["z", "parent_row", "y:detoxify_original@0.6"])
z = P.z.values; chi = np.flatnonzero(P.parent_row.values >= 0); par = P.parent_row.values[chi]; y = P["y:detoxify_original@0.6"].values
zc, zp, yc = z[chi], z[par], y[chi]
put("wDetoxTPRpos", f2(yc[(zc == 1) & (zp == 1)].mean())); put("wDetoxTPRneg", f2(yc[(zc == 1) & (zp == 0)].mean()))
# other corpora: how much of gold's escalation the instruments flatten
def flat(pop):
    r = D[pop]; gl = r["gold"]["depth"][0]
    v = [-(x["laws"]["depth"]["distortion"][0]) / gl for k, x in r["labelers"].items() if k != "human_lenient" and not k.endswith("@0.5")]
    return min(v), max(v)
qr, qg = flat("qian_reddit"), flat("qian_gab")
put("wQianRedFlatMin", f"{100 * qr[0]:.0f}"); put("wQianRedFlatMax", f"{100 * qr[1]:.0f}"); put("wGabFlatMin", f"{100 * qg[0]:.0f}"); put("wGabFlatMax", f"{100 * qg[1]:.0f}")
nrev = sum(1 for pop in ("qian_reddit", "qian_gab") for k, x in D[pop]["labelers"].items() if k != "human_lenient" and not k.endswith("@0.5")
           and -(x["laws"]["depth"]["distortion"][0]) / D[pop]["gold"]["depth"][0] > 1); put("wNHateReversed", str(nrev))
c_ = flat("cga"); put("wCgaFlatMin", f"{100 * c_[0]:.0f}"); put("wCgaFlatMax", f"{100 * c_[1]:.0f}")
# agreement vs distortion (Figure 2a; the registered H74f analysis over A1's settings, results/x74_eval/eval_A.json)
EA0 = load(f"{W}/results/x74_eval/eval_A.json") or {}
if "H74f" in EA0:
    f_ = EA0["H74f"]; put("wKappaRho", sg(f_["spearman_all"], 2)); put("wKappaRhoLo", sg(f_["spearman_all_ci"][0], 2)); put("wKappaRhoHi", sg(f_["spearman_all_ci"][1], 2))
    put("wNKappaSettings", str(f_["n_settings"])); put("wKappaTopZ", f"{max(f_[l]['top_max_z'] for l in ('depth', 'size', 'log_rr')):.0f}")
else:
    for k in ("wKappaRho", "wKappaRhoLo", "wKappaRhoHi", "wNKappaSettings", "wKappaTopZ"):
        put(k, "??")
# ---------------------------------------------------------------- second crowd and k-rater gold
C = load(f"{W}/results/x74_crowd/x74_crowd.json")
S = C["splits"]
med = lambda f: float(np.median([f(s) for s in S]))
cd = med(lambda s: abs(s["instr"]["crowd_A"]["depth_minus_goldB"])); hd = med(lambda s: abs(s["instr"]["haiku"]["depth_minus_goldB"]))
put("wCrowdDepthDist", f"{cd * PP:.3f}"); put("wCrowdDepthRatio", f"{hd / cd:.0f}")
put("wCrowdSizeDist", f"{med(lambda s: abs(s['instr']['crowd_A']['size_minus_goldB'])) * PP:.2f}")
v_ = med(lambda s: s["instr"]["crowd_A"]["placebo_depth_rel"]); put("wCrowdPDR", (sg(100 * v_, 1)) + "\\%")
put("wCrowdPRR", f2(np.exp(med(lambda s: s["instr"]["crowd_A"]["log_placebo_rr"]))))
put("wCrowdKappa", f2(med(lambda s: s["kappa_A_B"])))
put("wCrowdRRDist", f2(med(lambda s: abs(s["instr"]["crowd_A"]["log_rr_minus_goldB"]))))
put("wNSplits", str(len(S)))
for kk, nm in (("1", "One"), ("3", "Three"), ("5", "Five")):
    put(f"wK{nm}RR", f2(np.exp(np.median([r["log_rr"] for r in C["k_rater"][kk]]))))
# ---------------------------------------------------------------- in-context gold (CCC)
X = load(f"{W}/results/x74_ccc/x74_ccc.json")["estimates"]
put("wCccHumIC", sg(X["slope_Y_hic"][0] * PP, 2)); put("wCccHumICLo", sg(X["slope_Y_hic"][1] * PP, 2)); put("wCccHumICHi", sg(X["slope_Y_hic"][2] * PP, 2))
put("wCccHaikuIC", sg(X["slope_Y_haiku_ic"][0] * PP, 2)); put("wCccHaikuOC", sg(X["slope_Y_haiku_oc"][0] * PP, 2)); put("wCccQwenIC", sg(X["slope_Y_qwen_ic"][0] * PP, 2))
# ---------------------------------------------------------------- pending: placebo diagnostics with the fresh classifiers, edits, Part B
PL = load(f"{W}/results/x74_placebo/x74_placebo.json")
if PL:
    Lp = PL["cc_threads"]["labelers"]
    up = [k for k in CLS if Lp[k]["law_rr_minus_gold"][1] > 0]; put("wNClsRRUp", str(len(up)))
    prr = [Lp[k]["placebo_rr"][0] for k in CLS]; put("wClsPRRMin", f2(min(prr))); put("wClsPRRMax", f2(max(prr)))
    hp = [k for k in CLS if Lp[k]["vs_human"]["prr"][1] > 0]; put("wNClsPRRUp", str(len(hp)))
    gz = PL["cc_threads"]["gold"]
    put("wHaikuMatchedDepth", sg(Lp["haiku@m"]["law_depth_slope"][0] * PP, 2)); put("wHaikuCtxMatchedDepth", sg(Lp["haiku_ctx@m"]["law_depth_slope"][0] * PP, 2))
    put("wHaikuStrictDepth", sg(Lp["haiku@A"]["law_depth_slope"][0] * PP, 2)); put("wHaikuStrictFlag", pct(Lp["haiku@A"]["flag_rate"][0], 1))
    put("wHaikuMatchedSize", f"{Lp['haiku@m']['law_size_slope'][0] / gz['size_slope'][0]:.1f}")
    # TPR after a toxic vs a civil parent: every classifier setting finds more toxic replies after toxic parents
    import pandas as pd
    P2 = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
    z2 = P2.z.values; ch2 = np.flatnonzero(P2.parent_row.values >= 0); pa2 = P2.parent_row.values[ch2]; zc2, zp2 = z2[ch2], z2[pa2]
    def ycol(k):
        if k.endswith("@0.5"):
            return (P2[f"s:{k[:-4]}@m"].values >= 0.5).astype(float)
        return P2[f"y:{k}"].values
    diffs = [ycol(k)[ch2][(zc2 == 1) & (zp2 == 1)].mean() - ycol(k)[ch2][(zc2 == 1) & (zp2 == 0)].mean() for k in CLS]
    put("wNClsTPRUp", str(sum(d > 0 for d in diffs))); put("wClsTPRGapMin", f"{100 * min(diffs):.0f}"); put("wClsTPRGapMax", f"{100 * max(diffs):.0f}")
else:
    for k in ("wNClsRRUp", "wClsPRRMin", "wClsPRRMax", "wNClsPRRUp", "wHaikuMatchedDepth", "wHaikuCtxMatchedDepth", "wHaikuStrictDepth", "wHaikuStrictFlag",
              "wHaikuMatchedSize", "wNClsTPRUp", "wClsTPRGapMin", "wClsTPRGapMax"):
        put(k, "??")
E = load(f"{W}/results/x74_edit/x74_edit.json")
if E:
    h = E["h11"]["neg_deep"]; r_ = E["rev"]["neg_deep"]
    put("wEditOneBefore", f2(h["flag_E0"])); put("wEditOneAfter", f2(h["flag_E1"]))
    d = h["E4_minus_E3"]; put("wEditParaDiff", f"{-100 * d['diff']:.1f}"); put("wEditParaLo", f"{-100 * d['ci'][1]:.1f}"); put("wEditParaHi", f"{-100 * d['ci'][0]:.1f}")
    put("wEditRevBefore", f2(r_["flag_E0"])); put("wEditRevAfter", f2(r_["flag_E2"]))
else:
    for k in ("wEditOneBefore", "wEditOneAfter", "wEditParaDiff", "wEditParaLo", "wEditParaHi", "wEditRevBefore", "wEditRevAfter"):
        put(k, "??")
import pandas as _pd
_m = _pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id").toxicity_annotator_count
_ids = _pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id"]).id.values
put("wMedRaters", f"{int(np.median(_m.reindex(_ids).values))}")
# ---------------------------------------------------------------- Part B (filled by x74_eval_B.py into results/x74_eval/numbers_B.json)
NB_ = load(f"{W}/results/x74_eval/numbers_B.json") or {}
for k in ("wAblMinCov", "wAdvEdgeCov", "wAdvEdgeMin", "wAdvItemCov", "wAdvItemMin", "wAdvStratMin", "wAdvWorstUnif", "wCalagCovLaws",
          "wCalagShareBad", "wEdgeLawsDSL", "wEdgePlacDSL", "wItemLawsAI", "wItemLawsDSL", "wItemPlacAI", "wItemPlacDSL", "wBiasItemsMax", "wBiasItemsN", "wBiasMCMax", "wBiasEdgeMax",
          "wPpiCovMean", "wPpiCovMin", "wCompCovMinAll", "wCalagSCovLaws", "wCalagSShareBad", "wAblUnifLaws", "wAblUnifPlac", "wAblOneLaws", "wAblPlainMin"):
    put(k, NB_.get(k, "??"))
# ---------------------------------------------------------------- Table 1: placebo diagnostics on Civil Comments (x74_placebo, x74_crowd)
if PL:
    Lc_, L5_ = PL["cc_threads"]["labelers"], PL["cc_threads_s5"]["labelers"]; G_, G5_ = PL["cc_threads"]["gold"], PL["cc_threads_s5"]["gold"]
    def sgp(x, d=1):
        return sg(100 * x, d) + "\\%"
    def rat_(x):
        return f"{x:.1f}" if x >= 0 else f"$-${abs(x):.1f}"
    def kap(r, Gd, prev):
        if "kappa" in r:
            return r["kappa"][0]
        t, f_ = r["tpr"][0], r["fpr"][0]; q = prev * t + (1 - prev) * f_; po = prev * t + (1 - prev) * (1 - f_); pe = prev * q + (1 - prev) * (1 - q)
        return (po - pe) / (1 - pe)
    def row(name, r, Gd, s5=False):
        art = r.get("law_article_size_slope")
        prev = (PL["cc_threads_s5"] if s5 else PL["cc_threads"])["gold_prev"]
        a_ = "--" if (art is None or s5 or "article_size_slope" not in Gd) else rat_(art[0] / Gd['article_size_slope'][0])
        sz = "--" if s5 else rat_(r['law_size_slope'][0] / Gd['size_slope'][0])
        return (f"{name} & {kap(r, Gd, prev):.2f} & {r['fpr'][0]:.3f} & {sgp(r['placebo_depth_rel'][0])} & {sgp(r['placebo_size_rel'][0])} & {r['placebo_rr'][0]:.2f} & "
                f"{sg(r['law_depth_slope'][0] * PP, 2)} & {sz} & {a_} & {r['law_rr'][0]:.2f} \\\\").replace("$-$0.0\\%", "0.0\\%").replace("+0.0\\%", "0.0\\%")
    T = [f"Gold & -- & -- & -- & -- & -- & {sg(G_['depth_slope'][0] * PP, 2)} & 1.0 & 1.0 & {G_['rr'][0]:.2f} \\\\"]
    # revision (Addendum 75, B75b): a single rater against the other raters' majority, median of 20 draws
    O_ = load(f"{W}/results/x75_onerater/x75_onerater.json")["splits"]; mO = lambda f: float(np.median([f(x) for x in O_]))
    T.append(f"One rater$^{{\\ddagger}}$ & {mO(lambda x: x['kappa_A_B']):.2f} & {mO(lambda x: x['instr']['crowd_A']['fpr']):.3f} & "
             f"{sgp(mO(lambda x: x['instr']['crowd_A']['placebo_depth_rel']))} & {sgp(mO(lambda x: x['instr']['crowd_A']['placebo_size_rel']))} & "
             f"{np.exp(mO(lambda x: x['instr']['crowd_A']['log_placebo_rr'])):.2f} & {sg(mO(lambda x: x['instr']['crowd_A']['depth']) * PP, 2)} & "
             f"{rat_(mO(lambda x: x['instr']['crowd_A']['size'] / x['gold_B']['size']))} & {rat_(mO(lambda x: x['instr']['crowd_A']['article_size'] / x['gold_B']['article_size']))} & "
             f"{np.exp(mO(lambda x: x['instr']['crowd_A']['log_rr'])):.2f} \\\\".replace("$-$0.0\\%", "0.0\\%").replace("+0.0\\%", "0.0\\%"))
    T.append("\\midrule")
    RT_ = load(f"{W}/results/x76_rethreshold/x76_rethreshold.json")
    def row_own(name, pop, k):                                       # revision (Addendum 76, B76e): own gold-prevalence threshold
        r = RT_[pop]["labelers"][k]; g_ = RT_[pop]["gold"]; s5 = pop == "cc_s5"
        lw = lambda law: g_[law][0] + r["laws"][law]["distortion"][0]
        sz = "--" if s5 else rat_(lw("size") / g_["size"][0]); a_ = "--" if s5 else rat_(lw("article_size") / g_["article_size"][0])
        return (f"{name} & {r['kappa'][0]:.2f} & {r['fpr'][0]:.3f} & {sgp(r['placebo_depth_rel'])} & {sgp(r['placebo_size_rel_unit'])} & {r['placebo_rr'][0]:.2f} & "
                f"{sg(lw('depth') * PP, 2)} & {sz} & {a_} & {np.exp(lw('log_rr')):.2f} \\\\").replace("$-$0.0\\%", "0.0\\%").replace("+0.0\\%", "0.0\\%")
    for nm, k, s5 in [("Haiku", "haiku", 0), ("\\quad + parent", "haiku_ctx", 0), ("Sonnet$^{\\dagger}$", "sonnet", 1), ("\\quad + parent$^{\\dagger}$", "sonnet_ctx", 1),
                      ("Qwen", "qwen7b_m", 0), ("\\quad + parent", None, 0), ("Gemma$^{\\dagger}$", "gemma_m", 1), ("\\quad + parent$^{\\dagger}$", None, 1)]:
        if k is None:
            T.append(row_own(nm, "cc_s5" if s5 else "cc_threads", "gemma_m_ctx@own" if s5 else "qwen7b_m_ctx@own"))
        else:
            T.append(row(nm, (L5_ if s5 else Lc_)[k], G5_ if s5 else G_, bool(s5)))
    T.append("\\midrule")
    for nm, k in [("Detoxify (0.6)", "detoxify_original@0.6"), ("Detoxify", "detoxify_original@m"), ("Detoxify-unbiased$^{\\S}$", "detoxify_unbiased@m"), ("IMSYPP (argmax)", "imsypp@argmax"), ("IMSYPP", "imsypp@m"),
                  ("s-nlp RoBERTa$^{*}$", "sn_rob@m"), ("martin-ha DistilBERT$^{*}$", "mh_dist@m"), ("Dynabench hate$^{*}$", "fb_hate@m"), ("CardiffNLP offensive$^{*}$", "cn_off@m")]:
        T.append(row(nm, Lc_[k], G_))
    open(f"{OUTD}/tables/placebo_v4.tex", "w").write("\n".join(T) + "\n")
# ---------------------------------------------------------------- registered hypotheses (Appendix, Table hyp)
EA = load(f"{W}/results/x74_eval/eval_A.json") or {}; EB = load(f"{W}/results/x74_eval/eval_B.json") or {}
V = lambda b: "PASS" if b else "FAIL"
VV = lambda a, b: V(a) if bool(a) == bool(b) else V(a) + ", " + V(b)
HA = load(f"{W}/results/x11_conf_edge_all/eval_H71.json"); HC = load(f"{W}/results/x72_eval/eval_H72.json"); HU = load(f"{W}/results/x11_conf_unit/eval_H71e.json")["summary"]
vi = lambda E, r: E[r]["validity"]; f3_ = lambda x: f"{x:.3f}"; pc_ = lambda x: f"{100 * x:.0f}\\%" if x == 1 else f"{100 * x:.1f}\\%"
assert HA["item"]["vs_upgrade_lw"] == HC["item"]["vs_upgrade_lw"]
H = [f"H71a & Items, $B \\ge 1{{,}}000$: cov.\\ mean $\\ge 0.935$, 97\\% $\\ge 0.90$, min $\\ge 0.85$ & {f3_(vi(HC, 'item')['mean'])}; {pc_(vi(HC, 'item')['share_ge_090'])}; {f3_(vi(HC, 'item')['min'])} (run: {f3_(vi(HA, 'item')['min'])}) & {V(vi(HC, 'item')['passes'] and vi(HA, 'item')['passes'])} \\\\",
     f"H71b & Width $\\div$ stratified: laws $\\le 0.85$, placebo slopes $\\le 0.90$ & {HC['item']['vs_upgrade_lw']['laws']:.3f}; {HC['item']['vs_upgrade_lw']['placebo']:.3f} & {V(HC['item']['efficiency_passes'])} \\\\",
     f"H71c & Contagion, $B \\ge 2{{,}}000$: as H71a & {f3_(vi(HC, 'edge')['mean'])}; {pc_(vi(HC, 'edge')['share_ge_090'])}; {f3_(vi(HC, 'edge')['min'])} (run: {pc_(vi(HA, 'edge')['share_ge_090'])}; {f3_(vi(HA, 'edge')['min'])}) & {V(vi(HC, 'edge')['passes'] and vi(HA, 'edge')['passes'])} \\\\",
     f"H71d & Width $\\div$ stratified: reply risk ratio $\\le 0.97$, placebo $\\le 0.95$ & {HC['edge']['vs_upgrade_lw']['laws']:.3f}; {HC['edge']['vs_upgrade_lw']['placebo']:.3f} (run: {HA['edge']['vs_upgrade_lw']['laws']:.3f}; {HA['edge']['vs_upgrade_lw']['placebo']:.3f}) & PASS \\\\",
     f"H71e & Units, $B \\ge 500$: cov.\\ mean $\\ge 0.935$, min $\\ge 0.90$, width $\\le 0.97\\times$ own order's & {f3_(HU['mean_cov'])}; {f3_(HU['min_cov'])}; {f3_(HU['median_ratio'])} & {V(HU['mean_cov'] >= 0.935 and HU['min_cov'] >= 0.90 and HU['median_ratio'] <= 0.97)} \\\\"]
assert HC["edge"]["vs_upgrade_lw"]["laws"] <= 0.97 and HC["edge"]["vs_upgrade_lw"]["placebo"] <= 0.95 and HA["edge"]["vs_upgrade_lw"]["laws"] <= 0.97 and HA["edge"]["vs_upgrade_lw"]["placebo"] <= 0.95
if "H74a" in EA:
    H.append(f"H74a & $\\ge 9$ of 12 classifiers: placebo RR above lenient raters' & {EA['H74a']['n_pass']} of 12 & {V(EA['H74a']['pass_'])} \\\\")
    H.append(f"H74b & $\\ge 9$ of 12 classifiers: reply RR above gold's & {EA['H74b']['n_pass']} of 12 & {V(EA['H74b']['pass_'])} \\\\")
if "H74c" in EA:
    c_ = EA["H74c"]["counts"]; worst = min(v for d in c_.values() for v in d.values())
    H.append(f"H74c & Second crowd below named instruments, every law and split & all but one entry ({c_['log_rr']['detoxify_original@m']} of 20) & {V(EA['H74c']['pass_'])} \\\\")
if "H74d" in EA:
    H.append("H74d,e & Haiku: third person flagged less, added ``you'' more & " + f"{100 * EA['H74d']['diff']:+.1f} [{100 * EA['H74d']['ci'][0]:+.1f}, {100 * EA['H74d']['ci'][1]:+.1f}]; {100 * EA['H74e']['diff']:+.1f} [{100 * EA['H74e']['ci'][0]:+.1f}, {100 * EA['H74e']['ci'][1]:+.1f}]".replace("-", "$-$") + f" & {VV(EA['H74d']['pass_'], EA['H74e']['pass_'])} \\\\")
if "H74f" in EA:
    f_ = EA["H74f"]; H.append(f"H74f & Per law: a top-$\\kappa$-quartile setting $|z| > 3$, a bottom one $|z| < 1$ & max $|z|$ {f_['depth']['top_max_z']:.0f}, {f_['size']['top_max_z']:.0f}, {f_['log_rr']['top_max_z']:.0f}; min {f_['depth']['bottom_min_z']:.1f}, {f_['size']['bottom_min_z']:.1f}, {f_['log_rr']['bottom_min_z']:.1f} & {V(f_['pass_'])} \\\\")
if "H74g" in EA:
    H.append(f"H74g & Haiku's depth, size laws above gold's, strict and gold prev. & all four CIs above 0 & {V(EA['H74g']['pass_'])} \\\\" if EA["H74g"]["pass_"] else
             f"H74g & Haiku's depth, size laws above gold's, strict and gold prev. & not all & FAIL \\\\")
if "H74h" in EB:
    it_, ed_ = EB["items"], EB["edges"]; f2_ = lambda x: "--" if x is None else f"{x:.2f}"
    H.append(f"H74h & Width $\\div$ DSL $\\le 0.95$ (items), $0.97$, $0.95$ (RRs); $\\div$ active $\\le 0.90$, $0.95$ & DSL {f2_(it_['dsl']['laws'])}, {f2_(it_['dsl']['placebo'])}, {f2_(ed_['edge_dsl']['gold'])}, {f2_(ed_['edge_dsl']['placebo'])}; AI {f2_(it_['ai']['laws'])}, {f2_(it_['ai']['placebo'])}, {f2_(ed_['edge_ai']['gold'])}, {f2_(ed_['edge_ai']['placebo'])} & {V(EB['H74h']['pass'])} \\\\")
    H.append(f"H74i & Calibrate, aggregate: cov.\\ $< 0.90$ in $\\ge 25\\%$ of entries & {100 * (EB['H74i']['calag'] or 0):.0f}\\%, {100 * (EB['H74i']['calag_s'] or 0):.0f}\\% & {V(EB['H74i']['pass_'])} \\\\")
    a_, e_ = EB["H74j"]["items"], EB["H74j"]["edges"]
    if a_[0] is not None and e_[0] is not None:
        H.append(f"H74j & Adversarial labelers: cov.\\ mean $\\ge 0.935$, min $\\ge 0.90$ (laws, contagion) & {a_[0]:.3f}, {a_[1]:.3f}; {e_[0]:.3f}, {e_[1]:.3f} & {V(EB['H74j']['pass_'])} \\\\")
# Addendum 75 and 76 hypotheses (revision)
RA_ = load(f"{W}/results/x75_rating/x75_rating_eval.json")
if RA_:
    I_ = RA_["instruments"]; lo_ = lambda k: I_[k]["matched"]["depth_minus_gold"][1] * PP
    H.append(f"H75a,b & Ratings at gold prev.: depth law above gold's & lower ends {lo_('haiku_r'):+.2f}, {lo_('haiku_r_ctx'):+.2f}; {lo_('sonnet_r'):+.2f}, {lo_('sonnet_r_ctx'):+.2f} & {V(RA_['H75a']) if RA_['H75a'] == RA_['H75b'] else V(RA_['H75a']) + ', ' + V(RA_['H75b'])} \\\\")
    d_ = RA_["H75c_detail"]["diff"]; H.append(f"H75c & Haiku at gold prev.: placebo slope above single rater's & {100 * d_[0]:+.1f}\\% [{100 * d_[1]:+.1f}, {100 * d_[2]:+.1f}] & {V(RA_['H75c'])} \\\\")
O2 = load(f"{W}/results/x75_onerater/x75_onerater.json")
if O2:
    S2 = O2["splits"]; LL = ["haiku", "haiku_ctx", "qwen7b_m_ctx"]
    nd = min(sum(sp["instr"]["crowd_A"][st] < sp["instr"][l][st] for sp in S2) for l in LL for st in ("placebo_depth_rel", "placebo_size_rel"))
    ne = min(sum(abs(sp["instr"]["crowd_A"][f"{law}_minus_goldB"]) < abs(sp["instr"][l][f"{law}_minus_goldB"]) for sp in S2) for l in LL for law in ("depth", "size"))
    H.append(f"H75d,e & Single rater below three LLMs, every draw & {nd} and {ne} of {len(S2)} & {VV(nd == len(S2), ne == len(S2))} \\\\")
EV_ = load(f"{W}/results/x76_edit/x76_edit.json")
if EV_ and "H76a" in EV_:
    s_ = EV_["instruments"]["sonnet"]; a_ = s_["h11/neg_deep"]["E4_minus_E3"]; b_ = s_["rev/neg_deep"]["E2_minus_E0"]
    H.append(f"H76a,b & Sonnet: third person flagged less, added ``you'' more & " + f"{100 * a_['diff']:+.1f} [{100 * a_['ci'][0]:+.1f}, {100 * a_['ci'][1]:+.1f}]; {100 * b_['diff']:+.1f} [{100 * b_['ci'][0]:+.1f}, {100 * b_['ci'][1]:+.1f}]".replace("-", "$-$") + f" & {V(EV_['H76a']) if EV_['H76a'] == EV_['H76b'] else V(EV_['H76a']) + ', ' + V(EV_['H76b'])} \\\\")
HR_ = load(f"{W}/results/x77_humanrate/x77_humanrate.json")
mn = lambda v: f"{v:+.2f}".replace("-", "$-$")
m3 = lambda v: f"{v:+.3f}".replace("-", "$-$")
if HR_:
    if "H77a" in HR_ and "H77b" in HR_:
        la_ = [100 * v for v in HR_["H77a"]["lower_ends"]]; lb_ = [100 * v for v in HR_["H77b"]["lower_ends"]]
        H.append(f"H77a,b & Ratings above the human instrument at their rate & {mn(min(la_))} to {mn(max(la_))}; {mn(min(lb_))} to {mn(max(lb_))} & {VV(HR_['H77a']['pass_'], HR_['H77b']['pass_'])} \\\\")
CO_ = load(f"{W}/results/x77_correct/x77_correct.json")
if CO_:
    c_ = CO_["H77c"]; H.append(f"H77c & Validate and correct: cov.\\ $< 0.90$ in $\\ge 25\\%$ & {100 * c_['share_below_90']:.0f}\\% of {c_['n']} & {V(c_['pass_'])} \\\\")
YF_ = load(f"{W}/results/x77_youfree/x77_youfree.json")
if YF_:
    m_ = YF_["laws"]["depth"]["edit_minus_orig"]
    H.append(f"H77d & No ``you'' anywhere shrinks Haiku's distortion & {mn(100 * m_[0])} points [{mn(100 * m_[1])}, {mn(100 * m_[2])}] & {V(YF_['H77d'])} \\\\")
UV_ = load(f"{W}/results/x76_eval/x76_eval.json")
if UV_:
    c_ = UV_["H76c"]; H.append(f"H76c,d & Unplanned laws: as H71a; width $\\div$ PPI++ $\\le 1$ & {c_['mean']:.3f}; {100 * c_['share90']:.0f}\\%; {c_['min']:.3f}; {UV_['H76d']['median']:.2f} & {VV(c_['pass_'], UV_['H76d']['pass_'])} \\\\")
PRM_ = load(f"{W}/results/f124_prompts.json")                                                           # Addenda 24 and 30: the prompt study
if PRM_:
    cb_ = PRM_["audit"]["c_slope_bin1_minus_bin0"]; nh_ = sum(bool(PRM_[k_]) for k_ in ("H27a", "H27b", "H27c", "H27d"))
    H.append(f"H27a--d & Three Haiku prompts: share, RR, slope, with parent & {nh_} of 4; H27c {mn(100 * cb_[0])} [{mn(100 * cb_[1])}, {mn(100 * cb_[2])}] & {V(nh_ == 4)} \\\\")
X80_ = f"{W}/results/x80"                                                                                   # Addendum 80
MA_ = load(f"{X80_}/x80_matched.json")
if MA_:
    H.append(f"H80a & $\\ge 9$ of 12 classifiers: RR above matched human & {MA_['H80a']['n_pass']} of 12 & {V(MA_['H80a']['pass_'])} \\\\")
    lo_ = [100 * d_["lower"] for d_ in MA_["H80b"]["detail"]]
    H.append(f"H80b & 5 escalating LLMs: depth law above matched human & {sum(d_['above'] for d_ in MA_['H80b']['detail'])} of 5, lower ends {mn(min(lo_))} to {mn(max(lo_))} & {V(MA_['H80b']['pass_'])} \\\\")
TH_ = load(f"{X80_}/x80_threshold.json")
if TH_:
    bt_ = TH_["H80c"]["by_t"]
    H.append(f"H80c & Gold at thresholds 0.4--0.7: excess $> 0$, $\\ge 4$ of 7 & {min(v_['median'] for v_ in bt_.values()):+.2f} to {max(v_['median'] for v_ in bt_.values()):+.2f}; {min(v_['n_ci_above'] for v_ in bt_.values())}--{max(v_['n_ci_above'] for v_ in bt_.values())} of 7 & {V(TH_['H80c']['pass_'])} \\\\")
NZ_ = load(f"{X80_}/x80_noise.json")
if NZ_:
    H.append(f"H80d & Reweighted many-rater gold below $\\ge 9$ of 12 & infeasible (eff.\\ $n$ {NZ_['weights']['ess']:.0f}); {NZ_['H80d']['n_above']} of 12 & {V(NZ_['H80d']['pass_'])} \\\\")
YPE_ = load(f"{X80_}/x80_youprompt_eval.json")
if YPE_ and "H80e" in YPE_:
    ce_, cf_ = YPE_["H80e"]["change"], YPE_["H80f"]["change"]
    H.append(f"H80e,f & ``You'' note shrinks Haiku's, Sonnet's distortion & {mn(100 * ce_[0])} [{mn(100 * ce_[1])}, {mn(100 * ce_[2])}]; {mn(100 * cf_[0])} [{mn(100 * cf_[1])}, {mn(100 * cf_[2])}] & {VV(YPE_['H80e']['pass_'], YPE_['H80f']['pass_'])} \\\\")
else:
    H.append(f"H80e,f & ``You'' note shrinks Haiku's, Sonnet's distortion & pending & -- \\\\")
LO_ = load(f"{X80_}/x80_lopo_registered.json")
if LO_:
    H.append(f"H80g & One population held out: designs re-picked $\\ge 6$ times & {LO_['item']['n_frozen_adopted']} of {LO_['item']['n_runs']}, {LO_['edge']['n_frozen_adopted']} of {LO_['edge']['n_runs']}; gaps $\\le$ {100 * (LO_['item']['max_frozen_over_adopted'] - 1):.1f}, {100 * (LO_['edge']['max_frozen_over_adopted'] - 1):.1f}\\% & {V(LO_['H80g']['pass_'])} \\\\")
UC_ = load(f"{X80_}/x80_units_cluster.json")
if UC_:
    bb_ = UC_["summary"]["by_B"]
    H.append(f"H80h & Units: width below cluster PPI's in $\\ge 80\\%$ of cells & {100 * min(v_['share_below_1'] for v_ in bb_.values()):.0f}\\% (median {bb_['500']['median']:.2f} at 500) & {V(UC_['H80h']['pass_'])} \\\\")
CLS12_H = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m_}@{t_}" for m_ in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t_ in ("0.5", "m")]
X81_ = load(f"{W}/results/x81/x81_eval.json")
if X81_:
    c_ = X81_["H81a"]["change"]; d_ = X81_["H81b"]["detail"]
    H.append(f"H81a,b & Note, one batch: Haiku's change $> 0$; both stay $> 75\\%$ & {mn(100 * c_[0])} [{mn(100 * c_[1])}, {mn(100 * c_[2])}]; {mn(100 * d_['haiku']['you'][0])}, {mn(100 * d_['sonnet']['you'][0])} (main {mn(100 * d_['haiku']['main'][0])}, {mn(100 * d_['sonnet']['main'][0])}) & {VV(X81_['H81a']['pass_'], X81_['H81b']['pass_'])} \\\\")
X82L_ = load(f"{W}/results/x82/x82_latent.json"); X82P_ = load(f"{W}/results/x82/x82_protocol.json")
if X82L_ and X82P_:
    g_ = X82L_["H82b"]["gaps"]; a_ = X82P_["H82a"]
    H.append(f"H82b & Latent model: 1-, 3-rater, gold log RRs within 0.05 & {m3(g_['k1'])}, {m3(g_['k3'])}, {m3(g_['gold'])} & {V(X82L_['H82b']['pass_'])} \\\\")
    H.append(f"H82b$'$ & Same under the adaptive protocol (gold) & {m3(X82P_['H82b_prime']['gap'])} & {V(X82P_['H82b_prime']['pass_'])} \\\\")
    C85_ = load(f"{W}/results/x85/x85_rrinf.json")                                                  # Addendum 85, B85a: converged re-evaluation
    cv_ = f" (converged {C85_['rr_inf']['rr'][0]:.2f}: {C85_['H82a_converged']['n_above']}, {C85_['H82a_converged']['n_below']})" if C85_ else ""
    H.append(f"H82a & $\\ge 9$ of 12 classifiers: RR above noise-free {a_['rr_inf'][0]:.2f} & {a_['n_above']} above, {a_['n_below']} below{cv_} & {V(a_['pass_'])} \\\\")
X83_ = load(f"{W}/results/x83/x83_construct.json")
if X83_:
    A_ = X83_["H83a"]; B_ = X83_["H83b"]; R_ = X83_["grids"]["20"]["instruments"]
    ea_ = [R_[k_]["placebo_lrr"]["excess"][0] for k_ in CLS12_H]; eb_ = [100 * R_[k_]["placebo_slope"]["excess"][1] for k_ in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx")]
    H.append(f"H83a & $\\ge 9$ of 12 classifiers: placebo RR above construct-only & {A_['n_above']} of 12 (excess {mn(min(ea_))} to {mn(max(ea_))}) & {V(A_['pass_'])} \\\\")
    H.append(f"H83b & Haiku, Sonnet ($\\pm$ parent): placebo slope above it & {B_['n_above']} of 4, lower ends {mn(min(eb_))} to {mn(max(eb_))} & {V(B_['pass_'])} \\\\")
MN_ = load(f"{W}/results/x83/x83d_par.json")
if MN_ and "H83f" in MN_:
    nc_ = [MN_["grids"][m_]["contagion"]["n_above"] for m_ in ("20", "10", "40")]; ne_ = [MN_["grids"][m_]["escalation"]["n_above"] for m_ in ("20", "10", "40")]
    CV_ = {m_: load(f"{W}/results/x85/x85_construct_{m_}.json") for m_ in ("20", "10", "40")}                  # Addendum 85, B85a
    cvs_ = "" if not all(CV_.values()) else "; converged: " + ", ".join(f"{CV_[m_]['grids'][m_]['contagion']['n_above']}, {CV_[m_]['grids'][m_]['escalation']['n_above']}" for m_ in ("20", "10", "40"))
    H.append(f"H83d--f & Monotone $f$: H83a, b at $m = 20$; same at 10, 40 & {nc_[0]} of 12, {ne_[0]} of 4; {nc_[1]}, {ne_[1]}; {nc_[2]}, {ne_[2]}{cvs_} & {V(MN_['H83d']['pass_'] and MN_['H83e']['pass_'] and MN_['H83f']['pass_'])} \\\\")
WA_ = load(f"{W}/results/x86/x86_within.json")                                                           # Addendum 86, B86a
if WA_:
    ia_ = WA_["strata"]["article"]["instruments"]; st_ = [ia_[k_]["fp"] + ia_[k_]["tp"] for k_ in CLS12_H]
    H.append(f"H86a & Within articles: structured log-RR terms $> 0$, $\\ge 9$ of 12 & {WA_['H86a']['n_structured_pos']} of 12 ({mn(min(st_))} to {mn(max(st_))}) & {V(WA_['H86a']['pass_'])} \\\\")
FR_ = load(f"{W}/results/x85/x85_construct_frac.json")                                                  # Addendum 85, B85c'
if FR_:
    hc_ = FR_["H85c_prime"]["diff"]; si_ = FR_["slopes"]["insult_any"]; sh_ = FR_["slopes"]["haiku"]
    H.append(f"H85c$'$ & Gold negatives: any-insult depth slope $< \\frac14$ of Haiku's & {mn(si_[0])} vs {mn(sh_[0])}; gap {mn(hc_[0])} [{mn(hc_[1])}, {mn(hc_[2])}] & {V(FR_['H85c_prime']['pass_'])} \\\\")
MSZ_ = load(f"{W}/results/x83/x83_matched_size.json")
if MSZ_:
    I_ = MSZ_["cc_threads"]["instruments"]; lo_ = [100 * I_[k_]["thread_size_minus_human"][1] for k_ in ("haiku", "haiku_ctx", "qwen7b_m_ctx@own")]
    H.append(f"H83c & 3 LLM settings: thread size law above matched human & {MSZ_['H83c']['n_above']} of 3, lower ends {mn(min(lo_))} to {mn(max(lo_))} & {V(MSZ_['H83c']['pass_'])} \\\\")
# page limit: related registered hypotheses share a row (predictions, results and verdicts kept in order)
MERGE = {("H71a", "H71c"): ("H71a,c", "Coverage, items ($B \\ge 1{,}000$) and contagion ($B \\ge 2{,}000$): mean $\\ge 0.935$, 97\\% $\\ge 0.90$, min $\\ge 0.85$"),
         ("H71b", "H71d"): ("H71b,d", "Width $\\div$ stratified: laws $\\le 0.85$, placebo slopes $\\le 0.90$; reply RR $\\le 0.97$, placebo RR $\\le 0.95$"),
         ("H74a", "H74b"): ("H74a,b", "$\\ge 9$ of 12 classifiers: placebo RR above lenient raters'; reply RR above gold's"),
         ("H80a", "H80b"): ("H80a,b", "Above matched human: $\\ge 9$ of 12 classifiers' RR; 5 escalating LLMs' depth law"),
         ("H82b", "H82b$'$"): ("H82b,b$'$", "Latent model: 1-, 3-rater, gold log RRs within 0.05; gold under the adaptive protocol"),
         ("H83a", "H83b"): ("H83a,b", "Above construct-only (free $f$): $\\ge 9$ of 12 classifiers' placebo RR (log scale); Haiku, Sonnet ($\\pm$ parent) placebo slope (points per level)"),
         ("H75a,b", "H75c"): ("H75a--c", "Ratings at gold prev.: depth law above gold's; Haiku's placebo slope above single rater's"),
         ("H75a--c", "H75d,e"): ("H75a--e", "Ratings at gold prev.: depth law above gold's; Haiku's placebo slope above single rater's; single rater below three LLMs, every draw"),
         ("H80c", "H80d"): ("H80c,d", "Gold at thresholds 0.4--0.7: excess $> 0$, $\\ge 4$ of 7; reweighted many-rater gold below $\\ge 9$ of 12"),
         ("H74h", "H74i"): ("H74h,i", "Width $\\div$ DSL $\\le 0.95$ (items), $0.97$, $0.95$ (RRs); $\\div$ active $\\le 0.90$, $0.95$; calibrate, aggregate: cov.\\ $< 0.90$ in $\\ge 25\\%$"),
         ("H74d,e", "H76a,b"): ("H74d,e; H76a,b", "Haiku; Sonnet: third person flagged less, added ``you'' more"),
         ("H80e,f", "H81a,b"): ("H80e,f; H81a,b", "``You'' note shrinks Haiku's, Sonnet's distortion; one batch: Haiku's change $> 0$, both stay $> 75\\%$")}
rows_ = {r.split(" & ")[0]: r for r in H}
for (a_, b_), (nid_, pred_) in MERGE.items():
    if a_ in rows_ and b_ in rows_:
        pa_ = rows_.pop(a_).rstrip().rstrip("\\").rstrip().split(" & "); pb_ = rows_.pop(b_).rstrip().rstrip("\\").rstrip().split(" & ")
        ver_ = pa_[3] if pa_[3] == pb_[3] else f"{pa_[3]}, {pb_[3]}"
        rows_[nid_] = f"{nid_} & {pred_} & {pa_[2]} / {pb_[2]} & {ver_} \\\\"
H = list(rows_.values())
H.sort(key=lambda r: r.split(" & ")[0])
open(f"{OUTD}/tables/hypotheses_v4.tex", "w").write("\n".join(H) + "\n")
with open(f"{OUTD}/numbers_v4.tex", "w") as f:
    f.write("% generated by scripts/x16_paper_numbers.py\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros ->", f"{OUTD}/numbers_v4.tex")
for k, v in M.items():
    print(f"  {k} = {v}")
