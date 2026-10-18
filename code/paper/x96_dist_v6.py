"""v6 (readability round): Table 2 in distortions. Each LLM cell of Table 2 is the labeler's distortion of the depth law (its law
minus the human-labeled law on the same comments, points per level), so that every labeler number in Section 3 is the same quantity.
Sources (unrounded, the same files the earlier macros came from):
  full reply threads: results/x90/x90_dose.json (law_distortion); within threads: results/x75_within/x75_within.json;
  5% sample: results/x92/x92_matched.json (depth_excess of the own cut, the 0-100 rating at gold's flag rate, "very toxic"),
  results/x88/x88_calibrated.json (the kappa-maximizing 0-100 cut on 500 gold labels, median excess), results/x88/x88_rerun.json,
  results/x91/x91_mitigate.json (prompt variants, relabeled with the 5% sample's gold);
  replies shown the parent: results/x89/x89_cc_reference.json (and results/b84/x84_families.json for the 5% sample's gold among replies;
  Qwen recomputed from results/x14_pops/cc_threads.parquet exactly as x18_numbers_v5.py does);
  CCC: results/x74_ccc/x74_ccc.json; the separate prompt sample: results/f124_prompts.json.
Writes paper_v5/numbers_v6_dist.tex.
usage: python3 x96_dist_v6.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json
import numpy as np, pandas as pd
W = _ROOT; R = f"{W}/results"; P = f"{W}/paper_arxiv"
L = lambda p: json.load(open(f"{R}/{p}"))
sg = lambda v: "0.00" if abs(v) < 0.005 else (f"+{v:.2f}" if v > 0 else f"$-${abs(v):.2f}")
D = {}
do = L("x90/x90_dose.json")["corpora"]
D["dHaikuMain"] = do["cc_threads"]["labelers"]["haiku"]["law_distortion"][0]
mt = L("x92/x92_matched.json"); rows = {(r["model"], r["op"]): r for r in mt["rows"]}; g5 = mt["gold"]["depth"][0]
D["dHaikuMainSfive"] = rows[("haiku", "native")]["depth_excess"][0]; D["dSonnetMain"] = rows[("sonnet", "native")]["depth_excess"][0]
assert abs(D["dSonnetMain"] - do["cc_s5"]["labelers"]["sonnet"]["law_distortion"][0]) < 0.006
Wn = L("x75_within/x75_within.json")["within"]
D["dHaikuWithin"] = 100 * Wn["cc_threads"]["labelers"]["haiku"]["distortion"][0]; D["dSonnetWithin"] = 100 * Wn["cc_s5"]["labelers"]["sonnet"]["distortion"][0]
cr = L("x89/x89_cc_reference.json"); fam = L("b84/x84_families.json")
gr = cr["cc_threads"]["gold"]["depth_replies"][0]; gr5 = fam["settings"]["olmo2"]["depth_replies_gold"][0]
D["dHaikuCtx"] = cr["cc_threads"]["instruments"]["haiku_tox_ctx"]["depth_replies"][0] - gr
D["dSonnetCtx"] = cr["cc_s5"]["instruments"]["sonnet_tox_ctx"]["depth_replies"][0] - gr5
Pq = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["depth", "z", "s:qwen7b_m_ctx"]); zq = Pq.z.values; Nq = len(zq)
sq = Pq["s:qwen7b_m_ctx"].values.astype(float); sq = np.where(np.isnan(sq), -np.inf, sq); kq = int(round(zq.mean() * Nq))
oq = np.lexsort((np.random.default_rng(7650).random(Nq), -sq)); fq = np.zeros(Nq); fq[oq[:kq]] = 1.0
xq = np.minimum(Pq.depth.values, 10).astype(float); rq = xq >= 1; xr = xq[rq] - xq[rq].mean()
D["dQwenCtx"] = 100 * float((xr * (fq[rq] - fq[rq].mean())).sum() / (xr * xr).sum()) - gr
# v7: Qwen among replies without the parent (flags at gold's flag rate over all comments, as Table 1), the like-for-like comparison
Pq2 = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["s:qwen7b_m"]); s2 = Pq2["s:qwen7b_m"].values.astype(float); s2 = np.where(np.isnan(s2), -np.inf, s2)
o2 = np.lexsort((np.random.default_rng(7650).random(Nq), -s2)); f2 = np.zeros(Nq); f2[o2[:kq]] = 1.0
D["dQwenRepl"] = 100 * float((xr * (f2[rq] - f2[rq].mean())).sum() / (xr * xr).sum()) - gr
X = L("x74_ccc/x74_ccc.json")["estimates"]; D["dHaikuCcc"] = 100 * (X["slope_Y_haiku_ic"][0] - X["slope_Y_hic"][0])
rr = L("x88/x88_rerun.json"); D["dHaikuRerun"] = rr["haiku"]["depth_new"][0] - g5; D["dSonnetRerun"] = rr["sonnet"]["depth_new"][0] - g5
S5 = L("x88/x88_calibrated.json")["pops"]["cc_s5"]["instruments"]
D["dHaikuKmax"] = S5["haiku:rating"]["calibrated"]["kappa@500"]["excess_median"]; D["dSonnetKmax"] = S5["sonnet:rating"]["calibrated"]["kappa@500"]["excess_median"]
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    D[f"d{nm}Prev"] = rows[(m_, "rating at gold prevalence")]["depth_excess"][0]; D[f"d{nm}Vt"] = rows[(m_, "very toxic")]["depth_excess"][0]
I = L("x91/x91_mitigate.json")["instruments"]
for nm, k in (("Tuned", "T8"), ("Think", "TR"), ("MitOne", "M1"), ("MitTwo", "M2"), ("Batch", "T0")):
    D[f"dHaiku{nm}"] = I[k]["distortion"][0]
A = L("f124_prompts.json")["audit"]
D["dHaikuPrMain"] = 100 * (A["slope_main0"][0] - A["slope_z"][0]); D["dHaikuPrBin"] = 100 * (A["slope_bin0"][0] - A["slope_z"][0]); D["dHaikuPrRev"] = 100 * (A["slope_rev0"][0] - A["slope_z"][0])
# replies only, without the parent (the like-for-like comparison for Table 2's "shown the parent" row)
D["dHaikuRepl"] = cr["cc_threads"]["instruments"]["haiku_tox"]["depth_replies"][0] - gr
D["dSonnetRepl"] = cr["cc_s5"]["instruments"]["sonnet_tox"]["depth_replies"][0] - cr["cc_s5"]["gold"]["depth_replies"][0]
D["dSonnetCtx"] = cr["cc_s5"]["instruments"]["sonnet_tox_ctx"]["depth_replies"][0] - cr["cc_s5"]["gold"]["depth_replies"][0]
# Figure 2: the share of structured false positives in the escalation distortions (Haiku and Sonnet; shown the parent, among replies, x97)
dec = L("x74_decomp/x74_decomp.json"); rp = L("x97/x97_decomp_replies.json")
sh = [dec["cc_threads"]["labelers"]["haiku"]["laws"]["depth"]["fp_term"][0] / dec["cc_threads"]["labelers"]["haiku"]["laws"]["depth"]["distortion"][0],
      dec["cc_s5"]["labelers"]["sonnet"]["laws"]["depth"]["fp_term"][0] / dec["cc_s5"]["labelers"]["sonnet"]["laws"]["depth"]["distortion"][0]] + \
     [rp[p_]["labelers"][k_]["fp_share"] for p_, k_ in (("cc_threads", "haiku_ctx"), ("cc_s5", "sonnet_ctx"), ("cc_threads", "qwen7b_m_ctx@own"))]
for p_, k_, nm_ in (("cc_threads", "haiku_ctx", "HaikuCtx"), ("cc_s5", "sonnet_ctx", "SonnetCtx"), ("cc_threads", "qwen7b_m_ctx@own", "QwenCtx")):
    assert abs(rp[p_]["labelers"][k_]["distortion"][0] - D[f"d{nm_}"]) < 0.006                    # the same distortions as Table 2 and the text
T = {}
T["dFPShareMin"] = f"{100 * min(sh):.0f}\\%"; T["dFPShareMax"] = f"{100 * max(sh):.0f}\\%"
T["dFPShareHaiku"] = f"{100 * sh[0]:.0f}\\%"; T["dFPShareSonnet"] = f"{100 * sh[1]:.0f}\\%"                     # v7: the intro's two distortions
# v7: the prompt chosen for its agreement (T8) against the main prompt, on the 5% sample: kappa and flag rate (B91 / B89b)
T["dKapMainSfive"] = f"{I['stored']['kappa']:.2f}"; T["dKapTuned"] = f"{I['T8']['kappa']:.2f}"; T["dTunedFlag"] = f"{100 * I['T8']['flag_rate']:.0f}\\%"
assert I["T8"]["kappa"] == max(I[k]["kappa"] for k in ("T0", "T8", "T16", "TR")) and I["T8"]["kappa"] > I["stored"]["kappa"]
# v7: the paraphrase control (x76_edit.json, Haiku, clearly non-toxic second-person replies at depth >= 2): flags on Sonnet's paraphrases
ed = L("x76_edit/x76_edit.json")["instruments"]["haiku"]["h11/neg_deep"]
T["dParaN"] = f"{ed['n']:,}".replace(",", "{,}"); T["dParaYou"] = f"{100 * ed['E3']:.1f}\\%"; T["dParaThey"] = f"{100 * ed['E4']:.1f}\\%"
assert abs(100 * (ed["E3"] - ed["E4"]) - 8.7) < 0.05                                                         # the text's 8.7 points (wEditParaDiff)
# classifiers that report more contagion than gold (reply risk ratio's CI above gold's), by training data
Lp = L("x74_placebo/x74_placebo.json")["cc_threads"]["labelers"]
CC = ["detoxify_unbiased@0.6", "detoxify_unbiased@m", "sn_rob@0.5", "sn_rob@m", "mh_dist@0.5", "mh_dist@m"]          # trained in part on Civil Comments labels
NOT = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m", "fb_hate@0.5", "fb_hate@m", "cn_off@0.5", "cn_off@m"]
up = lambda ks: sum(Lp[k]["law_rr_minus_gold"][1] > 0 for k in ks)
T["vClsRRUpAll"] = str(up(CC + NOT)); T["vClsN"] = str(len(CC + NOT)); T["vClsCcN"] = str(len(CC)); T["vClsCcUp"] = str(up(CC))
assert up(CC) == len(CC)
# the seven LLMs at gold's flag rate: kappa against the distortion of the depth law (Table 3)
from scipy.stats import spearmanr
sel = [("haiku", "rating at gold prevalence"), ("sonnet", "rating at gold prevalence")] + [(m_, "gold prevalence") for m_ in ("qwen", "gemma", "olmo2", "granite", "llama")]
rho = spearmanr([rows[s_]["kappa"] for s_ in sel], [rows[s_]["depth_excess"][0] for s_ in sel]).correlation
T["dLlmKapRho"] = f"+{rho:.2f}" if rho > 0 else f"$-${abs(rho):.2f}"
assert max(sel, key=lambda s_: rows[s_]["kappa"])[0] == "sonnet" and max(sel, key=lambda s_: rows[s_]["depth_excess"][0])[0] == "sonnet"
T["dKapHaikuPrev"] = f"{rows[('haiku', 'rating at gold prevalence')]['kappa']:.2f}"; T["dKapGemmaPrev"] = f"{rows[('gemma', 'gold prevalence')]['kappa']:.2f}"
T["dLawHaikuPrev"] = sg(g5 + rows[("haiku", "rating at gold prevalence")]["depth_excess"][0]); T["dLawGemmaPrev"] = sg(g5 + rows[("gemma", "gold prevalence")]["depth_excess"][0])
# flag rates of the prompt variants (B91) and of the main prompt (5% sample)
fr_ = [I[k]["flag_rate"] for k in ("T8", "TR", "M1", "M2")]
T["dPromptFlagMin"] = f"{100 * min(fr_):.0f}\\%"; T["dPromptFlagMax"] = f"{100 * max(fr_):.0f}\\%"; T["dMainFlag"] = f"{100 * rows[('haiku', 'native')]['flag_rate']:.0f}\\%"
# v7: the classifiers' 4+:0 ratios at gold's flag rate (x90_dose.json, 5% sample): Detoxify's (the largest) and the next largest
MTd = do["cc_s5"]["matched"]; rc_ = sorted(((v["ratio_4_0"][0], k) for k, v in MTd.items() if v["kind"] == "classifier"), reverse=True)
assert rc_[0][1] == "detoxify_original@m"; T["dClsRatioSecond"] = f"{rc_[1][0]:.1f}"
# v7: flag rates of the own cuts and gold's prevalence (Figure 1 caption, Section 3)
Pf = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["z", "y:haiku"]); Sf = pd.read_parquet(f"{R}/x14_pops/cc_s5.parquet", columns=["z", "y:sonnet"])
T["dGoldPrev"] = f"{100 * Pf.z.mean():.1f}\\%"; T["dHaikuFlagAll"] = f"{100 * Pf['y:haiku'].mean():.0f}\\%"; T["dSonnetFlagSfive"] = f"{100 * Sf['y:sonnet'].mean():.0f}\\%"
# v7e: the paired change of the prompt chosen for agreement (T8) against the same-batch rerun of the main prompt (T0), B91
ch = I["T8"]["change_vs_T0"]; T["dTunedChg"] = sg(ch[0]); T["dTunedChgLo"] = sg(ch[1]); T["dTunedChgHi"] = sg(ch[2])
assert ch[1] > 0 and I["T8"]["kappa"] == max(I[k]["kappa"] for k in ("T0", "T8", "T16", "TR", "M1", "M2"))
# v7e: the kappa-maximizing four-level cut is Haiku's own cut in this share of draws at 200 gold labels (B88a, all reply threads)
Hc = L("x88/x88_calibrated.json")["pops"]["cc_threads"]["instruments"]["haiku:level"]["calibrated"]
T["dCalNativeTwo"] = f"{100 * Hc['kappa@200']['cut_counts'].get('2', 0) / 500:.0f}\\%"
# v7e: the thread size law keeps gold's sign in this many of the 14 classifier settings (x74_placebo.json, Table 7)
Lp2 = L("x74_placebo/x74_placebo.json")["cc_threads"]; gs = Lp2["gold"]["size_slope"][0]
T["vClsThreadSizeSign"] = str(sum((Lp2["labelers"][k]["law_size_slope"][0] > 0) == (gs > 0) for k in CC + NOT))
# v7e: the paraphrase rates to two decimals (appendix)
T["dParaYouP"] = f"{100 * ed['E3']:.2f}\\%"; T["dParaTheyP"] = f"{100 * ed['E4']:.2f}\\%"
# v7e: the lowest coverage that Monte Carlo error alone would produce over the evaluation's combinations (500 audits each, true coverage 95%)
import re as _re
nC = int(_re.search(r"\\newcommand\{\\vItemEntries\}\{(\d+)\}", "".join(open(f"{W}/results/reference/{f_}").read() for f_ in ("numbers_v4b.tex", "numbers_v5.tex", "numbers_v6_eval.tex", "numbers_v2.tex", "numbers_v3.tex", "numbers_v4.tex"))).group(1))
mins = np.random.default_rng(7700).binomial(500, 0.95, size=(4000, nC)).min(axis=1) / 500
T["dMCMinCov"] = f"{100 * np.median(mins):.1f}\\%"
with open(f"{P}/numbers_v6_dist.tex", "w") as fo:
    for k, v in D.items():
        fo.write(f"\\newcommand{{\\{k}}}{{{sg(v)}}}\n")
    for k, v in T.items():
        fo.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print({k: sg(v) for k, v in D.items()}); print(T)
