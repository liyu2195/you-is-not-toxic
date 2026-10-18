"""Tables of the v5 draft built from Addendum 88-89 results.
tables/platforms_v5.tex: the second-person bias on six platforms under the same toxicity question (results/x89/x89_cc_reference.json
for Civil Comments, results/x89/x89_other_tox.json for the others; Haiku's placebo slope at a fixed number of second-person words from
results/x90/x90_dose.json).
usage: python3 x18_tables_v5.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os
W = _ROOT; R = f"{W}/results"; T = f"{W}/paper_arxiv/tables"
def sg(x, d=2):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"                                                      # no signed zero
    return s.replace("-", "$-$") if s.startswith("-") else s
def f(x, d=1):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s
cc = json.load(open(f"{R}/x89/x89_cc_reference.json")); ot = json.load(open(f"{R}/x89/x89_other_tox.json"))
DO = json.load(open(f"{R}/x90/x90_dose.json"))["corpora"]                                     # within the number of second-person words (Addendum 90)
WC = {"cc": DO["cc_threads"], "cad": DO["cad"], "qr": DO["qian_reddit"], "qg": DO["qian_gab"], "cga": DO["cga"], "bad": DO["bad"]}
ROWS = [("News comments (Civil Comments)", cc["cc_threads"], "cc"), ("Reddit (CAD)", ot["cad"], "cad"), ("Reddit (Qian et al.)", ot["qian_reddit"], "qr"),
        ("Gab (Qian et al.)", ot["qian_gab"], "qg"), ("Wikipedia talk (CGA-Wiki)", ot["cga"], "cga"), ("Chatbot dialogue (BAD)", ot["bad"], "bad")]
lines = []
for name, Rr, tag in ROWS:
    I = Rr["instruments"]; h = I["haiku_tox"]; s = I.get("sonnet_tox") or (cc["cc_s5"]["instruments"]["sonnet_tox"] if tag == "cc" else None)
    d = I.get("ref:detoxify_original@m"); hc = I.get("haiku_tox_ctx")
    gap = lambda v: f(100 * v["fpr_you_minus_other"][0]) if v else "--"
    cells = [name, f"{Rr['N']:,}".replace(",", "{,}"), f"{Rr['you_root']:.2f}--{Rr['you_deep']:.2f}" if Rr.get("you_deep") is not None else "--",
             gap(h), gap(s), gap(d), f"{sg(h['placebo'][0])} [{sg(h['placebo'][1])}, {sg(h['placebo'][2])}]", sg(WC[tag]["labelers"]["haiku"]["within_count"][0]),
             sg(Rr["gold"]["depth"][0]), sg(h["depth"][0]), f"{sg(hc['depth_replies'][0])} ({sg(Rr['gold']['depth_replies'][0])})" if hc else "--"]
    lines.append(" & ".join(cells) + " \\\\")
open(f"{T}/platforms_v5.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- tables/examples_v5.tex: minimal pairs from the edit experiment
import pandas as pd, re as _re
items = {}; lab = {}
for l in open(f"{R}/x74_edit/label_items/shard_000.jsonl"):
    r = json.loads(l); items[int(r["id"])] = r["text"]
for l in open(f"{R}/x74_edit/haiku_tox/shard_000.jsonl"):
    r = json.loads(l); lab[int(r["id"])] = r.get("ans")
S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id")
pairs = [cid for cid in S11.index.astype(int) if lab.get(cid * 10) in ("A", "B") and lab.get(cid * 10 + 1) == "D" and S11.loc[cid, "tox"] == 0]
short = [cid for cid in pairs if 30 < len(items[cid * 10]) < 170]
PICK = [5878842, 1064184, 6272211]                                 # three clearly benign short pairs, chosen by reading the 24 short ones (v5b: one fewer, for space)
assert all(c in short for c in PICK)
def tex(t):
    t = _re.sub(r"https?://\S+", "[link]", t).replace("\n", " ")
    for a, b in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_")):
        t = t.replace(a, b)
    return " ".join(t.split())
REDACT_SHA256 = {"9dc51a7cf32335d044e9abf51b7f282c8957acfaf1f8c06bcc122bc19be2ed3e": (12, '[name]')}   # v6: a private person's name is removed from the printed pair (release: given by the SHA-256 of the name, not spelled out)
def red(t):
    """replace every occurrence of a redacted string, found by its SHA-256 and length (as str.replace, left to right)"""
    import hashlib as _hl
    for dg_, (n_, b_) in REDACT_SHA256.items():
        out_, i_ = [], 0
        while i_ <= len(t) - n_:
            if _hl.sha256(t[i_:i_ + n_].encode()).hexdigest() == dg_:
                out_.append(b_); i_ += n_
            else:
                out_.append(t[i_]); i_ += 1
        t = "".join(out_) + t[i_:]
    return t
rows = [f"{red(tex(items[c * 10]))} & {red(tex(items[c * 10 + 1]))} \\\\" for c in PICK]
open(f"{T}/examples_v5.tex", "w").write("\n".join(rows) + "\n")
with open(f"{W}/paper_arxiv/numbers_v5_examples.tex", "w") as fo:
    fo.write(f"\\newcommand{{\\cExPairs}}{{{len(pairs)}}}\n\\newcommand{{\\cExShort}}{{{len(short)}}}\n\\newcommand{{\\cExItems}}{{{len(S11)}}}\n")
print(len(pairs), "pairs,", len(short), "short"); print("\n".join(rows))

# ---------------------------------------------------------------- tables/matched_v5.tex: every LLM at matched operating points (B92a)
mt = json.load(open(f"{R}/x92/x92_matched.json"))
NM = {"haiku": "Haiku", "sonnet": "Sonnet", "qwen": "Qwen", "gemma": "Gemma", "olmo2": "OLMo-2", "granite": "Granite", "llama": "Llama"}
OP = {"native": "own cut", "very toxic": "``very toxic''", "rating at gold prevalence": "rating, gold prev.", "rating at max kappa": "rating, max $\\kappa$",
      "gold prevalence": "gold prevalence", "max kappa": "max $\\kappa$"}
out = []; last = None
for r in mt["rows"]:
    if r["op"] == "max kappa":                       # v5b: the open models' max-kappa rows stay in x92_matched.json (space)
        continue
    nm = NM[r["model"]] if r["model"] != last else ""; last = r["model"]
    d, fx = r["depth_excess"], r["depth_excess_fixed"]
    out.append(" & ".join([nm, OP[r["op"]], f(100 * r["flag_rate"], 1), f(r["kappa"], 2), f"{sg(d[0])} [{sg(d[1])}, {sg(d[2])}]", sg(fx[0]), sg(r["thread_size_excess"][0]), sg(r["log_rr_excess"][0])]) + " \\\\")
open(f"{T}/matched_v5.tex", "w").write("\n".join(out) + "\n"); print("\n".join(out[:4]))

# ================================================================ v6 (readability round): tables with one unit per quantity
import numpy as _np
# ---------------------------------------------------------------- tables/laws_v6.tex: Table 1, each instrument's placebo statistics and laws on Civil Comments
PL = json.load(open(f"{R}/x74_placebo/x74_placebo.json")); Lc, L5 = PL["cc_threads"]["labelers"], PL["cc_threads_s5"]["labelers"]; G = PL["cc_threads"]["gold"]
pts = lambda v: sg(100 * v, 2)
def kap(r, prev):
    if "kappa" in r:
        return r["kappa"][0]
    t, f_ = r["tpr"][0], r["fpr"][0]; q = prev * t + (1 - prev) * f_; po = prev * t + (1 - prev) * (1 - f_); pe = prev * q + (1 - prev) * (1 - q)
    return (po - pe) / (1 - pe)
def lrow(name, r, s5=False):
    prev = (PL["cc_threads_s5"] if s5 else PL["cc_threads"])["gold_prev"]
    return " & ".join([name, f"{kap(r, prev):.2f}", f"{r['fpr'][0]:.3f}", pts(r["placebo_depth_slope"][0]), f"{r['placebo_rr'][0]:.2f}", pts(r["law_depth_slope"][0]),
                       "--" if s5 else pts(r["law_size_slope"][0]), f"{r['law_rr'][0]:.2f}"]) + " \\\\"
T6 = [" & ".join(["Gold", "--", "--", "--", "--", pts(G["depth_slope"][0]), pts(G["size_slope"][0]), f"{G['rr'][0]:.2f}"]) + " \\\\"]
O_ = json.load(open(f"{R}/x75_onerater/x75_onerater.json"))["splits"]; mO = lambda g: float(_np.median([g(x) for x in O_]))
T6.append(" & ".join(["One rater", f"{mO(lambda x: x['kappa_A_B']):.2f}", f"{mO(lambda x: x['instr']['crowd_A']['fpr']):.3f}",
                      pts(mO(lambda x: x['instr']['crowd_A']['placebo_depth_rel'] * x['instr']['crowd_A']['fpr'])), f"{_np.exp(mO(lambda x: x['instr']['crowd_A']['log_placebo_rr'])):.2f}",
                      pts(mO(lambda x: x['instr']['crowd_A']['depth'])), pts(mO(lambda x: x['instr']['crowd_A']['size'])), f"{_np.exp(mO(lambda x: x['instr']['crowd_A']['log_rr'])):.2f}"]) + " \\\\")
T6.append("\\midrule")
for nm, k, s5 in [("Haiku", "haiku", 0), ("Sonnet$^{\\dagger}$", "sonnet", 1), ("Qwen", "qwen7b_m", 0), ("Gemma$^{\\dagger}$", "gemma_m", 1)]:
    T6.append(lrow(nm, (L5 if s5 else Lc)[k], bool(s5)))
T6.append("\\midrule")
for nm, k in [("Detoxify (0.6)", "detoxify_original@0.6"), ("Detoxify", "detoxify_original@m"), ("Detoxify-unbiased$^{\\S}$", "detoxify_unbiased@m"), ("IMSYPP (argmax)", "imsypp@argmax"),
              ("IMSYPP", "imsypp@m"), ("s-nlp RoBERTa$^{\\S}$", "sn_rob@m"), ("martin-ha DistilBERT$^{\\S}$", "mh_dist@m"), ("Dynabench hate", "fb_hate@m"), ("CardiffNLP offensive", "cn_off@m")]:
    T6.append(lrow(nm, Lc[k]))
open(f"{T}/laws_v6.tex", "w").write("\n".join(T6) + "\n"); print("\n".join(T6[:6]))
# v7: Appendix table, all 14 classifier settings (x74_placebo.json): the claims "12 of the 14 settings report more contagion than gold"
CLS = [("Detoxify", "detoxify_original", "0.6"), ("Detoxify-unbiased$^{\\S}$", "detoxify_unbiased", "0.6"), ("IMSYPP", "imsypp", "argmax"),
       ("s-nlp RoBERTa$^{\\S}$", "sn_rob", "0.5"), ("martin-ha DistilBERT$^{\\S}$", "mh_dist", "0.5"), ("Dynabench hate", "fb_hate", "0.5"), ("CardiffNLP offensive", "cn_off", "0.5")]
TC = []; nup = 0
for nm, k, thr in CLS:
    for j, (lab, key) in enumerate(((thr, f"{k}@{thr}"), ("gold's rate", f"{k}@m"))):
        r = Lc[key]; d = r["law_rr_minus_gold"]; nup += d[1] > 0
        TC.append(" & ".join([nm if j == 0 else "", lab, f"{kap(r, PL['cc_threads']['gold_prev']):.2f}", f"{r['fpr'][0]:.3f}", pts(r["placebo_depth_slope"][0]), f"{r['placebo_rr'][0]:.2f}",
                              pts(r["law_depth_slope"][0]), pts(r["law_size_slope"][0]), f"{r['law_rr'][0]:.2f}", f"{sg(d[0], 2)} [{sg(d[1], 2)}, {sg(d[2], 2)}]"]) + " \\\\")
    TC.append("\\addlinespace[1pt]")
assert nup == 12
open(f"{T}/classifiers_v7.tex", "w").write("\n".join(TC[:-1]) + "\n"); print("\n".join(TC[:3]))
with open(f"{W}/paper_arxiv/numbers_v6_tables.tex", "w") as fo:                      # values the Table 1 caption needs
    fo.write(f"\\newcommand{{\\tOneGoldBDepth}}{{{pts(mO(lambda x: x['gold_B']['depth']))}}}\n\\newcommand{{\\tOneGoldBSize}}{{{pts(mO(lambda x: x['gold_B']['size']))}}}\n"
             f"\\newcommand{{\\tOneGoldBRR}}{{{_np.exp(mO(lambda x: x['gold_B']['log_rr'])):.2f}}}\n"
             f"\\newcommand{{\\tOneGoldSfiveDepth}}{{{pts(PL['cc_threads_s5']['gold']['depth_slope'][0])}}}\n\\newcommand{{\\tOneGoldSfiveRR}}{{{PL['cc_threads_s5']['gold']['rr'][0]:.2f}}}\n")
    fo.write(f"\\newcommand{{\\tOneSrPlacDepth}}{{{pts(mO(lambda x: x['instr']['crowd_A']['placebo_depth_rel'] * x['instr']['crowd_A']['fpr']))}}}\n")
    kk_ = sorted(["detoxify_original@0.6", "detoxify_original@m", "detoxify_unbiased@m", "imsypp@argmax", "imsypp@m", "sn_rob@m", "mh_dist@m", "fb_hate@m", "cn_off@m"], key=lambda k: -kap(Lc[k], PL["cc_threads"]["gold_prev"]))[:2]
    assert sorted(kk_) == ["detoxify_unbiased@m", "sn_rob@m"] and all(f"{kap(Lc[k], PL['cc_threads']['gold_prev']):.2f}" == "0.71" for k in kk_)
    fo.write(f"\\newcommand{{\\tOneTopKappa}}{{0.71}}\n\\newcommand{{\\tOneDetUnbRR}}{{{Lc['detoxify_unbiased@m']['law_rr'][0]:.2f}}}\n\\newcommand{{\\tOneSnRobRR}}{{{Lc['sn_rob@m']['law_rr'][0]:.2f}}}\n\\newcommand{{\\tOneHaikuRR}}{{{Lc['haiku']['law_rr'][0]:.2f}}}\n")
    fv = [Lc[k][x][0] for k in ("sn_rob@0.5", "mh_dist@0.5", "fb_hate@0.5", "cn_off@0.5") for x in ("placebo_rr",)]; rv = [Lc[k]["law_rr"][0] for k in ("sn_rob@0.5", "mh_dist@0.5", "fb_hate@0.5", "cn_off@0.5")]
    fo.write(f"\\newcommand{{\\tOneHalfPRRMin}}{{{min(fv):.2f}}}\n\\newcommand{{\\tOneHalfPRRMax}}{{{max(fv):.2f}}}\n\\newcommand{{\\tOneHalfRRMin}}{{{min(rv):.2f}}}\n\\newcommand{{\\tOneHalfRRMax}}{{{max(rv):.2f}}}\n")
# ---------------------------------------------------------------- tables/llms_v6.tex: the seven LLMs at gold's flag rate (x92_matched.json; 4+:0 ratios from x90_dose.json, x91_mitigate.json)
mt = json.load(open(f"{R}/x92/x92_matched.json")); rows_ = {(r["model"], r["op"]): r for r in mt["rows"]}
MT = DO["cc_s5"]["matched"]; S5L = DO["cc_s5"]["labelers"]; MI = json.load(open(f"{R}/x91/x91_mitigate.json"))["instruments"]
def ci3(v):
    return f"{sg(v[0])} [{sg(v[1])}, {sg(v[2])}]"
def mrow(name, r, ratio):
    return " & ".join([name, f(100 * r["flag_rate"], 1), f(r["kappa"], 2), f(ratio, 1), ci3(r["depth_excess"]), sg(r["depth_excess_fixed"][0]), sg(r["log_rr_excess"][0])]) + " \\\\"
fr_s = S5L["sonnet"]["fpr_by_count"]
LL = [mrow("Haiku", rows_[("haiku", "native")], MI["stored"]["ratio_4_0"][0]), mrow("Haiku-100", rows_[("haiku", "rating at gold prevalence")], MT["haiku"]["ratio_4_0"][0]),
      mrow("Sonnet", rows_[("sonnet", "native")], fr_s[4][0] / fr_s[0][0]), mrow("Sonnet-100", rows_[("sonnet", "rating at gold prevalence")], MT["sonnet"]["ratio_4_0"][0]), "\\midrule"]
for nm, k, kd in (("Qwen", "qwen", "qwen"), ("Gemma", "gemma", "gemma"), ("OLMo-2", "olmo2", "olmo2"), ("Granite", "granite", "granite"), ("Llama", "llama", "llama31")):
    LL.append(mrow(nm, rows_[(k, "gold prevalence")], MT[kd]["ratio_4_0"][0]))
open(f"{T}/llms_v6.tex", "w").write("\n".join(LL) + "\n"); print("\n".join(LL))
# ---------------------------------------------------------------- tables/platforms_v6.tex: Table 2 with one quantity per column
NMC = {"cc": "Civil Comments (toxicity)", "cad": "CAD (abuse)", "qr": "Qian et al. Reddit (hate)", "qg": "Qian et al. Gab (hate)", "cga": "CGA-Wiki (attack)", "bad": "BAD (unsafe turn)"}
lines = []
for name, Rr, tag in ROWS:
    I = Rr["instruments"]; h = I["haiku_tox"]; s = I.get("sonnet_tox") or (cc["cc_s5"]["instruments"]["sonnet_tox"] if tag == "cc" else None)
    d = I.get("ref:detoxify_original@m")
    gap = lambda v: f(100 * v["fpr_you_minus_other"][0]) if v else "--"
    cells = [NMC[tag], f"{100 * Rr['you_root']:.0f}\\%", f"{100 * Rr['you_deep']:.0f}\\%", gap(h), gap(s), gap(d), f"{sg(h['placebo'][0])} [{sg(h['placebo'][1])}, {sg(h['placebo'][2])}]",
             sg(WC[tag]["labelers"]["haiku"]["within_count"][0]), sg(Rr["gold"]["depth"][0]), sg(h["depth"][0])]
    lines.append(" & ".join(cells) + " \\\\")
open(f"{T}/platforms_v6.tex", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
