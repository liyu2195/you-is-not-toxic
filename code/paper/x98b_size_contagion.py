"""arXiv extended version, numbers agent B: size laws, contagion, the other corpora and the latent model (inventory items E05, E06,
E33, E35 (size-law, contagion and unit-share parts), E36 (contagion part), E37-E54).
Every number comes from a stored result file. Deterministic re-analyses of stored labels (seeded resampling) are:
  (1) the thread size laws of Haiku and Sonnet under the Civil Comments toxicity question on CAD, Qian et al.'s Reddit and Gab, CGA-Wiki
      and BAD, from results/x89/labels/<corpus>_<model>_ctx0 (the slope definition of scripts/x74_decomp.py, reproduced exactly on the
      stored gold and corpus-question labels before use; 2,000 conversation-cluster Poisson resamples, seed 9821);
  (2) the rewrite effect on gold-negative second-person comments of the Civil Comments 5% sample, from the stored labels of
      results/x77_youfree (the statistic of scripts/x89_rewrite_other.py; 2,000 article-cluster Poisson resamples, seed 9822);
  (3) the severity reading of B75d (scripts/x75_decomp.py) for the seventh classifier at gold's flag rate, Detoxify-unbiased, from
      results/x14_pops/cc_threads.parquet (no resampling; the six stored settings are reproduced to 1e-12 first);
  (4) Qwen's second-person false-positive gap on the 5% sample, from the stored rates and counts of results/x90/x90_dose.json.
Writes
  paper_arxiv/numbers_arxiv_b.tex     LaTeX macros, prefix \\xb (letters only)
  paper_arxiv/tables/xb_*.tex          table bodies (rows only)
  paper_arxiv/provenance_xb.md         value, source and key, transformation, cross-check and caveats of every macro and table
Rules applied throughout:
  - current populations only (results/x14_pops: cc_threads 1,251,487 comments and 980,520 reply edges; its 5% sample 60,668); f96's
    article part (old train-split sections) is not used, its thread part (270,967 threads) only as a replicate of the shuffle test;
  - no pooled "+ parent" value: a labeler shown the parent labels roots without it and replies with it, so a law pooled over both
    carries the root/reply prompt switch. Size laws, unit shares, the parent-flag many-rater check, the construct-only escalation test
    and the within-thread law are therefore given without the parent only; with the parent, only statistics among replies are used
    (depth among replies; the many-rater check on the parent's gold label; the contrast effect; f122's reply RR, whose parent label is
    the context-free one);
  - nothing from the keep-out list K1-K30 (in particular not results/x82/x82_latent.json, K5; the raw within-article Mantel-Haenszel
    comparisons, K4; x80_noise's reweighting, K2; the free-f construct test, K8; x74_crowd's split halves, K9);
  - classifier ranges cover the 14 settings (Detoxify-unbiased included wherever it is stored, which is everywhere used here);
  - where an earlier-draft macro (paper_v5/numbers_*.tex except numbers_v2.tex) holds the same number, chk() asserts equality at the earlier-draft macro's
    precision or records the explained difference in the provenance file.
usage: python3 -I scripts/x98b_size_contagion.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, glob, os, re, math
import numpy as np, pandas as pd

W = _ROOT; RES = f"{W}/results"; PA = f"{W}/paper_arxiv"
OUT_NUM = f"{PA}/numbers_arxiv_b.tex"; OUT_TAB = f"{PA}/tables"; OUT_PROV = f"{PA}/provenance_xb.md"
for p_ in (OUT_NUM, OUT_TAB + "/x", OUT_PROV):
    assert os.path.abspath(p_).startswith(PA + "/"), p_                                    # write nowhere else
os.makedirs(OUT_TAB, exist_ok=True)
PP = 100.0


def load(p):
    return json.load(open(f"{RES}/{p}"))


# ---------------------------------------------------------------- formatting (as scripts/x18_numbers_v5.py)
def sg(x, d=2):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"                                                                    # no signed zero
    return s.replace("-", "$-$") if s.startswith("-") else s


def f(x, d=2):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pct(x, d=0):
    v = round(100 * x + 1e-9, d)
    return (f"{v:.{d}f}" if d else f"{int(round(100 * x + 1e-9))}") + "\\%"


def spct(x, d=0):                                                                              # signed percentage: +6\%, $-$7\%
    return sg(100 * x, d) + "\\%"


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def f2(x):
    return f"{x:.2f}"


def pv(p):                                                                                     # p-values: 0.20, 0.017, $<$0.001
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def pvd(p, tau):                                                                              # p-value, marked when the trend falls
    return pv(p) + ("$^{\\downarrow}$" if (p < 0.05 and tau < 0) else "")


def zf(z):
    return f(z, 1)


def ci3(v, fn=sg):
    return f"{fn(v[0])} [{fn(v[1])}, {fn(v[2])}]"


def sci(x):                                                                                    # 9.8e-09 -> $9.8\times 10^{-9}$
    e_ = int(math.floor(math.log10(x))); return f"${x / 10 ** e_:.1f}\\times 10^{{{e_}}}$"


def num(s):                                                                                    # a formatted macro back to a number
    t = s.replace("$-$", "-").replace("\\%", "").replace("{,}", "").replace("$<$", "").replace("<", "").strip()
    return float(t)


def ndec(s):
    t = s.replace("$-$", "-").replace("\\%", "").replace("{,}", "").strip()
    return len(t.split(".")[1]) if "." in t else 0


# ---------------------------------------------------------------- the earlier-draft macros, for cross-checks
EX = {}
for fn_ in sorted(glob.glob(f"{W}/results/reference/numbers_*.tex")):
    if os.path.basename(fn_) == "numbers_v2.tex":                                              # the old 1.78M population
        continue
    for l in open(fn_):
        m_ = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", l)
        if m_:
            EX[m_.group(1)] = (m_.group(2), os.path.basename(fn_))

# ---------------------------------------------------------------- registry of macros, cross-checks, discrepancies and tables
M, PROV, SECS, DISC, TABLES, DISC_EX = {}, {}, [], [], [], []


def sec(title):
    SECS.append((title, []))


def put(name, val, src, how, cav=""):
    assert re.fullmatch(r"xb[A-Z][A-Za-z]*", name), name
    assert name not in M, name
    assert isinstance(val, str) and "nan" not in val.lower(), (name, val)
    M[name] = val; PROV[name] = dict(src=src, how=how, chk=[], cav=cav); SECS[-1][1].append(name)


def put3(name, v, fn, src, how, cav=""):
    """name, nameLo, nameHi from a triple [estimate, lower, upper]."""
    put(name, fn(v[0]), src, how, cav)
    put(name + "Lo", fn(v[1]), "as \\" + name, f"lower end of the 95% CI of \\{name}")
    put(name + "Hi", fn(v[2]), "as \\" + name, f"upper end of the 95% CI of \\{name}")


def chk(name, ex, note=None, scale=1.0):
    """Assert that macro `name` equals the earlier-draft macro `ex` at the earlier-draft macro's precision, or record the explained difference."""
    assert ex in EX, ex
    a, (b, fb) = M[name], EX[ex]
    same = a == b
    if not same:
        try:
            same = round(num(a) * scale, ndec(b)) == round(num(b), ndec(b)) or abs(num(a) * scale - num(b)) < 0.5 * 10 ** (-ndec(b)) + 1e-12
        except ValueError:
            same = False
    if same:
        PROV[name]["chk"].append(f"equals \\{ex} = {b} ({fb})")
    else:
        assert note, f"{name} = {a} differs from \\{ex} = {b} ({fb}) without an explanation"
        PROV[name]["chk"].append(f"DIFFERS from \\{ex} = {b} ({fb}): {note}")
        DISC.append(f"\\{name} = {a} against \\{ex} = {b} ({fb}): {note}"); DISC_EX.append(ex)


def note(name, text):
    PROV[name]["chk"].append(text)


def table(name, caption, headers, rows, cols, cav=""):
    TABLES.append(dict(name=name, caption=caption, headers=headers, rows=rows, cols=cols, cav=cav))
    with open(f"{OUT_TAB}/{name}.tex", "w") as fo:
        fo.write("\n".join(rows) + "\n")                                                     # rows only, as paper_v5/tables/laws_v6.tex


# ---------------------------------------------------------------- labelers: the 14 classifier settings and display names
FAM = [("Detoxify", "detoxify_original", "0.6", False), ("Detoxify-unbiased", "detoxify_unbiased", "0.6", True), ("IMSYPP", "imsypp", "argmax", False),
       ("s-nlp RoBERTa", "sn_rob", "0.5", True), ("martin-ha DistilBERT", "mh_dist", "0.5", True), ("Dynabench hate", "fb_hate", "0.5", False),
       ("CardiffNLP offensive", "cn_off", "0.5", False)]
CLS14 = [f"{k}@{t}" for _, k, t, _ in FAM] + [f"{k}@m" for _, k, _, _ in FAM]
CLS12 = [c for c in CLS14 if not c.startswith("detoxify_unbiased")]
CLSM = [f"{k}@m" for _, k, _, _ in FAM]                                                          # the 7 settings at gold's flag rate
assert len(CLS14) == 14 and len(CLS12) == 12 and len(set(CLS14)) == 14


def fam_name(k, sect=True):
    for nm, kk, _, cc in FAM:
        if kk == k:
            return nm + ("$^{\\S}$" if (cc and sect) else "")
    raise KeyError(k)


def thr_name(setting):
    k, t = setting.split("@")
    return "gold's rate" if t == "m" else t


PROV_14 = "over the 14 classifier settings (Detoxify-unbiased included)"

# ================================================================ A. Size laws on Civil Comments (E05, E06; E35 size-law and unit-share parts)
sec("A. Size laws on Civil Comments (E05, E06; E35 size-law and unit-share parts)")
PLJ = load("x74_placebo/x74_placebo.json"); PL = PLJ["cc_threads"]; PL5 = PLJ["cc_threads_s5"]
assert PL["n_items"] == 1251487 and PL["n_edges"] == 980520 and PL["n_units"] == 270967 and PL5["n_items"] == 60668   # current population
G, L, G5, L5 = PL["gold"], PL["labelers"], PL5["gold"], PL5["labelers"]
G_PREV = PL["gold_prev"]


def avj(P_, lev):                                                                               # Avalle et al.'s tests: both runs of x74_placebo
    A_ = dict(P_["avalle"][lev])
    for k_, v_ in P_["avalle_added"][lev].items():
        if k_ in A_ and isinstance(v_, dict):
            assert abs(A_[k_]["slope"] - v_["slope"]) < 1e-12                                # the same gold in both runs
        A_.setdefault(k_, v_)
    return A_


AVT, AVA, AVT5, AVA5 = avj(PL, "threads"), avj(PL, "articles"), avj(PL5, "threads"), avj(PL5, "articles")
assert AVT["n_conversations"] == 270967 and AVT["n_comments"] == 1251487 and AVA["n_comments"] == 1251487 and AVA["n_conversations"] == 57769
assert AVT5["n_comments"] == 60668 and AVA5["n_comments"] == 60668
for k_ in ["haiku", "qwen7b_m"] + CLS14:                                                        # the tests' slopes are Table 1's laws
    assert abs(AVT[k_]["slope"] - L[k_]["law_size_slope"][0]) < 1e-12 and abs(AVA[k_]["slope"] - L[k_]["law_article_size_slope"][0]) < 1e-12
assert abs(AVT["z"]["slope"] - G["size_slope"][0]) < 1e-12 and abs(AVA["z"]["slope"] - G["article_size_slope"][0]) < 1e-12
assert abs(AVT5["sonnet"]["slope"] - L5["sonnet"]["law_size_slope"][0]) < 1e-12
SRC_PL = "results/x74_placebo/x74_placebo.json"
pp = lambda x: sg(PP * x)
put3("xbGoldThreadSize", G["size_slope"], pp, SRC_PL, "cc_threads/gold/size_slope x 100 (points per doubling; OLS slope of a reply thread's toxic share on log2 of its size, threads with >= 2 comments; 95% CI from the stored article-cluster resamples)")
chk("xbGoldThreadSize", "wGoldSizePP")
put3("xbGoldArtSize", G["article_size_slope"], pp, SRC_PL, "cc_threads/gold/article_size_slope x 100 (unit = an article's comment section within the reply-thread population)")
chk("xbGoldArtSize", "wGoldArtPP")
put("xbGoldArtTauLog", f2(AVA["z"]["mk_tau"]), SRC_PL, "cc_threads/avalle/articles/z/mk_tau: Kendall tau of the log2-bin means of the sections' toxic shares (bins with >= 30 sections)")
put("xbGoldArtTauTwentyOne", f2(AVA["z"]["bins21"]["mk_tau"]), SRC_PL, "cc_threads/avalle/articles/z/bins21/mk_tau: Kendall tau over Avalle et al.'s 21 log-spaced size bins (bins with >= 30 sections)")
chk("xbGoldArtTauTwentyOne", "gArtTau")
for nm_, k_ in (("Haiku", "haiku"), ("Qwen", "qwen7b_m")):
    r_ = L[k_]
    put3(f"xb{nm_}ThreadSize", r_["law_size_slope"], pp, SRC_PL, f"cc_threads/labelers/{k_}/law_size_slope x 100")
    put3(f"xb{nm_}ArtSize", r_["law_article_size_slope"], pp, SRC_PL, f"cc_threads/labelers/{k_}/law_article_size_slope x 100")
    put(f"xb{nm_}ThreadX", f(r_["law_size_slope"][0] / G["size_slope"][0], 1), SRC_PL, f"law_size_slope of {k_} / gold's size_slope (point estimates)")
    put(f"xb{nm_}ArtX", f(r_["law_article_size_slope"][0] / G["article_size_slope"][0], 1), SRC_PL, f"law_article_size_slope of {k_} / gold's article_size_slope")
chk("xbHaikuThreadSize", "qMsHaikuThread"); chk("xbHaikuArtSize", "qMsHaikuArt"); chk("xbHaikuThreadX", "pLSHaiku"); chk("xbHaikuArtX", "pLAHaiku")
chk("xbQwenThreadX", "pLSQwen"); chk("xbQwenArtX", "pLAQwen")
put3("xbSonnetThreadSize", L5["sonnet"]["law_size_slope"], pp, SRC_PL, "cc_threads_s5/labelers/sonnet/law_size_slope x 100 (5% sample)")
put3("xbSonnetArtSize", L5["sonnet"]["law_article_size_slope"], pp, SRC_PL, "cc_threads_s5/labelers/sonnet/law_article_size_slope x 100 (5% sample)")
chk("xbSonnetThreadSize", "qMsSonnetThread"); chk("xbSonnetArtSize", "qMsSonnetArt")
assert G5["size_slope"][1] < 0 < G5["size_slope"][2] and G5["article_size_slope"][1] < 0 < G5["article_size_slope"][2]       # the 5% sample's gold laws: CIs cover 0
put3("xbGoldSfiveThreadSize", G5["size_slope"], pp, SRC_PL, "cc_threads_s5/gold/size_slope x 100 (5% sample)", "its CI covers 0, so no ratio to it is given")
put3("xbGoldSfiveArtSize", G5["article_size_slope"], pp, SRC_PL, "cc_threads_s5/gold/article_size_slope x 100 (5% sample)", "its CI covers 0")
ts_ = {k: L[k]["law_size_slope"][0] for k in CLS14}; as_ = {k: L[k]["law_article_size_slope"][0] for k in CLS14}
put("xbClsThreadSizeMin", pp(min(ts_.values())), SRC_PL, f"min of law_size_slope x 100 {PROV_14} ({min(ts_, key=ts_.get)})")
put("xbClsThreadSizeMax", pp(max(ts_.values())), SRC_PL, f"max of law_size_slope x 100 {PROV_14} ({max(ts_, key=ts_.get)})")
put("xbClsArtSizeMin", pp(min(as_.values())), SRC_PL, f"min of law_article_size_slope x 100 {PROV_14} ({min(as_, key=as_.get)})")
put("xbClsArtSizeMax", pp(max(as_.values())), SRC_PL, f"max of law_article_size_slope x 100 {PROV_14} ({max(as_, key=as_.get)})")
ax_ = {k: v / G["article_size_slope"][0] for k, v in as_.items()}
put("xbClsArtXMin", f(min(ax_.values()), 1), SRC_PL, f"min over the 14 settings of the article size law / gold's ({min(ax_, key=ax_.get)})")
put("xbClsArtXMax", f(max(ax_.values()), 1), SRC_PL, f"max over the 14 settings of the article size law / gold's ({max(ax_, key=ax_.get)})")
put("xbClsThreadSignN", str(sum((ts_[k] > 0) == (G["size_slope"][0] > 0) for k in CLS14)), SRC_PL, "number of the 14 settings whose thread size law has gold's sign (point estimates)")
chk("xbClsThreadSignN", "vClsThreadSizeSign")
ad_ = {k: L[k]["distortion"]["article_size"]["ci"] for k in CLS14}
put("xbClsArtAboveN", str(sum(v[0] > 0 for v in ad_.values())), SRC_PL, "number of the 14 settings whose article-size distortion (law minus gold's, labelers/<k>/distortion/article_size/ci) has a 95% CI above 0")
put("xbClsArtBelowN", str(sum(v[1] < 0 for v in ad_.values())), SRC_PL, "number of the 14 settings whose article-size distortion has a 95% CI below 0")
put("xbDetoxSixArtX", f(ax_["detoxify_original@0.6"], 1), SRC_PL, "Detoxify at 0.6: article size law / gold's (one of Avalle et al.'s two robustness instruments)")
put("xbImsArgArtX", f(ax_["imsypp@argmax"], 1), SRC_PL, "IMSYPP at argmax: article size law / gold's (Avalle et al.'s other robustness instrument)")
put("xbAvalleArtRatio", f"{ax_['imsypp@argmax'] / ax_['detoxify_original@0.6']:.0f}", SRC_PL, f"IMSYPP-at-argmax article law / Detoxify-at-0.6 article law (unrounded {ax_['imsypp@argmax'] / ax_['detoxify_original@0.6']:.2f})",
    "the two checks disagree about tenfold; the unrounded ratio is given in the transformation")
chk("xbDetoxSixArtX", "qAvDetoxArt"); chk("xbImsArgArtX", "qAvImsArt"); chk("xbAvalleArtRatio", "qAvRatio")
# ---- the human instrument matched to each labeler's flag rate (x83, B83c)
MS = load("x83/x83_matched_size.json"); SRC_MS = "results/x83/x83_matched_size.json"
assert MS["cc_threads"]["N"] == 1251487 and MS["cc_s5"]["N"] == 60668
MH, MS5 = MS["cc_threads"]["instruments"]["haiku"], MS["cc_s5"]["instruments"]["sonnet"]
assert abs(MH["thread_size"][0] - L["haiku"]["law_size_slope"][0]) < 1e-12 and abs(MS5["thread_size"][0] - L5["sonnet"]["law_size_slope"][0]) < 1e-12
assert abs(MS["cc_threads"]["gold"]["thread_size"][0] - G["size_slope"][0]) < 1e-12
HUMNOTE = "the matched human instrument flags the round(qN) comments with the highest crowd vote fractions (ties at random, seed 8000 + j), q = the labeler's flag rate"
put("xbHaikuFlag", pct(MH["flag_rate"]), SRC_MS, "cc_threads/instruments/haiku/flag_rate"); chk("xbHaikuFlag", "qMatchHaikuFlag")
put("xbSonnetFlag", pct(MS5["flag_rate"]), SRC_MS, "cc_s5/instruments/sonnet/flag_rate (5% sample)"); chk("xbSonnetFlag", "qMatchSonnetFlag")
for nm_, r_, pop_ in (("Haiku", MH, "cc_threads"), ("Sonnet", MS5, "cc_s5")):
    put3(f"xbHum{nm_}ThreadSize", r_["thread_size_human"], pp, SRC_MS, f"{pop_}/instruments/{nm_.lower()}/thread_size_human x 100; {HUMNOTE}")
    put3(f"xbHum{nm_}ArtSize", r_["article_size_human"], pp, SRC_MS, f"{pop_}/instruments/{nm_.lower()}/article_size_human x 100")
    put3(f"xb{nm_}HumThreadDiff", r_["thread_size_minus_human"], pp, SRC_MS, f"{pop_}/instruments/{nm_.lower()}/thread_size_minus_human x 100 (paired over 500 article-cluster resamples, seed 8001)")
    put3(f"xb{nm_}HumArtDiff", r_["article_size_minus_human"], pp, SRC_MS, f"{pop_}/instruments/{nm_.lower()}/article_size_minus_human x 100 (paired)")
chk("xbHumHaikuThreadSize", "qMsHaikuThreadHum"); chk("xbHumHaikuArtSize", "qMsHaikuArtHum"); chk("xbHumSonnetThreadSize", "qMsSonnetThreadHum"); chk("xbHumSonnetArtSize", "qMsSonnetArtHum")
put("xbHumHaikuThreadX", f(MH["thread_size_human"][0] / G["size_slope"][0], 1), SRC_MS, "Haiku-matched human instrument's thread size law / gold's (thread_size_human_ratio_to_gold)")
put("xbHumHaikuArtX", f(MH["article_size_human"][0] / G["article_size_slope"][0], 1), SRC_MS, "Haiku-matched human instrument's article size law / gold's")
put("xbHaikuOverHumThreadX", f(MH["thread_size"][0] / MH["thread_size_human"][0], 1), SRC_MS, "Haiku's thread size law / its matched human instrument's")
put("xbHaikuOverHumArtX", f(MH["article_size"][0] / MH["article_size_human"][0], 1), SRC_MS, "Haiku's article size law / its matched human instrument's")
chk("xbHumHaikuThreadX", "qMsHumThreadX"); chk("xbHaikuOverHumThreadX", "qMsHaikuThreadX")
# ---- unit shares (E35's unit-share part; E62 is not in this script)
UG = MS["cc_threads"]["gold"]["unit_share"]
put("xbUnitGold", f2(UG[0]), SRC_MS, "cc_threads/gold/unit_share: share of reply threads with at least one toxic comment"); chk("xbUnitGold", "qUnitGold")
put("xbUnitHaiku", f2(MH["unit_share"][0]), SRC_MS, "cc_threads/instruments/haiku/unit_share"); chk("xbUnitHaiku", "qMsHaikuUnit")
put("xbUnitHumHaiku", f2(MH["unit_share_human"][0]), SRC_MS, "cc_threads/instruments/haiku/unit_share_human"); chk("xbUnitHumHaiku", "qMsHaikuUnitHum")
put3("xbUnitHaikuHumDiff", MH["unit_share_minus_human"], lambda x: sg(PP * x, 1), SRC_MS, "cc_threads/instruments/haiku/unit_share_minus_human x 100 (percentage points of reply threads; paired)")
put("xbUnitSonnet", f2(MS5["unit_share"][0]), SRC_MS, "cc_s5/instruments/sonnet/unit_share (5% sample)"); chk("xbUnitSonnet", "qMsSonnetUnit")
put("xbUnitHumSonnet", f2(MS5["unit_share_human"][0]), SRC_MS, "cc_s5/instruments/sonnet/unit_share_human"); chk("xbUnitHumSonnet", "qMsSonnetUnitHum")
put("xbUnitGoldSfive", f2(MS["cc_s5"]["gold"]["unit_share"][0]), SRC_MS, "cc_s5/gold/unit_share"); chk("xbUnitGoldSfive", "qMsGoldUnitSfive")
ucm = {k: MS["cc_threads"]["instruments"][k]["unit_share"][0] for k in CLSM}
put("xbUnitClsGpMin", f2(min(ucm.values())), SRC_MS, "min unit_share over the 7 classifier settings at gold's flag rate (@m)")
put("xbUnitClsGpMax", f2(max(ucm.values())), SRC_MS, "max unit_share over the 7 classifier settings at gold's flag rate (@m)")
chk("xbUnitClsGpMin", "qUnitClsMin")
chk("xbUnitClsGpMax", "qUnitClsMax")
# ---- Avalle et al.'s tests: Mann-Kendall over 21 log bins and over log2 bins, and the 200-shuffle z (x74_placebo; f96 as replicate)
TESTN = "(x74_placebo's avalle(), which reruns f96's pipeline on the current population: conversations with >= 2 comments, 200 label shuffles, seed 96)"
put("xbMKTwentyOneBinsThread", str(AVT["z"]["bins21"]["n_bins"]), SRC_PL, "cc_threads/avalle/threads/z/bins21/n_bins: of Avalle et al.'s 21 log-spaced size bins, those with >= 30 threads")
chk("xbMKTwentyOneBinsThread", "qMKBins")
put("xbMKLogBinsThread", str(len(AVT["z"]["bin_means"])), SRC_PL, "number of log2 size bins (floor(log2 size)) with >= 30 threads: len(cc_threads/avalle/threads/z/bin_means)")
put("xbMKTwentyOneBinsArt", str(AVA["z"]["bins21"]["n_bins"]), SRC_PL, "cc_threads/avalle/articles/z/bins21/n_bins")
put("xbMKLogBinsArt", str(len(AVA["z"]["bin_means"])), SRC_PL, "number of log2 bins with >= 30 sections")
for nm_, k_ in (("Gold", "z"), ("Haiku", "haiku"), ("Qwen", "qwen7b_m")):
    put(f"xbMKTwentyOne{nm_}", pv(AVT[k_]["bins21"]["mk_p"]), SRC_PL, f"cc_threads/avalle/threads/{k_}/bins21/mk_p (Mann-Kendall p, 21 log bins) {TESTN}")
    put(f"xbMKLog{nm_}", pv(AVT[k_]["mk_p"]), SRC_PL, f"cc_threads/avalle/threads/{k_}/mk_p (Mann-Kendall p, log2 bins)")
    put(f"xbShufZ{nm_}", zf(AVT[k_]["z_vs_shuffle"]), SRC_PL, f"cc_threads/avalle/threads/{k_}/z_vs_shuffle: (slope - shuffle mean) / shuffle SD")
    put(f"xbMKArtTwentyOne{nm_}", pv(AVA[k_]["bins21"]["mk_p"]), SRC_PL, f"cc_threads/avalle/articles/{k_}/bins21/mk_p")
    put(f"xbMKArtLog{nm_}", pv(AVA[k_]["mk_p"]), SRC_PL, f"cc_threads/avalle/articles/{k_}/mk_p")
    put(f"xbShufZArt{nm_}", zf(AVA[k_]["z_vs_shuffle"]), SRC_PL, f"cc_threads/avalle/articles/{k_}/z_vs_shuffle")
chk("xbMKTwentyOneGold", "qMKPGold"); chk("xbMKTwentyOneHaiku", "qMKPHaiku"); chk("xbMKTwentyOneQwen", "qMKPQwen")
put("xbMKTwentyOneGoldTau", f2(AVT["z"]["bins21"]["mk_tau"]), SRC_PL, "cc_threads/avalle/threads/z/bins21/mk_tau"); chk("xbMKTwentyOneGoldTau", "bThrTauGold")
put("xbMKLogGoldTau", sg(AVT["z"]["mk_tau"]), SRC_PL, "cc_threads/avalle/threads/z/mk_tau (log2 bins)")
put("xbMKLogHaikuTau", sg(AVT["haiku"]["mk_tau"]), SRC_PL, "cc_threads/avalle/threads/haiku/mk_tau (log2 bins)")
put("xbShufSigGold", "yes" if AVT["z"]["significant_increase"] else "no", SRC_PL, "cc_threads/avalle/threads/z/significant_increase (slope >= shuffle mean + 2 SD)")
# the 5% sample: gold and Sonnet
for nm_, k_ in (("GoldSfive", "z"), ("Sonnet", "sonnet")):
    put(f"xbMKTwentyOne{nm_}", pv(AVT5[k_]["bins21"]["mk_p"]), SRC_PL, f"cc_threads_s5/avalle/threads/{k_}/bins21/mk_p (5% sample)")
    put(f"xbMKLog{nm_}", pv(AVT5[k_]["mk_p"]), SRC_PL, f"cc_threads_s5/avalle/threads/{k_}/mk_p (5% sample)")
    put(f"xbShufZ{nm_}", zf(AVT5[k_]["z_vs_shuffle"]), SRC_PL, f"cc_threads_s5/avalle/threads/{k_}/z_vs_shuffle (5% sample)")
assert AVT5["z"]["bins21"]["mk_p"] < 0.05 and AVT5["z"]["bins21"]["mk_tau"] > 0 and AVT5["z"]["mk_p"] >= 0.05 and not AVT5["z"]["significant_increase"]
note("xbMKTwentyOneGoldSfive", "on the 5% sample gold's thread law is detected by the 21-bin test but not by the log2-bin test or the shuffle test: the binned test's verdict depends on the binning")
put("xbMKTwentyOneBinsSfive", str(AVT5["z"]["bins21"]["n_bins"]), SRC_PL, "cc_threads_s5/avalle/threads/z/bins21/n_bins")
# f96 (thread part only; 270,967 threads = the current population) as a replicate of the shuffle test
F96 = load("f96_avalle.json"); SRC_F96 = "results/f96_avalle.json"
assert F96["threads"]["n_conversations"] == 270967 and F96["threads"]["n_comments"] == 1251487
assert F96["articles"]["n_comments"] == 1061408                                                # old train-split sections: not used
for k96, kpl in (("Z", "z"), ("haiku", "haiku"), ("haiku_ctx", "haiku_ctx")):
    assert abs(F96["threads"][k96]["slope"] - AVT[kpl]["slope"]) < 1e-12 and abs(F96["threads"][k96]["mk_p"] - AVT[kpl]["mk_p"]) < 1e-12   # deterministic parts agree
put("xbShufZGoldRep", zf(F96["threads"]["Z"]["z_vs_shuffle"]), SRC_F96, "threads/Z/z_vs_shuffle (f96's own 200 shuffles on the same 270,967 threads)",
    "a second shuffle draw of the same test; it differs from \\xbShufZGold only by the shuffles' Monte Carlo error")
put("xbShufZHaikuRep", zf(F96["threads"]["haiku"]["z_vs_shuffle"]), SRC_F96, "threads/haiku/z_vs_shuffle (f96)")
# classifier counts over the 14 settings (threads and articles)
sigT = [k for k in CLS14 if AVT[k]["significant_increase"]]
put("xbClsShufSigN", str(len(sigT)), SRC_PL, "number of the 14 settings whose thread law the shuffle test flags as a significant increase (cc_threads/avalle(_added)/threads/<k>/significant_increase)")
up21 = [k for k in CLS14 if AVT[k]["bins21"]["mk_p"] < 0.05 and AVT[k]["bins21"]["mk_tau"] > 0]; dn21 = [k for k in CLS14 if AVT[k]["bins21"]["mk_p"] < 0.05 and AVT[k]["bins21"]["mk_tau"] < 0]
put("xbClsMKTwentyOneUpN", str(len(up21)), SRC_PL, f"number of the 14 settings whose 21-bin Mann-Kendall test finds a significant increase of the thread law (p < 0.05, tau > 0): {', '.join(up21)}")
put("xbClsMKTwentyOneDownN", str(len(dn21)), SRC_PL, f"... a significant decrease (p < 0.05, tau < 0): {', '.join(dn21)}")
upL = [k for k in CLS14 if AVT[k]["mk_p"] < 0.05 and AVT[k]["mk_tau"] > 0]
put("xbClsMKLogUpN", str(len(upL)), SRC_PL, f"number of the 14 settings whose log2-bin Mann-Kendall test finds a significant increase of the thread law: {', '.join(upL)}")
artsig = [k for k in CLS14 if AVA[k]["bins21"]["mk_p"] < 0.05 and AVA[k]["bins21"]["mk_tau"] > 0]
put("xbClsArtMKUpN", str(len(artsig)), SRC_PL, "number of the 14 settings whose 21-bin test finds a significant increase of the article law")
put("xbClsArtShufSigN", str(sum(AVA[k]["significant_increase"] for k in CLS14)), SRC_PL, "number of the 14 settings whose article law the shuffle test flags")
# ---- table xb_size
def size_cells(av_th, av_ar, th, ar, xg_th=None, xg_ar=None):
    return [th, xg_th or "--", pvd(av_th["bins21"]["mk_p"], av_th["bins21"]["mk_tau"]), pvd(av_th["mk_p"], av_th["mk_tau"]), zf(av_th["z_vs_shuffle"]),
            ar, xg_ar or "--", pvd(av_ar["bins21"]["mk_p"], av_ar["bins21"]["mk_tau"]), pvd(av_ar["mk_p"], av_ar["mk_tau"]), zf(av_ar["z_vs_shuffle"])]


ppci = lambda v: ci3([PP * t for t in v])
TS = [" & ".join(["Gold"] + size_cells(AVT["z"], AVA["z"], ppci(G["size_slope"]), ppci(G["article_size_slope"]), "1.0", "1.0")) + " \\\\"]
for nm_, k_ in (("Haiku", "haiku"), ("Qwen", "qwen7b_m")):
    r_ = L[k_]
    TS.append(" & ".join([nm_] + size_cells(AVT[k_], AVA[k_], ppci(r_["law_size_slope"]), ppci(r_["law_article_size_slope"]),
                                              f(r_["law_size_slope"][0] / G["size_slope"][0], 1), f(r_["law_article_size_slope"][0] / G["article_size_slope"][0], 1))) + " \\\\")
TS.append(" & ".join(["Vote-fraction instrument, Haiku's rate"] + [ppci(MH["thread_size_human"]), f(MH["thread_size_human"][0] / G["size_slope"][0], 1), "--", "--", "--",
                                                    ppci(MH["article_size_human"]), f(MH["article_size_human"][0] / G["article_size_slope"][0], 1), "--", "--", "--"]) + " \\\\")
TS.append("\\midrule")
TS.append(" & ".join(["Gold$^{\\dagger}$"] + size_cells(AVT5["z"], AVA5["z"], ppci(G5["size_slope"]), ppci(G5["article_size_slope"]))) + " \\\\")
TS.append(" & ".join(["Sonnet$^{\\dagger}$"] + size_cells(AVT5["sonnet"], AVA5["sonnet"], ppci(L5["sonnet"]["law_size_slope"]), ppci(L5["sonnet"]["law_article_size_slope"]))) + " \\\\")
TS.append(" & ".join(["Vote-fraction instrument, Sonnet's rate$^{\\dagger}$"] + [ppci(MS5["thread_size_human"]), "--", "--", "--", "--", ppci(MS5["article_size_human"]), "--", "--", "--", "--"]) + " \\\\")
TS.append("\\midrule")
for nm_, k_, t_, cc_ in FAM:
    a_, b_ = f"{k_}@{t_}", f"{k_}@m"; two = lambda fn: f"{fn(a_)} / {fn(b_)}"
    TS.append(" & ".join([f"{fam_name(k_)} ({t_} / gold's rate)",
                          two(lambda s: pp(L[s]["law_size_slope"][0])), two(lambda s: f(L[s]["law_size_slope"][0] / G["size_slope"][0], 1)),
                          two(lambda s: pvd(AVT[s]["bins21"]["mk_p"], AVT[s]["bins21"]["mk_tau"])), two(lambda s: pvd(AVT[s]["mk_p"], AVT[s]["mk_tau"])), two(lambda s: zf(AVT[s]["z_vs_shuffle"])),
                          two(lambda s: pp(L[s]["law_article_size_slope"][0])), two(lambda s: f(L[s]["law_article_size_slope"][0] / G["article_size_slope"][0], 1)),
                          two(lambda s: pvd(AVA[s]["bins21"]["mk_p"], AVA[s]["bins21"]["mk_tau"])), two(lambda s: pvd(AVA[s]["mk_p"], AVA[s]["mk_tau"])), two(lambda s: zf(AVA[s]["z_vs_shuffle"]))]) + " \\\\")
table("xb_size",
      "Size laws on Civil Comments: the reply-thread size law and the article size law (points per doubling of the unit's size, 95\\% CI), each as a multiple of gold's, "
      "and Avalle et al.'s two tests of a rising law: the Mann--Kendall trend test over their 21 logarithmic size bins and over log$_2$ size bins (bins with at least 30 units), "
      "and the z-score of the slope against 200 shuffles of the labels ($^{\\downarrow}$: a significantly falling trend). Haiku, Sonnet and the human instruments flag at their own rate, the classifiers at their own "
      "threshold and at gold's flag rate (left and right of each slash). The human instrument flags the comments with the most toxic votes, as many as the labeler. "
      "Labelers shown the parent comment are not listed, because they label top-level comments without it. $^{\\dagger}$5\\% sample. $^{\\S}$Trained at least in part on Civil Comments labels.",
      ["Labeler", "Thread size law [95% CI]", "x gold", "MK p, 21 bins", "MK p, log2 bins", "Shuffle z", "Article size law [95% CI]", "x gold", "MK p, 21 bins", "MK p, log2 bins", "Shuffle z"],
      TS, [("Thread/article size law and CI", f"{SRC_PL}: gold size_slope / article_size_slope; labelers law_size_slope / law_article_size_slope (x 100); human rows {SRC_MS} thread_size_human / article_size_human"),
           ("x gold", "law / gold's law (point estimates); not given on the 5% sample, where gold's laws have CIs covering 0"),
           ("MK p, 21 bins", f"{SRC_PL}: avalle(_added)/<level>/<labeler>/bins21/mk_p"), ("MK p, log2 bins", "avalle(_added)/<level>/<labeler>/mk_p"),
           ("Shuffle z", "avalle(_added)/<level>/<labeler>/z_vs_shuffle"), ("Human rows", "tests not stored for the matched human instruments: --")],
      "The + parent variants are omitted: no size law among replies is stored, and the pooled ones carry the root/reply prompt switch. "
      "Haiku's matched human row uses x83's article-cluster CIs (500 resamples, seed 8001), the other rows x74_placebo's.")

# ================================================================ B. Contagion on Civil Comments (E33, E35 contagion part, E36 contagion part, E37, E38, E39, E42, E43, E44)
sec("B. Contagion on Civil Comments (E33, E35 contagion part, E36 contagion part, E37-E39, E42-E44)")
put3("xbGoldRR", G["rr"], f2, SRC_PL, "cc_threads/gold/rr: the reply risk ratio P(Z_c = 1 | Z_p = 1) / P(Z_c = 1 | Z_p = 0) over the 980,520 reply edges")
chk("xbGoldRR", "gRR"); chk("xbGoldRRLo", "gRRLo"); chk("xbGoldRRHi", "gRRHi"); chk("xbGoldRR", "qLatGoldRR"); chk("xbGoldRR", "qSevRRFive")
put("xbHaikuRR", f2(L["haiku"]["law_rr"][0]), SRC_PL, "cc_threads/labelers/haiku/law_rr"); chk("xbHaikuRR", "tOneHaikuRR"); chk("xbHaikuRR", "qMatchHaikuRR"); chk("xbHaikuRR", "pLRRHaiku"); chk("xbHaikuRR", "qLatHaikuRR")
put3("xbHaikuRRExcess", L["haiku"]["law_rr_minus_gold"], sg, SRC_PL, "cc_threads/labelers/haiku/law_rr_minus_gold (RR minus gold's RR, ratio scale, as Table 7)")
d_ = L["haiku"]["distortion"]["log_rr"]
put3("xbHaikuLogRRDist", [d_["est"]] + d_["ci"], sg, SRC_PL, "cc_threads/labelers/haiku/distortion/log_rr (log RR minus gold's log RR)")
put("xbSonnetRR", f2(L5["sonnet"]["law_rr"][0]), SRC_PL, "cc_threads_s5/labelers/sonnet/law_rr (5% sample)"); chk("xbSonnetRR", "pLRRSonnet")
put("xbGoldSfiveRR", f2(G5["rr"][0]), SRC_PL, "cc_threads_s5/gold/rr"); chk("xbGoldSfiveRR", "qGoldRRSfive")
put3("xbSonnetRRExcess", L5["sonnet"]["law_rr_minus_gold"], sg, SRC_PL, f"cc_threads_s5/labelers/sonnet/law_rr_minus_gold (against the 5% sample's gold, {M['xbGoldSfiveRR']})")
OR_ = load("x75_onerater/x75_onerater.json"); SRC_OR = "results/x75_onerater/x75_onerater.json"; SPL = OR_["splits"]
assert OR_["n_comments"] == 1251487 and len(SPL) == 20
med = lambda fn_: float(np.median([fn_(s_) for s_ in SPL]))
put("xbSrRR", f2(np.exp(med(lambda s_: s_["instr"]["crowd_A"]["log_rr"]))), SRC_OR, "exp(median over the 20 draws of splits/<s>/instr/crowd_A/log_rr): one random rater per comment"); chk("xbSrRR", "qSrRR")
put("xbGoldBRR", f2(np.exp(med(lambda s_: s_["gold_B"]["log_rr"]))), SRC_OR, "exp(median of splits/<s>/gold_B/log_rr): the other raters' majority"); chk("xbGoldBRR", "qGoldLooRR")
put("xbSrRRExcess", sg(med(lambda s_: np.exp(s_["instr"]["crowd_A"]["log_rr"]) - np.exp(s_["gold_B"]["log_rr"]))), SRC_OR, "median over the 20 draws of RR(single rater) - RR(gold B), ratio scale",
    "against its own gold (the others' majority); no CI is stored on the ratio scale")
put("xbSrLogRRDist", sg(med(lambda s_: s_["instr"]["crowd_A"]["log_rr_minus_goldB"])), SRC_OR, "median of splits/<s>/instr/crowd_A/log_rr_minus_goldB"); chk("xbSrLogRRDist", "qSrLrrDist")
put("xbSrPlacRR", f2(np.exp(med(lambda s_: s_["instr"]["crowd_A"]["log_placebo_rr"]))), SRC_OR, "exp(median of crowd_A/log_placebo_rr)"); chk("xbSrPlacRR", "qSrPRR")
rr14 = {k: L[k]["law_rr"][0] for k in CLS14}; ex14 = {k: L[k]["law_rr_minus_gold"] for k in CLS14}
put("xbClsRRMin", f2(min(rr14.values())), SRC_PL, f"min of law_rr {PROV_14} ({min(rr14, key=rr14.get)})")
put("xbClsRRMax", f2(max(rr14.values())), SRC_PL, f"max of law_rr {PROV_14} ({max(rr14, key=rr14.get)})")
put("xbClsRRUpN", str(sum(v[1] > 0 for v in ex14.values())), SRC_PL, "number of the 14 settings whose law_rr_minus_gold has a 95% CI above 0"); chk("xbClsRRUpN", "vClsRRUpAll")
put("xbClsRRDownN", str(sum(v[2] < 0 for v in ex14.values())), SRC_PL, f"number of the 14 settings whose law_rr_minus_gold has a 95% CI below 0 ({', '.join(k for k in CLS14 if ex14[k][2] < 0)})")
CCS = [c for c in CLS14 if c.split("@")[0] in ("detoxify_unbiased", "sn_rob", "mh_dist")]
put("xbClsCcUpN", str(sum(ex14[k][1] > 0 for k in CCS)), SRC_PL, f"of the {len(CCS)} settings trained at least in part on Civil Comments labels, those with more contagion than gold"); chk("xbClsCcUpN", "vClsCcUp")
pr14 = {k: L[k]["placebo_rr"][0] for k in CLS14}
put("xbClsPlacRRMin", f2(min(pr14.values())), SRC_PL, f"min of placebo_rr {PROV_14} ({min(pr14, key=pr14.get)})"); chk("xbClsPlacRRMin", "wClsPRRMin")
put("xbClsPlacRRMax", f2(max(pr14.values())), SRC_PL, f"max of placebo_rr {PROV_14} ({max(pr14, key=pr14.get)})"); chk("xbClsPlacRRMax", "wClsPRRMax")
# ---- E35, contagion: the human instrument matched to each labeler's flag rate (x80_matched, B80a)
MA = load("x80/x80_matched.json"); SRC_MA = "results/x80/x80_matched.json"; IA, IS = MA["cc_threads"]["instruments"], MA["cc_s5"]["instruments"]
assert MA["cc_threads"]["N"] == 1251487 and MA["cc_s5"]["N"] == 60668
for k in CLSM:                                                                                  # at gold's flag rate the matched instrument is gold, up to ties
    assert abs(IA[k]["lrr_human"][0] - math.log(G["rr"][0])) < 1e-3 and abs(IA[k]["flag_rate"] - G_PREV) < 2e-5, k
upm = [k for k in CLS14 if IA[k]["lrr_minus_human"][1] > 0]
put("xbMatchUpN", str(len(upm)), SRC_MA, "number of the 14 settings whose log RR exceeds its matched human instrument's (cc_threads/instruments/<k>/lrr_minus_human, paired 95% CI above 0; 500 article-cluster resamples, seed 8001)")
assert "detoxify_unbiased@0.6" in upm and "detoxify_unbiased@m" in upm and len(CLS12) == int(EX["qMatchNCls"][0])
chk("xbMatchUpN", "qMatchNCls", f"the earlier-draft macro counts the {len(CLS12)} settings of H80a; with Detoxify-unbiased ({ci3(IA['detoxify_unbiased@0.6']['lrr_minus_human'])} at 0.6, "
    f"{ci3(IA['detoxify_unbiased@m']['lrr_minus_human'])} at gold's rate, log scale) all {len(upm)} are above")
own7 = [f"{k}@{t}" for _, k, t, _ in FAM]
put("xbMatchOwnN", str(sum(IA[k]["lrr_minus_human"][1] > 0 for k in own7)), SRC_MA, f"of the {len(own7)} settings at their own threshold (the ones whose matched instrument is not gold), those above it")
assert int(M["xbMatchOwnN"]) == len(own7) == int(EX["qMatchNOwn"][0]) + 1
chk("xbMatchOwnN", "qMatchNOwn", f"the earlier-draft macro counts {EX['qMatchNOwn'][0]} own-threshold settings (without Detoxify-unbiased at 0.6); all {M['xbMatchOwnN']} are above")
lmh = {k: IA[k]["lrr_minus_human"][0] for k in CLS14}
put("xbMatchDiffMin", sg(min(lmh.values())), SRC_MA, f"min over the 14 settings of lrr_minus_human ({min(lmh, key=lmh.get)})")
put("xbMatchDiffMax", sg(max(lmh.values())), SRC_MA, f"max over the 14 settings of lrr_minus_human ({max(lmh, key=lmh.get)})")
for nm_, k_ in (("Detox", "detoxify_original@0.6"), ("ImsArg", "imsypp@argmax")):
    put(f"xbMatch{nm_}RR", f2(math.exp(IA[k_]["lrr"][0])), SRC_MA, f"exp(cc_threads/instruments/{k_}/lrr)")
    put(f"xbMatch{nm_}Hum", f2(math.exp(IA[k_]["lrr_human"][0])), SRC_MA, f"exp(cc_threads/instruments/{k_}/lrr_human)")
chk("xbMatchDetoxRR", "qMatchDetoxRR"); chk("xbMatchDetoxHum", "qMatchDetoxHum")
put("xbMatchHaikuHum", f2(math.exp(IA["haiku"]["lrr_human"][0])), SRC_MA, "exp(cc_threads/instruments/haiku/lrr_human)"); chk("xbMatchHaikuHum", "qMatchHaikuHum")
put3("xbMatchHaikuDiff", IA["haiku"]["lrr_minus_human"], sg, SRC_MA, "cc_threads/instruments/haiku/lrr_minus_human (log scale, paired)")
put("xbMatchSonnetHum", f2(math.exp(IS["sonnet"]["lrr_human"][0])), SRC_MA, "exp(cc_s5/instruments/sonnet/lrr_human) (5% sample)"); chk("xbMatchSonnetHum", "qMatchSonnetHum")
put3("xbMatchSonnetDiff", IS["sonnet"]["lrr_minus_human"], sg, SRC_MA, "cc_s5/instruments/sonnet/lrr_minus_human (log scale, paired)")
note("xbMatchHaikuDiff", "at their own flag rates the Claude models report more contagion than a human instrument flagging as many comments: their weaker contagion against gold is the flag rate")
# ---- E36, contagion part: gold thresholds 0.3-0.8 (x80_threshold, B80b)
TH = load("x80/x80_threshold.json"); SRC_TH = "results/x80/x80_threshold.json"
assert TH["pops"]["cc_threads"]["N"] == 1251487 and TH["pops"]["cc_s5"]["N"] == 60668
CLS7 = ["detoxify_original", "detoxify_unbiased", "imsypp", "sn_rob", "mh_dist", "fb_hate", "cn_off"]
TW = {"0.3": "Three", "0.4": "Four", "0.5": "Five", "0.6": "Six", "0.7": "Seven", "0.8": "Eight"}
BT, BT5 = TH["pops"]["cc_threads"]["by_t"], TH["pops"]["cc_s5"]["by_t"]
assert list(BT) == list(TW) and abs(BT["0.5"]["gold_rr"] - G["rr"][0]) < 1e-9
THR = {}
for t_, nm_ in TW.items():
    b_ = BT[t_]; d7 = [b_["instruments"][c]["lrr_minus_gold"] for c in CLS7]
    THR[t_] = dict(prev=b_["prev"], rr=b_["gold_rr"], med=float(np.median([x[0] for x in d7])), n=int(sum(x[1] > 0 for x in d7)),
                   hk=BT5[t_]["instruments"]["haiku_rating"]["lrr_minus_gold"], sn=BT5[t_]["instruments"]["sonnet_rating"]["lrr_minus_gold"],
                   qw=b_["instruments"]["qwen7b_m"]["lrr_minus_gold"], rr5=BT5[t_]["gold_rr"])
    put(f"xbThrGoldRR{nm_}", f2(b_["gold_rr"]), SRC_TH, f"pops/cc_threads/by_t/{t_}/gold_rr: gold redefined as vote fraction >= {t_}")
    put(f"xbThrPrev{nm_}", pct(b_["prev"], 1), SRC_TH, f"pops/cc_threads/by_t/{t_}/prev")
    put(f"xbThrMed{nm_}", sg(THR[t_]["med"]), SRC_TH, f"median over the 7 classifiers of by_t/{t_}/instruments/<c>/lrr_minus_gold[0] (each flagging gold_{t_}'s share, top-k by score)")
    put(f"xbThrN{nm_}", str(THR[t_]["n"]), SRC_TH, f"number of the 7 classifiers whose paired 95% CI (500 article resamples, seed 8011) is above 0 at t = {t_}")
meds = [v["med"] for v in THR.values()]; ns = [v["n"] for v in THR.values()]
put("xbThrMedMin", sg(min(meds)), SRC_TH, "min over t = 0.3-0.8 of the classifiers' median log-RR distortion"); chk("xbThrMedMin", "qThrMedMin")
put("xbThrMedMax", sg(max(meds)), SRC_TH, "max over t of the median"); chk("xbThrMedMax", "qThrMedMax")
put("xbThrNMin", str(min(ns)), SRC_TH, "min over t of the count of CIs above 0 (of 7)"); chk("xbThrNMin", "qThrNMin")
put("xbThrNMax", str(max(ns)), SRC_TH, "max over t of the count"); chk("xbThrNMax", "qThrNMax")
chk("xbThrGoldRRThree", "qThrGoldRRLo"); chk("xbThrGoldRREight", "qThrGoldRRHi")
for nm_, kk in (("Son", "sn"), ("Haiku", "hk")):
    vv = [v[kk] for v in THR.values()]
    put(f"xbThr{nm_}LrrMin", sg(min(x[0] for x in vv)), SRC_TH, f"min over t of pops/cc_s5/by_t/<t>/instruments/{'sonnet' if kk == 'sn' else 'haiku'}_rating/lrr_minus_gold (0-100 rating flagging gold_t's share, 5% sample)")
    put(f"xbThr{nm_}LrrMax", sg(max(x[0] for x in vv)), SRC_TH, "max over t of the same")
    put(f"xbThr{nm_}AboveN", str(sum(x[1] > 0 for x in vv)), SRC_TH, "number of the 6 thresholds with the paired CI above 0")
    put(f"xbThr{nm_}BelowN", str(sum(x[2] < 0 for x in vv)), SRC_TH, "number of the 6 thresholds with the paired CI below 0")
chk("xbThrSonLrrMin", "qThrSonLrrMin"); chk("xbThrSonLrrMax", "qThrSonLrrMax")
qv = [v["qw"] for v in THR.values()]
put("xbThrQwenMin", sg(min(x[0] for x in qv)), SRC_TH, "min over t of pops/cc_threads/by_t/<t>/instruments/qwen7b_m/lrr_minus_gold (Qwen without the parent at gold_t's share)")
put("xbThrQwenMax", sg(max(x[0] for x in qv)), SRC_TH, "max over t of the same"); assert all(x[2] < 0 for x in qv)
put("xbThrGoldRRSfiveMin", f2(min(v["rr5"] for v in THR.values())), SRC_TH, "min over t of pops/cc_s5/by_t/<t>/gold_rr (5% sample)")
assert THR["0.8"]["rr5"] < THR["0.7"]["rr5"] and THR["0.8"]["rr"] > THR["0.7"]["rr"]                      # falls back at 0.8 on the 5% sample only
put("xbThrGoldRRSfiveMax", f2(max(v["rr5"] for v in THR.values())), SRC_TH, "max over t of the same (it falls back at t = 0.8 on the small sample)")
# ---- E37: severity (x75_decomp B75d), with Detoxify-unbiased recomputed from the stored flags (x75_decomp.py's lrr2, no resampling)
DC = load("x75_decomp/x75_decomp.json"); SRC_DC = "results/x75_decomp/x75_decomp.json"; SV = DC["severity"]
PCC = pd.read_parquet(f"{RES}/x14_pops/cc_threads.parquet", columns=["tox", "parent_row"] + [f"y:{k}" for k in CLSM + ["detoxify_original@0.6", "haiku"]])
assert len(PCC) == 1251487
tox_ = PCC.tox.values; chs = np.flatnonzero(PCC.parent_row.values >= 0); pas = PCC.parent_row.values[chs]; assert len(chs) == 980520


def lrr2(v):                                                                                    # as x75_decomp.py (B75d)
    vp, vc = v[pas], v[chs]; return float(np.log((vc[vp == 1].mean()) / (vc[vp == 0].mean())))


SEV = {}
for t_ in ("0.5", "0.6", "0.7", "0.8"):
    g_ = (tox_ >= float(t_) - 1e-9).astype(float); lg_ = lrr2(g_); assert abs(np.exp(lg_) - SV[t_]["gold_rr"]) < 1e-9
    SEV[t_] = {k: lrr2(np.nan_to_num(PCC[f"y:{k}"].values.astype(float))) - lg_ for k in CLSM + ["detoxify_original@0.6", "haiku"]}
    for k, v in SEV[t_].items():
        if k in SV[t_]:
            assert abs(v - SV[t_][k]) < 1e-12, (t_, k, v, SV[t_][k])                          # the six stored settings, reproduced
SRC_SEV = f"{SRC_DC} (key severity); Detoxify-unbiased at gold's rate recomputed from results/x14_pops/cc_threads.parquet (y:detoxify_unbiased@m) with x75_decomp.py's lrr2"
for t_, nm_ in (("0.5", "Five"), ("0.6", "Six"), ("0.7", "Seven"), ("0.8", "Eight")):
    put(f"xbSevRR{nm_}", f2(SV[t_]["gold_rr"]), SRC_DC, f"severity/{t_}/gold_rr"); chk(f"xbSevRR{nm_}", f"qSevRR{nm_}")
    put(f"xbSevPrev{nm_}", pct(SV[t_]["prev"], 1), SRC_DC, f"severity/{t_}/prev"); chk(f"xbSevPrev{nm_}", f"qSevPrev{nm_}")
    v7 = [SEV[t_][k] for k in CLSM]
    put(f"xbSev{nm_}Min", sg(min(v7)), SRC_SEV, f"min over the 7 settings at gold's 0.5 flag rate of their log RR minus gold_{t_}'s log RR")
    put(f"xbSev{nm_}Max", sg(max(v7)), SRC_SEV, f"max over the same 7 settings")
chk("xbSevSixMin", "qSevSixMin"); chk("xbSevSixMax", "qSevSixMax"); chk("xbSevFiveMin", "qSevFiveMin"); chk("xbSevFiveMax", "qSevFiveMax")
put("xbSevSixDetox", sg(SEV["0.6"]["detoxify_original@0.6"]), SRC_DC, "severity/0.6/detoxify_original@0.6: Detoxify at Avalle et al.'s threshold against a 0.6 gold"); chk("xbSevSixDetox", "qSevSixDetox")
put("xbSevSixUnb", sg(SEV["0.6"]["detoxify_unbiased@m"]), SRC_SEV, "Detoxify-unbiased at gold's 0.5 rate against a 0.6 gold (recomputed; not stored)", "deterministic re-analysis of stored flags")
put("xbSevSixHaiku", sg(SEV["0.6"]["haiku"]), SRC_DC, "severity/0.6/haiku")
note("xbSevSixMin", "point values without CIs; read with E36: matched to gold_t's own flag rate, the classifiers' excess persists (\\xbThrMedSix, \\xbThrNSix of 7 CIs above 0 at t = 0.6)")
# ---- table xb_thresholds_rr (E36 contagion part, E37)
TT = []
for t_, nm_ in TW.items():
    v_ = THR[t_]; sv_ = SEV.get(t_)
    TT.append(" & ".join([t_, pct(v_["prev"], 1), f2(v_["rr"]), sg(v_["med"]), f"{v_['n']} of 7", ci3(v_["hk"]), ci3(v_["sn"]), f2(v_["rr5"]),
                          f"{sg(min(sv_[k] for k in CLSM))} to {sg(max(sv_[k] for k in CLSM))}" if sv_ else "--", sg(sv_["detoxify_original@0.6"]) if sv_ else "--"]) + " \\\\")
table("xb_thresholds_rr",
      "Contagion under stricter or more lenient gold. Gold$_t$ calls a comment toxic when at least a share $t$ of its raters do; gold's share of toxic comments and reply risk ratio (Civil Comments "
      "reply threads). Each classifier, and the Haiku and Sonnet 0--100 ratings, flags as many comments as gold$_t$ (the comments it scores highest): the median over the seven classifiers of the "
      "log risk ratio minus gold$_t$'s, and how many of the seven have a 95\\% CI above 0; the ratings' log-RR distortions (95\\% CI) on the 5\\% sample, where gold$_t$'s risk ratio is given next. "
      "Last two columns: the seven classifier settings that flag as many comments as gold$_{0.5}$, and Detoxify at 0.6, against gold$_t$ (log RR minus gold$_t$'s, range over the seven).",
      ["t", "Gold_t prevalence", "Gold_t RR", "Classifiers at gold_t's rate: median log-RR distortion", "CIs above 0", "Haiku-100 [95% CI]", "Sonnet-100 [95% CI]", "Gold_t RR, 5% sample",
       "Classifiers at gold_0.5's rate against gold_t", "Detoxify (0.6) against gold_t"],
      TT, [("t, prevalence, RR, median, count", f"{SRC_TH}: pops/cc_threads/by_t/<t>/(prev, gold_rr, instruments/<classifier>/lrr_minus_gold) over the 7 classifiers (score thresholded at gold_t's share)"),
           ("Haiku-100, Sonnet-100, 5% sample RR", f"{SRC_TH}: pops/cc_s5/by_t/<t>/instruments/<haiku|sonnet>_rating/lrr_minus_gold and gold_rr"),
           ("Last two columns", SRC_SEV + "; given for t >= 0.5 as stored (B75d)")],
      "The last two columns are point values without CIs. The 7 settings at gold_0.5's rate include Detoxify-unbiased, recomputed from the stored flags (the six stored values reproduce to 1e-12).")
# ---- E38: k-rater gold
NZ = load("x80/x80_noise.json"); SRC_NZ = "results/x80/x80_noise.json (key full_population only; its reweighted part is K2)"; FPk = NZ["full_population"]["k"]
assert NZ["n_edges"] == 980520 and abs(NZ["full_population"]["gold_rr"] - G["rr"][0]) < 1e-9
for kk, nm_ in (("1", "One"), ("2", "Two"), ("3", "Three")):
    put(f"xbK{nm_}RR", f2(FPk[kk]), SRC_NZ, f"full_population/k/{kk}: median over 20 draws (seed 8021) of the reply RR when gold is the majority of {kk} raters drawn without replacement (ties at k = 2 broken at random)")
chk("xbKOneRR", "qNoiseKOne"); chk("xbKTwoRR", "qNoiseKTwo"); chk("xbKThreeRR", "qNoiseKThree")
CR = load("x74_crowd/x74_crowd.json"); SRC_CR = "results/x74_crowd/x74_crowd.json (key k_rater only; its split halves are K9)"; assert CR["n_comments"] == 1251487
for kk, nm_ in (("1", "One"), ("3", "Three"), ("5", "Five")):
    put(f"xbK{nm_}RRCrowd" if kk != "5" else "xbKFiveRR", f2(np.exp(np.median([r_["log_rr"] for r_ in CR["k_rater"][kk]]))), SRC_CR,
        f"exp(median over 20 draws of k_rater/{kk}/<s>/log_rr) (seeds 7600 + 10k + s; a tie counts as toxic)")
chk("xbKFiveRR", "wKFiveRR"); chk("xbKOneRRCrowd", "wKOneRR"); chk("xbKThreeRRCrowd", "wKThreeRR")
note("xbKOneRRCrowd", f"a second set of 20 draws gives {M['xbKOneRRCrowd']} where x80_noise gives {M['xbKOneRR']} (\\xbKOneRR)" + (": draw-to-draw variation" if M["xbKOneRRCrowd"] != M["xbKOneRR"] else ""))
# ---- E39: the model-free many-rater check (x88_manyraters, B88c)
MR = load("x88/x88_manyraters.json"); SRC_MR = "results/x88/x88_manyraters.json"; IMR = MR["instruments"]; assert MR["n_edges"] == 980520
NS = {"4": "Four", "6": "Six", "10": "Ten", "20": "Twenty", "50": "Fifty"}; assert [str(x) for x in MR["nstar"]] == list(NS)
for n_, nm_ in NS.items():
    v14 = {k: IMR[k][n_]["rr_parent_flag"] for k in CLS14}
    put(f"xbMrEdges{nm_}", comma(IMR["haiku"][n_]["n_edges"]), SRC_MR, f"instruments/haiku/{n_}/n_edges: reply edges whose child has >= {n_} raters")
    put(f"xbMrClsMin{nm_}", f2(min(x[0] for x in v14.values())), SRC_MR, f"min over the 14 settings of instruments/<k>/{n_}/rr_parent_flag (Mantel-Haenszel RR of the child's flag on the parent's flag, strata = the child's exact votes)")
    put(f"xbMrClsMax{nm_}", f2(max(x[0] for x in v14.values())), SRC_MR, f"max over the 14 settings ({max(v14, key=lambda k: v14[k][0])})")
    put(f"xbMrClsAbove{nm_}", str(sum(x[1] > 1 for x in v14.values())), SRC_MR, f"number of the 14 settings whose 95% CI (200 article resamples, seed 8804) is above 1 at n* = {n_}")
for n_ in ("10", "20", "50"):
    put3(f"xbMrSr{NS[n_]}", IMR["single_rater"][n_]["rr_parent_flag"], f2, SRC_MR, f"instruments/single_rater/{n_}/rr_parent_flag (one random vote per comment; strata = the child's other votes; seed 8803)")
    put(f"xbMrHaiku{NS[n_]}", f2(IMR["haiku"][n_]["rr_parent_flag"][0]), SRC_MR, f"instruments/haiku/{n_}/rr_parent_flag")
    put(f"xbMrQwen{NS[n_]}", f2(IMR["qwen7b_m"][n_]["rr_parent_flag"][0]), SRC_MR, f"instruments/qwen7b_m/{n_}/rr_parent_flag (Qwen at gold's flag rate, without the parent)")
chk("xbMrClsMinTwenty", "cMrClsMin"); chk("xbMrClsMaxTwenty", "cMrClsMax"); chk("xbMrClsAboveTwenty", "cMrClsN"); chk("xbMrClsMinTen", "cMrTenMin"); chk("xbMrClsMaxTen", "cMrTenMax")
chk("xbMrSrTwenty", "cMrSr"); chk("xbMrSrTwentyLo", "cMrSrLo"); chk("xbMrSrTwentyHi", "cMrSrHi")
g14 = {k: IMR[k]["20"]["rr_parent_gold"] for k in CLS14}
put("xbMrGoldClsMin", f2(min(x[0] for x in g14.values())), SRC_MR, "min over the 14 settings of instruments/<k>/20/rr_parent_gold (the parent's gold label in place of its flag)")
put("xbMrGoldClsMax", f2(max(x[0] for x in g14.values())), SRC_MR, "max over the 14 settings of the same")
put("xbMrGoldClsAboveN", str(sum(x[1] > 1 for x in g14.values())), SRC_MR, "number of the 14 settings whose parent-gold RR at n* = 20 has a CI above 1")
put("xbMrGoldClsBelowN", str(sum(x[2] < 1 for x in g14.values())), SRC_MR, "number of the 14 settings whose parent-gold RR at n* = 20 has a CI below 1")
put3("xbMrGoldHaiku", IMR["haiku"]["20"]["rr_parent_gold"], f2, SRC_MR, "instruments/haiku/20/rr_parent_gold")
put3("xbMrGoldHaikuCtx", IMR["haiku_ctx"]["20"]["rr_parent_gold"], f2, SRC_MR, "instruments/haiku_ctx/20/rr_parent_gold: Haiku shown the parent (every child is a reply, so the child's flag is always the with-parent one)")
chk("xbMrGoldHaikuCtx", "cMrCtxHaiku")
QCTX_RATE = float(pd.read_parquet(f"{RES}/x14_pops/cc_threads.parquet", columns=["y:qwen7b_m_ctx"])["y:qwen7b_m_ctx"].mean())   # the no-context-threshold variant's flag rate
put3("xbMrGoldQwenCtx", IMR["qwen7b_m_ctx@own"]["20"]["rr_parent_gold"], f2, SRC_MR, "instruments/qwen7b_m_ctx@own/20/rr_parent_gold: Qwen shown the parent at its own gold-prevalence threshold "
     f"({pct(G_PREV, 1)} flagged: x88_manyraters.py flags as many comments as gold, those Qwen scores highest)")
chk("xbMrGoldQwenCtx", "cMrCtxQwen", f"x18_numbers_v5.py takes instruments/qwen7b_m_ctx (Qwen shown the parent at the no-context threshold, {pct(QCTX_RATE, 1)} flagged "
    f"(results/x14_pops/cc_threads.parquet, y:qwen7b_m_ctx): {IMR['qwen7b_m_ctx']['20']['rr_parent_gold'][0]:.2f}); the paper's convention since Addendum 79 is the own threshold "
    "(qwen7b_m_ctx@own), which gives the value here")
put3("xbMrGoldSr", IMR["single_rater"]["20"]["rr_parent_gold"], f2, SRC_MR, "instruments/single_rater/20/rr_parent_gold")
note("xbMrClsMinTwenty", "the parent-flag RR of a labeler shown the parent is not reported: the parent's flag mixes the root (no context) and reply (with context) prompts")
# ---- table xb_manyraters (E39 at every n*; appendix)
TMR = [" & ".join(["Reply edges", "--"] + [comma(IMR["haiku"][n_]["n_edges"]) for n_ in NS] + ["--"]) + " \\\\", "\\midrule"]
for nm_, k_, t_, cc_ in FAM:
    for j_, s_ in enumerate((f"{k_}@{t_}", f"{k_}@m")):
        TMR.append(" & ".join([fam_name(k_) if j_ == 0 else "", thr_name(s_)] + [ci3(IMR[s_][n_]["rr_parent_flag"], f2) if n_ in ("10", "20") else f2(IMR[s_][n_]["rr_parent_flag"][0]) for n_ in NS]
                              + [ci3(IMR[s_]["20"]["rr_parent_gold"], f2)]) + " \\\\")
    TMR.append("\\addlinespace[1pt]")
TMR = TMR[:-1] + ["\\midrule"]
for nm_, k_, thr_ in (("Haiku", "haiku", "own cut"), ("Qwen", "qwen7b_m", "gold's rate")):
    TMR.append(" & ".join([nm_, thr_] + [ci3(IMR[k_][n_]["rr_parent_flag"], f2) if n_ in ("10", "20") else f2(IMR[k_][n_]["rr_parent_flag"][0]) for n_ in NS] + [ci3(IMR[k_]["20"]["rr_parent_gold"], f2)]) + " \\\\")
for nm_, k_, thr_ in (("Haiku + parent", "haiku_ctx", "own cut"), ("Qwen + parent", "qwen7b_m_ctx@own", "gold's rate")):
    TMR.append(" & ".join([nm_, thr_] + ["--"] * 5 + [ci3(IMR[k_]["20"]["rr_parent_gold"], f2)]) + " \\\\")
TMR.append(" & ".join(["One rater", "--"] + [ci3(IMR["single_rater"][n_]["rr_parent_flag"], f2) if n_ in ("10", "20") else f2(IMR["single_rater"][n_]["rr_parent_flag"][0]) for n_ in NS]
                      + [ci3(IMR["single_rater"]["20"]["rr_parent_gold"], f2)]) + " \\\\")
table("xb_manyraters",
      "The many-rater check on Civil Comments reply edges: among edges whose reply has at least $n^\\ast$ raters, the Mantel--Haenszel risk ratio of the reply's flag on its parent's flag, "
      "stratified by the reply's exact votes (number of raters and of toxic votes), with 95\\% CIs at $n^\\ast = 10$ and 20 (200 article resamples); last column, the same with the parent's gold "
      "label in place of its flag, at $n^\\ast = 20$. A labeler that reads a comment only through how toxic its raters find it gives 1 as $n^\\ast$ grows; the one rater (a random vote, against "
      "the reply's other votes) is that case. Shown the parent, only the parent-gold version is given, since the parent's own flag comes from the prompt without context when the parent is a "
      "top-level comment. $^{\\S}$Trained at least in part on Civil Comments labels.",
      ["Labeler", "Threshold", "n* = 4", "n* = 6", "n* = 10 [95% CI]", "n* = 20 [95% CI]", "n* = 50", "Parent's gold label, n* = 20 [95% CI]"],
      TMR, [("All cells", f"{SRC_MR}: instruments/<k>/<n*>/rr_parent_flag and /20/rr_parent_gold; edges: instruments/haiku/<n*>/n_edges")],
      "Qwen shown the parent is at its own gold-prevalence threshold (qwen7b_m_ctx@own); the no-context-threshold variant stored as qwen7b_m_ctx is not used.")
# ---- E42: contagion within articles against its within-article permutation null (x87_within_null, B87a)
WN = load("x87/x87_within_null.json"); SRC_WN = "results/x87/x87_within_null.json"; IWN = WN["instruments"]; gw_ = WN["gold"]
assert WN["n_edges"] == 980520 and WN["n_articles"] == 57769 and abs(gw_["pooled_log_rr"] - math.log(G["rr"][0])) < 1e-9
put3("xbWnGoldRR", gw_["rr_mh"], f2, SRC_WN, "gold/rr_mh: Mantel-Haenszel reply RR stratified by article (200 article resamples, seed 8611)"); chk("xbWnGoldRR", "qWaGold"); chk("xbWnGoldRRLo", "qWaGoldLo"); chk("xbWnGoldRRHi", "qWaGoldHi")
chk("xbWnGoldRR", "qWithinGoldRR")
put3("xbWnGoldNull", gw_["null_rr"], f2, SRC_WN, "gold/null_rr: the same statistic averaged over 50 permutations of the flags within each article (seed 8701)"); chk("xbWnGoldNull", "qWnNull")
put3("xbWnGoldExc", gw_["excess"], sg, SRC_WN, "gold/excess: log RR_MH minus its within-article null"); chk("xbWnGoldExc", "qWnExc")
put("xbWnBetween", pct(gw_["between_share"]), SRC_WN, "gold/between_share = 1 - excess / pooled log RR"); chk("xbWnBetween", "qWnBetween")
put("xbWnWithinShare", pct(1 - gw_["between_share"]), SRC_WN, "1 - gold/between_share: the share of gold's pooled log RR that lies within articles")
e14 = {k: IWN[k]["excess"][0] for k in CLS14}
put("xbWnClsExcMin", sg(min(e14.values())), SRC_WN, f"min over the 14 settings of instruments/<k>/excess ({min(e14, key=e14.get)})")
put("xbWnClsExcMax", sg(max(e14.values())), SRC_WN, f"max over the 14 settings ({max(e14, key=e14.get)})")
ab_ = [k for k in CLS14 if IWN[k]["minus_gold"][1] > 0]; bl_ = [k for k in CLS14 if IWN[k]["minus_gold"][2] < 0]
put("xbWnClsAboveN", str(len(ab_)), SRC_WN, f"number of the 14 settings whose excess exceeds gold's (instruments/<k>/minus_gold, paired CI above 0): {', '.join(ab_)}")
put("xbWnClsBelowN", str(len(bl_)), SRC_WN, f"number below gold's (paired CI below 0): {', '.join(bl_)}")
assert "detoxify_unbiased@0.6" in ab_ and "detoxify_unbiased@m" in ab_
chk("xbWnClsAboveN", "qWnAbove", f"the earlier-draft macro counts the {len(CLS12)} settings of B87a ({EX['qWnAbove'][0]} above); Detoxify-unbiased is above gold's at both thresholds "
    f"({sg(IWN['detoxify_unbiased@0.6']['minus_gold'][0])} and {sg(IWN['detoxify_unbiased@m']['minus_gold'][0])} on the log scale), so {len(ab_)} of {len(CLS14)}")
chk("xbWnClsBelowN", "qWnBelow")
p14 = {k: IWN[k]["placebo_excess"] for k in CLS14}
put("xbWnPlacMin", sg(min(v[0] for v in p14.values())), SRC_WN, "min over the 14 settings of instruments/<k>/placebo_excess (placebo RR_MH over gold-negative edges against its within-article null)"); chk("xbWnPlacMin", "qWnPlacMin")
put("xbWnPlacMax", sg(max(v[0] for v in p14.values())), SRC_WN, "max over the 14 settings"); chk("xbWnPlacMax", "qWnPlacMax")
put("xbWnPlacPosN", str(sum(v[1] > 0 for v in p14.values())), SRC_WN, "number of the 14 settings whose placebo excess has a CI above 0")
put3("xbWnSrPlac", IWN["single_rater"]["placebo_excess"], sg, SRC_WN, "instruments/single_rater/placebo_excess (against gold B)"); chk("xbWnSrPlac", "qWnSrPlac")
put3("xbWnSrExc", IWN["single_rater"]["excess"], sg, SRC_WN, "instruments/single_rater/excess (its flags against its own gold, the others' majority)")
put3("xbWnGoldBExc", WN["gold_B"]["excess"], sg, SRC_WN, "gold_B/excess: the others' majority")
put3("xbWnHaikuExc", IWN["haiku"]["excess"], sg, SRC_WN, "instruments/haiku/excess")
put3("xbWnQwenExc", IWN["qwen7b_m"]["excess"], sg, SRC_WN, "instruments/qwen7b_m/excess (Qwen at gold's flag rate, without the parent)")
note("xbWnHaikuExc", "labelers shown the parent are not reported (their parent labels mix the two prompts)")
# ---- E43: identity terms and sibling pairs (x80_mech, B80d)
MC = load("x80/x80_mech.json"); SRC_MC = "results/x80/x80_mech.json"; IMC = MC["instruments"]; SR1 = MC["single_rater"]
assert abs(IMC["haiku"]["placebo_rr"] - L["haiku"]["placebo_rr"][0]) < 1e-9 and all(abs(IMC[k]["placebo_rr"] - L[k]["placebo_rr"][0]) < 1e-9 for k in CLS14)
put("xbIdShare", pct(MC["identity_share"]), SRC_MC, "identity_share: comments with an identity-term mention (Borkan et al.'s subgroups, whole words)"); chk("xbIdShare", "qIdShare")
put("xbIdEdgeShare", pct(IMC["haiku"]["share_edges_with_identity"]), SRC_MC, "instruments/<any>/share_edges_with_identity: gold-negative edges with a mention at either end")
fr14 = {k: IMC[k]["fpr_ratio"] for k in CLS14}
put("xbIdFprMin", f(min(fr14.values()), 1), SRC_MC, f"min over the 14 settings of instruments/<k>/fpr_ratio (FPR on gold negatives with a mention / without) ({min(fr14, key=fr14.get)})")
put("xbIdFprMax", f(max(fr14.values()), 1), SRC_MC, f"max over the 14 settings ({max(fr14, key=fr14.get)})")
fr12 = [fr14[k] for k in CLS12]
put("xbIdFprMinTwelve", f(min(fr12), 1), SRC_MC, "min over the 12 settings without Detoxify-unbiased")
chk("xbIdFprMin", "qIdFprMin", f"the earlier-draft macro is the minimum over the {len(CLS12)} settings without Detoxify-unbiased ({M['xbIdFprMinTwelve']}, \\xbIdFprMinTwelve); Detoxify-unbiased, trained to reduce identity bias, "
    f"has no identity gap ({f2(fr14['detoxify_unbiased@0.6'])} at 0.6, {f2(fr14['detoxify_unbiased@m'])} at gold's rate), so over the {len(CLS14)} settings the range starts at {M['xbIdFprMin']}")
chk("xbIdFprMinTwelve", "qIdFprMin"); chk("xbIdFprMax", "qIdFprMax")
put("xbIdFprSr", f(SR1["fpr_ratio"], 1), SRC_MC, "single_rater/fpr_ratio"); chk("xbIdFprSr", "qIdFprSr")
ch14 = {k: IMC[k]["placebo_rr_no_identity"] / IMC[k]["placebo_rr"] - 1 for k in CLS14}
put("xbIdChgMin", spct(min(ch14.values())), SRC_MC, f"min over the 14 settings of placebo_rr_no_identity / placebo_rr - 1 ({min(ch14, key=ch14.get)})"); chk("xbIdChgMin", "qIdChgMin")
put("xbIdChgMax", spct(max(ch14.values())), SRC_MC, f"max over the 14 settings ({max(ch14, key=ch14.get)})"); chk("xbIdChgMax", "qIdChgMax")
put("xbIdChgSr", spct(SR1["placebo_rr_no_identity"] / SR1["placebo_rr"] - 1), SRC_MC, "single_rater: placebo_rr_no_identity / placebo_rr - 1")
put("xbIdChgHaiku", spct(IMC["haiku"]["placebo_rr_no_identity"] / IMC["haiku"]["placebo_rr"] - 1), SRC_MC, "haiku: placebo_rr_no_identity / placebo_rr - 1")
sb14 = {k: IMC[k]["placebo_rr_siblings"] for k in CLS14}
put("xbSibMin", f2(min(sb14.values())), SRC_MC, f"min over the 14 settings of placebo_rr_siblings (consecutive gold-negative replies to the same parent) ({min(sb14, key=sb14.get)})")
put("xbSibMax", f2(max(sb14.values())), SRC_MC, f"max over the 14 settings ({max(sb14, key=sb14.get)})")
put("xbSibSr", f2(SR1["placebo_rr_siblings"]), SRC_MC, "single_rater/placebo_rr_siblings"); put("xbSibHaiku", f2(IMC["haiku"]["placebo_rr_siblings"]), SRC_MC, "haiku/placebo_rr_siblings")
put("xbSibPairs", comma(MC["n_sibling_pairs"]), SRC_MC, "n_sibling_pairs")
note("xbSibMin", "x80_mech's article-stratified Mantel-Haenszel placebo RR (placebo_rr_mh_article) is the raw within-article statistic that Addendum 87 found biased toward 0; it is not used (E42 gives the corrected comparison)")
# ---- E44: the parent's contrast effect (x88_breakdown, B88h)
BD = load("x88/x88_breakdown.json"); SRC_BD = "results/x88/x88_breakdown.json (key B88h)"; HC = BD["B88h"]
for nm_, k_ in (("HaikuCtx", "haiku_ctx"), ("Haiku", "haiku"), ("QwenCtx", "qwen7b_m_ctx@own"), ("Qwen", "qwen7b_m")):
    for c_, zc, zp in (("TT", 1, 1), ("TC", 1, 0), ("CT", 0, 1), ("CC", 0, 0)):
        put(f"xbCtr{nm_}{c_}", pct(HC[k_][f"child{zc}_parent{zp}"]), SRC_BD, f"B88h/{k_}/child{zc}_parent{zp}: share of {'toxic' if zc else 'civil'} replies flagged when the parent is gold-{'toxic' if zp else 'civil'}")
chk("xbCtrHaikuCtxTT", "cCtrHaikuTT"); chk("xbCtrHaikuCtxTC", "cCtrHaikuTC"); chk("xbCtrQwenCtxTT", "cCtrQwenTT", scale=0.01); chk("xbCtrQwenCtxTC", "cCtrQwenTC", scale=0.01)
note("xbCtrHaikuCtxTT", "every child is a reply, so the with-parent flags here are among replies; point values, no CIs stored")
# ---- E33: how context changes contagion, humans against LLMs (CCC; f31)
SRC_F31 = "results/f31_ccc_compare.json (Haiku), f31_ccc_compare_qwen.json, f31_ccc_compare_gemma.json"
C31 = {m_: load(fn_) for m_, fn_ in (("haiku", "f31_ccc_compare.json"), ("qwen", "f31_ccc_compare_qwen.json"), ("gemma", "f31_ccc_compare_gemma.json"))}
assert C31["haiku"]["n"] == 9903 and C31["qwen"]["n"] == C31["gemma"]["n"] == 9908
H31 = C31["haiku"]["h"]
put("xbCccN", comma(C31["haiku"]["n"]), SRC_F31, "n: CCC posts with both human labelings, both Haiku labels and the parent's gold label")
put("xbCccNOpen", comma(C31["qwen"]["n"]), SRC_F31, "n of the Qwen and Gemma files (five more posts, which have no Haiku label)")
put("xbCccHumRROut", f2(H31["assoc_oc"]["rr"]), SRC_F31, "h/assoc_oc/rr: the original crowd (no context), reply flag on the parent's gold label")
put("xbCccHumRRIn", f2(H31["assoc_ic"]["rr"]), SRC_F31, "h/assoc_ic/rr: CCC raters shown the parent")
put("xbCccHumPrevOut", pct(H31["prev_oc"], 1), SRC_F31, "h/prev_oc"); put("xbCccHumPrevIn", pct(H31["prev_ic"], 1), SRC_F31, "h/prev_ic")
put("xbCccParentToxic", pct(C31["haiku"]["parent_toxic_share"], 1), SRC_F31, "parent_toxic_share (parent's Civil Comments gold)")
for m_, nm_ in (("haiku", "Haiku"), ("qwen", "Qwen"), ("gemma", "Gemma")):
    l_ = C31[m_]["l"]
    put(f"xbCcc{nm_}RROut", f2(l_["assoc_oc"]["rr"]), SRC_F31, f"{m_} file: l/assoc_oc/rr (labeled without the parent)")
    put(f"xbCcc{nm_}RRIn", f2(l_["assoc_ic"]["rr"]), SRC_F31, f"{m_} file: l/assoc_ic/rr (labeled with the parent)")
    put(f"xbCcc{nm_}PrevOut", pct(l_["prev_oc"], 1), SRC_F31, f"{m_} file: l/prev_oc"); put(f"xbCcc{nm_}PrevIn", pct(l_["prev_ic"], 1), SRC_F31, f"{m_} file: l/prev_ic")
    put3(f"xbCcc{nm_}DiffChg", l_["assoc_change_ic_minus_oc"], lambda x: sg(PP * x, 1), SRC_F31, f"{m_} file: l/assoc_change_ic_minus_oc x 100 (change of the risk difference with context, points; 500 post resamples, seed 0)")
put3("xbCccHumDiffChg", H31["assoc_change_ic_minus_oc"], lambda x: sg(PP * x, 1), SRC_F31, "h/assoc_change_ic_minus_oc x 100 (Haiku file)")
note("xbCccHaikuDiffChg", "on the difference scale the association falls with context for the humans and for Qwen and Gemma, but Haiku's CI covers 0; on the ratio scale it rises for the humans and falls for the LLMs (prevalence falls sixfold for the humans)")
note("xbCccN", "f31 analyses CCC's own posts (human labels from the original crowd and from CCC's in-context raters; parent toxicity from cc_meta's crowd fraction), not the train-split reply-thread population, so the old-population issue does not apply; Haiku flags 'toxic' or 'very toxic', while Qwen's and Gemma's scores are thresholded at the humans' out-of-context prevalence (their no-context flag rates are " + M["xbCccQwenPrevOut"] + " and " + M["xbCccGemmaPrevOut"] + " against the humans' " + M["xbCccHumPrevOut"] + ")")

# ---- table xb_contagion
def lrd(r_):                                                                                   # arXiv revision: the log-scale distortion (paired bootstrap CI)
    d_ = r_["distortion"]["log_rr"]; return ci3([d_["est"]] + d_["ci"])


def crow(name, thr, rr, exc, wn, mr, hum, idc):
    return " & ".join([name, thr, rr, exc, wn, mr, hum, idc]) + " \\\\"


TC = [crow("Gold", "--", f2(G["rr"][0]), "--", ci3(gw_["excess"]), "--", "--", "--"),
      crow("One rater", "--", M["xbSrRR"], M["xbSrLogRRDist"], ci3(IWN["single_rater"]["excess"]), ci3(IMR["single_rater"]["20"]["rr_parent_flag"], f2), "--", M["xbIdChgSr"]),
      crow("Haiku", "own cut", f2(L["haiku"]["law_rr"][0]), lrd(L["haiku"]), ci3(IWN["haiku"]["excess"]), ci3(IMR["haiku"]["20"]["rr_parent_flag"], f2),
           f2(math.exp(IA["haiku"]["lrr_human"][0])), M["xbIdChgHaiku"]),
      crow("Sonnet$^{\\dagger}$", "own cut", f2(L5["sonnet"]["law_rr"][0]), lrd(L5["sonnet"]), "--", "--", f2(math.exp(IS["sonnet"]["lrr_human"][0])), "--"),
      "\\midrule"]
for nm_, k_, t_, cc_ in FAM:
    for j_, s_ in enumerate((f"{k_}@{t_}", f"{k_}@m")):
        TC.append(crow(fam_name(k_) if j_ == 0 else "", thr_name(s_), f2(L[s_]["law_rr"][0]), lrd(L[s_]), ci3(IWN[s_]["excess"]),
                       ci3(IMR[s_]["20"]["rr_parent_flag"], f2), f2(math.exp(IA[s_]["lrr_human"][0])), spct(ch14[s_])))
    TC.append("\\addlinespace[1pt]")
TC = TC[:-1]
table("xb_contagion",
      f"Contagion on Civil Comments reply threads: the reply risk ratio under each labeler and its excess over gold's ({M['xbGoldRR']}; the One rater row against the other raters' majority, {M['xbGoldBRR']}; "
      f"Sonnet against the 5\\% sample's gold, {M['xbGoldSfiveRR']}); within articles, the log Mantel--Haenszel risk ratio stratified by article minus its value when the flags are permuted within each article; "
      "the many-rater check, the Mantel--Haenszel risk ratio of a reply's flag on its parent's flag among replies with at least 20 raters, stratified by the reply's exact votes (1 if the "
      "labeler reads only the raters' construct); the risk ratio of a human instrument flagging as many comments, those with the most toxic votes (gold itself at gold's flag rate); and "
      "the change in the placebo risk ratio when edges with an identity-term mention are left out. 95\\% CIs in brackets. $^{\\dagger}$5\\% sample. $^{\\S}$Trained at least in part on Civil Comments labels.",
      ["Labeler", "Threshold", "Reply RR", "Excess over gold [95% CI]", "Within articles: log RR minus null [95% CI]", "Many-rater RR, n* = 20 [95% CI]", "Matched human RR", "Placebo RR without identity edges"],
      TC, [("Reply RR, distortion", f"{SRC_PL}: law_rr, distortion/log_rr (log RR minus gold's log RR, paired CI; gold: gold/rr; One rater: {SRC_OR}, median over 20 draws of log_rr_minus_goldB, no CI)"),
           ("Within articles", f"{SRC_WN}: instruments/<k>/excess and gold/excess (One rater: against gold B)"),
           ("Many-rater RR", f"{SRC_MR}: instruments/<k>/20/rr_parent_flag"), ("Matched human RR", f"{SRC_MA}: exp(instruments/<k>/lrr_human)"),
           ("Placebo RR without identity edges", f"{SRC_MC}: placebo_rr_no_identity / placebo_rr - 1"),
           ("Blank cells", "Sonnet: within-article, many-rater and identity analyses were run on the full reply-thread population only; gold and the One rater have no matched human instrument")],
      "Sonnet's within-article excess, many-rater RR and identity change are not stored (those analyses ran on cc_threads only).")

# ================================================================ C. The latent model of the raters (E40, E41; appendix, with the model's caveats)
sec("C. The latent propensity model: noise-free contagion and the construct-only test (E40, E41)")
RI = load("x85/x85_rrinf.json"); SRC_RI = "results/x85/x85_rrinf.json (B85a, NPMLE solved to the KKT condition by SQP)"; IRI = RI["instruments"]
assert abs(IRI["haiku"]["rr"] - L["haiku"]["law_rr"][0]) < 1e-9 and all(abs(IRI[k]["rr"] - L[k]["law_rr"][0]) < 1e-9 for k in CLS14)   # cc_threads, all 980,520 edges
assert all(RI["fits"][m_]["converged"] for m_ in ("10", "20", "40"))
LATCAV = "model-based: rests on the approximate latent propensity model (Binomial votes given a comment's propensity; NPMLE of the joint parent-reply propensities on an m-point grid)"
put3("xbLatRR", RI["rr_inf"]["rr"], f2, SRC_RI, "rr_inf/rr: RR of the noise-free gold 1{pi >= 0.5} under the fitted joint distribution of (pi_p, pi_c), m = 20; CI over the 100 article-cluster resamples of B82a (seed 8201)", LATCAV)
chk("xbLatRR", "qLatRR"); chk("xbLatRRLo", "qLatRRLo"); chk("xbLatRRHi", "qLatRRHi")
put("xbLatRRTen", f2(RI["fits"]["10"]["rr_inf"]), SRC_RI, "fits/10/rr_inf (m = 10)", LATCAV); chk("xbLatRRTen", "qLatRRTen")
put("xbLatRRForty", f2(RI["fits"]["40"]["rr_inf"]), SRC_RI, "fits/40/rr_inf (m = 40)", LATCAV); chk("xbLatRRForty", "qLatRRForty")
put("xbLatKKT", sci(max([RI["fits"][m_]["kkt"] for m_ in ("10", "20", "40")] + [RI["rr_inf"]["max_resample_kkt"]])), SRC_RI, "max over the three fits and the 100 resamples of the KKT residual max_k d_k - 1")
GPJ = load("x87/x87_gap.json")["B87b"]; SRC_GP = "results/x87/x87_gap.json (B87b)"; assert abs(GPJ["rr_inf"] - RI["rr_inf"]["rr"][0]) < 1e-9 and abs(GPJ["rr_gold"] - G["rr"][0]) < 1e-9
put3("xbLatGap", GPJ["gap"], sg, SRC_GP, "B87b/gap: log RR_inf - gold's log RR, paired over the same 100 resamples", LATCAV)
chk("xbLatGap", "qGapLog"); chk("xbLatGapLo", "qGapLo"); chk("xbLatGapHi", "qGapHi"); chk("xbLatGap", "qLatAttLog")
put("xbLatGapShare", pct(GPJ["share_inf_le_gold"]), SRC_GP, "B87b/share_inf_le_gold: share of resamples with RR_inf <= gold's RR")
ab14 = [k for k in CLS14 if IRI[k]["minus_inf"][1] > 0]; bl14 = [k for k in CLS14 if IRI[k]["minus_inf"][2] < 0]
put("xbLatAboveN", str(len(ab14)), SRC_RI, f"number of the 14 settings whose log RR - log RR_inf (instruments/<k>/minus_inf, paired) has a CI above 0: {', '.join(ab14)}", LATCAV)
put("xbLatBelowN", str(len(bl14)), SRC_RI, f"number with a CI below 0: {', '.join(bl14)}", LATCAV)
put("xbLatSameN", str(14 - len(ab14) - len(bl14)), SRC_RI, "the other settings of the 14 (CI covers 0)", LATCAV)
chk("xbLatAboveN", "qLatNAbove"); chk("xbLatBelowN", "qLatNBelow")
assert not {"detoxify_unbiased@0.6", "detoxify_unbiased@m"} & (set(ab14) | set(bl14))
chk("xbLatSameN", "qLatNSame", f"the earlier-draft macro counts the {len(CLS12)} settings of H82a ({EX['qLatNSame'][0]} indistinguishable); both Detoxify-unbiased settings are also indistinguishable "
    f"from RR_inf, so {M['xbLatSameN']} of {len(CLS14)}")
put("xbLatAboveMin", f2(min(IRI[k]["rr"] for k in ab14)), SRC_RI, "smallest reply RR among the settings above RR_inf"); chk("xbLatAboveMin", "qLatAboveMin")
put("xbLatAboveMax", f2(max(IRI[k]["rr"] for k in ab14)), SRC_RI, "largest reply RR among them"); chk("xbLatAboveMax", "qLatAboveMax")
put3("xbLatHaikuGap", IRI["haiku"]["minus_inf"], sg, SRC_RI, "instruments/haiku/minus_inf: Haiku's log RR - log RR_inf", LATCAV)
assert set(ab14) == {"detoxify_original@0.6", "mh_dist@0.5"} and set(RI["H82a_converged"]["above"]) == set(ab14)
# ---- the model's fit checks (B82e, x82_protocol; x82_latent's fixed-n check is K5 and is not used)
PRO = load("x82/x82_protocol.json"); SRC_PRO = "results/x82/x82_protocol.json (B82e: the votes simulated under the inferred adaptive rater protocol)"
assert abs(PRO["obs"]["gold_rr"] - G["rr"][0]) < 1e-9
PROCAV = f"the protocol check simulates from the EM fit of B82a at m = 20 (RR_inf {f2(PRO['rr_inf'])}), not from the converged fit ({f2(RI['rr_inf']['rr'][0])})"
put("xbLatSimGold", f2(PRO["sim"]["gold_rr"]), SRC_PRO, "sim/gold_rr: gold's RR when the raters' votes are simulated under the fitted model and the inferred protocol", PROCAV); chk("xbLatSimGold", "qLatSimGold")
put("xbLatSimGap", f(abs(PRO["gaps"]["gold"]), 3), SRC_PRO, "|gaps/gold|: |simulated - observed| gold log RR", PROCAV); chk("xbLatSimGap", "qLatLogGap")
put("xbLatSimOne", f2(PRO["sim"]["k1_rr"]), SRC_PRO, "sim/k1_rr: the one-rater RR, simulated", PROCAV); chk("xbLatSimOne", "qLatSimOne")
put("xbLatObsOne", f2(PRO["obs"]["k1_rr"]), SRC_PRO, "obs/k1_rr: the one-rater RR, observed"); chk("xbLatObsOne", "qLatOneObs")
put("xbLatFiveSixOneObs", pct(np.mean([PRO["realism"]["5"]["K_obs"]["1"], PRO["realism"]["6"]["K_obs"]["1"]])), SRC_PRO, "mean of realism/5/K_obs/1 and realism/6/K_obs/1: share of 5- and 6-rater comments with exactly one toxic vote, observed")
put("xbLatFiveSixOneSim", pct(np.mean([PRO["realism"]["5"]["K_sim"]["1"], PRO["realism"]["6"]["K_sim"]["1"]])), SRC_PRO, "the same share, simulated", "the fitted model misses the vote patterns given the number of raters")
chk("xbLatFiveSixOneObs", "qLatFiveSixOne"); chk("xbLatFiveSixOneSim", "qLatFiveSixOneSim")
put("xbLatTenObs", pct(PRO["realism"]["10"]["share_obs"]), SRC_PRO, "realism/10/share_obs: share of reply-edge children with 10 or more raters, observed"); chk("xbLatTenObs", "qLatTenObs")
put("xbLatTenSim", pct(PRO["realism"]["10"]["share_sim"]), SRC_PRO, "realism/10/share_sim: simulated"); chk("xbLatTenSim", "qLatTenSim")
put("xbLatFourObs", pct(PRO["realism"]["4"]["share_obs"]), SRC_PRO, "realism/4/share_obs: share with exactly 4 raters, observed"); chk("xbLatFourObs", "qFourShare")
put("xbLatFourSim", pct(PRO["realism"]["4"]["share_sim"]), SRC_PRO, "realism/4/share_sim: simulated")
put("xbLatProtoRR", f2(PRO["rr_inf"]), SRC_PRO, "rr_inf: the EM fit the protocol check starts from"); chk("xbLatProtoRR", "qLatRREM")
note("xbLatSimGold", "x82_latent.json's check at the observed rater counts (the inventory's E40 fit check) is not used: the keep-out list (K5) supersedes that fit and that check")
# ---- the construct-only test with a monotone f, solved to convergence (x85_construct_{10,20,40}, B85a)
CG = {m_: load(f"x85/x85_construct_{m_}.json") for m_ in ("10", "20", "40")}; SRC_CG = "results/x85/x85_construct_{10,20,40}.json (B85a, grids m = 10, 20, 40)"
nf_ = 0
for m_, J_ in CG.items():
    assert J_["n_comments"] == 1251487 and J_["n_edges"] == 980520 and list(J_["grids"]) == [m_]
    assert J_["monotone_f"] is False and "SLSQP on the increments of the monotone f" in J_["solver"]
    for k_, v_ in J_["grids"][m_]["instruments"].items():
        assert np.all(np.diff(np.array(v_["f"])) >= -1e-12), (m_, k_); nf_ += 1                # every stored f is non-decreasing
CGI = {m_: CG[m_]["grids"][m_]["instruments"] for m_ in CG}; C20 = CGI["20"]
for k in CLS14 + ["haiku", "sonnet"]:
    assert abs(C20[k]["placebo_slope"]["obs"][0] - (L5 if k == "sonnet" else L)[k]["placebo_depth_slope"][0]) < 1e-9 or k in CLS14
nset_ = {m_: len(J_["grids"][m_]["instruments"]) for m_, J_ in CG.items()}; assert len(set(nset_.values())) == 1 and sum(nset_.values()) == nf_
put("xbConsMonoN", str(nf_), SRC_CG, f"number of fitted f vectors stored in the three files ({nset_['20']} settings x {len(CG)} grids), each checked non-decreasing",
    "the files' top-level flag \"monotone_f\": false is a label error: x83d_mono.run sets it to (em_f is em_f_mono), and x85_converged.py passes its own monotone solver em_f_sq "
    "(f = 1e-7 + cumsum of non-negative increments, by SLSQP), so the flag is False although f is constrained monotone; the solver field, Addendum 85 and every stored f agree that f is monotone")
CONSCAV = "model-based (the same latent model, with f non-decreasing in pi); the classifiers' test compares each setting with an instrument of the raters' construct at its own flag rate for each pi"
pe20 = {k: C20[k]["placebo_lrr"]["excess"] for k in CLS14}
put("xbConsExcMin", sg(min(pe20[k][0] for k in CLS12)), SRC_CG, "min over the 12 settings (without Detoxify-unbiased) of instruments/<k>/placebo_lrr/excess at m = 20 (observed minus construct-only placebo log RR)", CONSCAV)
put("xbConsExcMax", sg(max(pe20[k][0] for k in CLS12)), SRC_CG, "max over the 12 settings at m = 20", CONSCAV)
chk("xbConsExcMin", "qMonoExcMin"); chk("xbConsExcMax", "qMonoExcMax")
for m_, nm_ in (("10", "Ten"), ("20", "Twenty"), ("40", "Forty")):
    put(f"xbConsAbove{nm_}", str(sum(CGI[m_][k]["placebo_lrr"]["excess"][1] > 0 for k in CLS14)), SRC_CG, f"number of the 14 settings whose placebo log-RR excess has a CI above 0 at m = {m_}", CONSCAV)
    put(f"xbConsAboveTwelve{nm_}", str(sum(CGI[m_][k]["placebo_lrr"]["excess"][1] > 0 for k in CLS12)), SRC_CG, f"the same over the 12 settings without Detoxify-unbiased at m = {m_}")
chk("xbConsAboveTwelveTwenty", "qMonoNTwenty"); chk("xbConsAboveTwelveTen", "qMonoNTen"); chk("xbConsAboveTwelveForty", "qMonoNForty")
ea_ = [CGI[m_][k]["placebo_lrr"]["excess"][0] for m_ in CGI for k in CLS12]
put("xbConsExcAllMin", sg(min(ea_)), SRC_CG, "min over the 12 settings and the three grids"); chk("xbConsExcAllMin", "qMonoExcAllMin")
put("xbConsExcAllMax", sg(max(ea_)), SRC_CG, "max over the 12 settings and the three grids"); chk("xbConsExcAllMax", "qMonoExcAllMax")
for nm_, k_ in (("Unb", "detoxify_unbiased@m"), ("UnbSix", "detoxify_unbiased@0.6")):
    put3(f"xbCons{nm_}Exc", C20[k_]["placebo_lrr"]["excess"], sg, SRC_CG, f"m = 20: instruments/{k_}/placebo_lrr/excess", CONSCAV + "; Detoxify-unbiased was trained on these raters' labels")
    put(f"xbCons{nm_}ExcTen", sg(CGI["10"][k_]["placebo_lrr"]["excess"][0]), SRC_CG, f"m = 10: instruments/{k_}/placebo_lrr/excess[0]")
    put(f"xbCons{nm_}ExcForty", sg(CGI["40"][k_]["placebo_lrr"]["excess"][0]), SRC_CG, f"m = 40: instruments/{k_}/placebo_lrr/excess[0]")
chk("xbConsUnbExc", "qMonoUnbExc"); chk("xbConsUnbExcLo", "qMonoUnbLo"); chk("xbConsUnbExcHi", "qMonoUnbHi"); chk("xbConsUnbExcTen", "qMonoUnbTenExc")
chk("xbConsUnbSixExc", "qMonoUnbSixExc"); chk("xbConsUnbSixExcLo", "qMonoUnbSixLo"); chk("xbConsUnbSixExcHi", "qMonoUnbSixHi")
eu_ = {m_: CGI[m_]["detoxify_unbiased@m"]["placebo_lrr"]["excess"] for m_ in CGI}; assert eu_["20"][1] > 0 and eu_["10"][1] <= 0 and eu_["40"][1] <= 0
note("xbConsUnbExc", f"Detoxify-unbiased at gold's rate is significant at m = 20 only (lower end {sg(eu_['20'][1], 3)}); not at m = 10 or 40 (lower ends {sg(eu_['10'][1], 3)} and {sg(eu_['40'][1], 3)})")
put("xbConsUnbSigGrids", str(sum(CGI[m_]["detoxify_unbiased@m"]["placebo_lrr"]["excess"][1] > 0 for m_ in CGI)), SRC_CG, "number of the three grids on which Detoxify-unbiased at gold's rate has an excess with a CI above 0")
chk("xbConsUnbSigGrids", "qMonoUnbNSig")
put("xbConsExcLoMin", sg(min(CGI[m_][k]["placebo_lrr"]["excess"][1] for m_ in CGI for k in CLS12), 3), SRC_CG, "smallest lower CI end of the excess over the 12 settings and the three grids")
chk("xbConsExcLoMin", "qMonoExcLoMin")
po_ = {k: math.exp(C20[k]["placebo_lrr"]["obs"][0]) for k in CLS14}; pr_ = {k: math.exp(C20[k]["placebo_lrr"]["pred"][0]) for k in CLS14}
put("xbConsObsMin", f2(min(po_.values())), SRC_CG, "min over the 14 settings of exp(placebo_lrr/obs) (the observed placebo RR)"); put("xbConsObsMax", f2(max(po_.values())), SRC_CG, "max over the 14 settings")
chk("xbConsObsMin", "qConsObsMin"); chk("xbConsObsMax", "qConsObsMax")
put("xbConsPredMin", f2(min(pr_[k] for k in CLS12)), SRC_CG, "min over the 12 settings of exp(placebo_lrr/pred) at m = 20 (the construct-only placebo RR)", CONSCAV)
put("xbConsPredMax", f2(max(pr_[k] for k in CLS12)), SRC_CG, "max over the 12 settings at m = 20", CONSCAV)
chk("xbConsPredMin", "qMonoPredMin"); chk("xbConsPredMax", "qMonoPredMax")
put("xbConsPredFourteenMax", f2(max(pr_.values())), SRC_CG, "max over the 14 settings at m = 20 (Detoxify-unbiased included)")
rp_ = {k: math.exp(C20[k]["lrr"]["pred"][0]) for k in CLS14}
put("xbConsRRPredMin", f2(min(rp_[k] for k in CLS12)), SRC_CG, "min over the 12 settings of exp(lrr/pred): the reply RR an instrument of the raters' construct with the setting's f would report", CONSCAV)
put("xbConsRRPredMax", f2(max(rp_[k] for k in CLS12)), SRC_CG, "max over the 12 settings", CONSCAV); chk("xbConsRRPredMin", "qMonoRRPredMin"); chk("xbConsRRPredMax", "qMonoRRPredMax")
T20 = CG["20"]["grids"]["20"]["thresholds"]["pi>=0.5"]
put3("xbConsThrPlac", [math.exp(t) for t in T20["placebo_lrr"]["pred"]], f2, SRC_CG, "m = 20: thresholds/pi>=0.5/placebo_lrr/pred: the placebo RR of a noise-free instrument of the raters' construct, 1{pi >= 0.5}", CONSCAV)
chk("xbConsThrPlac", "qConsThrPlac"); chk("xbConsThrPlacLo", "qConsThrPlacLo"); chk("xbConsThrPlacHi", "qConsThrPlacHi")
put("xbConsThrPlacTen", f2(math.exp(CG["10"]["grids"]["10"]["thresholds"]["pi>=0.5"]["placebo_lrr"]["pred"][0])), SRC_CG, "m = 10: the same")
put("xbConsThrPlacForty", f2(math.exp(CG["40"]["grids"]["40"]["thresholds"]["pi>=0.5"]["placebo_lrr"]["pred"][0])), SRC_CG, "m = 40: the same")
note("xbConsThrPlac", f"so a placebo RR above the single rater's {M['xbSrPlacRR']} does not by itself show clustering beyond the raters' construct; the construct-only test (each setting against its own f) or the many-rater check is needed")
# the Claude models (escalation): without the parent only
LLMC = [("Haiku", "haiku"), ("Sonnet", "sonnet")]
for nm_, k_ in LLMC:
    r20 = C20[k_]["placebo_slope"]
    put(f"xbCons{nm_}Obs", pp(r20["obs"][0]), SRC_CG, f"m = 20: instruments/{k_}/placebo_slope/obs x 100 (placebo depth slope, points per level){' (5% sample)' if k_ == 'sonnet' else ''}")
    put(f"xbCons{nm_}Pred", pp(r20["pred"][0]), SRC_CG, f"m = 20: instruments/{k_}/placebo_slope/pred x 100 (construct-only)", CONSCAV)
    put3(f"xbCons{nm_}Exc", r20["excess"], pp, SRC_CG, f"m = 20: instruments/{k_}/placebo_slope/excess x 100", CONSCAV)
    put(f"xbCons{nm_}PredTen", pp(CGI["10"][k_]["placebo_slope"]["pred"][0]), SRC_CG, f"m = 10: placebo_slope/pred x 100")
    put(f"xbCons{nm_}PredForty", pp(CGI["40"][k_]["placebo_slope"]["pred"][0]), SRC_CG, f"m = 40: placebo_slope/pred x 100")
chk("xbConsHaikuObs", "qMonoHaikuObs"); chk("xbConsHaikuPred", "qMonoHaikuPred"); chk("xbConsHaikuExcLo", "qMonoHaikuExcLo"); chk("xbConsHaikuExcHi", "qMonoHaikuExcHi")
chk("xbConsSonnetObs", "qMonoSonnetObs"); chk("xbConsSonnetPred", "qMonoSonnetPred"); chk("xbConsSonnetExc", "qMonoSonnetExc")
sh_ = {(m_, k_): CGI[m_][k_]["placebo_slope"]["pred"][0] / CGI[m_][k_]["placebo_slope"]["obs"][0] for m_ in CGI for _, k_ in LLMC}
put("xbConsShareMin", pct(max(0.0, min(sh_.values()))), SRC_CG, "min over Haiku and Sonnet (without the parent) and the three grids of predicted / observed placebo slope, floored at 0", CONSCAV)
put("xbConsShareMax", pct(max(sh_.values())), SRC_CG, "max over the same", CONSCAV)
put("xbConsShareTwentyMin", pct(min(sh_[("20", k_)] for _, k_ in LLMC)), SRC_CG, "min at m = 20"); put("xbConsShareTwentyMax", pct(max(sh_[("20", k_)] for _, k_ in LLMC)), SRC_CG, "max at m = 20")
chk("xbConsShareMin", "qMonoShareMin"); chk("xbConsShareMax", "qMonoShareMax",
    "the earlier-draft macro ranges over the four Claude settings, including the two shown the parent, whose pooled placebo slopes mix the two prompts; without them the maximum is the same (Haiku at m = 40)")
put("xbConsHaikuFourZeroObs", pp(C20["haiku"]["slope_40"]["obs"][0]), SRC_CG, "m = 20: instruments/haiku/slope_40/obs x 100: Haiku's depth slope among comments whose four raters all said civil"); chk("xbConsHaikuFourZeroObs", "qMonoFortyHaikuObs")
put("xbConsHaikuFourZeroPred", pp(C20["haiku"]["slope_40"]["pred"][0]), SRC_CG, "m = 20: slope_40/pred x 100", CONSCAV); chk("xbConsHaikuFourZeroPred", "qMonoFortyHaikuPred")
f4_ = [CGI[m_]["haiku"]["slope_40"]["pred"][0] for m_ in CGI]
put("xbConsHaikuFourZeroPredMin", pp(min(f4_)), SRC_CG, "min over the grids"); put("xbConsHaikuFourZeroPredMax", pp(max(f4_)), SRC_CG, "max over the grids")
chk("xbConsHaikuFourZeroPredMin", "qMonoFortyHaikuPredMin"); chk("xbConsHaikuFourZeroPredMax", "qMonoFortyHaikuPredMax")
note("xbConsHaikuObs", "Haiku and Sonnet shown the parent are left out: their placebo slopes pool roots labeled without the parent and replies labeled with it")
# ---- synthetic checks of the monotone test (x83d_synth: simulated data only; EM solver of x83d_mono)
SDB, SDC = load("x83/synthd_binary.json"), load("x83/synthd_cat.json"); SRC_SY = "results/x83/synthd_binary.json and synthd_cat.json (x83d_synth.py, simulated populations, 30 resamples)"
SYCAV = "simulated data; the monotone EM solver of x83d_mono.py, not the converged SQP of B85a"
for nm_, e_ in (("SydHz", SDB["instruments"]["h0"]["placebo_lrr"]["excess"]), ("SydTop", SDB["instruments"]["clustered"]["placebo_lrr"]["excess"]),
                ("SydCatHz", SDC["instruments"]["h0"]["placebo_lrr"]["excess"]), ("SydCatTop", SDC["instruments"]["clustered"]["placebo_lrr"]["excess"]),
                ("SydCatDrift", SDC["instruments"]["depthy"]["placebo_lrr"]["excess"]), ("SydHzForty", SDB["instruments_m40"]["h0"]["placebo_lrr"]["excess"]),
                ("SydTopForty", SDB["instruments_m40"]["clustered"]["placebo_lrr"]["excess"])):
    put3(f"xb{nm_}", e_, sg, SRC_SY, f"placebo log-RR excess of the simulated instrument ({nm_})", SYCAV)
    if not nm_.endswith("Forty"):
        chk(f"xb{nm_}", f"q{nm_}"); chk(f"xb{nm_}Lo", f"q{nm_}Lo"); chk(f"xb{nm_}Hi", f"q{nm_}Hi")
for nm_, e_ in (("SydDepth", SDB["instruments"]["depthy"]["placebo_slope"]["excess"]), ("SydCatDepth", SDC["instruments"]["depthy"]["placebo_slope"]["excess"]),
                ("SydSfForty", SDB["instruments_m40"]["sonnet"]["placebo_slope"]["excess"])):
    put3(f"xb{nm_}", e_, pp, SRC_SY, f"placebo depth-slope excess x 100 of the simulated instrument ({nm_})", SYCAV)
chk("xbSydDepth", "qSydDepth"); chk("xbSydDepthLo", "qSydDepthLo"); chk("xbSydSfForty", "qSydSfForty"); chk("xbSydSfFortyLo", "qSydSfFortyLo"); chk("xbSydSfFortyHi", "qSydSfFortyHi")
note("xbSydCatDrift", "under the categorical protocol the depth-drifting instrument shows a spurious negative placebo log-RR excess (the conservative direction for the contagion claim)")
note("xbSydSfForty", "a 5%-sized H0 instrument's slope excess is marginally above 0 at m = 40")

# ---- tables: xb_latent (classifiers, contagion), xb_latent_llm (Claude models, escalation), xb_latent_checks
TL = [" & ".join(["Gold", "--", f2(G["rr"][0]), ci3([-x for x in GPJ["gap"][:1]] + [-GPJ["gap"][2], -GPJ["gap"][1]]), "--", "--", "--", "--", "--"]) + " \\\\",
      " & ".join(["Noise-free majority, $1\\{\\theta \\ge 0.5\\}$", "--", ci3(RI["rr_inf"]["rr"], f2), "0", "--", ci3([math.exp(t) for t in T20["placebo_lrr"]["pred"]], f2), "--", "--", "--"]) + " \\\\",
      "\\midrule"]
for nm_, k_, t_, cc_ in FAM:
    for j_, s_ in enumerate((f"{k_}@{t_}", f"{k_}@m")):
        TL.append(" & ".join([fam_name(k_) if j_ == 0 else "", thr_name(s_), f2(IRI[s_]["rr"]), ci3(IRI[s_]["minus_inf"]), f2(po_[s_]), f2(pr_[s_]), ci3(pe20[s_]),
                              sg(CGI["10"][s_]["placebo_lrr"]["excess"][0]), sg(CGI["40"][s_]["placebo_lrr"]["excess"][0])]) + " \\\\")
    TL.append("\\addlinespace[1pt]")
TL = TL[:-1]
table("xb_latent",
      "Contagion against a latent model of the raters. Each comment has a propensity $\\pi$, the share of the rater pool that would call it toxic, and its votes are Binomial given $\\pi$; "
      "the joint distribution of parent and reply propensities over all " + comma(len(chs)) + " reply edges is fitted by nonparametric maximum likelihood on a grid of $m$ points. "
      "Noise-free gold flags $\\pi \\ge 0.5$, and its risk ratio is $\\mathrm{RR}_\\infty$. For each classifier setting: its reply risk ratio and its log ratio to $\\mathrm{RR}_\\infty$ "
      "(95\\% CI over 100 article resamples, $m = 20$), and its placebo risk ratio, observed and as predicted for an instrument that flags a comment with a probability that depends "
      "only on $\\pi$, rises with $\\pi$ and matches the setting's flag rate, with the log excess at $m = 20$ (95\\% CI) and at $m = 10$ and $40$. "
      "The model reproduces gold's risk ratio under the raters' inferred stopping rule but not the vote patterns (Table~\\ref{tab:latentchecks}). $^{\\S}$Trained at least in part on Civil Comments labels.",
      ["Labeler", "Threshold", "Reply RR", "log RR - log RR_inf [95% CI]", "Placebo RR, observed", "Placebo RR, construct only (m = 20)", "Excess, log [95% CI] (m = 20)", "Excess (m = 10)", "Excess (m = 40)"],
      TL, [("Reply RR, log RR - log RR_inf", f"{SRC_RI}: instruments/<k>/rr and minus_inf; RR_inf: rr_inf/rr; gold's row: -(B87b gap) from {SRC_GP}"),
           ("Placebo RR observed / construct only / excess", f"{SRC_CG}: exp(placebo_lrr/obs), exp(placebo_lrr/pred), placebo_lrr/excess at m = 20; excess[0] at m = 10 and 40"),
           ("Noise-free gold row", "thresholds/pi>=0.5/placebo_lrr/pred at m = 20 (its reply RR is RR_inf by construction)")],
      LATCAV + "; the fit checks are in xb_latent_checks.")
TLL = []
for nm_, k_ in LLMC:
    r10, r20, r40 = (CGI[m_][k_] for m_ in ("10", "20", "40"))
    TLL.append(" & ".join([nm_ + ("$^{\\dagger}$" if k_ == "sonnet" else ""), pp(r20["placebo_slope"]["obs"][0]), pp(r10["placebo_slope"]["pred"][0]), pp(r20["placebo_slope"]["pred"][0]),
                           pp(r40["placebo_slope"]["pred"][0]), ci3([PP * t for t in r20["placebo_slope"]["excess"]]),
                           " / ".join(pct(max(0.0, sh_[(m_, k_)])) for m_ in ("10", "20", "40")), pp(r20["slope_40"]["obs"][0]), pp(r20["slope_40"]["pred"][0])]) + " \\\\")
table("xb_latent_llm",
      "The Claude models' false rise against the raters' construct: the placebo depth slope (points per level) observed and as predicted for an instrument that depends on a comment "
      "only through its propensity $\\pi$ (rising with $\\pi$, at the model's flag rate for each $\\pi$), on grids of $m = 10$, 20 and 40 points; the excess at $m = 20$ (95\\% CI); "
      "the predicted share of the observed slope; and the depth slope among comments whose four raters all called them not toxic, observed and predicted ($m = 20$). "
      "Models shown the parent comment are not listed, because they label top-level comments without it. $^{\\dagger}$5\\% sample.",
      ["Labeler", "Placebo depth slope, observed", "Construct only, m = 10", "m = 20", "m = 40", "Excess [95% CI] (m = 20)", "Share predicted (m = 10 / 20 / 40)", "Slope, four civil votes: observed", "predicted (m = 20)"],
      TLL, [("All columns", f"{SRC_CG}: instruments/<haiku|sonnet>/placebo_slope (obs, pred, excess) and slope_40 (obs, pred), x 100; shares pred/obs floored at 0")],
      CONSCAV + "; the + parent rows are omitted because their placebo slopes are pooled over roots and replies.")
TCK = [" & ".join(["$\\mathrm{RR}_\\infty$, $K = 20$", ci3(RI["rr_inf"]["rr"], f2), f"gold {f2(G['rr'][0])}"]) + " \\\\",
       " & ".join(["$\\mathrm{RR}_\\infty$, $K = 10$ and $40$", f"{f2(RI['fits']['10']['rr_inf'])} and {f2(RI['fits']['40']['rr_inf'])}", "--"]) + " \\\\",
       " & ".join(["$\\log \\mathrm{RR}_\\infty - \\log \\mathrm{RR}$(gold)", ci3(GPJ["gap"]), f"$\\mathrm{{RR}}_\\infty \\le$ gold's in {pct(GPJ['share_inf_le_gold'])} of resamples"]) + " \\\\",
       " & ".join(["Classifier settings above / below / within $\\mathrm{RR}_\\infty$", f"{len(ab14)} / {len(bl14)} / {14 - len(ab14) - len(bl14)} of 14", "--"]) + " \\\\",
       "\\midrule",
       " & ".join(["Gold's RR, votes simulated under the inferred stopping rule", f2(PRO["sim"]["gold_rr"]), f"observed {f2(PRO['obs']['gold_rr'])}"]) + " \\\\",
       " & ".join(["One rater's RR, simulated", f2(PRO["sim"]["k1_rr"]), f"observed {f2(PRO['obs']['k1_rr'])}"]) + " \\\\",
       " & ".join(["5- and 6-rater comments with one toxic vote", M["xbLatFiveSixOneSim"], f"observed {M['xbLatFiveSixOneObs']}"]) + " \\\\",
       " & ".join(["Comments with 10 or more raters", M["xbLatTenSim"], f"observed {M['xbLatTenObs']}"]) + " \\\\",
       " & ".join(["Comments with exactly 4 raters", M["xbLatFourSim"], f"observed {M['xbLatFourObs']}"]) + " \\\\",
       "\\midrule",
       " & ".join(["Simulated, two-stage stopping: instrument of the construct", ci3(SDB["instruments"]["h0"]["placebo_lrr"]["excess"]), "truth 0"]) + " \\\\",
       " & ".join(["\\quad instrument with a topic effect", ci3(SDB["instruments"]["clustered"]["placebo_lrr"]["excess"]), "truth $> 0$"]) + " \\\\",
       " & ".join(["\\quad instrument drifting with depth (slope, points per level)", ci3([PP * t for t in SDB["instruments"]["depthy"]["placebo_slope"]["excess"]]), "truth $> 0$"]) + " \\\\",
       " & ".join(["Simulated, categorical answers: instrument of the construct", ci3(SDC["instruments"]["h0"]["placebo_lrr"]["excess"]), "truth 0"]) + " \\\\",
       " & ".join(["\\quad instrument with a topic effect", ci3(SDC["instruments"]["clustered"]["placebo_lrr"]["excess"]), "truth $> 0$"]) + " \\\\",
       " & ".join(["\\quad instrument drifting with depth (placebo log RR)", ci3(SDC["instruments"]["depthy"]["placebo_lrr"]["excess"]), "none built in"]) + " \\\\",
       " & ".join(["Simulated 5\\% sample, instrument of the construct (slope, $K = 40$)", ci3([PP * t for t in SDB["instruments_m40"]["sonnet"]["placebo_slope"]["excess"]]), "truth 0"]) + " \\\\"]
table("xb_latent_checks",
      "Checks of the latent model. Top: noise-free contagion on Civil Comments (95\\% CIs over 100 article resamples). Middle: votes simulated from the fitted model under the stopping rule "
      "inferred from the data (four raters, more when they disagree) against the observed votes; the simulation starts from an earlier fit with $\\mathrm{RR}_\\infty = " + M["xbLatProtoRR"] + "$. "
      "Bottom: the construct-only test on simulated populations with a known model (placebo log-RR excess unless stated; $m = 20$ unless stated; 30 resamples).",
      ["Check", "Model or test [95% CI]", "Observed or truth"],
      TCK, [("Top", f"{SRC_RI}; {SRC_GP}"), ("Middle", SRC_PRO + " (sim, obs, realism)"), ("Bottom", SRC_SY + " (instruments/h0, clustered, depthy; instruments_m40/sonnet)")],
      PROCAV + "; " + SYCAV + ".")

# ================================================================ D. The other corpora (E45-E54) and the rewrite across corpora
sec("D. The other corpora (E45-E54): own question, construct control, dose, laws under the toxicity question, contagion, rewrite")
X89 = load("x89/x89_other_tox.json"); SRC_X89 = "results/x89/x89_other_tox.json (B89a: Haiku and Sonnet asked the Civil Comments toxicity question; gold = each corpus's own)"
X89C = load("x89/x89_cc_reference.json"); SRC_X89C = "results/x89/x89_cc_reference.json (the same statistics on Civil Comments: cc_threads, and cc_s5 for Sonnet)"
assert X89C["cc_threads"]["N"] == 1251487 and X89C["cc_s5"]["N"] == 60668 and X89C["cc_threads"]["edges"] == 980520
OC = [("cad", "Cad", "CAD", "abuse"), ("qian_reddit", "Qr", "Qian et al.'s Reddit", "hate"), ("qian_gab", "Gab", "Qian et al.'s Gab", "hate"), ("cga", "Cga", "CGA-Wiki", "attack"),
      ("bad", "Bad", "BAD", "unsafe turn")]
NPOP = {"cad": 26550, "qian_reddit": 22309, "qian_gab": 33776, "cga": 5340, "bad": 78874}
POPS = {}
for c_, *_ in OC:
    POPS[c_] = pd.read_parquet(f"{RES}/x14_pops/{c_}.parquet", columns=["id", "unit", "cluster", "depth", "z", "parent_row"])
    assert len(POPS[c_]) == NPOP[c_] == X89[c_]["N"], c_
# per-corpus records (Civil Comments from x89_cc_reference: cc_threads for gold and Haiku, cc_s5 for Sonnet)
REC = {"cc": dict(gold=X89C["cc_threads"]["gold"], haiku=X89C["cc_threads"]["instruments"]["haiku_tox"], sonnet=X89C["cc_s5"]["instruments"]["sonnet_tox"],
                  haiku_ctx=X89C["cc_threads"]["instruments"]["haiku_tox_ctx"], sonnet_ctx=X89C["cc_s5"]["instruments"]["sonnet_tox_ctx"], gold5=X89C["cc_s5"]["gold"])}
for c_, *_ in OC:
    I_ = X89[c_]["instruments"]
    REC[c_] = dict(gold=X89[c_]["gold"], haiku=I_["haiku_tox"], sonnet=I_["sonnet_tox"], haiku_ctx=I_["haiku_tox_ctx"], sonnet_ctx=I_["sonnet_tox_ctx"])
CTAG = {"cc": "Cc"} | {c_: t_ for c_, t_, _, _ in OC}
QN = "Haiku and Sonnet under the Civil Comments toxicity question"
for c_, t_ in CTAG.items():
    R_ = REC[c_]; srcR = SRC_X89C if c_ == "cc" else SRC_X89; s5 = " (Sonnet: 5% sample)" if c_ == "cc" else ""
    for law_, key_, kd_ in (("Depth", "depth", "distortion"), ("Replies", "depth_replies", "depth_replies_distortion"), ("Within", "within", "within_distortion")):
        put3(f"xb{t_}Gold{law_}", R_["gold"][key_], sg, srcR, f"{c_}/gold/{key_} (points per level; {'depth >= 1' if key_ == 'depth_replies' else ('thread fixed effects' if key_ == 'within' else 'all comments')})")
        for m_, nm_ in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
            put(f"xb{t_}{nm_}{law_}", sg(R_[m_][key_][0]), srcR, f"{c_}/instruments/{m_}_tox/{key_}[0]{s5}", QN)
            put3(f"xb{t_}{nm_}{law_}Dist", R_[m_][kd_], sg, srcR, f"{c_}/instruments/{m_}_tox/{kd_} (labeler minus gold, paired; 500 conversation-cluster resamples, seed 8901){s5}", QN)
    for m_, nm_ in (("haiku_ctx", "HaikuCtx"), ("sonnet_ctx", "SonnetCtx")):
        put3(f"xb{t_}{nm_}RepliesDist", R_[m_]["depth_replies_distortion"], sg, srcR, f"{c_}/instruments/{m_.replace('_ctx', '_tox_ctx')}/depth_replies_distortion: shown the parent, among replies{s5}", QN)
    if c_ != "cga":
        put3(f"xb{t_}GoldLogRR", R_["gold"]["log_rr"], sg, srcR, f"{c_}/gold/log_rr (log reply RR)")
        put(f"xb{t_}GoldRR", f2(math.exp(R_["gold"]["log_rr"][0])), srcR, f"exp({c_}/gold/log_rr[0])")
        for m_, nm_ in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
            put(f"xb{t_}{nm_}LogRR", sg(R_[m_]["log_rr"][0]), srcR, f"{c_}/instruments/{m_}_tox/log_rr[0]{s5}", QN)
            put3(f"xb{t_}{nm_}LrrDist", R_[m_]["log_rr_distortion"], sg, srcR, f"{c_}/instruments/{m_}_tox/log_rr_distortion (paired){s5}", QN)
    put3(f"xb{t_}Plac", R_["haiku"]["placebo"], sg, srcR, f"{c_}/instruments/haiku_tox/placebo: Haiku's placebo depth slope among gold negatives (points per level)", QN)
    put3(f"xb{t_}PlacYou", R_["haiku"]["placebo_within_you"], sg, srcR, f"{c_}/instruments/haiku_tox/placebo_within_you: pooled within-stratum slope, strata = with and without a second-person word", QN)
    put3(f"xb{t_}ToxGapHaiku", R_["haiku"]["fpr_you_minus_other"], lambda x: f(PP * x, 1), srcR, f"{c_}/instruments/haiku_tox/fpr_you_minus_other x 100 (points)", QN)
    put3(f"xb{t_}ToxGapSonnet", R_["sonnet"]["fpr_you_minus_other"], lambda x: f(PP * x, 1), srcR, f"{c_}/instruments/sonnet_tox/fpr_you_minus_other x 100{s5}", QN)
put3("xbCcGoldSfiveDepth", REC["cc"]["gold5"]["depth"], sg, SRC_X89C, "cc_s5/gold/depth (the 5% sample's gold, Sonnet's reference)")
put3("xbCcGoldSfiveReplies", REC["cc"]["gold5"]["depth_replies"], sg, SRC_X89C, "cc_s5/gold/depth_replies")
put3("xbCcGoldSfiveWithin", REC["cc"]["gold5"]["within"], sg, SRC_X89C, "cc_s5/gold/within")
put3("xbCcGoldSfiveLogRR", REC["cc"]["gold5"]["log_rr"], sg, SRC_X89C, "cc_s5/gold/log_rr")
# cross-checks with the earlier-draft macros (Table 4, Section 3, Table 2 rows)
chk("xbCcGoldDepth", "wGoldDepthPP"); chk("xbCcHaikuDepth", "wHaikuDepthPP")
chk("xbGabGoldDepth", "cGabGold"); chk("xbGabHaikuDepth", "cGabHaiku"); chk("xbBadGoldDepth", "cBadGold"); chk("xbBadHaikuDepth", "cBadHaiku")
chk("xbQrGoldDepth", "qRedditPooledPP"); chk("xbGabGoldDepth", "qGabPooledPP"); chk("xbCgaGoldDepth", "qCgaPooledPP"); chk("xbBadGoldDepth", "qBadPooledPP"); chk("xbCadGoldDepth", "qCadPooledPP")
chk("xbQrGoldWithin", "qRedditWithinPP"); chk("xbGabGoldWithin", "qGabWithinPP"); chk("xbCgaGoldWithin", "qCgaWithinPP"); chk("xbBadGoldWithin", "qBadWithinPP"); chk("xbCadGoldWithin", "qCadWithinPP")
chk("xbCcGoldWithin", "qGoldWithinPP"); chk("xbCcHaikuWithin", "qHaikuWithinPP"); chk("xbCcSonnetWithin", "qSonnetWithinPP"); chk("xbCcGoldSfiveWithin", "qGoldWithinSfivePP")
chk("xbQrGoldReplies", "cQrRepGold"); chk("xbCcGoldReplies", "cGoldReplies"); chk("xbCcHaikuReplies", "cHaikuReplies")
chk("xbCcHaikuWithinDist", "dHaikuWithin"); chk("xbCcSonnetWithinDist", "dSonnetWithin")
chk("xbCcHaikuCtxRepliesDist", "dHaikuCtx"); chk("xbCcSonnetCtxRepliesDist", "dSonnetCtx")
chk("xbQrPlac", "cQrPlac"); chk("xbGabPlac", "cGabPlac"); chk("xbCgaPlac", "cCgaPlac"); chk("xbCadPlac", "cCadPlac"); chk("xbCcPlac", "cPlacHaiku")
chk("xbQrPlacYou", "cQrPlacYou"); chk("xbGabPlacYou", "cGabPlacYou"); chk("xbCcPlacYou", "cPlacHaikuYou")
chk("xbCcToxGapHaiku", "cYouGapHaiku"); chk("xbCcToxGapSonnet", "cYouGapSonnet")
og_ = [REC[c_]["haiku"]["fpr_you_minus_other"][0] for c_, *_ in OC if c_ != "bad"]
put("xbOthGapMin", f(PP * min(og_), 0), SRC_X89, "min over CAD, Qian Reddit, Qian Gab and CGA-Wiki of Haiku's fpr_you_minus_other x 100 (toxicity question)"); chk("xbOthGapMin", "cOthGapMin")
put("xbOthGapMax", f(PP * max(og_), 0), SRC_X89, "max over the four forums"); chk("xbOthGapMax", "cOthGapMax")
note("xbCadHaikuDepth", "E54: shown the parent, CAD's pooled depth distortion (\\xbCadHaikuCtxPooledDist) is the prompt switch at depth 1; among replies it is \\xbCadHaikuCtxRepliesDist")
put3("xbCadHaikuCtxPooledDist", REC["cad"]["haiku_ctx"]["distortion"], sg, SRC_X89, "cad/instruments/haiku_tox_ctx/distortion: POOLED over roots (no parent) and replies (with parent)",
     "pooled + parent value, reported only to show the root/reply prompt switch; not a law distortion")
# ---- E52: thread size laws under the toxicity question (re-analysis of results/x89/labels, ctx0) and under each corpus's own question (x74_decomp)
DEC = load("x74_decomp/x74_decomp.json"); SRC_DEC = "results/x74_decomp/x74_decomp.json (labelers' corpus-question labels and the classifiers; units with >= 2 comments)"
SRC_LAB = "re-analysis of results/x89/labels/<corpus>_<haiku|sonnet>_ctx0/shard_*.jsonl (flag = answer A or B; unparsed answers are not flags, as x89_other_tox.py)"


def load_ans(corpus, model):
    out = {}
    for fn_ in sorted(glob.glob(f"{RES}/x89/labels/{corpus}_{model}_ctx0/shard_*.jsonl")):
        for l in open(fn_):
            r_ = json.loads(l); out[int(r_["id"])] = r_.get("ans")
    return out


def size_laws(P, ys, nb=2000, seed=9821, chunk=500):
    """Thread size law as x74_decomp.py: OLS slope of a unit's flag share on log2 of its size over units with >= 2 comments, each unit
    once; 2,000 Poisson resamples of the population's clusters (conversations here), the same weights for every labeler (paired)."""
    uu = pd.factorize(P.unit.values)[0]; ku = np.bincount(uu).astype(float); ok = ku >= 2; xu = np.log2(np.maximum(ku, 1))
    cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); ucl = np.zeros(ku.size, np.int64); ucl[uu] = cl
    base = [np.bincount(ucl[ok], w_, nC) for w_ in (np.ones(int(ok.sum())), xu[ok], xu[ok] ** 2)]
    S_ = {}
    for nm_, y_ in ys.items():
        fr = np.bincount(uu, y_, ku.size) / ku; S_[nm_] = [np.bincount(ucl[ok], fr[ok], nC), np.bincount(ucl[ok], xu[ok] * fr[ok], nC)]
    def slope(w_):
        n_, sx_, sxx_ = (w_ @ b_ for b_ in base); return {nm_: (n_ * (w_ @ v_[1]) - sx_ * (w_ @ v_[0])) / (n_ * sxx_ - sx_ ** 2) for nm_, v_ in S_.items()}
    est = {k: float(v) for k, v in slope(np.ones(nC)).items()}; rng_ = np.random.default_rng(seed); draws = {k: [] for k in ys}
    for _ in range(nb // chunk):
        wb_ = rng_.poisson(1.0, size=(chunk, nC)).astype(float); sb_ = slope(wb_)
        for k in ys:
            draws[k].append(sb_[k])
    draws = {k: np.concatenate(v) for k, v in draws.items()}
    return est, draws, int(ok.sum())


def ci_from(est, dr):
    return [float(est), float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]


SZ = {}
for c_, t_, nmc_, q_ in OC:
    P_ = POPS[c_]; ids_ = P_.id.astype(np.int64).values; ys = {"gold": P_.z.values.astype(float)}
    for m_ in ("haiku", "sonnet"):
        A_ = load_ans(c_, m_); assert len(A_) == len(P_), (c_, m_, len(A_))
        a_ = np.array([A_.get(int(i)) for i in ids_], dtype=object); ys[m_] = np.isin(a_, ["A", "B"]).astype(float)
        assert abs(ys[m_].mean() - REC[c_][m_]["flag_rate"][0]) < 1e-12                      # the same flags as x89_other_tox.py
    PF_ = pd.read_parquet(f"{RES}/x14_pops/{c_}.parquet")
    for col_ in ("haiku", "sonnet"):                                                          # the corpus-question labels, to reproduce x74_decomp exactly
        if f"y:{col_}" in PF_.columns:
            ys[f"own_{col_}"] = np.nan_to_num(PF_[f"y:{col_}"].values.astype(float))
    est_, dr_, nu_ = size_laws(P_, ys)
    assert abs(est_["gold"] - DEC[c_]["gold"]["size"][0]) < 1e-12                               # reproduces x74_decomp's gold size law
    for k in [k for k in est_ if k.startswith("own_")]:
        assert abs(est_[k] - DEC[c_]["labelers"][k[4:]]["laws"]["size"]["law"][0]) < 1e-12      # and its corpus-question labelers' laws
    SZ[c_] = dict(est=est_, dr=dr_, nu=nu_)
    put3(f"xb{t_}GoldSize", ci_from(est_["gold"], dr_["gold"]), pp, f"{SRC_DEC} for the estimate (reproduced to 1e-12); CI from this script's 2,000 conversation-cluster resamples (seed 9821)",
         f"{c_}: gold's thread size law x 100 (points per doubling; {nu_} units with >= 2 comments)")
    for m_, nm_ in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
        put3(f"xb{t_}{nm_}Size", ci_from(est_[m_], dr_[m_]), pp, SRC_LAB, f"{c_}: {m_}'s thread size law x 100 under the toxicity question, without the parent", QN)
        put3(f"xb{t_}{nm_}SizeDist", ci_from(est_[m_] - est_["gold"], dr_[m_] - dr_["gold"]), pp, SRC_LAB, f"{c_}: {m_}'s size law minus gold's, x 100 (paired resamples)", QN)
# the corpus question (x74_decomp): Haiku (and Sonnet on CAD), and how many settings without the parent attenuate gold's size law
for c_, t_, nmc_, q_ in OC:
    Ld_ = DEC[c_]["labelers"]; gsz = DEC[c_]["gold"]["size"][0]
    if "haiku" in Ld_:
        put3(f"xb{t_}HaikuSizeOwn", Ld_["haiku"]["laws"]["size"]["law"], pp, SRC_DEC, f"{c_}/labelers/haiku/laws/size/law x 100: Haiku asked the corpus's own question ({q_})", "own question, not the toxicity question")
    nop_ = [k for k in Ld_ if k != "human_lenient" and not k.endswith("_ctx") and "_ctx@" not in k]
    att_ = [k for k in nop_ if abs(Ld_[k]["laws"]["size"]["law"][0]) < abs(gsz) and Ld_[k]["laws"]["size"]["law"][0] * gsz > 0]
    inf_ = [k for k in nop_ if abs(Ld_[k]["laws"]["size"]["law"][0]) > abs(gsz) and Ld_[k]["laws"]["size"]["law"][0] * gsz > 0]
    rev_ = [k for k in nop_ if Ld_[k]["laws"]["size"]["law"][0] * gsz < 0]
    put(f"xb{t_}SizeAttenN", str(len(att_)), SRC_DEC, f"{c_}: settings without the parent (corpus-question LLMs and the 14 classifier settings) whose size law has gold's sign and a smaller magnitude", "own question")
    put(f"xb{t_}SizeInflN", str(len(inf_)), SRC_DEC, f"{c_}: those with gold's sign and a larger magnitude", "own question")
    put(f"xb{t_}SizeRevN", str(len(rev_)), SRC_DEC, f"{c_}: those with the opposite sign", "own question")
    put(f"xb{t_}SizeOf", str(len(nop_)), SRC_DEC, f"{c_}: number of settings without the parent")
put3("xbCadSonnetSizeOwn", DEC["cad"]["labelers"]["sonnet"]["laws"]["size"]["law"], pp, SRC_DEC, "cad/labelers/sonnet/laws/size/law x 100 (abuse question)", "own question")
bims = DEC["bad"]["labelers"]["imsypp@m"]["laws"]["size"]["law"][0]
put("xbBadImsSizeOwn", pp(bims), SRC_DEC, "bad/labelers/imsypp@m/laws/size/law[0] x 100 (IMSYPP at gold's flag rate)")
note("xbBadSizeAttenN", "on BAD the only stored corpus-question LLM labels are Haiku shown the previous turn (pooled), so BAD's count is over the 14 classifier settings")
# ---- E46: each corpus's own question (x92_extra, B92d)
XE = load("x92/x92_extra.json"); SRC_XE = "results/x92/x92_extra.json (B92d: stored corpus-question labels; 500 conversation-cluster resamples, seed 9203)"
OWN2 = {"cad": ("sonnet", "Sonnet"), "qian_reddit": ("gemma_m", "Gemma"), "qian_gab": ("gemma_m", "Gemma"), "cga": ("qwen7b_m", "Qwen")}
ogs_ = []
for c_, t_, nmc_, q_ in OC[:4]:
    E_ = XE[c_]; put3(f"xb{t_}OwnGapHaiku", E_["haiku"]["gap"], lambda x: f(PP * x, 1), SRC_XE, f"{c_}/haiku/gap x 100: Haiku's FPR on gold negatives with a second-person word minus without, under the {q_} question", "own question")
    k2, n2 = OWN2[c_]
    put3(f"xb{t_}OwnGap{n2}", E_[k2]["gap"], lambda x: f(PP * x, 1), SRC_XE, f"{c_}/{k2}/gap x 100 ({n2}{' at gold' + chr(39) + 's flag rate' if k2 != 'sonnet' else ''}, {q_} question)", "own question")
    ogs_ += [E_["haiku"]["gap"], E_[k2]["gap"]]
assert all(g_[1] > 0 for g_ in ogs_)
put("xbOwnGapMin", f(PP * min(g_[0] for g_ in ogs_), 1), SRC_XE, "min over the eight own-question gaps (all CIs above 0)"); chk("xbOwnGapMin", "cOwnGapMin")
put("xbOwnGapMax", f(PP * max(g_[0] for g_ in ogs_), 1), SRC_XE, "max over the eight"); chk("xbOwnGapMax", "cOwnGapMax")
# Qwen on Civil Comments (5% sample, gold's flag rate, no parent) from x90's stored rates by number of second-person words
DO = load("x90/x90_dose.json")["corpora"]; SRC_DO = "results/x90/x90_dose.json"
qw_ = DO["cc_s5"]["labelers"]["qwen"]; fq_ = [t[0] for t in qw_["fpr_by_count"]]; nq_ = qw_["n_by_count"]
put("xbCcGapQwen", f(PP * ((sum(fq_[j] * nq_[j] for j in range(1, 5)) / sum(nq_[1:])) - fq_[0]), 1), SRC_DO,
    "cc_s5/labelers/qwen: FPR over 1-4+ second-person words (weighted by n_by_count) minus FPR at 0, x 100 (Qwen at gold's flag rate, without the parent; 5% sample)",
    "point value only (no CI stored); deterministic function of stored rates and counts")
put("xbCcRatioQwen", f(fq_[4] / fq_[0], 1), SRC_DO, "cc_s5/labelers/qwen: FPR at 4+ second-person words over FPR at 0 (Qwen at gold's flag rate, 5% sample)")
_lq = [l_ for l_ in open(f"{W}/results/reference/tables/llms_v6.tex") if l_.startswith("Qwen &")][0]
assert [c_.strip() for c_ in _lq.split("&")][3] == M["xbCcRatioQwen"], (_lq, M["xbCcRatioQwen"])          # = Table llms's 4+ : 0 cell
# ---- E47: the construct control, E48: dose ratios and the share of the law excess carried by "you" (x92_construct, x90)
XC = load("x92/x92_construct.json"); SRC_XC = "results/x92/x92_construct.json (B92b: Haiku without the parent under the toxicity question; 500 conversation-cluster resamples, seed 9202)"
assert XC["cc_threads"]["n_neg"] == 1156062
rr3, gp3, ds3 = {}, {}, {}
for c_, t_ in [("cc_threads", "Cc")] + [(c_, t_) for c_, t_, _, _ in OC]:
    B_ = XC[c_]["below_median"]["haiku"]; A_ = XC[c_]["all"]["haiku"]
    put3(f"xb{t_}CtrlRatio", B_["ratio3"], lambda x: f(x, 1), SRC_XC, f"{c_}/below_median/haiku/ratio3: FPR with 3+ second-person words / with none, among gold negatives whose Detoxify score is below the median of the corpus's gold negatives", QN)
    put3(f"xb{t_}CtrlGap", B_["gap"], lambda x: f(PP * x, 1), SRC_XC, f"{c_}/below_median/haiku/gap x 100 (points)", QN)
    put3(f"xb{t_}DoseRatio", A_["ratio3"], lambda x: f(x, 1), SRC_XC, f"{c_}/all/haiku/ratio3: FPR with 3+ second-person words / with none, all gold negatives", QN)
    h_ = DO["cc_threads" if c_ == "cc_threads" else c_]["labelers"]["haiku"]; fr_ = [t[0] for t in h_["fpr_by_count"]]; n_ = h_["n_by_count"]
    assert abs(((fr_[3] * n_[3] + fr_[4] * n_[4]) / (n_[3] + n_[4])) / fr_[0] - A_["ratio3"][0]) < 1e-9           # the same ratio as x90 (cOthDose*)
    rr3[c_] = B_["ratio3"]; gp3[c_] = B_["gap"]; ds3[c_] = A_["ratio3"]
four = [c_ for c_, *_ in OC[:4]]
put("xbCtrlRatioMin", f(min(rr3[c_][0] for c_ in four), 1), SRC_XC, "min over the four forums of the construct-control ratio"); chk("xbCtrlRatioMin", "cDetoxCtrlMin")
put("xbCtrlRatioMax", f(max(rr3[c_][0] for c_ in four), 0), SRC_XC, "max over the four forums (0 decimals, as the earlier-draft macro)"); chk("xbCtrlRatioMax", "cDetoxCtrlMax")
put("xbCtrlGapMin", f(PP * min(gp3[c_][0] for c_ in four), 1), SRC_XC, "min over the four forums of the construct-control gap (points)"); chk("xbCtrlGapMin", "cDetoxGapMin")
put("xbCtrlGapMax", f(PP * max(gp3[c_][0] for c_ in four), 0), SRC_XC, "max over the four forums"); chk("xbCtrlGapMax", "cDetoxGapMax")
assert all(gp3[c_][1] > 0 for c_ in four + ["cc_threads"]) and gp3["bad"][2] < 0
put("xbDoseRatioMin", f(min(v[0] for v in ds3.values()), 1), SRC_XC, "min over the six corpora of the 3+ : 0 ratio"); chk("xbDoseRatioMin", "cOthDoseMin")
put("xbDoseRatioMax", f(max(v[0] for v in ds3.values()), 0), SRC_XC, "max over the six corpora"); chk("xbDoseRatioMax", "cOthDoseMax")
for c_, t_ in (("cc_threads", "Cc"), ("qian_reddit", "Qr"), ("qian_gab", "Gab")):
    h_ = DO[c_]["labelers"]["haiku"]
    put3(f"xb{t_}LawEx", h_["law_distortion"], sg, SRC_DO, f"{c_}/labelers/haiku/law_distortion: Haiku's depth-law distortion", QN)
    put3(f"xb{t_}LawExFix", h_["law_distortion_within_count"], sg, SRC_DO, f"{c_}/labelers/haiku/law_distortion_within_count: the same at a fixed number of second-person words", QN)
    put(f"xb{t_}LawShare", pct(1 - h_["law_distortion_within_count"][0] / h_["law_distortion"][0]), SRC_DO, f"1 - law_distortion_within_count / law_distortion ({c_})", QN)
chk("xbQrLawEx", "cQrLawEx"); chk("xbQrLawExFix", "cQrLawExFix"); chk("xbQrLawShare", "cQrLawShare"); chk("xbGabLawEx", "cGabLawEx"); chk("xbGabLawExFix", "cGabLawExFix"); chk("xbGabLawShare", "cGabLawShare")
chk("xbCcLawEx", "cDoseLawHaiku"); chk("xbCcLawExFix", "cDoseLawHaikuFix")
_fx = DO["cc_threads"]["labelers"]["haiku"]["law_distortion_within_count"]; assert _fx[0] < 0 and _fx[1] < 0 < _fx[2]
note("xbCcLawShare", "above 100%: at a fixed number of second-person words Haiku's distortion is slightly negative (CI covers 0)")
for c_, t_, *_ in OC:
    put3(f"xb{t_}PlacCount", DO[c_]["labelers"]["haiku"]["within_count"], sg, SRC_DO, f"{c_}/labelers/haiku/within_count: placebo slope pooled within the number of second-person words (0-4+)", QN)
put3("xbCcPlacCount", DO["cc_threads"]["labelers"]["haiku"]["within_count"], sg, SRC_DO, "cc_threads/labelers/haiku/within_count")
chk("xbQrPlacCount", "cQrPlacCount"); chk("xbGabPlacCount", "cGabPlacCount"); chk("xbCadPlacCount", "cCadPlacCount"); chk("xbCgaPlacCount", "cCgaPlacCount"); chk("xbCcPlacCount", "cDoseWithinHaiku")
yb_ = DO["bad"]["you_share_by_depth"]
put("xbBadYouMin", pct(min(yb_)), SRC_DO, "bad/you_share_by_depth: min over depths of the share of gold-negative turns with a second-person word"); chk("xbBadYouMin", "cBadYouMin")
put("xbBadYouMax", pct(max(yb_)), SRC_DO, "max over depths"); chk("xbBadYouMax", "cBadYouMax")
# ---- E50: within-thread laws under each corpus's own question, all settings without the parent (x75_within, B75e)
WI = load("x75_within/x75_within.json")["within"]; SRC_WI = "results/x75_within/x75_within.json (B75e: corpus-question labels and the 14 classifier settings; thread fixed effects)"
ESC = ["qian_reddit", "qian_gab", "cga", "bad"]; tot_n = tot_w = tot_r = tot_all = tot_wall = tot_rall = 0; shr_ = []
for c_ in ESC:
    gw = WI[c_]["gold_within"][0]; assert abs(PP * gw - REC[c_]["gold"]["within"][0]) < 1e-9                                  # the same gold as x89
    nop_ = {k: v for k, v in WI[c_]["labelers"].items() if k != "human_lenient" and not k.endswith("_ctx") and "_ctx@" not in k}
    alls_ = {k: v for k, v in WI[c_]["labelers"].items() if k != "human_lenient"}
    w_ = [k for k, v in nop_.items() if abs(gw + v["distortion"][0]) < abs(gw)]; r_ = [k for k, v in nop_.items() if (gw + v["distortion"][0]) * gw < 0]
    assert all(gw + v["distortion"][0] < gw for v in nop_.values())                            # every one is lower
    put(f"xb{CTAG[c_]}WeakerN", str(len(w_)), SRC_WI, f"{c_}: settings without the parent whose within-thread law is weaker than gold's (|law| < |gold's|)", "own question")
    put(f"xb{CTAG[c_]}WeakerOf", str(len(nop_)), SRC_WI, f"{c_}: number of settings without the parent")
    tot_n += len(nop_); tot_w += len(w_); tot_r += len(r_)
    tot_all += len(alls_); tot_wall += sum(abs(gw + v["distortion"][0]) < abs(gw) for v in alls_.values()); tot_rall += sum((gw + v["distortion"][0]) * gw < 0 for v in alls_.values())
    shr_ += [(v["attenuation_term"][0] - gw) / v["distortion"][0] for v in nop_.values() if abs(v["distortion"][0]) > 0]
put("xbWeakerN", str(tot_w), SRC_WI, "over Qian Reddit, Qian Gab, CGA-Wiki and BAD (where gold escalates within threads): settings without the parent weaker than gold's within-thread law", "own question")
put("xbWeakerOf", str(tot_n), SRC_WI, "number of settings without the parent on the four corpora")
put("xbWeakerRevN", str(tot_r), SRC_WI, "of them, those that reverse the sign of gold's within-thread law")
put("xbWeakerAllN", str(tot_wall), SRC_WI, "the same count including the settings shown the parent (pooled), as the earlier-draft macro"); put("xbWeakerAllOf", str(tot_all), SRC_WI, "all stored settings on the four corpora")
chk("xbWeakerAllN", "qNFlatten"); chk("xbWeakerAllOf", "qNFlattenOf")
put("xbWeakerRevAllN", str(tot_rall), SRC_WI, "sign reversals including the settings shown the parent"); chk("xbWeakerRevAllN", "qNFlattenRev")
put("xbAttShareMed", pct(float(np.median(shr_))), SRC_WI, "median over the settings without the parent of (attenuation_term - gold's law) / distortion: the share of the within-thread shortfall due to unstructured errors")
assert int(EX["qNFlattenOf"][0]) == tot_all
chk("xbAttShareMed", "qAttShareMed", f"the earlier-draft macro takes the median over all {tot_all} settings, including the {tot_all - tot_n} shown the parent")
note("xbWeakerN", "every setting without the parent is lower than gold's within-thread law on the four corpora (asserted)")
# ---- E53: contagion under the hate question on Qian et al.'s corpora (f122, final run; K16's first run is not used)
F122 = load("f122_qian.json"); SRC_122 = "results/f122_qian.json (final Qian run, hate question; 2,000 conversation-cluster resamples)"
assert F122["platforms"]["gab"]["n_posts"] == 33776 and F122["platforms"]["reddit"]["n_posts"] == 22309
for pl_, t_ in (("reddit", "Qr"), ("gab", "Gab")):
    a_ = F122["platforms"][pl_]["audit"]
    put(f"xb{t_}HateRRGold", f2(a_["rr_z"][0]), SRC_122, f"platforms/{pl_}/audit/rr_z[0]: gold's reply RR")
    put(f"xb{t_}HateRRHaiku", f2(a_["rr_y0"][0]), SRC_122, f"platforms/{pl_}/audit/rr_y0[0]: Haiku asked the hate question, without the parent", "own question")
    put3(f"xb{t_}HateLrrDist", a_["H25e_logrr_y0_minus_gold"], sg, SRC_122, f"platforms/{pl_}/audit/H25e_logrr_y0_minus_gold: Haiku's log RR minus gold's", "own question")
    put3(f"xb{t_}HateCtxChg", a_["H25f_logrr_y1_minus_y0"], sg, SRC_122, f"platforms/{pl_}/audit/H25f_logrr_y1_minus_y0: log RR with the parent shown minus without (the parent's label is the context-free one, so this is among replies)", "own question")
    assert abs(math.log(a_["rr_y0"][0]) - DEC[f"qian_{pl_}"]["gold"]["log_rr"][0] - DEC[f"qian_{pl_}"]["labelers"]["haiku"]["laws"]["log_rr"]["distortion"][0]) < 1e-6   # = x74_decomp's hate labels
# ---- the rewrite of "you" on gold-negative second-person comments (x89_rewrite_other; Civil Comments re-analysed from the x77_youfree labels)
RWO = load("x89/x89_rewrite_other.json"); SRC_RWO = "results/x89/x89_rewrite_other.json (B89d; 2,000 conversation-cluster resamples, seed 8909)"
for c_, t_, *_ in OC[:4]:
    for part_, nm_ in (("all", "All"), ("deep", "Deep")):
        put3(f"xb{t_}Rw{nm_}", RWO[c_][part_]["diff"], lambda x: sg(PP * x, 1), SRC_RWO, f"{c_}/{part_}/diff x 100: change in Haiku's flag rate (edited minus original){' at depth >= 2' if part_ == 'deep' else ''}", QN)
    put(f"xb{t_}RwN", comma(RWO[c_]["all"]["n"]), SRC_RWO, f"{c_}/all/n: gold-negative comments with a second-person word")
rw4 = [-RWO[c_]["all"]["diff"][0] for c_, *_ in OC[:4]]; assert all(RWO[c_]["all"]["diff"][2] < 0 and RWO[c_]["deep"]["diff"][2] < 0 for c_, *_ in OC[:4])
put("xbRwAllMin", f(PP * min(rw4), 1), SRC_RWO, "min over the four forums of the drop in Haiku's flag rate (points)"); chk("xbRwAllMin", "cRoMin")
put("xbRwAllMax", f(PP * max(rw4), 1), SRC_RWO, "max over the four forums"); chk("xbRwAllMax", "cRoMax")
rwd = [-RWO[c_]["deep"]["diff"][0] for c_, *_ in OC[:4]]
put("xbRwDeepMin", f(PP * min(rwd), 1), SRC_RWO, "min over the four forums of the drop at depth >= 2"); put("xbRwDeepMax", f(PP * max(rwd), 1), SRC_RWO, "max over the four forums at depth >= 2")
P5 = pd.read_parquet(f"{RES}/x14_pops/cc_s5.parquet", columns=["id", "z", "depth", "cluster"]); assert len(P5) == 60668
Rx = pd.DataFrame([json.loads(l) for l in open(f"{RES}/x77_youfree/haiku_tox/shard_000.jsonl")])
if "error" in Rx:
    Rx = Rx[Rx["error"].isna()]
Rx = Rx.drop_duplicates("id", keep="last"); Rx["flag"] = Rx.ans.isin(["A", "B"]).astype(float); Rx["cid"] = Rx.id // 10; Rx["var"] = Rx.id % 10
Pv = Rx.pivot(index="cid", columns="var", values="flag"); assert len(Pv) == 24116
Q5 = P5.set_index("id").join(Pv, how="inner"); Q5 = Q5[(Q5.z == 0) & Q5[0].notna() & Q5[1].notna()]
o5, e5 = Q5[0].values, Q5[1].values; cl5 = pd.factorize(Q5.cluster.values)[0]; nC5 = int(cl5.max() + 1); dp5 = Q5.depth.values
Wb5 = np.vstack([np.ones(nC5), np.random.default_rng(9822).poisson(1.0, size=(2000, nC5))])
RW5 = {}
for part_, mask_ in (("all", np.ones(len(o5), bool)), ("deep", dp5 >= 2)):
    n_ = Wb5 @ np.bincount(cl5, mask_.astype(float), nC5); so_ = Wb5 @ np.bincount(cl5, o5 * mask_, nC5); se_ = Wb5 @ np.bincount(cl5, e5 * mask_, nC5); d_ = (se_ - so_) / n_
    RW5[part_] = dict(n=int(mask_.sum()), orig=float(so_[0] / n_[0]), edit=float(se_[0] / n_[0]), diff=[float(d_[0]), float(np.percentile(d_[1:], 2.5)), float(np.percentile(d_[1:], 97.5))])
SRC_RW5 = ("re-analysis of results/x77_youfree/haiku_tox/shard_000.jsonl (B77d: every second-person comment of the 5% sample, original id 10i and rule-rewritten id 10i+1, labeled by Haiku "
           "without context in one shuffled batch) on the gold negatives of results/x14_pops/cc_s5.parquet; the statistic of x89_rewrite_other.py; 2,000 article-cluster resamples, seed 9822")
for part_, nm_ in (("all", "All"), ("deep", "Deep")):
    put3(f"xbCcRw{nm_}", RW5[part_]["diff"], lambda x: sg(PP * x, 1), SRC_RW5, f"change in Haiku's flag rate on gold-negative second-person comments, edited minus original{' at depth >= 2' if part_ == 'deep' else ''} (5% sample)",
         "deterministic re-analysis of stored labels")
put("xbCcRwN", comma(RW5["all"]["n"]), SRC_RW5, "gold-negative second-person comments of the 5% sample with both labels")
put("xbCcRwOrig", pct(RW5["all"]["orig"], 1), SRC_RW5, "flag rate of the originals"); put("xbCcRwEdit", pct(RW5["all"]["edit"], 1), SRC_RW5, "flag rate of the rewrites")
assert abs(np.mean(o5) - RW5["all"]["orig"]) < 1e-12

# ---- tables xb_corpora_laws and xb_corpora_you
NMC = {"cc": "Civil Comments (toxicity)", "cad": "CAD (abuse)", "qian_reddit": "Qian et al.'s Reddit (hate)", "qian_gab": "Qian et al.'s Gab (hate)", "cga": "CGA-Wiki (attack)", "bad": "BAD (unsafe turn)"}
star = lambda txt, dci: txt + ("$^{*}$" if (dci[1] > 0 or dci[2] < 0) else "")                    # labeler's value, starred when its distortion's CI excludes 0
TCL = []
for c_ in ["cc"] + [c_ for c_, *_ in OC]:
    R_ = REC[c_]; cells = [NMC[c_]]
    for key_, kd_ in (("depth", "distortion"), ("depth_replies", "depth_replies_distortion"), ("within", "within_distortion")):
        cells += [sg(R_["gold"][key_][0]), star(sg(R_["haiku"][key_][0]), R_["haiku"][kd_]), star(sg(R_["sonnet"][key_][0]), R_["sonnet"][kd_])]
    if c_ == "cc":
        dh_ = L["haiku"]["distortion"]["size"]; ds_ = L5["sonnet"]["distortion"]["size"]
        cells += [pp(G["size_slope"][0]), star(pp(L["haiku"]["law_size_slope"][0]), [dh_["est"]] + dh_["ci"]), star(pp(L5["sonnet"]["law_size_slope"][0]), [ds_["est"]] + ds_["ci"])]
    else:
        e_, d_ = SZ[c_]["est"], SZ[c_]["dr"]
        cells += [pp(e_["gold"])] + [star(pp(e_[m_]), ci_from(e_[m_] - e_["gold"], d_[m_] - d_["gold"])) for m_ in ("haiku", "sonnet")]
    if c_ == "cga":
        cells += ["--", "--", "--"]
    else:
        cells += [sg(R_["gold"]["log_rr"][0]), star(sg(R_["haiku"]["log_rr"][0]), R_["haiku"]["log_rr_distortion"]), star(sg(R_["sonnet"]["log_rr"][0]), R_["sonnet"]["log_rr_distortion"])]
    cells += [f"{M[f'xb{CTAG[c_]}WeakerN']}/{M[f'xb{CTAG[c_]}WeakerOf']}" if c_ in ESC else "--"]
    TCL.append(" & ".join(cells) + " \\\\")
table("xb_corpora_laws",
      "Laws on six corpora under gold (each corpus's own construct) and under Haiku and Sonnet asked the Civil Comments toxicity question without the parent comment: the depth law over all comments, "
      "among replies and within threads (points per level), the thread size law (points per doubling) and the log reply risk ratio. $^{*}$The labeler's law differs from gold's (95\\% CI of the "
      "paired difference excludes 0). Last column: of the instrument settings asked the corpus's own question without the parent (Haiku and the open LLMs stored for the corpus, and 14 classifier "
      "settings), how many report a weaker within-thread law than gold's, on the four corpora where gold's rises within threads. On Civil Comments, Sonnet's values are on the 5\\% sample, where gold's "
      f"laws are {M['xbCcGoldSfiveDepth']}, {M['xbCcGoldSfiveReplies']}, {M['xbCcGoldSfiveWithin']}, {M['xbGoldSfiveThreadSize']} and {M['xbCcGoldSfiveLogRR']}. CGA-Wiki's conversations end at the attack, so it has no reply risk ratio.",
      ["Corpus (gold construct)", "Depth law: gold", "Haiku", "Sonnet", "Among replies: gold", "Haiku", "Sonnet", "Within threads: gold", "Haiku", "Sonnet",
       "Thread size law: gold", "Haiku", "Sonnet", "Log reply RR: gold", "Haiku", "Sonnet", "Weaker within threads (own question)"],
      TCL, [("Depth laws (pooled, replies, within) and log RR", f"{SRC_X89} and, for Civil Comments, {SRC_X89C}: gold/<law>, instruments/<haiku|sonnet>_tox/<law>; stars from the paired distortion CIs"),
            ("Thread size law", f"Civil Comments: {SRC_PL} (gold size_slope, law_size_slope; distortion/size CI for the star); other corpora: {SRC_LAB}, gold's estimate as {SRC_DEC}"),
            ("Weaker within threads", SRC_WI + " (settings shown the parent excluded)")],
      "Every labeler column uses the toxicity question; the last column alone uses each corpus's own question. The + parent runs are not shown (their pooled laws carry the prompt switch); "
      "their among-replies distortions are the \\xb<C>HaikuCtxRepliesDist macros.")
TCY = []
for c_ in ["cc"] + [c_ for c_, *_ in OC]:
    t_ = CTAG[c_]; ck_ = "cc_threads" if c_ == "cc" else c_
    mci_ = lambda k_: f"{M[k_]} [{M[k_ + 'Lo']}, {M[k_ + 'Hi']}]"                         # arXiv revision: the own-question gaps with their 95% CIs
    own_h = mci_("xbCcToxGapHaiku") if c_ == "cc" else (mci_(f"xb{t_}OwnGapHaiku") if c_ in OWN2 else "--")
    if c_ == "cc":
        own_n, own_o = "Qwen$^{\\dagger}$", M['xbCcGapQwen']                                     # derived from stored rates: no CI
    elif c_ in OWN2:
        own_n, own_o = OWN2[c_][1], mci_(f"xb{t_}OwnGap{OWN2[c_][1]}")
    else:
        own_n, own_o = "--", "--"
    share_ = M.get(f"xb{t_}LawShare", "--")
    rwa = ci3(RW5["all"]["diff"], lambda x: sg(PP * x, 1)) + "$^{\\dagger}$" if c_ == "cc" else (ci3(RWO[c_]["all"]["diff"], lambda x: sg(PP * x, 1)) if c_ in RWO else "--")
    rwd_ = ci3(RW5["deep"]["diff"], lambda x: sg(PP * x, 1)) + "$^{\\dagger}$" if c_ == "cc" else (ci3(RWO[c_]["deep"]["diff"], lambda x: sg(PP * x, 1)) if c_ in RWO else "--")
    TCY.append(" & ".join([NMC[c_], M[f"xb{t_}ToxGapHaiku"], own_h, own_n, own_o, ci3(rr3[ck_], lambda x: f(x, 1)), f(ds3[ck_][0], 1), share_, rwa, rwd_]) + " \\\\")
table("xb_corpora_you",
      "Over-flagging of second-person address on six corpora, among gold negatives. How many points more often Haiku flags those with a second-person word than those without, asked the Civil "
      "Comments toxicity question and asked the corpus's own question (with a second LLM stored under that question, at gold's flag rate except Sonnet); Haiku's false-positive rate with three or "
      "more second-person words over that with none, among the gold negatives that Detoxify scores below the median of the corpus's gold negatives (95\\% CI) and among all; the share of Haiku's depth-law distortion that "
      "disappears at a fixed number of second-person words (where the distortion is positive); and the change in Haiku's flag rate, in points, when the same comments are rewritten to the third "
      "person, overall and at depth 2 or more (95\\% CI). Except for the own-question columns, Haiku answers the toxicity question without the parent. $^{\\dagger}$5\\% sample.",
      ["Corpus (own construct)", "Extra flags on \"you\": toxicity question", "own question, Haiku [95% CI]", "second LLM", "own question, second LLM [95% CI]", "3+ : 0, Detoxify below median [95% CI]", "3+ : 0, all",
       "Share of the distortion carried by \"you\"", "Rewrite: all [95% CI]", "Rewrite: depth >= 2 [95% CI]"],
      TCY, [("Extra flags, toxicity question", f"{SRC_X89} / {SRC_X89C}: instruments/haiku_tox/fpr_you_minus_other x 100"),
            ("Own question", f"{SRC_XE}: <corpus>/<labeler>/gap x 100; Civil Comments' own question is the toxicity question, and its second labeler is Qwen at gold's flag rate on the 5% sample ({SRC_DO}, derived from fpr_by_count and n_by_count; no CI)"),
            ("3+ : 0 ratios", f"{SRC_XC}: below_median/haiku/ratio3 and all/haiku/ratio3 (the latter equals x90's dose ratio, asserted)"),
            ("Share carried by you", f"{SRC_DO}: 1 - law_distortion_within_count / law_distortion; given only where the distortion's CI is above 0"),
            ("Rewrite", f"{SRC_RWO}: <corpus>/all|deep/diff; Civil Comments: {SRC_RW5}")],
      "BAD's own-question labels are stored only with the previous turn shown (pooled), so its own-question gaps are blank; the rewrite was not run on BAD. On BAD Haiku does not over-flag "
      f"second-person turns (toxicity-question gap {M['xbBadToxGapHaiku']} [{M['xbBadToxGapHaikuLo']}, {M['xbBadToxGapHaikuHi']}] points); among turns Detoxify scores below the median the "
      f"3+ : 0 ratio's CI covers 1 and the gap is below 0 ({M['xbBadCtrlGap']} [{M['xbBadCtrlGapLo']}, {M['xbBadCtrlGapHi']}] points); its 3+ : 0 ratios rest on few turns "
      f"(of all gold-negative turns, {comma(DO['bad']['labelers']['haiku']['n_by_count'][3])} and {comma(DO['bad']['labelers']['haiku']['n_by_count'][4])} have 3 and 4+ second-person words).")
_xr = XC["bad"]["below_median"]["haiku"]["ratio3"]; _xg = REC["bad"]["haiku"]["fpr_you_minus_other"]
assert _xr[1] < 1 < _xr[2] and gp3["bad"][2] < 0 and _xg[1] < 0 < _xg[2]                          # the BAD caveat as worded
assert all(DO[c_]["labelers"]["haiku"]["law_distortion"][1] > 0 for c_ in ("cc_threads", "qian_reddit", "qian_gab"))
assert all(DO[c_]["labelers"]["haiku"]["law_distortion"][1] <= 0 for c_ in ("cad", "cga", "bad"))   # the share is given exactly where the distortion's CI is above 0

# ================================================================ E. A check of v7f's '+ parent' contagion range, and counts
sec("E. Check of the '+ parent' reply RR among replies; counts")
PC2 = pd.read_parquet(f"{RES}/x14_pops/cc_threads.parquet", columns=["z", "depth", "parent_row", "y:haiku", "y:haiku_ctx"])
dpar_ = PC2.depth.values[pas]; mrep_ = dpar_ >= 1                                                 # edges whose parent is itself a reply: both ends labeled with their parent


def rr_m(v, m):
    vp, vc = v[pas][m], v[chs][m]; return float(vc[vp == 1].mean() / vc[vp == 0].mean())


SRC_CTX = "re-analysis of results/x14_pops/cc_threads.parquet (z, y:haiku, y:haiku_ctx; no resampling)"
_cr = [float(EX[k][0]) for k in ("qCtxRRMin", "qCtxRRMax", "cCtxRROpenMin", "cCtxRROpenMax")]
CTXR_ = f"{EX['cCtxRROpenMin'][0]}-{EX['qCtxRRMax'][0]}"; assert min(_cr) == _cr[2] and max(_cr) == _cr[1]     # v7f's range: \cCtxRROpenMin to \qCtxRRMax
put("xbCtxRepliesEdges", comma(int(mrep_.sum())), SRC_CTX, "reply edges whose parent has depth >= 1 (both ends are replies, so a labeler shown the parent labels both with their parent)")
put("xbCtxRepliesRRGold", f2(rr_m(PC2.z.values.astype(float), mrep_)), SRC_CTX, "gold's reply RR over those edges")
put("xbCtxRepliesRRHaiku", f2(rr_m(np.nan_to_num(PC2["y:haiku"].values.astype(float)), mrep_)), SRC_CTX, "Haiku's (without the parent) reply RR over those edges")
put("xbCtxRepliesRRHaikuCtx", f2(rr_m(np.nan_to_num(PC2["y:haiku_ctx"].values.astype(float)), mrep_)), SRC_CTX, "Haiku shown the parent: reply RR over those edges",
    f"a check of v7f's '+ parent' contagion range ({CTXR_}), which pools edges with top-level parents labeled without the parent")
assert abs(rr_m(np.nan_to_num(PC2["y:haiku_ctx"].values.astype(float)), np.ones(len(chs), bool)) - L["haiku_ctx"]["law_rr"][0]) < 1e-9   # pooled value = x74_placebo's
put("xbNClsSettings", str(len(CLS14)), "this script (CLS14)", "the classifier settings over which every range above is taken")
for c_, *_ in OC:                                                                                   # E52's wording: every Haiku size distortion excludes 0
    dz_ = ci_from(SZ[c_]["est"]["haiku"] - SZ[c_]["est"]["gold"], SZ[c_]["dr"]["haiku"] - SZ[c_]["dr"]["gold"]); assert dz_[1] > 0 or dz_[2] < 0, c_
A96_, T96_ = F96["articles"]["Z"], F96["threads"]
assert len(T96_["Z"]["bin_means"]) == len(T96_["haiku"]["bin_means"]) == len(T96_["haiku_ctx"]["bin_means"]) == int(M["xbMKLogBinsThread"])
mx_k = max(g14, key=lambda k: g14[k][0]); mx50 = IMR[mx_k]["50"]["rr_parent_gold"][0]
gmin50 = min(IMR[k]["50"]["rr_parent_gold"][0] for k in CLS14); gmax50 = max(IMR[k]["50"]["rr_parent_gold"][0] for k in CLS14)
INV = [
    "E05 (inventory): 'The f96 version over complete sections gives +0.27, tau 1.00, shuffle z = 10' comes from f96's article part "
    f"({sg(PP * A96_['slope'])}, tau {f2(A96_['mk_tau'])}, z {zf(A96_['z_vs_shuffle'])}), which uses the old train-split sections ({F96['articles']['n_comments']:,} comments); "
    f"not used. On the current population x74_placebo gives gold's article law {M['xbGoldArtSize']} with Mann-Kendall tau {M['xbGoldArtTauLog']} (log2 bins) and shuffle z {M['xbShufZArtGold']}.",
    f"E06 (inventory): 'On f96's 8 log2 bins' — at the thread level only {M['xbMKLogBinsThread']} log2 bins hold >= 30 threads (the article level has {M['xbMKLogBinsArt']}); the p-values quoted "
    f"(1.00, 0.24, 0.03) are f96's over those {M['xbMKLogBinsThread']} bins ({pv(T96_['Z']['mk_p'])}, {pv(T96_['haiku']['mk_p'])}, {pv(T96_['haiku_ctx']['mk_p'])} for gold, Haiku and "
    "Haiku shown the parent, the last pooled).",
    f"E06 (inventory and plan): gold's thread shuffle z = 5.9 is f96's draw (\\xbShufZGoldRep = {M['xbShufZGoldRep']}); x74_placebo's draw on the same {F96['threads']['n_conversations']:,} threads gives "
    f"{M['xbShufZGold']} (\\xbShufZGold). Both exceed 2; the tables use x74_placebo for every labeler.",
    f"E35 (inventory): 'All 12 classifier settings exceed their matched human instrument' — with Detoxify-unbiased (stored) all {M['xbMatchUpN']} do (\\xbMatchUpN).",
    "E35 (inventory): the size-law and unit-share comparisons for Haiku and Qwen shown the parent (quoted there as +3.65 against +0.89; +1.95 against +0.15; 0.57 against 0.62) are pooled "
    "'+ parent' values and are not used.",
    f"E39 (inventory): parent-gold RRs '0.92-1.04' — at n* = 20 the {len(CLS14)} settings give {M['xbMrGoldClsMin']}-{M['xbMrGoldClsMax']} (the maximum, {mx_k}, is {g14[mx_k][0]:.5f}); "
    f"at n* = 50 they give {f2(gmin50)}-{f2(gmax50)} ({mx_k}: {mx50:.5f}).",
    f"E39 (inventory) and \\cMrCtxQwen (numbers_v5.tex): 'Qwen shown the parent 0.81' uses Qwen at the abandoned no-context threshold ({pct(QCTX_RATE, 1)} flagged); at its own threshold, "
    f"the paper's convention, it is {M['xbMrGoldQwenCtx']}.",
    f"E40 (inventory): '2 of 12 settings are above it, 2 below and 8 indistinguishable' — over the {len(CLS14)} stored settings {M['xbLatAboveN']} / {M['xbLatBelowN']} / {M['xbLatSameN']}.",
    "E40 (inventory): the fit check 'the fitted model misses gold's log risk ratio by -0.177 at the observed rater counts' comes from results/x82/x82_latent.json, which the keep-out list (K5) supersedes; "
    f"only the adaptive-protocol check (x82_protocol, B82e) is used, and it starts from the EM fit (RR_inf {M['xbLatProtoRR']}), not the converged one ({M['xbLatRR']}).",
    "E41 (inventory note on \"monotone_f\": false): the flag is wrong and the solver field is right. x83d_mono.run records (em_f is em_f_mono); x85_converged.py passes its own solver em_f_sq, which "
    f"parametrizes f as 1e-7 plus the cumulative sum of non-negative increments (SLSQP), so f is monotone by construction; all {M['xbConsMonoN']} stored f vectors are non-decreasing (asserted), "
    "and Addendum 85 describes the same solver.",
    "E41 (inventory): Detoxify-unbiased's construct-only excess at gold's rate, quoted as '+0.09 [+0.01, +0.16] at m = 20, not significant at m = 10 and 40', is "
    f"{M['xbConsUnbExc']} [{M['xbConsUnbExcLo']}, {M['xbConsUnbExcHi']}] at m = 20 and has a CI above 0 on {M['xbConsUnbSigGrids']} of the {len(CG)} grids (m = 20 only); so "
    f"{M['xbConsAboveTwenty']} of the {len(CLS14)} settings have an excess above 0 at m = 20, {M['xbConsAboveTen']} at m = 10 and {M['xbConsAboveForty']} at m = 40.",
    f"E42 (inventory and Addendum 87): classifier excesses '+0.32 to +0.76' — the smallest, {min(e14, key=e14.get)}'s, is {min(e14.values()):.4f}, which prints as {M['xbWnClsExcMin']} "
    f"(the largest is {M['xbWnClsExcMax']}); and with Detoxify-unbiased {M['xbWnClsAboveN']} of {len(CLS14)} settings (not {EX['qWnAbove'][0]} of {len(CLS12)}) are above gold's.",
    f"E43 (inventory): identity mentions 'raise the classifiers' FPR 1.5-3.6-fold' — over the {len(CLS14)} settings {M['xbIdFprMin']}-{M['xbIdFprMax']}: Detoxify-unbiased has no identity gap "
    f"({f2(fr14['detoxify_unbiased@0.6'])} at 0.6 and {f2(fr14['detoxify_unbiased@m'])} at gold's rate).",
    "E43 (inventory): the article-stratified Mantel-Haenszel placebo RRs it quotes (1.11-1.43 against 0.95) are the raw within-article statistic that Addendum 87 found biased toward 0; not used "
    "(E42 replaces them).",
    f"E33 (inventory): 'On the difference scale, the association falls with context for humans and LLMs alike' — Haiku's change, {M['xbCccHaikuDiffChg']} points [{M['xbCccHaikuDiffChgLo']}, "
    f"{M['xbCccHaikuDiffChgHi']}], has a CI covering 0.",
    f"E50 (inventory): '67/67 settings weaker' ({M['xbWeakerAllN']}/{M['xbWeakerAllOf']} here) includes {tot_all - tot_n} settings shown the parent (pooled); without them "
    f"{M['xbWeakerN']}/{M['xbWeakerOf']}, with {M['xbWeakerRevN']} sign reversals; the uniform-error share of the shortfall has median {M['xbAttShareMed']} (earlier draft: {EX['qAttShareMed'][0]} over {tot_all}).",
    "E52 (inventory): 'Haiku shown the parent on BAD: +4.38' is a pooled '+ parent' size law; not used. B89a's toxicity question had no size laws; computed here from results/x89/labels: "
    f"Haiku {M['xbCadHaikuSize']} (CAD; gold {M['xbCadGoldSize']}), {M['xbQrHaikuSize']} (Reddit; {M['xbQrGoldSize']}), {M['xbGabHaikuSize']} (Gab; {M['xbGabGoldSize']}), "
    f"{M['xbCgaHaikuSize']} [{M['xbCgaHaikuSizeLo']}, {M['xbCgaHaikuSizeHi']}] (CGA-Wiki; gold {M['xbCgaGoldSize']} [{M['xbCgaGoldSizeLo']}, {M['xbCgaGoldSizeHi']}]: opposite sign in the point "
    f"estimate, but Haiku's CI covers 0) and {M['xbBadHaikuSize']} [{M['xbBadHaikuSizeLo']}, {M['xbBadHaikuSizeHi']}] (BAD; gold {M['xbBadGoldSize']}); every distortion's paired CI excludes 0.",
    f"E53 (inventory): 'On Reddit, Haiku inflates a weak association, from 1.14 to 1.50' is under the hate question, and the stored RR is {F122['platforms']['reddit']['audit']['rr_y0'][0]:.5f}, "
    f"which prints as {M['xbQrHateRRHaiku']} (1.50 rounds an already rounded {F122['platforms']['reddit']['audit']['rr_y0'][0]:.3f}); under the toxicity question Haiku's log-RR distortion on Reddit "
    f"is {M['xbQrHaikuLrrDist']} [{M['xbQrHaikuLrrDistLo']}, {M['xbQrHaikuLrrDistHi']}] (lower end {REC['qian_reddit']['haiku']['log_rr_distortion'][1]:.4f}).",
    f"v7f (Section 3): 'shown the parent, ... their risk ratios fall to {CTXR_}' rests on reply risk ratios pooled over all reply edges, whose parent label is the no-context label for top-level parents and "
    f"the with-context label for replies (\\qCtxRRMin/Max, \\cCtxRROpenMin/Max). Checked here for Haiku over the {M['xbCtxRepliesEdges']} edges whose parent is a reply: shown the parent "
    f"{M['xbCtxRepliesRRHaikuCtx']}, without it {M['xbCtxRepliesRRHaiku']}, gold {M['xbCtxRepliesRRGold']}; so the fall holds among replies, but the printed range is pooled and should be "
    "restated among replies (the open models' and Sonnet's among-replies values are not computed here).",
]

PLN = {"cad": "CAD", "qian_reddit": "Qian Reddit", "qian_gab": "Qian Gab", "cga": "CGA-Wiki", "bad": "BAD"}
andj = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
FLAT_ = [c_ for c_ in ESC if REC[c_]["haiku"]["within_distortion"][2] < 0]                         # Haiku's within-thread distortion significantly below 0
assert all(REC[c_]["gold"]["within"][1] > 0 for c_ in ESC)                                          # gold rises within threads on these four
assert IA["haiku"]["lrr_minus_human"][1] > 0 and IS["sonnet"]["lrr_minus_human"][1] > 0             # QUAL: more contagion than the matched human instrument
for c_, sgn_ in (("qian_gab", 1), ("qian_reddit", 1), ("cad", -1), ("bad", -1)):                   # QUAL: inflated on Gab and Reddit, weakened on CAD and BAD (CIs exclude 0)
    for m_ in ("haiku", "sonnet"):
        d_ = REC[c_][m_]["log_rr_distortion"]; assert (d_[1] > 0) if sgn_ > 0 else (d_[2] < 0), (c_, m_, d_)
e_ = {c_: SZ[c_]["est"] for c_, *_ in OC}; d0_ = {c_: ci_from(SZ[c_]["est"]["haiku"], SZ[c_]["dr"]["haiku"]) for c_, *_ in OC}
dg_ = {c_: ci_from(SZ[c_]["est"]["gold"], SZ[c_]["dr"]["gold"]) for c_, *_ in OC}
assert all(e_[c_]["gold"] * e_[c_]["haiku"] > 0 and abs(e_[c_]["haiku"]) < abs(e_[c_]["gold"]) for c_ in ("qian_reddit", "qian_gab"))       # attenuated
assert all(d0_[c_][1] < 0 < d0_[c_][2] and (dg_[c_][1] > 0 or dg_[c_][2] < 0) for c_ in ("bad", "cga"))                                   # erased
assert e_["cad"]["haiku"] > e_["cad"]["gold"] > 0                                                                                            # inflated
QUAL = [
    f"'Avalle et al.'s binned trend test misses gold's law (p = 0.20) but detects Haiku's (p = 0.017)': on log2 bins Haiku's thread law is not detected either (p = {M['xbMKLogHaiku']}), "
    f"and Avalle et al.'s shuffle test flags gold's tiny thread law ({M['xbGoldThreadSize']}) as a significant rise (z = {M['xbShufZGold']}; f96's draw {M['xbShufZGoldRep']}), as it flags "
    f"{M['xbClsShufSigN']} of the {len(CLS14)} classifier settings. "
    f"At the article level, gold's law is real ({M['xbGoldArtSize']} [{M['xbGoldArtSizeLo']}, {M['xbGoldArtSizeHi']}]); Haiku inflates it {M['xbHaikuArtX']}-fold, and Avalle et al.'s two robustness "
    f"instruments give {M['xbDetoxSixArtX']} and {M['xbImsArgArtX']} times gold's.",
    f"'one rater, or the majority of three, gives a risk ratio of 1.63 or 2.10, against all raters' 2.39': under the latent model, noise-free gold's RR is {M['xbLatRR']} [{M['xbLatRRLo']}, {M['xbLatRRHi']}] "
    f"(log gap {M['xbLatGap']} [{M['xbLatGapLo']}, {M['xbLatGapHi']}]); against it only {M['xbLatAboveN']} of the {len(CLS14)} classifier settings report more contagion and {M['xbLatSameN']} are indistinguishable "
    "(model-based; the model misses the vote patterns given the number of raters).",
    f"'Their placebo risk ratios are 1.75-4.01, against 1.43 for a single rater': a noise-free instrument of the raters' construct would show {M['xbConsThrPlac']} [{M['xbConsThrPlacLo']}, {M['xbConsThrPlacHi']}], "
    f"so the comparison alone does not show clustering beyond the construct; the construct-only test (excess {M['xbConsExcMin']} to {M['xbConsExcMax']}, {M['xbConsAboveTwelveTwenty']} of "
    f"{len(CLS12)} settings above 0; Detoxify-unbiased {M['xbConsUnbExc']} [{M['xbConsUnbExcLo']}, {M['xbConsUnbExcHi']}] at gold's rate and {M['xbConsUnbSixExc']} "
    f"[{M['xbConsUnbSixExcLo']}, {M['xbConsUnbSixExcHi']}] at 0.6) and the many-rater check ({M['xbMrClsMinTwenty']}-{M['xbMrClsMaxTwenty']} at n* = 20, "
    f"{M['xbMrClsAboveTwenty']} of {len(CLS14)} above 1; one rater {M['xbMrSrTwenty']}) do.",
    f"'12 of the 14 settings report more contagion than gold': classifiers flagging as many comments as gold look like a stricter gold (against gold at 0.6: {M['xbSevSixMin']} to {M['xbSevSixMax']}); "
    f"matched to gold_t's own flag rate their excess persists (median {M['xbThrMedMin']} to {M['xbThrMedMax']}, {M['xbThrNMin']}-{M['xbThrNMax']} of {len(CLS7)} CIs above 0); within articles, "
    f"{M['xbWnClsAboveN']} of {len(CLS14)} exceed gold's excess and {M['xbWnClsBelowN']} fall below; {M['xbWnBetween']} of gold's own contagion lies between articles.",
    f"'the Claude models at their own cut weaken it': against a human instrument flagging as many comments, Haiku and Sonnet report more contagion ({M['xbHaikuRR']} against {M['xbMatchHaikuHum']}; "
    f"{M['xbSonnetRR']} against {M['xbMatchSonnetHum']}; log differences {M['xbMatchHaikuDiff']} [{M['xbMatchHaikuDiffLo']}, {M['xbMatchHaikuDiffHi']}] and {M['xbMatchSonnetDiff']} "
    f"[{M['xbMatchSonnetDiffLo']}, {M['xbMatchSonnetDiffHi']}]); and outside Civil Comments, under the same question, they inflate it on Qian et al.'s Gab ({M['xbGabHaikuLrrDist']} "
    f"[{M['xbGabHaikuLrrDistLo']}, {M['xbGabHaikuLrrDistHi']}] and {M['xbGabSonnetLrrDist']} [{M['xbGabSonnetLrrDistLo']}, {M['xbGabSonnetLrrDistHi']}] on the log scale) and Reddit "
    f"({M['xbQrHaikuLrrDist']} [{M['xbQrHaikuLrrDistLo']}, {M['xbQrHaikuLrrDistHi']}], at the margin; {M['xbQrSonnetLrrDist']} [{M['xbQrSonnetLrrDistLo']}, {M['xbQrSonnetLrrDistHi']}]), "
    f"while weakening it on CAD ({M['xbCadHaikuLrrDist']}, {M['xbCadSonnetLrrDist']}) and BAD ({M['xbBadHaikuLrrDist']}, {M['xbBadSonnetLrrDist']}).",
    f"'The bias travels': within threads, where gold's rate rises with depth ({andj([PLN[c_] for c_ in ESC])}), every instrument setting without the parent reports a weaker rise "
    f"({M['xbWeakerN']} of {M['xbWeakerOf']}, own questions; {M['xbWeakerRevN']} reverse its sign), and Haiku under the toxicity question flattens it significantly on "
    + andj([f"{PLN[c_]} ({M['xb' + CTAG[c_] + 'HaikuWithinDist']})" for c_ in FLAT_]) + f" (on {andj([PLN[c_] for c_ in ESC if c_ not in FLAT_])} the CI covers 0); "
    f"the pooled false rise on Reddit and Gab sits among replies ({M['xbQrHaikuRepliesDist']}, {M['xbGabHaikuRepliesDist']}).",
    f"Size laws outside Civil Comments: under the toxicity question Haiku attenuates gold's strong size laws on Reddit and Gab ({M['xbQrHaikuSize']} against {M['xbQrGoldSize']}; {M['xbGabHaikuSize']} against "
    f"{M['xbGabGoldSize']}), erases BAD's ({M['xbBadHaikuSize']} [{M['xbBadHaikuSizeLo']}, {M['xbBadHaikuSizeHi']}] against {M['xbBadGoldSize']}) and CGA-Wiki's ({M['xbCgaHaikuSize']} "
    f"[{M['xbCgaHaikuSizeLo']}, {M['xbCgaHaikuSizeHi']}] against {M['xbCgaGoldSize']} [{M['xbCgaGoldSizeLo']}, {M['xbCgaGoldSizeHi']}]) and inflates CAD's "
    f"({M['xbCadHaikuSize']} against {M['xbCadGoldSize']}): the inflated size law is specific to Civil Comments and CAD.",
    f"Context and contagion (CCC): with the parent shown, human raters' reply RR rises ({M['xbCccHumRROut']} to {M['xbCccHumRRIn']}) while the LLMs' falls (Haiku {M['xbCccHaikuRROut']} to {M['xbCccHaikuRRIn']}); "
    f"on the difference scale it falls for both (humans {M['xbCccHumDiffChg']} points; Haiku {M['xbCccHaikuDiffChg']}, CI covering 0), so the comparison depends on the scale.",
]

# ================================================================ R. arXiv revision, review round 1 (numbers the revised text needs)
sec("R. arXiv revision: review-round numbers")
lc14 = {k: L[k]["distortion"]["log_rr"] for k in CLS14}
put("xbClsLogUpN", str(sum(v["ci"][0] > 0 for v in lc14.values())), SRC_PL, "number of the 14 settings whose log-RR distortion (distortion/log_rr, paired CI) lies above 0")
put("xbClsLogDownN", str(sum(v["ci"][1] < 0 for v in lc14.values())), SRC_PL, "number of the 14 settings whose log-RR distortion lies below 0 (" + ", ".join(k for k in CLS14 if lc14[k]["ci"][1] < 0) + ")")
assert M["xbClsLogUpN"] == M["xbClsRRUpN"] and M["xbClsLogDownN"] == M["xbClsRRDownN"]       # the ratio-scale and log-scale counts agree
# review round 2: counts written as words where the text needs them (sentence-initial, or below ten)
WORDS_ = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"]
put("xbClsRRUpNWord", WORDS_[int(M["xbClsRRUpN"])].capitalize(), SRC_PL, "xbClsRRUpN as a capitalized word (it opens a sentence)")
put("xbLatAboveNWord", WORDS_[int(M["xbLatAboveN"])], SRC_RI, "xbLatAboveN as a word")
put("xbWeakerRevNWord", WORDS_[int(M["xbWeakerRevN"])], PROV["xbWeakerRevN"]["src"], "xbWeakerRevN as a word")
for k_ in ("xbLatBelowN", "xbLatSameN", "xbWnClsAboveN", "xbWnClsBelowN", "xbClsCcUpN"):
    put(k_ + "Word", WORDS_[int(M[k_])], PROV[k_]["src"], f"{k_} as a word")
put("xbClsCcLogUpN", str(sum(lc14[k]["ci"][0] > 0 for k in CCS)), SRC_PL, "of the settings trained in part on Civil Comments labels, those whose log-RR distortion lies above 0")
assert M["xbClsCcLogUpN"] == M["xbClsCcUpN"]
put("xbMrClsLoMinTwenty", f"{min(IMR[k]['20']['rr_parent_flag'][1] for k in CLS14):.3f}", SRC_MR, "smallest lower CI end of instruments/<k>/20/rr_parent_flag over the 14 settings (three decimals)")
assert float(M["xbMrClsLoMinTwenty"]) > 1
put("xbConsPredFourteenMin", f2(min(pr_.values())), SRC_CG, "min over the 14 settings at m = 20 of the construct-only placebo RR (Detoxify-unbiased included)", CONSCAV)
put("xbConsExcFourteenMin", sg(min(pe20[k][0] for k in CLS14)), SRC_CG, "min over the 14 settings of the placebo log-RR excess at m = 20 (" + min(CLS14, key=lambda k: pe20[k][0]) + ")", CONSCAV)
put("xbConsExcFourteenMax", sg(max(pe20[k][0] for k in CLS14)), SRC_CG, "max over the 14 settings of the placebo log-RR excess at m = 20", CONSCAV)
assert M["xbConsExcFourteenMin"] == M["xbConsUnbSixExc"]
for k_, nm_ in (("detoxify_original@0.6", "DetoxSix"), ("detoxify_original@m", "DetoxM"), ("imsypp@argmax", "ImsArg"), ("imsypp@m", "ImsM")):
    r_ = L[k_]
    put(f"xb{nm_}Flag", pct(r_["flag_rate"][0], 1), SRC_PL, f"cc_threads/labelers/{k_}/flag_rate", "share of comments flagged")
    put3(f"xb{nm_}Art", r_["law_article_size_slope"], pp, SRC_PL, f"cc_threads/labelers/{k_}/law_article_size_slope x 100 (points per doubling)")
    put3(f"xb{nm_}Depth", r_["law_depth_slope"], pp, SRC_PL, f"cc_threads/labelers/{k_}/law_depth_slope x 100 (points per level)")
    put3(f"xb{nm_}DepthDist", [r_["distortion"]["depth"]["est"]] + r_["distortion"]["depth"]["ci"], pp, SRC_PL, f"cc_threads/labelers/{k_}/distortion/depth x 100")
    put3(f"xb{nm_}Plac", r_["placebo_depth_slope"], pp, SRC_PL, f"cc_threads/labelers/{k_}/placebo_depth_slope x 100 (points per level)")
    put(f"xb{nm_}RR", f2(r_["law_rr"][0]), SRC_PL, f"cc_threads/labelers/{k_}/law_rr")
hA = L["haiku@A"]
put("xbStrictRR", f2(hA["law_rr"][0]), SRC_PL, "cc_threads/labelers/haiku@A/law_rr: Haiku flagging only 'very toxic'")
put3("xbStrictLrrDist", [hA["distortion"]["log_rr"]["est"]] + hA["distortion"]["log_rr"]["ci"], sg, SRC_PL, "cc_threads/labelers/haiku@A/distortion/log_rr")
# Haiku asked each corpus's own question (x74_placebo's labelers on the other corpora are the corpus-question labels)
for c_, t_ in (("qian_reddit", "Qr"), ("qian_gab", "Gab"), ("cad_threads", "Cad"), ("cga", "Cga")):
    r_ = PLJ[c_]["labelers"]["haiku"]
    put3(f"xb{t_}OwnPlac", r_["placebo_depth_slope"], pp, SRC_PL, f"{c_}/labelers/haiku/placebo_depth_slope x 100: Haiku asked the corpus's own question, without the parent")
    put3(f"xb{t_}OwnDepth", r_["law_depth_slope"], pp, SRC_PL, f"{c_}/labelers/haiku/law_depth_slope x 100 (own question)")
    put(f"xb{t_}OwnFlag", pct(r_["flag_rate"][0], 1), SRC_PL, f"{c_}/labelers/haiku/flag_rate (own question)")
    put(f"xb{t_}ToxFlag", pct(X89[{"cad_threads": "cad"}.get(c_, c_)]["instruments"]["haiku_tox"]["flag_rate"][0], 1), SRC_X89, f"{c_}/instruments/haiku_tox/flag_rate[0] (toxicity question)")
    if c_ != "cga":
        put(f"xb{t_}OwnRR", f2(r_["law_rr"][0]), SRC_PL, f"{c_}/labelers/haiku/law_rr (own question)")
        put3(f"xb{t_}OwnLrrDist", [r_["distortion"]["log_rr"]["est"]] + r_["distortion"]["log_rr"]["ci"], sg, SRC_PL, f"{c_}/labelers/haiku/distortion/log_rr (own question; log RR minus gold's, paired CI)")
rb_ = PLJ["bad"]["labelers"]["haiku_ctx"]                                                          # BAD's own-question labels exist only with the previous turn shown
put("xbBadOwnCtxRR", f2(rb_["law_rr"][0]), SRC_PL, "bad/labelers/haiku_ctx/law_rr (own question, previous turn shown)")
put3("xbBadOwnCtxLrrDist", [rb_["distortion"]["log_rr"]["est"]] + rb_["distortion"]["log_rr"]["ci"], sg, SRC_PL, "bad/labelers/haiku_ctx/distortion/log_rr")
put3("xbBadOwnCtxPlac", rb_["placebo_depth_slope"], pp, SRC_PL, "bad/labelers/haiku_ctx/placebo_depth_slope x 100")

# ---- arXiv revision: Appendix D's table of all 14 classifier settings, with the reply risk ratio's distortion on the log scale
#      (paper_v5/tables/classifiers_v7.tex, the earlier-draft table, gives RR minus gold's RR; every other cell is asserted equal to it)
CLSV = [("Detoxify", "detoxify_original", "0.6"), ("Detoxify-unbiased$^{\\S}$", "detoxify_unbiased", "0.6"), ("IMSYPP", "imsypp", "argmax"),
        ("s-nlp RoBERTa$^{\\S}$", "sn_rob", "0.5"), ("martin-ha DistilBERT$^{\\S}$", "mh_dist", "0.5"), ("Dynabench hate", "fb_hate", "0.5"), ("CardiffNLP offensive", "cn_off", "0.5")]
OLD7 = [l_.rstrip() for l_ in open(f"{W}/results/reference/tables/classifiers_v7.tex") if "&" in l_]
TK, i7 = [], 0
for nm_, k_, th_ in CLSV:
    for j_, (lab_, key_) in enumerate(((th_, f"{k_}@{th_}"), ("gold's rate", f"{k_}@m"))):
        r_ = L[key_]
        cells = [nm_ if j_ == 0 else "", lab_, f"{r_['kappa'][0]:.2f}", f"{r_['fpr'][0]:.3f}", sg(PP * r_["placebo_depth_slope"][0]), f"{r_['placebo_rr'][0]:.2f}",
                 sg(PP * r_["law_depth_slope"][0]), sg(PP * r_["law_size_slope"][0]), f"{r_['law_rr'][0]:.2f}", lrd(r_)]
        old_ = [c.strip() for c in OLD7[i7].rstrip("\\").split("&")]; i7 += 1
        assert [c.strip() for c in cells[:9]] == old_[:9], (cells, old_)
        TK.append(" & ".join(cells) + " \\\\")
    TK.append("\\addlinespace[1pt]")
TK = TK[:-1]
table("xb_classifiers", "All 14 classifier settings on Civil Comments reply threads: kappa, FPR, placebo depth slope and risk ratio, depth law, thread size law, reply risk ratio and its distortion on the log scale (95% CI).",
      ["Classifier", "Threshold", "kappa", "FPR", "Placebo depth slope", "Placebo RR", "Depth law", "Thread size law", "Reply RR", "Distortion (log RR)"], TK,
      [("All cells but the last", "results/x74_placebo/x74_placebo.json cc_threads/labelers (asserted equal to paper_v5/tables/classifiers_v7.tex)"), ("Last column", f"{SRC_PL}: distortion/log_rr")],
      "The earlier-draft table's last column is RR minus gold's RR; this one is the log distortion, as every other table.")
# ---- arXiv revision: the latent-model checks quoted in Appendix C (the same values as tables/xb_latent_checks.tex)
for nm_, v_, how_ in (("ConsTwo", SDB["instruments"]["h0"]["placebo_lrr"]["excess"], "two-stage stopping: instrument of the construct, placebo log-RR excess"),
                      ("ConsCat", SDC["instruments"]["h0"]["placebo_lrr"]["excess"], "categorical answers: instrument of the construct, placebo log-RR excess"),
                      ("DriftCat", SDC["instruments"]["depthy"]["placebo_lrr"]["excess"], "categorical answers: instrument drifting with depth, placebo log-RR excess (no clustering built in)"),
                      ("ConsSfive", [PP * t for t in SDB["instruments_m40"]["sonnet"]["placebo_slope"]["excess"]], "simulated 5% sample, instrument of the construct, placebo slope excess at m = 40 (points per level)")):
    put3(f"xbLatChk{nm_}", v_, sg, "results/x85 latent-model simulations (as tables/xb_latent_checks.tex)", how_)
# ---- arXiv revision: flag rates among gold negatives under both questions; Qwen's placebo slope; own-question size laws and contagion (Appendix E)
for c_, t_ in (("qian_reddit", "Qr"), ("qian_gab", "Gab"), ("cad_threads", "Cad"), ("cga", "Cga")):
    xk_ = {"cad_threads": "cad"}.get(c_, c_)
    fr_ = X89[xk_]["instruments"]["haiku_tox"]["fpr"]; fr_ = fr_[0] if isinstance(fr_, list) else fr_
    put(f"xb{t_}ToxFPR", pct(fr_, 1), SRC_X89, f"{xk_}/instruments/haiku_tox/fpr: share of gold negatives Haiku flags under the toxicity question")
    put(f"xb{t_}OwnFPR", pct(PLJ[c_]["labelers"]["haiku"]["fpr"][0], 1), SRC_PL, f"{c_}/labelers/haiku/fpr: the same under the corpus's own question")
put3("xbQwenPlac", L["qwen7b_m"]["placebo_depth_slope"], pp, SRC_PL, "cc_threads/labelers/qwen7b_m/placebo_depth_slope x 100: Qwen at gold's flag rate, all reply threads")
TOL = []
for c_, nm_ in (("cad_threads", "CAD (abuse)"), ("qian_reddit", "Qian et al.'s Reddit (hate)"), ("qian_gab", "Qian et al.'s Gab (hate)"), ("cga", "CGA-Wiki (attack)"), ("bad", "BAD (unsafe turn)")):
    G_ = PLJ[c_]["gold"]; lab_ = "haiku" if "haiku" in PLJ[c_]["labelers"] else "haiku_ctx"; r_ = PLJ[c_]["labelers"][lab_]
    mk_ = "" if lab_ == "haiku" else "$^{\\|}$"
    dz_ = r_["distortion"]["size"]; sz_ = sg(PP * r_["law_size_slope"][0]) + ("$^{*}$" if (dz_["ci"][0] > 0 or dz_["ci"][1] < 0) else "")
    if c_ != "cga":
        dl_ = r_["distortion"]["log_rr"]; lr_ = sg(math.log(r_["law_rr"][0])) + ("$^{*}$" if (dl_["ci"][0] > 0 or dl_["ci"][1] < 0) else "")
        glr_ = sg(math.log(G_["rr"][0]))
    else:
        lr_ = glr_ = "--"
    TOL.append(" & ".join([nm_ + mk_, sg(PP * G_["size_slope"][0]), sz_, glr_, lr_]) + " \\\\")
table("xb_corpora_ownlaws", "The thread size law (points per doubling) and the log reply risk ratio under gold and under Haiku asked each corpus's own question without the parent "
      "(BAD: with the previous turn shown); star: the paired 95% CI of the distortion excludes 0.", ["Corpus", "Gold size law", "Haiku size law", "Gold log RR", "Haiku log RR"], TOL,
      [("All cells", f"{SRC_PL}: <corpus>/gold/(size_slope, rr) and labelers/haiku (BAD: haiku_ctx)/(law_size_slope, law_rr, distortion)")])

# ---- arXiv revision: the cross-corpus table read on each corpus's own construct (main text, Section 6)
OWNP = {"cc": "cc_threads", "cad": "cad_threads", "qian_reddit": "qian_reddit", "qian_gab": "qian_gab", "cga": "cga", "bad": "bad"}
NMO = {"cc": "Civil Comments (toxicity)", "cad": "CAD (abuse)", "qian_reddit": "Qian et al.'s Reddit (hate)", "qian_gab": "Qian et al.'s Gab (hate)", "cga": "CGA-Wiki (attack)", "bad": "BAD (unsafe turn)"}
XCC = load("x89/x89_cc_reference.json")
TO = []
for c_ in ("cc", "cad", "qian_reddit", "qian_gab", "cga", "bad"):
    R_ = REC[c_]; yr_ = XCC["cc_threads"] if c_ == "cc" else X89[c_]
    lab_ = "haiku" if "haiku" in PLJ[OWNP[c_]]["labelers"] else "haiku_ctx"; ro_ = PLJ[OWNP[c_]]["labelers"][lab_]; mk_ = "" if lab_ == "haiku" else "$^{\\|}$"
    tg_ = CTAG[c_]
    own_gap = f(PP * R_["haiku"]["fpr_you_minus_other"][0], 1) if c_ == "cc" else M.get(f"xb{tg_}OwnGapHaiku", "--")          # BAD: no own-question label without the previous turn
    TO.append(" & ".join([NMO[c_], f"{100 * yr_['you_root']:.0f}\\%", f"{100 * yr_['you_deep']:.0f}\\%", f(PP * R_["haiku"]["fpr_you_minus_other"][0], 1), own_gap,
                          (ppci(ro_["placebo_depth_slope"]) if c_ == "cc" else ci3(R_["haiku"]["placebo"])), ppci(ro_["placebo_depth_slope"]) + mk_,   # Civil Comments: Table 4's interval (x74_placebo) in both columns
                          sg(R_["gold"]["depth"][0]), sg(R_["haiku"]["depth"][0]), sg(PP * ro_["law_depth_slope"][0]) + mk_]) + " \\\\")
    if c_ == "cc":
        assert abs(ro_["placebo_depth_slope"][0] * PP - R_["haiku"]["placebo"][0]) < 1e-6, "Civil Comments: the own question is the toxicity question"
    put(f"xb{tg_}YouRoot", f"{100 * yr_['you_root']:.0f}\\%", "results/x89/x89_other_tox.json (x89_cc_reference.json on Civil Comments)", "you_root: share of gold negatives at the root with a second-person word")
    put(f"xb{tg_}YouDeep", f"{100 * yr_['you_deep']:.0f}\\%", "same", "you_deep: the same at depth 6 or more")
table("xb_corpora_own", "The bias across corpora, under the Civil Comments toxicity question and under each corpus's own question (Haiku without the parent): second-person shares of gold negatives at the root and at depth 6+; "
      "extra flags on 'you' (points); Haiku's placebo depth slope (points per level, 95% CI); the depth law under gold, Haiku asked the toxicity question and Haiku asked the corpus's question. Double bar: BAD's own-question labels exist only with the previous turn shown.",
      ["Corpus", "You root", "You 6+", "Gap tox", "Gap own", "Placebo tox", "Placebo own", "Gold law", "Haiku law tox", "Haiku law own"], TO,
      [("Shares", "results/x89/x89_other_tox.json and x89_cc_reference.json: you_root, you_deep"), ("Gaps", "toxicity: x89 fpr_you_minus_other; own: x92_extra (macros xb<tag>OwnGapHaiku)"),
       ("Placebo, laws", "toxicity: x89 placebo, depth; own: results/x74_placebo/x74_placebo.json labelers/haiku (BAD: haiku_ctx)")],
      "On Civil Comments the corpus's own question is the toxicity question, so both columns hold the same labels.")

# ================================================================ outputs
with open(OUT_NUM, "w") as fo:
    fo.write("% generated by scripts/x98b_size_contagion.py (python3 -I); provenance in provenance_xb.md\n")
    for k, v in M.items():
        fo.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
md = lambda s_: str(s_).replace("|", "\\|")
with open(OUT_PROV, "w") as fo:
    fo.write("# Provenance of numbers_arxiv_b.tex and tables/xb_*.tex\n\n")
    fo.write("Generated by `scripts/x98b_size_contagion.py` (run with `python3 -I`; re-runnable, all resampling seeded). Every value below is written by that script from the stored file named; "
             "nothing is typed by hand. Values are given exactly as the macro prints them.\n\n")
    fo.write("## Conventions\n\n"
             "- Populations: Civil Comments reply threads `cc_threads` (1,251,487 comments, 270,967 threads, 980,520 reply edges) and its 5% sample `cc_s5` (60,668); CAD, Qian et al.'s Reddit and Gab, "
             "CGA-Wiki and BAD as in `results/x14_pops`. Each source's population is asserted in the script.\n"
             "- Laws: depth law in points per reply level (depth capped at 10), size laws in points per doubling of the unit's size (OLS slope of the unit's toxic share on log2 of its size, units with "
             ">= 2 comments, each unit once), contagion as the reply risk ratio (RR) or its log.\n"
             "- 95% CIs are the stored percentile intervals of each source's cluster resamples (articles on Civil Comments, conversations elsewhere). New resampling in this script: 2,000 Poisson "
             "cluster resamples, seeds 9821 (size laws, other corpora) and 9822 (rewrite, 5% sample).\n"
             "- The 14 classifier settings: Detoxify (0.6, gold's rate), Detoxify-unbiased§ (0.6, gold's rate), IMSYPP (argmax, gold's rate), s-nlp RoBERTa§, martin-ha DistilBERT§, Dynabench hate "
             "and CardiffNLP offensive (0.5, gold's rate); § = trained at least in part on Civil Comments labels. Detoxify-unbiased is stored in every source used, so every range covers all 14 "
             "settings unless the macro says '12' (then it is the earlier-draft macro's 12-setting range, kept for the cross-check).\n"
             "- '+ parent' labelers: no pooled value is used. Size laws, unit shares, within-thread laws, the parent-flag many-rater RR, within-article contagion and the construct-only escalation test "
             "are reported without the parent only; shown the parent, only among-replies statistics (depth among replies, the parent-gold many-rater RR, the contrast effect, f122's reply RR with a "
             "context-free parent label) are used. No size law among replies is stored, so the '+ parent' size laws are omitted.\n"
             "- f-series files: f96's thread part is on the current population (asserted: 270,967 threads, 1,251,487 comments) and is used only as a replicate of the shuffle test; its article "
             "part (1,061,408 comments of old train-split sections) is not used. f31 (E33) analyses CCC's own 9,903 posts (9,908 for Qwen and Gemma), with human labels from CCC and parent "
             "toxicity from cc_meta; it does not depend on the reply-thread population, so the old-population issue does not affect it. f122 (E53) is on the current Qian populations (asserted).\n"
             "- Keep-out items not used: K2 (x80_noise reweighting), K4 (raw within-article MH comparisons, incl. x80_mech's placebo_rr_mh_article and x80_mech2), K5 (x82_latent.json), "
             "K8 (x83_construct.json), K9 (x74_crowd split halves), K10, K11, K12, K15, K16.\n\n")
    fo.write("## Discrepancies with the earlier-draft macros (verified by the script)\n\n")
    for d_ in DISC:
        fo.write(f"- {d_}\n")
    if os.path.exists(f"{W}/results/reference/main.tex"):                     # release: the earlier draft's text is not distributed
        draft_ = "".join(open(p_).read() for p_ in sorted(glob.glob(f"{W}/results/reference/sec/*.tex") + glob.glob(f"{W}/results/reference/tables/*.tex")) + [f"{W}/results/reference/main.tex"])
        used_ = [e_ for e_ in DISC_EX if re.search(r"\\" + e_ + r"(?![A-Za-z])", draft_)]
        fo.write(f"\nOf these {len(DISC_EX)} earlier-draft macros, " + ("none is printed in the earlier draft's text (paper_v5/main.tex, sec/*.tex, tables/*.tex; checked by the script), so the earlier draft itself "
                 "is not contradicted by them; they matter only if the arXiv text reuses them.\n" if not used_ else ", ".join("\\" + e_ for e_ in used_)
                 + " are printed in the earlier draft's text (paper_v5/main.tex, sec/*.tex, tables/*.tex; checked by the script): those statements need the value given here.\n"))
    else:
        fo.write(f"\nWhether any of these {len(DISC_EX)} earlier-draft macros is printed in the earlier draft's text was not checked: that text is not part of this repository.\n")
    fo.write("\n## Discrepancies and qualifications against the inventory and v7f\n\n")
    for d_ in INV:
        fo.write(f"- {d_}\n")
    fo.write("\n## v7f statements these numbers qualify (candidates for Limitations; no number in v7f is contradicted except where listed above)\n\n")
    for d_ in QUAL:
        fo.write(f"- {d_}\n")
    fo.write("\n## Macros\n\n")
    for title, names in SECS:
        if not names:
            continue
        fo.write(f"### {title}\n\n| Macro | Value | Source | Transformation | Cross-check | Caveat |\n|---|---|---|---|---|---|\n")
        for k in names:
            P_ = PROV[k]
            fo.write(f"| `\\{k}` | `{md(M[k])}` | {md(P_['src'])} | {md(P_['how'])} | {md('; '.join(P_['chk'])) or '--'} | {md(P_['cav']) or '--'} |\n")
        fo.write("\n")
    fo.write("## Tables\n\n")
    for T_ in TABLES:
        fo.write(f"### tables/{T_['name']}.tex ({len([r for r in T_['rows'] if r.endswith(chr(92) * 2)])} rows)\n\n")
        fo.write(f"**Proposed caption.** {T_['caption']}\n\n")
        fo.write("**Columns.** " + " | ".join(T_["headers"]) + f" ({len(T_['headers'])} columns)\n\n")
        fo.write("**Sources and transformations.**\n\n")
        for a_, b_ in T_["cols"]:
            fo.write(f"- {a_}: {b_}\n")
        fo.write(f"\n**Caveats.** {T_['cav'] or '--'}\n\n")
        n_amp = {r.count('&') for r in T_["rows"] if r.endswith(chr(92) * 2)}
        assert n_amp == {len(T_["headers"]) - 1}, (T_["name"], n_amp)                          # every row has one cell per header
print(len(M), "macros ->", OUT_NUM)
print(len(TABLES), "tables ->", ", ".join(f"tables/{T_['name']}.tex" for T_ in TABLES))
print(len(DISC), "differences from earlier-draft macros (explained); provenance ->", OUT_PROV)
