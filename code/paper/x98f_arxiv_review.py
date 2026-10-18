"""arXiv version, review round 1: the numbers the revised text needs that no other x98 script writes (no new labels; stored
labels and results only). Writes paper_arxiv/numbers_arxiv_f.tex (prefix xf) and paper_arxiv/provenance_xf.md.
  - the train file's reply threads before the label-availability rule (data/cc_meta.parquet, threads rebuilt as f10_population.py does)
  - the share of matched gold negatives that the platform rejected, by depth (as x85_construct_frac.py)
  - the 198 threads of the sampled articles that the 5% sample leaves out (results/x81/prn.json, results/second/s5_articles.txt)
  - comments without a reply edge in CAD and CGA-Wiki (results/x14_pops)
  - Haiku's unstructured share of its depth distortion (results/x74_placebo/x74_placebo.json, decomposition)
  - the 0-100 ratings at flag rates near the own cut (results/x75_rating/x75_rating_eval.json) and the matched LLM laws (results/x92/x92_matched.json)
  - the edit sample's asymmetry (results/x76_edit/x76_edit.json)
  - percentage versions of coverage macros that the other scripts print as proportions (read from paper_arxiv/numbers_arxiv_c.tex)
usage: python3 -I scripts/x98f_arxiv_review.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, re
import numpy as np, pandas as pd
W = _ROOT; R = f"{W}/results"; PA = f"{W}/paper_arxiv"
M, PROV = {}, []
th = lambda n: f"{int(n):,}".replace(",", "{,}")
pc0 = lambda v: f"{100 * v:.0f}\\%"
pc1 = lambda v: f"{100 * v:.1f}\\%"


def sg(x, d=2):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", "$-$")


def put(name, val, src, how):
    assert re.fullmatch(r"xf[A-Za-z]+", name) and name not in M, name
    M[name] = val; PROV.append((name, val, src, how))


# ---------------------------------------------------------------- 1. the train file's reply threads (W10)
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id", "split", "Z", "rating"]).set_index("id")
D = meta[meta.split == "train"]
par = D.parent_id.where(D.parent_id.isin(D.index)).astype("Int64")                       # as f10_population.py: a parent outside the file makes a root
pmap = par.dropna().astype("int64"); cur = pd.Series(D.index.values, index=D.index)
for _ in range(200):
    nxt = cur.map(pmap).fillna(cur).astype("int64")
    if (nxt == cur).all():
        break
    cur = nxt
tk = cur.map(cur.value_counts())
roots_all = cur[tk >= 2].unique(); n_thr_all = len(roots_all); n_com_all = int((tk >= 2).sum())
F10 = json.load(open(f"{R}/f10_population.json"))
assert F10["n_comments_total"] == len(D) and F10["n_threads_k2"] == 270967 and F10["n_comments_in_threads"] == 1251487
P = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["id", "unit", "depth", "z", "article"])
kept_roots = set(cur.loc[P.id.values].unique()); assert len(kept_roots) == 270967
dropped = [r for r in roots_all if r not in kept_roots]
root_has_outside_parent = D.parent_id.loc[dropped].notna()
put("xfTrainThreadsAll", th(n_thr_all), "data/cc_meta.parquet (split train)", "reply threads of at least two comments in the training file, rebuilt as f10_population.py does (a comment whose parent is outside the file starts a thread)")
put("xfTrainThreadsAllComments", th(n_com_all), "same", "comments in those threads")
put("xfTrainThreadsDropped", th(len(dropped)), "same and results/x14_pops/cc_threads.parquet", "threads of at least two comments that the population leaves out because some comment lacks a label")
put("xfTrainThreadsDroppedShare", pc1(len(dropped) / n_thr_all), "same", "their share of the threads of at least two comments")
put("xfTrainCommentsDropped", th(n_com_all - 1251487), "same", "comments in the dropped threads")
put("xfTrainDroppedOutsideShare", pc0(root_has_outside_parent.mean()), "same", "share of the dropped threads whose root replies to a comment outside the training file (so it has no text to show as parent)")
put("xfTrainDroppedOutsideN", th(int(root_has_outside_parent.sum())), "same", "number of the dropped threads whose root replies to a comment outside the training file")
print("dropped threads with an outside parent:", int(root_has_outside_parent.sum()), "of", len(dropped))
put("xfPopShareTrain", pc0(1251487 / len(D)), "same", "the population's share of the training file's comments")
put("xfTrainSingleShare", pc0((len(D) - n_com_all) / len(D)), "same", "share of the training file's comments that are single comments (a thread of one comment: no parent in the file and no reply)")
ma_ = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["split", "article_id"]).dropna()
assert not (set(ma_[ma_.split == "train"].article_id) & set(ma_[ma_.split != "train"].article_id))   # every article's comments are in one file, so the train file holds its articles' whole released sections
assert n_thr_all - 270967 == len(dropped)

# ---------------------------------------------------------------- 2. rejected share among matched gold negatives, by depth (W5)
Fm = pd.read_parquet(f"{W}/data/cc_attr_frac.parquet", columns=["id"]).set_index("id")
P2 = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["id", "depth", "z"])
rej = (meta.rating.reindex(P2.id.values).values == "rejected")
neg = (P2.z.values == 0) & P2.id.isin(Fm.index).values
x = np.minimum(P2.depth.values, 10)
lv = float(rej[neg].mean()); X85 = json.load(open(f"{R}/x85/x85_construct_frac.json"))
assert abs(100 * lv - X85["levels"]["rejected"][0]) < 1e-9
put("xfRejRoot", pc1(rej[neg & (x == 0)].mean()), "data/cc_meta.parquet rating; results/x14_pops/cc_threads.parquet; data/cc_attr_frac.parquet (matched gold negatives, as x85_construct_frac.py)", "share rejected by the platform among matched gold negatives at the root")
put("xfRejDepthOne", pc1(rej[neg & (x == 1)].mean()), "same", "at depth 1")
put("xfRejAll", pc1(lv), "same", "over all matched gold negatives (= x85 levels.rejected)")

# ---------------------------------------------------------------- 3. the threads of the sampled articles that the 5% sample leaves out
arts = {int(l) for l in open(f"{R}/second/s5_articles.txt") if l.strip()}
PR = json.load(open(f"{R}/x81/prn.json"))
P3 = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet", columns=["id", "unit", "z", "article"])
S5 = pd.read_parquet(f"{R}/x14_pops/cc_s5.parquet", columns=["id", "z"])
art_id = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "article_id"]).set_index("id").article_id
inart = P3[art_id.reindex(P3.id.values).isin(arts).values]                               # the real article ids (the parquet's article column is a factorized index)
assert len(inart) == PR["n_slope"] and inart.unit.nunique() == PR["n_threads"], (len(inart), inart.unit.nunique())
out = inart[~inart.id.isin(S5.id)]
assert len(out) == PR["n_slope"] - PR["n_cc_s5"]
put("xfSfiveDropTox", pc0(out.z.mean()), "results/x14_pops/cc_threads.parquet, results/second/s5_articles.txt, results/x14_pops/cc_s5.parquet", "gold's toxic share among the comments of the threads that the 5% sample leaves out")
put("xfSfiveTox", pc1(S5.z.mean()), "results/x14_pops/cc_s5.parquet", "gold's toxic share in the 5% sample")

# ---------------------------------------------------------------- 4. comments without a reply edge in CAD and CGA-Wiki
for f_, t_ in (("cad", "Cad"), ("cga", "Cga")):
    Q = pd.read_parquet(f"{R}/x14_pops/{f_}.parquet", columns=["unit", "depth", "parent_row"])
    put(f"xf{t_}NoEdge", th(int((Q.parent_row < 0).sum()) - Q.unit.nunique()), f"results/x14_pops/{f_}.parquet", "comments without a parent in the data beyond one root per unit")
    put(f"xf{t_}Roots", th(int((Q.depth == 0).sum())), f"results/x14_pops/{f_}.parquet", "comments at depth 0")

# ---------------------------------------------------------------- 5. Haiku's unstructured share; laws over five levels
PL = json.load(open(f"{R}/x74_placebo/x74_placebo.json"))
G = PL["cc_threads"]["gold"]; H = PL["cc_threads"]["labelers"]["haiku"]
DX = json.load(open(f"{R}/x74_decomp/x74_decomp.json"))["cc_threads"]["labelers"]["haiku"]["laws"]["depth"]        # the census's terms (x98c)
u_ = DX["attenuation_term"][0] - DX["gold"][0]; sfp, stp = DX["fp_term"][0], DX["tp_term"][0]
assert abs(u_ + sfp + stp - DX["distortion"][0]) < 1e-12 and abs(DX["distortion"][0] - H["distortion"]["depth"]["est"]) < 1e-12
put("xfHaikuUnifDepth", pc0(abs(u_) / (abs(u_) + abs(sfp) + abs(stp))), "results/x74_decomp/x74_decomp.json", "cc_threads Haiku depth law: |unstructured| / (|unstructured| + |S_FP| + |S_TP|), the terms of x98c's census")
Q7 = PL["cc_threads"]["labelers"]["qwen7b"]
put("xfQwenOwnDepth", sg(100 * Q7["law_depth_slope"][0]), "results/x74_placebo/x74_placebo.json", "cc_threads/labelers/qwen7b/law_depth_slope x 100: Qwen at its own cut (p > 0.5)")
put("xfQwenOwnDist", sg(100 * Q7["distortion"]["depth"]["est"]), "same", "its depth distortion x 100")
put("xfPrItems", th(PR["n_items"]), "results/x81/prn.json", "n_items: the comments of the second run's articles (single comments included), over which its flag rates and kappa are computed")
put("xfHaikuFiveLevels", sg(500 * H["law_depth_slope"][0], 1), "results/x74_placebo/x74_placebo.json", "cc_threads/labelers/haiku/law_depth_slope x 500: Haiku's depth law times five levels (points)")
put("xfGoldFiveLevels", sg(500 * G["depth_slope"][0], 1), "results/x74_placebo/x74_placebo.json", "cc_threads/gold/depth_slope x 500: gold's depth law times five levels (points)")
assert H["law_depth_slope"][0] > 0 > G["depth_slope"][0]
put("xfHaikuFiveLevelsAbs", f"{abs(500 * H['law_depth_slope'][0]):.1f}", "results/x74_placebo/x74_placebo.json", "the same without sign (Haiku's law adds this many points over five levels)")
put("xfGoldFiveLevelsAbs", f"{abs(500 * G['depth_slope'][0]):.1f}", "results/x74_placebo/x74_placebo.json", "the same without sign (gold's law takes away this many points over five levels)")

# ---------------------------------------------------------------- 6. the 0-100 ratings near the own cut; the matched laws on the 5% sample
RT = json.load(open(f"{R}/x75_rating/x75_rating_eval.json"))["instruments"]
for k_, t_, r_ in (("haiku_r", "HaikuHundredTwenty", "0.2"), ("sonnet_r", "SonnetHundredThirty", "0.3")):
    v = RT[k_][r_]["depth_minus_gold"]
    put(f"xf{t_}", sg(100 * v[0]), "results/x75_rating/x75_rating_eval.json", f"instruments.{k_}['{r_}'].depth_minus_gold x 100 (5% sample)")
    put(f"xf{t_}Lo", sg(100 * v[1]), "same", "2.5th percentile"); put(f"xf{t_}Hi", sg(100 * v[2]), "same", "97.5th percentile")
MT = json.load(open(f"{R}/x92/x92_matched.json")); rows = {(r["model"], r["op"]): r for r in MT["rows"]}; g5 = MT["gold"]["depth"][0]
put("xfHaikuHundredLaw", sg(g5 + rows[("haiku", "rating at gold prevalence")]["depth_excess"][0]), "results/x92/x92_matched.json", "gold's 5%-sample depth law plus Haiku-100's distortion")
put("xfGemmaLaw", sg(g5 + rows[("gemma", "gold prevalence")]["depth_excess"][0]), "same", "gold's 5%-sample depth law plus Gemma's distortion at gold's flag rate")
put("xfGoldSfiveDepth", sg(g5), "same", "gold's depth law on the 5% sample")

# ---------------------------------------------------------------- 7. the edit sample: removing and adding "you"
E = json.load(open(f"{R}/x76_edit/x76_edit.json"))["instruments"]
for m_, t_ in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    e_ = E[m_]
    put(f"xf{t_}RmDeep", f"{-100 * e_['h11/neg_deep']['E1_minus_E0']['diff']:.1f}", "results/x76_edit/x76_edit.json", f"instruments.{m_}['h11/neg_deep'].E1_minus_E0 x -100: points by which the rule lowers the flag rate on clearly non-toxic replies at depth >= 2")
    put(f"xf{t_}RmRoot", f"{-100 * e_['h11/neg_root']['E1_minus_E0']['diff']:.1f}", "same", "the same on clearly non-toxic roots")
    put(f"xf{t_}AddDeep", f"{100 * e_['rev/neg_deep']['E2_minus_E0']['diff']:.1f}", "same", "points by which the reverse rule raises the flag rate at depth >= 2")
    put(f"xf{t_}AddRoot", f"{100 * e_['rev/neg_root']['E2_minus_E0']['diff']:.1f}", "same", "the same at the root")
    put(f"xf{t_}PosBefore", pc1(float(e_["h11/pos_any"]["E0"])), "same", "flag rate on the edit sample's second-person gold positives as written")
    put(f"xf{t_}PosAfter", pc1(float(e_["h11/pos_any"]["E1"])), "same", "after the rule rewrite")
    put(f"xf{t_}ParaYou", pc1(float(e_["h11/neg_deep"]["E3"])), "same", "flag rate on Sonnet's paraphrases that keep 'you' (clearly non-toxic replies at depth >= 2)")
    put(f"xf{t_}ParaThey", pc1(float(e_["h11/neg_deep"]["E4"])), "same", "on the paraphrases in the third person")

# ---------------------------------------------------------------- 7b. review round 2: removing "you" at a fixed number of second-person words
#      The edit sample's labels as x76_edit_eval.py reads them (train split, comments with all four versions labeled); each comment's number of
#      second-person words is counted on its text as written with Appendix A's detector.
import glob
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id")
S11["train"] = meta.split.reindex(S11.index).eq("train").values
TX = pd.DataFrame([json.loads(l) for l in open(f"{R}/x74_edit/texts.jsonl")]); TX["cid"] = TX.id // 10
assert (TX.id % 10 == TX.variant).all()
T0 = TX[(TX["sample"] == "h11") & (TX.variant == 0)].set_index("cid")
S11["nyou"] = T0.text.reindex(S11.index).map(lambda t: len(YOU.findall(t)))


def letters(path):                                                                                  # as x76_edit_eval.py
    Rr = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{path}/shard_*.jsonl")) for l in open(f)])
    if "error" in Rr:
        Rr = Rr[Rr["error"].isna()]
    Rr = Rr.drop_duplicates("id", keep="last"); Rr["flag"] = Rr.ans.isin(["A", "B"]).astype(float); Rr["cid"] = Rr.id // 10; Rr["var"] = Rr.id % 10
    return Rr.pivot(index="cid", columns="var", values="flag")


rng7 = np.random.default_rng(9871); NY = {}
for m_, t_ in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    Pv = letters(f"{R}/x74_edit/{m_}_tox")
    J = S11[S11.train].join(Pv, how="inner"); J = J[J[[0, 1, 3, 4]].notna().all(axis=1)]
    rt, dp = J[J.stratum == "neg_root"], J[J.stratum == "neg_deep"]
    assert rt.nyou.min() >= 1 and dp.nyou.min() >= 1
    for st_, g_ in (("neg_deep", dp), ("neg_root", rt)):                                            # the stored effects, reproduced
        assert abs((g_[1] - g_[0]).mean() - E[m_][f"h11/{st_}"]["E1_minus_E0"]["diff"]) < 1e-12 and len(g_) == E[m_][f"h11/{st_}"]["n"]
    cat = lambda x: x.nyou.clip(upper=3)
    w_ = cat(rt).value_counts(normalize=True).sort_index()
    dd_ = {k: (dp[cat(dp) == k][1] - dp[cat(dp) == k][0]).values for k in w_.index}
    std_ = sum(w_[k] * dd_[k].mean() for k in w_.index)
    bs_ = [sum(w_[k] * dd_[k][rng7.integers(0, len(dd_[k]), len(dd_[k]))].mean() for k in w_.index) for _ in range(2000)]
    put(f"xf{t_}RmDeepStd", f"{-100 * std_:.1f}", "results/x74_edit (labels), data/you_edit_meta.parquet, results/x74_edit/texts.jsonl",
        "points by which the rule lowers the flag rate on clearly non-toxic replies at depth >= 2, standardized to the roots' mix of second-person words (1, 2, 3 or more)")
    put(f"xf{t_}RmDeepStdLo", f"{-100 * np.percentile(bs_, 97.5):.1f}", "same", "its 95% interval (2,000 resamples of the deep comments within count strata, seed 9871), lower end in points lowered")
    put(f"xf{t_}RmDeepStdHi", f"{-100 * np.percentile(bs_, 2.5):.1f}", "same", "upper end")
    for k_, nm_ in ((1, "One"),):
        put(f"xf{t_}RmDeep{nm_}", f"{-100 * dd_[k_].mean():.1f}", "same", f"the rule's effect on deep replies with exactly {k_} second-person word (points lowered)")
        put(f"xf{t_}RmRoot{nm_}", f"{-100 * (rt[cat(rt) == k_][1] - rt[cat(rt) == k_][0]).mean():.1f}", "same", f"on roots with exactly {k_} second-person word")
    NY[m_] = (dp.nyou.mean(), rt.nyou.mean(), len(dp), len(rt))
assert NY["haiku"] == NY["sonnet"]
put("xfEditNyouDeep", f"{NY['haiku'][0]:.2f}", "same", "mean number of second-person words of the edit sample's clearly non-toxic replies at depth >= 2 (as written)")
put("xfEditNyouRoot", f"{NY['haiku'][1]:.2f}", "same", "the same for its clearly non-toxic roots")

# ---------------------------------------------------------------- 7c. review round 2: Figure 1b's standardized rates (the figure's own data file)
FD = json.load(open(f"{W}/results/reference/figs/dose.json")); hs_ = FD["haiku_standardized"]; assert FD["depth"][0] == 0 and len(hs_) == 9
put("xfStdRoot", f"{hs_[0]:.1f}\\%", "paper_arxiv/figs/dose.json", "haiku_standardized[0]: Haiku's false-positive rate at the root (Figure 1b)")
put("xfStdRepMin", f"{min(hs_[1:]):.1f}\\%", "same", "the lowest standardized rate over depths 1 to 8+")
put("xfStdRepMax", f"{max(hs_[1:]):.1f}\\%", "same", "the highest")

# ---------------------------------------------------------------- 7d. review round 3: Qwen's dose on all reply threads (x98g_qwen_dose.py, same code as x90)
QD = json.load(open(f"{R}/x98g/x98g_qwen_dose.json")); assert QD["N"] == 1251487 and "asserted" in QD["check"]
QG, QO = QD["labelers"]["qwen_gold_rate"], QD["labelers"]["qwen_own_cut"]
assert abs(QO["flag_rate"] - 0.5704) < 5e-5 and abs(QG["flag_rate"] - 0.0748) < 5e-5
for nm_, r_ in (("QwenGoldDose", QG), ("QwenOwnDose", QO)):
    for key_, lab_, how_ in (("law_distortion", "Dist", "depth-law distortion (labeler minus gold), points per level"),
                             ("law_distortion_within_count", "DistCount", "the same, pooled within the number of second-person words (0..4+)"),
                             ("placebo", "Plac", "placebo depth slope among gold negatives, points per level"),
                             ("within_count", "PlacCount", "placebo depth slope pooled within the number of second-person words")):
        v_ = r_[key_]
        put(f"xf{nm_}{lab_}", sg(v_[0]), "results/x98g/x98g_qwen_dose.json", f"labelers.{'qwen_gold_rate' if nm_ == 'QwenGoldDose' else 'qwen_own_cut'}.{key_}[0]: {how_}, all reply threads")
        put(f"xf{nm_}{lab_}Lo", sg(v_[1]), "same", "2.5th percentile of 500 cluster resamples (seed 9001)")
        put(f"xf{nm_}{lab_}Hi", sg(v_[2]), "same", "97.5th percentile")
assert QG["law_distortion"][1] > 0 and QG["law_distortion_within_count"][2] < 0             # a false rise that turns negative at a fixed number
assert QO["within_count"][2] < QO["placebo"][1]                                                # at its own cut, the false positives fall faster at a fixed number

# ---------------------------------------------------------------- 8. coverage macros as percentages
NC = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", open(f"{PA}/numbers_arxiv_c.tex").read())}
for k_ in ("xcPpiSmallCov", "xcPpiSmallMin", "xcFcCovLaws", "xcFcMinLaws", "xcOcCovLaws", "xcOcMinLaws", "xcUnCovMin", "xcAdvItemMin", "xcAdvEdgeMin", "xcAdvPpiMin", "xcAdvUnifMin",
           "xcOneFoldTsCovMin", "xcOneFoldCovMin"):
    v = NC[k_]; assert re.fullmatch(r"0\.\d{3}", v), (k_, v)
    put("xf" + k_[2:] + "Pct", f"{100 * float(v):.1f}\\%", "paper_arxiv/numbers_arxiv_c.tex", f"\\{k_} = {v} as a percentage")

txt = "% generated by scripts/x98f_arxiv_review.py (python3 -I); provenance in provenance_xf.md\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items())
open(f"{PA}/numbers_arxiv_f.tex", "w").write(txt)
with open(f"{PA}/provenance_xf.md", "w") as fo:
    fo.write("# Provenance of numbers_arxiv_f.tex\n\nGenerated by `scripts/x98f_arxiv_review.py` (deterministic; its one bootstrap is seeded).\n\n| Macro | Value | Source | How |\n|---|---|---|---|\n")
    for n_, v_, s_, h_ in PROV:
        fo.write(f"| `\\{n_}` | `{v_}` | {s_} | {h_} |\n")
for k, v in M.items():
    print(k, "=", v)
