"""arXiv extended version, numbers agent C: what agreement can certify, what the distortions do to conclusions, and the evaluation
of Threadsafe (inventory items E55-E63 and M01-M14). Every number comes from a stored result file; the only re-analysis of stored
labels is E63's reply-association region (the logic of scripts/f14_regions.py, rerun on results/x14_pops/cc_threads.parquet).
Writes
  paper_arxiv/numbers_arxiv_c.tex       LaTeX macros, prefix \\xc (letters only)
  paper_arxiv/tables/xc_*.tex            table bodies (rows only)
  paper_arxiv/figs/kappa_distortion.pdf  kappa against the depth-law distortion, one panel per population
  paper_arxiv/provenance_xc.md           value, source and key, transformation, cross-check and caveats of every macro and table
Rules applied throughout:
  - current populations only (results/x14_pops; cc_threads = 1,251,487 comments); no f-series result of the old population;
  - no pooled "+ parent" law value: a labeler shown the parent labels roots without it and replies with it, so a law pooled over
    both modes carries a root/reply level shift. Such entries are dropped from every census, correlation, selection and figure,
    or replaced by the among-replies values where those are stored (results/x97/x97_decomp_replies.json). In the simulated audits
    a labeler shown the parent only steers the design, so its cells stay in the coverage and width evaluation; in the n80 table
    (which tests the labeler's own law) those rows are relabeled, and the cell whose labeler uses an abandoned threshold
    (Qwen shown the parent at the no-context threshold, 21.9% flagged) is dropped;
  - nothing from the keep-out list K1-K30 (in particular not the lenient human benchmark, K11, nor results/x74_crowd splits, K9);
  - where an earlier-draft macro (paper_v5/numbers_*.tex except numbers_v2.tex) holds the same number, chk() asserts equality at the earlier-draft
    macro's precision, or records the explained difference in the provenance file.
usage: python3 -I x98c_agree_eval.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

W = _ROOT; RES = f"{W}/results"; PA = f"{W}/paper_arxiv"
OUT_NUM = f"{PA}/numbers_arxiv_c.tex"; OUT_TAB = f"{PA}/tables"; OUT_FIG = f"{PA}/figs/kappa_distortion.pdf"; OUT_PROV = f"{PA}/provenance_xc.md"
PREVIEW = (_ROOT + "/scratch/xc/kappa_distortion.png")   # preview only, outside the paper
for p_ in (OUT_NUM, OUT_TAB + "/x", OUT_FIG, OUT_PROV):
    assert os.path.abspath(p_).startswith(PA + "/"), p_                                    # write nowhere else
os.makedirs(OUT_TAB, exist_ok=True); os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True); os.makedirs(os.path.dirname(PREVIEW), exist_ok=True)


def load(p):
    return json.load(open(f"{RES}/{p}"))


# ---------------------------------------------------------------- formatting (as scripts/x18_numbers_v5.py and x95_eval_v6.py)
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


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def cov3(x):                                                                                    # coverage as in Table 5: .949
    return f"{x:.3f}".replace("0.", ".", 1) if x < 1 else f"{x:.3f}"


def cov2(x):
    return f"{x:.2f}".replace("0.", ".", 1) if x < 1 else f"{x:.2f}"


def c3(m, lo):                                                                                  # ".949 (.92)", as x95_eval_v6.c3
    return f"{cov3(m)} ({cov2(lo)})"


def rng(lo, hi, d=2):
    return f"{sg(lo, d)} to {sg(hi, d)}"


f2 = lambda x: "--" if x is None else f"{x:.2f}"
i0 = lambda x: str(int(x))

# ---------------------------------------------------------------- the earlier-draft macros, for cross-checks
OLD, OLDF = {}, {}
for fn in sorted(glob.glob(f"{W}/results/reference/numbers_*.tex")):
    if os.path.basename(fn) == "numbers_v2.tex":                                               # old population (1.78M): never used
        continue
    if os.path.basename(fn).startswith(("numbers_arxiv_", "numbers_v7_fix")):                 # copies of this pipeline's own outputs (the earlier draft v7h inputs them)
        continue
    for line in open(fn):
        m_ = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line.strip())
        if m_:
            assert m_.group(1) not in OLD, m_.group(1)
            OLD[m_.group(1)] = m_.group(2); OLDF[m_.group(1)] = os.path.basename(fn)
V7F = open(f"{W}/docs/review_v7/flat_v7f.tex").read() if os.path.exists(f"{W}/docs/review_v7/flat_v7f.tex") else None   # release: the draft is not distributed

# ---------------------------------------------------------------- macros with provenance
M, PROV, DISC, TABLES, SKIP, SECT = {}, {}, [], {}, [], {}
CUR = ["setup"]


def section(name):
    CUR[0] = name; SECT.setdefault(name, [])


def put(name, raw, fmt, src, key, how, caveat=""):
    assert re.fullmatch(r"xc[A-Za-z]+", name), name
    assert name not in M, f"duplicate macro {name}"
    val = fmt(raw) if fmt is not None else raw
    assert isinstance(val, str) and "nan" not in val.lower() and "??" not in val, (name, val)
    M[name] = val; PROV[name] = dict(value=val, raw=raw, src=src, key=key, how=how, checks=[], caveat=caveat); SECT[CUR[0]].append(name)
    return val


def _clean(s):
    return s.replace("$-$", "-").replace("\\%", "").replace("{,}", "").replace("$", "").replace("{", "").replace("}", "").strip()


def num(s):
    return float(_clean(s).replace("+", ""))


def ndec(s):
    t = _clean(s)
    return len(t.split(".")[1]) if "." in t else 0


def chk(name, old, scale=1.0, why=None):
    """Compare \\name with the earlier-draft macro \\old: they must agree at \\old's precision (|raw * scale - old| <= half a unit of its last
    digit), unless `why` explains the difference, which is then recorded as a discrepancy."""
    ov = OLD[old]; p = PROV[name]; raw = p["raw"]
    if isinstance(raw, str):
        same = M[name] == ov
    else:
        d = ndec(ov); same = abs(float(raw) * scale - num(ov)) <= 0.5 * 10 ** -d + 1e-9
    if same:
        p["checks"].append(f"equals `\\{old}` = {ov} ({OLDF[old]})" + (f"; {why}" if why else ""))
    elif why:
        p["checks"].append(f"DIFFERS from `\\{old}` = {ov} ({OLDF[old]}): {why}")
        DISC.append(f"`\\{name}` = {M[name]} against `\\{old}` = {ov} ({OLDF[old]}): {why}")
    else:
        raise AssertionError(f"\\{name} = {M[name]} (raw {raw}) disagrees with \\{old} = {ov} ({OLDF[old]})")


def note(name, text):
    PROV[name]["checks"].append(text)


def v7f(name, snippet, same=True, why=""):
    """The flattened text of the earlier draft (v7f) prints `snippet`; same=False records that the number printed there differs from this macro."""
    assert V7F is None or snippet in V7F, (name, snippet)
    if same:
        note(name, f"printed in v7f as '{snippet}'")
    else:
        note(name, f"v7f prints '{snippet}': {why}"); DISC.append(f"v7f prints '{snippet}', `\\{name}` = {M[name]}: {why}")


def table(name, rows, caption, headers, src, how, checks, caveats):
    path = f"{OUT_TAB}/xc_{name}.tex"
    open(path, "w").write("\n".join(rows) + "\n")
    TABLES[name] = dict(path=path, rows=sum(r.count(" \\\\") for r in rows), caption=caption, headers=headers, src=src, how=how,
                        checks=checks, caveats=caveats, section=CUR[0])


CTX = lambda L: "_ctx" in L                                                                     # a labeler shown the parent
LLMP = ("haiku", "sonnet", "qwen", "gemma")
POPS = [("cc_threads", "Civil Comments"), ("cc_s5", "Civil Comments, 5\\% sample"), ("cad", "CAD"), ("qian_reddit", "Qian et al.'s Reddit"),
        ("qian_gab", "Qian et al.'s Gab"), ("cga", "CGA-Wiki"), ("bad", "BAD")]
PNAME = dict(POPS)
NICE = {"haiku": "Haiku", "sonnet": "Sonnet", "qwen7b_m": "Qwen", "qwen7b": "Qwen (own cut)", "gemma_m": "Gemma",
        "haiku_ctx": "Haiku, parent shown", "sonnet_ctx": "Sonnet, parent shown", "qwen7b_m_ctx": "Qwen, parent shown",
        "detoxify_original@0.6": "Detoxify (0.6)", "detoxify_original@m": "Detoxify", "detoxify_unbiased@0.6": "Detoxify-unbiased (0.6)",
        "detoxify_unbiased@m": "Detoxify-unbiased", "imsypp@argmax": "IMSYPP (argmax)", "imsypp@m": "IMSYPP", "sn_rob@m": "s-nlp RoBERTa",
        "sn_rob@0.5": "s-nlp RoBERTa (0.5)", "mh_dist@m": "martin-ha DistilBERT", "mh_dist@0.5": "martin-ha DistilBERT (0.5)",
        "fb_hate@m": "Dynabench hate", "fb_hate@0.5": "Dynabench hate (0.5)", "cn_off@m": "CardiffNLP offensive", "cn_off@0.5": "CardiffNLP offensive (0.5)"}
CLS14 = ["detoxify_original@0.6", "detoxify_original@m", "detoxify_unbiased@0.6", "detoxify_unbiased@m", "imsypp@argmax", "imsypp@m"] + \
        [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
S_BD, S_DEC, S_WI, S_AG, S_SE = ("results/x74_bounds/x74_bounds.json", "results/x74_decomp/x74_decomp.json", "results/x75_within/x75_within.json",
                                 "results/x79/x79_agree.json", "results/x76_select/x76_select.json")

# ======================================================================== E55: agreement bounds (Proposition 2)
section("E55 agreement bounds (Proposition 2)")
BD = load("x74_bounds/x74_bounds.json"); assert BD["cc_threads"]["N"] == 1251487
b = BD["cc_threads"]; H = b["labelers"]["haiku"]
k9, k5 = b["by_kappa"]["0.9"], b["by_kappa"]["0.5"]
put("xcBoundGoldDepth", 100 * b["gold"]["depth"], sg, S_BD, "cc_threads.gold.depth", "x100 (points per level)"); chk("xcBoundGoldDepth", "gDepthK")
for law, nm in (("depth", "Depth"), ("size", "Size"), ("article_size", "Art")):
    put(f"xcBoundKnine{nm}Lo", 100 * k9[law][0], sg, S_BD, f"cc_threads.by_kappa['0.9'].{law}[0]", "x100; sharp lower bound over all labelers with kappa 0.9 at gold's flag rate")
    put(f"xcBoundKnine{nm}Hi", 100 * k9[law][1], sg, S_BD, f"cc_threads.by_kappa['0.9'].{law}[1]", "x100; sharp upper bound, same counts")
    put(f"xcBoundKfive{nm}Lo", 100 * k5[law][0], sg, S_BD, f"cc_threads.by_kappa['0.5'].{law}[0]", "x100; kappa 0.5 at gold's flag rate")
    put(f"xcBoundKfive{nm}Hi", 100 * k5[law][1], sg, S_BD, f"cc_threads.by_kappa['0.5'].{law}[1]", "x100; kappa 0.5 at gold's flag rate")
for a_, o_ in (("xcBoundKnineDepthLo", "bKnineDepthLo"), ("xcBoundKnineDepthHi", "bKnineDepthHi"), ("xcBoundKfiveDepthLo", "bKfiveDepthLo"), ("xcBoundKfiveDepthHi", "bKfiveDepthHi")):
    chk(a_, o_)
v7f("xcBoundKnineDepthLo", "any depth law from $-$1.30 to +1.65 points per level")
put("xcBoundRatio", (k9["depth"][1] - k9["depth"][0]) / abs(b["gold"]["depth"]), lambda x: f"{x:.0f}", S_BD,
    "cc_threads.by_kappa['0.9'].depth, cc_threads.gold.depth", "width of the kappa-0.9 interval divided by |gold's depth law|, as x17_numbers_v4b.py"); chk("xcBoundRatio", "qBoundRatio")
put("xcBoundHaikuKappa", H["kappa"], f2, S_BD, "cc_threads.labelers.haiku.kappa", "Haiku's own counts (own cut)")
put("xcBoundHaikuFlag", H["flag_rate"], pct, S_BD, "cc_threads.labelers.haiku.flag_rate", "share of comments Haiku flags")
put("xcBoundHaikuDepth", 100 * H["depth"]["labeler"], sg, S_BD, "cc_threads.labelers.haiku.depth.labeler", "x100: Haiku's own depth law"); chk("xcBoundHaikuDepth", "wHaikuDepthPP")
put("xcBoundHaikuDepthLo", 100 * H["depth"]["lo"], sg, S_BD, "cc_threads.labelers.haiku.depth.lo", "x100: the lowest depth law of any labeler with Haiku's counts (n_TP, n_FP)")
put("xcBoundHaikuDepthHi", 100 * H["depth"]["hi"], sg, S_BD, "cc_threads.labelers.haiku.depth.hi", "x100: the highest, same counts")
put("xcBoundHaikuGoldDepthLo", 100 * H["depth"]["gold_lo"], sg, S_BD, "cc_threads.labelers.haiku.depth.gold_lo",
    "x100: dual form; Haiku's flags fixed, the lowest gold depth law of any gold labeling with Haiku's counts"); chk("xcBoundHaikuGoldDepthLo", "bHaikuGoldLo")
put("xcBoundHaikuGoldDepthHi", 100 * H["depth"]["gold_hi"], sg, S_BD, "cc_threads.labelers.haiku.depth.gold_hi", "x100: dual form, highest"); chk("xcBoundHaikuGoldDepthHi", "bHaikuGoldHi")
strad9 = [p for p, _ in POPS if BD[p]["by_kappa"]["0.9"]["depth"][0] < 0 < BD[p]["by_kappa"]["0.9"]["depth"][1]]
strad5 = [p for p, _ in POPS if BD[p]["by_kappa"]["0.5"]["depth"][0] < 0 < BD[p]["by_kappa"]["0.5"]["depth"][1]]
put("xcBoundNPops", len(POPS), i0, S_BD, "population keys", "number of populations")
put("xcBoundNSignNine", len(strad9), i0, S_BD, "<pop>.by_kappa['0.9'].depth", "populations whose kappa-0.9 depth-law interval contains 0 (the sign is not identified)",
    caveat="populations: " + ", ".join(strad9))
put("xcBoundNSignFive", len(strad5), i0, S_BD, "<pop>.by_kappa['0.5'].depth", "populations whose kappa-0.5 depth-law interval contains 0", caveat="populations: " + ", ".join(strad5))
LAWN = {"depth": "Depth", "size": "Thread size", "article_size": "Article size"}
rows = []
for pop, nm in POPS:
    bb = BD[pop]; Hh = bb["labelers"].get("haiku"); first = True
    for law in ("depth", "size", "article_size"):
        if law not in bb["gold"]:
            continue
        g9, g5 = bb["by_kappa"]["0.9"][law], bb["by_kappa"]["0.5"][law]
        hk = (f2(Hh["kappa"]) if first else "") if Hh else ("--" if first else "")
        hl = rng(100 * Hh[law]["lo"], 100 * Hh[law]["hi"]) if Hh else "--"
        hg = rng(100 * Hh[law]["gold_lo"], 100 * Hh[law]["gold_hi"]) if Hh else "--"
        rows.append(f"{nm if first else ''} & {LAWN[law]} & {sg(100 * bb['gold'][law])} & {rng(100 * g9[0], 100 * g9[1])} & {rng(100 * g5[0], 100 * g5[1])} & {hk} & {hl} & {hg} \\\\")
        first = False
    if pop != POPS[-1][0]:
        rows.append("\\addlinespace[2pt]")
table("bounds", rows,
      caption=("What overall agreement leaves open (Proposition 2). For each population and law: gold's law; the range of the law over all labelers "
               "that flag as many comments as gold and agree with it at kappa = 0.9 or 0.5; Haiku's kappa at its own cut; the range of the law over all "
               "labelers with Haiku's counts of true and false positives; and, with Haiku's flags fixed, the range of gold's law over every gold labeling "
               "with the same counts. Depth laws in points per level, size laws in points per doubling. Haiku is not shown where only Haiku shown the "
               "parent was run (BAD)."),
      headers=["Population", "Law", "Gold", "kappa = 0.9", "kappa = 0.5", "Haiku's kappa", "Haiku's counts: labeler's law", "Haiku's flags: gold's law"],
      src=f"{S_BD}: <pop>.gold.<law>, <pop>.by_kappa['0.9'|'0.5'].<law> = [lo, hi], <pop>.labelers.haiku.{{kappa, <law>.lo/hi, <law>.gold_lo/gold_hi}}",
      how="x100 for every law value; ranges printed 'lo to hi' with sg(., 2). The kappa-fixed ranges are at gold's flag rate (TPR = 1 - (1 - kappa)(1 - p), FPR = (1 - kappa) p, as x74_bounds.counts_for_kappa).",
      checks="Civil Comments depth row equals bKnineDepthLo/Hi, bKfiveDepthLo/Hi, bHaikuGoldLo/Hi and gDepthK (asserted through the macros); every population's kappa ranges match results/x74_bounds/x74_bounds.log (rounded there to 0.01 per mille).",
      caveats="Haiku's columns are filled for every population with a stored Haiku labeler without the parent (an extension of the requested Civil Comments row); "
              "BAD has only Haiku shown the parent, whose pooled flags mix two labeling modes, so its row has no Haiku columns. Article size laws exist on Civil Comments only.")
# arXiv revision (editor 4.13): BAD's thread-size range ends at the same value at kappa 0.9 and 0.5. That value is the largest size law of any labeling that
# flags gold's share (the sum of the K largest weights, x74_bounds' size weights); the smallest is reached at kappa 0.5. Most dialogues have 14 turns, so the
# weights take few values and the kappa constraint stops binding.
_Pb = pd.read_parquet(f"{RES}/x14_pops/bad.parquet", columns=["z", "unit"]); _zb = _Pb.z.values.astype(int); _Kb = int(_zb.sum())
_ub = pd.factorize(_Pb.unit.values)[0]; _kub = np.bincount(_ub).astype(float); _xub = np.log2(np.maximum(_kub, 1)); _okb = _kub >= 2
_wb = np.where(_okb[_ub], (_xub[_ub] - _xub[_okb].mean()) / (_kub[_ub] * ((_xub[_okb] - _xub[_okb].mean()) ** 2).sum()), 0.0)
_sb = np.sort(_wb); _gmax, _gmin = float(_sb[::-1][:_Kb].sum()), float(_sb[:_Kb].sum())
assert abs(_wb @ _zb - BD["bad"]["gold"]["size"]) < 1e-12
assert abs(_gmax - BD["bad"]["by_kappa"]["0.9"]["size"][1]) < 1e-9 and abs(_gmax - BD["bad"]["by_kappa"]["0.5"]["size"][1]) < 1e-9
assert abs(_gmin - BD["bad"]["by_kappa"]["0.5"]["size"][0]) < 1e-9 and BD["bad"]["by_kappa"]["0.9"]["size"][0] > _gmin + 1e-6
put("xcBoundBadDialogues", int(len(_kub)), comma, "results/x14_pops/bad.parquet", "unit", "BAD's dialogues")
put("xcBoundBadFourteen", int((_kub == 14).sum()), comma, "results/x14_pops/bad.parquet", "unit", "BAD's dialogues with 14 turns")
put("xcBoundBadSizeMax", 100 * _gmax, sg, "results/x14_pops/bad.parquet", "z, unit (x74_bounds size weights)",
    "x100: the largest thread size law of any labeling that flags gold's count, = the kappa-0.9 and kappa-0.5 upper bounds (asserted)")

# ======================================================================== E56: kappa against distortion, within populations
section("E56 kappa against distortion")
DEC = load("x74_decomp/x74_decomp.json"); WI = load("x75_within/x75_within.json")["within"]; AG = load("x79/x79_agree.json")
LAWS5 = ["depth", "size", "article_size", "log_rr", "depth_within"]


def dist_of(pop, L, law):
    if law == "depth_within":
        return (WI[pop]["labelers"].get(L) or {}).get("distortion", [None])[0]
    return (DEC[pop]["labelers"][L]["laws"].get(law) or {}).get("distortion", [None])[0]


def rho(pop, law, excl_ctx):
    ks, ds, names = [], [], []
    for L, r in DEC[pop]["labelers"].items():
        if L == "human_lenient" or (excl_ctx and CTX(L)):
            continue
        d = dist_of(pop, L, law)
        if d is None or not np.isfinite(d) or (law == "log_rr" and pop == "cga"):
            continue
        ks.append(r["kappa"][0]); ds.append(abs(d)); names.append(L)
    if len(ks) < 5:
        return None
    return dict(rho=float(spearmanr(ks, ds).correlation), n=len(ks), names=names)


for pop, _ in POPS:                                                                             # reproduce x79_agree.py exactly first
    for law in LAWS5:
        r_ = rho(pop, law, False)
        if law in AG[pop]:
            assert r_ and abs(r_["rho"] - AG[pop][law]["spearman"]) < 1e-12 and r_["n"] == AG[pop][law]["n"], (pop, law)
        else:
            assert r_ is None
RH = {pop: {law: rho(pop, law, True) for law in LAWS5} for pop, _ in POPS}
nx = {pop: sum(CTX(L) for L in DEC[pop]["labelers"]) for pop, _ in POPS}
six = [p for p, _ in POPS if p != "cad"]
for law, nm in (("depth", "Depth"), ("size", "Size"), ("depth_within", "Within")):
    v = [RH[p][law]["rho"] for p in six if RH[p][law]]
    put(f"xcRho{nm}Weakest", max(v), sg, S_DEC + " + " + S_WI, f"<pop>.labelers.<L>.kappa[0], |distortion| of {law}",
        f"Spearman rho(kappa, |distortion|) within each population other than CAD, over its LLM and classifier settings without the parent; the value closest to 0",
        caveat="the lenient human (K11) and every labeler shown the parent are left out")
    put(f"xcRho{nm}Strongest", min(v), sg, S_DEC + " + " + S_WI, "as above", "the most negative of the same six correlations")
    put(f"xcRhoCad{nm}", RH["cad"][law]["rho"], sg, S_DEC + " + " + S_WI, f"cad, {law}", "Spearman rho on CAD, same labelers")
    put(f"xcRhoNNeg{nm}", sum(x < 0 for x in v), i0, S_DEC, "as above", "number of the six non-CAD populations with a negative correlation")
vl = {p: RH[p]["log_rr"]["rho"] for p, _ in POPS if RH[p]["log_rr"]}
put("xcRhoLrrWeakest", max(vl.values()), sg, S_DEC, "<pop>.labelers.<L>.laws.log_rr.distortion[0]", "Spearman rho over the populations with a reply risk ratio (all but CGA-Wiki, CAD included, as x17); closest to 0")
put("xcRhoLrrStrongest", min(vl.values()), sg, S_DEC, "as above", "most negative")
put("xcRhoLrrNPops", len(vl), i0, S_DEC, "as above", "populations with a reply risk ratio")
put("xcRhoCadLrr", RH["cad"]["log_rr"]["rho"], sg, S_DEC, "cad", "CAD's log-RR correlation")
va = {p: RH[p]["article_size"]["rho"] for p, _ in POPS if RH[p]["article_size"]}
put("xcRhoArtCc", va["cc_threads"], sg, S_DEC, "cc_threads, article_size", "Spearman rho for the article size law (Civil Comments reply threads)")
put("xcRhoArtSfive", va["cc_s5"], sg, S_DEC, "cc_s5, article_size", "the same on the 5% sample")
WHY_RHO = ("the stored correlations (x79_agree.json, reproduced exactly here) include the pooled '+ parent' labelers "
           f"({', '.join(f'{p}: {nx[p]}' for p, _ in POPS)} per population); without them the correlation changes")
for a_, o_ in (("xcRhoDepthWeakest", "qRhoDepthMin"), ("xcRhoDepthStrongest", "qRhoDepthMax"), ("xcRhoSizeWeakest", "qRhoSizeMin"), ("xcRhoSizeStrongest", "qRhoSizeMax"),
               ("xcRhoWithinWeakest", "qRhoWithinMin"), ("xcRhoWithinStrongest", "qRhoWithinMax"), ("xcRhoLrrWeakest", "qRhoLrrMin"), ("xcRhoLrrStrongest", "qRhoLrrMax"),
               ("xcRhoCadDepth", "qRhoCadDepth"), ("xcRhoCadSize", "qRhoCadSize"), ("xcRhoCadWithin", "qRhoCadWithin"), ("xcRhoCadLrr", "qRhoCadLrr"), ("xcRhoLrrNPops", "qRhoLrrNPops")):
    chk(a_, o_, why=WHY_RHO)
v7f("xcRhoDepthWeakest", "rank correlations $-$0.59 to $-$0.91 on six of the seven populations", same=False,
    why="computed with the pooled '+ parent' labelers; see xcRhoDepthWeakest/Strongest for the values without them")
v7f("xcRhoCadDepth", "but not on CAD (+0.56)", same=(M["xcRhoCadDepth"] == "+0.56"), why="CAD's value without the pooled Haiku shown the parent")
MT = load("x92/x92_matched.json"); rows92 = {(r["model"], r["op"]): r for r in MT["rows"]}
sel = [("haiku", "rating at gold prevalence"), ("sonnet", "rating at gold prevalence")] + [(m_, "gold prevalence") for m_ in ("qwen", "gemma", "olmo2", "granite", "llama")]
for pop, tg in (("cc_threads", "Cc"), ("cc_s5", "Sfive"), ("cad", "CadAll"), ("qian_reddit", "Reddit"), ("qian_gab", "Gab"), ("cga", "Cga"), ("bad", "Bad")):
    if pop != "cad":
        put(f"xcRhoDepth{tg}", RH[pop]["depth"]["rho"], sg, S_DEC, f"{pop}, depth", f"Spearman rho(kappa, |depth-law distortion|) on {pop}, labelers without the parent (n = {RH[pop]['depth']['n']})")
put("xcRhoLlm", float(spearmanr([rows92[s]["kappa"] for s in sel], [rows92[s]["depth_excess"][0] for s in sel]).correlation), sg, "results/x92/x92_matched.json",
    "rows[(model, op)].kappa, .depth_excess[0]", "Spearman rho(kappa, depth-law distortion) over the seven LLMs at gold's flag rate on the 5% sample (Table 3), as x96_dist_v6.py")
chk("xcRhoLlm", "dLlmKapRho"); v7f("xcRhoLlm", "rank correlation +0.71")
FIGDATA = {}
for pop, _ in POPS:
    g_ = DEC[pop]["gold"]["depth"]
    pts = [(L, r["kappa"][0], 100 * r["laws"]["depth"]["distortion"][0], 100 * r["laws"]["depth"]["distortion"][1], 100 * r["laws"]["depth"]["distortion"][2], L.startswith(LLMP))
           for L, r in DEC[pop]["labelers"].items() if L != "human_lenient" and not CTX(L)]
    FIGDATA[pop] = dict(gold=100 * g_[0], pts=pts, rho=RH[pop]["depth"]["rho"], n=RH[pop]["depth"]["n"])

# ======================================================================== E57: the highest-kappa instrument in each population
section("E57 the highest-kappa instrument")
SE = load("x76_select/x76_select.json")
QF = 100 * DEC["cc_threads"]["labelers"]["qwen7b_m_ctx"]["flag_rate"][0]; QFs = f"{QF:.1f}%"                  # the abandoned no-context threshold's flag rate
BADK = f"{SE['bad']['all/top_kappa']['kappa']:.2f}"; assert SE["bad"]["all/top_kappa"]["instrument"] == "haiku_ctx"


def zof(pop, L, law):
    if law == "depth_within":
        r = WI[pop]["labelers"].get(L)
        return (r["distortion"][0], r["z"]) if r else (None, None)
    if law == "log_rr" and pop == "cga":
        return (None, None)
    r = DEC[pop]["labelers"][L]["laws"].get(law)
    return (r["distortion"][0], r["z"]) if r else (None, None)


def select(pop, excl):
    cand = {L: r for L, r in DEC[pop]["labelers"].items() if L != "human_lenient" and L not in excl}
    best = max(cand, key=lambda L: cand[L]["kappa"][0]); out = dict(instrument=best, kappa=cand[best]["kappa"][0], flag_rate=cand[best]["flag_rate"][0], laws={})
    for law in LAWS5:
        d, z = zof(pop, best, law)
        if d is None:
            continue
        allv = [abs(zof(pop, L, law)[0]) for L in cand if zof(pop, L, law)[0] is not None]
        out["laws"][law] = dict(distortion=d, z=z, rank=int(1 + sum(v < abs(d) for v in allv)), n=len(allv))
    return out


SEL = {}
for pop, _ in POPS:
    a_ = select(pop, set()); s_ = SE[pop]["all/top_kappa"]                                     # reproduce x76_select.py first
    assert a_["instrument"] == s_["instrument"] and all(abs(a_["laws"][l]["distortion"] - s_["laws"][l]["distortion"]) < 1e-15 and a_["laws"][l]["rank"] == s_["laws"][l]["rank"] for l in s_["laws"])
    SEL[pop] = select(pop, {L for L in DEC[pop]["labelers"] if CTX(L)})
    if not CTX(s_["instrument"]):
        assert SEL[pop]["instrument"] == s_["instrument"]
rebad = [p for p, _ in POPS if CTX(SE[p]["all/top_kappa"]["instrument"])]
LAWL = {"depth": "Depth law", "size": "Thread size law", "article_size": "Article size law", "log_rr": "Log reply RR", "depth_within": "Within-thread depth law"}
QOWN = {"cad": "abuse question", "qian_reddit": "hate-speech question", "qian_gab": "hate-speech question", "cga": "personal-attack question", "bad": "unsafe-turn question"}
def bk_name(pop, L):                                                                           # arXiv revision: the cut, and the question an LLM answered
    nm_ = NICE[L] + (" (gold's rate)" if L.endswith("@m") else "")
    return nm_ + (f", {QOWN[pop]}" if L.startswith(LLMP) and pop in QOWN else "")


rows = []; nz3 = 0; ne = 0; npz = 0; rk = []
for pop, nm in POPS:
    s_ = SEL[pop]; lw = s_["laws"]; worst = max(lw, key=lambda l: abs(lw[l]["z"]))
    n3 = sum(abs(v["z"]) > 3 for v in lw.values()); nz3 += n3; ne += len(lw); npz += n3 > 0; rk += [v["rank"] / v["n"] for v in lw.values()]
    dd = lw[worst]["distortion"] * (1 if worst == "log_rr" else 100)
    rows.append(f"{nm} & {bk_name(pop, s_['instrument'])} & {s_['kappa']:.2f} & {100 * s_['flag_rate']:.1f} & {LAWL[worst]} & {sg(dd)} & {abs(lw[worst]['z']):.1f} & {n3} of {len(lw)} \\\\")
table("bestkappa", rows,
      caption=("The labeler that agrees best with gold in each population, among the LLM and classifier settings without the parent: its kappa, its "
               "flag rate, the law it distorts most in standard-error units, that distortion (points per level for depth laws, points per doubling for "
               "size laws, log units for the risk ratio), its |z|, and how many of its laws it distorts by |z| > 3."),
      headers=["Population", "Highest-kappa labeler", "kappa", "Flags (%)", "Most distorted law", "Distortion", "|z|", "Laws with |z| > 3"],
      src=f"{S_DEC} (<pop>.labelers.<L>.kappa[0], flag_rate[0], laws.<law>.distortion[0], .z) and {S_WI} (<pop>.labelers.<L>.distortion[0], .z) for the within-thread law",
      how="the x76_select.py selection (highest kappa[0]) rerun with every labeler shown the parent removed from the candidates; z = distortion / paired bootstrap SD (stored); the most distorted law is the one with the largest |z|.",
      checks=f"the selection with all candidates reproduces {S_SE} 'all/top_kappa' exactly (instrument, distortions, ranks; asserted); it changes only where the stored choice was a pooled '+ parent' labeler: {', '.join(rebad)}.",
      caveats=f"On BAD the stored best-kappa instrument was Haiku shown the parent (kappa {BADK}), whose depth and within-thread laws pool roots labeled without the parent with replies labeled with it; it is replaced by the next highest kappa. "
              "Civil Comments' choice, Detoxify-unbiased, is trained in part on these raters' labels; without it the choice is s-nlp RoBERTa (macros xcSelCcNo...).")
put("xcSelNPops", len(POPS), i0, S_DEC, "populations", "number of populations")
put("xcSelNPopsZthree", npz, i0, S_DEC + " + " + S_WI, "re-selection", "populations whose highest-kappa labeler distorts at least one law by |z| > 3")
put("xcSelNEntries", ne, i0, S_DEC + " + " + S_WI, "re-selection", "law entries of the seven selected labelers")
put("xcSelNEntriesZthree", nz3, i0, S_DEC + " + " + S_WI, "re-selection", "of them, entries with |z| > 3")
put("xcSelRankShare", float(np.median(rk)), pct, S_DEC + " + " + S_WI, "re-selection", "median over the entries of rank/n, the rank of the selected labeler's |distortion| among the population's candidates (1 = smallest)")
WHY_SEL = "BAD's stored best-kappa instrument is the pooled Haiku shown the parent; the re-selection without labelers shown the parent picks " + NICE[SEL["bad"]["instrument"]]
for a_, o_ in (("xcSelNPops", "qSelNPops"), ("xcSelNPopsZthree", "qSelNPopsZthree"), ("xcSelNEntries", "qSelNEntries"), ("xcSelNEntriesZthree", "qSelNEntriesZthree"), ("xcSelRankShare", "qSelRankShare")):
    chk(a_, o_, why=WHY_SEL + "; ranks are also taken among candidates without the parent")
for pop, tag, laws_ in (("cga", "Cga", ("depth", "depth_within")), ("qian_gab", "Gab", ("depth", "size")), ("cc_threads", "Ccu", ("log_rr",))):
    s_ = SEL[pop]; put(f"xcSel{tag}Kappa", s_["kappa"], f2, S_DEC, f"{pop}.labelers.{s_['instrument']}.kappa[0]", f"kappa of the selected labeler ({s_['instrument']})")
    chk(f"xcSel{tag}Kappa", f"qSel{tag}Kappa")
    for law in laws_:
        ln = {"depth": "Depth", "depth_within": "Within", "size": "Size", "log_rr": "Lrr"}[law]; v = s_["laws"][law]
        put(f"xcSel{tag}{ln}Z", abs(v["z"]), (lambda x: f"{x:.1f}") if pop == "cga" else (lambda x: f"{x:.0f}"), S_DEC, f"{pop}.labelers.{s_['instrument']}.laws.{law}.z", "|z|")
        chk(f"xcSel{tag}{ln}Z", f"qSel{tag}{ln}Z")
        if pop != "qian_gab":
            put(f"xcSel{tag}{ln}", v["distortion"] * (1 if law == "log_rr" else 100), sg, S_DEC, f"{pop}.labelers.{s_['instrument']}.laws.{law}.distortion[0]",
                "x100 (points per level)" if law != "log_rr" else "log units")
            chk(f"xcSel{tag}{ln}", f"qSel{tag}{ln}" if law == "log_rr" else f"qSel{tag}{ln}PP")
sb = SEL["bad"]; wb = max(sb["laws"], key=lambda l: abs(sb["laws"][l]["z"]))
put("xcSelBadName", NICE[sb["instrument"]], None, S_DEC, f"bad.labelers.{sb['instrument']}", "the highest-kappa BAD labeler without the parent")
put("xcSelBadKappa", sb["kappa"], f2, S_DEC, f"bad.labelers.{sb['instrument']}.kappa[0]", "its kappa")
put("xcSelBadWorstZ", abs(sb["laws"][wb]["z"]), lambda x: f"{x:.0f}", S_DEC + " + " + S_WI, f"bad, {wb}", f"its largest |z| ({wb})")
for tg, L in (("Ccu", "detoxify_unbiased@m"), ("CcNo", "sn_rob@m")):
    put(f"xcSel{tg}RR", float(np.exp(DEC["cc_threads"]["labelers"][L]["laws"]["log_rr"]["law"][0])), f2, S_DEC, f"cc_threads.labelers['{L}'].laws.log_rr.law[0]", "exp: the reply risk ratio under this labeler")
chk("xcSelCcuRR", "tOneDetUnbRR"); chk("xcSelCcNoRR", "tOneSnRobRR"); v7f("xcSelCcuRR", "report reply risk ratios of 2.65 and 2.76")
put("xcSelTopKappa", SEL["cc_threads"]["kappa"], f2, S_DEC, "cc_threads.labelers['detoxify_unbiased@m'].kappa[0]", "the highest kappa on Civil Comments"); chk("xcSelTopKappa", "tOneTopKappa")
nd = select("cc_threads", {"detoxify_unbiased@0.6", "detoxify_unbiased@m"} | {L for L in DEC["cc_threads"]["labelers"] if CTX(L)})
assert nd["instrument"] == SE["cc_threads"]["no_detox_unbiased/top_kappa"]["instrument"]
put("xcSelCcNoKappa", nd["kappa"], f2, S_DEC, f"cc_threads.labelers.{nd['instrument']}.kappa[0]", "highest kappa on Civil Comments without Detoxify-unbiased"); chk("xcSelCcNoKappa", "qSelCcKappa")
put("xcSelCcNoLrr", nd["laws"]["log_rr"]["distortion"], sg, S_DEC, f"cc_threads.labelers.{nd['instrument']}.laws.log_rr.distortion[0]", "its log-RR distortion"); chk("xcSelCcNoLrr", "qSelCcLrr")
put("xcSelCcNoLrrZ", abs(nd["laws"]["log_rr"]["z"]), lambda x: f"{x:.0f}", S_DEC, "same, .z", "|z|"); chk("xcSelCcNoLrrZ", "qSelCcLrrZ")

# ======================================================================== E58: validate and correct
section("E58 validate and correct")
CO = load("x77_correct/x77_correct.json"); S_CO = "results/x77_correct/x77_correct.json"
LAW6 = ["depth", "depth_within", "size", "article_size", "log_rr"]


def corr_summary(excl):
    out = {}
    ent = [e for e in CO["entries"] if not (excl and CTX(e["lab"]))]
    for law in LAW6 + ["all"]:
        E_ = [e for e in ent if law == "all" or e["law"] == law]; cv = np.array([e["coverage"] for e in E_])
        ws, rl, ru = [], [], []
        for p_, P_ in CO["pops"].items():                                                    # as x17_numbers_v4b.py, by law
            for L_, r_ in P_.items():
                if excl and CTX(L_):
                    continue
                for law_, v_ in r_["laws"].items():
                    if law != "all" and law_ != law:
                        continue
                    if v_["corrected_pop_minus_gold"] is not None:
                        ws.append(abs(v_["corrected_pop_minus_gold"]) > abs(v_["uncorrected_minus_gold"]))
                    if v_.get("bias") is not None and v_["gold"] != 0:
                        rl.append(abs(v_["bias"]) / abs(v_["gold"])); ru.append(abs(v_["uncorrected_minus_gold"]) / abs(v_["gold"]))
        out[law] = dict(n=len(E_), below=float(np.mean(cv < 0.90)), med=float(np.median(cv)), n_ws=len(ws), worse=float(np.mean(ws)),
                        rel=float(np.median(rl)), unrel=float(np.median(ru)), n_rel=len(rl))
    byp = {p_: float(np.mean(np.array([e["coverage"] for e in ent if e["pop"] == p_]) >= 0.90)) for p_ in CO["pops"]}
    return out, byp


CA, CPA = corr_summary(False); CX, CPX = corr_summary(True)
assert CA["all"]["n"] == 523 and abs(CA["all"]["below"] - CO["H77c"]["share_below_90"]) < 1e-15 and CA["all"]["med"] == CO["H77c"]["median"]
for law in LAW6:
    assert abs(CA[law]["below"] - CO["H77c"][f"share_below_90_{law}"]) < 1e-12, law                # reproduces H77c by law
for nm, fn, raw_a, raw_x, old in (("N", i0, CA["all"]["n"], CX["all"]["n"], "qCorrN"), ("ShareBelow", pct, CA["all"]["below"], CX["all"]["below"], "qCorrShareBelow"),
                                  ("MedianCov", f2, CA["all"]["med"], CX["all"]["med"], "qCorrMedianCov"), ("ShareWorse", pct, CA["all"]["worse"], CX["all"]["worse"], "qCorrShareWorse"),
                                  ("MedRelBias", pct, CA["all"]["rel"], CX["all"]["rel"], "qCorrMedRelBias"), ("UncMedRelBias", pct, CA["all"]["unrel"], CX["all"]["unrel"], "qUncMedRelBias")):
    put(f"xcCorrAll{nm}", raw_a, fn, S_CO, "entries[].coverage; pops.<pop>.<lab>.laws.<law>", f"all {CA['all']['n']} stored entries, as x17_numbers_v4b.py (reproduction)",
        caveat="includes the pooled '+ parent' entries; for reference only")
    chk(f"xcCorrAll{nm}", old, scale=100 if fn is pct else 1)
    put(f"xcCorr{nm}", raw_x, fn, S_CO, "the same, without the labelers shown the parent", "the same summary over the entries of labelers without the parent")
    chk(f"xcCorr{nm}", old, scale=100 if fn is pct else 1, why=f"{CA['all']['n'] - CX['all']['n']} pooled '+ parent' entries are left out")
put("xcCorrMedianCovPct", CX["all"]["med"], pct, S_CO, "the same, without the labelers shown the parent", "median coverage of the corrected interval, as a percentage")   # review round 2: prose uses percentages
for law in LAW6:
    tg = {"depth": "Depth", "depth_within": "Within", "size": "Size", "article_size": "Art", "log_rr": "Lrr"}[law]
    put(f"xcCorrBelow{tg}", CX[law]["below"], pct, S_CO, f"entries[law={law}].coverage", "share of entries whose corrected 95% interval covers gold's law in fewer than 90% of 500 validations")
    put(f"xcCorrWorse{tg}", CX[law]["worse"], pct, S_CO, f"pops.*.*.laws.{law}.corrected_pop_minus_gold, uncorrected_minus_gold", "share of entries whose population-level corrected law is further from gold's than the uncorrected one")
put("xcCorrCgaOK", CPX["cga"], pct, S_CO, "entries[pop=cga].coverage", "share of CGA-Wiki's entries covering at least 0.90"); chk("xcCorrCgaOK", "qCorrCgaOK", scale=100, why="without CGA-Wiki's two labelers shown the parent")
put("xcCorrOtherOKMax", max(v for k, v in CPX.items() if k != "cga"), pct, S_CO, "entries[pop!=cga]", "the largest such share among the other populations")
chk("xcCorrOtherOKMax", "qCorrOtherOKMax", scale=100, why="without the labelers shown the parent")
hz = CO["pops"]["cc_threads"]["haiku"]["laws"]["depth"]
put("xcCorrHaikuDepth", 100 * (hz["gold"] + hz["bias"]), sg, S_CO, "pops.cc_threads.haiku.laws.depth.{gold, bias}", "x100: gold's law plus the mean bias of the corrected law = Haiku's corrected escalation")
chk("xcCorrHaikuDepth", "qCorrHaikuDepthPP"); v7f("xcCorrHaikuDepth", "Haiku's corrected escalation is +2.02 points per level")
rows = []
for law in LAW6 + ["all"]:
    c_ = CX[law]
    rows.append(("\\midrule\n" if law == "all" else "") + f"{LAWL.get(law, 'All laws')} & {c_['n']} & {pct(c_['below'])} & {f2(c_['med'])} & {pct(c_['worse'])} ({c_['n_ws']}) & {pct(c_['rel'])} & {pct(c_['unrel'])} \\\\")
table("correct", rows,
      caption=("Validating the labeler's error rates on 2,000 uniformly sampled gold labels and correcting the law (divided by TPR - FPR for linear laws; "
               "the matrix method for the reply risk ratio), over the labeler and population entries of the evidence, without labelers shown the parent: "
               "the share of entries whose corrected 95% interval covers gold's law in fewer than 90% of 500 simulated validations, the median coverage, the "
               "share whose corrected law (at the true error rates) lies further from gold's than the uncorrected one, and the median |bias| relative to "
               "|gold's law| after and before correction."),
      headers=["Law", "Entries", "Covering below 0.90", "Median coverage", "Moved away from gold (entries)", "|Bias| / |gold|, corrected", "|Bias| / |gold|, uncorrected"],
      src=f"{S_CO}: entries[] (pop, lab, law, coverage) and pops.<pop>.<lab>.laws.<law>.{{gold, bias, corrected_pop_minus_gold, uncorrected_minus_gold}}",
      how="shares and medians over entries; the 'moved away' share counts entries with a defined population-level correction (an undefined matrix correction, a negative cell share, is excluded); relative bias over entries with gold's law nonzero and a defined bias.",
      checks=f"with all {CA['all']['n']} entries the script reproduces H77c's shares by law and qCorrN, qCorrShareBelow, qCorrMedianCov, qCorrShareWorse, qCorrMedRelBias, qUncMedRelBias (asserted)",
      caveats=f"{CA['all']['n'] - CX['all']['n']} pooled '+ parent' entries are left out. For the log RR, undefined corrections count as not covering (as stored); the 'moved away' share for log RR uses only the {CX['log_rr']['n_ws']} defined corrections.")

# ======================================================================== E59: census of the law estimates, with the single-rater reference
section("E59 census of law estimates")
MG = load("x80/x80_magnitude.json"); DS = load("x82/x82_desc.json")["B82c"]; HRf = load("x85/x85_human_ref.json"); P1 = HRf["part1"]
C75 = load("x75_decomp/x75_decomp.json"); S_MG = "results/x80/x80_magnitude.json"; S_HR = "results/x85/x85_human_ref.json"


def census_rows(excl):
    rows_ = []
    for pop, Rr in DEC.items():
        g = Rr["gold"]
        for L, r in Rr["labelers"].items():
            if L == "human_lenient" or (excl and CTX(L)):
                continue
            for law, v in r["laws"].items():
                if law not in g:
                    continue
                if law == "log_rr":
                    pm = C75.get(pop, {}).get("labelers", {}).get(L)
                    terms = (pm["uniform"], pm["fp"], pm["tp"]) if pm else None
                else:
                    terms = (v["attenuation_term"][0] - v["gold"][0], v["fp_term"][0], v["tp_term"][0])
                rows_.append(dict(pop=pop, labeler=L, law=law, kappa=r["kappa"][0], gold=g[law][0], gold_lo=g[law][1], gold_hi=g[law][2], est=v["law"][0], lo=v["law"][1],
                                  hi=v["law"][2], z=v["z"], terms=terms))
            w = WI[pop]["labelers"].get(L)
            if w:
                gw = WI[pop]["gold_within"]
                rows_.append(dict(pop=pop, labeler=L, law="depth_within", kappa=r["kappa"][0], gold=gw[0], gold_lo=gw[1], gold_hi=gw[2], est=w["law"][0], lo=w["law"][1], hi=w["law"][2],
                                  z=w["z"] or 0.0, terms=(w["attenuation_term"][0] - gw[0], w["fp_term"][0], w["tp_term"][0])))
    for r in rows_:
        r["rel"] = (r["est"] - r["gold"]) / abs(r["gold"]) if r["gold"] != 0 else None
        si = r["lo"] > 0 or r["hi"] < 0; sgd = r["gold_lo"] > 0 or r["gold_hi"] < 0
        r.update(sign_flip=bool(r["est"] * r["gold"] < 0 and si), manuf=bool(si and not sgd), lost=bool(sgd and not si), gold_sig=sgd,
                 family="llm" if r["labeler"].startswith(LLMP) else "classifier")
        t = r["terms"]; r["umass"] = (abs(t[0]) / (abs(t[0]) + abs(t[1]) + abs(t[2]))) if t and (abs(t[0]) + abs(t[1]) + abs(t[2])) > 0 else None
    return rows_


def csumm(R_):
    rel = np.array([abs(r["rel"]) for r in R_ if r["rel"] is not None]); um = [r["umass"] for r in R_ if r["umass"] is not None]
    return dict(n=len(R_), share_rel_gt_25=float(np.mean(rel > 0.25)), median_abs_rel=float(np.median(rel)), n_sign_flip=int(sum(r["sign_flip"] for r in R_)),
                n_sign_flip_gold_sig=int(sum(r["sign_flip"] and r["gold_sig"] for r in R_)), n_manufactured_sig=int(sum(r["manuf"] for r in R_)),
                n_lost_sig=int(sum(r["lost"] for r in R_)), umass=float(np.median(um)) if um else None, n_um=len(um))


RA = census_rows(False); RX = census_rows(True)
for key_, sub in (("all", RA), ("llm", [r for r in RA if r["family"] == "llm"]), ("classifier", [r for r in RA if r["family"] == "classifier"])):
    ref = MG["all"] if key_ == "all" else MG["by_family"][key_]; mine = csumm(sub)                 # reproduce x80_magnitude.py
    for k_ in ("n", "share_rel_gt_25", "median_abs_rel", "n_sign_flip", "n_sign_flip_gold_sig", "n_manufactured_sig", "n_lost_sig"):
        assert abs(mine[k_] - ref[k_]) < 1e-12, (key_, k_, mine[k_], ref[k_])
hkA = csumm([r for r in RA if r["kappa"] >= 0.6]); dk = DS["kappa_ge_06"]                         # reproduce B82c (kappa >= 0.6)
assert hkA["n"] == dk["n"] and abs(hkA["share_rel_gt_25"] - dk["share_rel_gt_25"]) < 1e-12 and hkA["n_sign_flip"] == dk["n_flip"] and hkA["n_manufactured_sig"] == dk["n_manuf"] and hkA["n_lost_sig"] == dk["n_lost"]
for L, mass in P1["instrument_uniform_share"]["depth"]["by_inst"].items():                      # reproduce x85_human_ref.py's uniform shares
    mine = [r["umass"] for r in RA if r["pop"] == "cc_threads" and r["labeler"] == L and r["law"] == "depth"][0]
    assert abs(mine - mass) < 1e-12, L
for L, mass in P1["instrument_uniform_share"]["log_rr"]["by_inst"].items():
    mine = [r["umass"] for r in RA if r["pop"] == "cc_threads" and r["labeler"] == L and r["law"] == "log_rr"][0]
    assert abs(mine - mass) < 1e-12, L
FAM = {"all": RX, "llm": [r for r in RX if r["family"] == "llm"], "classifier": [r for r in RX if r["family"] == "classifier"], "hik": [r for r in RX if r["kappa"] >= 0.6]}
CS = {k: csumm(v) for k, v in FAM.items()}
ORX = load("x75_onerater/x75_onerater_decomp.json"); S_OR = "results/x75_onerater/x75_onerater_decomp.json"
srm = {law: [abs(D_[law]["uniform"]) / (abs(D_[law]["uniform"]) + abs(D_[law]["fp"]) + abs(D_[law]["tp"])) for D_ in ORX["draws"]] for law in ("depth", "size", "log_rr")}
for law in srm:                                                                                 # x85's per-law medians
    assert abs(float(np.median(srm[law])) - P1["single_rater_uniform_share"][law]["median_mass"]) < 1e-12
sr1 = P1["single_rater"]; sr_um = float(np.median([x for v in srm.values() for x in v]))
FAMN = {"all": "All labelers", "llm": "LLMs", "classifier": "Classifiers", "hik": "$\\kappa \\ge 0.6$"}
rows = []
for k, nm in FAMN.items():
    c_ = CS[k]
    rows.append(f"{nm} & {c_['n']} & {pct(c_['share_rel_gt_25'])} & {pct(c_['median_abs_rel'])} & {c_['n_sign_flip']} ({c_['n_sign_flip_gold_sig']}) & {c_['n_manufactured_sig']} & {c_['n_lost_sig']} & {pct(c_['umass'])} \\\\")
rows.append("\\midrule")
rows.append(f"One rater & {sr1['n']} & {pct(sr1['share_rel_gt_25'])} & {pct(sr1['median_rel'])} & -- & -- & -- & {pct(sr_um)} \\\\")
table("census", rows,
      caption=("How far each law estimate falls from gold's, over every labeler setting, population and law of the evidence (depth, within-thread "
               "depth, thread size, article size, log reply risk ratio), without labelers shown the parent: the number of estimates; the share off by "
               "more than a quarter of gold's law and the median |distortion| relative to |gold's law|; estimates of the opposite sign with the "
               "labeler's 95% CI excluding 0 (in parentheses: gold's CI also excludes 0); significance where gold's law is not significant "
               "(manufactured) and the reverse (lost); and the median share of the distortion's three terms (Proposition 1) that is unstructured "
               "error. Single rater: one random rater against the majority of the others, 20 draws, Civil Comments reply threads only."),
      headers=["Labelers", "Estimates", "Off by > 25%", "Median |distortion| / |gold|", "Opposite sign", "Significance manufactured", "Significance lost", "Uniform share"],
      src=f"{S_DEC} (laws, CIs, terms, kappa), {S_WI} (within-thread laws and terms), results/x75_decomp/x75_decomp.json (log-RR permutation terms), {S_HR} part1.single_rater, {S_OR} (single-rater terms)",
      how="the x80_magnitude.py rows rebuilt without the labelers shown the parent; uniform share of an entry = |U| / (|U| + |S_FP| + |S_TP|) with U = (TPR - FPR - 1) L(Z), as x85_human_ref.py, median over the entries that have the three terms (the log-RR terms exist for every population with a risk ratio).",
      checks=f"with the pooled '+ parent' entries the script reproduces x80_magnitude.json all/by_family (n {MG['all']['n']}), B82c's kappa >= 0.6 summary (n {DS['kappa_ge_06']['n']}) and x85's per-instrument uniform shares (asserted)",
      caveats="The single rater's estimates are judged against gold B (the others' majority), whose intervals are not stored, so its sign, manufactured and lost columns are not computable ('--'); its 80 estimates are 20 draws x 4 laws (depth, thread size, article size, log RR) and its uniform share pools depth, thread size and log RR (60 values). "
              "kappa >= 0.6 includes LLM settings on Gab, Reddit and CGA-Wiki.")
for k, tg in (("all", ""), ("llm", "LLM"), ("classifier", "Cls"), ("hik", "HiKap")):
    c_ = CS[k]
    put(f"xcMag{tg}N", c_["n"], i0, S_DEC, "census rows", f"number of estimates ({FAMN[k]}), without labelers shown the parent")
    put(f"xcMag{tg}Share", c_["share_rel_gt_25"], pct, S_DEC, "census rows", "share off by more than 25% of |gold's law|")
    put(f"xcMag{tg}MedRel", c_["median_abs_rel"], pct, S_DEC, "census rows", "median |distortion| / |gold's law|")
    put(f"xcMag{tg}Flip", c_["n_sign_flip"], i0, S_DEC, "census rows", "opposite sign, labeler's CI excluding 0")
    put(f"xcMag{tg}FlipSig", c_["n_sign_flip_gold_sig"], i0, S_DEC, "census rows", "of them, gold's CI also excluding 0")
    put(f"xcMag{tg}Manuf", c_["n_manufactured_sig"], i0, S_DEC, "census rows", "significance manufactured")
    put(f"xcMag{tg}Lost", c_["n_lost_sig"], i0, S_DEC, "census rows", "significance lost")
    put(f"xcMag{tg}Unif", c_["umass"], pct, S_DEC, "census rows", "median uniform share of the distortion")
WHY_MAG = "the stored census counts the pooled '+ parent' entries"
for a_, o_ in (("xcMagN", "qMagN"), ("xcMagShare", "qMagShare"), ("xcMagMedRel", "qMagMedRel"), ("xcMagFlip", "qMagFlip"), ("xcMagFlipSig", "qMagFlipSig"), ("xcMagManuf", "qMagManuf"),
               ("xcMagLost", "qMagLost"), ("xcMagLLMShare", "qMagLLMShare"), ("xcMagClsShare", "qMagClsShare"), ("xcMagHiKapN", "qMagHiN"), ("xcMagHiKapShare", "qMagHiShare"),
               ("xcMagHiKapMedRel", "qMagHiMed"), ("xcMagHiKapFlip", "qMagHiFlip"), ("xcMagHiKapManuf", "qMagHiManuf")):
    chk(a_, o_, scale=100 if a_.endswith(("Share", "MedRel")) else 1, why=WHY_MAG)
hkx = [r for r in RX if r["kappa"] >= 0.6]
gse = lambda r: (r["gold_hi"] - r["gold_lo"]) / (2 * 1.96)
put("xcMagHiKapSE", float(np.median([abs(r["est"] - r["gold"]) / gse(r) for r in hkx])), lambda x: f"{x:.1f}", S_DEC, "census rows, kappa >= 0.6",
    "median |distortion| in gold standard errors, SE = (gold's CI width)/3.92, as B82c"); chk("xcMagHiKapSE", "qMagHiSE", why=WHY_MAG)
put("xcSrOff", sr1["share_rel_gt_25"], pct, S_HR, "part1.single_rater.share_rel_gt_25", "share of the single rater's 80 estimates off by more than 25% of gold B's law"); chk("xcSrOff", "qHrSrOff", scale=100)
put("xcSrOffMed", sr1["median_rel"], pct, S_HR, "part1.single_rater.median_rel", "median relative distortion"); chk("xcSrOffMed", "qHrSrOffMed", scale=100)
put("xcSrN", sr1["n"], i0, S_HR, "part1.single_rater.n", "estimates (20 draws x 4 laws)")
for law, tg in (("depth", "Depth"), ("log_rr", "Lrr"), ("size", "Size")):
    put(f"xcSrUnif{tg}", float(np.median(srm[law])), pct, S_OR, f"draws[].{law}.{{uniform, fp, tp}}", "median over the 20 draws of the single rater's uniform share")
chk("xcSrUnifDepth", "qHrSrUnifDepth", scale=100); chk("xcSrUnifLrr", "qHrSrUnifRR", scale=100)
put("xcSrUnif", sr_um, pct, S_OR, "draws[].{depth,size,log_rr}", "median over the 60 values (3 laws x 20 draws)")
llm_cc_depth = [r["umass"] for r in RX if r["pop"] == "cc_threads" and r["law"] == "depth" and r["family"] == "llm"]
put("xcLlmUnifDepthCc", float(np.median(llm_cc_depth)), pct, S_DEC, "cc_threads LLMs without the parent: haiku, qwen7b_m, qwen7b", "median uniform share of the depth-law distortion")
chk("xcLlmUnifDepthCc", "qHrLLMUnifDepth", scale=100, why="x85 takes the median over five LLM settings including the pooled Haiku and Qwen shown the parent (3% each); without them the median is Qwen at its own answer's")
llm_depth = [r["umass"] for r in RX if r["law"] == "depth" and r["family"] == "llm"]
put("xcLlmUnifDepth", float(np.median(llm_depth)), pct, S_DEC, "all populations, LLM depth entries without the parent", "median uniform share of the LLMs' depth-law distortions (all populations)")
best = []
for pop, _ in POPS:
    s_ = SEL[pop]
    for law, e in s_["laws"].items():
        if abs(e["z"]) > 3:
            best += [r for r in RX if r["pop"] == pop and r["labeler"] == s_["instrument"] and r["law"] == law]
put("xcBestN", len(best), i0, S_DEC, "re-selected best-kappa labelers", "their entries with |z| > 3"); chk("xcBestN", "qBestN", why=WHY_SEL)
put("xcBestRel", sum(abs(r["rel"]) > 0.25 for r in best if r["rel"] is not None), i0, S_DEC, "same", "of them, off by more than 25%"); chk("xcBestRel", "qBestRel", why=WHY_SEL)
put("xcBestFlip", sum(r["sign_flip"] for r in best), i0, S_DEC, "same", "of them, opposite sign"); chk("xcBestFlip", "qBestFlip", why=WHY_SEL)

# ======================================================================== E60: decisions at 10,000-100,000 comments
section("E60 decisions at sample sizes")
DCS = load("x86/x86_decisions.json"); P2 = HRf["part2"]; S_DC = "results/x86/x86_decisions.json"
SZ = [("10000", DCS["sizes"]["10000"], S_DC + " sizes['10000']"), ("20000", P2, S_HR + " part2"), ("50000", DCS["sizes"]["50000"], S_DC + " sizes['50000']"),
      ("100000", DCS["sizes"]["100000"], S_DC + " sizes['100000']")]
HK = sorted(k for k, v in DCS["kappa"].items() if v >= 0.6 and not k.startswith(LLMP) and not CTX(k))
assert HK == sorted(P2["pooled"]["kappa_ge_06"]["keys"]) and all(not k.startswith(LLMP) for k in HK)
DEC_T = {}
for s, S, src in SZ:
    for law in ("depth", "size", "log_rr"):
        h_ = S["instruments"]["haiku"][law]; hd = float(np.mean([S["instruments"][k][law]["differ"] for k in HK])); hr = float(np.mean([S["instruments"][k][law]["reversal"] for k in HK]))
        pk = S["pooled"]["kappa_ge_06"][law]; pk = pk["differ"] if isinstance(pk, dict) else pk
        assert abs(hd - pk) < 1e-12, (s, law)                                                   # the pooled kappa >= 0.6 value is the mean over these settings
        sr_ = S["single_rater_vs_goldB"][law]; gv = S["gold_verdicts"][law]
        DEC_T[(s, law)] = dict(h=h_["differ"], hr=h_["reversal"], k=hd, kr=hr, s=sr_["differ"], sr=sr_["reversal"], g=1 - gv["0"], hci=h_.get("ci_excludes_gold"),
                               kci=float(np.mean([S["instruments"][k][law]["ci_excludes_gold"] for k in HK])) if "ci_excludes_gold" in h_ else None, sci=sr_.get("ci_excludes_gold"))
rows = []
for law in ("depth", "size", "log_rr"):
    rows.append(f"{LAWL[law]} & Haiku & " + " & ".join(f"{pct(DEC_T[(s, law)]['h'])} ({pct(DEC_T[(s, law)]['hr'])})" for s, _, _ in SZ) + " \\\\")
    rows.append(f" & $\\kappa \\ge 0.6$ classifiers & " + " & ".join(f"{pct(DEC_T[(s, law)]['k'])} ({pct(DEC_T[(s, law)]['kr'])})" for s, _, _ in SZ) + " \\\\")
    rows.append(f" & One rater & " + " & ".join(f"{pct(DEC_T[(s, law)]['s'])} ({pct(DEC_T[(s, law)]['sr'])})" for s, _, _ in SZ) + " \\\\")
    rows.append(f" & Gold significant & " + " & ".join(pct(DEC_T[(s, law)]["g"]) for s, _, _ in SZ) + " \\\\")
    if law != "log_rr":
        rows.append("\\addlinespace[2pt]")
    else:                                                                                       # the verdicts agree; what differs is the estimate
        for who, key_ in (("Haiku: CI excludes gold's estimate", "hci"), ("$\\kappa \\ge 0.6$ classifiers: CI excludes gold's estimate", "kci"), ("One rater: CI excludes gold B's estimate", "sci")):
            rows.append(f" & {who} & " + " & ".join("--" if DEC_T[(s, law)][key_] is None else pct(DEC_T[(s, law)][key_]) for s, _, _ in SZ) + " \\\\")
table("decisions", rows,
      caption=("Conclusions on samples of Civil Comments reply threads (whole articles, 200 samples per size): the share of samples in which a labeler's "
               "verdict on a law (a significant rise, fall or neither; article-clustered Wald test) differs from gold's on the same sample, with the share "
               "in which it reverses gold's in parentheses, and the share of samples in which gold's own law is significant; for the log risk ratio, also the share of samples in which the labeler's 95% CI excludes gold's estimate (not stored at 20,000). kappa >= 0.6 classifiers: "
               "the mean over Detoxify-unbiased and s-nlp RoBERTa at both thresholds. Single rater: one random rater against the majority of the others."),
      headers=["Law", "Labeler", "10,000 comments", "20,000", "50,000", "100,000"],
      src=f"{S_DC} sizes.<10000|50000|100000>.{{instruments.<L>.<law>.{{differ, reversal}}, single_rater_vs_goldB, gold_verdicts, pooled.kappa_ge_06}}; {S_HR} part2 (20,000)",
      how="shares as stored; the kappa >= 0.6 cell is the mean over the four settings (equal to the stored pooled value, asserted), its reversal the mean of their reversals; gold significant = 1 - share of samples with gold's verdict 'neither'.",
      checks="Haiku depth 70%-100% and reversal up to 48%, single rater 15%-37%: qDcHaikuMin/Max, qDcHaikuRevMax, qDcSrMin/Max (asserted through the macros); 20,000 column: qHr* macros",
      caveats="Haiku shown the parent is not shown: the stored decision analysis uses its pooled labels (roots without the parent, replies with it) over all comments, not among replies. The single rater's verdicts are compared with gold B, the majority of the other raters, not with gold; gold B's own significance shares are stored only at 20,000.")
for law, tg in (("depth", "Depth"), ("size", "Size"), ("log_rr", "Lrr")):
    for s, _, src in SZ:
        st = {"10000": "Ten", "20000": "Twenty", "50000": "Fifty", "100000": "Hundred"}[s]; v = DEC_T[(s, law)]
        put(f"xcDc{tg}Haiku{st}", v["h"], pct, src, f"instruments.haiku.{law}.differ", f"share of {s}-comment samples whose Haiku verdict differs from gold's")
        put(f"xcDc{tg}HaikuRev{st}", v["hr"], pct, src, f"instruments.haiku.{law}.reversal", "share in which Haiku reverses gold's verdict")
        put(f"xcDc{tg}HiKap{st}", v["k"], pct, src, f"mean over {', '.join(HK)} of instruments.<L>.{law}.differ", "kappa >= 0.6 classifiers")
        put(f"xcDc{tg}Sr{st}", v["s"], pct, src, f"single_rater_vs_goldB.{law}.differ", "single rater against gold B")
        put(f"xcDc{tg}Gold{st}", v["g"], pct, src, f"1 - gold_verdicts.{law}['0']", "share of samples in which gold's law is significant")
hd_ = [DEC_T[(s, "depth")]["h"] for s, _, _ in SZ]; sd_ = [DEC_T[(s, "depth")]["s"] for s, _, _ in SZ]
put("xcDcHaikuMin", min(hd_), pct, S_DC + " + " + S_HR, "instruments.haiku.depth.differ over the four sizes", "smallest share"); chk("xcDcHaikuMin", "qDcHaikuMin", scale=100)
put("xcDcHaikuMax", max(hd_), pct, S_DC + " + " + S_HR, "same", "largest share"); chk("xcDcHaikuMax", "qDcHaikuMax", scale=100)
put("xcDcHaikuRevMax", max(DEC_T[(s, "depth")]["hr"] for s, _, _ in SZ), pct, S_DC + " + " + S_HR, "instruments.haiku.depth.reversal", "largest reversal share"); chk("xcDcHaikuRevMax", "qDcHaikuRevMax", scale=100)
put("xcDcSrMin", min(sd_), pct, S_DC + " + " + S_HR, "single_rater_vs_goldB.depth.differ", "smallest share"); chk("xcDcSrMin", "qDcSrMin", scale=100)
put("xcDcSrMax", max(sd_), pct, S_DC + " + " + S_HR, "same", "largest share"); chk("xcDcSrMax", "qDcSrMax", scale=100)
v7f("xcDcSrMax", "a single rater's verdict differs in 15\\%--37\\% and never reverses it")
assert all(DEC_T[(s, l)]["sr"] == 0 for s, _, _ in SZ for l in ("depth", "size", "log_rr"))
for a_, o_ in (("xcDcDepthHaikuTwenty", "qHrHaikuDepth"), ("xcDcDepthHaikuRevTwenty", "qHrHaikuRev"), ("xcDcSizeHaikuTwenty", "qHrHaikuSize"), ("xcDcDepthSrTwenty", "qHrSrDepth"),
               ("xcDcSizeSrTwenty", "qHrSrSize"), ("xcDcDepthHiKapTwenty", "qHrHiKapDepth"), ("xcDcSizeHiKapTwenty", "qHrHiKapSize")):
    chk(a_, o_, scale=100)
put("xcDcHiKapN", len(HK), i0, S_DC, "kappa", "number of kappa >= 0.6 classifier settings on cc_threads"); chk("xcDcHiKapN", "qHrHiKapN")
for who, key_, tg in (("Haiku", "hci", "Haiku"), ("kappa >= 0.6", "kci", "HiKap"), ("single rater", "sci", "Sr")):
    v = [DEC_T[(s, "log_rr")][key_] for s, _, _ in SZ if s != "20000"]
    put(f"xcDcLrrCi{tg}Min", min(v), pct, S_DC, "sizes.<10000|50000|100000>....log_rr.ci_excludes_gold", f"smallest share of samples in which the {who} 95% CI excludes gold's estimate (log RR)")
    put(f"xcDcLrrCi{tg}Max", max(v), pct, S_DC, "same", "largest share", caveat="not stored at 20,000 comments")
gd20 = P2["gold_verdicts"]
put("xcDcGoldDepthNegTwenty", gd20["depth"]["-1"], pct, S_HR, "part2.gold_verdicts.depth['-1']", "share of 20,000-comment samples where gold's depth law is significantly negative"); chk("xcDcGoldDepthNegTwenty", "qHrGoldDepthNeg", scale=100)
put("xcDcGoldSizePosTwenty", gd20["size"]["1"], pct, S_HR, "part2.gold_verdicts.size['1']", "significantly positive thread size law"); chk("xcDcGoldSizePosTwenty", "qHrGoldSizePos", scale=100)

# ======================================================================== E61: comparisons between publications
section("E61 publication comparisons")
PB = load("x78/x78_pubs.json"); Mt = PB["metrics"]; S_PB = "results/x78/x78_pubs.json"
put("xcPubN", PB["n_pubs"], i0, S_PB, "n_pubs", "publications with >= 1,000 comments"); chk("xcPubN", "qPubN")
put("xcPubPairs", PB["n_pairs"], i0, S_PB, "n_pairs", "pairs of publications")
for met, tg in (("item", "Item"), ("thread", "Thr"), ("article", "Art")):
    put(f"xcPub{tg}Dec", Mt[met]["n_decided"], i0, S_PB, f"metrics.{met}.n_decided", "pairs whose gold difference has a 95% article-bootstrap CI excluding 0")
    chk(f"xcPub{tg}Dec", f"qPub{tg}Dec")
    put(f"xcPubHaiku{tg}", Mt[met]["reversed"]["haiku"], i0, S_PB, f"metrics.{met}.reversed.haiku", "decided pairs whose order Haiku's point difference reverses")
    chk(f"xcPubHaiku{tg}", f"qPubHaiku{tg}")
    put(f"xcPubFb{tg}", Mt[met]["reversed"]["fb_hate@m"], i0, S_PB, f"metrics.{met}.reversed['fb_hate@m']", "the same for the Dynabench hate classifier at gold's flag rate")
    if tg != "Art":
        chk(f"xcPubFb{tg}", f"qPubFb{tg}")
    cl_ = {k: v for k, v in Mt[met]["reversed"].items() if not k.startswith(LLMP)}
    put(f"xcPubCls{tg}Max", max(cl_.values()), i0, S_PB, f"metrics.{met}.reversed over the 14 classifier settings", "largest number of reversals by a classifier setting")
zero = [k for k in CLS14 if all(Mt[m]["reversed"][k] == 0 for m in ("item", "thread", "article"))]
put("xcPubNZero", len(zero), i0, S_PB, "metrics.*.reversed", "classifier settings that reverse no decided pair on any metric", caveat="settings: " + ", ".join(zero))
put("xcPubNoneNames", " and ".join(sorted({NICE[k.split("@")[0] + "@m"] for k in zero})), None, S_PB, "same", "their classifiers (both thresholds)")
SKIP.append("E61: Haiku and Qwen shown the parent (qPubHaikuCtxItem = 22) are not reported: their publication rates pool roots labeled without the parent with replies labeled with it.")

# ======================================================================== E62: unit shares
section("E62 unit shares")
UN = load("x77_units/x77_units.json"); MS = load("x83/x83_matched_size.json")
put("xcUnitGold", UN["cc_threads"]["all"]["gold"], f2, "results/x77_units/x77_units.json", "cc_threads.all.gold", "share of reply threads with at least one gold-toxic comment"); chk("xcUnitGold", "qUnitGold")
put("xcUnitHaiku", UN["cc_threads"]["all"]["labelers"]["haiku"]["share"], f2, "results/x77_units/x77_units.json", "cc_threads.all.labelers.haiku.share", "the same under Haiku's flags"); chk("xcUnitHaiku", "qUnitHaiku")
put("xcUnitHaikuHum", MS["cc_threads"]["instruments"]["haiku"]["unit_share_human"][0], f2, "results/x83/x83_matched_size.json", "cc_threads.instruments.haiku.unit_share_human[0]",
    "the same under the human vote-fraction instrument flagging as many comments as Haiku"); chk("xcUnitHaikuHum", "qMsHaikuUnitHum")
assert abs(MS["cc_threads"]["gold"]["unit_share"][0] - UN["cc_threads"]["all"]["gold"]) < 1e-12

# ======================================================================== E63: Frechet bounds (Proposition 3)
section("E63 Frechet bounds")
FR = load("x78/x78_frechet.json"); S_FR = "results/x78/x78_frechet.json"
for unit, tg, old in (("unit", "Thr", "qFrThr"), ("article", "Art", "qFrArt")):
    h_ = FR[unit]["instruments"]["haiku"]
    for k_, nm in (("lo", "Lo"), ("hi", "Hi"), ("independence", "Ind")):
        put(f"xcFr{tg}{nm}", h_[k_], f2, S_FR, f"{unit}.instruments.haiku.{k_}", f"{'thread' if unit == 'unit' else 'article-section'} share with a toxic comment under calibrated Haiku levels: {k_}")
        chk(f"xcFr{tg}{nm}", f"{old}{nm}")
    put(f"xcFr{tg}Gold", FR[unit]["gold"], f2, S_FR, f"{unit}.gold", "gold's share"); chk(f"xcFr{tg}Gold", f"{old}Gold")
sn_ = FR["unit"]["instruments"]["sn_rob@m"]
put("xcFrThrSnLo", sn_["lo"], f2, S_FR, "unit.instruments['sn_rob@m'].lo", "s-nlp RoBERTa's calibrated flags: lower bound"); chk("xcFrThrSnLo", "qFrThrSnLo")
put("xcFrThrSnHi", sn_["hi"], f2, S_FR, "unit.instruments['sn_rob@m'].hi", "upper bound"); chk("xcFrThrSnHi", "qFrThrSnHi")
# the reply-association region of f14_regions.py, rerun on the current population (deterministic; seconds)
Pq = pd.read_parquet(f"{RES}/x14_pops/cc_threads.parquet", columns=["z", "s:haiku", "parent_row"]); assert len(Pq) == 1251487
zq = Pq.z.values.astype(float); lev = Pq["s:haiku"].values.astype(float); assert not np.isnan(lev).any()
keys_, inv_ = np.unique(lev, return_inverse=True); rate_ = np.bincount(inv_, zq) / np.bincount(inv_)
assert np.all(np.diff(rate_) > 0)                                                               # monotone in the level: isotonic = the per-level gold rates
from sklearn.isotonic import IsotonicRegression
hq = np.clip(rate_[inv_], 1e-6, 1 - 1e-6)
hi_ = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip").fit(lev, zq).predict(lev)  # f14's calibration
assert np.max(np.abs(hi_ - hq)) < 1e-9
chq = np.flatnonzero(Pq.parent_row.values >= 0); paq = Pq.parent_row.values[chq]; keep = paq != chq; chq, paq = chq[keep], paq[keep]; assert len(chq) == 980520
hp, hc, zp, zc = hq[paq], hq[chq], zq[paq], zq[chq]
assoc = lambda j11: float(np.sum(j11) / np.sum(hp) - np.sum(hc - j11) / np.sum(1 - hp))
arr = lambda j11: float((np.sum(j11) / np.sum(hp)) / (np.sum(hc - j11) / np.sum(1 - hp)))
jlo, jhi, jin = np.maximum(0, hp + hc - 1), np.minimum(hp, hc), hp * hc
g1, g0 = zc[zp == 1].mean(), zc[zp == 0].mean()
S_F14 = "results/x14_pops/cc_threads.parquet (z, s:haiku, parent_row), logic of scripts/f14_regions.py"
put("xcFrAssocN", len(chq), comma, S_F14, "reply edges (parent_row >= 0, self-edges dropped)", "reply edges of Civil Comments reply threads"); chk("xcFrAssocN", "qNEdgeChildren")
put("xcFrAssocLo", assoc(jlo), lambda x: sg(x, 3), S_F14, "per-edge Frechet lower bound max(0, hp + hc - 1)", "risk difference P(child toxic | parent toxic) - P(child toxic | parent not), lowest value given calibrated Haiku levels")
put("xcFrAssocHi", assoc(jhi), lambda x: sg(x, 3), S_F14, "per-edge Frechet upper bound min(hp, hc)", "highest value")
put("xcFrAssocInd", assoc(jin), lambda x: sg(x, 3), S_F14, "j11 = hp hc", "the independence point")
put("xcFrAssocIndRR", arr(jin), f2, S_F14, "j11 = hp hc", "risk ratio at the independence point")
put("xcFrAssocRRLo", arr(jlo), f2, S_F14, "j11 at the lower bound", "risk ratio at the lower bound")
put("xcFrAssocRRHi", arr(jhi), f2, S_F14, "j11 at the upper bound", "risk ratio at the upper bound")
put("xcFrAssocGold", g1 - g0, lambda x: sg(x, 3), S_F14, "gold labels", "gold's risk difference")
put("xcFrAssocGoldRR", g1 / g0, f2, S_F14, "gold labels", "gold's reply risk ratio"); chk("xcFrAssocGoldRR", "gRR")
F14o = json.load(open(f"{RES}/f14_regions_haiku.json"))["association"]
for nm_ in ("xcFrAssocLo", "xcFrAssocHi", "xcFrAssocInd", "xcFrAssocGold"):
    note(nm_, f"old population (f14_regions_haiku.json, {F14o['n']:,} edges): frechet_diff {F14o['frechet_diff'][0]:+.3f} to {F14o['frechet_diff'][1]:+.3f}, gold {F14o['gold_diff']:+.3f}, independence {F14o['independence_diff']:+.3f} (RR {F14o['independence_rr']:.2f}); not used")

# ======================================================================== M01-M03, M10: the confirmation audits (coverage, widths, small budgets, bias)
section("M01-M03 coverage and width by target, comparators, small budgets")
LAWS_I = ["depth_law", "size_law", "article_size_law"]; PLAC_I = ["placebo_depth", "placebo_size"]; ITEM_M = LAWS_I + PLAC_I; EDGE_M = ["log_rr", "log_placebo_rr"]
SIZE_M = {"size_law", "article_size_law", "placebo_size"}; V3I, V3E = "pps_seq4_ba", "edge_m3_c15"


def load_dir(d, reg, cells=None, exclude=None):
    """as scripts/x74_eval_B.py and x85_small_budgets.py"""
    cells = {} if cells is None else cells
    for fn in sorted(glob.glob(f"{d}/*.json")):
        if os.path.basename(fn).startswith("eval"):
            continue
        J = json.load(open(fn))
        for R_ in J["results"]:
            tag = f"{J['pop']}__{R_['labeler'].replace('@', '_at_')}"
            if exclude and J["pop"] == exclude:
                continue
            for B, dd in (R_.get(reg) or {}).items():
                for des, mm in dd.items():
                    cells.setdefault(tag, {}).setdefault(int(B), {})[des] = mm
            if reg == "item" and R_.get("truth_item"):
                cells.setdefault(tag, {}).setdefault("_truth", R_["truth_item"])
            if reg == "edge" and R_.get("truth_edge"):
                cells.setdefault(tag, {}).setdefault("_truth", {k: R_["truth_edge"][k] for k in ("log_rr", "log_placebo_rr")})
    return cells


def defined(C, m):
    t = C.get("_truth", {}).get(m)
    return t is None or bool(np.isfinite(t))


I = load_dir(f"{RES}/x11_conf_item_all", "item")
for fn in sorted(glob.glob(f"{RES}/x72_item_cga/*.json")):                                    # Addendum 72: CGA-Wiki items rerun with corrected reply counts
    J = json.load(open(fn))
    for R_ in J["results"]:
        tag = f"{J['pop']}__{R_['labeler'].replace('@', '_at_')}"
        for B, dd in R_["item"].items():
            for des, mm in dd.items():
                if tag in I and int(B) in I[tag]:
                    I[tag][int(B)][des] = mm
I = load_dir(f"{RES}/x74_b1b3", "item", I)
E = load_dir(f"{RES}/x11_conf_edge_all", "edge", exclude="cga"); E = load_dir(f"{RES}/x74_b1b3", "edge", E)
assert len(I) == 55 and len(E) == 48
# arXiv revision: the list of audit cells (Appendix H): evaluation cells for laws (I), for contagion (E), and the 17 development cells
DEVC = sorted(os.path.basename(f_)[:-5] for f_ in glob.glob(f"{RES}/x10_dev_r0/*__*.json")); assert len(DEVC) == 17


def cell_name(lab_, pop_=None):
    L_ = lab_.replace("_at_", "@")
    nm_ = NICE[L_].replace(", parent shown", " + parent")
    if (pop_, L_) == ("cc_threads", "qwen7b_m_ctx"):                                        # the evaluation cell at the no-context threshold (see ABANDONED)
        return nm_ + " (cut without the parent)"
    return nm_ + (" (gold's rate)" if L_.endswith("@m") else "")


CROWS = []
for pop_, pnm_ in POPS:
    it_ = sorted(t.split("__", 1)[1] for t in I if t.split("__", 1)[0] == pop_)
    ed_ = sorted(t.split("__", 1)[1] for t in E if t.split("__", 1)[0] == pop_)
    dv_ = sorted(t.split("__", 1)[1] for t in DEVC if t.split("__", 1)[0] == pop_)
    ec_ = "same" if ed_ == it_ else ("none" if not ed_ else "; ".join(cell_name(x, pop_) for x in ed_))
    CROWS.append(" & ".join([pnm_, "; ".join(cell_name(x, pop_) for x in it_), ec_, "; ".join(cell_name(x) for x in dv_) or "--"]) + " \\\\")
CELLS_ROWS = CROWS
open(f"{OUT_TAB}/xc_cells.tex", "w").write("\n".join(CROWS) + "\n")


def cov_vals(cells, d, metrics, bmin=0, bmax=10 ** 9):
    return [dd[d][m]["coverage"] for C in cells.values() for B, dd in C.items() if B != "_truth" and bmin <= B <= bmax and d in dd for m in metrics
            if m in dd[d] and defined(C, m)]


def covs(cells, d, metrics, bmin=0, bmax=10 ** 9):
    v = cov_vals(cells, d, metrics, bmin, bmax)
    return (float(np.mean(v)), float(np.min(v)), len(v), float(np.mean(np.array(v) < 0.90))) if v else (None, None, 0, None)


def ratios(cells, a, b, metrics, bmin=0, bmax=10 ** 9):
    out = []
    for tag, C in cells.items():
        for B, dd in C.items():
            if B == "_truth" or B < bmin or B > bmax or a not in dd or b not in dd:
                continue
            for m in metrics:
                if not defined(C, m):
                    continue
                x, y = dd[a].get(m), dd[b].get(m)
                if not x or not y or x.get("width") is None or y.get("width") is None or y["width"] <= 0:
                    continue
                out.append(x["width"] / y["width"])
    return out


med = lambda v: float(np.median(v)) if v else None
EB = load("x74_eval/eval_B.json"); SB = load("x85/x85_small_budgets.json"); S_EB = "results/x74_eval/eval_B.json"; S_SB = "results/x85/x85_small_budgets.json"
S_RAW = "results/x11_conf_item_all (+ x72_item_cga for CGA-Wiki), x11_conf_edge_all without CGA-Wiki, x74_b1b3 (loaded as x74_eval_B.py)"
DES = [("\\ts", V3I, V3E), ("PPI++", "ppi", "ppi"), ("DSL", "dsl", "edge_dsl"), ("Active inference", "ai", "edge_ai"), ("Stratified", "upgrade_lw", "upgrade_lw"),
       ("Uniform", "srs", "srs"), ("Calibrate \\& aggregate", "calag", "edge_calag"), ("Calibrated within strata", "calag_s", None)]
for _, di, de in DES:                                                                           # reproduce eval_B.json and x85_small_budgets.json
    ref = EB["items"]["v3_cov"] if di == V3I else EB["items"][di]["cov"]
    assert np.allclose(covs(I, di, ITEM_M, 1000), ref, rtol=0, atol=1e-12), di
    if di != V3I:
        assert abs(med(ratios(I, V3I, di, LAWS_I, 1000)) - EB["items"][di]["laws"]) < 1e-12 and abs(med(ratios(I, V3I, di, PLAC_I, 1000)) - EB["items"][di]["placebo"]) < 1e-12
    for B, v in SB["items"][di].items():
        for fam, ms in (("laws", LAWS_I), ("placebo", PLAC_I), ("all", ITEM_M)):
            c_ = covs(I, di, ms, int(B), int(B)); r_ = v[fam]
            assert c_[2] == r_["n"] and abs(c_[0] - r_["mean"]) < 1e-12 and abs(c_[1] - r_["min"]) < 1e-12 and abs(c_[3] - r_["share_lt_090"]) < 1e-12, (di, B, fam)
    if de:
        ref = EB["edges"]["v3_cov"] if de == V3E else EB["edges"][de]["cov"]
        assert np.allclose(covs(E, de, EDGE_M, 2000), ref, rtol=0, atol=1e-12), de
        if de != V3E:
            assert abs(med(ratios(E, V3E, de, ["log_rr"], 2000)) - EB["edges"][de]["gold"]) < 1e-12 and abs(med(ratios(E, V3E, de, ["log_placebo_rr"], 2000)) - EB["edges"][de]["placebo"]) < 1e-12
        for B, v in SB["edges"][de].items():
            for fam, ms in (("gold_rr", ["log_rr"]), ("placebo_rr", ["log_placebo_rr"])):
                c_ = covs(E, de, ms, int(B), int(B)); r_ = v[fam]
                assert c_[2] == r_["n"] and abs(c_[0] - r_["mean"]) < 1e-12 and abs(c_[1] - r_["min"]) < 1e-12, (de, B, fam)
BG = [("250", 250, 250), ("500", 500, 500), ("1k--8k", 1000, 8000)]
rc, rw = [], []
for nm, di, de in DES:
    cc_ = [c3(*covs(I, di, ms, lo, hi)[:2]) for ms in (LAWS_I, PLAC_I) for _, lo, hi in BG]
    cc_ += [c3(*covs(E, de, [m], 2000)[:2]) if de else "--" for m in EDGE_M]
    rc.append(f"{nm} & " + " & ".join(cc_) + " \\\\" + ("\n\\midrule" if di == V3I else ""))
    if di in (V3I, "calag", "calag_s"):
        ww = ["--"] * 8
    else:
        ww = [f2(med(ratios(I, V3I, di, ms, lo, hi))) for ms in (LAWS_I, PLAC_I) for _, lo, hi in BG] + [f2(med(ratios(E, V3E, de, [m], 2000))) for m in EDGE_M]
    if di not in (V3I, "calag", "calag_s"):                                                    # rows that would be all '--' are left out
        rw.append(f"{nm} & " + " & ".join(ww) + " \\\\")
HEAD_T = ["Design", "Laws: 250", "500", "1k-8k", "Placebo slopes: 250", "500", "1k-8k", "Reply RR: 2k-16k", "Placebo RR: 2k-16k"]
table("cov_target_cov", rc,
      caption=("Coverage of the 95% intervals by target and budget: mean over the combinations of cell, budget and target, with the lowest in "
               "parentheses, for laws (depth, thread size, article size) and placebo slopes at 250, 500 and 1,000-8,000 labels (to 2,000 on CGA-Wiki), "
               "and for the reply and placebo risk ratios at 2,000-16,000. Calibrated within strata: calibrate and aggregate within strata of score "
               "level, depth and size (laws and placebo slopes only)."),
      headers=HEAD_T, src=S_RAW + f"; reproduces {S_EB} (items/edges cov, widths) and {S_SB} (coverage by design and budget)",
      how="coverage = the stored share of 500 audits whose interval covers the full-gold value, averaged over (cell, budget, metric) entries with a defined estimand; minima over the same entries.",
      checks="the 1k-8k and 2k-16k columns pooled over both families equal Table 5 (confirm_v6.tex) and eval_B.json; the 250 and 500 columns equal x85_small_budgets.json (asserted entry by entry)",
      caveats="CGA-Wiki contributes to laws and placebo slopes only (budgets to 2,000). The placebo risk ratio of a labeler that flags nothing is undefined and excluded, as in eval_B.json.")
table("cov_target_width", rw,
      caption=("Width of Threadsafe's 95% interval divided by each design's (median over matched cell, budget and target entries; below 1: "
               "Threadsafe's is narrower), by target and budget as in the coverage table. Calibrate and aggregate is not shown: it covers too rarely "
               "for its width to be comparable."),
      headers=HEAD_T, src=S_RAW, how="ratio of the stored mean widths of Threadsafe and the design in the same cell, budget and metric (log scale for risk ratios); median over entries",
      checks="1k-8k laws and placebo columns and the risk-ratio columns equal Table 5 (confirm_v6.tex) and eval_B.json items.<d>.laws/placebo, edges.<d>.gold/placebo (asserted)",
      caveats="none beyond the coverage table's")
c_ = covs(I, V3I, ITEM_M, 1000); ce_ = covs(E, V3E, EDGE_M, 2000)
put("xcTsCov", c_[0], lambda x: pct(x, 1), S_RAW, "pps_seq4_ba, B >= 1,000, laws and placebo slopes", "mean coverage"); chk("xcTsCov", "eTsCov", scale=100)
put("xcTsCovMin", c_[1], lambda x: pct(x, 1), S_RAW, "same", "lowest entry"); chk("xcTsCovMin", "eTsCovMin", scale=100)
put("xcTsN", c_[2], comma, S_RAW, "same", "number of entries"); chk("xcTsN", "vItemEntries")
put("xcTsCovRR", ce_[0], lambda x: pct(x, 1), S_RAW, "edge_m3_c15, B >= 2,000, both risk ratios", "mean coverage"); chk("xcTsCovRR", "eTsCovRR", scale=100)
put("xcTsCovRRMin", ce_[1], lambda x: pct(x, 1), S_RAW, "same", "lowest entry"); chk("xcTsCovRRMin", "vEdgeCovMin")
put("xcTsRRN", ce_[2], i0, S_RAW, "same", "entries"); chk("xcTsRRN", "vEdgeEntries")
for fam, ms in (("Laws", LAWS_I), ("Plac", PLAC_I)):
    for tg, lo, hi in (("TwoFifty", 250, 250), ("FiveHundred", 500, 500)):
        v = covs(I, V3I, ms, lo, hi)
        put(f"xcTs{fam}Cov{tg}", v[0], lambda x: pct(x, 1), S_RAW, f"pps_seq4_ba, B = {lo}, {fam.lower()}", "mean coverage")
        put(f"xcTs{fam}Min{tg}", v[1], lambda x: pct(x, 1), S_RAW, "same", "lowest entry")
for tg, d_ in (("Gold", "log_rr"), ("Plac", "log_placebo_rr")):
    v = covs(E, V3E, [d_], 2000); put(f"xcTsRR{tg}Cov", v[0], lambda x: pct(x, 1), S_RAW, f"edge_m3_c15, B >= 2,000, {d_}", "mean coverage")
    put(f"xcTsRR{tg}Min", v[1], lambda x: pct(x, 1), S_RAW, "same", "lowest entry")
small = covs(I, V3I, ITEM_M, 250, 250)
put("xcTsCovSmall", small[0], lambda x: pct(x, 1), S_RAW, "pps_seq4_ba, B = 250", "mean coverage at 250 labels"); chk("xcTsCovSmall", "qSbTsCov")
put("xcTsMinSmall", min(covs(I, V3I, ITEM_M, 250, 250)[1], covs(I, V3I, ITEM_M, 500, 500)[1]), lambda x: pct(x, 1), S_RAW, "B = 250 and 500", "lowest entry at 250 or 500 labels")
chk("xcTsMinSmall", "qSbTsMin")
assert covs(I, V3I, ITEM_M, 250, 500)[3] == 0                                                  # no Threadsafe entry below 0.90 at 250-500
put("xcTsMinAll", min(c_[1], small[1]), pct, S_RAW, "B = 250 and B >= 1,000", "lowest entry at 250 labels or at 1,000-8,000, as x95"); chk("xcTsMinAll", "eTsMinAll", scale=100)
mins_ = np.random.default_rng(7700).binomial(500, 0.95, size=(4000, c_[2])).min(axis=1) / 500                 # as x96_dist_v6.py, with the entry count of the runs
put("xcMcMinCov", float(np.median(mins_)), lambda x: pct(x, 1), S_RAW, f"{c_[2]} entries (xcTsN), 500 audits each, seed 7700",
    "median over 4,000 replicates of the lowest of that many Binomial(500, 0.95)/500 coverages: the minimum that Monte Carlo error alone produces at nominal coverage")
chk("xcMcMinCov", "dMCMinCov", scale=100); v7f("xcMcMinCov", "the minimum that Monte Carlo error alone would produce (91.6\\%)")
for nm_, di in (("Ppi", "ppi"), ("Unif", "srs"), ("Dsl", "dsl"), ("Strat", "upgrade_lw"), ("Ai", "ai")):
    rr_ = [(t_, B, m, dd[di][m]["coverage"]) for t_, C in I.items() for B, dd in C.items() if B == 250 and di in dd for m in ITEM_M if m in dd[di] and defined(C, m)]
    low = [r for r in rr_ if r[3] < 0.90]
    put(f"xc{nm_}SmallCov", float(np.mean([r[3] for r in rr_])), lambda x: f"{x:.3f}", S_RAW, f"{di}, B = 250", "mean coverage at 250 labels"); chk(f"xc{nm_}SmallCov", f"qSb{nm_}Cov")
    put(f"xc{nm_}SmallMin", min(r[3] for r in rr_), lambda x: f"{x:.3f}", S_RAW, f"{di}, B = 250", "lowest entry"); chk(f"xc{nm_}SmallMin", f"qSb{nm_}Min")
    put(f"xc{nm_}SmallLowN", len(low), i0, S_RAW, f"{di}, B = 250", "entries below 0.90")
    put(f"xc{nm_}SmallN", len(rr_), i0, S_RAW, f"{di}, B = 250", "entries")
    put(f"xc{nm_}SmallSizeShare", sum(r[2] in SIZE_M for r in low) / len(low) if low else 0.0, pct, S_RAW, f"{di}, B = 250", "share of the entries below 0.90 that are size laws or the placebo size slope")
chk("xcPpiSmallLowN", "eLowPpiN"); chk("xcPpiSmallN", "eLowPpiAll"); chk("xcPpiSmallSizeShare", "eLowPpiSizeShare", scale=100); chk("xcPpiSmallMin", "eMinPpiPct", scale=100)
chk("xcPpiSmallCov", "ePpiMeanSmall", scale=100); chk("xcUnifSmallLowN", "eLowUnifN"); chk("xcUnifSmallN", "eLowUnifAll"); chk("xcUnifSmallSizeShare", "eLowUnifSizeShare", scale=100)
chk("xcUnifSmallMin", "eMinUnifPct", scale=100); chk("xcDslSmallMin", "eMinDslPct", scale=100)
v7f("xcPpiSmallCov", "PPI++ falls short at 250 labels (89\\% on average)")
v7f("xcPpiSmallMin", "cover in as few as 55\\% and 79\\% of the audits")
pm_ = SB["ppi_min"]["items"]
put("xcPpiNBelow", pm_["n_lt_090"], i0, S_SB, "ppi_min.items.n_lt_090", "PPI++ entries below 0.90 at B >= 1,000"); chk("xcPpiNBelow", "qPpiNBelow")
assert covs(I, "ppi", ITEM_M, 1000)[3] * covs(I, "ppi", ITEM_M, 1000)[2] == pm_["n_lt_090"]
put("xcPpiSizeShare", sum(v for k, v in pm_["by_metric"].items() if k in SIZE_M) / pm_["n_lt_090"], pct, S_SB, "ppi_min.items.by_metric", "share of them on size laws or the placebo size slope")
chk("xcPpiSizeShare", "qPpiSizeShare", scale=100)
for nm_, di in (("Ppi", "ppi"), ("Dsl", "dsl"), ("Ai", "ai"), ("Strat", "upgrade_lw"), ("Unif", "srs")):
    v = covs(I, di, ITEM_M, 1000); put(f"xc{nm_}Cov", v[0], lambda x: f"{x:.3f}", S_RAW, f"{di}, B >= 1,000", "mean coverage, laws and placebo slopes")
    put(f"xc{nm_}CovMin", v[1], f2, S_RAW, f"{di}, B >= 1,000", "lowest entry")
    put(f"xc{nm_}WLaws", med(ratios(I, V3I, di, LAWS_I, 1000)), f2, S_RAW, f"widths pps_seq4_ba / {di}, B >= 1,000, laws", "median width ratio")
    put(f"xc{nm_}WPlac", med(ratios(I, V3I, di, PLAC_I, 1000)), f2, S_RAW, "same, placebo slopes", "median width ratio")
for nm_, de in (("Ppi", "ppi"), ("Dsl", "edge_dsl"), ("Ai", "edge_ai"), ("Strat", "upgrade_lw"), ("Unif", "srs")):
    put(f"xc{nm_}WGold", med(ratios(E, V3E, de, ["log_rr"], 2000)), f2, S_RAW, f"widths edge_m3_c15 / {de}, B >= 2,000, log_rr", "median width ratio")
    put(f"xc{nm_}WPlacRR", med(ratios(E, V3E, de, ["log_placebo_rr"], 2000)), f2, S_RAW, "same, log_placebo_rr", "median width ratio")
for a_, o_ in (("xcPpiCov", "wPpiCovMean"), ("xcPpiCovMin", "wPpiCovMin"), ("xcStratCov", "vItemCovVtwo"), ("xcStratCovMin", "vItemCovMinVtwo"), ("xcUnifCov", "vItemCovUnif"),
               ("xcUnifCovMin", "vItemCovMinUnif"), ("xcPpiWLaws", "vItemLawsPPI"), ("xcPpiWPlac", "vItemPlacPPI"), ("xcDslWLaws", "wItemLawsDSL"), ("xcDslWPlac", "wItemPlacDSL"),
               ("xcAiWLaws", "wItemLawsAI"), ("xcAiWPlac", "wItemPlacAI"), ("xcStratWLaws", "vItemLawsVtwo"), ("xcStratWPlac", "vItemPlacVtwo"), ("xcUnifWLaws", "vItemLawsUnif"),
               ("xcUnifWPlac", "vItemPlacUnif"), ("xcPpiWGold", "vEdgeLawsPPI"), ("xcPpiWPlacRR", "vEdgePlacPPI"), ("xcDslWGold", "wEdgeLawsDSL"), ("xcDslWPlacRR", "wEdgePlacDSL"),
               ("xcStratWGold", "vEdgeLawsVtwo"), ("xcStratWPlacRR", "vEdgePlacVtwo"), ("xcUnifWGold", "vEdgeLawsUnif"), ("xcUnifWPlacRR", "vEdgePlacUnif")):
    chk(a_, o_)
v7f("xcDslWLaws", "intervals are 0.75 times as wide as DSL's")
CV4 = {r.split("&")[0].strip(): [x.strip() for x in r.split("&")[1:]] for r in open(f"{W}/results/reference/tables/confirm_v4.tex").read().split("\\\\") if "&" in r and "Comparator" not in r and "multicolumn" not in r}
MNAME = {"depth_law": "Escalation", "size_law": "Thread size law", "article_size_law": "Article size law", "placebo_depth": "Placebo depth slope", "placebo_size": "Placebo size slope"}
for m, tg in (("depth_law", "Depth"), ("size_law", "Size"), ("article_size_law", "Art"), ("placebo_depth", "PlacDepth"), ("placebo_size", "PlacSize")):
    put(f"xcPpiW{tg}", med(ratios(I, V3I, "ppi", [m], 1000)), f2, S_RAW, f"widths pps_seq4_ba / ppi, B >= 1,000, {m}", "median width ratio for this target")
    assert M[f"xcPpiW{tg}"] == CV4[MNAME[m]][2], (m, CV4[MNAME[m]])                              # the stored per-metric table, column PPI++
    note(f"xcPpiW{tg}", f"equals paper_v5/tables/confirm_v4.tex, row '{MNAME[m]}', PPI++ column")
    put(f"xcStratW{tg}", med(ratios(I, V3I, "upgrade_lw", [m], 1000)), f2, S_RAW, f"widths pps_seq4_ba / upgrade_lw, B >= 1,000, {m}", "median width ratio against the stratified design")
H72 = load("x72_eval/eval_H72.json")
for B, tg in (("1000", "OneK"), ("2000", "TwoK"), ("4000", "FourK"), ("8000", "EightK")):
    put(f"xcStratWLaws{tg}", med(ratios(I, V3I, "upgrade_lw", LAWS_I, int(B), int(B))), f2, S_RAW, f"B = {B}", "median width ratio against stratified, laws, by budget")
    assert abs(PROV[f"xcStratWLaws{tg}"]["raw"] - H72["item"]["desc_vs_upgrade_lw"]["by_B"][B]["laws"]) < 1e-12
for B, tg in (("2000", "TwoK"), ("4000", "FourK"), ("8000", "EightK"), ("16000", "SixteenK")):
    put(f"xcStratWGold{tg}", med(ratios(E, V3E, "upgrade_lw", ["log_rr"], int(B), int(B))), f2, S_RAW, f"B = {B}", "median width ratio against stratified, reply RR, by budget")
    assert abs(PROV[f"xcStratWGold{tg}"]["raw"] - H72["edge"]["desc_vs_upgrade_lw"]["by_B"][B]["laws"]) < 1e-12
cal = SB["items"]["calag"]; cals = SB["items"]["calag_s"]
put("xcCalagLo", cal["250"]["all"]["mean"], pct, S_SB, "items.calag['250'].all.mean", "calibrate and aggregate: mean coverage at 250 labels"); chk("xcCalagLo", "eCalagLo", scale=100)
put("xcCalagHi", cal["8000"]["all"]["mean"], pct, S_SB, "items.calag['8000'].all.mean", "at 8,000 labels"); chk("xcCalagHi", "eCalagHi", scale=100)
put("xcCalagSHi", cals["8000"]["all"]["mean"], pct, S_SB, "items.calag_s['8000'].all.mean", "calibrated within strata, at 8,000"); chk("xcCalagSHi", "eCalagSHi", scale=100)
for tg, d_ in (("", "calag"), ("S", "calag_s")):
    v = covs(I, d_, ["depth_law", "size_law"], 2000)
    put(f"xcCalag{tg}Laws", v[0], f2, S_RAW, f"{d_}, depth and thread size laws, B >= 2,000", "mean coverage"); chk(f"xcCalag{tg}Laws", f"wCalag{tg}CovLaws")
    put(f"xcCalag{tg}LawsBad", v[3], pct, S_RAW, "same", "share of entries below 0.90"); chk(f"xcCalag{tg}LawsBad", f"wCalag{tg}ShareBad", scale=100)
# M10: bias in standard-deviation units (as eval_B.json v3_bias_sd)
section("M10 bias of the estimates")
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
    mc = float(np.median(np.abs(np.random.default_rng(7475).normal(0, 1 / np.sqrt(500), (2000, len(bs)))).max(1)))
    ref = EB["v3_bias_sd"][nm]; assert len(bs) == ref["n"] and abs(np.max(bs) - ref["max"]) < 1e-12 and abs(np.median(bs) - ref["median"]) < 1e-12 and abs(mc - ref["mc_expected_max"]) < 1e-12
    tg = "Items" if nm == "items" else "Edge"
    put(f"xcBias{tg}N", len(bs), comma, S_RAW + "; " + S_EB + " v3_bias_sd", f"{d}: bias, rmse per entry", f"entries ({'every budget' if nm == 'items' else 'B >= 2,000'})")
    put(f"xcBias{tg}Med", float(np.median(bs)), f2, S_RAW, "same", "median |bias| / SD, SD^2 = rmse^2 - bias^2")
    put(f"xcBias{tg}Max", float(np.max(bs)), f2, S_RAW, "same", "largest |bias| / SD")
    put(f"xcBias{tg}MC", mc, f2, S_RAW, "seed 7475", "median over 2,000 Monte Carlo replicates of the largest |N(0, 1/500)| over that many entries: the largest |bias|/SD that 500 audits produce with no bias")
chk("xcBiasItemsN", "wBiasItemsN"); chk("xcBiasItemsMax", "wBiasItemsMax"); chk("xcBiasItemsMC", "wBiasMCMax"); chk("xcBiasEdgeN", "qBiasEdgeN"); chk("xcBiasEdgeMax", "wBiasEdgeMax")
chk("xcBiasEdgeMC", "qBiasEdgeMC")

# ======================================================================== M04: adversarial labelers
section("M04 adversarial labelers")
A2i = load_dir(f"{RES}/x74_b2", "item"); A2e = load_dir(f"{RES}/x74_b2", "edge"); S_B2 = "results/x74_b2 (loaded as x74_eval_B.py)"
ADV = [("adv_rev", "Reversed score"), ("adv_rand", "Random score"), ("adv_const", "Constant score"), ("adv_depth", "Flags every comment at depth $\\ge 2$"), ("adv_trap", "Gold-aware trap")]
for d in (V3I, "upgrade_lw", "ppi", "srs"):
    assert np.allclose(covs(A2i, d, ITEM_M, 1000), EB["adversarial"]["items"][d], atol=1e-12)
for d in (V3E, "upgrade_lw", "ppi", "srs"):
    assert np.allclose(covs(A2e, d, EDGE_M, 2000), EB["adversarial"]["edges"][d], atol=1e-12)
rows = []; worst = []
for lab, nm in ADV + [("all", "All five")]:
    si = {t: c for t, c in A2i.items() if lab == "all" or t.split("__")[1] == lab}; se = {t: c for t, c in A2e.items() if lab == "all" or t.split("__")[1] == lab}
    ci = [covs(si, d, ITEM_M, 1000) for d in (V3I, "ppi", "srs")]; ce = [covs(se, d, EDGE_M, 2000) for d in (V3E, "ppi", "srs")]
    w = [med(ratios(si, V3I, "srs", LAWS_I, 1000)), med(ratios(si, V3I, "srs", PLAC_I, 1000)), med(ratios(se, V3E, "srs", ["log_rr"], 2000)), med(ratios(se, V3E, "srs", ["log_placebo_rr"], 2000))]
    if lab != "all":
        bl = EB["adversarial"]["by_labeler"][lab]
        for a, b_ in zip(w, (bl["laws"], bl["placebo"], bl["gold"], bl["plac_rr"])):
            assert (a is None and b_ is None) or abs(a - b_) < 1e-12, (lab, a, b_)
        assert np.allclose(ci[0], bl["cov_v3"], atol=1e-12) and np.allclose(ce[0], bl["cov_v3_edge"], atol=1e-12)
        worst += [x for x in w[:2] if x is not None]
    rows.append(("\\midrule\n" if lab == "all" else "") + f"{nm} & " + " & ".join(c3(*x[:2]) for x in ci + ce) + " & " + " & ".join(f2(x) for x in w) + " \\\\")
table("adversaries", rows,
      caption=("Five labelers built from gold and structure: one minus Detoxify's score, a random score, a constant score, flags on exactly the comments "
               "at depth 2 or more, and a gold-aware trap whose score is the gold label at depth 0-1 and its opposite deeper. Coverage of Threadsafe, "
               "PPI++ and uniform sampling (mean, lowest in parentheses) for laws and placebo slopes at 1,000-8,000 labels and for the risk ratios at "
               "2,000-16,000, and the width of Threadsafe's interval divided by uniform sampling's (median)."),
      headers=["Labeler", "Laws and placebo slopes: Threadsafe", "PPI++", "Uniform", "Risk ratios: Threadsafe", "PPI++", "Uniform",
               "Width / uniform: laws", "placebo slopes", "reply RR", "placebo RR"],
      src=S_B2 + f"; reproduces {S_EB} adversarial.items/edges and adversarial.by_labeler", how="as the coverage and width tables, per labeler type (cell tag); populations: the 5% sample, CAD, Qian Reddit and Gab, CGA-Wiki (items only) and BAD",
      checks="Threadsafe's and the comparators' pooled coverage and the per-type width ratios equal eval_B.json (asserted)",
      caveats="The constant labeler flags nothing, so its placebo slopes and placebo risk ratio are undefined ('--'). Under the trap, the stratified design (not shown) has pure strata and returns the exact value (coverage 1.000).")
ai_, ae_ = EB["adversarial"]["items"], EB["adversarial"]["edges"]
put("xcAdvItemCov", ai_[V3I][0], lambda x: pct(x, 1), S_EB, "adversarial.items.pps_seq4_ba[0]", "Threadsafe's mean coverage under the five labelers"); chk("xcAdvItemCov", "eAdvItem", scale=100)
put("xcAdvItemMin", ai_[V3I][1], lambda x: f"{x:.3f}", S_EB, "adversarial.items.pps_seq4_ba[1]", "lowest entry"); chk("xcAdvItemMin", "wAdvItemMin")
put("xcAdvItemN", ai_[V3I][2], i0, S_EB, "adversarial.items.pps_seq4_ba[2]", "entries")
put("xcAdvEdgeCov", ae_[V3E][0], lambda x: pct(x, 1), S_EB, "adversarial.edges.edge_m3_c15[0]", "edges"); chk("xcAdvEdgeCov", "eAdvEdge", scale=100)
put("xcAdvEdgeMin", ae_[V3E][1], lambda x: f"{x:.3f}", S_EB, "adversarial.edges.edge_m3_c15[1]", "lowest edge entry"); chk("xcAdvEdgeMin", "wAdvEdgeMin")
put("xcAdvEdgeN", ae_[V3E][2], i0, S_EB, "adversarial.edges.edge_m3_c15[2]", "edge entries")
put("xcAdvPpiMin", ai_["ppi"][1], lambda x: f"{x:.3f}", S_EB, "adversarial.items.ppi[1]", "PPI++'s lowest entry under the adversaries")
put("xcAdvUnifMin", ai_["srs"][1], lambda x: f"{x:.3f}", S_EB, "adversarial.items.srs[1]", "uniform sampling's lowest entry")
put("xcAdvStratMin", ai_["upgrade_lw"][1], f2, S_EB, "adversarial.items.upgrade_lw[1]", "the stratified design's lowest entry"); chk("xcAdvStratMin", "wAdvStratMin")
put("xcAdvWorstUnif", max(worst), f2, S_EB, "adversarial.by_labeler.<lab>.{laws, placebo}", "largest median width ratio to uniform over the five labelers (laws and placebo slopes)"); chk("xcAdvWorstUnif", "wAdvWorstUnif")
put("xcAdvTrapStrat", EB["adversarial"]["by_labeler"]["adv_trap"]["cov_v2"][0], lambda x: f"{x:.3f}", S_EB, "adversarial.by_labeler.adv_trap.cov_v2[0]", "stratified design's coverage under the trap (pure strata: exact value)")

# ======================================================================== M05: ablations
section("M05 ablations")
AB = EB["ablations"]; ABL_I = ["pps_seq4_ba_unif", "pps_seq4_ba_noaux", "pps_seq1_ba", "pps_seq4_a"]
ABL_E = {"pps_seq4_ba_unif": "edge_m3_c15_unif", "pps_seq4_ba_noaux": "edge_m3_c15_noaux", "pps_seq1_ba": "edge_m1_c15", "pps_seq4_a": None}
regen = []
for a in ABL_I:
    ea = ABL_E[a]
    w = [med(ratios(I, a, V3I, LAWS_I, 1000)), med(ratios(I, a, V3I, PLAC_I, 1000)), med(ratios(E, ea, V3E, ["log_rr"], 2000)) if ea else None, med(ratios(E, ea, V3E, ["log_placebo_rr"], 2000)) if ea else None]
    for x, k in zip(w, ("laws", "placebo", "gold", "plac_rr")):
        assert (x is None and AB[a][k] is None) or abs(x - AB[a][k]) < 1e-12, (a, k)
    assert np.allclose(covs(I, a, ITEM_M, 1000), AB[a]["cov_items"], atol=1e-12)
    regen.append(f"{AB[a]['label']} & {f2(AB[a]['laws'])} & {f2(AB[a]['placebo'])} & {f2(AB[a]['gold'])} & {f2(AB[a]['plac_rr'])} \\\\")
STORED_ABL = open(f"{W}/results/reference/tables/ablation.tex").read()
assert "\n".join(regen) + "\n" == STORED_ABL                                                   # the earlier-draft ablation table is current
ABN = {"pps_seq4_ba_unif": "Weighted sampling (uniform probabilities instead)", "pps_seq4_ba_noaux": "Auxiliary classifiers (the labeler's scores only)",
       "pps_seq1_ba": "Four waves (one wave: the model fitted on the pilot only)", "pps_seq4_a": "Tempered predictions (plain predictions)"}
rows = []
for a in ABL_I:
    v = AB[a]; ce_ = v["cov_edges"]
    rows.append(f"{ABN[a]} & {f2(v['laws'])} & {f2(v['placebo'])} & {f2(v['gold'])} & {f2(v['plac_rr'])} & {c3(*v['cov_items'][:2])} & " + (c3(*ce_[:2]) if ce_ else "--") + " \\\\")
table("ablation", rows,
      caption=("Removing one part of Threadsafe at a time: the median width of the variant's 95% interval divided by Threadsafe's (above 1: wider) for laws "
               "and placebo slopes at 1,000-8,000 labels and the reply and placebo risk ratios at 2,000-16,000, and the variant's coverage (mean, lowest "
               "in parentheses). The edge design has no tempering variant."),
      headers=["Part removed (replacement)", "Laws", "Placebo slopes", "Reply RR", "Placebo RR", "Coverage: laws and placebo slopes", "Coverage: risk ratios"],
      src=f"{S_EB} ablations.<variant>.{{laws, placebo, gold, plac_rr, cov_items, cov_edges}}", how="as stored; widths: variant / Threadsafe in the same cell, budget and metric, median",
      checks="the four width columns regenerated with the earlier-draft labels are byte-identical to paper_v5/tables/ablation.tex (asserted), so that table is current; the widths and item coverage were also recomputed from the raw runs (asserted)",
      caveats="Row labels use the paper's terms (waves, auxiliary classifiers) instead of the earlier-draft table's (folds, instruments). The one-wave variant at 250 labels is in xcOneFold* (M06).")
put("xcAblUnifMin", min(AB["pps_seq4_ba_unif"][k] for k in ("laws", "placebo", "gold", "plac_rr")), f2, S_EB, "ablations.pps_seq4_ba_unif", "smallest width ratio without weighted sampling"); chk("xcAblUnifMin", "qAblUnifMin")
put("xcAblUnifMax", max(AB["pps_seq4_ba_unif"][k] for k in ("laws", "placebo", "gold", "plac_rr")), f2, S_EB, "same", "largest"); chk("xcAblUnifMax", "qAblUnifMax")
put("xcAblAuxMin", min(AB["pps_seq4_ba_noaux"][k] for k in ("laws", "placebo", "gold", "plac_rr")), f2, S_EB, "ablations.pps_seq4_ba_noaux", "without the auxiliary classifiers: smallest"); chk("xcAblAuxMin", "qAblAuxMin")
put("xcAblAuxMax", max(AB["pps_seq4_ba_noaux"][k] for k in ("laws", "placebo", "gold", "plac_rr")), f2, S_EB, "same", "largest"); chk("xcAblAuxMax", "qAblAuxMax")
put("xcAblOneMin", min(AB["pps_seq1_ba"][k] for k in ("laws", "placebo", "gold", "plac_rr")), f2, S_EB, "ablations.pps_seq1_ba", "one wave: smallest"); chk("xcAblOneMin", "qAblOneGold")
put("xcAblOneMax", max(AB["pps_seq1_ba"][k] for k in ("laws", "placebo", "gold", "plac_rr")), f2, S_EB, "same", "largest"); chk("xcAblOneMax", "qAblOneMax")
put("xcAblPlainMin", AB["pps_seq4_a"]["cov_items"][1], f2, S_EB, "ablations.pps_seq4_a.cov_items[1]", "plain predictions: lowest coverage entry"); chk("xcAblPlainMin", "wAblPlainMin")
put("xcAblPlainMinPct", AB["pps_seq4_a"]["cov_items"][1], lambda x: pct(x, 1), S_EB, "ablations.pps_seq4_a.cov_items[1]", "plain predictions: lowest coverage entry, as a percentage")
put("xcAblPlainW", min(AB["pps_seq4_a"]["laws"], AB["pps_seq4_a"]["placebo"]), f2, S_EB, "ablations.pps_seq4_a.{laws, placebo}", "plain predictions: width ratio")
covs_ = [v["cov_items"][0] for v in AB.values()] + [v["cov_edges"][0] for v in AB.values() if v["cov_edges"]]
put("xcAblMinCov", min(covs_), lambda x: f"{x:.3f}", S_EB, "ablations.*.cov_items[0], cov_edges[0]", "the lowest mean coverage of any variant"); chk("xcAblMinCov", "wAblMinCov")
v7f("xcAblUnifMin", "without weighted sampling they are 1.17 to 2.11 times as wide")

# ======================================================================== M06: one wave, by budget
section("M06 one wave by budget")
Bi = load("x88/x88_breakdown.json")["B88i"]; S_BI = "results/x88/x88_breakdown.json"
BW = {"250": "TwoFifty", "500": "FiveHundred", "1000": "OneK", "2000": "TwoK", "4000": "FourK", "8000": "EightK"}
for B, tg in BW.items():
    put(f"xcOneFoldLaws{tg}", Bi[B]["laws"]["median_ratio"], f2, S_BI, f"B88i['{B}'].laws.median_ratio", "width of the one-wave variant (pps_seq1_ba) / Threadsafe's, laws, median")
    put(f"xcOneFoldPlac{tg}", Bi[B]["placebo"]["median_ratio"], f2, S_BI, f"B88i['{B}'].placebo.median_ratio", "same, placebo slopes")
    a_ = med(ratios(I, "pps_seq1_ba", V3I, LAWS_I, int(B), int(B))); assert abs(a_ - Bi[B]["laws"]["median_ratio"]) < 1e-12, B   # recomputed from the raw runs
chk("xcOneFoldLawsTwoFifty", "cOneFoldSmall"); chk("xcOneFoldLawsOneK", "cOneFoldThousand"); chk("xcOneFoldLawsEightK", "cOneFoldLarge")
v7f("xcOneFoldLawsTwoFifty", "with one wave instead of four 1.30 times at 250 labels")
pv = [Bi[B]["placebo"]["median_ratio"] for B in BW]
put("xcOneFoldPlacMin", min(pv), f2, S_BI, "B88i[*].placebo.median_ratio", "smallest over budgets"); put("xcOneFoldPlacMax", max(pv), f2, S_BI, "same", "largest")
put("xcOneFoldCovMin", min(min(Bi[B][f_]["cov_single"][1] for f_ in ("laws", "placebo")) for B in BW), lambda x: f"{x:.3f}", S_BI, "B88i[*].{laws,placebo}.cov_single[1]", "the one-wave variant's lowest coverage at any budget")
put("xcOneFoldTsCovMin", min(min(Bi[B][f_]["cov_ts"][1] for f_ in ("laws", "placebo")) for B in BW), lambda x: f"{x:.3f}", S_BI, "B88i[*].{laws,placebo}.cov_ts[1]", "Threadsafe's lowest coverage in the same entries")

# ======================================================================== M07: laws added after the design was fixed
section("M07 laws added after the design was fixed")
UV = load("x76_eval/x76_eval.json"); S_UV = "results/x76_eval/x76_eval.json"
put("xcUnCovMean", UV["H76c"]["mean"], lambda x: pct(x, 1), S_UV, "H76c.mean", "Threadsafe's mean coverage of the within-thread law and prevalence (targets at weight 0)"); chk("xcUnCovMean", "qUnCovMean")
put("xcUnCovMin", UV["H76c"]["min"], lambda x: f"{x:.3f}", S_UV, "H76c.min", "lowest entry"); chk("xcUnCovMin", "qUnCovMin")
put("xcUnN", UV["H76c"]["n"], i0, S_UV, "H76c.n", "entries"); chk("xcUnN", "qUnNEntries")
put("xcUnShareNinety", UV["H76c"]["share90"], pct, S_UV, "H76c.share90", "share of entries at or above 0.90"); chk("xcUnShareNinety", "qUnShareNinety", scale=100)
for k_, tg, o_ in (("depth_within/ppi", "WithinPpi", "qUnWithinPPI"), ("depth_within/srs", "WithinUnif", "qUnWithinUnif"), ("prevalence/ppi", "PrevPpi", "qUnPrevPPI"), ("prevalence/srs", "PrevUnif", "qUnPrevUnif")):
    put(f"xcUn{tg}", UV["ratios"][k_]["median"], f2, S_UV, f"ratios['{k_}'].median", "median width ratio, Threadsafe / comparator"); chk(f"xcUn{tg}", o_)
put("xcUnCellsN", UV["n_cells"], i0, S_UV, "n_cells", "cells")

# ======================================================================== M08: fresh gold
section("M08 fresh gold by rater process")
FG = load("x88/x88_fresh_gold.json"); S_FG = "results/x88/x88_fresh_gold.json"; FC_ = FG["cells"]; BUDF = ("1000", "4000")
PRN = [("fixed", "Fixed gold (all raters, as in the data)"), ("one", "One rater"), ("three", "Majority of three raters"), ("adaptive", "Adaptive (three, and two more if they disagree)")]
mets = list(next(iter(FC_.values()))["truth"]["one"].keys()); assert sorted(mets) == sorted(ITEM_M)
rows = []; FGV = {}
for pr, nm in PRN:
    ent = [FC_[l][B][pr][m] for l in FC_ for B in BUDF for m in mets]
    cr = [e["cov_expected_rv"] for e in ent]; cn = [e["cov_expected"] for e in ent]; cz = [e["cov_realized"] for e in ent]
    if pr == "fixed":
        wl = wp = None; ra = None
    else:
        wl = med([FC_[l][B][pr][m]["width_rv_vs_fixed"] for l in FC_ for B in BUDF for m in LAWS_I]); wp = med([FC_[l][B][pr][m]["width_rv_vs_fixed"] for l in FC_ for B in BUDF for m in PLAC_I])
        ra = [FC_[l][B]["_raters_adaptive"] for l in FC_ for B in BUDF] if pr == "adaptive" else None
    FGV[pr] = dict(cr=cr, cn=cn, cz=cz, wl=wl, wp=wp, ra=ra, n=len(ent))
    rr = {"fixed": "all", "one": "1", "three": "3"}.get(pr) or f"{np.mean(ra):.1f} ({min(ra):.1f}--{max(ra):.1f})"
    rows.append(f"{nm} & " + ("--" if pr == "fixed" else c3(np.mean(cr), min(cr))) + f" & {c3(np.mean(cn), min(cn))} & {c3(np.mean(cz), min(cz))} & " +
                ("1.00 & 1.00" if pr == "fixed" else f"{f2(wl)} & {f2(wp)}") + f" & {rr} \\\\" + ("\n\\midrule" if pr == "fixed" else ""))
allr = [x for pr in ("one", "three", "adaptive") for x in FGV[pr]["cr"]]; alln = [x for pr in ("one", "three", "adaptive") for x in FGV[pr]["cn"]]
assert len(allr) == FG["B88f_prime"]["n"] and abs(np.mean(allr) - FG["B88f_prime"]["mean_rv"]) < 1e-12 and min(allr) == FG["B88f_prime"]["min_rv"]
assert abs(np.mean(alln) - FG["H88f"]["mean"]) < 1e-12 and min(alln) == FG["H88f"]["min"]
table("freshgold", rows,
      caption=("When each gold label bought is a fresh judgment drawn from the comment's raters (seven cells of the 5% sample, 1,000 and 4,000 labels, "
               "500 audits): coverage of the law of the expected labels with Theorem 1(c)'s rater-variance term added to the interval and without it, "
               "coverage of the law of the labels actually drawn, the median width relative to fixed gold (laws; placebo slopes), and the raters bought "
               "per label (adaptive: mean over cells, range in parentheses). Coverage: mean, lowest in parentheses."),
      headers=["Gold process", "Coverage with rater variance", "without", "Coverage of the realized law", "Width / fixed: laws", "placebo slopes", "Raters per label"],
      src=f"{S_FG} cells.<labeler>.<1000|4000>.<process>.<metric>.{{cov_expected_rv, cov_expected, cov_realized, width_rv_vs_fixed}}, ._raters_adaptive",
      how="means and minima over 7 cells x 2 budgets x 5 metrics per process; widths: median of the stored width-with-rater-variance / fixed-gold width over laws (depth, thread size, article size) or placebo slopes",
      checks=f"pooled over the three fresh processes: {FG['B88f_prime']['n']} entries, mean {FG['B88f_prime']['mean_rv']:.4f} and minimum {FG['B88f_prime']['min_rv']} with the term (B88f_prime), mean {FG['H88f']['mean']:.4f} and minimum {FG['H88f']['min']} without (H88f) (asserted)",
      caveats="For fixed gold no rater variance applies ('--'); its coverage equals that of the realized law. Raters per label for one rater and three raters are the processes' definitions, not measurements.")
put("xcFgN", len(allr), i0, S_FG, "B88f_prime.n", "entries (3 processes x 7 cells x 2 budgets x 5 metrics)"); chk("xcFgN", "cFgN")
put("xcFgMean", float(np.mean(allr)), lambda x: pct(x, 1), S_FG, "cov_expected_rv over the fresh processes", "mean coverage with the rater-variance term"); chk("xcFgMean", "cFgMean", scale=100)
put("xcFgMin", min(allr), lambda x: pct(x, 1), S_FG, "same", "lowest entry"); chk("xcFgMin", "cFgMin", scale=100)
put("xcFgNoRvMean", float(np.mean(alln)), lambda x: pct(x, 1), S_FG, "cov_expected", "mean coverage without the term")
put("xcFgNoRvMin", min(alln), lambda x: pct(x, 1), S_FG, "same", "lowest entry without the term"); chk("xcFgNoRvMin", "cFgNoRvMin", scale=100)
wl_ = [e for e in ((l, B, pr, m) for l in FC_ for B in BUDF for pr in ("one", "three", "adaptive") for m in mets) if FC_[e[0]][e[1]][e[2]][e[3]]["cov_expected"] == min(alln)]
PRW = {"one": "one rater", "three": "a majority of three raters", "adaptive": "the adaptive process"}
MW = {"depth_law": "depth law", "size_law": "thread size law", "article_size_law": "article size law", "placebo_depth": "placebo depth slope", "placebo_size": "placebo size slope"}
put("xcFgNoRvMinMetric", MW[wl_[0][3]], None, S_FG, "argmin of cov_expected", "the target with the lowest coverage without the term", caveat="labeler " + wl_[0][0])
put("xcFgNoRvMinBudget", int(wl_[0][1]), comma, S_FG, "argmin of cov_expected", "its budget")
put("xcFgNoRvMinProcess", PRW[wl_[0][2]], None, S_FG, "argmin of cov_expected", "its gold process")
put("xcFgReal", FG["B88f_prime"]["mean_realized"], lambda x: pct(x, 1), S_FG, "B88f_prime.mean_realized", "mean coverage of the realized law"); chk("xcFgReal", "cFgReal", scale=100)
for pr, tg in (("one", "One"), ("three", "Three"), ("adaptive", "Adapt")):
    put(f"xcFgW{tg}", FGV[pr]["wl"], f2, S_FG, f"cells.*.*.{pr}.<law>.width_rv_vs_fixed", "median width relative to fixed gold, laws"); chk(f"xcFgW{tg}", f"cFgW{tg}")
    put(f"xcFgWPlac{tg}", FGV[pr]["wp"], f2, S_FG, f"cells.*.*.{pr}.<placebo>.width_rv_vs_fixed", "same, placebo slopes")
    put(f"xcFgMin{tg}", min(FGV[pr]["cr"]), lambda x: f"{x:.3f}", S_FG, f"cells.*.*.{pr}.*.cov_expected_rv", "lowest coverage with the term, this process")
    put(f"xcFgNoRvMin{tg}", min(FGV[pr]["cn"]), lambda x: f"{x:.3f}", S_FG, f"cells.*.*.{pr}.*.cov_expected", "lowest coverage without the term, this process")
put("xcFgRaters", float(np.mean(FGV["adaptive"]["ra"])), lambda x: f"{x:.1f}", S_FG, "cells.*.*._raters_adaptive", "mean raters per label, adaptive process"); chk("xcFgRaters", "cFgRaters")
put("xcFgRatersLo", min(FGV["adaptive"]["ra"]), lambda x: f"{x:.1f}", S_FG, "same", "smallest over cells and budgets"); put("xcFgRatersHi", max(FGV["adaptive"]["ra"]), lambda x: f"{x:.1f}", S_FG, "same", "largest")

# ======================================================================== M09: held-out classifier cells against the others
section("M09 held-out cells against the others")
FCJ = load("x86/x86_fresh.json"); S_FCJ = "results/x86/x86_fresh.json"; FRESH = ("sn_rob", "mh_dist", "fb_hate", "cn_off")
isf = lambda t: t.split("__")[1].split("_at_")[0] in FRESH
for grp, sel_ in (("fresh", True), ("others", False)):
    si = {t: c for t, c in I.items() if isf(t) == sel_}; se = {t: c for t, c in E.items() if isf(t) == sel_}
    for fam, cells_, ms, bmin, des in (("laws", si, LAWS_I, 1000, ("ppi", "dsl", "ai", "upgrade_lw", "srs")), ("placebo", si, PLAC_I, 1000, ("ppi", "dsl", "ai", "upgrade_lw", "srs")),
                                       ("gold_rr", se, ["log_rr"], 2000, ("ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs")), ("placebo_rr", se, ["log_placebo_rr"], 2000, ("ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs"))):
        fin = V3I if fam in ("laws", "placebo") else V3E; c_ = covs(cells_, fin, ms, bmin); ref = FCJ[grp][fam]
        assert c_[2] == ref["n"] and abs(c_[0] - ref["cov_mean"]) < 1e-12 and abs(c_[1] - ref["cov_min"]) < 1e-12, (grp, fam)
        for d_ in des:
            assert abs(med(ratios(cells_, fin, d_, ms, bmin)) - ref["ratio"][d_]) < 1e-12, (grp, fam, d_)
for grp, tg in (("fresh", "Fc"), ("others", "Oc")):
    for fam, ft in (("laws", "Laws"), ("placebo", "Plac"), ("gold_rr", "Gold"), ("placebo_rr", "PlacRR")):
        r_ = FCJ[grp][fam]
        put(f"xc{tg}Cov{ft}", r_["cov_mean"], lambda x: f"{x:.3f}", S_FCJ, f"{grp}.{fam}.cov_mean", f"Threadsafe's mean coverage, {'the 28 held-out classifier cells' if grp == 'fresh' else 'the other 27 cells'}")
        put(f"xc{tg}Min{ft}", r_["cov_min"], lambda x: f"{x:.3f}", S_FCJ, f"{grp}.{fam}.cov_min", "lowest entry")
        put(f"xc{tg}Ppi{ft}", r_["ratio"]["ppi"], f2, S_FCJ, f"{grp}.{fam}.ratio.ppi", "median width ratio to PPI++")
    put(f"xc{tg}Cells", FCJ[grp]["laws"]["n_cells"], i0, S_FCJ, f"{grp}.laws.n_cells", "cells",
        caveat="" if grp == "fresh" else "the other 27 confirmation cells: their labelers (LLMs, Detoxify, IMSYPP) also appear in the 17 development cells, but these cells were not used in development")
for a_, o_ in (("xcFcCovLaws", "qFcCov"), ("xcFcMinLaws", "qFcMin"), ("xcFcPpiLaws", "qFcPPI"), ("xcOcPpiLaws", "qOcPPI"), ("xcFcPpiPlac", "qFcPlacPPI"), ("xcOcPpiPlac", "qOcPlacPPI"), ("xcFcCells", "vNfresh")):
    chk(a_, o_)
chk("xcFcCovLaws", "eFcCov", scale=100)
put("xcFcAiLaws", FCJ["fresh"]["laws"]["ratio"]["ai"], f2, S_FCJ, "fresh.laws.ratio.ai", "median width ratio to active inference, held-out cells"); chk("xcFcAiLaws", "qFcAI")
put("xcOcAiLaws", FCJ["others"]["laws"]["ratio"]["ai"], f2, S_FCJ, "others.laws.ratio.ai", "the same, other cells"); chk("xcOcAiLaws", "qOcAI")

# ======================================================================== M11-M12: labels needed to detect a distortion (n80); the edge design on CAD and BAD
section("M11 labels needed to detect a distortion (n80)")
S_N80 = "results/x11_conf_item_all, x11_conf_edge_all (without CGA-Wiki), x72_item_cga (loaded as x13_tables.py)"


def load_n80():
    """as scripts/x13_tables.py with --exclude-pop cga and the item rerun last (results/x72_eval/tables_H72.md)"""
    C = {}
    for d in (f"{RES}/x11_conf_item_all", f"{RES}/x11_conf_edge_all", f"{RES}/x72_item_cga"):
        for fn in sorted(glob.glob(f"{d}/*__*.json")):
            if os.path.basename(fn).startswith("eval"):
                continue
            R0 = json.load(open(fn))
            for r in R0["results"]:
                te = r.get("truth_edge") or {}; ti = r.get("truth_item") or {}; li = r.get("labeler_item") or {}; le = r.get("labeler_edge") or {}
                for reg in ("item", "edge"):
                    if reg == "edge" and R0["pop"] == "cga":
                        continue
                    for B, dd in (r.get(reg) or {}).items():
                        for des, mm in dd.items():
                            for m, c in mm.items():
                                if m == "_labels":
                                    continue
                                C[(R0["pop"], r["labeler"], reg, m, int(B), des)] = dict(c, truth=(ti.get(m) if reg == "item" else te.get("rr" if m == "log_rr" else "placebo_rr")),
                                                                                       lab_value=(li.get(m) if reg == "item" else (le.get("rr") if m == "log_rr" else None)))
    return C


CN = load_n80(); N80LAWS = ("depth_law", "size_law", "article_size_law", "log_rr")
N80 = {}
for key4 in sorted({k[:4] for k in CN if k[3] in N80LAWS}):
    fin = V3I if key4[2] == "item" else V3E; Bs = sorted({k[4] for k in CN if k[:4] == key4})
    def n80(des):
        for B in Bs:
            c = CN.get(key4 + (B, des))
            if c and c.get("detect") is not None and c["detect"] >= 0.8:
                return B
        return None
    v = {"ts": n80(fin), "v2": n80("upgrade_lw"), "ppi": n80("ppi"), "srs": n80("srs")}
    c0 = CN.get(key4 + (Bs[0], fin)) or next(CN[k] for k in CN if k[:4] == key4)
    N80[key4] = dict(v, Bmax=Bs[-1], lab=c0.get("lab_value"), truth=c0.get("truth"), any=any(x is not None for x in v.values()))
MDN = {"depth_law": "Depth law", "size_law": "Thread size law", "article_size_law": "Article size law", "log_rr": "Reply risk ratio"}
fmt_md = lambda x, Bm: f"{x:,}" if x else f">{Bm:,}"
mine_md = set()
for (pop, lab, reg, m), v in N80.items():
    if v["any"]:
        lv, gv = v["lab"], v["truth"]
        lvs = "" if lv is None else (f"{lv:+.4f}" if reg == "item" else f"{lv:.2f}"); gvs = "" if gv is None else (f"{gv:+.4f}" if reg == "item" else f"{gv:.2f}")
        mine_md.add(f"| {pop} | {lab} | {MDN[m]} | {lvs} | {gvs} | {fmt_md(v['ts'], v['Bmax'])} | {fmt_md(v['v2'], v['Bmax'])} | {fmt_md(v['ppi'], v['Bmax'])} | {fmt_md(v['srs'], v['Bmax'])} |")
stored_md = {l.strip() for l in open(f"{RES}/x72_eval/tables_H72.md") if l.startswith("| ") and l.count("|") == 10 and not l.startswith("| Population")}
assert mine_md == stored_md, (len(mine_md), len(stored_md), sorted(mine_md ^ stored_md)[:4])     # reproduces the stored n80 table row for row
inf = lambda x, v: x if x is not None else 2 * v["Bmax"]
ABANDONED = {("cc_threads", "qwen7b_m_ctx")}                                                    # Qwen shown the parent at the no-context threshold (21.9% flagged)


def n80_counts(reg, drop=frozenset()):
    rr_ = [v for k, v in N80.items() if k[2] == reg and v["any"] and (k[0], k[1]) not in drop]
    return dict(rows=len(rr_), better=sum(inf(v["ts"], v) < inf(v["v2"], v) for v in rr_), worse=sum(inf(v["ts"], v) > inf(v["v2"], v) for v in rr_),
                bppi=sum(inf(v["ts"], v) < inf(v["ppi"], v) for v in rr_), wppi=sum(inf(v["ts"], v) > inf(v["ppi"], v) for v in rr_),
                bytwok=sum(1 for v in rr_ if v["ts"] is not None and v["ts"] <= 2000))


for reg, tg, P_ in (("item", "Item", "vNItem"), ("edge", "Edge", "vNEdge")):
    a_ = n80_counts(reg); x_ = n80_counts(reg, ABANDONED)
    for k_, nm_, o_ in (("rows", "Rows", "Rows"), ("better", "Better", "Better"), ("worse", "Worse", "Worse"), ("bppi", "BetterPpi", "BetterPPI"), ("wppi", "WorsePpi", "WorsePPI"), ("bytwok", "ByTwoK", "ByTwoK")):
        put(f"xcNeighty{tg}{nm_}All", a_[k_], i0, S_N80, f"n80 rows ({reg})", f"as stored, all cells: {k_}", caveat="includes the cell at an abandoned threshold")
        chk(f"xcNeighty{tg}{nm_}All", P_ + o_)
        put(f"xcNeighty{tg}{nm_}", x_[k_], i0, S_N80, f"n80 rows ({reg})", f"without Qwen shown the parent at the no-context threshold: {k_}",
            caveat="rows: (cell, law) pairs in which some design reaches 80% detection within the budget grid; better/worse: Threadsafe needs fewer/more labels than the stratified design (not reached = twice the largest budget, as x15)")
        if a_[k_] != x_[k_]:
            note(f"xcNeighty{tg}{nm_}", f"differs from the stored \\{P_ + o_} = {OLD[P_ + o_]} because the abandoned-threshold cell is dropped")
newsRR = [v["ts"] for k, v in N80.items() if k[2] == "edge" and k[0].startswith("cc") and v["ts"] and (k[0], k[1]) not in ABANDONED]
newsRRa = [v["ts"] for k, v in N80.items() if k[2] == "edge" and k[0].startswith("cc") and v["ts"]]
put("xcNeightyNewsRRMinAll", min(newsRRa), comma, S_N80, "n80 of edge_m3_c15, log_rr, cc_threads and cc_s5 cells", "as stored: fewest labels at which Threadsafe detects a contagion distortion on news comments")
chk("xcNeightyNewsRRMinAll", "qEdgeNewsMin")
put("xcNeightyNewsRRMin", min(newsRR), comma, S_N80, "same, without the abandoned-threshold cell", "fewest labels")
chk("xcNeightyNewsRRMin", "qEdgeNewsMin", why=f"the stored minimum, {OLD['qEdgeNewsMin'].replace('{,}', ',')}, is Qwen shown the parent at its no-context threshold ({QFs} flagged), an abandoned operating point")
put("xcNeightyNewsRRMax", max(newsRR), comma, S_N80, "same", "most labels"); chk("xcNeightyNewsRRMax", "qEdgeNewsMax")
v7f("xcNeightyNewsRRMin", "Contagion is expensive to certify (2{,}000--16{,}000 labels on news comments)", same=False,
    why="the 2,000 is the abandoned-threshold Qwen cell; without it Threadsafe needs 4,000-16,000 labels (the 4,000 being Haiku shown the parent, whose law pools two labeling modes)")
wr_ = sorted((k[0], k[1]) for k, v in N80.items() if k[2] == "edge" and v["any"] and inf(v["ts"], v) > inf(v["v2"], v))
put("xcNeightyEdgeWorseWhere", "; ".join(f"{PNAME[p]} {NICE[l]}" for p, l in wr_), None, S_N80, "edge rows with Threadsafe slower than stratified", "where the stratified edge design detects contagion sooner")
for (pop, lab, m, tg) in (("cc_threads", "haiku_ctx", "depth_law", "HaikuCtxDepth"), ("bad", "imsypp@m", "size_law", "BadImsSize"), ("cc_s5", "sonnet", "depth_law", "SonnetDepth"),
                          ("cc_threads", "imsypp@m", "depth_law", "ImsDepth"), ("cad", "haiku", "log_rr", "CadHaikuRR")):
    v = N80[(pop, lab, "edge" if m == "log_rr" else "item", m)]
    put(f"xcNeighty{tg}Ts", v["ts"] if v["ts"] else 2 * v["Bmax"], lambda x, v=v: comma(x) if x <= v["Bmax"] else f"more than {comma(v['Bmax'])}", S_N80, f"{pop} {lab} {m}", "Threadsafe's n80")
    put(f"xcNeighty{tg}Strat", v["v2"] if v["v2"] else 2 * v["Bmax"], lambda x, v=v: comma(x) if x <= v["Bmax"] else f"more than {comma(v['Bmax'])}", S_N80, f"{pop} {lab} {m}", "the stratified design's n80",
        caveat="Haiku shown the parent's law pools roots labeled without the parent with replies labeled with it" if CTX(lab) else "")
CELLS = sorted({(k[0], k[1]) for k in N80}, key=lambda t: ([p for p, _ in POPS].index(t[0]), (["haiku", "haiku_ctx", "sonnet", "sonnet_ctx", "qwen7b_m", "qwen7b_m_ctx", "gemma_m"] + CLS14).index(t[1])))
gfmt = lambda x, Bm: comma(x) if x else f"${{>}}${comma(Bm)}"
rows = []; prev = None; nrow = 0
for pop, lab in CELLS:
    if (pop, lab) in ABANDONED:
        continue
    cells_ = []; anyr = False
    for m, reg in (("depth_law", "item"), ("size_law", "item"), ("article_size_law", "item"), ("log_rr", "edge")):
        v = N80.get((pop, lab, reg, m))
        if v is None:
            cells_ += ["--", "--"]; continue
        anyr |= v["any"]; cells_ += [gfmt(v["ts"], v["Bmax"]), gfmt(v["v2"], v["Bmax"])]
    if not anyr:
        continue
    nm_ = (NICE[lab].replace(", parent shown", " + parent") if CTX(lab) else NICE[lab] + (" (gold's rate)" if lab.endswith("@m") else ""))   # arXiv revision: defined labels
    if prev is not None and pop != prev:
        rows.append("\\addlinespace[2pt]")
    rows.append(f"{PNAME[pop] if pop != prev else ''} & {nm_} & " + " & ".join(cells_) + " \\\\"); prev = pop; nrow += 1
table("n80", rows,
      caption=("Labels needed to detect a labeler's distortion of a law: the smallest budget (250, 500, 1,000, ... labels for laws; 500 to 16,000 "
               "for the reply risk ratio; to 2,000 on CGA-Wiki) at which at least 80% of 500 audits give an interval that excludes the labeler's own "
               "value of the law, for Threadsafe and the stratified design. '>': not reached at the largest budget; '--': the law is not audited there. "
               "Rows: the audit cells in which some design reaches 80% for some law. $^{\\ddagger}$Replies labeled with their parent and roots without "
               "it; the audited law pools both modes."),
      headers=["Population", "Labeler", "Depth law: Threadsafe", "Stratified", "Thread size law: Threadsafe", "Stratified", "Article size law: Threadsafe", "Stratified",
               "Reply RR: Threadsafe", "Stratified"],
      src=S_N80 + ": <run>.results[].{item,edge}.<budget>.<design>.<law>.detect (share of audits whose interval excludes labeler_item / labeler_edge)",
      how="n80 = the first budget whose stored detection share is at least 0.80, as x13_tables.py / x15_paper_numbers.py",
      checks="the per-(cell, law) n80 values reproduce results/x72_eval/tables_H72.md row for row (asserted); counts reproduce vNItemRows/Better/Worse and vNEdgeRows/Better/Worse (39 of 78, 0; 9 of 41, 5)",
      caveats=f"The earlier-draft n80 table (paper_v5/tables/n80.tex) is partly stale: its 'Qwen + parent' contagion row is the audit cell of Qwen shown the parent at the no-context threshold ({QFs} flagged), an abandoned operating point; that cell is dropped here."
              " Its 'Haiku + parent' rows are kept but relabeled (‡): the law they test pools roots labeled without the parent and replies labeled with it, not the among-replies law. "
              "The stored audits target the pooled laws, so these rows cannot be recomputed among replies from stored results.")
put("xcNeightyTabRows", nrow, i0, S_N80, "xc_n80 rows", "cells shown in the n80 table")
section("M12 the edge design on CAD and BAD")
for pop, tg in (("cad", "Cad"), ("bad", "Bad")):
    se = {t: c for t, c in E.items() if t.startswith(pop + "__")}
    put(f"xcEdge{tg}Unif", med(ratios(se, V3E, "srs", ["log_rr"], 2000)), f2, S_RAW, f"{pop} cells, log_rr, B >= 2,000", "median width of Threadsafe's reply-RR interval divided by uniform sampling's")
    put(f"xcEdge{tg}Strat", med(ratios(se, V3E, "upgrade_lw", ["log_rr"], 2000)), f2, S_RAW, f"{pop} cells, log_rr, B >= 2,000", "the same against the stratified design")
    put(f"xcEdge{tg}UnifPlac", med(ratios(se, V3E, "srs", ["log_placebo_rr"], 2000)), f2, S_RAW, f"{pop} cells, log_placebo_rr", "placebo risk ratio against uniform")
oth = {t: c for t, c in E.items() if not t.startswith(("cad__", "bad__"))}
put("xcEdgeOtherUnif", med(ratios(oth, V3E, "srs", ["log_rr"], 2000)), f2, S_RAW, "cells of the other populations, log_rr", "median width ratio to uniform elsewhere")
note("xcEdgeCadUnif", "registration Addendum 72 says 'its median width there is 1.00-1.08x uniform sampling's'")

# ======================================================================== M13: leaving one population out
section("M13 leaving one population out")
LO = load("x80/x80_lopo_registered.json"); S_LO = "results/x80/x80_lopo_registered.json"
for reg, tg in (("item", "Item"), ("edge", "Edge")):
    L_ = LO[reg]; assert L_["n_frozen_qualifies"] == L_["n_runs"]
    put(f"xcLopo{tg}N", L_["n_frozen_adopted"], i0, S_LO, f"{reg}.n_frozen_adopted", "runs, each leaving one population out of the development cells, whose lowest weighted median width is the fixed design")
    chk(f"xcLopo{tg}N", f"qLopo{tg}N")
    put(f"xcLopo{tg}Runs", L_["n_runs"], i0, S_LO, f"{reg}.n_runs", "runs (one per population with development cells)"); chk(f"xcLopo{tg}Runs", f"qLopo{tg}Runs")
    put(f"xcLopo{tg}Max", L_["max_frozen_over_adopted"] - 1, lambda x: pct(x, 1), S_LO, f"{reg}.max_frozen_over_adopted", "how much wider the fixed design is than the narrowest variant, at most")
    chk(f"xcLopo{tg}Max", f"qLopo{tg}Max", scale=100)
    ch_ = [p for p, v in L_["runs"].items() if p != "none" and v["adopted"] != L_["frozen"]]
    LST = {"cc_threads": "Civil Comments", "cc_s5": "its 5\\% sample", "cad": "CAD", "qian_reddit": "Qian et al.'s Reddit", "qian_gab": "Qian et al.'s Gab", "cga": "CGA-Wiki", "bad": "BAD"}
    nm_ = [LST[p] for p in ch_]
    put(f"xcLopo{tg}Changed", (", ".join(nm_[:-1]) + " and " + nm_[-1]) if len(nm_) > 1 else nm_[0], None, S_LO, f"{reg}.runs.<pop>.adopted", "populations whose omission changes the narrowest variant")
put("xcLopoEdgeNone", LO["edge"]["runs"]["none"]["frozen_over_adopted"] - 1, lambda x: pct(x, 1), S_LO, "edge.runs.none.frozen_over_adopted", "with every population, how much wider the fixed edge design is than the narrowest edge variant of rounds E1-E5")
chk("xcLopoEdgeNone", "qLopoEdgeNone", scale=100)
put("xcLopoEdgeNoneVariant", LO["edge"]["runs"]["none"]["adopted"].replace("_", "\\_"), None, S_LO, "edge.runs.none.adopted", "that variant (label sharing, five waves)")
SKIP.append("M13: the unit-share regime's leave-one-out runs (qLopoUnitN = 7 of 7) are not reported: the unit-share regime (M16) stays out of the paper.")

# ======================================================================== M14: distance to an oracle
section("M14 distance to an oracle")
GA5, GA7, GE4 = (load(f"x15_paper/{n}.json") for n in ("gap_rA5", "gap_rA7", "gap_rE4")); S_GAP = "results/x15_paper/gap_rA5.json, gap_rA7.json, gap_rE4.json (x10_gap.py outputs)"


def gmed(rows_, Iv, des, metrics, excl_cga=False):
    v = [r["eff"] for r in rows_ if r["I"] == Iv and r["design"] == des and r["metric"] in metrics and not (excl_cga and r["cell"].startswith("cga__"))]
    return float(np.median(v))


lawsI = ("depth_law", "size_law")
for nm, rows_, des, old in (("Ts", GA5, "pps_seq4_ba", "cCeilLaws"), ("Strat", GA5, "upgrade_lw", "cCeilLawsVtwo"), ("Or", GA7, "pps_seq3_bc10_or", "cOrLaws"), ("Twin", GA7, "pps_seq3_bc10", "cTwinLaws")):
    v = [gmed(rows_, "I4", des, (m,)) for m in lawsI]
    put(f"xcCeil{nm}LawsMin", min(v), f2, S_GAP, f"rows[I=I4, design={des}, metric in depth/size law].eff", "median ceiling efficiency (oracle variance / design's MSE) over development cells and budgets: the smaller of the depth and size laws")
    put(f"xcCeil{nm}LawsMax", max(v), f2, S_GAP, "same", "the larger"); chk(f"xcCeil{nm}LawsMin", old + "Min"); chk(f"xcCeil{nm}LawsMax", old + "Max")
for nm, rows_, des, old in (("Ts", GA5, "pps_seq4_ba", "cCeilArt"), ("Strat", GA5, "upgrade_lw", "cCeilArtVtwo"), ("Or", GA7, "pps_seq3_bc10_or", "cOrArt")):
    put(f"xcCeil{nm}Art", gmed(rows_, "I4", des, ("article_size_law",)), f2, S_GAP, f"rows[I=I4, design={des}, metric=article_size_law].eff", "the same for the article size law"); chk(f"xcCeil{nm}Art", old)
v = [gmed(GA5, "I4", "pps_seq4_ba", (m,)) for m in PLAC_I]
put("xcCeilTsPlacMin", min(v), f2, S_GAP, "rows[I=I4, pps_seq4_ba, placebo slopes]", "placebo slopes: smaller"); chk("xcCeilTsPlacMin", "cCeilPlacMin")
put("xcCeilTsPlacMax", max(v), f2, S_GAP, "same", "larger"); chk("xcCeilTsPlacMax", "cCeilPlacMax")
for nm, des in (("Ts", "edge_m3_c15"), ("Strat", "upgrade_lw"), ("Or", "edge_m3_c25_or"), ("Twin", "edge_m3_c25")):
    old = {"Ts": "cEdgeCeil", "Strat": "cEdgeCeilVtwo", "Or": "cEdgeOr", "Twin": "cEdgeTwin"}[nm]
    put(f"xcCeil{nm}Gold", gmed(GE4, "I4m", des, ("log_rr",), True), f2, S_GAP, f"gap_rE4 rows[I=I4m, design={des}, log_rr], CGA-Wiki excluded", "edge ceiling efficiency, reply risk ratio"); chk(f"xcCeil{nm}Gold", old + "Gold")
    put(f"xcCeil{nm}PlacRR", gmed(GE4, "I4m", des, ("log_placebo_rr",), True), f2, S_GAP, "same, log_placebo_rr", "placebo risk ratio"); chk(f"xcCeil{nm}PlacRR", old + "Plac")
note("xcCeilTsGold", "Addendum 70 (registration) prints 0.70 for this efficiency computed with CGA-Wiki's development cells; x15 and this script exclude CGA-Wiki (self-reply correction, Addendum 72)")
A7 = load("x10_dev_rA7/eval_rA7.json")["summary"]; E4 = load("x10_dev_rE4/eval_rE4.json")["summary"]; E5 = load("x14_dev_rE5/eval_E5_pooled_nocga.json")["summary"]
put("xcLearnItems", (A7["pps_seq3_bc10_or"]["wmed"] / A7["pps_seq3_bc10"]["wmed"]) ** 2, f2, "results/x10_dev_rA7/eval_rA7.json", "summary.<design>.wmed",
    "(weighted median width of the oracle-model twin / the learned design)^2: the share of the variance left when the working model is known instead of learned (items)",
    caveat="measured on the sibling design pps_seq3_bc10 and its oracle twin, not on the fixed pps_seq4_ba")
put("xcLearnEdges", (E4["edge_m3_c25_or"]["wmed"] / E4["edge_m3_c25"]["wmed"]) ** 2, f2, "results/x10_dev_rE4/eval_rE4.json", "summary.<design>.wmed", "the same for edges (edge_m3_c25 and its oracle-law twin)",
    caveat="sibling design edge_m3_c25, not the fixed edge_m3_c15")
put("xcStopGainEfour", 1 - E4["edge_m3_c15"]["wmed"] / E4["edge_m3_c25"]["wmed"], lambda x: pct(x, 1), "results/x10_dev_rE4/eval_rE4.json", "summary.edge_m3_c15.wmed / summary.edge_m3_c25.wmed",
    "gain of round E4's best edge variant over the previous leader")
put("xcStopGainEfive", 1 - E5["edge_m5_c15_sh"]["wmed"] / E5["edge_m3_c15"]["wmed"], lambda x: pct(x, 1), "results/x14_dev_rE5/eval_E5_pooled_nocga.json", "summary.edge_m5_c15_sh.wmed / summary.edge_m3_c15.wmed",
    "gain of round E5's best variant (label sharing) over the fixed edge design in the same run")
SKIP.append("M14: the stopping bar (two consecutive rounds gaining less than 3%) is a rule stated in the registration log, not a stored result; no macro.")
SKIP.append("M14: the model-class factor (logistic against gradient boosting, about 0.86-0.96) and the item rounds' gains (A6, A7: 0%) are prose in the registration log, not fields of a stored result read here; no macro.")
SKIP.append("M14/M15: label sharing in the edge regime (round E5) appears only as the 2.2% gain (xcStopGainEfive); it was not adopted.")
for pop, tg in (("cad", "Cad"), ("bad", "Bad")):
    se = {t: c for t, c in E.items() if t.startswith(pop + "__")}
    bb_ = [med(ratios(se, V3E, "srs", ["log_rr"], B, B)) for B in (2000, 4000, 8000, 16000)]; ba_ = [med(ratios(se, V3E, "srs", ["log_rr"], B, B)) for B in (500, 1000, 2000, 4000, 8000, 16000)]
    put(f"xcEdge{tg}UnifLo", min(bb_), f2, S_RAW, f"{pop}, log_rr, median per budget 2,000-16,000", "smallest per-budget median width ratio to uniform")
    put(f"xcEdge{tg}UnifHi", max(bb_), f2, S_RAW, "same", "largest", caveat=f"with 500 and 1,000 labels included the per-budget medians run {min(ba_):.2f}-{max(ba_):.2f}")
    note(f"xcEdge{tg}UnifHi", f"the registration (Addendum 72) gives 1.00-1.08 for CAD and BAD together; the confirmation runs give {min(bb_):.2f}-{max(bb_):.2f} here at 2,000-16,000 and {min(ba_):.2f}-{max(ba_):.2f} at 500-16,000")
DISC.append(f"M12: the registration's 'median width 1.00-1.08x uniform sampling's' on CAD and BAD is not reproduced from the confirmation runs at 2,000-16,000 labels: "
            f"CAD {M['xcEdgeCadUnif']} ({M['xcEdgeCadUnifLo']}-{M['xcEdgeCadUnifHi']} by budget), BAD {M['xcEdgeBadUnif']} ({M['xcEdgeBadUnifLo']}-{M['xcEdgeBadUnifHi']}); the substance (no gain) holds")
# half-widths of Threadsafe's depth-law interval on Civil Comments reply threads (the earlier draft's 'What a certificate costs' paragraph)
section("M11 half-widths of the depth-law interval (what a certificate costs)")
MGh = MG["half_width_depth_cc_threads"]
for B, tg in BW.items():
    hw = float(np.median([C[int(B)][V3I]["depth_law"]["width"] / 2 for t, C in I.items() if t.startswith("cc_threads__") and int(B) in C]))
    assert abs(hw - MGh[B]) < 1e-15, B
    put(f"xcHwDepth{tg}", 100 * hw, f2, S_RAW + f"; equals {S_MG} half_width_depth_cc_threads", f"cc_threads cells, pps_seq4_ba, B = {B}, depth_law.width / 2",
        "x100: median over the Civil Comments reply-thread cells of half of Threadsafe's mean interval width (points per level)")
chk("xcHwDepthTwoFifty", "qHwDepthLo"); chk("xcHwDepthOneK", "qHwDepthMid"); chk("xcHwDepthEightK", "qHwDepthHi")
v7f("xcHwDepthTwoFifty", "is 1.34 points per level at 250 labels, 0.54 at 1{,}000 and 0.16 at 8{,}000")
hw_all = {m: {B: float(np.median([C[int(B)][V3I][m]["width"] / 2 for C in I.values() if int(B) in C and V3I in C[int(B)] and m in C[int(B)][V3I] and C[int(B)][V3I][m].get("width")]))
              for B in BW} for m in ITEM_M}
dif = {(m, B): (MG["half_widths"]["item"][m][B], hw_all[m][B]) for m in ITEM_M for B in BW if abs(MG["half_widths"]["item"][m][B] - hw_all[m][B]) > 1e-12}
if dif:
    DISC.append(f"x80_magnitude.py loads results/x72_item_cga/cga__all7.json as one bogus cell and keeps CGA-Wiki's uncorrected item cells: {len(dif)} all-population half-widths in x80_magnitude.json half_widths.item shift slightly "
                f"(largest relative change {max(abs(a / b - 1) for a, b in dif.values()):.1%}); the Civil Comments depth-law half-widths used in v7f are unaffected")

# ======================================================================== the figure: kappa against the depth-law distortion
section("Figure kappa_distortion")
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"; C_LLM, C_CLS = "#eb6834", "#2a78d6"     # p20_fig_decomp.py's palette
SHORT = {"cc_threads": "Civil Comments", "cc_s5": "Civil Comments,\n5% sample", "cad": "CAD", "qian_reddit": "Qian et al.'s\nReddit", "qian_gab": "Qian et al.'s\nGab", "cga": "CGA-Wiki", "bad": "BAD"}
fig, axes = plt.subplots(1, len(POPS), figsize=(7.0, 2.6), sharey=True, facecolor=SURF)
ylo = min(p[3] for v in FIGDATA.values() for p in v["pts"]); yhi = max(p[4] for v in FIGDATA.values() for p in v["pts"]); pad = 0.06 * (yhi - ylo)
for ax, (pop, _) in zip(axes, POPS):
    v = FIGDATA[pop]; ax.set_facecolor(SURF)
    ax.axhline(0, color=INK2, lw=0.7, zorder=2)
    ax.axhline(-v["gold"], color=INK2, lw=0.7, ls=(0, (1.2, 1.6)), zorder=2)                     # where the labeler's law would be 0
    for llm, col, mk, ms in ((False, C_CLS, "s", 3.8), (True, C_LLM, "o", 4.4)):
        q = [p for p in v["pts"] if p[5] == llm]
        if not q:
            continue
        x = np.array([p[1] for p in q]); y = np.array([p[2] for p in q]); lo = np.array([p[3] for p in q]); hi = np.array([p[4] for p in q])
        ax.vlines(x, lo, hi, color=col, lw=0.7, alpha=0.65, zorder=3)
        ax.plot(x, y, ls="", marker=mk, ms=ms, mfc=col, mec=SURF, mew=0.4, zorder=4)
    for L, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
        q = [p for p in v["pts"] if p[0] == L]
        if q and pop in ("cc_threads", "cc_s5") and (L == "haiku") == (pop == "cc_threads"):
            ax.annotate(nm, (q[0][1], q[0][2]), xytext=(4, 1), textcoords="offset points", fontsize=7, color=INK2, va="center")
    ax.set_xlim(-0.03, 1.0); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1" if pop == POPS[-1][0] else ""])   # only the last panel labels 1, so neighbours' labels never touch
    tl_ = ax.get_xticklabels(); tl_[0].set_ha("left"); tl_[-1].set_ha("right")
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_title(f"{SHORT[pop]}\n$\\rho$ = " + f"{v['rho']:+.2f}".replace("-", "\u2212"), fontsize=7.5, color=INK, pad=3)   # true minus in the value only (CGA-Wiki keeps its hyphen)
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.tick_params(colors=INK2, labelsize=7, pad=1.5, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("Depth-law distortion\n(points per reply level)", fontsize=7.5, color=INK2)
fig.text(0.5, 0.098, "Cohen's $\\kappa$ with gold", ha="center", va="center", fontsize=7.5, color=INK2)
from matplotlib.lines import Line2D
hh = [Line2D([], [], ls="", marker="o", ms=4.4, mfc=C_LLM, mec=SURF, label="LLM setting"), Line2D([], [], ls="", marker="s", ms=3.8, mfc=C_CLS, mec=SURF, label="classifier setting"),
      Line2D([], [], color=INK2, lw=0.7, label="no distortion"), Line2D([], [], color=INK2, lw=0.7, ls=(0, (1.2, 1.6)), label="labeler's law = 0"),
      Line2D([], [], color=INK2, lw=0.7, alpha=0.65, label="95% CI")]
fig.legend(handles=hh, loc="lower center", ncol=5, frameon=False, fontsize=7, labelcolor=INK2, handlelength=1.4, columnspacing=1.6, borderaxespad=0.1)
fig.tight_layout(rect=(0, 0.135, 1, 1), w_pad=0.45, pad=0.25)
fig.savefig(OUT_FIG, facecolor=SURF, metadata={"CreationDate": None}); fig.savefig(PREVIEW, dpi=200, facecolor=SURF); plt.close(fig)
npts = sum(len(v["pts"]) for v in FIGDATA.values())
put("xcFigNPts", npts, i0, S_DEC, "labelers without the parent, every population", "points in the kappa-distortion figure")
put("xcFigNPanels", len(POPS), i0, S_DEC, "populations", "panels")
FIG = dict(path=OUT_FIG, panels=[SHORT[p] for p, _ in POPS], n=npts,
           caption=("Agreement does not order labelers by how much they distort a law. Each panel is one population; each point is an LLM setting (orange "
                    "circles) or a classifier setting (blue squares) without the parent, at its kappa with gold (x) and its distortion of the depth law, "
                    "its law minus gold's in points per reply level, with the 95% bootstrap interval (y). The solid line marks no distortion; the dotted "
                    "line marks the distortion at which the labeler's law would be 0 (minus gold's law), so a point beyond it reverses gold's sign. rho: "
                    "the Spearman correlation between kappa and the absolute distortion in the panel. Labelers shown the parent are not plotted: their "
                    "depth laws pool roots labeled without the parent with replies labeled with it."),
           src=f"{S_DEC}: <pop>.labelers.<L>.kappa[0] and .laws.depth.distortion (point, 2.5% and 97.5%); rho as xcRho*",
           units="points per level (the stored distortion x100), chosen over gold standard errors because it is stored directly with its interval and a single y scale can be shared across populations "
                 f"(range {ylo:+.2f} to {yhi:+.2f}); in gold standard errors the Civil Comments panel would span about 100 units and CAD about 3, so no shared scale would be readable")

# ======================================================================== write the macros, and check the earlier-draft tables this script regenerates
# ======================================================================== arXiv revision, review round 1
section("arXiv revision R1")
gd_ = [DEC_T[(s_, "depth")]["g"] for s_, _, _ in SZ]
put("xcDcDepthGoldMin", min(gd_), pct, S_DC, "1 - gold_verdicts.depth['0'] over the four sample sizes", "smallest share of samples in which gold's depth law is significant")
put("xcDcDepthGoldMax", max(gd_), pct, S_DC, "same", "largest share")
put("xcQwenCtxOldFlag", QF / 100, lambda v_: f"{100 * v_:.1f}\\%", S_DEC, "cc_threads.labelers.qwen7b_m_ctx.flag_rate", "the flag rate of Qwen shown the parent at the earlier (no-context) threshold, the evaluation cell kept in coverage and width")
put("xcPubDuSixRev", Mt["item"]["reversed"]["detoxify_unbiased@0.6"], i0, S_PB, "metrics.item.reversed['detoxify_unbiased@0.6']", "decided pairs whose order Detoxify-unbiased at 0.6 reverses")
WORDS_ = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
put("xcPubDuSixRevWord", WORDS_[Mt["item"]["reversed"]["detoxify_unbiased@0.6"]], None, S_PB, "metrics.item.reversed['detoxify_unbiased@0.6']", "the same as a word")
put("xcPubQwenOwnRev", Mt["item"]["reversed"]["qwen7b"], i0, S_PB, "metrics.item.reversed.qwen7b", "decided pairs whose order Qwen's own answer reverses")
put("xcCorrDefN", CX["all"]["n_ws"], i0, S_CO, "entries with a defined population-level correction (corrected_pop_minus_gold not None), without pooled '+ parent' entries", "denominator of xcCorrShareWorse")
sep_ = {k: v["ts"] for k, v in N80.items() if k[2] == "edge" and k[0].startswith("cc") and v["ts"] and (k[0], k[1]) not in ABANDONED and not k[1].endswith("_ctx")}
put("xcNeightyNewsRRMinSep", min(sep_.values()), comma, S_N80, "n80 of the edge design on cc_threads and cc_s5 cells, without the abandoned cell and without '+ parent' cells (whose audited law pools roots without the parent and replies with it)",
    "fewest labels at which Threadsafe detects a contagion distortion on news comments: " + ", ".join(f"{k[0]}/{k[1]}" for k, v in sep_.items() if v == min(sep_.values())))
ne_ = sorted({(k[0], k[1]) for k in N80 if k[2] == "edge" and k[0].startswith("cc") and (k[0], k[1]) not in ABANDONED and not k[1].endswith("_ctx")})
nr_ = [c for c in ne_ if N80[c + ("edge", "log_rr")]["ts"] is None]
put("xcNeightyNewsRRCells", len(ne_), i0, S_N80, "news-comment edge cells audited for contagion (without the abandoned and the '+ parent' cells)", "cells: " + ", ".join(f"{a}/{b}" for a, b in ne_))
put("xcNeightyNewsRRNotReached", len(nr_), i0, S_N80, "of those, the cells in which Threadsafe does not reach 80% detection of the contagion distortion by the largest budget (16,000)", "cells: " + ", ".join(f"{a}/{b}" for a, b in nr_))
# ---- arXiv revision, review round 2
nrx_ = [c for c in nr_ if not any(N80.get(c + (("edge" if m_ == "log_rr" else "item"), m_), {}).get("any") for m_ in N80LAWS)]
put("xcNeightyNewsRRNotInTab", len(nrx_), i0, S_N80, "of the undetected news cells, those with no row in tables/xc_n80.tex (no design detects any of their laws)", "cells: " + ", ".join(f"{a}/{b}" for a, b in nrx_))
hc_ = N80[("cc_threads", "haiku_ctx", "edge", "log_rr")]
put("xcNeightyHaikuCtxRRTs", hc_["ts"] if hc_["ts"] else 2 * hc_["Bmax"], comma, S_N80, "cc_threads haiku_ctx log_rr", "Threadsafe's n80 for the reply risk ratio when Haiku shown the parent steers the audit",
    caveat="the audited law pools roots labeled without the parent with replies labeled with it")
assert hc_["ts"] == min(v["ts"] for k, v in N80.items() if k[2] == "edge" and k[0].startswith("cc") and v["ts"] and (k[0], k[1]) not in ABANDONED)
cgr_ = covs(E, "edge_calag", EDGE_M, 2000)
put("xcCalagRRCov", cgr_[0], pct, S_RAW, "edge_calag, B >= 2,000, both risk ratios", "calibrate and aggregate: mean coverage for the risk ratios")
put("xcCalagRRMin", cgr_[1], pct, S_RAW, "same", "lowest entry")
oth_ = [r for r in RX if not (r["sign_flip"] or r["manuf"] or r["lost"])]
shr_ = [r for r in oth_ if r["est"] * r["gold"] > 0 and abs(r["est"]) < abs(r["gold"])]
inf_ = [r for r in oth_ if r["est"] * r["gold"] > 0 and abs(r["est"]) >= abs(r["gold"])]
opp_ = [r for r in oth_ if r["est"] * r["gold"] <= 0]
assert len(shr_) + len(inf_) + len(opp_) == len(oth_)
put("xcMagOtherN", len(oth_), i0, S_DEC, "census rows (without '+ parent') that neither reverse a sign significantly nor manufacture or lose significance", "number")
put("xcMagShrunkN", len(shr_), i0, S_DEC, "of those, same sign as gold's law and smaller in absolute value", "shrunk toward zero")
put("xcMagInflN", len(inf_), i0, S_DEC, "of those, same sign and at least as large in absolute value", "inflated")
put("xcMagOppN", len(opp_), i0, S_DEC, "of those, the opposite sign without significance", "opposite sign, not significant")
# review round 2: counts that the prose spells out (numbers below ten, and the counts that sit next to them)
WORDN_ = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
for k_ in ("xcRhoNNegDepth", "xcNeightyNewsRRNotReached", "xcNeightyNewsRRNotInTab", "xcNeightyEdgeBetter", "xcNeightyEdgeWorse", "xcLopoItemN", "xcLopoItemRuns",
           "xcLopoEdgeN", "xcLopoEdgeRuns", "xcMagHiKapLost", "xcBoundNSignNine", "xcBoundNPops"):
    put(k_ + "Word", WORDN_[int(M[k_])], None, PROV[k_]["src"], PROV[k_]["key"], f"\\{k_} as a word")
with open(OUT_NUM, "w") as fo:
    fo.write("% generated by scripts/x98c_agree_eval.py from stored result files; do not edit (provenance: provenance_xc.md)\n")
    for k, v in M.items():
        fo.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
clash = [k for k in M if k in OLD]; assert not clash, clash


# ======================================================================== the provenance file
def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


INTRO = """# Provenance of `numbers_arxiv_c.tex`, `tables/xc_*.tex` and `figs/kappa_distortion.pdf`

Generated by `scripts/x98c_agree_eval.py` (run: `python3 -I scripts/x98c_agree_eval.py`; it reads stored result files only and rewrites every output). Paths are relative to `$YOU_IS_NOT_TOXIC_ROOT`.

## Rules applied

- **Population.** Current populations only: `results/x14_pops/` (Civil Comments reply threads, `cc_threads`, 1,251,487 comments and 980,520 reply edges; asserted). No f-series result of the old 1.78M population is used; E63's reply-association region, which existed only on the old population (`results/f14_regions_haiku.json`), is recomputed on `cc_threads` with the logic of `scripts/f14_regions.py` (calibrated Haiku levels, per-edge Fréchet bounds; seconds).
- **Labelers shown the parent.** They label roots without the parent and replies with it, so a law pooled over all comments carries a root/reply level shift. Their pooled law values are dropped from every census, correlation, selection, decision summary and from the figure; the among-replies decomposition (`results/x97/x97_decomp_replies.json`) is a different law and is not mixed in. In the simulated audits such a labeler only steers the design, so its cells stay in the coverage and width evaluation. In the n80 table, which tests the labeler's own law, its rows are kept and marked ‡ (relabeled), and the audit cell of Qwen shown the parent at the abandoned no-context threshold (21.9% flagged) is dropped.
- **Keep-out list.** Nothing from K1–K30 is used: no lenient human benchmark (K11), no split-half crowd (K9), no pooled κ–|z| correlation (K29), no B86a raw within-article ratios (K4), no uncorrected CGA-Wiki edge numbers (K13), no first LOPO run (K7). The unit-share regime (M16) stays out, so its leave-one-out result is not reported.
- **Classifier settings.** All 14 settings are in every stored analysis used here (`x74_decomp`, `x75_within`, `x77_correct`, `x78_pubs`, `x86_decisions`); x85's decision summaries and the many-rater check are not used for classifier ranges.
- **Cross-checks.** Every stored analysis that this script re-derives is first reproduced exactly (asserted): x79_agree (with the parent-shown labelers), x76_select, x77_correct's H77c, x80_magnitude, B82c, x85's uniform shares, eval_B (coverage, widths, ablations, adversaries, bias), x85_small_budgets, x86_fresh, B88i, the stored n80 table (`results/x72_eval/tables_H72.md`, row for row) and the earlier-draft ablation table (byte for byte). Where an earlier-draft macro (`paper_v5/numbers_*.tex` except `numbers_v2.tex`) holds the same number, `chk()` asserts equality at the earlier-draft macro's precision or records the explained difference below.
- **Formatting.** As `x18_numbers_v5.py` and `x95_eval_v6.py`: signed values `+0.93`, `$-$0.12` (no signed zero), percentages `67\\%` (half up), thousands `1{,}489`, coverage in tables `.949 (.92)`; coverage in macros as percentages with one decimal (`94.9\\%`) unless noted. Each interval end is its own macro.
"""
L_ = [INTRO]
L_.append("## Discrepancies found (earlier-draft macros, v7f's printed numbers, the inventory)\n")
plain = lambda s_: s_.replace("$-$", "\u2212").replace("\\%", "%").replace("{,}", ",").replace("$\\kappa \\ge 0.6$", "kappa >= 0.6")
inv = [
    f"E56: v7f's 'rank correlations {plain(OLD['qRhoDepthMin'])} to {plain(OLD['qRhoDepthMax'])} on six of the seven populations' and 'CAD ({plain(OLD['qRhoCadDepth'])})' (x79_agree.json) include the pooled '+ parent' labelers; without them the depth-law range is "
    f"{M['xcRhoDepthWeakest']} to {M['xcRhoDepthStrongest']} and CAD {M['xcRhoCadDepth']} (other laws: see xcRho*).",
    f"E57: BAD's stored highest-κ instrument (x76_select.json) is the pooled Haiku shown the parent (κ {BADK}); re-selected without it, BAD's is {M['xcSelBadName']} (κ {M['xcSelBadKappa']}). The totals stay 7 of 7 populations and 17 of 29 entries with |z| > 3; the median rank share becomes {M['xcSelRankShare']} (stored {plain(OLD['qSelRankShare'])}).",
    f"E58: the {CA['all']['n']} validate-and-correct entries include {CA['all']['n'] - CX['all']['n']} pooled '+ parent' entries; without them {M['xcCorrN']} entries, {M['xcCorrShareBelow']} cover below 0.90 (stored {plain(OLD['qCorrShareBelow'])}), {M['xcCorrShareWorse']} move away from gold ({plain(OLD['qCorrShareWorse'])}), median |bias|/|gold| {M['xcCorrMedRelBias']} after and {M['xcCorrUncMedRelBias']} before correction ({plain(OLD['qCorrMedRelBias'])}, {plain(OLD['qUncMedRelBias'])}), CGA-Wiki {M['xcCorrCgaOK']} ({plain(OLD['qCorrCgaOK'])}). The inventory's by-law shares (" + ", ".join(f"{LAWL[l].lower()} {plain(pct(CO['H77c'][f'share_below_90_{l}']))}" for l in LAW6) + f") are the all-entry values; without the pooled entries: {M['xcCorrBelowDepth']}, {M['xcCorrBelowWithin']}, {M['xcCorrBelowSize']}, {M['xcCorrBelowArt']}, {M['xcCorrBelowLrr']}.",
    f"E59: the census of {MG['all']['n']} estimates includes the same pooled entries; without them {M['xcMagN']} estimates, {M['xcMagShare']} off by more than 25% ({plain(OLD['qMagShare'])}), {M['xcMagFlip']} opposite signs ({M['xcMagFlipSig']} with gold's CI excluding 0; stored {OLD['qMagFlip']} and {OLD['qMagFlipSig']}), {M['xcMagManuf']} manufactured ({OLD['qMagManuf']}), {M['xcMagLost']} lost ({OLD['qMagLost']}); LLMs {M['xcMagLLMShare']} ({plain(OLD['qMagLLMShare'])}); κ ≥ 0.6: {M['xcMagHiKapN']} estimates ({OLD['qMagHiN']}), {M['xcMagHiKapShare']} ({plain(OLD['qMagHiShare'])}), {M['xcMagHiKapFlip']} sign flips ({OLD['qMagHiFlip']}). "
    f"The inventory's 'The LLMs' depth distortion is only {plain(OLD['qHrLLMUnifDepth'])} uniform' is the median over five Civil Comments LLM settings including the two pooled '+ parent' ones; without them {M['xcLlmUnifDepthCc']} on Civil Comments and {M['xcLlmUnifDepth']} over all populations (outside Civil Comments the LLMs mostly attenuate).",
    "E60: Haiku shown the parent is dropped from the decision table (the stored analysis uses its pooled labels, not replies only).",
    f"E61: Haiku shown the parent ({Mt['item']['reversed']['haiku_ctx']}, {Mt['thread']['reversed']['haiku_ctx']} and {Mt['article']['reversed']['haiku_ctx']} reversals) is not reported (pooled labeling modes).",
    f"E63: on the current population the reply-association region under calibrated Haiku is {M['xcFrAssocLo']} to {M['xcFrAssocHi']} (old population: {plain(sg(F14o['frechet_diff'][0], 3))} to {plain(sg(F14o['frechet_diff'][1], 3))}), the independence point {M['xcFrAssocInd']} (RR {M['xcFrAssocIndRR']}; old {plain(sg(F14o['independence_diff'], 3))}, RR {F14o['independence_rr']:.2f}), gold {M['xcFrAssocGold']} (RR {M['xcFrAssocGoldRR']}).",
    f"M11: `paper_v5/tables/n80.tex` is stale in two ways: its 'Qwen + parent§' contagion row is the cell at the abandoned no-context threshold (dropped here), and its 'Haiku + parent' rows test the pooled law (kept, relabeled ‡). v7f's Limitations sentence 'Contagion is expensive to certify (2,000–16,000 labels on news comments)' takes its 2,000 from the abandoned Qwen cell ({QFs} flagged); without it {M['xcNeightyNewsRRMin']}–{M['xcNeightyNewsRRMax']}. Without that cell the n80 counts are {M['xcNeightyItemBetter']} of {M['xcNeightyItemRows']} item rows better (stored {OLD['vNItemBetter']} of {OLD['vNItemRows']}) and {M['xcNeightyEdgeBetter']} of {M['xcNeightyEdgeRows']} better, {M['xcNeightyEdgeWorse']} worse for contagion ({OLD['vNEdgeBetter']} of {OLD['vNEdgeRows']}, {OLD['vNEdgeWorse']}).",
    f"M14: the inventory's 'v2 reaches 0.27–0.50' is the development figure of Addendum 64; the stored ceiling numbers give the stratified design {M['xcCeilStratLawsMin']}–{M['xcCeilStratLawsMax']} on the depth and size laws and {M['xcCeilStratArt']} on the article size law. Its 'Edges: 0.70 (gold RR)' is Addendum 70's value with CGA-Wiki; without CGA-Wiki (as x15 and here) {M['xcCeilTsGold']}.",
]
for x in inv:
    L_.append(f"- {esc(plain(x))}")
for x in DISC:
    L_.append(f"- {esc(x)}")
L_.append("\n## Items skipped or not computable\n")
for x in SKIP + ["E59: the single rater's opposite-sign, manufactured and lost counts are not computable: gold B (the other raters' majority) has no stored intervals ('--' in xc_census).",
                 "E60: the share of samples whose CI excludes gold's estimate is not stored at 20,000 comments ('--').",
                 "E63: the association region is an outer region: each reply edge's joint probability is bounded separately (as f14), although a comment enters several edges."]:
    L_.append(f"- {esc(x)}")
L_.append("\n## Tables\n")
for nm, t in TABLES.items():
    L_.append(f"### `tables/xc_{nm}.tex` ({t['rows']} rows; section {t['section']})\n")
    L_.append(f"- **Proposed caption.** {t['caption']}")
    L_.append(f"- **Column headers** (in order, separated by semicolons). " + "; ".join(f"'{h}'" for h in t["headers"]))
    L_.append(f"- **Source and keys.** {t['src']}")
    L_.append(f"- **Transformation.** {t['how']}")
    L_.append(f"- **Cross-check.** {t['checks']}")
    L_.append(f"- **Caveats.** {t['caveats']}\n")
L_.append("### Table layout notes\n")
L_.append("- `xc_cov_target` is split in two files, `xc_cov_target_cov.tex` (coverage) and `xc_cov_target_width.tex` (widths), each with 1 + 8 columns: Design | Laws (250, 500, 1k–8k) | Placebo slopes (250, 500, 1k–8k) | Reply RR (2k–16k) | Placebo RR (2k–16k). In the coverage file Threadsafe's row is followed by a `\\midrule`, as in `confirm_v6.tex`; the width file has only the five comparators with widths (PPI++, DSL, active inference, stratified, uniform).")
L_.append("- `xc_bounds`, `xc_decisions` and `xc_n80` contain `\\addlinespace` between groups (booktabs); `xc_correct`, `xc_census`, `xc_adversaries`, `xc_freshgold` contain `\\midrule` before their summary or reference row.\n")
L_.append("## Figure `figs/kappa_distortion.pdf`\n")
L_.append(f"- **Panels.** {', '.join(FIG['panels'])}; {FIG['n']} points (macro xcFigNPts). 7.0 × 2.4 in, fonts 7–7.5 pt, TrueType fonts embedded, palette of `p20_fig_decomp.py` (orange #eb6834, blue #2a78d6; colorblind-safe pair), LLMs as circles and classifiers as squares. A 200-dpi preview is written to the scratchpad, not to the paper.")
L_.append(f"- **Units.** {FIG['units']}.")
L_.append(f"- **Source.** {FIG['src']}.")
L_.append(f"- **Proposed caption.** {FIG['caption']}\n")
L_.append(f"## Macros ({len(M)})\n")
for sec, names in SECT.items():
    if not names:
        continue
    L_.append(f"### {sec}\n")
    L_.append("| Macro | Value | Source and key | Transformation | Cross-check | Caveats |")
    L_.append("|---|---|---|---|---|---|")
    for k in names:
        p = PROV[k]
        L_.append(f"| `\\{k}` | {esc(p['value'])} | {esc(p['src'])}: {esc(p['key'])} | {esc(p['how'])} | {esc('; '.join(p['checks']) or 'no earlier-draft counterpart')} | {esc(p['caveat'])} |")
    L_.append("")
open(OUT_PROV, "w").write("\n".join(L_) + "\n")
print(f"{len(M)} macros -> {OUT_NUM}; {len(TABLES)} tables -> {OUT_TAB}/xc_*.tex; figure -> {OUT_FIG}; provenance -> {OUT_PROV}")
print(f"{len(DISC)} recorded differences; {len(SKIP)} skipped items")
