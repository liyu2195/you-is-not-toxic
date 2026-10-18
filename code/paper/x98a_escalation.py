"""arXiv extension, agent A: numbers, tables and the operating-point figure for the news-comment escalation evidence
(inventory items E01-E04, E08-E19, E21-E26, E28-E32, E34 and the depth-law parts of E35 and E36), built only from stored results.

Reads (all under $YOU_IS_NOT_TOXIC_ROOT/results unless noted; every file is on the current population, cc_threads = 1,251,487 comments
and 980,520 reply edges, 5% sample = 60,668 comments, except where a note says otherwise):
  x74_placebo, x74_decomp, x75_decomp, x97, x75_within, x75_onerater, x92, x88, x89, x90, x91, x74_edit, x75_edit, x76_edit, x79,
  x77_youfree, x85, x87, x80, x81, x77_humanrate, x74_ccc, b84, x75_rating, x76_rethreshold, f81_you_edit.json (same comment sample as
  x74_edit), f124_prompts.json (sample containing the 5% sample, see x81/prn.json), x14_pops/*.parquet, data/second/s5 texts.
Deterministic re-analysis of stored labels (no randomness except replicated, seeded tie-breaking and the replicated seeded bootstrap of
x89_rewrite.py): flag rate, kappa, TPR and FPR among replies; open models shown the parent (x84_families.py flags); Haiku-100 and
Sonnet-100 flags (x92_matched.py); Sonnet's corpus-wide rewrite (x89_rewrite.py, analysis (i)) for the change CIs of the within-thread
and size laws and the gold positives' flag rates; the both-address detail of the rewrite shown the parent; kappa of the later rerun.
Writes:
  paper_arxiv/numbers_arxiv_a.tex (macros \\xa...), paper_arxiv/tables/xa_*.tex (table bodies), paper_arxiv/figs/oppoint_grid.pdf,
  paper_arxiv/provenance_xa.md.
Asserts equality with the existing macros of paper_v5/numbers_*.tex (numbers_v2.tex excluded: old population) wherever they hold the
same number; every difference is listed with its explanation in the provenance file.
usage: python3 -I x98a_escalation.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, re, glob
import numpy as np, pandas as pd

W = _ROOT; R = f"{W}/results"; PA = f"{W}/paper_arxiv"; P5 = f"{W}/results/reference"
OUT_NUM = f"{PA}/numbers_arxiv_a.tex"; TAB = f"{PA}/tables"; FIG = f"{PA}/figs"; OUT_PROV = f"{PA}/provenance_xa.md"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SCRIPT = "scripts/x98a_escalation.py"


# ================================================================ formatting (as x18_numbers_v5.py, x96_dist_v6.py)
def sg(x, d=2):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"                                                      # no signed zero
    return s.replace("-", "$-$") if s.startswith("-") else s
def f(x, d=2):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s
def pct(x, d=0):
    v = round(100 * x + 1e-9, d)
    return (f"{v:.{d}f}" if d else f"{int(round(100 * x + 1e-9))}") + "\\%"
def spct(x, d=0):                                                                  # signed percentage (as qSrPDR: +0.1\%)
    return sg(100 * x, d) + "\\%"
def th(n):
    return f"{int(n):,}".replace(",", "{,}")
def ci3(v, d=2, sc=1.0):                                                           # signed value with its CI, one cell
    return f"{sg(sc * v[0], d)} [{sg(sc * v[1], d)}, {sg(sc * v[2], d)}]"
def ciu(v, d=2, sc=1.0):                                                           # unsigned value (ratio, rate) with its CI
    return f"{f(sc * v[0], d)} [{f(sc * v[1], d)}, {f(sc * v[2], d)}]"
def plain(v_):                                                                      # a macro value as plain text for markdown
    return v_.replace("$-$", "\u2212").replace("\\%", "%").replace("{,}", ",")
def load(p):
    return json.load(open(f"{R}/{p}"))
def un(s):                                                                         # a formatted number back to float
    return float(s.replace("$-$", "-").replace("\\%", "").replace("{,}", ""))


# ================================================================ existing macros (earlier draft), for the cross-checks
EX, EXF = {}, {}
for fn in sorted(glob.glob(f"{P5}/numbers_*.tex")):
    if fn.endswith("numbers_v2.tex"):                                              # old 1.78M population
        continue
    s = open(fn).read(); i = 0
    while True:
        j = s.find("\\newcommand{\\", i)
        if j < 0:
            break
        k = s.find("}", j + 13); name = s[j + 13:k]; assert s[k + 1] == "{"
        dpt = 0; m = k + 1
        while True:
            c = s[m]
            if c == "\\":
                m += 2; continue
            if c == "{":
                dpt += 1
            elif c == "}":
                dpt -= 1
                if dpt == 0:
                    break
            m += 1
        EX[name] = s[k + 2:m]; EXF[name] = os.path.basename(fn); i = m


# ================================================================ macro registry and provenance
M = {}; PROV = []; DISC = []; ITEM = ["?"]; NOTES = []
def item(tag):
    ITEM[0] = tag
def put(name, value, src, key, how, caveat=""):
    assert re.fullmatch(r"xa[A-Za-z]+", name), name
    assert name not in M, f"duplicate macro {name}"
    assert isinstance(value, str) and value != "" and "nan" not in value.lower(), (name, value)
    M[name] = value
    PROV.append(dict(name=name, value=value, item=ITEM[0], src=src, key=key, how=how, check="", caveat=caveat))
def _rec(name):
    return next(p for p in PROV if p["name"] == name)
def same(name, ex):
    """the formatted value equals an existing macro of the earlier draft"""
    assert ex in EX, f"no existing macro {ex}"
    assert M[name] == EX[ex], f"{name}={M[name]} but \\{ex}={EX[ex]} ({EXF[ex]})"
    r = _rec(name); r["check"] += ("; " if r["check"] else "") + f"= \\{ex} ({EXF[ex]})"
def differs(name, ex, why):
    """the formatted value differs from an existing macro; the reason goes into the provenance and the discrepancy list"""
    assert ex in EX, f"no existing macro {ex}"
    assert M[name] != EX[ex], f"{name} now equals \\{ex}: remove the stale explanation"
    r = _rec(name); r["check"] += ("; " if r["check"] else "") + f"differs from \\{ex} = {EX[ex]} ({EXF[ex]}): {why}"
    DISC.append(dict(name=name, value=M[name], ex=ex, exv=EX[ex], exf=EXF[ex], why=why, item=ITEM[0]))
def logchk(name, text):
    r = _rec(name); r["check"] += ("; " if r["check"] else "") + text
def note(text):
    NOTES.append((ITEM[0], text))
def trip(base, v, src, key, how, d=2, sc=1.0, caveat="", signed=True):
    """a value and its 95% CI as three macros: base, baseLo, baseHi"""
    fm = (lambda x: sg(sc * x, d)) if signed else (lambda x: f(sc * x, d))
    put(base, fm(v[0]), src, key, how, caveat); put(base + "Lo", fm(v[1]), src, key, how + " (2.5th percentile)", caveat)
    put(base + "Hi", fm(v[2]), src, key, how + " (97.5th percentile)", caveat)
def same3(base, ex):
    for s_ in ("", "Lo", "Hi"):
        same(base + s_, ex + s_)

TABLES = {}                                                                        # name -> dict(rows, caption, headers, src)
def table(name, rows, caption, headers, src):
    for r_ in rows:
        assert "nan" not in r_.lower(), (name, r_)
    open(f"{TAB}/{name}.tex", "w").write("\n".join(rows) + "\n")
    TABLES[name] = dict(rows=rows, caption=caption, headers=headers, src=src)


# ================================================================ shared data: populations, labels, structure
PT = pd.read_parquet(f"{R}/x14_pops/cc_threads.parquet")
PS = pd.read_parquet(f"{R}/x14_pops/cc_s5.parquet")
assert len(PT) == 1251487 and int((PT.parent_row.values >= 0).sum()) == 980520 and len(PS) == 60668
NT, NS = len(PT), len(PS); zT = PT.z.values.astype(float); zS = PS.z.values.astype(float)
repT = PT.depth.values >= 1; repS = PS.depth.values >= 1
assert int(repT.sum()) == 980520 and np.array_equal(repT, PT.parent_row.values >= 0) and np.array_equal(repS, PS.parent_row.values >= 0)

def conf(y, z, m=None):
    y = np.asarray(y, float); z = np.asarray(z, float)
    if m is not None:
        y, z = y[m], z[m]
    q, p = y.mean(), z.mean(); po = (y == z).mean(); pe = q * p + (1 - q) * (1 - p)
    return dict(n=int(len(y)), flag=float(q), kappa=float((po - pe) / (1 - pe)), tpr=float(y[z == 1].mean()), fpr=float(y[z == 0].mean()), prev=float(p))

def flags_at_prev(score, N, k, seed, rng=None):
    """the k comments with the highest score, ties at random (np.lexsort as x76_rethreshold.py, x84_families.py, x92_matched.py)"""
    s = np.where(np.isnan(score), -np.inf, score)
    r = (rng if rng is not None else np.random.default_rng(seed)).random(N)
    o = np.lexsort((r, -s)); y = np.zeros(N); y[o[:k]] = 1.0
    return y

CLS = [("Detoxify", "detoxify_original", "0.6", "DetoxSix", "DetoxM"), ("Detoxify-unbiased$^{\\S}$", "detoxify_unbiased", "0.6", "UnbSix", "UnbM"),
       ("IMSYPP", "imsypp", "argmax", "ImsArg", "ImsM"), ("s-nlp RoBERTa$^{\\S}$", "sn_rob", "0.5", "SnlpHalf", "SnlpM"),
       ("martin-ha DistilBERT$^{\\S}$", "mh_dist", "0.5", "MhHalf", "MhM"), ("Dynabench hate", "fb_hate", "0.5", "DynHalf", "DynM"),
       ("CardiffNLP offensive", "cn_off", "0.5", "CnHalf", "CnM")]
CLS14 = [k for _, c, t, _, _ in CLS for k in (f"{c}@{t}", f"{c}@m")]
CLSNM = {f"{c}@{t}": a for _, c, t, a, _ in CLS} | {f"{c}@m": b for _, c, t, _, b in CLS}
CLSDISP = {f"{c}@{t}": (nm, t) for nm, c, t, _, _ in CLS} | {f"{c}@m": (nm, "gold's rate") for nm, c, t, _, _ in CLS}
assert len(CLS14) == 14 and sorted(CLS14) == sorted(["detoxify_original@0.6", "detoxify_original@m", "detoxify_unbiased@0.6", "detoxify_unbiased@m",
       "imsypp@argmax", "imsypp@m", "sn_rob@0.5", "sn_rob@m", "mh_dist@0.5", "mh_dist@m", "fb_hate@0.5", "fb_hate@m", "cn_off@0.5", "cn_off@m"])
CLS7M = [f"{c}@m" for _, c, _, _, _ in CLS]


# ================================================================ E01: Table 1 and Table 7 with CIs; relative placebo slopes
item("E01")
PL = load("x74_placebo/x74_placebo.json"); PLc, PL5 = PL["cc_threads"]["labelers"], PL["cc_threads_s5"]["labelers"]; GT = PL["cc_threads"]["gold"]
SRC_PL = "results/x74_placebo/x74_placebo.json"
O_ = load("x75_onerater/x75_onerater.json"); SPL = O_["splits"]; assert len(SPL) == 20
med = lambda g: float(np.median([g(s_) for s_ in SPL]))
srA = lambda k: med(lambda s_: s_["instr"]["crowd_A"][k])
srArng = lambda k, j: (min if j == 0 else max)(s_["instr"]["crowd_A"][k] for s_ in SPL)          # range over the 20 draws
SR0 = SPL[0]["instr"]["crowd_A"]; assert "placebo_depth_rel_ci" in SR0 and all("placebo_depth_rel_ci" not in s_["instr"]["crowd_A"] for s_ in SPL[1:])
ROWS1 = [("Haiku", "haiku", 0, "Haiku"), ("Sonnet$^{\\dagger}$", "sonnet", 1, "Sonnet"), ("Qwen", "qwen7b_m", 0, "Qwen"), ("", "qwen7b", 0, "QwenOwn"),   # arXiv: Qwen's own cut too
         ("Gemma$^{\\dagger}$", "gemma_m", 1, "Gemma")]
ROWS1 += [((CLSDISP[k][0] if k.endswith(("@0.6", "@0.5", "@argmax")) else ""), k, 0, CLSNM[k]) for k in CLS14]
# the existing Table 1 / Table 7 cells, to check the point values
def _cells(fn):
    out = []
    for l in open(f"{P5}/tables/{fn}"):
        l = l.strip()
        if l.endswith("\\\\") and "&" in l:
            out.append([c.strip() for c in l[:-2].split("&")])
    return out
T1 = {r_[0]: r_ for r_ in _cells("laws_v6.tex")}; T7 = _cells("classifiers_v7.tex")
T7d = {}; cur = None
for r_ in T7:
    cur = r_[0] or cur; T7d[(cur, r_[1])] = r_
rows = []
gcell = lambda k: ci3(GT[k], 2, 100)
rows.append(" & ".join(["Gold", "", "--", "--", "--", gcell("depth_slope"), gcell("size_slope"), ciu(GT["rr"])]) + " \\\\")
for k_, nm in (("depth_slope", "GoldDepth"), ("size_slope", "GoldSize")):
    trip(f"xa{nm}", GT[k_], SRC_PL, f"cc_threads.gold.{k_}", "x100 (points per level / per doubling); 500 article-cluster Poisson resamples", sc=100)
trip("xaGoldRR", GT["rr"], SRC_PL, "cc_threads.gold.rr", "reply risk ratio", signed=False)
same3("xaGoldDepth", "wGoldDepthPP"); same("xaGoldSize", "wGoldSizePP")
assert T1["Gold"][5] == M["xaGoldDepth"] and T1["Gold"][6] == M["xaGoldSize"] and T1["Gold"][7] == M["xaGoldRR"]
# one rater (median over the 20 draws of each draw's point value and of each draw's CI bounds; gold = the other raters' majority)
sr_pd_abs = med(lambda s_: s_["instr"]["crowd_A"]["placebo_depth_rel"] * s_["instr"]["crowd_A"]["fpr"])
sr_prr = [float(np.exp(srA("log_placebo_rr"))), float(np.exp(srArng("log_placebo_rr", 0))), float(np.exp(srArng("log_placebo_rr", 1)))]
sr_rel = [srA("placebo_depth_rel"), srArng("placebo_depth_rel", 0), srArng("placebo_depth_rel", 1)]
sr_dep = [srA("depth"), srArng("depth", 0), srArng("depth", 1)]; sr_siz = [srA("size"), srArng("size", 0), srArng("size", 1)]
sr_rr = [float(np.exp(srA("log_rr"))), float(np.exp(srArng("log_rr", 0))), float(np.exp(srArng("log_rr", 1)))]
rows.append(" & ".join(["One rater", "", sg(100 * sr_pd_abs), ci3(sr_rel, 1, 100), ciu(sr_prr), ci3(sr_dep, 2, 100), ci3(sr_siz, 2, 100), ciu(sr_rr)]) + " \\\\")
assert T1["One rater"][3] == sg(100 * sr_pd_abs) and T1["One rater"][4] == f(sr_prr[0]) and T1["One rater"][5] == sg(100 * sr_dep[0])
assert T1["One rater"][6] == sg(100 * sr_siz[0]) and T1["One rater"][7] == f(sr_rr[0])
rows.append("\\midrule")
for disp, k, s5, nm in ROWS1:
    r_ = (PL5 if s5 else PLc)[k]
    if k == "detoxify_original@0.6":
        rows.append("\\midrule")
    elif k in CLS14 and not k.endswith("@m"):
        rows.append("\\addlinespace[1pt]")
    cut = CLSDISP[k][1] if k in CLS14 else ("own cut" if nm in ("Haiku", "Sonnet") else ("own cut ($p > 0.5$)" if nm == "QwenOwn" else "gold's rate"))
    cells = [disp, cut, ci3(r_["placebo_depth_slope"], 2, 100), ci3(r_["placebo_depth_rel"], 1, 100), ciu(r_["placebo_rr"]),
             ci3(r_["law_depth_slope"], 2, 100), ("--" if s5 else ci3(r_["law_size_slope"], 2, 100)), ciu(r_["law_rr"])]
    rows.append(" & ".join(cells) + " \\\\")
    if nm in ("Haiku", "Sonnet", "Qwen", "Gemma") or k in CLS7M or k == "detoxify_original@0.6" or k == "imsypp@argmax":
        tname = {"Haiku": "Haiku", "Sonnet": "Sonnet$^{\\dagger}$", "Qwen": "Qwen", "Gemma": "Gemma$^{\\dagger}$", "DetoxSix": "Detoxify (0.6)", "DetoxM": "Detoxify",
                 "UnbM": "Detoxify-unbiased$^{\\S}$", "ImsArg": "IMSYPP (argmax)", "ImsM": "IMSYPP", "SnlpM": "s-nlp RoBERTa$^{\\S}$", "MhM": "martin-ha DistilBERT$^{\\S}$",
                 "DynM": "Dynabench hate", "CnM": "CardiffNLP offensive"}[nm]
        t1 = T1[tname]
        assert t1[3] == sg(100 * r_["placebo_depth_slope"][0]) and t1[4] == f(r_["placebo_rr"][0]) and t1[5] == sg(100 * r_["law_depth_slope"][0]), (tname, t1)
        assert t1[7] == f(r_["law_rr"][0]) and (s5 or t1[6] == sg(100 * r_["law_size_slope"][0]))
    if k in CLS14:                                                                  # Table 7's cells
        fam = CLSDISP[k][0] if k.endswith(("@0.6", "@0.5", "@argmax")) else next(CLSDISP[k2][0] for k2 in CLS14 if k2.split("@")[0] == k.split("@")[0] and not k2.endswith("@m"))
        t7 = T7d[(fam, CLSDISP[k][1])]
        assert t7[4] == sg(100 * r_["placebo_depth_slope"][0]) and t7[5] == f(r_["placebo_rr"][0]) and t7[6] == sg(100 * r_["law_depth_slope"][0]) and t7[8] == f(r_["law_rr"][0])
    pk = f"{'cc_threads_s5' if s5 else 'cc_threads'}.labelers.{k}"
    trip(f"xa{nm}PlacRel", r_["placebo_depth_rel"], SRC_PL, pk + ".placebo_depth_rel", "placebo depth slope / FPR, x100 (% of the FPR per level)", 1, 100,
         caveat="5% sample" if s5 else "")
    trip(f"xa{nm}PlacSizeRel", r_["placebo_size_rel"], SRC_PL, pk + ".placebo_size_rel", "placebo size slope / FPR, x100 (% of the FPR per doubling)", 1, 100,
         caveat="5% sample" if s5 else "")
for nm, k, s5 in (("Haiku", "haiku", 0), ("Sonnet", "sonnet", 1)):
    r_ = (PL5 if s5 else PLc)[k]; pk = f"{'cc_threads_s5' if s5 else 'cc_threads'}.labelers.{k}"; cv = "5% sample" if s5 else ""
    trip(f"xa{nm}Plac", r_["placebo_depth_slope"], SRC_PL, pk + ".placebo_depth_slope", "x100 (points per level)", caveat=cv, sc=100)
    trip(f"xa{nm}PlacRR", r_["placebo_rr"], SRC_PL, pk + ".placebo_rr", "placebo risk ratio", signed=False, caveat=cv)
    trip(f"xa{nm}Law", r_["law_depth_slope"], SRC_PL, pk + ".law_depth_slope", "x100 (points per level)", caveat=cv, sc=100)
    trip(f"xa{nm}SizeLaw", r_["law_size_slope"], SRC_PL, pk + ".law_size_slope", "x100 (points per doubling)", caveat=cv, sc=100)
    trip(f"xa{nm}RR", r_["law_rr"], SRC_PL, pk + ".law_rr", "reply risk ratio", signed=False, caveat=cv)
same("xaHaikuPlac", "cPlacHaiku"); same("xaHaikuLaw", "wHaikuDepthPP"); same("xaHaikuRR", "tOneHaikuRR"); same("xaSonnetLaw", "qSonnetDepthPP")
same("xaHaikuSizeLaw", "qMsHaikuThread"); same("xaSonnetPlac", "cDoseSonPlac")
put("xaSrPlacRel", spct(sr_rel[0], 1), "results/x75_onerater/x75_onerater.json", "splits[*].instr.crowd_A.placebo_depth_rel",
    "median over the 20 draws, x100 (% of the FPR per level); gold = majority of the other raters")
put("xaSrPlacRelLo", spct(SR0["placebo_depth_rel_ci"][0], 1), "results/x75_onerater/x75_onerater.json", "splits[0].instr.crowd_A.placebo_depth_rel_ci[0]",
    "draw 0's article-bootstrap 2.5th percentile (only draw 0 stores bootstrap intervals)", caveat="interval of one draw; the point value is the median over draws")
put("xaSrPlacRelHi", spct(SR0["placebo_depth_rel_ci"][1], 1), "results/x75_onerater/x75_onerater.json", "splits[0].instr.crowd_A.placebo_depth_rel_ci[1]",
    "draw 0's article-bootstrap 97.5th percentile", caveat="interval of one draw")
same("xaSrPlacRel", "qSrPDR")
note("xaHaikuPlacRel (\\xaHaikuPlacRel, against full gold, x74_placebo) is not the earlier-draft macro qHaikuPDRLoo (\\xaHaikuPlacRelLoo), which is against the draws' "
     "leave-one-out gold (x75_onerater); the latter is reproduced as xaHaikuPlacRelLoo. Relative-slope macros carry no percent sign (write \\\\%).")
table("xa_laws_ci", rows,
      "Placebo statistics and laws on Civil Comments reply threads, with 95% intervals from article-cluster resamples. The placebo depth slope is in "
      "points per level and also relative to the labeler's false-positive rate (percent of the FPR per level); depth laws in points per level, thread size "
      "laws in points per doubling. Rows flag at gold's flag rate except Haiku, Sonnet and Qwen's second row (own cut) and the thresholds in parentheses. One rater: a single rater "
      f"against the majority of the other raters: median over {len(SPL)} draws, with the range over the draws in brackets. Dagger: 5% sample (size laws not shown). "
      "Section sign: trained at least in part on Civil Comments labels.",
      "Labeler | Cut | Placebo depth slope [CI] | Relative to FPR, % [CI] | Placebo risk ratio [CI] | Depth law [CI] | Thread size law [CI] | Risk ratio [CI]",
      SRC_PL + "; results/x75_onerater/x75_onerater.json (one rater)")
# arXiv revision, review round 2: the main-text table keeps the human references, the LLMs and the two classifiers of Avalle et al.'s checks;
# Appendix D's table gives all 14 classifier settings with the same cells. Both are row subsets of xa_laws_ci (no cell changes).
_mid = [i_ for i_, r_ in enumerate(rows) if r_ == "\\midrule"]
assert len(_mid) == 2
_pre, _rest = rows[:_mid[1]], rows[_mid[1] + 1:]
_fams, _cur = [], []
for r_ in _rest:
    if r_ == "\\addlinespace[1pt]":
        _fams.append(_cur); _cur = []
    else:
        _cur.append(r_)
_fams.append(_cur)
_fn = [f_[0].split("&")[0].strip() for f_ in _fams]
assert _fn == ["Detoxify", "Detoxify-unbiased$^{\\S}$", "IMSYPP", "s-nlp RoBERTa$^{\\S}$", "martin-ha DistilBERT$^{\\S}$", "Dynabench hate", "CardiffNLP offensive"], _fn
assert all(len(f_) == 2 for f_ in _fams)
_main = _pre + ["\\midrule"] + _fams[0] + ["\\addlinespace[1pt]"] + _fams[2]
_cls = []
for f_ in _fams:
    _cls += (["\\addlinespace[1pt]"] if _cls else []) + f_
table("xa_laws_main", _main, "Table xa_laws_ci's rows for the human references, the LLMs, Detoxify and IMSYPP (main text).", TABLES["xa_laws_ci"]["headers"], TABLES["xa_laws_ci"]["src"])
table("xa_laws_cls", _cls, "Table xa_laws_ci's 14 classifier rows (Appendix D).", TABLES["xa_laws_ci"]["headers"], TABLES["xa_laws_ci"]["src"])


# ================================================================ E02: the single rater (B75b)
item("E02")
SO = "results/x75_onerater/x75_onerater.json"
put("xaSrNDraws", str(len(SPL)), SO, "splits", "number of draws"); same("xaSrNDraws", "qNDraws")
put("xaSrKappa", f(med(lambda s_: s_["kappa_A_B"])), SO, "splits[*].kappa_A_B", "median over draws"); same("xaSrKappa", "qSrKappa")
sr_fpr = srA("fpr"); put("xaSrFPR", f"{sr_fpr:.3f}", SO, "splits[*].instr.crowd_A.fpr", "median over draws"); same("xaSrFPR", "qSrFPR")
sr_tpr = med(lambda s_: (s_["prev_A"] - (1 - s_["prev_B"]) * s_["instr"]["crowd_A"]["fpr"]) / s_["prev_B"])
put("xaSrTPR", f"{sr_tpr:.3f}", SO, "splits[*].prev_A, prev_B, instr.crowd_A.fpr",
    "per draw TPR = (flag rate - (1 - prevalence of the others' majority) FPR) / that prevalence (exact identity); median over draws")
sr_flag = med(lambda s_: s_["prev_A"]); put("xaSrFlag", pct(sr_flag, 1), SO, "splits[*].prev_A", "median over draws of the single rater's flag rate")
put("xaSrGoldBPrev", pct(med(lambda s_: s_["prev_B"]), 1), SO, "splits[*].prev_B", "median over draws: prevalence of the other raters' majority")
put("xaSrPlacRR", f(sr_prr[0]), SO, "splits[*].instr.crowd_A.log_placebo_rr", "exp of the median log placebo RR"); same("xaSrPlacRR", "qSrPRR")
put("xaSrDepthDist", f(abs(100 * srA("depth_minus_goldB"))), SO, "splits[*].instr.crowd_A.depth_minus_goldB", "|median| x100 (points per level)"); same("xaSrDepthDist", "qSrDepthDist")
put("xaSrLrrDist", sg(srA("log_rr_minus_goldB")), SO, "splits[*].instr.crowd_A.log_rr_minus_goldB", "median over draws"); same("xaSrLrrDist", "qSrLrrDist")
put("xaHaikuLrrDistLoo", sg(med(lambda s_: s_["instr"]["haiku"]["log_rr_minus_goldB"])), SO, "splits[*].instr.haiku.log_rr_minus_goldB", "median over draws")
same("xaHaikuLrrDistLoo", "qHaikuLrrDistLoo")
put("xaHaikuDepthDistLoo", sg(100 * med(lambda s_: s_["instr"]["haiku"]["depth_minus_goldB"])), SO, "splits[*].instr.haiku.depth_minus_goldB", "median x100")
put("xaHaikuPlacRelLoo", spct(med(lambda s_: s_["instr"]["haiku"]["placebo_depth_rel"]), 1), SO, "splits[*].instr.haiku.placebo_depth_rel", "median x100")
same("xaHaikuPlacRelLoo", "qHaikuPDRLoo")
below = sum(all([s_["instr"]["crowd_A"]["placebo_depth_rel"] < s_["instr"]["haiku"]["placebo_depth_rel"],
                 s_["instr"]["crowd_A"]["placebo_size_rel"] < s_["instr"]["haiku"]["placebo_size_rel"],
                 abs(s_["instr"]["crowd_A"]["depth_minus_goldB"]) < abs(s_["instr"]["haiku"]["depth_minus_goldB"]),
                 abs(s_["instr"]["crowd_A"]["size_minus_goldB"]) < abs(s_["instr"]["haiku"]["size_minus_goldB"])]) for s_ in SPL)
put("xaSrBelowHaikuDraws", str(below), SO, "splits[*].instr.{crowd_A,haiku}.{placebo_depth_rel,placebo_size_rel,depth_minus_goldB,size_minus_goldB}",
    "number of draws in which the single rater's relative placebo depth and size slopes and its |depth| and |size| distortions are all below Haiku's",
    caveat="Haiku shown the parent and Qwen shown the parent are left out: their x75_onerater values are pooled (roots labeled without the parent) and Qwen's uses the old reused threshold")
rbd = list(O_["raters_by_depth"].values())
put("xaRatersMin", f(min(rbd), 1), SO, "raters_by_depth", "minimum over depths 0-10 of the mean raters per comment"); same("xaRatersMin", "qRatersMin")
put("xaRatersMax", f(max(rbd), 1), SO, "raters_by_depth", "maximum"); same("xaRatersMax", "qRatersMax")
put("xaShareFourRaters", pct(O_["share_4_raters"]), SO, "share_4_raters", "x100"); same("xaShareFourRaters", "qShareFourRaters")
sr_draw_rel = [100 * s_["instr"]["crowd_A"]["placebo_depth_rel"] for s_ in SPL]
put("xaSrPlacRelDrawMin", spct(min(sr_draw_rel) / 100, 1), SO, "splits[*].instr.crowd_A.placebo_depth_rel", "minimum over draws")
put("xaSrPlacRelDrawMax", spct(max(sr_draw_rel) / 100, 1), SO, "splits[*].instr.crowd_A.placebo_depth_rel", "maximum over draws")


# ================================================================ E03: the exact decomposition of every law (Prop. 1), the contagion terms
item("E03")
DEC = load("x74_decomp/x74_decomp.json"); DC = load("x75_decomp/x75_decomp.json"); RP = load("x97/x97_decomp_replies.json")
SR_DEC = load("x75_onerater/x75_onerater_decomp.json")["median"]
g0T, g0S = DC["cc_threads"]["gold_log_rr"], DC["cc_s5"]["gold_log_rr"]
DROWS = [("Haiku", "cc_threads", "haiku", None), ("Haiku + parent", "cc_threads", "haiku_ctx", "x97"),
         ("Sonnet$^{\\dagger}$", "cc_s5", "sonnet", None), ("Sonnet + parent$^{\\dagger}$", "cc_s5", "sonnet_ctx", "x97"),
         ("Qwen", "cc_threads", "qwen7b_m", None), ("Qwen + parent", "cc_threads", "qwen7b_m_ctx@own", "x97"),
         ("Gemma$^{\\dagger}$", "cc_s5", "gemma_m", None), ("Gemma + parent$^{\\dagger}$", "cc_s5", "gemma_m_ctx@own", "x97")]
DROWS += [(CLSDISP[k][0] + f" ({CLSDISP[k][1]})", "cc_threads", k, None) for k in CLS14]
rows = []; maxchk = 0.0; TERMS = {}
for disp, pop, k, how in DROWS:
    cells = [disp]; TERMS[k] = {}
    for law in ("depth", "size", "article_size", "log_rr"):
        if how == "x97":                                                           # shown the parent: depth law among replies only
            if law != "depth":
                cells += ["--"] * 4; continue
            q_ = RP[pop]["labelers"][k]; u, fp, tp, tot = q_["uniform"], q_["fp"], q_["tp"], q_["distortion"][0]
            maxchk = max(maxchk, abs(u + fp + tp - tot))
        elif law == "log_rr":
            q_ = DC[pop]["labelers"][k]; u, fp, tp, tot = q_["uniform"], q_["fp"], q_["tp"], q_["distortion"]
            assert abs(tot - DEC[pop]["labelers"][k]["laws"]["log_rr"]["distortion"][0]) < 1e-12; maxchk = max(maxchk, q_["check"])
        else:
            q_ = DEC[pop]["labelers"][k]["laws"][law]
            u, fp, tp, tot = 100 * (q_["attenuation_term"][0] - q_["gold"][0]), 100 * q_["fp_term"][0], 100 * q_["tp_term"][0], 100 * q_["distortion"][0]
            maxchk = max(maxchk, q_["check"], abs(u + fp + tp - tot) / 100)
        TERMS[k][law] = (u, fp, tp, tot)
        cells += [sg(u), sg(fp), sg(tp), sg(tot)]
    rows.append(" & ".join(cells) + " \\\\")
    if disp.startswith("Gemma + parent"):
        rows.append("\\midrule")
cells = ["One rater"]
for law in ("depth", "size", "article_size", "log_rr"):
    if law not in SR_DEC:
        cells += ["--"] * 4; continue
    m_ = SR_DEC[law]; sc = 1.0 if law == "log_rr" else 100.0
    cells += [sg(sc * m_["uniform"]), sg(sc * m_["fp"]), sg(sc * m_["tp"]), sg(sc * m_["distortion"])]
rows.insert(len(rows), "\\midrule"); rows.append(" & ".join(cells) + " \\\\")
assert maxchk < 1e-9, maxchk
table("xa_decomp_terms", rows,
      "The exact decomposition of each distortion on Civil Comments reply threads (Proposition 1): unstructured errors, (TPR - FPR - 1) times gold's law; "
      "structured false positives S_FP; structured true positives S_TP; and the total, the labeler's law minus gold's. Depth law in points per level, thread "
      "and article size laws in points per doubling, log reply risk ratio in log units; for contagion the unstructured term is the permutation baseline and "
      "the other two are Shapley values. Rows + parent label replies with the parent shown: their depth law is decomposed among replies, and their size laws "
      "and contagion, which mix roots labeled without the parent with replies labeled with it, are not shown. Qwen and Gemma shown the parent flag at their "
      f"own gold-prevalence threshold. Dagger: 5% sample. One rater: median of {len(SPL)} draws, against the other raters' majority.",
      "Labeler | Depth law: Unstructured, S_FP, S_TP, Total | Thread size law: same four | Article size law: same four | Log reply RR: same four",
      "results/x74_decomp/x74_decomp.json (linear laws), results/x75_decomp/x75_decomp.json (log RR), results/x97/x97_decomp_replies.json (+ parent, depth among "
      "replies), results/x75_onerater/x75_onerater_decomp.json (one rater)")
put("xaDecompCheck", "$10^{-9}$" if maxchk >= 1e-12 else "$10^{-12}$", "x74_decomp, x75_decomp, x97", "laws.*.check; recomputed sums",
    "largest |unstructured + S_FP + S_TP - total| over all rows (bound shown)")
logchk("xaDecompCheck", f"actual maximum {maxchk:.1e}; the earlier-draft macro qMaxCheck is 1e-16 (x74/x75 'check' fields only)")
# shares of structured false positives in the escalation distortions (Figure 2's caption): Claude without the parent, and every LLM shown the parent among replies
sh = {"Haiku": TERMS["haiku"]["depth"][1] / TERMS["haiku"]["depth"][3], "Sonnet": TERMS["sonnet"]["depth"][1] / TERMS["sonnet"]["depth"][3],
      "HaikuCtx": RP["cc_threads"]["labelers"]["haiku_ctx"]["fp_share"], "SonnetCtx": RP["cc_s5"]["labelers"]["sonnet_ctx"]["fp_share"],
      "QwenCtx": RP["cc_threads"]["labelers"]["qwen7b_m_ctx@own"]["fp_share"]}
for nm, v in sh.items():
    put(f"xaFPShare{nm}", pct(v), "x74_decomp / x97", "laws.depth.fp_term / distortion; x97 fp_share", "S_FP / total, depth law")
put("xaFPShareMin", pct(min(sh.values())), "x74_decomp / x97", "as above", "minimum over Haiku, Sonnet, and Haiku, Sonnet, Qwen shown the parent (among replies)")
put("xaFPShareMax", pct(max(sh.values())), "x74_decomp / x97", "as above", "maximum over the same five")
same("xaFPShareMin", "dFPShareMin"); same("xaFPShareMax", "dFPShareMax"); same("xaFPShareHaiku", "dFPShareHaiku"); same("xaFPShareSonnet", "dFPShareSonnet")
put("xaFPShareGemmaCtx", pct(RP["cc_s5"]["labelers"]["gemma_m_ctx@own"]["fp_share"]), "results/x97/x97_decomp_replies.json", "cc_s5.labelers.gemma_m_ctx@own.fp_share",
    "S_FP / total among replies", caveat="5% sample; Gemma's distortion among replies is small (CI covers 0), so the share is unstable")
# contagion ranges: LLMs without the parent (the +parent log RR mixes labeling modes on the parent side), all 14 classifier settings
LLM4 = [("cc_threads", "haiku"), ("cc_threads", "qwen7b_m"), ("cc_s5", "sonnet"), ("cc_s5", "gemma_m")]
gl = {"cc_threads": g0T, "cc_s5": g0S}
uni_l = [DC[p_]["labelers"][k_]["uniform"] for p_, k_ in LLM4]; uni_c = [DC["cc_threads"]["labelers"][k_]["uniform"] for k_ in CLS14]
fp_l = [DC[p_]["labelers"][k_]["fp"] for p_, k_ in LLM4]; fp_c = [DC["cc_threads"]["labelers"][k_]["fp"] for k_ in CLS14]
tp_l = [DC[p_]["labelers"][k_]["tp"] for p_, k_ in LLM4]; tp_c = [DC["cc_threads"]["labelers"][k_]["tp"] for k_ in CLS14]
rr_left = [float(np.exp(gl[p_] + DC[p_]["labelers"][k_]["uniform"])) for p_, k_ in LLM4] + [float(np.exp(g0T + DC["cc_threads"]["labelers"][k_]["uniform"])) for k_ in CLS14]
SD = "results/x75_decomp/x75_decomp.json"
put("xaGoldLogRR", f(g0T), SD, "cc_threads.gold_log_rr", "log of gold's reply risk ratio")
put("xaGoldLogRRSfive", f(g0S), SD, "cc_s5.gold_log_rr", "log of gold's reply RR on the 5% sample", caveat="5% sample")
put("xaUniLLMMin", sg(max(uni_l)), SD, "{cc_threads,cc_s5}.labelers.{haiku,qwen7b_m,sonnet,gemma_m}.uniform", "smallest cut (largest uniform term) over the four LLMs without the parent")
put("xaUniLLMMax", sg(min(uni_l)), SD, "same", "largest cut over the four LLMs")
put("xaUniClsMin", sg(max(uni_c)), SD, "cc_threads.labelers.<14 classifier settings>.uniform", "smallest cut over the 14 classifier settings")
put("xaUniClsMax", sg(min(uni_c)), SD, "same", "largest cut over the 14 classifier settings")
put("xaFPLLMMin", sg(min(fp_l)), SD, "...fp (LLMs)", "minimum S_FP over the four LLMs"); put("xaFPLLMMax", sg(max(fp_l)), SD, "...fp (LLMs)", "maximum")
put("xaFPClsMin", sg(min(fp_c)), SD, "...fp (14 classifiers)", "minimum S_FP"); put("xaFPClsMax", sg(max(fp_c)), SD, "...fp (14 classifiers)", "maximum")
put("xaTPLLMMin", sg(min(tp_l)), SD, "...tp (LLMs)", "minimum S_TP"); put("xaTPLLMMax", sg(max(tp_l)), SD, "...tp (LLMs)", "maximum")
put("xaTPClsMin", sg(min(tp_c)), SD, "...tp (14 classifiers)", "minimum S_TP"); put("xaTPClsMax", sg(max(tp_c)), SD, "...tp (14 classifiers)", "maximum")
put("xaUniRRMin", f(min(rr_left)), SD, "gold_log_rr + uniform", "exp(gold's log RR + unstructured term), minimum over the 4 LLMs and 14 classifier settings (each against its population's gold)")
put("xaUniRRMax", f(max(rr_left)), SD, "gold_log_rr + uniform", "maximum over the same 18 settings")
put("xaNClsTPPos", str(sum(t_ > 0 for t_ in tp_c)), SD, "...tp (14 classifiers)", "number of classifier settings with S_TP > 0")
put("xaNClsFPPos", str(sum(t_ > 0 for t_ in fp_c)), SD, "...fp (14 classifiers)", "number of classifier settings with S_FP > 0")
put("xaHaikuUniLRR", sg(DC["cc_threads"]["labelers"]["haiku"]["uniform"]), SD, "cc_threads.labelers.haiku.uniform", "Haiku's unstructured term")
put("xaHaikuFPLRR", sg(DC["cc_threads"]["labelers"]["haiku"]["fp"]), SD, "cc_threads.labelers.haiku.fp", "Haiku's S_FP"); same("xaHaikuFPLRR", "qFPHaiku")
same("xaFPClsMax", "qFPClsMax"); same("xaTPClsMax", "qTPClsMax"); same("xaUniClsMax", "qUniClsMax")
_old12 = [k_ for k_ in CLS14 if not k_.startswith("detoxify_unbiased")]
for nm_, ex_, why_ in (("xaUniLLMMax", "qUniLLMMax", "qUniLLMMax also covers the pooled with-parent LLMs, whose unstructured cut is larger (-0.85); here the four LLMs without the parent"),
                       ("xaUniLLMMin", "qUniLLMMin", "qUniLLMMin also covers the pooled with-parent LLMs (Haiku, Sonnet, Qwen, Gemma shown the parent); here the four LLMs without the parent only"),
                       ("xaUniClsMin", "qUniClsMin", "qUniClsMin covers 12 classifier settings (no Detoxify-unbiased); here all 14"),
                       ("xaFPLLMMin", "qFPLLMMin", "qFPLLMMin includes the pooled with-parent LLMs (Qwen shown the parent, +0.05); here the four LLMs without the parent"),
                       ("xaFPLLMMax", "qFPLLMMax", "qFPLLMMax includes Haiku and Sonnet shown the parent (pooled log RR); here the four LLMs without the parent, where Gemma's S_FP is the largest"),
                       ("xaFPClsMin", "qFPClsMin", "qFPClsMin covers 12 classifier settings; with Detoxify-unbiased at 0.6 and at gold's rate the minimum is lower"),
                       ("xaTPClsMin", "qTPClsMin", "qTPClsMin covers 12 classifier settings; here all 14"),
                       ("xaTPLLMMax", "qTPLLMMax", "qTPLLMMax includes the pooled with-parent LLMs; here the four LLMs without the parent"),
                       ("xaUniRRMin", "qUniRRMin", "qUniRRMin uses Haiku, Qwen, both shown the parent (pooled), and 12 classifiers on all reply threads; here the 4 LLMs without the parent (Sonnet and Gemma against the 5% sample's gold) and all 14 classifiers"),
                       ("xaUniRRMax", "qUniRRMax", "qUniRRMax uses Haiku, Qwen, both shown the parent (pooled), and 12 classifiers on all reply threads; here the 4 LLMs without the parent (Sonnet and Gemma against the 5% sample's gold) and all 14 classifiers"),
                       ("xaNClsTPPos", "qNClsTPPos", "qNClsTPPos counts 12 classifier settings; here 14")):
    if M[nm_] == EX[ex_]:
        logchk(nm_, f"= \\{ex_} ({EXF[ex_]}) although the set differs: {why_}")
    else:
        differs(nm_, ex_, why_)


# ================================================================ E04: escalation within threads (B75e)
item("E04")
WI = load("x75_within/x75_within.json")["within"]; SW = "results/x75_within/x75_within.json"
trip("xaGoldWithin", WI["cc_threads"]["gold_within"], SW, "within.cc_threads.gold_within", "x100; depth slope with thread fixed effects", sc=100)
trip("xaGoldWithinSfive", WI["cc_s5"]["gold_within"], SW, "within.cc_s5.gold_within", "x100", sc=100, caveat="5% sample")
same("xaGoldWithin", "qGoldWithinPP"); same("xaGoldWithinSfive", "qGoldWithinSfivePP")
for nm, pop, k in (("Haiku", "cc_threads", "haiku"), ("Qwen", "cc_threads", "qwen7b_m"), ("HaikuSfive", "cc_s5", "haiku"), ("Sonnet", "cc_s5", "sonnet"),
                   ("Gemma", "cc_s5", "gemma_m")):
    r_ = WI[pop]["labelers"][k]
    trip(f"xa{nm}WithinDist", r_["distortion"], SW, f"within.{pop}.labelers.{k}.distortion", "x100; labeler's within-thread law minus gold's", sc=100,
         caveat="5% sample" if pop == "cc_s5" else "")
    if nm in ("Haiku", "HaikuSfive", "Sonnet"):                                   # a share is meaningful only for the escalating labelers
        put(f"xa{nm}WithinFPShare", pct(r_["fp_term"][0] / r_["distortion"][0]), SW, f"within.{pop}.labelers.{k}.fp_term / distortion",
            "share of S_FP in the within-thread distortion", caveat="5% sample" if pop == "cc_s5" else "")
same("xaHaikuWithinDist", "dHaikuWithin"); same("xaSonnetWithinDist", "dSonnetWithin"); same("xaHaikuWithinFPShare", "qHaikuWithinFPShare")
put("xaWithinVarShare", pct(WI["cc_threads"]["share_of_variance_within"]), SW, "within.cc_threads.share_of_variance_within", "share of the depth variation within threads")
mx_ = max(abs(WI[p_]["labelers"][k_]["check"]) for p_ in ("cc_threads", "cc_s5") for k_ in WI[p_]["labelers"])
put("xaWithinCheck", "$10^{-12}$", SW, "within.*.labelers.*.check", "largest |sum of the three terms - distortion| is below this bound")
assert mx_ < 1e-12; logchk("xaWithinCheck", f"actual maximum {mx_:.1e}")
note("Within-thread values of the labelers shown the parent (x75_within haiku_ctx, qwen7b_m_ctx; qHaikuCtxWithinPP, qQwenCtxWithinPP; x76_rethreshold depth_within) "
     "mix the root, labeled without the parent, with replies labeled with it inside every thread, and are not used.")


# ================================================================ E08: the strictest level and prevalence matching (B92c, H74g, B88a)
item("E08")
EX2 = load("x92/x92_extra.json"); vt = EX2["cc_very_toxic"]; SX = "results/x92/x92_extra.json"
put("xaStrictFlag", pct(vt["flag_rate"], 1), SX, "cc_very_toxic.flag_rate", "x100 (share flagged at 'very toxic' only, all reply threads)"); same("xaStrictFlag", "cStrictFlag")
put("xaStrictTPR", pct(vt["tpr"], 1), SX, "cc_very_toxic.tpr", "x100"); put("xaStrictFPR", pct(vt["fpr"], 2), SX, "cc_very_toxic.fpr", "x100")
put("xaStrictLaw", sg(vt["law"], 3), SX, "cc_very_toxic.law", "points per level, 3 decimals")
put("xaStrictUni", sg(vt["uniform"]), SX, "cc_very_toxic.uniform", "unstructured term (points per level)"); same("xaStrictUni", "cStrictUni")
put("xaStrictUniThree", sg(vt["uniform"], 3), SX, "cc_very_toxic.uniform", "3 decimals")
put("xaStrictFP", sg(vt["S_FP"], 3), SX, "cc_very_toxic.S_FP", "S_FP, 3 decimals"); put("xaStrictTP", sg(vt["S_TP"], 3), SX, "cc_very_toxic.S_TP", "S_TP, 3 decimals")
differs("xaStrictFP", "cStrictFP", "the earlier-draft macro prints 0.00 (two decimals); here three decimals ($-$0.001), the same number")
hA = PLc["haiku@A"]["distortion"]
put("xaStrictExcess", sg(100 * hA["depth"]["est"]), SRC_PL, "cc_threads.labelers.haiku@A.distortion.depth.est", "x100 (depth law minus gold's)")
put("xaStrictExcessLo", sg(100 * hA["depth"]["ci"][0]), SRC_PL, "...depth.ci[0]", "x100"); put("xaStrictExcessHi", sg(100 * hA["depth"]["ci"][1]), SRC_PL, "...depth.ci[1]", "x100")
assert abs(100 * hA["depth"]["est"] - (vt["law"] - vt["gold"])) < 1e-9
put("xaStrictSizeDist", sg(100 * hA["size"]["est"]), SRC_PL, "cc_threads.labelers.haiku@A.distortion.size.est", "x100 (thread size law minus gold's, points per doubling)",
    caveat="contradicts or qualifies the story (Haiku's strict level reports a smaller size law than gold) -> Limitations")
put("xaStrictSizeDistLo", sg(100 * hA["size"]["ci"][0]), SRC_PL, "...size.ci[0]", "x100"); put("xaStrictSizeDistHi", sg(100 * hA["size"]["ci"][1]), SRC_PL, "...size.ci[1]", "x100")
EA = load("x74_eval/eval_A.json")["H74g"]
assert abs(EA["haiku@A"]["size"][0] - hA["size"]["ci"][0]) < 1e-12 and abs(EA["haiku@A"]["size"][1] - hA["size"]["ci"][1]) < 1e-12 and EA["pass_"] is False
logchk("xaStrictSizeDistLo", "equals results/x74_eval/eval_A.json H74g.haiku@A.size[0] (the log's 'CI -0.18 to -0.06')")
CA = load("x88/x88_calibrated.json")["pops"]; SC_ = "results/x88/x88_calibrated.json"
HL = CA["cc_threads"]["instruments"]["haiku:level"]["calibrated"]
vtc = [HL[f"prev@{n_}"]["cut_counts"].get("3", 0) for n_ in (200, 500, 1000)]
put("xaPrevVtMin", str(min(vtc)), SC_, "pops.cc_threads.instruments.haiku:level.calibrated.prev@{200,500,1000}.cut_counts['3']",
    "fewest of 500 draws in which prevalence matching on 200, 500 or 1,000 gold labels picks 'very toxic'")
put("xaPrevVtMax", str(max(vtc)), SC_, "same", "most of 500 draws")
put("xaPrevVtShareMin", pct(min(vtc) / 500, 1), SC_, "same", "fewest draws as a share"); same("xaPrevVtShareMin", "cCalPrevStrict")
S5L = CA["cc_s5"]["instruments"]["sonnet:level"]; pc3 = next(r_ for r_ in S5L["per_cut"] if r_["cut"] == 3.0)
put("xaSonnetVtFlag", pct(pc3["flag_rate"], 2), SC_, "pops.cc_s5.instruments.sonnet:level.per_cut[cut=3].flag_rate", "x100", caveat="5% sample")
put("xaSonnetVtExcess", sg(pc3["excess"]), SC_, "...per_cut[cut=3].excess", "depth law minus gold's (points per level)", caveat="5% sample")
put("xaSonnetVtZ", f(pc3["excess_z"], 1), SC_, "...per_cut[cut=3].excess_z", "article-clustered Wald z of the excess", caveat="5% sample")
put("xaSonnetPrevSigMax", pct(max(S5L["calibrated"][f"prev@{n_}"]["share_excess_sig_pos"] for n_ in (200, 500, 1000)), 1), SC_,
    "...sonnet:level.calibrated.prev@*.share_excess_sig_pos", "largest share of the 500 draws with a significant positive excess, prevalence matching")
same("xaSonnetVtExcess", "cCalPrevSonnet"); same("xaSonnetVtExcess", "dSonnetVt")
note("Sonnet's 'very toxic' excess \\xaSonnetVtExcess{} appears twice in the stored results with the same point value: x88_calibrated per_cut (z = \\xaSonnetVtZ) "
     "and x92_matched (CI \\xaSonnetVtDistLo{} to \\xaSonnetVtDistHi); both are on the 5% sample.")
MT2 = load("x92/x92_matched.json"); ROWS92 = {(r_["model"], r_["op"]): r_ for r_ in MT2["rows"]}; S92 = "results/x92/x92_matched.json"
for nm, m_ in (("Haiku", "haiku"), ("Sonnet", "sonnet")):
    r_ = ROWS92[(m_, "very toxic")]
    trip(f"xa{nm}VtDist", r_["depth_excess"], S92, f"rows[{m_}, very toxic].depth_excess", "depth law minus gold's (points per level)", caveat="5% sample")
    put(f"xa{nm}VtKappa", f(r_["kappa"]), S92, f"rows[{m_}, very toxic].kappa", "Cohen's kappa", caveat="5% sample")
same("xaHaikuVtDist", "dHaikuVt"); same("xaSonnetVtDist", "dSonnetVt")
note("C6: the excess of Haiku's 'very toxic' level over gold (\\xaStrictExcess{} on all reply threads) is unstructured error: the uniform term is \\xaStrictUniThree{} and S_FP is \\xaStrictFP; "
     "the level's thread size law is below gold's. This qualifies 'A positive distortion survives every threshold' -> Limitations.")


# ================================================================ E09: second-person address by depth (x79_you)
item("E09")
YO = load("x79/x79_you.json")["defs"]; SY = "results/x79/x79_you.json"
for dn, tag in (("gold_negative", "Neg"), ("tox_below_0.2", "Civil")):
    d0, dd = YO[dn]["d0_vs_d6plus"]["d0"], YO[dn]["d0_vs_d6plus"]["deep"]
    put(f"xaYouShareRoot{tag}", pct(d0["share_you"][0]), SY, f"defs.{dn}.d0_vs_d6plus.d0.share_you", "x100 (share with a second-person word, root comments)")
    put(f"xaYouShareDeep{tag}", pct(dd["share_you"][0]), SY, f"defs.{dn}.d0_vs_d6plus.deep.share_you", "x100 (depth 6 or more)")
    for w_, wn in (("haiku_on_you", "You"), ("haiku_on_other", "NoYou")):
        put(f"xaFpr{wn}Root{tag}", pct(d0[w_][0], 1), SY, f"defs.{dn}.d0_vs_d6plus.d0.{w_}", "x100 (Haiku's flag rate among these comments, root)")
        put(f"xaFpr{wn}Deep{tag}", pct(dd[w_][0], 1), SY, f"defs.{dn}.d0_vs_d6plus.deep.{w_}", "x100 (depth 6 or more)")
same("xaYouShareRootNeg", "cYouRootNeg"); same("xaYouShareDeepNeg", "cYouDeepNeg")
for a_, b_ in (("xaFprYouRootCivil", "qYouFprZero"), ("xaFprYouDeepCivil", "qYouFprSix"), ("xaFprNoYouRootCivil", "qNoYouFprZero"), ("xaFprNoYouDeepCivil", "qNoYouFprSix")):
    assert abs(un(M[a_]) / 100 - float(EX[b_])) < 0.0005 + 1e-12, (a_, b_); logchk(a_, f"= \\{b_} ({EX[b_]}, a proportion) in percent")
for a_, b_ in (("xaYouShareRootCivil", "qYouShareZero"), ("xaYouShareDeepCivil", "qYouShareSix")):
    assert abs(un(M[a_]) / 100 - float(EX[b_])) < 0.005 + 1e-12; logchk(a_, f"= \\{b_} ({EX[b_]}, a proportion) in percent")


# ================================================================ E10: dose and standardization (B90; x90_dose)
item("E10")
DO = load("x90/x90_dose.json")["corpora"]; SDO = "results/x90/x90_dose.json"; ct = DO["cc_threads"]; s5d = DO["cc_s5"]
WORDS = ("Zero", "One", "Two", "Three", "Four")
for nm, L_, pop in (("Haiku", ct["labelers"]["haiku"]["fpr_by_count"], "cc_threads.labelers.haiku"), ("Sonnet", s5d["labelers"]["sonnet"]["fpr_by_count"], "cc_s5.labelers.sonnet"),
                    ("Detox", ct["labelers"]["detoxify"]["fpr_by_count"], "cc_threads.labelers.detoxify"), ("Sr", ct["single_rater"]["fpr_by_count"], "cc_threads.single_rater"),
                    ("NoVote", ct["no_votes"]["haiku"], "cc_threads.no_votes.haiku")):
    for j, w_ in enumerate(WORDS):
        put(f"xaDose{nm}{w_}", pct(L_[j][0], 1), SDO, f"{pop}.fpr_by_count[{j}]", "x100 (FPR among gold negatives with 0,1,2,3,4+ second-person words)",
            caveat="5% sample" if nm == "Sonnet" else ("among gold negatives with no toxicity or subtype vote" if nm == "NoVote" else ""))
for a_, b_ in (("xaDoseDetoxZero", "cDoseDetZero"), ("xaDoseDetoxFour", "cDoseDetFour"), ("xaDoseSrZero", "cDoseSrZero"), ("xaDoseSrFour", "cDoseSrFour")):
    same(a_, b_)
for a_, b_ in (("xaDoseHaikuZero", "cDoseHaikuZero"), ("xaDoseHaikuOne", "cDoseHaikuOne"), ("xaDoseHaikuTwo", "cDoseHaikuTwo"), ("xaDoseHaikuThree", "cDoseHaikuThree"),
               ("xaDoseHaikuFour", "cDoseHaikuFour"), ("xaDoseSonnetZero", "cDoseSonZero"), ("xaDoseSonnetFour", "cDoseSonFour"), ("xaDoseNoVoteZero", "cDoseNoneZero"),
               ("xaDoseNoVoteFour", "cDoseNoneFour")):
    assert round(un(M[a_])) == un(EX[b_]); logchk(a_, f"rounds to \\{b_} ({EX[b_]}); one decimal here")
put("xaNoVoteShare", pct(ct["no_votes"]["share_of_neg"]), SDO, "cc_threads.no_votes.share_of_neg", "x100 (share of ALL gold negatives with no toxicity or subtype vote)")
_cf85 = load("x85/x85_construct_frac.json")
assert plain(pct(_cf85["share_none_among_neg"])) == plain(EX["qFrNone"])
differs("xaNoVoteShare", "qFrNone", f"same comments, different denominator: qFrNone (x85_construct_frac share_none_among_neg) divides by the gold negatives matched to the public "
        f"release ({plain(pct(_cf85['match_rate_neg'], 1))}), x90_dose by all gold negatives (unmatched ones count as voted). v7f says the climb holds among 'the 75% of gold "
        f"negatives' with no vote; as a share of all gold negatives it is {plain(M['xaNoVoteShare'])} ({plain(EX['qFrNone'])} of the matched ones)")
mcd = ct["mean_count_by_depth"]
put("xaDoseCountZero", f(mcd[0]), SDO, "cc_threads.mean_count_by_depth[0]", "mean number of second-person words, gold negatives, root")
put("xaDoseCountEight", f(mcd[8]), SDO, "cc_threads.mean_count_by_depth[8]", "depth 8 or more")
assert f(mcd[0], 1) == EX["cDoseCountZero"] and f(mcd[8], 1) == EX["cDoseCountEight"]; logchk("xaDoseCountZero", "one decimal = \\cDoseCountZero"); logchk("xaDoseCountEight", "one decimal = \\cDoseCountEight")
lb = ct["length_bands"]["haiku"]; assert all(all(b_[k_] < b_[k_ + 1] for k_ in range(4)) for b_ in lb)
put("xaDoseBands", str(len(lb)), SDO, "cc_threads.length_bands.haiku", "number of length bands in each of which Haiku's FPR rises with every added second-person word (asserted)")
hz = ct["labelers"]["haiku"]
trip("xaPlacWithinAny", hz["within_any"], SDO, "cc_threads.labelers.haiku.within_any", "Haiku's placebo depth slope pooled within presence of 'you' (points per level)")
trip("xaPlacWithinCount", hz["within_count"], SDO, "cc_threads.labelers.haiku.within_count", "pooled within the number of second-person words (0-4+)")
same("xaPlacWithinAny", "cPlacHaikuYou"); same3("xaPlacWithinCount", "cDoseWithinHaiku")
trip("xaDoseLawFix", hz["law_distortion_within_count"], SDO, "cc_threads.labelers.haiku.law_distortion_within_count", "depth-law distortion pooled within the number")
same("xaDoseLawFix", "cDoseLawHaikuFix")
sn = s5d["labelers"]["sonnet"]
trip("xaSonnetPlacWithinCount", sn["within_count"], SDO, "cc_s5.labelers.sonnet.within_count", "Sonnet's placebo slope within the number", caveat="5% sample")
same3("xaSonnetPlacWithinCount", "cDoseWithinSon")
trip("xaSonnetDoseLawFix", sn["law_distortion_within_count"], SDO, "cc_s5.labelers.sonnet.law_distortion_within_count", "Sonnet's depth-law distortion within the number", caveat="5% sample")
same("xaSonnetDoseLawFix", "cDoseLawSonFix")
for nm, k_, pop, cv in (("HaikuCtx", "haiku_ctx", ct, "among gold-negative replies, labeled with the parent"), ("QwenCtx", "qwen_ctx", ct, "among gold-negative replies; Qwen at its own gold-prevalence threshold"),
                        ("SonnetCtx", "sonnet_ctx", s5d, "5% sample; among gold-negative replies")):
    r_ = pop["labelers"][k_]; pp_ = "cc_s5" if pop is s5d else "cc_threads"
    trip(f"xa{nm}PlacReplies", r_["placebo"], SDO, f"{pp_}.labelers.{k_}.placebo", "placebo depth slope among gold-negative replies (points per level)", caveat=cv)
    trip(f"xa{nm}PlacWithinBoth", r_["within_count2"], SDO, f"{pp_}.labelers.{k_}.within_count2", "pooled within the reply's (0-4+) x the parent's (0-3+) number", caveat=cv)
same("xaHaikuCtxPlacReplies", "cDoseCtxPlac"); same("xaHaikuCtxPlacWithinBoth", "cDoseCtxWithin"); same("xaQwenCtxPlacReplies", "cDoseQwenPlac"); same("xaQwenCtxPlacWithinBoth", "cDoseQwenWithin")
assert ct["labelers"]["haiku_ctx"]["n"] == int(((zT == 0) & repT).sum())                         # among gold-negative replies, not pooled
MTd = s5d["matched"]; rl = {k_: v["ratio_4_0"] for k_, v in MTd.items() if v["kind"] == "llm"}; rc = {k_: v["ratio_4_0"] for k_, v in MTd.items() if v["kind"] == "classifier"}
assert sorted(rc) == sorted(CLS7M)
put("xaRatioLLMMin", f(min(v[0] for v in rl.values()), 1), SDO, "cc_s5.matched.<7 LLMs>.ratio_4_0", "smallest 4+:0 FPR ratio of the seven LLMs at gold's flag rate", caveat="5% sample")
put("xaRatioLLMMax", f(max(v[0] for v in rl.values()), 1), SDO, "same", "largest", caveat="5% sample")
put("xaRatioClsMin", f(min(v[0] for v in rc.values()), 1), SDO, "cc_s5.matched.<7 classifiers @ gold's rate>.ratio_4_0", "smallest of the 7 classifier settings at gold's flag rate",
    caveat="5% sample; only the 7 settings at gold's flag rate exist at one flag rate, the other 7 classifier settings flag at their usual thresholds")
put("xaRatioClsMax", f(max(v[0] for v in rc.values()), 1), SDO, "same", "largest (Detoxify)", caveat="5% sample")
same("xaRatioLLMMin", "cModRatioMin"); same("xaRatioLLMMax", "cModRatioMax"); same("xaRatioClsMin", "cClsRatioMin"); same("xaRatioClsMax", "cClsRatioMax")
assert max(rc, key=lambda k_: rc[k_][0]) == "detoxify_original@m"


# ================================================================ E11: construct divergence (B85c', B87c)
item("E11")
CF = load("x85/x85_construct_frac.json"); SCF = "results/x85/x85_construct_frac.json"; sl, lv = CF["slopes"], CF["levels"]
put("xaFracMatch", pct(CF["match_rate_neg"], 1), SCF, "match_rate_neg", "x100 (gold negatives matched to the public release)"); same("xaFracMatch", "qFrMatch")
put("xaInsLevel", pct(lv["insult_any"][0] / 100), SCF, "levels.insult_any", "share of gold negatives with at least one insult vote"); same("xaInsLevel", "qFrInsLevel")
trip("xaInsSlope", sl["insult_any"], SCF, "slopes.insult_any", "points per level"); same3("xaInsSlope", "qFrIns")
trip("xaFracHaikuSlope", sl["haiku"], SCF, "slopes.haiku", "Haiku's flag-rate slope among matched gold negatives, points per level"); same3("xaFracHaikuSlope", "qFrHaiku")
trip("xaFracHaikuNoVote", sl["haiku_zero_votes"], SCF, "slopes.haiku_zero_votes", "among gold negatives with no vote of any kind"); same3("xaFracHaikuNoVote", "qFrHaikuNone")
put("xaFracNoneShare", pct(CF["share_none_among_neg"]), SCF, "share_none_among_neg", "x100"); same("xaFracNoneShare", "qFrNone")
trip("xaIdaSlope", sl["identity_attack_any"], SCF, "slopes.identity_attack_any", "share with an identity-attack vote, points per level",
     caveat="contradicts or qualifies the story (real structure among borderline comments) -> Limitations")
same3("xaIdaSlope", "qFrIda")
trip("xaRejSlope", sl["rejected"], SCF, "slopes.rejected", "share rejected by the platform, points per level", caveat="contradicts or qualifies the story -> Limitations")
same3("xaRejSlope", "qFrRej")
put("xaRejLevel", pct(lv["rejected"][0] / 100, 1), SCF, "levels.rejected", "share rejected among gold negatives"); same("xaRejLevel", "qFrRejLevel")
put("xaRejSlopeShareOfHaiku", pct(sl["rejected"][0] / sl["haiku"][0]), SCF, "slopes.rejected / slopes.haiku", "the rejection slope as a share of Haiku's slope")
yu = CF["you"]
for k_, nm in (("insult_any_you", "InsYou"), ("insult_any_noyou", "InsNoYou"), ("rejected_you", "RejYou"), ("rejected_noyou", "RejNoYou"), ("haiku_you", "HaikuYou"), ("haiku_noyou", "HaikuNoYou")):
    put(f"xaCivil{nm}", pct(yu[k_], 1), SCF, f"you.{k_}", "x100, among matched clearly civil comments (vote fraction below 0.2)")
put("xaCivilYouShare", pct(yu["share_you"]), SCF, "you.share_you", "share of matched clearly civil comments with a second-person word")
G87 = load("x87/x87_gap.json")["B87c"]; S87 = "results/x87/x87_gap.json"
trip("xaApprovedSlope", G87["haiku"]["slope_approved"], S87, "B87c.haiku.slope_approved", "Haiku's slope among gold negatives with no vote that the platform approved, points per level")
put("xaApprovedN", th(G87["n_none_approved"]), S87, "B87c.n_none_approved", "number of such comments")
put("xaApprovedLevel", pct(G87["haiku"]["level_approved"][0] / 100, 1), S87, "B87c.haiku.level_approved", "Haiku's flag rate among them")
put("xaFracHaikuNoVoteLevel", pct(lv["haiku_zero_votes"][0] / 100, 1), SCF, "levels.haiku_zero_votes", "Haiku's flag rate among no-vote gold negatives"); same("xaFracHaikuNoVoteLevel", "qFrHaikuNoneLevel")
note("x85_construct_frac and x87_gap also hold Haiku shown the parent (haiku_ctx) slopes; these are pooled over roots labeled without the parent and are not used. "
     "results/x85/x85_construct_div.json is the buggy first run (K6) and is not read.")


# ================================================================ E12-E19: causal edits of "you"
item("E12")
F81A = load("f81_you_edit.json"); F81 = F81A["strata_train"]; S81 = "results/f81_you_edit.json"
CV81 = (f"first labeled run of the edit sample (f-series file; the same {F81A['n']:,} comments as x74_edit, data/you_edit_meta.parquet; train-split strata, the source of "
        "cc_threads); superseded by x74_edit except as a replication")
for st, nm in (("neg_deep", "Deep"), ("neg_root", "Root"), ("pos_any", "Pos")):
    r_ = F81[st]
    put(f"xaHeleven{nm}Before", pct(r_["flag_orig"], 1), S81, f"strata_train.{st}.flag_orig", "x100 (Haiku's flag rate, original text, same batch)", caveat=CV81)
    put(f"xaHeleven{nm}After", pct(r_["flag_edit"], 1), S81, f"strata_train.{st}.flag_edit", "x100 (after the rule rewrite)", caveat=CV81)
    trip(f"xaHeleven{nm}Diff", [r_["diff"]] + r_["ci"], S81, f"strata_train.{st}.diff, ci", "x100 (points; paired comment-level bootstrap, 2,000 resamples in scripts/f81_you_edit_analysis.py)", d=1, sc=100, caveat=CV81)
    put(f"xaHeleven{nm}N", th(r_["n"]), S81, f"strata_train.{st}.n", "comments", caveat=CV81)
put("xaHelevenGapBefore", sg(100 * (F81["neg_deep"]["flag_orig"] - F81["neg_root"]["flag_orig"]), 1), S81, "strata_train.neg_deep.flag_orig - neg_root.flag_orig",
    "x100: depth gap (deep minus root) on the originals", caveat=CV81)
put("xaHelevenGapAfter", sg(100 * (F81["neg_deep"]["flag_edit"] - F81["neg_root"]["flag_edit"]), 1), S81, "strata_train.neg_deep.flag_edit - neg_root.flag_edit",
    "x100: depth gap after the rewrite", caveat=CV81)

item("E13")
E74 = load("x74_edit/x74_edit.json"); S74E = "results/x74_edit/x74_edit.json"; EG = load("x75_edit/x75_edit_gaps.json"); SEG = "results/x75_edit/x75_edit_gaps.json"
E76 = load("x76_edit/x76_edit.json"); I76 = E76["instruments"]; S76 = "results/x76_edit/x76_edit.json"
EDCV = "edit sample: 1,489 paired clearly non-toxic second-person replies at depth >= 2, 1,082 roots, 876 gold positives (train split)"
for nm, m_ in (("Haiku", "haiku"), ("Sonnet", "sonnet")):
    I_ = I76[m_]
    for st, sn_ in (("neg_deep", "Deep"), ("neg_root", "Root"), ("pos_any", "Pos")):
        r_ = I_[f"h11/{st}"]
        put(f"xa{nm}Edit{sn_}Before", pct(r_["E0"], 1), S76, f"instruments.{m_}.h11/{st}.E0", "x100 (paired original, same batch)", caveat=EDCV)
        put(f"xa{nm}Edit{sn_}After", pct(r_["E1"], 1), S76, f"instruments.{m_}.h11/{st}.E1", "x100 (rule rewrite)", caveat=EDCV)
        trip(f"xa{nm}Edit{sn_}Diff", [r_["E1_minus_E0"]["diff"]] + r_["E1_minus_E0"]["ci"], S76, f"instruments.{m_}.h11/{st}.E1_minus_E0", "x100 (points)", d=1, sc=100, caveat=EDCV)
        put(f"xa{nm}Para{sn_}You", pct(r_["E3"], 1), S76, f"instruments.{m_}.h11/{st}.E3", "x100 (Sonnet's paraphrase keeping 'you')", caveat=EDCV)
        put(f"xa{nm}Para{sn_}They", pct(r_["E4"], 1), S76, f"instruments.{m_}.h11/{st}.E4", "x100 (paraphrase in the third person)", caveat=EDCV)
        trip(f"xa{nm}Para{sn_}Diff", [r_["E4_minus_E3"]["diff"]] + r_["E4_minus_E3"]["ci"], S76, f"instruments.{m_}.h11/{st}.E4_minus_E3", "x100 (points)", d=1, sc=100, caveat=EDCV)
        trip(f"xa{nm}ParaKeep{sn_}Diff", [r_["E3_minus_E0"]["diff"]] + r_["E3_minus_E0"]["ci"], S76, f"instruments.{m_}.h11/{st}.E3_minus_E0",
             "x100: paraphrase keeping 'you' minus the original (points)", d=1, sc=100, caveat=EDCV)
    for st, sn_ in (("neg_deep", "Deep"), ("neg_root", "Root"), ("pos_any", "Pos")):
        r_ = I_[f"rev/{st}"]; src_ = S76; kp_ = f"instruments.{m_}.rev/{st}"
        if m_ == "haiku":                                                             # x74_edit holds the same points (different bootstrap)
            assert abs(E74["rev"][st]["E2_minus_E0"]["diff"] - r_["E2_minus_E0"]["diff"]) < 1e-12 and abs(E74["rev"][st]["flag_E0"] - r_["E0"]) < 1e-12
        put(f"xa{nm}Rev{sn_}Before", pct(r_["E0"], 1), src_, kp_ + ".E0", "x100 (original, third person, same batch)",
            caveat="reverse edit: 1,500 clearly non-toxic replies at depth >= 2 without 'you', 1,500 roots, 1,000 gold positives")
        put(f"xa{nm}Rev{sn_}After", pct(r_["E2"], 1), src_, kp_ + ".E2", "x100 (third person rewritten as 'you')")
        trip(f"xa{nm}Rev{sn_}Diff", [r_["E2_minus_E0"]["diff"]] + r_["E2_minus_E0"]["ci"], src_, kp_ + ".E2_minus_E0", "x100 (points)", d=1, sc=100)
# the earlier-draft macros (x74_edit / x16 / x17), and the paired rates of x76_edit against x74_edit's unpaired ones
same("xaHaikuEditDeepDiff", "qEditOneDeep"); same("xaHaikuEditRootDiff", "qEditOneRoot"); same("xaHaikuRevDeepDiff", "qRevDeep"); same("xaHaikuRevRootDiff", "qRevRoot")
same("xaHaikuRevPosDiff", "qRevPos")
assert round(un(M["xaHaikuRevDeepBefore"])) == un(EX["cRevHaikuBefore"]) and round(un(M["xaHaikuRevDeepAfter"])) == un(EX["cRevHaikuAfter"])
logchk("xaHaikuRevDeepBefore", "rounds to \\cRevHaikuBefore (10\\%)"); logchk("xaHaikuRevDeepAfter", "rounds to \\cRevHaikuAfter (15\\%)")
assert M["xaHaikuParaDeepYou"] == EX["dParaYou"] and M["xaHaikuParaDeepThey"] == EX["dParaThey"]; logchk("xaHaikuParaDeepYou", "= \\dParaYou"); logchk("xaHaikuParaDeepThey", "= \\dParaThey")
assert M["xaHaikuParaDeepDiff"] == "$-$" + EX["wEditParaDiff"] and M["xaHaikuParaDeepDiffLo"] == "$-$" + EX["wEditParaHi"] and M["xaHaikuParaDeepDiffHi"] == "$-$" + EX["wEditParaLo"]
logchk("xaHaikuParaDeepDiff", "= -\\wEditParaDiff (the earlier-draft macro is the unsigned difference)")
for a_, b_ in (("xaSonnetParaDeepDiff", "qSonParaDiff"), ("xaSonnetParaDeepDiffLo", "qSonParaHi"), ("xaSonnetParaDeepDiffHi", "qSonParaLo")):
    assert M[a_] == "$-$" + EX[b_]; logchk(a_, f"= -\\{b_} (the earlier-draft macro is unsigned)")
for a_, b_ in (("xaSonnetRevDeepDiff", "qSonRevDiff"), ("xaSonnetRevDeepDiffLo", "qSonRevLo"), ("xaSonnetRevDeepDiffHi", "qSonRevHi")):
    same(a_, b_)
assert f(I76["sonnet"]["h11/neg_deep"]["E0"]) == EX["qSonOneBefore"] and f(I76["sonnet"]["h11/neg_deep"]["E1"]) == EX["qSonOneAfter"]
logchk("xaSonnetEditDeepBefore", "= \\qSonOneBefore as a proportion"); logchk("xaSonnetEditDeepAfter", "= \\qSonOneAfter as a proportion")
_e74 = E74["rev"]["neg_deep"]["E2_minus_E0"]
note(f"Haiku's reverse-rewrite intervals come from x76_edit (one bootstrap for all instruments); x74_edit holds the same point values with its own "
     f"bootstrap (deep: {sg(100 * _e74['diff'], 1)} [{sg(100 * _e74['ci'][0], 1)}, {sg(100 * _e74['ci'][1], 1)}] there, [\\xaHaikuRevDeepDiffLo, \\xaHaikuRevDeepDiffHi] here). "
     f"x74_edit's flag_E0 for the {th(E74['h11']['neg_deep']['n'])} deep comments ({pct(E74['h11']['neg_deep']['flag_E0'], 1)}) is unpaired; x76_edit's \\xaHaikuEditDeepBefore{{}} "
     f"is over the {th(I76['haiku']['h11/neg_deep']['n'])} comments with both an original and a rule rewrite, the comparison the paired change uses. The earlier-draft macro "
     f"wEditOneBefore ({EX['wEditOneBefore']}) is the unpaired rate.")
# depth gaps (deep minus root) on the edit sample
for nm, G_, src_ in (("Haiku", EG, SEG), ("Sonnet", load("x79/x79_songap.json"), "results/x79/x79_songap.json")):
    h_ = G_["h11_train"]
    for v_, vn in (("E0", "Orig"), ("E1", "Rule"), ("E3", "ParaYou"), ("E4", "ParaThey")):
        put(f"xa{nm}Gap{vn}", sg(100 * h_["variants"][v_]["gap"], 1), src_, f"h11_train.variants.{v_}.gap", "x100: flag rate at depth >= 2 minus roots (points)", caveat=EDCV)
        put(f"xa{nm}Gap{vn}Lo", sg(100 * h_["variants"][v_]["ci"][0], 1), src_, f"h11_train.variants.{v_}.ci[0]", "x100")
        put(f"xa{nm}Gap{vn}Hi", sg(100 * h_["variants"][v_]["ci"][1], 1), src_, f"h11_train.variants.{v_}.ci[1]", "x100")
    for c_, cn in (("gap_E1_minus_E0", "CloseRule"), ("gap_E4_minus_E3", "ClosePara")):
        trip(f"xa{nm}Gap{cn}", [h_[c_]["diff"]] + h_[c_]["ci"], src_, f"h11_train.{c_}", "x100: change of the gap (points)", d=1, sc=100)
    put(f"xa{nm}GapCloseShare", pct(-h_["gap_E1_minus_E0"]["diff"] / h_["variants"]["E0"]["gap"]), src_, "h11_train.gap_E1_minus_E0.diff / variants.E0.gap",
        "share of the original gap closed by the rule rewrite")
    rv_ = G_["rev"]
    put(f"xa{nm}RevGapOrig", sg(100 * rv_["variants"]["E0"]["gap"], 1), src_, "rev.variants.E0.gap", "x100 (comments without 'you': deep minus root)")
    put(f"xa{nm}RevGapYou", sg(100 * rv_["variants"]["E2"]["gap"], 1), src_, "rev.variants.E2.gap", "x100 (after adding 'you')")
    trip(f"xa{nm}RevGapChange", [rv_["gap_E2_minus_E0"]["diff"]] + rv_["gap_E2_minus_E0"]["ci"], src_, "rev.gap_E2_minus_E0", "x100", d=1, sc=100)
for a_, b_ in (("xaHaikuGapOrig", "qGapZero"), ("xaHaikuGapOrigLo", "qGapZeroLo"), ("xaHaikuGapOrigHi", "qGapZeroHi"), ("xaHaikuGapRule", "qGapOne"), ("xaHaikuGapRuleLo", "qGapOneLo"),
               ("xaHaikuGapRuleHi", "qGapOneHi"), ("xaHaikuGapParaYou", "qGapThree"), ("xaHaikuGapParaThey", "qGapFour"), ("xaHaikuGapParaTheyLo", "qGapFourLo"),
               ("xaHaikuGapParaTheyHi", "qGapFourHi"), ("xaHaikuRevGapOrig", "qRevGapZero"), ("xaHaikuRevGapYou", "qRevGapTwo"), ("xaSonnetGapOrig", "qSonGapZero"),
               ("xaSonnetGapOrigLo", "qSonGapZeroLo"), ("xaSonnetGapOrigHi", "qSonGapZeroHi"), ("xaSonnetGapRule", "qSonGapOne"), ("xaSonnetGapRuleLo", "qSonGapOneLo"),
               ("xaSonnetGapRuleHi", "qSonGapOneHi"), ("xaSonnetGapParaYou", "qSonGapThreeV"), ("xaSonnetGapParaYouLo", "qSonGapThreeLo"), ("xaSonnetGapParaYouHi", "qSonGapThreeHi"),
               ("xaSonnetGapParaThey", "qSonGapFourV"), ("xaSonnetGapParaTheyLo", "qSonGapFourLo"), ("xaSonnetGapParaTheyHi", "qSonGapFourHi"), ("xaSonnetGapCloseShare", "qSonGapCut")):
    same(a_, b_)
same3("xaHaikuGapCloseRule", "qGapCloseOne"); same3("xaHaikuGapClosePara", "qGapCloseFour"); same3("xaHaikuRevGapChange", "qRevGapChange")

item("E15")
ALLI = ["haiku", "sonnet"] + CLS14
add = {k_: I76[k_]["rev/neg_deep"] for k_ in ALLI}; rem = {k_: I76[k_]["h11/neg_deep"] for k_ in ALLI}
radd = {k_: add[k_]["E2_minus_E0"]["diff"] / add[k_]["E0"] for k_ in ALLI}; rrem = {k_: rem[k_]["E1_minus_E0"]["diff"] / rem[k_]["E0"] for k_ in ALLI}
put("xaClsAddMin", sg(100 * min(add[k_]["E2_minus_E0"]["diff"] for k_ in CLS14), 1), S76, "instruments.<14 settings>.rev/neg_deep.E2_minus_E0.diff", "x100; smallest change on adding 'you'")
put("xaClsAddMax", sg(100 * max(add[k_]["E2_minus_E0"]["diff"] for k_ in CLS14), 1), S76, "same", "largest (IMSYPP at argmax, which flags a third of comments)")
put("xaClsAddMMin", sg(100 * min(add[k_]["E2_minus_E0"]["diff"] for k_ in CLS7M), 1), S76, "instruments.<7 settings at gold's rate>.rev/neg_deep.E2_minus_E0.diff", "x100; smallest of the 7 at gold's rate")
put("xaClsAddMMax", sg(100 * max(add[k_]["E2_minus_E0"]["diff"] for k_ in CLS7M), 1), S76, "same", "largest of the 7")
put("xaClsRemMin", sg(100 * min(rem[k_]["E1_minus_E0"]["diff"] for k_ in CLS14), 1), S76, "instruments.<14 settings>.h11/neg_deep.E1_minus_E0.diff", "x100; change on removing 'you'")
put("xaClsRemMax", sg(100 * max(rem[k_]["E1_minus_E0"]["diff"] for k_ in CLS14), 1), S76, "same", "largest")
put("xaClsRemMMin", sg(100 * min(rem[k_]["E1_minus_E0"]["diff"] for k_ in CLS7M), 1), S76, "instruments.<7 at gold's rate>.h11/neg_deep.E1_minus_E0.diff", "x100")
put("xaClsRemMMax", sg(100 * max(rem[k_]["E1_minus_E0"]["diff"] for k_ in CLS7M), 1), S76, "same", "x100")
same("xaClsAddMMin", "qClsRevMin"); same("xaClsAddMMax", "qClsRevMax"); same("xaClsRemMMin", "qClsYouMin"); same("xaClsRemMMax", "qClsYouMax")
_amx = max(CLS14, key=lambda k_: add[k_]["E2_minus_E0"]["diff"]); _amn = min(CLS14, key=lambda k_: add[k_]["E2_minus_E0"]["diff"])
assert (_amx, _amn) == ("imsypp@argmax", "fb_hate@0.5")
differs("xaClsAddMax", "qClsRevMax", f"the earlier-draft range (qClsRevMin/Max, x76_edit macros) covers the 7 settings at gold's flag rate; over all 14 settings IMSYPP at argmax "
        f"(which flags {pct(np.nan_to_num(PT['y:imsypp@argmax'].values.astype(float)).mean())} of all comments) adds {M['xaClsAddMax']} points and Dynabench at 0.5 changes by {M['xaClsAddMin']}")
differs("xaClsAddMin", "qClsRevMin", "as xaClsAddMax: 14 settings here, 7 in the earlier-draft range")
logchk("xaClsRemMin", "over 14 settings equal to the 7-setting value \\qClsYouMin" if M["xaClsRemMin"] == EX["qClsYouMin"] else "differs from \\qClsYouMin")
logchk("xaClsRemMax", "over 14 settings equal to \\qClsYouMax" if M["xaClsRemMax"] == EX["qClsYouMax"] else "differs from \\qClsYouMax")
for k_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet"), ("detoxify_original@m", "DetoxM"), ("detoxify_original@0.6", "DetoxSix")):
    put(f"xa{nm}AddRel", spct(radd[k_]), S76, f"instruments.{k_}.rev/neg_deep.E2_minus_E0.diff / E0", "relative change on adding 'you' (x100)",
        caveat="Detoxify's base rate is 1% (0.3% at 0.6): the relative change rests on 16 (4) flagged comments of 1,500" if k_.startswith("detox") else "")
    put(f"xa{nm}RemRel", spct(rrem[k_]), S76, f"instruments.{k_}.h11/neg_deep.E1_minus_E0.diff / E0", "relative change on removing 'you' (x100)")
AC = E76["across"]
put("xaYouRho", sg(AC["spearman_abs"]), S76, "across.spearman_abs", "Spearman correlation over 9 instruments (Haiku, Sonnet, 7 classifiers at gold's rate) between the change on adding 'you' and the relative placebo depth slope")
put("xaYouRhoRel", sg(AC["spearman_rel"]), S76, "across.spearman_rel", "the same with the relative change")
put("xaYouRhoN", str(len(AC["instruments"])), S76, "across.instruments", "number of instruments")
same("xaYouRho", "qYouRho"); same("xaYouRhoN", "qYouRhoN")
from scipy.stats import spearmanr
prel = {k_: (PL5 if k_ == "sonnet" else PLc)[k_]["placebo_depth_rel"][0] for k_ in ALLI}
assert all(abs(prel[k_] - v_) < 1e-12 for k_, v_ in zip(AC["instruments"], AC["placebo_depth_rel"]))
rho16 = spearmanr([add[k_]["E2_minus_E0"]["diff"] for k_ in ALLI], [prel[k_] for k_ in ALLI]).correlation
rho16r = spearmanr([radd[k_] for k_ in ALLI], [prel[k_] for k_ in ALLI]).correlation
put("xaYouRhoAll", sg(rho16), S76 + "; " + SRC_PL, "instruments.<16>.rev/neg_deep.E2_minus_E0.diff; labelers.<16>.placebo_depth_rel",
    "Spearman over Haiku, Sonnet and all 14 classifier settings (computed here)", caveat="computed here, not in the registration log")
put("xaYouRhoAllRel", sg(rho16r), S76 + "; " + SRC_PL, "same, relative change", "Spearman with the relative change over 16 instruments", caveat="computed here")
put("xaYouRhoAllN", str(len(ALLI)), S76, "instruments", "16 instruments")
rows = []
for k_ in ALLI:
    if k_ in ("haiku", "sonnet"):
        nm_, cut_ = ("Haiku" if k_ == "haiku" else "Sonnet"), "own cut"
    else:
        nm_, cut_ = CLSDISP[k_][0] if not k_.endswith("@m") else "", CLSDISP[k_][1]
        if k_ == "detoxify_original@0.6":
            rows.append("\\midrule")
        elif not k_.endswith("@m"):
            rows.append("\\addlinespace[1pt]")
    a_, r_ = add[k_], rem[k_]
    rows.append(" & ".join([nm_, cut_, pct(a_["E0"], 1).replace("\\%", ""), ci3([a_["E2_minus_E0"]["diff"]] + a_["E2_minus_E0"]["ci"], 1, 100), sg(100 * radd[k_], 0),
                            pct(r_["E0"], 1).replace("\\%", ""), ci3([r_["E1_minus_E0"]["diff"]] + r_["E1_minus_E0"]["ci"], 1, 100), sg(100 * rrem[k_], 0),
                            sg(100 * prel[k_], 1)]) + " \\\\")
table("xa_rewrite_cls", rows,
      "How each labeler responds to editing 'you' on the edit sample: flag rate on clearly non-toxic replies at depth 2 or more, and its change when third-person "
      f"references are rewritten as 'you' ({th(add['haiku']['n'])} comments without 'you') and when 'you' is rewritten in the third person ({th(rem['haiku']['n'])} comments with 'you'), in points and "
      "relative to the base rate (%); last column, the labeler's relative placebo depth slope (% of its FPR per level). Classifiers at their usual threshold and at "
      "gold's flag rate; relative changes rest on few flags where the base rate is below 1%. Spearman correlation of the change on adding 'you' with the relative placebo depth slope: "
      f"\\xaYouRho{{}} over the {len(AC['instruments'])} instruments of the original analysis (Haiku, Sonnet and the {len(CLS7M)} classifiers at gold's rate), \\xaYouRhoAll{{}} over all {len(ALLI)}. "
      "Section sign: trained at least in part on Civil Comments labels.",
      "Labeler | Cut | Adding 'you': base (%) | change, points [CI] | relative (%) | Removing 'you': base (%) | change, points [CI] | relative (%) | Placebo depth slope (% of FPR)",
      S76 + "; " + SRC_PL + " (relative placebo slopes)")
_b7 = [add[k_]["E0"] for k_ in CLS7M]
note(f"C9 and E15: relative to their base rates, some classifiers react to adding 'you' as strongly as the LLMs (Detoxify at gold's rate \\xaDetoxMAddRel{{}} against "
     f"Haiku's \\xaHaikuAddRel); their absolute changes are small because, at gold's rate, their base rates are {pct(min(_b7), 1)}-{pct(max(_b7), 1)} on these comments. Report both.")

item("E16")
YF = load("x77_youfree/x77_youfree.json"); SYF = "results/x77_youfree/x77_youfree.json"; L77 = YF["laws"]
put("xaYfNYou", th(YF["n_you_labeled"]), SYF, "n_you_labeled", "second-person comments of the 5% sample labeled as written and rewritten"); same("xaYfNYou", "qYfNYou")
for law, nm, sc_ in (("depth", "", 100), ("depth_within", "Within", 100), ("size", "Size", 100)):
    r_ = L77[law]
    trip(f"xaYf{nm}Orig", r_["dist_orig"], SYF, f"laws.{law}.dist_orig", "x100: distortion as written (same batch)", sc=sc_, caveat="5% sample")
    trip(f"xaYf{nm}Edit", r_["dist_edit"], SYF, f"laws.{law}.dist_edit", "x100: distortion after the rewrite", sc=sc_, caveat="5% sample")
    trip(f"xaYf{nm}Chg", r_["edit_minus_orig"], SYF, f"laws.{law}.edit_minus_orig", "x100: paired change", sc=sc_, caveat="5% sample")
    put(f"xaYf{nm}Share", pct(r_["share_removed"]), SYF, f"laws.{law}.share_removed", "share of the distortion removed", caveat="5% sample")
same("xaYfOrig", "qYfOrig"); same("xaYfEdit", "qYfEdit"); same("xaYfShare", "qYfShare"); same("xaYfChgLo", "qYfLo"); same("xaYfChgHi", "qYfHi")
same("xaYfWithinShare", "qYfWithinShare"); same("xaYfSizeShare", "qYfSizeShare"); same("xaYfSizeShare", "cYfSizeShare")
logchk("xaYfWithinShare", "the inventory says 100%: the stored share is 1.005 (the within-thread distortion after the rewrite is $-$0.004 x 0.01 points), printed 101\\%")
fr77 = YF["flag_rate"]
for k_, nm in (("gold_pos_orig", "PosOrig"), ("gold_pos_edit", "PosEdit"), ("neg_orig", "NegOrig"), ("neg_edit", "NegEdit")):
    put(f"xaYf{nm}", pct(fr77[k_], 1), SYF, f"flag_rate.{k_}", "x100 (flag rate among gold positives / negatives of the 5% sample)", caveat="5% sample")
assert f(fr77["gold_pos_orig"]) == EX["qYfPosOrig"] and f(fr77["gold_pos_edit"]) == EX["qYfPosEdit"]
logchk("xaYfPosOrig", "= \\qYfPosOrig as a proportion"); logchk("xaYfPosEdit", "= \\qYfPosEdit as a proportion")
_rec("xaYfPosEdit")["caveat"] += "; contradicts or qualifies the story (the rewrite also lowers flags on toxic comments) -> Limitations"

item("E17")
RW = load("x89/x89_rewrite.json"); SRW = "results/x89/x89_rewrite.json"; sn_ = RW["sonnet_noctx"]
# replicate x89_rewrite.py analyze (i) on the stored answers, to get the paired CIs it did not store and the gold positives' flag rates
ANS = {}
for sub in ("out_s", "out_c"):
    for fn in sorted(glob.glob(f"{R}/x89/rewrite/{sub}/req_*.out.jsonl")):
        for l in open(fn):
            r_ = json.loads(l); ANS[r_["custom_id"]] = r_["ans"]
flg = lambda a: float(a in ("A", "B"))
idS = PS.id.astype(int).values; xS = np.minimum(PS.depth.values, 10).astype(float); clS = pd.factorize(PS.cluster.values)[0]; nCS = int(clS.max() + 1)
uS = pd.factorize(PS.unit.values)[0]; xtS = xS - (np.bincount(uS, xS) / np.bincount(uS))[uS]
Wb89 = np.vstack([np.ones(nCS), np.random.default_rng(8905).poisson(1.0, size=(500, nCS))])
S89 = lambda v: Wb89 @ np.bincount(clS, v, nCS)
def slope89(xv, v, mask):
    m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S89(m_ * a_) for a_ in (np.ones_like(xv), xv, v, xv * xv, xv * v))
    return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
def size89(v):
    ku = np.bincount(uS).astype(float); ok = ku >= 2; fu = np.bincount(uS, v) / ku; lx = np.log2(ku); ucl = np.zeros(ku.size, int); ucl[uS] = clS
    w = Wb89[:, ucl[ok]]; lxo, fuo = lx[ok], fu[ok]; sw = w.sum(1); mx = (w * lxo).sum(1) / sw; my = (w * fuo).sum(1) / sw
    return 100 * (w * (lxo - mx[:, None]) * (fuo - my[:, None])).sum(1) / (w * (lxo - mx[:, None]) ** 2).sum(1)
ci89 = lambda a: [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
allS = np.ones(NS, bool); g89 = slope89(xS, zS, allS); gw89 = slope89(xtS, zS, allS); gs89 = size89(zS)
yst = np.nan_to_num(PS["y:sonnet"].values.astype(float)); hv = np.array([f"o_{i}" in ANS for i in idS])
yo = yst.copy(); ye = yst.copy()
for j in np.flatnonzero(hv):
    yo[j] = flg(ANS[f"o_{idS[j]}"]); ye[j] = flg(ANS[f"e_{idS[j]}"])
do_, de_ = slope89(xS, yo, allS) - g89, slope89(xS, ye, allS) - g89; wo_, we_ = slope89(xtS, yo, allS) - gw89, slope89(xtS, ye, allS) - gw89
so_, se_ = size89(yo) - gs89, size89(ye) - gs89
for a_, b_ in ((ci89(do_), sn_["dist_orig"]), (ci89(de_), sn_["dist_edit"]), (ci89(de_ - do_), sn_["change"]), (ci89(wo_), sn_["within_orig"]), (ci89(we_), sn_["within_edit"]),
               (ci89(so_), sn_["size_orig"]), (ci89(se_), sn_["size_edit"])):
    assert np.allclose(a_, b_, rtol=0, atol=1e-9), (a_, b_)                      # the replication reproduces every stored value
assert int(hv.sum()) == sn_["n_you"]
CVS = "5% sample; recomputed from the stored answers (results/x89/rewrite/out_s) with x89_rewrite.py's resamples (seed 8905), which reproduce every stored value"
for nm, v_, how in (("Orig", ci89(do_), "distortion as written"), ("Edit", ci89(de_), "after the rewrite"), ("Chg", ci89(de_ - do_), "paired change"),
                    ("WithinOrig", ci89(wo_), "within-thread distortion as written"), ("WithinEdit", ci89(we_), "within-thread, rewritten"), ("WithinChg", ci89(we_ - wo_), "within-thread paired change"),
                    ("SizeOrig", ci89(so_), "thread size law distortion as written"), ("SizeEdit", ci89(se_), "thread size, rewritten"), ("SizeChg", ci89(se_ - so_), "thread size paired change")):
    stored_ = nm in ("Orig", "Edit", "Chg", "WithinOrig", "WithinEdit", "SizeOrig", "SizeEdit")
    trip(f"xaRwSon{nm}", v_, SRW if stored_ else "recomputed (x89_rewrite.py analysis (i))", f"sonnet_noctx.{ {'Orig': 'dist_orig', 'Edit': 'dist_edit', 'Chg': 'change', 'WithinOrig': 'within_orig', 'WithinEdit': 'within_edit', 'SizeOrig': 'size_orig', 'SizeEdit': 'size_edit'}.get(nm, '(not stored)') }",
         how + " (points per level; per doubling for the size law)", caveat="5% sample" if stored_ else CVS)
put("xaRwSonShare", pct(sn_["share_removed"]), SRW, "sonnet_noctx.share_removed", "share of the depth-law distortion removed", caveat="5% sample"); same("xaRwSonShare", "cRwSonShare")
put("xaRwSonWithinShare", pct(1 - sn_["within_edit"][0] / sn_["within_orig"][0]), SRW, "1 - sonnet_noctx.within_edit / within_orig", "share removed within threads", caveat="5% sample")
put("xaRwSonSizeShare", pct(sn_["size_share_removed"]), SRW, "sonnet_noctx.size_share_removed", "share of the size-law distortion removed", caveat="5% sample")
same("xaRwSonSizeShare", "cRwSonSize"); same("xaRwSonOrig", "cRwSonOrig"); same("xaRwSonEdit", "cRwSonEdit")
for nm, v_ in (("PosOrig", yo[zS == 1].mean()), ("PosEdit", ye[zS == 1].mean()), ("NegOrig", yo[zS == 0].mean()), ("NegEdit", ye[zS == 0].mean())):
    put(f"xaRwSon{nm}", pct(v_, 1), "recomputed (x89_rewrite.py analysis (i))", "flags yo / ye of analysis (i) among gold positives / negatives",
        "x100 (Sonnet's flag rate as written / rewritten)", caveat=CVS + ("; contradicts or qualifies the story -> Limitations" if nm == "PosEdit" else ""))
# the same construction for Haiku (x77_youfree) gives its stored gold-positive rates: a check of the construction
H77 = pd.DataFrame([json.loads(l) for fn in sorted(glob.glob(f"{R}/x77_youfree/haiku_tox/shard_*.jsonl")) for l in open(fn)])
if "error" in H77:
    H77 = H77[H77["error"].isna()]
H77 = H77.drop_duplicates("id", keep="last"); H77["flag"] = H77.ans.isin(["A", "B"]).astype(float); H77["cid"] = H77.id // 10; H77["var"] = H77.id % 10
Pv = H77.pivot(index="cid", columns="var", values="flag"); o77 = Pv.reindex(idS)[0].values; e77 = Pv.reindex(idS)[1].values; ok77 = np.isfinite(o77) & np.isfinite(e77)
base77 = np.nan_to_num(PS["y:haiku"].values.astype(float)); yo77 = np.where(ok77, o77, base77); ye77 = np.where(ok77, e77, base77)
assert abs(yo77[zS == 1].mean() - fr77["gold_pos_orig"]) < 1e-12 and abs(ye77[zS == 1].mean() - fr77["gold_pos_edit"]) < 1e-12

item("E18")
for nm, k_ in (("Haiku", "haiku_ctx"), ("Sonnet", "sonnet_ctx")):
    r_ = RW[k_]
    for v_ in ("v0", "v1", "v2"):
        trip(f"xaRwCtx{nm}{ {'v0': 'Orig', 'v1': 'Reply', 'v2': 'Both'}[v_] }", r_["dist_replies"][v_], SRW, f"{k_}.dist_replies.{v_}",
             "distortion of the depth law among replies (points per level): as written / reply rewritten / reply and parent rewritten", caveat="5% sample; among replies")
    trip(f"xaRwCtx{nm}ChgReply", r_["change_v1_replies"], SRW, f"{k_}.change_v1_replies", "paired change V1 - V0 among replies", caveat="5% sample")
    trip(f"xaRwCtx{nm}ChgBoth", r_["change_v2_replies"], SRW, f"{k_}.change_v2_replies", "paired change V2 - V0 among replies", caveat="5% sample")
    put(f"xaRwCtx{nm}ShareReply", pct(r_["share_removed_v1_replies"]), SRW, f"{k_}.share_removed_v1_replies", "share removed by rewriting the reply", caveat="5% sample")
    put(f"xaRwCtx{nm}ShareBoth", pct(r_["share_removed_v2_replies"]), SRW, f"{k_}.share_removed_v2_replies", "share removed by rewriting reply and parent",
        caveat="5% sample; contradicts or qualifies the story -> Limitations (already in v7f)")
    put(f"xaRwCtx{nm}N", th(r_["n_replies"]), SRW, f"{k_}.n_replies", "replies whose text or parent addresses 'you'")
same("xaRwCtxHaikuShareBoth", "cRwCtxShareH"); same("xaRwCtxSonnetShareBoth", "cRwCtxShareS")
assert M["xaRwCtxHaikuChgReply"] == "$-$" + EX["cRwCtxHaiku"] and M["xaRwCtxSonnetChgReply"] == "$-$" + EX["cRwCtxSonnet"]
logchk("xaRwCtxHaikuChgReply", "= -\\cRwCtxHaiku (unsigned in the earlier draft)"); logchk("xaRwCtxSonnetChgReply", "= -\\cRwCtxSonnet (unsigned)")
note(f"x89_rewrite.json also stores the pooled with-parent distortions (dist.v0 = {sg(RW['haiku_ctx']['dist']['v0'][0])} for Haiku, used by the inventory's E18 "
     "'+2.24 -> +1.69 -> +1.56'); they include roots labeled without the parent and are not used. The among-replies values are \\xaRwCtxHaikuOrig{} -> "
     "\\xaRwCtxHaikuReply{} -> \\xaRwCtxHaikuBoth.")
# the both-address detail: Haiku's (and Sonnet's) FPR among gold-negative replies whose reply and parent both address 'you', by depth (1 ... 5+)
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
TXT = {}
for fn in sorted(glob.glob(f"{W}/data/second/s5/shard_*.jsonl")):
    for l in open(fn):
        r_ = json.loads(l); TXT[int(r_["id"])] = r_.get("text") or ""
assert all(int(i) in TXT for i in idS)
youS = np.array([bool(YOU.search(TXT[int(i)])) for i in idS]); prS = PS.parent_row.values
pyS = np.zeros(NS, bool); pyS[repS] = youS[prS[repS]]; d5 = np.minimum(PS.depth.values, 5)
both = {}
for tag, nm in (("h", "Haiku"), ("s", "Sonnet")):
    V0 = np.full(NS, np.nan); V2 = np.full(NS, np.nan)
    for j in np.flatnonzero(repS & (zS == 0) & youS & pyS):
        i = int(idS[j]); k0 = f"{tag}v0_{i}"
        if k0 not in ANS:
            continue
        f0 = flg(ANS[k0]); f1 = flg(ANS[f"{tag}v1_{i}"]) if f"{tag}v1_{i}" in ANS else f0; f2 = flg(ANS[f"{tag}v2_{i}"]) if f"{tag}v2_{i}" in ANS else f1
        V0[j], V2[j] = f0, f2
    for dd, dn in ((1, "One"), (5, "Five")):
        m_ = (d5 == dd) & np.isfinite(V0)
        both[(nm, dn)] = (float(V0[m_].mean()), float(V2[m_].mean()), int(m_.sum()))
        put(f"xaBoth{nm}{dn}Orig", pct(both[(nm, dn)][0]), "recomputed: results/x89/rewrite/out_c + data/second/s5 texts", f"{tag}v0_<id>, depth {'5 or more' if dd == 5 else dd}",
            "FPR among gold-negative replies whose text and parent both address 'you', labeled with the parent, as written", caveat="5% sample; recomputed descriptive (Addendum 89 reading)")
        put(f"xa{'Both'}{nm}{dn}Both", pct(both[(nm, dn)][1]), "recomputed: results/x89/rewrite/out_c", f"{tag}v2_<id> (else v1, else v0)", "after rewriting reply and parent",
            caveat="5% sample; recomputed descriptive")
        put(f"xaBoth{nm}{dn}N", th(both[(nm, dn)][2]), "recomputed", "count", "number of such replies at that depth", caveat="5% sample")
logchk("xaBothHaikuOneOrig", "the registration log (Addendum 89 results) reports 36% at depth 1"); logchk("xaBothHaikuFiveOrig", "the log reports 50% at depth 5+")
logchk("xaBothHaikuOneBoth", "the log reports 30%"); logchk("xaBothHaikuFiveBoth", "the log reports 41%")
assert (M["xaBothHaikuOneOrig"], M["xaBothHaikuFiveOrig"], M["xaBothHaikuOneBoth"], M["xaBothHaikuFiveBoth"]) == ("36\\%", "50\\%", "30\\%", "41\\%"), \
    (M["xaBothHaikuOneOrig"], M["xaBothHaikuFiveOrig"], M["xaBothHaikuOneBoth"], M["xaBothHaikuFiveBoth"])

item("E19")
RB = s5d["rewrite_by_count"]; assert all(RB[t_][k_]["change"][2] < 0 for t_ in RB for k_ in RB[t_])
for t_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    for k_, w_ in (("1", "One"), ("2", "Two"), ("3", "Three"), ("4", "Four")):
        trip(f"xaRwDose{nm}{w_}", RB[t_][k_]["change"], SDO, f"cc_s5.rewrite_by_count.{t_}.{k_}.change", "x100: change in the flag rate of gold negatives with k second-person words (points)",
             d=1, sc=100, caveat="5% sample")
        if nm == "Haiku":
            put(f"xaRwDoseN{w_}", th(RB[t_][k_]["n"]), SDO, f"cc_s5.rewrite_by_count.haiku.{k_}.n", "gold negatives with k words", caveat="5% sample")
for a_, b_ in (("xaRwDoseHaikuOne", "cRwDoseHOne"), ("xaRwDoseHaikuThree", "cRwDoseHThree"), ("xaRwDoseSonnetOne", "cRwDoseSOne"), ("xaRwDoseSonnetFour", "cRwDoseSFour")):
    assert f"{abs(un(M[a_])):.0f}" == EX[b_]; logchk(a_, f"rounds to -\\{b_} ({EX[b_]}, unsigned, no decimals in the earlier draft)")

# ---------------------------------------------------------------- tables/xa_rewrite.tex (E12-E19)
item("E13")
arr = " $\\to$ "
def cic(b):                                                                        # "value [lo, hi]" from three macros
    return f"{M[b]} [{M[b + 'Lo']}, {M[b + 'Hi']}]"
rows = ["\\multicolumn{3}{@{}l}{\\emph{Every second-person comment of the 5\\% sample rewritten; distortion as written $\\to$ rewritten (change [CI]; share removed)}} \\\\"]
for lab, h_, s_ in (("Depth law", ("xaYfOrig", "xaYfEdit", "xaYfChg", "xaYfShare"), ("xaRwSonOrig", "xaRwSonEdit", "xaRwSonChg", "xaRwSonShare")),
                    ("Within threads", ("xaYfWithinOrig", "xaYfWithinEdit", "xaYfWithinChg", "xaYfWithinShare"), ("xaRwSonWithinOrig", "xaRwSonWithinEdit", "xaRwSonWithinChg", "xaRwSonWithinShare")),
                    ("Thread size law", ("xaYfSizeOrig", "xaYfSizeEdit", "xaYfSizeChg", "xaYfSizeShare"), ("xaRwSonSizeOrig", "xaRwSonSizeEdit", "xaRwSonSizeChg", "xaRwSonSizeShare"))):
    rows.append(" & ".join([lab] + [f"{M[a]}{arr}{M[b]} ({cic(c)}; {M[d]})" for a, b, c, d in (h_, s_)]) + " \\\\")
rows.append(" & ".join(["Gold positives flagged", f"{M['xaYfPosOrig']}{arr}{M['xaYfPosEdit']}", f"{M['xaRwSonPosOrig']}{arr}{M['xaRwSonPosEdit']}"]) + " \\\\")
rows.append(" & ".join(["Gold negatives flagged", f"{M['xaYfNegOrig']}{arr}{M['xaYfNegEdit']}", f"{M['xaRwSonNegOrig']}{arr}{M['xaRwSonNegEdit']}"]) + " \\\\")
rows.append("\\midrule")
rows.append("\\multicolumn{3}{@{}l}{\\emph{Edit sample; flag rate as written $\\to$ edited (change, points [CI])}} \\\\")
for lab, st in (("Rule rewrite, replies at depth $\\ge 2$", "Deep"), ("\\quad roots", "Root"), ("\\quad gold positives", "Pos")):
    rows.append(" & ".join([lab] + [f"{M[f'xa{m}Edit{st}Before']}{arr}{M[f'xa{m}Edit{st}After']} ({cic(f'xa{m}Edit{st}Diff')})" for m in ("Haiku", "Sonnet")]) + " \\\\")
for lab, st in (("Paraphrase with ``you'' $\\to$ in the third person", "Deep"), ("\\quad gold positives", "Pos")):
    rows.append(" & ".join([lab] + [f"{M[f'xa{m}Para{st}You']}{arr}{M[f'xa{m}Para{st}They']} ({cic(f'xa{m}Para{st}Diff')})" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["Depth gap (depth $\\ge 2$ minus roots), rule rewrite"] + [f"{cic(f'xa{m}GapOrig')}{arr}{cic(f'xa{m}GapRule')}" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["\\quad change of the gap"] + [cic(f"xa{m}GapCloseRule") for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["Depth gap, paraphrase with ``you'' $\\to$ third person"] + [f"{cic(f'xa{m}GapParaYou')}{arr}{cic(f'xa{m}GapParaThey')}" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["\\quad change of the gap"] + [cic(f"xa{m}GapClosePara") for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append("\\midrule")
rows.append("\\multicolumn{3}{@{}l}{\\emph{Reverse rewrite of clearly non-toxic comments without ``you'': third person $\\to$ ``you''}} \\\\")
for lab, st in (("Replies at depth $\\ge 2$", "Deep"), ("\\quad roots", "Root"), ("\\quad gold positives", "Pos")):
    rows.append(" & ".join([lab] + [f"{M[f'xa{m}Rev{st}Before']}{arr}{M[f'xa{m}Rev{st}After']} ({cic(f'xa{m}Rev{st}Diff')})" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["Depth gap (depth $\\ge 2$ minus roots)"] + [f"{M[f'xa{m}RevGapOrig']}{arr}{M[f'xa{m}RevGapYou']} ({cic(f'xa{m}RevGapChange')})" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append("\\midrule")
rows.append("\\multicolumn{3}{@{}l}{\\emph{Rewrite dose: change in the flag rate of gold negatives (points [CI]), by second-person words rewritten}} \\\\")
for w_, lab in (("One", "1"), ("Two", "2"), ("Three", "3"), ("Four", "4 or more")):
    rows.append(" & ".join([f"{lab} (n = {M[f'xaRwDoseN{w_}']})"] + [cic(f"xaRwDose{m}{w_}") for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append("\\midrule")
rows.append("\\multicolumn{3}{@{}l}{\\emph{Shown the parent, among replies: distortion as written $\\to$ reply rewritten $\\to$ reply and parent rewritten}} \\\\")
rows.append(" & ".join(["Depth law among replies"] + [f"{M[f'xaRwCtx{m}Orig']}{arr}{M[f'xaRwCtx{m}Reply']}{arr}{M[f'xaRwCtx{m}Both']}" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["\\quad change, both rewritten (share removed)"] + [f"{cic(f'xaRwCtx{m}ChgBoth')} ({M[f'xaRwCtx{m}ShareBoth']})" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["Reply and parent address ``you'': FPR at depth 1 $\\to$ 5+, as written"] + [f"{M[f'xaBoth{m}OneOrig']}{arr}{M[f'xaBoth{m}FiveOrig']}" for m in ("Haiku", "Sonnet")]) + " \\\\")
rows.append(" & ".join(["\\quad both rewritten"] + [f"{M[f'xaBoth{m}OneBoth']}{arr}{M[f'xaBoth{m}FiveBoth']}" for m in ("Haiku", "Sonnet")]) + " \\\\")
table("xa_rewrite", rows,
      "Rewriting ``you'' with a fixed rule (and, for the paraphrase control, with Sonnet's paraphrases) changes what the LLMs flag; each edit is labeled in the same "
      "batch as its original. Distortions are the labeler's law minus gold's, in points per level (per doubling for the thread size law); flag rates in percent, "
      f"changes in points with paired 95% intervals. Top: all {M['xaYfNYou']} second-person comments of the 5% sample rewritten, the other comments keeping their labels. "
      f"Edit sample: {th(I76['haiku']['h11/neg_deep']['n'])} clearly non-toxic second-person replies at depth 2 or more (fewer than 20% of raters voting toxic), "
      f"{th(I76['haiku']['h11/neg_root']['n'])} roots and {th(I76['haiku']['h11/pos_any']['n'])} gold positives, "
      f"paired; reverse rewrite: {th(I76['haiku']['rev/neg_deep']['n'])} such replies without ``you'', {th(I76['haiku']['rev/neg_root']['n'])} roots and {th(I76['haiku']['rev/pos_any']['n'])} gold positives. "
      "Dose: gold negatives of the 5% sample with that many "
      f"second-person words. Shown the parent: the {M['xaRwCtxHaikuN']} replies of the 5% sample whose text or parent addresses ``you'' are relabeled with the reply rewritten and "
      "with both rewritten; the distortion is measured among replies.",
      "(row label) | Haiku | Sonnet",
      f"{SYF}; {SRW}; recomputed from results/x89/rewrite (Sonnet's paired within-thread and size changes, gold-positive rates, the both-address rows); {S76} (edit sample, "
      f"reverse rewrite); {SEG}; results/x79/x79_songap.json; {SDO} (dose)")


# ================================================================ E21: minimal pairs (tables/xa_pairs.tex)
item("E21")
items_ = {}; lab_ = {}
for l in open(f"{R}/x74_edit/label_items/shard_000.jsonl"):
    r_ = json.loads(l); items_[int(r_["id"])] = r_["text"]
for l in open(f"{R}/x74_edit/haiku_tox/shard_000.jsonl"):
    r_ = json.loads(l); lab_[int(r_["id"])] = r_.get("ans")
S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id")
pairs = [c for c in S11.index.astype(int) if lab_.get(c * 10) in ("A", "B") and lab_.get(c * 10 + 1) == "D" and S11.loc[c, "tox"] == 0]   # as x18_tables_v5.py
short = [c for c in pairs if 30 < len(items_[c * 10]) < 170]
put("xaExPairs", str(len(pairs)), "results/x74_edit/{label_items,haiku_tox}/shard_000.jsonl; data/you_edit_meta.parquet", "pairs flagged as written, 'not toxic' after the rewrite, no toxic vote",
    "count (as x18_tables_v5.py)"); same("xaExPairs", "cExPairs")
put("xaExShort", str(len(short)), "same", "30 < length < 170 characters", "count"); same("xaExShort", "cExShort")
put("xaExItems", th(len(S11)), "data/you_edit_meta.parquet", "rows", "comments in the edit sample"); assert str(len(S11)) == EX["cExItems"]; logchk("xaExItems", "= \\cExItems (4000, without the thousands brace)")
assert F81A["n"] == len(S11)                                                         # f81 labeled the same edit sample
# pairs that name a person (public or private), address one by name or initials, or carry a user handle: read and listed here; never printed
NAMED = {255935: "a first name", 7043721: "initials", 7088436: "a user handle", 5738351: "a surname", 6272211: "a private person's full name (printed redacted in the earlier-draft table)",
         321135: "a surname", 5986762: "a full name", 630089: "a full name", 5734293: "a public figure's name", 862628: "a nickname", 7078233: "a user name and a surname",
         415956: "public figures' names"}   # read: all pairs up to 236 characters
def tex_(t):
    t = re.sub(r"https?://\S+", "[link]", t).replace("\n", " ")
    for a_, b_ in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        t = t.replace(a_, b_)
    t = " ".join(t.split())
    t = re.sub(r'(^|[\s(\[])"', r"\1``", t); t = t.replace('"', "''")                 # straight double quotes as TeX quotes
    return t
elig = [c for c in sorted(pairs, key=lambda c: (len(items_[c * 10]), c)) if c not in NAMED]
PICK = elig[:12]
LMAX = 236                                                                           # every pair up to this length was read for names (NAMED)
assert max(len(items_[c * 10]) for c in PICK) <= LMAX
for c in PICK:
    for t in (items_[c * 10], items_[c * 10 + 1]):
        assert not re.search(r"@\w|\b[\w.+-]+@[\w-]+\.\w+|www\.", t), (c, t)               # no handle, e-mail or bare web address
ALLOW = {"You're", "They're", "Dude,", "Kinda", "Clearly,", "Privilege", "Because", "Russian", "I", "The", "Conflict", "Interest", "Act", "Act?", "Do", "Enquirer,",
         "It", "If", "You", "They", "Me", "Windmills", "Anyone", "A", "Please", "[link]", "Haha", "LOL", "Act.", "Interest", "Of"}
for c in PICK:
    for t in (items_[c * 10], items_[c * 10 + 1]):
        caps = [w for w in re.findall(r"\S+", re.sub(r"https?://\S+", "[link]", t)) if w[:1].isupper() or w.strip('"').strip("(")[:1].isupper()]
        bad = [w for w in caps if w.strip('"').strip("(").strip(".,?!") not in {a.strip(".,?!") for a in ALLOW}]
        assert not bad, (c, bad)                                                    # every capitalized word was read and is not a name
rows = [f"{tex_(items_[c * 10])} & {tex_(items_[c * 10 + 1])} \\\\" for c in PICK]
_draft_red = [c for c in pairs if items_[c * 10].startswith("IRT ") and "Star-Advertiser" in items_[c * 10]]   # the pair the earlier-draft table prints with [name]
assert len(_draft_red) == 1 and _draft_red[0] in NAMED and _draft_red[0] not in PICK
put("xaExShown", str(len(PICK)), "selection rule", "the shortest eligible pairs", "number of pairs shown")
table("xa_pairs", rows,
      "Minimal pairs from the edit sample: of the comments that no rater called toxic, \\xaExPairs{} are toxic to Haiku as written and ``not toxic'' after the "
      "fixed rewrite; the \\xaExShown{} shortest that contain no name, initials or user handle, links shown as [link].",
      "As written (flagged) | Rewritten (``not toxic'')",
      "results/x74_edit/label_items/shard_000.jsonl (texts), results/x74_edit/haiku_tox/shard_000.jsonl (Haiku's answers), data/you_edit_meta.parquet (strata, vote fraction)")
table("xa_pairs_three", rows[:3], "The three shortest pairs of xa_pairs (main text, Section 4.2).", "As written (flagged) | Rewritten (``not toxic'')",
      "the first three rows of tables/xa_pairs.tex")


# ================================================================ E22-E26: LLMs and operating points; the settings table
item("E22")
NM7 = {"haiku": "Haiku", "sonnet": "Sonnet", "qwen": "Qwen", "gemma": "Gemma", "olmo2": "Olmo", "granite": "Granite", "llama": "Llama"}
gp = {m_: ROWS92[(m_, "rating at gold prevalence")] for m_ in ("haiku", "sonnet")} | {m_: ROWS92[(m_, "gold prevalence")] for m_ in ("qwen", "gemma", "olmo2", "granite", "llama")}
for m_, r_ in gp.items():
    nm = NM7[m_] + ("Hundred" if m_ in ("haiku", "sonnet") else "")
    trip(f"xa{nm}PrevDist", r_["depth_excess"], S92, f"rows[{m_}, {'rating at gold prevalence' if m_ in ('haiku', 'sonnet') else 'gold prevalence'}].depth_excess",
         "depth law minus gold's at gold's flag rate (points per level)", caveat="5% sample")
    trip(f"xa{nm}PrevFix", r_["depth_excess_fixed"], S92, "...depth_excess_fixed", "at a fixed number of second-person words", caveat="5% sample")
    trip(f"xa{nm}PrevLRR", r_["log_rr_excess"], S92, "...log_rr_excess", "log reply RR minus gold's", caveat="5% sample")
    put(f"xa{nm}PrevKappa", f(r_["kappa"]), S92, "...kappa", "Cohen's kappa", caveat="5% sample")
same("xaHaikuHundredPrevDist", "cMatHaiku"); same("xaSonnetHundredPrevDist", "cMatSonnet"); same("xaGemmaPrevDist", "cMatGemma"); same("xaSonnetHundredPrevFix", "cMatSonnetFix")
same3("xaSonnetHundredPrevLRR", "cMatSonnetRR"); same("xaHaikuHundredPrevKappa", "dKapHaikuPrev"); same("xaGemmaPrevKappa", "dKapGemmaPrev")
op4 = [gp[m_]["depth_excess"][0] for m_ in ("qwen", "olmo2", "granite", "llama")]
put("xaOpenPrevMin", sg(min(op4)), S92, "rows[qwen, olmo2, granite, llama; gold prevalence].depth_excess", "smallest of the four open models whose CI is above 0", caveat="5% sample")
put("xaOpenPrevMax", sg(max(op4)), S92, "same", "largest", caveat="5% sample")
assert all(gp[m_]["depth_excess"][1] > 0 for m_ in ("qwen", "olmo2", "granite", "llama")); same("xaOpenPrevMin", "cMatOpenMin"); same("xaOpenPrevMax", "cMatOpenMax")
for m_ in ("qwen", "gemma", "olmo2", "granite", "llama"):
    r_ = ROWS92[(m_, "max kappa")]; nm = NM7[m_]
    trip(f"xa{nm}KmaxDist", r_["depth_excess"], S92, f"rows[{m_}, max kappa].depth_excess", "depth law minus gold's at the kappa-maximizing cut over 400 score quantiles, all gold labels", caveat="5% sample")
    put(f"xa{nm}KmaxFlag", pct(r_["flag_rate"], 1), S92, f"rows[{m_}, max kappa].flag_rate", "share flagged at that cut", caveat="5% sample")
    put(f"xa{nm}KmaxKappa", f(r_["kappa"]), S92, f"rows[{m_}, max kappa].kappa", "kappa at that cut", caveat="5% sample")
km = [ROWS92[(m_, "max kappa")]["depth_excess"][0] for m_ in ("qwen", "gemma", "olmo2", "granite", "llama")]
put("xaOpenKmaxMin", sg(min(km)), S92, "rows[open models, max kappa].depth_excess", "smallest", caveat="5% sample"); put("xaOpenKmaxMax", sg(max(km)), S92, "same", "largest", caveat="5% sample")
same("xaOpenKmaxMin", "cMatOpenKmaxMin"); same("xaOpenKmaxMax", "cMatOpenKmaxMax")
sel7 = [("haiku", "rating at gold prevalence"), ("sonnet", "rating at gold prevalence")] + [(m_, "gold prevalence") for m_ in ("qwen", "gemma", "olmo2", "granite", "llama")]
rho7 = spearmanr([ROWS92[s_]["kappa"] for s_ in sel7], [ROWS92[s_]["depth_excess"][0] for s_ in sel7]).correlation
put("xaLlmKapRho", sg(rho7), S92, "rows[7 LLMs at gold's flag rate].kappa, depth_excess", "Spearman correlation of kappa with the depth-law distortion", caveat="5% sample")
same("xaLlmKapRho", "dLlmKapRho")

# ---- replicated flags for the settings table and the among-replies rows
def rating_scores(m_):
    d_ = {}
    for l in open(f"{R}/x75_rating/{m_}_ctx0.jsonl"):
        r_ = json.loads(l); v_ = r_.get("rating"); d_[int(r_["id"])] = float(v_) if v_ is not None else np.nan
    return np.array([d_.get(int(i), np.nan) for i in idS])
kS = int(round(zS.mean() * NS)); kT = int(round(zT.mean() * NT))
F100 = {m_: flags_at_prev(rating_scores(m_), NS, kS, 9010) for m_ in ("haiku", "sonnet")}            # as x92_matched.py at_prev(rating, 9010)
for m_ in ("haiku", "sonnet"):
    c_ = conf(F100[m_], zS); r_ = ROWS92[(m_, "rating at gold prevalence")]
    assert abs(c_["kappa"] - r_["kappa"]) < 1e-12 and abs(c_["fpr"] - r_["fpr"]) < 1e-12 and abs(c_["flag"] - r_["flag_rate"]) < 1e-12
def b84_scores(model, ctx):
    s_ = {}
    for fn in sorted(glob.glob(f"{R}/b84/{model}/shard_*_tox_ctx{ctx}.jsonl")):
        for l in open(fn):
            r_ = json.loads(l); s_[int(r_["id"])] = float(r_["p"][0] + r_["p"][1])
    return s_
FAM = load("b84/x84_families.json"); SFAM = "results/b84/x84_families.json"; FOPEN = {}
for model in ("olmo2", "granite", "llama31"):
    s0, s1 = b84_scores(model, 0), b84_scores(model, 1)
    a0 = np.array([s0.get(int(i), np.nan) for i in idS]); a1 = np.array([s1.get(int(i), np.nan) if prS[j] >= 0 and int(i) in s1 else s0.get(int(i), np.nan) for j, i in enumerate(idS)])
    for tag, sc_ in ((model, a0), (f"{model}_ctx", a1)):
        y_ = flags_at_prev(sc_, NS, kS, 8400); FOPEN[tag] = y_; c_ = conf(y_, zS); st_ = FAM["settings"][tag]
        assert all(abs(c_[a_] - st_[b_][0]) < 1e-12 for a_, b_ in (("flag", "flag_rate"), ("kappa", "kappa"), ("tpr", "tpr"), ("fpr", "fpr"))), tag   # x84's flags reproduced
mname = {"llama31": "llama"}
for model in ("olmo2", "granite", "llama31"):
    assert abs(conf(FOPEN[model], zS)["kappa"] - ROWS92[(mname.get(model, model), "gold prevalence")]["kappa"]) < 1e-12      # x92 uses the same flags
def own_ctx_flags(P_, z_):                                                         # as x76_rethreshold.py / x97_decomp_replies.py (seed 7650, columns in order)
    out = {}; rng = np.random.default_rng(7650); N_ = len(P_); k_ = int(round(z_.mean() * N_))
    for m_ in [c[2:] for c in P_.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
        out[m_] = flags_at_prev(P_[f"s:{m_}"].values.astype(float), N_, k_, None, rng=rng)
    return out
OWN_T = own_ctx_flags(PT, zT); OWN_S = own_ctx_flags(PS, zS)
for P_, z_, rep_, pop, k_, y_ in ((PT, zT, repT, "cc_threads", "qwen7b_m_ctx@own", OWN_T["qwen7b_m_ctx"]), (PS, zS, repS, "cc_s5", "gemma_m_ctx@own", OWN_S["gemma_m_ctx"])):
    c_ = conf(y_, z_, rep_); q_ = RP[pop]["labelers"][k_]
    assert abs(c_["tpr"] - q_["tpr"]) < 1e-12 and abs(c_["fpr"] - q_["fpr"]) < 1e-12, k_           # x97's flags reproduced
    RT_ = load("x76_rethreshold/x76_rethreshold.json")[pop]["labelers"][k_]; ca_ = conf(y_, z_)
    assert abs(ca_["kappa"] - RT_["kappa"][0]) < 1e-12 and abs(ca_["flag"] - RT_["flag_rate"][0]) < 1e-12

item("E23")
RA = load("x75_rating/x75_rating_eval.json"); SRA = "results/x75_rating/x75_rating_eval.json"; GRID = ["0.02", "0.05", "0.075", "0.1", "0.15", "0.2", "0.3"]
GRIDNM = {"0.02": "Two", "0.05": "Five", "0.075": "SevenHalf", "0.1": "Ten", "0.15": "Fifteen", "0.2": "Twenty", "0.3": "Thirty"}
gv = []
for m_, nm in (("haiku_r", "Haiku"), ("sonnet_r", "Sonnet")):
    for t_ in GRID:
        r_ = RA["instruments"][m_][t_]; gv.append(r_["depth_minus_gold"])
        trip(f"xaGrid{nm}{GRIDNM[t_]}", r_["depth_minus_gold"], SRA, f"instruments.{m_}.{t_}.depth_minus_gold", "x100: 0-100 rating's depth law minus gold's at that flag rate", sc=100,
             caveat="5% sample")
put("xaGridMin", sg(100 * min(v[0] for v in gv)), SRA, "instruments.{haiku_r,sonnet_r}.<7 rates>.depth_minus_gold", "smallest of the 14 grid distortions (points per level)", caveat="5% sample")
put("xaGridMax", sg(100 * max(v[0] for v in gv)), SRA, "same", "largest", caveat="5% sample")
put("xaGridNPos", str(sum(v[1] > 0 for v in gv)), SRA, "same", "grid points whose CI is above 0", caveat="5% sample"); put("xaGridN", str(len(gv)), SRA, "same", "grid points")
for a_, b_ in (("xaGridMin", "qRateGridMin"), ("xaGridMax", "qRateGridMax"), ("xaGridNPos", "qRateNGridPos"), ("xaGridN", "qRateNGrid")):
    if M[a_] == EX[b_]:
        logchk(a_, f"= \\{b_} although the set differs")
    else:
        differs(a_, b_, f"\\{b_} also covers Haiku and Sonnet rating with the parent, which x75_rating_eval.py flags over all comments with roots keeping their "
                        "no-context rating (pooled); those 14 points are left out here")
for m_, nm in (("haiku_r", "Haiku"), ("sonnet_r", "Sonnet")):
    r_ = RA["instruments"][m_]["matched"]
    put(f"xaRate{nm}Law", sg(100 * r_["depth"][0]), SRA, f"instruments.{m_}.matched.depth", "x100: depth law at gold's flag rate (ties broken with seed 7500)", caveat="5% sample")
    _r92 = ROWS92[({"haiku_r": "haiku", "sonnet_r": "sonnet"}[m_], "rating at gold prevalence")]["depth_excess"][0]   # x92 stores points per level
    trip(f"xaRate{nm}Dist", r_["depth_minus_gold"], SRA, f"instruments.{m_}.matched.depth_minus_gold", "x100", sc=100,
         caveat=f"5% sample; ties broken with seed 7500 (x92 uses seed 9010: {sg(_r92)})")
same("xaRateHaikuLaw", "qRateHaikuDepthPP"); same("xaRateSonnetLaw", "qRateSonnetDepthPP")
same("xaRateHaikuDistLo", "qRateHaikuDepthLo"); same("xaRateSonnetDistLo", "qRateSonnetDepthLo")
logchk("xaRateHaikuDistLo", "note: the earlier-draft pair (qRateHaikuDepthPP, qRateHaikuDepthLo) puts the law (+0.25) next to the lower CI end of the distortion (+0.23)")
hc = RA["H75c_detail"]["diff"]
trip("xaRatePlacOverSr", hc, SRA, "H75c_detail.diff", "x100: Haiku-100's relative placebo depth slope minus the single rater's (points of the FPR per level), both against draw 0's leave-one-out gold",
     d=1, sc=100, caveat="5% sample")
note("Figure: the 0-100 ratings with the parent (haiku_r_ctx, sonnet_r_ctx) are pooled in x75_rating_eval.json (flags drawn over all comments, roots keep "
     "their no-context rating), so the figure shows Haiku-100 and Sonnet-100 without the parent only.")

item("E24")
for n_, w_ in ((200, "Two"), (500, "Five"), (1000, "Thousand")):
    put(f"xaCalNative{w_}", pct(HL[f"kappa@{n_}"]["cut_counts"].get("2", 0) / 500, 1 if n_ == 1000 else 0), SC_, f"pops.cc_threads.instruments.haiku:level.calibrated.kappa@{n_}.cut_counts['2']",
        "share of 500 draws in which maximizing kappa on n gold labels picks Haiku's own cut")
same("xaCalNativeFive", "cCalNativeFive"); same("xaCalNativeThousand", "cCalNativeThousand"); same("xaCalNativeTwo", "dCalNativeTwo")
put("xaCalFOne", pct(HL["f1@500"]["cut_counts"].get("2", 0) / 500), SC_, "...haiku:level.calibrated.f1@500.cut_counts['2']", "share for the F1-maximizing cut"); same("xaCalFOne", "cCalFOne")
assert all(HL[f"{r_}@{n_}"]["share_excess_sig_pos"] == 1.0 for r_ in ("kappa", "f1", "prev") for n_ in (200, 500, 1000))
put("xaCalSigAll", "100\\%", SC_, "...haiku:level.calibrated.*.share_excess_sig_pos", "share of draws with a significant excess, under every rule and size (asserted 1.0)")
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    c_ = CA["cc_s5"]["instruments"][f"{m_}:rating"]["calibrated"]
    for r_, rn in (("kappa", "Kmax"), ("prev", "Prev")):
        e_ = c_[f"{r_}@500"]
        put(f"xaCal{nm}{rn}Ex", sg(e_["excess_median"]), SC_, f"pops.cc_s5.instruments.{m_}:rating.calibrated.{r_}@500.excess_median",
            "median over 500 draws of 500 gold labels of the depth-law excess at the chosen 0-100 cut", caveat="5% sample")
        put(f"xa{'Cal'}{nm}{rn}ExLo", sg(e_["excess_p05"]), SC_, "...excess_p05", "5th percentile over draws (not a confidence interval)", caveat="5% sample")
        put(f"xaCal{nm}{rn}ExHi", sg(e_["excess_p95"]), SC_, "...excess_p95", "95th percentile over draws", caveat="5% sample")
        put(f"xaCal{nm}{rn}Cut", f(e_["cut_median"], 1 if e_["cut_median"] % 1 else 0), SC_, "...cut_median", "median cut over draws", caveat="5% sample")
        put(f"xaCal{nm}{rn}Flag", pct(e_["flag_rate_median"], 1), SC_, "...flag_rate_median", "median flag rate", caveat="5% sample")
        put(f"xaCal{nm}{rn}Kappa", f(e_["kappa_median"]), SC_, "...kappa_median", "median kappa", caveat="5% sample")
        put(f"xaCal{nm}{rn}Sig", pct(e_["share_excess_sig_pos"]), SC_, "...share_excess_sig_pos", "share of draws with a significant positive excess", caveat="5% sample")
same("xaCalHaikuKmaxEx", "dHaikuKmax"); same("xaCalSonnetKmaxEx", "dSonnetKmax"); same("xaCalHaikuKmaxEx", "cRateHaikuEx")
DEC88 = CA["cc_threads"]["decisions"]
for n_, w_ in (("10000", "Ten"), ("20000", "Twenty")):
    put(f"xaCalDecHaiku{w_}", pct(DEC88[n_]["haiku:level"]["differ"]), SC_, f"pops.cc_threads.decisions.{n_}.haiku:level.differ",
        "share of 200 article samples of that size whose depth-law verdict at the modal kappa-maximizing cut (Haiku's own cut) differs from gold's")
    put(f"xaCalDecSr{w_}", pct(DEC88[n_]["single_rater"]["differ"]), SC_, f"pops.cc_threads.decisions.{n_}.single_rater.differ", "same for the single rater")
note("The decision shares at 10k and 20k comments (x88_calibrated) overlap E60 (x86_decisions, agent C), which reports 10k-100k; both use Haiku's own cut.")

item("E25")
RR8 = load("x88/x88_rerun.json"); SRR = "results/x88/x88_rerun.json"
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    r_ = RR8[m_]
    put(f"xaRerun{nm}FlagChg", pct(r_["flag_changed"], 1), SRR, f"{m_}.flag_changed", "share of comments whose flag changes between runs", caveat="5% sample")
    put(f"xaRerun{nm}AnsChg", pct(r_["answer_changed"], 1), SRR, f"{m_}.answer_changed", "share whose four-level answer changes", caveat="5% sample")
    put(f"xaRerun{nm}Kappa", f(r_["kappa_between_runs"]), SRR, f"{m_}.kappa_between_runs", "kappa between the two runs", caveat="5% sample")
    trip(f"xaRerun{nm}Diff", r_["depth_new_minus_old"], SRR, f"{m_}.depth_new_minus_old", "paired change of the depth law (points per level)", caveat="5% sample")
    trip(f"xaRerun{nm}Dist", r_["distortion_new"], SRR, f"{m_}.distortion_new", "distortion of the later batch (points per level)", caveat="5% sample")
    put(f"xaRerun{nm}Flag", pct(r_["flag_rate_new"], 1), SRR, f"{m_}.flag_rate_new", "flag rate of the later batch", caveat="5% sample")
same("xaRerunHaikuFlagChg", "cRerunFlagChg"); same("xaRerunHaikuKappa", "cRerunKappaH"); same("xaRerunSonnetKappa", "cRerunKappaS"); same3("xaRerunHaikuDiff", "cRerunDiff")
same("xaRerunHaikuDist", "dHaikuRerun"); same("xaRerunSonnetDist", "dSonnetRerun")
LET = {3.0: "A", 2.0: "B", 1.0: "C", 0.0: "D"}; RRK = {}
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):                          # kappa of the later batch against gold (stored answers, as x88_rerun.py reads them)
    new_ = {}
    for h_ in (0, 1):
        for l in open(f"{R}/x81/{m_}_{h_}.jsonl"):
            r_ = json.loads(l)
            if r_["task"] == "tox":
                new_[int(r_["id"])] = r_["ans"]
    a_old = np.array([LET.get(v) for v in PS[f"s:{m_}"].values.astype(float)], dtype=object); a_new = np.array([new_.get(int(i)) for i in idS], dtype=object)
    ok_ = np.array([(u is not None) and (w is not None) for u, w in zip(a_old, a_new)]); f_new = np.isin(a_new, ["A", "B"]).astype(float)
    assert int(ok_.sum()) == RR8[m_]["n"] and abs(f_new[ok_].mean() - RR8[m_]["flag_rate_new"]) < 1e-12
    RRK[m_] = conf(f_new, zS, ok_)
    put(f"xaRerun{nm}KappaGold", f(RRK[m_]["kappa"]), "recomputed: results/x81/{m}_{0,1}.jsonl (task tox)", "later batch's flags against gold", "Cohen's kappa over the comments with both answers",
        caveat="5% sample; recomputed")

item("E26")
for model, nm in (("olmo2", "Olmo"), ("granite", "Granite"), ("llama31", "Llama")):
    trip(f"xa{nm}RepliesDist", FAM["settings"][model]["depth_replies_excess"], SFAM, f"settings.{model}.depth_replies_excess", "depth law among replies minus gold's, without the parent",
         caveat="5% sample")
    trip(f"xa{nm}CtxRepliesDist", FAM["settings"][f"{model}_ctx"]["depth_replies_excess"], SFAM, f"settings.{model}_ctx.depth_replies_excess",
         "among replies, shown the parent (own gold-prevalence threshold)", caveat="5% sample")
    trip(f"xa{nm}WithinDist", FAM["settings"][model]["within_distortion"], SFAM, f"settings.{model}.within_distortion", "within-thread distortion without the parent", caveat="5% sample")
same3("xaLlamaCtxRepliesDist", "cLlamaCtxEx")
oc = [FAM["settings"][f"{m_}_ctx"]["depth_replies_excess"][0] for m_ in ("olmo2", "granite", "llama31")]
put("xaOpenCtxRepliesMax", sg(max(oc)), SFAM, "settings.*_ctx.depth_replies_excess", "largest of the three", caveat="5% sample"); same("xaOpenCtxRepliesMax", "cOpenCtxExMax")
for pop, k_, nm in (("cc_threads", "qwen7b_m_ctx@own", "QwenCtx"), ("cc_s5", "gemma_m_ctx@own", "GemmaCtx")):
    trip(f"xa{nm}RepliesDist", RP[pop]["labelers"][k_]["distortion"], "results/x97/x97_decomp_replies.json", f"{pop}.labelers.{k_}.distortion",
         "depth law among replies minus gold's, shown the parent, own gold-prevalence threshold", caveat="5% sample" if pop == "cc_s5" else "")
same("xaQwenCtxRepliesDist", "dQwenCtx")
put("xaQwenRepliesDist", EX["dQwenRepl"], "paper_v5/numbers_v6_dist.tex (x96_dist_v6.py)", "dQwenRepl", "copied: Qwen without the parent among replies (recomputed below)")
yq = np.nan_to_num(PT["y:qwen7b_m"].values.astype(float)); flq = flags_at_prev(PT["s:qwen7b_m"].values.astype(float), NT, kT, 7650)     # as x96_dist_v6.py
xT = np.minimum(PT.depth.values, 10).astype(float); xr_ = xT[repT] - xT[repT].mean()
slr = lambda v: 100 * float((xr_ * (v[repT] - v[repT].mean())).sum() / (xr_ * xr_).sum())
dq_ = slr(flq) - load("x89/x89_cc_reference.json")["cc_threads"]["gold"]["depth_replies"][0]
M["xaQwenRepliesDist"] = sg(dq_); _rec("xaQwenRepliesDist").update(src="recomputed as x96_dist_v6.py (s:qwen7b_m at gold prevalence over all comments, seed 7650)",
    key="slope among replies minus gold's (x89_cc_reference gold.depth_replies)", how="points per level, no CI (x96 stores none)")
same("xaQwenRepliesDist", "dQwenRepl")
note(f"The reply risk ratios of the LLMs shown the parent (x84 *_ctx log_rr; cCtxRROpenMin/Max = {EX['cCtxRROpenMin']}-{EX['cCtxRROpenMax']}; qCtxRRMin/Max = "
     f"{EX['qCtxRRMin']}-{EX['qCtxRRMax']}; v7f's '0.87--1.64') "
     "are pooled over edges whose parent is a root labeled without the parent; no among-replies version is stored, so no contagion number for these settings is given here.")

# ---------------------------------------------------------------- tables/xa_settings.tex: every named setting
item("E22")
def srow(name, cut, sample, n, c):
    return " & ".join([name, cut, sample, th(n), f(100 * c["flag"], 1), f(c["kappa"]), f"{c['tpr']:.3f}", f"{c['fpr']:.3f}"]) + " \\\\"
yT = lambda c_: np.nan_to_num(PT[f"y:{c_}"].values.astype(float)); yS = lambda c_: np.nan_to_num(PS[f"y:{c_}"].values.astype(float))
SET = {}
SET["Haiku"] = conf(yT("haiku"), zT); SET["HaikuCtx"] = conf(yT("haiku_ctx"), zT, repT); SET["Sonnet"] = conf(yS("sonnet"), zS); SET["SonnetCtx"] = conf(yS("sonnet_ctx"), zS, repS)
SET["HaikuHundred"] = conf(F100["haiku"], zS); SET["SonnetHundred"] = conf(F100["sonnet"], zS)
SET["Qwen"] = conf(yT("qwen7b_m"), zT); SET["QwenOwn"] = conf(yT("qwen7b"), zT); SET["QwenCtx"] = conf(OWN_T["qwen7b_m_ctx"], zT, repT); SET["Gemma"] = conf(yS("gemma_m"), zS); SET["GemmaCtx"] = conf(OWN_S["gemma_m_ctx"], zS, repS)
for model, nm in (("olmo2", "Olmo"), ("granite", "Granite"), ("llama31", "Llama")):
    SET[nm] = conf(FOPEN[model], zS); SET[nm + "Ctx"] = conf(FOPEN[f"{model}_ctx"], zS, repS)
for nm, k_, s5 in (("Haiku", "haiku", 0), ("Sonnet", "sonnet", 1), ("Qwen", "qwen7b_m", 0), ("QwenOwn", "qwen7b", 0), ("Gemma", "gemma_m", 1)):           # = Table 1's values (QwenOwn: x74's qwen7b)
    r_ = (PL5 if s5 else PLc)[k_]
    assert all(abs(SET[nm][a_] - r_[b_][0]) < 1e-12 for a_, b_ in (("flag", "flag_rate"), ("kappa", "kappa"), ("tpr", "tpr"), ("fpr", "fpr"))), nm
assert abs(SET["Haiku"]["kappa"] - ROWS92[("haiku", "native")]["kappa"]) > 0 and abs(conf(yS("haiku"), zS)["kappa"] - ROWS92[("haiku", "native")]["kappa"]) < 1e-12
assert abs(SET["HaikuCtx"]["tpr"] - RP["cc_threads"]["labelers"]["haiku_ctx"]["tpr"]) < 1e-12 and abs(SET["SonnetCtx"]["fpr"] - RP["cc_s5"]["labelers"]["sonnet_ctx"]["fpr"]) < 1e-12
for k_ in CLS14:
    r_ = PLc[k_]; SET[CLSNM[k_]] = dict(flag=r_["flag_rate"][0], kappa=r_["kappa"][0], tpr=r_["tpr"][0], fpr=r_["fpr"][0])
    if f"y:{k_}" in PT:
        c_ = conf(yT(k_), zT); assert all(abs(c_[a_] - SET[CLSNM[k_]][a_]) < 1e-12 for a_ in ("flag", "kappa", "tpr", "fpr")), k_
SET["Sr"] = dict(flag=sr_flag, kappa=med(lambda s_: s_["kappa_A_B"]), tpr=sr_tpr, fpr=sr_fpr)
LBL = {"all": "all reply threads", "s5": "5\\% sample", "repT": "replies, all threads", "repS": "replies, 5\\% sample"}
rows = [srow("Haiku", "own cut", LBL["all"], NT, SET["Haiku"]), srow("\\quad + parent", "own cut", LBL["repT"], int(repT.sum()), SET["HaikuCtx"]),
        srow("Sonnet", "own cut", LBL["s5"], NS, SET["Sonnet"]), srow("\\quad + parent", "own cut", LBL["repS"], int(repS.sum()), SET["SonnetCtx"]),
        srow("Haiku-100", "gold's rate", LBL["s5"], NS, SET["HaikuHundred"]), srow("Sonnet-100", "gold's rate", LBL["s5"], NS, SET["SonnetHundred"]), "\\midrule",
        srow("Qwen", "gold's rate", LBL["all"], NT, SET["Qwen"]), srow("", "own cut ($p > 0.5$)", LBL["all"], NT, SET["QwenOwn"]),
        srow("\\quad + parent", "gold's rate, all comments", LBL["repT"], int(repT.sum()), SET["QwenCtx"]),
        srow("Gemma", "gold's rate", LBL["s5"], NS, SET["Gemma"]), srow("\\quad + parent", "gold's rate, all comments", LBL["repS"], int(repS.sum()), SET["GemmaCtx"])]
for nm, disp in (("Olmo", "OLMo-2"), ("Granite", "Granite"), ("Llama", "Llama")):
    rows += [srow(disp, "gold's rate", LBL["s5"], NS, SET[nm]), srow("\\quad + parent", "gold's rate, all comments", LBL["repS"], int(repS.sum()), SET[nm + "Ctx"])]
rows.append("\\midrule")
for k_ in CLS14:
    if not k_.endswith("@m") and k_ != "detoxify_original@0.6":
        rows.append("\\addlinespace[1pt]")
    rows.append(srow(CLSDISP[k_][0] if not k_.endswith("@m") else "", CLSDISP[k_][1], LBL["all"], NT, SET[CLSNM[k_]]))
rows += ["\\midrule", srow("One rater", "--", LBL["all"], NT, SET["Sr"])]
assert rows[0].split(" & ")[4] == "22.3" and rows[2].split(" & ")[4] == "27.1"
SPS = ("x14_pops parquets (y: flags), recomputed; results/x74_placebo/x74_placebo.json (classifiers, identical where both exist); x92_matched.py's rating flags (seed 9010); "
       "x84_families.py's open-model flags (seed 8400); x76_rethreshold.py's flags for Qwen and Gemma shown the parent (seed 7650); x75_onerater.json (one rater)")
table("xa_settings", rows,
      "Labeling settings on Civil Comments: the comments each labels, the share flagged, Cohen's kappa with gold, and the true- and false-positive rates. Haiku and "
      "Sonnet flag what they call toxic (own cut); the 0-100 ratings (Haiku-100, Sonnet-100) and the open models flag as many comments as gold (gold's rate); the "
      "classifiers flag at their usual threshold and at gold's rate. Rows + parent label replies with the replied-to comment shown and are evaluated on replies; "
      "roots keep their label without the parent. *Threshold set so that all comments together, roots with their score without the parent, flag gold's share, "
      f"so replies flag a different share. One rater: one of a comment's raters drawn at random against the majority of the others (median of {len(SPL)} draws; TPR and "
      "FPR against that majority). Section sign: trained at least in part on Civil Comments labels.",
      "Labeler | Cut | Sample | Comments | Flags (%) | kappa | TPR | FPR", SPS)
for nm, c_ in SET.items():
    if nm in ("Haiku", "Sonnet", "Qwen", "Gemma") or nm in CLSNM.values():
        continue                                                                  # their flag rate, kappa, TPR, FPR are Table 1's
    cv = (("among replies" + ("" if nm in ("HaikuCtx", "QwenCtx") else ", 5% sample")) if nm.endswith("Ctx")      # HaikuCtx, QwenCtx: all reply threads (zT)
          else ("5% sample" if nm not in ("Sr", "QwenOwn") else ""))                                              # QwenOwn: all reply threads (conf(yT(...)))
    put(f"xaSet{nm}Flag", pct(c_["flag"], 1), "recomputed (see xa_settings)", nm, "share flagged", caveat=cv)
    put(f"xaSet{nm}Kappa", f(c_["kappa"]), "recomputed", nm, "Cohen's kappa", caveat=cv)
    put(f"xaSet{nm}TPR", f"{c_['tpr']:.3f}", "recomputed", nm, "true-positive rate", caveat=cv); put(f"xaSet{nm}FPR", f"{c_['fpr']:.3f}", "recomputed", nm, "false-positive rate", caveat=cv)
same("xaSetHaikuHundredKappa", "dKapHaikuPrev")
LLMSET = [k_ for k_ in SET if k_ not in CLSNM.values() and k_ != "Sr"]                    # arXiv revision: every LLM row of xa_settings
put("xaSetLlmKappaMin", f(min(SET[k_]["kappa"] for k_ in LLMSET)), "recomputed (see xa_settings)", "min over the LLM rows of xa_settings: " + ", ".join(LLMSET), "smallest kappa of an LLM setting")
put("xaSetLlmKappaMax", f(max(SET[k_]["kappa"] for k_ in LLMSET)), "recomputed (see xa_settings)", "max over the same rows", "largest kappa of an LLM setting")
put("xaSetMhKappaThree", f"{SET[CLSNM['mh_dist@m']]['kappa']:.3f}", "results/x74_placebo/x74_placebo.json", "cc_threads.labelers['mh_dist@m'].kappa", "martin-ha DistilBERT at gold's rate, three decimals")
assert float(M["xaSetLlmKappaMax"]) < float(M["xaSetSrKappa"])
logchk("xaSetSrKappa", "= \\qSrKappa"); assert M["xaSetSrKappa"] == EX["qSrKappa"] and M["xaSetSrFPR"] == EX["qSrFPR"]

# ---------------------------------------------------------------- tables/xa_oppoints.tex: every LLM at its operating points (E22-E24)
item("E22")
MI = load("x91/x91_mitigate.json")["instruments"]; S91 = "results/x91/x91_mitigate.json"
HR = load("x77_humanrate/x77_humanrate.json")["rates"]; SHR = "results/x77_humanrate/x77_humanrate.json"
def r92(m_, op):
    r_ = ROWS92[(m_, op)]
    return [f(100 * r_["flag_rate"], 1), f(r_["kappa"]), ci3(r_["depth_excess"]), None, ci3(r_["depth_excess_fixed"]), ci3(r_["thread_size_excess"]), ci3(r_["log_rr_excess"]), "--"]
def calrow(m_, r_):
    e_ = CA["cc_s5"]["instruments"][f"{m_}:rating"]["calibrated"][f"{r_}@500"]
    return [f(100 * e_["flag_rate_median"], 1), f(e_["kappa_median"]), f"{sg(e_['excess_median'])} [{sg(e_['excess_p05'])}, {sg(e_['excess_p95'])}]$^{{\\ddagger}}$", "--", "--", "--", "--", "--"]
NCALL = 500; NCALD = sum(CA["cc_s5"]["instruments"]["haiku:rating"]["calibrated"][f"kappa@{NCALL}"]["cut_counts"].values())      # from the key and the stored counts
OPR = []
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    r_ = r92(m_, "native"); r_[3] = ci3(WI["cc_s5"]["labelers"][m_]["distortion"], 2, 100)
    r_[7] = ciu(MI["stored"]["ratio_4_0"], 1) if m_ == "haiku" else f(s5d["labelers"]["sonnet"]["fpr_by_count"][4][0] / s5d["labelers"]["sonnet"]["fpr_by_count"][0][0], 1)
    OPR.append([nm, "own cut (max $\\kappa$)"] + r_)
    r_ = r92(m_, "very toxic"); OPR.append(["", "``very toxic'' (prevalence)"] + r_)
    r_ = r92(m_, "rating at gold prevalence"); r_[3] = ci3(HR["matched"][f"{m_}_r"]["depth_within_minus_human"], 2, 100) + "$^{\\P}$"          # arXiv revision: pilcrow, since the section sign marks Civil Comments training elsewhere
    r_[7] = ciu(MTd[m_]["ratio_4_0"], 1); OPR.append([f"{nm}-100", "gold's rate"] + r_)
    OPR.append(["", f"max $\\kappa$ (cut {ROWS92[(m_, 'rating at max kappa')]['cut']:.0f})"] + r92(m_, "rating at max kappa"))
    OPR.append(["", "max $\\kappa$, 500 labels"] + calrow(m_, "kappa")); OPR.append(["", "prevalence, 500 labels"] + calrow(m_, "prev"))
OPR.append("\\midrule")
for m_, disp, wkey in (("qwen", "Qwen", ("x75", "qwen7b_m")), ("gemma", "Gemma", ("x75", "gemma_m")), ("olmo2", "OLMo-2", ("x84", "olmo2")), ("granite", "Granite", ("x84", "granite")),
                       ("llama", "Llama", ("x84", "llama31"))):
    r_ = r92(m_, "gold prevalence")
    r_[3] = ci3(WI["cc_s5"]["labelers"][wkey[1]]["distortion"], 2, 100) if wkey[0] == "x75" else ci3(FAM["settings"][wkey[1]]["within_distortion"])
    r_[7] = ciu(MTd["llama31" if m_ == "llama" else m_]["ratio_4_0"], 1)
    OPR.append([disp, "gold's rate (prevalence)"] + r_); OPR.append(["", "max $\\kappa$"] + r92(m_, "max kappa"))
assert abs(conf(yS("qwen7b_m"), zS)["kappa"] - ROWS92[("qwen", "gold prevalence")]["kappa"]) < 1e-12 and abs(conf(yS("gemma_m"), zS)["kappa"] - ROWS92[("gemma", "gold prevalence")]["kappa"]) < 1e-12
rows = [r_ if isinstance(r_, str) else " & ".join("--" if c_ is None else c_ for c_ in r_) + " \\\\" for r_ in OPR]
assert rows[0].split(" & ")[4].startswith(EX["dHaikuMainSfive"]) and rows[2].split(" & ")[4].startswith(EX["dHaikuPrev"])
table("xa_oppoints", rows,
      "Every LLM at its operating points on the 5% sample: own cut (for the four-level answer, the cut that maximizes kappa against gold), ``very toxic'' only "
      "(the cut whose flag rate is closest to gold's), gold's flag rate, the kappa-maximizing cut of the 0-100 rating and of the open models' scores against all "
      f"gold labels of the sample, and the 0-100 cuts chosen on {NCALL} gold labels. Distortions, the labeler's law minus gold's, with 95% intervals: the depth law "
      "pooled, within threads and at a fixed number of second-person words (points per level), the thread size law (points per doubling) and the log reply risk "
      "ratio; last, how many times as often the labeler flags gold negatives with four or more second-person words as those with none. Double dagger: median "
      f"over {NCALD} draws of {NCALL} gold labels, with the 5th and 95th percentiles over draws. Section sign: within-thread distortion of the same rating cut with ties "
      f"broken by a different random draw. Gold's laws here: depth {sg(MT2['gold']['depth'][0])}, thread size {sg(MT2['gold']['thread_size'][0])}.",
      "LLM | Operating point | Flags (%) | kappa | Depth law [CI] | Within threads [CI] | At fixed no. [CI] | Thread size law [CI] | Log RR [CI] | 4+:0 [CI]",
      f"{S92}; {SW} (within threads, Haiku, Sonnet, Qwen, Gemma); {SFAM} (within threads, OLMo-2, Granite, Llama); {SHR} (within threads, ratings); {SC_} (500-label "
      f"cuts); {SDO} and {S91} (4+:0)")
# the operating-point labels of xa_oppoints: on the 5% sample the four-level answer's kappa-maximizing cut is the own cut, and the cut closest to gold's flag
# rate is 'very toxic'
for m_ in ("haiku", "sonnet"):
    pc_ = CA["cc_s5"]["instruments"][f"{m_}:level"]["per_cut"]
    assert max(pc_, key=lambda r_: r_["kappa"])["cut"] == 2.0 and min(pc_, key=lambda r_: abs(r_["flag_rate"] - CA["cc_s5"]["prevalence"]))["cut"] == 3.0


# ================================================================ E28-E31: prompts
item("E28")
F124 = load("f124_prompts.json"); AU = F124["audit"]; IT = F124["item"]; S124 = "results/f124_prompts.json"; PRN = load("x81/prn.json")
CV124 = (f"separate sample of {PRN['n_slope']:,} comments in threads of two or more (f-series file; x81/prn.json: it contains all {PRN['n_slope_in_cc_s5']:,} comments "
         f"of the 5% sample); flag rate and kappa over its {F124['n_items']:,} items")
assert PRN["n_slope_in_cc_s5"] == NS == PRN["n_cc_s5"] and PRN["n_items"] == F124["n_items"]
put("xaPrN", th(PRN["n_slope"]), "results/x81/prn.json", "n_slope", "comments the slopes are fit on"); same("xaPrN", "qPrN")
trip("xaPrGold", AU["slope_z"], S124, "audit.slope_z", "x100: gold's depth law on that sample", sc=100, caveat=CV124)
for k_, nm, lab in (("main", "Main", "main prompt"), ("bin", "Bin", "yes/no question"), ("rev", "Rev", "reversed answer scale")):
    trip(f"xaPr{nm}Dist", AU[f"c_slope_{k_}0_minus_gold"], S124, f"audit.c_slope_{k_}0_minus_gold", f"x100: depth-law distortion, {lab}, without the parent", sc=100, caveat=CV124)
    put(f"xaPr{nm}Flag", pct(IT[f"{k_}0"]["prev"], 1), S124, f"item.{k_}0.prev", "share flagged", caveat=CV124)
    put(f"xaPr{nm}Kappa", f(IT[f"{k_}0"]["kappa"]), S124, f"item.{k_}0.kappa", "Cohen's kappa", caveat=CV124)
same("xaPrMainDist", "dHaikuPrMain"); same("xaPrBinDist", "dHaikuPrBin"); same("xaPrRevDist", "dHaikuPrRev"); same("xaPrGold", "qPrGold")
note(f"C13: the with-parent prompt comparisons of f124 (main1, bin1, rev1; qPrBinCtxChg = {plain(EX['qPrBinCtxChg'])}, the 'yes/no prompt with the parent flattens the slope') "
     "label replies with the parent and keep roots without it, so they are pooled; no among-replies version is stored, and they are not used.")

item("E29")
Y80 = load("x80/x80_youprompt_eval.json")["models"]; S80Y = "results/x80/x80_youprompt_eval.json"; Y81 = load("x81/x81_eval.json")["models"]; S81Y = "results/x81/x81_eval.json"
trip("xaYpHaikuChg", Y80["haiku"]["depth_change"], S80Y, "models.haiku.depth_change", "x100: distortion with the note minus without (cross-batch, B80e)", sc=100, caveat="5% sample")
same3("xaYpHaikuChg", "qYpHaikuChg")
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    h_ = Y81[m_]
    trip(f"xaYr{nm}Chg", h_["depth_change"], S81Y, f"models.{m_}.depth_change", "x100: same-batch change of the depth-law distortion, note minus main prompt (B81a)", sc=100, caveat="5% sample")
    trip(f"xaYr{nm}WithinChg", h_["within_change"], S81Y, f"models.{m_}.within_change", "x100: same for the within-thread distortion", sc=100, caveat="5% sample")
    trip(f"xaYr{nm}DistYou", h_["depth_dist_you"], S81Y, f"models.{m_}.depth_dist_you", "x100: distortion with the note", sc=100, caveat="5% sample")
    trip(f"xaYr{nm}DistMain", h_["depth_dist_main"], S81Y, f"models.{m_}.depth_dist_main", "x100: distortion of the main prompt in the same batch", sc=100, caveat="5% sample")
    put(f"xaYr{nm}FlagYou", pct(h_["flag_you"], 1), S81Y, f"models.{m_}.flag_you", "share flagged with the note", caveat="5% sample")
    put(f"xaYr{nm}FlagMain", pct(h_["flag_main"], 1), S81Y, f"models.{m_}.flag_main", "share flagged, main prompt", caveat="5% sample")
same3("xaYrHaikuChg", "qYrHaikuChg"); same3("xaYrSonnetChg", "qYrSonnetChg")
put("xaYrRootNoYouMain", pct(Y81["haiku"]["civil_main"]["d0_noyou"], 1), S81Y, "models.haiku.civil_main.d0_noyou", "Haiku's flag rate on clearly civil roots without 'you', main prompt")
put("xaYrRootNoYouNote", pct(Y81["haiku"]["civil_you"]["d0_noyou"], 1), S81Y, "models.haiku.civil_you.d0_noyou", "with the note")
same("xaYrRootNoYouMain", "qYrShallowMain"); same("xaYrRootNoYouNote", "qYrShallowYou")
def kap_(t, f_, p):
    q = p * t + (1 - p) * f_; po = p * t + (1 - p) * (1 - f_); pe = p * q + (1 - p) * (1 - q); return (po - pe) / (1 - pe)
p5 = zS.mean()

item("E30")
TU = load("x89/x89_tuned.json"); S89T = "results/x89/x89_tuned.json"
put("xaTunedDevKappa", f(TU["candidates"]["T8"]["dev_kappa"]["2"], 3), S89T, "candidates.T8.dev_kappa['2']", "kappa of the selected prompt on 2,000 development labels, own cut")
assert f(TU["candidates"]["T8"]["dev_kappa"]["2"]) == EX["cTunedKappa"]; logchk("xaTunedDevKappa", "two decimals = \\cTunedKappa")
for t_, nm in (("T0", "Zero"), ("T8", "Eight"), ("T16", "Sixteen"), ("TR", "Think"), ("M1", "MitOne"), ("M2", "MitTwo")):
    r_ = MI[t_]
    item("E31" if t_ in ("M1", "M2") else "E30")                                   # E30 tuned prompts, E31 mitigation prompts (one batch, x91)
    trip(f"xaPrompt{nm}Dist", r_["distortion"], S91, f"instruments.{t_}.distortion", "depth-law distortion (points per level), 5% sample relabeled in one batch", caveat="5% sample")
    put(f"xaPrompt{nm}Flag", pct(r_["flag_rate"], 1), S91, f"instruments.{t_}.flag_rate", "share flagged", caveat="5% sample")
    put(f"xaPrompt{nm}Kappa", f(r_["kappa"]), S91, f"instruments.{t_}.kappa", "Cohen's kappa", caveat="5% sample")
    trip(f"xaPrompt{nm}Fix", r_["distortion_fixed"], S91, f"instruments.{t_}.distortion_fixed", "distortion at a fixed number of second-person words", caveat="5% sample")
    put(f"xaPrompt{nm}Ratio", f(r_["ratio_4_0"][0], 1), S91, f"instruments.{t_}.ratio_4_0", "4+:0 FPR ratio", caveat="5% sample")
    if t_ != "T0":
        trip(f"xaPrompt{nm}Chg", r_["change_vs_T0"], S91, f"instruments.{t_}.change_vs_T0", "paired change against the main prompt in the same batch", caveat="5% sample")
        trip(f"xaPrompt{nm}FixChg", r_["change_fixed_vs_T0"], S91, f"instruments.{t_}.change_fixed_vs_T0", "paired change at a fixed number", caveat="5% sample")
        put(f"xaPrompt{nm}FixShare", pct(r_["distortion_fixed"][0] / r_["distortion"][0]), S91, f"instruments.{t_}.distortion_fixed / distortion",
            "share of the distortion that remains at a fixed number", caveat="5% sample")
item("E30")
for a_, b_ in (("xaPromptZeroDist", "dHaikuBatch"), ("xaPromptEightDist", "dHaikuTuned"), ("xaPromptThinkDist", "dHaikuThink"), ("xaPromptMitOneDist", "dHaikuMitOne"),
               ("xaPromptMitTwoDist", "dHaikuMitTwo"), ("xaPromptEightKappa", "dKapTuned"), ("xaPromptMitOneFix", "cMitOneFix"), ("xaPromptMitTwoFix", "cMitTwoFix"),
               ("xaPromptZeroFix", "cMitZeroFix"), ("xaPromptMitOneRatio", "cMitOneRatio"), ("xaPromptMitTwoRatio", "cMitTwoRatio"), ("xaPromptZeroRatio", "cMitZeroRatio")):
    same(a_, b_)
same3("xaPromptEightChg", "dTunedChg"); same3("xaPromptMitOneChg", "cMitOneChg"); same3("xaPromptMitTwoChg", "cMitTwoChg"); same3("xaPromptZeroDist", "cMitZeroDist")
same3("xaPromptMitOneDist", "cMitOneDist"); same3("xaPromptMitTwoDist", "cMitTwoDist")
assert M["xaPromptEightDist"] == EX["cTunedDist"]
differs("xaPromptEightDistLo", "cTunedDistLo", "the earlier-draft interval of the tuned prompt (cTunedDist*) comes from x89_tuned.json's bootstrap; x91_mitigate.json's own resamples, "
        f"used here so that every prompt row and its paired change share one bootstrap, give [{M['xaPromptEightDistLo']}, {M['xaPromptEightDistHi']}] for the same point value {M['xaPromptEightDist']}")
differs("xaPromptEightDistHi", "cTunedDistHi", "as xaPromptEightDistLo")
same("xaPromptThinkDist", "cTunedThink")
item("E31")
note("The inventory's 'At a fixed number of second-person words: M1 +0.34, M2 +0.14, T8 +0.29, TR +0.65' mixes levels and changes: M1 \\xaPromptMitOneFix{} and "
     "M2 \\xaPromptMitTwoFix{} are the distortions at a fixed number (x91 distortion_fixed), while T8 \\xaPromptEightFixChg{} and TR \\xaPromptThinkFixChg{} are changes "
     "against T0 at a fixed number (change_fixed_vs_T0). The levels are T8 \\xaPromptEightFix, TR \\xaPromptThinkFix; the changes M1 \\xaPromptMitOneFixChg, M2 \\xaPromptMitTwoFixChg.")
alt = ("M1", "T8", "T16", "TR"); shf = [MI[t_]["distortion_fixed"][0] / MI[t_]["distortion"][0] for t_ in alt]
put("xaAltFixMin", pct(min(shf)), S91, "instruments.{M1,T8,T16,TR}.distortion_fixed / distortion", "smallest share remaining at a fixed number among the prompts that enlarge the distortion")
put("xaAltFixMax", pct(max(shf)), S91, "same", "largest"); same("xaAltFixMin", "cAltFixMin"); same("xaAltFixMax", "cAltFixMax")

item("E32")
CC_ = load("x74_ccc/x74_ccc.json"); SCC = "results/x74_ccc/x74_ccc.json"; Xc = CC_["estimates"]
put("xaCccN", th(CC_["n"]), SCC, "n", "CCC replies with depth (1-6)")
trip("xaCccHuman", Xc["slope_Y_hic"], SCC, "estimates.slope_Y_hic", "x100: depth slope of the in-context raters' majority (depths 1-6)", sc=100)
trip("xaCccHaikuLaw", Xc["slope_Y_haiku_ic"], SCC, "estimates.slope_Y_haiku_ic", "x100: Haiku shown the parent", sc=100)
trip("xaCccHaikuDist", Xc["diff_Y_haiku_ic_vs_Y_hic"], SCC, "estimates.diff_Y_haiku_ic_vs_Y_hic", "x100: paired difference (2,000 post resamples)", sc=100)
trip("xaCccHumanOut", Xc["slope_Y_hoc"], SCC, "estimates.slope_Y_hoc", "x100: the original raters (no parent shown) on the same replies", sc=100)
trip("xaCccHaikuOutLaw", Xc["slope_Y_haiku_oc"], SCC, "estimates.slope_Y_haiku_oc", "x100: Haiku without the parent on the same replies", sc=100)
same3("xaCccHuman", "wCccHumIC"); same("xaCccHaikuDist", "dHaikuCcc"); same("xaCccHaikuLaw", "wCccHaikuIC"); same("xaCccHaikuOutLaw", "wCccHaikuOC")
nbd = {int(k_): v_ for k_, v_ in CC_["n_by_depth"].items()}; rbd_ = CC_["rates_by_depth"]
wavg = lambda c_: sum(rbd_[c_][str(d_)] * n_ for d_, n_ in nbd.items()) / sum(nbd.values())
assert sum(nbd.values()) == CC_["n"]
for c_, nm in (("Y_hic", "PrevIn"), ("Y_hoc", "PrevOut"), ("Y_haiku_ic", "HaikuFlagIn"), ("Y_haiku_oc", "HaikuFlagOut")):
    put(f"xaCcc{nm}", pct(wavg(c_), 1), SCC, f"rates_by_depth.{c_} weighted by n_by_depth", "share flagged among the CCC replies")
_qr = list(CC_["rates_by_depth"]["Y_qwen_ic"].values())
note(f"x74_ccc's Qwen rows flag at a threshold matched to the out-of-context prevalence (reused for the in-context scores, {pct(min(_qr))}-{pct(max(_qr))} flagged by depth); they are not used.")

item("E34")
NF = load("x88/x88_noisefree_depth.json"); SNF = "results/x88/x88_noisefree_depth.json"
trip("xaNfDepth", NF["noise_free"], SNF, "noise_free", "depth law of the majority of infinitely many raters (NPMLE on 40 grid points; 100 article resamples)", caveat="model-based")
same3("xaNfDepth", "cNfDepth")
trip("xaNfMinusGold", NF["noise_free_minus_gold"], SNF, "noise_free_minus_gold", "paired difference from gold's law", caveat="model-based")
trip("xaRatersSlope", NF["raters"], SNF, "raters", "slope of the number of raters per comment on depth (raters per level)"); same("xaRatersSlope", "cRatersSlope")
bdn = NF["by_depth"]; nfprev = sum(v_["noise_free"] * v_["n"] for v_ in bdn.values()) / sum(v_["n"] for v_ in bdn.values())
assert sum(v_["n"] for v_ in bdn.values()) == NT
put("xaNfPrev", pct(nfprev, 1), SNF, "by_depth.*.noise_free weighted by n", "estimated share toxic under the majority of infinitely many raters", caveat="model-based")

item("E35")
MA80 = load("x80/x80_matched.json"); S80M = "results/x80/x80_matched.json"
for pop, m_, nm in (("cc_threads", "haiku", "Haiku"), ("cc_s5", "sonnet", "Sonnet")):
    r_ = MA80[pop]["instruments"][m_]
    trip(f"xaVf{nm}Law", r_["depth_human"], S80M, f"{pop}.instruments.{m_}.depth_human", "x100: depth law of the comments with the largest vote fractions, as many as the LLM flags", sc=100,
         caveat="5% sample" if pop == "cc_s5" else "")
    trip(f"xaVf{nm}Excess", r_["depth_minus_human"], S80M, f"{pop}.instruments.{m_}.depth_minus_human", "x100: the LLM's depth law minus that instrument's", sc=100,
         caveat="5% sample" if pop == "cc_s5" else "")
    put(f"xaVf{nm}Flag", pct(r_["flag_rate"], 1), S80M, f"{pop}.instruments.{m_}.flag_rate", "flag rate matched", caveat="5% sample" if pop == "cc_s5" else "")
same("xaVfHaikuLaw", "qMatchHumHaikuDepth"); same("xaVfSonnetLaw", "qMatchHumSonnetDepth")
_h80 = {d_["instrument"]: d_["lower"] for d_ in MA80["H80b"]["detail"]}
note(f"x80_matched's H80b list also has Haiku, Qwen and Sonnet shown the parent (lower ends {sg(100 * _h80['haiku_ctx'])}, {sg(100 * _h80['qwen7b_m_ctx@own'])}, "
     f"{sg(100 * _h80['sonnet_ctx'])}); those are pooled over roots labeled without the parent and are not used. Only Haiku (lower end \\xaVfHaikuExcessLo) and Sonnet "
     "(\\xaVfSonnetExcessLo) are reported.")
assert MA80["H80b"]["detail"][0]["instrument"] == "haiku"
for t_ in ["0.02", "0.05", "0.075", "0.1", "0.15", "0.2", "0.3"]:
    trip(f"xaVfLaw{GRIDNM[t_]}", HR[t_]["human"]["depth"], SHR, f"rates.{t_}.human.depth", "x100: depth law of the vote-fraction instrument flagging that share", sc=100, caveat="5% sample")
vfl = [HR[t_]["human"]["depth"][0] for t_ in GRID]
put("xaVfLawMin", sg(100 * max(vfl)), SHR, "rates.<2%-30%>.human.depth", "least negative of the seven", caveat="5% sample")
put("xaVfLawMax", sg(100 * min(vfl)), SHR, "same", "most negative", caveat="5% sample")
assert all(v_ < 0 for v_ in vfl)

item("E36")
TH = load("x80/x80_threshold.json"); S80T = "results/x80/x80_threshold.json"; BT = TH["pops"]["cc_s5"]["by_t"]; TT = [f"{t_}" for t_ in TH["t"]]
TNM = {"0.3": "Three", "0.4": "Four", "0.5": "Five", "0.6": "Six", "0.7": "Seven", "0.8": "Eight"}
thr = []
for t_ in TT:
    b_ = BT[t_]
    for m_, nm in (("haiku_rating", "Haiku"), ("sonnet_rating", "Sonnet")):
        trip(f"xaThr{nm}{TNM[t_]}", b_["instruments"][m_]["depth_minus_gold"], S80T, f"pops.cc_s5.by_t.{t_}.instruments.{m_}.depth_minus_gold",
             "x100: the 0-100 rating flagging gold_t's share, depth law minus gold_t's", sc=100, caveat="5% sample"); thr.append(b_["instruments"][m_]["depth_minus_gold"])
    put(f"xaThrGoldLaw{TNM[t_]}", sg(100 * b_["gold_depth"]), S80T, f"pops.cc_s5.by_t.{t_}.gold_depth", "x100: gold_t's depth law", caveat="5% sample")
    put(f"xaThrGoldPrev{TNM[t_]}", pct(b_["prev"], 1), S80T, f"pops.cc_s5.by_t.{t_}.prev", "share toxic under gold_t", caveat="5% sample")
put("xaThrMin", sg(100 * min(v_[0] for v_ in thr)), S80T, "pops.cc_s5.by_t.*.instruments.{haiku,sonnet}_rating.depth_minus_gold", "smallest of the 12", caveat="5% sample")
put("xaThrMax", sg(100 * max(v_[0] for v_ in thr)), S80T, "same", "largest", caveat="5% sample")
put("xaThrNPos", str(sum(v_[1] > 0 for v_ in thr)), S80T, "same", "CIs above 0", caveat="5% sample"); put("xaThrN", str(len(thr)), S80T, "same", "settings", caveat="5% sample")
same("xaThrMin", "qThrLLMMin"); same("xaThrMax", "qThrLLMMax")


# ---------------------------------------------------------------- B81a flags (main prompt and the earlier note, same batch): kappa from the stored answers
item("E29")
K81 = {}
for m_ in ("haiku", "sonnet"):
    an_ = {"tox": {}, "tox_you": {}}
    for l in open(f"{R}/x81/{m_}.jsonl"):
        r_ = json.loads(l); an_[r_["task"]][int(r_["id"])] = r_.get("ans")
    am = np.array([an_["tox"].get(int(i)) for i in idS], dtype=object); ay = np.array([an_["tox_you"].get(int(i)) for i in idS], dtype=object)
    okp = np.array([(u in ("A", "B", "C", "D")) and (w in ("A", "B", "C", "D")) for u, w in zip(am, ay)])
    fm = np.isin(am, ["A", "B"]).astype(float); fy = np.isin(ay, ["A", "B"]).astype(float)
    assert int(okp.sum()) == Y81[m_]["n_paired"] and abs(fm[okp].mean() - Y81[m_]["flag_main"]) < 1e-12 and abs(fy[okp].mean() - Y81[m_]["flag_you"]) < 1e-12
    cm_, cy_ = conf(fm, zS, okp), conf(fy, zS, okp)
    assert abs(cm_["tpr"] - Y81[m_]["tpr_main"][0]) < 1e-12 and abs(cy_["fpr"] - Y81[m_]["fpr_you"][0]) < 1e-12
    K81[m_] = (cm_, cy_)
    nm = {"haiku": "Haiku", "sonnet": "Sonnet"}[m_]
    put(f"xaYr{nm}KappaMain", f(cm_["kappa"]), "recomputed: results/x81/{m}.jsonl (task tox)", "paired comments", "Cohen's kappa, main prompt in the B81a batch", caveat="5% sample; recomputed")
    put(f"xaYr{nm}KappaYou", f(cy_["kappa"]), "recomputed: results/x81/{m}.jsonl (task tox_you)", "paired comments", "Cohen's kappa with the note", caveat="5% sample; recomputed")


# ---------------------------------------------------------------- tables/xa_esc_full.tex: earlier draft's Table 2 with flag rates, kappa and CIs
item("E01")
CR = load("x89/x89_cc_reference.json"); SCR = "results/x89/x89_cc_reference.json"
RR_ = {"hT": conf(yT("haiku"), zT, repT), "hS": conf(yS("haiku"), zS, repS), "sS": conf(yS("sonnet"), zS, repS),
       "hcT": SET["HaikuCtx"], "hcS": conf(yS("haiku_ctx"), zS, repS), "scS": SET["SonnetCtx"]}
cH = lambda c_: (f(100 * c_["flag"], 1), f(c_["kappa"]))
def er(label, hflag="", hlaw="", hk=("", "", ""), sn=("", "", "")):
    return " & ".join([label, hflag, hlaw, *hk, *sn]) + " \\\\"
g5 = MT2["gold"]["depth"]
nat = {m_: ROWS92[(m_, "native")] for m_ in ("haiku", "sonnet")}
E = []
E.append(er("Main prompt, own cut", f(100 * zT.mean(), 1), ci3(GT["depth_slope"], 2, 100), (*cH(SET["Haiku"]), ci3(DO["cc_threads"]["labelers"]["haiku"]["law_distortion"])), ("", "", "")))
E.append(er("\\quad 5\\% sample$^{\\dagger}$", f(100 * zS.mean(), 1), ci3(g5), (f(100 * nat["haiku"]["flag_rate"], 1), f(nat["haiku"]["kappa"]), ci3(nat["haiku"]["depth_excess"])),
            (f(100 * nat["sonnet"]["flag_rate"], 1), f(nat["sonnet"]["kappa"]), ci3(nat["sonnet"]["depth_excess"]))))
E.append(er("Within threads", "", ci3(WI["cc_threads"]["gold_within"], 2, 100), (*cH(SET["Haiku"]), ci3(WI["cc_threads"]["labelers"]["haiku"]["distortion"], 2, 100))))
E.append(er("\\quad 5\\% sample$^{\\dagger}$", "", ci3(WI["cc_s5"]["gold_within"], 2, 100), (f(100 * nat["haiku"]["flag_rate"], 1), f(nat["haiku"]["kappa"]), ci3(WI["cc_s5"]["labelers"]["haiku"]["distortion"], 2, 100)),
            (f(100 * nat["sonnet"]["flag_rate"], 1), f(nat["sonnet"]["kappa"]), ci3(WI["cc_s5"]["labelers"]["sonnet"]["distortion"], 2, 100))))
E.append(er("Among replies", f(100 * zT[repT].mean(), 1), ci3(CR["cc_threads"]["gold"]["depth_replies"]), (*cH(RR_["hT"]), ci3(CR["cc_threads"]["instruments"]["haiku_tox"]["depth_replies_distortion"]))))
E.append(er("\\quad 5\\% sample$^{\\dagger}$", f(100 * zS[repS].mean(), 1), ci3(CR["cc_s5"]["gold"]["depth_replies"]), (*cH(RR_["hS"]), ci3(CR["cc_s5"]["instruments"]["haiku_tox"]["depth_replies_distortion"])),
            (*cH(RR_["sS"]), ci3(CR["cc_s5"]["instruments"]["sonnet_tox"]["depth_replies_distortion"]))))
E.append(er("\\quad shown the parent", "", "", (*cH(RR_["hcT"]), ci3(CR["cc_threads"]["instruments"]["haiku_tox_ctx"]["depth_replies_distortion"]))))
E.append(er("\\quad shown the parent, 5\\% sample$^{\\dagger}$", "", "", (*cH(RR_["hcS"]), ci3(CR["cc_s5"]["instruments"]["haiku_tox_ctx"]["depth_replies_distortion"])),
            (*cH(RR_["scS"]), ci3(CR["cc_s5"]["instruments"]["sonnet_tox_ctx"]["depth_replies_distortion"]))))
E.append(er("\\quad shown the parent, vs.\\ CCC's raters", M["xaCccPrevIn"].replace("\\%", ""), ci3(Xc["slope_Y_hic"], 2, 100), (M["xaCccHaikuFlagIn"].replace("\\%", ""), "--", ci3(Xc["diff_Y_haiku_ic_vs_Y_hic"], 2, 100))))
E.append("\\midrule"); E.append("\\multicolumn{9}{@{}l}{\\emph{Other readings of the human votes}} \\\\")
E.append(er("Noise-free majority (infinitely many raters)", M["xaNfPrev"].replace("\\%", ""), ci3(NF["noise_free"])))
E.append(er("Vote-fraction instrument, Haiku's flag rate", f(100 * MA80["cc_threads"]["instruments"]["haiku"]["flag_rate"], 1), ci3(MA80["cc_threads"]["instruments"]["haiku"]["depth_human"], 2, 100),
            (*cH(SET["Haiku"]), ci3(MA80["cc_threads"]["instruments"]["haiku"]["depth_minus_human"], 2, 100))))
E.append(er("Vote-fraction instrument, Sonnet's rate$^{\\dagger}$", f(100 * MA80["cc_s5"]["instruments"]["sonnet"]["flag_rate"], 1), ci3(MA80["cc_s5"]["instruments"]["sonnet"]["depth_human"], 2, 100),
            ("", "", ""), (*cH(SET["Sonnet"]), ci3(MA80["cc_s5"]["instruments"]["sonnet"]["depth_minus_human"], 2, 100))))
E.append("\\midrule"); E.append("\\multicolumn{9}{@{}l}{\\emph{Other labeling settings, on the 5\\% sample}} \\\\")
E.append(er("Relabeled in a later batch", "", "", (f(100 * RR8["haiku"]["flag_rate_new"], 1), f(RRK["haiku"]["kappa"]), ci3(RR8["haiku"]["distortion_new"])),
            (f(100 * RR8["sonnet"]["flag_rate_new"], 1), f(RRK["sonnet"]["kappa"]), ci3(RR8["sonnet"]["distortion_new"]))))
ck_ = lambda m_: CA["cc_s5"]["instruments"][f"{m_}:rating"]["calibrated"]["kappa@500"]
E.append(er("0--100 rating, cut maximizing $\\kappa$", "", "", *[(f(100 * ck_(m_)["flag_rate_median"], 1), f(ck_(m_)["kappa_median"]),
            f"{sg(ck_(m_)['excess_median'])} [{sg(ck_(m_)['excess_p05'])}, {sg(ck_(m_)['excess_p95'])}]$^{{\\ddagger}}$") for m_ in ("haiku", "sonnet")]))
E.append(er("0--100 rating, gold's flag rate", "", "", *[(f(100 * gp[m_]["flag_rate"], 1), f(gp[m_]["kappa"]), ci3(gp[m_]["depth_excess"])) for m_ in ("haiku", "sonnet")]))
vt_ = {m_: ROWS92[(m_, "very toxic")] for m_ in ("haiku", "sonnet")}
E.append(er("``Very toxic'' only", "", "", *[(f(100 * vt_[m_]["flag_rate"], 1), f(vt_[m_]["kappa"]), ci3(vt_[m_]["depth_excess"])) for m_ in ("haiku", "sonnet")]))
for t_, lab in (("T0", "Main prompt, rerun with the variants below"), ("T8", "Eight examples (best of the four candidates)"), ("T16", "Sixteen examples"),
                ("TR", "Extended thinking"), ("M1", "Told address is not by itself toxic"), ("M2", "Third-person instructions")):
    r_ = MI[t_]; E.append(er(lab, "", "", (f(100 * r_["flag_rate"], 1), f(r_["kappa"]), ci3(r_["distortion"]))))
for k_, lab in (("you", "Told addressing someone is not toxic (earlier wording)"),):          # arXiv revision: the batch's main-prompt row is the later relabeling above (same labels)
    E.append(er(lab, "", "", *[(f(100 * Y81[m_][f"flag_{k_}"], 1), f(K81[m_][0 if k_ == "main" else 1]["kappa"]), ci3(Y81[m_][f"depth_dist_{k_}"], 2, 100)) for m_ in ("haiku", "sonnet")]))
E.append("\\midrule"); E.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Second run on the sampled articles ({th(PRN['n_slope'])} comments)}}}} \\\\")
E.append(er("Main prompt", f(100 * IT["z"]["prev"], 1), ci3(AU["slope_z"], 2, 100), (f(100 * IT["main0"]["prev"], 1), f(IT["main0"]["kappa"]), ci3(AU["c_slope_main0_minus_gold"], 2, 100))))
E.append(er("Yes/no question", "", "", (f(100 * IT["bin0"]["prev"], 1), f(IT["bin0"]["kappa"]), ci3(AU["c_slope_bin0_minus_gold"], 2, 100))))
E.append(er("Reversed answer scale", "", "", (f(100 * IT["rev0"]["prev"], 1), f(IT["rev0"]["kappa"]), ci3(AU["c_slope_rev0_minus_gold"], 2, 100))))
# every distortion of the earlier draft's Table 2 appears with the same rounding
chk2 = {"dHaikuMain": E[0], "dHaikuMainSfive": E[1], "dSonnetMain": E[1], "dHaikuWithin": E[2], "dSonnetWithin": E[3], "dHaikuRepl": E[4], "dSonnetRepl": E[5], "dHaikuCtx": E[6],
        "dSonnetCtx": E[7], "dHaikuCcc": E[8]}
for k_, r_ in chk2.items():
    assert (" " + EX[k_] + " [") in (" " + r_.replace(" & ", " ")), (k_, EX[k_], r_)
lawchk = {"wGoldDepthPP": E[0], "wGoldDepthSfivePP": E[1], "qGoldWithinPP": E[2], "qGoldWithinSfivePP": E[3], "cGoldReplies": E[4], "cGoldRepliesSfive": E[5], "wCccHumIC": E[8]}
for k_, r_ in lawchk.items():
    assert r_.split(" & ")[2].startswith(EX[k_] + " ["), (k_, r_)
for k_, i_ in (("cNfDepth", 11), ("qMatchHumHaikuDepth", 12), ("qMatchHumSonnetDepth", 13)):
    assert E[i_].split(" & ")[2].startswith(EX[k_] + " ["), (k_, E[i_])
for k_, i_, col in (("dHaikuRerun", 16, 5), ("dSonnetRerun", 16, 8), ("dHaikuKmax", 17, 5), ("dSonnetKmax", 17, 8), ("dHaikuPrev", 18, 5), ("dSonnetPrev", 18, 8), ("dHaikuVt", 19, 5),
                    ("dSonnetVt", 19, 8), ("dHaikuBatch", 20, 5), ("dHaikuTuned", 21, 5), ("dHaikuThink", 23, 5), ("dHaikuMitOne", 24, 5), ("dHaikuMitTwo", 25, 5),
                    ("dHaikuPrMain", 29, 5), ("dHaikuPrBin", 30, 5), ("dHaikuPrRev", 31, 5)):          # arXiv revision: one row fewer above (the duplicate relabeling row)
    assert E[i_].split(" & ")[col].startswith(EX[k_] + " ["), (k_, i_, E[i_])
assert E[0].split(" & ")[3] == "22.3" and E[20].split(" & ")[3] == "22.2"
table("xa_esc_full", E,
      "Escalation under alternatives: the depth law under human labels and each LLM's distortion of it (points per level, 95% intervals), with each setting's "
      "flag rate and Cohen's kappa against gold. Haiku on all reply threads unless marked; Sonnet and rows marked with a dagger on the 5% sample. Among replies "
      "and shown the parent: the depth law among replies, with flag rate and kappa among replies. In the rows for other readings of the votes, the human column "
      "is that reading and the distortion is the LLM's law minus it; the share toxic under infinitely many raters is model-based. Largest vote fractions: the "
      f"comments with the largest shares of toxic votes, as many as the LLM flags. Double dagger: the cut that maximizes kappa on {NCALL} gold labels, median over "
      f"{NCALD} draws with the 5th and 95th percentiles over draws. The main prompt rerun with the variants and the variants below it were labeled in one batch; "
      "the earlier wording of the note (``Addressing someone directly as `you' is not toxic in itself.'') was labeled in a batch with its own rerun of the main prompt.",
      "Setting | Human labels: flags (%), law [CI] | Haiku: flags (%), kappa, distortion [CI] | Sonnet: flags (%), kappa, distortion [CI]",
      f"{SRC_PL} (gold's law); {SDO} (Haiku's main distortion); {S92}; {SW}; {SCR}; {SCC}; {SNF}; {S80M}; {SRR} and results/x81/*.jsonl (rerun kappa); {SC_}; {S91}; "
      f"{S81Y} and results/x81/*.jsonl (kappa); {S124}; x14_pops parquets (flag rate and kappa among replies, recomputed)")


# ---------------------------------------------------------------- tables/xa_thresholds_depth.tex (E36 depth part, E35 depth part)
item("E36")
rows = ["\\multicolumn{5}{@{}l}{\\emph{Gold defined at vote threshold $t$; each rating flags gold$_t$'s share}} \\\\"]
for t_ in TT:
    b_ = BT[t_]
    rows.append(" & ".join([f"$t = {t_}$", f(100 * b_["prev"], 1), sg(100 * b_["gold_depth"]), ci3(b_["instruments"]["haiku_rating"]["depth_minus_gold"], 2, 100),
                            ci3(b_["instruments"]["sonnet_rating"]["depth_minus_gold"], 2, 100)]) + " \\\\")
rows += ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Gold at $t = 0.5$; the vote-fraction instrument and each rating flag the same share}} \\\\"]
for j_, t_ in enumerate(GRID):
    h_ = HR[t_]
    rows.append(" & ".join(["Vote-fraction instrument" if j_ == 0 else "", f(100 * float(t_), 1), ci3(h_["human"]["depth"], 2, 100), ci3(h_["haiku_r"]["depth_minus_human"], 2, 100),
                            ci3(h_["sonnet_r"]["depth_minus_human"], 2, 100)]) + " \\\\")
table("xa_thresholds_depth", rows,
      "The 0-100 ratings' depth law against other definitions of the human label, on the 5% sample (points per level, 95% intervals). Top: gold is the majority "
      "at vote threshold t (a comment is toxic when at least a share t of its raters say so); each rating flags as many comments as gold_t calls toxic, and the "
      "columns give gold_t's flag rate and depth law and each rating's depth law minus gold_t's. Bottom: at a fixed flag rate, the human instrument flags the "
      "comments with the largest shares of toxic votes; the columns give its depth law and each rating's depth law minus the instrument's at the same flag rate.",
      "Reference | Flags (%) | Reference's depth law [CI] | Haiku-100 minus reference [CI] | Sonnet-100 minus reference [CI]",
      f"{S80T} (top); {SHR} (bottom)")
assert all(v_[1] > 0 for v_ in thr)
item("E35")
note(f"Bottom panel of xa_thresholds_depth: x77_humanrate's ratings are flagged with ties broken by seed 7500 (as x75_rating_eval.py), not x92's seed 9010; at 7.5% Haiku-100's "
     f"law minus the instrument's is {sg(100 * HR['0.075']['haiku_r']['depth_minus_human'][0])} here (\\xaRateHaikuDist{{}} at the exact gold rate in x75_rating_eval).")


# ================================================================ figs/oppoint_grid.pdf (E23): distortion of the depth law against the flag rate
item("E23")
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"; C2, C3 = "#eb6834", "#1baf7a"   # p31_fig_dose.py: Haiku orange, Sonnet aqua (validated pair)
fig, ax = plt.subplots(figsize=(3.4, 2.1), facecolor=SURF); ax.set_facecolor(SURF)
FD = {}
for (m_, nm, col, mk, dx) in (("haiku_r", "Haiku-100", C2, "o", 0.965), ("sonnet_r", "Sonnet-100", C3, "s", 1.035)):
    xs = np.array([100 * RA["instruments"][m_][t_]["flag_rate"][0] for t_ in GRID]); v_ = np.array([RA["instruments"][m_][t_]["depth_minus_gold"] for t_ in GRID]) * 100
    FD[nm] = (xs, v_)
    ax.errorbar(xs * dx, v_[:, 0], yerr=[v_[:, 0] - v_[:, 1], v_[:, 2] - v_[:, 0]], color=col, lw=1.4, marker=mk, ms=3.6, mec=SURF, mew=0.6, ecolor=col, elinewidth=0.8,
                capsize=0, zorder=4, label=nm)
gold_rate = 100 * RA["prev"]
ax.axvline(gold_rate, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2); ax.axhline(0, color=INK2, lw=0.7, zorder=2)
ax.text(gold_rate * 1.04, 1.62, "gold's flag rate", fontsize=7, color=INK2, va="top", ha="left")
ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 30]); ax.set_xticklabels(["2", "5", "10", "20", "30"]); ax.minorticks_off()
ax.set_xlim(1.7, 36); ax.set_ylim(-0.12, 1.72)
ax.set_xlabel("share of comments flagged (%)", fontsize=7, color=INK2, labelpad=1)
ax.set_ylabel("depth law minus gold's\n(points per level)", fontsize=7, color=INK2, labelpad=2)
ax.grid(True, axis="y", color=GRIDC, lw=0.5, zorder=0); ax.tick_params(colors=INK2, labelsize=7, pad=1.5, length=2)
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
for sp_ in ("left", "bottom"):
    ax.spines[sp_].set_color(INK2); ax.spines[sp_].set_linewidth(0.6)
leg = ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.2, labelcolor=INK2)
fig.tight_layout(pad=0.3)
FIGPDF = f"{FIG}/oppoint_grid.pdf"; fig.savefig(FIGPDF, facecolor=SURF, metadata={"CreationDate": None})   # no timestamp: byte-identical reruns
plt.close(fig)
assert all(np.all(v_[:, 1] > 0) for _, v_ in FD.values())                           # every CI above 0 (the figure's message)
assert all(abs(FD[nm][0][j] - 100 * float(t_)) < 0.01 for nm in FD for j, t_ in enumerate(GRID))



# ================================================================ further cross-checks against the earlier-draft macros (numbers that the earlier draft also prints)
item("E01")
put("xaGoldPrev", pct(zT.mean(), 1), "results/x14_pops/cc_threads.parquet", "z", "gold prevalence, all reply threads"); same("xaGoldPrev", "dGoldPrev")
put("xaHaikuFlag", pct(SET["Haiku"]["flag"]), "results/x14_pops/cc_threads.parquet", "y:haiku", "share flagged, rounded"); same("xaHaikuFlag", "dHaikuFlagAll"); same("xaHaikuFlag", "dMainFlag")
same("xaHaikuFlag", "qMatchHaikuFlag")
put("xaSonnetFlag", pct(SET["Sonnet"]["flag"]), "results/x14_pops/cc_s5.parquet", "y:sonnet", "share flagged, rounded", caveat="5% sample"); same("xaSonnetFlag", "dSonnetFlagSfive")
same("xaSonnetFlag", "qMatchSonnetFlag")
put("xaFlagRatioHaiku", f(SET["Haiku"]["flag"] / zT.mean(), 1), "x14_pops", "y:haiku / z", "Haiku flags this many times gold's share"); same("xaFlagRatioHaiku", "qFlagRatioHaiku")
put("xaFlagRatioSonnet", f(SET["Sonnet"]["flag"] / zS.mean(), 1), "x14_pops", "y:sonnet / z", "Sonnet (5% sample)", caveat="5% sample"); same("xaFlagRatioSonnet", "qFlagRatioSonnet")
trip("xaGoldDepthSfive", g5, S92, "gold.depth", "gold's depth law on the 5% sample (points per level), x92's resamples", caveat="5% sample")
same("xaGoldDepthSfive", "wGoldDepthSfivePP"); same("xaGoldDepthSfive", "qRateGoldDepthPP"); same("xaGoldDepthSfive", "tOneGoldSfiveDepth")
put("xaGoldRRSfive", f(PL["cc_threads_s5"]["gold"]["rr"][0]), SRC_PL, "cc_threads_s5.gold.rr", "gold's reply RR, 5% sample", caveat="5% sample"); same("xaGoldRRSfive", "qGoldRRSfive")
for nm, k_, s5 in (("Qwen", "qwen7b_m", 0), ("Gemma", "gemma_m", 1)):
    r_ = (PL5 if s5 else PLc)[k_]; pk = f"{'cc_threads_s5' if s5 else 'cc_threads'}.labelers.{k_}"
    trip(f"xa{nm}RR", r_["law_rr"], SRC_PL, pk + ".law_rr", "reply risk ratio", signed=False, caveat="5% sample" if s5 else "")
    trip(f"xa{nm}Law", r_["law_depth_slope"], SRC_PL, pk + ".law_depth_slope", "x100 (points per level)", sc=100, caveat="5% sample" if s5 else "")
same("xaGemmaRR", "qGemmaRR")
trip("xaHaikuMainDist", DO["cc_threads"]["labelers"]["haiku"]["law_distortion"], SDO, "cc_threads.labelers.haiku.law_distortion", "Haiku's depth-law distortion, all reply threads")
same("xaHaikuMainDist", "dHaikuMain"); same("xaHaikuMainDist", "cDoseLawHaiku")
trip("xaHaikuMainDistSfive", nat["haiku"]["depth_excess"], S92, "rows[haiku, native].depth_excess", "on the 5% sample", caveat="5% sample")
trip("xaSonnetMainDist", nat["sonnet"]["depth_excess"], S92, "rows[sonnet, native].depth_excess", "Sonnet's depth-law distortion", caveat="5% sample")
same("xaHaikuMainDistSfive", "dHaikuMainSfive"); same("xaSonnetMainDist", "dSonnetMain"); same("xaSonnetMainDist", "cDoseLawSon"); same("xaHaikuMainDistSfive", "qYpHaikuDistMain")
put("xaHaikuSfiveKappa", f(nat["haiku"]["kappa"]), S92, "rows[haiku, native].kappa", "Haiku's kappa on the 5% sample", caveat="5% sample")
same("xaHaikuSfiveKappa", "cKapHaikuSfive"); same("xaHaikuSfiveKappa", "dKapMainSfive")
put("xaHaikuSfiveLaw", sg(g5[0] + nat["haiku"]["depth_excess"][0]), S92, "gold.depth + rows[haiku, native].depth_excess", "Haiku's depth law on the 5% sample", caveat="5% sample")
same("xaHaikuSfiveLaw", "cLawHaikuSfive")
put("xaHaikuHundredLaw", sg(g5[0] + gp["haiku"]["depth_excess"][0]), S92, "gold.depth + rows[haiku, rating at gold prevalence].depth_excess", "Haiku-100's depth law", caveat="5% sample")
put("xaGemmaPrevLaw", sg(g5[0] + gp["gemma"]["depth_excess"][0]), S92, "gold.depth + rows[gemma, gold prevalence].depth_excess", "Gemma's depth law", caveat="5% sample")
same("xaHaikuHundredLaw", "dLawHaikuPrev"); same("xaGemmaPrevLaw", "dLawGemmaPrev"); same("xaGemmaPrevLaw", "cLawGemmaSfive"); same("xaGemmaPrevKappa", "cKapGemmaSfive")
for nm, pop, k_ in (("HaikuReplies", "cc_threads", "haiku_tox"), ("HaikuCtxReplies", "cc_threads", "haiku_tox_ctx"), ("SonnetCtxReplies", "cc_s5", "sonnet_tox_ctx"),
                    ("SonnetReplies", "cc_s5", "sonnet_tox")):
    trip(f"xa{nm}Law", CR[pop]["instruments"][k_]["depth_replies"], SCR, f"{pop}.instruments.{k_}.depth_replies", "depth law among replies (points per level)",
         caveat="among replies" + ("; 5% sample" if pop == "cc_s5" else ""))
same("xaHaikuRepliesLaw", "cHaikuReplies"); same("xaHaikuCtxRepliesLaw", "cHaikuCtxReplies"); same("xaSonnetCtxRepliesLaw", "cSonnetCtxReplies")
put("xaQwenCtxRepliesLaw", sg(slr(OWN_T["qwen7b_m_ctx"])), "recomputed (x76_rethreshold.py's flags)", "slope among replies", "Qwen shown the parent: depth law among replies", caveat="among replies")
same("xaQwenCtxRepliesLaw", "cQwenCtxReplies")
item("E02")
put("xaSrPlac", sg(100 * sr_pd_abs), SO, "splits[*].instr.crowd_A.placebo_depth_rel x fpr", "median over draws, x100 (points per level)"); same("xaSrPlac", "tOneSrPlacDepth")
put("xaSrPlacSizeRel", spct(srA("placebo_size_rel"), 1), SO, "splits[*].instr.crowd_A.placebo_size_rel", "median, x100"); same("xaSrPlacSizeRel", "qSrPSR")
put("xaSrRR", f(sr_rr[0]), SO, "splits[*].instr.crowd_A.log_rr", "exp of the median"); same("xaSrRR", "qSrRR")
put("xaSrSizeDist", f(abs(100 * srA("size_minus_goldB"))), SO, "splits[*].instr.crowd_A.size_minus_goldB", "|median| x100"); same("xaSrSizeDist", "qSrSizeDist")
for k_, nm, sc_ in (("depth", "Depth", 100), ("size", "Size", 100)):
    put(f"xaSrGoldB{nm}", sg(sc_ * med(lambda s_: s_["gold_B"][k_])), SO, f"splits[*].gold_B.{k_}", "median x100: the other raters' majority's law")
put("xaSrGoldBRR", f(float(np.exp(med(lambda s_: s_["gold_B"]["log_rr"])))), SO, "splits[*].gold_B.log_rr", "exp of the median")
same("xaSrGoldBDepth", "tOneGoldBDepth"); same("xaSrGoldBSize", "tOneGoldBSize"); same("xaSrGoldBRR", "tOneGoldBRR"); same("xaSrGoldBRR", "qGoldLooRR")
rd_ = med(lambda s_: abs(s_["instr"]["haiku"]["depth_minus_goldB"])) / med(lambda s_: abs(s_["instr"]["crowd_A"]["depth_minus_goldB"]))
rs_ = med(lambda s_: abs(s_["instr"]["haiku"]["size_minus_goldB"])) / med(lambda s_: abs(s_["instr"]["crowd_A"]["size_minus_goldB"]))
put("xaSrDepthRatio", f"{rd_:.0f}", SO, "median |haiku depth_minus_goldB| / median |crowd_A depth_minus_goldB|", "how many times the single rater's depth distortion Haiku's is")
put("xaSrSizeRatio", f"{rs_:.0f}", SO, "same for the size law", "ratio"); same("xaSrDepthRatio", "qSrDepthRatio"); same("xaSrSizeRatio", "qSrSizeRatio")
pdw_ = [abs(s_["instr"]["haiku"]["depth_minus_goldB"]) / abs(s_["instr"]["crowd_A"]["depth_minus_goldB"]) for s_ in SPL]
put("xaSrDepthRatioDrawMin", f"{min(pdw_):.0f}", SO, "per draw ratio", "smallest over draws"); put("xaSrDepthRatioDrawMax", f"{max(pdw_):.0f}", SO, "per draw ratio", "largest")
same("xaSrDepthRatioDrawMin", "qSrDepthRatioDrawMin"); same("xaSrDepthRatioDrawMax", "qSrDepthRatioDrawMax")
note("qSrDepthRatioCtx*, qSrDepthRatioMin/Max and qLLMPDRMin/Max (numbers_v4b) include Haiku and Qwen shown the parent with pooled values and are not reproduced.")
item("E03")
left_ = [(gl[p_] + DC[p_]["labelers"][k_]["uniform"]) / gl[p_] for p_, k_ in LLM4] + [(g0T + DC["cc_threads"]["labelers"][k_]["uniform"]) / g0T for k_ in CLS14]
put("xaUniLeftMin", pct(min(left_)), SD, "(gold_log_rr + uniform) / gold_log_rr", "smallest share of gold's log RR left by unstructured errors (4 LLMs, 14 classifiers)")
put("xaUniLeftMax", pct(max(left_)), SD, "same", "largest")
for a_, b_ in (("xaUniLeftMin", "qUniLeftMin"), ("xaUniLeftMax", "qUniLeftMax")):
    if M[a_] == EX[b_]:
        logchk(a_, f"= \\{b_} although the set differs (12 classifier settings and pooled with-parent LLMs there)")
    else:
        differs(a_, b_, "the earlier-draft range covers 12 classifier settings and the pooled with-parent LLMs; here the 4 LLMs without the parent and all 14 classifier settings")
put("xaGemmaFPLRR", sg(DC["cc_s5"]["labelers"]["gemma_m"]["fp"]), SD, "cc_s5.labelers.gemma_m.fp", "Gemma's S_FP for the log RR", caveat="5% sample"); same("xaGemmaFPLRR", "qFPGemma")
note(f"qFPShareDepthMin/Max ({plain(EX['qFPShareDepthMin'])}-{plain(EX['qFPShareDepthMax'])}, numbers_v4b) take the with-parent shares from pooled decompositions (x74_decomp, "
     f"x76_rethreshold); v7f uses the among-replies dFPShareMin/Max ({plain(EX['dFPShareMin'])}-{plain(EX['dFPShareMax'])}), reproduced as xaFPShareMin/Max. qFPHaikuCtx and "
     "qFPQwenCtx are pooled.")
item("E04")
for nm, pop, k_ in (("Haiku", "cc_threads", "haiku"), ("Qwen", "cc_threads", "qwen7b_m"), ("Sonnet", "cc_s5", "sonnet")):
    put(f"xa{nm}WithinLaw", sg(100 * (WI[pop]["gold_within"][0] + WI[pop]["labelers"][k_]["distortion"][0])), SW, f"within.{pop}.gold_within + labelers.{k_}.distortion",
        "x100: the labeler's within-thread depth law", caveat="5% sample" if pop == "cc_s5" else "")
    put(f"xa{nm}WithinZ", f"{abs(WI[pop]['labelers'][k_]['z']):.0f}", SW, f"within.{pop}.labelers.{k_}.z", "|z| of the within-thread distortion")
same("xaHaikuWithinLaw", "qHaikuWithinPP"); same("xaQwenWithinLaw", "qQwenWithinPP"); same("xaSonnetWithinLaw", "qSonnetWithinPP"); same("xaHaikuWithinZ", "qHaikuWithinZ")
same("xaQwenWithinZ", "qQwenWithinZ")
item("E08")
same("xaStrictExcess", "cCalPrevHaiku")
assert sg(vt["law"]) in ("0.00", "+0.00") and EX["wHaikuStrictDepth"] == "+0.00"; logchk("xaStrictLaw", "= \\wHaikuStrictDepth (+0.00, two decimals, printed with a sign there)")
same("xaStrictFlag", "wHaikuStrictFlag")
item("E10")
assert pct(un(M["xaDoseSrZero"]) / 100) == EX["cDoseSrZeroR"] and pct(un(M["xaDoseSrFour"]) / 100) == EX["cDoseSrFourR"]
logchk("xaDoseSrZero", "rounds to \\cDoseSrZeroR"); logchk("xaDoseSrFour", "rounds to \\cDoseSrFourR")
put("xaPlacYouShare", pct(1 - hz["within_any"][0] / hz["placebo"][0]), SDO, "1 - cc_threads.labelers.haiku.within_any / placebo", "share of Haiku's placebo slope that presence of 'you' carries")
same("xaPlacYouShare", "cPlacYouShare")
rcs = sorted((v[0] for v in rc.values()), reverse=True)
put("xaRatioClsSecond", f(rcs[1], 1), SDO, "cc_s5.matched.<classifiers>.ratio_4_0", "second largest classifier 4+:0 ratio", caveat="5% sample"); same("xaRatioClsSecond", "dClsRatioSecond")
put("xaModFlag", pct(np.mean([v["flag_rate"] for v in MTd.values()]), 1), SDO, "cc_s5.matched.*.flag_rate", "mean flag rate of the settings at one flag rate", caveat="5% sample")
same("xaModFlag", "cModFlag")
cr_ = load("x89/x89_cc_reference.json")
for nm, pop, k_ in (("Haiku", "cc_threads", "haiku_tox"), ("Sonnet", "cc_s5", "sonnet_tox"), ("Detox", "cc_threads", "ref:detoxify_original@m")):
    trip(f"xaYouGap{nm}", cr_[pop]["instruments"][k_]["fpr_you_minus_other"], SCR, f"{pop}.instruments.{k_}.fpr_you_minus_other",
         "x100: FPR on gold negatives with a second-person word minus the others' (points)", d=1, sc=100, signed=False, caveat="5% sample" if pop == "cc_s5" else "")
same("xaYouGapHaiku", "cYouGapHaiku"); same("xaYouGapSonnet", "cYouGapSonnet"); same("xaYouGapDetox", "cYouGapDetox")
for nm, pop, k_ in (("HaikuSfive", "cc_s5", "haiku_tox"), ("DetoxSfive", "cc_s5", "ref:detoxify_original@m")):                    # review round 2: like-for-like with Qwen on the 5% sample
    trip(f"xaYouGap{nm}", cr_[pop]["instruments"][k_]["fpr_you_minus_other"], SCR, f"{pop}.instruments.{k_}.fpr_you_minus_other",
         "x100: FPR on gold negatives with a second-person word minus the others' (points)", d=1, sc=100, signed=False, caveat="5% sample")
item("E11")
ib = CF["insult_by_haiku"]
put("xaInsFlagged", pct(ib["any_flagged"]), SCF, "insult_by_haiku.any_flagged", "share with an insult vote among gold negatives Haiku flags"); same("xaInsFlagged", "qFrInsFlag")
put("xaInsNotFlagged", pct(ib["any_not_flagged"]), SCF, "insult_by_haiku.any_not_flagged", "among those Haiku does not flag"); same("xaInsNotFlagged", "qFrInsNoFlag")
item("E13")
assert f(E74["h11"]["neg_deep"]["flag_E0"]) == EX["wEditOneBefore"] and f(E74["h11"]["neg_deep"]["flag_E1"]) == EX["wEditOneAfter"]
assert f(E74["rev"]["neg_deep"]["flag_E0"]) == EX["wEditRevBefore"] and f(E74["rev"]["neg_deep"]["flag_E2"]) == EX["wEditRevAfter"]
logchk("xaHaikuEditDeepBefore", "x74_edit's unpaired rate (1,500 comments) is \\wEditOneBefore = 0.29; 28.4\\% here is paired (1,489)")
logchk("xaHaikuRevDeepBefore", "= \\wEditRevBefore (0.10) as a proportion"); logchk("xaHaikuRevDeepAfter", "= \\wEditRevAfter (0.15) as a proportion")
put("xaParaN", th(I76["haiku"]["h11/neg_deep"]["E4_minus_E3"]["n"]), S76, "instruments.haiku.h11/neg_deep.E4_minus_E3.n", "paired comments of the paraphrase control"); same("xaParaN", "dParaN")
put("xaParaYouTwo", pct(I76["haiku"]["h11/neg_deep"]["E3"], 2), S76, "instruments.haiku.h11/neg_deep.E3", "two decimals"); same("xaParaYouTwo", "dParaYouP")
put("xaParaTheyTwo", pct(I76["haiku"]["h11/neg_deep"]["E4"], 2), S76, "instruments.haiku.h11/neg_deep.E4", "two decimals"); same("xaParaTheyTwo", "dParaTheyP")
item("E23")
r7 = RA["instruments"]["haiku_r"]["matched"]
put("xaRateHaikuPlacRel", spct(r7["placebo_depth_rel"][0], 1), SRA, "instruments.haiku_r.matched.placebo_depth_rel", "x100: Haiku-100's relative placebo depth slope", caveat="5% sample")
same("xaRateHaikuPlacRel", "qRateHaikuPDR")
for t_, nm in (("0.3", "Thirty"),):
    put(f"xaRateHaikuLaw{nm}", sg(100 * HR[t_]["haiku_r"]["depth"][0]), SHR, f"rates.{t_}.haiku_r.depth", "x100: Haiku-100's depth law flagging 30%", caveat="5% sample")
    put(f"xaRateSonnetLaw{nm}", sg(100 * HR[t_]["sonnet_r"]["depth"][0]), SHR, f"rates.{t_}.sonnet_r.depth", "x100: Sonnet-100's", caveat="5% sample")
same("xaRateHaikuLawThirty", "qHrHaikuThirty"); same("xaRateSonnetLawThirty", "qHrSonnetThirty"); same("xaVfLawThirty", "qHrHumanThirty")
put("xaVfLawMatched", sg(100 * HR["matched"]["human"]["depth"][0]), SHR, "rates.matched.human.depth", "x100: the instrument at gold's flag rate (= gold)", caveat="5% sample")
same("xaVfLawMatched", "qHrHumanMatched")
item("E24")
nat_ = sum(HL[f"kappa@{n_}"]["cut_counts"].get("2", 0) for n_ in (200, 500, 1000)) / 1500
put("xaCalNativeAll", pct(nat_), SC_, "...haiku:level.calibrated.kappa@{200,500,1000}.cut_counts['2']", "share of all 1,500 draws picking the own cut"); same("xaCalNativeAll", "cCalNative")
same("xaCalSonnetKmaxEx", "cRateSonnetEx")
mins_ = min(CA["cc_s5"]["instruments"][f"{m_}:rating"]["calibrated"][f"kappa@{n_}"]["share_excess_sig_pos"] for m_ in ("haiku", "sonnet") for n_ in (200, 500, 1000))
put("xaCalRateSigMin", pct(mins_ - 0.0005, 1), SC_, "...{haiku,sonnet}:rating.calibrated.kappa@*.share_excess_sig_pos", "smallest share of draws with a significant excess (as x18)",
    caveat="5% sample"); same("xaCalRateSigMin", "cRateSigMin")
item("E25")
for m_, nm, ex_ in (("haiku", "Haiku", "cRerunHaiku"), ("sonnet", "Sonnet", "cRerunSonnet")):
    put(f"xaRerun{nm}Law", sg(RR8[m_]["depth_new"][0]), SRR, f"{m_}.depth_new", "depth law of the later batch", caveat="5% sample"); same(f"xaRerun{nm}Law", ex_)
put("xaYrAgreeMin", pct(min(Y81["haiku"]["drift"]["agreement"], Y81["sonnet"]["drift"]["agreement"]), 1), S81Y, "models.*.drift.agreement", "agreement of the later batch with the stored labels (smaller of the two)")
same("xaYrAgreeMin", "qYrAgreeMin")
item("E28")
for k_, nm, ex_ in (("main", "Main", "qPrMain"), ("bin", "Bin", "qPrBin"), ("rev", "Rev", "qPrRev")):
    put(f"xaPr{nm}Law", sg(100 * AU[f"slope_{k_}0"][0]), S124, f"audit.slope_{k_}0", "x100: the labeler's depth law on that sample", caveat=CV124); same(f"xaPr{nm}Law", ex_)
item("E29")
for m_, nm in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
    trip(f"xaYpB{nm}Dist", Y80[m_]["depth_dist_you"], S80Y, f"models.{m_}.depth_dist_you", "x100: distortion with the note (B80e, cross-batch)", sc=100, caveat="5% sample")
trip("xaYpSonnetChg", Y80["sonnet"]["depth_change"], S80Y, "models.sonnet.depth_change", "x100: B80e change for Sonnet", sc=100, caveat="5% sample")
same("xaYpBHaikuDist", "qYpHaikuDist"); same("xaYpBSonnetDist", "qYpSonnetDist"); same3("xaYpSonnetChg", "qYpSonnetChg")
put("xaYrDeepMain", pct(Y81["haiku"]["civil_main"]["d6_you"], 1), S81Y, "models.haiku.civil_main.d6_you", "Haiku's flag rate on clearly civil 'you' comments at depth 6+, main prompt")
put("xaYrDeepNote", pct(Y81["haiku"]["civil_you"]["d6_you"], 1), S81Y, "models.haiku.civil_you.d6_you", "with the note"); same("xaYrDeepMain", "qYrDeepMain"); same("xaYrDeepNote", "qYrDeepYou")
item("E30")
for t_, nm in (("T0", "Zero"), ("T8", "Eight"), ("M1", "MitOne"), ("M2", "MitTwo"), ("TR", "Think"), ("T16", "Sixteen")):
    put(f"xaPrompt{nm}Law", sg(MI[t_]["depth"][0]), S91, f"instruments.{t_}.depth", "depth law under the prompt", caveat="5% sample")
    put(f"xaPrompt{nm}FprZero", pct(MI[t_]["fpr_by_count"][0][0]), S91, f"instruments.{t_}.fpr_by_count[0]", "FPR on gold negatives without 'you'", caveat="5% sample")
    put(f"xaPrompt{nm}FprFour", pct(MI[t_]["fpr_by_count"][4][0]), S91, f"instruments.{t_}.fpr_by_count[4]", "with four or more", caveat="5% sample")
for a_, b_ in (("xaPromptZeroLaw", "cMitZeroLaw"), ("xaPromptEightLaw", "cTunedLaw"), ("xaPromptMitOneLaw", "cMitOneLaw"), ("xaPromptMitTwoLaw", "cMitTwoLaw"),
               ("xaPromptZeroFprZero", "cMitZeroFprZero"), ("xaPromptZeroFprFour", "cMitZeroFprFour"), ("xaPromptMitOneFprZero", "cMitOneFprZero"), ("xaPromptMitOneFprFour", "cMitOneFprFour"),
               ("xaPromptMitTwoFprZero", "cMitTwoFprZero"), ("xaPromptMitTwoFprFour", "cMitTwoFprFour"), ("xaPromptEightDist", "cTunedDist")):
    same(a_, b_)
same3("xaPromptEightChg", "cTunedChg")
assert pct(MI["T8"]["flag_rate"]) == EX["dTunedFlag"]; logchk("xaPromptEightFlag", "rounds to \\dTunedFlag")
fr4 = [MI[k_]["flag_rate"] for k_ in ("T8", "TR", "M1", "M2")]
assert pct(min(fr4)) == EX["dPromptFlagMin"] and pct(max(fr4)) == EX["dPromptFlagMax"]; logchk("xaPromptMitOneFlag", "the variants of earlier draft's Table 2 flag \\dPromptFlagMin--\\dPromptFlagMax")
gaps_ = [100 * c_["fpr_you_minus_other"][0] for c_ in TU["candidates"].values()]
put("xaTunedGapMin", f(min(gaps_), 0), S89T, "candidates.*.fpr_you_minus_other", "smallest 'you' FPR gap of the four tuned prompts (points)", caveat="5% sample")
put("xaTunedGapMax", f(max(gaps_), 0), S89T, "same", "largest", caveat="5% sample"); same("xaTunedGapMin", "cTunedGapMin"); same("xaTunedGapMax", "cTunedGapMax")
put("xaTunedAllPos", str(sum(c_["distortion"][1] > 0 for c_ in TU["candidates"].values())), S89T, "candidates.*.distortion[1]", "tuned prompts whose distortion CI is above 0")
same("xaTunedAllPos", "cTunedAllPos")
item("E36")
for nm, m_ in (("Haiku", "haiku_rating"), ("Son", "sonnet_rating")):
    v_ = [BT[t_]["instruments"][m_]["depth_minus_gold"][0] for t_ in TT]
    put(f"xaThr{nm}RangeMin", sg(100 * min(v_)), S80T, f"pops.cc_s5.by_t.*.instruments.{m_}.depth_minus_gold", "smallest over t", caveat="5% sample")
    put(f"xaThr{nm}RangeMax", sg(100 * max(v_)), S80T, "same", "largest over t", caveat="5% sample")
same("xaThrHaikuRangeMin", "qThrHaikuMin"); same("xaThrHaikuRangeMax", "qThrHaikuMax"); same("xaThrSonRangeMin", "qThrSonMin"); same("xaThrSonRangeMax", "qThrSonMax")
put("xaThrTLo", f(min(TH["t"]), 1), S80T, "t", "lowest threshold"); put("xaThrTHi", f(max(TH["t"]), 1), S80T, "t", "highest"); same("xaThrTLo", "qThrTLo"); same("xaThrTHi", "qThrTHi")
note("Not reproduced (pooled with-parent values or other agents' items): qQwenOwnDepthPP, qQwenOwnFPShare, qQwenOwnLRR, qQwenOwnPDR, qQwenOwnWithinPP (x76_rethreshold, "
     "pooled); qRateHaikuCtx*, qRateSonnetCtx*, qHrHaikuCtxMatched, wHaikuCtxDepthPP, cSonnetCtxDepth, qSonnetCtxWithinPP, qHaikuCtxWithinPP, qQwenCtxWithinPP (pooled); "
     "wHaikuMatched* (random thinning, keep-out K12); qMatchNLLM (counts pooled settings); qPrBinCtx* (pooled); the contagion, unit-share and census macros belong to agents B and C.")


# ================================================================ cross-check: the earlier draft's Table 3 rows are reproduced by xa_oppoints
item("E22")
T3 = {r_[0]: r_ for r_ in _cells("llms_v6.tex")}
OPd = {}
_op = [l_ for l_ in TABLES["xa_oppoints"]["rows"] if " & " in l_]
_opc = [[c_.strip() for c_ in l_[:-2].split("&")] for l_ in _op]
_cur = None
for c_ in _opc:
    _cur = c_[0] or _cur; OPd[(_cur, c_[1])] = c_
for nm, key in (("Haiku", ("Haiku", "own cut (max $\\kappa$)")), ("Haiku-100", ("Haiku-100", "gold's rate")), ("Sonnet", ("Sonnet", "own cut (max $\\kappa$)")),
                ("Sonnet-100", ("Sonnet-100", "gold's rate")), ("Qwen", ("Qwen", "gold's rate (prevalence)")), ("Gemma", ("Gemma", "gold's rate (prevalence)")),
                ("OLMo-2", ("OLMo-2", "gold's rate (prevalence)")), ("Granite", ("Granite", "gold's rate (prevalence)")), ("Llama", ("Llama", "gold's rate (prevalence)"))):
    t3, op = T3[nm], OPd[key]
    assert t3[1] == op[2] and t3[2] == op[3] and t3[4] == op[4] and op[6].startswith(t3[5] + " [") and op[8].startswith(t3[6] + " [") and op[9].split(" [")[0] == t3[3], (nm, t3, op)


# ================================================================ outputs: the macros and the provenance
SKIPPED = [
    ("E03", "Decomposition of the thread size, article size and contagion laws for labelers shown the parent", "x97 decomposes only the depth law among replies; the stored size "
     "laws and log RR of these labelers pool roots labeled without the parent with replies labeled with it. Not computed; shown as -- in xa_decomp_terms."),
    ("E04", "Within-thread distortions of Haiku and Qwen shown the parent (inventory: +1.70, +2.66)", "pooled (the root of each thread is labeled without the parent)."),
    ("E18", "Pooled with-parent rewrite distortions (+2.24 -> +1.69 -> +1.56)", "pooled; the among-replies values are reported instead."),
    ("E23", "The 14 with-parent points of the 0-100 grid and the parent variants of the figure", "pooled in x75_rating_eval.json (flags over all comments, roots keep their "
     "no-context rating); dropped from the figure and from the grid ranges."),
    ("E26", f"Reply risk ratios of the LLMs shown the parent ({EX['cCtxRROpenMin']}-{EX['cCtxRROpenMax']} for the open models, {EX['qCtxRRMin']}-{EX['qCtxRRMax']} for Haiku and Qwen; v7f's 0.87-1.64)", "pooled over edges whose parent is a root labeled without the parent; no "
     "among-replies (reply-parent edge) version is stored."),
    ("E28", "The with-parent prompt comparison (yes/no prompt flattens the slope, " + plain(ci3(AU["c_slope_bin1_minus_bin0"], 2, 100)) + "; C13)", "pooled (f124's P1 keeps roots at P0)."),
    ("E35", f"Matched human instruments for Haiku, Qwen and Sonnet shown the parent (lower ends {sg(100 * _h80['haiku_ctx'])}, {sg(100 * _h80['qwen7b_m_ctx@own'])}, {sg(100 * _h80['sonnet_ctx'])})", "pooled."),
    ("E02", "Single-rater comparisons with Haiku and Qwen shown the parent (20/20 draws)", "pooled and, for Qwen, at the old reused threshold; the 20/20 check is redone against Haiku only."),
    ("E32", "Qwen on CCC (x74_ccc qwen rows)", f"thresholded at the out-of-context prevalence and reused for the in-context scores ({plain(pct(min(_qr)))}-{plain(pct(max(_qr)))} flagged by depth), the superseded convention."),
    ("E32", "kappa of Haiku against CCC's in-context raters", "not stored (the CCC CSV is not in data/); the row shows the flag rate only."),
    ("E11", "Slopes of Haiku shown the parent in x85_construct_frac / x87_gap (haiku_ctx)", "pooled."),
    ("E36", "Classifier, Qwen and Gemma depth distortions against gold_t (also in x80_threshold.json)", "not tabulated: the assignment covers the Haiku and Sonnet ratings; the values are stored if wanted."),
]
INV_V7F = [
    "v7f, Section 3 ('The word you'): 'among the 75% of gold negatives on which no rater voted toxic or marked a toxicity subtype'. \\xaFracNoneShare{} is the share among gold negatives "
    "matched to the public release (x85, \\xaFracMatch{} matched); the dose statistics quoted for this subset (\\xaDoseNoVoteZero{} -> \\xaDoseNoVoteFour) are computed in x90 "
    "on the same comments, which are \\xaNoVoteShare{} of all gold negatives. Suggest 'the \\xaNoVoteShare{} of gold negatives' or 'three quarters'.",
    "v7f, Section 3 ('The size law and contagion') and Figure 2: 'shown the parent ... their risk ratios fall to 0.87--1.64', and the contagion bars of the '+ parent' rows "
    "in Figure 2, rest on pooled values: every reply-edge whose parent is a root pairs a no-context label (the root) with a with-context label (the reply). No among-replies "
    "contagion is stored. Figure 2's caption says only that their size law is not shown. The claim should be caveated or recomputed on edges whose parent is itself a reply.",
    "Inventory E03: the contagion ranges in numbers_v4b (qUni*, qFP*, qTP*, qUniRR*) cover 12 classifier settings (Detoxify-unbiased missing) and include the pooled "
    "with-parent LLMs (and qFPLLM* leaves Gemma out). Over the 4 LLMs without the parent and all 14 classifier settings: unstructured cut \\xaUniLLMMin{} to \\xaUniLLMMax{} "
    "(LLMs) and \\xaUniClsMin{} to \\xaUniClsMax{} (classifiers); S_FP \\xaFPLLMMin{} to \\xaFPLLMMax{} and \\xaFPClsMin{} to \\xaFPClsMax; S_TP \\xaTPLLMMin{} to "
    "\\xaTPLLMMax{} and \\xaTPClsMin{} to \\xaTPClsMax; risk ratio left by unstructured errors \\xaUniRRMin--\\xaUniRRMax{} (inventory: 1.02-1.66).",
    "Inventory E02: '20 of 20 draws ... below Haiku's, Haiku shown the parent's and Qwen shown the parent's' uses pooled with-parent values (and Qwen's old threshold); "
    "re-verified for Haiku alone (\\xaSrBelowHaikuDraws{} of 20).",
    "Inventory E04: 'Haiku shown the parent +1.70, Qwen shown the parent +2.66' within threads are pooled values (not usable).",
    f"Inventory E16: 'within-thread distortion removed entirely (+0.75 -> -0.00)': the stored share removed is {L77['depth_within']['share_removed']:.3f} (printed \\xaYfWithinShare, as qYfWithinShare); fine as 'all of it'.",
    "Inventory E18: '+2.24 (V0) -> +1.69 (V1) -> +1.56 (V2)' are pooled; among replies the distortion is \\xaRwCtxHaikuOrig{} -> \\xaRwCtxHaikuReply{} -> \\xaRwCtxHaikuBoth{} (the \\xaRwCtxHaikuShareReply{} and \\xaRwCtxHaikuShareBoth{} shares are among replies).",
    "Inventory E23: '28 operating points ... with and without the parent' and 'At gold prevalence: Haiku +0.25 (shown the parent +0.70), Sonnet +0.31': the with-parent "
    "points are pooled, and +0.25/+0.31 are laws, not distortions. The distortions at gold's flag rate are \\xaRateHaikuDist/\\xaRateSonnetDist{} (x75, seed 7500) or \\xaHaikuHundredPrevDist/\\xaSonnetHundredPrevDist{} (x92, seed 9010); "
    "14 of 14 points without the parent have CIs above 0 (\\xaGridMin{} to \\xaGridMax).",
    "numbers_v4b: qRateHaikuDepthPP (\\xaRateHaikuLaw) is the law at gold's flag rate while qRateHaikuDepthLo (\\xaRateHaikuDistLo) is the lower CI end of the distortion (\\xaRateHaikuDist): the pair mixes two "
    "quantities. Not used in v7f.",
    "Inventory E31: 'At a fixed number: M1 +0.34, M2 +0.14, T8 +0.29, TR +0.65' mixes levels (M1, M2) with changes against T0 (T8, TR). Levels: M1 \\xaPromptMitOneFix, "
    "M2 \\xaPromptMitTwoFix, T8 \\xaPromptEightFix, TR \\xaPromptThinkFix; changes: \\xaPromptMitOneFixChg, \\xaPromptMitTwoFixChg, \\xaPromptEightFixChg, \\xaPromptThinkFixChg.",
    "Inventory E15 / qClsRevMin-Max: 'adding you moves the classifiers' flag rates by -0.3 to +1.3 points' covers the 7 settings at gold's rate; over all 14 settings it is "
    "\\xaClsAddMin{} to \\xaClsAddMax{} (IMSYPP at argmax, which flags a third of comments). Removing: \\xaClsRemMin{} to \\xaClsRemMax{} over 14 (unchanged).",
    "Inventory E26, E35, E28 (C13): the with-parent risk ratios, matched-instrument comparisons and prompt comparison are pooled (see Skipped).",
    "Inventory E13: Haiku's reverse rewrite '+4.8 [3.7, 6.0]' is x74_edit's bootstrap; x76_edit's, used here for one bootstrap across all instruments, gives [\\xaHaikuRevDeepDiffLo, \\xaHaikuRevDeepDiffHi].",
    f"earlier-draft macro cTunedDist ({plain(EX['cTunedDist'])} [{plain(EX['cTunedDistLo'])}, {plain(EX['cTunedDistHi'])}], x89_tuned) and x91_mitigate's interval for the same prompt "
    "(\\xaPromptEightDist{} [\\xaPromptEightDistLo, \\xaPromptEightDistHi]) are two bootstraps; xa_esc_full uses x91 so "
    "that each prompt row and its paired change share one bootstrap.",
    "v7f Table 2 caption: 'the prompt variants flag 20%--24% of comments' covers the variants shown; the fourth tuned prompt (sixteen examples), added in xa_esc_full, flags "
    "\\xaPromptSixteenFlag.",
    f"Qwen and Gemma 'at gold's flag rate' are the stored matched thresholds and flag {pct(SET['Qwen']['flag'], 2)} (Qwen, all reply threads; {pct(ROWS92[('qwen', 'gold prevalence')]['flag_rate'], 2)} on the 5% sample) and "
    f"{pct(SET['Gemma']['flag'], 2)} (Gemma) against gold's {pct(zT.mean(), 2)}/{pct(zS.mean(), 2)}; the open models from B84/B93 flag exactly gold's share.",
    "The 0-100 ratings at gold's flag rate exist with two tie-breaking draws: x75_rating (seed 7500; \\xaRateHaikuDist{} / \\xaRateSonnetDist) and x92 (seed 9010; \\xaHaikuHundredPrevDist{} / \\xaSonnetHundredPrevDist, used in Table 3 and "
    "Table 2). xa_oppoints's within-thread values for the ratings come from x77_humanrate, which uses the seed-7500 flags (marked).",
]
with open(OUT_NUM, "w") as fo:
    fo.write("% generated by scripts/x98a_escalation.py (agent A: news-comment escalation, mechanism, causal edits, operating points, prompts, human readings); do not edit\n")
    for k_, v_ in M.items():
        fo.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
# provenance
def md(s_):
    return str(s_).replace("|", "\\|").replace("\n", " ")
def expand(s_):                                                                    # macro names in prose -> their values
    return re.sub(r"\\(xa[A-Za-z]+)(\{\})?", lambda m_: plain(M[m_.group(1)]), s_)
L = []
L.append("# Provenance: numbers_arxiv_a.tex, tables/xa_*.tex, figs/oppoint_grid.pdf (agent A)\n")
L.append(f"Generated by `{SCRIPT}` (`python3 -I scripts/x98a_escalation.py`, about two minutes; no network, no API calls). Rerunning rewrites every output; nothing is "
         f"edited by hand. Inputs are stored results under `results/` (listed per macro), the label parquets `results/x14_pops/cc_threads.parquet` ({NT:,} comments, "
         f"{int(repT.sum()):,} reply edges) and `cc_s5.parquet` ({NS:,}), and the 5% sample's texts `data/second/s5/` (only to find second-person words).\n")
L.append("**Population.** Every number is on the current population (cc_threads, its 5% sample, CCC on the train-split structure). Two f-series files are used, each "
         f"checked: `results/f81_you_edit.json` (E12; the same {F81A['n']:,} comments as x74_edit, train-split strata) and `results/f124_prompts.json` (E28; its slope sample of {PRN['n_slope']:,} "
         f"comments contains all {PRN['n_slope_in_cc_s5']:,} of the 5% sample, `results/x81/prn.json`). No `numbers_v2.tex` value is used or checked against.\n")
L.append("**Pooled with-parent values are never used.** Where a labeler is shown the parent, values are among replies (x97, x89_cc_reference depth_replies, x90 "
         "replies, x89_rewrite dist_replies, x84 depth_replies_excess, recomputed flag rates); everything else is listed under Skipped.\n")
L.append("**Formatting.** As `x18_numbers_v5.py`: signed values `+0.93`, `$-$0.12` (no signed zero), percentages `67\\%`, thousands `1{,}251{,}487`. Each 95% interval is "
         "a pair of macros `...Lo`/`...Hi` (2.5th and 97.5th percentiles of the stored resamples). Depth laws and distortions in points per level, size laws per "
         "doubling, flag-rate changes in points. Relative placebo slopes (`...PlacRel`) carry no percent sign.\n")
L.append(f"**Counts.** {len(M)} macros; {len(TABLES)} table bodies ({', '.join(sorted(TABLES))}); 1 figure.\n")
L.append("**Re-analyses of stored labels (deterministic).** (1) Flag rate, kappa, TPR, FPR among replies and of every setting (xa_settings, xa_esc_full); (2) the open "
         "models' flags with and without the parent (x84_families.py's rule, seed 8400; reproduces x84's stored flag rate, kappa, TPR and FPR exactly); (3) Qwen and Gemma "
         "shown the parent at their own threshold (x76_rethreshold.py's rule, seed 7650; reproduces x97's TPR and FPR exactly); (4) the 0-100 ratings at gold's flag rate "
         "(x92_matched.py, seed 9010; reproduces x92's kappa and FPR); (5) Sonnet's corpus-wide rewrite, x89_rewrite.py analysis (i) with its resamples (seed 8905), which "
         "reproduces every stored value to 1e-9 and adds the paired within-thread and size changes and the gold-positive flag rates; (6) the both-address FPRs of the rewrite "
         "shown the parent (reproduces Haiku's values in the registration log: \\xaBothHaikuOneOrig at depth 1 and \\xaBothHaikuFiveOrig at depth 5+ as written, "
         "\\xaBothHaikuOneBoth and \\xaBothHaikuFiveBoth with both rewritten); (7) kappa of the later batches (x81 answers); (8) the Spearman correlation over all 16 "
         "instruments (E15); (9) Qwen without the parent among replies (x96_dist_v6.py's rule).\n")
L.append("\n## Discrepancies with the earlier-draft macros (automatic: every macro whose value differs from an existing macro holding a related number)\n")
L.append("| Macro | Value | earlier-draft macro | earlier-draft value | Why |\n|---|---|---|---|---|")
for d_ in DISC:
    L.append(f"| `\\{d_['name']}` | `{md(d_['value'])}` | `\\{d_['ex']}` ({d_['exf']}) | `{md(d_['exv'])}` | {md(d_['why'])} |")
L.append("\n## Discrepancies with the inventory and with v7f\n")
for s_ in INV_V7F:
    L.append(f"- {md(expand(s_))}")
L.append("\n## Skipped, not computable, or suspect\n")
_ino = lambda t_: int(re.sub(r"\D", "", t_) or 0)                                # item order E01 < E02 < ... (stable within an item)
for it_, what, why in sorted(SKIPPED, key=lambda r_: _ino(r_[0])):
    L.append(f"- **{it_}.** {md(what)}: {md(why)}")
L.append("\n## Notes from the script\n")
for it_, t_ in sorted(NOTES, key=lambda r_: _ino(r_[0])):
    L.append(f"- **{it_}.** {md(expand(t_))}")
TCHK = {
    "xa_laws_ci": ("Every point value equals the earlier draft's Table 1 (paper_v5/tables/laws_v6.tex) and Table 7 (classifiers_v7.tex) cells (asserted), except Qwen's own-cut row, "
                   "which has no earlier-draft counterpart (x74_placebo cc_threads.labelers.qwen7b, as Figure 4's Qwen own-cut point); gold's interval equals \\wGoldDepthPPLo/Hi.", f"One rater: the bracket is the range over the {len(SPL)} draws (only draw 0 stores bootstrap intervals). Sonnet and Gemma on the 5% sample; "
                   "their size laws are not shown, as in Table 1."),
    "xa_laws_main": ("Rows of xa_laws_ci (Gold, One rater, the four LLMs with Qwen also at its own cut, Detoxify and IMSYPP at both cuts), cells unchanged (asserted by construction).", "As xa_laws_ci."),
    "xa_laws_cls": ("The 14 classifier rows of xa_laws_ci, cells unchanged (family order asserted).", "As xa_laws_ci."),
    "xa_decomp_terms": ("The three terms add up to the total within 1e-9 for every row (asserted); totals equal Table 1's distortions; the FP shares reproduce "
                        "\\dFPShareMin/Max; Haiku's and Gemma's log-RR S_FP equal \\qFPHaiku and \\qFPGemma.", "+ parent rows: depth law among replies only (x97); "
                        "contagion terms are the permutation baseline and Shapley values (50 permutations); x74_decomp's two-path split (K10) is not used."),
    "xa_settings": ("Haiku, Sonnet, Qwen, Gemma and the 14 classifier rows equal x74_placebo's flag rate, kappa, TPR and FPR to 1e-12; the recomputed open-model flags "
                    "reproduce x84's stored statistics, Qwen's and Gemma's with-parent flags reproduce x97's TPR and FPR and x76's kappa, and the rating flags reproduce "
                    "x92's kappa and FPR (all asserted); One rater kappa and FPR = \\qSrKappa, \\qSrFPR.", "+ parent rows are evaluated on replies; the single rater is "
                    "judged against the other raters' majority and its TPR is derived from its flag rate, FPR and that majority's prevalence."),
    "xa_esc_full": ("Every distortion and human law of the earlier draft's Table 2 is reproduced with the same rounding (asserted against the d*, w*, q* and c* macros); flag rates and "
                    "kappa of the main rows equal Tables 1 and 3.", "Each row's interval comes from its own source's bootstrap. Rows for other readings give the LLM's law "
                    "minus that reading; double-dagger rows are medians over calibration draws with 5th-95th percentiles; the noise-free share is model-based; kappa against "
                    "CCC's raters is not stored; the separate sample is an f-series file containing the 5% sample; the earlier-wording note was labeled with its own rerun of "
                    "the main prompt."),
    "xa_rewrite": ("Haiku's corpus-wide values equal \\qYf*; Sonnet's equal \\cRwSon*; the edit-sample values equal \\qGap*, \\qSonGap*, \\qEdit*, \\qRev*, "
                   "\\wEditPara* and \\qSonPara*/\\qSonRev* (signed here); the dose values round to \\cRwDose*; the with-parent shares equal \\cRwCtxShareH/S; the replication "
                   "of x89_rewrite.py analysis (i) reproduces every stored value; the both-address rows reproduce Haiku's values in the registration log (\\xaBothHaikuOneOrig -> \\xaBothHaikuFiveOrig as written, \\xaBothHaikuOneBoth -> "
                   "\\xaBothHaikuFiveBoth with both rewritten, depth 1 against 5+).",
                   "5% sample and the edit sample; the rewrite also lowers flags on gold positives (Limitations); no human judged the edited comments; the both-address rows "
                   "are a recomputed descriptive breakdown."),
    "xa_rewrite_cls": ("The 7-setting ranges equal \\qClsRevMin/Max and \\qClsYouMin/Max; the 9-instrument Spearman equals \\qYouRho; the placebo slopes equal "
                       "x76_edit's stored ones (asserted).", f"Relative changes rest on few flags where the base rate is below 1% (Detoxify at 0.6: {round(add['detoxify_original@0.6']['E0'] * add['detoxify_original@0.6']['n'])} "
                       f"of {add['detoxify_original@0.6']['n']:,} comments); "
                       "IMSYPP at argmax flags a third of comments; the 16-instrument Spearman is computed here."),
    "xa_pairs_three": ("The first three rows of xa_pairs, unchanged (the pairs are sorted by length).", "As xa_pairs."),
    "xa_pairs": ("\\xaExPairs{} pairs and \\xaExShort{} short ones (= \\cExPairs, \\cExShort); automatic checks for handles, e-mail and web addresses and an allow-list of every capitalized word "
                 "in the printed pairs; the redacted private name of the earlier-draft table is absent (asserted).", f"Pairs with a name, initials or a handle are left out ({len(NAMED)} among "
                 f"those up to {LMAX} characters, all of which were read); some kept pairs are rude although no rater called them toxic, and the rule rewrite is ungrammatical in places."),
    "xa_oppoints": ("The rows at gold's flag rate and the own cuts reproduce the earlier draft's Table 3 (llms_v6.tex) cells (asserted); the open models' max-kappa range equals "
                    "\\cMatOpenKmaxMin/Max.", "5% sample. Within-thread values of the ratings come from flags with a different tie-breaking draw (marked); double-dagger rows "
                    "are medians over calibration draws; Sonnet's own-cut 4+:0 ratio has no stored interval."),
    "xa_thresholds_depth": ("The 12 rating distortions span \\qThrLLMMin to \\qThrLLMMax; per model \\qThrHaikuMin/Max and \\qThrSonMin/Max; the instrument's law at 30% "
                            "equals \\qHrHumanThirty (asserted).", "5% sample; the bottom panel's ratings use seed-7500 tie-breaking; its 7.5% row approximates gold's "
                            f"flag rate ({plain(pct(zS.mean(), 2))})."),
}
assert sorted(TCHK) == sorted(TABLES)
L.append("\n## Tables\n")
for nm, T_ in TABLES.items():
    L.append(f"### tables/{nm}.tex ({sum(1 for r_ in T_['rows'] if ' & ' in r_)} data rows)\n")
    L.append(f"- Source: {md(T_['src'])}.")
    L.append(f"- Proposed caption: {md(T_['caption'])}")
    L.append(f"- Proposed column headers: {md(T_['headers'])}.")
    L.append(f"- Transformation: every cell is formatted by the script from the source (helpers sg/f/pct as x18_numbers_v5.py); no hand-typed number.")
    L.append(f"- Cross-checks: {md(expand(TCHK[nm][0]))}")
    L.append(f"- Caveats: {md(expand(TCHK[nm][1]))}\n")
L.append("### figs/oppoint_grid.pdf\n")
L.append(f"- Source: {SRA}, instruments.{{haiku_r,sonnet_r}}.<0.02 ... 0.3>.depth_minus_gold (x100), flag_rate; dashed line at gold's flag rate ({100 * RA['prev']:.2f}%).")
L.append("- Transformation: points with 95% intervals from 500 article-cluster resamples; log x axis; Haiku-100 orange circles, Sonnet-100 aqua squares (the colors of dose.pdf; "
         "palette validated with the dataviz validator: CVD separation 9.2). 3.4 x 2.1 in, 7 pt text.")
L.append("- The parent variants are dropped: x75_rating_eval.py flags them over all comments with roots keeping their no-context rating (pooled).")
L.append("- Proposed caption: Distortion of the depth law (the rating's law minus gold's, points per level, 95% intervals) when Haiku's and Sonnet's 0-100 ratings flag "
         "from 2% to 30% of the comments of the 5% sample; the dashed line marks gold's flag rate. Every point's interval is above 0.\n")
L.append("\n## Macros by item\n")
for it_ in sorted({p_["item"] for p_ in PROV}, key=lambda s_: (s_[0], int(re.sub(r"\D", "", s_) or 0))):
    L.append(f"\n### {it_}\n")
    L.append("| Macro | Value | Source | Key | Transformation | Check | Caveat |\n|---|---|---|---|---|---|---|")
    for p_ in PROV:
        if p_["item"] == it_:
            L.append(f"| `\\{p_['name']}` | `{md(M[p_['name']])}` | {md(p_['src'])} | {md(p_['key'])} | {md(p_['how'])} | {md(p_['check'])} | {md(p_['caveat'])} |")
open(OUT_PROV, "w").write("\n".join(L) + "\n")
print(len(M), "macros ->", OUT_NUM); print(len(TABLES), "tables ->", TAB, sorted(TABLES)); print("figure ->", FIGPDF); print("provenance ->", OUT_PROV)
print(len(DISC), "documented differences from earlier-draft macros;", len(NOTES), "notes")
