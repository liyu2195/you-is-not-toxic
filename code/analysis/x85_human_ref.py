"""Addendum 85, B85b: a human reference for the distortion summaries (descriptive; Civil Comments reply threads, cc_threads).
Part 1, single rater (stored files). The 20 single-rater draws of B75b (results/x75_onerater/x75_onerater.json; rater A one of the
comment's raters, gold B the majority of the others) for the depth, thread size and article size laws and the log reply RR:
  - the share of estimates off by more than a quarter of gold B's law (|A - B| / |B| > 0.25);
  - the share with |z| > 3, z = distortion / (CI width / 3.92), the CI width being draw 0's bootstrap width of the same law's
    distortion (500 article-cluster resamples; the width hardly depends on the draw);
  - the uniform share of the distortion (x75_onerater_decomp.json; Proposition 1 terms for depth and thread size, the permutation
    terms for the log RR): uniform / distortion and |uniform| / (|uniform| + |S_FP| + |S_TP|);
  - the same summaries for the instruments: (i) against gold B in the same 20 draws (same file), and (ii) the B82c rows of cc_threads
    against full gold (results/x74_decomp), overall, for kappa >= 0.6 and for the LLMs.
Part 2, decisions at a realistic sample size. 200 article-cluster subsamples of about 20,000 comments each (seed 8502; clusters in
random order, taken until 20,000 comments are reached). In each, gold's depth law, thread size law and log reply RR are classified as
significantly positive, significantly negative or not significant (Wald, |t| > 1.96, article-clustered sandwich SE: OLS influence
functions summed by cluster; the log RR by the delta method), and so is each instrument's law. Reported: the share of subsamples in
which the instrument's conclusion differs from gold's (and how: reversal, manufactured, lost), for every setting, pooled over the
kappa >= 0.6 settings and over the LLMs; and for the single rater (one draw, seed 8503: rater A against the majority B of the
others, both in the same subsample), together with how often gold's own conclusions differ between gold and gold B.
usage: python3 x85_human_ref.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NSUB = 200; TARGET = 20000
LAWS = ["depth", "size", "article_size", "log_rr"]; LLM = ("haiku", "sonnet", "qwen", "gemma")
res = {"part1": {}, "part2": {}}
# ======================================================================== part 1
O = json.load(open(f"{W}/results/x75_onerater/x75_onerater.json")); DC = json.load(open(f"{W}/results/x75_onerater/x75_onerater_decomp.json"))
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json"))["cc_threads"]
w0 = {law: O["splits"][0]["instr"]["crowd_A"][f"{law}_minus_goldB_ci"] for law in LAWS}
w0 = {law: v[1] - v[0] for law, v in w0.items()}
w0_inst = {L: {law: O["splits"][0]["instr"][L][f"{law}_minus_goldB_ci"][1] - O["splits"][0]["instr"][L][f"{law}_minus_goldB_ci"][0] for law in LAWS} for L in O["instruments"]}


def summ_rows(R):
    if not R:
        return None
    rel = np.array([r["rel"] for r in R]); z = np.array([r["z"] for r in R])
    out = dict(n=len(R), share_rel_gt_25=float(np.mean(rel > 0.25)), median_rel=float(np.median(rel)), share_absz_gt_3=float(np.mean(np.abs(z) > 3)),
               median_absz=float(np.median(np.abs(z))))
    out["by_law"] = {law: dict(n=sum(r["law"] == law for r in R), share_rel_gt_25=float(np.mean([r["rel"] > 0.25 for r in R if r["law"] == law])),
                               median_rel=float(np.median([r["rel"] for r in R if r["law"] == law])),
                               share_absz_gt_3=float(np.mean([abs(r["z"]) > 3 for r in R if r["law"] == law]))) for law in sorted({r["law"] for r in R})}
    return out


rows_A, rows_I = [], []
for s, S in enumerate(O["splits"]):
    gB = S["gold_B"]
    for name, o in S["instr"].items():
        for law in LAWS:
            d = o[f"{law}_minus_goldB"]; g = gB[law]
            width = w0[law] if name == "crowd_A" else w0_inst[name][law]
            r = dict(draw=s, inst=name, law=law, dist=d, gold=g, rel=abs(d) / abs(g) if g != 0 else np.inf, z=d / (width / 3.92))
            (rows_A if name == "crowd_A" else rows_I).append(r)
kap = {L: DEC["labelers"][L]["kappa"][0] for L in DEC["labelers"] if L != "human_lenient"}
res["part1"]["single_rater"] = summ_rows(rows_A)
res["part1"]["single_rater"]["kappa_vs_goldB"] = dict(mean=float(np.mean([S["kappa_A_B"] for S in O["splits"]])), min=float(min(S["kappa_A_B"] for S in O["splits"])),
                                                       max=float(max(S["kappa_A_B"] for S in O["splits"])))
res["part1"]["instruments_vs_goldB"] = {"all": summ_rows(rows_I), "kappa_ge_06": summ_rows([r for r in rows_I if kap.get(r["inst"], 0) >= 0.6]),
                                         "llm": summ_rows([r for r in rows_I if r["inst"].startswith(LLM)]),
                                         "classifier": summ_rows([r for r in rows_I if not r["inst"].startswith(LLM)])}
# B82c rows of cc_threads (against full gold)
g = DEC["gold"]; rows_F = []; PERM = json.load(open(f"{W}/results/x75_decomp/x75_decomp.json"))["cc_threads"]["labelers"]   # log-RR terms (B75c)
for L, r in DEC["labelers"].items():
    if L == "human_lenient":
        continue
    for law, v in r["laws"].items():
        gl = g[law]; il = v["law"]
        if law == "log_rr":
            pm = PERM.get(L, {}); u_, fp_, tp_ = pm.get("uniform", np.nan), pm.get("fp", np.nan), pm.get("tp", np.nan)
        else:
            u_, fp_, tp_ = v["attenuation_term"][0] - v["gold"][0], v["fp_term"][0], v["tp_term"][0]
        rows_F.append(dict(inst=L, law=law, dist=il[0] - gl[0], gold=gl[0], rel=abs(il[0] - gl[0]) / abs(gl[0]) if gl[0] != 0 else np.inf, z=v["z"],
                           uniform=u_, fp=fp_, tp=tp_, kappa=kap[L]))
res["part1"]["instruments_vs_gold_B82c"] = {"all": summ_rows(rows_F), "kappa_ge_06": summ_rows([r for r in rows_F if r["kappa"] >= 0.6]),
                                             "llm": summ_rows([r for r in rows_F if r["inst"].startswith(LLM)]), "kappa_ge_06_settings": sorted({r["inst"] for r in rows_F if r["kappa"] >= 0.6})}
# uniform share
def ushare(u, fp, tp, d):
    return dict(ratio=u / d if d != 0 else None, mass=abs(u) / (abs(u) + abs(fp) + abs(tp)) if (abs(u) + abs(fp) + abs(tp)) > 0 else None)
US = {}
for law in ("depth", "size", "log_rr"):
    v = [ushare(D[law]["uniform"], D[law]["fp"], D[law]["tp"], D[law]["distortion"]) for D in DC["draws"]]
    US[law] = dict(median_ratio=float(np.median([x["ratio"] for x in v])), median_mass=float(np.median([x["mass"] for x in v])),
                   min_mass=float(min(x["mass"] for x in v)), max_mass=float(max(x["mass"] for x in v)))
res["part1"]["single_rater_uniform_share"] = US
UI = {}
for law in ("depth", "size", "log_rr"):
    v = [(r["inst"], ushare(r["uniform"], r["fp"], r["tp"], r["dist"])) for r in rows_F if r["law"] == law and np.isfinite(r["uniform"])]
    UI[law] = dict(median_mass=float(np.median([x[1]["mass"] for x in v])), median_mass_kappa_ge_06=float(np.median([x[1]["mass"] for x in v if kap[x[0]] >= 0.6])) if any(kap[x[0]] >= 0.6 for x in v) else None,
                   median_mass_llm=float(np.median([x[1]["mass"] for x in v if x[0].startswith(LLM)])), by_inst={x[0]: x[1]["mass"] for x in v})
res["part1"]["instrument_uniform_share"] = UI
print("part 1 single rater:", json.dumps({k: v for k, v in res["part1"]["single_rater"].items() if k != "by_law"}), flush=True)
print("part 1 instruments vs gold B:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_law"} for k, v in res["part1"]["instruments_vs_goldB"].items() if v}), flush=True)
print("part 1 B82c cc_threads:", json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "by_law"} if isinstance(v, dict) else v) for k, v in res["part1"]["instruments_vs_gold_B82c"].items()}), flush=True)
print("uniform shares: single rater", US, "| instruments", {k: {kk: vv for kk, vv in v.items() if kk != "by_inst"} for k, v in UI.items()}, flush=True)
# ======================================================================== part 2
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); N = len(z)
cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); csize = np.bincount(cl, minlength=nC)
un = pd.factorize(P.unit.values)[0]; ku = np.bincount(un); ucl = np.zeros(ku.size, np.int64); ucl[un] = cl; lxu = np.log2(np.maximum(ku, 1)).astype(float)
ar = pd.factorize(P.article.values)[0]; ka = np.bincount(ar); acl = np.zeros(ka.size, np.int64); acl[ar] = cl; lxa = np.log2(np.maximum(ka, 1)).astype(float)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
INS = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    INS[f"{mm}@0.5"] = (P[f"s:{mm}@m"].values >= 0.5).astype(float)
rng = np.random.default_rng(8503); kA = rng.hypergeometric(kv, n - kv, np.ones_like(n)); A1 = (kA >= 1).astype(float)
kB = kv - kA; gB = (kB / np.maximum(n - 1, 1) >= 0.5 - 1e-12).astype(float)
kappa = {}
for k, v in list(INS.items()) + [("single_rater_vs_goldB", A1)]:
    gg = gB if k.startswith("single") else z
    po = float((v == gg).mean()); p1 = v.mean(); p2 = gg.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2); kappa[k] = (po - pe) / (1 - pe)


def ols_cl(xv, yv, cidx, mask):
    """OLS slope of y on x over mask, with the article-clustered sandwich SE (CR0 x G/(G-1))."""
    c = cidx[mask]; xx = xv[mask]; yy = yv[mask]
    nn = len(xx); xb = xx.mean(); yb = yy.mean(); sxx = np.sum((xx - xb) ** 2)
    if sxx <= 0:
        return np.nan, np.nan
    b = np.sum((xx - xb) * (yy - yb)) / sxx; e = (xx - xb) * (yy - yb - b * (xx - xb)) / sxx
    psi = np.bincount(c, weights=e); G = np.count_nonzero(np.bincount(c))
    return b, float(np.sqrt(np.sum(psi ** 2) * G / max(G - 1, 1)))


def lrr_cl(v, emask):
    vp, vc = v[pa[emask]], v[ch[emask]]; c = ecl[emask]
    a, b, cc, d = np.bincount(c, vp * vc), np.bincount(c, vp), np.bincount(c, (1 - vp) * vc), np.bincount(c, 1 - vp)
    A, B, C, D = a.sum(), b.sum(), cc.sum(), d.sum()
    if min(A, B, C, D) <= 0:
        return np.nan, np.nan
    psi = a / A - b / B - cc / C + d / D; G = np.count_nonzero(np.bincount(c))
    return float(np.log((A / B) / (C / D))), float(np.sqrt(np.sum(psi ** 2) * G / max(G - 1, 1)))


FU = {}


def laws(k, v, cmask, umask, emask):
    out = {}
    out["depth"] = ols_cl(x, v, cl, cmask)
    if k not in FU:
        FU[k] = np.bincount(un, weights=v, minlength=ku.size) / np.maximum(ku, 1)
    out["size"] = ols_cl(lxu, FU[k], ucl, umask & (ku >= 2))
    out["log_rr"] = lrr_cl(v, emask)
    return out


def verdict(b, se):
    if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
        return 0
    t = b / se
    return 1 if t > 1.96 else (-1 if t < -1.96 else 0)


rs = np.random.default_rng(8502); V = {k: {law: [] for law in ("depth", "size", "log_rr")} for k in ["gold", "gold_B", "single_rater"] + list(INS)}
sizes = []
for s in range(NSUB):
    order = rs.permutation(nC); cum = np.cumsum(csize[order]); take = order[: int(np.searchsorted(cum, TARGET) + 1)]
    inC = np.zeros(nC, bool); inC[take] = True; cmask = inC[cl]; umask = inC[ucl]; emask = inC[ecl]; sizes.append(int(cmask.sum()))
    for k, v in [("gold", z), ("gold_B", gB), ("single_rater", A1)] + list(INS.items()):
        L = laws(k, v, cmask, umask, emask)
        for law, (b, se) in L.items():
            V[k][law].append((verdict(b, se), b, se))
    if (s + 1) % 20 == 0:
        print("subsample", s + 1, "size", sizes[-1], flush=True)


def compare(k, ref):
    out = {}
    for law in ("depth", "size", "log_rr"):
        a = np.array([t[0] for t in V[k][law]]); g_ = np.array([t[0] for t in V[ref][law]])
        out[law] = dict(differ=float(np.mean(a != g_)), reversal=float(np.mean(a * g_ == -1)), manufactured=float(np.mean((a != 0) & (g_ == 0))),
                        lost=float(np.mean((a == 0) & (g_ != 0))))
    out["any_law"] = float(np.mean(np.any([[t[0] for t in V[k][law]] != np.array([t[0] for t in V[ref][law]]) for law in ("depth", "size", "log_rr")], axis=0)))
    out["mean_over_laws"] = float(np.mean([out[law]["differ"] for law in ("depth", "size", "log_rr")]))
    return out


R2 = {"n_subsamples": NSUB, "size_median": float(np.median(sizes)), "size_min": int(min(sizes)), "size_max": int(max(sizes)), "kappa": kappa,
      "gold_verdicts": {law: {str(c): float(np.mean(np.array([t[0] for t in V["gold"][law]]) == c)) for c in (-1, 0, 1)} for law in ("depth", "size", "log_rr")},
      "gold_B_verdicts": {law: {str(c): float(np.mean(np.array([t[0] for t in V["gold_B"][law]]) == c)) for c in (-1, 0, 1)} for law in ("depth", "size", "log_rr")},
      "instruments": {k: compare(k, "gold") for k in INS}, "single_rater_vs_goldB": compare("single_rater", "gold_B"),
      "single_rater_vs_gold": compare("single_rater", "gold"), "goldB_vs_gold": compare("gold_B", "gold")}
hk = [k for k in INS if kappa[k] >= 0.6]; ll = [k for k in INS if k.startswith(LLM)]; cls = [k for k in INS if not k.startswith(LLM)]
def pool(keys):
    return {law: float(np.mean([R2["instruments"][k][law]["differ"] for k in keys])) for law in ("depth", "size", "log_rr")} | \
           {"mean_over_laws": float(np.mean([R2["instruments"][k]["mean_over_laws"] for k in keys])), "n": len(keys), "keys": keys}
R2["pooled"] = {"kappa_ge_06": pool(hk) if hk else None, "llm": pool(ll), "classifier": pool(cls), "all": pool(list(INS))}
res["part2"] = R2
print("part 2:", json.dumps({k: v for k, v in R2.items() if k in ("size_median", "gold_verdicts", "pooled", "single_rater_vs_goldB", "goldB_vs_gold")}, indent=0), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
