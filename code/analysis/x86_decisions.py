"""Addendum 86, B86b: decisions at several sample sizes (descriptive; B85b's part 2 extended).
Article-cluster subsamples of about 10,000, 50,000 and 100,000 comments of Civil Comments reply threads, 200 each (seeds 8602,
8603, 8604; clusters in random order, taken until the size is reached). In each, the depth law, the thread size law and the log reply
RR of gold and of every instrument are classified as significantly positive, negative or neither (Wald, |t| > 1.96,
article-clustered sandwich SE; delta method for the log RR), exactly as x85_human_ref.py. Reported: the share of subsamples whose
conclusion differs from gold's (reversal, manufactured, lost), and the share in which the instrument's 95% CI excludes gold's
estimate on the same subsample; for every instrument, pooled over the kappa >= 0.6 settings, the LLMs and the classifiers, and for
the single rater (seed 8503) against the majority of the others (gold B).
usage: python3 x86_decisions.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NSUB = 200
SIZES = [(10000, 8602), (50000, 8603), (100000, 8604)]; LLM = ("haiku", "sonnet", "qwen", "gemma"); LAWS = ("depth", "size", "log_rr")
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float)
cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); csize = np.bincount(cl, minlength=nC)
un = pd.factorize(P.unit.values)[0]; ku = np.bincount(un); ucl = np.zeros(ku.size, np.int64); ucl[un] = cl; lxu = np.log2(np.maximum(ku, 1)).astype(float)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
INS = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    INS[f"{mm}@0.5"] = (P[f"s:{mm}@m"].values >= 0.5).astype(float)
rng = np.random.default_rng(8503); kA = rng.hypergeometric(kv, n - kv, np.ones_like(n)); A1 = (kA >= 1).astype(float)
kB = kv - kA; gB = (kB / np.maximum(n - 1, 1) >= 0.5 - 1e-12).astype(float)
kappa = {}
for k, v in INS.items():
    po = float((v == z).mean()); p1 = v.mean(); p2 = z.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2); kappa[k] = (po - pe) / (1 - pe)
FU = {}


def ols_cl(xv, yv, cidx, mask):
    c = cidx[mask]; xx = xv[mask]; yy = yv[mask]
    xb = xx.mean(); yb = yy.mean(); sxx = np.sum((xx - xb) ** 2)
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


def laws(k, v, cmask, umask, emask):
    if k not in FU:
        FU[k] = np.bincount(un, weights=v, minlength=ku.size) / np.maximum(ku, 1)
    return {"depth": ols_cl(x, v, cl, cmask), "size": ols_cl(lxu, FU[k], ucl, umask & (ku >= 2)), "log_rr": lrr_cl(v, emask)}


def verdict(b, se):
    if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
        return 0
    t = b / se
    return 1 if t > 1.96 else (-1 if t < -1.96 else 0)


def compare(V, k, ref):
    out = {}
    for law in LAWS:
        a = np.array([t[0] for t in V[k][law]]); g_ = np.array([t[0] for t in V[ref][law]])
        bi = np.array([t[1] for t in V[k][law]]); si = np.array([t[2] for t in V[k][law]]); bg = np.array([t[1] for t in V[ref][law]])
        out[law] = dict(differ=float(np.mean(a != g_)), reversal=float(np.mean(a * g_ == -1)), manufactured=float(np.mean((a != 0) & (g_ == 0))),
                        lost=float(np.mean((a == 0) & (g_ != 0))), ci_excludes_gold=float(np.nanmean(np.abs(bi - bg) > 1.96 * si)))
    out["mean_over_laws"] = float(np.mean([out[law]["differ"] for law in LAWS]))
    return out


res = {"sizes": {}, "kappa": kappa}
for target, seed in SIZES:
    rs = np.random.default_rng(seed); V = {k: {law: [] for law in LAWS} for k in ["gold", "gold_B", "single_rater"] + list(INS)}; sizes = []
    for s in range(NSUB):
        order = rs.permutation(nC); cum = np.cumsum(csize[order]); take = order[: int(np.searchsorted(cum, target) + 1)]
        inC = np.zeros(nC, bool); inC[take] = True; cmask = inC[cl]; umask = inC[ucl]; emask = inC[ecl]; sizes.append(int(cmask.sum()))
        for k, v in [("gold", z), ("gold_B", gB), ("single_rater", A1)] + list(INS.items()):
            for law, (b, se) in laws(k, v, cmask, umask, emask).items():
                V[k][law].append((verdict(b, se), b, se))
    R = {"n_subsamples": NSUB, "size_median": float(np.median(sizes)),
         "gold_verdicts": {law: {str(c): float(np.mean(np.array([t[0] for t in V["gold"][law]]) == c)) for c in (-1, 0, 1)} for law in LAWS},
         "instruments": {k: compare(V, k, "gold") for k in INS}, "single_rater_vs_goldB": compare(V, "single_rater", "gold_B"), "goldB_vs_gold": compare(V, "gold_B", "gold")}
    hk = [k for k in INS if kappa[k] >= 0.6]; ll = [k for k in INS if k.startswith(LLM)]; cls = [k for k in INS if not k.startswith(LLM)]
    pool = lambda keys: {law: dict(differ=float(np.mean([R["instruments"][k][law]["differ"] for k in keys])),
                                   ci_excludes_gold=float(np.mean([R["instruments"][k][law]["ci_excludes_gold"] for k in keys]))) for law in LAWS} | {"n": len(keys)}
    R["pooled"] = {"kappa_ge_06": pool(hk), "llm": pool(ll), "classifier": pool(cls)}
    res["sizes"][str(target)] = R
    sr = R["single_rater_vs_goldB"]; hk_ = R["instruments"]["haiku"]
    print(f"size {target} (median {R['size_median']:.0f}): gold depth verdicts {R['gold_verdicts']['depth']}, size {R['gold_verdicts']['size']}", flush=True)
    for law in LAWS:
        print(f"   {law:7s} single rater differ {sr[law]['differ']:.3f} (CI excl {sr[law]['ci_excludes_gold']:.3f}) | Haiku differ {hk_[law]['differ']:.3f} rev {hk_[law]['reversal']:.3f} (CI excl {hk_[law]['ci_excludes_gold']:.3f}) | "
              f"kappa>=0.6 {R['pooled']['kappa_ge_06'][law]['differ']:.3f} (CI excl {R['pooled']['kappa_ge_06'][law]['ci_excludes_gold']:.3f}) | classifiers {R['pooled']['classifier'][law]['differ']:.3f} "
              f"(CI excl {R['pooled']['classifier'][law]['ci_excludes_gold']:.3f}) | LLMs {R['pooled']['llm'][law]['differ']:.3f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
