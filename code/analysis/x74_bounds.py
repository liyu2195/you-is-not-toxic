"""Addendum 74, Part C: sharp agreement bounds. For a law linear in the item labels, L(Y) = sum_i w_i Y_i, every instrument with
n_TP true positives and n_FP false positives (equivalently, with a given kappa and flag rate) has
  L(Y) in [ sum of the n_TP smallest w over gold positives + sum of the n_FP smallest w over gold negatives,
            sum of the n_TP largest  w over gold positives + sum of the n_FP largest  w over gold negatives ],
and both ends are attained (by the instrument that flags exactly those comments), so the bounds are sharp.
Laws: the depth law (OLS slope on min(depth, 10)), the thread size law and the article size law (unit share on log2 size, units
with >= 2 comments, each unit once). For every population and labeler of results/x14_pops/: the bounds at the labeler's own
counts, gold's value and the labeler's; and, at flag rate = gold prevalence, the bounds as a function of kappa.
Dual form (identification): with the labeler's flags fixed, every gold configuration with the labeler's counts (n_TP gold positives
among the flagged comments, n_FN among the others) gives a gold law in [gold_lo, gold_hi], by the same sorting over the flagged and
the unflagged comments: what an exact item-level validation leaves open.
usage: python3 x74_bounds.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]


def size_w(unit):
    u = pd.factorize(unit)[0]; ku = np.bincount(u).astype(float); xu = np.log2(np.maximum(ku, 1)); ok = ku >= 2
    xb = xu[ok].mean(); ss = ((xu[ok] - xb) ** 2).sum()
    return np.where(ok[u], (xu[u] - xb) / (ku[u] * ss), 0.0)


def bounds(w, z, ntp, nfp):
    wp = np.sort(w[z == 1]); wn = np.sort(w[z == 0])
    cp = np.r_[0, np.cumsum(wp)]; cn = np.r_[0, np.cumsum(wn)]; rp = np.r_[0, np.cumsum(wp[::-1])]; rn = np.r_[0, np.cumsum(wn[::-1])]
    return float(cp[ntp] + cn[nfp]), float(rp[ntp] + rn[nfp])


def counts_for_kappa(kap, p, N):
    """Flag rate = prevalence p: TPR = 1 - (1 - kappa)(1 - p), FPR = (1 - kappa) p (from kappa = (po - pe) / (1 - pe), q = p)."""
    tpr = 1 - (1 - kap) * (1 - p); fpr = (1 - kap) * p
    return int(round(tpr * p * N)), int(round(fpr * (1 - p) * N))


res = {}
for pop in ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]:
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(int); N = len(z); p = float(z.mean())
    x = np.minimum(P.depth.values, 10).astype(float); wd = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    laws = {"depth": wd, "size": size_w(P.unit.values)}
    if "article" in P:
        laws["article_size"] = size_w(P.article.values)
    out = {"N": N, "prev": p, "gold": {k: float(w @ z) for k, w in laws.items()}, "labelers": {}, "by_kappa": {}}
    for c in [c for c in P.columns if c.startswith("y:")]:
        y = np.nan_to_num(P[c].values.astype(float)).astype(int); ntp = int(((y == 1) & (z == 1)).sum()); nfp = int(((y == 1) & (z == 0)).sum())
        q = y.mean(); tpr = ntp / max(z.sum(), 1); fpr = nfp / max((z == 0).sum(), 1)
        po = p * tpr + (1 - p) * (1 - fpr); pe = p * q + (1 - p) * (1 - q); kap = (po - pe) / (1 - pe)
        r = {"kappa": float(kap), "flag_rate": float(q), "tpr": float(tpr), "fpr": float(fpr)}
        for k, w in laws.items():
            lo, hi = bounds(w, z, ntp, nfp); r[k] = dict(lo=lo, hi=hi, labeler=float(w @ y), gold=float(w @ z))
            # dual (identification): the labeler's flags fixed, every gold configuration with the same counts (n_TP gold positives
            # among the flagged, n_FN among the others) -- what exact item-level validation leaves open about the gold law
            nfn = int(z.sum()) - ntp; glo, ghi = bounds(w, y, ntp, nfn); r[k].update(gold_lo=glo, gold_hi=ghi)
        out["labelers"][c[2:]] = r
    for kap in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        ntp, nfp = counts_for_kappa(kap, p, N); out["by_kappa"][kap] = {k: bounds(w, z, ntp, nfp) for k, w in laws.items()}
    res[pop] = out
    print(pop, "gold", {k: round(v * 1e3, 3) for k, v in out["gold"].items()}, "| kappa 0.5:", {k: tuple(round(b * 1e3, 2) for b in v) for k, v in out["by_kappa"][0.5].items()},
          "| kappa 0.9:", {k: tuple(round(b * 1e3, 2) for b in v) for k, v in out["by_kappa"][0.9].items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
