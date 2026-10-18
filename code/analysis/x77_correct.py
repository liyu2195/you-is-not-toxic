"""Addendum 77, B77b: validate and correct. For every instrument of results/x14_pops/<pop>.parquet (plus the fresh classifiers at
0.5; the lenient human excluded) and every population, a uniform validation sample of 2,000 comments estimates TPR and FPR
(500 simulated validations, seed 7710 + population index). The instrument labels every comment, so its own law L(Y) is exact.
  linear laws (depth, within-thread depth, thread size, article size): L(Y) / (TPR^ - FPR^) (Rogan-Gladen), 95% delta-method
    interval from the binomial variances of TPR^ and FPR^;
  contagion: the matrix method under independent misclassification of the two ends: the gold edge-cell shares are
    (M^ x M^)^-1 applied to the flags' four edge-cell shares, M^ = [[1 - FPR^, 1 - TPR^], [FPR^, TPR^]]; the 95% interval is the
    percentile interval of a parametric bootstrap of (TPR^, FPR^) (binomial, 200 draws); an estimate with a negative cell share
    is undefined and does not cover.
Reported per entry: bias of the corrected law (mean over validations minus gold's law), in natural units and relative to gold's
law; its SD over validations; coverage of gold's law; the population-level corrected law (true TPR, FPR) and its distortion.
H77c: the corrected intervals cover gold's law below 0.90 in at least 25% of (instrument, population, law) entries.
usage: python3 x77_correct.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NV = 500; NVAL = 2000; NBT = 200
FRESH = ["sn_rob", "mh_dist", "fb_hate", "cn_off"]
res = {"entries": [], "pops": {}}
for pi_, pop in enumerate(["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]):
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(float); N = len(z)
    labs = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    for m in FRESH:
        if f"s:{m}@m" in P:
            labs[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
    x = np.minimum(P.depth.values, 10).astype(float); uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
    W_ = {"depth": (x - x.mean()) / ((x - x.mean()) ** 2).sum(), "depth_within": xt / (xt ** 2).sum()}
    for nm, u in (("size", P.unit.values),) + ((("article_size", P.article.values),) if "article" in P else ()):
        f = pd.factorize(u)[0]; ku = np.bincount(f).astype(float); ok = ku >= 2; xu = np.log2(np.maximum(ku, 1)); xb = xu[ok].mean(); ss = ((xu[ok] - xb) ** 2).sum()
        W_[nm] = np.where(ok[f], (xu[f] - xb) / (ku[f] * ss), 0.0)
    ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
    def cells(v):
        a, b = v[pa], v[ch]; return np.array([((1 - a) * (1 - b)).mean(), ((1 - a) * b).mean(), (a * (1 - b)).mean(), (a * b).mean()])  # (yp, yc) = 00, 01, 10, 11
    def rr_from(c):
        return (c[..., 3] / (c[..., 2] + c[..., 3])) / (c[..., 1] / (c[..., 0] + c[..., 1]))
    gold_rr = float(rr_from(cells(z))) if pop != "cga" else None
    rng = np.random.default_rng(7710 + pi_)
    VAL = [rng.choice(N, NVAL, replace=False) for _ in range(NV)]
    out = {}
    for L, y in labs.items():
        tpr, fpr = float((z * y).sum() / z.sum()), float(((1 - z) * y).sum() / (1 - z).sum())
        T = np.array([(z[s] * y[s]).sum() / max(z[s].sum(), 1) for s in VAL]); F = np.array([((1 - z[s]) * y[s]).sum() / max((1 - z[s]).sum(), 1) for s in VAL])
        npos = np.array([z[s].sum() for s in VAL]); nneg = NVAL - npos
        r = {"tpr": tpr, "fpr": fpr, "laws": {}}
        for law, w in W_.items():
            lY, lZ = float(w @ y), float(w @ z); d = T - F
            est = lY / d; var = (lY ** 2 / d ** 4) * (T * (1 - T) / np.maximum(npos, 1) + F * (1 - F) / np.maximum(nneg, 1))
            lo, hi = est - 1.96 * np.sqrt(var), est + 1.96 * np.sqrt(var); cov = float(np.mean((lo <= lZ) & (lZ <= hi)))
            pop_corr = lY / (tpr - fpr)
            r["laws"][law] = dict(gold=lZ, instrument=lY, corrected_pop=pop_corr, corrected_pop_minus_gold=pop_corr - lZ, bias=float(np.mean(est) - lZ),
                                  bias_rel=float((np.mean(est) - lZ) / lZ) if lZ != 0 else None, sd=float(np.std(est)), coverage=cov,
                                  uncorrected_minus_gold=lY - lZ)
            res["entries"].append(dict(pop=pop, lab=L, law=law, coverage=cov))
        if gold_rr is not None:
            cY = cells(y)
            def corr(Tv, Fv):
                det = Tv - Fv; inv = np.stack([np.stack([Tv, -(1 - Tv)], -1), np.stack([-Fv, 1 - Fv], -1)], -2) / det[..., None, None]   # M^-1, rows z, cols y
                K = np.einsum("...ab,...cd->...acbd", inv, inv).reshape(inv.shape[:-2] + (4, 4))
                return K @ cY
            cz = corr(T, F); ok_ = (cz > 0).all(-1); est = np.where(ok_, rr_from(np.clip(cz, 1e-12, None)), np.nan)
            br = np.random.default_rng(7710 + pi_ + 100); cov_ = []
            for v in range(NV):
                Tb = br.binomial(int(npos[v]), T[v], NBT) / max(npos[v], 1); Fb = br.binomial(int(nneg[v]), F[v], NBT) / max(nneg[v], 1)
                cb = corr(Tb, Fb); okb = (cb > 0).all(-1); rb = np.where(okb, rr_from(np.clip(cb, 1e-12, None)), np.nan)
                if not ok_[v] or np.isnan(rb).mean() > 0.5:
                    cov_.append(False); continue
                lo, hi = np.nanpercentile(rb, 2.5), np.nanpercentile(rb, 97.5); cov_.append(bool(lo <= gold_rr <= hi))
            cp = corr(np.array(tpr), np.array(fpr)); pop_corr = float(rr_from(cp)) if (cp > 0).all() else None
            r["laws"]["log_rr"] = dict(gold=float(np.log(gold_rr)), instrument=float(np.log(rr_from(cY))), corrected_pop=None if pop_corr is None else float(np.log(pop_corr)),
                                       corrected_pop_minus_gold=None if pop_corr is None else float(np.log(pop_corr) - np.log(gold_rr)),
                                       bias=float(np.nanmean(np.log(est)) - np.log(gold_rr)) if np.isfinite(est).any() else None, undefined=float(1 - ok_.mean()),
                                       sd=float(np.nanstd(np.log(est))) if np.isfinite(est).any() else None, coverage=float(np.mean(cov_)),
                                       uncorrected_minus_gold=float(np.log(rr_from(cY)) - np.log(gold_rr)))
            res["entries"].append(dict(pop=pop, lab=L, law="log_rr", coverage=float(np.mean(cov_))))
        out[L] = r
        print(pop, f"{L:22s}", " ".join(f"{law}: cov {v['coverage']:.2f} bias {v['bias'] if v['bias'] is None else round(v['bias'] * (1 if law == 'log_rr' else 100), 3)}" for law, v in r["laws"].items()), flush=True)
    res["pops"][pop] = out
cv = np.array([e["coverage"] for e in res["entries"]]); res["H77c"] = dict(n=len(cv), share_below_90=float(np.mean(cv < 0.90)), median=float(np.median(cv)), pass_=bool(np.mean(cv < 0.90) >= 0.25))
for law in ("depth", "depth_within", "size", "article_size", "log_rr"):
    v = np.array([e["coverage"] for e in res["entries"] if e["law"] == law]); res["H77c"][f"share_below_90_{law}"] = float(np.mean(v < 0.90)) if len(v) else None
print(json.dumps(res["H77c"], indent=1))
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
