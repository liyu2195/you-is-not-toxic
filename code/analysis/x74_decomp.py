"""Addendum 74, A5/A6 (memory-light companion of x74_placebo.py): the exact decomposition of every law, kappa, and paired
law distortions with cluster-bootstrap SDs, for every labeler of results/x14_pops/<pop>.parquet (plus the fresh classifiers at
0.5 and, on Civil Comments, the lenient human benchmark).
Linear laws (the depth law; the thread and article size laws, unit share on log2 size over units with >= 2 comments, each unit
once). With TPR and FPR the labeler's item-level rates, for every linear law with sum of weights 0, exactly,
    law(Y) = (TPR - FPR) law(Z) + S_FP + S_TP,
    S_FP = law((1 - Z) Y) + FPR law(Z)   (structured false positives: the law of the false-positive indicator, centred),
    S_TP = law(Z Y) - TPR law(Z)          (structured true positives; minus the structured misses).
For the depth law, S_FP = (SS_neg / SS) x the placebo depth slope (absolute), with SS the sum of squared depth deviations.
The log reply RR: the two-path (Shapley) split of log RR(Y) - log RR(Z) into a false-positive part and a miss part through
Z or Y and Z and Y. 500 Poisson cluster-bootstrap resamples (clusters: articles on Civil Comments, units elsewhere), seed 7440.
usage: python3 x74_decomp.py OUT_JSON [--nb 500] [--pops a,b]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
NB = int(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else 500
POPS = sys.argv[sys.argv.index("--pops") + 1].split(",") if "--pops" in sys.argv else ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]
FRESH = ["sn_rob", "mh_dist", "fb_hate", "cn_off"]


def run(pop):
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet")
    z = P.z.values.astype(float); N = len(z)
    labs = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    for m in FRESH:
        if f"s:{m}@m" in P:
            labs[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
    if "tox" in P:
        labs["human_lenient"] = np.where(z == 1, 1.0, (P.tox.values >= 0.3).astype(float))
    cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
    rng = np.random.default_rng(7440); Wb = np.vstack([np.ones(nC), rng.poisson(1.0, size=(NB, nC))])
    x = np.minimum(P.depth.values, 10).astype(float)
    levels = {"depth": None, "size": P.unit.values}
    if "article" in P:
        levels["article_size"] = P.article.values
    # per-cluster sufficient statistics of an OLS slope: item level (depth) or unit level (size: unit share of the outcome)
    base = {}
    base["depth"] = dict(n=np.bincount(cl, minlength=nC).astype(float), sx=np.bincount(cl, x, nC), sxx=np.bincount(cl, x * x, nC))
    unit_maps = {}
    for nm, u in levels.items():
        if u is None:
            continue
        uu = pd.factorize(u)[0]; ku = np.bincount(uu).astype(float); ucl = np.zeros(ku.size, np.int64); ucl[uu] = cl; ok = ku >= 2
        xu = np.log2(np.maximum(ku, 1)); unit_maps[nm] = (uu, ku, ucl, ok, xu)
        base[nm] = dict(n=np.bincount(ucl[ok], minlength=nC).astype(float), sx=np.bincount(ucl[ok], xu[ok], nC), sxx=np.bincount(ucl[ok], xu[ok] ** 2, nC))
    def suff(nm, v):
        if nm == "depth":
            return np.bincount(cl, v, nC), np.bincount(cl, x * v, nC)
        uu, ku, ucl, ok, xu = unit_maps[nm]; f = np.bincount(uu, v, ku.size) / ku
        return np.bincount(ucl[ok], f[ok], nC), np.bincount(ucl[ok], xu[ok] * f[ok], nC)
    def slope(nm, sv):
        b = base[nm]; n, sx, sxx = Wb @ b["n"], Wb @ b["sx"], Wb @ b["sxx"]; sy, sxy = Wb @ sv[0], Wb @ sv[1]
        return (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
    def lrr(v):
        vp, vc = v[pa], v[ch]
        p1, p1c1, p0, p0c1 = (Wb @ np.bincount(ecl, w, nC) for w in (vp, vp * vc, 1 - vp, (1 - vp) * vc))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((p1c1 / p1) / (p0c1 / p0))
    zsum = Wb @ np.bincount(cl, z, nC); nsum = Wb @ np.bincount(cl, np.ones(N), nC)
    ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    sd = lambda a: float(np.nanstd(a[1:], ddof=1))
    gl = {nm: slope(nm, suff(nm, z)) for nm in base}
    have_edges = len(ch) > 0 and z[pa].sum() > 0 and (z[pa] * z[ch]).sum() > 0
    glr = lrr(z) if have_edges else None
    out = {"N": N, "prev": float(z.mean()), "gold": {nm: ci(v) for nm, v in gl.items()} | ({"log_rr": ci(glr)} if have_edges else {}), "labelers": {}}
    for L, y in labs.items():
        tp = np.bincount(cl, z * y, nC); fp = np.bincount(cl, (1 - z) * y, nC)
        TPR = (Wb @ tp) / zsum; FPR = (Wb @ fp) / (nsum - zsum); q = (Wb @ (tp + fp)) / nsum; p = zsum / nsum
        po = p * TPR + (1 - p) * (1 - FPR); pe = p * q + (1 - p) * (1 - q); kap = (po - pe) / (1 - pe)
        r = {"kappa": ci(kap), "tpr": ci(TPR), "fpr": ci(FPR), "flag_rate": ci(q), "laws": {}}
        for nm in base:
            lY = slope(nm, suff(nm, y)); lZY = slope(nm, suff(nm, z * y)); lFP = slope(nm, suff(nm, (1 - z) * y)); lZ = gl[nm]
            att = (TPR - FPR) * lZ; SFP = lFP + FPR * lZ; STP = lZY - TPR * lZ
            r["laws"][nm] = dict(law=ci(lY), gold=ci(lZ), distortion=ci(lY - lZ), distortion_sd=sd(lY - lZ),
                                 z=float((lY[0] - lZ[0]) / sd(lY - lZ)) if sd(lY - lZ) > 0 else None,
                                 attenuation_term=ci(att), fp_term=ci(SFP), tp_term=ci(STP), fp_law=ci(lFP), miss_law=ci(lZ - lZY),
                                 check=float(np.nanmax(np.abs(att + SFP + STP - lY))))
            if nm == "depth":                                           # S_FP = (SS_neg / SS) x the placebo depth slope
                neg = z == 0; xn = x[neg]; yn = y[neg]
                b = float(((xn - xn.mean()) * (yn - yn.mean())).sum() / ((xn - xn.mean()) ** 2).sum())
                r["laws"][nm]["placebo_depth_slope"] = b; r["laws"][nm]["ss_ratio"] = float(((xn - xn.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
        if have_edges:
            lY = lrr(y); lor = lrr(np.maximum(z, y)); lan = lrr(z * y)
            fpp = 0.5 * ((lor - glr) + (lY - lan)); msp = 0.5 * ((lY - lor) + (lan - glr))
            r["laws"]["log_rr"] = dict(law=ci(lY), gold=ci(glr), distortion=ci(lY - glr), distortion_sd=sd(lY - glr),
                                       z=float((lY[0] - glr[0]) / sd(lY - glr)) if sd(lY - glr) > 0 else None,
                                       fp_part=ci(fpp), miss_part=ci(msp), check=float(np.nanmax(np.abs(fpp + msp - (lY - glr)))))
        out["labelers"][L] = r
    return out


res = {}
for pop in POPS:
    res[pop] = run(pop)
    g = res[pop]["gold"]
    print(pop, "gold", {k: round(v[0] * (1e3 if k != "log_rr" else 1), 3) for k, v in g.items()}, flush=True)
    for L, r in res[pop]["labelers"].items():
        d = r["laws"]["depth"]
        print(f"  {L:24s} k={r['kappa'][0]:.3f} depth: dist={d['distortion'][0]*1e3:+7.2f} att={d['attenuation_term'][0]*1e3:+7.2f} FP={d['fp_term'][0]*1e3:+7.2f} TP={d['tp_term'][0]*1e3:+7.2f} z={d['z']:+7.1f}"
              + (f" | lrr dist={r['laws']['log_rr']['distortion'][0]:+.3f} FP={r['laws']['log_rr']['fp_part'][0]:+.3f} miss={r['laws']['log_rr']['miss_part'][0]:+.3f}" if "log_rr" in r["laws"] else ""), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
