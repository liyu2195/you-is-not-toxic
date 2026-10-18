"""Addendum 76, B76e: LLMs shown the parent at their own gold-prevalence threshold.
For each population with a context-probability column s:<llm>_m_ctx (Qwen; Gemma on the 5% sample), the new flags mark the
round(prevalence x N) comments with the highest probability, ties broken at random (seed 7650). Previously the no-context
threshold was reused. Against gold, with 500 Poisson cluster-bootstrap resamples (seed 7651; clusters as in x74_decomp.py):
  flag rate, kappa, TPR, FPR;
  the depth law, the thread size law, the article size law (Civil Comments), the log reply RR (not CGA-Wiki) and the within-thread
  depth law (B75e), each with its distortion, CI and z, and the exact decomposition of the linear laws (Proposition 1);
  placebo statistics: the depth slope and the size slope among gold negatives relative to the FPR, and the placebo RR over edges
  with two gold-negative ends (definitions of x74_placebo.py);
  the B75c contagion terms (uniform errors, structured false positives, structured hits; 50 permutations, seed 7530).
The same is reported for the original flags, for comparison.
usage: python3 x76_rethreshold.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500; NP = 50
res = {}
for pop in ["cc_threads", "cc_s5", "cga"]:
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(float); N = len(z); prev = z.mean()
    llms = [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]
    if not llms:
        continue
    labs = {}
    rng = np.random.default_rng(7650)
    for m in llms:
        s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(prev * N))
        order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0
        labs[f"{m}@own"] = f; labs[m] = np.nan_to_num(P[f"y:{m}"].values.astype(float))
    cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
    rb = np.random.default_rng(7651); Wb = np.vstack([np.ones(nC), rb.poisson(1.0, size=(NB, nC))])
    x = np.minimum(P.depth.values, 10).astype(float); uu = pd.factorize(P.unit.values)[0]
    xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
    levels = {"size": P.unit.values} | ({"article_size": P.article.values} if "article" in P else {})
    base = {"depth": dict(n=np.bincount(cl, minlength=nC).astype(float), sx=np.bincount(cl, x, nC), sxx=np.bincount(cl, x * x, nC))}
    umaps = {}
    for nm, u in levels.items():
        uf = pd.factorize(u)[0]; ku = np.bincount(uf).astype(float); ucl = np.zeros(ku.size, np.int64); ucl[uf] = cl; ok = ku >= 2; xu = np.log2(np.maximum(ku, 1))
        umaps[nm] = (uf, ku, ucl, ok, xu)
        base[nm] = dict(n=np.bincount(ucl[ok], minlength=nC).astype(float), sx=np.bincount(ucl[ok], xu[ok], nC), sxx=np.bincount(ucl[ok], xu[ok] ** 2, nC))
    sxxw = Wb @ np.bincount(cl, xt * xt, nC)
    def law(nm, v):
        if nm == "depth_within":
            return (Wb @ np.bincount(cl, xt * v, nC)) / sxxw
        if nm == "depth":
            sy, sxy = np.bincount(cl, v, nC), np.bincount(cl, x * v, nC)
        else:
            uf, ku, ucl, ok, xu = umaps[nm]; f = np.bincount(uf, v, ku.size) / ku
            sy, sxy = np.bincount(ucl[ok], f[ok], nC), np.bincount(ucl[ok], xu[ok] * f[ok], nC)
        b = base[nm]; n, sx, sxx = Wb @ b["n"], Wb @ b["sx"], Wb @ b["sxx"]; sy, sxy = Wb @ sy, Wb @ sxy
        return (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
    def lrr_b(v, mask=None):
        vp, vc = v[pa], v[ch]; mk = np.ones(len(ch), bool) if mask is None else mask
        p1, p1c1, p0, p0c1 = (Wb @ np.bincount(ecl[mk], w[mk], nC) for w in (vp, vp * vc, 1 - vp, (1 - vp) * vc))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((p1c1 / p1) / (p0c1 / p0))
    def lrr1(v):
        vp, vc = v[pa], v[ch]; a = (vp * vc).sum(); b = vp.sum(); c = ((1 - vp) * vc).sum(); d = (1 - vp).sum()
        return float(np.log((a / b) / (c / d))) if min(a, b, c, d) > 0 else np.nan
    zsum = Wb @ np.bincount(cl, z, nC); nsum = Wb @ np.bincount(cl, np.ones(N), nC)
    ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    sd = lambda a: float(np.nanstd(a[1:], ddof=1))
    LAWS = ["depth", "depth_within"] + list(levels)
    gold = {nm: law(nm, z) for nm in LAWS}
    have_rr = pop != "cga"
    if have_rr:
        gold["log_rr"] = lrr_b(z)
    negboth = (z[pa] == 0) & (z[ch] == 0)
    out = {"N": N, "prev": float(prev), "gold": {k: ci(v) for k, v in gold.items()}, "labelers": {}}
    pos, neg = np.flatnonzero(z == 1), np.flatnonzero(z == 0)
    for L, y in labs.items():
        TPR = (Wb @ np.bincount(cl, z * y, nC)) / zsum; FPR = (Wb @ np.bincount(cl, (1 - z) * y, nC)) / (nsum - zsum)
        q = (Wb @ np.bincount(cl, y, nC)) / nsum; p = zsum / nsum
        po = p * TPR + (1 - p) * (1 - FPR); pe = p * q + (1 - p) * (1 - q); kap = (po - pe) / (1 - pe)
        r = {"flag_rate": ci(q), "kappa": ci(kap), "tpr": ci(TPR), "fpr": ci(FPR), "laws": {}}
        for nm in LAWS:
            lY, lZ = law(nm, y), gold[nm]; d = lY - lZ
            att = (TPR - FPR) * lZ; SFP = law(nm, (1 - z) * y) + FPR * lZ; STP = law(nm, z * y) - TPR * lZ
            r["laws"][nm] = dict(law=ci(lY), distortion=ci(d), z=float(d[0] / sd(d)), attenuation_term=ci(att), fp_term=ci(SFP), tp_term=ci(STP),
                                 check=float(np.nanmax(np.abs(att + SFP + STP - lY))))
        if have_rr:
            lY = lrr_b(y); d = lY - gold["log_rr"]
            r["laws"]["log_rr"] = dict(law=ci(lY), distortion=ci(d), z=float(d[0] / sd(d)))
            prr = lrr_b(y, negboth); r["placebo_rr"] = ci(np.exp(prr))
            pr = np.random.default_rng(7530); pp, ap, pa_ = [], [], []
            for _ in range(NP):
                yp = y.copy(); yp[pos] = pr.permutation(y[pos]); yp[neg] = pr.permutation(y[neg])
                yap = y.copy(); yap[pos] = yp[pos]; ypa = y.copy(); ypa[neg] = yp[neg]
                pp.append(lrr1(yp)); ap.append(lrr1(yap)); pa_.append(lrr1(ypa))
            epp, eap, epa, ly, lz = np.nanmean(pp), np.nanmean(ap), np.nanmean(pa_), lrr1(y), lrr1(z)
            r["contagion_terms"] = dict(distortion=ly - lz, uniform=epp - lz, fp=0.5 * ((eap - epp) + (ly - epa)), tp=0.5 * ((epa - epp) + (ly - eap)))
        # placebo slopes among gold negatives, relative to the FPR (point values)
        xn, yn = x[neg], y[neg]; fpr0 = float(yn.mean())
        r["placebo_depth_rel"] = float((((xn - xn.mean()) * (yn - yn.mean())).sum() / ((xn - xn.mean()) ** 2).sum()) / fpr0)
        uf, ku, ucl, ok, xu = umaps["size"]; nn = np.bincount(uf, (z == 0).astype(float), ku.size); fn = np.bincount(uf, (z == 0) * y, ku.size)
        mm = ok & (nn > 0); yy = fn[mm] / nn[mm]; xx = xu[mm]
        r["placebo_size_rel"] = float((((xx - xx.mean()) * (yy - yy.mean())).sum() / ((xx - xx.mean()) ** 2).sum()) / fpr0)
        # Addendum 79 (E6): the normalization of x74_placebo.py, the mean over units of each unit's gold-negative flag rate
        r["placebo_size_rel_unit"] = float((((xx - xx.mean()) * (yy - yy.mean())).sum() / ((xx - xx.mean()) ** 2).sum()) / yy.mean())
        out["labelers"][L] = r
        print(pop, f"{L:22s} flag={q[0]:.4f} k={kap[0]:.3f} FPR={FPR[0]:.4f}",
              " ".join(f"{nm}: {r['laws'][nm]['distortion'][0]*1e3:+.2f} (z {r['laws'][nm]['z']:+.1f}, FP {r['laws'][nm]['fp_term'][0]*1e3:+.2f})" for nm in LAWS),
              (f"| lrr {r['laws']['log_rr']['distortion'][0]:+.3f} prr {r['placebo_rr'][0]:.2f} terms " + json.dumps({k: round(v, 3) for k, v in r['contagion_terms'].items()})) if have_rr else "",
              f"| placebo depth {r['placebo_depth_rel']*100:+.2f}% size {r['placebo_size_rel']*100:+.2f}% (unit-mean scale {r['placebo_size_rel_unit']*100:+.2f}%)", flush=True)
    res[pop] = out
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
