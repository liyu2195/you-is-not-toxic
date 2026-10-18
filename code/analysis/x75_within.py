"""Addendum 75, B75e: the within-thread escalation law, and B75f: agreement against distortion within populations.
B75e. For every labeler of results/x14_pops/<pop>.parquet (plus the fresh classifiers at 0.5 and, on Civil Comments, the lenient
human benchmark), the depth slope with thread fixed effects: with x = min(depth, 10) and x~ = x minus its thread mean,
    law_w(V) = sum x~ V / sum x~^2.
Its weights sum to 0 within every thread, so Proposition 1 holds exactly:
    law_w(Y) = (TPR - FPR) law_w(Z) + S_FP + S_TP,  S_FP = law_w((1 - Z) Y) + FPR law_w(Z),  S_TP = law_w(Z Y) - TPR law_w(Z).
500 Poisson cluster-bootstrap resamples (clusters as in x74_decomp.py; threads nest in clusters), seed 7550.
B75f. Within each population, Spearman rho between kappa and the absolute distortion of each law in its natural units
(x74_decomp.json for the pooled laws; B75e for the within-thread law), over that population's labelers.
usage: python3 x75_within.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
W = _ROOT; OUT = sys.argv[1]; NB = 500
FRESH = ["sn_rob", "mh_dist", "fb_hate", "cn_off"]
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json"))
res = {"within": {}, "agreement": {}}
for pop in ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]:
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet")
    z = P.z.values.astype(float)
    labs = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    for m in FRESH:
        if f"s:{m}@m" in P:
            labs[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
    if "tox" in P:
        labs["human_lenient"] = np.where(z == 1, 1.0, (P.tox.values >= 0.3).astype(float))
    cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1; uu = pd.factorize(P.unit.values)[0]
    x = np.minimum(P.depth.values, 10).astype(float); xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
    rng = np.random.default_rng(7550); Wb = np.vstack([np.ones(nC), rng.poisson(1.0, size=(NB, nC))])
    sxx = Wb @ np.bincount(cl, xt * xt, nC)
    law = lambda v: (Wb @ np.bincount(cl, xt * v, nC)) / sxx
    zsum = Wb @ np.bincount(cl, z, nC); nsum = Wb @ np.bincount(cl, np.ones(len(z)), nC)
    ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    sd = lambda a: float(np.nanstd(a[1:], ddof=1))
    lZ = law(z); pooled = DEC[pop]["gold"]["depth"][0]
    out = {"gold_within": ci(lZ), "gold_pooled": pooled, "share_of_variance_within": float((xt ** 2).sum() / ((x - x.mean()) ** 2).sum()), "labelers": {}}
    for L, y in labs.items():
        TPR = (Wb @ np.bincount(cl, z * y, nC)) / zsum; FPR = (Wb @ np.bincount(cl, (1 - z) * y, nC)) / (nsum - zsum)
        lY = law(y); att = (TPR - FPR) * lZ; SFP = law((1 - z) * y) + FPR * lZ; STP = law(z * y) - TPR * lZ; d = lY - lZ
        pd_ = DEC[pop]["labelers"].get(L, {}).get("laws", {}).get("depth", {})
        out["labelers"][L] = dict(law=ci(lY), distortion=ci(d), z=float(d[0] / sd(d)) if sd(d) > 0 else None, attenuation_term=ci(att),
                                  fp_term=ci(SFP), tp_term=ci(STP), check=float(np.nanmax(np.abs(att + SFP + STP - lY))),
                                  pooled_distortion=pd_.get("distortion", [None])[0], pooled_z=pd_.get("z"))
    res["within"][pop] = out
    print(pop, "gold within", round(lZ[0] * 1e3, 3), "pooled", round(pooled * 1e3, 3), "within share", round(out["share_of_variance_within"], 3), flush=True)
    for L, r in out["labelers"].items():
        print(f"   {L:24s} within dist={r['distortion'][0]*1e3:+7.2f} [{r['distortion'][1]*1e3:+.2f},{r['distortion'][2]*1e3:+.2f}] z={r['z'] or 0:+6.1f} FP={r['fp_term'][0]*1e3:+6.2f}"
              f" | pooled dist={(r['pooled_distortion'] or 0)*1e3:+7.2f} z={r['pooled_z'] or 0:+6.1f}", flush=True)
    # B75f
    ag = {}
    for law_nm in ["depth", "size", "article_size", "log_rr", "depth_within"]:
        ks, ds = [], []
        for L, r in DEC[pop]["labelers"].items():
            if law_nm == "depth_within":
                d = out["labelers"].get(L, {}).get("distortion", [None])[0]
            else:
                d = r["laws"].get(law_nm, {}).get("distortion", [None])[0]
            if d is None or not np.isfinite(d) or (law_nm == "log_rr" and pop == "cga"):
                continue
            ks.append(r["kappa"][0]); ds.append(abs(d))
        if len(ks) >= 5:
            ks, ds = np.array(ks), np.array(ds); top = ds[ks >= np.quantile(ks, 0.75)]
            ag[law_nm] = dict(n=len(ks), spearman=float(spearmanr(ks, ds).correlation), p=float(spearmanr(ks, ds).pvalue),
                              max_abs_dist_top_quartile=float(top.max()), min_abs_dist_bottom_quartile=float(ds[ks <= np.quantile(ks, 0.25)].min()))
    res["agreement"][pop] = ag
    print("   agreement", {k: (v["n"], round(v["spearman"], 2)) for k, v in ag.items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
