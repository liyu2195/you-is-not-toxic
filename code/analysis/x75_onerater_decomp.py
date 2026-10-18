"""Addendum 75, B75b output 'the decomposition (B75c)': the exact decomposition of the single rater's distortion against the
leave-one-out gold, on Civil Comments reply threads, for the 20 draws of x75_onerater.py (seeds 7510-7529, the same draws).
Linear laws (depth, thread size; Proposition 1): uniform errors (TPR - FPR - 1) law(B), structured false positives, structured
hits. The log reply RR: the B75c terms (50 permutations within the classes of B, seed 7530). Point values (no bootstrap).
usage: python3 x75_onerater_decomp.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NP = 50
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "tox", "depth", "unit", "parent_row"])
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
x = np.minimum(P.depth.values, 10).astype(float); N = len(x)
un = pd.factorize(P.unit.values)[0]; ku = np.bincount(un).astype(float); okU = ku >= 2; xu = np.log2(np.maximum(ku, 1))
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
wd = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
xb = xu[okU].mean(); ss = ((xu[okU] - xb) ** 2).sum(); ws = np.where(okU[un], (xu[un] - xb) / (ku[un] * ss), 0.0)
def lrr(v):
    vp, vc = v[pa], v[ch]; a = (vp * vc).sum(); b = vp.sum(); c = ((1 - vp) * vc).sum(); d = (1 - vp).sum()
    return float(np.log((a / b) / (c / d)))
out = []
for s in range(20):
    rng = np.random.default_rng(7510 + s); nA = np.ones_like(n); kA = rng.hypergeometric(kv, n - kv, nA); kB = kv - kA
    y = (kA >= 0.5 - 1e-12).astype(float); z = (kB / np.maximum(n - 1, 1) >= 0.5 - 1e-12).astype(float)
    TPR = (z * y).sum() / z.sum(); FPR = ((1 - z) * y).sum() / (1 - z).sum(); r = {"tpr": float(TPR), "fpr": float(FPR)}
    for nm, w in (("depth", wd), ("size", ws)):
        lZ = float(w @ z); lY = float(w @ y)
        r[nm] = dict(distortion=lY - lZ, uniform=(TPR - FPR - 1) * lZ, fp=float(w @ ((1 - z) * y)) + FPR * lZ, tp=float(w @ (z * y)) - TPR * lZ)
    pos, neg = np.flatnonzero(z == 1), np.flatnonzero(z == 0); pr = np.random.default_rng(7530); pp, ap, pa_ = [], [], []
    for _ in range(NP):
        yp = y.copy(); yp[pos] = pr.permutation(y[pos]); yp[neg] = pr.permutation(y[neg])
        yap = y.copy(); yap[pos] = yp[pos]; ypa = y.copy(); ypa[neg] = yp[neg]
        pp.append(lrr(yp)); ap.append(lrr(yap)); pa_.append(lrr(ypa))
    epp, eap, epa, ly, lz = np.mean(pp), np.mean(ap), np.mean(pa_), lrr(y), lrr(z)
    r["log_rr"] = dict(distortion=ly - lz, uniform=epp - lz, fp=0.5 * ((eap - epp) + (ly - epa)), tp=0.5 * ((epa - epp) + (ly - eap)))
    out.append(r); print(s, {k: ({kk: round(vv * (100 if k != "log_rr" else 1), 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)) for k, v in r.items()}, flush=True)
med = {law: {t: float(np.median([o[law][t] for o in out])) for t in ("distortion", "uniform", "fp", "tp")} for law in ("depth", "size", "log_rr")}
json.dump({"draws": out, "median": med}, open(OUT, "w"), indent=1); print("median", med); print("saved", OUT)
