"""Addendum 75, B75c: a uniform-error baseline for contagion, and B75d: contagion of severe toxicity.
For each labeler of results/x14_pops/<pop>.parquet (plus the fresh classifiers at 0.5): its flags are permuted at random within gold
positives and within gold negatives (TPR and FPR kept), 50 permutations (seed 7530). With Y_pp the permuted flags, Y_ap the actual
flags on gold negatives and permuted ones on gold positives, and Y_pa the reverse,
    log RR(Y) - log RR(Z) = [E log RR(Y_pp) - log RR(Z)]                                    uniform errors
                          + 1/2 [(E lRR(Y_ap) - E lRR(Y_pp)) + (lRR(Y) - E lRR(Y_pa))]       structured false positives
                          + 1/2 [(E lRR(Y_pa) - E lRR(Y_pp)) + (lRR(Y) - E lRR(Y_ap))]       structured hits (minus structured misses)
exactly (a two-player Shapley split of the remainder). For linear laws the same construction gives (TPR - FPR) L(Z), S_FP and S_TP
in expectation (Proposition 1), which x74_decomp.py reports exactly. B75d (Civil Comments): gold's RR with gold redefined at vote
thresholds 0.5-0.8, and each classifier's log-RR distortion against each.
usage: python3 x75_decomp.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NP = 50
FRESH = ["sn_rob", "mh_dist", "fb_hate", "cn_off"]
res = {}
for pop in ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "bad"]:
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(float)
    labs = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    for m in FRESH:
        if f"s:{m}@m" in P:
            labs[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
    ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
    def lrr(v):
        vp, vc = v[pa], v[ch]; a = (vp * vc).sum(); b = vp.sum(); c = ((1 - vp) * vc).sum(); d = (1 - vp).sum()
        return float(np.log((a / b) / (c / d))) if min(a, b, c, d) > 0 else np.nan
    pos, neg = np.flatnonzero(z == 1), np.flatnonzero(z == 0); lz = lrr(z); out = {"gold_log_rr": lz, "labelers": {}}
    rng = np.random.default_rng(7530)
    for L, y in labs.items():
        pp, ap, pa_ = [], [], []
        for _ in range(NP):
            yp = y.copy(); yp[pos] = rng.permutation(y[pos]); yp[neg] = rng.permutation(y[neg])
            yap = y.copy(); yap[pos] = yp[pos]                 # actual false positives, permuted hits
            ypa = y.copy(); ypa[neg] = yp[neg]                 # permuted false positives, actual hits
            pp.append(lrr(yp)); ap.append(lrr(yap)); pa_.append(lrr(ypa))
        epp, eap, epa, ly = np.nanmean(pp), np.nanmean(ap), np.nanmean(pa_), lrr(y)
        uni = epp - lz; fp = 0.5 * ((eap - epp) + (ly - epa)); tp = 0.5 * ((epa - epp) + (ly - eap))
        out["labelers"][L] = dict(distortion=ly - lz, uniform=uni, fp=fp, tp=tp, check=abs(uni + fp + tp - (ly - lz)), perm_sd=float(np.nanstd(pp)))
    res[pop] = out
    print(pop, "gold log RR", round(lz, 3), {L: (round(v["distortion"], 3), round(v["uniform"], 3), round(v["fp"], 3), round(v["tp"], 3)) for L, v in out["labelers"].items() if not L.endswith("@0.5")}, flush=True)
# B75d: severity
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet"); tox = P.tox.values
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]
def lrr2(v):
    vp, vc = v[pa], v[ch]; return float(np.log((vc[vp == 1].mean()) / (vc[vp == 0].mean())))
sev = {}
for t in (0.5, 0.6, 0.7, 0.8):
    g = (tox >= t - 1e-9).astype(float); sev[t] = {"prev": float(g.mean()), "gold_rr": float(np.exp(lrr2(g)))}
    for L in ["detoxify_original@0.6", "detoxify_original@m", "sn_rob@m", "mh_dist@m", "fb_hate@m", "cn_off@m", "imsypp@m", "haiku", "haiku_ctx"]:
        sev[t][L] = float(lrr2(np.nan_to_num(P[f"y:{L}"].values.astype(float))) - lrr2(g))
res["severity"] = sev
print("severity", {t: (round(v["prev"], 4), round(v["gold_rr"], 2)) for t, v in sev.items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
