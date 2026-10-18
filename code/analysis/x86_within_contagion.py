"""Addendum 86, B86a: contagion within articles and within threads (Civil Comments reply threads, all reply edges).
Statistic: the Mantel-Haenszel reply risk ratio stratified by article (cluster) or by reply thread (unit),
    RR_MH = [sum_s a_s n0_s / N_s] / [sum_s c_s n1_s / N_s]
(a_s: edges with flagged parent and child; n1_s: edges with a flagged parent; c_s: unflagged parent, flagged child; n0_s: unflagged
parent; N_s = n0_s + n1_s; strata without both kinds of parent drop out), and the pooled RR for reference. For gold, the 12
classifier settings, Detoxify-unbiased (2), the LLMs on cc_threads and the single rater (one draw, seed 8503, against the others'
majority gold B, whose own RR_MH is also reported). Decomposition as B75c: flags permuted within gold positives and within gold
negatives over the whole population, 50 permutations (seed 8601): uniform = E lRR(Y_pp) - lRR(Z); structured false positives and
hits by the two-player Shapley split of the remainder. Placebo RR_MH over edges with two gold-negative ends. Intervals: 200
article-cluster Poisson resamples (seed 8611); a resampled article repeats its strata (for threads, all threads of the article).
H86a: within articles, S_FP + S_TP > 0 for >= 9 of the 12 classifier settings (point values).
usage: python3 x86_within_contagion.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NP = 50; NB = 200
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
z = P.z.values.astype(float); N = len(z)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
un = pd.factorize(P.unit.values)[0]; nU = int(un.max() + 1); ucl = np.zeros(nU, np.int64); ucl[un] = cl
STRATA = {"article": (cl[ch], nC, np.arange(nC)), "thread": (un[ch], nU, ucl), "pooled": (np.zeros(len(ch), np.int64), 1, None)}
# instruments
INS = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    INS[f"{mm}@0.5"] = (P[f"s:{mm}@m"].values >= 0.5).astype(float)
rng = np.random.default_rng(7650)                                                   # as x76_rethreshold.py (B76e)
for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
    s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(z.mean() * N))
    order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0; INS[f"{m}@own"] = f
r1 = np.random.default_rng(8503); kA = r1.hypergeometric(kv, n - kv, np.ones_like(n)); A1 = (kA >= 1).astype(float)
gB = ((kv - kA) / np.maximum(n - 1, 1) >= 0.5 - 1e-12).astype(float)
Wb = np.vstack([np.ones(nC), np.random.default_rng(8611).poisson(1.0, size=(NB, nC))])


def terms(v, sk, emask=None):
    """per-stratum MH numerator and denominator terms (and the stratum ids' cluster map applies later)"""
    s_, nS, _ = STRATA[sk]; vp, vc = v[pa], v[ch]
    if emask is not None:
        s_ = s_[emask]; vp = vp[emask]; vc = vc[emask]
    a = np.bincount(s_, vp * vc, nS); n1 = np.bincount(s_, vp, nS); c = np.bincount(s_, (1 - vp) * vc, nS); n0 = np.bincount(s_, 1 - vp, nS)
    Nn = n1 + n0; ok = (n1 > 0) & (n0 > 0)
    num = np.where(ok, a * n0 / np.where(Nn > 0, Nn, 1), 0.0); den = np.where(ok, c * n1 / np.where(Nn > 0, Nn, 1), 0.0)
    return num, den


def lrr(v, sk, emask=None):
    num, den = terms(v, sk, emask)
    return float(np.log(num.sum() / den.sum())) if den.sum() > 0 and num.sum() > 0 else np.nan


def lrr_boot(v, sk, emask=None):
    num, den = terms(v, sk, emask); _, nS, smap = STRATA[sk]
    if sk == "pooled":                                                           # pooled: per-cluster sums of the four counts
        vp, vc = v[pa], v[ch]; e = cl[ch]
        if emask is not None:
            vp, vc, e = vp[emask], vc[emask], e[emask]
        a, b, c, d = (np.bincount(e, w_, nC) for w_ in (vp * vc, vp, (1 - vp) * vc, 1 - vp))
        A, B, C, D = Wb @ a, Wb @ b, Wb @ c, Wb @ d
        return np.log((A / B) / (C / D))
    numc = np.bincount(smap, num, nC); denc = np.bincount(smap, den, nC)
    return np.log((Wb @ numc) / (Wb @ denc))


ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
neg_e = (z[pa] == 0) & (z[ch] == 0)
pos = np.flatnonzero(z == 1); negi = np.flatnonzero(z == 0); rp = np.random.default_rng(8601)
PERMS = [(rp.permutation(len(pos)), rp.permutation(len(negi))) for _ in range(NP)]
res = {"n_edges": int(len(ch)), "n_articles": nC, "n_threads": nU, "strata": {}}
for sk in ("article", "thread", "pooled"):
    R = {"gold": {}, "gold_B": {}, "instruments": {}}
    gb_ = lrr_boot(z, sk); R["gold"] = dict(rr=[float(np.exp(x)) for x in ci(gb_)])
    R["gold_B"] = dict(rr=float(np.exp(lrr(gB, sk))))
    for name, v in list(INS.items()) + [("single_rater", A1)]:
        ref = gB if name == "single_rater" else z
        lv = lrr_boot(v, sk); lg = lrr_boot(ref, sk) if name == "single_rater" else gb_
        out = dict(rr=[float(np.exp(x)) for x in ci(lv)], minus_gold=ci(lv - lg), placebo_rr=float(np.exp(lrr(v, sk, (ref[pa] == 0) & (ref[ch] == 0)))))
        if name != "single_rater":
            vpp, vap, vpa = [], [], []
            for pp_, pn_ in PERMS:
                yp = v.copy(); yp[pos] = v[pos][pp_]; yp[negi] = v[negi][pn_]
                ya = v.copy(); ya[pos] = v[pos][pp_]                                     # actual false positives, permuted hits
                yb = v.copy(); yb[negi] = v[negi][pn_]                                   # permuted false positives, actual hits
                vpp.append(lrr(yp, sk)); vap.append(lrr(ya, sk)); vpa.append(lrr(yb, sk))
            Epp, Eap, Epa = np.nanmean(vpp), np.nanmean(vap), np.nanmean(vpa); lY = lv[0]; lZ = gb_[0]
            out.update(uniform=float(Epp - lZ), fp=float(0.5 * ((Eap - Epp) + (lY - Epa))), tp=float(0.5 * ((Epa - Epp) + (lY - Eap))))
            out["check"] = float(out["uniform"] + out["fp"] + out["tp"] - (lY - lZ))
        R["instruments"][name] = out
    R["gold"]["placebo_rr"] = 1.0
    st_ = [R["instruments"][k]["fp"] + R["instruments"][k]["tp"] for k in CLS12]
    R["n_cls_structured_pos"] = int(sum(x > 0 for x in st_))
    R["n_cls_above_gold"] = int(sum(R["instruments"][k]["minus_gold"][1] > 0 for k in CLS12)); R["n_cls_below_gold"] = int(sum(R["instruments"][k]["minus_gold"][2] < 0 for k in CLS12))
    res["strata"][sk] = R
    print(sk, "gold RR", R["gold"]["rr"], "| structured>0:", R["n_cls_structured_pos"], "/12 | above gold:", R["n_cls_above_gold"], "below:", R["n_cls_below_gold"], flush=True)
    for k in CLS12 + ["haiku", "haiku_ctx", "qwen7b_m", "single_rater"]:
        o = R["instruments"][k]
        print(f"   {k:24s} RR {o['rr'][0]:.2f} [{o['rr'][1]:.2f},{o['rr'][2]:.2f}] minus gold {o['minus_gold'][0]:+.3f} [{o['minus_gold'][1]:+.3f},{o['minus_gold'][2]:+.3f}] placebo {o['placebo_rr']:.2f}"
              + (f" | uni {o['uniform']:+.3f} fp {o['fp']:+.3f} tp {o['tp']:+.3f} chk {o['check']:.1e}" if "uniform" in o else ""), flush=True)
res["H86a"] = dict(n_structured_pos=res["strata"]["article"]["n_cls_structured_pos"], pass_=bool(res["strata"]["article"]["n_cls_structured_pos"] >= 9))
print("H86a", res["H86a"])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
