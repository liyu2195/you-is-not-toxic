"""Addendum 87, B87a: within-article contagion against its within-article permutation null (Civil Comments reply threads).
For each flag vector v (gold, the 12 classifier settings, Detoxify-unbiased, the LLMs on cc_threads, the single rater and gold B,
seed 8503 as B85b/B86a): the within-article Mantel-Haenszel log RR (strata = articles, i.e. clusters; as x86_within_contagion.py)
minus its null, the mean of the same statistic over 50 permutations of v within each article (each article keeps its flag count;
seed 8701). Placebo RR_MH over edges with two gold-negative ends, its null permuting v within (article, gold class). Intervals: 200
article-cluster Poisson resamples (seed 8611), reweighting the per-article terms of the observed and of every permuted vector.
Paired differences instrument - gold (single rater - gold B). Also the between-article share of gold's pooled log RR,
1 - (within-article excess) / (pooled log RR).
usage: python3 x87_within_null.py OUT_JSON"""
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
cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); ecl = cl[ch]
INS = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    INS[f"{mm}@0.5"] = (P[f"s:{mm}@m"].values >= 0.5).astype(float)
rng = np.random.default_rng(7650)                                                   # as x76_rethreshold.py and x86_within_contagion.py
for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
    s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(z.mean() * N))
    order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0; INS[f"{m}@own"] = f
r1 = np.random.default_rng(8503); kA = r1.hypergeometric(kv, n - kv, np.ones_like(n)); A1 = (kA >= 1).astype(float)
gB = ((kv - kA) / np.maximum(n - 1, 1) >= 0.5 - 1e-12).astype(float)
Wb = np.vstack([np.ones(nC), np.random.default_rng(8611).poisson(1.0, size=(NB, nC))])
# permutation orders, shared by all flag vectors: within article (contagion) and within (article, gold class) (placebo)
rp = np.random.default_rng(8701)
grp_c = cl; grp_p = cl * 2 + z.astype(np.int64)
base_c = np.lexsort((np.arange(N), grp_c)); base_p = np.lexsort((np.arange(N), grp_p))
PERM_C = [np.lexsort((rp.random(N), grp_c)) for _ in range(NP)]
PERM_P = [np.lexsort((rp.random(N), grp_p)) for _ in range(NP)]


def permuted(v, base, perm):
    out = np.empty_like(v); out[perm] = v[base]; return out


def terms(v, emask=None):
    vp, vc = v[pa], v[ch]; s_ = ecl
    if emask is not None:
        vp, vc, s_ = vp[emask], vc[emask], s_[emask]
    a = np.bincount(s_, vp * vc, nC); n1 = np.bincount(s_, vp, nC); c = np.bincount(s_, (1 - vp) * vc, nC); n0 = np.bincount(s_, 1 - vp, nC)
    Nn = n1 + n0; ok = (n1 > 0) & (n0 > 0); Nd = np.where(Nn > 0, Nn, 1)
    return np.where(ok, a * n0 / Nd, 0.0), np.where(ok, c * n1 / Nd, 0.0)


def lrr_boot(v, emask=None):
    num, den = terms(v, emask)
    return np.log((Wb @ num) / (Wb @ den))


def excess(v, placebo_ref=None):
    """observed within-article log RR_MH (per resample) minus the mean over permutations within articles (or within article x class)"""
    if placebo_ref is None:
        obs = lrr_boot(v); nul = np.mean([lrr_boot(permuted(v, base_c, pk)) for pk in PERM_C], axis=0)
    else:
        em = (placebo_ref[pa] == 0) & (placebo_ref[ch] == 0)
        obs = lrr_boot(v, em)
        if placebo_ref is z:
            nul = np.mean([lrr_boot(permuted(v, base_p, pk), em) for pk in PERM_P], axis=0)
        else:                                                                    # gold B classes for the single rater
            g2 = cl * 2 + placebo_ref.astype(np.int64); b2 = np.lexsort((np.arange(N), g2)); rq = np.random.default_rng(8702)
            nul = np.mean([lrr_boot(permuted(v, b2, np.lexsort((rq.random(N), g2))), em) for _ in range(NP)], axis=0)
    return obs, nul, obs - nul


ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
res = {"n_edges": int(len(ch)), "n_articles": nC, "instruments": {}}
pooled_gold = float(np.log(((z[pa] * z[ch]).sum() / z[pa].sum()) / (((1 - z[pa]) * z[ch]).sum() / (1 - z[pa]).sum())))
go, gn, ge = excess(z); bo, bn, be = excess(gB)
res["gold"] = dict(rr_mh=ci(np.exp(go)), null_rr=ci(np.exp(gn)), excess=ci(ge), pooled_log_rr=pooled_gold, between_share=float(1 - ge[0] / pooled_gold))
res["gold_B"] = dict(rr_mh=ci(np.exp(bo)), null_rr=ci(np.exp(bn)), excess=ci(be))
print("gold: RR_MH", res["gold"]["rr_mh"], "null", res["gold"]["null_rr"], "excess", res["gold"]["excess"], "between share", round(res["gold"]["between_share"], 3), flush=True)
for name, v in list(INS.items()) + [("single_rater", A1)]:
    ref_e = be if name == "single_rater" else ge; ref = gB if name == "single_rater" else z
    o, nl, e = excess(v); po, pn, pe = excess(v, placebo_ref=ref)
    res["instruments"][name] = dict(flag_rate=float(v.mean()), rr_mh=ci(np.exp(o)), null_rr=ci(np.exp(nl)), excess=ci(e), minus_gold=ci(e - ref_e),
                                    placebo_rr_mh=ci(np.exp(po)), placebo_null=ci(np.exp(pn)), placebo_excess=ci(pe))
    r_ = res["instruments"][name]
    print(f"{name:24s} q {r_['flag_rate']:.3f} RR_MH {r_['rr_mh'][0]:.2f} null {r_['null_rr'][0]:.2f} excess {r_['excess'][0]:+.3f} minus gold {r_['minus_gold'][0]:+.3f} [{r_['minus_gold'][1]:+.3f},{r_['minus_gold'][2]:+.3f}] | placebo {r_['placebo_rr_mh'][0]:.2f} null {r_['placebo_null'][0]:.2f} excess {r_['placebo_excess'][0]:+.3f} [{r_['placebo_excess'][1]:+.3f},{r_['placebo_excess'][2]:+.3f}]", flush=True)
I = res["instruments"]
res["summary"] = dict(n_above=int(sum(I[k]["minus_gold"][1] > 0 for k in CLS12)), n_below=int(sum(I[k]["minus_gold"][2] < 0 for k in CLS12)),
                      above=[k for k in CLS12 if I[k]["minus_gold"][1] > 0], below=[k for k in CLS12 if I[k]["minus_gold"][2] < 0],
                      placebo_excess_min=float(min(I[k]["placebo_excess"][0] for k in CLS12)), placebo_excess_max=float(max(I[k]["placebo_excess"][0] for k in CLS12)),
                      n_placebo_excess_pos=int(sum(I[k]["placebo_excess"][1] > 0 for k in CLS12)))
print("summary", res["summary"], flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
