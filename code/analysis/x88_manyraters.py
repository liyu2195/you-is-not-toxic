"""Addendum 88, B88c: contagion beyond the raters' construct without the latent model.
Under B83's construct-only hypothesis (an instrument's flag depends on a comment only through its rater probability pi, with
independent noise), a reply's flag is independent of everything about its parent given the reply's pi; the reply's own votes pin pi
as their number grows. Over Civil Comments reply edges whose child has at least n* raters (n* = 4, 6, 10, 20, 50): the
Mantel-Haenszel risk ratio of the child's flag on the parent's flag, strata = the child's exact votes (n_c, k_c); the same with the
parent's gold label in place of its flag. For the 12 classifier settings, Detoxify-unbiased (2), the LLMs on cc_threads, and a single
rater as a construct-only positive control (child: one random vote, strata = its other votes; parent: one random vote of the parent;
seed 8803). 200 article-cluster Poisson resamples (seed 8804).
H88c: at n* = 10 the RR on the parent's flag has a CI above 1 for at least 9 of the 12 classifier settings, while the single rater's CI
at n* = 50 includes 1.
usage: python3 x88_manyraters.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
from scipy import sparse
W = _ROOT; OUT = sys.argv[1]; NB = 200; NSTAR = (4, 6, 10, 20, 50)
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
rng = np.random.default_rng(7650)                                                   # as x76_rethreshold.py, x86 and x87
for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
    s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(z.mean() * N))
    order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0; INS[f"{m}@own"] = f
r1 = np.random.default_rng(8803); kA = r1.hypergeometric(kv, n - kv, np.ones_like(n)); A1 = (kA >= 1).astype(float)
Wb = np.vstack([np.ones(nC), np.random.default_rng(8804).poisson(1.0, size=(NB, nC))]).T          # clusters x (1 + NB)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]


def mh(vc, vp, sn, sk, emask):
    """MH RR of vc on vp within strata (sn, sk), over edges in emask; point value and the NB resamples."""
    e = np.flatnonzero(emask); s = pd.factorize(pd.MultiIndex.from_arrays([sn[e], sk[e]]))[0]; S = int(s.max() + 1)
    cell = (vp[e] * 2 + vc[e]).astype(np.int64)                                    # 0: p0c0, 1: p0c1, 2: p1c0, 3: p1c1
    M = sparse.csr_matrix((np.ones(len(e)), (s * 4 + cell, ecl[e])), shape=(S * 4, nC))
    C = (M @ Wb).reshape(S, 4, -1)                                                 # strata x cells x resamples
    a = C[:, 3]; n1 = C[:, 2] + C[:, 3]; c = C[:, 1]; n0 = C[:, 0] + C[:, 1]; Nn = n0 + n1
    ok = (n1 > 0) & (n0 > 0); Nd = np.where(Nn > 0, Nn, 1)
    num = np.where(ok, a * n0 / Nd, 0).sum(0); den = np.where(ok, c * n1 / Nd, 0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den, len(e)


res = {"n_edges": int(len(ch)), "nstar": list(NSTAR), "instruments": {}}
nc_, kc_ = n[ch], kv[ch]
for name, v in list(INS.items()) + [("single_rater", A1)]:
    out = {}
    if name == "single_rater":
        vc, vp = A1[ch], A1[pa]; sn, sk = nc_ - 1, kc_ - kA[ch]
    else:
        vc, vp = v[ch], v[pa]; sn, sk = nc_, kc_
    for ns in NSTAR:
        em = nc_ >= ns
        r_f, ne = mh(vc, vp, sn, sk, em); r_g, _ = mh(vc, z[pa], sn, sk, em)
        out[str(ns)] = dict(n_edges=int(ne), rr_parent_flag=ci(r_f), rr_parent_gold=ci(r_g))
    res["instruments"][name] = out
    print(f"{name:24s}", " | ".join(f"n*={ns}: {out[str(ns)]['rr_parent_flag'][0]:.2f} [{out[str(ns)]['rr_parent_flag'][1]:.2f},{out[str(ns)]['rr_parent_flag'][2]:.2f}] g {out[str(ns)]['rr_parent_gold'][0]:.2f}" for ns in NSTAR), flush=True)
I = res["instruments"]
res["H88c"] = dict(n_cls_ci_above_1_at_10=int(sum(I[k]["10"]["rr_parent_flag"][1] > 1 for k in CLS12)),
                   single_rater_at_50=I["single_rater"]["50"]["rr_parent_flag"])
res["H88c"]["pass"] = bool(res["H88c"]["n_cls_ci_above_1_at_10"] >= 9 and res["H88c"]["single_rater_at_50"][1] <= 1 <= res["H88c"]["single_rater_at_50"][2])
print("H88c", res["H88c"], flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
