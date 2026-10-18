"""Addendum 80, B80c: a less noisy gold on representative edges (Civil Comments reply threads).
(i) Edges whose two ends have >= 10 raters (the B77c set) are reweighted to the full population by inverse probability weights from a
    logistic model of being in the set on the parent's and child's vote-fraction bins (0, (0, .2), [.2, .5), [.5, .8), >= .8), capped
    depth, log2 thread size and publication. On the reweighted edges: the k-rater majority RR (k = 1, 3, 5, 7, 9; raters drawn without
    replacement, toxic when at least half are; median of 20 draws, seed 8020), the full majority (gold), the 1/k extrapolation to
    k -> inf (log RR = a + b/k over the five k and the full majority at its mean rater count, as B77c), and every instrument's RR.
(ii) All edges: the k-rater majority RR for k = 1, 2, 3 (ties at k = 2 broken at random), 20 draws, seed 8021, and the 1/k fit over
    k = 1, 2, 3 (descriptive).
H80d: on the reweighted edges, the extrapolated noise-free RR is below the RR of at least 9 of the 12 classifier settings.
usage: python3 x80_noise.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
W = _ROOT; OUT = sys.argv[1]; K = 10
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count", "publication_id"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
pub = meta.publication_id.reindex(P.id.values).fillna(-1).astype(int).values
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
uu = pd.factorize(P.unit.values)[0]; tsize = np.bincount(uu)[uu]
def vbin(t):
    return np.digitize(t, [1e-9, 0.2, 0.5, 0.8])
sel = (n[ch] >= K) & (n[pa] >= K)
X = pd.DataFrame({"bp": vbin(P.tox.values[pa]).astype(str), "bc": vbin(P.tox.values[ch]).astype(str), "d": np.minimum(P.depth.values[ch], 10).astype(str),
                  "pub": pub[ch].astype(str)})
Xd = pd.get_dummies(X, drop_first=True).astype(float); Xd["lsize"] = np.log2(tsize[ch])
lr = LogisticRegression(max_iter=2000, C=10.0).fit(Xd.values, sel.astype(int)); ph = lr.predict_proba(Xd.values)[:, 1]
w = np.where(sel, sel.mean() / np.maximum(ph, 1e-6), 0.0)
print("edges", len(ch), "selected", int(sel.sum()), "weight quantiles", np.quantile(w[sel], [0, .5, .9, .99, 1]).round(2), flush=True)
def rr(v, ws, mask):
    a, b, ww = v[pa][mask], v[ch][mask], ws[mask]
    return float((np.sum(ww * a * b) / np.sum(ww * a)) / (np.sum(ww * (1 - a) * b) / np.sum(ww * (1 - a))))
z = P.z.values.astype(float); one = np.ones(len(ch))
res = {"n_edges": int(len(ch)), "n_selected": int(sel.sum()), "weights": {"max": float(w[sel].max()), "ess": float(w[sel].sum() ** 2 / (w[sel] ** 2).sum())}}
# balance check: the reweighted selected edges against all edges
bal = {}
for nm, v in (("parent_gold", z[pa]), ("child_gold", z[ch]), ("parent_vote", P.tox.values[pa]), ("depth", np.minimum(P.depth.values[ch], 10).astype(float)), ("log2_size", np.log2(tsize[ch]))):
    bal[nm] = dict(all=float(v.mean()), selected=float(v[sel].mean()), reweighted=float(np.sum(w[sel] * v[sel]) / np.sum(w[sel])))
res["balance"] = bal; print(json.dumps(bal), flush=True)
rows = np.unique(np.r_[ch[sel], pa[sel]])
rw = {"gold_rr_full_population": rr(z, one, np.ones(len(ch), bool)), "gold_rr_selected": rr(z, one, sel), "gold_rr_reweighted": rr(z, w, sel), "k": {}}
rng = np.random.default_rng(8020)
for k in (1, 3, 5, 7, 9):
    vals = []
    for _ in range(20):
        t = rng.hypergeometric(kv, n - kv, np.minimum(k, n)); y = (t / np.minimum(k, n) >= 0.5 - 1e-12).astype(float); vals.append(rr(y, w, sel))
    rw["k"][k] = float(np.median(vals))
kk = np.array([1, 3, 5, 7, 9, n[rows].mean()]); ll = np.log(np.array([rw["k"][k] for k in (1, 3, 5, 7, 9)] + [rw["gold_rr_reweighted"]]))
A = np.column_stack([np.ones(len(kk)), 1 / kk]); coef = np.linalg.lstsq(A, ll, rcond=None)[0]; rw["rr_inf"] = float(np.exp(coef[0])); rw["fit_b"] = float(coef[1])
ins = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    ins[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
rw["instruments"] = {k: rr(v, w, sel) for k, v in ins.items()}
res["reweighted"] = rw
up = [c for c in CLS12 if rw["instruments"][c] > rw["rr_inf"]]
res["H80d"] = dict(rr_inf=rw["rr_inf"], n_above=len(up), settings=up, pass_=len(up) >= 9)
print(json.dumps({k: v for k, v in rw.items() if k != "instruments"}), flush=True); print({k: round(v, 2) for k, v in rw["instruments"].items()}, flush=True)
# (ii) all edges, k = 1, 2, 3
full = {"k": {}}; rng2 = np.random.default_rng(8021); allm = np.ones(len(ch), bool)
for k in (1, 2, 3):
    vals = []
    for _ in range(20):
        t = rng2.hypergeometric(kv, n - kv, np.minimum(k, n)); fr = t / np.minimum(k, n)
        y = np.where(np.abs(fr - 0.5) < 1e-12, (rng2.random(len(fr)) < 0.5).astype(float), (fr > 0.5).astype(float)); vals.append(rr(y, one, allm))
    full["k"][k] = float(np.median(vals))
kk = np.array([1, 2, 3]); ll = np.log(np.array([full["k"][k] for k in (1, 2, 3)])); A = np.column_stack([np.ones(3), 1 / kk]); coef = np.linalg.lstsq(A, ll, rcond=None)[0]
full["rr_inf_k123"] = float(np.exp(coef[0])); full["gold_rr"] = rw["gold_rr_full_population"]; res["full_population"] = full
print("full population", json.dumps(full)); print("H80d", res["H80d"])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
