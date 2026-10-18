"""Addendum 61: information ceilings of the item and edge regimes for one population and labeler.

For each metric, the influence of item i's gold label at the full-gold value is z_i * phi_i (item metrics) or v_e(z_p, z_c)
(edge metrics). Given an information set I with per-item gold probabilities p_i = P(z_i = 1 | I_i), the best unbiased design
using I (difference estimator, inclusion probabilities pi_i ~ sigma_i, capped at 1) has variance
    V*_I(B) = sum_i sigma_i^2 (1/pi_i - 1),   sigma_i^2 = p_i (1 - p_i) phi_i^2     (items)
and, per edge, sigma_e^2 = Var(v_e | I_e) at cost 2 per edge (edges). Metric sets use the compromise pi ~ sqrt(sum_m w_m sigma_im^2)
with the laws-first weights normalized by each metric's own optimum.
Information sets: I0 flag; I1 own score (levels, or quantile bins of a continuous score refined at the top); I2 I1 x depth x
size buckets (cells shrunk toward I1 with weight 20); I3 gradient boosting on score + structure, cross-fitted 5-fold.
usage: python3 x10_ceiling.py POP LABELER OUT_JSON [--budgets 1000,4000] [--extra lab1,lab2]   (I4: plus auxiliary instruments)"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code"))
from threadsafe2 import laws as L, edges as E
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

W = _ROOT
pop, lab, OUT = sys.argv[1:4]
BUD = [int(b) for b in sys.argv[sys.argv.index("--budgets") + 1].split(",")] if "--budgets" in sys.argv else [1000, 4000]
PD = sys.argv[sys.argv.index("--pops-dir") + 1] if "--pops-dir" in sys.argv else "x10_pops"
P = pd.read_parquet(f"{W}/results/{PD}/{pop}.parquet")
z = P.z.values.astype(float); y = np.nan_to_num(P[f"y:{lab}"].values.astype(float)); s = P[f"s:{lab}"].values.astype(float)
s = np.where(np.isnan(s), np.nanmin(s), s)
N = len(P); depth = P.depth.values; k = P.k.values.astype(float); xd = np.minimum(depth, 10).astype(float)
big = k >= 2; logk = np.where(big, np.log2(np.maximum(k, 1)), 0.0)


def size_phi(unit):
    u = pd.factorize(unit)[0]; ku = np.bincount(u).astype(float); xu = np.log2(np.maximum(ku, 1)); ok = ku >= 2
    xb = xu[ok].mean(); ss = ((xu[ok] - xb) ** 2).sum()
    return np.where(ok[u], (xu[u] - xb) / (ku[u] * ss), 0.0)


ms = [L.Linear("depth_law", (xd - xd.mean()) / ((xd - xd.mean()) ** 2).sum()), L.Linear("size_law", size_phi(P.unit.values))]
if "article" in P:
    ms.append(L.Linear("article_size_law", size_phi(P.article.values)))
ms += [L.PlaceboSlope("placebo_depth", xd, y, np.ones(N)), L.PlaceboSlope("placebo_size", logk, y, np.where(big, 1.0 / np.maximum(k, 1), 0.0))]
LAWS = {"depth_law", "size_law", "article_size_law"}
PHI = {}
for m in ms:
    th, c = m.theta_c(z @ m.B); PHI[m.name] = m.B @ c


def score_bins(s):
    u = np.unique(s)
    if len(u) <= 8:
        return np.searchsorted(u, s)
    q = np.quantile(s, [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995]); return np.searchsorted(np.unique(q), s, side="right")


def cell_rate(cell, shrink_to=None, kappa=20.0):
    ids, inv = np.unique(cell, return_inverse=True); n = np.bincount(inv).astype(float); x = np.bincount(inv, weights=z)
    p = x / n
    if shrink_to is not None:
        prior = np.bincount(inv, weights=shrink_to) / n; p = (x + kappa * prior) / (n + kappa)
    return p[inv]


sb = score_bins(s); db = L.depth_bucket(depth); zb = np.searchsorted([2, 3, 4, 6], logk, side="right")
INFO = {"I0": cell_rate(y.astype(int)), "I1": cell_rate(sb)}
INFO["I2"] = cell_rate((sb * 5 + db) * 5 + zb, shrink_to=INFO["I1"])
X = np.column_stack([s, y, xd, logk] + ([np.log2(np.maximum(P.groupby("article").article.transform("size").values, 1))] if "article" in P else []))
def cv_model(X):
    out = np.zeros(N)
    for tr, te in KFold(5, shuffle=True, random_state=61).split(X):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=61).fit(X[tr], z[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return np.clip(out, 1e-6, 1 - 1e-6)
INFO["I3"] = cv_model(X)
EXTRA = sys.argv[sys.argv.index("--extra") + 1].split(",") if "--extra" in sys.argv else []
if EXTRA:   # I4: the audited labeler plus auxiliary instruments' scores (committed side information)
    Xa = np.column_stack([X] + [np.nan_to_num(P[f"s:{e}"].values.astype(float), nan=-1.0) for e in EXTRA])
    INFO["I4"] = cv_model(Xa)


def ceiling(sig2, B):
    """V* = sum sigma^2 (1/pi - 1) with pi = min(1, c * sigma), sum pi = B."""
    sg = np.sqrt(sig2); tot = sg.sum()
    if tot <= 0:
        return 0.0
    c = B / tot; pi = np.minimum(1.0, c * sg)
    for _ in range(50):                                                   # water-fill the caps
        free = pi < 1.0; left = B - (~free).sum()
        if left <= 0 or not free.any():
            break
        c = left / sg[free].sum(); new = np.minimum(1.0, c * sg)
        if np.allclose(new, pi):
            break
        pi = new
    pi = np.maximum(pi, 1e-12)
    return float((sig2 * (1 / pi - 1)).sum())


res = {"pop": pop, "labeler": lab, "N": int(N), "item": {}, "edge": {}}
for I, p in INFO.items():
    S2 = {mn: p * (1 - p) * PHI[mn] ** 2 for mn in PHI}
    for B in BUD:
        own = {mn: ceiling(S2[mn], B) for mn in S2}
        w = {mn: (1.0 if mn in LAWS else 0.25) / max(own[mn], 1e-300) for mn in S2}
        comp = sum(w[mn] * S2[mn] for mn in S2); sg = np.sqrt(comp); c = B / sg.sum(); pi = np.minimum(1.0, c * sg)
        for _ in range(50):
            free = pi < 1.0; left = B - (~free).sum(); c = left / max(sg[free].sum(), 1e-300); new = np.minimum(1.0, c * sg)
            if np.allclose(new, pi):
                break
            pi = new
        pi = np.maximum(pi, 1e-12)
        res["item"].setdefault(I, {})[B] = {mn: dict(own=own[mn], compromise=float((S2[mn] * (1 / pi - 1)).sum())) for mn in S2}
# ---- edges: per-edge linearized variables of the two log risk ratios, joint (z_p, z_c) law per information cell
cr = np.flatnonzero(P.parent_row.values >= 0); pr = P.parent_row.values[cr]
pop_e = E.EdgePop(pr, cr, y); Hz = E.contributions(z[pr], z[cr], y[pr], y[cr]); T = Hz.sum(0)
cells4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)


def edge_sig2(cell):
    """Per-edge Var(v_e | cell) for both log RRs: the cell's empirical joint law of (z_p, z_c), each edge's own flags."""
    ids, inv = np.unique(cell, return_inverse=True); zz = (2 * z[pr] + z[cr]).astype(int)
    cnt = np.zeros((len(ids), 4)); np.add.at(cnt, (inv, zz), 1); law = (cnt / cnt.sum(1, keepdims=True))[inv]
    V = np.zeros((2, len(pr), 4))
    for q in range(4):
        Hc = E.contributions(np.full(len(pr), cells4[q, 0]), np.full(len(pr), cells4[q, 1]), y[pr], y[cr])
        g_, q_ = E.influence(Hc, T); V[0, :, q] = g_; V[1, :, q] = q_
    return [np.maximum((law * V[j] ** 2).sum(1) - ((law * V[j]).sum(1)) ** 2, 0) for j in range(2)]


ecells = {"I0": 2 * y[pr] + y[cr], "I1": sb[pr] * 100 + sb[cr]}
ecells["I2"] = (sb[pr] * 100 + sb[cr]) * 25 + db[cr] * 5 + zb[cr]
for I, cell in ecells.items():
    g, q = edge_sig2(cell)
    for B in [b * 4 for b in BUD]:
        res["edge"].setdefault(I, {})[B] = {"log_rr": ceiling(g, B / 2.0), "log_placebo_rr": ceiling(q, B / 2.0)}   # 2 labels per edge
json.dump(res, open(OUT, "w"), indent=1)
for I in INFO:
    print(pop, lab, I, {B: {mn: f"{v['own']:.3g}/{v['compromise']:.3g}" for mn, v in d.items()} for B, d in res["item"][I].items()})
for I in ecells:
    print(pop, lab, "edge", I, {B: {mn: f"{v:.3g}" for mn, v in d.items()} for B, d in res["edge"][I].items()})
