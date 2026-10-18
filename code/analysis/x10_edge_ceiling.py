"""Round E1 diagnosis: information ceilings of the edge regime with model-based information sets.

Per reply edge e, the influence of its two gold labels on each log risk ratio is the linearized variable v_e(z_p, z_c) at the
full-gold totals. Given a model law of (z_p, z_c) per edge, sigma_e^2 = Var(v_e | I_e), and the best edge-level design at
2 labels per edge has V* = sum sigma_e^2 (1/pi_e - 1), pi_e ~ sigma_e capped at 1, sum pi_e = B / 2 (as in x10_ceiling.py; it
ignores the sharing of comments between edges, so it is a bound only when the sampled share is small).
Information sets:
  I0 the two flags; I1 the two score levels (empirical cell laws, as in x10_ceiling.py);
  I3m gradient boosting, cross-fitted 5-fold: parent P(z = 1 | own score, flag, depth, size) and child
      P(z_c = 1 | child's and parent's score, flag, depth, size, z_p);
  I4m I3m plus the auxiliary instruments' scores of both ends.
Also reported: the per-edge share of each metric's ceiling variance by flag cell.
usage: python3 x10_edge_ceiling.py POP LABELER OUT_JSON [--budgets 2000,4000,8000,16000] [--extra a,b]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code"))
from threadsafe2 import edges as E
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

W = _ROOT
pop, lab, OUT = sys.argv[1:4]
BUD = [int(b) for b in sys.argv[sys.argv.index("--budgets") + 1].split(",")] if "--budgets" in sys.argv else [2000, 4000, 8000, 16000]
EXTRA = sys.argv[sys.argv.index("--extra") + 1].split(",") if "--extra" in sys.argv else []
P = pd.read_parquet(f"{W}/results/x10_pops/{pop}.parquet")
z = P.z.values.astype(float); y = np.nan_to_num(P[f"y:{lab}"].values.astype(float)); s = P[f"s:{lab}"].values.astype(float)
s = np.where(np.isnan(s), np.nanmin(s), s); N = len(P)
depth = P.depth.values; k = P.k.values.astype(float); xd = np.minimum(depth, 10).astype(float); logk = np.where(k >= 2, np.log2(np.maximum(k, 1)), 0.0)
cr = np.flatnonzero(P.parent_row.values >= 0); pr = P.parent_row.values[cr]; Ed = len(pr)
Hz = E.contributions(z[pr], z[cr], y[pr], y[cr]); T = Hz.sum(0)
CELLS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
V = np.zeros((2, Ed, 4))
for q in range(4):
    Hc = E.contributions(np.full(Ed, CELLS[q, 0]), np.full(Ed, CELLS[q, 1]), y[pr], y[cr]); g_, q_ = E.influence(Hc, T); V[0, :, q] = g_; V[1, :, q] = q_


def score_bins(s):
    u = np.unique(s)
    if len(u) <= 8:
        return np.searchsorted(u, s)
    q = np.quantile(s, [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995]); return np.searchsorted(np.unique(q), s, side="right")


def ceiling(sig2, B):
    sg = np.sqrt(sig2); tot = sg.sum()
    if tot <= 0:
        return 0.0
    pi = np.minimum(1.0, B / tot * sg)
    for _ in range(60):
        free = pi < 1.0; left = B - (~free).sum()
        if left <= 0 or not free.any():
            break
        new = np.minimum(1.0, left / sg[free].sum() * sg)
        if np.allclose(new, pi):
            break
        pi = new
    return float((sig2 * (1 / np.maximum(pi, 1e-12) - 1)).sum())


def sig2_from_law(law):
    return [np.maximum((law * V[j] ** 2).sum(1) - ((law * V[j]).sum(1)) ** 2, 0) for j in range(2)]


def cell_law(cell):
    ids, inv = np.unique(cell, return_inverse=True); zz = (2 * z[pr] + z[cr]).astype(int)
    cnt = np.zeros((len(ids), 4)); np.add.at(cnt, (inv, zz), 1); return (cnt / cnt.sum(1, keepdims=True))[inv]


def gb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=61)


SAVE = {}


def model_law(aux_cols, save_as=None):
    """Cross-fitted parent (item) and child (edge) models; returns the per-edge law over the 4 cells."""
    Xi = np.column_stack([s, y, xd, logk] + aux_cols)
    pz = np.zeros(N)
    for tr, te in KFold(5, shuffle=True, random_state=61).split(Xi):
        pz[te] = gb().fit(Xi[tr], z[tr]).predict_proba(Xi[te])[:, 1]
    a = np.clip(pz[pr], 1e-6, 1 - 1e-6)
    Xe = np.column_stack([Xi[cr], Xi[pr], z[pr]]); r1 = np.zeros(Ed); r0 = np.zeros(Ed)
    for tr, te in KFold(5, shuffle=True, random_state=62).split(Xe):
        m = gb().fit(Xe[tr], z[cr][tr]); X1 = Xe[te].copy(); X1[:, -1] = 1; X0 = Xe[te].copy(); X0[:, -1] = 0
        r1[te] = m.predict_proba(X1)[:, 1]; r0[te] = m.predict_proba(X0)[:, 1]
    r1 = np.clip(r1, 1e-6, 1 - 1e-6); r0 = np.clip(r0, 1e-6, 1 - 1e-6)
    if save_as:
        SAVE[save_as] = (a, r1, r0)
    return np.column_stack([(1 - a) * (1 - r0), (1 - a) * r0, a * (1 - r1), a * r1])


sb = score_bins(s)
laws = {"I0": cell_law(2 * y[pr] + y[cr]), "I1": cell_law(sb[pr] * 100 + sb[cr]), "I3m": model_law([])}
if EXTRA:
    laws["I4m"] = model_law([np.nan_to_num(P[f"s:{e}"].values.astype(float), nan=-1.0) for e in EXTRA], save_as="I4m")
if "--save-law" in sys.argv and "I4m" in SAVE:             # the I4m oracle edge law, for the learning diagnostic (round E4)
    import os
    os.makedirs(f"{W}/results/x10_oracle_edge", exist_ok=True); a_, r1_, r0_ = SAVE["I4m"]
    np.savez_compressed(f"{W}/results/x10_oracle_edge/{pop}__{lab.replace('@', '_at_')}.npz", a=a_.astype(np.float32), r1=r1_.astype(np.float32), r0=r0_.astype(np.float32))
fc = (2 * y[pr] + y[cr]).astype(int)
res = {"pop": pop, "labeler": lab, "E": int(Ed), "extra": EXTRA, "edge": {}, "share_by_flag_cell": {}}
for I, law in laws.items():
    g, q = sig2_from_law(law)
    for B in BUD:
        res["edge"].setdefault(I, {})[B] = {"log_rr": ceiling(g, B / 2.0), "log_placebo_rr": ceiling(q, B / 2.0)}
    res["share_by_flag_cell"][I] = {m: [float(np.sqrt(v[fc == c]).sum() / max(np.sqrt(v).sum(), 1e-300)) for c in range(4)] for m, v in (("log_rr", g), ("log_placebo_rr", q))}
json.dump(res, open(OUT, "w"), indent=1)
for I in laws:
    print(pop, lab, I, {B: {m: f"{v:.3g}" for m, v in d.items()} for B, d in res["edge"][I].items()}, "sd share by flag cell", {m: np.round(v, 2).tolist() for m, v in res["share_by_flag_cell"][I].items()})
