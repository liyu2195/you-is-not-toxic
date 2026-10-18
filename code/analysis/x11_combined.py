"""Round U1: one gold sample for every law. The edge audit runs first (edges_multi.run_edge, B_E labels); the item audit then
treats every comment the edge audit labeled as already known (exact, and training data for the working model) and spends
B_I new labels (pps.run_seq with prelabeled=...). Compared with the item audit alone at B_I and at B_I + B_E labels.
usage: python3 x11_combined.py POP OUT_JSON --labeler L --aux a,b --reps 300 --seed 6200 --pairs 4000:1000,8000:1000
       [--item-design seq3] [--edge-design m3]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, time, zlib, argparse
import numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code"))
from threadsafe2 import laws as L, pps as PPS, edges as E, gated_edges as G, edges_multi as EM

W = _ROOT
ap = argparse.ArgumentParser(); ap.add_argument("pop"); ap.add_argument("out")
ap.add_argument("--labeler", required=True); ap.add_argument("--aux", default=""); ap.add_argument("--reps", type=int, default=300)
ap.add_argument("--seed", type=int, default=6200); ap.add_argument("--pairs", default="4000:1000"); ap.add_argument("--bayes", action="store_true")
a = ap.parse_args()
P = pd.read_parquet(f"{W}/results/x10_pops/{a.pop}.parquet"); N = len(P); lab = a.labeler
z = P.z.values.astype(float); y = np.nan_to_num(P[f"y:{lab}"].values.astype(float)); s = P[f"s:{lab}"].values
depth = P.depth.values; k = P.k.values.astype(float); xd = np.minimum(depth, 10).astype(float); big = k >= 2; logk = np.where(big, np.log2(np.maximum(k, 1)), 0.0)


def size_phi(unit):
    u = pd.factorize(unit)[0]; ku = np.bincount(u).astype(float); xu = np.log2(np.maximum(ku, 1)); ok = ku >= 2
    xb = xu[ok].mean(); ss = ((xu[ok] - xb) ** 2).sum()
    return np.where(ok[u], (xu[u] - xb) / (ku[u] * ss), 0.0)


ms = [L.Linear("depth_law", (xd - xd.mean()) / ((xd - xd.mean()) ** 2).sum()), L.Linear("size_law", size_phi(P.unit.values))]
art = None
if "article" in P:
    ms.append(L.Linear("article_size_law", size_phi(P.article.values))); art = P.groupby("article").article.transform("size").values
ms += [L.PlaceboSlope("placebo_depth", xd, y, np.ones(N)), L.PlaceboSlope("placebo_size", logk, y, np.where(big, 1.0 / np.maximum(k, 1), 0.0))]
truth = {m.name: L.value(m, z) for m in ms}
LAWS = ("depth_law", "size_law", "article_size_law"); mult = [1.0 if m.name in LAWS else 0.25 for m in ms]
aux = [P[f"s:{e}"].values for e in a.aux.split(",") if e and e.split("@")[0] != lab.split("@")[0]]
cr = np.flatnonzero(P.parent_row.values >= 0); pr = P.parent_row.values[cr]; replies = np.bincount(pr, minlength=N)
lev = PPS.score_levels(s); dbk = L.depth_bucket(depth)
X = PPS.design_matrix(lev, depth, logk, art, aux=aux, replies=replies)
cp = G.ClusterPop(P.parent_row.values, N, y, s); tg, tq = E.EdgePop(pr, cr, y).truth(z)
key = [a.seed, zlib.crc32(a.pop.encode()), zlib.crc32(lab.encode())]


def item(B, rng, pre=None):
    return PPS.run_seq(ms, lev, B, rng, z, X, mult=mult, pilot_share=0.10, K=3, pilot_cells=lev * 5 + dbk, prelabeled=pre, bayes=a.bayes)


def summarize(rows, t):
    R = np.array(rows, float); th, v = R[:, 0], R[:, 1]; ok = np.isfinite(th) & np.isfinite(v) & (v >= 0)
    se = np.sqrt(v[ok]); return dict(coverage=float(np.mean(np.abs(th[ok] - t) <= 1.96 * se)), width=float(np.mean(2 * 1.96 * se)),
                                    rmse=float(np.sqrt(np.mean((th[ok] - t) ** 2))), n=int(ok.sum()))


res = {"pop": a.pop, "labeler": lab, "args": vars(a), "pairs": {}}
t0 = time.time()
for pair in a.pairs.split(","):
    BE, BI = (int(x) for x in pair.split(":"))
    rows = {d: {m.name: [] for m in ms} for d in ("item_alone", "combined", "item_alone_total")}; edge_rows = {mn: [] for mn in E.METRICS}
    used = {d: [] for d in rows}
    for r in range(a.reps):
        rng = np.random.default_rng(key + [BE, BI, r])
        oe = EM.run_edge(cp, z, BE, rng, X, K=3, bayes=a.bayes)
        for j, mn in enumerate(E.METRICS):
            edge_rows[mn].append(oe[mn])
        oc = item(BI, rng, pre=oe["_labeled"]); used["combined"].append(oc["_labels"] + oe["_labels"])
        oa = item(BI, np.random.default_rng(key + [BE, BI, r, 1])); used["item_alone"].append(oa["_labels"])
        ot = item(BI + BE, np.random.default_rng(key + [BE, BI, r, 2])); used["item_alone_total"].append(ot["_labels"])
        for d, o in (("combined", oc), ("item_alone", oa), ("item_alone_total", ot)):
            for m in ms:
                rows[d][m.name].append(o[m.name])
    out = {d: {mn: summarize(rr, truth[mn]) for mn, rr in rows[d].items()} | {"_labels": float(np.mean(used[d]))} for d in rows}
    out["edge"] = {mn: summarize(edge_rows[mn], tg if mn == "log_rr" else tq) for mn in E.METRICS}
    res["pairs"][pair] = out
    print(a.pop, lab, pair, f"{time.time() - t0:.0f}s", json.dumps({d: {mn: (round(v["coverage"], 3), round(v["width"], 6)) for mn, v in out[d].items() if not mn.startswith("_")} for d in out}), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("saved", a.out)
