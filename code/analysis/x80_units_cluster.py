"""Addendum 80, B80g: unit shares against cluster PPI on the 55 confirmation cells (the cells of results/x11_conf_unit/).
Cluster PPI (as f98_threadsafe2_sim.py): units (reply threads, conversations, dialogues) in uniformly random order while the
cumulative cost (comments labeled = unit size) stays within the budget (at least 2 units); every comment of a sampled unit is labeled;
PPI++ mean of the unit's gold answer with the labeler's unit answer as control variate (CV1) or the vector control variate (CV4: unit
answer, flag count capped at 5, log size, maximum score); Wald interval, Korn-Graubard below ten effective positive or negative units.
500 audits per budget (250, 500, 1,000, 2,000), seed 8030 + a hash of the cell. Compared with THREADSAFE's stored confirmation
widths (ts_rank) in the same cells and budgets.
H80h: ts_rank's width is below cluster PPI's (the narrower of CV1 and CV4) at 500 labels and above in at least 80% of cells.
usage: python3 x80_units_cluster.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, os, zlib, numpy as np, pandas as pd
from scipy.stats import beta as _beta
W = _ROOT; OUT = sys.argv[1]; REPS = 500; BUD = [250, 500, 1000, 2000]
def kg(est, se, n):
    p = float(np.clip(est, 0, 1)); v = max(se ** 2, 1e-15); ne = n if p in (0.0, 1.0) else min(n, p * (1 - p) / v); x = ne * p
    lo = _beta.ppf(0.025, max(x, 1e-12), ne - x + 1) if x > 0 else 0.0; hi = _beta.ppf(0.975, x + 1, max(ne - x, 1e-12)) if x < ne else 1.0
    return lo, hi
def ppi(t, X, mu, Sig, N):
    n = len(t)
    if n < X.shape[1] + 3:
        return t.mean(), t.std(ddof=1) / np.sqrt(n)
    Xc = X - X.mean(0); Cxx = Xc.T @ Xc / (n - 1); Cxt = Xc.T @ (t - t.mean()) / (n - 1)
    beta = np.linalg.lstsq(Cxx * (1 + n / N) + 1e-10 * np.eye(len(mu)), Cxt, rcond=None)[0]
    est = t.mean() - (X.mean(0) - mu) @ beta; res = t - X @ beta
    return est, np.sqrt(np.var(res, ddof=1) / n + beta @ Sig @ beta / N)
out = {"cells": {}}
for f in sorted(glob.glob(f"{W}/results/x11_conf_unit/*__*.json")):
    if os.path.basename(f).startswith("eval"):
        continue
    C = json.load(open(f)); pop, lab = C["pop"], C["labeler"]; tag = os.path.basename(f)[:-5]
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet", columns=["unit", "z", f"y:{lab}", f"s:{lab}"])
    u = pd.factorize(P.unit.values)[0]; N = int(u.max() + 1); k = np.bincount(u).astype(float)
    gold = np.zeros(N); np.maximum.at(gold, u, P.z.values.astype(float)); theta = float(gold.mean())
    y = np.nan_to_num(P[f"y:{lab}"].values.astype(float)); s = np.nan_to_num(P[f"s:{lab}"].values.astype(float), nan=-1.0)
    Yt = np.zeros(N); np.maximum.at(Yt, u, y); nfl = np.minimum(np.bincount(u, y, N), 5); smax = np.full(N, -np.inf); np.maximum.at(smax, u, s)
    X1 = Yt[:, None]; X4 = np.column_stack([Yt, nfl, np.log(k), smax]); POP = {1: (X1.mean(0), np.atleast_2d(np.cov(X1.T))), 4: (X4.mean(0), np.cov(X4.T))}
    rng = np.random.default_rng([8030, zlib.crc32(tag.encode())]); cell = {"theta": theta, "n_units": N, "unit": {}}
    for B in BUD:
        r = {}
        for nm, X, p in (("cluster_ppi", X1, 1), ("cluster_ppi4", X4, 4)):
            cov = []; wid = []
            for _ in range(REPS):
                order = rng.permutation(N); n = max(2, int(np.searchsorted(np.cumsum(k[order]), B, side="right"))); ix = order[:n]; t = gold[ix]
                est, se = ppi(t, X[ix], *POP[p], N)
                npos, nneg = t.sum(), n - t.sum()
                if min(npos, nneg) < 10:
                    lo, hi = kg(est, se, n)
                else:
                    lo, hi = est - 1.96 * se, est + 1.96 * se
                cov.append(lo <= theta <= hi); wid.append(hi - lo)
            r[nm] = dict(coverage=float(np.mean(cov)), width=float(np.mean(wid)))
        ts = C["unit"].get(str(B), {}).get("ts_rank", {}); best = min(r.values(), key=lambda v: v["width"])
        r["ts_rank"] = dict(coverage=ts.get("coverage"), width=ts.get("width")); r["ratio_ts_to_best_cluster"] = (ts["width"] / best["width"]) if ts.get("width") else None
        cell["unit"][str(B)] = r
    out["cells"][tag] = cell
    print(tag, {B: (round(v["cluster_ppi"]["coverage"], 3), round(v["cluster_ppi4"]["coverage"], 3), round(v["ratio_ts_to_best_cluster"], 3)) for B, v in cell["unit"].items()}, flush=True)
H = {}
for B in ("500", "1000", "2000"):
    rr = [c["unit"][B]["ratio_ts_to_best_cluster"] for c in out["cells"].values() if c["unit"][B]["ratio_ts_to_best_cluster"]]
    H[B] = dict(n=len(rr), share_below_1=float(np.mean(np.array(rr) < 1)), median=float(np.median(rr)), min=float(np.min(rr)), max=float(np.max(rr)))
covs = [v[m]["coverage"] for c in out["cells"].values() for B, v in c["unit"].items() if int(B) >= 500 for m in ("cluster_ppi", "cluster_ppi4")]
out["summary"] = dict(by_B=H, cluster_cov_mean=float(np.mean(covs)), cluster_cov_min=float(np.min(covs)))
out["H80h"] = dict(pass_=bool(all(v["share_below_1"] >= 0.8 for v in H.values())), by_B={B: v["share_below_1"] for B, v in H.items()})
print(json.dumps(out["summary"]), out["H80h"]); json.dump(out, open(OUT, "w"), indent=1); print("saved", OUT)
