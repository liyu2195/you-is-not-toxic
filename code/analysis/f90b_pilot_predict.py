"""Addendum 9, J (reported, no hypothesis): can a 50-unit pilot predict THREADSAFE-2's gain over cluster PPI before labeling?
Pilot: 50 random units, fully labeled (cost = all their comments). From them, estimate s2_cv (single plug-in CV), h, c and r
at the THREADSAFE-2 checkpoint, and predict the width ratio sqrt[(s2_cv + h (1/q - 1)) (c + q r) / (s2_cv * kbar)], q = .5,
with kbar the population mean unit size (known). Compared with the simulated ratio ts2_cv1 / cluster_ppi at 500 labels
(results/f98_ts2_*.json). 200 pilots per setting.
usage: f90b_pilot_predict.py OUT_JSON NAME=UNITS_PARQUET ..."""
import os, sys, json, numpy as np, pandas as pd
OUT = sys.argv[1]; res = {}; rng = np.random.default_rng(901 + int(os.environ.get('SEED_OFFSET', 0))); Q = 0.5
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1); simf = f"results/f98_ts2_{name}.json"
    if ":" in path: path, simf = path.split(":", 1)
    U = pd.read_parquet(path, columns=["unit", "z", "s", "y"]); U["s2"] = U.s.values + 1e-7 * np.random.default_rng(90).random(len(U))
    U = U.sort_values(["unit", "s2"], ascending=[True, False]).reset_index(drop=True); U["u"] = pd.factorize(U.unit)[0]; U["rk"] = U.groupby("u").cumcount()
    lv = np.round(U.s.values, 4); uq = np.unique(lv)
    cands = [L for L in uq if (lv >= L).mean() <= 0.40]                     # fallback (Addendum 17)
    inC = lv >= (float(min(cands)) if cands else float(uq.max())) if len(uq) <= 6 else U.s.values >= np.quantile(U.s.values, 0.60)
    g = U.assign(F=inC).groupby("u"); k = g.size().values; nF = g.F.sum().values; Y = g.z.max().values.astype(float); Yt = g.y.max().values.astype(float)
    fpos = U.rk.where(U.z == 1).groupby(U.u).min().reindex(np.arange(len(k))).fillna(np.inf).values
    S = fpos < nF; H = ((Y == 1) & ~S).astype(float); c1 = np.where(S, fpos + 1, nF); c2 = np.where(H == 1, fpos - nF + 1, np.where(S, 0, k - nF))
    kbar = k.mean(); sim = json.load(open(simf))["results"]; target = sim["ts2_cv1@500"]["width"] / sim["cluster_ppi@500"]["width"]
    def pred(ix):
        y, f = Y[ix], Yt[ix]; lam = np.cov(y, f)[0, 1] / np.var(f, ddof=1) if np.var(f) > 0 else 0.0; s2 = np.var(y - lam * f, ddof=1)
        h, c, r = H[ix].mean(), c1[ix].mean(), c2[ix].mean(); return float(np.sqrt((s2 + h * (1 / Q - 1)) * (c + Q * r) / max(s2 * kbar, 1e-12))) if s2 > 0 else np.nan
    P = np.array([pred(rng.choice(len(k), 50, replace=False)) for _ in range(200)]); P = P[np.isfinite(P)]
    pop = pred(np.arange(len(k)))
    res[name] = dict(simulated_ratio=float(target), population_formula_ratio=pop, pilot_median=float(np.median(P)), pilot_q10=float(np.quantile(P, .1)), pilot_q90=float(np.quantile(P, .9)),
                     pilot_mae=float(np.mean(np.abs(P - target))), share_pilots_predicting_gain_10pct=float(np.mean(P < 0.9)), pilot_labels_mean=float(50 * kbar))
    print(name, json.dumps({kk: round(v, 3) for kk, v in res[name].items()}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
