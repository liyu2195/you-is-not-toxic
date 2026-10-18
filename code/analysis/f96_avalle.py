"""Addendum 9, I (H15): re-running the design of Avalle et al. (Nature 2024) on Civil Comments (train split).
Conversations: an article's comment section, or a reply thread (>= 2 comments); every comment labeled by Haiku with and
without the parent. Statistic: each conversation's toxic fraction, averaged within log2 size bins; Mann-Kendall (Kendall tau
of bin means on bin index); OLS slope of the conversation fraction on log2(size); null = 200 shuffles of the labels across
comments (Avalle's randomization). 'Significant increase' = observed slope >= shuffled mean + 2 SD.
H15a: at the thread level, Haiku with the parent shows a significant increase and gold does not.
H15b: at 2,000 labels (300 runs), THREADSAFE's contrast regime covers the gold thread-level slope in >= 90% of runs and
announces an increase (lower bound > 0) in <= 10%; validate-then-calibrate announces it in > 50%.
  THREADSAFE: sample threads at random, label one random comment of each (E[Z | thread] = its toxic fraction); PPI for
  Cov(x, f) with the LLM's conversation fraction as predictor: theta = mean_N[(x - xbar) fhat] + mean_n[(x - xbar)(Z - Y)].
  Calibrate: 2,000 random comments, P(Z | letter) from them, applied to every comment; slope of the calibrated fractions;
  percentile bootstrap (200) over the validation sample.
usage: f96_avalle.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, numpy as np, pandas as pd
from scipy.stats import kendalltau
OUT = sys.argv[1]; W = _ROOT; rng = np.random.default_rng(96)
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "tox", "parent_id", "article_id", "split"]]
meta = meta[meta.split == "train"].drop(columns="split").set_index("id")
def letters(files):
    R = pd.DataFrame([json.loads(l) for f in files for l in open(f)])
    if "error" in R: R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last").set_index("id"); return R.ans.map({"A": 3, "B": 2, "C": 1, "D": 0}).fillna(0).astype(np.int8)
F = f"{W}/results/full"
L0 = letters(sorted(glob.glob(f"{F}/haiku_tox_ctx0/shard_*.jsonl"))); L1 = letters(sorted(glob.glob(f"{F}/haiku_tox_ctx1/shard_*.jsonl")))
D = meta.copy(); D["l0"] = L0.reindex(D.index); c = L1.reindex(D.index); D["l1"] = c.where(c.notna(), D.l0); D.loc[D.parent_id.notna() & c.isna(), "l1"] = np.nan
pm = D.parent_id.where(D.parent_id.isin(D.index)).dropna().astype("int64"); cur = pd.Series(D.index.values, index=D.index)
for _ in range(200):
    has = cur.isin(pm.index)
    if not has.any(): break
    cur[has] = pm.reindex(cur[has].values).values
D["root"] = cur.values; D["Z"] = (D.tox >= .5).astype(np.int8)
D["haiku"] = (D.l0 >= 2).astype(float).where(D.l0.notna()); D["haiku_ctx"] = (D.l1 >= 2).astype(float).where(D.l1.notna())
SRC = ["Z", "haiku", "haiku_ctx"]
def conv_level(unit):
    ok = D[["haiku", "haiku_ctx"]].notna().all(axis=1).groupby(D[unit]).all(); X = D[D[unit].map(ok)].copy()
    X["k"] = X.groupby(unit).Z.transform("size"); X = X[X.k >= 2]
    return X
res = {}
for level, unit in [("articles", "article_id"), ("threads", "root")]:
    X = conv_level(unit); u = pd.factorize(X[unit])[0]; NU = u.max() + 1; k = np.bincount(u); x = np.log2(k)
    b = np.floor(x).astype(int); out = {"n_conversations": int(NU), "n_comments": int(len(X))}
    def frac(v): return np.bincount(u, weights=v, minlength=NU) / k
    def slope(f): xc = x - x.mean(); return float((xc * (f - f.mean())).sum() / (xc ** 2).sum())
    for s in SRC:
        v = X[s].values.astype(float); f = frac(v); bm = pd.Series(f).groupby(b).mean(); bn = pd.Series(f).groupby(b).size()
        bm = bm[bn >= 30]; tau, p = kendalltau(bm.index.values, bm.values); obs = slope(f)
        sh = [slope(frac(rng.permutation(v))) for _ in range(200)]; mu, sd = float(np.mean(sh)), float(np.std(sh, ddof=1))
        out[s] = dict(bin_means={int(i): float(m) for i, m in bm.items()}, mk_tau=float(tau), mk_p=float(p), slope=obs, shuffle_mean=mu, shuffle_sd=sd,
                      z_vs_shuffle=float((obs - mu) / sd) if sd > 0 else None, significant_increase=bool(obs >= mu + 2 * sd))
        print(level, s, f"slope {obs:+.5f} shuffle {mu:+.5f} ± {sd:.5f} | MK tau {tau:+.2f} p {p:.3g} | bins {({i: round(m, 3) for i, m in bm.items()})}", flush=True)
    res[level] = out
    if level == "threads":
        # H15b: design simulation for the thread-level slope
        zt = X.Z.values.astype(float); yt = X.haiku_ctx.values.astype(float); lt = X.l1.values.astype(int)
        gold_slope = out["Z"]["slope"]; fhat = frac(yt); xc = x - x.mean(); vx = (xc ** 2).mean(); pop_term = (xc * fhat).mean()
        start = np.r_[0, np.cumsum(k)[:-1]]                 # rows of each thread are contiguous after sorting
        order_rows = np.argsort(u, kind="stable"); zt_s, yt_s, lt_s = zt[order_rows], yt[order_rows], lt[order_rows]
        CNT = np.zeros((NU, 4)); np.add.at(CNT, (u, lt), 1)
        def ts(m):
            th = rng.choice(NU, m, replace=False); j = start[th] + (rng.random(m) * k[th]).astype(int)
            d = xc[th] * (zt_s[j] - yt_s[j]); est = (pop_term + d.mean()) / vx; se = d.std(ddof=1) / np.sqrt(m) / vx; return est, est - 1.96 * se, est + 1.96 * se
        def cal_est(ix):
            n1 = np.bincount(lt[ix], weights=zt[ix], minlength=4); n = np.bincount(lt[ix], minlength=4)
            pz = np.where(n > 0, n1 / np.maximum(n, 1), zt[ix].mean()); f = (CNT @ pz) / k; return slope(f)
        def cal(m):
            ix = rng.choice(len(zt), m, replace=False); est = cal_est(ix); bs = [cal_est(ix[rng.integers(0, m, m)]) for _ in range(200)]
            return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        sim = {}
        for name, fn in [("threadsafe", ts), ("calibrate", cal)]:
            for m in [500, 2000]:
                R = np.array([fn(m) for _ in range(300)]); est, lo, hi = R.T
                sim[f"{name}@{m}"] = dict(bias=float(est.mean() - gold_slope), coverage=float(np.mean((lo <= gold_slope) & (gold_slope <= hi))), width=float(np.mean(hi - lo)),
                                          announces_increase=float(np.mean(lo > 0)), announces_decrease=float(np.mean(hi < 0)))
                print(name, m, json.dumps({kk: round(vv, 4) for kk, vv in sim[f"{name}@{m}"].items()}), flush=True)
        out["design"] = dict(gold_slope=gold_slope, plugin_slope=out["haiku_ctx"]["slope"], results=sim)
r = res["threads"]
res["H15a"] = bool(r["haiku_ctx"]["significant_increase"] and not r["Z"]["significant_increase"])
s = r["design"]["results"]; res["H15b"] = bool(s["threadsafe@2000"]["coverage"] >= .90 and s["threadsafe@2000"]["announces_increase"] <= .10 and s["calibrate@2000"]["announces_increase"] > .50)
json.dump(res, open(OUT, "w"), indent=1); print("H15a", res["H15a"], "H15b", res["H15b"])
