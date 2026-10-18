"""Addendum 77, B77a: human instruments at matched flag rates (5% Civil Comments sample, results/x14_pops/cc_s5.parquet).
The human instrument at flag rate r flags the round(r N) comments with the highest crowd vote fraction ('tox'), ties broken at
random (seed 7700). Compared, at gold prevalence and at the B75a grid rates (2%, 5%, 7.5%, 10%, 15%, 20%, 30%), with the B75a rating
flags of Haiku and Sonnet (x75_rating_eval.py's flags: highest rating, ties at random, seed 7500; root comments keep their rating
without context in the 'with parent' rows). Statistics: the depth law, the within-thread depth law, the placebo depth slope relative
to the FPR and the thread size law, with 500 article-cluster Poisson bootstrap resamples (seed 7701), paired.
H77a: at gold prevalence and at 10%, 15%, 20% and 30%, the depth law of Haiku rated without the parent exceeds the human
instrument's at the same rate (paired CI above 0). H77b: the same for Sonnet without the parent.
usage: python3 x77_humanrate.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x75_rating"; OUT = sys.argv[1]; NB = 500; PP = 100.0
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); z = P.z.values.astype(float); N = len(z); prev = float(z.mean())
ids = P.id.astype(np.int64).values; has_par = P.parent_row.values >= 0; tox = P.tox.values.astype(float)
def ratings(name):
    R = {}
    try:
        for l in open(f"{D}/{name}.jsonl"):
            r = json.loads(l)
            if r.get("rating") is not None:
                R[int(r["id"])] = float(r["rating"])
    except FileNotFoundError:
        return None
    return R
S = {}
for m in ("haiku", "sonnet"):
    r0 = ratings(f"{m}_ctx0"); r1 = ratings(f"{m}_ctx1")
    if r0 is None or r1 is None:
        continue
    S[f"{m}_r"] = np.nan_to_num(np.array([r0.get(i, np.nan) for i in ids]), nan=0.0)
    S[f"{m}_r_ctx"] = np.nan_to_num(np.array([r1.get(i, np.nan) if hp else r0.get(i, np.nan) for i, hp in zip(ids, has_par)]), nan=0.0)
def flags(s, rate, seed):
    k = int(round(rate * N)); rng = np.random.default_rng(seed); order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0
    return f
cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1; x = np.minimum(P.depth.values, 10).astype(float)
uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
ku = np.bincount(uu).astype(float); ucl = np.zeros(ku.size, np.int64); ucl[uu] = cl; okU = ku >= 2; xu = np.log2(np.maximum(ku, 1))
Wb = np.vstack([np.ones(nC), np.random.default_rng(7701).poisson(1.0, size=(NB, nC))])
bc = lambda w, m=None: Wb @ np.bincount(cl if m is None else cl[m], w if m is None else w[m], nC)
def ols(v, mask):
    n, sx, sy, sxx, sxy = (bc(a, mask) for a in (np.ones(N), x, v, x * x, x * v)); return (n * sxy - sx * sy) / (n * sxx - sx ** 2), sy / n
def stats(v):
    d, _ = ols(v, np.ones(N, bool)); pdl, fpr = ols(v, z == 0); wt = bc(xt * v) / bc(xt * xt)
    f = np.bincount(uu, v, ku.size) / ku; bu = lambda w: Wb @ np.bincount(ucl[okU], w[okU], nC)
    n, sx, sy, sxx, sxy = bu(np.ones(ku.size)), bu(xu), bu(f), bu(xu * xu), bu(xu * f)
    return dict(depth=d, depth_within=wt, placebo_depth_rel=pdl / fpr, size=(n * sxy - sx * sy) / (n * sxx - sx ** 2))
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
RATES = [("matched", prev)] + [(f"{r:g}", r) for r in (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)]
res = {"prev": prev, "rates": {}}
for tag, rate in RATES:
    hs = stats(flags(tox, rate, 7700)); row = {"human": {k: ci(v) for k, v in hs.items()}}
    for name, s in S.items():
        st = stats(flags(s, rate, 7500))
        row[name] = {k: ci(v) for k, v in st.items()} | {f"{k}_minus_human": ci(st[k] - hs[k]) for k in st}
    res["rates"][tag] = row
    print(tag, "human depth", round(hs["depth"][0] * PP, 2), "placebo", round(100 * hs["placebo_depth_rel"][0], 1),
          {n_: (round(row[n_]["depth"][0] * PP, 2), [round(v * PP, 2) for v in row[n_]["depth_minus_human"][1:]]) for n_ in S}, flush=True)
ABOVE = ["matched", "0.1", "0.15", "0.2", "0.3"]
for H, name in (("H77a", "haiku_r"), ("H77b", "sonnet_r")):
    if name in S:
        res[H] = dict(lower_ends=[res["rates"][t][name]["depth_minus_human"][1] for t in ABOVE], pass_=bool(all(res["rates"][t][name]["depth_minus_human"][1] > 0 for t in ABOVE)))
        print(H, res[H])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
