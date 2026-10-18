"""Addendum 83, B83c: size laws and unit shares against human instruments matched to each instrument's flag rate (Civil Comments
reply threads; the 5% sample for Sonnet and Gemma), and the 0-100 rating flags at gold prevalence against gold (5% sample).
Matched human instrument: as B80a (x80_matched.py): the round(qN) comments with the highest crowd vote fractions, ties at random
(seed 8000 + j over the instruments in sorted order). Qwen and Gemma shown the parent at their own gold-prevalence threshold
(x76_rethreshold.py's flags, seed 7650). Rating flags as x75_rating_eval.py (highest rating, ties at random, seed 7500; root
comments keep their rating without context in the 'with parent' rows).
Statistics (as x74_decomp.py and x77_units.py): the thread size law (unit = reply thread) and the article size law (unit =
article), each the OLS slope of a unit's flag fraction on log2 of its size over units with >= 2 comments, each unit once; and the
unit share, the share of reply threads with at least one flag (all threads). 500 article-cluster Poisson resamples (seed 8001, as
B80a), paired.
H83c: the thread size laws of Haiku, Haiku shown the parent and Qwen shown the parent exceed their matched human instruments'
(paired 95% CI above 0), all three.
usage: python3 x83_matched_size.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code/analysis"))
from x80_common import Pop, ci, instruments, top_share
W = _ROOT; OUT = sys.argv[1]; PP = 100.0
H83C = ["haiku", "haiku_ctx", "qwen7b_m_ctx@own"]


def unit_stats(S, level):
    u = pd.factorize(S.P[level].values)[0]; ku = np.bincount(u).astype(float); ucl = np.zeros(ku.size, np.int64); ucl[u] = S.cl
    ok = ku >= 2; xu = np.log2(np.maximum(ku, 1))
    base = {k: S.Wb @ np.bincount(ucl[ok], w, S.nC) for k, w in (("n", np.ones(ok.sum())), ("sx", xu[ok]), ("sxx", xu[ok] ** 2))}
    return u, ku, ucl, ok, xu, base


def size_law(v, us, S):
    u, ku, ucl, ok, xu, b = us; f = np.bincount(u, v, ku.size) / ku
    sy = S.Wb @ np.bincount(ucl[ok], f[ok], S.nC); sxy = S.Wb @ np.bincount(ucl[ok], xu[ok] * f[ok], S.nC)
    return (b["n"] * sxy - b["sx"] * sy) / (b["n"] * b["sxx"] - b["sx"] ** 2)


def unit_share(v, us, S):
    u, ku, ucl, ok, xu, b = us; a = (np.bincount(u, v, ku.size) > 0).astype(float)
    return (S.Wb @ np.bincount(ucl, a, S.nC)) / (S.Wb @ np.bincount(ucl, np.ones(ku.size), S.nC))


def ratings(name):
    R = {}
    for l in open(f"{W}/results/x75_rating/{name}.jsonl"):
        r = json.loads(l)
        if r.get("rating") is not None:
            R[int(r["id"])] = float(r["rating"])
    return R


res = {}
for pop in ("cc_threads", "cc_s5"):
    S = Pop(pop); P = S.P; L = instruments(pop, P); prev = S.z.mean()
    rng = np.random.default_rng(7650)
    for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
        s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(prev * S.N))
        order = np.lexsort((rng.random(S.N), -s)); f = np.zeros(S.N); f[order[:k]] = 1.0; L[f"{m}@own"] = f
    if pop == "cc_s5":
        L = {k: v for k, v in L.items() if k.startswith(("sonnet", "gemma"))}
    th = unit_stats(S, "unit"); ar = unit_stats(S, "article")
    g = {"thread_size": size_law(S.z, th, S), "article_size": size_law(S.z, ar, S), "unit_share": unit_share(S.z, th, S)}
    out = {"N": S.N, "prev": float(prev), "gold": {k: ci(v) for k, v in g.items()}, "instruments": {}}
    for j, (name, y) in enumerate(sorted(L.items())):
        q = float(y.mean()); k = int(round(q * S.N)); h = top_share(S.tox, k, 8000 + j); r = {"flag_rate": q}
        for stat, fn, us in (("thread_size", size_law, th), ("article_size", size_law, ar), ("unit_share", unit_share, th)):
            a, b = fn(y, us, S), fn(h, us, S); r[stat] = ci(a); r[f"{stat}_human"] = ci(b); r[f"{stat}_minus_human"] = ci(a - b)
            r[f"{stat}_ratio_to_gold"] = float(a[0] / g[stat][0]); r[f"{stat}_human_ratio_to_gold"] = float(b[0] / g[stat][0])
        out["instruments"][name] = r
    if pop == "cc_s5":
        ids = P.id.astype(np.int64).values; has_par = P.parent_row.values >= 0; out["ratings"] = {}
        for m in ("haiku", "sonnet"):
            r0 = ratings(f"{m}_ctx0"); r1 = ratings(f"{m}_ctx1")
            a0 = np.nan_to_num(np.array([r0.get(i, np.nan) for i in ids]), nan=0.0)
            a1 = np.nan_to_num(np.array([r1.get(i, np.nan) if hp else r0.get(i, np.nan) for i, hp in zip(ids, has_par)]), nan=0.0)
            for nm, s in ((f"{m}_r", a0), (f"{m}_r_ctx", a1)):
                k = int(round(prev * S.N)); rr = np.random.default_rng(7500); order = np.lexsort((rr.random(S.N), -s)); f = np.zeros(S.N); f[order[:k]] = 1.0
                r = {"flag_rate": float(f.mean())}
                for stat, fn, us in (("thread_size", size_law, th), ("article_size", size_law, ar), ("unit_share", unit_share, th)):
                    a = fn(f, us, S); r[stat] = ci(a); r[f"{stat}_minus_gold"] = ci(a - g[stat]); r[f"{stat}_ratio_to_gold"] = float(a[0] / g[stat][0])
                out["ratings"][nm] = r
    res[pop] = out
    print(pop, "gold", {k: [round(x * PP, 3) for x in v] for k, v in out["gold"].items()}, flush=True)
    for name, r in out["instruments"].items():
        print(f"  {name:24s} q={r['flag_rate']:.3f} thread {r['thread_size'][0]*PP:+.3f} (human {r['thread_size_human'][0]*PP:+.3f}; diff CI [{r['thread_size_minus_human'][1]*PP:+.3f},{r['thread_size_minus_human'][2]*PP:+.3f}])"
              f" article {r['article_size'][0]*PP:+.3f} (human {r['article_size_human'][0]*PP:+.3f}) unit {r['unit_share'][0]:.3f} (human {r['unit_share_human'][0]:.3f})", flush=True)
    for name, r in out.get("ratings", {}).items():
        print(f"  rating {name:16s} thread {r['thread_size'][0]*PP:+.3f} [{r['thread_size_minus_gold'][1]*PP:+.3f},{r['thread_size_minus_gold'][2]*PP:+.3f}] vs gold; article {r['article_size'][0]*PP:+.3f}; unit {r['unit_share'][0]:.3f}", flush=True)
R = res["cc_threads"]["instruments"]
okc = [k for k in H83C if R[k]["thread_size_minus_human"][1] > 0]
res["H83c"] = dict(n_above=len(okc), above=okc, pass_=bool(len(okc) == len(H83C)))
print("H83c", res["H83c"], flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
