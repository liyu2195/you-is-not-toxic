"""Addendum 88, B88b: run-to-run variation of the LLM labels on the 5% Civil Comments sample.
B81a re-labeled the sample with Haiku 5.5 and Sonnet 5.5 under the main prompt (task tox) in later Message Batches with the same
settings (results/x81/{haiku,sonnet}_{0,1}.jsonl). Against the original labels (s:<model> levels 3/2/1/0 = A/B/C/D): the share of
comments whose answer or flag changes and kappa between the runs; in each run the depth law, its distortion and the placebo depth
slope (points per level); the paired difference of the depth law between runs; 500 article-cluster resamples (seed 8802).
H88b: for both models the depth laws of the two runs differ by less than 0.2 points per level and the paired CI includes 0.
usage: python3 x88_rerun.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(np.int64).values; z = P.z.values.astype(float)
x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
Wb = np.vstack([np.ones(nC), np.random.default_rng(8802).poisson(1.0, size=(NB, nC))])
LET = {3.0: "A", 2.0: "B", 1.0: "C", 0.0: "D"}


def slope_b(v, mask):
    m_ = mask.astype(float); v = np.nan_to_num(v)
    n_, sx, sy, sxx, sxy = (Wb @ np.bincount(cl, m_ * a_, nC) for a_ in (np.ones_like(x), x, v, x * x, x * v))
    return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)


ci = lambda a: [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
res = {}
for m in ("haiku", "sonnet"):
    new = {}
    for h in (0, 1):
        for l in open(f"{W}/results/x81/{m}_{h}.jsonl"):
            r = json.loads(l)
            if r["task"] == "tox":
                new[int(r["id"])] = r["ans"]
    a_old = np.array([LET.get(v) for v in P[f"s:{m}"].values.astype(float)], dtype=object); a_new = np.array([new.get(int(i)) for i in ids], dtype=object)
    ok = np.array([(u is not None) and (w is not None) for u, w in zip(a_old, a_new)])
    f_old = np.isin(a_old, ["A", "B"]).astype(float); f_new = np.isin(a_new, ["A", "B"]).astype(float)
    po = float((f_old[ok] == f_new[ok]).mean()); p1, p2 = f_old[ok].mean(), f_new[ok].mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
    all_ = ok; neg = ok & (z == 0)
    g = slope_b(z, all_); lo = slope_b(f_old, all_); ln = slope_b(f_new, all_)
    res[m] = dict(n=int(ok.sum()), answer_changed=float((a_old[ok] != a_new[ok]).mean()), flag_changed=float((f_old[ok] != f_new[ok]).mean()),
                  kappa_between_runs=float((po - pe) / (1 - pe)), flag_rate_old=float(p1), flag_rate_new=float(p2),
                  depth_gold=ci(g), depth_old=ci(lo), depth_new=ci(ln), distortion_old=ci(lo - g), distortion_new=ci(ln - g), depth_new_minus_old=ci(ln - lo),
                  placebo_old=ci(slope_b(f_old, neg)), placebo_new=ci(slope_b(f_new, neg)))
    d = res[m]["depth_new_minus_old"]; res[m]["H88b"] = bool(abs(d[0]) < 0.2 and d[1] <= 0 <= d[2])
    print(m, {k: (round(v, 4) if isinstance(v, float) else ([round(t, 3) for t in v] if isinstance(v, list) else v)) for k, v in res[m].items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
