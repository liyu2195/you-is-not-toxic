"""Addendum 91, B92c-d (descriptive). (c) Haiku's strictest level ('very toxic') on Civil Comments reply threads: flag rate, TPR, FPR,
the depth law and Proposition 1's three terms (point values). (d) Under each other corpus's own question (stored labels: CAD abuse,
Qian et al. hate speech, CGA-Wiki personal attack; Haiku and the Gemma/Qwen settings stored for them), the FPR gap between gold
negatives with and without a second-person word and the 3+:0 ratio; 500 conversation-cluster resamples (seed 9203).
usage: python3 x92_extra.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
res = {}
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "depth", "z", "s:haiku", "y:haiku"])
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); lev = np.nan_to_num(P["s:haiku"].values.astype(float), nan=-1)
w = (x - x.mean()) / ((x - x.mean()) ** 2).sum(); L = lambda v: 100 * float(w @ v)
for nm, y in (("very_toxic", (lev >= 3).astype(float)), ("native", (lev >= 2).astype(float))):
    TPR, FPR = float(y[z == 1].mean()), float(y[z == 0].mean()); neg = z == 0
    res[f"cc_{nm}"] = dict(flag_rate=float(y.mean()), tpr=TPR, fpr=FPR, law=L(y), gold=L(z), uniform=(TPR - FPR - 1) * L(z),
                           S_FP=100 * float(w[neg] @ (y[neg] - FPR)), S_TP=100 * float(w[~neg] @ (y[~neg] - TPR)))
    print(nm, {k: round(v, 4) for k, v in res[f"cc_{nm}"].items()}, flush=True)
for corpus in ("cad", "qian_reddit", "qian_gab", "cga"):
    P = pd.read_parquet(f"{W}/results/x14_pops/{corpus}.parquet"); ids = P.id.astype(np.int64).values
    z = P.z.values; cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); neg = z == 0
    need = set(ids.tolist()); C = {}
    for f in sorted(glob.glob(f"{W}/results/x89/shards/{corpus}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"])
            if i in need: C[i] = len(YOU.findall(r.get("text") or ""))
    cnt = np.array([C.get(int(i), 0) for i in ids])
    Wb = np.vstack([np.ones(nC), np.random.default_rng(9203).poisson(1.0, size=(NB, nC))]); S = lambda v: Wb @ np.bincount(cl, v, nC)
    def rate(v, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            return S(v * m) / S(m.astype(float))
    R = {}
    for col in [c for c in P.columns if c.startswith("y:") and c[2:].split("_")[0] in ("haiku", "sonnet", "qwen7b", "gemma") and not c.endswith("_ctx")]:
        y = np.nan_to_num(P[col].values.astype(float))
        R[col[2:]] = dict(fpr=float(y[neg].mean()), gap=ci(rate(y, neg & (cnt > 0)) - rate(y, neg & (cnt == 0))), ratio3=ci(rate(y, neg & (cnt >= 3)) / rate(y, neg & (cnt == 0))))
        print(corpus, col, "FPR %.3f" % R[col[2:]]["fpr"], "gap %+.1f [%+.1f, %+.1f]" % tuple(100 * t for t in R[col[2:]]["gap"]), "ratio3 %.2f" % R[col[2:]]["ratio3"][0], flush=True)
    res[corpus] = R
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
