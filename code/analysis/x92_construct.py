"""Addendum 91, B92b (descriptive): is the 'you' gap on the other corpora construct mismatch? Among each corpus's gold negatives, keep those
that a toxicity classifier also calls clearly non-toxic: Detoxify-original's toxicity score below the median of the corpus's gold negatives
(and, stricter, below its 25th percentile). Haiku (and Sonnet) without the parent: FPR on comments with a second-person word minus FPR on the
others, and the FPR ratio of comments with 3+ second-person words to those with none; 500 conversation-cluster resamples (seed 9202).
usage: python3 x92_construct.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
FULL = (_ROOT + "/scratch/full")
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
def load_labels(corpus, model):
    out = {}
    for f in sorted(glob.glob(f"{W}/results/x89/labels/{corpus}_{model}_ctx0/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); out[int(r["id"])] = r.get("ans")
    return out
res = {}
for corpus in ("cc_threads", "cad", "qian_reddit", "qian_gab", "cga", "bad"):
    P = pd.read_parquet(f"{W}/results/x14_pops/{corpus}.parquet"); ids = P.id.astype(np.int64).values; N = len(P)
    z = P.z.values; cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); neg = z == 0
    need = set(ids.tolist()); C = {}
    TXT = FULL if corpus == "cc_threads" else f"{W}/results/x89/shards/{corpus}"
    for f in sorted(glob.glob(f"{TXT}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"])
            if i in need:
                C[i] = len(YOU.findall(r.get("text") or ""))
    cnt = np.array([C.get(int(i), 0) for i in ids])
    dx = P["s:detoxify_original@m"].values.astype(float)
    Wb = np.vstack([np.ones(nC), np.random.default_rng(9202).poisson(1.0, size=(NB, nC))])
    S = lambda v: Wb @ np.bincount(cl, v, nC)
    def rate(v, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            return S(v * m) / S(m.astype(float))
    LAB = {}
    if corpus == "cc_threads":
        LAB["haiku"] = np.nan_to_num(P["y:haiku"].values.astype(float))
    else:
        for model in ("haiku", "sonnet"):
            L0 = load_labels(corpus, model)
            if L0:
                LAB[model] = np.array([float(L0.get(int(i)) in ("A", "B")) for i in ids])
    R = {"n_neg": int(neg.sum())}
    for nm, q in (("all", None), ("below_median", 0.5), ("below_q25", 0.25)):
        keep = neg if q is None else neg & (dx < np.nanquantile(dx[neg], q))
        R[nm] = {"n": int(keep.sum()), "you_share": float((cnt[keep] > 0).mean())}
        for model, y in LAB.items():
            R[nm][model] = dict(gap=ci(rate(y, keep & (cnt > 0)) - rate(y, keep & (cnt == 0))), ratio3=ci(rate(y, keep & (cnt >= 3)) / rate(y, keep & (cnt == 0))),
                                fpr0=float(y[keep & (cnt == 0)].mean()), fpr3=float(y[keep & (cnt >= 3)].mean()))
        print(corpus, nm, "n", R[nm]["n"], {m: ("gap %+.1f [%+.1f, %+.1f]" % tuple(100 * t for t in R[nm][m]["gap"]), "ratio3 %.2f" % R[nm][m]["ratio3"][0], "fpr0 %.3f" % R[nm][m]["fpr0"]) for m in LAB}, flush=True)
    res[corpus] = R
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
