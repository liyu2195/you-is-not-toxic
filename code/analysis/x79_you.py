"""Addendum 79, C1: the second-person descriptive numbers of Section 4 on the v4 population (results/x14_pops/cc_threads.parquet).
Civil: crowd vote fraction ('tox') below 0.2, the definition used to sample the edits (alternative, for the record: gold negatives).
Detector: the Appendix A 'you' detector (x74_edit.py, x77_youfree.py). Depths: 0 against >= 6 (exact 6 for the record).
Statistics among civil comments: the share with second-person address, and Haiku's flag rate (y:haiku) on those with it and
on those without it. 500 article-cluster Poisson bootstrap resamples (seed 7900).
usage: python3 x79_you.py TEXT_SHARD_DIR OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; SH, OUT = sys.argv[1], sys.argv[2]; NB = 500
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "cluster", "depth", "z", "tox", "y:haiku"])
idx = pd.Series(np.arange(len(P)), index=P.id.astype(np.int64).values)
you = np.full(len(P), -1, np.int8)
for f in sorted(glob.glob(f"{SH}/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = idx.get(int(r["id"]))
        if i is not None:
            you[i] = bool(YOU.search(r.get("text") or ""))
have = you >= 0; print("population", len(P), "texts found", int(have.sum()), flush=True)
d = P.depth.values; z = P.z.values; tox = P.tox.values; y = np.nan_to_num(P["y:haiku"].values.astype(float))
cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
Wb = np.vstack([np.ones(nC), np.random.default_rng(7900).poisson(1.0, size=(NB, nC))])
bc = lambda w, m: Wb @ np.bincount(cl[m], w[m], nC)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
res = {"n": int(len(P)), "n_texts": int(have.sum()), "defs": {}}
for dname, civ in (("tox_below_0.2", tox < 0.2), ("gold_negative", z == 0)):
    out = {}
    for bname, bins in (("d0_vs_d6plus", (d == 0, d >= 6)), ("d0_vs_d6", (d == 0, d == 6))):
        r = {}
        for tag, b in zip(("d0", "deep"), bins):
            m = civ & have & b; yu = (you == 1)
            n = bc(np.ones(len(P)), m); ny = bc(yu.astype(float), m)
            r[tag] = dict(n=int(m.sum()), share_you=ci(ny / n),
                          haiku_on_you=ci(bc(y * yu, m) / ny), haiku_on_other=ci(bc(y * (~yu), m) / (n - ny)))
        out[bname] = r
        print(dname, bname, json.dumps({t: {k: (round(v[0], 3) if isinstance(v, list) else v) for k, v in r[t].items()} for t in r}), flush=True)
    res["defs"][dname] = out
M = res["defs"]["tox_below_0.2"]["d0_vs_d6plus"]; f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"
res["macros"] = dict(nyouShareZero=f2(M["d0"]["share_you"][0]), nyouShareSix=f2(M["deep"]["share_you"][0]),
                     nfprYouZero=f3(M["d0"]["haiku_on_you"][0]), nfprYouSix=f3(M["deep"]["haiku_on_you"][0]),
                     nfprNoYouZero=f3(M["d0"]["haiku_on_other"][0]), nfprNoYouSix=f3(M["deep"]["haiku_on_other"][0]))
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res["macros"]), "\nsaved", OUT)
