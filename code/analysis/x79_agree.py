"""Addendum 79, C4: the within-population agreement correlations of B75f without the lenient human benchmark.
B75f (x75_within.py) computed, within each population, Spearman rho between kappa and the absolute distortion of each law in its
natural units over the population's labelers, including 'human_lenient' on Civil Comments, which is built from gold's own votes.
This recomputes it from the same stored files (x74_decomp.json for the pooled laws, x75_within.json for the within-thread law)
over the LLMs and classifiers only. Laws with fewer than 5 labelers are skipped, as in B75f; CGA-Wiki has no reply risk ratio.
usage: python3 x79_agree.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np
from scipy.stats import spearmanr
W = _ROOT; OUT = sys.argv[1]
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json")); WI = json.load(open(f"{W}/results/x75_within/x75_within.json"))["within"]
res = {}
for pop in ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]:
    ag = {}
    for law in ["depth", "size", "article_size", "log_rr", "depth_within"]:
        ks, ds, names = [], [], []
        for L, r in DEC[pop]["labelers"].items():
            if L == "human_lenient":
                continue
            if law == "depth_within":
                d = WI[pop]["labelers"].get(L, {}).get("distortion", [None])[0]
            else:
                d = r["laws"].get(law, {}).get("distortion", [None])[0]
            if d is None or not np.isfinite(d) or (law == "log_rr" and pop == "cga"):
                continue
            ks.append(r["kappa"][0]); ds.append(abs(d)); names.append(L)
        if len(ks) >= 5:
            sp = spearmanr(ks, ds); ag[law] = dict(n=len(ks), spearman=float(sp.correlation), p=float(sp.pvalue), labelers=names)
    res[pop] = ag
    print(pop, {k: (v["n"], round(v["spearman"], 2)) for k, v in ag.items()}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
