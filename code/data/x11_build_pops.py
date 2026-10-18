"""Addendum 69: add the fresh labelers (results/x69_fresh/*/<tag>.parquet, code/score_fresh.py) to the audit populations,
results/x10_pops -> results/x11_pops. For each fresh model <m>: s:<m>@m = its probability; y:<m>@m = 1 if the probability is
above the threshold matched to the population's gold prevalence (as for the Detoxify and IMSYPP @m labelers).
Civil Comments populations read the cc_000..cc_079 shards; the others their own tag.
usage: python3 x11_build_pops.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import glob, os
import numpy as np, pandas as pd
W = _ROOT; IN, OUT = f"{W}/results/x10_pops", f"{W}/results/x11_pops"; os.makedirs(OUT, exist_ok=True)
MODELS = ["sn_rob", "mh_dist", "fb_hate", "cn_off"]
TAGS = {"cc_threads": [f"cc_{i:03d}" for i in range(80)], "cc_s5": [f"cc_{i:03d}" for i in range(80)], "cad": ["cad"], "cga": ["cga"],
        "bad": ["bad"], "qian_reddit": ["qian_reddit"], "qian_gab": ["qian_gab"]}
cache = {}


def scores(tags):
    key = tuple(tags)
    if key not in cache:
        fs = [f for t in tags for f in glob.glob(f"{W}/results/x69_fresh/*/{t}.parquet")]
        cache[key] = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("id").set_index("id")
    return cache[key]


report = {}
for pop, tags in TAGS.items():
    P = pd.read_parquet(f"{IN}/{pop}.parquet"); S = scores(tags).reindex(P.id.values); prev = float(P.z.mean())
    for m in MODELS:
        v = S[m].values.astype(float); t = float(np.nanquantile(v, 1 - prev))
        P[f"s:{m}@m"] = v; P[f"y:{m}@m"] = np.where(np.isnan(v), np.nan, (v > t).astype(float))
        report[(pop, m)] = dict(missing=int(np.isnan(v).sum()), threshold=t, flag_rate=float(np.nanmean(P[f"y:{m}@m"])), prev=prev)
    P.to_parquet(f"{OUT}/{pop}.parquet")
    print(pop, len(P), {m: f"miss {report[(pop, m)]['missing']} t {report[(pop, m)]['threshold']:.3f}" for m in MODELS}, flush=True)
pd.DataFrame([dict(pop=k[0], model=k[1], **v) for k, v in report.items()]).to_csv(f"{OUT}/fresh_report.csv", index=False)
