"""Addendum 24: the Civil Comments samples for the second batch and the dry run, and JSONL shards for labeling.
S10 = 10% of articles (seed 124) among the train-split items of f51 (complete threads); S5 = the half of S10 with a second
uniform draw < .5. Writes results/second/s10_articles.txt, s5_articles.txt, s10_ids.txt, s5_ids.txt, and shards of 25,000
items (id, text, parent_id, parent_text) for S10 and S5 into SHARD_ROOT/{s10,s5}/ (copied from the population shards).
usage: f121_cc_samples.py POP_SHARD_DIR SHARD_ROOT"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, glob, numpy as np, pandas as pd
POP, ROOT = sys.argv[1], sys.argv[2]; W = _ROOT; os.makedirs(f"{W}/results/second", exist_ok=True)
I = pd.read_parquet(f"{W}/results/f51_items_haiku.parquet", columns=["id", "article"])
arts = np.sort(I.article.unique()); rng = np.random.default_rng(124); u = rng.random(len(arts)); v = rng.random(len(arts))
S = {"s10": arts[u < .10], "s5": arts[(u < .10) & (v < .5)]}
for nm, a in S.items():
    np.savetxt(f"{W}/results/second/{nm}_articles.txt", a, fmt="%d"); ids = I.id[I.article.isin(set(a))].values
    np.savetxt(f"{W}/results/second/{nm}_ids.txt", np.sort(ids), fmt="%d"); S[nm] = set(ids.tolist()); print(nm, len(a), "articles", len(ids), "items")
for nm, keep in S.items():
    od = f"{ROOT}/{nm}"; os.makedirs(od, exist_ok=True); buf = []; k = 0
    def flush():
        global buf, k
        with open(f"{od}/shard_{k:03d}.jsonl", "w") as fo: fo.writelines(buf)
        k += 1; buf = []
    for f in sorted(glob.glob(f"{POP}/shard_*.jsonl")):
        for l in open(f):
            if json.loads(l)["id"] in keep:
                buf.append(l)
                if len(buf) == 25000: flush()
    if buf: flush()
    print(nm, "shards", k)
