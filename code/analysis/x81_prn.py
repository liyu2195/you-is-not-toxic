"""Addendum 80 verification follow-up (E2): the sample behind the prompt-study slopes (results/f124_prompts.json). f124 fits the
depth slopes on comments in threads with at least two comments, after dropping threads with a missing label; this script repeats
that preprocessing (lines 18-49 of scripts/f124_prompts.py, unchanged) and records the counts.
usage: python3 x81_prn.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
ids = set(np.loadtxt(f"{W}/results/second/s5_ids.txt", dtype=np.int64).tolist())
I = pd.read_parquet(f"{W}/results/f51_items_haiku.parquet"); I = I[I.id.isin(ids)].set_index("id")
MAPS = {"tox": {"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}, "tox_rev": {"D": 1.0, "C": 2 / 3, "B": 1 / 3, "A": 0.0}, "tox_bin": {"A": 1.0, "B": 0.0}}
def read(pattern, task):
    R = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("id") in ids: R.append(r)
    R = pd.DataFrame(R)
    if "error" in R: R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last").set_index("id"); return R.ans.map(MAPS[task]).fillna(0.0)
SRC = {"main": ("tox", f"{W}/results/full/haiku_tox_ctx0/shard_*.jsonl", f"{W}/results/full/haiku_tox_ctx1/shard_*.jsonl"),
       "bin": ("tox_bin", f"{W}/results/second/s5/haiku_toxbin_ctx0/shard_*.jsonl", f"{W}/results/second/s5/haiku_toxbin_ctx1/shard_*.jsonl"),
       "rev": ("tox_rev", f"{W}/results/second/s5/haiku_toxrev_ctx0/shard_*.jsonl", f"{W}/results/second/s5/haiku_toxrev_ctx1/shard_*.jsonl")}
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id"]).set_index("id").parent_id
I["has_parent"] = meta.reindex(I.index).notna().values
for p, (task, f0, f1) in SRC.items():
    s0 = read(f0, task).reindex(I.index); s1 = read(f1, task).reindex(I.index)
    I[f"{p}0"] = (s0 >= 2 / 3 - 1e-9).astype(float).where(s0.notna()).values
    c = (s1 >= 2 / 3 - 1e-9).astype(float).where(s1.notna()); I[f"{p}1"] = np.where(I.has_parent, c.values, I[f"{p}0"].values)
LAB = ["z"] + [f"{p}{c}" for p in SRC for c in [0, 1]]; I["z"] = I.z.astype(float)
ok = I[LAB].notna().all(axis=1); ok_root = ok.groupby(I.root).all(); I = I[I.root.map(ok_root)].copy()
I["tk"] = I.groupby("root").z.transform("size"); T = I[I.tk >= 2].copy()
x = T.depth.clip(upper=10).astype(float).values
slopes = {c: float(np.cov(x, T[c].values, bias=True)[0, 1] / np.var(x)) for c in ("z", "main0", "bin0", "rev0", "main1", "bin1", "rev1")}
S5 = set(pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet", columns=["id"]).id.astype(np.int64))
res = dict(n_items=int(len(I)), n_slope=int(len(T)), n_threads=int(T.root.nunique()), n_articles=int(T.article.nunique()),
           n_slope_in_cc_s5=int(len(set(T.index.astype(np.int64)) & S5)), n_cc_s5=len(S5), slopes=slopes)
F = json.load(open(f"{W}/results/f124_prompts.json"))["audit"]
for c, k in (("z", "slope_z"), ("main0", "slope_main0"), ("bin0", "slope_bin0"), ("rev0", "slope_rev0")):
    assert abs(slopes[c] - F[k][0]) < 1e-9, (c, slopes[c], F[k][0])         # the stored slopes are reproduced on this set
print(json.dumps(res, indent=1)); json.dump(res, open(OUT, "w"), indent=1)
