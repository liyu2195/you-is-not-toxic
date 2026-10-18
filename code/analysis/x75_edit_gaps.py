"""Addendum 75, B75g: the depth gap of Haiku's flag rate under each edit variant (part of A4's analysis).
Gap = flag rate on civil comments at depth >= 2 minus flag rate on civil comments at depth 0 (strata neg_deep, neg_root).
H11 sample (comments with 'you'): variants E0 (original), E1 (regex edit you -> third person), E3 (paraphrase keeping 'you'),
E4 (paraphrase in the third person). New sample (comments with 'they' and no 'you'): E0 and E2 (reverse edit they -> you).
Also the change of the gap between variants (a difference in differences), paired within comments. 2,000 bootstrap resamples
drawn within strata (seed 7570).
usage: python3 x75_edit_gaps.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, glob, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x74_edit"; OUT = sys.argv[1]; B = 2000
R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{D}/haiku_tox/shard_*.jsonl")) for l in open(f)])
R = R.drop_duplicates("id", keep="last"); R["flag"] = R.ans.isin(["A", "B"]).astype(float); R["cid"] = R.id // 10; R["var"] = R.id % 10
Pv = R.pivot(index="cid", columns="var", values="flag")
S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id"); Sr = pd.read_parquet(f"{D}/meta_rev.parquet").set_index("id")
sp = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "split"]).set_index("id").split
S11["train"] = sp.reindex(S11.index).eq("train").values
rng = np.random.default_rng(7570); res = {}
for key, S, vs, pairs in [("h11_train", S11[S11.train], [0, 1, 3, 4], [(0, 1), (3, 4), (0, 4)]),
                          ("h11_all", S11, [0, 1, 3, 4], [(0, 1), (3, 4), (0, 4)]), ("rev", Sr, [0, 2], [(0, 2)])]:
    J = S.join(Pv, how="inner"); J = J[J[vs].notna().all(axis=1)]
    dp = J[J.stratum == "neg_deep"][vs].values; rt = J[J.stratum == "neg_root"][vs].values
    gap = dp.mean(0) - rt.mean(0); bs = []
    for _ in range(B):
        a = dp[rng.integers(0, len(dp), len(dp))]; b = rt[rng.integers(0, len(rt), len(rt))]; bs.append(a.mean(0) - b.mean(0))
    bs = np.array(bs); r = {"n_deep": int(len(dp)), "n_root": int(len(rt)), "variants": {}}
    for j, v in enumerate(vs):
        r["variants"][f"E{v}"] = dict(deep=float(dp[:, j].mean()), root=float(rt[:, j].mean()), gap=float(gap[j]),
                                      ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))])
    for a_, b_ in pairs:
        ja, jb = vs.index(a_), vs.index(b_); d = bs[:, jb] - bs[:, ja]
        r[f"gap_E{b_}_minus_E{a_}"] = dict(diff=float(gap[jb] - gap[ja]), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                          share_of_E0_gap=float((gap[jb] - gap[ja]) / gap[vs.index(0)]) if a_ == 0 and gap[vs.index(0)] != 0 else None)
    res[key] = r
    print(key, json.dumps(r, default=float))
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
