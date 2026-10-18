"""Analysis of the 'you' counterfactual (registered H11). Paired flag rates (Haiku 5.5, Toxic or Very toxic) for the original
(same-batch retest) and the edited text, by stratum, with paired bootstrap CIs; retest agreement with the full-run label.
usage: f81_you_edit_analysis.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; rng = np.random.default_rng(81)
S = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id")
R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{W}/results/you_edit/haiku_tox/shard_*.jsonl")) for l in open(f)])
if "error" in R: R = R[R["error"].isna()]
R = R.drop_duplicates("id", keep="last"); R["flag"] = R.ans.isin(["A", "B"]).astype(float)       # no letter = not toxic
R["cid"] = R.id // 10; R["var"] = R.id % 10
P = R.pivot(index="cid", columns="var", values="flag").rename(columns={0: "orig", 1: "edit"}).dropna()
F = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{W}/results/full/haiku_tox_ctx0/shard_*.jsonl")) for l in open(f)])
if "error" in F: F = F[F["error"].isna()]
F = F.drop_duplicates("id", keep="last").set_index("id"); full = F.ans.isin(["A", "B"]).astype(float)
S = S.join(P, how="inner"); S["full"] = full.reindex(S.index)
sp = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "split"]).set_index("id").split
S["train"] = sp.reindex(S.index).eq("train").values      # the test splits release replies without parents (depth 0 is then wrong)
res = {"n": int(len(S)), "n_train": int(S.train.sum()), "strata": {}, "strata_train": {}}
def summarize(st, g):
    d = (g.edit - g.orig).values; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    ok = g.full.notna()
    return dict(n=int(len(g)), flag_orig=float(g.orig.mean()), flag_edit=float(g.edit.mean()), diff=float(d.mean()),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], rel_change=float(d.mean() / g.orig.mean()) if g.orig.mean() > 0 else None,
                retest_agreement=float((g.orig[ok] == g.full[ok]).mean()) if ok.any() else None, flag_full_run=float(g.full[ok].mean()) if ok.any() else None)
for st, g in S[S.train].groupby("stratum"):
    res["strata_train"][st] = summarize(st, g); print("train", st, json.dumps(res["strata_train"][st]), flush=True)
for st, g in S.groupby("stratum"):
    d = (g.edit - g.orig).values; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    ok = g.full.notna()
    res["strata"][st] = dict(n=int(len(g)), flag_orig=float(g.orig.mean()), flag_edit=float(g.edit.mean()), diff=float(d.mean()),
                             ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], rel_change=float(d.mean() / g.orig.mean()) if g.orig.mean() > 0 else None,
                             retest_agreement=float((g.orig[ok] == g.full[ok]).mean()) if ok.any() else None, flag_full_run=float(g.full[ok].mean()) if ok.any() else None)
    print(st, json.dumps(res["strata"][st]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
