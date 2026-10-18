"""Addendum 76, B76b and B76c: Sonnet 5.5 and the text classifiers on the edits of A4 (results/x74_edit).
Sonnet: flags from the registered four-level toxicity prompt (A or B), as for Haiku. Classifiers (x76_edit_cls/*.parquet): flags
at the gold-prevalence threshold of each classifier on Civil Comments reply threads (score above the (1 - prevalence) quantile of
its scores there, the '@m' rule) and at its own threshold (Detoxify 0.6, IMSYPP argmax not 'acceptable', the fresh ones 0.5).
For every instrument, as in x74_edit.py analyze (paired bootstrap, 2,000 resamples, seed 7412 for Sonnet as for Haiku) and
x75_edit_gaps.py: E1 - E0, E3 - E0, E4 - E3 on the H11 sample (train split), E2 - E0 on the new sample, per stratum, and the
depth gaps. H76a: Sonnet's E4 - E3 on deep civil comments has CI below 0. H76b: Sonnet's E2 - E0 on deep civil comments
without 'you' has CI above 0. Across instruments: Spearman correlation between E2 - E0 (deep civil, absolute; also relative to
E0) and the relative placebo depth slope on Civil Comments reply threads (x74_placebo.json; Sonnet from the 5% sample).
usage: python3 x76_edit_eval.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
W = _ROOT; D = f"{W}/results/x74_edit"; OUT = sys.argv[1]
S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id"); Sr = pd.read_parquet(f"{D}/meta_rev.parquet").set_index("id")
sp = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "split"]).set_index("id").split
S11["train"] = sp.reindex(S11.index).eq("train").values
def pivot_from(df, col):
    df = df.copy(); df["cid"] = df.id // 10; df["var"] = df.id % 10
    return df.pivot(index="cid", columns="var", values=col)
def letters(path):
    R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{path}/shard_*.jsonl")) for l in open(f)])
    if "error" in R:
        R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last"); R["flag"] = R.ans.isin(["A", "B"]).astype(float)
    return pivot_from(R[["id", "flag"]], "flag")
INSTR = {}
for nm, path in (("haiku", f"{D}/haiku_tox"), ("sonnet", f"{D}/sonnet_tox")):
    if glob.glob(f"{path}/shard_*.jsonl"):
        INSTR[nm] = letters(path)
# classifiers
CC = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["z"] + [f"s:{k}" for k in ("detoxify_original@m", "detoxify_unbiased@m", "imsypp@m", "sn_rob@m", "mh_dist@m", "fb_hate@m", "cn_off@m")])
prev = CC.z.mean()
thr = {k: float(np.nanquantile(CC[f"s:{k}@m"].values.astype(float), 1 - prev)) for k in ("detoxify_original", "detoxify_unbiased", "imsypp", "sn_rob", "mh_dist", "fb_hate", "cn_off")}
fo, ff = f"{W}/results/x76_edit_cls/edit_open.parquet", f"{W}/results/x76_edit_cls/edit_fresh.parquet"
if os.path.exists(fo) and os.path.exists(ff):
    O = pd.read_parquet(fo); F = pd.read_parquet(ff)
    O["imsypp"] = 1 - O.ims_p0; O["imsypp_arg"] = (O[[f"ims_p{k}" for k in range(4)]].values.argmax(1) != 0).astype(float)
    sc = O[["id", "det_orig", "det_unb", "imsypp", "imsypp_arg"]].rename(columns={"det_orig": "detoxify_original", "det_unb": "detoxify_unbiased"}).merge(F, on="id")
    for k in thr:
        sc[f"{k}@m"] = (sc[k] > thr[k]).astype(float)
    sc["detoxify_original@0.6"] = (sc.detoxify_original >= 0.6).astype(float); sc["detoxify_unbiased@0.6"] = (sc.detoxify_unbiased >= 0.6).astype(float)
    sc["imsypp@argmax"] = sc.imsypp_arg
    for k in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
        sc[f"{k}@0.5"] = (sc[k] >= 0.5).astype(float)
    for col in [c for c in sc.columns if "@" in c]:
        INSTR[col] = pivot_from(sc[["id", col]], col)
    for k in thr:
        INSTR[f"{k}:score"] = pivot_from(sc[["id", k]], k)
rng = np.random.default_rng(7412)
def paired(a, b):
    d = b - a; d = d[np.isfinite(d)]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return dict(n=int(len(d)), diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
res = {"thresholds": thr, "instruments": {}}
for nm, Pv in INSTR.items():
    r = {}
    J = S11[S11.train].join(Pv, how="inner")
    for st, g in J.groupby("stratum"):
        g = g[g[[0, 1, 3, 4]].notna().all(axis=1)] if all(v in g for v in (0, 1, 3, 4)) else g
        r[f"h11/{st}"] = {f"E{v}": float(g[v].mean()) for v in (0, 1, 3, 4) if v in g} | {"n": int(len(g))}
        r[f"h11/{st}"]["E1_minus_E0"] = paired(g[0].values, g[1].values)
        if 3 in g and 4 in g:
            r[f"h11/{st}"]["E4_minus_E3"] = paired(g[3].values, g[4].values); r[f"h11/{st}"]["E3_minus_E0"] = paired(g[0].values, g[3].values)
    J = Sr.join(Pv, how="inner")
    for st, g in J.groupby("stratum"):
        g = g[g[[0, 2]].notna().all(axis=1)]
        r[f"rev/{st}"] = {"E0": float(g[0].mean()), "E2": float(g[2].mean()), "n": int(len(g)), "E2_minus_E0": paired(g[0].values, g[2].values)}
    for key in ("h11", "rev"):
        vs = ["E0", "E1", "E3", "E4"] if key == "h11" else ["E0", "E2"]
        r[f"{key}/gap"] = {v: r[f"{key}/neg_deep"][v] - r[f"{key}/neg_root"][v] for v in vs if v in r[f"{key}/neg_deep"]}
    res["instruments"][nm] = r
    print(nm, "| E1-E0 deep", round(100 * r["h11/neg_deep"]["E1_minus_E0"]["diff"], 1), "| E4-E3 deep", round(100 * r["h11/neg_deep"].get("E4_minus_E3", {}).get("diff", np.nan), 1),
          "| E2-E0 deep", round(100 * r["rev/neg_deep"]["E2_minus_E0"]["diff"], 1), r["rev/neg_deep"]["E2_minus_E0"]["ci"], "| gaps", {k: round(100 * v, 1) for k, v in r["h11/gap"].items()}, flush=True)
if "sonnet" in res["instruments"]:
    s_ = res["instruments"]["sonnet"]
    res["H76a"] = bool(s_["h11/neg_deep"]["E4_minus_E3"]["ci"][1] < 0); res["H76b"] = bool(s_["rev/neg_deep"]["E2_minus_E0"]["ci"][0] > 0)
    print("H76a", res["H76a"], "H76b", res["H76b"])
PL = json.load(open(f"{W}/results/x74_placebo/x74_placebo.json"))
pdr = {k: v["placebo_depth_rel"][0] for k, v in PL["cc_threads"]["labelers"].items()}; pdr["sonnet"] = PL["cc_threads_s5"]["labelers"]["sonnet"]["placebo_depth_rel"][0]
xs, ya, yr, names = [], [], [], []
for nm in res["instruments"]:
    if nm in pdr and ":" not in nm and (nm in ("haiku", "sonnet") or nm.endswith("@m")):
        e = res["instruments"][nm]["rev/neg_deep"]; xs.append(pdr[nm]); ya.append(e["E2_minus_E0"]["diff"]); yr.append(e["E2_minus_E0"]["diff"] / max(e["E0"], 1e-9)); names.append(nm)
if len(xs) >= 4:
    res["across"] = dict(instruments=names, placebo_depth_rel=xs, you_effect=ya, you_effect_rel=yr,
                         spearman_abs=float(spearmanr(xs, ya).correlation), spearman_rel=float(spearmanr(xs, yr).correlation))
    print("across instruments", json.dumps(res["across"], default=float)[:600])
# macros
sg = lambda v, dd: (f"+{v:.{dd}f}" if v >= 0 else f"$-${abs(v):.{dd}f}")
mm = {}
if "sonnet" in res["instruments"]:
    s_ = res["instruments"]["sonnet"]; d = s_["h11/neg_deep"]["E4_minus_E3"]; e = s_["rev/neg_deep"]["E2_minus_E0"]
    mm.update(qSonParaDiff=f"{-100 * d['diff']:.1f}", qSonParaLo=f"{-100 * d['ci'][1]:.1f}", qSonParaHi=f"{-100 * d['ci'][0]:.1f}",
              qSonRevDiff=sg(100 * e["diff"], 1), qSonRevLo=sg(100 * e["ci"][0], 1), qSonRevHi=sg(100 * e["ci"][1], 1),
              qSonGapZero=sg(100 * s_["h11/gap"]["E0"], 1), qSonGapOne=sg(100 * s_["h11/gap"]["E1"], 1),
              qSonOneBefore=f"{s_['h11/neg_deep']['E0']:.2f}", qSonOneAfter=f"{s_['h11/neg_deep']['E1']:.2f}")
cm = [k for k in res["instruments"] if k.endswith("@m") and k not in ("haiku", "sonnet")]
if cm:
    v1 = [100 * res["instruments"][k]["h11/neg_deep"]["E1_minus_E0"]["diff"] for k in cm]; v2 = [100 * res["instruments"][k]["rev/neg_deep"]["E2_minus_E0"]["diff"] for k in cm]
    mm.update(qClsYouMin=sg(min(v1), 1), qClsYouMax=sg(max(v1), 1), qClsRevMin=sg(min(v2), 1), qClsRevMax=sg(max(v2), 1))
if "across" in res:
    mm.update(qYouRho=sg(res["across"]["spearman_abs"], 2), qYouRhoN=str(len(res["across"]["instruments"])))
res["macros"] = mm
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(res, open(OUT, "w"), indent=1, default=float); print("saved", OUT, mm)
