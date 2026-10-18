"""Humans vs LLMs, with and without the parent, on the same 9,929 CCC posts.
Human out-of-context = original Civil Comments crowd (CCC tox_codes_oc); human in-context = CCC raters shown the parent.
LLM out-of-context / in-context = Haiku 5.5 without / with the same parent text. Parent toxicity = the parent's Civil Comments
gold (crowd target >= .5), and the CCC raters' rating of the parent as a check.
Reports: context shifts by parent toxicity, OLS of the shift on parent toxicity (controlling for the post's own
out-of-context score), and the parent -> post association under each of the four labelings."""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, ast, numpy as np, pandas as pd
CCC, HA0, HA1, OUT = sys.argv[1:5]
d = pd.read_csv(CCC); m = lambda s: float(np.mean(ast.literal_eval(s))) if isinstance(s, str) else np.nan
d["h_oc"] = d.tox_codes_oc.map(m); d["h_ic"] = d.tox_codes_ic.map(m); d["par_ccc"] = d.tox_codes_parent.map(m)
meta = pd.read_parquet((_ROOT + "/data/cc_meta.parquet"))[["id", "tox", "parent_id"]].set_index("id")
d = d.join(meta[["parent_id"]], on="id"); d["par_gold"] = meta.tox.reindex(d.parent_id.astype("Int64")).values
def rd(f):
    rows = []
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: continue                    # a line cut off by a sync
    R = pd.DataFrame(rows)
    if "p" in R:                                      # Qwen2.5-7B option probabilities: score = P(very toxic) + P(toxic)
        R = R.drop_duplicates("id", keep="last").set_index("id"); return pd.Series([p[0] + p[1] for p in R.p], index=R.index)
    R = R[R.ans.notna()].drop_duplicates("id", keep="last").set_index("id")
    return R.ans.map({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0})
d["l_oc"] = rd(HA0).reindex(d.id).values; d["l_ic"] = rd(HA1).reindex(d.id).values
d = d.dropna(subset=["h_oc", "h_ic", "l_oc", "l_ic", "par_gold"])
d["yh_oc"] = (d.h_oc >= .5).astype(int); d["yh_ic"] = (d.h_ic >= .5).astype(int)
QWEN = "--qwen" in sys.argv                            # LLM flag: letters A/B for Haiku; for Qwen a threshold matched to the human out-of-context prevalence
tl = float(np.quantile(d.l_oc, 1 - d.yh_oc.mean())) if QWEN else 2 / 3 - 1e-9
d["yl_oc"] = (d.l_oc > tl).astype(int); d["yl_ic"] = (d.l_ic > tl).astype(int)
d["ptox"] = (d.par_gold >= .5).astype(int)
r = {"n": int(len(d)), "parent_toxic_share": float(d.ptox.mean())}
for who in ["h", "l"]:
    r[who] = {"prev_oc": float(d[f"y{who}_oc"].mean()), "prev_ic": float(d[f"y{who}_ic"].mean()),
              "up": float(((d[f"y{who}_ic"] == 1) & (d[f"y{who}_oc"] == 0)).mean()), "down": float(((d[f"y{who}_ic"] == 0) & (d[f"y{who}_oc"] == 1)).mean())}
    for lab in ["oc", "ic"]:
        y = d[f"y{who}_{lab}"]; p1 = y[d.ptox == 1].mean(); p0 = y[d.ptox == 0].mean()
        r[who][f"assoc_{lab}"] = {"p1": float(p1), "p0": float(p0), "diff": float(p1 - p0), "rr": float(p1 / p0) if p0 > 0 else None}
    # shift (in-context minus out-of-context score) by parent toxicity, among posts not toxic out of context
    for pt in [0, 1]:
        x = d[(d.ptox == pt)]
        r[who][f"shift_parent{pt}"] = {"n": int(len(x)), "mean_shift": float((x[f"{who}_ic"] - x[f"{who}_oc"]).mean()),
                                       "up": float(((x[f"y{who}_ic"] == 1) & (x[f"y{who}_oc"] == 0)).mean()), "down": float(((x[f"y{who}_ic"] == 0) & (x[f"y{who}_oc"] == 1)).mean())}
    X = np.column_stack([np.ones(len(d)), d.par_gold.values, d[f"{who}_oc"].values]); yv = (d[f"{who}_ic"] - d[f"{who}_oc"]).values
    b = np.linalg.lstsq(X, yv, rcond=None)[0]; res = yv - X @ b; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
    r[who]["ols_shift_on_parent_gold"] = [float(b[1]), float(se[1])]
    rng = np.random.default_rng(0); B = []
    for _ in range(500):
        ix = rng.integers(0, len(d), len(d)); dd = d.iloc[ix]
        a_ic = dd[f"y{who}_ic"][dd.ptox == 1].mean() - dd[f"y{who}_ic"][dd.ptox == 0].mean(); a_oc = dd[f"y{who}_oc"][dd.ptox == 1].mean() - dd[f"y{who}_oc"][dd.ptox == 0].mean()
        B.append(a_ic - a_oc)
    r[who]["assoc_change_ic_minus_oc"] = [float(np.mean(B)), float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))]
json.dump(r, open(OUT, "w"), indent=1); print(json.dumps(r, indent=1))
