"""Addendum 74, Part A evaluation: hypotheses H74a-H74g, as registered.
  H74a  of the 12 text-classifier settings on Civil Comments reply threads, >= 9 have a placebo RR above the lenient human's
        (CI of the difference above 0)                                   results/x74_placebo/x74_placebo.json
  H74b  of the same 12, >= 9 report a reply RR above gold's (CI of the paired difference above 0)
  H74c  a second crowd distorts less than the named instruments in each of 20 splits     results/x74_crowd/x74_crowd.json
  H74d  deep civil comments: Haiku flags the third-person paraphrase less than the you-keeping one (CI below 0)
  H74e  deep civil comments without 'you': the reverse edit raises Haiku's flag rate (CI above 0)   results/x74_edit/x74_edit.json
  H74f  for each of depth, thread size and log RR: a top-kappa-quartile setting with |z| > 3 and a bottom-quartile one with |z| < 1
        (A1's settings: every labeler x population of x74_placebo.json; z from its paired bootstrap)
  H74g  Haiku's depth law and thread size law exceed gold's (CI above 0) at the strict level and at gold prevalence
usage: python3 x74_eval_A.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, numpy as np
from scipy.stats import spearmanr
W = _ROOT; OUT = f"{W}/results/x74_eval"; os.makedirs(OUT, exist_ok=True)
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
res = {}
CLS = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
PL = load(f"{W}/results/x74_placebo/x74_placebo.json")
if PL:
    L = PL["cc_threads"]["labelers"]
    a = {k: L[k]["vs_human"]["prr"] for k in CLS}; b = {k: L[k]["law_rr_minus_gold"] for k in CLS}
    res["H74a"] = dict(n_pass=sum(v[1] > 0 for v in a.values()), per_setting=a, pass_=sum(v[1] > 0 for v in a.values()) >= 9)
    res["H74b"] = dict(n_pass=sum(v[1] > 0 for v in b.values()), per_setting=b, pass_=sum(v[1] > 0 for v in b.values()) >= 9)
    # H74f over every labeler x population of A1
    rows = []
    for pop, P in PL.items():
        for k, r in P["labelers"].items():
            if "distortion" not in r:
                continue
            for law in ("depth", "size", "log_rr"):
                if law == "log_rr" and pop == "cga":                      # no gold risk ratio on CGA-Wiki (Addendum 72)
                    continue
                d = r["distortion"].get(law)
                if d and d.get("z") is not None and np.isfinite(d["z"]):
                    rows.append((pop, k, law, r["kappa"][0], abs(d["z"])))
    f = {}
    for law in ("depth", "size", "log_rr"):
        R = [x for x in rows if x[2] == law]; ks = np.array([x[3] for x in R]); zs = np.array([x[4] for x in R])
        q1, q3 = np.quantile(ks, [0.25, 0.75])
        f[law] = dict(n=len(R), top_max_z=float(zs[ks >= q3].max()), bottom_min_z=float(zs[ks <= q1].min()),
                      spearman=float(spearmanr(ks, zs).correlation), pass_=bool(zs[ks >= q3].max() > 3 and zs[ks <= q1].min() < 1))
    ks_all = np.array([x[3] for x in rows]); zs_all = np.array([x[4] for x in rows])
    f["spearman_all"] = float(spearmanr(ks_all, zs_all).correlation); f["n_settings"] = len({(x[0], x[1]) for x in rows})
    bs = []
    rng = np.random.default_rng(7476)
    for _ in range(2000):
        ix = rng.integers(0, len(rows), len(rows)); bs.append(spearmanr(ks_all[ix], zs_all[ix]).correlation)
    f["spearman_all_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res["H74f"] = f | {"pass_": all(f[l]["pass_"] for l in ("depth", "size", "log_rr"))}
    g = {}
    for k in ("haiku@A", "haiku@m"):
        r = L[k]; dd = r["distortion"]; g[k] = dict(depth=dd["depth"]["ci"], size=dd["size"]["ci"], flag_rate=r["flag_rate"][0])
    res["H74g"] = g | {"pass_": all(g[k]["depth"][0] > 0 and g[k]["size"][0] > 0 for k in ("haiku@A", "haiku@m"))}
C = load(f"{W}/results/x74_crowd/x74_crowd.json")
if C:
    S = C["splits"]; parts = {}
    for law, labs in [("depth", ["haiku", "haiku_ctx", "qwen7b_m_ctx"]), ("size", ["haiku", "haiku_ctx", "qwen7b_m_ctx"]),
                      ("log_rr", ["detoxify_original@0.6", "detoxify_original@m", "haiku_ctx", "qwen7b_m_ctx"])]:
        parts[law] = {l: sum(abs(s["instr"][l][f"{law}_minus_goldB"]) > abs(s["instr"]["crowd_A"][f"{law}_minus_goldB"]) for s in S) for l in labs}
    for st in ("placebo_depth_rel", "placebo_size_rel"):
        parts[st] = {l: sum(s["instr"][l][st] > s["instr"]["crowd_A"][st] for s in S) for l in ["haiku", "haiku_ctx", "qwen7b_m_ctx"]}
    res["H74c"] = dict(splits=len(S), counts=parts, pass_=all(v == len(S) for d in parts.values() for v in d.values()))
E = load(f"{W}/results/x74_edit/x74_edit.json")
if E:
    res["H74d"] = dict(**E["h11"]["neg_deep"]["E4_minus_E3"], pass_=E["H74d"]); res["H74e"] = dict(**E["rev"]["neg_deep"]["E2_minus_E0"], pass_=E["H74e"])
json.dump(res, open(f"{OUT}/eval_A.json", "w"), indent=1, default=float)
for k, v in res.items():
    print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("per_setting",)}, default=float)[:600])
