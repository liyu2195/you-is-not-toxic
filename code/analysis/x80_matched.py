"""Addendum 80, B80a: human instruments matched to each instrument's flag rate (Civil Comments reply threads; the 5% sample for
Sonnet and Gemma). For an instrument flagging a share q of the N comments, the matched human instrument flags the round(qN) comments
with the highest crowd vote fractions, ties at random (seed 8000). Both are scored on the same 500 article-cluster Poisson
bootstrap resamples (seed 8001): depth law, log reply RR, placebo depth slope relative to the FPR and placebo log RR (the last two
only where the human instrument has false positives, i.e. q above gold prevalence).
H80a: >= 9 of the 12 classifier settings have a log RR above their matched human instrument's (paired 95% CI above 0).
H80b: the five LLM settings that manufacture escalation have depth laws above their matched human instrument's (paired CI above 0).
Qwen and Gemma shown the parent are at their own gold-prevalence threshold (the flags of x76_rethreshold.py, seed 7650).
usage: python3 x80_matched.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code/analysis"))
from x80_common import Pop, ci, instruments, top_share
OUT = sys.argv[1]; PP = 100.0
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
MAN5 = [("cc_threads", "haiku"), ("cc_threads", "haiku_ctx"), ("cc_threads", "qwen7b_m_ctx@own"), ("cc_s5", "sonnet"), ("cc_s5", "sonnet_ctx")]
res = {}
for pop in ("cc_threads", "cc_s5"):
    S = Pop(pop); P = S.P; L = instruments(pop, P); prev = S.z.mean()
    rng = np.random.default_rng(7650)                                   # x76_rethreshold.py's own-threshold flags
    for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
        s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(prev * S.N))
        order = np.lexsort((rng.random(S.N), -s)); f = np.zeros(S.N); f[order[:k]] = 1.0; L[f"{m}@own"] = f
    if pop == "cc_s5":
        L = {k: v for k, v in L.items() if k.startswith(("sonnet", "gemma"))}
    out = {"N": S.N, "prev": float(prev), "instruments": {}}
    for j, (name, y) in enumerate(sorted(L.items())):
        q = float(y.mean()); k = int(round(q * S.N)); h = top_share(S.tox, k, 8000 + j)
        r = {"flag_rate": q, "human_fp": bool(q > prev + 1e-9)}
        for stat, fn in (("depth", S.depth), ("lrr", S.lrr)):
            a, b = fn(y), fn(h); r[stat] = ci(a); r[f"{stat}_human"] = ci(b); r[f"{stat}_minus_human"] = ci(a - b)
        if r["human_fp"]:
            for stat, fn in (("placebo_depth_rel", S.placebo_depth_rel), ("placebo_lrr", S.placebo_lrr)):
                a, b = fn(y), fn(h); r[stat] = ci(a); r[f"{stat}_human"] = ci(b); r[f"{stat}_minus_human"] = ci(a - b)
        out["instruments"][name] = r
        print(pop, f"{name:24s} q={q:.3f} depth {r['depth'][0]*PP:+.2f} vs human {r['depth_human'][0]*PP:+.2f} | RR {np.exp(r['lrr'][0]):.2f} vs human {np.exp(r['lrr_human'][0]):.2f} "
              f"diff CI [{r['lrr_minus_human'][1]:+.3f}, {r['lrr_minus_human'][2]:+.3f}]" + (f" | placebo RR {np.exp(r['placebo_lrr'][0]):.2f} vs {np.exp(r['placebo_lrr_human'][0]):.2f}" if r["human_fp"] else ""), flush=True)
    res[pop] = out
I = res["cc_threads"]["instruments"]
up = [k for k in CLS12 if I[k]["lrr_minus_human"][1] > 0]
res["H80a"] = dict(n_pass=len(up), settings=up, pass_=len(up) >= 9)
man = [(p, k, res[p]["instruments"][k]["depth_minus_human"][1] > 0) for p, k in MAN5]
res["H80b"] = dict(detail=[dict(pop=p, instrument=k, lower=res[p]["instruments"][k]["depth_minus_human"][1], above=a) for p, k, a in man], pass_=all(a for _, _, a in man))
print("H80a", res["H80a"]); print("H80b", res["H80b"])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
