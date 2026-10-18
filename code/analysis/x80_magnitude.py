"""Addendum 80, B80h: substantive size (descriptive).
(1) Every instrument-law entry of the evidence (results/x74_decomp: depth, thread size, article size, log reply RR; x75_within: the
    within-thread depth law), LLMs and classifiers (not the lenient benchmark): the distortion as a share of |gold's law|; a sign
    flip (the instrument's law has the opposite sign of gold's and its 95% CI excludes 0); a manufactured significance (gold's CI
    includes 0, the instrument's excludes it) and a lost one (the reverse).
(2) The best-kappa instruments' entries with |z| > 3 (results/x76_select): how many flip the sign or move the law by > 25% of gold's.
(3) THREADSAFE's absolute 95% half-widths: medians over the confirmation cells per metric and budget (results/x11_conf_item_all,
    x72_item_cga for CGA-Wiki; x11_conf_edge_all without CGA-Wiki), and on Civil Comments reply-thread cells alone for the depth law.
usage: python3 x80_magnitude.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, os, re, numpy as np
W = _ROOT; OUT = sys.argv[1]
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json")); WI = json.load(open(f"{W}/results/x75_within/x75_within.json"))["within"]
SE = json.load(open(f"{W}/results/x76_select/x76_select.json"))
LLM = ("haiku", "sonnet", "qwen", "gemma")
rows = []
for pop, R in DEC.items():
    if not isinstance(R, dict) or "labelers" not in R:
        continue
    g = R["gold"]
    for L, r in R["labelers"].items():
        if L == "human_lenient":
            continue
        for law, v in r["laws"].items():
            if law not in g:
                continue
            gl, il = g[law], v["law"]
            rows.append(dict(pop=pop, labeler=L, law=law, gold=gl[0], gold_lo=gl[1], gold_hi=gl[2], law_pt=il[0], lo=il[1], hi=il[2], z=v["z"]))
        w = WI[pop]["labelers"].get(L)
        if w:
            gw = WI[pop]["gold_within"]; d = w["distortion"]; lw = w["law"]
            rows.append(dict(pop=pop, labeler=L, law="depth_within", gold=gw[0], gold_lo=gw[1], gold_hi=gw[2], law_pt=lw[0], lo=lw[1], hi=lw[2], z=w["z"] or 0.0))
for r in rows:
    r["rel"] = (r["law_pt"] - r["gold"]) / abs(r["gold"]) if r["gold"] != 0 else None
    sig_i = r["lo"] > 0 or r["hi"] < 0; sig_g = r["gold_lo"] > 0 or r["gold_hi"] < 0
    r["sign_flip"] = bool(r["law_pt"] * r["gold"] < 0 and sig_i); r["manufactured_sig"] = bool(sig_i and not sig_g); r["lost_sig"] = bool(sig_g and not sig_i)
    r["family"] = "llm" if r["labeler"].startswith(LLM) else "classifier"
def summ(R):
    rel = np.array([abs(r["rel"]) for r in R if r["rel"] is not None])
    return dict(n=len(R), share_rel_gt_25=float(np.mean(rel > 0.25)), median_abs_rel=float(np.median(rel)), n_sign_flip=int(sum(r["sign_flip"] for r in R)),
                n_sign_flip_gold_sig=int(sum(r["sign_flip"] and (r["gold_lo"] > 0 or r["gold_hi"] < 0) for r in R)),
                share_rel_gt_25_gold_sig=float(np.mean([abs(r["rel"]) > 0.25 for r in R if r["rel"] is not None and (r["gold_lo"] > 0 or r["gold_hi"] < 0)])),
                n_manufactured_sig=int(sum(r["manufactured_sig"] for r in R)), n_lost_sig=int(sum(r["lost_sig"] for r in R)))
res = {"all": summ(rows), "by_family": {f: summ([r for r in rows if r["family"] == f]) for f in ("llm", "classifier")},
       "by_law": {l: summ([r for r in rows if r["law"] == l]) for l in sorted({r["law"] for r in rows})}}
# (2) the best-kappa instruments
best = []
for pop, v in SE.items():
    if pop == "summary":
        continue
    tk = v["all/top_kappa"]; L = tk["instrument"]
    for law, e in tk["laws"].items():
        if e.get("z") is not None and np.isfinite(e["z"]) and abs(e["z"]) > 3:
            rr_ = [r for r in rows if r["pop"] == pop and r["labeler"] == L and r["law"] == law]
            if rr_:
                best.append(rr_[0])
res["best_kappa_z3"] = dict(n=len(best), n_rel_gt_25=int(sum(abs(r["rel"]) > 0.25 for r in best if r["rel"] is not None)), n_sign_flip=int(sum(r["sign_flip"] for r in best)),
                            n_manufactured_sig=int(sum(r["manufactured_sig"] for r in best)), n_lost_sig=int(sum(r["lost_sig"] for r in best)),
                            entries=[dict(pop=r["pop"], labeler=r["labeler"], law=r["law"], rel=r["rel"], z=r["z"], sign_flip=r["sign_flip"]) for r in best])
# (3) absolute half-widths of THREADSAFE in the confirmation
def cells(dirs, reg, design, excl=None):
    C = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*__*.json")):
            if os.path.basename(f).startswith("eval"):
                continue
            tag = re.sub(r"__g\d+$", "", os.path.basename(f)[:-5])
            if excl and tag.split("__")[0] == excl:
                continue
            R = json.load(open(f))["results"][0]
            for B, dd in (R.get(reg) or {}).items():
                if design in dd:
                    C.setdefault(tag, {}).setdefault(int(B), {}).update(dd[design])
    return C
IT = cells([f"{W}/results/x11_conf_item_all", f"{W}/results/x72_item_cga"], "item", "pps_seq4_ba"); ED = cells([f"{W}/results/x11_conf_edge_all"], "edge", "edge_m3_c15", excl="cga")
hw = {}
for nm, C in (("item", IT), ("edge", ED)):
    mets = sorted({m for c in C.values() for b in c.values() for m in b if m != "_labels"})
    hw[nm] = {m: {B: float(np.median([c[B][m]["width"] / 2 for c in C.values() if B in c and m in c[B] and c[B][m].get("width")]))
                  for B in sorted({b for c in C.values() for b in c})} for m in mets}
cct = {B: float(np.median([c[B]["depth_law"]["width"] / 2 for t, c in IT.items() if t.startswith("cc_threads") and B in c])) for B in sorted({b for c in IT.values() for b in c})}
res["half_widths"] = hw; res["half_width_depth_cc_threads"] = cct; res["gold_depth_cc_threads"] = DEC["cc_threads"]["gold"]["depth"][0]
print(json.dumps({k: v for k, v in res.items() if k not in ("half_widths",)}, indent=1)[:3000]); print(json.dumps(hw, indent=0)[:2000])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
