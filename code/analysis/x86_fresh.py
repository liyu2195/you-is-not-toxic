"""Addendum 86, B86c: the 28 fresh-classifier cells against the 27 others (descriptive, from the stored confirmation runs, loaded as in
x74_eval_B.py). Coverage (mean, minimum) of Threadsafe and median width ratios to PPI++, DSL, active inference, stratified and
uniform sampling, at the registered budgets (laws and placebo slopes B >= 1,000; contagion B >= 2,000).
usage: python3 x86_fresh.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, os
import numpy as np
W = _ROOT; OUT = sys.argv[1]
sys.argv = [sys.argv[0]]
FRESH = ("sn_rob", "mh_dist", "fb_hate", "cn_off")
LAWS = ["depth_law", "size_law", "article_size_law"]; PLAC = ["placebo_depth", "placebo_size"]; EDGE = ["log_rr", "log_placebo_rr"]
V3I, V3E = "pps_seq4_ba", "edge_m3_c15"


def load_dir(d, reg, cells=None, exclude=None):
    cells = {} if cells is None else cells
    for f in sorted(glob.glob(f"{d}/*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        J = json.load(open(f))
        for R in J["results"]:
            tag = f"{J['pop']}__{R['labeler'].replace('@', '_at_')}"
            if exclude and J["pop"] == exclude:
                continue
            for B, dd in (R.get(reg) or {}).items():
                for des, mm in dd.items():
                    cells.setdefault(tag, {}).setdefault(int(B), {})[des] = mm
            if reg == "item" and R.get("truth_item"):
                cells.setdefault(tag, {}).setdefault("_truth", R["truth_item"])
            if reg == "edge" and R.get("truth_edge"):
                cells.setdefault(tag, {}).setdefault("_truth", {k: R["truth_edge"][k] for k in ("log_rr", "log_placebo_rr")})
    return cells


def defined(C, m):
    t = C.get("_truth", {}).get(m)
    return t is None or bool(np.isfinite(t))


I = load_dir(f"{W}/results/x11_conf_item_all", "item")
for f in sorted(glob.glob(f"{W}/results/x72_item_cga/*.json")):
    J = json.load(open(f))
    for R in J["results"]:
        tag = f"{J['pop']}__{R['labeler'].replace('@', '_at_')}"
        for B, dd in R["item"].items():
            for des, mm in dd.items():
                if tag in I and int(B) in I[tag]:
                    I[tag][int(B)][des] = mm
I = load_dir(f"{W}/results/x74_b1b3", "item", I)
E = load_dir(f"{W}/results/x11_conf_edge_all", "edge", exclude="cga")
E = load_dir(f"{W}/results/x74_b1b3", "edge", E)
fresh = lambda tag: tag.split("__")[1].split("_at_")[0] in FRESH


def summary(cells, d, comps, metrics, bmin, grp):
    cov = []; rat = {c: [] for c in comps}
    for tag, C in cells.items():
        if fresh(tag) != grp:
            continue
        for B, dd in C.items():
            if B == "_truth" or B < bmin or d not in dd:
                continue
            for m in metrics:
                x = dd[d].get(m)
                if not x or x.get("coverage") is None or not defined(C, m):
                    continue
                cov.append(x["coverage"])
                for c in comps:
                    y = dd.get(c, {}).get(m)
                    if y and y.get("width") and x.get("width"):
                        rat[c].append(x["width"] / y["width"])
    return dict(n=len(cov), cov_mean=float(np.mean(cov)), cov_min=float(np.min(cov)), n_cells=len({t for t in cells if fresh(t) == grp}),
                ratio={c: (float(np.median(v)) if v else None) for c, v in rat.items()})


res = {}
for grp, nm in ((True, "fresh"), (False, "others")):
    res[nm] = {"laws": summary(I, V3I, ["ppi", "dsl", "ai", "upgrade_lw", "srs"], LAWS, 1000, grp),
               "placebo": summary(I, V3I, ["ppi", "dsl", "ai", "upgrade_lw", "srs"], PLAC, 1000, grp),
               "gold_rr": summary(E, V3E, ["ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs"], ["log_rr"], 2000, grp),
               "placebo_rr": summary(E, V3E, ["ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs"], ["log_placebo_rr"], 2000, grp)}
for nm, R in res.items():
    for fam, v in R.items():
        print(nm, fam, f"cells {v['n_cells']} entries {v['n']} cov {v['cov_mean']:.3f} min {v['cov_min']:.3f}", {c: (round(r, 2) if r else None) for c, r in v["ratio"].items()})
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
