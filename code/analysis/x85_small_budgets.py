"""Addendum 85, B85d: coverage at small budgets and PPI++'s minimum (descriptive, from the stored confirmation runs).
Loads the confirmation entries exactly as x74_eval_B.py does (items: results/x11_conf_item_all with CGA-Wiki's entries from
results/x72_item_cga, plus the B1/B3 comparators in results/x74_b1b3; edges: results/x11_conf_edge_all without CGA-Wiki, plus
results/x74_b1b3; units: results/x11_conf_unit and the cluster-PPI comparators of results/x80/x80_units_cluster.json).
For every design and budget: mean and minimum coverage, number of entries, share below 0.90 (laws, placebo slopes, both; gold and
placebo risk ratios; unit shares). The registered small budgets are 250 and 500 labels (laws, placebo slopes), 500 and 1,000
(contagion) and 250 (units); every budget is reported. Then the entries behind PPI++'s minimum coverage at the registered budgets
(B >= 1,000 for items, B >= 2,000 for edges): the ten lowest, with Threadsafe's coverage in the same entry, and the share of PPI++'s
entries below 0.90 by population, labeler and metric.
usage: python3 x85_small_budgets.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, os
import numpy as np
W = _ROOT; OUT = sys.argv[1]
sys.path.insert(0, f"{W}/code/analysis")
LAWS = ["depth_law", "size_law", "article_size_law"]; PLAC = ["placebo_depth", "placebo_size"]; ITEM_M = LAWS + PLAC; EDGE_M = ["log_rr", "log_placebo_rr"]
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


def rows(cells, d, metrics):
    out = []
    for tag, C in cells.items():
        for B, dd in C.items():
            if B == "_truth" or d not in dd:
                continue
            for m in metrics:
                x = dd[d].get(m)
                if x and x.get("coverage") is not None and defined(C, m):
                    out.append(dict(cell=tag, B=int(B), metric=m, coverage=float(x["coverage"])))
    return out


def summ(rr):
    if not rr:
        return None
    v = np.array([r["coverage"] for r in rr]); lo = min(rr, key=lambda r: r["coverage"])
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()), share_lt_090=float(np.mean(v < 0.90)), share_lt_085=float(np.mean(v < 0.85)),
                min_at=f"{lo['cell']} B={lo['B']} {lo['metric']}")


def by_budget(cells, designs, fams):
    out = {}
    for d in designs:
        R = {fam: rows(cells, d, ms) for fam, ms in fams.items()}
        Bs = sorted({r["B"] for rr in R.values() for r in rr})
        out[d] = {str(B): {fam: summ([r for r in rr if r["B"] == B]) for fam, rr in R.items()} for B in Bs}
    return out


# ---------------------------------------------------------------- items and edges, as x74_eval_B.py
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
COMP_I = [V3I, "ppi", "dsl", "ai", "upgrade_lw", "srs", "calag", "calag_s"]; COMP_E = [V3E, "ppi", "edge_dsl", "edge_ai", "upgrade_lw", "srs", "edge_calag"]
res = {"n_item_cells": len(I), "n_edge_cells": len(E)}
res["items"] = by_budget(I, COMP_I, {"laws": LAWS, "placebo": PLAC, "all": ITEM_M})
res["edges"] = by_budget(E, COMP_E, {"gold_rr": ["log_rr"], "placebo_rr": ["log_placebo_rr"], "all": EDGE_M})
# ---------------------------------------------------------------- units
U = {}
for f in sorted(glob.glob(f"{W}/results/x11_conf_unit/*__*.json")):
    if os.path.basename(f).startswith("eval"):
        continue
    R = json.load(open(f)); tag = os.path.basename(f)[:-5]
    for B, dd in R["unit"].items():
        for des, mm in dd.items():
            U.setdefault(tag, {}).setdefault(int(B), {})[des] = {"share": mm}
UC = json.load(open(f"{W}/results/x80/x80_units_cluster.json"))
for tag, c in UC["cells"].items():
    for B, dd in c["unit"].items():
        for des in ("cluster_ppi", "cluster_ppi4"):
            if des in dd:
                U.setdefault(tag, {}).setdefault(int(B), {})[des] = {"share": dd[des]}
res["units"] = by_budget(U, ["ts_rank", "ts_own", "cluster_ppi", "cluster_ppi4"], {"share": ["share"]})
res["n_unit_cells"] = len(U)
# ---------------------------------------------------------------- PPI++'s minimum at the registered budgets
def lowest(cells, d, ref, metrics, bmin, k=10):
    rr = [r for r in rows(cells, d, metrics) if r["B"] >= bmin]
    rr.sort(key=lambda r: r["coverage"]); out = []
    for r in rr[:k]:
        x = cells[r["cell"]][r["B"]].get(ref, {}).get(r["metric"], {})
        out.append(dict(r, threadsafe=x.get("coverage"), uniform=cells[r["cell"]][r["B"]].get("srs", {}).get(r["metric"], {}).get("coverage")))
    low = [r for r in rr if r["coverage"] < 0.90]
    cnt = lambda key: {v: sum(1 for r in low if key(r) == v) for v in sorted({key(r) for r in low})}
    return dict(n=len(rr), n_lt_090=len(low), lowest=out, by_pop=cnt(lambda r: r["cell"].split("__")[0]), by_labeler=cnt(lambda r: r["cell"].split("__")[1]),
                by_metric=cnt(lambda r: r["metric"]), by_B=cnt(lambda r: r["B"]))
res["ppi_min"] = {"items": lowest(I, "ppi", V3I, ITEM_M, 1000), "edges": lowest(E, "ppi", V3E, EDGE_M, 2000)}
res["srs_min"] = {"items": lowest(I, "srs", V3I, ITEM_M, 1000), "edges": lowest(E, "srs", V3E, EDGE_M, 2000)}
json.dump(res, open(OUT, "w"), indent=1)
# ---------------------------------------------------------------- print
for reg, fams in (("items", ("laws", "placebo")), ("edges", ("gold_rr", "placebo_rr")), ("units", ("share",))):
    print(f"== {reg}")
    for d, bb in res[reg].items():
        s = " | ".join(f"B={B}: " + ", ".join(f"{fam} {v[fam]['mean']:.3f}/{v[fam]['min']:.3f} (n={v[fam]['n']})" for fam in fams if v.get(fam)) for B, v in bb.items())
        print(f"{d:14s} {s}")
for reg in ("items", "edges"):
    P = res["ppi_min"][reg]; print(f"PPI++ {reg}: {P['n_lt_090']} of {P['n']} below 0.90; by pop {P['by_pop']}; by metric {P['by_metric']}; by B {P['by_B']}")
    for r in P["lowest"][:5]:
        print("   ", r)
print("saved", OUT)
