"""v6 (readability round): the evaluation table with one row per design, and the numbers the rewritten Section 5 cites.
Reads the stored confirmation summaries only (no new runs): results/x85/x85_small_budgets.json (coverage by design and budget)
and results/x74_eval/eval_B.json (coverage at the registered budgets, pooled median width ratios, adversarial labelers), and the
confirmation entries themselves (loaded exactly as x85_small_budgets.py does) for the share of a comparator's undercovering
entries at small budgets that are size laws.
Writes paper_v5/tables/confirm_v6.tex and paper_v5/numbers_v6_eval.tex.
usage: python3 x95_eval_v6.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, sys
import numpy as np
W = _ROOT
sys.argv = [sys.argv[0], "/dev/null"]                       # x85's module writes its JSON to argv[1]; we only reuse its loaders
import importlib.util
spec = importlib.util.spec_from_file_location("x85", f"{W}/code/analysis/x85_small_budgets.py")
SB = json.load(open(f"{W}/results/x85/x85_small_budgets.json")); EB = json.load(open(f"{W}/results/x74_eval/eval_B.json"))
ITEM_M = ["depth_law", "size_law", "article_size_law", "placebo_depth", "placebo_size"]; SIZE_M = {"size_law", "article_size_law", "placebo_size"}
V3I, V3E = "pps_seq4_ba", "edge_m3_c15"
ROWS = [("\\ts", V3I, V3E), ("PPI++", "ppi", "ppi"), ("DSL", "dsl", "edge_dsl"), ("Active inference", "ai", "edge_ai"),
        ("Stratified", "upgrade_lw", "upgrade_lw"), ("Uniform", "srs", "srs"), ("Calibrate \\& aggregate", "calag", "edge_calag")]
c3 = lambda m, lo: f"{m:.3f} ({lo:.2f})".replace("0.", ".", 2) if lo is not None else "--"
f2 = lambda x: "--" if x is None else f"{x:.2f}"
lines = []
for name, di, de in ROWS:
    s250 = SB["items"][di]["250"]["all"]
    if di == V3I:
        ci, ce = EB["items"]["v3_cov"], EB["edges"]["v3_cov"]; w = [None] * 4
    else:
        ci, ce = EB["items"][di]["cov"], EB["edges"][de]["cov"]
        w = [None] * 4 if di == "calag" else [EB["items"][di]["laws"], EB["items"][di]["placebo"], EB["edges"][de]["gold"], EB["edges"][de]["placebo"]]
    cells = [c3(s250["mean"], s250["min"]), c3(ci[0], ci[1]), c3(ce[0], ce[1])] + [f2(x) for x in w]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    if di == V3I:
        lines.append("\\midrule")
open(f"{W}/paper_arxiv/tables/confirm_v6.tex", "w").write("\n".join(lines) + "\n")
assert all(EB["items"][d]["laws"] < 1 and EB["items"][d]["placebo"] < 1 for _, d, _ in ROWS[1:6])
assert all(EB["edges"][d]["gold"] < 1 and EB["edges"][d]["placebo"] < 1 for _, _, d in ROWS[1:6])
# ---------------------------------------------------------------- the size laws behind the comparators' undercoverage at 250 labels
mod = importlib.util.module_from_spec(spec)
src = open(f"{W}/code/analysis/x85_small_budgets.py").read().split("# ---------------------------------------------------------------- units")[0]
exec(compile(src, "x85_loaders", "exec"), mod.__dict__)
I = mod.I
M = {}
pct = lambda x: f"{100 * x:.0f}\\%"
for nm, d in (("Ppi", "ppi"), ("Unif", "srs")):
    rr = [r for r in mod.rows(I, d, ITEM_M) if r["B"] == 250]
    low = [r for r in rr if r["coverage"] < 0.90]
    M[f"eLow{nm}N"] = str(len(low)); M[f"eLow{nm}All"] = str(len(rr))
    M[f"eLow{nm}SizeShare"] = pct(sum(r["metric"] in SIZE_M for r in low) / len(low))
    M[f"eMin{nm}Pct"] = pct(min(r["coverage"] for r in rr))
    lo = min(rr, key=lambda r: r["coverage"]); print(nm, "lowest at 250:", lo)
M["eMinDslPct"] = pct(min(r["coverage"] for r in mod.rows(I, "dsl", ITEM_M) if r["B"] == 250))
M["ePpiMeanSmall"] = pct(SB["items"]["ppi"]["250"]["all"]["mean"])
ts250 = [r for r in mod.rows(I, V3I, ITEM_M) if r["B"] == 250]
assert min(r["coverage"] for r in ts250) >= 0.90
M["eTsMinAll"] = pct(min(EB["items"]["v3_cov"][1], SB["items"][V3I]["250"]["all"]["min"]))
M["eTsCov"] = f"{100 * EB['items']['v3_cov'][0]:.1f}\\%"; M["eTsCovRR"] = f"{100 * EB['edges']['v3_cov'][0]:.1f}\\%"
M["eTsCovMin"] = f"{100 * EB['items']['v3_cov'][1]:.1f}\\%"
cal = SB["items"]["calag"]; cals = SB["items"]["calag_s"]
M["eCalagLo"] = pct(cal["250"]["all"]["mean"]); M["eCalagHi"] = pct(cal["8000"]["all"]["mean"]); M["eCalagSHi"] = pct(cals["8000"]["all"]["mean"])
assert cal["250"]["all"]["mean"] > cal["1000"]["all"]["mean"] > cal["8000"]["all"]["mean"]          # coverage falls as labels grow
FC = json.load(open(f"{W}/results/x86/x86_fresh.json")); UV = json.load(open(f"{W}/results/x76_eval/x76_eval.json"))
M["eFcCov"] = f"{100 * FC['fresh']['laws']['cov_mean']:.1f}\\%"; M["eUnCov"] = f"{100 * float(UV['macros']['qUnCovMean']):.1f}\\%"
adv = EB["adversarial"]
M["eAdvItem"] = f"{100 * adv['items'][V3I][0]:.1f}\\%"; M["eAdvEdge"] = f"{100 * adv['edges'][V3E][0]:.1f}\\%"
with open(f"{W}/paper_arxiv/numbers_v6_eval.tex", "w") as fh:
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(lines)); print(json.dumps(M, indent=1))
