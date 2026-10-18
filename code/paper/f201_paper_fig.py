"""Figure 1 of the paper, from stored population results (no random seeds).
(a) Toxicity rate by reply depth relative to depth 0 on all Civil Comments reply threads (f10): gold, Haiku 5.5, Qwen2.5-7B matched.
(b) Depth slope of the toxicity rate minus gold's, with 95% article-cluster bootstrap intervals, against item agreement (kappa):
    Haiku and Qwen conditions on the full population (f13, f10); Gemma 3 27B on the 5% sample (f129, Addendum 34).
Also writes paper/numbers_fig.tex (Haiku's kappa on the 5% sample, recomputed here from the stored labels).
usage: python3 f201_paper_fig.py ROOT_DIR PAPER_DIR"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
W, PD = sys.argv[1], sys.argv[2]
R = lambda f: json.load(open(os.path.join(W, "results", f)))
P, H3, G = R("f10_population.json"), R("f13_depth_slope_ci.json"), R("f129_open_model.json")["s5"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"; BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlesize": 7.5})
fig = plt.figure(figsize=(7.0, 2.0)); a1 = fig.add_axes([0.062, 0.2, 0.285, 0.68]); a2 = fig.add_axes([0.70, 0.2, 0.285, 0.68])
def style(ax):
    ax.tick_params(colors=INK2, length=2); ax.grid(True, axis="y" if ax is a1 else "x", color=GRID, lw=0.6)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(GRID)
# (a) curves
E = P["escalation"]; x = np.arange(11); K = {c: P["item"][c]["kappa"] for c in P["item"] if c != "gold_prev"}
SER = [("haiku", f"Haiku 5.5 (κ = {K['haiku']:.2f})", BLUE, "s"), ("qwen7b_m", f"Qwen2.5-7B (κ = {K['qwen7b_m']:.2f})", AQUA, "D"), ("Z", "Gold labels", INK, "o")]
end = {}
for c, lab, col, mk in SER:
    y = np.array([E[c]["by_depth"][str(i)] for i in x]); y = y / y[0]; end[c] = y[-1]
    a1.plot(x, y, color=col, lw=1.5, marker=mk, ms=3.2, mec="white", mew=0.5, zorder=3)
    off = {"Z": -5, "qwen7b_m": 4, "haiku": 0}[c]
    a1.annotate(lab, (x[-1], y[-1]), xytext=(5, off), textcoords="offset points", va="center", ha="left", color=INK, fontsize=6.6, annotation_clip=False)
a1.axhline(1, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
a1.set_xticks(x); a1.set_xticklabels([str(i) for i in range(10)] + ["10+"]); a1.set_xlim(-0.4, 10.4); a1.set_ylim(0.8, 1.47)
a1.set_xlabel("Reply depth", color=INK2); a1.set_ylabel("Toxicity rate ÷ rate at depth 0", color=INK2)
a1.set_title(f"(a) The same {P['n_comments_in_threads'] / 1e6:.1f}M comments under three labelers", loc="left", color=INK); style(a1)
# (b) forest
mg = H3["minus_gold"]; Kg = G["item"]
rows = [("Haiku 5.5", K["haiku"], mg["haiku"], BLUE, "s", True), ("Haiku 5.5 + parent", K["haiku_ctx"], mg["haiku_ctx"], BLUE, "s", False),
        ("Qwen2.5-7B", K["qwen7b_m"], mg["qwen7b_m"], AQUA, "D", True), ("Qwen2.5-7B + parent", K["qwen7b_m_ctx"], mg["qwen7b_m_ctx"], AQUA, "D", False),
        ("Gemma 3 27B", Kg["m0"]["kappa"], G["audit"]["slope_m0_minus_gold"], ORANGE, "^", True),
        ("Gemma 3 27B + parent", Kg["m1"]["kappa"], G["audit"]["G1b_slope_m1_minus_gold"], ORANGE, "^", False)]
for i, (lab, k, (m, lo, hi), col, mk, filled) in enumerate(rows):
    yy = len(rows) - 1 - i
    a2.plot([1e3 * lo, 1e3 * hi], [yy, yy], color=col, lw=1.4, solid_capstyle="butt")
    a2.plot(1e3 * m, yy, marker=mk, ms=4.2, color=col if filled else "white", mec=col, mew=1.1, zorder=3)
a2.set_yticks(range(len(rows))); a2.set_yticklabels([f"{r[0]}   κ = {r[1]:.2f}" for r in rows][::-1], color=INK, fontsize=6.6)
a2.axvline(0, color=INK, lw=0.8); a2.annotate("gold", (0, len(rows) - 0.45), xytext=(3, 0), textcoords="offset points", fontsize=6.6, color=INK, va="center")
a2.set_ylim(-0.6, len(rows) - 0.2); a2.set_xlabel("Depth slope minus gold's (×10⁻³)", color=INK2)
a2.set_title("(b) Item agreement does not predict the trend", loc="left", color=INK, x=-0.62); style(a2)
os.makedirs(f"{PD}/figs", exist_ok=True); fig.savefig(f"{PD}/figs/escalation.pdf"); fig.savefig(f"{PD}/figs/escalation.png", dpi=220); plt.close(fig)
# Haiku's kappa on the same 5% sample as Gemma's (stored labels)
ids = set(np.loadtxt(f"{W}/results/second/s5_ids.txt", dtype=np.int64).tolist())
I = pd.read_parquet(f"{W}/results/f51_items_haiku.parquet", columns=["id", "z", "y"]); I = I[I.id.isin(ids)]
z, y = I.z.values.astype(float), I.y.values.astype(float); pe = z.mean() * y.mean() + (1 - z.mean()) * (1 - y.mean())
num = {"figKappaHaikuSfive": f"{((z == y).mean() - pe) / (1 - pe):.2f}", "figGoldEnd": f"{end['Z']:.2f}", "figHaikuEnd": f"{end['haiku']:.2f}", "figQwenEnd": f"{end['qwen7b_m']:.2f}", "figComments": f"{P['n_comments_in_threads']:,}".replace(",", "{,}"), "figCommentsM": f"{P['n_comments_in_threads'] / 1e6:.1f}", "figHaikuRise": f"{100 * (end['haiku'] - 1):.0f}\\%", "figGoldFall": f"{100 * (1 - end['Z']):.0f}\\%"}
with open(f"{PD}/numbers_fig.tex", "w") as fh:
    fh.write("% generated by scripts/f201_paper_fig.py; do not edit\n")
    for k, v in num.items():
        assert k.isalpha(), k; fh.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print(num)
