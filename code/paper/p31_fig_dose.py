"""Paper v5, Figure 2: the dose of second-person address, on Civil Comments (results/x90/x90_dose.json, scripts/x90_dose.py).
(a) the false-positive rate among gold negatives by the number of second-person words (0, 1, 2, 3, 4+): Haiku (reply threads, native
cut), Sonnet (5% sample, native cut), Detoxify-original (reply threads, at gold prevalence), and a single human rater against the other
raters' majority (reply threads, exact). (b) Haiku's false-positive rate by reply depth
(8 = 8 or more), observed and directly standardized to the root's distribution of the number of second-person words.
usage: python3 p31_fig_dose.py OUT_PDF"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = _ROOT; OUT = sys.argv[1]
d = json.load(open(f"{W}/results/x90/x90_dose.json"))["corpora"]; ct = d["cc_threads"]; s5 = d["cc_s5"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"; C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated: blue, orange, aqua
K = list(range(5)); pc = lambda L: [100 * t[0] for t in L]
A = {"Haiku": (pc(ct["labelers"]["haiku"]["fpr_by_count"]), C2, "o"), "Sonnet": (pc(s5["labelers"]["sonnet"]["fpr_by_count"]), C3, "s"),
     "Detoxify": (pc(ct["labelers"]["detoxify"]["fpr_by_count"]), C1, "D"), "one rater": (pc(ct["single_rater"]["fpr_by_count"]), INK2, "^")}
raw = pc(ct["figure"]["raw"]); std = pc(ct["figure"]["standardized"]); D = ct["figure"]["depth"]
json.dump(dict(count=K, fpr={k: v[0] for k, v in A.items()}, depth=D, haiku_raw=raw, haiku_standardized=std, mean_count_by_depth=ct["mean_count_by_depth"]),
          open(OUT.replace(".pdf", ".json"), "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(3.33, 1.62), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.5, zorder=0); a.tick_params(colors=INK2, labelsize=6.0, pad=1.5, length=2)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(INK2); a.spines[sp].set_linewidth(0.6)
for name, (v, c, mk) in A.items():
    ax[0].plot(K, v, color=c, lw=1.6, marker=mk, ms=3.0, mec=SURF, mew=0.6, zorder=4)
    ax[0].text(4.25, v[-1] + {"Sonnet": 3.0, "Haiku": -3.0, "Detoxify": -2.2, "one rater": 2.4}.get(name, 0.0), name, fontsize=5.8, color=INK2, va="center")   # ends close together
ax[0].set_xticks(K); ax[0].set_xticklabels(["0", "1", "2", "3", "4+"]); ax[0].set_xlim(-0.3, 5.6); ax[0].set_ylim(0, 60)
ax[0].set_xlabel("second-person words in the comment", fontsize=6.2, color=INK2, labelpad=1)
ax[0].set_title("(a) false-positive rate (%) by\nnumber of second-person words", fontsize=6.4, color=INK, loc="left", pad=2)
ax[1].plot(D, raw, color=C2, lw=1.6, marker="o", ms=3.0, mec=SURF, mew=0.6, zorder=4)
ax[1].plot(D, std, color=INK, lw=1.2, ls=(0, (3, 1.6)), zorder=3)
ax[1].text(8.3, raw[-1], "observed", fontsize=5.8, color=INK2, va="center"); ax[1].text(8.3, std[-1] - 0.6, "roots' mix", fontsize=5.8, color=INK2, va="center")
ax[1].set_xticks([0, 2, 4, 6, 8]); ax[1].set_xticklabels(["0", "2", "4", "6", "8+"]); ax[1].set_xlim(-0.4, 11.6); ax[1].set_ylim(0, 30)
ax[1].set_xlabel("reply depth", fontsize=6.2, color=INK2, labelpad=1)
ax[1].set_title("(b) Haiku's false-positive\nrate (%) by depth", fontsize=6.4, color=INK, loc="left", pad=2)
fig.tight_layout(pad=0.25, w_pad=0.9); fig.savefig(OUT, facecolor=SURF); fig.savefig(OUT.replace(".pdf", ".png"), dpi=220, facecolor=SURF)
print("a", {k: [round(t, 1) for t in v[0]] for k, v in A.items()}); print("b raw", [round(t, 1) for t in raw], "std", [round(t, 1) for t in std])
