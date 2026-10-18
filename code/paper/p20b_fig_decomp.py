"""Paper v7h (earlier draft) and arXiv, Figure 2 (from p20_fig_decomp.py; v7h: "+ parent" contagion over edges between replies): which part of each law the instrument writes, on Civil Comments (exact decomposition, Proposition 1).
Data: results/x74_decomp/x74_decomp.json (cc_threads; Sonnet and Gemma from the 5% sample, cc_s5).
Per instrument, the distortion law(Y) - law(Z) split into: uniform errors (TPR - FPR - 1) law(Z), structured false positives
S_FP, structured hits S_TP (escalation, thread size law); for contagion the same three terms from the permutation baseline of
B75c (results/x75_decomp/x75_decomp.json). Qwen and Gemma shown the parent are at their own gold-prevalence threshold (B76e,
results/x76_rethreshold/x76_rethreshold.json). The human row is a single rater against the other raters' majority (B75b; median of
20 draws, results/x75_onerater/x75_onerater_decomp.json and x75_onerater.json).
usage: python3 p20_fig_decomp.py OUT_PDF"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open((_ROOT + "/results/x74_decomp/x74_decomp.json"))); OUT = sys.argv[1]
RC = json.load(open((_ROOT + "/results/x75_decomp/x75_decomp.json")))
SR = json.load(open((_ROOT + "/results/x75_onerater/x75_onerater_decomp.json")))["median"]
RT = json.load(open((_ROOT + "/results/x76_rethreshold/x76_rethreshold.json")))
RP = json.load(open((_ROOT + "/results/x97/x97_decomp_replies.json")))            # v6: "+ parent" escalation rows among replies only
RX = json.load(open((_ROOT + "/results/x99/x99_ctx_replies.json")))              # v7h: "+ parent" contagion over edges between replies (x99_ctx_replies.py)
CTX = {"haiku_ctx": ("cc_threads", "haiku_ctx"), "sonnet_ctx": ("cc_s5", "sonnet_ctx"), "qwen7b_m_ctx@own": ("cc_threads", "qwen7b_m_ctx@own"),
       "gemma_m_ctx@own": ("cc_s5", "gemma_m_ctx@own")}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C_U, C_FP, C_TP = "#9aa0a6", "#eb6834", "#2a78d6"
ROWS = [("Haiku", "cc_threads", "haiku"), ("Haiku + parent", "cc_threads", "haiku_ctx"),
        ("Sonnet$^{\\dagger}$", "cc_s5", "sonnet"), ("Sonnet + parent$^{\\dagger}$", "cc_s5", "sonnet_ctx"),
        ("Qwen", "cc_threads", "qwen7b_m"), ("Qwen + parent", "own:cc_threads", "qwen7b_m_ctx@own"),
        ("Gemma$^{\\dagger}$", "cc_s5", "gemma_m"), ("Gemma + parent$^{\\dagger}$", "own:cc_s5", "gemma_m_ctx@own"),
        ("Detoxify (0.6)", "cc_threads", "detoxify_original@0.6"), ("Detoxify", "cc_threads", "detoxify_original@m"), ("Detoxify-unbiased$^{\\S}$", "cc_threads", "detoxify_unbiased@m"),
        ("IMSYPP (argmax)", "cc_threads", "imsypp@argmax"), ("IMSYPP", "cc_threads", "imsypp@m"),
        ("s-nlp RoBERTa$^{\\S}$", "cc_threads", "sn_rob@m"), ("martin-ha DistilBERT$^{\\S}$", "cc_threads", "mh_dist@m"),
        ("Dynabench hate", "cc_threads", "fb_hate@m"), ("CardiffNLP offensive", "cc_threads", "cn_off@m"),
        ("One rater", "sr", "single")]
SEPS = [7.5, 16.5]                                                   # rule lines between the LLMs, the classifiers and the human row (row index from the top)
PANELS = [("depth", "Escalation: depth law minus gold's\n(percentage points per reply level)", 100.0),
          ("size", "Thread size law minus gold's\n(percentage points per doubling)", 100.0),
          ("log_rr", "Contagion: log reply risk ratio\nminus gold's", 1.0)]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), facecolor=SURF, sharey=True)          # true print size (ACM \textwidth)
ypos = np.arange(len(ROWS))[::-1]
for ax, (law, title, sc) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    for y, (lab, pop, L) in zip(ypos, ROWS):
        if pop is None:
            continue
        if L in CTX and law == "size":                                     # the size law of a thread labeled in two modes is not shown
            continue
        if L in CTX and law == "log_rr":                                   # v7h: both ends labeled with their parent shown
            q_ = RX[CTX[L][0]]["labelers"][CTX[L][1]]
            parts = [(q_["uniform"], C_U), (q_["fp"], C_FP), (q_["tp"], C_TP)]; d = q_["log_rr_distortion"]
        elif L in CTX and law == "depth":
            q_ = RP[CTX[L][0]]["labelers"][CTX[L][1]]
            parts = [(q_["uniform"] / 100, C_U), (q_["fp"] / 100, C_FP), (q_["tp"] / 100, C_TP)]; d = [t / 100 for t in q_["distortion"]]
        elif pop.startswith("own:"):
            q_ = RT[pop[4:]]["labelers"][L]; g_ = RT[pop[4:]]["gold"]
            if law == "log_rr":
                c_ = q_["contagion_terms"]; parts = [(c_["uniform"], C_U), (c_["fp"], C_FP), (c_["tp"], C_TP)]; d = q_["laws"]["log_rr"]["distortion"]
            else:
                r = q_["laws"][law]; parts = [((r["attenuation_term"][0] - g_[law][0]), C_U), (r["fp_term"][0], C_FP), (r["tp_term"][0], C_TP)]; d = r["distortion"]
        elif pop == "sr":
            m_ = SR[law]; parts = [(m_["uniform"], C_U), (m_["fp"], C_FP), (m_["tp"], C_TP)]; d = [m_["distortion"]] * 3
        elif law == "log_rr":
            r = RC[pop]["labelers"][L]; parts = [(r["uniform"], C_U), (r["fp"], C_FP), (r["tp"], C_TP)]
            d = R[pop]["labelers"][L]["laws"][law]["distortion"]
        else:
            r = R[pop]["labelers"][L]["laws"][law]
            parts = [((r["attenuation_term"][0] - r["gold"][0]), C_U), (r["fp_term"][0], C_FP), (r["tp_term"][0], C_TP)]; d = r["distortion"]
        pos, neg = 0.0, 0.0
        for v, col in parts:
            v = v * sc
            if v >= 0:
                ax.barh(y, v, left=pos, height=0.66, color=col, edgecolor=SURF, linewidth=0.6, zorder=3); pos += v
            else:
                ax.barh(y, v, left=neg, height=0.66, color=col, edgecolor=SURF, linewidth=0.6, zorder=3); neg += v
        ax.errorbar(d[0] * sc, y, xerr=[[d[0] * sc - d[1] * sc], [d[2] * sc - d[0] * sc]], fmt="o", ms=2.9, color=INK,
                                         mec=SURF, mew=0.5, ecolor=INK, elinewidth=0.8, capsize=0, zorder=5)
    ax.axvline(0, color=INK2, lw=0.7, zorder=2); ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    for sp_ in SEPS:
        ax.axhline(len(ROWS) - 1 - sp_, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_title(title, fontsize=6.9, color=INK, loc="left", pad=3); ax.tick_params(colors=INK2, labelsize=6.2, pad=1.5); ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_yticks(ypos); axes[0].set_yticklabels([r[0] for r in ROWS], fontsize=6.3); axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
for t, (lab, pop, L) in zip(axes[0].get_yticklabels(), ROWS):
    if pop is None:
        t.set_fontweight("bold"); t.set_color(INK)
    else:
        t.set_color(INK2)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h = [Patch(color=C_U, label="unstructured errors"), Patch(color=C_FP, label="structured false positives"),
     Patch(color=C_TP, label="structured true positives"), Line2D([], [], marker="o", ls="", color=INK, label="total distortion (95% CI)")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.4, labelcolor=INK2, handlelength=1.2, columnspacing=1.4, borderaxespad=0.1)
fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.5, pad=0.2); fig.savefig(OUT, facecolor=SURF); fig.savefig(OUT.replace(".pdf", ".png"), dpi=150, facecolor=SURF)
print("saved", OUT)
