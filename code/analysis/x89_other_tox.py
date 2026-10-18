"""Addendum 89, B89a: the toxicity question on the other corpora (CAD, Qian Reddit, Qian Gab, BAD, CGA-Wiki).
Labels: results/x89/labels/<corpus>_<model>_ctx<c>/shard_*.jsonl (batch_manager.py, task tox, the Civil Comments prompt verbatim).
Flags: answer A or B; with the parent, replies take the context label and roots the no-context label. Against each corpus's own gold,
500 conversation-cluster Poisson resamples (seed 8901; clusters = the population's cluster column):
  kappa, TPR, FPR; the depth law (points per level, depth capped at 10) with Proposition 1's terms (uniform, S_FP, S_TP); the
  within-thread depth law; the placebo depth slope (points per level, and relative to the FPR); the log reply RR and placebo RR (not
  CGA-Wiki); among gold negatives the FPR on comments that address 'you' and on the others, the share addressing 'you' at the root and
  at depth >= 6, and the pooled within-'you'-stratum placebo slope. The same for the stored corpus-question LLM labels and for
  Detoxify-original at gold prevalence (references).
H89a: Haiku without the parent, FPR('you') - FPR(other) CI above 0 on >= 4 of 5 corpora. H89b: its placebo depth slope CI above 0 on
>= 2 of 5 corpora.
usage: python3 x89_other_tox.py OUT_JSON [corpus ...]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
CORP = sys.argv[2:] or ["cad", "qian_reddit", "qian_gab", "bad", "cga"]
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)


def load_labels(corpus, model, ctx):
    d = f"{W}/results/x89/labels/{corpus}_{model}_ctx{ctx}"; out = {}
    for f in sorted(glob.glob(f"{d}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            out[int(r["id"])] = r.get("ans")
    return out


ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
res = {}
for corpus in CORP:
    P = pd.read_parquet(f"{W}/results/x14_pops/{corpus}.parquet"); ids = P.id.astype(np.int64).values; N = len(P)
    z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
    uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
    pr = P.parent_row.values; ch = np.flatnonzero(pr >= 0); pa = pr[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
    T = {}
    TXT = {"cc_threads": (_ROOT + "/scratch/full"), "cc_s5": f"{W}/data/second/s5"}
    for f in sorted(glob.glob(f"{TXT.get(corpus, W + '/results/x89/shards/' + corpus)}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); T[int(r["id"])] = r.get("text") or ""
    you = np.array([bool(YOU.search(T.get(int(i), ""))) for i in ids]); neg = z == 0
    Wb = np.vstack([np.ones(nC), np.random.default_rng(8901).poisson(1.0, size=(NB, nC))])
    def S(v):                                                          # per-resample weighted sums of v
        return Wb @ np.bincount(cl, v, nC)
    def slope(xv, v, mask):
        m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones_like(xv), xv, v, xv * xv, xv * v))
        return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
    def slope_strata(v, mask, strat):                                  # pooled within-stratum OLS slope (stratum fixed effects)
        num = 0; den = 0
        for s_ in (0, 1):
            m_ = (mask & (strat == s_)).astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones_like(x), x, v, x * x, x * v))
            num = num + (sxy - sx * sy / n_); den = den + (sxx - sx ** 2 / n_)
        return 100 * num / den
    def rr(v, emask=None):
        vp, vc, c_ = v[pa], v[ch], ecl
        if emask is not None:
            vp, vc, c_ = vp[emask], vc[emask], c_[emask]
        a, b, c, d = (Wb @ np.bincount(c_, t, nC) for t in (vp * vc, vp, (1 - vp) * vc, 1 - vp))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log((a / b) / (c / d))
    def rate(v, mask):
        return S(v * mask) / S(mask.astype(float))
    def stats(y):
        tpr = rate(y, z == 1); fpr = rate(y, neg); py = S(y) / S(np.ones(N)); pz = S(z) / S(np.ones(N))
        po = S((y == z).astype(float)) / S(np.ones(N)); pe = py * pz + (1 - py) * (1 - pz)
        g = slope(x, z, np.ones(N, bool)); l = slope(x, y, np.ones(N, bool))
        # Proposition 1 at the point estimate (weights from the full population)
        w = (x - x.mean()) / ((x - x.mean()) ** 2).sum(); TPR, FPR = float(y[z == 1].mean()), float(y[neg].mean())
        terms = dict(uniform=100 * (TPR - FPR - 1) * float(w @ z), S_FP=100 * float(w[neg] @ (y[neg] - FPR)), S_TP=100 * float(w[~neg] @ (y[~neg] - TPR)))
        out = dict(flag_rate=ci(py), kappa=ci((po - pe) / (1 - pe)), tpr=ci(tpr), fpr=ci(fpr), depth=ci(l), distortion=ci(l - g), terms=terms,
                   within=ci(slope(xt, y, np.ones(N, bool))), within_distortion=ci(slope(xt, y, np.ones(N, bool)) - slope(xt, z, np.ones(N, bool))),
                   placebo=ci(slope(x, y, neg)), placebo_rel=ci(slope(x, y, neg) / (100 * fpr)),
                   fpr_you=ci(rate(y, neg & you)), fpr_other=ci(rate(y, neg & ~you)), fpr_you_minus_other=ci(rate(y, neg & you) - rate(y, neg & ~you)),
                   placebo_within_you=ci(slope_strata(y, neg, you.astype(int))),
                   depth_replies=ci(slope(x, y, x >= 1)), depth_replies_distortion=ci(slope(x, y, x >= 1) - slope(x, z, x >= 1)), placebo_replies=ci(slope(x, y, neg & (x >= 1))))
        if corpus != "cga":
            out.update(log_rr=ci(rr(y)), log_rr_distortion=ci(rr(y) - rr(z)), placebo_rr=ci(np.exp(rr(y, (z[pa] == 0) & (z[ch] == 0)))))
        return out
    R = {"N": N, "edges": int(len(ch)), "prevalence": float(z.mean()), "you_share_neg": float(you[neg].mean()),
         "you_root": float(you[neg & (x == 0)].mean()), "you_deep": float(you[neg & (x >= 6)].mean()) if (neg & (x >= 6)).any() else None,
         "n_deep_neg": int((neg & (x >= 6)).sum()), "gold": dict(depth=ci(slope(x, z, np.ones(N, bool))), within=ci(slope(xt, z, np.ones(N, bool))), depth_replies=ci(slope(x, z, x >= 1)))}
    if corpus != "cga":
        R["gold"]["log_rr"] = ci(rr(z))
    R["instruments"] = {}
    for model in ("haiku", "sonnet"):
        if corpus in ("cc_threads", "cc_s5"):                          # Civil Comments: the stored main labels (same prompt)
            if f"y:{model}" not in P:
                continue
            for tag, col in ((f"{model}_tox", f"y:{model}"), (f"{model}_tox_ctx", f"y:{model}_ctx")):
                R["instruments"][tag] = stats(np.nan_to_num(P[col].values.astype(float))) | {"missing": int(P[col].isna().sum())}
            continue
        L0 = load_labels(corpus, model, 0); L1 = load_labels(corpus, model, 1)
        if not L0:
            continue
        a0 = np.array([L0.get(int(i)) for i in ids], dtype=object); y0 = np.isin(a0, ["A", "B"]).astype(float)
        R["instruments"][f"{model}_tox"] = stats(y0) | {"missing": int(sum(v is None for v in a0))}
        if L1:
            a1 = np.array([L1.get(int(i)) if (pr[j] >= 0 and int(i) in L1) else L0.get(int(i)) for j, i in enumerate(ids)], dtype=object)   # no parent text: no-context label
            y1 = np.isin(a1, ["A", "B"]).astype(float)
            R["instruments"][f"{model}_tox_ctx"] = stats(y1) | {"missing": int(sum(v is None for v in a1))}
    for c in P.columns:                                                # references: stored corpus-question LLM labels, Detoxify
        if corpus in ("cc_threads", "cc_s5") and c != "y:detoxify_original@m":
            continue
        if c.startswith("y:") and (c[2:].startswith(("haiku", "sonnet", "gemma", "qwen")) or c[2:] == "detoxify_original@m"):
            R["instruments"][f"ref:{c[2:]}"] = stats(np.nan_to_num(P[c].values.astype(float)))
    res[corpus] = R
    for k_, v in R["instruments"].items():
        print(corpus, f"{k_:28s}", "kappa", round(v["kappa"][0], 2), "FPR", round(v["fpr"][0], 3), "depth", [round(t, 2) for t in v["depth"]], "gold", round(R["gold"]["depth"][0], 2),
              "S_FP", round(v["terms"]["S_FP"], 2), "placebo", [round(t, 2) for t in v["placebo"]], "you-other", [round(t, 3) for t in v["fpr_you_minus_other"]],
              "plac|you", [round(t, 2) for t in v["placebo_within_you"]], "replies", [round(t, 2) for t in v["depth_replies"]], "plac rep", [round(t, 2) for t in v["placebo_replies"]], flush=True)
H = {}
for corpus in res:
    v = res[corpus]["instruments"].get("haiku_tox")
    if v:
        H[corpus] = dict(you_gap_lo=v["fpr_you_minus_other"][1], placebo_lo=v["placebo"][1])
res["H89"] = dict(per_corpus=H, H89a_n=int(sum(h["you_gap_lo"] > 0 for h in H.values())), H89b_n=int(sum(h["placebo_lo"] > 0 for h in H.values())))
res["H89"]["H89a_pass"] = res["H89"]["H89a_n"] >= 4; res["H89"]["H89b_pass"] = res["H89"]["H89b_n"] >= 2
print("H89", res["H89"], flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
