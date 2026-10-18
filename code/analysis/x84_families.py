"""Addendum 84, B84: OLMo-2-1124-7B-Instruct and Granite-3.1-8B-Instruct on the 5% Civil Comments sample (descriptive).
Labels: results/b84/<model>/shard_00k_tox_ctx<c>.jsonl from one of our GPU machines (label_vllm.py: p = option probabilities A-D renormalized;
score = P(A) + P(B)). Flags at gold prevalence (ties at random, seed 8400); with the parent, replies take the context score and roots
the plain score, at the setting's own gold-prevalence threshold. For the 4 settings, against gold, 500 article-cluster resamples (seed
8401): kappa, TPR, FPR; the depth law with Proposition 1's terms; the placebo depth slope (and relative to the FPR); the within-thread
distortion; the thread size law (point value); the log reply RR and the placebo RR; added for the v5 draft (descriptive, as B89a): the
FPR on gold negatives that address 'you' minus the others', the within-'you' placebo slope, and the depth law among replies.
B83d's construct-only test is not run: the paper no longer uses the latent model (Addendum 88).
Added for v5b: the excess of the depth law among replies over gold's, paired over the same resamples; Llama-3.1-8B-Instruct (B93) as model llama31.
usage: python3 x84_families.py OUT_JSON [model ...]   (models: olmo2 granite llama31; default olmo2 granite)"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(np.int64).values; N = len(P)
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]; neg = z == 0
pr = P.parent_row.values; ch = np.flatnonzero(pr >= 0); pa = pr[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
T = {}
for f in sorted(glob.glob(f"{W}/data/second/s5/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); T[int(r["id"])] = r.get("text") or ""
you = np.array([bool(YOU.search(T.get(int(i), ""))) for i in ids])
Wb = np.vstack([np.ones(nC), np.random.default_rng(8401).poisson(1.0, size=(NB, nC))])
S = lambda v: Wb @ np.bincount(cl, v, nC)
def slope(xv, v, mask):
    m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones_like(xv), xv, v, xv * xv, xv * v))
    return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
def slope_strata(v, mask, strat):
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
    return np.log((a / b) / (c / d))
rate = lambda v, m: S(v * m) / S(m.astype(float))
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
def size_law(v):
    ku = np.bincount(uu).astype(float); ok = ku >= 2; fu = np.bincount(uu, v) / ku; lx = np.log2(ku)
    lx, fu = lx[ok], fu[ok]; return 100 * float(((lx - lx.mean()) * (fu - fu.mean())).sum() / ((lx - lx.mean()) ** 2).sum())
def stats(y):
    tpr = rate(y, z == 1); fpr = rate(y, neg); py = S(y) / S(np.ones(N)); pz = S(z) / S(np.ones(N))
    po = S((y == z).astype(float)) / S(np.ones(N)); pe = py * pz + (1 - py) * (1 - pz)
    g = slope(x, z, np.ones(N, bool)); l = slope(x, y, np.ones(N, bool))
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum(); TPR, FPR = float(y[z == 1].mean()), float(y[neg].mean())
    return dict(flag_rate=ci(py), kappa=ci((po - pe) / (1 - pe)), tpr=ci(tpr), fpr=ci(fpr), depth=ci(l), distortion=ci(l - g),
                terms=dict(uniform=100 * (TPR - FPR - 1) * float(w @ z), S_FP=100 * float(w[neg] @ (y[neg] - FPR)), S_TP=100 * float(w[~neg] @ (y[~neg] - TPR))),
                within_distortion=ci(slope(xt, y, np.ones(N, bool)) - slope(xt, z, np.ones(N, bool))), placebo=ci(slope(x, y, neg)), placebo_rel=ci(slope(x, y, neg) / (100 * fpr)),
                size_law=size_law(y), size_law_gold=size_law(z), log_rr=ci(rr(y)), placebo_rr=ci(np.exp(rr(y, (z[pa] == 0) & (z[ch] == 0)))),
                fpr_you_minus_other=ci(rate(y, neg & you) - rate(y, neg & ~you)), placebo_within_you=ci(slope_strata(y, neg, you.astype(int))),
                depth_replies=ci(slope(x, y, x >= 1)), depth_replies_gold=ci(slope(x, z, x >= 1)), depth_replies_excess=ci(slope(x, y, x >= 1) - slope(x, z, x >= 1)))
def scores(model, ctx):
    s = {}
    for f in sorted(glob.glob(f"{W}/results/b84/{model}/shard_*_tox_ctx{ctx}.jsonl")):
        for l in open(f):
            r = json.loads(l); s[int(r["id"])] = float(r["p"][0] + r["p"][1])
    return s
res = {"N": N, "prevalence": float(z.mean()), "gold": dict(depth=ci(slope(x, z, np.ones(N, bool))), log_rr=ci(rr(z)), size_law=size_law(z)), "settings": {}}
k = int(round(z.mean() * N))
for model in (sys.argv[2:] or ["olmo2", "granite"]):
    s0 = scores(model, 0); s1 = scores(model, 1)
    a0 = np.array([s0.get(int(i), np.nan) for i in ids]); a1 = np.array([s1.get(int(i), np.nan) if pr[j] >= 0 and int(i) in s1 else s0.get(int(i), np.nan) for j, i in enumerate(ids)])
    for tag, sc in ((model, a0), (f"{model}_ctx", a1)):
        miss = int(np.isnan(sc).sum()); sc = np.where(np.isnan(sc), -np.inf, sc)
        order = np.lexsort((np.random.default_rng(8400).random(N), -sc)); y = np.zeros(N); y[order[:k]] = 1.0
        res["settings"][tag] = stats(y) | {"missing_scores": miss}
        v = res["settings"][tag]
        print(tag, "missing", miss, "kappa", round(v["kappa"][0], 2), "FPR", round(v["fpr"][0], 3), "depth", [round(t, 2) for t in v["depth"]], "gold", round(res["gold"]["depth"][0], 2),
              "S_FP", round(v["terms"]["S_FP"], 2), "placebo", [round(t, 2) for t in v["placebo"]], "you-other", [round(t, 3) for t in v["fpr_you_minus_other"]],
              "replies", [round(t, 2) for t in v["depth_replies"]], "RR", round(float(np.exp(v["log_rr"][0])), 2), "size", round(v["size_law"], 3), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
