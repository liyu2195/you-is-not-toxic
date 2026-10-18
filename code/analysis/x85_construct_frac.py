"""Addendum 85, B85c': B85c recomputed with the subtype vote fractions of the public release (data/cc_attr_frac.parquet, matched by
x85_match_fractions.py), on the matched gold negatives of Civil Comments reply threads (cc_threads); x = min(depth, 10); 500
article-cluster Poisson resamples (seed 8501, as B85c, drawn over all of cc_threads' articles).
For insult, identity attack, obscene and threat: depth slopes (points per level, OLS on x) of the share with any vote (fraction > 0),
of the mean fraction and of the share with a majority (fraction >= 0.5); the share rejected by the platform's civility review;
Haiku's (and Haiku shown the parent's) flag rate, depth slope and level among all matched gold negatives, among those with zero insult
votes and among those with zero votes of any kind (toxicity and every subtype); the mean insult fraction and any-insult share among the
gold negatives Haiku flags and does not flag; and, on the comments with texts, civil comments (toxicity fraction < 0.2) that address
someone as "you" against the others.
H85c': the depth slope of the any-insult share among matched gold negatives is below a quarter of Haiku's placebo depth slope on the
same comments (paired 95% CI of 0.25 x Haiku's slope - the insult slope above 0).
usage: python3 -I x85_construct_frac.py TEXT_SHARD_DIR OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; SH, OUT = sys.argv[1], sys.argv[2]; PP = 100.0; NB = 500
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "cluster", "depth", "z", "tox", "y:haiku", "y:haiku_ctx"])
F = pd.read_parquet(f"{W}/data/cc_attr_frac.parquet").set_index("id")
M = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "rating"]).set_index("id").reindex(P.id.values)
matched = P.id.isin(F.index).values; A = F.reindex(P.id.values)
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float)
neg_all = z == 0; neg = neg_all & matched
cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
Wb = np.vstack([np.ones(nC), np.random.default_rng(8501).poisson(1.0, size=(NB, nC))])


def slope(v, mask):
    m = mask.astype(float); v = np.nan_to_num(v)
    n, sx, sy, sxx, sxy = (Wb @ np.bincount(cl, m * a, nC) for a in (np.ones_like(x), x, v, x * x, x * v))
    return (n * sxy - sx * sy) / (n * sxx - sx ** 2), sy / n


ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
fr = {k: A[k].values.astype(float) for k in ("insult", "identity_attack", "obscene", "threat", "severe_toxicity", "sexual_explicit")}
tox_hf = A.toxicity_hf.values.astype(float); tox = P.tox.values.astype(float)
rej = (M.rating.values == "rejected").astype(float)
none = matched & (tox == 0) & np.all([np.nan_to_num(fr[k], nan=1.0) == 0 for k in fr], axis=0)
hk = np.nan_to_num(P["y:haiku"].values.astype(float)); hkc = np.nan_to_num(P["y:haiku_ctx"].values.astype(float))
res = {"n_neg_all": int(neg_all.sum()), "n_neg_matched": int(neg.sum()), "match_rate_neg": float(neg.sum() / neg_all.sum()),
       "match_rate_cc_threads": float(matched.mean()), "share_none_among_neg": float(none[neg].mean()),
       "max_abs_tox_diff": float(np.nanmax(np.abs(tox_hf[matched] - tox[matched]))), "slopes": {}, "levels": {}}
S = {}
series = {"rejected": rej, "haiku": hk, "haiku_ctx": hkc}
for k in ("insult", "identity_attack", "obscene", "threat"):
    v = fr[k]; series[f"{k}_any"] = (v > 0).astype(float); series[f"{k}_mean"] = v; series[f"{k}_major"] = (v >= 0.5).astype(float)
for k, v in series.items():
    s, mu = slope(v, neg); S[k] = s; res["slopes"][k] = ci(s * PP); res["levels"][k] = ci(mu * PP)
ins = np.nan_to_num(fr["insult"])
for k, v in (("haiku", hk), ("haiku_ctx", hkc)):
    s0, mu0 = slope(v, neg & (ins == 0)); s1, mu1 = slope(v, neg & none)
    res["slopes"][f"{k}_zero_insult"] = ci(s0 * PP); res["levels"][f"{k}_zero_insult"] = ci(mu0 * PP)
    res["slopes"][f"{k}_zero_votes"] = ci(s1 * PP); res["levels"][f"{k}_zero_votes"] = ci(mu1 * PP)
    S[f"{k}_zero_votes"] = s1
res["share_zero_insult_among_neg"] = float((ins[neg] == 0).mean())
res["insult_by_haiku"] = {"mean_flagged": float(ins[neg & (hk == 1)].mean()), "mean_not_flagged": float(ins[neg & (hk == 0)].mean()),
                          "any_flagged": float((ins[neg & (hk == 1)] > 0).mean()), "any_not_flagged": float((ins[neg & (hk == 0)] > 0).mean()),
                          "none_flagged": float(none[neg & (hk == 1)].mean()), "none_not_flagged": float(none[neg & (hk == 0)].mean())}
d = 0.25 * S["haiku"] - S["insult_any"]; res["H85c_prime"] = dict(diff=ci(d * PP), pass_=bool(np.nanpercentile(d[1:], 2.5) > 0))
res["ratio_insult_any_to_haiku"] = ci(S["insult_any"] / S["haiku"])
# "you" on comments with texts (civil comments: toxicity fraction < 0.2), matched only
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
pos = pd.Series(np.arange(len(P)), index=P.id.astype(np.int64).values); you = np.full(len(P), np.nan)
for f in sorted(glob.glob(f"{SH}/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = pos.get(int(r["id"]))
        if i is not None:
            you[i] = float(bool(YOU.search(r.get("text") or "")))
have = ~np.isnan(you); civ = (tox < 0.2) & have & matched
res["you"] = dict(coverage=float(have.mean()), n_civil=int(civ.sum()), share_you=float((you[civ] == 1).mean()),
                  insult_any_you=float((ins[civ & (you == 1)] > 0).mean()), insult_any_noyou=float((ins[civ & (you == 0)] > 0).mean()),
                  insult_mean_you=float(ins[civ & (you == 1)].mean()), insult_mean_noyou=float(ins[civ & (you == 0)].mean()),
                  haiku_you=float(hk[civ & (you == 1)].mean()), haiku_noyou=float(hk[civ & (you == 0)].mean()),
                  rejected_you=float(rej[civ & (you == 1)].mean()), rejected_noyou=float(rej[civ & (you == 0)].mean()))
print(json.dumps(res, indent=1)[:6000])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
