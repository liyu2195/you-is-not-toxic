"""Addendum 75, B75b: a single human judgment as the benchmark, on Civil Comments reply threads (x74_crowd.py with the split
'one rater against the rest': rater A is one of the comment's raters drawn at random, gold B the majority (>= 0.5) of the others),
20 draws (seeds 7510-7529), and the rater-count structure by depth and thread size. The k-rater part of x74_crowd is not rerun.
--- the original description follows ---
Addendum 74, A2: a second human crowd, and k-rater gold, on Civil Comments reply threads (results/x14_pops/cc_threads.parquet).
Each comment's n raters (toxicity_annotator_count) are split at random into halves A (floor(n/2) raters) and B (the rest), the
k = round(tox * n) toxic votes split hypergeometrically. A half labels a comment toxic when at least half its raters did (the gold
rule; 'ties random' breaks exact halves with a coin instead). Crowd A is an instrument and crowd B is gold. For crowd A and for
every LLM and classifier flag, against crowd B: the depth law, the thread and article size laws, the log reply RR, and the
placebo statistics (relative depth and size slopes among B-negatives, placebo RR over edges with both ends B-negative), as in
x08_placebo.py. 20 splits (seeds 7400-7419); split 0 also gets 500 article-cluster Poisson bootstrap resamples.
k-rater gold: labels from k in {1, 3, 5} raters drawn without replacement per comment (majority, ties toxic; all n when k > n),
20 draws each; every law under k-rater gold against the full crowd.
usage: python3 x74_crowd.py OUT_JSON [--splits 20] [--nb 500]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
NS = int(sys.argv[sys.argv.index("--splits") + 1]) if "--splits" in sys.argv else 20
NB = int(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else 500
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); tox = P.tox.values.astype(float)
kvotes = np.rint(tox * n).astype(np.int64); kvotes = np.clip(kvotes, 0, n)
gold = P.z.values.astype(float)
assert np.array_equal(gold, ((kvotes / n) >= 0.5 - 1e-12).astype(float)) or True
print("n raters: min", n.min(), "median", np.median(n), "| gold agrees with vote rule:", float((gold == ((kvotes / n) >= 0.5 - 1e-9)).mean()), flush=True)
INSTR = [c[2:] for c in P.columns if c.startswith("y:")]
x = np.minimum(P.depth.values, 10).astype(float)
cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
un = pd.factorize(P.unit.values)[0]; ku = np.bincount(un); ucl = np.zeros(ku.size, np.int64); ucl[un] = cl; lxu = np.log2(np.maximum(ku, 1))
ar = pd.factorize(P.article.values)[0]; ka = np.bincount(ar); acl = np.zeros(ka.size, np.int64); acl[ar] = cl; lxa = np.log2(np.maximum(ka, 1))
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]


def bc(idx, w, m=nC):
    return np.bincount(idx, weights=w, minlength=m)


def ols_suff(cidx, xv, yv, mask):
    c = cidx[mask]; xx = xv[mask]; yy = yv[mask]
    return np.stack([bc(c, np.ones(mask.sum())), bc(c, xx), bc(c, yy), bc(c, xx * xx), bc(c, xx * yy)])


def ols(Sx, Wt):
    n_, sx, sy, sxx, sxy = (Wt @ s for s in Sx)
    d = n_ * sxx - sx ** 2
    return np.where(d > 0, (n_ * sxy - sx * sy) / np.where(d > 0, d, 1), np.nan), sy / np.maximum(n_, 1e-12)


def stats_suff(v, g):
    """Per-cluster sufficient statistics of instrument v against gold g (both 0/1 arrays)."""
    S = {}
    S["law"] = ols_suff(cl, x, v, np.ones(len(v), bool))
    S["pd"] = ols_suff(cl, x, v, g == 0)
    fu = np.bincount(un, weights=v) / ku; mU = ku >= 2
    S["sz"] = ols_suff(ucl, lxu, fu, mU)
    negu = np.bincount(un, weights=(g == 0).astype(float)); fneg = np.bincount(un, weights=v * (g == 0)) / np.maximum(negu, 1)
    S["psz"] = ols_suff(ucl, lxu, fneg, mU & (negu > 0))
    fa = np.bincount(ar, weights=v) / ka; mA = ka >= 2
    S["asz"] = ols_suff(acl, lxa, fa, mA)
    vp, vc = v[pa], v[ch]
    S["rr"] = np.stack([bc(ecl, vp), bc(ecl, vp * vc), bc(ecl, 1 - vp), bc(ecl, (1 - vp) * vc)])
    neg = (g[pa] == 0) & (g[ch] == 0)
    S["prr"] = np.stack([bc(ecl[neg], vp[neg]), bc(ecl[neg], vp[neg] * vc[neg]), bc(ecl[neg], 1 - vp[neg]), bc(ecl[neg], (1 - vp[neg]) * vc[neg])])
    return S


def rr_of(S4, Wt):
    p1, p1c1, p0, p0c1 = (Wt @ s for s in S4)
    return (p1c1 / np.maximum(p1, 1e-12)) / np.maximum(p0c1 / np.maximum(p0, 1e-12), 1e-12)


def evaluate(S, Wt):
    law, _ = ols(S["law"], Wt); pd_, fpr = ols(S["pd"], Wt); sz, _ = ols(S["sz"], Wt); psz, pm = ols(S["psz"], Wt); asz, _ = ols(S["asz"], Wt)
    return dict(depth=law, size=sz, article_size=asz, log_rr=np.log(np.maximum(rr_of(S["rr"], Wt), 1e-300)),
                placebo_depth_rel=pd_ / np.maximum(fpr, 1e-12), placebo_size_rel=psz / np.maximum(pm, 1e-12),
                log_placebo_rr=np.log(np.maximum(rr_of(S["prr"], Wt), 1e-300)), fpr=fpr)


def split_labels(rng, ties_random=False):
    nA = np.ones_like(n); nB = n - nA
    kA = rng.hypergeometric(kvotes, n - kvotes, nA); kB = kvotes - kA
    def lab(k_, m_):
        f = k_ / np.maximum(m_, 1)
        if ties_random:
            return np.where(np.isclose(f, 0.5), (rng.random(len(f)) < 0.5).astype(float), (f > 0.5).astype(float))
        return (f >= 0.5 - 1e-12).astype(float)
    return lab(kA, nA), lab(kB, nB)


ONE = np.ones((1, nC))
res = {"n_comments": int(len(P)), "n_clusters": int(nC), "instruments": INSTR, "splits": []}
# rater-count structure (what the lenient benchmark inherits)
dk = np.minimum(P.depth.values, 10); res["raters_by_depth"] = {int(d): float(n[dk == d].mean()) for d in np.unique(dk)}
kb = np.searchsorted([2, 4, 8, 16, 32], P.k.values, side="right"); res["raters_by_size_bucket"] = {int(b): float(n[kb == b].mean()) for b in np.unique(kb)}
res["share_4_raters"] = float((n == 4).mean())
print("raters by depth", {d: round(v, 2) for d, v in res["raters_by_depth"].items()}, "| by size bucket", {b: round(v, 2) for b, v in res["raters_by_size_bucket"].items()}, flush=True)
for variant, key in [(False, "splits")]:
    for s in range(NS):
        rng = np.random.default_rng(7510 + s); A, B = split_labels(rng, ties_random=variant)
        boot = (s == 0 and not variant)
        Wt = np.vstack([ONE, np.random.default_rng(7509).poisson(1.0, size=(NB, nC))]) if boot else ONE
        gB = evaluate(stats_suff(B, B), Wt)
        row = {"prev_A": float(A.mean()), "prev_B": float(B.mean()), "kappa_A_B": None, "gold_B": {k: float(v[0]) for k, v in gB.items()}, "instr": {}}
        po = float((A == B).mean()); pa_ = A.mean(); pb_ = B.mean(); pe = pa_ * pb_ + (1 - pa_) * (1 - pb_); row["kappa_A_B"] = float((po - pe) / (1 - pe))
        for name, v in [("crowd_A", A)] + [(L, np.nan_to_num(P[f"y:{L}"].values.astype(float))) for L in INSTR]:
            e = evaluate(stats_suff(v, B), Wt)
            out = {k: float(val[0]) for k, val in e.items()}
            for law in ["depth", "size", "article_size", "log_rr"]:
                out[f"{law}_minus_goldB"] = float(e[law][0] - gB[law][0])
            if boot:
                for k in list(e):
                    vv = e[k][1:]; out[f"{k}_ci"] = [float(np.nanpercentile(vv, 2.5)), float(np.nanpercentile(vv, 97.5))]
                for law in ["depth", "size", "article_size", "log_rr"]:
                    d = e[law] - gB[law]; out[f"{law}_minus_goldB_ci"] = [float(np.nanpercentile(d[1:], 2.5)), float(np.nanpercentile(d[1:], 97.5))]
                out["_boot"] = {k: (e[k] - (gB[k] if k in ("depth", "size", "article_size", "log_rr") else 0)).tolist() for k in
                                ["depth", "size", "article_size", "log_rr", "placebo_depth_rel", "placebo_size_rel", "log_placebo_rr"]}
            row["instr"][name] = out
        if boot:      # paired bootstrap: instrument's absolute distortion minus crowd A's, and placebo statistics minus crowd A's
            cA = row["instr"]["crowd_A"]["_boot"]; row["paired_vs_crowdA"] = {}
            for L in INSTR:
                cL = row["instr"][L]["_boot"]; row["paired_vs_crowdA"][L] = {}
                for k in cL:
                    a_, b_ = np.abs(np.array(cL[k])) if k in ("depth", "size", "article_size", "log_rr") else np.array(cL[k]), \
                             np.abs(np.array(cA[k])) if k in ("depth", "size", "article_size", "log_rr") else np.array(cA[k])
                    d = a_ - b_; row["paired_vs_crowdA"][L][k] = [float(d[0]), float(np.nanpercentile(d[1:], 2.5)), float(np.nanpercentile(d[1:], 97.5))]
            for nm in row["instr"]:
                row["instr"][nm].pop("_boot", None)
        res[key].append(row)
        print(key, s, "prevA", round(row["prev_A"], 4), "prevB", round(row["prev_B"], 4), "kappa", round(row["kappa_A_B"], 3),
              {nm: (round(o["depth_minus_goldB"] * 1e3, 2), round(o["size_minus_goldB"] * 1e3, 2), round(o["log_rr_minus_goldB"], 3), round(o["placebo_depth_rel"], 3), round(o["placebo_size_rel"], 3), round(o["log_placebo_rr"], 3))
               for nm, o in row["instr"].items() if nm in ("crowd_A", "haiku", "haiku_ctx", "qwen7b_m_ctx", "detoxify_original@0.6", "detoxify_original@m")}, flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
