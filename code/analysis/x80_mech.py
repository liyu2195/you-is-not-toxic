"""Addendum 80, B80d: what makes classifiers' false positives cluster (descriptive; Civil Comments reply threads).
(1) Identity terms: a lexical detector (whole words, case-insensitive) for the identity subgroups of Borkan et al. (2019): gender,
    sexual orientation, religion, race or ethnicity, disability. For each instrument: the FPR on gold negatives with and without a
    mention, and the placebo RR over all gold-negative edges and over those where neither end mentions an identity.
(2) Shared parent: the placebo RR over sibling pairs (consecutive replies to the same parent, both gold-negative; each pair counted in
    both orders) against reply edges.
(3) Shared article: the Mantel-Haenszel placebo RR stratified by article.
Benchmark: the single rater (one rater per comment at random, gold = the others' majority; 20 draws, seed 8050), whose placebo
statistics use its own gold. Point estimates; 200 article-cluster Poisson resamples (seed 8051) for the no-identity ratio.
usage: python3 x80_mech.py TEXT_SHARD_DIR OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; SH, OUT = sys.argv[1], sys.argv[2]
TERMS = {
 "gender": r"men|man|women|woman|male|males|female|females|boy|boys|girl|girls|transgender|trans|feminist|feminists",
 "sexual_orientation": r"gay|gays|lesbian|lesbians|homosexual|homosexuals|bisexual|lgbt|lgbtq|queer|heterosexual",
 "religion": r"christian|christians|christianity|catholic|catholics|protestant|protestants|jew|jews|jewish|judaism|muslim|muslims|islam|islamic|hindu|hindus|buddhist|buddhists|atheist|atheists",
 "race": r"black|blacks|white|whites|asian|asians|latino|latinos|latina|hispanic|hispanics|african|africans|mexican|mexicans|indigenous|arab|arabs|immigrant|immigrants|racist|racists",
 "disability": r"disabled|disability|disabilities|autistic|mentally ill|mental illness"}
IDRE = re.compile(r"\b(" + "|".join(TERMS.values()) + r")\b", re.I)
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet"); N = len(P)
idx = pd.Series(np.arange(N), index=P.id.astype(np.int64).values); ident = np.zeros(N, bool)
for f in sorted(glob.glob(f"{SH}/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = idx.get(int(r["id"]))
        if i is not None:
            ident[i] = bool(IDRE.search(r.get("text") or ""))
z = P.z.values.astype(float); neg = z == 0
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
art = pd.factorize(P.article.values)[0]; cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1
# sibling pairs: consecutive replies (by row) to the same parent
o = np.lexsort((ch, pa)); pa_s, ch_s = pa[o], ch[o]; same = pa_s[1:] == pa_s[:-1]; s1, s2 = ch_s[:-1][same], ch_s[1:][same]
sa, sb = np.r_[s1, s2], np.r_[s2, s1]
print("comments", N, "identity share", ident.mean().round(3), "edges", len(ch), "sibling pairs", len(s1), flush=True)
def rr_pairs(y, a, b, m):
    ya, yb = y[a][m], y[b][m]
    return float((yb[ya == 1].mean()) / (yb[ya == 0].mean())) if (ya == 1).any() and (ya == 0).any() and yb[ya == 0].mean() > 0 else float("nan")
def mh_rr(y, a, b, m, strata):
    ya, yb, s = y[a][m], y[b][m], strata[b][m]; ns = s.max() + 1
    a_ = np.bincount(s, ya * yb, ns); n1 = np.bincount(s, ya, ns); c_ = np.bincount(s, (1 - ya) * yb, ns); n0 = np.bincount(s, 1 - ya, ns); nn = n1 + n0
    ok = nn > 0; num = np.sum(a_[ok] * n0[ok] / nn[ok]); den = np.sum(c_[ok] * n1[ok] / nn[ok])
    return float(num / den) if den > 0 else float("nan")
def stats(y, g):
    """placebo statistics of flags y against gold g (gold negatives: g == 0)."""
    ng = g == 0; nb = ng[pa] & ng[ch]; nb_noid = nb & ~ident[pa] & ~ident[ch]; sib = ng[sa] & ng[sb]
    fpr_id = float(y[ng & ident].mean()); fpr_no = float(y[ng & ~ident].mean())
    return dict(fpr_identity=fpr_id, fpr_no_identity=fpr_no, fpr_ratio=fpr_id / fpr_no if fpr_no > 0 else float("nan"),
                placebo_rr=rr_pairs(y, pa, ch, nb), placebo_rr_no_identity=rr_pairs(y, pa, ch, nb_noid),
                placebo_rr_siblings=rr_pairs(y, sa, sb, sib), placebo_rr_mh_article=mh_rr(y, pa, ch, nb, art),
                share_edges_with_identity=float(1 - nb_noid.sum() / nb.sum()))
ins = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    ins[f"{m}@0.5"] = (P[f"s:{m}@m"].values >= 0.5).astype(float)
res = {"identity_share": float(ident.mean()), "identity_share_gold_neg": float(ident[neg].mean()), "gold_prev_identity": float(z[ident].mean()), "gold_prev_no_identity": float(z[~ident].mean()),
       "n_sibling_pairs": int(len(s1)), "instruments": {}}
for k, y in ins.items():
    res["instruments"][k] = stats(y, z)
    r = res["instruments"][k]; print(f"{k:24s} FPR id/no {r['fpr_identity']:.3f}/{r['fpr_no_identity']:.3f} (x{r['fpr_ratio']:.2f}) | placebo RR {r['placebo_rr']:.2f} no-id {r['placebo_rr_no_identity']:.2f} siblings {r['placebo_rr_siblings']:.2f} MH-article {r['placebo_rr_mh_article']:.2f}", flush=True)
# the single rater against the others' majority
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
rng = np.random.default_rng(8050); draws = []
for _ in range(20):
    y1 = (rng.random(N) < kv / n).astype(float); gB = ((kv - y1) / (n - 1) >= 0.5 - 1e-12).astype(float); draws.append(stats(y1, gB))
res["single_rater"] = {k: float(np.median([d[k] for d in draws])) for k in draws[0]}
r = res["single_rater"]; print(f"{'single rater':24s} FPR id/no {r['fpr_identity']:.3f}/{r['fpr_no_identity']:.3f} (x{r['fpr_ratio']:.2f}) | placebo RR {r['placebo_rr']:.2f} no-id {r['placebo_rr_no_identity']:.2f} siblings {r['placebo_rr_siblings']:.2f} MH-article {r['placebo_rr_mh_article']:.2f}", flush=True)
# 200 resamples for the no-identity placebo RR minus the overall one, for the twelve classifier settings
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
Wb = np.random.default_rng(8051).poisson(1.0, size=(200, nC)); ecl = cl[ch]; nb = neg[pa] & neg[ch]; nb_noid = nb & ~ident[pa] & ~ident[ch]
def lrr_b(y, m):
    yp, yc = y[pa], y[ch]
    p1, p1c1, p0, p0c1 = (Wb @ np.bincount(ecl[m], w[m], nC) for w in (yp, yp * yc, 1 - yp, (1 - yp) * yc))
    return np.log((p1c1 / p1) / (p0c1 / p0))
for k in CLS12:
    d = lrr_b(ins[k], nb_noid) - lrr_b(ins[k], nb); res["instruments"][k]["no_identity_minus_all_log"] = [float(np.log(res["instruments"][k]["placebo_rr_no_identity"] / res["instruments"][k]["placebo_rr"])),
                                                                                                      float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
