"""Addendum 82, B82a: noise-free contagion from a latent propensity model (Civil Comments reply threads, all reply edges).
Each comment has a latent propensity pi (the share of the rater pool that would call it toxic); its toxic votes are Binomial(n, pi)
with n = toxicity_annotator_count and K = round(vote fraction x n). Over reply edges the parent's and child's propensities follow an
unknown joint distribution G on an m x m grid of cell midpoints, estimated by nonparametric maximum likelihood (EM over the distinct
(n_p, K_p, n_c, K_c) patterns; each edge counts once). Noise-free gold is 1{pi >= 0.5}; RR_inf follows from G.
H82b (adequacy): at each edge's observed rater counts the fit reproduces the observed log RR of the 1-rater majority, the 3-rater
majority (results/x80/x80_noise.json, full population) and gold within 0.05.
H82a (if H82b passes): >= 9 of the 12 classifier settings have log RR above log RR_inf (paired 95% CI of the difference above 0),
over 100 article-cluster Poisson resamples (seed 8201), each EM warm-started from the full-data fit.
Descriptive: m = 10 and 40; RR_inf against the LLMs and the single rater; classifiers' placebo RR on latent negatives (edges weighted
by the posterior probability that both ends have pi < 0.5); the 2-rater curve.
usage: python3 x82_latent.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, time, numpy as np, pandas as pd
from scipy.special import gammaln
W = _ROOT; OUT = sys.argv[1]; NB = 100; TOL = 1e-10; MAXIT = 20000
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); K = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
cl = pd.factorize(P.cluster.values)[0]; ecl = cl[ch]; nC = int(cl.max() + 1)
z = P.z.values.astype(float)
print("gold matches K >= n/2 on", float(np.mean(z == (K >= n / 2))), flush=True); assert np.mean(z == (K >= n / 2)) > 0.999
pat = pd.DataFrame({"np": n[pa], "kp": K[pa], "nc": n[ch], "kc": K[ch]}); codes, uniq = pd.factorize(pd.MultiIndex.from_frame(pat))
U = np.array(uniq.tolist()); S = len(U); print("edges", len(ch), "patterns", S, flush=True)
def logbinom(k, nn, g):
    return gammaln(nn + 1) - gammaln(k + 1) - gammaln(nn - k + 1) + k * np.log(g) + (nn - k) * np.log1p(-g)
def emissions(m):
    g = (np.arange(m) + 0.5) / m
    Lp = logbinom(U[:, 1][:, None], U[:, 0][:, None], g[None, :]); Lc = logbinom(U[:, 3][:, None], U[:, 2][:, None], g[None, :])
    Ep = np.exp(Lp - Lp.max(1, keepdims=True)); Ec = np.exp(Lc - Lc.max(1, keepdims=True)); off = Lp.max(1) + Lc.max(1)
    return g, Ep, Ec, off
def em(w, Ep, Ec, G=None, maxit=MAXIT, tol=TOL):
    m = Ep.shape[1]; G = np.full((m, m), 1.0 / m ** 2) if G is None else G.copy(); prev = -np.inf; wt = w / w.sum()
    for it in range(maxit):
        Zs = np.sum((Ep @ G) * Ec, axis=1)
        ll = float(np.sum(wt * np.log(Zs)))
        if np.isfinite(prev) and abs(ll - prev) < tol * abs(prev):
            break
        prev = ll; G = G * (Ep.T @ (Ec * (wt / Zs)[:, None])); G /= G.sum()
    return G, it + 1, ll
def rr_from_G(G, g):
    hi = g >= 0.5
    p11 = G[np.ix_(hi, hi)].sum(); p1 = G[hi, :].sum(); p01 = G[np.ix_(~hi, hi)].sum(); p0 = G[~hi, :].sum()
    return float(np.log((p11 / p1) / (p01 / p0)))
from scipy.stats import binom
def pmaj(nvals, g, rule):
    out = np.zeros((len(nvals), len(g)))
    for i, nn in enumerate(nvals):
        if rule == "ge_half":
            out[i] = binom.sf(np.ceil(nn / 2) - 1, nn, g)
        else:                                                            # 'tie_half': strict majority plus half the ties
            out[i] = binom.sf(np.floor(nn / 2), nn, g) + (0.5 * binom.pmf(nn // 2, nn, g) if nn % 2 == 0 else 0.0)
    return out
def predicted_lrr(G, g, n_p, n_c, rule):
    """Expected majority-gold log RR at the given rater counts (per edge), from the fitted G."""
    vals = np.unique(np.r_[n_p, n_c]); A = pmaj(vals, g, rule); idx = {v: i for i, v in enumerate(vals)}
    ip = np.array([idx[v] for v in n_p]); ic = np.array([idx[v] for v in n_c])
    # group edges by (n_p, n_c)
    key = pd.factorize(pd.MultiIndex.from_arrays([ip, ic]))[0]; cnt = np.bincount(key); first = np.unique(key, return_index=True)[1]
    s11 = s1 = s01 = s0 = 0.0
    for kk_, f_ in zip(range(len(cnt)), first):
        a, c = A[ip[f_]], A[ic[f_]]; w_ = cnt[kk_]
        s11 += w_ * a @ G @ c; s1 += w_ * a @ G.sum(1); s01 += w_ * (1 - a) @ G @ c; s0 += w_ * (1 - a) @ G.sum(1)
    return float(np.log((s11 / s1) / (s01 / s0)))
res = {"n_edges": int(len(ch)), "n_patterns": int(S), "fits": {}}
w0 = np.bincount(codes, minlength=S).astype(float)
NZ = json.load(open(f"{W}/results/x80/x80_noise.json"))["full_population"]
obs = {"k1": float(np.log(NZ["k"]["1"])), "k3": float(np.log(NZ["k"]["3"])), "gold": float(np.log(NZ["gold_rr"]))}
for m in (10, 20, 40):
    t0 = time.time(); g, Ep, Ec, off = emissions(m); G, it, ll = em(w0, Ep, Ec)
    one = np.ones(len(ch), int)
    pred = {"k1": predicted_lrr(G, g, one, one, "ge_half"), "k3": predicted_lrr(G, g, 3 * one, 3 * one, "ge_half"),
            "k2_tie_half": predicted_lrr(G, g, 2 * one, 2 * one, "tie_half"), "gold": predicted_lrr(G, g, n[pa], n[ch], "ge_half")}
    r = dict(m=m, iterations=it, loglik=ll, rr_inf=float(np.exp(rr_from_G(G, g))), log_rr_inf=rr_from_G(G, g), predicted=pred, observed=obs,
             adequacy_gaps={k: pred[k] - obs[k] for k in ("k1", "k3", "gold")}, marginal_parent=G.sum(1).tolist(), secs=time.time() - t0)
    res["fits"][str(m)] = r
    print(f"m={m} it={it} RR_inf={r['rr_inf']:.3f} | predicted k1 {np.exp(pred['k1']):.3f} (obs {np.exp(obs['k1']):.3f}) k3 {np.exp(pred['k3']):.3f} (obs {np.exp(obs['k3']):.3f}) gold {np.exp(pred['gold']):.3f} (obs {np.exp(obs['gold']):.3f}) k2 {np.exp(pred['k2_tie_half']):.3f} | {r['secs']:.0f}s", flush=True)
    if m == 20:
        G20, g20, Ep20, Ec20 = G, g, Ep, Ec
H82b = all(abs(v) <= 0.05 for v in res["fits"]["20"]["adequacy_gaps"].values()); res["H82b"] = dict(gaps=res["fits"]["20"]["adequacy_gaps"], pass_=bool(H82b))
print("H82b", H82b, res["fits"]["20"]["adequacy_gaps"], flush=True)
# instruments: flags and log RR on the same resamples
ins = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
    ins[f"{mm}@0.5"] = (P[f"s:{mm}@m"].values >= 0.5).astype(float)
Wb = np.vstack([np.ones(nC), np.random.default_rng(8201).poisson(1.0, size=(NB, nC))]).astype(float)
def lrr_w(v, we):
    vp, vc = v[pa], v[ch]
    a = np.bincount(ecl, vp * vc, nC); b = np.bincount(ecl, vp, nC); c = np.bincount(ecl, (1 - vp) * vc, nC); d = np.bincount(ecl, 1 - vp, nC)
    A, B, C, D = we @ a, we @ b, we @ c, we @ d
    return np.log((A / B) / (C / D))
L = {k: lrr_w(v, Wb) for k, v in ins.items()}; Lg = lrr_w(z, Wb)
# RR_inf on the resamples (m = 20), warm start
rinf = [rr_from_G(G20, g20)]; t0 = time.time()
for b in range(1, NB + 1):
    wb = np.bincount(codes, weights=Wb[b][ecl], minlength=S); Gb, _, _ = em(wb, Ep20, Ec20, G=G20, maxit=3000, tol=1e-9); rinf.append(rr_from_G(Gb, g20))
    if b % 20 == 0:
        print("resample", b, f"{time.time() - t0:.0f}s", flush=True)
rinf = np.array(rinf)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
res["rr_inf"] = dict(log=ci(rinf), rr=[float(np.exp(x)) for x in ci(rinf)]); res["gold"] = dict(log=ci(Lg), rr=[float(np.exp(x)) for x in ci(Lg)])
res["instruments"] = {k: dict(rr=float(np.exp(v[0])), minus_inf=ci(v - rinf)) for k, v in L.items()}
up = [k for k in CLS12 if res["instruments"][k]["minus_inf"][1] > 0]; dn = [k for k in CLS12 if res["instruments"][k]["minus_inf"][2] < 0]
res["H82a"] = dict(evaluated=bool(H82b), n_above=len(up), n_below=len(dn), above=up, below=dn, pass_=bool(H82b and len(up) >= 9))
# classifiers' placebo RR on latent negatives (posterior weights, m = 20)
post_neg = np.zeros(S)
lo = g20 < 0.5
Zs = np.sum((Ep20 @ G20) * Ec20, axis=1); post_neg = np.sum((Ep20[:, lo] @ G20[np.ix_(lo, lo)]) * Ec20[:, lo], axis=1) / Zs
wneg = post_neg[codes]
def plrr(v, w_):
    vp, vc = v[pa], v[ch]
    return float(np.log((np.sum(w_ * vp * vc) / np.sum(w_ * vp)) / (np.sum(w_ * (1 - vp) * vc) / np.sum(w_ * (1 - vp)))))
res["placebo_latent"] = {k: float(np.exp(plrr(v, wneg))) for k, v in ins.items()}
res["placebo_gold_neg"] = {k: float(np.exp(plrr(v, ((z[pa] == 0) & (z[ch] == 0)).astype(float)))) for k, v in ins.items()}
print("RR_inf", res["rr_inf"]["rr"], "gold", res["gold"]["rr"], "| H82a", res["H82a"], flush=True)
for k in sorted(res["instruments"]):
    r_ = res["instruments"][k]; print(f"  {k:24s} RR {r_['rr']:.2f}  log(RR/RR_inf) {r_['minus_inf'][0]:+.3f} [{r_['minus_inf'][1]:+.3f}, {r_['minus_inf'][2]:+.3f}]  placebo latent {res['placebo_latent'][k]:.2f} (gold-neg {res['placebo_gold_neg'][k]:.2f})", flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
