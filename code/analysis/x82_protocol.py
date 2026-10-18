"""Addendum 82, B82e: adequacy of the latent propensity model (B82a, m = 20) under Civil Comments' adaptive annotation protocol.
Protocol (inferred from the (n, K) support): an initial batch of n0 in {4, 5, 6} raters with probabilities q4, q5, q6 independent of
pi; stop at n0 if no vote dissents from the batch majority (n0 = 4) or at most one does (n0 = 5, 6); otherwise continue to 10 raters.
Comments with any other final count (2.3%) keep their observed count and are drawn as Binomial(n, pi). q4..q6 are fitted by least
squares to the observed shares of final n = 4, 5, 6, 10 among edge children, under the child marginal of the fitted G.
4,000,000 edges are drawn from G (cell midpoints) and both ends are annotated by the protocol; gold is a final vote fraction >= 0.5.
H82b': the simulated gold log RR is within 0.05 of the observed log RR (2.389). If it passes, H82a is evaluated with B82a's RR_inf
(m = 20, results/x82/x82_latent.json, 100 resamples).
usage: python3 x82_protocol.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
from scipy.special import gammaln
from scipy.stats import binom
from scipy.optimize import minimize
W = _ROOT; OUT = sys.argv[1]; NSIM = 4_000_000; TOL = 1e-10; MAXIT = 20000
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "tox", "parent_row", "z"])
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); K = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
z = P.z.values.astype(float)
pat = pd.DataFrame({"np": n[pa], "kp": K[pa], "nc": n[ch], "kc": K[ch]}); codes, uniq = pd.factorize(pd.MultiIndex.from_frame(pat)); U = np.array(uniq.tolist()); S = len(U)
def logbinom(k, nn, g):
    return gammaln(nn + 1) - gammaln(k + 1) - gammaln(nn - k + 1) + k * np.log(g) + (nn - k) * np.log1p(-g)
m = 20; g = (np.arange(m) + 0.5) / m
Lp = logbinom(U[:, 1][:, None], U[:, 0][:, None], g[None, :]); Lc = logbinom(U[:, 3][:, None], U[:, 2][:, None], g[None, :])
Ep = np.exp(Lp - Lp.max(1, keepdims=True)); Ec = np.exp(Lc - Lc.max(1, keepdims=True)); w = np.bincount(codes, minlength=S).astype(float); wt = w / w.sum()
G = np.full((m, m), 1.0 / m ** 2); prev = -np.inf
for it in range(MAXIT):                                   # the same EM as x82_latent.py (m = 20)
    Zs = np.sum((Ep @ G) * Ec, axis=1); ll = float(np.sum(wt * np.log(Zs)))
    if np.isfinite(prev) and abs(ll - prev) < TOL * abs(prev):
        break
    prev = ll; G = G * (Ep.T @ (Ec * (wt / Zs)[:, None])); G /= G.sum()
hi = g >= 0.5; rr_inf = (G[np.ix_(hi, hi)].sum() / G[hi].sum()) / (G[np.ix_(~hi, hi)].sum() / G[~hi].sum())
print(f"EM iterations {it + 1}, RR_inf {rr_inf:.4f}", flush=True)
# protocol fit on edge children
nc_obs = n[ch]; reg = np.isin(nc_obs, (4, 5, 6, 10)); s_obs = {j: float(np.mean(nc_obs[reg] == j)) for j in (4, 5, 6, 10)}; irr_share = float(1 - reg.mean())
gc = G.sum(0)                                              # child marginal
def p_stop(n0):
    k = np.arange(n0 + 1)[:, None]; pm = binom.pmf(k, n0, g[None, :]); dis = np.minimum(np.arange(n0 + 1), n0 - np.arange(n0 + 1))
    ok = dis == 0 if n0 == 4 else dis <= 1
    return (pm * ok[:, None]).sum(0)                       # per grid point
ps = {j: float(p_stop(j) @ gc) for j in (4, 5, 6)}
def shares(q):
    q4, q5 = q; q6 = 1 - q4 - q5
    return np.array([q4 * ps[4], q5 * ps[5], q6 * ps[6], q4 * (1 - ps[4]) + q5 * (1 - ps[5]) + q6 * (1 - ps[6])])
obs_v = np.array([s_obs[j] for j in (4, 5, 6, 10)])
fit = minimize(lambda q: np.sum((shares(q) - obs_v) ** 2) + (1e3 * max(0, q[0] + q[1] - 1)) ** 2 + (1e3 * max(0, -q[0])) ** 2 + (1e3 * max(0, -q[1])) ** 2, x0=[0.7, 0.1], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000))
q4, q5 = fit.x; q6 = 1 - q4 - q5; print("q", round(q4, 4), round(q5, 4), round(q6, 4), "fitted shares", shares(fit.x).round(4), "observed", obs_v.round(4), "stop probs", {k: round(v, 4) for k, v in ps.items()}, flush=True)
irr_n = nc_obs[~reg]
rng = np.random.default_rng(8202)
def annotate(pi):
    N = len(pi); out_n = np.zeros(N, np.int64); out_k = np.zeros(N, np.int64)
    irr = rng.random(N) < irr_share; ni = rng.choice(irr_n, size=irr.sum()); out_n[irr] = ni; out_k[irr] = rng.binomial(ni, pi[irr])
    r = ~irr; u = rng.random(r.sum()); n0 = np.where(u < q4, 4, np.where(u < q4 + q5, 5, 6)); k0 = rng.binomial(n0, pi[r]); dis = np.minimum(k0, n0 - k0)
    stop = np.where(n0 == 4, dis == 0, dis <= 1); kk = np.where(stop, k0, k0 + rng.binomial(10 - n0, pi[r])); nn = np.where(stop, n0, 10)
    out_n[r] = nn; out_k[r] = kk
    return out_n, out_k
cells = rng.choice(m * m, size=NSIM, p=G.ravel() / G.sum()); jp, jc = np.divmod(cells, m)
n_p, k_p = annotate(g[jp]); n_c, k_c = annotate(g[jc])
gp = (k_p >= n_p / 2).astype(float); gcv = (k_c >= n_c / 2).astype(float)
def lrr(a, b):
    return float(np.log((np.sum(a * b) / np.sum(a)) / (np.sum((1 - a) * b) / np.sum(1 - a))))
sim_gold = lrr(gp, gcv); one_p = (rng.random(NSIM) < k_p / n_p).astype(float); one_c = (rng.random(NSIM) < k_c / n_c).astype(float); sim_k1 = lrr(one_p, one_c)
obs_gold = lrr(z[pa], z[ch]); NZ = json.load(open(f"{W}/results/x80/x80_noise.json"))["full_population"]; obs_k1 = float(np.log(NZ["k"]["1"]))
true_p = (g[jp] >= 0.5).astype(float); true_c = (g[jc] >= 0.5).astype(float); sim_inf = lrr(true_p, true_c)
# realism: shares of final n and K | n among simulated children vs observed
real = {}
for nn in (4, 5, 6, 10):
    so = K[ch][nc_obs == nn]; ss = k_c[n_c == nn]
    real[str(nn)] = dict(share_obs=float(np.mean(nc_obs == nn)), share_sim=float(np.mean(n_c == nn)),
                         K_obs={str(k): float(np.mean(so == k)) for k in range(nn + 1) if np.mean(so == k) > 0.001}, K_sim={str(k): float(np.mean(ss == k)) for k in range(nn + 1) if np.mean(ss == k) > 0.001})
res = dict(em_iterations=it + 1, rr_inf=float(rr_inf), q=dict(q4=float(q4), q5=float(q5), q6=float(q6)), stop_probs=ps, irregular_share=irr_share,
           sim=dict(gold_rr=float(np.exp(sim_gold)), k1_rr=float(np.exp(sim_k1)), inf_rr=float(np.exp(sim_inf)), gold_prev=float(gcv.mean())),
           obs=dict(gold_rr=float(np.exp(obs_gold)), k1_rr=float(np.exp(obs_k1)), gold_prev=float(z[ch].mean())),
           gaps=dict(gold=sim_gold - obs_gold, k1=sim_k1 - obs_k1), realism=real)
res["H82b_prime"] = dict(gap=sim_gold - obs_gold, pass_=bool(abs(sim_gold - obs_gold) <= 0.05))
L = json.load(open(f"{W}/results/x82/x82_latent.json")) if __import__("os").path.exists(f"{W}/results/x82/x82_latent.json") else None
if L and res["H82b_prime"]["pass_"]:
    CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{mm}@{t}" for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
    up = [k for k in CLS12 if L["instruments"][k]["minus_inf"][1] > 0]; dn = [k for k in CLS12 if L["instruments"][k]["minus_inf"][2] < 0]
    res["H82a"] = dict(evaluated=True, n_above=len(up), n_below=len(dn), above=up, below=dn, rr_inf=L["rr_inf"]["rr"], pass_=bool(len(up) >= 9))
else:
    res["H82a"] = dict(evaluated=False, reason="H82b' failed" if not res["H82b_prime"]["pass_"] else "x82_latent.json not found yet")
print(json.dumps({k: v for k, v in res.items() if k != "realism"}, indent=1)); print(json.dumps(real)[:1500])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
