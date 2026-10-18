"""Addendum 85, B85a: the NPMLE of B82a and the monotone f-MLE of B83d solved to convergence.
Solvers (deviation from the registered SQUAREM, which stalled at the boundary of the simplex): G (edges) and g_x (depth levels) by
sequential quadratic programming (x85_sqp.py, mixsqp-type) to the KKT condition of the NPMLE, max_k d_k - 1 <= 1e-8; f (monotone,
in [1e-7, 1 - 1e-7]) by SLSQP on its increments (ftol 1e-15), started from 200 EM steps.
Part 1 (--part rrinf): RR_inf at m = 10, 20, 40; at m = 20 the 100 resamples of B82a (seed 8201, the same weights), the classifiers'
log RR minus log RR_inf (paired) and H82a re-evaluated.
Part 2 (--part construct --grid m --nb NB): B83d's monotone construct-only test at grid m with converged G, g_x and f (resample
weights default_rng(8303 + m), as B83d), H83d / H83e at m = 20 re-evaluated.
usage: python3 x85_converged.py OUT_JSON --part rrinf | --part construct --grid 20 --nb 100"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code/analysis"))
OUT = sys.argv[1]; PART = sys.argv[sys.argv.index("--part") + 1]
GRID = int(sys.argv[sys.argv.index("--grid") + 1]) if "--grid" in sys.argv else 20
NBA = int(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else 100
_argv = sys.argv; sys.argv = sys.argv[:2]
import x83_construct as X
import x83d_mono as MO
sys.argv = _argv
KKT = 1e-8; FTOL = 1e-13; MAXF = 50000; MAXG = 200000


def squarem(F, ll, proj, x0, conv, maxit):
    """Generic SQUAREM: F the EM map, ll the log-likelihood, proj the feasibility projection, conv(x) -> True when converged."""
    x = x0.copy(); lx = ll(x); it = 0
    while it < maxit:
        if conv(x):
            return x, it, True
        x1 = F(x); x2 = F(x1); it += 2
        r = x1 - x; v = x2 - 2 * x1 + x; nr = np.sqrt(np.sum(r * r)); nv = np.sqrt(np.sum(v * v))
        if nv == 0 or nr == 0:
            x = x2; lx = ll(x); continue
        a = min(-nr / nv, -1.0)
        while True:                                                   # step halving toward -1 keeps the extrapolation feasible
            xp = x - 2 * a * r + a * a * v
            if a >= -1.0 or proj(xp) is not None:
                break
            a = (a - 1.0) / 2.0
        xp = x2 if a >= -1.0 else proj(xp)
        xn = F(xp); it += 1; ln = ll(xn)
        l2 = ll(x2)
        if not np.isfinite(ln) or ln < l2:
            xn, ln = x2, l2
        x, lx = xn, ln
    return x, it, conv(x)


def simplex_proj(x):
    """Feasibility check and normalization for a mixing distribution: None if any mass is not positive."""
    if np.min(x) <= 0:
        return None
    return x / x.sum()


from x85_sqp import npmle_sqp
from scipy.optimize import minimize


def em_G_sq(w, Ep, Ec, G=None, maxit=None, tol=None):
    """Edge NPMLE solved to the KKT tolerance by SQP (x85_sqp.py), warm-started from G."""
    S, m = Ep.shape; L = (Ep[:, :, None] * Ec[:, None, :]).reshape(S, m * m)
    x, info = npmle_sqp(L, w, x0=None if G is None else G.ravel(), tol=KKT)
    em_G_sq.last = info
    return x.reshape(m, m)


def em_gx_sq(wc, pop, Lc, gx=None, maxit=None, tol=None):
    m = Lc.shape[1]; out = np.full((X.XMAX + 1, m), 1.0 / m) if gx is None else gx.copy(); info = []
    for d in range(X.XMAX + 1):
        s = pop.cU[:, 2] == d
        if not s.any() or wc[s].sum() <= 0:
            continue
        x, inf_ = npmle_sqp(Lc[s], wc[s], x0=out[d], tol=KKT); out[d] = x; info.append(inf_)
    em_gx_sq.last = info
    return out


def em_f_sq(pop, Lc, gx, y, wcl, f=None, maxit=None, tol=None):
    """Monotone f-MLE in [1e-7, 1 - 1e-7] by SLSQP on the increments: f = 1e-7 + cumsum(delta), delta >= 0, sum(delta) <= 1 - 2e-7."""
    code = pop.ccode * 2 + y.astype(np.int64); w = np.bincount(code, weights=wcl, minlength=2 * pop.cS)
    j = np.flatnonzero(w > 0); w = w[j]; cp = j // 2; yy = (j % 2).astype(float)
    base = gx[pop.cU[cp, 2]] * Lc[cp]; m = Lc.shape[1]; wt = w / w.sum()
    al = np.where(yy == 1, 0.0, base.sum(1)); be = np.where((yy == 1)[:, None], base, -base)       # term = al + be . f
    lo, hi = 1e-7, 1 - 1e-7; T = np.tril(np.ones((m, m)))                                       # f = lo + T delta
    def nll(dl):
        f_ = lo + T @ dl; v = al + be @ f_
        if np.any(v <= 0):
            return 1e30, np.zeros(m)
        return float(-np.sum(wt * np.log(v))), -(T.T @ (be.T @ (wt / v)))
    f0 = MO.em_f_mono(pop, Lc, gx, y, wcl, f=f, maxit=200) if f is None else np.clip(np.maximum.accumulate(f), lo, hi)
    d0 = np.diff(np.r_[lo, f0]); d0 = np.maximum(d0, 0)
    r = minimize(nll, d0, jac=True, method="SLSQP", bounds=[(0, None)] * m,
                 constraints=[{"type": "ineq", "fun": lambda dl: (hi - lo) - dl.sum(), "jac": lambda dl: -np.ones(m)}],
                 options=dict(ftol=1e-15, maxiter=5000))
    f1 = lo + T @ np.maximum(r.x, 0)
    em_f_sq.last = dict(success=bool(r.success), nit=int(r.nit), message=str(r.message)[:60])
    return np.clip(f1, lo, hi)


if __name__ == "__main__" and PART == "rrinf":
    P = pd.read_parquet(f"{X.W}/results/x14_pops/cc_threads.parquet")
    meta = pd.read_parquet(f"{X.W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
    n = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); K = np.clip(np.rint(P.tox.values * n).astype(np.int64), 0, n)
    ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
    cl = pd.factorize(P.cluster.values)[0]; ecl = cl[ch]; nC = int(cl.max() + 1); z = P.z.values.astype(float)
    pat = pd.DataFrame({"np": n[pa], "kp": K[pa], "nc": n[ch], "kc": K[ch]}); codes, uniq = pd.factorize(pd.MultiIndex.from_frame(pat)); U = np.array(uniq.tolist()); S = len(U)
    w0 = np.bincount(codes, minlength=S).astype(float); res = {"fits": {}}
    def emis(m):
        g = (np.arange(m) + 0.5) / m
        Ep = X.norm_rows(X.logbinom(U[:, 1][:, None], U[:, 0][:, None], g[None, :])); Ec = X.norm_rows(X.logbinom(U[:, 3][:, None], U[:, 2][:, None], g[None, :]))
        return g, Ep, Ec
    def rr(G, g):
        hi = g >= 0.5; return float(np.log((G[np.ix_(hi, hi)].sum() / G[hi].sum()) / (G[np.ix_(~hi, hi)].sum() / G[~hi].sum())))
    for m in (10, 20, 40):
        t0 = time.time(); g, Ep, Ec = emis(m); G = em_G_sq(w0, Ep, Ec)
        res["fits"][str(m)] = dict(rr_inf=float(np.exp(rr(G, g))), **em_G_sq.last, secs=time.time() - t0)
        print(f"m={m} RR_inf {np.exp(rr(G, g)):.4f} {em_G_sq.last} {time.time() - t0:.0f}s", flush=True)
        if m == 20:
            G20, g20, Ep20, Ec20 = G, g, Ep, Ec
    NB = 100; Wb = np.vstack([np.ones(nC), np.random.default_rng(8201).poisson(1.0, size=(NB, nC))]).astype(float)
    ins = {c[2:]: np.nan_to_num(P[c].values.astype(float)) for c in P.columns if c.startswith("y:")}
    for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
        ins[f"{mm}@0.5"] = (P[f"s:{mm}@m"].values >= 0.5).astype(float)
    def lrr_w(v, we):
        vp, vc = v[pa], v[ch]
        a = np.bincount(ecl, vp * vc, nC); b = np.bincount(ecl, vp, nC); c = np.bincount(ecl, (1 - vp) * vc, nC); d = np.bincount(ecl, 1 - vp, nC)
        A, B, C, D = we @ a, we @ b, we @ c, we @ d
        return np.log((A / B) / (C / D))
    L = {k: lrr_w(v, Wb) for k, v in ins.items()}
    rinf = [rr(G20, g20)]; t0 = time.time(); kk = []
    for b in range(1, NB + 1):
        wb = np.bincount(codes, weights=Wb[b][ecl], minlength=S); Gb = em_G_sq(wb, Ep20, Ec20, G=G20); rinf.append(rr(Gb, g20)); kk.append(em_G_sq.last["kkt"])
        if b % 10 == 0:
            print("resample", b, f"{time.time() - t0:.0f}s", "max kkt", max(kk), flush=True)
    rinf = np.array(rinf); ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    res["rr_inf"] = dict(log=ci(rinf), rr=[float(np.exp(x)) for x in ci(rinf)], max_resample_kkt=float(max(kk)))
    res["instruments"] = {k: dict(rr=float(np.exp(v[0])), minus_inf=ci(v - rinf)) for k, v in L.items()}
    up = [k for k in X.CLS12 if res["instruments"][k]["minus_inf"][1] > 0]; dn = [k for k in X.CLS12 if res["instruments"][k]["minus_inf"][2] < 0]
    res["H82a_converged"] = dict(n_above=len(up), n_below=len(dn), above=up, below=dn, pass_=bool(len(up) >= 9))
    print("RR_inf", res["rr_inf"], "H82a", res["H82a_converged"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)

if __name__ == "__main__" and PART == "construct":
    X.em_G = em_G_sq; X.em_gx = em_gx_sq                                           # MO.run calls X.em_G / X.em_gx
    full, s5 = X.load(); print("loaded", full.N, len(full.ch), "grid", GRID, "nb", NBA, flush=True)
    res = MO.run(full, s5, [GRID], [NBA], log=lambda s: print(s, flush=True), em_f=em_f_sq)
    res["solver"] = "SQP (x85_sqp.py) to KKT 1e-8 for G and g_x; SLSQP on the increments of the monotone f (ftol 1e-15)"
    R = res["grids"][str(GRID)]
    print("contagion", R["contagion"], "escalation", R["escalation"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
