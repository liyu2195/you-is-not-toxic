"""Addendum 83, B83a-b: are instruments' errors explained by the raters' own uncertainty? A construct-only test under the latent
propensity model of Addendum 82 (Civil Comments reply threads).
Model: each comment has a latent propensity pi on the m = 20 grid of cell midpoints; its toxic votes are Binomial(n, pi) up to the
stopping-rule factor (the likelihood principle under vote-dependent stopping); over reply edges (pi_p, pi_c) ~ G (the edge NPMLE of
x82_latent.py); at depth level x = min(depth, 10) a comment's pi ~ g_x (one NPMLE per level, all comments of cc_threads).
H0 (construct only): an instrument's flag depends on the comment only through pi, P(Y = 1 | pi) = f(pi), independently across
comments given their propensities. f is estimated on the grid by EM over (n, K, x, Y) patterns with g_x fixed (two-step).
Under H0 the expected flags given the votes are E[Y | votes, x] = sum_a P(pi = a | votes, x) f(a) for a comment and
E[Y_p Y_c | votes] = sum_ab Q(a, b) f(a) f(b) for an edge, Q the posterior under G; the H0-predicted statistic is the statistic of
these expectations. Excess = observed minus H0-predicted.
  B83a contagion: placebo log RR (edges with both ends gold-negative, as x74_placebo.py); descriptive: all edges (law), edges whose
       two comments both had 4 of 4 raters say civil, grids m = 10 and 40.
  B83b escalation: placebo depth slope (OLS of Y on x among gold negatives, as x74_decomp.py); descriptive: the depth law (all
       comments), grids m = 10 and 40.
Reference instruments without data: the noise-free thresholds f = 1{pi >= t}, t = 0.3, 0.4, 0.5 (predicted statistics only).
Uncertainty: 100 article-cluster Poisson resamples (seed 8301); each re-estimates G (warm start, as x82_latent.py), g_x and every f,
and recomputes observed and predicted statistics. Instruments labeled only on the 5% sample (cc_s5: Sonnet, Gemma) use the same
article weights, the resample's G and g_x, and f re-estimated on cc_s5.
H83a: >= 9 of the 12 classifier settings (x82's list) have a placebo log RR excess with a 95% CI above 0.
H83b: Haiku and Sonnet without and with the parent (4 settings) all have a placebo depth-slope excess with a 95% CI above 0.
usage: python3 x83_construct.py OUT_JSON [--nb 100]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, time, numpy as np, pandas as pd
from scipy.special import gammaln
W = _ROOT; OUT = sys.argv[1] if len(sys.argv) > 1 else None; NB = int(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else 100
TOL = 1e-10; MAXIT = 20000; XMAX = 10
CLS12 = ["detoxify_original@0.6", "detoxify_original@m", "imsypp@argmax", "imsypp@m"] + [f"{m}@{t}" for m in ("sn_rob", "mh_dist", "fb_hate", "cn_off") for t in ("0.5", "m")]
H83B = ["haiku", "haiku_ctx", "sonnet", "sonnet_ctx"]
THR = (0.3, 0.4, 0.5)


def logbinom(k, nn, g):
    return gammaln(nn + 1) - gammaln(k + 1) - gammaln(nn - k + 1) + k * np.log(g) + (nn - k) * np.log1p(-g)


def norm_rows(L):
    return np.exp(L - L.max(1, keepdims=True))


class Pop:
    """Comments with votes (n, K), depth level x, parent index (or -1), article cluster, and flags."""
    def __init__(self, n, K, x, parent, cl, flags):
        self.n, self.K, self.x, self.cl, self.flags = n, K, x, cl, flags
        self.N = len(n); self.z = (K >= n / 2).astype(float)
        ch = np.flatnonzero(parent >= 0); pa = parent[ch]; keep = pa != ch; self.ch, self.pa = ch[keep], pa[keep]; self.ecl = cl[self.ch]
        cp = pd.factorize(pd.MultiIndex.from_arrays([n, K, x]))
        self.ccode = cp[0]; self.cU = np.array(cp[1].tolist()); self.cS = len(self.cU)
        ep = pd.factorize(pd.MultiIndex.from_arrays([n[self.pa], K[self.pa], n[self.ch], K[self.ch]]))
        self.ecode = ep[0]; self.eU = np.array(ep[1].tolist()); self.eS = len(self.eU)
        self.eneg = (self.eU[:, 1] < self.eU[:, 0] / 2) & (self.eU[:, 3] < self.eU[:, 2] / 2)          # both gold-negative
        self.e44 = (self.eU[:, 0] == 4) & (self.eU[:, 1] == 0) & (self.eU[:, 2] == 4) & (self.eU[:, 3] == 0)
        self.cneg = self.cU[:, 1] < self.cU[:, 0] / 2


def emis(pop, g):
    Lc = norm_rows(logbinom(pop.cU[:, 1][:, None], pop.cU[:, 0][:, None], g[None, :]))
    Ep = norm_rows(logbinom(pop.eU[:, 1][:, None], pop.eU[:, 0][:, None], g[None, :]))
    Ec = norm_rows(logbinom(pop.eU[:, 3][:, None], pop.eU[:, 2][:, None], g[None, :]))
    return Lc, Ep, Ec


def em_G(w, Ep, Ec, G=None, maxit=MAXIT, tol=TOL):
    m = Ep.shape[1]; G = np.full((m, m), 1.0 / m ** 2) if G is None else G.copy(); prev = -np.inf; wt = w / w.sum()
    for it in range(maxit):
        Zs = np.sum((Ep @ G) * Ec, axis=1); ll = float(np.sum(wt * np.log(Zs)))
        if np.isfinite(prev) and abs(ll - prev) < tol * abs(prev):
            break
        prev = ll; G = G * (Ep.T @ (Ec * (wt / Zs)[:, None])); G /= G.sum()
    return G


def em_gx(wc, pop, Lc, gx=None, maxit=MAXIT, tol=TOL):
    """One NPMLE of pi per depth level; wc: weights of the comment patterns."""
    m = Lc.shape[1]; out = np.full((XMAX + 1, m), 1.0 / m) if gx is None else gx.copy()
    for d in range(XMAX + 1):
        s = pop.cU[:, 2] == d
        if not s.any() or wc[s].sum() <= 0:
            continue
        L = Lc[s]; w = wc[s] / wc[s].sum(); g = out[d].copy(); prev = -np.inf
        for it in range(maxit):
            Zs = L @ g; ll = float(np.sum(w * np.log(Zs)))
            if np.isfinite(prev) and abs(ll - prev) < tol * abs(prev):
                break
            prev = ll; g = g * (L.T @ (w / Zs)); g /= g.sum()
        out[d] = g
    return out


def em_f(pop, Lc, gx, y, wcl, f=None, maxit=5000, tol=TOL):
    """f on the grid: EM over (comment pattern, Y) with pi ~ g_x fixed. wcl: per-comment weights."""
    code = pop.ccode * 2 + y.astype(np.int64); w = np.bincount(code, weights=wcl, minlength=2 * pop.cS)
    j = np.flatnonzero(w > 0); w = w[j]; cp = j // 2; yy = (j % 2).astype(float)
    base = gx[pop.cU[cp, 2]] * Lc[cp]; m = Lc.shape[1]
    f = np.full(m, max(float(np.sum(w * yy) / w.sum()), 1e-4)) if f is None else f.copy(); prev = -np.inf; wt = w / w.sum()
    for it in range(maxit):
        lk = base * np.where(yy[:, None] == 1, f[None, :], 1 - f[None, :]); Zs = lk.sum(1); ll = float(np.sum(wt * np.log(Zs)))
        if np.isfinite(prev) and abs(ll - prev) < tol * abs(prev):
            break
        prev = ll; r = lk * (wt / Zs)[:, None]; f = np.clip((r * yy[:, None]).sum(0) / np.maximum(r.sum(0), 1e-300), 1e-7, 1 - 1e-7)
    return f


def pred_comment(pop, Lc, gx, f):
    """E[Y | votes, x] per comment pattern under H0."""
    post = gx[pop.cU[:, 2]] * Lc; return (post @ f) / post.sum(1)


def pred_edge(Ep, Ec, G, f):
    """Per edge pattern: E[Y_p], E[Y_c], E[Y_p Y_c] under H0 and the posterior under G."""
    EG = Ep @ G; Zs = np.sum(EG * Ec, 1); EfG = (Ep * f) @ G
    return np.sum(EfG * Ec, 1) / Zs, np.sum(EG * (Ec * f), 1) / Zs, np.sum(EfG * (Ec * f), 1) / Zs


def lrr_from(p1, p1c1, p0, p0c1):
    return float(np.log((p1c1 / p1) / (p0c1 / p0)))


def slope_from(n, sx, sxx, sy, sxy):
    return float((n * sxy - sx * sy) / (n * sxx - sx ** 2))


def statistics(pop, Lc, Ep, Ec, G, gx, f, we_pat, wc_pat):
    """H0-predicted statistics from pattern weights (edge patterns we_pat, comment patterns wc_pat)."""
    Epp, Ecc, Epc = pred_edge(Ep, Ec, G, f); out = {}
    for nm, msk in (("placebo_lrr", pop.eneg), ("lrr", np.ones(pop.eS, bool)), ("lrr_44", pop.e44)):
        w = we_pat * msk; out[nm] = lrr_from(np.sum(w * Epp), np.sum(w * Epc), np.sum(w * (1 - Epp)), np.sum(w * (Ecc - Epc)))
    Ec_ = pred_comment(pop, Lc, gx, f); x = pop.cU[:, 2].astype(float)
    for nm, msk in (("placebo_slope", pop.cneg), ("depth_law", np.ones(pop.cS, bool))):
        w = wc_pat * msk; out[nm] = slope_from(w.sum(), np.sum(w * x), np.sum(w * x * x), np.sum(w * Ec_), np.sum(w * x * Ec_))
    out["flag_rate"] = float(np.sum(wc_pat * Ec_) / wc_pat.sum())
    return out


def observed(pop, y, wcl, we):
    """Observed statistics with per-comment weights wcl and per-edge weights we."""
    out = {}; yp, yc = y[pop.pa], y[pop.ch]; eneg = pop.eneg[pop.ecode]; e44 = pop.e44[pop.ecode]
    for nm, msk in (("placebo_lrr", eneg), ("lrr", np.ones(len(yp), bool)), ("lrr_44", e44)):
        w = we * msk; out[nm] = lrr_from(np.sum(w * yp), np.sum(w * yp * yc), np.sum(w * (1 - yp)), np.sum(w * (1 - yp) * yc))
    x = pop.x.astype(float); cneg = pop.z == 0
    for nm, msk in (("placebo_slope", cneg), ("depth_law", np.ones(pop.N, bool))):
        w = wcl * msk; out[nm] = slope_from(w.sum(), np.sum(w * x), np.sum(w * x * x), np.sum(w * y), np.sum(w * x * y))
    out["flag_rate"] = float(np.sum(wcl * y) / wcl.sum())
    return out


def load():
    P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet")
    meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
    def build(D, clmap=None):
        n = meta.toxicity_annotator_count.reindex(D.id.values).values.astype(np.int64); K = np.clip(np.rint(D.tox.values * n).astype(np.int64), 0, n)
        x = np.minimum(D.depth.values, XMAX).astype(np.int64); cl = pd.factorize(D.cluster.values)[0] if clmap is None else clmap
        fl = {c[2:]: np.nan_to_num(D[c].values.astype(float)) for c in D.columns if c.startswith("y:")}
        for mm in ("sn_rob", "mh_dist", "fb_hate", "cn_off"):
            if f"s:{mm}@m" in D:
                fl[f"{mm}@0.5"] = (D[f"s:{mm}@m"].values >= 0.5).astype(float)
        pop = Pop(n, K, x, D.parent_row.values.astype(np.int64), cl, fl)
        assert np.mean(pop.z == D.z.values) > 0.999
        return pop
    full = build(P)
    S = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); row = pd.Series(np.arange(len(P)), index=P.id.values)
    s5 = build(S, clmap=full.cl[row.reindex(S.id.values).values])
    s5.flags = {k: v for k, v in s5.flags.items() if k in ("sonnet", "sonnet_ctx", "gemma_m", "gemma_m_ctx")}
    return full, s5


def run(full, s5, nb, seed=8301, grids=(20, 10, 40), log=print):
    res = {"n_comments": full.N, "n_edges": int(len(full.ch)), "n_s5": s5.N, "n_s5_edges": int(len(s5.ch)), "grids": {}}
    nC = int(full.cl.max() + 1)
    for m in grids:
        t0 = time.time(); g = (np.arange(m) + 0.5) / m
        Lc, Ep, Ec = emis(full, g); Lc5, Ep5, Ec5 = emis(s5, g)
        we1 = np.bincount(full.ecode, minlength=full.eS).astype(float); wc1 = np.bincount(full.ccode, minlength=full.cS).astype(float)
        G = em_G(we1, Ep, Ec); gx = em_gx(wc1, full, Lc)
        F = {k: em_f(full, Lc, gx, y, np.ones(full.N)) for k, y in full.flags.items()}
        F5 = {k: em_f(s5, Lc5, gx, y, np.ones(s5.N)) for k, y in s5.flags.items()}
        we5 = np.bincount(s5.ecode, minlength=s5.eS).astype(float); wc5 = np.bincount(s5.ccode, minlength=s5.cS).astype(float)
        R = {"instruments": {}, "thresholds": {}}
        def one(pop, Lc_, Ep_, Ec_, G_, gx_, f_, y, we_pat, wc_pat, wcl, we):
            pr = statistics(pop, Lc_, Ep_, Ec_, G_, gx_, f_, we_pat, wc_pat); ob = observed(pop, y, wcl, we) if y is not None else None
            return pr, ob
        draws = {}
        for k, y in full.flags.items():
            pr, ob = one(full, Lc, Ep, Ec, G, gx, F[k], y, we1, wc1, np.ones(full.N), np.ones(len(full.ch))); draws[k] = [(pr, ob)]
        for k, y in s5.flags.items():
            pr, ob = one(s5, Lc5, Ep5, Ec5, G, gx, F5[k], y, we5, wc5, np.ones(s5.N), np.ones(len(s5.ch))); draws[k] = [(pr, ob)]
        for t in THR:
            ft = (g >= t).astype(float); pr, _ = one(full, Lc, Ep, Ec, G, gx, ft, None, we1, wc1, None, None); draws[f"pi>={t}"] = [(pr, None)]
        log(f"m={m} full fit {time.time() - t0:.0f}s")
        if m == 20 and nb > 0:
            rng = np.random.default_rng(seed); Wb = rng.poisson(1.0, size=(nb, nC)).astype(float); t1 = time.time()
            for b in range(nb):
                wcl = Wb[b][full.cl]; we = Wb[b][full.ecl]; wcl5 = Wb[b][s5.cl]; we5b = Wb[b][s5.ecl]
                wep = np.bincount(full.ecode, weights=we, minlength=full.eS); wcp = np.bincount(full.ccode, weights=wcl, minlength=full.cS)
                wep5 = np.bincount(s5.ecode, weights=we5b, minlength=s5.eS); wcp5 = np.bincount(s5.ccode, weights=wcl5, minlength=s5.cS)
                Gb = em_G(wep, Ep, Ec, G=G, maxit=3000, tol=1e-9); gxb = em_gx(wcp, full, Lc, gx=gx, maxit=3000, tol=1e-10)
                for k, y in full.flags.items():
                    fb = em_f(full, Lc, gxb, y, wcl, f=F[k], maxit=2000); draws[k].append(one(full, Lc, Ep, Ec, Gb, gxb, fb, y, wep, wcp, wcl, we))
                for k, y in s5.flags.items():
                    fb = em_f(s5, Lc5, gxb, y, wcl5, f=F5[k], maxit=2000); draws[k].append(one(s5, Lc5, Ep5, Ec5, Gb, gxb, fb, y, wep5, wcp5, wcl5, we5b))
                for t in THR:
                    ft = (g >= t).astype(float); draws[f"pi>={t}"].append(one(full, Lc, Ep, Ec, Gb, gxb, ft, None, wep, wcp, None, None))
                if (b + 1) % 10 == 0:
                    log(f"resample {b + 1} {time.time() - t1:.0f}s")
        ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))] if len(a) > 1 else [float(a[0])]
        for k, D in draws.items():
            stats = D[0][0].keys(); r = {}
            for s in stats:
                p = np.array([d[0][s] for d in D]); r[s] = {"pred": ci(p)}
                if D[0][1] is not None:
                    o = np.array([d[1][s] for d in D]); r[s]["obs"] = ci(o); r[s]["excess"] = ci(o - p)
            if k in F:
                r["f"] = F[k].tolist()
            elif k in F5:
                r["f"] = F5[k].tolist()
            (R["thresholds"] if k.startswith("pi>=") else R["instruments"])[k] = r
        hi = g >= 0.5; R["rr_inf"] = float((G[np.ix_(hi, hi)].sum() / G[hi].sum()) / (G[np.ix_(~hi, hi)].sum() / G[~hi].sum()))
        R["gx_mean"] = (gx @ g).tolist(); R["gx_share_ge_0.25"] = gx[:, g >= 0.25].sum(1).tolist()
        res["grids"][str(m)] = R
    R = res["grids"]["20"]["instruments"]
    up = [k for k in CLS12 if k in R and len(R[k]["placebo_lrr"]["excess"]) == 3 and R[k]["placebo_lrr"]["excess"][1] > 0]
    res["H83a"] = dict(n_above=len(up), above=up, pass_=bool(len(up) >= 9))
    okb = [k for k in H83B if k in R and len(R[k]["placebo_slope"]["excess"]) == 3 and R[k]["placebo_slope"]["excess"][1] > 0]
    res["H83b"] = dict(n_above=len(okb), above=okb, pass_=bool(len(okb) == len(H83B)))
    return res


if __name__ == "__main__":
    full, s5 = load(); print("comments", full.N, "edges", len(full.ch), "comment patterns", full.cS, "edge patterns", full.eS, "| s5", s5.N, len(s5.ch), flush=True)
    res = run(full, s5, NB, log=lambda s: print(s, flush=True))
    R = res["grids"]["20"]["instruments"]; pp = 100.0
    for k in sorted(R):
        r = R[k]; e = r["placebo_lrr"].get("excess"); s = r["placebo_slope"].get("excess")
        print(f"{k:24s} placebo RR obs {np.exp(r['placebo_lrr']['obs'][0]):.2f} pred {np.exp(r['placebo_lrr']['pred'][0]):.2f} excess {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}] | "
              f"placebo slope obs {r['placebo_slope']['obs'][0]*pp:+.3f} pred {r['placebo_slope']['pred'][0]*pp:+.3f} excess [{s[1]*pp:+.3f},{s[2]*pp:+.3f}] | RR obs {np.exp(r['lrr']['obs'][0]):.2f} pred {np.exp(r['lrr']['pred'][0]):.2f}", flush=True)
    for k, r in res["grids"]["20"]["thresholds"].items():
        print(k, {s: round(v["pred"][0], 4) for s, v in r.items()}, flush=True)
    print("H83a", res["H83a"], "H83b", res["H83b"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
