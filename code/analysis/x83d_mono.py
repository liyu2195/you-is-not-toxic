"""Addendum 83, B83d: the construct-only test of B83a-b with f constrained to be non-decreasing in pi (an instrument of the raters'
construct flags a comment at least as often when more of the rater pool would call it toxic). The EM M-step for f is the weighted
isotonic regression (pool adjacent violators) of the cell rates with the expected cell counts as weights, which maximizes the
expected complete-data Bernoulli likelihood under the order constraint. Everything else is x83_construct.py: G, the depth-level
priors g_x, the H0 predictions, the statistics and the instruments.
Grids m = 10, 20 and 40, each with article-cluster Poisson resamples (seed 8303 + m): 100 at m = 10 and 20, 50 at m = 40.
Descriptive addition: the depth slope among comments whose 4 raters all said civil, (n, K) = (4, 0) (observed and predicted).
H83d: at m = 20, >= 9 of the 12 classifier settings have a placebo log-RR excess with a 95% CI above 0.
H83e: at m = 20, Haiku and Sonnet without and with the parent all have a placebo depth-slope excess with a 95% CI above 0.
H83f: the verdicts of H83d and H83e are the same at m = 10 and m = 40.
usage: python3 x83d_mono.py OUT_JSON --grids 20,10 [--nb 100,100]   (run m = 40 separately: --grids 40 --nb 50)"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, time, numpy as np
sys.path.insert(0, (_ROOT + "/code/analysis"))
GR = [int(v) for v in sys.argv[sys.argv.index("--grids") + 1].split(",")] if "--grids" in sys.argv else [20, 10]
NBS = [int(v) for v in sys.argv[sys.argv.index("--nb") + 1].split(",")] if "--nb" in sys.argv else [100] * len(GR)
_argv = sys.argv; sys.argv = sys.argv[:2]                    # x83_construct parses its own --nb at import
import x83_construct as X
sys.argv = _argv


def pav(y, w):
    """Weighted isotonic (non-decreasing) regression by pool adjacent violators."""
    v, ww, n = [], [], []
    for yi, wi in zip(y, w):
        v.append(float(yi)); ww.append(float(wi)); n.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            tw = ww[-2] + ww[-1]; tv = (v[-2] * ww[-2] + v[-1] * ww[-1]) / tw if tw > 0 else 0.5 * (v[-2] + v[-1])
            v[-2:] = [tv]; ww[-2:] = [tw]; n[-2:] = [n[-2] + n[-1]]
    return np.repeat(v, n)


def em_f_mono(pop, Lc, gx, y, wcl, f=None, maxit=5000, tol=X.TOL):
    code = pop.ccode * 2 + y.astype(np.int64); w = np.bincount(code, weights=wcl, minlength=2 * pop.cS)
    j = np.flatnonzero(w > 0); w = w[j]; cp = j // 2; yy = (j % 2).astype(float)
    base = gx[pop.cU[cp, 2]] * Lc[cp]; m = Lc.shape[1]
    f = np.full(m, max(float(np.sum(w * yy) / w.sum()), 1e-4)) if f is None else np.maximum.accumulate(f.copy()); prev = -np.inf; wt = w / w.sum()
    for it in range(maxit):
        lk = base * np.where(yy[:, None] == 1, f[None, :], 1 - f[None, :]); Zs = lk.sum(1); ll = float(np.sum(wt * np.log(Zs)))
        if np.isfinite(prev) and abs(ll - prev) < tol * abs(prev):
            break
        prev = ll; r = lk * (wt / Zs)[:, None]; num = (r * yy[:, None]).sum(0); den = r.sum(0)
        f = np.clip(pav(num / np.maximum(den, 1e-300), den), 1e-7, 1 - 1e-7)
    return f


def statistics(pop, Lc, Ep, Ec, G, gx, f, we_pat, wc_pat):
    out = X.statistics(pop, Lc, Ep, Ec, G, gx, f, we_pat, wc_pat)
    E = X.pred_comment(pop, Lc, gx, f); x = pop.cU[:, 2].astype(float); s40 = (pop.cU[:, 0] == 4) & (pop.cU[:, 1] == 0); w = wc_pat * s40
    out["slope_40"] = X.slope_from(w.sum(), np.sum(w * x), np.sum(w * x * x), np.sum(w * E), np.sum(w * x * E))
    return out


def observed(pop, y, wcl, we):
    out = X.observed(pop, y, wcl, we)
    x = pop.x.astype(float); s40 = (pop.n == 4) & (pop.K == 0); w = wcl * s40
    out["slope_40"] = X.slope_from(w.sum(), np.sum(w * x), np.sum(w * x * x), np.sum(w * y), np.sum(w * x * y))
    return out


def run(full, s5, grids, nbs, seed=8303, log=print, em_f=em_f_mono):
    res = {"n_comments": full.N, "n_edges": int(len(full.ch)), "grids": {}, "monotone_f": em_f is em_f_mono}
    nC = int(full.cl.max() + 1)
    for m, nb in zip(grids, nbs):
        t0 = time.time(); g = (np.arange(m) + 0.5) / m
        Lc, Ep, Ec = X.emis(full, g); Lc5, Ep5, Ec5 = X.emis(s5, g)
        we1 = np.bincount(full.ecode, minlength=full.eS).astype(float); wc1 = np.bincount(full.ccode, minlength=full.cS).astype(float)
        we5 = np.bincount(s5.ecode, minlength=s5.eS).astype(float); wc5 = np.bincount(s5.ccode, minlength=s5.cS).astype(float)
        G = X.em_G(we1, Ep, Ec); gx = X.em_gx(wc1, full, Lc)
        F = {k: em_f(full, Lc, gx, y, np.ones(full.N)) for k, y in full.flags.items()}; F5 = {k: em_f(s5, Lc5, gx, y, np.ones(s5.N)) for k, y in s5.flags.items()}
        def one(pop, Lc_, Ep_, Ec_, G_, gx_, f_, y, we_pat, wc_pat, wcl, we):
            return statistics(pop, Lc_, Ep_, Ec_, G_, gx_, f_, we_pat, wc_pat), (observed(pop, y, wcl, we) if y is not None else None)
        D = {}
        for k, y in full.flags.items():
            D[k] = [one(full, Lc, Ep, Ec, G, gx, F[k], y, we1, wc1, np.ones(full.N), np.ones(len(full.ch)))]
        for k, y in s5.flags.items():
            D[k] = [one(s5, Lc5, Ep5, Ec5, G, gx, F5[k], y, we5, wc5, np.ones(s5.N), np.ones(len(s5.ch)))]
        for t in X.THR:
            D[f"pi>={t}"] = [one(full, Lc, Ep, Ec, G, gx, (g >= t).astype(float), None, we1, wc1, None, None)]
        log(f"m={m} full fit {time.time() - t0:.0f}s")
        rng = np.random.default_rng(seed + m); Wb = rng.poisson(1.0, size=(nb, nC)).astype(float); t1 = time.time()
        for b in range(nb):
            wcl = Wb[b][full.cl]; we = Wb[b][full.ecl]; wcl5 = Wb[b][s5.cl]; we5b = Wb[b][s5.ecl]
            wep = np.bincount(full.ecode, weights=we, minlength=full.eS); wcp = np.bincount(full.ccode, weights=wcl, minlength=full.cS)
            wep5 = np.bincount(s5.ecode, weights=we5b, minlength=s5.eS); wcp5 = np.bincount(s5.ccode, weights=wcl5, minlength=s5.cS)
            Gb = X.em_G(wep, Ep, Ec, G=G, maxit=3000, tol=1e-9); gxb = X.em_gx(wcp, full, Lc, gx=gx, maxit=3000, tol=1e-10)
            for k, y in full.flags.items():
                D[k].append(one(full, Lc, Ep, Ec, Gb, gxb, em_f(full, Lc, gxb, y, wcl, f=F[k], maxit=2000), y, wep, wcp, wcl, we))
            for k, y in s5.flags.items():
                D[k].append(one(s5, Lc5, Ep5, Ec5, Gb, gxb, em_f(s5, Lc5, gxb, y, wcl5, f=F5[k], maxit=2000), y, wep5, wcp5, wcl5, we5b))
            for t in X.THR:
                D[f"pi>={t}"].append(one(full, Lc, Ep, Ec, Gb, gxb, (g >= t).astype(float), None, wep, wcp, None, None))
            if (b + 1) % 10 == 0:
                log(f"m={m} resample {b + 1} {time.time() - t1:.0f}s")
        ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))] if len(a) > 1 else [float(a[0])]
        R = {"instruments": {}, "thresholds": {}, "n_resamples": nb}
        for k, DD in D.items():
            r = {}
            for s in DD[0][0]:
                p = np.array([d[0][s] for d in DD]); r[s] = {"pred": ci(p)}
                if DD[0][1] is not None:
                    o = np.array([d[1][s] for d in DD]); r[s]["obs"] = ci(o); r[s]["excess"] = ci(o - p)
            if k in F:
                r["f"] = F[k].tolist()
            elif k in F5:
                r["f"] = F5[k].tolist()
            (R["thresholds"] if k.startswith("pi>=") else R["instruments"])[k] = r
        hi = g >= 0.5; R["rr_inf"] = float((G[np.ix_(hi, hi)].sum() / G[hi].sum()) / (G[np.ix_(~hi, hi)].sum() / G[~hi].sum()))
        Ri = R["instruments"]
        up = [k for k in X.CLS12 if k in Ri and Ri[k]["placebo_lrr"]["excess"][1] > 0]; okb = [k for k in X.H83B if k in Ri and Ri[k]["placebo_slope"]["excess"][1] > 0]
        R["contagion"] = dict(n_above=len(up), above=up, pass_=bool(len(up) >= 9)); R["escalation"] = dict(n_above=len(okb), above=okb, pass_=bool(len(okb) == len(X.H83B)))
        res["grids"][str(m)] = R
        log(f"m={m} contagion {R['contagion']['n_above']}/12, escalation {R['escalation']['n_above']}/4")
    return res


if __name__ == "__main__":
    OUT = sys.argv[1]; full, s5 = X.load(); print("loaded", full.N, len(full.ch), flush=True)
    res = run(full, s5, GR, NBS, log=lambda s: print(s, flush=True))
    pp = 100.0
    for m, R in res["grids"].items():
        for k in [k_ for k_ in X.CLS12 + X.H83B if k_ in R["instruments"]]:
            r = R["instruments"][k]; e = r["placebo_lrr"]["excess"]; s = r["placebo_slope"]["excess"]
            print(f"m={m} {k:24s} placebo RR obs {np.exp(r['placebo_lrr']['obs'][0]):.2f} pred {np.exp(r['placebo_lrr']['pred'][0]):.2f} excess {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}] | "
                  f"slope obs {r['placebo_slope']['obs'][0]*pp:+.3f} pred {r['placebo_slope']['pred'][0]*pp:+.3f} [{s[1]*pp:+.3f},{s[2]*pp:+.3f}] | (4,0) slope obs {r['slope_40']['obs'][0]*pp:+.3f} pred {r['slope_40']['pred'][0]*pp:+.3f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
