"""Addendum 83, B83d run in parallel: the same computation as x83d_mono.py (same full-data fits, same resample weights drawn from
default_rng(8303 + m), same per-resample warm-started fits and statistics), with the resamples spread over worker processes.
The draws are assembled in resample order, so the output equals a sequential run's.
usage: python3 x83d_parallel.py OUT_JSON --grids 20,10,40 --nb 100,100,50 [--procs 46]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, time, os, numpy as np
from multiprocessing import get_context
sys.path.insert(0, (_ROOT + "/code/analysis"))
GR = [int(v) for v in sys.argv[sys.argv.index("--grids") + 1].split(",")]
NBS = [int(v) for v in sys.argv[sys.argv.index("--nb") + 1].split(",")]
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else max(1, os.cpu_count() - 2)
OUT = sys.argv[1]
_argv = sys.argv; sys.argv = sys.argv[:2]
import x83_construct as X
import x83d_mono as MO
sys.argv = _argv
S = {}                                                         # shared with forked workers


def one(pop, Lc_, Ep_, Ec_, G_, gx_, f_, y, we_pat, wc_pat, wcl, we):
    return MO.statistics(pop, Lc_, Ep_, Ec_, G_, gx_, f_, we_pat, wc_pat), (MO.observed(pop, y, wcl, we) if y is not None else None)


def resample(b):
    full, s5, g, Lc, Ep, Ec, Lc5, Ep5, Ec5, G, gx, F, F5, Wb = (S[k] for k in ("full", "s5", "g", "Lc", "Ep", "Ec", "Lc5", "Ep5", "Ec5", "G", "gx", "F", "F5", "Wb"))
    wcl = Wb[b][full.cl]; we = Wb[b][full.ecl]; wcl5 = Wb[b][s5.cl]; we5b = Wb[b][s5.ecl]
    wep = np.bincount(full.ecode, weights=we, minlength=full.eS); wcp = np.bincount(full.ccode, weights=wcl, minlength=full.cS)
    wep5 = np.bincount(s5.ecode, weights=we5b, minlength=s5.eS); wcp5 = np.bincount(s5.ccode, weights=wcl5, minlength=s5.cS)
    Gb = X.em_G(wep, Ep, Ec, G=G, maxit=3000, tol=1e-9); gxb = X.em_gx(wcp, full, Lc, gx=gx, maxit=3000, tol=1e-10)
    out = {}
    for k, y in full.flags.items():
        out[k] = one(full, Lc, Ep, Ec, Gb, gxb, MO.em_f_mono(full, Lc, gxb, y, wcl, f=F[k], maxit=2000), y, wep, wcp, wcl, we)
    for k, y in s5.flags.items():
        out[k] = one(s5, Lc5, Ep5, Ec5, Gb, gxb, MO.em_f_mono(s5, Lc5, gxb, y, wcl5, f=F5[k], maxit=2000), y, wep5, wcp5, wcl5, we5b)
    for t in X.THR:
        out[f"pi>={t}"] = one(full, Lc, Ep, Ec, Gb, gxb, (g >= t).astype(float), None, wep, wcp, None, None)
    return b, out


def main():
    full, s5 = X.load(); print("loaded", full.N, len(full.ch), "procs", PROCS, flush=True)
    res = {"n_comments": full.N, "n_edges": int(len(full.ch)), "grids": {}, "monotone_f": True, "parallel": True}
    nC = int(full.cl.max() + 1)
    for m, nb in zip(GR, NBS):
        t0 = time.time(); g = (np.arange(m) + 0.5) / m
        Lc, Ep, Ec = X.emis(full, g); Lc5, Ep5, Ec5 = X.emis(s5, g)
        we1 = np.bincount(full.ecode, minlength=full.eS).astype(float); wc1 = np.bincount(full.ccode, minlength=full.cS).astype(float)
        we5 = np.bincount(s5.ecode, minlength=s5.eS).astype(float); wc5 = np.bincount(s5.ccode, minlength=s5.cS).astype(float)
        G = X.em_G(we1, Ep, Ec); gx = X.em_gx(wc1, full, Lc)
        F = {k: MO.em_f_mono(full, Lc, gx, y, np.ones(full.N)) for k, y in full.flags.items()}; F5 = {k: MO.em_f_mono(s5, Lc5, gx, y, np.ones(s5.N)) for k, y in s5.flags.items()}
        D = {}
        for k, y in full.flags.items():
            D[k] = [one(full, Lc, Ep, Ec, G, gx, F[k], y, we1, wc1, np.ones(full.N), np.ones(len(full.ch)))]
        for k, y in s5.flags.items():
            D[k] = [one(s5, Lc5, Ep5, Ec5, G, gx, F5[k], y, we5, wc5, np.ones(s5.N), np.ones(len(s5.ch)))]
        for t in X.THR:
            D[f"pi>={t}"] = [one(full, Lc, Ep, Ec, G, gx, (g >= t).astype(float), None, we1, wc1, None, None)]
        print(f"m={m} full fit {time.time() - t0:.0f}s", flush=True)
        rng = np.random.default_rng(8303 + m); Wb = rng.poisson(1.0, size=(nb, nC)).astype(float)
        S.update(full=full, s5=s5, g=g, Lc=Lc, Ep=Ep, Ec=Ec, Lc5=Lc5, Ep5=Ep5, Ec5=Ec5, G=G, gx=gx, F=F, F5=F5, Wb=Wb)
        t1 = time.time(); outs = {}
        with get_context("fork").Pool(min(PROCS, nb)) as pool:
            for b, o in pool.imap_unordered(resample, range(nb)):
                outs[b] = o
                if len(outs) % 10 == 0:
                    print(f"m={m} resamples done {len(outs)} {time.time() - t1:.0f}s", flush=True)
        for b in range(nb):
            for k in D:
                D[k].append(outs[b][k])
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
        print(f"m={m} contagion {R['contagion']['n_above']}/12, escalation {R['escalation']['n_above']}/4, {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    v = {m: res["grids"][m] for m in res["grids"]}
    if "20" in v:
        res["H83d"] = v["20"]["contagion"]; res["H83e"] = v["20"]["escalation"]
        if "10" in v and "40" in v:
            same = all(v[m]["contagion"]["pass_"] == v["20"]["contagion"]["pass_"] and v[m]["escalation"]["pass_"] == v["20"]["escalation"]["pass_"] for m in ("10", "40"))
            res["H83f"] = dict(pass_=bool(same))
    json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT, {k: res.get(k) for k in ("H83d", "H83e", "H83f")}, flush=True)


if __name__ == "__main__":
    main()
