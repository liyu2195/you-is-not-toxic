"""Synthetic check of x83d_mono.py (monotone f) on the simulated populations of x83_synth.py (no real data; Addendum 83, B83d): a known latent model with binary-vote stopping (PROTO unset) or a
categorical protocol with 'hard to say' answers (PROTO=cat); an H0 instrument should show no excess, a between-article clustered
instrument a placebo-RR excess, and a depth-drifting instrument a placebo-slope excess. 30 resamples.
usage: [PROTO=cat] python3 x83_synth.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np
sys.path.insert(0, (_ROOT + "/code/analysis"))
import x83_construct as X
import x83d_mono as MO
rng = np.random.default_rng(1)
A = 12000; per = 25; N = A * per
art = np.repeat(np.arange(A), per); par = np.full(N, -1); dep = np.zeros(N, np.int64)
for a in range(A):
    o = a * per
    for j in range(1, per):
        if rng.random() < 0.15:
            continue
        p = o + rng.integers(0, j); par[o + j] = p; dep[o + j] = dep[p] + 1
alpha = rng.normal(0, 0.7, A)[art]; tau = (rng.random(A) < 0.2)[art]
lg = np.zeros(N); mu = -3.0
for i in range(N):
    base = mu + alpha[i] + 0.06 * min(dep[i], 10)
    lg[i] = base + (0.55 * (lg[par[i]] - (mu + alpha[par[i]] + 0.06 * min(dep[par[i]], 10))) if par[i] >= 0 else 0.0) + rng.normal(0, 1.1)
pi = 1 / (1 + np.exp(-lg))
import os
if os.environ.get("PROTO") == "cat":
    # categorical protocol: toxic (pi), hard to say (h), civil; stop at 4 if all four agree in category; else add 1 or 2 raters and
    # stop if at most one rater differs from the modal category; else continue to 10 raters
    h = 0.04 + 0.5 * pi * (1 - pi)
    def cats(m_):
        u = rng.random((N, m_)); return np.where(u < pi[:, None], 2, np.where(u < (pi + h)[:, None], 1, 0))
    C = cats(10); a4 = (C[:, :4] == C[:, :1]).all(1); extra = np.where(rng.random(N) < 1 / 3, 1, 2)
    nn = np.where(a4, 4, 4 + extra)
    def maj_ok(row_n):
        out = np.zeros(N, bool)
        for e in (5, 6):
            s = row_n == e; c = C[s, :e]; cnt = np.stack([(c == k).sum(1) for k in range(3)], 1); out[s] = (e - cnt.max(1)) <= 1
        return out
    ok56 = maj_ok(nn); n = np.where(a4, 4, np.where(ok56, nn, 10)); K = np.array([(C[i, :n[i]] == 2).sum() for i in range(N)])
    import collections
    for e in (4, 5, 6, 10):
        s = n == e; print("n", e, "share", round(s.mean(), 3), "K dist", {k: round(v / s.sum(), 3) for k, v in sorted(collections.Counter(K[s]).items()) if v / s.sum() > 0.005})
else:
    k0 = rng.binomial(4, pi); stop = (k0 == 0) | (k0 == 4); n = np.where(stop, 4, 10); K = np.where(stop, k0, k0 + rng.binomial(6, pi))
f0 = lambda p: 0.02 + 0.6 / (1 + np.exp(-12 * (p - 0.45)))
flags = {"h0": (rng.random(N) < f0(pi)).astype(float),
         "clustered": (rng.random(N) < np.clip(f0(pi) + 0.06 * tau, 0, 1)).astype(float),
         "depthy": (rng.random(N) < np.clip(f0(pi) + 0.006 * np.minimum(dep, 10), 0, 1)).astype(float)}
x = np.minimum(dep, 10)
full = X.Pop(n, K, x, par, art, flags)
sel = np.flatnonzero(np.isin(art, np.arange(0, A, 20))); mp = -np.ones(N, np.int64); mp[sel] = np.arange(len(sel))
par5 = np.where(par[sel] >= 0, mp[np.maximum(par[sel], 0)], -1)
s5 = X.Pop(n[sel], K[sel], x[sel], par5, art[sel], {"sonnet": flags["h0"][sel]})
print("prev", full.z.mean(), "edges", len(full.ch), flush=True)
res = MO.run(full, s5, [20, 40], [30, 30], log=lambda s: print(s, flush=True))
R = res["grids"]["20"]["instruments"]
for k, r in R.items():
    print(k, "placebo lrr obs %.3f pred %.3f excess %s" % (r["placebo_lrr"]["obs"][0], r["placebo_lrr"]["pred"][0], np.round(r["placebo_lrr"]["excess"], 3)),
          "| slope obs %.5f pred %.5f excess %s" % (r["placebo_slope"]["obs"][0], r["placebo_slope"]["pred"][0], np.round(r["placebo_slope"]["excess"], 5)),
          "| lrr obs %.3f pred %.3f" % (r["lrr"]["obs"][0], r["lrr"]["pred"][0]), "| law obs %.5f pred %.5f" % (r["depth_law"]["obs"][0], r["depth_law"]["pred"][0]))
for k, r in res["grids"]["40"]["instruments"].items():
    print("m=40", k, "placebo lrr excess", np.round(r["placebo_lrr"]["excess"], 3), "slope obs %.5f pred %.5f excess %s" % (r["placebo_slope"]["obs"][0], r["placebo_slope"]["pred"][0], np.round(r["placebo_slope"]["excess"], 5)))
for k, r in res["grids"]["20"]["thresholds"].items():
    print(k, {s: np.round(v["pred"], 4).tolist() for s, v in r.items()})
# truth for the threshold instrument pi >= 0.5 (placebo lrr among gold-negative edges)
yv = (pi >= 0.5).astype(float); neg = (full.z[full.pa] == 0) & (full.z[full.ch] == 0); yp, yc = yv[full.pa][neg], yv[full.ch][neg]
print("true placebo lrr of 1{pi>=0.5}", np.log((np.sum(yp * yc) / yp.sum()) / (np.sum((1 - yp) * yc) / np.sum(1 - yp))))
cn = full.z == 0; print("true placebo slope of 1{pi>=0.5}", np.polyfit(x[cn], yv[cn], 1)[0])

yv = (pi >= 0.5).astype(float)
json.dump({"protocol": os.environ.get("PROTO", "binary"), "instruments": {k: {s_: v_ for s_, v_ in r.items() if s_ != "f"} for k, r in R.items()},
           "instruments_m40": {k: {s_: v_ for s_, v_ in r.items() if s_ != "f"} for k, r in res["grids"]["40"]["instruments"].items()},
           "thresholds": res["grids"]["20"]["thresholds"],
           "truth_pi05": {"placebo_lrr": float(np.log((np.sum(yp * yc) / yp.sum()) / (np.sum((1 - yp) * yc) / np.sum(1 - yp)))), "placebo_slope": float(np.polyfit(x[cn], yv[cn], 1)[0])}},
          open(sys.argv[1], "w"), indent=1)
print("saved", sys.argv[1])
