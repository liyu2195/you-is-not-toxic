"""Addendum 88, B88f' run in parts (same computation, seeds and outputs as scripts/x88_fresh_gold.py; the parts are merged by
scripts/x88_fresh_gold_merge.py). One part = one labeler, one budget, audits r0..r1-1; it saves every audit's estimate, intervals
and realized law so the merge reproduces the single-process summary exactly.
usage: python3 x88_fresh_gold_part.py OUT_JSON LABELER BUDGET R0 R1

The original docstring follows.
Addendum 88, B88f: audits that buy fresh, noisier gold.
Each purchased label is drawn when bought from the comment's own rater panel (k toxic votes of n): (i) one rater; (ii) the majority
of three drawn without replacement; (iii) adaptive: three raters, two more if they disagree (all n if n < 5), majority (vote
fraction >= 0.5, as gold). Target: the law of the expected labels under the process (exact, hypergeometric); also the realized law of
the drawn labels. THREADSAFE's frozen item design (pps_seq4_ba: 4 folds, pilot 10%, delta auto, Bayesian predictive model, the
confirmation's features and auxiliaries) and fixed gold for comparison, on the seven 5% Civil Comments cells of the confirmation;
budgets 1,000 and 4,000; 500 audits per entry (seed 8807). Reported: coverage of the expected and of the realized law, mean width
relative to the fixed-gold audit, raters per label under (iii), the share of purchased labels that are gold-positive.
H88f: over the 210 entries THREADSAFE's coverage of the expected law averages >= 0.94 with no entry below 0.90.
usage: python3 x88_fresh_gold.py OUT_JSON [labeler ...]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, zlib, time, numpy as np, pandas as pd
from scipy.stats import hypergeom
sys.path.insert(0, (_ROOT + "/code"))
from threadsafe2 import laws as L, pps as PPS
import os
W = _ROOT; OUT = sys.argv[1]; REPS = int(os.environ.get("X88F_REPS", 500)); BUD = (1000, 4000)
LABS = sys.argv[2:] or ["cn_off@m", "fb_hate@m", "gemma_m", "mh_dist@m", "sn_rob@m", "sonnet", "sonnet_ctx"]
AUX = ["detoxify_original@m", "imsypp@m"]
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet")
z = P.z.values.astype(float); depth = P.depth.values; kk = P.k.values.astype(float); N = len(P)
xd = np.minimum(depth, 10).astype(float); big = kk >= 2; logk = np.where(big, np.log2(np.maximum(kk, 1)), 0.0)
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
nv = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * nv).astype(np.int64), 0, nv)
REPLIES = np.bincount(P.parent_row.values[P.parent_row.values >= 0], minlength=N)
asz = P.groupby("article").article.transform("size").values


def size_phi(unit):
    u = pd.factorize(unit)[0]; ku = np.bincount(u).astype(float); xu = np.log2(np.maximum(ku, 1)); ok = ku >= 2
    xb = xu[ok].mean(); ss = ((xu[ok] - xb) ** 2).sum()
    return np.where(ok[u], (xu[u] - xb) / (ku[u] * ss), 0.0)


def metrics_for(y):                                                     # as x09_audit_sim.py
    return [L.Linear("depth_law", (xd - xd.mean()) / ((xd - xd.mean()) ** 2).sum()), L.Linear("size_law", size_phi(P.unit.values)),
            L.Linear("article_size_law", size_phi(P.article.values)),
            L.PlaceboSlope("placebo_depth", xd, y, np.ones(N)), L.PlaceboSlope("placebo_size", logk, y, np.where(big, 1.0 / np.maximum(kk, 1), 0.0))]


# exact expected labels under each process
def p_one():
    return kv / nv
def p_three():
    out = np.zeros(N)
    for h in (2, 3):
        out += hypergeom.pmf(h, nv, kv, np.minimum(3, nv))
    return np.where(nv >= 3, out, (kv / nv >= 0.5).astype(float))
def p_adaptive():
    out = np.zeros(N); m3 = np.minimum(3, nv)
    for h3 in range(4):
        ph3 = hypergeom.pmf(h3, nv, kv, m3); unanimous = (h3 == 0) | (h3 == m3)
        r = np.where(unanimous, 0, np.minimum(2, nv - m3))
        for h2 in range(3):
            ph2 = np.where(r == 0, float(h2 == 0), hypergeom.pmf(h2, np.maximum(nv - m3, 1), np.clip(kv - h3, 0, np.maximum(nv - m3, 1)), np.maximum(r, 1)))
            tox = ((h3 + h2) / (m3 + r) >= 0.5 - 1e-12)
            out += np.nan_to_num(ph3 * ph2) * tox
    return out
PEXP = {"one": p_one(), "three": p_three(), "adaptive": p_adaptive()}


def draw(proc, rng):
    if proc == "one":
        return (rng.hypergeometric(kv, nv - kv, np.ones_like(nv)) >= 1).astype(float), np.ones(N)
    m3 = np.minimum(3, nv); h3 = rng.hypergeometric(kv, nv - kv, m3)
    if proc == "three":
        return (h3 / m3 >= 0.5 - 1e-12).astype(float), m3.astype(float)
    unanimous = (h3 == 0) | (h3 == m3); r = np.where(unanimous, 0, np.minimum(2, nv - m3))
    rem_k = kv - h3; rem_n = nv - m3; h2 = np.zeros(N, np.int64); ok = r > 0
    h2[ok] = rng.hypergeometric(rem_k[ok], rem_n[ok] - rem_k[ok], r[ok])
    return ((h3 + h2) / (m3 + r) >= 0.5 - 1e-12).astype(float), (m3 + r).astype(float)


class Tracked(np.ndarray):                                              # records which labels the design reads (exposure)
    seen = []
    def __getitem__(self, idx):
        r = super().__getitem__(idx)
        i0 = idx[0] if isinstance(idx, tuple) else idx
        if isinstance(i0, np.ndarray) and i0.dtype.kind in "iu":
            Tracked.seen.append(i0)
        return np.asarray(r)


def wrap(m):                                                             # records the last plug-in (the final one) of theta_c
    f0 = m.theta_c
    def g(T):
        th, c = f0(T); m._last = (th, c); return th, c
    m.theta_c = g
    return m


def run_one(lab, B, zz, rng):
    lev = PPS.score_levels(P[f"s:{lab}"].values); ms = [wrap(m) for m in metrics_for(np.nan_to_num(P[f"y:{lab}"].values.astype(float)))]
    aux = [P[f"s:{e}"].values for e in AUX if e.split("@")[0] != lab.split("@")[0]]
    key = ("X", lab)
    if key not in CACHE:
        CACHE[key] = PPS.design_matrix(lev, depth, logk, asz, aux=aux, replies=REPLIES)
    zt = zz.view(Tracked); Tracked.seen = []
    o = PPS.run_seq(ms, lev, B, rng, zt, CACHE[key], mult=[1.0, 1.0, 1.0, 0.25, 0.25], pilot_share=0.10, K=4, pilot_cells=lev * 5 + L.depth_bucket(depth),
                    delta="auto", bayes=True)
    lab_idx = np.unique(np.concatenate(Tracked.seen)) if Tracked.seen else np.array([], int)
    # B88f' (added after B88f's first cells): the rater variance sum_i b_i^2 p_i (1 - p_i) that the folds' variances omit for the
    # expected law (Theorem 2(d)), with p from the working model refitted on every label bought (conservative by Jensen)
    beta, Hb = PPS.logit_fit(CACHE[key][lab_idx], zz[lab_idx], lam=1.0, return_hessian=True)
    ph = np.clip(PPS.predictive_p(CACHE[key], beta, Hb), 1e-3, 1 - 1e-3)
    o["_vr"] = {m.name: float(((m.B @ m._last[1]) ** 2 * ph * (1 - ph)).sum()) for m in ms}
    return ms, o, lab_idx


CACHE = {}; PROCS = ["fixed", "one", "three", "adaptive"]
LAB, B, R0, R1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
t0 = time.time(); lab = LAB
ms = metrics_for(np.nan_to_num(P[f"y:{lab}"].values.astype(float)))
truth = {"fixed": {m.name: L.value(m, z) for m in ms}} | {pr: {m.name: L.value(m, PEXP[pr]) for m in ms} for pr in PEXP}
acc = {pr: {m.name: [] for m in ms} for pr in PROCS}; raters = []; expo = {pr: [] for pr in PROCS}
for r in range(R0, R1):
    for pr in PROCS:
        rng = np.random.default_rng([8807, zlib.crc32(lab.encode()), B, r, zlib.crc32(pr.encode())])
        if pr == "fixed":
            zz, cost = z.copy(), None
        else:
            zz, cost = draw(pr, rng)
        ms_, o, li = run_one(lab, B, zz, rng)
        if pr == "adaptive" and len(li):
            raters.append(float(cost[li].mean()))
        if len(li):
            expo[pr].append(float(z[li].mean()))
        for m in ms_:
            th, v = o[m.name]; lo, hi = L.wald(th, v); lo2, hi2 = L.wald(th, v + o["_vr"][m.name])
            acc[pr][m.name].append([float(t) for t in (th, lo, hi, L.value(m, zz), lo2, hi2)])
    if (r - R0) % 25 == 24:
        print(lab, B, r + 1, f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(lab=lab, B=B, r0=R0, r1=R1, truth=truth, metrics=[m.name for m in ms], acc=acc, raters=raters, expo=expo), open(OUT, "w"))
print("saved", OUT, f"{time.time() - t0:.0f}s", flush=True)
