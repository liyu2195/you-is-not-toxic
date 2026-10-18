"""arXiv revision (re-analysis of stored labels, no new labeling): the dose analysis of x90_dose.py for Qwen on all Civil Comments
reply threads, at gold's flag rate (y:qwen7b_m) and at its own cut (y:qwen7b, p > 0.5). x90 has these quantities for Qwen only on the
5% sample. Same detector, same statistics and the same 500 cluster resamples (seed 9001) as x90_dose.py:
  - the placebo depth slope among gold negatives, overall and pooled within the number of second-person words (0..4+);
  - the depth law's distortion (labeler minus gold), overall and pooled within the number;
  - the false-positive rate by number.
usage: python3 -I x98g_qwen_dose.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
FULL = (_ROOT + "/scratch/full")           # the comment texts of cc_threads (as x90)
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]

P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet"); ids = P.id.astype(np.int64).values; N = len(P); assert N == 1251487
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
need = set(ids.tolist()); cnt_d = {}
for f in sorted(glob.glob(f"{FULL}/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = int(r["id"])
        if i in need:
            cnt_d[i] = len(YOU.findall(r.get("text") or ""))
assert len(cnt_d) == N, (len(cnt_d), N)
cnt = np.array([cnt_d[int(i)] for i in ids]); cb = np.minimum(cnt, 4); neg = z == 0
Wb = np.vstack([np.ones(nC), np.random.default_rng(9001).poisson(1.0, size=(NB, nC))])
S = lambda v: Wb @ np.bincount(cl, v, nC)


def rate(v, mask):
    with np.errstate(divide="ignore", invalid="ignore"):
        return S(v * mask) / S(mask.astype(float))


def slope(v, mask):
    m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
    return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)


def pooled(v, mask, strata):                                                 # pooled within-stratum OLS slope (stratum fixed effects), as x90
    num = 0.0; den = 0.0
    for s_ in np.unique(strata[mask]):
        m_ = (mask & (strata == s_)).astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
        ok = n_ > 0; nn = np.where(ok, n_, 1.0)
        num = num + np.where(ok, sxy - sx * sy / nn, 0.0); den = den + np.where(ok, sxx - sx ** 2 / nn, 0.0)
    return 100 * num / den


def stats(y):
    allm = np.ones(N, bool)
    o = dict(flag_rate=float(y.mean()), fpr=float(y[neg].mean()),
             fpr_by_count=[ci(rate(y, neg & (cb == k))) for k in range(5)], n_by_count=[int((neg & (cb == k)).sum()) for k in range(5)],
             placebo=ci(slope(y, neg)), within_count=ci(pooled(y, neg, cb)),
             law_distortion=ci(slope(y, allm) - slope(z, allm)), law_distortion_within_count=ci(pooled(y, allm, cb) - pooled(z, allm, cb)))
    return o


res = {"population": "cc_threads", "N": N, "labelers": {}}
for nm, col in (("qwen_gold_rate", "y:qwen7b_m"), ("qwen_own_cut", "y:qwen7b")):
    res["labelers"][nm] = stats(np.nan_to_num(P[col].values.astype(float)))
    r = res["labelers"][nm]
    print(nm, "flags", round(r["flag_rate"], 4), "| placebo", [round(v, 3) for v in r["placebo"]], "at fixed no.", [round(v, 3) for v in r["within_count"]],
          "| law distortion", [round(v, 3) for v in r["law_distortion"]], "at fixed no.", [round(v, 3) for v in r["law_distortion_within_count"]], flush=True)
# the stored Haiku values reproduce x90 (same code path): a check that the counts and resamples are the same
h = stats(np.nan_to_num(P["y:haiku"].values.astype(float))); X90 = json.load(open(f"{W}/results/x90/x90_dose.json"))["corpora"]["cc_threads"]["labelers"]["haiku"]
assert all(abs(a - b) < 1e-9 for a, b in zip(h["law_distortion_within_count"], X90["law_distortion_within_count"])), (h["law_distortion_within_count"], X90["law_distortion_within_count"])
assert all(abs(a - b) < 1e-9 for a, b in zip(h["within_count"], X90["within_count"]))
res["check"] = "Haiku's law_distortion_within_count and within_count equal x90_dose.json's (asserted to 1e-9)"
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
