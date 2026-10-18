"""Addendum 91, B92a (descriptive): every LLM at matched operating points on the 5% Civil Comments sample (60,668 comments).
Operating points:
  Haiku and Sonnet (four-level answer): native ("toxic" or "very toxic"); "very toxic" only (what prevalence matching picks);
     their 0-100 rating (results/x75_rating) at gold prevalence (ties at random, seed 9010, as x90_dose.py) and at the cut that
     maximizes kappa against all gold labels of the sample.
  Qwen2.5-7B, Gemma 3 (stored scores), OLMo-2, Granite (B84 scores, P(A) + P(B)), Llama-3.1-8B (B93, same reading): at gold prevalence (Qwen and Gemma as stored;
     OLMo-2 and Granite as x84_families.py) and at the kappa-maximizing cut over 400 score quantiles.
For each: flag rate, kappa; excess of the depth law over gold's (points per level) with 500 article-cluster resamples (seed 9201), and
the same excess at a fixed number of second-person words (0..4+, pooled within-stratum slopes, as x90_dose.py); excess of the thread
and article size laws (points per doubling, units weighted by their cluster's resample weight); excess of the log reply risk ratio.
usage: python3 x92_matched.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(np.int64).values; N = len(P)
z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
T = {}
for f in sorted(glob.glob(f"{W}/data/second/s5/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); T[int(r["id"])] = r.get("text") or ""
cb = np.minimum(np.array([len(YOU.findall(T.get(int(i), ""))) for i in ids]), 4)
pr = P.parent_row.values; ch = np.flatnonzero(pr >= 0); pa = pr[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]; ecl = cl[ch]
Wb = np.vstack([np.ones(nC), np.random.default_rng(9201).poisson(1.0, size=(NB, nC))])
S = lambda v: Wb @ np.bincount(cl, v, nC)
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
allm = np.ones(N, bool)
DIR = {"llama": "llama31"}   # Llama-3.1-8B-Instruct (B93)


def slope(v, m):
    m_ = m.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
    return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)


def pooled(v, m, st):
    num = 0.0; den = 0.0
    for s_ in np.unique(st[m]):
        m_ = (m & (st == s_)).astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
        ok = n_ > 0; nn = np.where(ok, n_, 1.0); num = num + np.where(ok, sxy - sx * sy / nn, 0.0); den = den + np.where(ok, sxx - sx ** 2 / nn, 0.0)
    return 100 * num / den


def size_law_b(v, unit):
    uu = pd.factorize(unit)[0]; ku = np.bincount(uu).astype(float); ok = ku >= 2; fu = np.bincount(uu, v) / ku; lx = np.log2(ku)
    ucl = np.zeros(ku.size, int); ucl[uu] = cl; w = Wb[:, ucl[ok]]; lxo, fuo = lx[ok], fu[ok]
    sw = w.sum(1); mx = (w * lxo).sum(1) / sw; my = (w * fuo).sum(1) / sw
    return 100 * (w * (lxo - mx[:, None]) * (fuo - my[:, None])).sum(1) / (w * (lxo - mx[:, None]) ** 2).sum(1)


def log_rr(v):
    vp, vc = v[pa], v[ch]
    a, b, c, d = (Wb @ np.bincount(ecl, t, nC) for t in (vp * vc, vp, (1 - vp) * vc, 1 - vp))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((a / b) / (c / d))


def kappa(y):
    po = float((y == z).mean()); p1, p2 = y.mean(), z.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2); return (po - pe) / (1 - pe)


K = int(round(z.mean() * N))
def at_prev(s, seed):
    s = np.where(np.isnan(s), -np.inf, s); o = np.lexsort((np.random.default_rng(seed).random(N), -s)); y = np.zeros(N); y[o[:K]] = 1.0; return y
def kmax(s, cuts):
    best = None
    for c in cuts:
        y = (s >= c).astype(float)
        if 0 < y.sum() < N:
            k_ = kappa(y)
            if best is None or k_ > best[0]:
                best = (k_, float(c), y)
    return best
def rating(model):
    d = {}
    for l in open(f"{W}/results/x75_rating/{model}_ctx0.jsonl"):
        r = json.loads(l); v = r.get("rating"); d[int(r["id"])] = float(v) if v is not None else np.nan
    return np.array([d.get(int(i), np.nan) for i in ids])
def b84(model):
    sc = {}
    for f in sorted(glob.glob(f"{W}/results/b84/{model}/shard_*_tox_ctx0.jsonl")):
        for l in open(f):
            r = json.loads(l); sc[int(r["id"])] = float(r["p"][0] + r["p"][1])
    return np.array([sc.get(int(i), np.nan) for i in ids])


g_d = slope(z, allm); g_fix = pooled(z, allm, cb); g_ts = size_law_b(z, P.unit.values); g_as = size_law_b(z, P.article.values); g_rr = log_rr(z)
SET = []
for m in ("haiku", "sonnet"):
    lev = np.nan_to_num(P[f"s:{m}"].values.astype(float), nan=-1)
    SET += [(m, "native", (lev >= 2).astype(float), None), (m, "very toxic", (lev >= 3).astype(float), None)]
    rt = rating(m); SET.append((m, "rating at gold prevalence", at_prev(rt, 9010), None))
    kb = kmax(np.nan_to_num(rt, nan=-1), np.arange(1, 101)); SET.append((m, "rating at max kappa", kb[2], kb[1]))
for m, col in (("qwen", "s:qwen7b_m"), ("gemma", "s:gemma_m"), ("olmo2", None), ("granite", None), ("llama", None)):
    if not col and len(glob.glob(f"{W}/results/b84/{DIR.get(m, m)}/shard_*_tox_ctx0.jsonl")) < 4:
        continue
    s_ = P[col].values.astype(float) if col else b84(DIR.get(m, m))
    y_ = np.nan_to_num(P[f"y:{col[2:]}"].values.astype(float)) if col else at_prev(s_, 8400)
    SET.append((m, "gold prevalence", y_, None))
    qs = np.unique(np.nanquantile(s_, np.linspace(0.5, 0.995, 400))); kb = kmax(np.nan_to_num(s_, nan=-np.inf), qs); SET.append((m, "max kappa", kb[2], kb[1]))
res = {"N": N, "gold": dict(depth=ci(g_d), thread_size=ci(g_ts), article_size=ci(g_as), rr=float(np.exp(g_rr[0]))), "rows": []}
for m, op, y, cut in SET:
    row = dict(model=m, op=op, cut=cut, flag_rate=float(y.mean()), kappa=kappa(y), fpr=float(y[z == 0].mean()),
               depth_excess=ci(slope(y, allm) - g_d), depth_excess_fixed=ci(pooled(y, allm, cb) - g_fix),
               thread_size_excess=ci(size_law_b(y, P.unit.values) - g_ts), article_size_excess=ci(size_law_b(y, P.article.values) - g_as),
               log_rr_excess=ci(log_rr(y) - g_rr))
    res["rows"].append(row)
    print(f"{m:8s} {op:26s} flag {row['flag_rate']:.3f} kappa {row['kappa']:.3f} depth {row['depth_excess'][0]:+.2f} [{row['depth_excess'][1]:+.2f}, {row['depth_excess'][2]:+.2f}]"
          f" fixed {row['depth_excess_fixed'][0]:+.2f} [{row['depth_excess_fixed'][1]:+.2f}, {row['depth_excess_fixed'][2]:+.2f}] thread {row['thread_size_excess'][0]:+.2f}"
          f" article {row['article_size_excess'][0]:+.2f} logRR {row['log_rr_excess'][0]:+.2f}", flush=True)
print("gold depth %.2f thread %.2f article %.2f RR %.2f" % (g_d[0], g_ts[0], g_as[0], np.exp(g_rr[0])))
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
