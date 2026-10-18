"""Addendum 55, check 2: placebo diagnostics of conversation laws, for every stored labeler and the open classifiers.
Per dataset an item table (unit, cluster, depth, parent, z, labeler flags). Per labeler, statistics with 95% Poisson cluster
bootstrap CIs (500 resamples; clusters = articles on Civil Comments, units elsewhere), from per-cluster sufficient statistics:
  placebo depth slope : OLS slope of the flag on min(depth, 10) among gold negatives (absolute, and relative = slope / mean);
  placebo size slope  : OLS slope over units (>= 2 comments) of the unit's flag fraction among its gold negatives on log2(size);
  placebo RR          : P(child flag | parent flag) / P(child flag | parent no flag), edges with both ends gold-negative;
  sibling RR          : the same with a random earlier gold-negative non-parent comment of the unit in place of the parent;
  law slopes and RR   : depth slope of the flag (all comments), size slope of the unit flag fraction, RR over all edges;
                        the same for gold; decomposition of the depth law: (TPR-FPR) slope(p) + (1-p) slope(FPR) + p slope(TPR).
Lenient-human benchmark on Civil Comments: flag = crowd vote fraction >= 0.3 (gold negatives have fraction < 0.5).
Open classifiers (results/x55_open/*/*.parquet): det_orig, det_unb (contaminated on Civil Comments), ims; flags at 0.6
(Detoxify), argmax != acceptable (IMSYPP), and prevalence-matched thresholds ('@m').
Addendum 56 completes the registered specification: Qwen on CGA-Wiki; the f96 Avalle pipeline (conversation fraction on
log2 size, Mann-Kendall tau on log2-bin means with >= 30 conversations, 200-shuffle null) under gold and every labeler; the
size-slope decomposition. It adds an FPR-matched human benchmark on Civil Comments: per labeler, gold positives plus the gold
negatives with the highest crowd vote fractions, up to the labeler's FPR (ties at random, seed 56); saturated (unused) when
that FPR exceeds the share of gold negatives with any toxic vote.
usage: python3 x08_placebo.py OUT_JSON [--nb 500]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, glob, numpy as np, pandas as pd
from scipy.stats import kendalltau
W = _ROOT; OUT = sys.argv[1]; NB = int(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else 500
rng = np.random.default_rng(55)

def open_scores(tags):
    fs = [f for t in tags for f in glob.glob(f"{W}/results/x55_open/*/{t}.parquet")]
    if not fs: return None
    return pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("id").set_index("id")
def add_open(D, S, prev, contaminated=False):
    if S is None: return []
    S = S.reindex(D.index); names = []
    for col, nm in [("det_orig", "detoxify_original"), ("det_unb", "detoxify_unbiased")]:
        v = S[col]; t = float(np.nanquantile(v, 1 - prev))
        D[f"{nm}@0.6"] = (v >= 0.6).astype(float).where(v.notna()); D[f"{nm}@m"] = (v > t).astype(float).where(v.notna()); names += [f"{nm}@0.6", f"{nm}@m"]
    P = S[[f"ims_p{k}" for k in range(4)]].values; ok = ~np.isnan(P).any(1); am = np.where(ok, P.argmax(1), -1); notacc = 1 - S.ims_p0
    D["imsypp@argmax"] = np.where(ok, (am != 0).astype(float), np.nan); t = float(np.nanquantile(notacc, 1 - prev))
    D["imsypp@m"] = (notacc > t).astype(float).where(notacc.notna()); names += ["imsypp@argmax", "imsypp@m"]
    return names
def letters_tox(files):
    R = pd.DataFrame([json.loads(l) for f in files for l in open(f)])
    if "error" in R: R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last").set_index("id"); return R.ans.map({"A": 3, "B": 2, "C": 1, "D": 0})

def sibling_parents(D, unit, pos, cand_mask, child_idx, parent_idx):
    """For each child row, a uniformly random earlier candidate row (pos < child's pos) of the same unit that is not its parent; -1 if none."""
    uc = pd.factorize(D[unit])[0].astype(np.int64); p = np.asarray(D[pos].values, dtype=np.int64); p = p - p.min()
    key = uc * (int(p.max()) + 2) + p
    crow = np.flatnonzero(np.asarray(cand_mask.values)); o = np.argsort(key[crow], kind="stable"); crow = crow[o]; ck = key[crow]; cu = uc[crow]
    ustart = np.searchsorted(cu, uc[child_idx], side="left"); before = np.maximum(np.searchsorted(ck, key[child_idx], side="left") - ustart, 0)
    r = np.floor(rng.random(len(child_idx)) * before).astype(int)
    cand = np.where(before > 0, crow[np.clip(ustart + r, 0, len(crow) - 1)], -1)
    clash = (cand == parent_idx) & (before > 0); alt_r = np.where(r + 1 < before, r + 1, r - 1)
    alt = np.where(alt_r >= 0, crow[np.clip(ustart + alt_r, 0, len(crow) - 1)], -1)
    return np.where(clash, alt, cand)

def suff(D, labs, unit, clus, depth, pos, edges, extra=()):
    """Per-cluster sufficient statistics for every labeler (and gold)."""
    cl = pd.factorize(D[clus])[0]; nC = cl.max() + 1; x = np.minimum(D[depth].values.astype(float), 10); z = D.z.values.astype(float)
    k = D.groupby(unit)[unit].transform("size").values; un = pd.factorize(D[unit])[0]
    S = {}
    def acc(name, mask, xv, yv):
        S[name + "_n"] = np.bincount(cl[mask], minlength=nC).astype(float); S[name + "_sx"] = np.bincount(cl[mask], weights=xv[mask], minlength=nC)
        S[name + "_sy"] = np.bincount(cl[mask], weights=yv[mask], minlength=nC); S[name + "_sxx"] = np.bincount(cl[mask], weights=xv[mask] ** 2, minlength=nC)
        S[name + "_sxy"] = np.bincount(cl[mask], weights=xv[mask] * yv[mask], minlength=nC)
    ea, eb = edges                                                 # child rows, parent rows
    ecl = cl[ea]; zc, zp = z[ea], z[eb]
    sib = sibling_parents(D, unit, pos, pd.Series(z == 0, index=D.index), ea[(zc == 0) & (zp == 0)], eb[(zc == 0) & (zp == 0)])
    neg_e = (zc == 0) & (zp == 0); ea_n, eb_n = ea[neg_e], eb[neg_e]; okS = sib >= 0
    allL = ["z"] + list(labs) + list(extra)
    U = pd.DataFrame({"un": un, "cl": cl, "k": k, "neg": (z == 0).astype(float)})
    big = U.k.values >= 2
    for L in allL:
        f = z if L == "z" else D[L].values.astype(float)
        good = ~np.isnan(f); fz = np.nan_to_num(f)
        acc(f"law_{L}", good, x, fz)
        if L != "z":
            acc(f"pd_{L}", good & (z == 0), x, fz); acc(f"tp_{L}", good & (z == 1), x, fz)
        # unit-level: flag fraction (all) and among negatives, vs log2 size
        g = pd.DataFrame({"un": un, "f": fz, "neg": (z == 0) & good, "good": good}).groupby("un")
        uf = g.f.sum().values / np.maximum(g.good.sum().values, 1); uneg = pd.DataFrame({"un": un, "fn": fz * ((z == 0) & good)}).groupby("un").fn.sum().values
        nneg = g.neg.sum().values; ucl = pd.Series(cl).groupby(un).first().values; uk = pd.Series(k).groupby(un).first().values; lx = np.log2(uk)
        upos = pd.DataFrame({"un": un, "fp": fz * ((z == 1) & good)}).groupby("un").fp.sum().values; npos = pd.DataFrame({"un": un, "np": ((z == 1) & good).astype(float)}).groupby("un").np.sum().values
        mU = uk >= 2
        for nm, yv, mm in [(f"sz_{L}", uf, mU), (f"psz_{L}", uneg / np.maximum(nneg, 1), mU & (nneg > 0)), (f"tsz_{L}", upos / np.maximum(npos, 1), mU & (npos > 0))]:
            S[nm + "_n"] = np.bincount(ucl[mm], minlength=nC).astype(float); S[nm + "_sx"] = np.bincount(ucl[mm], weights=lx[mm], minlength=nC)
            S[nm + "_sy"] = np.bincount(ucl[mm], weights=yv[mm], minlength=nC); S[nm + "_sxx"] = np.bincount(ucl[mm], weights=lx[mm] ** 2, minlength=nC)
            S[nm + "_sxy"] = np.bincount(ucl[mm], weights=lx[mm] * yv[mm], minlength=nC)
        # edges: law RR (all edges), placebo RR (both negative), sibling RR
        for nm, (aa, bb) in [("rr", (ea, eb)), ("prr", (ea_n, eb_n)), ("srr", (ea_n[okS], sib[okS]))]:
            fc, fp = np.nan_to_num(f[aa]), np.nan_to_num(f[bb]); okk = ~np.isnan(f[aa]) & ~np.isnan(f[bb]); cc = cl[aa]
            for t_, w in [("p1", (fp == 1) & okk), ("p1c1", (fp == 1) & (fc == 1) & okk), ("p0", (fp == 0) & okk), ("p0c1", (fp == 0) & (fc == 1) & okk)]:
                S[f"{nm}_{L}_{t_}"] = np.bincount(cc, weights=w.astype(float), minlength=nC)
    return S, nC

def ols(S, name, w):
    n, sx, sy, sxx, sxy = (w @ S[f"{name}_{s}"] for s in ["n", "sx", "sy", "sxx", "sxy"])
    d = n * sxx - sx ** 2; return np.where(d > 0, (n * sxy - sx * sy) / np.where(d > 0, d, 1), np.nan), sy / np.maximum(n, 1e-12)
def rr(S, nm, L, w):
    p1, p1c1, p0, p0c1 = (w @ S[f"{nm}_{L}_{t}"] for t in ["p1", "p1c1", "p0", "p0c1"])
    return (p1c1 / np.maximum(p1, 1e-12)) / np.maximum(p0c1 / np.maximum(p0, 1e-12), 1e-12)

def avalle(D, labs, unit, nshuf=200, seed=96):
    """The f96 Avalle pipeline on conversations (>= 2 comments) labeled in full by every labeler: conversation fraction on log2
    size (OLS slope), Mann-Kendall tau of log2-bin means (bins of >= 30 conversations), 200-shuffle null of labels across comments."""
    r = np.random.default_rng(seed); cols = ["z"] + list(labs)
    full = D[cols].notna().all(axis=1).groupby(D[unit]).transform("all").values; k_all = D.groupby(unit)[unit].transform("size").values
    m = full & (k_all >= 2); out = {"n_conversations": 0}
    if m.sum() == 0: return out
    u = pd.factorize(D[unit].values[m])[0]; NU = u.max() + 1; k = np.bincount(u); x = np.log2(k); xc = x - x.mean(); sxx = (xc ** 2).sum(); b = np.floor(x).astype(int)
    out["n_conversations"] = int(NU); out["n_comments"] = int(m.sum())
    frac = lambda vv: np.bincount(u, weights=vv, minlength=NU) / k
    slope = lambda f: float((xc * (f - f.mean())).sum() / sxx)
    for c in cols:
        v = D[c].values[m].astype(float); f = frac(v); obs = slope(f)
        bm = pd.Series(f).groupby(b).mean(); bn = pd.Series(f).groupby(b).size(); bm = bm[bn >= 30]
        tau, p = kendalltau(bm.index.values, bm.values) if len(bm) >= 3 else (np.nan, np.nan)
        sh = np.array([slope(frac(r.permutation(v))) for _ in range(nshuf)]); mu, sd = float(sh.mean()), float(sh.std(ddof=1))
        # Avalle et al.'s binning: 21 log-spaced size bins between the smallest and largest conversation, bins of >= 30 conversations
        e21 = np.logspace(np.log2(k.min()), np.log2(k.max()) + 1e-9, 22, base=2.0); b21 = np.clip(np.searchsorted(e21, k, side="right") - 1, 0, 20)
        g21 = pd.DataFrame({"b": b21, "f": f, "lx": x}).groupby("b").agg(m=("f", "mean"), n=("f", "size"), c=("lx", "mean")); g21 = g21[g21.n >= 30]
        sl21 = float(np.polyfit(g21.c.values, g21.m.values, 1)[0]) if len(g21) >= 3 else np.nan
        tau21, p21 = kendalltau(g21.index.values, g21.m.values) if len(g21) >= 3 else (np.nan, np.nan)
        out[c] = dict(slope=obs, shuffle_mean=mu, shuffle_sd=sd, z_vs_shuffle=(obs - mu) / sd if sd > 0 else None, significant_increase=bool(obs >= mu + 2 * sd),
                      mk_tau=float(tau), mk_p=float(p), bin_means={int(i): round(float(mm), 5) for i, mm in bm.items()},
                      bins21=dict(n_bins=int(len(g21)), slope=sl21, mk_tau=float(tau21), mk_p=float(p21), bin_means={int(i): round(float(mm), 5) for i, mm in g21.m.items()}))
    return out

def matched_human(D, labs, seed=56):
    """FPR-matched human benchmark (Addendum 56): per labeler, gold positives plus the gold negatives with the highest crowd vote
    fractions up to the labeler's FPR, ties at random; saturated when that FPR exceeds the share of negatives with any toxic vote."""
    r = np.random.default_rng(seed); M, info = {}, {}
    z = D.z.values; tox = D.tox.values.astype(float)
    for L in labs:
        ok = D[L].notna().values; neg = (z == 0) & ok; f = float(D[L].values[neg].mean()); share = float((tox[neg] > 0).mean())
        info[L] = dict(fpr=f, share_any_vote=share, saturated=bool(f > share))
        if f > share: continue
        idx = np.flatnonzero(neg); n = int(round(f * len(idx))); key = tox[idx] + r.random(len(idx)) * 1e-6
        h = np.where(ok, (z == 1).astype(float), np.nan); h[idx[np.argsort(-key, kind="stable")[:n]]] = 1.0
        D[f"hm_{L}"] = h; M[L] = f"hm_{L}"
    return M, info

def analyze(name, D, labs, unit, clus, depth, pos, edges, human=None, matched=None, levels=None):
    matched = matched or {}; extra = ([human] if human else []) + list(matched.values())
    S, nC = suff(D, labs, unit, clus, depth, pos, edges, extra)
    Wb = np.vstack([np.ones(nC), rng.poisson(1.0, size=(NB, nC))])
    out = {"n_items": int(len(D)), "n_units": int(D[unit].nunique()), "n_edges": int(len(edges[0])), "gold_prev": float(D.z.mean()), "labelers": {}}
    ci = lambda v: [float(v[0]), float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))]
    gl, gm = ols(S, "law_z", Wb); gsz, gszm = ols(S, "sz_z", Wb); grr = rr(S, "rr", "z", Wb)
    out["gold"] = dict(depth_slope=ci(gl), size_slope=ci(gsz), rr=ci(grr))
    hum = {}
    for L in labs + extra:
        pd_, fpr = ols(S, f"pd_{L}", Wb); tp_, tpr = ols(S, f"tp_{L}", Wb); ll, fl = ols(S, f"law_{L}", Wb); sz, _ = ols(S, f"sz_{L}", Wb); psz, pm = ols(S, f"psz_{L}", Wb)
        tsz, _ = ols(S, f"tsz_{L}", Wb)
        prr, srr, lrr = rr(S, "prr", L, Wb), rr(S, "srr", L, Wb), rr(S, "rr", L, Wb)
        hum[L] = dict(prr=prr, pdr=pd_ / np.maximum(fpr, 1e-12), psr=psz / np.maximum(pm, 1e-12), pd=pd_, ps=psz)
        if L not in labs and L != human: continue
        p = gm; comp = dict(gold_part=ci((tpr - fpr) * gl), fpr_part=ci((1 - p) * pd_), tpr_part=ci(p * tp_))
        ps_ = gszm; comp_size = dict(gold_part=ci((tpr - fpr) * gsz), fpr_part=ci((1 - ps_) * psz), tpr_part=ci(ps_ * tsz),
                                     sum_of_parts=ci((tpr - fpr) * gsz + (1 - ps_) * psz + ps_ * tsz), law=ci(sz))
        r = dict(flag_rate=ci(fl), fpr=ci(fpr), tpr=ci(tpr), placebo_depth_slope=ci(pd_), placebo_depth_rel=ci(pd_ / np.maximum(fpr, 1e-12)),
                 placebo_size_slope=ci(psz), placebo_size_rel=ci(psz / np.maximum(pm, 1e-12)), placebo_rr=ci(prr), sibling_rr=ci(srr),
                 law_depth_slope=ci(ll), law_depth_minus_gold=ci(ll - gl), law_size_slope=ci(sz), law_size_ratio_to_gold=ci(sz / np.where(np.abs(gsz) > 1e-12, gsz, np.nan)),
                 law_rr=ci(lrr), decomposition=comp, decomposition_size=comp_size, positives_size_slope=ci(tsz))
        out["labelers"][L] = r
    if human:
        H = hum[human]
        for L in labs:
            out["labelers"][L]["vs_human"] = {k: ci(hum[L][k] - H[k]) for k in ["prr", "pdr", "psr", "pd", "ps"]}
    for L, hm in matched.items():
        out["labelers"][L]["vs_human_matched"] = {k: ci(hum[L][k] - hum[hm][k]) for k in ["prr", "pdr", "psr", "pd", "ps"]}
        out["labelers"][L]["matched_benchmark"] = {k: ci(hum[hm][k]) for k in ["prr", "pdr", "psr", "pd", "ps"]}
    out["avalle"] = {lv: avalle(D, labs, u_) for lv, u_ in (levels or {"units": unit}).items()}
    print(name, json.dumps({L: (round(v["placebo_rr"][0], 3), round(v["placebo_depth_rel"][0], 3), round(v["placebo_size_rel"][0], 3)) for L, v in out["labelers"].items()}), flush=True)
    return out

def edges_from(D, parent_col, key=None):
    idx = pd.Series(np.arange(len(D)), index=(D[key].values if key else D.index))
    par = D[parent_col]; m = par.notna() & par.isin(idx.index)
    ch, pa = np.flatnonzero(m.values), idx.reindex(par[m].values).values.astype(int)
    keep = pa != ch                                   # Addendum 72: drop self-replies (CGA-Wiki: 84 utterances reply to themselves)
    return ch[keep], pa[keep]

RES = {}
# ---------------------------------------------------------------- Civil Comments threads (all) and the 5% sample
T = pd.read_parquet(f"{W}/results/f10_population_thread_items.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "tox", "parent_id", "created_date"]).set_index("id")
T["tox"] = meta.tox.reindex(T.index).values; T["parent"] = meta.parent_id.reindex(T.index).values; T["z"] = T.Z
T["human_lenient"] = np.where(T.z == 1, 1.0, (T.tox >= 0.3).astype(float))
labs = ["haiku", "haiku_ctx", "qwen7b_m", "qwen7b_m_ctx", "qwen7b"]
O = open_scores([f"cc_{i:03d}" for i in range(80)]); labs += add_open(T, O, float(T.z.mean()))
T["pos"] = meta.created_date.reindex(T.index).values.astype("datetime64[s]").astype(np.int64)
E = edges_from(T, "parent")
labs = [l for l in labs if l in T]; MH, MI = matched_human(T, labs)
RES["cc_threads"] = analyze("cc_threads", T, labs, "root", "article_id", "depth", "pos", E, human="human_lenient", matched=MH,
                            levels={"threads": "root", "articles": "article_id"}); RES["cc_threads"]["matched_info"] = MI
ids5 = set(np.loadtxt(f"{W}/results/second/s5_ids.txt", dtype=np.int64).tolist())
T5 = T[T.index.isin(ids5)].copy()
G0 = {}; LD = f"{W}/results/g27"
def gload(pattern):
    R = {}
    for f in sorted(glob.glob(f"{LD}/**/{pattern}", recursive=True)):
        for l in open(f):
            r = json.loads(l)
            if r.get("p") is not None: R[r["id"]] = r["p"][0] + r["p"][1]
    return pd.Series(R)
g0, g1 = gload("shard_*_tox_ctx0.jsonl"), gload("shard_*_tox_ctx1.jsonl")
if len(g0):
    s0 = g0.reindex(T5.index); t = float(np.nanquantile(s0, 1 - T5.z.mean())); T5["gemma_m"] = (s0 > t).astype(float).where(s0.notna())
    c = g1.reindex(T5.index); T5["gemma_m_ctx"] = np.where(T5.parent.notna(), (c > t).astype(float).where(c.notna()).values, T5.gemma_m.values)
for ctx in (0, 1):
    fs = sorted(glob.glob(f"{W}/results/second/s5/sonnet_tox_ctx{ctx}/shard_*.jsonl"))
    if fs:
        L_ = letters_tox(fs).reindex(T5.index); T5[f"sonnet{'_ctx' if ctx else ''}"] = (L_ >= 2).astype(float).where(L_.notna())
if "sonnet_ctx" in T5: T5["sonnet_ctx"] = np.where(T5.parent.notna(), T5.sonnet_ctx.values, T5.get("sonnet", pd.Series(np.nan, index=T5.index)).values)
labs5 = [l for l in ["haiku", "haiku_ctx", "qwen7b_m", "qwen7b_m_ctx", "gemma_m", "gemma_m_ctx", "sonnet", "sonnet_ctx"] + [l for l in labs if "@" in l] if l in T5 and T5[l].notna().mean() >= 0.95]
print("s5 labelers", labs5, {l: round(float(T5[l].notna().mean()), 3) for l in ["gemma_m", "gemma_m_ctx", "sonnet", "sonnet_ctx"] if l in T5}, flush=True)
ok = T5[labs5].notna().all(axis=1).groupby(T5.root).transform("all"); T5 = T5[ok].copy()
E5 = edges_from(T5, "parent")
T5 = T5.drop(columns=[c for c in T5.columns if c.startswith("hm_")]); MH5, MI5 = matched_human(T5, labs5)
RES["cc_threads_s5"] = analyze("cc_threads_s5", T5, labs5, "root", "article_id", "depth", "pos", E5, human="human_lenient", matched=MH5,
                               levels={"threads": "root", "articles": "article_id"}); RES["cc_threads_s5"]["matched_info"] = MI5
# ---------------------------------------------------------------- CAD
g = pd.read_parquet(f"{W}/data/cad_items.parquet").set_index("id"); g["z"] = g.Z
def rd(f):
    R = pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates("id", keep="last").set_index("id"); return (R.ans == "A").astype(float)
g["haiku"] = rd(f"{W}/results/cad/haiku_cad_ctx0.jsonl").reindex(g.index)
c1 = rd(f"{W}/results/cad/haiku_cad_ctx1.jsonl").reindex(g.index); g["haiku_ctx"] = c1.where(c1.notna(), g.haiku)
g["sonnet"] = rd(f"{W}/results/cad/sonnet_cad_ctx0.jsonl").reindex(g.index)
labc = ["haiku", "haiku_ctx", "sonnet"] + add_open(g, open_scores(["cad"]), float(g.z.mean()))
g = g[g[["haiku", "sonnet"]].notna().all(axis=1)].copy(); g["pos"] = pd.to_numeric(g.created, errors="coerce").fillna(0).astype(np.int64)
Ec = edges_from(g, "parent_iid", key="info_id")
RES["cad_threads"] = analyze("cad_threads", g, [l for l in labc if l in g], "thread", "thread", "depth", "pos", Ec)
# ---------------------------------------------------------------- Qian Reddit and Gab
M = pd.read_parquet(f"{W}/data/qian/qian_meta.parquet").set_index("id")
def qlab(d):
    R = []
    for f in sorted(glob.glob(f"{d}/shard_*.jsonl")):
        for l in open(f):
            try: R.append(json.loads(l))
            except Exception: pass
    R = pd.DataFrame(R).drop_duplicates("id", keep="last").set_index("id"); return (R.get("ans") == "A").astype(float)
a0 = qlab(f"{W}/results/second/qian/haiku_hate_ctx0"); a1 = qlab(f"{W}/results/second/qian/haiku_hate_ctx1")
M["haiku"] = a0.reindex(M.index).fillna(0.0).values; rep = M.parent >= 0; has1 = M.index.isin(a1.index)
M["haiku_ctx"] = np.where(rep & has1, a1.reindex(M.index).fillna(0.0).values, M.haiku.values)
q0, q1 = gload("shard_*_hate_ctx0.jsonl"), gload("shard_*_hate_ctx1.jsonl")
for plat in ["reddit", "gab"]:
    Q = M[M.platform == plat].copy(); Q["parent"] = Q.parent.where(Q.parent >= 0); labq = ["haiku", "haiku_ctx"]
    if len(q0):
        # hate task: Gemma P(A) (option probabilities saved as p; gload sums p[0]+p[1], so reload P(A) only)
        R = {}
        for f in sorted(glob.glob(f"{LD}/**/shard_*_hate_ctx0.jsonl", recursive=True)):
            for l in open(f):
                r = json.loads(l)
                if r.get("p") is not None: R[r["id"]] = r["p"][0]
        sg = pd.Series(R).reindex(Q.index)
        if sg.notna().mean() > 0.5:
            t = float(np.nanquantile(sg, 1 - Q.z.mean())); Q["gemma_m"] = (sg > t).astype(float).where(sg.notna()); labq.append("gemma_m")
    labq += add_open(Q, open_scores([f"qian_{plat}"]), float(Q.z.mean()))
    Eq = edges_from(Q, "parent")
    RES[f"qian_{plat}"] = analyze(f"qian_{plat}", Q, [l for l in labq if l in Q], "conv", "conv", "depth", "pos", Eq)
# ---------------------------------------------------------------- CGA-Wiki (verified conversations)
C = pd.read_parquet(f"{W}/data/cga_meta.parquet").set_index("id"); C["z"] = C.pa
def cl_(d):
    fs = sorted(glob.glob(f"{d}/shard_*.jsonl"))
    R = pd.DataFrame([json.loads(l) for f in fs for l in open(f)])
    if "error" in R: R = R[R["error"].isna()]
    return (R.drop_duplicates("id", keep="last").set_index("id").ans == "A").astype(float)
C["haiku"] = cl_(f"{W}/results/cga/haiku_pa_ctx0").reindex(C.index); h1 = cl_(f"{W}/results/cga/haiku_pa_ctx1").reindex(C.index)
C["haiku_ctx"] = h1.where(h1.notna(), C.haiku); C.loc[C.parent_uid.notna() & h1.isna(), "haiku_ctx"] = np.nan
C = C[C.verified.eq(1) & C.pair_verified.eq(1)].copy()
labg = ["haiku", "haiku_ctx"]
qf0, qf1 = sorted(glob.glob(f"{W}/results/cga/qwen7b/*_ctx0.jsonl")), sorted(glob.glob(f"{W}/results/cga/qwen7b/*_ctx1.jsonl"))
if qf0:                                                            # as in f71: option probability, threshold matched to gold prevalence on the verified set
    qrd = lambda F: pd.Series({json.loads(l)["id"]: json.loads(l)["p"][0] for f in F for l in open(f)})
    sp = qrd(qf0).reindex(C.index); t = float(np.nanquantile(sp, 1 - C.z.mean())); C["qwen7b_m"] = (sp > t).astype(float).where(sp.notna()); labg.append("qwen7b_m")
    if qf1:
        cq = qrd(qf1).reindex(C.index); C["qwen7b_m_ctx"] = (cq > t).astype(float).where(cq.notna(), C.qwen7b_m); labg.append("qwen7b_m_ctx")
labg += add_open(C, open_scores(["cga"]), float(C.z.mean()))
okc = C[["haiku", "haiku_ctx"]].notna().all(axis=1).groupby(C.conv).transform("all"); C = C[okc].copy(); C["depth"] = C.pos - C.pos.groupby(C.conv).transform("min")
Eg = edges_from(C, "parent_uid", key="uid")
RES["cga"] = analyze("cga", C, [l for l in labg if l in C], "conv", "conv", "depth", "pos", Eg)
# ---------------------------------------------------------------- BAD (bot and human turns; parent = previous turn)
B = pd.read_parquet(f"{W}/data/bad_meta.parquet").set_index("id"); B["z"] = B.unsafe.astype(int)
Rb = pd.concat([pd.DataFrame([json.loads(l) for l in open(f)]) for f in [f"{W}/results/bad/haiku_unsafe_ctx1.jsonl", f"{W}/results/bad/haiku_unsafe_ctx0_first.jsonl"]])
if "error" in Rb: Rb = Rb[Rb["error"].isna()]
sb = Rb.drop_duplicates("id", keep="last").set_index("id").ans.map({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}).reindex(B.index)
B["haiku_ctx"] = (sb >= 2 / 3).astype(float).where(sb.notna())
labb = ["haiku_ctx"] + add_open(B, open_scores(["bad"]), float(B.z.mean()))
B["depth"] = B.pos - 1; B = B.sort_values(["dlg", "pos"]); B["prev"] = B.groupby("dlg").apply(lambda x: pd.Series(x.index.values, index=x.index).shift(1)).reset_index(level=0, drop=True).reindex(B.index)
Eb = edges_from(B, "prev")
RES["bad"] = analyze("bad", B, [l for l in labb if l in B], "dlg", "dlg", "depth", "pos", Eb)
json.dump(RES, open(OUT, "w"), indent=1); print("saved", OUT)
