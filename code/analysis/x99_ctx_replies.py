"""v7h / arXiv: contagion of the LLMs shown the parent, over reply edges between replies (no new labels).
A labeler shown the parent labels a reply with its parent and a root alone, so a reply risk ratio over all edges mixes parents labeled
in two modes (roots without context, replies with it). Here every LLM shown the parent is measured over the reply edges whose parent is
itself a reply, so that both ends of every edge are labeled with their parent shown: the reply risk ratio and its log-RR distortion
against gold on the same edges (500 Poisson article-cluster resamples, seed 9901, paired), and the split of the log-RR distortion into
Proposition 1's three terms by permuting the flags within gold positives and within gold negatives among replies (50 permutations,
seed 9902), as x75_decomp.py does over all edges. The same LLM without the parent is measured on the same edges for comparison.
Flags: Haiku and Sonnet as stored; Qwen and Gemma shown the parent at their own gold-prevalence threshold (seed 7650, as
x76_rethreshold.py and x97_decomp_replies.py draw them); OLMo-2, Granite and Llama as x84_families.py builds them (seed 8400).
Writes results/x99/x99_ctx_replies.json.
usage: python3 x99_ctx_replies.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, os, glob
import numpy as np, pandas as pd
W = _ROOT; OUT = f"{W}/results/x99/x99_ctx_replies.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
NB, NP = 500, 50
res = {}


def own_threshold_flags(P):
    """Qwen/Gemma shown the parent at their own gold-prevalence threshold, exactly as x97_decomp_replies.py (seed 7650)."""
    z = P.z.values.astype(float); N = len(z); prev = z.mean(); out = {}
    rng = np.random.default_rng(7650)
    for m in [c[2:] for c in P.columns if c.startswith("s:") and c.endswith("_m_ctx")]:
        s = P[f"s:{m}"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(prev * N))
        order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0; out[f"{m}@own"] = f
    return out


def family_flags(P, model):
    """OLMo-2, Granite, Llama with and without the parent, exactly as x84_families.py (seed 8400)."""
    ids = P.id.astype(np.int64).values; N = len(P); pr = P.parent_row.values; z = P.z.values.astype(float); k = int(round(z.mean() * N))
    def scores(ctx):
        s = {}
        for f in sorted(glob.glob(f"{W}/results/b84/{model}/shard_*_tox_ctx{ctx}.jsonl")):
            for l in open(f):
                r = json.loads(l); s[int(r["id"])] = float(r["p"][0] + r["p"][1])
        return s
    s0 = scores(0); s1 = scores(1)
    a0 = np.array([s0.get(int(i), np.nan) for i in ids])
    a1 = np.array([s1.get(int(i), np.nan) if pr[j] >= 0 and int(i) in s1 else s0.get(int(i), np.nan) for j, i in enumerate(ids)])
    out = {}
    for tag, sc in ((model, a0), (f"{model}_ctx", a1)):
        sc = np.where(np.isnan(sc), -np.inf, sc)
        order = np.lexsort((np.random.default_rng(8400).random(N), -sc)); y = np.zeros(N); y[order[:k]] = 1.0; out[tag] = y
    return out


for pop, pairs in (("cc_threads", (("haiku", "haiku_ctx"), ("qwen7b_m", "qwen7b_m_ctx@own"))),
                   ("cc_s5", (("sonnet", "sonnet_ctx"), ("gemma_m", "gemma_m_ctx@own"), ("olmo2", "olmo2_ctx"), ("granite", "granite_ctx"), ("llama31", "llama31_ctx")))):
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet"); z = P.z.values.astype(float); N = len(z)
    depth = P.depth.values; pr = P.parent_row.values
    ch = np.flatnonzero(pr >= 0); pa = pr[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
    rep = depth[pa] >= 1                                                       # edges whose parent is a reply
    cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); ecl = cl[ch]
    Wb = np.vstack([np.ones(nC), np.random.default_rng(9901).poisson(1.0, size=(NB, nC))])
    flags = {}
    for c in P.columns:
        if c.startswith("y:") and c[2:] in ("haiku", "haiku_ctx", "sonnet", "sonnet_ctx", "qwen7b_m", "gemma_m"):
            flags[c[2:]] = np.nan_to_num(P[c].values.astype(float))
    flags.update(own_threshold_flags(P))
    if pop == "cc_s5":
        for model in ("olmo2", "granite", "llama31"):
            flags.update(family_flags(P, model))

    def lrr_boot(v, emask):
        vp, vc, c_ = v[pa][emask], v[ch][emask], ecl[emask]
        a, b, c, d = (Wb @ np.bincount(c_, t, nC) for t in (vp * vc, vp, (1 - vp) * vc, 1 - vp))
        return np.log((a / b) / (c / d))

    def lrr(v, emask):
        vp, vc = v[pa][emask], v[ch][emask]; a = (vp * vc).sum(); b = vp.sum(); c = ((1 - vp) * vc).sum(); d = (1 - vp).sum()
        return float(np.log((a / b) / (c / d))) if min(a, b, c, d) > 0 else np.nan

    ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    g_rep = lrr_boot(z, rep); g_all = lrr_boot(z, np.ones(len(ch), bool))
    out = {"n_edges_all": int(len(ch)), "n_edges_between_replies": int(rep.sum()),
           "gold": {"rr_between_replies": ci(np.exp(g_rep)), "rr_all_edges": ci(np.exp(g_all))}, "labelers": {}}
    reply_rows = np.flatnonzero(depth >= 1); zr = z[reply_rows]
    pos, neg = reply_rows[zr == 1], reply_rows[zr == 0]
    rng = np.random.default_rng(9902); lz = lrr(z, rep)
    for base, ctx in pairs:
        for L in (base, ctx):
            y = flags[L]; lb = lrr_boot(y, rep)
            pp, ap, pa_ = [], [], []
            for _ in range(NP):
                yp = y.copy(); yp[pos] = rng.permutation(y[pos]); yp[neg] = rng.permutation(y[neg])
                yap = y.copy(); yap[pos] = yp[pos]; ypa = y.copy(); ypa[neg] = yp[neg]
                pp.append(lrr(yp, rep)); ap.append(lrr(yap, rep)); pa_.append(lrr(ypa, rep))
            epp, eap, epa, ly = np.nanmean(pp), np.nanmean(ap), np.nanmean(pa_), lrr(y, rep)
            uni = epp - lz; fp = 0.5 * ((eap - epp) + (ly - epa)); tp = 0.5 * ((epa - epp) + (ly - eap))
            assert abs(uni + fp + tp - (ly - lz)) < 1e-9
            out["labelers"][L] = dict(rr_between_replies=ci(np.exp(lb)), log_rr_distortion=ci(lb - g_rep), uniform=float(uni), fp=float(fp), tp=float(tp),
                                      rr_all_edges=float(np.exp(lrr(y, np.ones(len(ch), bool)))), flag_rate=float(y.mean()))
            print(pop, L, "RR between replies", [round(t, 3) for t in out["labelers"][L]["rr_between_replies"]],
                  "dist", [round(t, 3) for t in out["labelers"][L]["log_rr_distortion"]], "terms", round(uni, 3), round(fp, 3), round(tp, 3),
                  "all-edge RR", round(out["labelers"][L]["rr_all_edges"], 3), flush=True)
    print(pop, "gold RR between replies", [round(t, 3) for t in out["gold"]["rr_between_replies"]], "edges", out["n_edges_between_replies"], "of", out["n_edges_all"])
    res[pop] = out
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
