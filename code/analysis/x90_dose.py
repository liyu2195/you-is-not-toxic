"""Addendum 90 (exploratory, descriptive): the dose of second-person address.
The number of second-person words of a comment (the 'you' detector of Appendix A, counted: you, your, yours, yourself, yourselves and
the contractions you're, youre, you've, you'd, you'll, as whole words) and of its parent (counted on the shard's parent_text).
For each corpus and labeler, among gold negatives (with the parent shown: gold-negative replies, depth >= 1):
  - the false-positive rate by number (0, 1, 2, 3, 4+);
  - the placebo depth slope (points per level, depth capped at 10): overall, pooled within the presence of 'you' (as B89a), pooled within
    the number (0..4+), and with the parent shown also pooled within the reply's number x the parent's number (0..4+ x 0..3+).
Labelers. Civil Comments threads: Haiku, Haiku shown the parent, Qwen2.5-7B shown the parent at gold prevalence (as x76_rethreshold.py,
seed 7650), Detoxify-original at gold prevalence. The 5% sample: Sonnet and Sonnet shown the parent. CAD, Qian et al.'s Reddit and Gab,
CGA-Wiki, BAD: Haiku and Sonnet without and with the parent (B89a labels; replies without parent text keep the no-context label) and
Detoxify-original at gold prevalence.
Also, the mean number by depth (gold negatives); on the 5% sample the rewrite's change in flags of gold negatives by number (B77d Haiku,
B89c (i) Sonnet); and Figure 2's data: Haiku's false-positive rate by depth on the threads, raw and standardized to the root's
distribution of the number (direct standardization over 0..4+). 500 cluster resamples (seed 9001; the populations' cluster column).
usage: python3 x90_dose.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; NB = 500
FULL = (_ROOT + "/scratch/full")
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
TXT = {"cc_threads": FULL, "cc_s5": f"{W}/data/second/s5"}
CORP = ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]
ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
flag = lambda a: float(a in ("A", "B"))


def load_labels(corpus, model, ctx):
    out = {}
    for f in sorted(glob.glob(f"{W}/results/x89/labels/{corpus}_{model}_ctx{ctx}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); out[int(r["id"])] = r.get("ans")
    return out


res = {"corpora": {}}
for corpus in CORP:
    P = pd.read_parquet(f"{W}/results/x14_pops/{corpus}.parquet"); ids = P.id.astype(np.int64).values; N = len(P)
    z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
    need = set(ids.tolist()); cnt_d = {}; pcnt_d = {}; nw_d = {}
    for f in sorted(glob.glob(f"{TXT.get(corpus, W + '/results/x89/shards/' + corpus)}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"])
            if i in need:
                t_ = r.get("text") or ""; cnt_d[i] = len(YOU.findall(t_)); pt = r.get("parent_text"); pcnt_d[i] = len(YOU.findall(pt)) if pt else 0
                nw_d[i] = len(t_.split())
    cnt = np.array([cnt_d.get(int(i), 0) for i in ids]); pcnt = np.array([pcnt_d.get(int(i), 0) for i in ids]); nw = np.array([nw_d.get(int(i), 0) for i in ids])
    cb = np.minimum(cnt, 4); pcb = np.minimum(pcnt, 3); neg = z == 0; rep = x >= 1
    Wb = np.vstack([np.ones(nC), np.random.default_rng(9001).poisson(1.0, size=(NB, nC))])
    S = lambda v: Wb @ np.bincount(cl, v, nC)
    def rate(v, mask):
        with np.errstate(divide="ignore", invalid="ignore"):
            return S(v * mask) / S(mask.astype(float))
    def slope(v, mask):
        m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
        return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
    def pooled(v, mask, strata):                                       # pooled within-stratum OLS slope (stratum fixed effects)
        num = 0.0; den = 0.0
        for s_ in np.unique(strata[mask]):
            m_ = (mask & (strata == s_)).astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
            ok = n_ > 0; nn = np.where(ok, n_, 1.0)
            num = num + np.where(ok, sxy - sx * sy / nn, 0.0); den = den + np.where(ok, sxx - sx ** 2 / nn, 0.0)
        return 100 * num / den
    def stats(y, ctx):
        M = neg & rep if ctx else neg
        o = dict(n=int(M.sum()), n_by_count=[int((M & (cb == k)).sum()) for k in range(5)],
                 fpr_by_count=[ci(rate(y, M & (cb == k))) for k in range(5)], placebo=ci(slope(y, M)),
                 within_any=ci(pooled(y, M, (cnt > 0).astype(int))), within_count=ci(pooled(y, M, cb)))
        if not ctx:                                                     # the depth law's excess over gold, overall and at a fixed number
            allm = np.ones(N, bool)
            o["law_distortion"] = ci(slope(y, allm) - slope(z, allm)); o["law_distortion_within_count"] = ci(pooled(y, allm, cb) - pooled(z, allm, cb))
        if ctx:
            o["within_any2"] = ci(pooled(y, M, (cnt > 0).astype(int) * 2 + (pcnt > 0).astype(int))); o["within_count2"] = ci(pooled(y, M, cb * 4 + pcb))
        p0 = o["placebo"][0]
        o["share_any"] = float(1 - o["within_any"][0] / p0) if p0 > 0 else None; o["share_count"] = float(1 - o["within_count"][0] / p0) if p0 > 0 else None
        if ctx and p0 > 0:
            o["share_count2"] = float(1 - o["within_count2"][0] / p0)
        return o
    R = {"N": N, "n_neg": int(neg.sum()), "mean_count_by_depth": [float(cnt[neg & (np.minimum(x, 8) == d)].mean()) if (neg & (np.minimum(x, 8) == d)).any() else None for d in range(9)],
         "n_neg_by_depth": [int((neg & (np.minimum(x, 8) == d)).sum()) for d in range(9)],
         "you_share_by_depth": [float((cnt[neg & (np.minimum(x, 8) == d)] > 0).mean()) if (neg & (np.minimum(x, 8) == d)).any() else None for d in range(9)], "share_count_neg": [float((cb[neg] == k).mean()) for k in range(5)], "labelers": {}}
    L = R["labelers"]
    if corpus == "cc_threads":
        L["haiku"] = stats(np.nan_to_num(P["y:haiku"].values.astype(float)), False)
        L["haiku_ctx"] = stats(np.nan_to_num(P["y:haiku_ctx"].values.astype(float)), True)
        sq = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["s:qwen7b_m_ctx"])["s:qwen7b_m_ctx"].values.astype(float)
        sq = np.where(np.isnan(sq), -np.inf, sq); kq = int(round(z.mean() * N)); oq = np.lexsort((np.random.default_rng(7650).random(N), -sq))
        fq = np.zeros(N); fq[oq[:kq]] = 1.0; L["qwen_ctx"] = stats(fq, True)
        L["detoxify"] = stats(np.nan_to_num(P["y:detoxify_original@m"].values.astype(float)), False)
        # Figure 2: Haiku's false-positive rate by depth, raw and standardized to the root's distribution of the number
        y = np.nan_to_num(P["y:haiku"].values.astype(float)); d8 = np.minimum(x, 8); w0 = np.array([(neg & (d8 == 0) & (cb == k)).sum() for k in range(5)], float); w0 /= w0.sum()
        raw = [ci(rate(y, neg & (d8 == d))) for d in range(9)]
        std = [ci(sum(w0[k] * rate(y, neg & (d8 == d) & (cb == k)) for k in range(5))) for d in range(9)]
        R["figure"] = dict(depth=list(range(9)), raw=raw, standardized=std, root_dist=w0.tolist())
        # not a proxy for borderline votes or length: gold negatives with no toxicity vote and no subtype vote (as B85c'), and length bands
        F = pd.read_parquet(f"{W}/data/cc_attr_frac.parquet").set_index("id").reindex(ids)
        none = F.index.notna() & np.asarray(P.id.isin(pd.read_parquet(f"{W}/data/cc_attr_frac.parquet", columns=["id"]).id).values) & (P.tox.values == 0) & \
            np.all([np.nan_to_num(F[k].values.astype(float), nan=1.0) == 0 for k in ("insult", "identity_attack", "obscene", "threat", "severe_toxicity", "sexual_explicit")], axis=0)
        R["no_votes"] = dict(share_of_neg=float(none[neg].mean()), haiku=[ci(rate(y, neg & none & (cb == k))) for k in range(5)],
                             detoxify=[ci(rate(np.nan_to_num(P["y:detoxify_original@m"].values.astype(float)), neg & none & (cb == k))) for k in range(5)])
        # a single human rater (one of the comment's raters at random) judged against the majority of the others, exactly (as the
        # paper's single-rater benchmark): its false-positive rate by number = sum P(vote 1, others' majority 0) / sum P(others' majority 0)
        meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
        nr = meta.toxicity_annotator_count.reindex(ids).values.astype(float); kr = np.clip(np.rint(P.tox.values * nr), 0, nr); okr = nr >= 2
        p1 = np.where(okr, kr / np.maximum(nr, 1), 0.0); n1 = np.maximum(nr - 1, 1)
        fp1 = np.where(okr, p1 * ((kr - 1) / n1 < 0.5), 0.0); ng1 = np.where(okr, p1 * ((kr - 1) / n1 < 0.5) + (1 - p1) * (kr / n1 < 0.5), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            R["single_rater"] = dict(fpr_by_count=[ci(S(fp1 * (cb == k)) / S(ng1 * (cb == k))) for k in range(5)])
        lb = np.digitize(nw, [20, 40, 80, 160])
        R["length_bands"] = dict(edges=[20, 40, 80, 160], haiku=[[float(y[neg & (lb == b) & (cb == k)].mean()) if (neg & (lb == b) & (cb == k)).sum() >= 200 else None for k in range(5)] for b in range(5)],
                                 n=[[int((neg & (lb == b) & (cb == k)).sum()) for k in range(5)] for b in range(5)])
    elif corpus == "cc_s5":
        L["sonnet"] = stats(np.nan_to_num(P["y:sonnet"].values.astype(float)), False)
        L["sonnet_ctx"] = stats(np.nan_to_num(P["y:sonnet_ctx"].values.astype(float)), True)
        # the open LLMs without the parent, at gold prevalence: Qwen2.5-7B and Gemma 3 (stored), OLMo-2 and Granite (B84 scores, flags as x84_families.py)
        L["qwen"] = stats(np.nan_to_num(P["y:qwen7b_m"].values.astype(float)), False); L["gemma"] = stats(np.nan_to_num(P["y:gemma_m"].values.astype(float)), False)
        kk = int(round(z.mean() * N))
        for model in ("olmo2", "granite", "llama31"):
            sc = {}
            for f in sorted(glob.glob(f"{W}/results/b84/{model}/shard_*_tox_ctx0.jsonl")):
                for l in open(f):
                    r = json.loads(l); sc[int(r["id"])] = float(r["p"][0] + r["p"][1])
            if len(sc) < N:
                continue
            a0 = np.array([sc.get(int(i), -np.inf) for i in ids]); order = np.lexsort((np.random.default_rng(8400).random(N), -a0)); yy = np.zeros(N); yy[order[:kk]] = 1.0
            L[model] = stats(yy, False)
        # at one flag rate: every labeler flagging as many comments as gold calls toxic (Haiku and Sonnet through their 0-100 ratings,
        # ties at random, seed 9010; the others as stored or above); false-positive rate by number and its ratio 4+ : 0
        def rating_flags(model):
            d = {}
            for l in open(f"{W}/results/x75_rating/{model}_ctx0.jsonl"):
                r = json.loads(l); v = r.get("rating"); d[int(r["id"])] = float(v) if v is not None else -np.inf
            a = np.array([d.get(int(i), -np.inf) for i in ids]); o_ = np.lexsort((np.random.default_rng(9010).random(N), -a)); yy = np.zeros(N); yy[o_[:kk]] = 1.0
            return yy
        MT = {"haiku": rating_flags("haiku"), "sonnet": rating_flags("sonnet"), "qwen": np.nan_to_num(P["y:qwen7b_m"].values.astype(float)), "gemma": np.nan_to_num(P["y:gemma_m"].values.astype(float))}
        for model in ("olmo2", "granite", "llama31"):
            if model in L:
                sc = {}
                for f in sorted(glob.glob(f"{W}/results/b84/{model}/shard_*_tox_ctx0.jsonl")):
                    for l in open(f):
                        r = json.loads(l); sc[int(r["id"])] = float(r["p"][0] + r["p"][1])
                a0 = np.array([sc.get(int(i), -np.inf) for i in ids]); order = np.lexsort((np.random.default_rng(8400).random(N), -a0)); yy = np.zeros(N); yy[order[:kk]] = 1.0
                MT[model] = yy
        CLS = ["detoxify_original@m", "detoxify_unbiased@m", "imsypp@m", "sn_rob@m", "mh_dist@m", "fb_hate@m", "cn_off@m"]
        for c in CLS:
            MT[c] = np.nan_to_num(P[f"y:{c}"].values.astype(float))
        R["matched"] = {}
        for k_, yy in MT.items():
            fr_ = [ci(rate(yy, neg & (cb == j))) for j in range(5)]
            R["matched"][k_] = dict(kind="classifier" if k_ in CLS else "llm", flag_rate=float(yy.mean()), fpr_by_count=fr_,
                                    ratio_4_0=ci(rate(yy, neg & (cb == 4)) / rate(yy, neg & (cb == 0))), diff_4_0=ci(rate(yy, neg & (cb == 4)) - rate(yy, neg & (cb == 0))))
        # the rewrite's change in flags of gold negatives by number (B77d Haiku; B89c (i) Sonnet)
        H = {}
        for l in open(f"{W}/results/x77_youfree/haiku_tox/shard_000.jsonl"):
            r = json.loads(l); H[int(r["id"])] = r.get("ans")
        A = {}
        for f in sorted(glob.glob(f"{W}/results/x89/rewrite/out_s/req_*.out.jsonl")):
            for l in open(f):
                r = json.loads(l); A[r["custom_id"]] = r["ans"]
        R["rewrite_by_count"] = {}
        for tag, get in (("haiku", lambda i: (H.get(10 * i), H.get(10 * i + 1))), ("sonnet", lambda i: (A.get(f"o_{i}"), A.get(f"e_{i}")))):
            yo = np.zeros(N); ye = np.zeros(N); have = np.zeros(N, bool)
            for j in np.flatnonzero(neg & (cnt > 0)):
                a, b = get(int(ids[j]))
                if a is not None or b is not None:
                    yo[j], ye[j], have[j] = flag(a), flag(b), True
            R["rewrite_by_count"][tag] = {str(k): dict(n=int((have & (cb == k)).sum()), orig=ci(rate(yo, have & (cb == k))), edit=ci(rate(ye, have & (cb == k))),
                                                     change=ci(rate(ye - yo, have & (cb == k)))) for k in range(1, 5)}
    else:
        for model in ("haiku", "sonnet"):
            L0 = load_labels(corpus, model, 0); L1 = load_labels(corpus, model, 1)
            if not L0:
                continue
            L[model] = stats(np.array([flag(L0.get(int(i))) for i in ids]), False)
            if L1:
                L[f"{model}_ctx"] = stats(np.array([flag(L1.get(int(i)) if int(i) in L1 else L0.get(int(i))) for i in ids]), True)
        L["detoxify"] = stats(np.nan_to_num(P["y:detoxify_original@m"].values.astype(float)), False)
    res["corpora"][corpus] = R
    print(corpus, "N", N, "mean count by depth", [round(v, 2) if v is not None else None for v in R["mean_count_by_depth"]], flush=True)
    for k, v in L.items():
        print(f"  {k:12s} FPR by count", [round(100 * t[0], 1) for t in v["fpr_by_count"]], "placebo", [round(t, 2) for t in v["placebo"]],
              "within any", round(v["within_any"][0], 2), "within count", [round(t, 2) for t in v["within_count"]],
              ("within count2 " + str([round(t, 2) for t in v["within_count2"]])) if "within_count2" in v else "",
              ("law distortion %+.2f -> %+.2f" % (v["law_distortion"][0], v["law_distortion_within_count"][0])) if "law_distortion" in v else "", flush=True)
    if "matched" in R:
        for k_, v in R["matched"].items():
            print("  matched", f"{k_:22s}", round(100 * v["flag_rate"], 1), [round(100 * t[0], 1) for t in v["fpr_by_count"]], "ratio", [round(t, 2) for t in v["ratio_4_0"]], flush=True)
    if "rewrite_by_count" in R:
        for tag, v in R["rewrite_by_count"].items():
            print("  rewrite", tag, {k: (w["n"], round(100 * w["orig"][0], 1), round(100 * w["edit"][0], 1), [round(100 * t, 1) for t in w["change"]]) for k, w in v.items()}, flush=True)
    if "figure" in R:
        print("  figure raw", [round(100 * t[0], 1) for t in R["figure"]["raw"]], "std", [round(100 * t[0], 1) for t in R["figure"]["standardized"]], flush=True)
        print("  no votes: share", round(R["no_votes"]["share_of_neg"], 3), "haiku", [round(100 * t[0], 1) for t in R["no_votes"]["haiku"]], "detoxify", [round(100 * t[0], 1) for t in R["no_votes"]["detoxify"]])
        print("  single rater", [round(100 * t[0], 2) for t in R["single_rater"]["fpr_by_count"]])
        print("  length bands", [[round(100 * v, 1) if v is not None else None for v in b] for b in R["length_bands"]["haiku"]], flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
