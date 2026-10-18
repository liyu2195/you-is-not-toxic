"""Addendum 82, B82b-d (descriptive, from stored results).
B82b: the structured false-positive term S_FP of the pooled depth law (results/x74_decomp) and of the within-thread law
      (results/x75_within) for every LLM setting on CAD, Qian Reddit, Qian Gab, CGA-Wiki and BAD; and the share of civil comments
      (gold-negative) that address someone as "you" by depth where texts are in hand.
B82c: the B80h size summaries (results/x80/x80_magnitude rows, recomputed here) stratified by kappa (>= 0.6 vs below), restricted to
      laws whose gold CI excludes 0, with one setting per instrument (gold prevalence where it exists, else native), and with
      distortions in units of |gold's law| and in gold standard errors.
B82d: kappa, TPR and FPR of the Haiku and Sonnet 0-100 ratings thresholded at gold prevalence (x75_rating_eval.py's flags, seed 7500).
usage: python3 x82_desc.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, re, glob, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; PP = 100.0
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json")); WI = json.load(open(f"{W}/results/x75_within/x75_within.json"))["within"]
LLM = ("haiku", "sonnet", "qwen", "gemma"); OTH = ("cad", "qian_reddit", "qian_gab", "cga", "bad")
res = {"B82b": {}, "B82c": {}, "B82d": {}}
# ---------------- B82b
for pop in OTH:
    R = DEC[pop]; out = {}
    for L, r in R["labelers"].items():
        if not L.startswith(LLM):
            continue
        d = r["laws"]["depth"]; w = WI[pop]["labelers"].get(L, {})
        out[L] = dict(gold=R["gold"]["depth"][0], distortion=d["distortion"][0], fp_term=d["fp_term"][0], tp_term=d["tp_term"][0],
                      attenuation=d["attenuation_term"][0] - d["gold"][0], within_fp_term=(w.get("fp_term") or [None])[0], within_distortion=(w.get("distortion") or [None])[0])
    res["B82b"][pop] = out
fp = [v["fp_term"] for p in OTH for v in res["B82b"][p].values()]; wfp = [v["within_fp_term"] for p in OTH for v in res["B82b"][p].values() if v["within_fp_term"] is not None]
res["B82b"]["summary"] = dict(n=len(fp), fp_min=min(fp), fp_max=max(fp), n_fp_pos=int(sum(f > 0 for f in fp)), within_fp_min=min(wfp), within_fp_max=max(wfp), n_within_fp_pos=int(sum(f > 0 for f in wfp)))
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
for pop in ("cad", "qian_reddit", "qian_gab"):
    try:
        P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet", columns=["id", "z", "depth"])
    except Exception:
        continue
    txt = {}
    for f in glob.glob(f"{W}/data/{pop.split('_')[0]}*items.jsonl") + glob.glob(f"{W}/data/qian/*{pop.split('_')[-1]}*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "id" in r and ("text" in r):
                txt[str(r["id"])] = r["text"] or ""
    have = np.array([str(i) in txt for i in P.id.values])
    if have.mean() < 0.9:
        res["B82b"][f"you_{pop}"] = dict(coverage=float(have.mean())); continue
    you = np.array([bool(YOU.search(txt.get(str(i), ""))) for i in P.id.values]); civ = P.z.values == 0; d = P.depth.values
    res["B82b"][f"you_{pop}"] = dict(coverage=float(have.mean()), by_depth={str(k): float(you[civ & (np.minimum(d, 6) == k)].mean()) for k in range(7) if (civ & (np.minimum(d, 6) == k)).sum() > 50})
# ---------------- B82c
SE = json.load(open(f"{W}/results/x76_select/x76_select.json"))
rows = []
for pop, R in DEC.items():
    if not isinstance(R, dict) or "labelers" not in R:
        continue
    g = R["gold"]
    for L, r in R["labelers"].items():
        if L == "human_lenient":
            continue
        kap = r.get("kappa", [None])[0] if isinstance(r.get("kappa"), list) else r.get("kappa")
        for law, v in r["laws"].items():
            if law not in g:
                continue
            gl = g[law]; il = v["law"]; gse = (gl[2] - gl[1]) / (2 * 1.96)
            rows.append(dict(pop=pop, labeler=L, law=law, kappa=kap, gold=gl[0], gold_lo=gl[1], gold_hi=gl[2], est=il[0], lo=il[1], hi=il[2], gold_se=gse))
        w = WI[pop]["labelers"].get(L)
        if w:
            gw = WI[pop]["gold_within"]; lw = w["law"]; gse = (gw[2] - gw[1]) / (2 * 1.96)
            rows.append(dict(pop=pop, labeler=L, law="depth_within", kappa=kap, gold=gw[0], gold_lo=gw[1], gold_hi=gw[2], est=lw[0], lo=lw[1], hi=lw[2], gold_se=gse))
for r in rows:
    r["rel"] = abs(r["est"] - r["gold"]) / abs(r["gold"]) if r["gold"] != 0 else np.inf
    r["gold_sig"] = bool(r["gold_lo"] > 0 or r["gold_hi"] < 0); r["inst_sig"] = bool(r["lo"] > 0 or r["hi"] < 0)
    r["flip"] = bool(r["est"] * r["gold"] < 0 and r["inst_sig"]); r["dist_gold_se"] = abs(r["est"] - r["gold"]) / r["gold_se"] if r["gold_se"] > 0 else np.inf
    r["base"] = r["labeler"].split("@")[0]
def summ(R):
    if not R:
        return {}
    rel = np.array([r["rel"] for r in R]); dg = np.array([r["dist_gold_se"] for r in R])
    return dict(n=len(R), share_rel_gt_25=float(np.mean(rel > 0.25)), median_rel=float(np.median(rel)), n_flip=int(sum(r["flip"] for r in R)),
                n_flip_gold_sig=int(sum(r["flip"] and r["gold_sig"] for r in R)), n_manuf=int(sum(r["inst_sig"] and not r["gold_sig"] for r in R)),
                n_lost=int(sum(r["gold_sig"] and not r["inst_sig"] for r in R)), median_gold_se=float(np.median(dg)), share_gold_se_gt_2=float(np.mean(dg > 2)))
assert len(rows) == 523, len(rows)
hk = [r for r in rows if r["kappa"] is not None and r["kappa"] >= 0.6]; lk = [r for r in rows if r["kappa"] is not None and r["kappa"] < 0.6]
gs = [r for r in rows if r["gold_sig"]]
# one setting per instrument: the gold-prevalence setting ('@m' or '_m') where it exists, else the native one; LLMs without the parent
def pick(L):
    return L.endswith("@m") or L in ("haiku", "sonnet", "qwen7b_m", "gemma_m") or L.startswith("detoxify_unbiased@m")
dd = [r for r in rows if pick(r["labeler"])]
res["B82c"] = dict(all=summ(rows), kappa_ge_06=summ(hk), kappa_lt_06=summ(lk), gold_significant=summ(gs), gold_significant_kappa_ge_06=summ([r for r in gs if r in hk]),
                   one_per_instrument=summ(dd), n_kappa_missing=int(sum(r["kappa"] is None for r in rows)),
                   by_law_kappa_ge_06={l: summ([r for r in hk if r["law"] == l]) for l in sorted({r["law"] for r in rows})})
# ---------------- B82d
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); z = P.z.values.astype(float); N = len(z); prev = float(z.mean())
ids = P.id.astype(np.int64).values; has_par = P.parent_row.values >= 0
def ratings(name):
    R = {}
    for l in open(f"{W}/results/x75_rating/{name}.jsonl"):
        r = json.loads(l)
        if r.get("rating") is not None:
            R[int(r["id"])] = float(r["rating"])
    return R
for m in ("haiku", "sonnet"):
    r0 = ratings(f"{m}_ctx0"); r1 = ratings(f"{m}_ctx1")
    a0 = np.nan_to_num(np.array([r0.get(i, np.nan) for i in ids]), nan=0.0); a1 = np.nan_to_num(np.array([r1.get(i, np.nan) if hp else r0.get(i, np.nan) for i, hp in zip(ids, has_par)]), nan=0.0)
    for nm, s in ((f"{m}_r", a0), (f"{m}_r_ctx", a1)):
        k = int(round(prev * N)); rng = np.random.default_rng(7500); order = np.lexsort((rng.random(N), -s)); f = np.zeros(N); f[order[:k]] = 1.0
        q = f.mean(); po = float((f == z).mean()); pe = prev * q + (1 - prev) * (1 - q)
        res["B82d"][nm] = dict(kappa=(po - pe) / (1 - pe), tpr=float(f[z == 1].mean()), fpr=float(f[z == 0].mean()), flag_rate=float(q))
P2 = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet")
for m in ("haiku", "sonnet"):
    for nm in (m, f"{m}_ctx"):
        if f"y:{nm}" in P2:
            f = np.nan_to_num(P2[f"y:{nm}"].values.astype(float)); q = f.mean(); po = float((f == z).mean()); pe = prev * q + (1 - prev) * (1 - q)
            res["B82d"][f"{nm}_native"] = dict(kappa=(po - pe) / (1 - pe), tpr=float(f[z == 1].mean()), fpr=float(f[z == 0].mean()), flag_rate=float(q))
print(json.dumps(res["B82b"]["summary"]), json.dumps({k: v for k, v in res["B82b"].items() if k.startswith("you_")})[:800])
print(json.dumps({k: v for k, v in res["B82c"].items() if k != "by_law_kappa_ge_06"}, indent=0)[:3000])
print(json.dumps(res["B82d"], indent=0))
json.dump(res, open(OUT, "w"), indent=1, default=float); print("saved", OUT)
