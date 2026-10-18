"""Addendum 88, B88a: practitioner-calibrated operating points.
A practitioner labels a uniform random calibration sample (200, 500 or 1,000 comments; 500 draws each, seed 8801), picks the labeler's
cut on it (kappa-max, F1-max, or prevalence matching: calibration flag rate closest to the calibration gold rate; ties to the higher
cut) and measures the depth law on the whole population. Per draw: population flag rate, kappa, the depth law (points per level,
depth capped at 10), its excess over gold's (OLS slope of flag - gold on depth) with its article-clustered Wald z.
Instruments: Haiku and Haiku shown the parent on cc_threads (four-level answer, cuts >= 1, 2, 3); Haiku, Sonnet and both shown the
parent on cc_s5 (0-100 rating, cuts 1..100; and the four-level answer).
Decisions: for the cut chosen most often under kappa-max at 500 labels, the share of article-cluster subsamples whose depth-law
conclusion (Wald, |t| > 1.96, article-clustered) differs from gold's; the single rater against gold B (seed 8503) on the same
subsamples. cc_threads: 10,000 (seed 8602) and 20,000 (8810) comments; cc_s5: 10,000 (8811) and 20,000 (8812); 200 each.
usage: python3 x88_calibrated.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]; ND = 500; SIZES = (200, 500, 1000); NSUB = 200
RULES = ("kappa", "f1", "prev")


def ols_cl(xv, yv, c):
    xb = xv.mean(); yb = yv.mean(); sxx = np.sum((xv - xb) ** 2)
    if sxx <= 0:
        return 0.0, 0.0
    b = np.sum((xv - xb) * (yv - yb)) / sxx; e = (xv - xb) * (yv - yb - b * (xv - xb)) / sxx
    psi = np.bincount(c, weights=e); G = np.count_nonzero(np.bincount(c))
    return float(b), float(np.sqrt(np.sum(psi ** 2) * G / max(G - 1, 1)))


def kappa_of(tp, fp, fn, tn):
    n = tp + fp + fn + tn; po = (tp + tn) / n; py = (tp + fp) / n; pz = (tp + fn) / n; pe = py * pz + (1 - py) * (1 - pz)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def load_rating(model, ctx, ids):
    r = {}
    for l in open(f"{W}/results/x75_rating/{model}_ctx{ctx}.jsonl"):
        d = json.loads(l); v = d.get("rating")
        r[int(d["id"])] = v if v is not None else -1
    return np.array([r.get(int(i), -1) for i in ids], float)


res = {"pops": {}}
for pop in ("cc_threads", "cc_s5"):
    P = pd.read_parquet(f"{W}/results/x14_pops/{pop}.parquet")
    z = P.z.values.astype(float); N = len(z); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]
    nC = int(cl.max() + 1); csize = np.bincount(cl, minlength=nC)
    meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "toxicity_annotator_count"]).set_index("id")
    nv = meta.toxicity_annotator_count.reindex(P.id.values).values.astype(np.int64); kv = np.clip(np.rint(P.tox.values * nv).astype(np.int64), 0, nv)
    gold_b, gold_se = ols_cl(x, z, cl)
    R = {"N": N, "prevalence": float(z.mean()), "gold_depth_pp": [100 * gold_b, 100 * gold_se], "instruments": {}}
    INS = {}
    llms = ("haiku", "haiku_ctx") if pop == "cc_threads" else ("haiku", "sonnet", "haiku_ctx", "sonnet_ctx")
    for m in llms:
        INS[f"{m}:level"] = (np.nan_to_num(P[f"s:{m}"].values.astype(float), nan=-1.0), np.array([1.0, 2.0, 3.0]))
    if pop == "cc_s5":
        for m in ("haiku", "sonnet"):
            for c in (0, 1):
                INS[f"{m}{'_ctx' if c else ''}:rating"] = (load_rating(m, c, P.id.values), np.arange(1.0, 101.0))
    rng = np.random.default_rng(8801)
    draws = {n: [rng.choice(N, n, replace=False) for _ in range(ND)] for n in SIZES}
    for name, (s, cuts) in INS.items():
        # population quantities per cut
        pc = []
        for c in cuts:
            y = (s >= c).astype(float); tp = float((y * z).sum()); fp = float((y * (1 - z)).sum()); fn = float(((1 - y) * z).sum()); tn = N - tp - fp - fn
            b, se = ols_cl(x, y, cl); e, se_e = ols_cl(x, y - z, cl)
            pc.append(dict(cut=float(c), flag_rate=float(y.mean()), kappa=float(kappa_of(tp, fp, fn, tn)), law=100 * b, law_z=b / se if se > 0 else 0.0,
                           excess=100 * e, excess_z=e / se_e if se_e > 0 else 0.0))
        out = {"per_cut": pc if len(cuts) <= 3 else None, "native": None, "calibrated": {}}
        if name.endswith(":level"):
            out["native"] = pc[1]
        # calibration draws: counts on S for every cut via cumulative histograms
        smax = int(cuts.max()); sb = np.clip(s, -1, smax).astype(int) + 1             # bins 0..smax+1 (0 = missing)
        for n in SIZES:
            ch_ = {r: [] for r in RULES}
            for S in draws[n]:
                hp = np.bincount(sb[S][z[S] == 1], minlength=smax + 2); hn = np.bincount(sb[S][z[S] == 0], minlength=smax + 2)
                ge_p = np.cumsum(hp[::-1])[::-1]; ge_n = np.cumsum(hn[::-1])[::-1]          # counts with bin >= j
                idx = (cuts.astype(int) + 1)                                             # bin of cut c is c + 1
                tp = ge_p[idx].astype(float); fp = ge_n[idx].astype(float); P1 = hp.sum(); N0 = hn.sum(); fn = P1 - tp; tn = N0 - fp
                kap = np.nan_to_num(kappa_of(tp, fp, fn, tn), nan=-np.inf)
                f1 = np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
                prv = -np.abs((tp + fp) / n - P1 / n)
                for r, v in (("kappa", kap), ("f1", f1), ("prev", prv)):
                    j = len(v) - 1 - int(np.argmax(v[::-1]))                              # ties to the higher cut
                    ch_[r].append(j)
            for r in RULES:
                J = np.array(ch_[r]); ex = np.array([pc[j]["excess"] for j in J]); exz = np.array([pc[j]["excess_z"] for j in J])
                lz = np.array([pc[j]["law_z"] for j in J]); vals, cnt = np.unique(cuts[J], return_counts=True)
                out["calibrated"][f"{r}@{n}"] = dict(
                    cut_counts={str(int(v)): int(c_) for v, c_ in zip(vals, cnt)}, cut_median=float(np.median(cuts[J])),
                    flag_rate_median=float(np.median([pc[j]["flag_rate"] for j in J])), kappa_median=float(np.median([pc[j]["kappa"] for j in J])),
                    excess_median=float(np.median(ex)), excess_p05=float(np.percentile(ex, 5)), excess_p95=float(np.percentile(ex, 95)),
                    share_excess_sig_pos=float(np.mean(exz > 1.96)), share_law_sig_pos=float(np.mean(lz > 1.96)),
                    share_law_sig_neg=float(np.mean(lz < -1.96)), modal_cut=float(vals[np.argmax(cnt)]))
            k5 = out["calibrated"][f"kappa@{n}"]
            print(pop, name, n, "kappa-max: cut", k5["modal_cut"], "excess", round(k5["excess_median"], 3), [round(k5["excess_p05"], 3), round(k5["excess_p95"], 3)],
                  "sig+", k5["share_excess_sig_pos"], "| f1 sig+", out["calibrated"][f"f1@{n}"]["share_excess_sig_pos"], "| prev sig+",
                  out["calibrated"][f"prev@{n}"]["share_excess_sig_pos"], flush=True)
        R["instruments"][name] = out
    # decisions at the modal kappa-max cut (500 labels) vs the single rater against gold B
    r1 = np.random.default_rng(8503); kA = r1.hypergeometric(kv, nv - kv, np.ones_like(nv)); A1 = (kA >= 1).astype(float)
    gB = ((kv - kA) / np.maximum(nv - 1, 1) >= 0.5 - 1e-12).astype(float)
    FL = {"gold": z, "gold_B": gB, "single_rater": A1}
    for name, (s, cuts) in INS.items():
        FL[name] = (s >= R["instruments"][name]["calibrated"]["kappa@500"]["modal_cut"]).astype(float)
    sizes = ((10000, 8602), (20000, 8810)) if pop == "cc_threads" else ((10000, 8811), (20000, 8812))
    dec = {}
    for target, seed in sizes:
        rs = np.random.default_rng(seed); V = {k: [] for k in FL}; got = []
        for _ in range(NSUB):
            order = rs.permutation(nC); cum = np.cumsum(csize[order]); take = order[: int(np.searchsorted(cum, target) + 1)]
            inC = np.zeros(nC, bool); inC[take] = True; msk = inC[cl]; got.append(int(msk.sum())); cc = pd.factorize(cl[msk])[0]
            for k, v in FL.items():
                b, se = ols_cl(x[msk], v[msk], cc); t = b / se if se > 0 else 0.0
                V[k].append(1 if t > 1.96 else (-1 if t < -1.96 else 0))
        V = {k: np.array(v) for k, v in V.items()}
        dd = {"mean_size": float(np.mean(got)), "gold_shares": {str(s_): float(np.mean(V["gold"] == s_)) for s_ in (-1, 0, 1)}}
        for k in FL:
            if k in ("gold", "gold_B"):
                continue
            ref = V["gold_B"] if k == "single_rater" else V["gold"]
            dd[k] = dict(differ=float(np.mean(V[k] != ref)), reversal=float(np.mean(V[k] * ref == -1)), manufactured=float(np.mean((V[k] != 0) & (ref == 0))),
                         lost=float(np.mean((V[k] == 0) & (ref != 0))), sig_pos=float(np.mean(V[k] == 1)))
        dec[str(target)] = dd
        print(pop, "decisions", target, {k: (round(v["differ"], 3), round(v["reversal"], 3)) for k, v in dd.items() if isinstance(v, dict) and "differ" in v}, flush=True)
    R["decisions"] = dec
    res["pops"][pop] = R
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
