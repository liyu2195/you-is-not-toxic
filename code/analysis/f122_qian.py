"""Addendum 24, B1 (H25a-g): hate speech in the Qian et al. (2019) Reddit and Gab conversations, with in-context crowd gold.
Labels: Haiku 5.5, task `hate`; y0 = context-free flag; y1 = "with the parent" (replies labeled with the parent shown, first posts
keep y0, as f10). Missing or errored answers count as not hateful (counted and reported).
Audit (per platform; conversation-cluster Poisson bootstrap, 2,000 reps):
  item prevalence, FPR, TPR, kappa; share of conversations (>= 2 posts) with >= 1 hate post;
  reply RR = P(reply = 1 | parent = 1) / P(reply = 1 | parent = 0), parent label context-free (gold for gold);
    H25e: log RR(y0) - log RR(gold) < 0;  H25f: log RR(y1) - log RR(y0) < 0;
  depth gap = share at depth >= 2 minus share at depth 0 (all posts);  H25g: gap(y1) - gap(gold) > 0.
THREADSAFE, frozen (units = conversations with >= 2 posts; guidance = y0): f98 (the final design ts2_a100c, cluster_ppi, cluster_ppi4, ts2_cv1) at
250/500/1000 labels -> H25a, H25b; f90b pilot; pairs f24h at 500/1000/2000 -> H25c; depth contrast f54b at 500/2000 -> H25d.
REG = the design the registered tests use (frozen at Addendum 24); FIN = the final design (Addendum 33), run separately with the same seed.
usage: f122_qian.py OUT_JSON [--reps 300] [--boot 2000]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, glob, subprocess, numpy as np, pandas as pd
args = sys.argv[1:]; OUT = args[0]
def opt(k, d): return args[args.index(k) + 1] if k in args else d
REPS = int(opt("--reps", 300)); B = int(opt("--boot", 2000)); W = _ROOT; REG = "ts2_a100c"; FIN = os.environ.get("FINAL_METHOD", "ts2_a100cy"); LD = os.environ.get("QIAN_LABELS", f"{W}/results/second/qian")
WD = os.path.join(os.path.dirname(os.path.abspath(OUT)), "f122_work"); os.makedirs(WD, exist_ok=True)
rng = np.random.default_rng(122 + int(os.environ.get("SEED_OFFSET", 0)))
def labels(d):
    R = []
    for f in sorted(glob.glob(f"{d}/shard_*.jsonl")):
        for l in open(f):
            try: R.append(json.loads(l))
            except Exception: pass
    R = pd.DataFrame(R).drop_duplicates("id", keep="last").set_index("id")
    ok = R["ans"].notna() if "ans" in R else pd.Series(False, index=R.index)
    return (R.get("ans") == "A").astype(float), ok
M = pd.read_parquet(f"{W}/data/qian/qian_meta.parquet").set_index("id")
a0, ok0 = labels(f"{LD}/haiku_hate_ctx0"); a1, ok1 = labels(f"{LD}/haiku_hate_ctx1")
M["y0"] = a0.reindex(M.index).fillna(0.0).values; M["ok0"] = ok0.reindex(M.index).fillna(False).values
rep = M.parent >= 0
# replies to an empty post were never shown a parent (no context to show): they keep y0, like first posts
has1 = M.index.isin(a1.index); M["y1"] = np.where(rep & has1, a1.reindex(M.index).fillna(0.0).values, M.y0.values)
M["ok1"] = np.where(rep & has1, ok1.reindex(M.index).fillna(False).values, M.ok0.values)
def kappa(z, y):
    po = (z == y).mean(); pe = z.mean() * y.mean() + (1 - z.mean()) * (1 - y.mean()); return float((po - pe) / (1 - pe))
def ci(base, bs): return [float(base), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
res = {"n_labels_missing_ctx0": int((~M.ok0).sum()), "n_labels_missing_ctx1": int((rep & has1 & ~M.ok1.astype(bool)).sum()), "n_replies_empty_parent": int((rep & ~has1).sum()), "platforms": {}, "reps": REPS, "boot": B}
for plat, D in M.groupby("platform"):
    D = D.copy(); conv = pd.factorize(D.conv)[0]; nC = conv.max() + 1; D["c"] = conv
    o = {"n_conversations": int(nC), "n_posts": int(len(D)), "n_replies": int((D.parent >= 0).sum())}
    o["item"] = {src: dict(prev=float(D[src].mean()), **({} if src == "z" else dict(fpr=float(D[src][D.z == 0].mean()), tpr=float(D[src][D.z == 1].mean()), kappa=kappa(D.z.values, D[src].values)))) for src in ["z", "y0", "y1"]}
    # per-conversation sufficient statistics for the bootstrap
    k = D.groupby("c").size().values; big = k >= 2
    S = {}
    for src in ["z", "y0", "y1"]:
        S[f"any_{src}"] = (D.groupby("c")[src].max().values * big)
        S[f"d2_{src}"] = D[src].where(D.depth >= 2, 0).groupby(D.c).sum().values; S[f"d0_{src}"] = D[src].where(D.depth == 0, 0).groupby(D.c).sum().values
    S["n_big"] = big.astype(float); S["n_d2"] = (D.depth >= 2).groupby(D.c).sum().values.astype(float); S["n_d0"] = (D.depth == 0).groupby(D.c).sum().values.astype(float)
    E = D[D.parent >= 0]; P_ = D.reindex(E.parent.values)
    for src, psrc in [("z", "z"), ("y0", "y0"), ("y1", "y0")]:                  # parent labels context-free (gold for gold)
        pp = P_[psrc].values; ch = E[src].values; ce = E.c.values
        S[f"e1_{src}"] = np.bincount(ce, weights=(pp == 1).astype(float), minlength=nC); S[f"e1c_{src}"] = np.bincount(ce, weights=((pp == 1) & (ch == 1)).astype(float), minlength=nC)
        S[f"e0_{src}"] = np.bincount(ce, weights=(pp == 0).astype(float), minlength=nC); S[f"e0c_{src}"] = np.bincount(ce, weights=((pp == 0) & (ch == 1)).astype(float), minlength=nC)
    names = list(S); A = np.column_stack([S[n] for n in names]); ix = {n: i for i, n in enumerate(names)}
    def stats(T):                                                              # T: (..., n_stats) weighted totals
        g = lambda n: T[..., ix[n]]; out = {}
        for src in ["z", "y0", "y1"]:
            out[f"share_{src}"] = g(f"any_{src}") / g("n_big")
            out[f"gap_{src}"] = g(f"d2_{src}") / g("n_d2") - g(f"d0_{src}") / g("n_d0")
            p1 = g(f"e1c_{src}") / g(f"e1_{src}"); p0 = g(f"e0c_{src}") / g(f"e0_{src}"); out[f"rr_{src}"] = p1 / p0; out[f"diff_{src}"] = p1 - p0
        out["H25e_logrr_y0_minus_gold"] = np.log(out["rr_y0"]) - np.log(out["rr_z"]); out["H25f_logrr_y1_minus_y0"] = np.log(out["rr_y1"]) - np.log(out["rr_y0"])
        out["H25g_gap_y1_minus_gold"] = out["gap_y1"] - out["gap_z"]; out["gap_y0_minus_gold"] = out["gap_y0"] - out["gap_z"]
        out["share_y0_minus_gold"] = out["share_y0"] - out["share_z"]; out["share_y1_minus_gold"] = out["share_y1"] - out["share_z"]
        return out
    base = stats(A.sum(0)); bs = {kk: [] for kk in base}
    for _ in range(B // 100):
        Wt = rng.poisson(1.0, size=(100, nC)).astype(float); sb = stats(Wt @ A)
        for kk in base: bs[kk].append(sb[kk])
    o["audit"] = {kk: ci(base[kk], np.concatenate(v)) for kk, v in bs.items()}
    o["H25e"] = bool(o["audit"]["H25e_logrr_y0_minus_gold"][2] < 0); o["H25f"] = bool(o["audit"]["H25f_logrr_y1_minus_y0"][2] < 0); o["H25g"] = bool(o["audit"]["H25g_gap_y1_minus_gold"][1] > 0)
    o["by_depth"] = {src: {int(kk): float(v) for kk, v in D.groupby(D.depth.clip(upper=5))[src].mean().items()} for src in ["z", "y0", "y1"]}
    # THREADSAFE inputs (guidance: the context-free flag)
    Ub = D[D.groupby("c").z.transform("size") >= 2]
    up = f"{WD}/units_{plat}.parquet"; Ub.assign(unit=Ub.conv, s=Ub.y0, y=Ub.y0.astype(int))[["unit", "z", "s", "y"]].to_parquet(up)
    ep = f"{WD}/edges_{plat}.parquet"; pd.DataFrame({"parent": E.parent.values, "child": E.index.values, "Zp": P_.z.values.astype(int), "Zc": E.z.values.astype(int), "Ap": P_.y0.values.astype(int), "Ac": E.y0.values.astype(int)}).to_parquet(ep)
    ip = f"{WD}/items_{plat}.parquet"; D.reset_index().assign(s=D.y0.values, y=D.y0.values.astype(int))[["id", "z", "s", "y", "depth"]].to_parquet(ip)
    run = lambda cmd: subprocess.run(["python3", "-I"] + cmd, check=True, capture_output=True, text=True)
    run([f"{W}/code/analysis/f98_threadsafe2_sim.py", up, f"{WD}/f98_{plat}.json", "--reps", str(REPS), "--budgets", "250,500,1000", "--methods", f"{REG},cluster_ppi,cluster_ppi4,ts2_cv1"])
    run([f"{W}/code/analysis/f90b_pilot_predict.py", f"{WD}/f90b_{plat}.json", f"{plat}={up}:{WD}/f98_{plat}.json"])
    run([f"{W}/code/analysis/f24h_edges_final.py", ep, "none", f"{WD}/f24h_{plat}.json", "--reps", str(REPS), "--budgets", "500,1000,2000"])
    run([f"{W}/code/analysis/f54b_depth_upgrade.py", ip, f"{WD}/f54b_{plat}.json", "--reps", str(max(REPS, 500) if REPS >= 300 else REPS), "--budgets", "500,2000"])
    o["f98"] = json.load(open(f"{WD}/f98_{plat}.json")); o["f90b"] = json.load(open(f"{WD}/f90b_{plat}.json"))[plat]
    o["f24h"] = json.load(open(f"{WD}/f24h_{plat}.json")); o["f54b"] = json.load(open(f"{WD}/f54b_{plat}.json"))
    r = o["f98"]["results"]; bc = lambda m: min([r[f"cluster_ppi@{m}"], r[f"cluster_ppi4@{m}"]], key=lambda x: x["width_hyb"])
    o["threadsafe"] = dict(coverage={m: r[f"{REG}@{m}"]["coverage_hyb"] for m in [250, 500, 1000]}, ratio_500=r[f"{REG}@500"]["width_hyb"] / bc(500)["width_hyb"],
                          pilot_pred=o["f90b"]["pilot_median"], pair_cov={m: o["f24h"]["results"][f"final@{m}"]["cov_score"] for m in [500, 1000, 2000]},
                          depth_cov={m: o["f54b"]["results"][f"strat_ppi_hyb@{m}"]["coverage"] for m in [500, 2000]})
    if FIN != REG:                                                     # the final design, separate run with the same seed (common random numbers with REG)
        run([f"{W}/code/analysis/f98_threadsafe2_sim.py", up, f"{WD}/f98_{plat}_final.json", "--reps", str(REPS), "--budgets", "250,500,1000", "--methods", FIN + (",ts2_a100cy" if FIN.endswith("x") else "")])
        rf = json.load(open(f"{WD}/f98_{plat}_final.json"))["results"]
        o["threadsafe_final"] = dict(method=FIN, coverage={m: rf[f"{FIN}@{m}"]["coverage_hyb"] for m in [250, 500, 1000]}, width={m: rf[f"{FIN}@{m}"]["width_hyb"] for m in [250, 500, 1000]},
                                    ratio_500=rf[f"{FIN}@500"]["width_hyb"] / bc(500)["width_hyb"], ratio_to_reg_500=rf[f"{FIN}@500"]["width_hyb"] / r[f"{REG}@500"]["width_hyb"], share_q1_500=rf[f"{FIN}@500"].get("share_q1"))
    res["platforms"][plat] = o
    print(plat, json.dumps({kk: [round(x, 4) for x in v] for kk, v in o["audit"].items() if kk.startswith("H25") or kk.startswith("share") or kk.startswith("rr")}), flush=True)
    print(plat, "THREADSAFE", json.dumps(o["threadsafe"]), "item", json.dumps(o["item"]), flush=True)
cov = [c for o in res["platforms"].values() for c in o["threadsafe"]["coverage"].values()]
res["H25a"] = bool(np.mean(cov) >= .90 and min(cov) >= .85); res["H25b"] = bool(all(o["threadsafe"]["ratio_500"] <= 1.0 for o in res["platforms"].values()))
res["H25c"] = bool(all(c >= .90 for o in res["platforms"].values() for c in o["threadsafe"]["pair_cov"].values()))
res["H25d"] = bool(all(c >= .90 for o in res["platforms"].values() for c in o["threadsafe"]["depth_cov"].values()))
for h in ["H25e", "H25f", "H25g"]: res[h] = bool(all(o[h] for o in res["platforms"].values()))
json.dump(res, open(OUT, "w"), indent=1); print({h: res[h] for h in ["H25a", "H25b", "H25c", "H25d", "H25e", "H25f", "H25g"]})
