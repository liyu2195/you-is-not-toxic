"""Addendum 24, B3 (H27a-d): prompt sensitivity on the S5 article sample (Civil Comments, train split).
Annotators: the main prompt (`tox`, population labels), `tox_bin` (yes/no, Perspective definition) and `tox_rev` (reversed 4-point
scale, new wording and system prompt); each without context (P0) and with the parent for replies (P1; first comments keep P0).
Flags: toxic or very toxic, or "Yes". Missing letters count as not toxic; errored requests are missing (as f51), and threads with
a missing label are dropped (as f10). Definitions follow f10/f13: threads have >= 2 comments; slope on min(depth, 10);
reply RR with context-free parent labels (gold for gold). Article-cluster Poisson bootstrap, 1,000 reps.
  H27a: thread share with >= 1 flag minus gold's > 0, for P0 and P1;   H27b: log RR(P0) - log RR(gold) < 0;
  H27c: slope(P0) - slope(gold) > 0, and slope(P1) - slope(P0) > 0.   (all CIs exclude 0; both alternative prompts)
  H27d: THREADSAFE guided by each prompt's P0 score on the sample's threads and articles (f98: the final design ts2_a100c, cluster_ppi, cluster_ppi4;
        250/500/1000 labels): coverage (hybrid) mean >= .90 and minimum >= .85; width at 500 labels <= cluster PPI's in every setting.
REG = the design the registered tests use (frozen at Addendum 24); FIN = the final design (Addendum 33), run separately with the same seed.
usage: f124_prompts.py OUT_JSON [--reps 300] [--boot 1000]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, glob, subprocess, numpy as np, pandas as pd
args = sys.argv[1:]; OUT = args[0]
def opt(k, d): return args[args.index(k) + 1] if k in args else d
REPS = int(opt("--reps", 300)); B = int(opt("--boot", 1000)); W = _ROOT; REG = "ts2_a100c"; FIN = os.environ.get("FINAL_METHOD", "ts2_a100cy")
WD = os.path.join(os.path.dirname(os.path.abspath(OUT)), "f124_work"); os.makedirs(WD, exist_ok=True)
rng = np.random.default_rng(124 + int(os.environ.get("SEED_OFFSET", 0)))
ids = set(np.loadtxt(f"{W}/results/second/s5_ids.txt", dtype=np.int64).tolist())
I = pd.read_parquet(f"{W}/results/f51_items_haiku.parquet"); I = I[I.id.isin(ids)].set_index("id")
MAPS = {"tox": {"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}, "tox_rev": {"D": 1.0, "C": 2 / 3, "B": 1 / 3, "A": 0.0}, "tox_bin": {"A": 1.0, "B": 0.0}}
def read(pattern, task):
    R = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("id") in ids: R.append(r)
    R = pd.DataFrame(R)
    if "error" in R: R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last").set_index("id"); return R.ans.map(MAPS[task]).fillna(0.0)
SRC = {"main": ("tox", f"{W}/results/full/haiku_tox_ctx0/shard_*.jsonl", f"{W}/results/full/haiku_tox_ctx1/shard_*.jsonl"),
       "bin": ("tox_bin", f"{W}/results/second/s5/haiku_toxbin_ctx0/shard_*.jsonl", f"{W}/results/second/s5/haiku_toxbin_ctx1/shard_*.jsonl"),
       "rev": ("tox_rev", f"{W}/results/second/s5/haiku_toxrev_ctx0/shard_*.jsonl", f"{W}/results/second/s5/haiku_toxrev_ctx1/shard_*.jsonl")}
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id"]).set_index("id").parent_id
I["has_parent"] = meta.reindex(I.index).notna().values
for p, (task, f0, f1) in SRC.items():
    s0 = read(f0, task).reindex(I.index); s1 = read(f1, task).reindex(I.index)
    I[f"s_{p}0"] = s0.values; I[f"{p}0"] = (s0 >= 2 / 3 - 1e-9).astype(float).where(s0.notna()).values
    c = (s1 >= 2 / 3 - 1e-9).astype(float).where(s1.notna()); I[f"{p}1"] = np.where(I.has_parent, c.values, I[f"{p}0"].values)
LAB = ["z"] + [f"{p}{c}" for p in SRC for c in [0, 1]]; I["z"] = I.z.astype(float)
ok = I[LAB].notna().all(axis=1); ok_root = ok.groupby(I.root).all(); n_drop = int((~I.root.map(ok_root)).sum())
I = I[I.root.map(ok_root)].copy()
def kappa(z, y):
    po = (z == y).mean(); pe = z.mean() * y.mean() + (1 - z.mean()) * (1 - y.mean()); return float((po - pe) / (1 - pe))
res = dict(n_items=int(len(I)), n_items_dropped_incomplete_threads=n_drop, reps=REPS, boot=B,
           item={c: dict(prev=float(I[c].mean()), **({} if c == "z" else dict(kappa=kappa(I.z.values, I[c].values), fpr=float(I[c][I.z == 0].mean()), tpr=float(I[c][I.z == 1].mean())))) for c in LAB})
# per-article sufficient statistics
I["tk"] = I.groupby("root").z.transform("size"); T = I[I.tk >= 2].copy(); T["x"] = T.depth.clip(upper=10).astype(float)
art = pd.factorize(I.article)[0]; I["a_"] = art; nA = art.max() + 1; T["a_"] = I.a_.reindex(T.index).values
S = {"n_thr": np.bincount(T.drop_duplicates("root").a_.values, minlength=nA).astype(float), "n": np.bincount(T.a_, minlength=nA).astype(float),
     "sx": np.bincount(T.a_, weights=T.x, minlength=nA), "sxx": np.bincount(T.a_, weights=T.x ** 2, minlength=nA)}
first = T.groupby("root").a_.first()
for c in LAB:
    S[f"any_{c}"] = np.bincount(first.values, weights=T.groupby("root")[c].max().reindex(first.index).values, minlength=nA)
    S[f"sy_{c}"] = np.bincount(T.a_, weights=T[c], minlength=nA); S[f"sxy_{c}"] = np.bincount(T.a_, weights=T.x * T[c], minlength=nA)
E = I[I.parent.isin(I.index)]; Pp = I.reindex(E.parent.values)
for c in LAB:
    pc = "z" if c == "z" else c[:-1] + "0"; pp = Pp[pc].values; ch = E[c].values; ea = E.a_.values
    for nm_, w in [("e1", pp == 1), ("e1c", (pp == 1) & (ch == 1)), ("e0", pp == 0), ("e0c", (pp == 0) & (ch == 1))]:
        S[f"{nm_}_{c}"] = np.bincount(ea, weights=w.astype(float), minlength=nA)
names = list(S); A = np.column_stack([S[n] for n in names]); ix = {n: i for i, n in enumerate(names)}
def stats(Tt):
    g = lambda n: Tt[..., ix[n]]; out = {}; vx = g("sxx") / g("n") - (g("sx") / g("n")) ** 2
    for c in LAB:
        out[f"share_{c}"] = g(f"any_{c}") / g("n_thr"); out[f"slope_{c}"] = (g(f"sxy_{c}") / g("n") - g("sx") * g(f"sy_{c}") / g("n") ** 2) / vx
        out[f"rr_{c}"] = (g(f"e1c_{c}") / g(f"e1_{c}")) / (g(f"e0c_{c}") / g(f"e0_{c}"))
    for p in SRC:
        for cc in [0, 1]: out[f"a_share_{p}{cc}_minus_gold"] = out[f"share_{p}{cc}"] - out["share_z"]
        out[f"b_logrr_{p}0_minus_gold"] = np.log(out[f"rr_{p}0"]) - np.log(out["rr_z"])
        out[f"c_slope_{p}0_minus_gold"] = out[f"slope_{p}0"] - out["slope_z"]; out[f"c_slope_{p}1_minus_{p}0"] = out[f"slope_{p}1"] - out[f"slope_{p}0"]
    return out
base = stats(A.sum(0)); bs = {k: [] for k in base}
for _ in range(B // 50):
    Wt = rng.poisson(1.0, size=(50, nA)).astype(float); sb = stats(Wt @ A)
    for k in base: bs[k].append(sb[k])
res["audit"] = {k: [float(base[k]), float(np.percentile(np.concatenate(v), 2.5)), float(np.percentile(np.concatenate(v), 97.5))] for k, v in bs.items()}
au = res["audit"]; H = {}
for p in ["bin", "rev", "main"]:
    H[p] = dict(a=bool(au[f"a_share_{p}0_minus_gold"][1] > 0 and au[f"a_share_{p}1_minus_gold"][1] > 0), b=bool(au[f"b_logrr_{p}0_minus_gold"][2] < 0),
                c=bool(au[f"c_slope_{p}0_minus_gold"][1] > 0 and au[f"c_slope_{p}1_minus_{p}0"][1] > 0))
res["verdicts_by_prompt"] = H
for h in "abc": res[f"H27{h}"] = bool(H["bin"][h] and H["rev"][h])
# H27d: THREADSAFE guided by each prompt's context-free score
run = lambda cmd: subprocess.run(["python3", "-I"] + cmd, check=True, capture_output=True, text=True)
I["k_art"] = I.groupby("article").z.transform("size"); res["sims"] = {}
for p in ["bin", "rev", "main"]:
    for u, col, kc in [("threads", "root", "tk"), ("articles", "article", "k_art")]:
        X = I[I[kc] >= 2]; up = f"{WD}/units_{u}_{p}.parquet"
        pd.DataFrame({"unit": X[col].values, "z": X.z.astype(int).values, "s": X[f"s_{p}0"].values, "y": X[f"{p}0"].astype(int).values}).to_parquet(up)
        run([f"{W}/code/analysis/f98_threadsafe2_sim.py", up, f"{WD}/f98_{u}_{p}.json", "--reps", str(REPS), "--budgets", "250,500,1000", "--methods", f"{REG},cluster_ppi,cluster_ppi4"])
        res["sims"][f"{u}_{p}"] = json.load(open(f"{WD}/f98_{u}_{p}.json"))
        if FIN != REG:                                                 # the final design, separate run with the same seed
            run([f"{W}/code/analysis/f98_threadsafe2_sim.py", up, f"{WD}/f98_{u}_{p}_final.json", "--reps", str(REPS), "--budgets", "250,500,1000", "--methods", FIN + (",ts2_a100cy" if FIN.endswith("x") else "")])
            res["sims"][f"{u}_{p}_final"] = json.load(open(f"{WD}/f98_{u}_{p}_final.json"))
SM = res["sims"]; bc = lambda r, m: min(r[f"cluster_ppi@{m}"]["width_hyb"], r[f"cluster_ppi4@{m}"]["width_hyb"])
cov = [SM[f"{u}_{p}"]["results"][f"{REG}@{m}"]["coverage_hyb"] for p in ["bin", "rev"] for u in ["threads", "articles"] for m in [250, 500, 1000]]
rat = {f"{u}_{p}": SM[f"{u}_{p}"]["results"][f"{REG}@500"]["width_hyb"] / bc(SM[f"{u}_{p}"]["results"], 500) for p in ["bin", "rev", "main"] for u in ["threads", "articles"]}
if FIN != REG:
    res["final_design"] = {f"{u}_{p}": dict(method=FIN, coverage=[SM[f"{u}_{p}_final"]["results"][f"{FIN}@{m}"]["coverage_hyb"] for m in [250, 500, 1000]],
                                           ratio_to_cluster_500=SM[f"{u}_{p}_final"]["results"][f"{FIN}@500"]["width_hyb"] / bc(SM[f"{u}_{p}"]["results"], 500),
                                           ratio_to_reg_500=SM[f"{u}_{p}_final"]["results"][f"{FIN}@500"]["width_hyb"] / SM[f"{u}_{p}"]["results"][f"{REG}@500"]["width_hyb"])
                           for p in ["bin", "rev", "main"] for u in ["threads", "articles"]}
res["H27d"] = bool(np.mean(cov) >= .90 and min(cov) >= .85 and all(rat[f"{u}_{p}"] <= 1.0 for p in ["bin", "rev"] for u in ["threads", "articles"]))
res["summary"] = dict(item={c: res["item"][c] for c in LAB}, audit={k: v for k, v in au.items() if k[:2] in ("a_", "b_", "c_")}, shares={c: au[f"share_{c}"] for c in LAB},
                      rr={c: au[f"rr_{c}"] for c in LAB}, slopes={c: au[f"slope_{c}"] for c in LAB}, threadsafe_coverage=cov, threadsafe_ratio_500=rat, verdicts=H)
json.dump(res, open(OUT, "w"), indent=1)
for k, v in au.items(): print(f"{k:34s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
print("ratios", json.dumps({k: round(v, 3) for k, v in rat.items()}), "coverage mean %.3f min %.3f" % (np.mean(cov), min(cov)))
print({h: res[h] for h in ["H27a", "H27b", "H27c", "H27d"]}, json.dumps(H))
