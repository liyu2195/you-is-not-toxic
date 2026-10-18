"""Addendum 34 (G1-G4): the larger open model from another family, Gemma 3 27B (instruction-tuned, W4A16), on CCC, the Qian et al.
Reddit and Gab conversations, and the S5 Civil Comments sample. Same prompts as Haiku; next-token option probabilities (code/label_vllm.py).
Score: tox = p(very toxic) + p(toxic); hate = p(yes). Flags: "matched" = threshold giving the gold prevalence on the sample
(CCC: the out-of-context crowd's prevalence, as Qwen in f93); "default" = score > .5 (reported).
  G1 (S5, matched; article-cluster Poisson bootstrap, 1,000): (a) log RR(ctx0) - log RR(gold) < 0; (b) slope(with parent) - slope(gold) > 0;
     (c) thread share with >= 1 flag, with the parent shown, minus gold's > 0.  Definitions as f10/f13/f124.
  G2 (CCC, matched as f93): gap(with parent) - gap(in-context humans) > 0, gap = share at depth >= 3 minus depth 1 (iid bootstrap, 2,000).
  G3 (Qian, matched per platform): depth gap (>= 2 minus 0) with the parent minus gold's > 0 on both platforms (conversation bootstrap, 2,000).
  G4 (final THREADSAFE, f98 ts2_a100cy, guided by the context-free score): S5 threads and articles, Qian Reddit and Gab conversations;
     coverage (hybrid) mean >= .90 and minimum >= .85 over 250/500/1000 labels; width at 500 <= cluster PPI's in every setting.
usage: f129_open_model.py LABEL_DIR CCC_CSV OUT_JSON [--reps 300]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, glob, ast, subprocess, numpy as np, pandas as pd
args = sys.argv[1:]; LD, CCC, OUT = args[:3]
def opt(k, d): return args[args.index(k) + 1] if k in args else d
REPS = int(opt("--reps", 300)); W = _ROOT; FIN = os.environ.get("FINAL_METHOD", "ts2_a100cy")
WD = os.path.join(os.path.dirname(os.path.abspath(OUT)), "f129_work"); os.makedirs(WD, exist_ok=True)
rng = np.random.default_rng(129 + int(os.environ.get("SEED_OFFSET", 0)))
def load(pattern):
    R = {}; mass = []
    for f in sorted(glob.glob(f"{LD}/**/{pattern}", recursive=True)):
        for l in open(f):
            r = json.loads(l)
            if r.get("p") is None: continue
            R[r["id"]] = r["p"]; mass.append(r["mass"])
    return R, (float(np.mean(mass)) if mass else float("nan"))
def score(R, task):
    return pd.Series({k: (v[0] + v[1]) if task == "tox" else v[0] for k, v in R.items()})
def kappa(z, y):
    po = (z == y).mean(); pe = z.mean() * y.mean() + (1 - z.mean()) * (1 - y.mean()); return float((po - pe) / (1 - pe))
res = {"model": "RedHatAI/gemma-3-27b-it-quantized.w4a16", "reps": REPS}
# ---------------- G1: S5 ----------------
s0d, m0 = load("shard_*_tox_ctx0.jsonl"); s1d, m1 = load("shard_*_tox_ctx1.jsonl")
ids = set(np.loadtxt(f"{W}/results/second/s5_ids.txt", dtype=np.int64).tolist())
I = pd.read_parquet(f"{W}/results/f51_items_haiku.parquet"); I = I[I.id.isin(ids)].set_index("id")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id"]).set_index("id").parent_id
I["has_parent"] = meta.reindex(I.index).notna().values
I["s0"] = score(s0d, "tox").reindex(I.index).values; s1 = score(s1d, "tox").reindex(I.index)
t = float(np.quantile(I.s0.dropna(), 1 - I.z.mean())); res["s5"] = dict(threshold_matched=t, mean_option_mass_ctx0=m0, mean_option_mass_ctx1=m1)
for tag, thr in [("m", t), ("d", 0.5)]:
    I[f"{tag}0"] = (I.s0 > thr).astype(float).where(I.s0.notna()); c = (s1 > thr).astype(float).where(s1.notna())
    I[f"{tag}1"] = np.where(I.has_parent, c.values, I[f"{tag}0"].values)
LAB = ["z", "m0", "m1", "d0", "d1"]; I["z"] = I.z.astype(float)
ok = I[LAB].notna().all(axis=1); okr = ok.groupby(I.root).all(); res["s5"]["n_dropped"] = int((~I.root.map(okr)).sum()); I = I[I.root.map(okr)].copy()
res["s5"]["item"] = {c: dict(prev=float(I[c].mean()), **({} if c == "z" else dict(kappa=kappa(I.z.values, I[c].values)))) for c in LAB}
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
    pc = "z" if c == "z" else c[0] + "0"; pp = Pp[pc].values; ch = E[c].values; ea = E.a_.values
    for nm_, w in [("e1", pp == 1), ("e1c", (pp == 1) & (ch == 1)), ("e0", pp == 0), ("e0c", (pp == 0) & (ch == 1))]:
        S[f"{nm_}_{c}"] = np.bincount(ea, weights=w.astype(float), minlength=nA)
names = list(S); A = np.column_stack([S[n] for n in names]); ix = {n: i for i, n in enumerate(names)}
def st1(Tt):
    g = lambda n: Tt[..., ix[n]]; out = {}; vx = g("sxx") / g("n") - (g("sx") / g("n")) ** 2
    for c in LAB:
        out[f"share_{c}"] = g(f"any_{c}") / g("n_thr"); out[f"slope_{c}"] = (g(f"sxy_{c}") / g("n") - g("sx") * g(f"sy_{c}") / g("n") ** 2) / vx
        out[f"rr_{c}"] = (g(f"e1c_{c}") / g(f"e1_{c}")) / (g(f"e0c_{c}") / g(f"e0_{c}"))
    out["G1a_logrr_m0_minus_gold"] = np.log(out["rr_m0"]) - np.log(out["rr_z"]); out["G1b_slope_m1_minus_gold"] = out["slope_m1"] - out["slope_z"]
    out["G1c_share_m1_minus_gold"] = out["share_m1"] - out["share_z"]; out["share_m0_minus_gold"] = out["share_m0"] - out["share_z"]
    out["slope_m0_minus_gold"] = out["slope_m0"] - out["slope_z"]; out["slope_m1_minus_m0"] = out["slope_m1"] - out["slope_m0"]
    return out
base = st1(A.sum(0)); bs = {k: [] for k in base}
for _ in range(20):
    Wt = rng.poisson(1.0, size=(50, nA)).astype(float); sb = st1(Wt @ A)
    for k in base: bs[k].append(sb[k])
res["s5"]["audit"] = {k: [float(base[k]), float(np.percentile(np.concatenate(v), 2.5)), float(np.percentile(np.concatenate(v), 97.5))] for k, v in bs.items()}
au = res["s5"]["audit"]
res["G1a"] = bool(au["G1a_logrr_m0_minus_gold"][2] < 0); res["G1b"] = bool(au["G1b_slope_m1_minus_gold"][1] > 0); res["G1c"] = bool(au["G1c_share_m1_minus_gold"][1] > 0)
# ---------------- G2: CCC ----------------
c0d, mc0 = load("ccc_items_tox_ctx0.jsonl"); c1d, mc1 = load("ccc_items_tox_ctx1.jsonl")
d = pd.read_csv(CCC); mean = lambda s: float(np.mean(ast.literal_eval(s))) if isinstance(s, str) else np.nan
d["h_oc"] = d.tox_codes_oc.map(mean); d["h_ic"] = d.tox_codes_ic.map(mean)
cm = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id", "split"]); cm = cm[cm.split == "train"].set_index("id")
par = cm.parent_id.where(cm.parent_id.isin(cm.index)).dropna().astype("int64"); cur = pd.Series(cm.index.values, index=cm.index); dep = pd.Series(0, index=cm.index)
for _ in range(200):
    has = cur.isin(par.index)
    if not has.any(): break
    cur[has] = par.reindex(cur[has].values).values; dep[has.values] += 1
d["depth"] = dep.reindex(d.id).values; d["g0"] = score(c0d, "tox").reindex(d.id).values; d["g1"] = score(c1d, "tox").reindex(d.id).values
d = d.dropna(subset=["h_oc", "h_ic", "g0", "g1", "depth"]); d = d[d.depth >= 1].copy()
d["Y_hic"] = (d.h_ic >= .5).astype(int); d["Y_hoc"] = (d.h_oc >= .5).astype(int)
tc = float(np.quantile(d.g0, 1 - d.Y_hoc.mean())); d["Y_g_oc"] = (d.g0 > tc).astype(int); d["Y_g_ic"] = (d.g1 > tc).astype(int)
deep = (d.depth >= 3).values; one = (d.depth == 1).values; V = {c: d[c].values.astype(float) for c in ["Y_hic", "Y_hoc", "Y_g_oc", "Y_g_ic"]}
def st2(ix_):
    g = lambda x: x[ix_][deep[ix_]].mean() - x[ix_][one[ix_]].mean(); o = {f"gap_{c}": g(v) for c, v in V.items()}
    o["G2_gap_g_ic_minus_human_ic"] = o["gap_Y_g_ic"] - o["gap_Y_hic"]; o["gap_g_oc_minus_human_oc"] = o["gap_Y_g_oc"] - o["gap_Y_hoc"]; return o
b0 = st2(np.arange(len(d))); B2 = [st2(rng.integers(0, len(d), len(d))) for _ in range(2000)]
res["ccc"] = dict(n=int(len(d)), threshold=tc, mean_option_mass=[mc0, mc1], item=dict(kappa_oc_vs_hoc=kappa(d.Y_hoc.values, d.Y_g_oc.values), kappa_ic_vs_hic=kappa(d.Y_hic.values, d.Y_g_ic.values)),
                  estimates={k: [float(b0[k]), float(np.percentile([b[k] for b in B2], 2.5)), float(np.percentile([b[k] for b in B2], 97.5))] for k in b0})
res["G2"] = bool(res["ccc"]["estimates"]["G2_gap_g_ic_minus_human_ic"][1] > 0)
# ---------------- G3: Qian ----------------
q0d, mq0 = load("shard_*_hate_ctx0.jsonl"); q1d, mq1 = load("shard_*_hate_ctx1.jsonl")
M = pd.read_parquet(f"{W}/data/qian/qian_meta.parquet").set_index("id"); M["s0"] = score(q0d, "hate").reindex(M.index).values; s1q = score(q1d, "hate").reindex(M.index)
rep = M.parent >= 0; has1 = s1q.notna().values; res["qian"] = {"mean_option_mass": [mq0, mq1], "platforms": {}}
for plat, D in M.groupby("platform"):
    D = D.copy(); tq = float(np.quantile(D.s0.dropna(), 1 - D.z.mean())); D["y0"] = (D.s0 > tq).astype(float).where(D.s0.notna()).fillna(0.0)
    s1p = s1q.reindex(D.index); D["y1"] = np.where((D.parent >= 0) & s1p.notna(), (s1p > tq).astype(float), D.y0)
    conv = pd.factorize(D.conv)[0]; nC = conv.max() + 1; D["c"] = conv
    St = {}
    for src in ["z", "y0", "y1"]:
        St[f"d2_{src}"] = D[src].where(D.depth >= 2, 0).groupby(D.c).sum().values; St[f"d0_{src}"] = D[src].where(D.depth == 0, 0).groupby(D.c).sum().values
    St["n_d2"] = (D.depth >= 2).groupby(D.c).sum().values.astype(float); St["n_d0"] = (D.depth == 0).groupby(D.c).sum().values.astype(float)
    nm3 = list(St); A3 = np.column_stack([St[n] for n in nm3]); i3 = {n: i for i, n in enumerate(nm3)}
    def st3(Tt):
        g = lambda n: Tt[..., i3[n]]; o = {f"gap_{s}": g(f"d2_{s}") / g("n_d2") - g(f"d0_{s}") / g("n_d0") for s in ["z", "y0", "y1"]}
        o["G3_gap_y1_minus_gold"] = o["gap_y1"] - o["gap_z"]; o["gap_y0_minus_gold"] = o["gap_y0"] - o["gap_z"]; return o
    b3 = st3(A3.sum(0)); bb = {k: [] for k in b3}
    for _ in range(20):
        Wt = rng.poisson(1.0, size=(100, nC)).astype(float); sb = st3(Wt @ A3)
        for k in b3: bb[k].append(sb[k])
    ests = {k: [float(b3[k]), float(np.percentile(np.concatenate(v), 2.5)), float(np.percentile(np.concatenate(v), 97.5))] for k, v in bb.items()}
    k_ = D.groupby("c").size(); big = k_[k_ >= 2].index
    res["qian"]["platforms"][plat] = dict(threshold=tq, kappa=kappa(D.z.values, D.y0.values), prev=dict(z=float(D.z.mean()), y0=float(D.y0.mean()), y1=float(D.y1.mean())),
                                          share=dict(z=float(D[D.c.isin(big)].groupby("c").z.max().mean()), y0=float(D[D.c.isin(big)].groupby("c").y0.max().mean())), estimates=ests)
    Ub = D[D.c.isin(big)]; up = f"{WD}/units_qian_{plat}.parquet"; Ub.assign(unit=Ub.conv, s=Ub.s0.fillna(0.0), y=Ub.y0.astype(int))[["unit", "z", "s", "y"]].to_parquet(up)
res["G3"] = bool(all(v["estimates"]["G3_gap_y1_minus_gold"][1] > 0 for v in res["qian"]["platforms"].values()))
# ---------------- G4: THREADSAFE with Gemma guidance ----------------
I["k_art"] = I.groupby("article").z.transform("size")
for u, col, kc in [("threads", "root", "tk"), ("articles", "article", "k_art")]:
    X = I[I[kc] >= 2]; pd.DataFrame({"unit": X[col].values, "z": X.z.astype(int).values, "s": X.s0.values, "y": X.m0.astype(int).values}).to_parquet(f"{WD}/units_s5_{u}.parquet")
run = lambda cmd: subprocess.run(["python3", "-I"] + cmd, check=True, capture_output=True, text=True)
res["sims"] = {}
for nm in ["s5_threads", "s5_articles", "qian_reddit", "qian_gab"]:
    up = f"{WD}/units_{nm}.parquet"
    run([f"{W}/code/analysis/f98_threadsafe2_sim.py", up, f"{WD}/f98_{nm}.json", "--reps", str(REPS), "--budgets", "250,500,1000", "--methods", f"{FIN},cluster_ppi,cluster_ppi4" + (",ts2_a100cy" if FIN.endswith("x") else "")])
    res["sims"][nm] = json.load(open(f"{WD}/f98_{nm}.json"))
bc = lambda r, m: min(r[f"cluster_ppi@{m}"]["width_hyb"], r[f"cluster_ppi4@{m}"]["width_hyb"])
cov = [v["results"][f"{FIN}@{m}"]["coverage_hyb"] for v in res["sims"].values() for m in [250, 500, 1000]]
rat = {k: v["results"][f"{FIN}@500"]["width_hyb"] / bc(v["results"], 500) for k, v in res["sims"].items()}
res["G4"] = bool(np.mean(cov) >= .90 and min(cov) >= .85 and all(r <= 1.0 for r in rat.values())); res["G4_detail"] = dict(coverage=cov, ratio_500=rat)
json.dump(res, open(OUT, "w"), indent=1)
for k in ["G1a_logrr_m0_minus_gold", "G1b_slope_m1_minus_gold", "G1c_share_m1_minus_gold", "share_m0_minus_gold", "slope_m0_minus_gold", "slope_m1_minus_m0"]: print("s5", k, au[k])
print("ccc", res["ccc"]["estimates"]["G2_gap_g_ic_minus_human_ic"], {k: round(v, 3) for k, v in res["ccc"]["item"].items()})
for p, v in res["qian"]["platforms"].items(): print("qian", p, v["estimates"]["G3_gap_y1_minus_gold"], "kappa %.2f" % v["kappa"])
print("G4", json.dumps({k: round(v, 3) for k, v in rat.items()}), "coverage mean %.3f min %.3f" % (np.mean(cov), min(cov)))
print({k: res[k] for k in ["G1a", "G1b", "G1c", "G2", "G3", "G4"]})
