"""Addendum 57: item tables for the upgraded audit, built exactly as x08_placebo.py builds its datasets (same labelers,
flags and thresholds), saved to results/x09_pops/<name>.parquet: id, unit, cluster, depth, parent_row (-1 if none), z, k (unit
size) and one column 'y:<labeler>' per labeler; Civil Comments also 'article' and 'tox'.
usage: python3 x09_build_pops.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, glob, numpy as np, pandas as pd
W = _ROOT; rng = np.random.default_rng(55)
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

def edges_from(D, parent_col, key=None):
    idx = pd.Series(np.arange(len(D)), index=(D[key].values if key else D.index))
    par = D[parent_col]; m = par.notna() & par.isin(idx.index)
    ch, pa = np.flatnonzero(m.values), idx.reindex(par[m].values).values.astype(int)
    keep = pa != ch                                   # Addendum 72: drop self-replies (CGA-Wiki: 84 utterances reply to themselves)
    return ch[keep], pa[keep]

RES = {}
OUTD = f"{W}/results/x09_pops"; os.makedirs(OUTD, exist_ok=True)
def save(name, D, labs, unit, clus, depth, edges, extra=()):
    n = len(D); pr = np.full(n, -1, dtype=np.int64); pr[edges[0]] = edges[1]
    out = pd.DataFrame({"id": D.index.values, "unit": pd.factorize(D[unit])[0], "cluster": pd.factorize(D[clus])[0],
                        "depth": np.asarray(D[depth].values, dtype=np.int64), "parent_row": pr, "z": D.z.values.astype(np.int8)})
    out["k"] = out.groupby("unit").unit.transform("size").values
    for c in extra: out[c] = D[c].values
    for L in labs: out["y:" + L] = D[L].values.astype(float)
    out.to_parquet(f"{OUTD}/{name}.parquet"); META[name] = dict(n=int(n), labelers=list(labs), n_edges=int(len(edges[0])), units=int(out.unit.nunique()))
    print(name, META[name], flush=True)
META = {}

# ---------------------------------------------------------------- Civil Comments threads (all) and the 5% sample
T = pd.read_parquet(f"{W}/results/f10_population_thread_items.parquet")
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "tox", "parent_id", "created_date"]).set_index("id")
T["tox"] = meta.tox.reindex(T.index).values; T["parent"] = meta.parent_id.reindex(T.index).values; T["z"] = T.Z
T["human_lenient"] = np.where(T.z == 1, 1.0, (T.tox >= 0.3).astype(float))
labs = ["haiku", "haiku_ctx", "qwen7b_m", "qwen7b_m_ctx", "qwen7b"]
O = open_scores([f"cc_{i:03d}" for i in range(80)]); labs += add_open(T, O, float(T.z.mean()))
T["pos"] = meta.created_date.reindex(T.index).values.astype("datetime64[s]").astype(np.int64)
E = edges_from(T, "parent")
labs = [l for l in labs if l in T]
T["article"] = pd.factorize(T.article_id)[0]
save("cc_threads", T, labs, "root", "article_id", "depth", E, extra=("article", "tox"))
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
T5["article"] = pd.factorize(T5.article_id)[0]
save("cc_s5", T5, labs5, "root", "article_id", "depth", E5, extra=("article", "tox"))
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
save("cad", g, [l for l in labc if l in g], "thread", "thread", "depth", Ec)
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
    save(f"qian_{plat}", Q, [l for l in labq if l in Q], "conv", "conv", "depth", Eq)
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
save("cga", C, [l for l in labg if l in C], "conv", "conv", "depth", Eg)
# ---------------------------------------------------------------- BAD (bot and human turns; parent = previous turn)
B = pd.read_parquet(f"{W}/data/bad_meta.parquet").set_index("id"); B["z"] = B.unsafe.astype(int)
Rb = pd.concat([pd.DataFrame([json.loads(l) for l in open(f)]) for f in [f"{W}/results/bad/haiku_unsafe_ctx1.jsonl", f"{W}/results/bad/haiku_unsafe_ctx0_first.jsonl"]])
if "error" in Rb: Rb = Rb[Rb["error"].isna()]
sb = Rb.drop_duplicates("id", keep="last").set_index("id").ans.map({"A": 1.0, "B": 2 / 3, "C": 1 / 3, "D": 0.0}).reindex(B.index)
B["haiku_ctx"] = (sb >= 2 / 3).astype(float).where(sb.notna())
labb = ["haiku_ctx"] + add_open(B, open_scores(["bad"]), float(B.z.mean()))
B["depth"] = B.pos - 1; B = B.sort_values(["dlg", "pos"]); B["prev"] = B.groupby("dlg").apply(lambda x: pd.Series(x.index.values, index=x.index).shift(1)).reset_index(level=0, drop=True).reindex(B.index)
Eb = edges_from(B, "prev")
save("bad", B, [l for l in labb if l in B], "dlg", "dlg", "depth", Eb)
json.dump(META, open(f"{OUTD}/meta.json", "w"), indent=1); print("saved", OUTD)
