"""Population-level audit of LLM labels on Civil Comments (all comments with labels from every annotator considered).
usage: f10_population.py OUT_JSON [--partial]   (annotator label globs are set in ANN below)
Units: threads (root + descendants via parent_id), articles, recipients; reply edges. Structural variables: thread size,
reply depth, position in the thread (time order). Each annotator gives a 0/1 label per comment, without context (ctx0) and
with the parent shown for replies (ctx1; roots keep their ctx0 label). Qwen also gets a prevalence-matched threshold."""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, glob, numpy as np, pandas as pd
from scipy.stats import kendalltau
OUT = sys.argv[1]; PARTIAL = "--partial" in sys.argv
W = _ROOT
import os
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "Z", "parent_id", "article_id", "publication_id", "created_date", "split"]].set_index("id")
if os.environ.get("CC_SPLIT", "train") != "all": meta = meta[meta.split == os.environ.get("CC_SPLIT", "train")]   # test splits: replies released without parents
meta = meta.drop(columns="split")
def read_letters(files):
    # an answered request without a letter (mostly empty comments) counts as 'not toxic'; errored requests stay missing
    R = [json.loads(l) for f in files for l in open(f)]
    R = pd.DataFrame(R)
    if "error" in R: R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last").set_index("id")
    return R.ans.isin(["A", "B"]).astype(np.int8)
def read_probs(files):
    ids, s = [], []
    for f in files:
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            ids.append(r["id"]); s.append(r["p"][0] + r["p"][1])
    return pd.Series(np.array(s), index=np.array(ids)).groupby(level=0).last()
ANN = {}
import os
FULL = os.environ.get("FULLDIR", f"{W}/results/full")
h0 = sorted(glob.glob(f"{FULL}/haiku_tox_ctx0/shard_*.jsonl")); h1 = sorted(glob.glob(f"{FULL}/haiku_tox_ctx1/shard_*.jsonl"))
q0 = sorted(glob.glob(f"{FULL}/qwen7b/*/shard_*_ctx0.jsonl")); q1 = sorted(glob.glob(f"{FULL}/qwen7b/*/shard_*_ctx1.jsonl"))
if h0: ANN["haiku"] = ("letters", h0, h1)
if q0: ANN["qwen7b"] = ("probs", q0, q1)
print("files:", {k: (len(v[1]), len(v[2])) for k, v in ANN.items()}, flush=True)
D = meta.copy()
for name, (kind, f0, f1) in ANN.items():
    if kind == "letters":
        y0 = read_letters(f0); y1 = read_letters(f1) if f1 else pd.Series(dtype=np.int8)
        D[name] = y0.reindex(D.index); c = y1.reindex(D.index); D[name + "_ctx"] = c.where(c.notna(), D[name])
        D.loc[D.parent_id.notna() & c.isna(), name + "_ctx"] = np.nan          # replies need their context label
    else:
        s0 = read_probs(f0); s1 = read_probs(f1) if f1 else pd.Series(dtype=float)
        sp = s0.reindex(D.index); t = float(np.nanquantile(sp, 1 - D.Z[sp.notna()].mean()))   # matched to gold prevalence
        D[name + "_m"] = (sp > t).astype(float).where(sp.notna()); c = s1.reindex(D.index)
        D[name + "_m_ctx"] = (c > t).astype(float).where(c.notna(), D[name + "_m"])
        D.loc[D.parent_id.notna() & c.isna(), name + "_m_ctx"] = np.nan
        D[name] = (sp > .5).astype(float).where(sp.notna())
LAB = ["Z"] + [c for c in D.columns if any(c == k or c.startswith(k + "_") for k in ANN)]
# --- thread structure on all comments: root, depth, size, time position
par = D.parent_id.where(D.parent_id.isin(D.index)).astype("Int64")
root = pd.Series(D.index.values, index=D.index); depth = pd.Series(0, index=D.index)
pmap = par.dropna().astype("int64")
cur = root.copy()
for _ in range(200):
    has = cur.isin(pmap.index)
    if not has.any(): break
    cur[has] = pmap.reindex(cur[has].values).values; depth[has] += 1
D["root"] = cur.values; D["depth"] = depth.values
D["tk"] = D.groupby("root").Z.transform("size")
# keep threads (and articles) whose every comment is labeled by every annotator
lab_ok = D[LAB].notna().all(axis=1)
ok_root = lab_ok.groupby(D.root).all(); ok_art = lab_ok.groupby(D.article_id).all()
res = {"n_comments_total": int(len(D)), "n_comments_labeled": int(lab_ok.sum()), "n_threads_ok": int(ok_root.sum()), "n_articles_ok": int(ok_art.sum()), "labels": LAB}
T = D[D.root.map(ok_root) & (D.tk >= 2)].copy()
T = T.sort_values("created_date"); T["pos"] = T.groupby("root").cumcount() + 1
res["n_threads_k2"] = int(T.root.nunique()); res["n_comments_in_threads"] = int(len(T))
T[["root", "article_id", "depth", "tk"] + LAB].to_parquet(OUT.rsplit(".", 1)[0] + "_thread_items.parquet")   # for the H3 bootstrap (f13)
# item level (all labeled comments)
L = D[lab_ok]
def item(z, y):
    z = np.asarray(z); y = np.asarray(y); po = (z == y).mean(); pe = z.mean() * y.mean() + (1 - z.mean()) * (1 - y.mean())
    return dict(prev=float(y.mean()), fpr=float(y[z == 0].mean()), tpr=float(y[z == 1].mean()), kappa=float((po - pe) / (1 - pe)))
res["item"] = {c: item(L.Z, L[c]) for c in LAB if c != "Z"}; res["item"]["gold_prev"] = float(L.Z.mean())
# (1) escalation: toxic share by thread size (log2 bins) and depth; slope on log size; Mann-Kendall over bins
edges = [2, 3, 4, 6, 9, 14, 22, 34, 54, 86, 140, 10**6]
T["sbin"] = pd.cut(T.tk, edges, right=False, labels=False)
esc = {}
for c in LAB:
    g = T.groupby("sbin")[c].mean(); X = np.log(T.tk.values); y = T[c].values
    sl = float(np.cov(X, y)[0, 1] / np.var(X, ddof=1)); tau, p = kendalltau(g.index.values, g.values)
    gd = T.groupby(T.depth.clip(upper=10))[c].mean(); tau_d, p_d = kendalltau(gd.index.values, gd.values)
    sld = float(np.cov(T.depth.clip(upper=10), y)[0, 1] / np.var(T.depth.clip(upper=10), ddof=1))
    esc[c] = dict(by_size={int(k): round(float(v), 4) for k, v in g.items()}, slope_logsize=sl, mk_tau=float(tau), mk_p=float(p),
                  by_depth={int(k): round(float(v), 4) for k, v in gd.items()}, slope_depth=sld, mk_depth_tau=float(tau_d), mk_depth_p=float(p_d))
res["escalation"] = esc; res["size_bin_edges"] = edges
# (1b) Avalle-style: conversations = an article's whole comment section; toxic share by log-binned conversation size
A_ = D[D.article_id.map(ok_art)].copy(); A_["ak"] = A_.groupby("article_id").Z.transform("size")
aedges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 10**6]; A_["abin"] = pd.cut(A_.ak, aedges, right=False, labels=False)
aesc = {}
for c in LAB:
    g = A_.groupby("abin")[c].mean(); X = np.log(A_.ak.values); y = A_[c].values
    tau, p = kendalltau(g.index.values, g.values)
    aesc[c] = dict(by_size={int(k): round(float(v), 4) for k, v in g.items()}, slope_logsize=float(np.cov(X, y)[0, 1] / np.var(X, ddof=1)), mk_tau=float(tau), mk_p=float(p))
res["escalation_articles"] = aesc; res["article_bin_edges"] = aedges
res["n_comments_by_article_bin"] = {int(k): int(v) for k, v in A_.groupby("abin").size().items()}
res["n_by_size_bin"] = {int(k): int(v) for k, v in T.groupby("sbin").size().items()}
# by publication (8 largest): Mann-Kendall on size bins and slope, per label
pubs = D.publication_id.value_counts().index[:8].tolist(); bp = {}
for p_ in pubs:
    Tp = T[T.publication_id == p_]; bp[int(p_)] = {"n_threads": int(Tp.root.nunique())}
    for c in LAB:
        g = Tp.groupby("sbin")[c].mean(); X = np.log(Tp.tk.values); y = Tp[c].values
        tau, p = kendalltau(g.index.values, g.values) if len(g) > 2 else (np.nan, np.nan)
        bp[int(p_)][c] = dict(slope=float(np.cov(X, y)[0, 1] / np.var(X, ddof=1)), mk_tau=float(tau), mk_p=float(p))
res["escalation_by_publication"] = bp
# (2) derailment curve and onset hazard by position
der = {}
for c in LAB:
    cm = T.groupby("root")[c].cummax(); prev = cm.groupby(T.root).shift(1).fillna(0)
    d = {}
    for t in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        m = T.pos == t
        if m.sum() < 200: continue
        d[t] = dict(derailed=float(cm[m].mean()), hazard=float(T.loc[m & (prev == 0), c].mean()), n=int(m.sum()))
    der[c] = d
res["derailment"] = der
# (3) contagion on reply edges (both ends labeled; child label ctx/no-ctx as named, parent label without context)
E = D[lab_ok & D.parent_id.isin(D.index[lab_ok])].copy(); E["pid"] = E.parent_id.astype("int64")
con = {}
for c in LAB:
    base = c.replace("_ctx", "")
    zp = D[base].reindex(E.pid).values; zc = E[c].values
    p1 = zc[zp == 1].mean(); p0 = zc[zp == 0].mean(); con[c] = dict(p1=float(p1), p0=float(p0), diff=float(p1 - p0), rr=float(p1 / p0), n=int(len(E)))
res["contagion"] = con
# (4) 'at least one' shares: articles, threads, recipients; and publication comparisons
A = D[D.article_id.map(ok_art)]
sh = {}
for c in LAB:
    art = A.groupby("article_id")[c].max(); thr = T.groupby("root")[c].max()
    rec = E.groupby("pid")[c].max(); a2 = A.groupby("article_id")[c].size() >= 2
    sh[c] = dict(articles=float(art.mean()), threads=float(thr.mean()), recipients=float(rec.mean()),
                 articles_k2=float(art[a2].mean()))          # articles with >= 2 comments (the THREADSAFE evaluation units)
res["or_shares"] = sh
pubsh = {}
for c in LAB:
    pubsh[c] = {int(p_): dict(articles=float(A[A.publication_id == p_].groupby("article_id")[c].max().mean()),
                              threads=float(T[T.publication_id == p_].groupby("root")[c].max().mean()),
                              item=float(A.loc[A.publication_id == p_, c].mean())) for p_ in pubs}
res["or_by_publication"] = pubsh
flips = {}
for c in LAB:
    if c == "Z": continue
    f = {}
    for m in ["articles", "threads", "item"]:
        n_mat = n_fl = 0
        for i in range(len(pubs)):
            for j in range(i + 1, len(pubs)):
                dg = pubsh["Z"][int(pubs[i])][m] - pubsh["Z"][int(pubs[j])][m]; dl = pubsh[c][int(pubs[i])][m] - pubsh[c][int(pubs[j])][m]
                if abs(dg) >= .02: n_mat += 1; n_fl += int(np.sign(dg) != np.sign(dl))
        f[m] = [n_fl, n_mat]
    flips[c] = f
res["publication_flips"] = flips
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["n_comments_labeled", "n_threads_k2", "n_articles_ok"]}))
for c in LAB:
    e = esc[c]; print(f"{c:12s} slope/logsize {e['slope_logsize']:+.4f} MK p {e['mk_p']:.3g} | slope/depth {e['slope_depth']:+.4f} | contagion RR {con[c]['rr']:.2f} | "
                      f"threads>=1 {sh[c]['threads']:.3f} articles {sh[c]['articles']:.3f} | derailed@5 {der[c].get(5, {}).get('derailed', float('nan')):.3f}")
