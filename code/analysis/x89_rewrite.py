"""Addendum 89, B89c: the deterministic rewrite (B77d's fixed word-substitution rule, f80_you_edit_lib.edit) with Sonnet and with
the parent shown, on the 5% Civil Comments sample.
  build TEXT_SHARD_DIR
    (i)  Sonnet 5.5 without context: every comment that the 'you' detector matches, original (o_<id>) and edited (e_<id>), shuffled
         together (seed 8906) -> results/x89/rewrite/req_s/req_*.jsonl.
    (ii) Haiku 5.5 and Sonnet 5.5 shown the parent: every reply whose text or parent matches the detector. V0 original reply and
         parent (v0_<id>); V1 edited reply, original parent (v1_<id>; only when the reply matches, otherwise V1 = V0); V2 edited
         reply and edited parent (v2_<id>; only when the parent matches, otherwise V2 = V1). Shuffled together (seed 8907)
         -> results/x89/rewrite/req_c/req_*.jsonl.
    Settings as the main labels: max_tokens 5, the main system prompt, thinking disabled for Haiku and between_tools for Sonnet.
  analyze OUT_JSON  (after x89_batch.py)
    (i)  as B77d: comments without 'you' keep Sonnet's stored labels (y:sonnet) in both instruments; comments with 'you' take this
         batch's original label in one and its edited label in the other. Depth law, within-thread depth law, distortions and
         their paired change; 500 article-cluster resamples (seed 8905). H89d: the change in the depth-law distortion has CI < 0.
    (ii) the with-parent instrument (replies: context labels; roots: the stored no-context labels): replies without 'you' in the
         reply or the parent keep the stored context labels; the others take V0, V1 or V2 from this batch. H89e (Haiku): V1 - V0
         change in the depth-law distortion has CI < 0.
usage: python3 x89_rewrite.py build TEXT_SHARD_DIR | analyze OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x89/rewrite"
sys.path.insert(0, f"{W}/code/labeling"); sys.path.insert(0, f"{W}/code/analysis")
from prompts import SYSTEM, tox_prompt
from f80_you_edit_lib import edit
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
THINK = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}


def req(cid, model, text, parent=None):
    return {"custom_id": cid, "params": dict(model=model, max_tokens=5, system=SYSTEM, thinking=THINK[model],
                                             messages=[dict(role="user", content=tox_prompt(text, parent))])}


def write(rows, d, per=30000):
    os.makedirs(d, exist_ok=True)
    for j in range(0, len(rows), per):
        with open(f"{d}/req_{j // per:03d}.jsonl", "w") as fo:
            for r in rows[j: j + per]:
                fo.write(json.dumps(r) + "\n")


if sys.argv[1] == "build":
    T = {}
    for f in sorted(glob.glob(f"{sys.argv[2]}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); T[int(r["id"])] = (r.get("text") or "", r.get("parent_text"))
    P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet", columns=["id", "parent_row"]); ids = P.id.astype(int).values; pr = P.parent_row.values
    yc = np.array([bool(YOU.search(T[i][0])) for i in ids]); yp = np.array([bool(T[i][1]) and bool(YOU.search(T[i][1])) for i in ids])
    rows = []
    for i in ids[yc]:
        t = T[i][0]; rows += [req(f"o_{i}", "claude-sonnet-5-5", t), req(f"e_{i}", "claude-sonnet-5-5", edit(t))]
    rows = [rows[j] for j in np.random.default_rng(8906).permutation(len(rows))]; write(rows, f"{D}/req_s")
    print("(i) Sonnet requests", len(rows), "comments", int(yc.sum()), flush=True)
    rows = []; rep = pr >= 0
    for m in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        tag = m.split("-")[1][0]
        for j in np.flatnonzero(rep & (yc | yp)):
            i = int(ids[j]); t, p = T[i]; p = p if p else None             # no parent text: no-context prompt, as batch_manager.py
            rows.append(req(f"{tag}v0_{i}", m, t, p))
            if yc[j]:
                rows.append(req(f"{tag}v1_{i}", m, edit(t), p))
            if yp[j]:
                rows.append(req(f"{tag}v2_{i}", m, edit(t), edit(p) if p else None))
    rows = [rows[j] for j in np.random.default_rng(8907).permutation(len(rows))]; write(rows, f"{D}/req_c")
    print("(ii) requests", len(rows), "replies", int((rep & (yc | yp)).sum()), "child", int((rep & yc).sum()), "parent", int((rep & yp).sum()), flush=True)
    json.dump(dict(n_you=int(yc.sum()), n_rep=int((rep & (yc | yp)).sum())), open(f"{D}/build.json", "w"))

if sys.argv[1] == "analyze":
    OUT = sys.argv[2]; NB = 500
    A = {}
    for sub in ("out_s", "out_c"):
        for f in sorted(glob.glob(f"{D}/{sub}/req_*.out.jsonl")):
            for l in open(f):
                r = json.loads(l); A[r["custom_id"]] = r["ans"]
    flag = lambda a: float(a in ("A", "B"))
    P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(int).values; N = len(P); pr = P.parent_row.values
    z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
    uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
    Wb = np.vstack([np.ones(nC), np.random.default_rng(8905).poisson(1.0, size=(NB, nC))])
    S = lambda v: Wb @ np.bincount(cl, v, nC)
    def slope(xv, v, mask):
        m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones_like(xv), xv, v, xv * xv, xv * v))
        return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
    def size_law_b(v):                                                   # thread size law per resample (units weighted by cluster weights)
        ku = np.bincount(uu).astype(float); ok = ku >= 2; fu = np.bincount(uu, v) / ku; lx = np.log2(ku); ucl = np.zeros(ku.size, int); ucl[uu] = cl
        w = Wb[:, ucl[ok]]; lxo, fuo = lx[ok], fu[ok]
        sw = w.sum(1); mx = (w * lxo).sum(1) / sw; my = (w * fuo).sum(1) / sw
        return 100 * (w * (lxo - mx[:, None]) * (fuo - my[:, None])).sum(1) / (w * (lxo - mx[:, None]) ** 2).sum(1)
    ci = lambda a: [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
    allm = np.ones(N, bool); g = slope(x, z, allm); gw = slope(xt, z, allm); gs = size_law_b(z); gr = slope(x, z, x >= 1)
    res = {}
    # (i) Sonnet without context, as B77d
    y_st = np.nan_to_num(P["y:sonnet"].values.astype(float)); have = np.array([f"o_{i}" in A for i in ids])
    yo = y_st.copy(); ye = y_st.copy()
    for j in np.flatnonzero(have):
        yo[j] = flag(A[f"o_{ids[j]}"]); ye[j] = flag(A[f"e_{ids[j]}"])
    do, de = slope(x, yo, allm) - g, slope(x, ye, allm) - g; wo, we = slope(xt, yo, allm) - gw, slope(xt, ye, allm) - gw; so, se_ = size_law_b(yo) - gs, size_law_b(ye) - gs
    res["sonnet_noctx"] = dict(n_you=int(have.sum()), dist_orig=ci(do), dist_edit=ci(de), change=ci(de - do), share_removed=float(1 - de[0] / do[0]),
                               within_orig=ci(wo), within_edit=ci(we), size_orig=ci(so), size_edit=ci(se_), size_share_removed=float(1 - se_[0] / so[0]))
    res["H89d"] = bool(res["sonnet_noctx"]["change"][2] < 0)
    print("(i)", res["sonnet_noctx"], "H89d", res["H89d"], flush=True)
    # (ii) shown the parent: V0, V1, V2 for replies whose text or parent addresses 'you'
    for tag, model in (("h", "haiku"), ("s", "sonnet")):
        y_ctx = np.nan_to_num(P[f"y:{model}_ctx"].values.astype(float)); V = {k: y_ctx.copy() for k in ("v0", "v1", "v2")}; n_aff = 0
        for j in range(N):
            i = ids[j]; k0 = f"{tag}v0_{i}"
            if pr[j] < 0 or k0 not in A:
                continue
            n_aff += 1; f0 = flag(A[k0]); f1 = flag(A[f"{tag}v1_{i}"]) if f"{tag}v1_{i}" in A else f0; f2 = flag(A[f"{tag}v2_{i}"]) if f"{tag}v2_{i}" in A else f1
            V["v0"][j], V["v1"][j], V["v2"][j] = f0, f1, f2
        rep = x >= 1; d = {k: slope(x, v, allm) - g for k, v in V.items()}; dr = {k: slope(x, v, rep) - gr for k, v in V.items()}
        res[f"{model}_ctx"] = dict(n_replies=n_aff, dist={k: ci(v) for k, v in d.items()}, dist_replies={k: ci(v) for k, v in dr.items()},
                                   change_v1=ci(d["v1"] - d["v0"]), change_v2=ci(d["v2"] - d["v0"]), change_v1_replies=ci(dr["v1"] - dr["v0"]), change_v2_replies=ci(dr["v2"] - dr["v0"]),
                                   share_removed_v1_replies=float(1 - dr["v1"][0] / dr["v0"][0]), share_removed_v2_replies=float(1 - dr["v2"][0] / dr["v0"][0]))
        print(f"(ii) {model}", {k: v for k, v in res[f"{model}_ctx"].items() if k.startswith(("change", "share", "n_"))}, flush=True)
    res["H89e"] = bool(res["haiku_ctx"]["change_v1"][2] < 0)
    nb = json.load(open(f"{D}/build.json"))["n_rep"]; res["_complete"] = bool(res["haiku_ctx"]["n_replies"] >= 0.99 * nb and res["sonnet_ctx"]["n_replies"] >= 0.99 * nb)
    print("H89e", res["H89e"]); json.dump(res, open(OUT, "w"), indent=1)
