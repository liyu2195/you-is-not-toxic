"""Addendum 77, B77d: a corpus without second-person address (5% Civil Comments sample, results/x14_pops/cc_s5.parquet).
  build     every comment that the 'you' detector of x74_edit.py matches is rewritten with the H11 regex edit
            (f80_you_edit_lib.edit); originals (id 10*i) and edits (id 10*i + 1) go into one shuffled shard (seed 7731)
            results/x77_youfree/label_items/shard_000.jsonl, labeled by code/batch_manager.py (Haiku 5.5, task tox, no context).
  analyze   two instruments on the sample: comments without 'you' keep Haiku's existing labels (y:haiku) in both; comments
            with 'you' take the new batch's original label in one and its edit label in the other. Against gold: the depth law,
            the within-thread depth law, S_FP of the depth law and the thread size law, with 500 article-cluster Poisson bootstrap
            resamples (seed 7730), paired. H77d: the edited corpus's depth-law distortion minus the original's has CI below 0.
usage: python3 x77_youfree.py build TEXT_SHARD_DIR | analyze OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x77_youfree"; os.makedirs(D, exist_ok=True)
sys.path.insert(0, f"{W}/code/analysis")
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(np.int64).values
if sys.argv[1] == "build":
    from f80_you_edit_lib import edit as you_edit
    need = set(ids.tolist()); txt = {}
    for f in sorted(glob.glob(f"{sys.argv[2]}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"])
            if i in need:
                txt[i] = r.get("text") or ""
    print("texts", len(txt), "of", len(need))
    items = []; n_you = 0; n_changed = 0
    for i in ids:
        t = txt.get(int(i), "")
        if YOU.search(t):
            e = you_edit(t); n_you += 1; n_changed += e != t
            items += [dict(id=10 * int(i), text=t, parent_text=None), dict(id=10 * int(i) + 1, text=e, parent_text=None)]
    rng = np.random.default_rng(7731); order = rng.permutation(len(items)); os.makedirs(f"{D}/label_items", exist_ok=True)
    with open(f"{D}/label_items/shard_000.jsonl", "w") as fo:
        for k in order:
            fo.write(json.dumps(items[k]) + "\n")
    json.dump(dict(n_comments=len(ids), n_you=n_you, n_changed=n_changed, n_items=len(items)), open(f"{D}/build.json", "w"), indent=1)
    print("comments with 'you':", n_you, "changed by the edit:", n_changed, "items:", len(items))
else:
    OUT = sys.argv[2]
    R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{D}/haiku_tox/shard_*.jsonl")) for l in open(f)])
    if "error" in R:
        R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last"); R["flag"] = R.ans.isin(["A", "B"]).astype(float); R["cid"] = R.id // 10; R["var"] = R.id % 10
    Pv = R.pivot(index="cid", columns="var", values="flag")
    base = np.nan_to_num(P["y:haiku"].values.astype(float)); has = pd.Series(ids).isin(Pv.index).values
    o_ = Pv.reindex(ids)[0].values; e_ = Pv.reindex(ids)[1].values
    ok = has & np.isfinite(o_) & np.isfinite(e_)
    yo = np.where(ok, o_, base); ye = np.where(ok, e_, base); z = P.z.values.astype(float)
    cl = pd.factorize(P.cluster.values)[0]; nC = cl.max() + 1; x = np.minimum(P.depth.values, 10).astype(float)
    uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]
    ku = np.bincount(uu).astype(float); ucl = np.zeros(ku.size, np.int64); ucl[uu] = cl; okU = ku >= 2; xu = np.log2(np.maximum(ku, 1))
    Wb = np.vstack([np.ones(nC), np.random.default_rng(7730).poisson(1.0, size=(500, nC))])
    bc = lambda w: Wb @ np.bincount(cl, w, nC)
    def depth(v):
        n, sx, sy, sxx, sxy = bc(np.ones(len(v))), bc(x), bc(v), bc(x * x), bc(x * v); return (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    within = lambda v: bc(xt * v) / bc(xt * xt)
    def size(v):
        f = np.bincount(uu, v, ku.size) / ku; bu = lambda w: Wb @ np.bincount(ucl[okU], w[okU], nC)
        n, sx, sy, sxx, sxy = bu(np.ones(ku.size)), bu(xu), bu(f), bu(xu * xu), bu(xu * f); return (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    res = {"n_you_labeled": int(ok.sum()), "laws": {}}
    zsum = bc(z); nsum = bc(np.ones(len(z)))
    for nm, f in (("depth", depth), ("depth_within", within), ("size", size)):
        g = f(z); do = f(yo) - g; de = f(ye) - g
        r = dict(gold=ci(g), dist_orig=ci(do), dist_edit=ci(de), edit_minus_orig=ci(de - do), share_removed=float(1 - de[0] / do[0]))
        if nm in ("depth", "depth_within"):                     # S_FP = law((1 - Z) Y) + FPR law(Z)
            fo = bc((1 - z) * yo) / (nsum - zsum); fe = bc((1 - z) * ye) / (nsum - zsum)
            r["sfp_orig"] = ci(f((1 - z) * yo) + fo * g); r["sfp_edit"] = ci(f((1 - z) * ye) + fe * g)
        res["laws"][nm] = r
        print(nm, json.dumps({k: (np.round(v, 5).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r.items()}), flush=True)
    res["flag_rate"] = dict(orig=float(yo.mean()), edit=float(ye.mean()), gold_pos_orig=float(yo[z == 1].mean()), gold_pos_edit=float(ye[z == 1].mean()),
                            neg_orig=float(yo[z == 0].mean()), neg_edit=float(ye[z == 0].mean()))
    res["H77d"] = bool(res["laws"]["depth"]["edit_minus_orig"][2] < 0)
    print("flag rates", res["flag_rate"], "H77d", res["H77d"])
    json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
