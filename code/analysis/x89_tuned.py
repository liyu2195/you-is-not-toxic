"""Addendum 89, B89b: a tuned Haiku.
  build     development pool: Civil Comments reply-thread comments in articles outside the 5% sample. Development set: 2,000
            comments, uniform (seed 8902). Few-shot examples from the rest of the pool (seed 8903), half gold positives (shown as
            "B"), half with a vote fraction below 0.2 (shown as "D"), in random order; T16's first 8 are T8's.
            Candidates (main prompt, system prompt and options): T0 zero-shot; T8 and T16 with 8 and 16 examples as earlier
            user/assistant turns (cache_control on the last example); TR zero-shot with adaptive thinking at effort 'high'
            (max_tokens 2,048; Addendum 89 run log: 'enabled' thinking with a token budget is not supported for Haiku 5.5). Requests for the development set and the whole 5% sample -> results/x89/tuned/req/req_*.jsonl
            (custom_id <cand>_<d|e>_<comment id>).
  analyze OUT_JSON   (after x89_batch.py) development kappa per cut (A; A-B; A-C) and candidate; selection; on the 5% sample at the
            development cut: kappa, TPR, FPR, the depth law and its distortion, S_FP, the placebo depth slope, the within-thread
            distortion; 500 article-cluster resamples (seed 8904).
usage: python3 x89_tuned.py build TEXT_SHARD_DIR | analyze OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, glob, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x89/tuned"; os.makedirs(f"{D}/req", exist_ok=True)
sys.path.insert(0, f"{W}/code/labeling")
from prompts import SYSTEM, tox_prompt
MODEL = "claude-haiku-5-5"; CANDS = ("T0", "T8", "T16", "TR")
if sys.argv[1] == "build":
    A = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "z", "tox"])
    S = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet", columns=["id"])
    art = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "article_id"]).set_index("id").article_id   # real article ids
    A["art"] = art.reindex(A.id.values).values; S["art"] = art.reindex(S.id.values).values
    pool = A[~A.art.isin(set(S.art))].reset_index(drop=True); assert not pool.id.isin(S.id).any()
    rng = np.random.default_rng(8902); dev = pool.iloc[rng.choice(len(pool), 2000, replace=False)]
    rest = pool[~pool.id.isin(set(dev.id))]
    r2 = np.random.default_rng(8903)
    pos = rest[rest.z == 1].sample(8, random_state=r2.integers(2**31)).id.tolist(); civ = rest[rest.tox < 0.2].sample(8, random_state=r2.integers(2**31)).id.tolist()
    ex16 = [(i, "B") for i in pos] + [(i, "D") for i in civ]
    first8 = [ex16[j] for j in (0, 1, 2, 3, 8, 9, 10, 11)]; rest8 = [e for e in ex16 if e not in first8]
    o8 = r2.permutation(8); first8 = [first8[j] for j in o8]; o16 = r2.permutation(8); ex16 = first8 + [rest8[j] for j in o16]
    need = set(dev.id.astype(int)) | {int(i) for i, _ in ex16} | set(S.id.astype(int))
    T = {}
    for f in sorted(glob.glob(f"{sys.argv[2]}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"])
            if i in need:
                T[i] = r.get("text") or ""
    print("texts", len(T), "of", len(need), flush=True)
    def shots(ex):
        msgs = []
        for i, lab in ex:
            msgs += [dict(role="user", content=tox_prompt(T[int(i)])), dict(role="assistant", content=lab)]
        msgs[-1] = dict(role="assistant", content=[dict(type="text", text=msgs[-1]["content"], cache_control={"type": "ephemeral"})])
        return msgs
    SH = {"T0": [], "T8": shots(first8), "TR": [], "T16": shots(ex16)}
    json.dump(dict(dev=dev.id.astype(int).tolist(), ex8=[[int(i), l] for i, l in first8], ex16=[[int(i), l] for i, l in ex16]), open(f"{D}/build.json", "w"))
    per = {"T0": 30000, "TR": 30000, "T8": 15000, "T16": 8000}; nf = 0
    for c in CANDS:
        rows = []
        for tag, ids in (("d", dev.id.astype(int).tolist()), ("e", S.id.astype(int).tolist())):
            for i in ids:
                p = dict(model=MODEL, max_tokens=5, system=SYSTEM, thinking={"type": "disabled"}, messages=SH[c] + [dict(role="user", content=tox_prompt(T.get(i, "")))])
                if c == "TR":
                    p.update(max_tokens=2048, thinking={"type": "adaptive"}, output_config={"effort": "high"})
                rows.append({"custom_id": f"{c}_{tag}_{i}", "params": p})
        for j in range(0, len(rows), per[c]):
            with open(f"{D}/req/req_{c}_{j // per[c]:03d}.jsonl", "w") as fo:
                for r in rows[j: j + per[c]]:
                    fo.write(json.dumps(r) + "\n")
            nf += 1
        print(c, len(rows), "requests", flush=True)
    print("files", nf)

if sys.argv[1] == "analyze":
    import re as _re
    OUT = sys.argv[2]; NB = 500
    A = {}
    for f in sorted(glob.glob(f"{D}/out/req_*.out.jsonl")) + sorted(glob.glob(f"{D}/out_retry/req_*.out.jsonl")):   # retries fill errored rows
        for l in open(f):
            r = json.loads(l)
            if r.get("error") is None or r["custom_id"] not in A:
                A[r["custom_id"]] = r["ans"]
    b = json.load(open(f"{D}/build.json")); dev = b["dev"]
    G = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["id", "z"]).set_index("id").z
    zd = G.reindex(dev).values.astype(float)
    LEV = {"A": 3, "B": 2, "C": 1, "D": 0}
    P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(int).values; N = len(P)
    z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1)
    uu = pd.factorize(P.unit.values)[0]; xt = x - (np.bincount(uu, x) / np.bincount(uu))[uu]; neg = z == 0
    YOU = _re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", _re.I)
    T = {}
    for f in sorted(glob.glob(f"{W}/data/second/s5/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); T[int(r["id"])] = r.get("text") or ""
    you = np.array([bool(YOU.search(T.get(int(i), ""))) for i in ids])
    Wb = np.vstack([np.ones(nC), np.random.default_rng(8904).poisson(1.0, size=(NB, nC))])
    S = lambda v: Wb @ np.bincount(cl, v, nC)
    def slope(xv, v, mask):
        m_ = mask.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones_like(xv), xv, v, xv * xv, xv * v))
        return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
    rate = lambda v, m: S(v * m) / S(m.astype(float))
    ci = lambda a: [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
    def kappa(y, zz):
        po = float((y == zz).mean()); p1, p2 = y.mean(), zz.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")
    res = {"candidates": {}}; allm = np.ones(N, bool); g = slope(x, z, allm); gw = slope(xt, z, allm)
    for c in CANDS:
        ld = np.array([LEV.get(A.get(f"{c}_d_{i}"), -1) for i in dev]); le = np.array([LEV.get(A.get(f"{c}_e_{i}"), -1) for i in ids])
        if (ld >= 0).sum() == 0:
            continue
        kd = {cut: kappa((ld >= cut).astype(float), zd) for cut in (3, 2, 1)}; best = max(kd, key=lambda k_: (kd[k_], k_))
        y = (le >= best).astype(float); l_ = slope(x, y, allm); TPR, FPR = float(y[z == 1].mean()), float(y[neg].mean())
        w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
        res["candidates"][c] = dict(dev_kappa=kd, cut=int(best), dev_missing=int((ld < 0).sum()), eval_missing=int((le < 0).sum()), kappa=kappa(y, z), flag_rate=float(y.mean()),
                                    tpr=TPR, fpr=FPR, depth=ci(l_), distortion=ci(l_ - g), S_FP=100 * float(w[neg] @ (y[neg] - FPR)), placebo=ci(slope(x, y, neg)),
                                    within_distortion=ci(slope(xt, y, allm) - gw), fpr_you_minus_other=ci(rate(y, neg & you) - rate(y, neg & ~you)))
        v = res["candidates"][c]
        print(c, "dev kappa", {k_: round(t, 3) for k_, t in kd.items()}, "cut", best, "kappa", round(v["kappa"], 3), "flag", round(v["flag_rate"], 3), "distortion", [round(t, 2) for t in v["distortion"]],
              "placebo", [round(t, 2) for t in v["placebo"]], "you gap", [round(t, 3) for t in v["fpr_you_minus_other"]], "missing", v["eval_missing"], flush=True)
    sel = max(res["candidates"], key=lambda c: res["candidates"][c]["dev_kappa"][res["candidates"][c]["cut"]])
    res["selected"] = sel; res["H89c"] = bool(res["candidates"][sel]["distortion"][1] > 0); res["gold_depth"] = ci(g)
    print("selected", sel, "H89c", res["H89c"]); json.dump(res, open(OUT, "w"), indent=1)
