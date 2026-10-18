"""Addendum 91, B91: can a prompt remove the second-person bias? Haiku 5.5 without the parent on the 5% Civil Comments sample (60,668
comments), the main settings (max_tokens 5, thinking disabled), two variants of the main prompt (code/prompts.py):
  M1 instruction: the main system prompt and question, with one added sentence before the comment:
     'Addressing someone directly, for example as "you", is not by itself toxic; judge what the comment says to or about them.'
  M2 third-person framing: no second person in the instructions (system prompt 'This task rates comments posted under online news
     articles for toxicity.'; the scale says 'readers' where the raters' wording says 'you'), and a note before the comment:
     'The comment was written by one reader of the article. Any "you" in it addresses another reader, not the person rating it.'
  build TEXT_SHARD_DIR -> results/x91/req/req_<M>_*.jsonl (custom_id <M>_<comment id>), run with scripts/x89_batch.py
  analyze OUT_JSON: for M1, M2, the B89b zero-shot rerun T0 (results/x89/tuned, same week) and the stored main labels (y:haiku):
     flag rate, kappa, FPR; the depth law, its distortion (500 article-cluster resamples, seed 9101); the placebo depth slope; the 'you'
     gap; the FPR by number of second-person words (0..4+) and the 4+:0 ratio; the placebo slope and the law's distortion at a fixed
     number; the paired change in distortion of M1 and M2 against T0.
  H91a: under M1 Haiku's depth-law distortion has CI above 0. H91b: the same under M2. (Descriptive otherwise.)
  Added after the v5b verification (descriptive): the same statistics for B89b's other candidates (T8, T16, TR; flags at their selected
  cut, toxic or very toxic), the paired changes against T0 of the distortion at a fixed number, and of the false-positive rate at the root
  and among replies.
usage: python3 x91_mitigate.py build TEXT_SHARD_DIR | analyze OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x91"; os.makedirs(f"{D}/req", exist_ok=True)
sys.path.insert(0, f"{W}/code/labeling")
from prompts import SYSTEM, TOX_SCALE, _q
MODEL = "claude-haiku-5-5"
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
M1_NOTE = 'Addressing someone directly, for example as "you", is not by itself toxic; judge what the comment says to or about them.'
M2_SYSTEM = "This task rates comments posted under online news articles for toxicity."
M2_SCALE = TOX_SCALE.replace("make you leave", "make readers leave")
M2_NOTE = 'The comment was written by one reader of the article. Any "you" in it addresses another reader, not the person rating it.'
assert M2_SCALE != TOX_SCALE and "you" not in M2_SCALE.lower().replace("you\"", "")


def m1_prompt(text):
    return (f"Rate the toxicity of the comment below.\n\n{TOX_SCALE}\n\n{M1_NOTE}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\n"
            "Answer with a single letter (A, B, C or D).")


def m2_prompt(text):
    return (f"Rate the toxicity of the comment below.\n\n{M2_SCALE}\n\n{M2_NOTE}\n\nComment:\n\"\"\"{_q(text)}\"\"\"\n\n"
            "Answer with a single letter (A, B, C or D).")


if sys.argv[1] == "build":
    S = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet", columns=["id"]); ids = S.id.astype(int).tolist(); need = set(ids); T = {}
    for f in sorted(glob.glob(f"{sys.argv[2]}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"])
            if i in need:
                T[i] = r.get("text") or ""
    print("texts", len(T), "of", len(need))
    rng = np.random.default_rng(9100)
    for tag, sysp, fn in (("M1", SYSTEM, m1_prompt), ("M2", M2_SYSTEM, m2_prompt)):
        order = rng.permutation(len(ids)); rows = []
        for j in order:
            i = ids[j]
            rows.append({"custom_id": f"{tag}_{i}", "params": dict(model=MODEL, max_tokens=5, system=sysp, thinking={"type": "disabled"},
                                                                     messages=[dict(role="user", content=fn(T.get(i, "")))])})
        for k in range(0, len(rows), 30000):
            with open(f"{D}/req/req_{tag}_{k // 30000:03d}.jsonl", "w") as fo:
                for r in rows[k: k + 30000]:
                    fo.write(json.dumps(r) + "\n")
        print(tag, len(rows))
    print("example M1:\n", m1_prompt("You are missing the point."), "\nexample M2 system:", M2_SYSTEM, "\n", m2_prompt("You are missing the point."))

if sys.argv[1] == "analyze":
    OUT = sys.argv[2]; NB = 500
    A = {}
    for f in (sorted(glob.glob(f"{D}/out/req_*.out.jsonl")) + sorted(glob.glob(f"{W}/results/x89/tuned/out/req_*.out.jsonl"))
              + sorted(glob.glob(f"{W}/results/x89/tuned/out_retry/req_*.out.jsonl"))):   # B89b's candidates; retries fill errored rows
        for l in open(f):
            r = json.loads(l)
            if r.get("error") is None or r["custom_id"] not in A:
                A[r["custom_id"]] = r["ans"]
    P = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet"); ids = P.id.astype(int).values; N = len(P)
    z = P.z.values.astype(float); x = np.minimum(P.depth.values, 10).astype(float); cl = pd.factorize(P.cluster.values)[0]; nC = int(cl.max() + 1); neg = z == 0
    T = {}
    for f in sorted(glob.glob(f"{W}/data/second/s5/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); T[int(r["id"])] = r.get("text") or ""
    cnt = np.array([len(YOU.findall(T.get(int(i), ""))) for i in ids]); cb = np.minimum(cnt, 4)
    Wb = np.vstack([np.ones(nC), np.random.default_rng(9101).poisson(1.0, size=(NB, nC))])
    S = lambda v: Wb @ np.bincount(cl, v, nC)
    ci = lambda a: [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    def rate(v, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            return S(v * m) / S(m.astype(float))
    def slope(v, m):
        m_ = m.astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
        return 100 * (n_ * sxy - sx * sy) / (n_ * sxx - sx ** 2)
    def pooled(v, m, st):
        num = 0.0; den = 0.0
        for s_ in np.unique(st[m]):
            m_ = (m & (st == s_)).astype(float); n_, sx, sy, sxx, sxy = (S(m_ * a_) for a_ in (np.ones(N), x, v, x * x, x * v))
            ok = n_ > 0; nn = np.where(ok, n_, 1.0); num = num + np.where(ok, sxy - sx * sy / nn, 0.0); den = den + np.where(ok, sxx - sx ** 2 / nn, 0.0)
        return 100 * num / den
    allm = np.ones(N, bool); g = slope(z, allm); gfix = pooled(z, allm, cb)
    def kappa(y):
        po = float((y == z).mean()); p1, p2 = y.mean(), z.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2); return (po - pe) / (1 - pe)
    INS = {}
    for tag in ("M1", "M2", "T0", "T8", "T16", "TR"):
        a_ = [A.get(f"{tag}_{i}" if tag in ("M1", "M2") else f"{tag}_e_{i}") for i in ids]
        INS[tag] = (np.array([float(v in ("A", "B")) for v in a_]), int(sum(v is None for v in a_)))
    INS["stored"] = (np.nan_to_num(P["y:haiku"].values.astype(float)), int(P["y:haiku"].isna().sum()))
    res = {"N": N, "gold_depth": ci(g), "instruments": {}}; D0 = {}
    for tag, (y, miss) in INS.items():
        d_ = slope(y, allm) - g; D0[tag] = d_
        fr = [ci(rate(y, neg & (cb == k))) for k in range(5)]
        res["instruments"][tag] = dict(missing=miss, flag_rate=float(y.mean()), kappa=kappa(y), fpr=float(y[neg].mean()), depth=ci(slope(y, allm)), distortion=ci(d_),
                                       placebo=ci(slope(y, neg)), you_gap=ci(rate(y, neg & (cnt > 0)) - rate(y, neg & (cnt == 0))), fpr_by_count=fr,
                                       ratio_4_0=ci(rate(y, neg & (cb == 4)) / rate(y, neg & (cb == 0))), placebo_fixed=ci(pooled(y, neg, cb)),
                                       distortion_fixed=ci(pooled(y, allm, cb) - gfix), fpr_root=ci(rate(y, neg & (x == 0))), fpr_replies=ci(rate(y, neg & (x >= 1))))
        D0[tag + "_fix"] = pooled(y, allm, cb) - gfix; D0[tag + "_root"] = rate(y, neg & (x == 0)); D0[tag + "_rep"] = rate(y, neg & (x >= 1))
        v = res["instruments"][tag]
        print(tag, "missing", miss, "flag %.3f kappa %.3f FPR %.3f" % (v["flag_rate"], v["kappa"], v["fpr"]), "distortion", [round(t, 2) for t in v["distortion"]],
              "placebo", [round(t, 2) for t in v["placebo"]], "you gap", [round(100 * t, 1) for t in v["you_gap"]], "FPR by count", [round(100 * t[0], 1) for t in fr],
              "ratio", round(v["ratio_4_0"][0], 2), "fixed: placebo %.2f distortion %.2f" % (v["placebo_fixed"][0], v["distortion_fixed"][0]), flush=True)
    for tag in ("M1", "M2", "T8", "T16", "TR"):
        res["instruments"][tag]["change_vs_T0"] = ci(D0[tag] - D0["T0"]); res["instruments"][tag]["change_fixed_vs_T0"] = ci(D0[tag + "_fix"] - D0["T0_fix"])
        res["instruments"][tag]["change_fpr_root_vs_T0"] = ci(D0[tag + "_root"] - D0["T0_root"]); res["instruments"][tag]["change_fpr_replies_vs_T0"] = ci(D0[tag + "_rep"] - D0["T0_rep"])
        print(tag, "change vs T0: distortion", [round(u, 2) for u in res["instruments"][tag]["change_vs_T0"]], "at a fixed number", [round(u, 2) for u in res["instruments"][tag]["change_fixed_vs_T0"]],
              "FPR root", [round(100 * u, 1) for u in res["instruments"][tag]["change_fpr_root_vs_T0"]], "replies", [round(100 * u, 1) for u in res["instruments"][tag]["change_fpr_replies_vs_T0"]], flush=True)
    res["H91a"] = bool(res["instruments"]["M1"]["distortion"][1] > 0); res["H91b"] = bool(res["instruments"]["M2"]["distortion"][1] > 0)
    print("change vs T0", {t: [round(u, 2) for u in res["instruments"][t]["change_vs_T0"]] for t in ("M1", "M2")}, "H91a", res["H91a"], "H91b", res["H91b"])
    json.dump(res, open(OUT, "w"), indent=1)
