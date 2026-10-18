"""Addendum 61: attach each labeler's own score to the audit populations (results/x09_pops -> results/x10_pops).
Scores: letters A-D as levels 3..0 (Haiku, Sonnet; binary tasks give 1/0), option probabilities (Qwen, Gemma: P(A)+P(B) on the
4-option toxicity task, P(A) on 2-option tasks), continuous classifier scores (Detoxify toxicity; IMSYPP 1 - P(acceptable)).
With the parent ('_ctx'): the context run's score for replies, the plain run's for thread starts, as for the flags.
Check: the flag implied by each score at the original threshold must reproduce the stored flag ('y:') on every item.
usage: python3 x10_scores.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import json, glob, os
import numpy as np, pandas as pd
W = _ROOT; IN, OUT = f"{W}/results/x09_pops", f"{W}/results/x10_pops"; os.makedirs(OUT, exist_ok=True)
LV = {"A": 3, "B": 2, "C": 1, "D": 0}


def letters(files, binary=False):
    R = pd.DataFrame([json.loads(l) for f in files for l in open(f)])
    if "error" in R:
        R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last").set_index("id")
    if binary:
        return (R.ans == "A").astype(float)
    return R.ans.map(LV).fillna(0).astype(float)          # an answer without a letter counts as 'not toxic', as in f10


def probs(files, k=2):
    S = {}
    for f in files:
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("p") is not None:
                S[r["id"]] = float(sum(r["p"][:k]))
    return pd.Series(S)


def ctxmix(s0, s1, has_parent):
    """Context score for replies (falling back to the plain score if missing), plain score for thread starts."""
    a, b = s0.values, s1.values
    return np.where(has_parent, np.where(np.isnan(b), a, b), a)


def open_scores(tags):
    fs = [f for t in tags for f in glob.glob(f"{W}/results/x55_open/*/{t}.parquet")]
    S = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("id").set_index("id")
    return S


meta = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "parent_id"]).set_index("id")
F = f"{W}/results/full"
CC = {}
CC["h0"] = letters(sorted(glob.glob(f"{F}/haiku_tox_ctx0/shard_*.jsonl"))); CC["h1"] = letters(sorted(glob.glob(f"{F}/haiku_tox_ctx1/shard_*.jsonl")))
CC["q0"] = probs(sorted(glob.glob(f"{F}/qwen7b/*/shard_*_ctx0.jsonl"))); CC["q1"] = probs(sorted(glob.glob(f"{F}/qwen7b/*/shard_*_ctx1.jsonl")))
OC = open_scores([f"cc_{i:03d}" for i in range(80)])
report = {}
for pop in ["cc_threads", "cc_s5", "cad", "qian_reddit", "qian_gab", "cga", "bad"]:
    P = pd.read_parquet(f"{IN}/{pop}.parquet"); ids = P.id.values; labs = [c[2:] for c in P.columns if c.startswith("y:")]
    S = {}
    if pop in ("cc_threads", "cc_s5"):
        hp = meta.parent_id.reindex(ids).notna().values
        g = lambda s: s.reindex(ids)
        S["haiku"] = g(CC["h0"]).values; S["haiku_ctx"] = ctxmix(g(CC["h0"]), g(CC["h1"]), hp)
        S["qwen7b_m"] = S["qwen7b"] = g(CC["q0"]).values; S["qwen7b_m_ctx"] = ctxmix(g(CC["q0"]), g(CC["q1"]), hp)
        O = OC.reindex(ids)
        if pop == "cc_s5":
            LD = f"{W}/results/g27"
            g0 = probs(sorted(glob.glob(f"{LD}/**/shard_*_tox_ctx0.jsonl", recursive=True))); g1 = probs(sorted(glob.glob(f"{LD}/**/shard_*_tox_ctx1.jsonl", recursive=True)))
            S["gemma_m"] = g(g0).values; S["gemma_m_ctx"] = ctxmix(g(g0), g(g1), hp)
            s0 = letters(sorted(glob.glob(f"{W}/results/second/s5/sonnet_tox_ctx0/shard_*.jsonl"))); s1 = letters(sorted(glob.glob(f"{W}/results/second/s5/sonnet_tox_ctx1/shard_*.jsonl")))
            S["sonnet"] = g(s0).values; S["sonnet_ctx"] = ctxmix(g(s0), g(s1), hp)
    else:
        O = open_scores([pop]).reindex(ids)
        if pop == "cad":
            rd = lambda f: letters([f], binary=True)
            h0 = rd(f"{W}/results/cad/haiku_cad_ctx0.jsonl").reindex(ids); h1 = rd(f"{W}/results/cad/haiku_cad_ctx1.jsonl").reindex(ids)
            S["haiku"] = h0.values; S["haiku_ctx"] = np.where(np.isnan(h1.values), h0.values, h1.values)
            S["sonnet"] = rd(f"{W}/results/cad/sonnet_cad_ctx0.jsonl").reindex(ids).values
        elif pop.startswith("qian"):
            def ql(d):
                R = []
                for f in sorted(glob.glob(f"{d}/shard_*.jsonl")):
                    for l in open(f):
                        try:
                            R.append(json.loads(l))
                        except Exception:
                            pass
                R = pd.DataFrame(R).drop_duplicates("id", keep="last").set_index("id"); return (R.get("ans") == "A").astype(float)
            a0 = ql(f"{W}/results/second/qian/haiku_hate_ctx0").reindex(ids).fillna(0.0); a1 = ql(f"{W}/results/second/qian/haiku_hate_ctx1").reindex(ids)
            par = P.parent_row.values >= 0
            S["haiku"] = a0.values; S["haiku_ctx"] = np.where(par & ~np.isnan(a1.values), np.nan_to_num(a1.values), a0.values)
            S["gemma_m"] = probs(sorted(glob.glob(f"{W}/results/g27/**/shard_*_hate_ctx0.jsonl", recursive=True)), k=1).reindex(ids).values
        elif pop == "cga":
            cl = lambda d: letters(sorted(glob.glob(f"{d}/shard_*.jsonl")), binary=True)
            h0 = cl(f"{W}/results/cga/haiku_pa_ctx0").reindex(ids); h1 = cl(f"{W}/results/cga/haiku_pa_ctx1").reindex(ids)
            S["haiku"] = h0.values; S["haiku_ctx"] = np.where(np.isnan(h1.values), h0.values, h1.values)
            qr = lambda F_: pd.Series({json.loads(l)["id"]: json.loads(l)["p"][0] for f in F_ for l in open(f)})
            q0 = qr(sorted(glob.glob(f"{W}/results/cga/qwen7b/*_ctx0.jsonl"))).reindex(ids); q1 = qr(sorted(glob.glob(f"{W}/results/cga/qwen7b/*_ctx1.jsonl"))).reindex(ids)
            S["qwen7b_m"] = q0.values; S["qwen7b_m_ctx"] = np.where(np.isnan(q1.values), q0.values, q1.values)
        elif pop == "bad":
            Rb = pd.concat([pd.DataFrame([json.loads(l) for l in open(f)]) for f in [f"{W}/results/bad/haiku_unsafe_ctx1.jsonl", f"{W}/results/bad/haiku_unsafe_ctx0_first.jsonl"]])
            if "error" in Rb:
                Rb = Rb[Rb["error"].isna()]
            S["haiku_ctx"] = Rb.drop_duplicates("id", keep="last").set_index("id").ans.map(LV).reindex(ids).values.astype(float)
    for nm, col in [("detoxify_original", "det_orig"), ("detoxify_unbiased", "det_unb")]:
        S[f"{nm}@0.6"] = S[f"{nm}@m"] = O[col].values
    S["imsypp@m"] = S["imsypp@argmax"] = 1 - O["ims_p0"].values
    rep = {}
    for L in labs:
        if L not in S:
            rep[L] = "no score"; continue
        s = np.asarray(S[L], float); y = P[f"y:{L}"].values
        ok = ~np.isnan(y) & ~np.isnan(s)
        # the original threshold: flags are monotone in the score; check that no flagged item scores below an unflagged one
        mono = bool((s[ok & (y == 1)].min() >= s[ok & (y == 0)].max()) if (ok & (y == 1)).any() and (ok & (y == 0)).any() else True)
        rep[L] = dict(coverage=float(ok.mean()), flag_monotone=mono, levels=int(len(np.unique(s[ok]))))
        P[f"s:{L}"] = s
    P.to_parquet(f"{OUT}/{pop}.parquet"); report[pop] = rep
    print(pop, json.dumps(rep), flush=True)
json.dump(report, open(f"{OUT}/scores_report.json", "w"), indent=1)
