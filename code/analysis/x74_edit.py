"""Addendum 74, A4: control edits for the second-person mechanism.
  build SHARD_DIR   texts of the H11 comments (data/you_edit_meta.parquet) and a new sample (seed 7410, train split) of comments
                    without second-person address but with a third-person pronoun: 1,500 civil (crowd score < 0.2) at depth >= 2,
                    1,500 civil at depth 0, 1,000 gold positives (score >= 0.5). Variants written to results/x74_edit/:
                      E0 original (both samples), E1 the H11 regex edit (H11 sample), E2 the reverse regex edit (new sample).
                    Ids: 10 * comment id + variant (0..4).
  para              Sonnet 5.5 paraphrases E3 (keep "you") and E4 (third person) of the H11 comments, by Message Batches.
  shard             all variants in one shuffled shard for the Haiku label batch (results/x74_edit/label_items/shard_000.jsonl).
  analyze OUT_JSON  paired flag-rate differences by stratum, 2,000 paired bootstrap resamples (H74d, H74e).
usage: python3 x74_edit.py build|para|shard|analyze ..."""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, time, subprocess, urllib.request, urllib.error
import numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x74_edit"; os.makedirs(D, exist_ok=True)
sys.path.insert(0, f"{W}/code/labeling"); sys.path.insert(0, f"{W}/code/analysis")
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
THEY = re.compile(r"\b(they|them|their|theirs|themselves|they're|theyre|they've|they'd|they'll)\b", re.I)
REV = {"they": "you", "them": "you", "their": "your", "theirs": "yours", "themselves": "yourselves", "they're": "you're", "theyre": "youre",
       "they've": "you've", "they'd": "you'd", "they'll": "you'll"}


def reverse_edit(t):
    def sub(m):
        w = m.group(0); rep = REV[w.lower().replace("’", "'")]
        return rep.capitalize() if w[0].isupper() else rep
    return THEY.sub(sub, t)


def depth_table():
    meta = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "tox", "parent_id", "split"]].set_index("id")
    par = meta.parent_id.where(meta.parent_id.isin(meta.index)).dropna().astype("int64")
    cur = pd.Series(meta.index.values, index=meta.index); dep = pd.Series(0, index=meta.index)
    for _ in range(200):
        has = cur.isin(par.index)
        if not has.any():
            break
        cur[has] = par.reindex(cur[has].values).values; dep[has] += 1
    meta["depth"] = dep
    return meta


def build(shd):
    from f80_you_edit_lib import edit as you_edit                      # the H11 regex edit, verbatim
    S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet"); need = set(S11.id.tolist())
    meta = depth_table(); txt11 = {}; cand = {}
    for f in sorted(glob.glob(f"{shd}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); i = int(r["id"]); t = r["text"] or ""
            if i in need:
                txt11[i] = t
            elif not YOU.search(t) and THEY.search(t):
                cand[i] = t
    M = meta.loc[meta.index.isin(list(cand))].copy(); M = M[M.split == "train"]
    rng = np.random.default_rng(7410); S = []
    for name, sel, n in [("neg_deep", (M.tox < .2) & (M.depth >= 2), 1500), ("neg_root", (M.tox < .2) & (M.depth == 0), 1500), ("pos_any", M.tox >= .5, 1000)]:
        ids = M.index[sel].values; take = rng.choice(ids, min(n, len(ids)), replace=False)
        S += [dict(id=int(i), stratum=name, tox=float(M.tox[i]), depth=int(M.depth[i])) for i in take]
    S = pd.DataFrame(S); S.to_parquet(f"{D}/meta_rev.parquet")
    with open(f"{D}/texts.jsonl", "w") as fo:
        for i in S11.id:
            t = txt11[int(i)]; fo.write(json.dumps(dict(id=10 * int(i), text=t, sample="h11", variant=0)) + "\n")
            fo.write(json.dumps(dict(id=10 * int(i) + 1, text=you_edit(t), sample="h11", variant=1)) + "\n")
        n_changed = 0
        for i in S.id:
            t = cand[int(i)]; e = reverse_edit(t); n_changed += e != t
            fo.write(json.dumps(dict(id=10 * int(i), text=t, sample="rev", variant=0)) + "\n")
            fo.write(json.dumps(dict(id=10 * int(i) + 2, text=e, sample="rev", variant=2)) + "\n")
    print("h11", len(txt11), "of", len(need), "| rev sample", S.stratum.value_counts().to_dict(), "edited differs:", n_changed)
    ex = S.sample(3, random_state=2)
    for r in ex.itertuples():
        print("---", r.stratum, r.depth, "\n", cand[r.id][:200], "\n=>", reverse_edit(cand[r.id])[:200])


PARA = ("Rewrite the comment below in different words. Keep its meaning, its tone and its level of politeness or rudeness exactly as "
        "they are. {person} Output only the rewritten comment, with no preamble.\n\nComment:\n\"\"\"{text}\"\"\"")
PERSON = {3: "Keep addressing the person the comment is addressed to as \"you\".",
          4: "Refer to the person the comment is addressed to in the third person (they, them, their) instead of \"you\"."}


def api():
    from label_claude import api_key, WORKSPACE
    key = api_key(); H = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json", **({"anthropic-workspace-id": WORKSPACE} if WORKSPACE else {})}
    def call(url, data=None, tries=8):
        for k in range(tries):
            try:
                req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers=H)
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                if e.code in (429, 500, 502, 503, 504, 529):
                    time.sleep(min(300, 15 * 2 ** k)); continue
                raise RuntimeError(f"HTTP {e.code}: {body}")
            except Exception:
                time.sleep(min(300, 15 * 2 ** k))
        raise RuntimeError("retries")
    return call


def para():
    T = [json.loads(l) for l in open(f"{D}/texts.jsonl")]; orig = {r["id"] // 10: r["text"] for r in T if r["sample"] == "h11" and r["variant"] == 0}
    st = f"{D}/para_state.json"; state = json.load(open(st)) if os.path.exists(st) else {}
    call = api()
    if "batch" not in state:
        reqs = []
        for cid, t in orig.items():
            for v in (3, 4):
                reqs.append({"custom_id": f"c{10 * cid + v}", "params": dict(model="claude-sonnet-5-5", max_tokens=min(1200, 60 + 2 * len(t) // 3),
                             thinking={"type": "between_tools"}, system="You rewrite online comments for a linguistics study.",
                             messages=[dict(role="user", content=PARA.format(person=PERSON[v], text=t[:2000]))])})
        b = json.loads(call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs}))
        state = {"batch": b["id"], "n": len(reqs), "t": time.time()}; json.dump(state, open(st, "w")); print("submitted", b["id"], len(reqs), flush=True)
    while True:
        b = json.loads(call(f"https://api.anthropic.com/v1/messages/batches/{state['batch']}"))
        if b["processing_status"] == "ended":
            break
        print(time.strftime("%H:%M:%S"), b["request_counts"], flush=True); time.sleep(60)
    raw = call(b["results_url"]); n = 0
    with open(f"{D}/para.jsonl", "w") as fo:
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line); iid = int(r["custom_id"][1:])
            if r["result"]["type"] != "succeeded":
                fo.write(json.dumps(dict(id=iid, error=r["result"]["type"])) + "\n"); continue
            m = r["result"]["message"]; txt = "".join(x.get("text", "") for x in m.get("content", []) if x.get("type") == "text").strip()
            fo.write(json.dumps(dict(id=iid, text=txt, stop=m.get("stop_reason"))) + "\n"); n += 1
    print("paraphrases", n, b["request_counts"])


REFUSE = re.compile(r"^(i can['’]?t|i cannot|i won['’]?t|i['’]m (not able|unable|sorry)|sorry|i apologi[sz]e|i will not|i['’]m not going to)", re.I)


def shard():
    T = [json.loads(l) for l in open(f"{D}/texts.jsonl")]
    P = [json.loads(l) for l in open(f"{D}/para.jsonl")]
    items = [dict(id=r["id"], text=r["text"], parent_text=None) for r in T]
    bad = 0
    for r in P:
        if "text" not in r or not r["text"] or REFUSE.search(r["text"].strip().strip('"')) or r.get("stop") == "max_tokens":
            bad += 1; continue
        t = r["text"].strip()
        if t.startswith('"""') and t.endswith('"""'):
            t = t[3:-3].strip()
        items.append(dict(id=r["id"], text=t, parent_text=None))
    rng = np.random.default_rng(7411); order = rng.permutation(len(items))
    os.makedirs(f"{D}/label_items", exist_ok=True)
    with open(f"{D}/label_items/shard_000.jsonl", "w") as fo:
        for k in order:
            fo.write(json.dumps(items[k]) + "\n")
    print("items", len(items), "paraphrases dropped (refusal, truncation or empty):", bad)


def analyze(out):
    rng = np.random.default_rng(7412)
    R = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{D}/haiku_tox/shard_*.jsonl")) for l in open(f)])
    if "error" in R:
        R = R[R["error"].isna()]
    R = R.drop_duplicates("id", keep="last"); R["flag"] = R.ans.isin(["A", "B"]).astype(float); R["cid"] = R.id // 10; R["var"] = R.id % 10
    Pv = R.pivot(index="cid", columns="var", values="flag")
    S11 = pd.read_parquet(f"{W}/data/you_edit_meta.parquet").set_index("id"); Sr = pd.read_parquet(f"{D}/meta_rev.parquet").set_index("id")
    sp = pd.read_parquet(f"{W}/data/cc_meta.parquet", columns=["id", "split"]).set_index("id").split
    S11["train"] = sp.reindex(S11.index).eq("train").values
    res = {"h11": {}, "rev": {}, "h11_all_splits": {}}
    def paired(a, b):
        d = (b - a); d = d[np.isfinite(d)]
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        return dict(n=int(len(d)), diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    for key, S, keep in [("h11", S11[S11.train], (0, 1, 3, 4)), ("h11_all_splits", S11, (0, 1, 3, 4)), ("rev", Sr, (0, 2))]:
        J = S.join(Pv, how="inner")
        for st, g in J.groupby("stratum"):
            r = {f"flag_E{v}": float(g[v].mean()) for v in keep if v in g}
            r["n"] = int(len(g))
            if key.startswith("h11"):
                ok = g[[0, 1, 3, 4]].notna().all(axis=1) if all(v in g for v in (0, 1, 3, 4)) else g[0].notna()
                gg = g[ok]
                r["E1_minus_E0"] = paired(gg[0].values, gg[1].values)
                if 3 in gg and 4 in gg:
                    r["E3_minus_E0"] = paired(gg[0].values, gg[3].values); r["E4_minus_E3"] = paired(gg[3].values, gg[4].values)
                    r["E4_minus_E0"] = paired(gg[0].values, gg[4].values); r["n_paired_all"] = int(len(gg))
            else:
                gg = g[g[[0, 2]].notna().all(axis=1)]; r["E2_minus_E0"] = paired(gg[0].values, gg[2].values)
            res[key][st] = r
            print(key, st, json.dumps(r))
    res["H74d"] = bool(res["h11"]["neg_deep"]["E4_minus_E3"]["ci"][1] < 0)
    res["H74e"] = bool(res["rev"]["neg_deep"]["E2_minus_E0"]["ci"][0] > 0)
    print("H74d", res["H74d"], "H74e", res["H74e"])
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "para":
        para()
    elif cmd == "shard":
        shard()
    elif cmd == "analyze":
        analyze(sys.argv[2])
