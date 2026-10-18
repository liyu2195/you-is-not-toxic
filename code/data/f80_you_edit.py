"""Counterfactual test of the mechanism (registered H11): does second-person address cause Haiku's false flags?
Sample Civil Comments that address 'you' and edit them minimally so they refer to a third party instead
(you -> they/them, your -> their, yourself -> themselves, you people -> those people, ...). Label the edited text and,
in the same batch, the original again (retest), so the paired difference isolates the edit from sampling noise.
Strata (seed 80): clear negatives (crowd score < .2) at depth >= 2 (1,500) and at depth 0 (1,500); gold positives
(score >= .5) at any depth (1,000). Ids: 10 * comment id + variant (0 = original retest, 1 = edited).
usage: f80_you_edit.py SHARD_DIR OUT_DIR   -> OUT_DIR/shard_000.jsonl, data/you_edit_meta.parquet"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
SHD, OUTD = sys.argv[1], sys.argv[2]; W = _ROOT; rng = np.random.default_rng(80)
meta = pd.read_parquet(f"{W}/data/cc_meta.parquet")[["id", "tox", "parent_id"]].set_index("id")
par = meta.parent_id.where(meta.parent_id.isin(meta.index)).dropna().astype("int64")
cur = pd.Series(meta.index.values, index=meta.index); dep = pd.Series(0, index=meta.index)
for _ in range(200):
    has = cur.isin(par.index)
    if not has.any(): break
    cur[has] = par.reindex(cur[has].values).values; dep[has] += 1
meta["depth"] = dep
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
txt = {}
for f in sorted(glob.glob(f"{SHD}/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); t = r["text"] or ""
        if YOU.search(t): txt[int(r["id"])] = t
M = meta.loc[meta.index.isin(list(txt))].copy()
OBJ_PREV = set("to with for at about of from on in by like than against toward towards upon without into onto over under behind beside "
               "tell told ask asked thank thanks give gave show showed let see saw hear heard call called love hate help helped remind "
               "reminded warn warned assure assured bet owe believe trust pay paid send sent bring brought teach taught invite invited "
               "join forgive blame congratulate congrats mock mocked call meet met follow followed".split())
def edit(t):
    def sub(m):
        w = m.group(0); lw = w.lower().replace("’", "'")
        rep = {"your": "their", "yours": "theirs", "yourself": "themselves", "yourselves": "themselves", "you're": "they're", "youre": "theyre",
               "you've": "they've", "you'd": "they'd", "you'll": "they'll"}.get(lw)
        if rep is None:                                  # bare 'you': object after a preposition/verb or before punctuation, else subject
            pre = t[: m.start()].split()[-1:] if t[: m.start()].split() else [""]
            prev = re.sub(r"[^a-z]", "", pre[0].lower()); nxt = t[m.end():m.end() + 2]
            rep = "them" if (prev in OBJ_PREV or not nxt.strip() or nxt.strip()[:1] in ".,;:!?)\"") else "they"
        return rep.capitalize() if w[0].isupper() else rep
    t2 = re.sub(r"\byou people\b", lambda m: "Those people" if m.group(0)[0].isupper() else "those people", t, flags=re.I)
    return YOU.sub(sub, t2)
S = []
for name, sel, n in [("neg_deep", (M.tox < .2) & (M.depth >= 2), 1500), ("neg_root", (M.tox < .2) & (M.depth == 0), 1500), ("pos_any", M.tox >= .5, 1000)]:
    ids = M.index[sel].values; take = rng.choice(ids, min(n, len(ids)), replace=False)
    S += [dict(id=int(i), stratum=name, tox=float(M.tox[i]), depth=int(M.depth[i])) for i in take]
S = pd.DataFrame(S); os.makedirs(OUTD, exist_ok=True); n_changed = 0
with open(f"{OUTD}/shard_000.jsonl", "w") as f:
    for r in S.itertuples():
        t = txt[r.id]; e = edit(t); n_changed += e != t
        f.write(json.dumps(dict(id=10 * r.id, text=t, parent_text=None)) + "\n"); f.write(json.dumps(dict(id=10 * r.id + 1, text=e, parent_text=None)) + "\n")
S.to_parquet(f"{W}/data/you_edit_meta.parquet")
print(S.stratum.value_counts().to_dict(), "edited differs from original:", n_changed, "of", len(S))
ex = S.sample(4, random_state=1)
for r in ex.itertuples(): print("---", r.stratum, "\n", txt[r.id][:220], "\n=>", edit(txt[r.id])[:220])
