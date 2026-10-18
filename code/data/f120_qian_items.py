"""Addendum 24, B1: parse the Qian et al. (2019) Reddit and Gab conversations into items.
Each conversation's text lists posts as "k. <tabs>text"; the number of leading tabs is the reply depth. A post's parent is the
nearest preceding post one level up. Gold z = 1 when the post's index is in hate_speech_idx (majority of 3 in-context workers).
Writes data/qian/{reddit,gab}_items.jsonl (id, text, parent_id, parent_text) and data/qian/qian_meta.parquet
(id, platform, conv, pos, depth, parent, z). Ids: platform offset (reddit 1e7, gab 2e7) + conversation row * 100 + position.
usage: f120_qian_items.py RAW_DIR"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, ast, re, numpy as np, pandas as pd
RAW = sys.argv[1]; W = _ROOT; rows = []; bad = 0
for plat, off in [("reddit", 10_000_000), ("gab", 20_000_000)]:
    d = pd.read_csv(f"{RAW}/{plat}.csv")
    for ci, r in enumerate(d.itertuples(index=False)):
        hs = set(ast.literal_eval(r.hate_speech_idx)) if isinstance(r.hate_speech_idx, str) else set()
        posts = []
        for line in str(r.text).split("\n"):
            m = re.match(r"^(\d+)\. ?(\t*)(.*)$", line)
            if m: posts.append([int(m.group(1)), len(m.group(2)), m.group(3).strip()])
            elif line.strip() and posts: posts[-1][2] += " " + line.strip()      # continuation of a multi-line post
        if not posts or [p[0] for p in posts] != list(range(1, len(posts) + 1)): bad += 1; continue
        base = min(p[1] for p in posts); last = {}
        for k, t, txt in posts:
            dep = t - base; pid = last.get(dep - 1) if dep > 0 else None
            if dep > 0 and pid is None: dep = 0                                   # orphan indentation: treat as a new top-level post
            iid = off + ci * 100 + k; last[dep] = iid
            for dd in [x for x in last if x > dep]: del last[dd]
            rows.append(dict(id=iid, platform=plat, conv=off + ci, pos=k, depth=dep, parent=-1 if pid is None else pid, z=int(k in hs), text=txt))
M = pd.DataFrame(rows); txt = M.set_index("id").text
_os.makedirs(f"{W}/data/qian", exist_ok=True)                                    # release: the output folder is created if missing
for plat in ["reddit", "gab"]:
    with open(f"{W}/data/qian/{plat}_items.jsonl", "w") as fo:
        for x in M[M.platform == plat].itertuples(index=False):
            fo.write(json.dumps(dict(id=int(x.id), text=x.text, parent_id=None if x.parent < 0 else int(x.parent), parent_text=None if x.parent < 0 else txt[x.parent])) + "\n")
M.drop(columns="text").to_parquet(f"{W}/data/qian/qian_meta.parquet")
print("unparsed conversations", bad)
for plat, g in M.groupby("platform"):
    k = g.groupby("conv").size()
    print(plat, "conversations", g.conv.nunique(), "posts", len(g), "conversations with >= 2 posts", int((k >= 2).sum()), "posts in them", int(k[k >= 2].sum()),
          "replies", int((g.parent >= 0).sum()), "depth counts", g.depth.clip(upper=5).value_counts().sort_index().to_dict())
