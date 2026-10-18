"""Build CGA-Wiki items (Conversations Gone Awry, ConvoKit corpus v6) for labeling: non-header comments with integer ids,
the replied-to comment as context (None when it is a section header or outside the corpus), and a meta table with the gold.
Gold: conversation_has_personal_attack (awry) and comment_has_personal_attack; 'verified' marks crowd-verified conversations.
usage: f70_cga_items.py CORPUS_DIR   -> data/cga_items.jsonl, data/cga_meta.parquet, SHARD_DIR/shard_000.jsonl"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, ast, pandas as pd
D = sys.argv[1]; W = _ROOT; SH = sys.argv[2]
U = [json.loads(l) for l in open(f"{D}/utterances.jsonl")]; C = json.load(open(f"{D}/conversations.json"))
rows = []
for u in U:
    m = u["meta"] if isinstance(u["meta"], dict) else ast.literal_eval(u["meta"])
    rows.append(dict(uid=u["id"], conv=u["conversation_id"], rt=u["reply-to"], ts=float(u["timestamp"]), hdr=bool(m["is_section_header"]),
                     pa=int(bool(m["comment_has_personal_attack"])), speaker=u["speaker"], text=u["text"]))
X = pd.DataFrame(rows)
cm = pd.DataFrame([dict(conv=k, **{kk: (v.get("meta", v)).get(kk) for kk in ["conversation_has_personal_attack", "verified", "pair_verified", "pair_id", "page_title", "split"]}) for k, v in C.items()])
for c in ["conversation_has_personal_attack", "verified", "pair_verified"]: cm[c] = cm[c].astype(str).eq("True").astype(int)
cm = cm.rename(columns={"conversation_has_personal_attack": "awry"})
txt = dict(zip(X.uid, X.text)); hdr = dict(zip(X.uid, X.hdr))
Y = X[~X.hdr].sort_values(["conv", "ts", "uid"]).reset_index(drop=True); Y["id"] = range(1, len(Y) + 1)
Y["pos"] = Y.groupby("conv").cumcount() + 1; Y["k"] = Y.groupby("conv").uid.transform("size")
Y["parent_uid"] = Y.rt.where(Y.rt.isin(Y.uid))                      # reply-to another comment (not a header)
Y["parent_text"] = Y.parent_uid.map(txt)
Y = Y.merge(cm, on="conv", how="left")
Y[["id", "uid", "conv", "pos", "k", "parent_uid", "speaker", "ts", "pa", "awry", "verified", "pair_verified", "pair_id", "page_title", "split"]].to_parquet(f"{W}/data/cga_meta.parquet")
os.makedirs(SH, exist_ok=True)
with open(f"{W}/data/cga_items.jsonl", "w") as f, open(f"{SH}/shard_000.jsonl", "w") as g:
    for r in Y.itertuples():
        l = json.dumps(dict(id=int(r.id), text=r.text, parent_text=r.parent_text if isinstance(r.parent_text, str) else None)) + "\n"; f.write(l); g.write(l)
print("comments", len(Y), "conversations", Y.conv.nunique(), "with parent", int(Y.parent_text.notna().sum()), "verified convs", int(cm.verified.sum()), "pair_verified", int(cm.pair_verified.sum()))
