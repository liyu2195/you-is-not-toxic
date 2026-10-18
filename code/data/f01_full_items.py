"""All 1,999,514 Civil Comments as JSONL shards of 25,000 (id, text, parent_id, parent_text), sorted by id.
parent_text is set when the parent exists in the data. Writes OUTDIR/shard_000.jsonl ... and OUTDIR/manifest.json."""
import sys, json, os, pandas as pd, numpy as np
src, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
parts = []
for s, tcol in [("train", "comment_text"), ("test_public_expanded", "comment_text"), ("test_private_expanded", "comment_text")]:
    parts.append(pd.read_csv(f"{src}/{s}.csv.gz", usecols=["id", "comment_text", "parent_id"], dtype={"comment_text": "string"}))
df = pd.concat(parts, ignore_index=True); del parts
df["parent_id"] = df.parent_id.astype("Int64")
df = df.sort_values("id").reset_index(drop=True)
txt = pd.Series(df.comment_text.values, index=df.id.values)
has = df.parent_id.notna() & df.parent_id.isin(txt.index)
df["parent_text"] = pd.array([None] * len(df), dtype="string")
df.loc[has, "parent_text"] = txt.reindex(df.loc[has, "parent_id"].astype("int64").values).values
print("comments", len(df), "with parent text", int(has.sum()), flush=True)
N = 25000; man = []
for k, s0 in enumerate(range(0, len(df), N)):
    d = df.iloc[s0:s0 + N]; fn = f"{outdir}/shard_{k:03d}.jsonl"
    with open(fn, "w") as fo:
        for r in d.itertuples(index=False):
            fo.write(json.dumps({"id": int(r.id), "text": "" if pd.isna(r.comment_text) else str(r.comment_text),
                                 "parent_id": None if pd.isna(r.parent_id) else int(r.parent_id),
                                 "parent_text": None if pd.isna(r.parent_text) else str(r.parent_text)}) + "\n")
    man.append({"shard": k, "n": len(d), "n_ctx": int(d.parent_text.notna().sum())})
json.dump(man, open(f"{outdir}/manifest.json", "w"), indent=0)
print("shards", len(man), "ctx total", sum(m["n_ctx"] for m in man), flush=True)
