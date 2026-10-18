"""Load Civil Comments (Jigsaw Unintended Bias; HF mirror shuttie/jigsaw-unintended-bias) metadata into one parquet.
Gold toxic := target >= 0.5 (fraction of raters saying toxic or very toxic)."""
import sys, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]
cols_tr = ["id","target","created_date","publication_id","parent_id","article_id","rating","toxicity_annotator_count"]
tr = pd.read_csv(f"{src}/train.csv.gz", usecols=cols_tr).rename(columns={"target":"tox"})
tr["split"] = "train"
parts = [tr]
for s in ["test_public_expanded","test_private_expanded"]:
    t = pd.read_csv(f"{src}/{s}.csv.gz", usecols=["id","toxicity","created_date","publication_id","parent_id","article_id","rating","toxicity_annotator_count"]).rename(columns={"toxicity":"tox"})
    t["split"] = s
    parts.append(t)
df = pd.concat(parts, ignore_index=True)
df["parent_id"] = df["parent_id"].astype("Int64")
df["created_date"] = pd.to_datetime(df["created_date"], format="mixed", utc=True)
df["Z"] = (df["tox"] >= 0.5).astype(np.int8)
print(df.shape, df["id"].is_unique)
print(df.groupby("split").agg(n=("id","size"), prev=("Z","mean"), raters_med=("toxicity_annotator_count","median")))
df.to_parquet(out, index=False)
