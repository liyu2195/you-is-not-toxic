"""Rebuild, from the original Civil Comments download, the few comment texts that three paper scripts read. This repository
contains no comment text; Civil Comments itself is released under CC0.

  code/paper/x98a_escalation.py reads  data/second/s5/shard_*.jsonl                 texts of the 5% sample (the "you" detector)
                                       results/x74_edit/label_items/shard_000.jsonl  the edit sample as written (E0) and with
                                                                                     the rule rewrite (E1), for the minimal pairs
  code/paper/x98f_arxiv_review.py reads results/x74_edit/texts.jsonl                 the edit sample as written (E0)
  code/paper/x18_tables_v5.py reads    results/x74_edit/label_items/shard_000.jsonl  E0 and E1, for the earlier draft's
                                                                                     examples table (tables/examples_v5.tex)

The files written here hold exactly these records (the pipeline's files held more: the whole 5% sample's articles, and Sonnet's
paraphrases E3/E4 and the reverse sample, which are not rebuilt and which the three scripts do not read). Texts are read as
code/data/f01_full_items.py reads them (comment_text of train.csv.gz, test_public_expanded.csv.gz and test_private_expanded.csv.gz
of the Hugging Face mirror shuttie/jigsaw-unintended-bias; a missing text is the empty string), and E1 applies the rule of
code/analysis/f80_you_edit_lib.py, as code/analysis/x74_edit.py did. Prints counts only.

usage: python3 -I code/data/rebuild_cc_texts.py CIVIL_COMMENTS_DIR
  root: $YOU_IS_NOT_TOXIC_ROOT (default: the repository root); reads labels/cc_s5/structure.parquet and data/you_edit_meta.parquet."""
import json, os, sys
import pandas as pd

ROOT = os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "analysis"))
FILES = ("train", "test_public_expanded", "test_private_expanded")


def load(src):
    """id -> text and id -> parent_id over the three files (as f01_full_items.py)"""
    parts = [pd.read_csv(os.path.join(src, f"{s}.csv.gz"), usecols=["id", "comment_text", "parent_id"], dtype={"comment_text": "string"}) for s in FILES]
    df = pd.concat(parts, ignore_index=True)
    df["parent_id"] = df.parent_id.astype("Int64")
    txt = {int(i): ("" if pd.isna(t) else str(t)) for i, t in zip(df.id.values, df.comment_text.values)}
    par = {int(i): (None if pd.isna(p) else int(p)) for i, p in zip(df.id.values, df.parent_id.values)}
    return txt, par


def s5_records(txt, par, ids):
    out = []
    for i in sorted(int(x) for x in ids):
        p = par[i]
        out.append({"id": i, "text": txt[i], "parent_id": p, "parent_text": txt[p] if p is not None and p in txt else None})
    return out


def edit_records(txt, ids):
    from f80_you_edit_lib import edit as you_edit                                            # the rewriting rule, verbatim
    texts, items = [], []
    for c in ids:
        t = txt[int(c)]
        for v, s in ((0, t), (1, you_edit(t))):
            texts.append({"id": 10 * int(c) + v, "text": s, "sample": "h11", "variant": v})
            items.append({"id": 10 * int(c) + v, "text": s, "parent_text": None})
    return texts, items


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    txt, par = load(sys.argv[1])
    s5 = pd.read_parquet(os.path.join(ROOT, "labels", "cc_s5", "structure.parquet"), columns=["id"]).id.values
    S11 = pd.read_parquet(os.path.join(ROOT, "data", "you_edit_meta.parquet")).id.values
    missing = [int(i) for i in list(s5) + list(S11) if int(i) not in txt]
    if missing:
        raise SystemExit(f"{len(missing)} comments are missing from the download")
    d = os.path.join(ROOT, "data", "second", "s5"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "shard_000.jsonl"), "w") as fo:
        for r in s5_records(txt, par, s5):
            fo.write(json.dumps(r) + "\n")
    texts, items = edit_records(txt, S11)
    d = os.path.join(ROOT, "results", "x74_edit"); os.makedirs(os.path.join(d, "label_items"), exist_ok=True)
    with open(os.path.join(d, "texts.jsonl"), "w") as fo:
        for r in texts:
            fo.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "label_items", "shard_000.jsonl"), "w") as fo:
        for r in items:
            fo.write(json.dumps(r) + "\n")
    print(f"5% sample: {len(s5)} texts -> data/second/s5/shard_000.jsonl; edit sample: {len(S11)} comments x 2 versions -> "
          "results/x74_edit/texts.jsonl and results/x74_edit/label_items/shard_000.jsonl")


if __name__ == "__main__":
    main()
