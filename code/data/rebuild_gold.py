"""Gold labels of the three corpora whose labels this repository does not redistribute, read from their original downloads.

  Qian et al. (2019), Reddit and Gab   reddit.csv and gab.csv of https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech
  CGA-Wiki (Zhang et al. 2018)          ConvoKit's conversations-gone-awry-corpus (utterances.jsonl, conversations.json)
  BAD (Xu et al. 2021)                  ParlAI's bot_adversarial_dialogue_datasets_with_persona (train.txt, valid.txt, test.txt)

Each reader parses the files exactly as the paper's pipeline did (Qian: code/data/f120_qian_items.py; CGA-Wiki:
code/data/f70_cga_items.py; BAD: one line per turn, Appendix A of the paper) and returns the keys of labels/<population>/structure.parquet
with the gold label, so that code/data/assemble_populations.py can rebuild results/x14_pops/<population>.parquet. Nothing is printed
but counts; no text leaves the reader.

usage (writes one gold file per population, id + gold, to OUT_DIR):
  python3 -I rebuild_gold.py OUT_DIR [--qian DIR] [--cga DIR] [--bad DIR]"""
import argparse, ast, json, os, re
import pandas as pd

QIAN_OFFSET = {"reddit": 10_000_000, "gab": 20_000_000}


def qian(raw_dir, platform):
    """Posts of Qian et al.'s <platform>.csv: conversation (0-based row), post (1-based number), depth, parent_post, post_id and
    gold (1 when the post's number is in the conversation's hate_speech_idx). The pipeline id is offset + 100 * row + post."""
    d = pd.read_csv(os.path.join(raw_dir, f"{platform}.csv"))
    rows = []
    for ci, r in enumerate(d.itertuples(index=False)):
        hs = set(ast.literal_eval(r.hate_speech_idx)) if isinstance(r.hate_speech_idx, str) else set()
        posts = []
        for line in str(r.text).split("\n"):
            m = re.match(r"^(\d+)\. ?(\t*)(.*)$", line)
            if m:
                posts.append([int(m.group(1)), len(m.group(2))])
        if not posts or [p[0] for p in posts] != list(range(1, len(posts) + 1)):
            continue                                                          # skipped by the pipeline too (none is)
        ids = [m.group(3) for line in str(r.id).split("\n") for m in [re.match(r"^(\d+)\. ?(\t*)(\S*)\s*$", line)] if m]
        base = min(p[1] for p in posts); last = {}
        for j, (k, t) in enumerate(posts):
            dep = t - base; pk = last.get(dep - 1) if dep > 0 else None
            if dep > 0 and pk is None:
                dep = 0                                                       # orphan indentation: a new top-level post (none is)
            last[dep] = k
            for dd in [x for x in last if x > dep]:
                del last[dd]
            rows.append(dict(id=QIAN_OFFSET[platform] + ci * 100 + k, conversation=ci, post=k, depth=dep,
                             parent_post=pk, post_id=ids[j] if len(ids) == len(posts) else None, gold=int(k in hs)))
    out = pd.DataFrame(rows)
    out["parent_post"] = out.parent_post.astype("Int64")
    return out


def cga(corpus_dir):
    """Non-header utterances of ConvoKit's CGA-Wiki corpus: utterance_id, conversation_id, position (1-based, in order of timestamp
    then utterance id), reply_to (the replied-to utterance when it is another non-header utterance) and gold
    (comment_has_personal_attack), plus the conversation's verified and pair_verified flags."""
    U = []
    with open(os.path.join(corpus_dir, "utterances.jsonl")) as f:
        for line in f:
            u = json.loads(line); m = u["meta"] if isinstance(u["meta"], dict) else ast.literal_eval(u["meta"])
            U.append(dict(utterance_id=u["id"], conversation_id=u["conversation_id"], rt=u["reply-to"], ts=float(u["timestamp"]),
                          hdr=bool(m["is_section_header"]), gold=int(bool(m["comment_has_personal_attack"]))))
    X = pd.DataFrame(U)
    Y = X[~X.hdr].sort_values(["conversation_id", "ts", "utterance_id"]).reset_index(drop=True)
    Y["id"] = range(1, len(Y) + 1)
    Y["position"] = Y.groupby("conversation_id").cumcount() + 1
    Y["reply_to"] = Y.rt.where(Y.rt.isin(Y.utterance_id))
    C = json.load(open(os.path.join(corpus_dir, "conversations.json")))
    flags = {k: {kk: str((v.get("meta", v)).get(kk)) == "True" for kk in ("verified", "pair_verified")} for k, v in C.items()}
    Y["verified"] = Y.conversation_id.map(lambda c: int(flags.get(c, {}).get("verified", False)))
    Y["pair_verified"] = Y.conversation_id.map(lambda c: int(flags.get(c, {}).get("pair_verified", False)))
    return Y[["id", "utterance_id", "conversation_id", "position", "reply_to", "verified", "pair_verified", "gold"]]


def bad(parlai_dir):
    """Turns of the Bot-Adversarial Dialogue files with personas: one line per turn, the line's text holding the dialogue so far.
    id = 0-based line number over train.txt, valid.txt and test.txt in that order; a line with one turn starts a dialogue;
    dialogue = 0-based number of the dialogue within its split; turn = number of turns in the line's text; speaker =
    speaker_to_eval; gold = 1 when the line's label is __notok__."""
    rows = []; i = 0
    for split in ("train", "valid", "test"):
        k = -1
        with open(os.path.join(parlai_dir, f"{split}.txt"), encoding="utf-8") as f:
            for line in f:
                fld = dict(p.split(":", 1) for p in line.rstrip("\n").split("\t"))
                n = len(fld["text"].split("\\n"))
                if n == 1:
                    k += 1
                rows.append(dict(id=i, split=split, dialogue=k, turn=n, speaker=fld["speaker_to_eval"], gold=int(fld["labels"] == "__notok__")))
                i += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="gold labels from the original downloads")
    ap.add_argument("out"); ap.add_argument("--qian"); ap.add_argument("--cga"); ap.add_argument("--bad")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.qian:
        for plat in ("reddit", "gab"):
            g = qian(a.qian, plat); g[["id", "gold"]].to_parquet(os.path.join(a.out, f"gold_qian_{plat}.parquet"), index=False)
            print(f"qian_{plat}: {len(g)} posts in {g.conversation.nunique()} conversations")
    if a.cga:
        g = cga(a.cga); g[["id", "gold"]].to_parquet(os.path.join(a.out, "gold_cga.parquet"), index=False)
        print(f"cga: {len(g)} comments in {g.conversation_id.nunique()} conversations")
    if a.bad:
        g = bad(a.bad); g[["id", "gold"]].to_parquet(os.path.join(a.out, "gold_bad.parquet"), index=False)
        print(f"bad: {len(g)} turns in {g.groupby('split').dialogue.nunique().sum()} dialogues")


if __name__ == "__main__":
    main()
