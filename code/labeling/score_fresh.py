"""Final confirmation (Addendum 69): score comments with fresh open classifiers, never used in development, on text alone.
Models (Hugging Face, public): s-nlp/roberta_toxicity_classifier, martin-ha/toxic-comment-model,
facebook/roberta-hate-speech-dynabench-r4-target, cardiffnlp/twitter-roberta-base-offensive. Each model's positive class is
found by name in its config (toxic / hate / offensive); the score is that class's softmax probability.
Input: JSONL files with {id, text}. Output per file: OUT/<tag>.parquet with id and one column per model.
Resumable per file; GPU fp16; length-sorted batches.
usage: python score_fresh.py --out OUT --files TAG=PATH [TAG=PATH ...] [--bs 256] [--models a,b]"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODELS = {"sn_rob": "s-nlp/roberta_toxicity_classifier", "mh_dist": "martin-ha/toxic-comment-model",
          "fb_hate": "facebook/roberta-hate-speech-dynabench-r4-target", "cn_off": "cardiffnlp/twitter-roberta-base-offensive"}
POS = ("toxic", "hate", "offensive", "label_1")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--maxlen", type=int, default=256); ap.add_argument("--models", default=",".join(MODELS))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
LOADED = {}


def load(name):
    if name not in LOADED:
        path = MODELS[name]; tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
        if dev == "cuda":
            m.half()
        labels = {int(k): str(v).lower() for k, v in m.config.id2label.items()}
        pos = [k for k, v in labels.items() if any(v == p or v.startswith(p) for p in POS) and not v.startswith("non") and not v.startswith("not")]
        if len(pos) != 1:
            raise SystemExit(f"{name}: cannot find the positive class in {labels}")
        LOADED[name] = (tok, m, pos[0]); print("loaded", name, path, labels, "positive", pos[0], flush=True)
    return LOADED[name]


@torch.no_grad()
def score(name, texts):
    tok, m, pos = load(name); order = np.argsort([len(t) for t in texts]); out = np.zeros(len(texts), np.float32)
    for b in range(0, len(texts), a.bs):
        ix = order[b:b + a.bs]; batch = [texts[i] if texts[i] else " " for i in ix]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=a.maxlen, padding=True).to(dev)
        p = torch.softmax(m(**enc).logits.float(), -1)[:, pos].cpu().numpy()
        out[ix] = p
    return out


names = [n for n in a.models.split(",") if n]
for spec in a.files:
    tag, path = spec.split("=", 1); dst = os.path.join(a.out, f"{tag}.parquet")
    if os.path.exists(dst):
        print("skip", tag, flush=True); continue
    t0 = time.time(); rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame({"id": [r["id"] for r in rows]}); texts = [r.get("text") or "" for r in rows]
    for n in names:
        df[n] = score(n, texts)
    df.to_parquet(dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print(f"{tag}: {len(df)} items in {time.time() - t0:.0f}s", flush=True)
print("scoring complete", flush=True)
