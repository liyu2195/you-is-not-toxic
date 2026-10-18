"""Addendum 55, D1: score comments with open classifiers on text alone, no context (as Avalle et al. 2024).
Models: Detoxify 'original' (Jigsaw 2018, Wikipedia talk) and 'unbiased' (Jigsaw 2019, Civil Comments: contaminated on
Civil Comments), toxicity heads; IMSyPP/hate_speech_en (4 classes: 0 acceptable, 1 inappropriate, 2 offensive, 3 violent).
Input: JSONL files with {id, text}. Output per file: OUT/<tag>.parquet with id, det_orig, det_unb, ims_p0..ims_p3.
Resumable per file; GPU fp16; length-sorted batches.
usage: python score_open.py --out OUT --files TAG=PATH [TAG=PATH ...] [--bs 256]"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--maxlen", type=int, default=256); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)

def load_detoxify(kind):
    from detoxify import Detoxify
    d = Detoxify(kind, device=dev); d.model.eval()
    if dev == "cuda": d.model.half()
    return d.tokenizer, d.model, list(d.class_names).index("toxicity" if "toxicity" in d.class_names else d.class_names[0]), "sigmoid"
def load_imsypp():
    tok = AutoTokenizer.from_pretrained("IMSyPP/hate_speech_en"); m = AutoModelForSequenceClassification.from_pretrained("IMSyPP/hate_speech_en").to(dev).eval()
    if dev == "cuda": m.half()
    return tok, m, None, "softmax"
MODELS = {}
def model(name):
    if name not in MODELS:
        MODELS[name] = {"det_orig": lambda: load_detoxify("original"), "det_unb": lambda: load_detoxify("unbiased"), "ims": load_imsypp}[name]()
        print("loaded", name, flush=True)
    return MODELS[name]

@torch.no_grad()
def score(name, texts):
    tok, m, head, act = model(name); order = np.argsort([len(t) for t in texts]); out = [None] * len(texts)
    for b in range(0, len(texts), a.bs):
        ix = order[b:b + a.bs]; batch = [texts[i] if texts[i] else " " for i in ix]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=a.maxlen, padding=True).to(dev)
        o = m(**enc); logits = (o.logits if hasattr(o, "logits") else o[0]).float()
        p = torch.sigmoid(logits[:, head]) if act == "sigmoid" else torch.softmax(logits, -1)
        p = p.cpu().numpy()
        for j, i in enumerate(ix): out[i] = p[j]
    return np.array(out)

for spec in a.files:
    tag, path = spec.split("=", 1); dst = os.path.join(a.out, f"{tag}.parquet")
    if os.path.exists(dst): print("skip", tag, flush=True); continue
    t0 = time.time(); rows = [json.loads(l) for l in open(path)]
    ids = [r["id"] for r in rows]; texts = [r.get("text") or "" for r in rows]
    df = pd.DataFrame({"id": ids})
    df["det_orig"] = score("det_orig", texts); df["det_unb"] = score("det_unb", texts)
    P = score("ims", texts)
    for k in range(P.shape[1]): df[f"ims_p{k}"] = P[:, k]
    df.to_parquet(dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print(f"{tag}: {len(df)} items in {time.time() - t0:.0f}s", flush=True)
print("scoring complete", flush=True)
