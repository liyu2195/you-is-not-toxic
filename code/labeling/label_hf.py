"""Next-token option probabilities from an open instruct LLM (GPU). Resumable JSONL output.
usage: python label_hf.py ITEMS OUT --model Qwen/Qwen2.5-7B-Instruct --task tox|ida --context 0|1 [--bs 16]
Items with --context 1 are only those with parent_text; their prompt shows the parent."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TASKS, SYSTEM
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out")
ap.add_argument("--model", required=True); ap.add_argument("--task", required=True); ap.add_argument("--context", type=int, default=0)
ap.add_argument("--bs", type=int, default=16); a = ap.parse_args()
fn, opts = TASKS[a.task]
items = [json.loads(l) for l in open(a.items)]
if a.context: items = [it for it in items if it.get("parent_text")]
done = set()
if os.path.exists(a.out): done = {json.loads(l)["id"] for l in open(a.out)}
todo = [it for it in items if it["id"] not in done]
print(f"{a.model} {a.task} ctx={a.context}: {len(todo)} to do of {len(items)}", flush=True)
if not todo: sys.exit(0)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda"); model.eval()
opt_ids = []
for o in opts:
    ids = tok.encode(o, add_special_tokens=False); assert len(ids) == 1, (o, ids); opt_ids.append(ids[0])
def build(it):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": fn(it["text"], it["parent_text"] if a.context else None)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
P = [(it["id"], build(it)) for it in todo]
P.sort(key=lambda x: len(x[1]))
t0 = time.time(); n = 0
with open(a.out, "a") as fo, torch.inference_mode():
    for s in range(0, len(P), a.bs):
        chunk = P[s:s + a.bs]
        enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        pr = torch.softmax(logits, dim=-1)
        po = pr[:, opt_ids]; mass = po.sum(1)
        pn = (po / mass[:, None]).cpu().tolist(); mass = mass.cpu().tolist()
        for (iid, _), p, m in zip(chunk, pn, mass):
            fo.write(json.dumps({"id": iid, "p": [round(x, 6) for x in p], "mass": round(m, 4)}) + "\n")
        n += len(chunk)
        if (s // a.bs) % 50 == 0:
            fo.flush(); print(f"{n}/{len(P)} {n/(time.time()-t0):.1f}/s", flush=True)
print("done", n, f"{time.time()-t0:.0f}s", flush=True)
