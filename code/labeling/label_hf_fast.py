"""Fast next-token option probabilities for many shards on one GPU (last-token logits only, no KV cache).
usage: python label_hf_fast.py --model Qwen/Qwen2.5-7B-Instruct --task tox --shards S1.jsonl S2.jsonl ... --outdir OUT [--bs 32]
For each shard writes OUT/<shard>_ctx0.jsonl (all items) and OUT/<shard>_ctx1.jsonl (items with parent text). Resumable."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TASKS, SYSTEM
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--task", default="tox")
ap.add_argument("--shards", nargs="+", required=True); ap.add_argument("--outdir", required=True); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--maxlen", type=int, default=1536); a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
fn, opts = TASKS[a.task]
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
opt_ids = [tok.encode(o, add_special_tokens=False)[0] for o in opts]
KEEP = True
def run(prompts):
    global KEEP
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=a.maxlen, add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        if KEEP:
            try: lg = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError: KEEP = False
        if not KEEP: lg = model(**enc, use_cache=False).logits[:, -1, :].float()
    pr = torch.softmax(lg, -1)[:, opt_ids]; mass = pr.sum(1)
    return (pr / mass[:, None]).cpu().tolist(), mass.cpu().tolist()
t0 = time.time(); total = 0
for sh in a.shards:
    name = os.path.basename(sh)[:-6]; items = [json.loads(l) for l in open(sh)]
    for ctx in [0, 1]:
        outp = f"{a.outdir}/{name}_ctx{ctx}.jsonl"
        if os.path.exists(outp + ".done"): continue
        its = [it for it in items if (it.get("parent_text") or not ctx)]
        done = set()
        if os.path.exists(outp):
            for l in open(outp):
                try: done.add(json.loads(l)["id"])
                except Exception: pass          # a line cut off by a VM stop is simply redone
        todo = [it for it in its if it["id"] not in done]
        P = [(it["id"], tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": fn(it["text"], it["parent_text"] if ctx else None)}],
                                                 tokenize=False, add_generation_prompt=True)) for it in todo]
        P.sort(key=lambda x: len(x[1]))
        with open(outp, "a") as fo:
            for s in range(0, len(P), a.bs):
                ch = P[s:s + a.bs]
                try: pn, ms = run([c[1] for c in ch])
                except torch.cuda.OutOfMemoryError:          # very long prompts: fall back to one prompt at a time
                    torch.cuda.empty_cache(); pn, ms = [], []
                    for c in ch:
                        p1, m1 = run([c[1]]); pn += p1; ms += m1
                    torch.cuda.empty_cache()
                for (iid, _), p, m in zip(ch, pn, ms): fo.write(json.dumps({"id": iid, "p": [round(x, 6) for x in p], "mass": round(m, 4)}) + "\n")
                total += len(ch)
                if (s // a.bs) % 100 == 0: fo.flush(); print(f"{name} ctx{ctx} {s + len(ch)}/{len(P)} | {total / (time.time() - t0):.1f}/s", flush=True)
        open(outp + ".done", "w").write("ok\n"); print(f"done {name} ctx{ctx} {len(P)}", flush=True)
print("all shards done", total, f"{time.time() - t0:.0f}s", flush=True)
