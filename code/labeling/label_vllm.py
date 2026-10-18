"""Next-token option probabilities with vLLM (open models on one GPU). Same prompts as the Haiku labels (code/prompts.py),
with the task's system prompt, rendered through the model's chat template. For each item: the probability of each option
letter at the first answer position, renormalized over the options ("p"), and the mass the options received ("mass").
Resumable: items already in the output are skipped. Writes OUTDIR/<name>_ctx<c>.jsonl per work item.
usage: python label_vllm.py --model M --work FILE:TASK:CTX[,FILE:TASK:CTX...] --outdir OUT [--chunk 2000]"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TASKS, SYSTEM, SYSTEMS
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--work", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--chunk", type=int, default=2000); ap.add_argument("--maxlen", type=int, default=2048); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-num-seqs", type=int, default=64); ap.add_argument("--max-batched-tokens", type=int, default=0); ap.add_argument("--eager", action="store_true")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
from vllm import LLM, SamplingParams
kw = dict(model=a.model, max_model_len=a.maxlen, gpu_memory_utilization=0.92, enable_prefix_caching=True, max_num_seqs=a.max_num_seqs)
if a.max_batched_tokens: kw["max_num_batched_tokens"] = a.max_batched_tokens
if a.eager: kw["enforce_eager"] = True
try: llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
except Exception as e:
    print("retrying without limit_mm_per_prompt:", str(e)[:200], flush=True); llm = LLM(**kw)
tok = llm.get_tokenizer(); sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
t0 = time.time(); total = 0
for w in a.work.split(","):
    path, task, ctx = w.split(":"); ctx = int(ctx); fn, opts = TASKS[task]; system = SYSTEMS.get(task, SYSTEM)
    name = os.path.basename(path).rsplit(".", 1)[0]; outp = f"{a.outdir}/{name}_{task}_ctx{ctx}.jsonl"
    items = [json.loads(l) for l in open(path)]
    if ctx: items = [it for it in items if it.get("parent_text")]
    if a.limit: items = items[:a.limit]
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    todo = [it for it in items if it["id"] not in done]
    print(f"{name} {task} ctx{ctx}: {len(todo)} to do of {len(items)}", flush=True)
    for s0 in range(0, len(todo), a.chunk):
        ch = todo[s0:s0 + a.chunk]
        prompts = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": fn(it["text"], it["parent_text"] if ctx else None)}],
                                           tokenize=False, add_generation_prompt=True) for it in ch]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        with open(outp, "a") as fo:
            for it, o in zip(ch, outs):
                lp = o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}
                pr = {k: 0.0 for k in opts}; top = None; best = -1e9
                for v in lp.values():
                    t = (v.decoded_token or "").strip()
                    if v.logprob > best: best, top = v.logprob, t
                    if t in pr: pr[t] += math.exp(v.logprob)
                mass = sum(pr.values()); p = [pr[k] / mass for k in opts] if mass > 0 else None
                fo.write(json.dumps(dict(id=it["id"], p=p, mass=mass, top=top)) + "\n")
        total += len(ch); el = time.time() - t0
        print(f"  {name} ctx{ctx} {s0 + len(ch)}/{len(todo)}  {total / el:.1f} items/s", flush=True)
print(f"all work done: {total} items in {time.time() - t0:.0f}s", flush=True)
