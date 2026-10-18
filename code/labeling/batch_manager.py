"""Resumable Message Batches manager for labeling sharded items at scale (claude-haiku-5-5 / claude-sonnet-5-5 only).
One batch per shard. State is kept in OUT/state.json and mirrored to the bucket, so a later session can collect results
from batch ids even if this process dies. Results go to OUT/shard_XXX.jsonl in the pilot format {id, ans, raw, in_tok, out_tok}.
usage: python -I batch_manager.py SHARD_DIR OUT --model claude-haiku-5-5 --task tox --context 0|1 [--ids FILE] [--max_inflight 16]
       [--bucket $YOU_IS_NOT_TOXIC_BUCKET/results/full/NAME]"""
import argparse, glob, json, os, re, subprocess, sys, time, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TASKS, SYSTEM, SYSTEMS
from label_claude import api_key, WORKSPACE, ALLOWED
ap = argparse.ArgumentParser(); ap.add_argument("shards"); ap.add_argument("out")
ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--task", default="tox"); ap.add_argument("--context", type=int, default=0)
ap.add_argument("--ids", default=None); ap.add_argument("--max_inflight", type=int, default=16); ap.add_argument("--bucket", default=None)
a = ap.parse_args(); assert a.model in ALLOWED
os.makedirs(a.out, exist_ok=True); SP = f"{a.out}/state.json"
fn, opts = TASKS[a.task]; pat = re.compile(r"\b([" + "".join(opts) + r"])\b")
think = {"type": "disabled"} if "haiku" in a.model else {"type": "between_tools"}
keep = set(int(x) for x in open(a.ids).read().split()) if a.ids else None
state = json.load(open(SP)) if os.path.exists(SP) else {}
GENV = {k: v for k, v in os.environ.items() if k != "CLOUDSDK_AUTH_ACCESS_TOKEN"}
def gsync(path):
    if a.bucket:
        subprocess.run(["timeout", "60", "gcloud", "storage", "cp", path, f"{a.bucket}/{os.path.basename(path)}"], env=GENV, capture_output=True)
def save():
    json.dump(state, open(SP + ".tmp", "w"), indent=0); os.replace(SP + ".tmp", SP); gsync(SP)
key = api_key(); H = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json", **({"anthropic-workspace-id": WORKSPACE} if WORKSPACE else {})}
def call(url, data=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers=H)
            with urllib.request.urlopen(req, timeout=600) as r: return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503, 504, 529):
                if data is not None and e.code == 429: return ("RATE", body)
                time.sleep(min(300, 10 * 2 ** k)); continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except Exception as e:
            time.sleep(min(300, 10 * 2 ** k))
    return ("FAIL", "retries")
shards = sorted(glob.glob(f"{a.shards}/shard_*.jsonl"))
def requests_for(sh):
    reqs = []
    for l in open(sh):
        it = json.loads(l)
        if keep is not None and it["id"] not in keep: continue
        if a.context and not it.get("parent_text"): continue
        reqs.append({"custom_id": f"c{it['id']}", "params": dict(model=a.model, max_tokens=5, system=SYSTEMS.get(a.task, SYSTEM), thinking=think,
                     messages=[dict(role="user", content=fn(it["text"], it["parent_text"] if a.context else None))])})
    return reqs
def collect(name, b):
    raw = call(b["results_url"])
    if isinstance(raw, tuple): return False
    n = 0; outp = f"{a.out}/{name}.jsonl"
    with open(outp + ".tmp", "w") as fo:
        for line in raw.decode().splitlines():
            if not line.strip(): continue
            r = json.loads(line); iid = int(r["custom_id"][1:])
            if r["result"]["type"] != "succeeded": fo.write(json.dumps(dict(id=iid, error=r["result"]["type"])) + "\n"); continue
            m = r["result"]["message"]; txt = "".join(x.get("text", "") for x in m.get("content", [])).strip(); mm = pat.search(txt); us = m.get("usage", {})
            fo.write(json.dumps(dict(id=iid, ans=mm.group(1) if mm else None, raw=txt[:20], in_tok=us.get("input_tokens"), out_tok=us.get("output_tokens"))) + "\n"); n += 1
    os.replace(outp + ".tmp", outp); gsync(outp); return n
t_rate = 0
while True:
    pending = [s for s in shards if os.path.basename(s)[:-6] not in state]
    inflight = [k for k, v in state.items() if v["status"] == "submitted"]
    # submit
    while pending and len(inflight) < a.max_inflight and time.time() >= t_rate:
        sh = pending.pop(0); name = os.path.basename(sh)[:-6]; reqs = requests_for(sh)
        if not reqs: state[name] = {"status": "empty"}; save(); continue
        r = call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs})
        if isinstance(r, tuple):
            print(time.strftime("%H:%M:%S"), "submit deferred", name, r[0], r[1][:300], flush=True); t_rate = time.time() + 300; break
        bid = json.loads(r)["id"]; state[name] = {"status": "submitted", "batch": bid, "n": len(reqs), "t": time.time()}; save(); inflight.append(name)
        print(time.strftime("%H:%M:%S"), "submitted", name, bid, len(reqs), flush=True)
    # poll
    for name in list(inflight):
        r = call(f"https://api.anthropic.com/v1/messages/batches/{state[name]['batch']}")
        if isinstance(r, tuple): continue
        b = json.loads(r)
        if b["processing_status"] == "ended":
            n = collect(name, b)
            if n is not False:
                state[name].update(status="done", n_ok=n, counts=b["request_counts"]); save()
                print(time.strftime("%H:%M:%S"), "done", name, n, b["request_counts"], flush=True)
    done = sum(v["status"] in ("done", "empty") for v in state.values())
    if done == len(shards): print("all done", flush=True); break
    time.sleep(60)
