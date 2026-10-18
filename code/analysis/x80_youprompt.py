"""Addendum 80, B80e: the main toxicity prompt plus an instruction against the second-person mechanism (task tox_you in
code/prompts.py), Haiku 5.5 and Sonnet 5.5 on the 5% Civil Comments sample (cc_s5), without context, by Message Batches (one batch
per model; requests in shuffled order, seed 8040). Answers are parsed like the main prompt's (first letter A-D).
  submit SHARD_DIR   build requests from the full text shards (id, text, parent_text) and submit the two batches
  collect            poll until the batches end and write results/x80/youprompt/<model>.jsonl  {id, ans}
usage: python3 x80_youprompt.py submit SHARD_DIR | collect"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, json, glob, time, urllib.request, urllib.error
import numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x80/youprompt"; os.makedirs(D, exist_ok=True); ST = f"{D}/state.json"
sys.path.insert(0, f"{W}/code/labeling")
from label_claude import api_key, WORKSPACE
from prompts import SYSTEM, TASKS
prompt = TASKS["tox_you"][0]
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
key = api_key(); H = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json", **({"anthropic-workspace-id": WORKSPACE} if WORKSPACE else {})}
def call(url, data=None, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers=H)
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503, 504, 529):
                time.sleep(min(300, 20 * 2 ** k)); continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except Exception:
            time.sleep(min(300, 20 * 2 ** k))
    raise RuntimeError("retries")
if sys.argv[1] == "submit":
    ids = pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet", columns=["id"]).id.astype(int).tolist(); need = set(ids)
    items = {}
    for f in sorted(glob.glob(f"{sys.argv[2]}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if int(r["id"]) in need:
                items[int(r["id"])] = r.get("text") or ""
    order = np.random.default_rng(8040).permutation(sorted(items)); print("items", len(items), "of", len(need), flush=True)
    state = json.load(open(ST)) if os.path.exists(ST) else {}
    for m in MODELS:
        name = m.split("-")[1]
        if name in state:
            continue
        reqs = [{"custom_id": f"c{int(i)}", "params": dict(model=m, max_tokens=5, system=SYSTEM,
                 thinking={"type": "disabled"} if "haiku" in m else {"type": "between_tools"},
                 messages=[dict(role="user", content=prompt(items[int(i)]))])} for i in order]
        b = json.loads(call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs}))
        state[name] = {"batch": b["id"], "n": len(reqs), "t": time.time()}; json.dump(state, open(ST, "w"), indent=1)
        print("submitted", name, b["id"], len(reqs), flush=True)
else:
    state = json.load(open(ST)); pat = re.compile(r"\b([ABCD])\b")
    while True:
        done = True
        for name, v in state.items():
            if v.get("collected"):
                continue
            b = json.loads(call(f"https://api.anthropic.com/v1/messages/batches/{v['batch']}"))
            if b["processing_status"] != "ended":
                done = False; print(time.strftime("%H:%M:%S"), name, b["request_counts"], flush=True); continue
            raw = call(b["results_url"]); n = 0
            with open(f"{D}/{name}.jsonl", "w") as fo:
                for line in raw.decode().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line); iid = int(r["custom_id"][1:])
                    if r["result"]["type"] != "succeeded":
                        fo.write(json.dumps(dict(id=iid, error=r["result"]["type"])) + "\n"); continue
                    txt = "".join(x.get("text", "") for x in r["result"]["message"].get("content", []) if x.get("type") == "text").strip()
                    mm = pat.search(txt); fo.write(json.dumps(dict(id=iid, ans=mm.group(1) if mm else None, raw=txt[:20])) + "\n"); n += 1
            v["collected"] = True; json.dump(state, open(ST, "w"), indent=1); print("collected", name, n, flush=True)
        if done:
            break
        time.sleep(300)
