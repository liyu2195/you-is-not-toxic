"""Addendum 81, B81a: within-batch replication of B80e. Haiku 5.5 and Sonnet 5.5 label every comment of the 5% Civil Comments
sample (cc_s5) without context under the main prompt (task tox) and under the note (task tox_you), both prompts shuffled together
(seed 8100) and split into two Message Batches per model (each a random half of both prompts' requests). Settings as the main
labels: max_tokens 5, the main system prompt, thinking disabled for Haiku and between_tools for Sonnet.
  submit SHARD_DIR   build and submit the four batches (state in results/x81/state.json)
  collect            poll until all batches end and write results/x81/<model>.jsonl  {id, task, ans}
usage: python3 x81_youprompt_rep.py submit SHARD_DIR | collect"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, json, glob, time, urllib.request, urllib.error
import numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x81"; os.makedirs(D, exist_ok=True); ST = f"{D}/state.json"
sys.path.insert(0, f"{W}/code/labeling")
from label_claude import api_key, WORKSPACE
from prompts import SYSTEM, TASKS
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]; TASK = {"m": "tox", "y": "tox_you"}
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
    keys = [(t, i) for t in ("m", "y") for i in sorted(items)]
    perm = np.random.default_rng(8100).permutation(len(keys)); keys = [keys[j] for j in perm]
    halves = [keys[: len(keys) // 2], keys[len(keys) // 2:]]
    print("items", len(items), "of", len(need), "requests", len(keys), flush=True)
    state = json.load(open(ST)) if os.path.exists(ST) else {}
    for m in MODELS:
        name = m.split("-")[1]
        for h, part in enumerate(halves):
            tag = f"{name}_{h}"
            if tag in state:
                continue
            reqs = [{"custom_id": f"{t}{int(i)}", "params": dict(model=m, max_tokens=5, system=SYSTEM,
                     thinking={"type": "disabled"} if "haiku" in m else {"type": "between_tools"},
                     messages=[dict(role="user", content=TASKS[TASK[t]][0](items[int(i)]))])} for t, i in part]
            b = json.loads(call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs}))
            state[tag] = {"batch": b["id"], "n": len(reqs), "t": time.time()}; json.dump(state, open(ST, "w"), indent=1)
            print("submitted", tag, b["id"], len(reqs), flush=True)
else:
    state = json.load(open(ST)); pat = re.compile(r"\b([ABCD])\b")
    while True:
        done = True
        for tag, v in state.items():
            if v.get("collected"):
                continue
            b = json.loads(call(f"https://api.anthropic.com/v1/messages/batches/{v['batch']}"))
            if b["processing_status"] != "ended":
                done = False; print(time.strftime("%H:%M:%S"), tag, b["request_counts"], flush=True); continue
            raw = call(b["results_url"]); n = 0
            with open(f"{D}/{tag}.jsonl", "w") as f:
                for line in raw.decode().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line); cid = r["custom_id"]; ans = None; txt = ""
                    if r["result"]["type"] == "succeeded":
                        txt = "".join(c.get("text", "") for c in r["result"]["message"]["content"] if c.get("type") == "text").strip()
                        mm = pat.search(txt); ans = mm.group(1) if mm else None
                    f.write(json.dumps({"id": int(cid[1:]), "task": TASK[cid[0]], "ans": ans, "raw": txt[:20]}) + "\n"); n += 1
            v["collected"] = True; json.dump(state, open(ST, "w"), indent=1); print("collected", tag, n, flush=True)
        if done:
            break
        time.sleep(300)
    for name in ("haiku", "sonnet"):
        with open(f"{D}/{name}.jsonl", "w") as f:
            for h in (0, 1):
                f.write(open(f"{D}/{name}_{h}.jsonl").read())
    print("merged", flush=True)
