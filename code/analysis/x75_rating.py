"""Addendum 75, B75a: 0-100 toxicity ratings from Haiku 5.5 and Sonnet 5.5 on the 5% Civil Comments sample (cc_s5), without and
with the parent, by Message Batches (one batch per model x context). Ratings are parsed as the first integer in the answer.
  submit SHARD_DIR   build requests from the full text shards (id, text, parent_text) and submit the four batches
  collect            poll until the batches end and write results/x75_rating/<model>_ctx<c>.jsonl  {id, rating}
usage: python3 x75_rating.py submit|collect [SHARD_DIR]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, json, glob, time, urllib.request, urllib.error
import numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x75_rating"; os.makedirs(D, exist_ok=True); ST = f"{D}/state.json"
sys.path.insert(0, f"{W}/code/labeling")
from label_claude import api_key, WORKSPACE
from prompts import SYSTEM, _q
DEF = "A toxic comment is a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
def prompt(text, parent=None):
    if parent is None:
        return (f"Rate how toxic the comment below is, on a scale from 0 (not toxic at all) to 100 (extremely toxic). {DEF}\n\n"
                f"Comment:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a number from 0 to 100 only.")
    return (f"The comment below is a reply to an earlier comment, which is shown first for context. Rate how toxic the reply is, on a scale "
            f"from 0 (not toxic at all) to 100 (extremely toxic). {DEF}\n\nEarlier comment (context):\n\"\"\"{_q(parent)}\"\"\"\n\n"
            f"Reply to rate:\n\"\"\"{_q(text)}\"\"\"\n\nAnswer with a number from 0 to 100 only.")
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
    ids = set(pd.read_parquet(f"{W}/results/x14_pops/cc_s5.parquet", columns=["id"]).id.astype(int).tolist())
    items = {}
    for f in sorted(glob.glob(f"{sys.argv[2]}/shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if int(r["id"]) in ids:
                items[int(r["id"])] = (r.get("text") or "", r.get("parent_text"))
    print("items", len(items), "of", len(ids), "with parent", sum(1 for t, p in items.values() if p), flush=True)
    state = json.load(open(ST)) if os.path.exists(ST) else {}
    for m in MODELS:
        for ctx in (0, 1):
            name = f"{m.split('-')[1]}_ctx{ctx}"
            if name in state:
                continue
            reqs = []
            for i, (t, p) in items.items():
                if ctx and not p:
                    continue
                reqs.append({"custom_id": f"c{i}", "params": dict(model=m, max_tokens=5, system=SYSTEM,
                             thinking={"type": "disabled"} if "haiku" in m else {"type": "between_tools"},
                             messages=[dict(role="user", content=prompt(t, p if ctx else None))])})
            b = json.loads(call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs}))
            state[name] = {"batch": b["id"], "n": len(reqs), "t": time.time()}; json.dump(state, open(ST, "w"), indent=1)
            print("submitted", name, b["id"], len(reqs), flush=True)
else:
    state = json.load(open(ST)); pat = re.compile(r"\d{1,3}")
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
                    mm = pat.search(txt); val = int(mm.group(0)) if mm else None
                    fo.write(json.dumps(dict(id=iid, rating=val if (val is not None and val <= 100) else None, raw=txt[:12])) + "\n"); n += 1
            v["collected"] = True; json.dump(state, open(ST, "w"), indent=1); print("collected", name, n, b["request_counts"], flush=True)
        if done:
            break
        time.sleep(120)
    print("all collected")
