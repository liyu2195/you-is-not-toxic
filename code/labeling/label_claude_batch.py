"""Claude annotator through the Message Batches API (same params as label_claude.py). Appends to the same JSONL format.
usage: python -I label_claude_batch.py ITEMS OUT --model claude-haiku-5-5 --task tox|ida --context 0|1 [--ids FILE]"""
import argparse, json, os, re, subprocess, sys, time, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TASKS, SYSTEM, SYSTEMS
from label_claude import api_key, WORKSPACE, ALLOWED
ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out")
ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--task", required=True); ap.add_argument("--context", type=int, default=0)
ap.add_argument("--ids", default=None); a = ap.parse_args()
assert a.model in ALLOWED
fn, opts = TASKS[a.task]; pat = re.compile(r"\b([" + "".join(opts) + r"])\b")
items = [json.loads(l) for l in open(a.items)]
if a.context: items = [it for it in items if it.get("parent_text")]
if a.ids: keep = set(int(x) for x in open(a.ids).read().split()); items = [it for it in items if it["id"] in keep]
done = set()
if os.path.exists(a.out): done = {json.loads(l)["id"] for l in open(a.out) if '"error"' not in l}
todo = [it for it in items if it["id"] not in done]
print(f"{a.model} {a.task} ctx={a.context}: {len(todo)} to do of {len(items)}", flush=True)
if not todo: sys.exit(0)
key = api_key()
H = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json", **({"anthropic-workspace-id": WORKSPACE} if WORKSPACE else {})}
def call(url, data=None, method=None):
    for k in range(8):
        try:
            req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers=H, method=method)
            with urllib.request.urlopen(req, timeout=300) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529): time.sleep(min(120, 5 * 2 ** k)); continue
            raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except Exception as e:
            time.sleep(min(120, 5 * 2 ** k))
    raise SystemExit("retries")
pref = f"{a.task}{a.context}_"
reqs = [{"custom_id": pref + str(it["id"]),
         "params": dict(model=a.model, max_tokens=5, system=SYSTEMS.get(a.task, SYSTEM), thinking={"type": "disabled" if "haiku" in a.model else "between_tools"},
                        messages=[dict(role="user", content=fn(it["text"], it["parent_text"] if a.context else None))])} for it in todo]
bid = json.loads(call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs}))["id"]
print("batch", bid, len(reqs), flush=True)
while True:
    b = json.loads(call(f"https://api.anthropic.com/v1/messages/batches/{bid}"))
    print(time.strftime("%H:%M:%S"), b["processing_status"], b["request_counts"], flush=True)
    if b["processing_status"] == "ended": break
    time.sleep(60)
raw = call(b["results_url"]).decode()
n = ni = no = 0
with open(a.out, "a") as fo:
    for line in raw.splitlines():
        if not line.strip(): continue
        r = json.loads(line); iid = int(r["custom_id"][len(pref):])
        if r["result"]["type"] != "succeeded":
            fo.write(json.dumps(dict(id=iid, error=r["result"]["type"])) + "\n"); continue
        m = r["result"]["message"]; txt = "".join(x.get("text", "") for x in m.get("content", [])).strip()
        mm = pat.search(txt); us = m.get("usage", {})
        fo.write(json.dumps(dict(id=iid, ans=mm.group(1) if mm else None, raw=txt[:20], in_tok=us.get("input_tokens"), out_tok=us.get("output_tokens"), batch=1)) + "\n")
        n += 1; ni += us.get("input_tokens") or 0; no += us.get("output_tokens") or 0
print(f"done {n} in_tok={ni} out_tok={no}", flush=True)
