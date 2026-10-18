"""Unit-level LLM judgment: show a whole thread (comments in time order, with reply links) and ask whether at least one
comment meets the gold's own definition (Toxic or Very toxic). Batches API; claude-haiku-5-5 / claude-sonnet-5-5 only.
usage: python -I label_threads_batch.py ITEMS OUT --model claude-haiku-5-5 [--maxk 40] [--cap 600]"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM, TOX_SCALE
from label_claude import api_key, WORKSPACE, ALLOWED
ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out")
ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--maxk", type=int, default=40); ap.add_argument("--cap", type=int, default=600)
a = ap.parse_args(); assert a.model in ALLOWED
it = [json.loads(l) for l in open(a.items)]; it = [r for r in it if r["src"] == "pilot"]; by = {r["id"]: r for r in it}
def root(i, d=0):
    p = by[i]["parent_id"]
    return i if (p is None or p not in by or d > 200) else root(p, d + 1)
th = {}
for r in it: th.setdefault(root(r["id"]), []).append(r["id"])
th = {k: sorted(v, key=lambda i: by[i]["created"]) for k, v in th.items() if len(v) >= 2}
def q(t): t = (t or "").strip().replace('"""', "'''"); return t[:a.cap] + (" [...]" if len(t) > a.cap else "")
def prompt(ids):
    ids = ids[:a.maxk]; pos = {i: n + 1 for n, i in enumerate(ids)}; lines = []
    for i in ids:
        p = by[i]["parent_id"]; tag = f" (reply to [{pos[p]}])" if p in pos else ""
        lines.append(f"[{pos[i]}]{tag}: \"\"\"{q(by[i]['text'])}\"\"\"")
    return ("Below is a conversation thread from the comment section of a news article, in time order.\n\n"
            "Each comment can be rated on this scale:\n" + TOX_SCALE + "\n\n" + "\n\n".join(lines) +
            "\n\nIs at least one comment in this thread Toxic or Very toxic?\nA) Yes\nB) No\n\nAnswer with a single letter (A or B).")
done = set()
if os.path.exists(a.out): done = {json.loads(l)["root"] for l in open(a.out) if '"error"' not in l}
todo = [k for k in th if k not in done]
print(f"{len(todo)} threads to do of {len(th)}; truncated (> maxk): {sum(len(th[k]) > a.maxk for k in todo)}", flush=True)
if not todo: sys.exit(0)
key = api_key(); H = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json", **({"anthropic-workspace-id": WORKSPACE} if WORKSPACE else {})}
def call(url, data=None):
    for k in range(8):
        try:
            req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(), headers=H)
            with urllib.request.urlopen(req, timeout=300) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529): time.sleep(min(120, 5 * 2 ** k)); continue
            raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except Exception: time.sleep(min(120, 5 * 2 ** k))
    raise SystemExit("retries")
think = {"type": "disabled"} if "haiku" in a.model else {"type": "between_tools"}
reqs = [{"custom_id": f"t_{k}", "params": dict(model=a.model, max_tokens=5, system=SYSTEM, thinking=think,
         messages=[dict(role="user", content=prompt(th[k]))])} for k in todo]
bid = json.loads(call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs}))["id"]; print("batch", bid, len(reqs), flush=True)
while True:
    b = json.loads(call(f"https://api.anthropic.com/v1/messages/batches/{bid}"))
    print(time.strftime("%H:%M:%S"), b["processing_status"], b["request_counts"], flush=True)
    if b["processing_status"] == "ended": break
    time.sleep(60)
n = ni = 0
with open(a.out, "a") as fo:
    for line in call(b["results_url"]).decode().splitlines():
        if not line.strip(): continue
        r = json.loads(line); k = int(r["custom_id"][2:])
        if r["result"]["type"] != "succeeded": fo.write(json.dumps(dict(root=k, error=r["result"]["type"])) + "\n"); continue
        m = r["result"]["message"]; txt = "".join(x.get("text", "") for x in m.get("content", [])).strip(); mm = re.search(r"\b([AB])\b", txt)
        fo.write(json.dumps(dict(root=k, k=len(th[k]), ans=mm.group(1) if mm else None, raw=txt[:20], in_tok=m["usage"].get("input_tokens"))) + "\n")
        n += 1; ni += m["usage"].get("input_tokens") or 0
print(f"done {n} in_tok={ni}", flush=True)
