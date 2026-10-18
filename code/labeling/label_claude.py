"""Claude annotator (claude-haiku-5-5 or claude-sonnet-5-5 only). Key from the environment variable ANTHROPIC_API_KEY into memory; never printed.
usage: python -I label_claude.py ITEMS OUT --model claude-haiku-5-5 --task tox|ida --context 0|1 [--workers 16] [--ids FILE]"""
import argparse, json, os, random, re, subprocess, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import TASKS, SYSTEM, SYSTEMS
WORKSPACE = os.environ.get("ANTHROPIC_WORKSPACE_ID", ""); ALLOWED = {"claude-haiku-5-5", "claude-sonnet-5-5"}
def api_key():
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k: raise SystemExit("no API key (set ANTHROPIC_API_KEY)")
    return k
def ask(it, a, key, fn, opts):
    prompt = fn(it["text"], it["parent_text"] if a.context else None)
    body = dict(model=a.model, max_tokens=5, system=SYSTEMS.get(a.task, SYSTEM), thinking={"type": "disabled" if "haiku" in a.model else "between_tools"}, messages=[dict(role="user", content=prompt)])
    h = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json", **({"anthropic-workspace-id": WORKSPACE} if WORKSPACE else {})}
    pat = re.compile(r"\b([" + "".join(opts) + r"])\b")
    for k in range(10):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(), headers=h)
            r = json.load(urllib.request.urlopen(req, timeout=120))
            txt = "".join(x.get("text", "") for x in r.get("content", [])).strip()
            m = pat.search(txt); us = r.get("usage", {})
            return dict(id=it["id"], ans=m.group(1) if m else None, raw=txt[:20], in_tok=us.get("input_tokens"), out_tok=us.get("output_tokens"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529):
                time.sleep(min(60, 2 ** k + random.random())); continue
            return dict(id=it["id"], error=str(e.code))
        except Exception:
            time.sleep(min(60, 2 ** k))
    return dict(id=it["id"], error="retries")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--model", default="claude-haiku-5-5"); ap.add_argument("--task", required=True); ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--ids", default=None); a = ap.parse_args()
    assert a.model in ALLOWED
    fn, opts = TASKS[a.task]
    items = [json.loads(l) for l in open(a.items)]
    if a.context: items = [it for it in items if it.get("parent_text")]
    if a.ids: keep = set(int(x) for x in open(a.ids).read().split()); items = [it for it in items if it["id"] in keep]
    done = set()
    if os.path.exists(a.out): done = {json.loads(l)["id"] for l in open(a.out) if '"error"' not in l}
    todo = [it for it in items if it["id"] not in done]
    print(f"{a.model} {a.task} ctx={a.context}: {len(todo)} to do of {len(items)}", flush=True)
    key = api_key(); lock = threading.Lock(); n = [0, 0, 0]; t0 = time.time()
    with open(a.out, "a") as fo, ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lambda it: ask(it, a, key, fn, opts), todo):
            with lock:
                fo.write(json.dumps(r) + "\n"); n[0] += 1; n[1] += r.get("in_tok") or 0; n[2] += r.get("out_tok") or 0
                if n[0] % 1000 == 0:
                    fo.flush(); print(f"{n[0]}/{len(todo)} {n[0]/(time.time()-t0):.1f}/s in={n[1]} out={n[2]}", flush=True)
    print(f"done {n[0]} in_tok={n[1]} out_tok={n[2]} {time.time()-t0:.0f}s", flush=True)
