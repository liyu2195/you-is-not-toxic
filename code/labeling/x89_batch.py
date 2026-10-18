"""Addendum 89: a generic, resumable Message Batches runner for prepared requests (claude-haiku-5-5 / claude-sonnet-5-5 only).
REQ_DIR holds req_XXX.jsonl files, one batch each, every line {"custom_id": ..., "params": {...}} (<= 100,000 lines, < 256 MB).
State: OUT_DIR/state.json (mirrored to the bucket if --bucket). Results: OUT_DIR/req_XXX.out.jsonl, one line per request:
{custom_id, ans, raw, error, in_tok, out_tok, cache_read}. The answer is the whole text if it is a single option letter, else the
last standalone option letter in the text (None if there is none).
usage: python3 -I x89_batch.py REQ_DIR OUT_DIR [--opts ABCD] [--max_inflight 4] [--bucket $YOU_IS_NOT_TOXIC_BUCKET/results/x89/NAME]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import argparse, glob, json, os, re, subprocess, sys, time, urllib.error, urllib.request
sys.path.insert(0, (_ROOT + "/code/labeling"))
from label_claude import api_key, WORKSPACE, ALLOWED
ap = argparse.ArgumentParser(); ap.add_argument("reqs"); ap.add_argument("out"); ap.add_argument("--opts", default="ABCD")
ap.add_argument("--max_inflight", type=int, default=4); ap.add_argument("--bucket", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); SP = f"{a.out}/state.json"
state = json.load(open(SP)) if os.path.exists(SP) else {}
GENV = {k: v for k, v in os.environ.items() if k != "CLOUDSDK_AUTH_ACCESS_TOKEN"}
one = re.compile(r"^\W*([" + a.opts + r"])\W*$"); anyl = re.compile(r"\b([" + a.opts + r"])\b")


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
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503, 504, 529):
                if data is not None and e.code == 429:
                    return ("RATE", body)
                time.sleep(min(300, 10 * 2 ** k)); continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except Exception:
            time.sleep(min(300, 10 * 2 ** k))
    return ("FAIL", "retries")


def parse(txt):
    m = one.match(txt)
    if m:
        return m.group(1)
    ms = anyl.findall(txt)
    return ms[-1] if ms else None


def collect(name, b):
    raw = call(b["results_url"])
    if isinstance(raw, tuple):
        return False
    n = 0; outp = f"{a.out}/{name}.out.jsonl"
    with open(outp + ".tmp", "w") as fo:
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line); cid = r["custom_id"]
            if r["result"]["type"] != "succeeded":
                fo.write(json.dumps(dict(custom_id=cid, ans=None, error=r["result"]["type"])) + "\n"); continue
            m = r["result"]["message"]; txt = "".join(x.get("text", "") for x in m.get("content", []) if x.get("type") == "text").strip(); us = m.get("usage", {})
            fo.write(json.dumps(dict(custom_id=cid, ans=parse(txt), raw=txt[:40], error=None, in_tok=us.get("input_tokens"), out_tok=us.get("output_tokens"),
                                     cache_read=us.get("cache_read_input_tokens"))) + "\n"); n += 1
    os.replace(outp + ".tmp", outp); gsync(outp); return n


files = sorted(glob.glob(f"{a.reqs}/req_*.jsonl")); t_rate = 0
while True:
    pending = [f for f in files if os.path.basename(f)[:-6] not in state]
    inflight = [k for k, v in state.items() if v["status"] == "submitted"]
    while pending and len(inflight) < a.max_inflight and time.time() >= t_rate:
        f = pending.pop(0); name = os.path.basename(f)[:-6]; reqs = [json.loads(l) for l in open(f)]
        assert all(r["params"]["model"] in ALLOWED for r in reqs)
        r = call("https://api.anthropic.com/v1/messages/batches", {"requests": reqs})
        if isinstance(r, tuple):
            print(time.strftime("%H:%M:%S"), "submit deferred", name, r[0], r[1][:160], flush=True); t_rate = time.time() + 300; break
        bid = json.loads(r)["id"]; state[name] = {"status": "submitted", "batch": bid, "n": len(reqs), "t": time.time()}; save(); inflight.append(name)
        print(time.strftime("%H:%M:%S"), "submitted", name, bid, len(reqs), flush=True)
    for name in list(inflight):
        r = call(f"https://api.anthropic.com/v1/messages/batches/{state[name]['batch']}")
        if isinstance(r, tuple):
            continue
        b = json.loads(r)
        if b["processing_status"] == "ended":
            n = collect(name, b)
            if n is not False:
                state[name].update(status="done", n_ok=n, counts=b["request_counts"]); save()
                print(time.strftime("%H:%M:%S"), "done", name, n, b["request_counts"], flush=True)
    if sum(v["status"] == "done" for v in state.values()) == len(files):
        print("all done", flush=True); break
    time.sleep(60)
