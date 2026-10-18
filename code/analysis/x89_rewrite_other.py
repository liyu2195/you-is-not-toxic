"""Addendum 89, B89d: the fixed-rule rewrite on CAD, Qian Reddit, Qian Gab and CGA-Wiki.
  build     every gold-negative comment that the 'you' detector matches, original (o_<corpus>_<id>) and edited (e_<corpus>_<id>) by
            f80_you_edit_lib.edit, labeled by Haiku 5.5 without context (task tox, B89a settings), shuffled together (seed 8908)
            -> results/x89/rewrite_other/req/req_*.jsonl
  analyze OUT_JSON  per corpus: flag rates of originals and edits, paired difference, 2,000 conversation-cluster resamples (seed 8909);
            also at depth >= 2. H89f: the difference has CI below 0 on each corpus.
usage: python3 x89_rewrite_other.py build | analyze OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, re, glob, json, numpy as np, pandas as pd
W = _ROOT; D = f"{W}/results/x89/rewrite_other"
sys.path.insert(0, f"{W}/code/labeling"); sys.path.insert(0, f"{W}/code/analysis")
from prompts import SYSTEM, tox_prompt
from f80_you_edit_lib import edit
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
CORP = ["cad", "qian_reddit", "qian_gab", "cga"]
if sys.argv[1] == "build":
    rows = []
    for c in CORP:
        P = pd.read_parquet(f"{W}/results/x14_pops/{c}.parquet", columns=["id", "z"]); neg = set(P.id[P.z == 0].astype(int))
        for f in sorted(glob.glob(f"{W}/results/x89/shards/{c}/shard_*.jsonl")):
            for l in open(f):
                r = json.loads(l); i = int(r["id"]); t = r.get("text") or ""
                if i in neg and YOU.search(t):
                    for tag, tx in (("o", t), ("e", edit(t))):
                        rows.append({"custom_id": f"{tag}_{c}_{i}", "params": dict(model="claude-haiku-5-5", max_tokens=5, system=SYSTEM, thinking={"type": "disabled"},
                                                                                messages=[dict(role="user", content=tox_prompt(tx))])})
    rows = [rows[j] for j in np.random.default_rng(8908).permutation(len(rows))]
    os.makedirs(f"{D}/req", exist_ok=True)
    for j in range(0, len(rows), 30000):
        with open(f"{D}/req/req_{j // 30000:03d}.jsonl", "w") as fo:
            for r in rows[j: j + 30000]:
                fo.write(json.dumps(r) + "\n")
    print("requests", len(rows))
else:
    OUT = sys.argv[2]; A = {}
    for f in sorted(glob.glob(f"{D}/out/req_*.out.jsonl")):
        for l in open(f):
            r = json.loads(l); A[r["custom_id"]] = r["ans"]
    res = {}
    for c in CORP:
        P = pd.read_parquet(f"{W}/results/x14_pops/{c}.parquet", columns=["id", "depth", "cluster"]).set_index("id")
        ids = sorted({int(k.split("_")[-1]) for k in A if k.split("_", 1)[1].rsplit("_", 1)[0] == c})
        o = np.array([float(A.get(f"o_{c}_{i}") in ("A", "B")) for i in ids]); e = np.array([float(A.get(f"e_{c}_{i}") in ("A", "B")) for i in ids])
        cl = pd.factorize(P.loc[ids, "cluster"].values)[0]; dep = P.loc[ids, "depth"].values; nC = int(cl.max() + 1)
        Wb = np.vstack([np.ones(nC), np.random.default_rng(8909).poisson(1.0, size=(2000, nC))])
        def ci(mask):
            n_ = Wb @ np.bincount(cl, mask.astype(float), nC); so = Wb @ np.bincount(cl, o * mask, nC); se = Wb @ np.bincount(cl, e * mask, nC)
            d = (se - so) / n_
            return dict(n=int(mask.sum()), orig=float(so[0] / n_[0]), edit=float(se[0] / n_[0]), diff=[float(d[0]), float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))])
        res[c] = dict(all=ci(np.ones(len(ids), bool)), deep=ci(dep >= 2))
        print(c, res[c], flush=True)
    res["H89f"] = dict(n_pass=int(sum(res[c]["all"]["diff"][2] < 0 for c in CORP)), pass_=bool(all(res[c]["all"]["diff"][2] < 0 for c in CORP)))
    print("H89f", res["H89f"]); json.dump(res, open(OUT, "w"), indent=1)
