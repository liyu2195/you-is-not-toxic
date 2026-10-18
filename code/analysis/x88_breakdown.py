"""Addendum 88, B88g-i (descriptive).
B88g: B80h's law estimates (results/x80/x80_magnitude rows rebuilt as in x80_magnitude.py) by population, law and family.
B88h: the parent's contrast effect on reply threads: Haiku and Qwen (gold prevalence), with and without the parent, flag rates on
      gold-negative and gold-positive replies by the parent's gold label.
B88i: active sampling with THREADSAFE's own working model (the single-fold ablation pps_seq1_ba, results/x74_b1b3) against
      THREADSAFE (pps_seq4_ba, results/x11_conf_item_all + x72_item_cga), per budget: median width ratio and coverage, laws and
      placebo slopes; edges: edge_m1_c15 against edge_m3_c15 (results/x11_conf_edge_all without CGA-Wiki).
usage: python3 x88_breakdown.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np, pandas as pd
W = _ROOT; OUT = sys.argv[1]
import glob, os


def load_dir(d, reg, cells=None, exclude=None):                         # as x85_small_budgets.py
    cells = {} if cells is None else cells
    for f in sorted(glob.glob(f"{d}/*.json")):
        if os.path.basename(f).startswith("eval"):
            continue
        J = json.load(open(f))
        for R in J["results"]:
            tag = f"{J['pop']}__{R['labeler'].replace('@', '_at_')}"
            if exclude and J["pop"] == exclude:
                continue
            for B, dd in (R.get(reg) or {}).items():
                for des, mm in dd.items():
                    cells.setdefault(tag, {}).setdefault(int(B), {})[des] = mm
            if reg == "item" and R.get("truth_item"):
                cells.setdefault(tag, {}).setdefault("_truth", R["truth_item"])
    return cells


def defined(C, m):
    t = C.get("_truth", {}).get(m)
    return t is None or bool(np.isfinite(t))


res = {}
# ---------------------------------------------------------------- B88g
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json")); WI = json.load(open(f"{W}/results/x75_within/x75_within.json"))["within"]
LLM = ("haiku", "sonnet", "qwen", "gemma"); rows = []
for pop, R in DEC.items():
    if not isinstance(R, dict) or "labelers" not in R:
        continue
    g = R["gold"]
    for L_, r in R["labelers"].items():
        if L_ == "human_lenient":
            continue
        for law, v in r["laws"].items():
            if law in g:
                rows.append(dict(pop=pop, labeler=L_, law=law, gold=g[law], ins=v["law"]))
        w = WI[pop]["labelers"].get(L_)
        if w:
            rows.append(dict(pop=pop, labeler=L_, law="depth_within", gold=WI[pop]["gold_within"], ins=w["law"]))
def sig(c):
    return 1 if c[1] > 0 else (-1 if c[2] < 0 else 0)
tab = {}
for r in rows:
    fam = "llm" if r["labeler"].startswith(LLM) else "classifier"; sg_, si = sig(r["gold"]), sig(r["ins"])
    k = (r["pop"], r["law"], fam); t = tab.setdefault("|".join(k), dict(n=0, off25=0, flip=0, manuf=0, lost=0))
    t["n"] += 1; t["off25"] += int(r["gold"][0] != 0 and abs(r["ins"][0] - r["gold"][0]) > 0.25 * abs(r["gold"][0]))
    t["flip"] += int(si != 0 and sg_ != 0 and si != sg_); t["manuf"] += int(si != 0 and sg_ == 0); t["lost"] += int(si == 0 and sg_ != 0)
res["B88g"] = dict(n=len(rows), table=tab, totals={k: int(sum(t[k] for t in tab.values())) for k in ("n", "off25", "flip", "manuf", "lost")})
print("B88g totals", res["B88g"]["totals"], flush=True)
# ---------------------------------------------------------------- B88h
P = pd.read_parquet(f"{W}/results/x14_pops/cc_threads.parquet", columns=["z", "parent_row", "y:haiku", "y:haiku_ctx", "y:qwen7b_m", "s:qwen7b_m_ctx"])
z = P.z.values; N = len(z); ch = np.flatnonzero(P.parent_row.values >= 0); pa = P.parent_row.values[ch]; keep = pa != ch; ch, pa = ch[keep], pa[keep]
s = P["s:qwen7b_m_ctx"].values.astype(float); s = np.where(np.isnan(s), -np.inf, s); k = int(round(z.mean() * N))
order = np.lexsort((np.random.default_rng(7650).random(N), -s)); qo = np.zeros(N); qo[order[:k]] = 1.0
INS = {"haiku": np.nan_to_num(P["y:haiku"].values.astype(float)), "haiku_ctx": np.nan_to_num(P["y:haiku_ctx"].values.astype(float)),
       "qwen7b_m": np.nan_to_num(P["y:qwen7b_m"].values.astype(float)), "qwen7b_m_ctx@own": qo}
H = {}
for nm, y in INS.items():
    H[nm] = {f"child{zc}_parent{zp}": float(y[ch][(z[ch] == zc) & (z[pa] == zp)].mean()) for zc in (0, 1) for zp in (0, 1)}
res["B88h"] = H; print("B88h", json.dumps(H), flush=True)
# ---------------------------------------------------------------- B88i
LAWS = ["depth_law", "size_law", "article_size_law"]; PLAC = ["placebo_depth", "placebo_size"]
I4 = load_dir(f"{W}/results/x11_conf_item_all", "item", exclude="cga"); I4 = load_dir(f"{W}/results/x72_item_cga", "item", I4)
I1 = load_dir(f"{W}/results/x74_b1b3", "item")
out = {}
for B in (250, 500, 1000, 2000, 4000, 8000):
    rat = {"laws": [], "placebo": []}; cov4 = {"laws": [], "placebo": []}; cov1 = {"laws": [], "placebo": []}
    for tag, C in I4.items():
        if B not in C or "pps_seq4_ba" not in C[B] or tag not in I1 or B not in I1[tag] or "pps_seq1_ba" not in I1[tag][B]:
            continue
        for grp, ms in (("laws", LAWS), ("placebo", PLAC)):
            for m in ms:
                a = C[B]["pps_seq4_ba"].get(m); b = I1[tag][B]["pps_seq1_ba"].get(m)
                if not a or not b or a.get("width") in (None, 0) or b.get("width") is None or not defined(C, m):
                    continue
                rat[grp].append(b["width"] / a["width"]); cov4[grp].append(a["coverage"]); cov1[grp].append(b["coverage"])
    out[str(B)] = {g: dict(n=len(rat[g]), median_ratio=float(np.median(rat[g])) if rat[g] else None, cov_ts=[float(np.mean(cov4[g])), float(np.min(cov4[g]))] if cov4[g] else None,
                           cov_single=[float(np.mean(cov1[g])), float(np.min(cov1[g]))] if cov1[g] else None) for g in rat}
    print("B88i", B, out[str(B)], flush=True)
res["B88i"] = out
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
