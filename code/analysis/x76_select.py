"""Addendum 76, B76a: the instrument that agreement selects (descriptive).
For each population, the instrument with the highest kappa (all instruments of x74_decomp.json except the lenient human, which is
built from the gold raters' votes), and the one with the highest kappa among instruments flagging within +-25% of gold prevalence.
For each law (depth, thread size, article size, log RR, within-thread depth from x75_within.json): its distortion in natural units
and in SE units, and the rank of its |distortion| among the population's instruments (1 = the smallest). Also, as a variant that
was not registered, the same with Detoxify-unbiased excluded on Civil Comments (it is trained on that gold).
usage: python3 x76_select.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, numpy as np
W = _ROOT; OUT = sys.argv[1]
DEC = json.load(open(f"{W}/results/x74_decomp/x74_decomp.json")); WI = json.load(open(f"{W}/results/x75_within/x75_within.json"))["within"]
LAWS = ["depth", "size", "article_size", "log_rr", "depth_within"]
res = {}
for pop, D in DEC.items():
    prev = D["prev"]; labs = {L: r for L, r in D["labelers"].items() if L != "human_lenient"}
    def dist(L, law):
        if law == "depth_within":
            r = WI[pop]["labelers"].get(L)
            return (r["distortion"][0], r["z"]) if r else (None, None)
        if law == "log_rr" and pop == "cga":
            return (None, None)
        r = labs[L]["laws"].get(law)
        return (r["distortion"][0], r["z"]) if r else (None, None)
    out = {}
    for variant, excl in [("all", set()), ("no_detox_unbiased", {"detoxify_unbiased@0.6", "detoxify_unbiased@m"} if pop.startswith("cc_") else set())]:
        cand = {L: r for L, r in labs.items() if L not in excl}
        for sel, pool in [("top_kappa", cand), ("top_kappa_matched", {L: r for L, r in cand.items() if abs(r["flag_rate"][0] / prev - 1) <= 0.25})]:
            if not pool:
                continue
            best = max(pool, key=lambda L: pool[L]["kappa"][0]); row = {"instrument": best, "kappa": pool[best]["kappa"][0], "flag_rate": pool[best]["flag_rate"][0], "laws": {}}
            for law in LAWS:
                d, zz = dist(best, law)
                if d is None:
                    continue
                allv = [abs(dist(L, law)[0]) for L in cand if dist(L, law)[0] is not None]
                row["laws"][law] = dict(distortion=d, z=zz, rank=int(1 + sum(v < abs(d) for v in allv)), n=len(allv), min_abs=float(min(allv)))
            out[f"{variant}/{sel}"] = row
    res[pop] = out
    for k, row in out.items():
        print(pop, k, row["instrument"], f"k={row['kappa']:.3f} flag={row['flag_rate']/prev:.2f}x",
              {law: (f"{v['distortion']*(1 if law == 'log_rr' else 1e3):+.2f}", f"z={v['z']:+.1f}", f"rank {v['rank']}/{v['n']}") for law, v in row["laws"].items()})
summ = {}
for key in ["all/top_kappa", "all/top_kappa_matched", "no_detox_unbiased/top_kappa"]:
    rows = [(p, r[key]) for p, r in res.items() if key in r]
    summ[key] = dict(n_pops=len(rows), n_pops_any_z3=sum(any(abs(v["z"]) > 3 for v in r["laws"].values() if v["z"] is not None) for _, r in rows),
                     n_law_entries=sum(len(r["laws"]) for _, r in rows), n_entries_z3=sum(sum(abs(v["z"]) > 3 for v in r["laws"].values() if v["z"] is not None) for _, r in rows),
                     median_rank_share=float(np.median([v["rank"] / v["n"] for _, r in rows for v in r["laws"].values()])))
res["summary"] = summ
print(json.dumps(summ, indent=1))
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
