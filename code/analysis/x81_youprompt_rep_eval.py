"""Addendum 81, B81a evaluation: the note (task tox_you) against the main prompt (task tox), both labeled in the same batches
(results/x81/<model>.jsonl), Haiku 5.5 and Sonnet 5.5 on the 5% sample (cc_s5), without context. Flags: A or B. A comment without a
valid answer under a prompt keeps that prompt's flag missing and is dropped from that model's paired comparison (reported).
Statistics as x80_youprompt_eval.py, on 500 article-cluster Poisson resamples (seed 8041), paired: the depth-law and within-thread
distortions under each prompt and their change; TPR and FPR; civil-comment flag rates by depth and second-person address.
Batch drift: the new main-prompt flags against the stored y:<model> (flag rates, agreement, depth-law distortion difference).
H81a: Haiku's change (note minus main) in the depth-law distortion has a 95% CI above 0.
H81b: under the note, both models' depth-law distortions have CIs above 0 and point values above 75% of their main-prompt ones.
usage: python3 x81_youprompt_rep_eval.py TEXT_SHARD_DIR OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, re, glob, json, numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code/analysis"))
from x80_common import Pop, ci
W = _ROOT; SH, OUT = sys.argv[1], sys.argv[2]; PP = 100.0
YOU = re.compile(r"\b(you|your|yours|yourself|yourselves|you're|youre|you've|you'd|you'll)\b", re.I)
S = Pop("cc_s5", seed=8041); P = S.P; ids = P.id.astype(np.int64).values; pos = pd.Series(np.arange(S.N), index=ids)
you = np.zeros(S.N, bool)
for f in sorted(glob.glob(f"{SH}/shard_*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = pos.get(int(r["id"]))
        if i is not None:
            you[i] = bool(YOU.search(r.get("text") or ""))
uu = pd.factorize(P.unit.values)[0]; xt = S.x - (np.bincount(uu, S.x) / np.bincount(uu))[uu]
def within(v, m):
    w = m.astype(float); sxx = S.Wb @ np.bincount(S.cl, w * xt * xt, S.nC)
    return (S.Wb @ np.bincount(S.cl, w * xt * v, S.nC)) / sxx
def depth_m(v, m):
    """Depth slope over the comments in mask m (Pop.ols with a mask)."""
    return S.ols(v, m)[0]
res = {"n": S.N, "you_share": float(you.mean()), "models": {}}
for m in ("haiku", "sonnet"):
    F = {"tox": np.full(S.N, np.nan), "tox_you": np.full(S.N, np.nan)}
    for l in open(f"{W}/results/x81/{m}.jsonl"):
        r = json.loads(l); i = pos.get(int(r["id"]))
        if i is not None and r.get("ans") in ("A", "B", "C", "D"):
            F[r["task"]][i] = float(r["ans"] in ("A", "B"))
    ok = ~np.isnan(F["tox"]) & ~np.isnan(F["tox_you"]); main = np.nan_to_num(F["tox"]); new = np.nan_to_num(F["tox_you"])
    z = S.z; gd = depth_m(z, ok); gw = within(z, ok)
    dm, dy = depth_m(main, ok) - gd, depth_m(new, ok) - gd; wm, wy = within(main, ok) - gw, within(new, ok) - gw
    out = {"n_paired": int(ok.sum()), "n_missing_main": int(np.isnan(F["tox"]).sum()), "n_missing_you": int(np.isnan(F["tox_you"]).sum()),
           "flag_main": float(main[ok].mean()), "flag_you": float(new[ok].mean()),
           "depth_dist_main": ci(dm), "depth_dist_you": ci(dy), "depth_change": ci(dy - dm), "within_dist_main": ci(wm), "within_dist_you": ci(wy), "within_change": ci(wy - wm)}
    zs = S.bc(z * ok); ns = S.bc(ok.astype(float))
    for nm, v in (("main", main), ("you", new)):
        out[f"tpr_{nm}"] = ci(S.bc(z * v * ok) / zs); out[f"fpr_{nm}"] = ci(S.bc((1 - z) * v * ok) / (ns - zs))
    civ = (S.tox < 0.2) & ok; d = P.depth.values
    for nm, v in (("main", main), ("you", new)):
        out[f"civil_{nm}"] = {f"{dl}_{yl}": float(v[civ & dmask & ymask].mean()) for dl, dmask in (("d0", d == 0), ("d6", d >= 6)) for yl, ymask in (("you", you), ("noyou", ~you))}
    stored = P[f"y:{m}"].values.astype(float); sv = ~np.isnan(stored) & ~np.isnan(F["tox"]); st = np.nan_to_num(stored)
    out["drift"] = dict(n=int(sv.sum()), flag_stored=float(st[sv].mean()), flag_new=float(main[sv].mean()), agreement=float((st[sv] == main[sv]).mean()),
                        depth_dist_stored=ci(depth_m(st, sv) - depth_m(z, sv)), depth_dist_new=ci(depth_m(main, sv) - depth_m(z, sv)),
                        depth_new_minus_stored=ci(depth_m(main, sv) - depth_m(st, sv)))
    out["share_removed"] = float(1 - dy[0] / dm[0])
    res["models"][m] = out
    print(m, f"paired {ok.sum()} | flags {out['flag_main']:.3f} -> {out['flag_you']:.3f} | depth distortion {dm[0]*PP:+.2f} -> {dy[0]*PP:+.2f}, change {(dy-dm)[0]*PP:+.2f} [{np.percentile((dy-dm)[1:], 2.5)*PP:+.2f}, {np.percentile((dy-dm)[1:], 97.5)*PP:+.2f}]",
          f"| within {wm[0]*PP:+.2f} -> {wy[0]*PP:+.2f} | drift: agree {out['drift']['agreement']:.3f}, flags {out['drift']['flag_stored']:.3f} vs {out['drift']['flag_new']:.3f}, depth new-stored {out['drift']['depth_new_minus_stored'][0]*PP:+.2f}",
          json.dumps({k: {kk: round(vv, 3) for kk, vv in out[k].items()} for k in ("civil_main", "civil_you")}), flush=True)
H_ = res["models"]["haiku"]; So = res["models"]["sonnet"]
res["H81a"] = dict(change=H_["depth_change"], pass_=bool(H_["depth_change"][1] > 0))
res["H81b"] = dict(detail={m: dict(you=res["models"][m]["depth_dist_you"], main=res["models"][m]["depth_dist_main"]) for m in ("haiku", "sonnet")},
                   pass_=bool(all(res["models"][m]["depth_dist_you"][1] > 0 and res["models"][m]["depth_dist_you"][0] > 0.75 * res["models"][m]["depth_dist_main"][0] for m in ("haiku", "sonnet"))))
print("H81a", res["H81a"]["pass_"], "H81b", res["H81b"]["pass_"]); json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
