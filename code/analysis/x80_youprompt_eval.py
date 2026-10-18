"""Addendum 80, B80e evaluation: the main toxicity prompt against the same prompt plus "Addressing someone directly as 'you' is not
toxic in itself" (task tox_you), Haiku 5.5 and Sonnet 5.5 without context on the 5% sample (cc_s5).
Flags: A or B. Main-prompt flags are the stored y:haiku and y:sonnet. A comment without a valid tox_you answer keeps its main-prompt
flag (reported). Against gold, on 500 article-cluster Poisson resamples (seed 8041), paired: the depth-law distortion under each
prompt and its change (tox_you minus main), the within-thread distortion, TPR and FPR; among civil comments (vote fraction < 0.2) the
flag rate with and without second-person address at depth 0 and depth >= 6.
H80e: Haiku's change has a 95% CI below 0. H80f: the same for Sonnet.
usage: python3 x80_youprompt_eval.py TEXT_SHARD_DIR OUT_JSON"""
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
uu = pd.factorize(P.unit.values)[0]; xt = S.x - (np.bincount(uu, S.x) / np.bincount(uu))[uu]; sxx = S.Wb @ np.bincount(S.cl, xt * xt, S.nC)
within = lambda v: (S.Wb @ np.bincount(S.cl, xt * v, S.nC)) / sxx
zsum = S.bc(S.z); nsum = S.bc(np.ones(S.N))
res = {"n": S.N, "you_share": float(you.mean()), "models": {}}
gd, gw = S.depth(S.z), within(S.z)
for m in ("haiku", "sonnet"):
    main = np.nan_to_num(P[f"y:{m}"].values.astype(float)); new = main.copy(); seen = np.zeros(S.N, bool)
    for l in open(f"{W}/results/x80/youprompt/{m}.jsonl"):
        r = json.loads(l); i = pos.get(int(r["id"]))
        if i is not None and r.get("ans") in ("A", "B", "C", "D"):
            new[i] = float(r["ans"] in ("A", "B")); seen[i] = True
    out = {"n_valid": int(seen.sum()), "flag_main": float(main.mean()), "flag_you": float(new.mean())}
    dm, dy = S.depth(main) - gd, S.depth(new) - gd; wm, wy = within(main) - gw, within(new) - gw
    out.update(depth_dist_main=ci(dm), depth_dist_you=ci(dy), depth_change=ci(dy - dm), within_dist_main=ci(wm), within_dist_you=ci(wy), within_change=ci(wy - wm))
    for nm, v in (("main", main), ("you", new)):
        out[f"tpr_{nm}"] = ci(S.bc(S.z * v) / zsum); out[f"fpr_{nm}"] = ci(S.bc((1 - S.z) * v) / (nsum - zsum))
    civ = S.tox < 0.2; d = P.depth.values
    for nm, v in (("main", main), ("you", new)):
        out[f"civil_{nm}"] = {f"{dl}_{yl}": float(v[civ & dm_ & ym_].mean()) for dl, dm_ in (("d0", d == 0), ("d6", d >= 6)) for yl, ym_ in (("you", you), ("noyou", ~you))}
    out["share_removed"] = float(1 - dy[0] / dm[0])
    res["models"][m] = out
    print(m, f"valid {seen.sum()}/{S.N} flags {main.mean():.3f}->{new.mean():.3f} | depth distortion {dm[0]*PP:+.2f} -> {dy[0]*PP:+.2f} change CI [{(dy-dm)[1:].min()*0:+.0f}]",
          f"change {out['depth_change'][0]*PP:+.2f} [{out['depth_change'][1]*PP:+.2f}, {out['depth_change'][2]*PP:+.2f}] removed {100*out['share_removed']:.0f}% | within {wm[0]*PP:+.2f} -> {wy[0]*PP:+.2f}",
          f"| TPR {out['tpr_main'][0]:.3f}->{out['tpr_you'][0]:.3f} FPR {out['fpr_main'][0]:.3f}->{out['fpr_you'][0]:.3f}", json.dumps({k: {kk: round(vv, 3) for kk, vv in out[k].items()} for k in ('civil_main', 'civil_you')}), flush=True)
res["H80e"] = dict(change=res["models"]["haiku"]["depth_change"], pass_=bool(res["models"]["haiku"]["depth_change"][2] < 0))
res["H80f"] = dict(change=res["models"]["sonnet"]["depth_change"], pass_=bool(res["models"]["sonnet"]["depth_change"][2] < 0))
sg = lambda x, d: (f"+{x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}")
H, So = res["models"]["haiku"], res["models"]["sonnet"]
res["macros"] = dict(qYpHaikuDist=sg(H["depth_dist_you"][0] * PP, 2), qYpHaikuDistMain=sg(H["depth_dist_main"][0] * PP, 2), qYpHaikuChg=sg(H["depth_change"][0] * PP, 2),
                     qYpHaikuChgLo=sg(H["depth_change"][1] * PP, 2), qYpHaikuChgHi=sg(H["depth_change"][2] * PP, 2),
                     qYpSonnetDist=sg(So["depth_dist_you"][0] * PP, 2), qYpSonnetDistMain=sg(So["depth_dist_main"][0] * PP, 2), qYpSonnetChg=sg(So["depth_change"][0] * PP, 2),
                     qYpSonnetChgLo=sg(So["depth_change"][1] * PP, 2), qYpSonnetChgHi=sg(So["depth_change"][2] * PP, 2),
                     qYpHaikuShare=f"{100 * H['share_removed']:.0f}\\%", qYpSonnetShare=f"{100 * So['share_removed']:.0f}\\%",
                     qYpHaikuTPR=f"{H['tpr_you'][0]:.2f}", qYpHaikuTPRMain=f"{H['tpr_main'][0]:.2f}")
print("H80e", res["H80e"]["pass_"], "H80f", res["H80f"]["pass_"]); json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT)
