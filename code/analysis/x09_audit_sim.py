"""Addendum 57 (N1, development) and Addendum 58 (confirmation): simulated audits of conversation laws and placebo statistics.

For one population (results/x09_pops/<pop>.parquet) and its labelers, runs `reps` audits per (labeler, design, budget) in the
item regime (depth law, f96 size law, article size law when available, placebo depth and size slopes) and the edge regime
(gold and placebo reply risk ratios), and records per cell: coverage of the full-gold value, mean width, and the rate at which
the interval excludes the labeler's own value of the law ("distortion detected").
usage: python3 x09_audit_sim.py POP OUT_JSON --labelers a,b --reps 300 --seed 5700 [--item-budgets ..] [--edge-budgets ..]
       [--item-designs ..] [--edge-designs ..] [--procs 1]"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os, json, time, argparse, zlib
import numpy as np, pandas as pd
sys.path.insert(0, (_ROOT + "/code"))
from threadsafe2 import laws as L, edges as E, pps as PPS, gated_edges as G, edges_multi as EM, baselines as BL

W = _ROOT
ap = argparse.ArgumentParser(); ap.add_argument("pop"); ap.add_argument("out")
ap.add_argument("--labelers", required=True); ap.add_argument("--reps", type=int, default=300); ap.add_argument("--seed", type=int, default=5700)
ap.add_argument("--item-budgets", default="250,500,1000,2000,4000,8000"); ap.add_argument("--edge-budgets", default="500,1000,2000,4000,8000,16000")
ap.add_argument("--item-designs", default="srs,ppi,dstrat,flagprop,upgrade,upgrade_lw"); ap.add_argument("--edge-designs", default="srs,ppi,flagprop,upgrade,upgrade_lw")
ap.add_argument("--procs", type=int, default=1); ap.add_argument("--regimes", default="item,edge")
ap.add_argument("--edge-fill", action="store_true", help="Addendum 60: stratified edge designs spend the whole budget")
ap.add_argument("--pops-dir", default="x09_pops", help="x10_pops adds the labelers' scores (Addendum 61)")
ap.add_argument("--aux", default="", help="Addendum 64: auxiliary instruments (score columns) for the pps_multi working model")
ap.add_argument("--score-onesided", action="store_true", help="Addendum 60: score one-sided (lo, inf) intervals by lo <= truth")
ap.add_argument("--extra", default="", help="Addendum 76, B76d: unplanned targets with weight 0 in every allocation (within, prev)")
a = ap.parse_args()

P = pd.read_parquet(f"{W}/results/{a.pops_dir}/{a.pop}.parquet")
z = P.z.values.astype(float); depth = P.depth.values; k = P.k.values.astype(float); N = len(P)
xd = np.minimum(depth, 10).astype(float); big = k >= 2; logk = np.where(big, np.log2(np.maximum(k, 1)), 0.0)


def size_phi(unit):
    """f96 weights: theta = sum_i z_i (x_u - xbar)/(k_u SS) over units with >= 2 items."""
    u = pd.factorize(unit)[0]; ku = np.bincount(u).astype(float); xu = np.log2(np.maximum(ku, 1)); ok = ku >= 2
    xb = xu[ok].mean(); ss = ((xu[ok] - xb) ** 2).sum()
    return np.where(ok[u], (xu[u] - xb) / (ku[u] * ss), 0.0)


def metrics_for(y):
    ms = [L.Linear("depth_law", (xd - xd.mean()) / ((xd - xd.mean()) ** 2).sum()), L.Linear("size_law", size_phi(P.unit.values))]
    if "article" in P:
        ms.append(L.Linear("article_size_law", size_phi(P.article.values)))
    ms += [L.PlaceboSlope("placebo_depth", xd, y, np.ones(N)), L.PlaceboSlope("placebo_size", logk, y, np.where(big, 1.0 / np.maximum(k, 1), 0.0))]
    ex = [e for e in a.extra.split(",") if e]
    if "within" in ex:                                    # Addendum 76, B76d: the within-thread depth law (thread fixed effects)
        uu = pd.factorize(P.unit.values)[0]; xt = xd - (np.bincount(uu, xd) / np.bincount(uu))[uu]
        ms.append(L.Linear("depth_within", xt / (xt ** 2).sum()))
    if "prev" in ex:                                      # prevalence, the share of toxic comments
        ms.append(L.Linear("prevalence", np.full(N, 1.0 / N)))
    return ms


LAWS = ("depth_law", "size_law", "article_size_law")
EXTRA = ("depth_within", "prevalence")                   # Addendum 76: weight 0 in every allocation
small_art = None
if "article" in P:
    ka = P.groupby("article").article.transform("size").values; small_art = (ka <= 4).astype(int)
dbk = L.depth_bucket(depth)
cr = np.flatnonzero(P.parent_row.values >= 0); pr = P.parent_row.values[cr]
SEQ_ITEM = {"pps_seq3": (3, 0.10, 0.10), "pps_seq4": (4, 0.10, 0.10), "pps_seq3_d05": (3, 0.10, 0.05)}     # (folds, pilot share, delta)
SEQ_ITEM_X = {"pps_seq3_b": dict(K=3, delta=0.10, bayes=True), "pps_seq3_a": dict(K=3, delta="auto"),            # round A5
              "pps_seq3_ba": dict(K=3, delta="auto", bayes=True), "pps_seq4_ba": dict(K=4, delta="auto", bayes=True),
              "pps_seq3_or": dict(K=3, delta=0.10), "pps_seq3_ba_or": dict(K=3, delta="auto"),                  # diagnostics: oracle model
              "pps_seq3_bc10": dict(K=3, delta=0.10, bayes=True, floor=True), "pps_seq3_bc25": dict(K=3, delta=0.25, bayes=True, floor=True),   # round A6
              "pps_seq3_c25": dict(K=3, delta=0.25, floor=True), "pps_seq3_b_or": dict(K=3, delta=0.10), "pps_seq3_bc25_or": dict(K=3, delta=0.25, floor=True),
              "pps_seq3_bc05": dict(K=3, delta=0.05, bayes=True, floor=True), "pps_seq3_bc10_p05": dict(K=3, delta=0.10, bayes=True, floor=True, rho=0.05),   # round A7
              "pps_seq4_bc10": dict(K=4, delta=0.10, bayes=True, floor=True), "pps_seq3_bc10_or": dict(K=3, delta=0.10, floor=True),
              # Addendum 74, B3: ablations of the frozen pps_seq4_ba
              "pps_seq4_ba_noaux": dict(K=4, delta="auto", bayes=True, noaux=True), "pps_seq1_ba": dict(K=1, delta="auto", bayes=True),
              "pps_seq4_a": dict(K=4, delta="auto"), "pps_seq4_ba_unif": dict(K=4, delta=1.0, bayes=True)}
CELLS_I = None


def item_cells(y):
    """Round A6: the defensive stratified share's cells, flag x depth bucket x thread-size bucket (as the I2 information set)."""
    zb = np.searchsorted([2, 3, 4, 6], logk, side="right")
    return ((y.astype(int) * 5 + dbk) * 5 + zb).astype(int)


def oracle_edge(lab_name):
    """The I4m oracle edge law (scripts/x10_edge_ceiling.py --save-law), in the order of cluster_pop's edges."""
    key = ("ORACLE_E", lab_name)
    if key not in _CACHE:
        Z = np.load(f"{W}/results/x10_oracle_edge/{a.pop}__{lab_name.replace('@', '_at_')}.npz")
        cp_ = cluster_pop(lab_name, np.nan_to_num(P[f"y:{lab_name}"].values.astype(float)))
        # the ceiling script indexes edges by child row order (cr = rows with a parent, ascending); ClusterPop sorts by parent
        child_rows = np.flatnonzero(P.parent_row.values >= 0); pos = np.empty(N, np.int64); pos[child_rows] = np.arange(len(child_rows))
        ix = pos[cp_.child]; _CACHE[key] = (Z["a"][ix], Z["r1"][ix], Z["r0"][ix])
    return _CACHE[key]


def oracle_p(lab_name):
    """The I4 oracle working model (scripts/x10_oracle_p.py), for the learning-efficiency diagnostic only."""
    key = ("ORACLE", lab_name)
    if key not in _CACHE:
        _CACHE[key] = np.load(f"{W}/results/x10_oracle_p/{a.pop}__{lab_name.replace('@', '_at_')}.npy").astype(float)
    return _CACHE[key]
for _d, _kw in SEQ_ITEM_X.items():
    SEQ_ITEM[_d] = (_kw["K"], _kw.get("rho", 0.10), _kw["delta"])
SEQ_EDGE = {"edge_seq3": (3, 0.10, 0.10), "edge_seq4": (4, 0.10, 0.10), "edge_seq3_d05": (3, 0.10, 0.05)}
EDGE_GEN = {"edge_m1": dict(K=1, pilot_by="cell", post=True), "edge_m3": dict(K=3, pilot_by="cell", post=True),       # round E1
            "edge_m4": dict(K=4, pilot_by="cell", post=True), "edge_m3_np": dict(K=3, pilot_by="cell", post=False),
            "edge_m3_b": dict(K=3, pilot_by="cell", post=True, bayes=True), "edge_m3_a": dict(K=3, pilot_by="cell", post=True, delta="auto"),   # round E2
            "edge_m3_ba": dict(K=3, pilot_by="cell", post=True, bayes=True, delta="auto"), "edge_m4_ba": dict(K=4, pilot_by="cell", post=True, bayes=True, delta="auto"),
            "edge_m3_c25": dict(K=3, pilot_by="cell", post=True, delta=0.25, floor="cells"),                                      # round E3
            "edge_m3_ca": dict(K=3, pilot_by="cell", post=True, delta="auto", floor="cells"),
            "edge_m3_c10": dict(K=3, pilot_by="cell", post=True, delta=0.10, floor="cells"),
            "edge_m3_c25_or": dict(K=3, pilot_by="cell", post=True, delta=0.25, floor="cells"),                                  # diagnostic: oracle law
            "edge_m3_c15": dict(K=3, pilot_by="cell", post=True, delta=0.15, floor="cells"),                                     # round E4
            "edge_m4_c25": dict(K=4, pilot_by="cell", post=True, delta=0.25, floor="cells"),
            "edge_m3_c25_p05": dict(K=3, pilot_by="cell", post=True, delta=0.25, floor="cells", pilot_share=0.05),
            # round E5 (Addendum 73): label sharing (sh) and power tuning (pt)
            "edge_m3_c15_sh": dict(K=3, pilot_by="cell", post=True, delta=0.15, floor="cells", cost=True),
            "edge_m3_c15_pt": dict(K=3, pilot_by="cell", post=True, delta=0.15, floor="cells", tune=True),
            "edge_m3_c15_sh_pt": dict(K=3, pilot_by="cell", post=True, delta=0.15, floor="cells", cost=True, tune=True),
            "edge_m4_sh_pt": dict(K=4, pilot_by="cell", post=True, cost=True, tune=True),
            "edge_m4_c15_sh_pt": dict(K=4, pilot_by="cell", post=True, delta=0.15, floor="cells", cost=True, tune=True),
            "edge_m4_c15_sh": dict(K=4, pilot_by="cell", post=True, delta=0.15, floor="cells", cost=True),
            "edge_m5_c15_sh": dict(K=5, pilot_by="cell", post=True, delta=0.15, floor="cells", cost=True),
            "edge_m4_sh": dict(K=4, pilot_by="cell", post=True, cost=True),
            "edge_m4_c15_sh_p05": dict(K=4, pilot_by="cell", post=True, delta=0.15, floor="cells", cost=True, pilot_share=0.05),
            # Addendum 74, B3: ablations of the frozen edge_m3_c15
            "edge_m3_c15_noaux": dict(K=3, pilot_by="cell", post=True, delta=0.15, floor="cells", noaux=True),
            "edge_m1_c15": dict(K=1, pilot_by="cell", post=True, delta=0.15, floor="cells"),
            "edge_m3_c15_unif": dict(K=3, pilot_by="cell", post=True, delta=1.0)}
_CACHE = {}


REPLIES = np.bincount(P.parent_row.values[P.parent_row.values >= 0], minlength=N)       # each comment's number of replies


def item_X(lab_name, lev, multi, replies=False):
    """Working-model features (pps.design_matrix): score levels, depth, size, article size, (multi) auxiliary instruments'
    scores other than the audited labeler's own family, and (round A4, replies) the comment's number of replies."""
    key = ("X", lab_name, multi, replies)
    if key not in _CACHE:
        aux = [P[f"s:{e}"].values for e in a.aux.split(",") if e and multi and f"s:{e}" in P and e.split("@")[0] != lab_name.split("@")[0]]
        _CACHE[key] = PPS.design_matrix(lev, depth, logk, P.groupby("article").article.transform("size").values if "article" in P else None, aux=aux,
                                        replies=REPLIES if replies else None)
    return _CACHE[key]


def calag_cells(lab_name, d, ms):
    """Addendum 74, B1: calibration cells of calibrate-and-aggregate: the score level ('calag'), or level x depth bucket x
    thread-size bucket ('calag_s', falling back to the level when a cell is empty)."""
    key = ("CALAG", lab_name, d)
    if key not in _CACHE:
        lev = PPS.score_levels(P[f"s:{lab_name}"].values)
        if d == "calag":
            fine, coarse = lev, lev
        else:
            zb = np.searchsorted([2, 3, 4, 6], logk, side="right"); fine = (lev * 5 + dbk) * 5 + zb; coarse = lev
        _CACHE[key] = BL.CalagCells(ms, fine, coarse)
    return _CACHE[key]


def cluster_pop(lab_name, y):
    key = ("CP", lab_name)
    if key not in _CACHE:
        _CACHE[key] = G.ClusterPop(P.parent_row.values, len(P), y, P[f"s:{lab_name}"].values)
    return _CACHE[key]


def summarize(rows, truth, lab, log=False):
    """rows: list of (estimate, lo, hi). Coverage (undefined intervals count as covering), mean width (log scale for risk
    ratios, over intervals with lo > 0), distortion-detection rate, undefined rate, bias and RMSE of the point estimate."""
    R = np.array(rows, float); est, lo, hi = R.T; ok = np.isfinite(lo) & np.isfinite(hi)
    tol = 1e-9 * abs(truth) + 1e-15                       # Addendum 74: a zero-width interval at the truth (up to rounding) covers
    if a.score_onesided:                                  # (lo, inf) covers iff lo <= truth; only a missing lo counts as covering
        cov = float(np.mean(np.isnan(lo) | ((lo - tol <= truth) & (truth <= hi + tol))))
    else:
        cov = float(np.mean(~ok | ((lo - tol <= truth) & (truth <= hi + tol))))
    if log:
        wok = ok & (lo > 0); width = float(np.mean(np.log(hi[wok] / lo[wok]))) if wok.any() else None
    else:
        width = float(np.mean((hi - lo)[ok])) if ok.any() else None
    fe = np.isfinite(est)
    out = dict(coverage=cov, width=width, undefined=float(np.mean(~ok)), bias=float(np.mean(est[fe]) - truth) if fe.any() else None,
               rmse=float(np.sqrt(np.mean((est[fe] - truth) ** 2))) if fe.any() else None)
    if lab is not None:
        okd = (np.isfinite(lo) & ~np.isnan(hi)) if a.score_onesided else ok
        out["detect"] = float(np.mean(okd & ((lab < lo) | (lab > hi))))
    return out


def run_labeler(lab_name):
    key = [a.seed, zlib.crc32(a.pop.encode()), zlib.crc32(lab_name.encode())]          # deterministic seeds per cell and audit
    y = np.nan_to_num(P[f"y:{lab_name}"].values.astype(float))
    res = {"labeler": lab_name, "item": {}, "edge": {}}
    t0 = time.time()
    if "item" in a.regimes:
        ms = metrics_for(y); truth = {m.name: L.value(m, z) for m in ms}; labv = {m.name: (L.value(m, y) if (m.name in LAWS or m.name in EXTRA) else None) for m in ms}
        res["truth_item"] = truth; res["labeler_item"] = labv
        Y_tot = [(y[:, None] * m.B).sum(0) for m in ms]
        dmem = [np.flatnonzero(dbk == h) for h in range(5)]; Nd = np.array([len(m) for m in dmem], float)
        Yd = [[(y[mm, None] * m.B[mm]).sum(0) for mm in dmem] for m in ms]
        tiers = {}
        for B in [int(b) for b in a.item_budgets.split(",")]:
            tier = 0 if B < 500 else (1 if B < 2000 else 2)
            if tier not in tiers:
                tiers[tier] = L.Strata(L.upgrade_strata(y, depth, logk, B, small_art), ms, y)
                tiers[tier].flag = np.array([int(y[mm[0]]) for mm in tiers[tier].members])   # pooling groups for the allocation
            st = tiers[tier]; cell = {d: {m.name: [] for m in ms} for d in a.item_designs.split(",")}; labs_used = {d: [] for d in cell}
            for r in range(a.reps):
                for di, d in enumerate(cell):
                    rng = np.random.default_rng(key + [1, B, r, zlib.crc32(d.encode())])
                    if d in ("srs", "ppi"):
                        s = rng.choice(N, B, replace=False)
                        for j, m in enumerate(ms):
                            if d == "srs":
                                th, v = L.estimate_strat(m, [s], [np.arange(N)], np.array([float(N)]), z)
                            else:
                                th, v = L.estimate_ppi(m, s, N, z, y, Y_tot[j])
                            cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(B)
                    elif d == "upgrade_lw_score":                         # Addendum 63: score-level strata, pooled by level
                        lev = PPS.score_levels(P[f"s:{lab_name}"].values); sk = ("score", B < 1000, B < 4000)
                        if sk not in tiers:
                            tiers[sk] = L.Strata(L.score_strata(lev, depth, logk, B, small_art), ms, y); tiers[sk].flag = np.array([int(lev[mm[0]]) for mm in tiers[sk].members])
                        stS = tiers[sk]; mult = [0.0 if m.name in EXTRA else (1.0 if m.name in LAWS else 0.25) for m in ms]
                        o = L.run_strat(ms, stS, B, rng, z, mode="upgrade", mult=mult, pool=stS.flag)
                        for m in ms:
                            th, v = o[m.name]; cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
                    elif d in ("ai", "dsl", "calag", "calag_s"):                   # Addendum 74, B1: comparators
                        lev = PPS.score_levels(P[f"s:{lab_name}"].values); mult = [0.0 if m.name in EXTRA else (1.0 if m.name in LAWS else 0.25) for m in ms]
                        if d == "ai":
                            o = BL.run_ai(ms, P[f"s:{lab_name}"].values, B, rng, z, mult=mult)
                        elif d == "dsl":
                            o = BL.run_dsl(ms, item_X(lab_name, lev, True, replies=True), B, rng, z)
                        else:
                            o = BL.run_calag(ms, calag_cells(lab_name, d, ms), B, rng, z)
                        for m in ms:
                            th, v = o[m.name]
                            cell[d][m.name].append((th, *o["_ci"][m.name]) if "_ci" in o else (th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
                    elif d.startswith("pps_seq"):                                  # round A4: sequential folds, refitted working model
                        K, rho, dl = SEQ_ITEM[d]; lev = PPS.score_levels(P[f"s:{lab_name}"].values); mult = [0.0 if m.name in EXTRA else (1.0 if m.name in LAWS else 0.25) for m in ms]
                        noaux = SEQ_ITEM_X.get(d, {}).get("noaux", False)
                        o = PPS.run_seq(ms, lev, B, rng, z, item_X(lab_name, lev, not noaux, replies=True), mult=mult, pilot_share=rho, K=K, pilot_cells=lev * 5 + dbk, delta=dl,
                                        bayes=SEQ_ITEM_X.get(d, {}).get("bayes", False), p_fixed=oracle_p(lab_name) if d.endswith("_or") else None,
                                        floor_cells=item_cells(y) if SEQ_ITEM_X.get(d, {}).get("floor") else None)
                        for m in ms:
                            th, v = o[m.name]; cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
                    elif d == "pps_multi_cr":                                       # round A4: capped normalization, guard, reply features
                        lev = PPS.score_levels(P[f"s:{lab_name}"].values); mult = [0.0 if m.name in EXTRA else (1.0 if m.name in LAWS else 0.25) for m in ms]
                        o = PPS.run(ms, lev, B, rng, z, mult=mult, X=item_X(lab_name, lev, True, replies=True), pilot_cells=lev * 5 + dbk, norm="capped", guard=4.0)
                        for m in ms:
                            th, v = o[m.name]; cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
                    elif d in ("pps", "pps_flag", "pps_logit", "pps_logit_d", "pps_multi", "pps_multi_c"):   # Addenda 62-64: two-phase PPS by influence
                        lev = PPS.score_levels(P[f"s:{lab_name}"].values) if d != "pps_flag" else y.astype(int)
                        mult = [0.0 if m.name in EXTRA else (1.0 if m.name in LAWS else 0.25) for m in ms]
                        multi = d in ("pps_multi", "pps_multi_c"); xk = ("_XM_" if multi else "_X_") + lab_name
                        if (d.startswith("pps_logit") or multi) and xk not in globals():
                            aux = [P[f"s:{e}"].values for e in a.aux.split(",") if e and multi and f"s:{e}" in P and e.split("@")[0] != lab_name.split("@")[0]]
                            globals()[xk] = PPS.design_matrix(lev, depth, logk, P.groupby("article").article.transform("size").values if "article" in P else None, aux=aux)
                        o = PPS.run(ms, lev, B, rng, z, mult=mult, X=globals()[xk] if (d.startswith("pps_logit") or multi) else None,
                                    pilot_cells=(lev * 5 + dbk) if d in ("pps_logit_d", "pps_multi", "pps_multi_c") else None,
                                    norm="capped" if d == "pps_multi_c" else "uncapped", guard=4.0 if d == "pps_multi_c" else None)
                        for m in ms:
                            th, v = o[m.name]; cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
                    elif d == "dstrat":
                        o = L.run_dstrat(ms, dmem, Nd, B, rng, z, y, Yd)
                        for m in ms:
                            th, v = o[m.name]; cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
                    else:
                        mult = [0.0 if m.name in EXTRA else (1.0 if m.name in LAWS else 0.25) for m in ms] if d == "upgrade_lw" else None
                        o = L.run_strat(ms, st, B, rng, z, mode="flagprop" if d == "flagprop" else "upgrade", mult=mult, pool=st.flag)
                        for m in ms:
                            th, v = o[m.name]; cell[d][m.name].append((th, *L.wald(th, v)))
                        labs_used[d].append(o["_labels"])
            res["item"][B] = {d: {mn: summarize(rows, truth[mn], labv[mn]) for mn, rows in cell[d].items()} | {"_labels": float(np.mean(labs_used[d]))} for d in cell}
            print(a.pop, lab_name, "item", B, f"{time.time() - t0:.0f}s", json.dumps({d: {mn: (round(v["coverage"], 3), None if v["width"] is None else round(v["width"], 5), v.get("detect")) for mn, v in res["item"][B][d].items() if mn != "_labels"} for d in cell}), flush=True)
    if "edge" in a.regimes:
        pop = E.EdgePop(pr, cr, y); tg, tq = pop.truth(z); lab_rr = pop.labeler_log_rr()
        res["truth_edge"] = {"log_rr": tg, "log_placebo_rr": tq, "rr": float(np.exp(tg)), "placebo_rr": float(np.exp(tq))}; res["labeler_edge"] = {"rr": float(np.exp(lab_rr))}
        for B in [int(b) for b in a.edge_budgets.split(",")]:
            cell = {d: {mn: [] for mn in E.METRICS} for d in a.edge_designs.split(",")}; used = {d: [] for d in cell}
            for r in range(a.reps):
                for d in cell:
                    rng = np.random.default_rng(key + [2, B, r, zlib.crc32(d.encode())])
                    if d == "edge_pps":                                            # round B2: level working model
                        o = G.run_edge_pps(cluster_pop(lab_name, y), z, B, rng)
                    elif d == "edge_pps_multi":                                    # round E1: logistic working model with auxiliaries
                        o = EM.run_pps(cluster_pop(lab_name, y), z, B, rng, item_X(lab_name, PPS.score_levels(P[f"s:{lab_name}"].values), True), pilot_share=0.10)
                    elif d == "edge_ai":                                           # Addendum 74, B1: comparators
                        o = BL.run_edge_ai(cluster_pop(lab_name, y), z, B, rng)
                    elif d == "edge_dsl":
                        o = BL.run_edge_dsl(cluster_pop(lab_name, y), z, B, rng, item_X(lab_name, PPS.score_levels(P[f"s:{lab_name}"].values), True, replies=True))
                    elif d == "edge_calag":
                        o = BL.run_edge_calag(cluster_pop(lab_name, y), z, B, rng)
                    elif d in EDGE_GEN:                                            # round E1: edges_multi.run_edge
                        kw = dict(EDGE_GEN[d]); noaux = kw.pop("noaux", False)
                        o = EM.run_edge(cluster_pop(lab_name, y), z, B, rng, item_X(lab_name, PPS.score_levels(P[f"s:{lab_name}"].values), not noaux, replies=True), **kw,
                                        law_fixed=oracle_edge(lab_name) if d.endswith("_or") else None)
                    elif d.startswith("edge_seq"):                                 # round E1: sequential folds
                        K, rho, dl = SEQ_EDGE[d]
                        o = EM.run_pps_seq(cluster_pop(lab_name, y), z, B, rng, item_X(lab_name, PPS.score_levels(P[f"s:{lab_name}"].values), True), K=K, pilot_share=rho, delta=dl)
                    else:
                        o = E.run(pop, z, B, rng, design="upgrade" if d == "upgrade_lw" else d, mult=[1.0, 0.25] if d == "upgrade_lw" else None, fill=a.edge_fill)
                    for jm, mn in enumerate(E.METRICS):
                        th, v = o[mn]
                        if "_ci" in o:                                             # Addendum 74: bootstrap interval (calibrate and aggregate)
                            lo, hi = o["_ci"][mn]
                        else:
                            lo, hi = E.rr_interval(th, v, o["_props"][2 * jm: 2 * jm + 2]) if o.get("_props") is not None else E.log_wald(th, v)
                        cell[d][mn].append((np.exp(th) if np.isfinite(th) else np.nan, lo, hi if (np.isfinite(hi) or a.score_onesided) else np.nan))
                    used[d].append(o["_labels"])
            tr = {"log_rr": np.exp(tg), "log_placebo_rr": np.exp(tq)}; lb = {"log_rr": np.exp(lab_rr), "log_placebo_rr": None}
            res["edge"][B] = {d: {mn: summarize(rows, tr[mn], lb[mn], log=True) for mn, rows in cell[d].items()} | {"_labels": float(np.mean(used[d]))} for d in cell}
            print(a.pop, lab_name, "edge", B, f"{time.time() - t0:.0f}s", json.dumps({d: {mn: (round(v["coverage"], 3), None if v["width"] is None else round(v["width"], 3), v.get("detect")) for mn, v in res["edge"][B][d].items() if mn != "_labels"} for d in cell}), flush=True)
    return res


if __name__ == "__main__":
    labs = a.labelers.split(",")
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as pool:
            out = pool.map(run_labeler, labs)
    else:
        out = [run_labeler(l) for l in labs]
    json.dump({"pop": a.pop, "args": vars(a), "results": out}, open(a.out, "w"), indent=1)
    print("saved", a.out)
