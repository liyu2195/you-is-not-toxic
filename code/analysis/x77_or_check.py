"""Addendum 77, B77f: Theorem 1's certificate against the exact Bellman value of ALL adaptive policies, for k = 7-10 (200 random
instances each, double precision, seed 7750). Same construction as part A of f213_or_machine_check.py (gates, lambda, R0, M and the
certificate direction xi; V(0) = max over policies of E (E[xi|O])^2 - lambda E cost over every partial assignment in {*,0,1}^k):
checks V(0) <= M, <xi, Y - mu> = R0 and lambda B + M = R0, to a relative tolerance of 1e-9.
usage: python3 x77_or_check.py OUT_JSON"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, json, itertools, random, time, math
OUT = sys.argv[1]; rnd = random.Random(7750)
src = open((_ROOT + "/code/analysis/f213_or_machine_check.py")).read()
ns = {}; exec(src[src.index("def frontier_quantities"):src.index("def part_A")], ns)
frontier_quantities, gates_for_budget = ns["frontier_quantities"], ns["gates_for_budget"]
res = {}
for k in (7, 8, 9, 10):
    worst = dict(V_minus_M=-1.0, pair=0.0, dual=0.0); t0 = time.time(); n = 200
    for it in range(n):
        p = [rnd.randint(20, 900) / 1000 for _ in range(k)]; c = [rnd.randint(200, 3000) / 1000 for _ in range(k)]
        order = sorted(range(k), key=lambda i: c[i] / p[i]); p = [p[i] for i in order]; c = [c[i] for i in order]
        A, d, e, blocks = frontier_quantities(p, c, 1.0, math.sqrt); B = sum(e) * rnd.randint(50, 980) / 1000
        lam, QG, j = gates_for_budget(blocks, B, 1.0, math.sqrt)
        Q = [None] * k
        for t, b in enumerate(blocks):
            for i in range(b[0], b[1] + 1): Q[i] = QG[t]
        R0 = sum(d[i] / Q[i] for i in range(k)); Afin = A[-1]; mu = 1 - Afin
        M = sum(d[i] - lam * e[i] for i in range(blocks[j - 1][1] + 1)) if j > 0 else 0.0
        G = ([(0, blocks[j - 1][1], 1.0)] if j > 0 else []) + [(b[0], b[1], QG[t]) for t, b in enumerate(blocks) if t >= j]
        Gs = [(l, u, Afin / (A[u + 1] * qg), A[u + 1] / A[l]) for (l, u, qg) in G]
        def Exi(s):
            tot = 0.0
            for (l, u, beta, r) in Gs:
                pz = 1.0
                for i in range(l, u + 1): pz *= (0.0 if s[i] == 1 else (1.0 if s[i] == 0 else 1 - p[i]))
                tot += beta * ((1 - pz) - (1 - r))
            return tot
        memo = {}
        def V(s):
            if s in memo: return memo[s]
            best = Exi(s) ** 2
            for i in range(k):
                if s[i] is None:
                    best = max(best, p[i] * V(s[:i] + (1,) + s[i + 1:]) + (1 - p[i]) * V(s[:i] + (0,) + s[i + 1:]) - lam * c[i])
            memo[s] = best; return best
        sys.setrecursionlimit(10000); v0 = V(tuple([None] * k))
        pair = 0.0
        for z in itertools.product([0, 1], repeat=k):
            pz = 1.0
            for i in range(k): pz *= p[i] if z[i] else 1 - p[i]
            pair += pz * Exi(tuple(z)) * ((1.0 if any(z) else 0.0) - mu)
        sc = max(abs(R0), 1e-300)
        worst["V_minus_M"] = max(worst["V_minus_M"], (v0 - M) / sc); worst["pair"] = max(worst["pair"], abs(pair - R0) / sc); worst["dual"] = max(worst["dual"], abs(lam * B + M - R0) / sc)
    res[k] = dict(n=n, max_rel_V_minus_M=worst["V_minus_M"], max_rel_pairing_error=worst["pair"], max_rel_dual_error=worst["dual"], seconds=round(time.time() - t0, 1),
                  PASS=bool(worst["V_minus_M"] <= 1e-9 and worst["pair"] <= 1e-9 and worst["dual"] <= 1e-9))
    print(k, res[k], flush=True)
res["PASS"] = all(v["PASS"] for v in res.values() if isinstance(v, dict)); json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT, res["PASS"])
