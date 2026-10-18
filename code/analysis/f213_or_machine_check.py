"""Addendum 42, E4: machine check of Theorem 2 (the complete "at least one" frontier) and Corollary 2 (its hazard form).
A. Certificate, 2,000 random instances, 60-digit decimal arithmetic: Theorem 2's gates, lambda, R0, M and the certificate direction xi;
   the Bellman value over ALL adaptive policies (every partial assignment in {*,0,1}^k), V(0) = max_pi [E (E[xi|O])^2 - lambda E cost];
   checks V(0) <= M, <xi, Y - mu> = R0 and lambda B + M = R0 (so R*(B) >= R0^2 / I_B(xi) >= R0), each to 1e-30.
B. Achievability, 500 random instances in exact rational arithmetic, random nested gates: on every gold vector z, E[T | z] = Y(z);
   E cost = sum e_i Q_i and E (T - mu)^2 = sum d_i / Q_i exactly.
C. Hazard form, 500 random dependent laws on {0,1}^k in exact rational arithmetic, fixed order, random nested gates: unbiased on every z;
   cost and risk equal the hazard formulas exactly; and (floating point) the pooled gates are not beaten by 300 random feasible nested gates.
usage: python3 f213_or_machine_check.py OUT_JSON [--nA 2000 --nB 500 --nC 500]"""
import sys, json, itertools, random, time
from fractions import Fraction as F
from decimal import Decimal as D, getcontext
getcontext().prec = 60
args = sys.argv[1:]; OUT = args[0]
opt = lambda k, d: int(args[args.index(k) + 1]) if k in args else d
NA, NB, NC = opt("--nA", 2000), opt("--nB", 500), opt("--nC", 500)
rnd = random.Random(42 + 4242)

def frontier_quantities(p, c, one, sqrt):
    """Theorem 2 in the given (already c/p-sorted, or fixed) order: A_i, d_i, e_i, pooled blocks, and helpers."""
    k = len(p); A = [one]
    for x in p: A.append(A[-1] * (one - x))
    Afin = A[-1]; d = [Afin * Afin * (one / A[i + 1] - one / A[i]) for i in range(k)]; e = [A[i] * c[i] for i in range(k)]
    blocks = []                                   # pool adjacent violators so that block ratios D/E are nonincreasing
    for i in range(k):
        blocks.append([i, i, d[i], e[i]])
        while len(blocks) > 1 and blocks[-2][2] * blocks[-1][3] < blocks[-1][2] * blocks[-2][3]:
            b2 = blocks.pop(); b1 = blocks.pop(); blocks.append([b1[0], b2[1], b1[2] + b2[2], b1[3] + b2[3]])
    return A, d, e, blocks

def gates_for_budget(blocks, B, one, sqrt):
    """Q per block: Q = min(1, sqrt(rho / lambda)) with sum e_i Q_i = B; the capped blocks are the leading ones."""
    nb = len(blocks)
    for j in range(nb + 1):                       # j leading capped blocks
        Ecap = sum((b[3] for b in blocks[:j]), 0 * one); Sfree = sum((sqrt(b[2] * b[3]) for b in blocks[j:]), 0 * one)
        if j == nb or B - Ecap <= 0: continue
        lam = (Sfree / (B - Ecap)) ** 2
        rho = [b[2] / b[3] for b in blocks]
        if all(rho[t] >= lam for t in range(j)) and all(rho[t] < lam for t in range(j, nb)):
            return lam, [one if t < j else sqrt(rho[t] / lam) for t in range(nb)], j
    raise ValueError("no consistent cap set")

def part_A(n, control=False):
    worst = {"V_minus_M": D(-1), "pair": D(0), "dual": D(0)}; t0 = time.time(); viol = 0
    for it in range(n):
        k = rnd.randint(2, 6)
        p = [D(rnd.randint(20, 900)) / D(1000) for _ in range(k)]; c = [D(rnd.randint(200, 3000)) / D(1000) for _ in range(k)]
        order = sorted(range(k), key=lambda i: c[i] / p[i], reverse=control); p = [p[i] for i in order]; c = [c[i] for i in order]
        one = D(1); sq = lambda x: x.sqrt()
        A, d, e, blocks = frontier_quantities(p, c, one, sq); Cfull = sum(e)
        B = Cfull * D(rnd.randint(50, 980)) / D(1000)
        lam, QG, j = gates_for_budget(blocks, B, one, sq)
        Q = [None] * k
        for t, b in enumerate(blocks):
            for i in range(b[0], b[1] + 1): Q[i] = QG[t]
        R0 = sum(d[i] / Q[i] for i in range(k)); Afin = A[-1]; mu = one - Afin
        M = sum((d[i] - lam * e[i] for i in range(blocks[j - 1][1] + 1)), D(0)) if j > 0 else D(0)
        # certificate direction: one combined capped block [0, k0] (beta = A/A_k0) and each uncapped pooled block
        G = ([(0, blocks[j - 1][1], one)] if j > 0 else []) + [(b[0], b[1], QG[t]) for t, b in enumerate(blocks) if t >= j]
        Gs = [(l, u, Afin / (A[u + 1] * qg), A[u + 1] / A[l]) for (l, u, qg) in G]          # (l, u, beta_G, r_G)
        def Exi(s):                                # E[xi | partial assignment s] under the product law; s[i] in {None, 0, 1}
            tot = D(0)
            for (l, u, beta, r) in Gs:
                pz = one
                for i in range(l, u + 1): pz *= (D(0) if s[i] == 1 else (one if s[i] == 0 else one - p[i]))
                tot += beta * ((one - pz) - (one - r))
            return tot
        memo = {}
        def V(s):
            if s in memo: return memo[s]
            best = Exi(s) ** 2
            for i in range(k):
                if s[i] is None:
                    s1 = s[:i] + (1,) + s[i + 1:]; s0 = s[:i] + (0,) + s[i + 1:]
                    best = max(best, p[i] * V(s1) + (one - p[i]) * V(s0) - lam * c[i])
            memo[s] = best; return best
        v0 = V(tuple([None] * k))
        pair = D(0)                                # <xi, Y - mu> = sum_z p(z) xi(z) (Y(z) - mu)
        for z in itertools.product([0, 1], repeat=k):
            pz = one
            for i in range(k): pz *= p[i] if z[i] else one - p[i]
            pair += pz * Exi(tuple(z)) * ((one if any(z) else D(0)) - mu)
        viol += int(v0 - M > D("1e-30")); worst["V_minus_M"] = max(worst["V_minus_M"], v0 - M); worst["pair"] = max(worst["pair"], abs(pair - R0)); worst["dual"] = max(worst["dual"], abs(lam * B + M - R0))
    ok = worst["V_minus_M"] <= D("1e-30") and worst["pair"] <= D("1e-30") and worst["dual"] <= D("1e-30")
    if control: return dict(n=n, wrong_order_certificate_violations=viol, share=round(viol / n, 3), PASS=bool(viol > 0))
    return dict(n=n, max_V_minus_M=float(worst["V_minus_M"]), max_pairing_error=float(worst["pair"]), max_dual_error=float(worst["dual"]), seconds=round(time.time() - t0, 1), PASS=bool(ok))

def policy_moments(law, k, Q, A, Afin, mu, h, c):
    """Exact moments of the fixed-order, first-positive, nested-gate policy and output mu + sum_queried A/(A_i Q_i)(Z_i - h_i), under `law` (dict z -> prob)."""
    EC = F(0); risk = F(0); unb = True
    for z, pz in law.items():
        J = next((i for i in range(k) if z[i]), k)          # first positive (k = none)
        L = min(J + 1, k)                                     # most queries possible: up to and including the first positive
        cond_ET = F(0); cond_ET2 = F(0); cond_C = F(0)
        for m in range(0, L + 1):                             # m = number of queries made
            pm = Q[L - 1] if m == L else (Q[m - 1] if m > 0 else F(1)) - Q[m]     # passed gates 0..m-1, failed gate m (or reached the end)
            if pm == 0: continue
            T = mu + sum(Afin / (A[i + 1] * Q[i]) * ((1 if z[i] else 0) - h[i]) for i in range(m))
            cond_ET += pm * T; cond_ET2 += pm * (T - mu) ** 2; cond_C += pm * sum(c[i] for i in range(m))
        Y = 1 if any(z) else 0
        unb = unb and (cond_ET == Y)
        EC += pz * cond_C; risk += pz * cond_ET2
    return EC, risk, unb

def part_BC(n, dependent):
    bad = 0; worst_pool = -1.0; t0 = time.time()
    for it in range(n):
        k = rnd.randint(2, 5)
        if dependent:
            w = {z: F(rnd.randint(1, 60)) for z in itertools.product([0, 1], repeat=k)}; tot = sum(w.values()); law = {z: v / tot for z, v in w.items()}
            h = []
            for i in range(k):                                # hazards along the fixed order
                num = sum(v for z, v in law.items() if all(z[j] == 0 for j in range(i)) and z[i] == 1)
                den = sum(v for z, v in law.items() if all(z[j] == 0 for j in range(i)))
                h.append(num / den if den else F(1, 2))
        else:
            h = [F(rnd.randint(1, 19), 20) for _ in range(k)]
            law = {}
            for z in itertools.product([0, 1], repeat=k):
                pz = F(1)
                for i in range(k): pz *= h[i] if z[i] else 1 - h[i]
                law[z] = pz
        c = [F(rnd.randint(1, 30), 10) for _ in range(k)]
        A = [F(1)]
        for x in h: A.append(A[-1] * (1 - x))
        Afin = A[-1]; mu = 1 - Afin; d = [Afin * Afin * (1 / A[i + 1] - 1 / A[i]) for i in range(k)]; e = [A[i] * c[i] for i in range(k)]
        qs = sorted([F(rnd.randint(1, 20), 20) for _ in range(k)], reverse=True); Q = qs
        EC, risk, unb = policy_moments(law, k, Q, A, Afin, mu, h, c)
        ok = unb and EC == sum(e[i] * Q[i] for i in range(k)) and risk == sum(d[i] / Q[i] for i in range(k))
        bad += (not ok)
        if dependent:                                        # pooled gates vs random feasible nested gates (floating point)
            df, ef = [float(x) for x in d], [float(x) for x in e]
            _, _, _, blocks = frontier_quantities([float(x) for x in h], [float(x) for x in c], 1.0, lambda x: x ** 0.5)
            Bf = sum(ef) * rnd.uniform(0.05, 0.95)
            try:
                lam, QG, j = gates_for_budget(blocks, Bf, 1.0, lambda x: x ** 0.5)
            except ValueError:
                continue
            Qp = [None] * k
            for t, b in enumerate(blocks):
                for i in range(b[0], b[1] + 1): Qp[i] = QG[t]
            Rp = sum(df[i] / Qp[i] for i in range(k))
            for _ in range(300):
                qq = sorted([rnd.random() for _ in range(k)], reverse=True); s = sum(ef[i] * qq[i] for i in range(k)); qq = [min(1.0, x * Bf / s) for x in qq]
                qq = [max(min(qq[:i + 1]), 1e-9) for i in range(k)]                        # nested and capped at 1
                if sum(ef[i] * qq[i] for i in range(k)) <= Bf * (1 + 1e-12):
                    worst_pool = max(worst_pool, (Rp - sum(df[i] / qq[i] for i in range(k))) / Rp)
    res = dict(n=n, failures=int(bad), seconds=round(time.time() - t0, 1), PASS=bool(bad == 0 and (not dependent or worst_pool <= 1e-9)))
    if dependent: res["max_relative_gain_of_random_gates_over_pooled"] = worst_pool   # <= 0: no random feasible nested gates beat the pooled gates
    return res

out = {"A_certificate_all_adaptive_policies": part_A(NA), "A_negative_control_reversed_order": part_A(200, control=True), "B_achievability_exact": part_BC(NB, False), "C_hazard_form_exact": part_BC(NC, True)}
out["PASS"] = all(v["PASS"] for v in out.values() if isinstance(v, dict))
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out, indent=1))
