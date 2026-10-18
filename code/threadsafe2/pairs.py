"""THREADSAFE, Algorithm 3: the reply association, Delta = P(reply positive | parent positive) - P(reply positive | parent not).

Reply edges are sampled uniformly at random; each sampled edge costs two labels, or one when its parent's label is already
known. The LLM is not used: no LLM-guided allocation of pair labels was reliably better (ablation), because the right
allocation depends on gold rates the LLM does not identify. Delta gets Newcombe's score interval; the risk ratio MOVER-R.
"""
import numpy as np
from .intervals import wilson, newcombe, mover_rr, Z95


def run(parent, child, oracle, budget, rng=None, z=Z95):
    """parent, child: comment ids of every reply edge. Returns a dict with Delta, its interval, the RR and its interval."""
    rng = np.random.default_rng() if rng is None else rng
    known, used, zp, zc = {}, 0, [], []
    for e in rng.permutation(len(parent)):
        cost = 1 + (parent[e] not in known)
        if used + cost > budget:
            break
        used += cost
        if parent[e] not in known:
            known[parent[e]] = int(oracle(parent[e]))
        zp.append(known[parent[e]])
        zc.append(int(oracle(child[e])))
    a, b = np.array(zp), np.array(zc)
    n1, x1, n0, x0 = int(a.sum()), int((a * b).sum()), int((1 - a).sum()), int(((1 - a) * b).sum())
    p1 = x1 / n1 if n1 else 0.0
    p0 = x0 / n0 if n0 else 0.0
    clip = lambda lim: (max(0.0, lim[0]), min(1.0, lim[1]))
    (l1, u1), (l0, u0) = clip(wilson(p1, n1, z)), clip(wilson(p0, n0, z))
    lo, hi = newcombe(p1, n1, p0, n0, z, clip=True)
    rlo, rhi = mover_rr(p1, l1, u1, p0, l0, u0)
    return dict(delta=p1 - p0, lo=lo, hi=hi, rr=p1 / p0 if p0 > 0 else float("nan"), rr_lo=rlo, rr_hi=rhi,
                labels=used, edges=len(zp), toxic_parent_edges=n1)
