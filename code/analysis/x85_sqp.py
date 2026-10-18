"""Addendum 85, B85a: a sequential quadratic programming solver for the NPMLE of a mixing distribution on a fixed grid (the mixsqp
algorithm of Kim, Carbonetto and Stephens 2020, without its low-rank tricks): minimize f(x) = -sum_u w_u log((L x)_u) + sum_k x_k
over x >= 0 (at the optimum sum_k x_k = 1). Each iteration solves the quadratic model over x >= 0 exactly (non-negative least squares
on the Cholesky factor of the Hessian) and takes an Armijo backtracking step. Converged when max_k d_k - 1 <= tol, with
d_k = sum_u w_u L_uk / (L x)_u, the KKT condition of the NPMLE.
usage: from x85_sqp import npmle_sqp"""
import numpy as np
from scipy.optimize import nnls
from scipy.linalg import cholesky, solve_triangular


def npmle_sqp(L, w, x0=None, tol=1e-8, maxit=500, ridge=1e-12):
    S, K = L.shape; w = np.asarray(w, float); w = w / w.sum()
    x = np.full(K, 1.0 / K) if x0 is None else np.maximum(np.asarray(x0, float).ravel(), 1e-12 / K)
    x = x / x.sum()
    def f(x_):
        Lx = L @ x_
        if np.any(Lx <= 1e-150):                                      # keeps every pattern's likelihood away from underflow
            return np.inf
        return float(-np.sum(w * np.log(Lx)) + x_.sum())
    fx = f(x); it = 0; kkt = np.inf
    for it in range(1, maxit + 1):
        Lx = L @ x; d = L.T @ (w / Lx); kkt = float(d.max() - 1.0)
        if kkt <= tol:
            break
        g = 1.0 - d
        A = L * (np.sqrt(w) / Lx)[:, None]; H = A.T @ A
        H[np.diag_indices(K)] += ridge * max(np.trace(H) / K, 1e-300)
        R = cholesky(H, lower=False)                                  # H = R^T R
        b = g - H @ x                                                 # minimize 0.5 y'Hy + b'y over y >= 0
        c = -solve_triangular(R, b, trans="T", lower=False)           # = -R^{-T} b
        y, _ = nnls(R, c, maxiter=50 * K)
        dlt = y - x; gd = float(g @ dlt)
        if gd >= 0:                                                   # no descent direction: fall back to an EM step
            x = x * d; x /= x.sum(); fx = f(x); continue
        a = 1.0; ok = False
        while a >= 1e-10:
            xn = x + a * dlt; fn = f(xn)
            if np.isfinite(fn) and fn <= fx + 0.01 * a * gd:
                ok = True; break
            a *= 0.5
        if not ok:                                                    # no acceptable step: an EM step instead
            x = x * d; x /= x.sum(); fx = f(x); continue
        x, fx = xn, fn
    x = np.maximum(x, 0); x = x / x.sum()
    Lx = L @ x; kkt = float((L.T @ (w / Lx)).max() - 1.0)
    return x, dict(iterations=it, kkt=kkt, converged=bool(kkt <= tol))
