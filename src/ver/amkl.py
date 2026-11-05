"""A-MKL (Section 4, Algorithm 1): multiple base kernels plus pre-learned classifiers, with the domain mismatch minimized."""

from typing import NamedTuple

import numpy as np
from sklearn.svm import SVC

from ver.svm import C

THETA, LAM, EPS, TAU, T_MAX = 1e-5, 20.0, 1e-5, 1e-5, 15
MAX_HALVINGS = 10  # line search gives up after eta = 2**-10


class AMKLResult(NamedTuple):
    scores: np.ndarray  # (N,) decision values on all videos
    objective: list  # G(d) per iteration, starting at d = 1/M
    d: np.ndarray  # (M,) base kernel weights on the simplex
    beta: np.ndarray  # (P,) coefficients on the pre-learned classifiers


def project_simplex(x):
    """Euclidean projection onto {d >= 0, sum(d) = 1} (Duchi et al., 2008)."""
    u = np.sort(x)[::-1]
    css = np.cumsum(u) - 1
    rho = np.flatnonzero(u - css / np.arange(1, len(x) + 1) > 0)[-1]
    return np.maximum(x - css[rho] / (rho + 1), 0)


def amkl_solve(kernels, F, y, train_idx, target):
    """Algorithm 1. kernels: {(feature, level): [K, ...]} over all N videos; F: (N, P) pre-learned classifier outputs.

    The mismatch vector h uses all videos, labeled or not (the paper's transductive setting).
    """
    Ks = [K for ks in kernels.values() for K in ks]
    s = np.where(target, -1 / target.sum(), 1 / (~target).sum())
    h = np.array([s @ K @ s for K in Ks])  # h_m = tr(K_m S) with S = s s'
    FF = F @ F.T / LAM
    tt = np.ix_(train_idx, train_idx)

    def solve(d):
        """Inner SVM with kernel sum_m d_m K_m + F F' / lam; returns (alpha * y on train_idx, bias, G(d), combined kernel)."""
        K = sum(dm * Km for dm, Km in zip(d, Ks)) + FF
        svm = SVC(kernel="precomputed", C=C).fit(K[tt], y[train_idx])
        v = np.zeros(len(train_idx))
        v[svm.support_] = svm.dual_coef_[0]
        J = np.abs(v).sum() - 0.5 * v @ K[tt] @ v
        return v, svm.intercept_[0], 0.5 * (h @ d) ** 2 + THETA * J, K

    d = np.full(len(Ks), 1 / len(Ks))
    v, b, G, K = solve(d)
    objective = [G]
    while len(objective) < T_MAX:
        sv = train_idx[v != 0]
        q = np.array([0.5 * v[v != 0] @ Km[np.ix_(sv, sv)] @ v[v != 0] for Km in Ks])
        d_new = project_simplex(THETA * np.linalg.solve(np.outer(h, h) + EPS * np.eye(len(Ks)), q))
        for eta in 0.5 ** np.arange(MAX_HALVINGS + 1):  # backtracking: accept the first step that lowers G
            d_try = (1 - eta) * d + eta * d_new
            v_try, b_try, G_try, K_try = solve(d_try)
            if G_try < G:
                break
        else:
            break  # no descent step left: converged
        d, v, b, K = d_try, v_try, b_try, K_try
        G, G_prev = G_try, G
        objective.append(G)
        if G_prev - G <= TAU:
            break

    scores = K[:, train_idx] @ v + b
    return AMKLResult(scores, objective, d, F[train_idx].T @ v / LAM)
