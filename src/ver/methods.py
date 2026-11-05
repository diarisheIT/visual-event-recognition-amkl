"""Classifiers behind one interface, so experiment scripts can treat them interchangeably.

Every method is called as method(kernels, y, train_idx, target) and returns decision values for all N videos:
  kernels:   {(feature, level): [K, ...]}, base kernels (N, N) over all videos
  y:         (N,) +1/-1 labels for one event (only y[train_idx] is used)
  train_idx: indices of the labeled training videos
  target:    (N,) bool, True for target-domain videos
"""

import numpy as np
from scipy.optimize import minimize

from ver.amkl import amkl_solve
from ver.kernels import default_gaussian, feature_replication
from ver.svm import C, average_classifier


def prelearned_classifiers(kernels, y, train_idx):
    """(N, P) outputs of the pre-learned classifiers, one per (feature, level)."""
    return np.column_stack([average_classifier(ks, y, train_idx) for ks in kernels.values()])


def _fused_average_classifiers(kernels, y, train_idx):
    """Equal-weight fusion of one average classifier per (feature, level)."""
    return prelearned_classifiers(kernels, y, train_idx).mean(axis=1)


def svm_at(kernels, y, train_idx, target):
    """SVM_AT: trained on auxiliary + labeled target videos."""
    return _fused_average_classifiers(kernels, y, train_idx)


def svm_t(kernels, y, train_idx, target):
    """SVM_T: trained on labeled target videos only."""
    return _fused_average_classifiers(kernels, y, train_idx[target[train_idx]])


def fr(kernels, y, train_idx, target):
    """FR: the SVM_AT recipe on feature-replication kernels."""
    replicated = {key: [feature_replication(K, target) for K in ks] for key, ks in kernels.items()}
    return svm_at(replicated, y, train_idx, target)


def asvm(kernels, y, train_idx, target, C=C):
    """A-SVM: per (feature, level), a pre-learned classifier f trained on auxiliary videos only plus a perturbation
    learned on the labeled target videos with the default Gaussian kernel; adapted classifiers fused with equal weights.

    Perturbation dual (no bias): max_a sum_i a_i (1 - y_i f_i) - 1/2 a' (yy' * K) a,  0 <= a <= C.
    """
    aux, tgt = train_idx[~target[train_idx]], train_idx[target[train_idx]]
    yt = y[tgt]
    adapted = []
    for ks, f in zip(kernels.values(), prelearned_classifiers(kernels, y, aux).T):
        K = default_gaussian(ks, train_idx)[:, tgt]
        Q = np.outer(yt, yt) * K[tgt]
        p = 1 - yt * f[tgt]
        a = minimize(lambda a: (0.5 * a @ Q @ a - p @ a, Q @ a - p), np.zeros(len(tgt)), jac=True,
                     method="L-BFGS-B", bounds=[(0, C)] * len(tgt)).x
        adapted.append(f + K @ (a * yt))
    return np.mean(adapted, axis=0)


def amkl(kernels, y, train_idx, target):
    """A-MKL with the pre-learned classifiers of this event (2 for one feature, 4 for both)."""
    return amkl_solve(kernels, prelearned_classifiers(kernels, y, train_idx), y, train_idx, target).scores


def mkl(kernels, y, train_idx, target):
    """MKL: DTSVM without the domain mismatch term."""
    return amkl_solve(kernels, np.empty((len(y), 0)), y, train_idx, target, mismatch=False).scores


def dtsvm(kernels, y, train_idx, target):
    """DTSVM: A-MKL without pre-learned classifiers."""
    return amkl_solve(kernels, np.empty((len(y), 0)), y, train_idx, target).scores


METHODS = {"SVM_T": svm_t, "SVM_AT": svm_at, "FR": fr, "A-SVM": asvm, "MKL": mkl, "DTSVM": dtsvm, "A-MKL": amkl}
