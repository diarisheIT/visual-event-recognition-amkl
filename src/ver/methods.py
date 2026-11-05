"""Classifiers behind one interface, so experiment scripts can treat them interchangeably.

Every method is called as method(kernels, y, train_idx, target) and returns decision values for all N videos:
  kernels:   {(feature, level): [K, ...]}, base kernels (N, N) over all videos
  y:         (N,) +1/-1 labels for one event (only y[train_idx] is used)
  train_idx: indices of the labeled training videos
  target:    (N,) bool, True for target-domain videos
"""

import numpy as np

from ver.amkl import amkl_solve
from ver.svm import average_classifier


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


def amkl(kernels, y, train_idx, target):
    """A-MKL with the pre-learned classifiers of this event (2 for one feature, 4 for both)."""
    return amkl_solve(kernels, prelearned_classifiers(kernels, y, train_idx), y, train_idx, target).scores


def mkl(kernels, y, train_idx, target):
    """MKL: DTSVM without the domain mismatch term."""
    return amkl_solve(kernels, np.empty((len(y), 0)), y, train_idx, target, mismatch=False).scores


def dtsvm(kernels, y, train_idx, target):
    """DTSVM: A-MKL without pre-learned classifiers."""
    return amkl_solve(kernels, np.empty((len(y), 0)), y, train_idx, target).scores


METHODS = {"SVM_T": svm_t, "SVM_AT": svm_at, "MKL": mkl, "DTSVM": dtsvm, "A-MKL": amkl}
