"""Classifiers behind one interface, so experiment scripts can treat them interchangeably.

Every method is called as method(kernels, y, train_idx, target) and returns decision values for all N videos:
  kernels:   {(feature, level): [K, ...]}, base kernels (N, N) over all videos
  y:         (N,) +1/-1 labels for one event (only y[train_idx] is used)
  train_idx: indices of the labeled training videos
  target:    (N,) bool, True for target-domain videos
"""

import numpy as np

from ver.svm import average_classifier


def _fused_average_classifiers(kernels, y, train_idx):
    """Equal-weight fusion of one average classifier per (feature, level)."""
    return np.mean([average_classifier(ks, y, train_idx) for ks in kernels.values()], axis=0)


def svm_at(kernels, y, train_idx, target):
    """SVM_AT: trained on auxiliary + labeled target videos."""
    return _fused_average_classifiers(kernels, y, train_idx)


def svm_t(kernels, y, train_idx, target):
    """SVM_T: trained on labeled target videos only."""
    return _fused_average_classifiers(kernels, y, train_idx[target[train_idx]])


METHODS = {"SVM_T": svm_t, "SVM_AT": svm_at}
