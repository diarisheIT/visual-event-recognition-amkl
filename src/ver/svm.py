"""Precomputed-kernel SVMs and the equal-weight average classifiers used by SVM_T / SVM_AT."""

import numpy as np
from sklearn.svm import SVC

C = 1.0


def svm_scores(K, y, train_idx, C=C):
    """Train one SVM on train_idx with kernel K (N, N); return decision values on all N videos."""
    clf = SVC(kernel="precomputed", C=C).fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
    return clf.decision_function(K[:, train_idx])


def average_classifier(kernels, y, train_idx):
    """Equal-weight average of one SVM per kernel (the paper's prelearned average classifier)."""
    return np.mean([svm_scores(K, y, train_idx) for K in kernels], axis=0)
