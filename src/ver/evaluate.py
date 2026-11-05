"""Non-interpolated average precision per event and MAP, as in Section 5."""

import numpy as np
from sklearn.metrics import average_precision_score


def average_precisions(scores, labels, test_idx):
    """AP per event on the test videos; scores and labels are (N, n_events)."""
    return np.array([
        average_precision_score(labels[test_idx, e] == 1, scores[test_idx, e])
        for e in range(labels.shape[1])
    ])


def mean_std(values):
    """Mean and std over rounds, in percent."""
    values = 100 * np.asarray(values)
    return values.mean(0), values.std(0)
