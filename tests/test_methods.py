import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from ver.kernels import base_kernels
from ver.methods import svm_at, svm_t


def two_domain_problem(seed=0, n_target=40, n_aux=200):
    """Two classes in 2-D; auxiliary videos shifted away from the target domain. Returns kernels, y, train, target."""
    rng = np.random.default_rng(seed)
    n = n_target + n_aux
    y = np.where(np.arange(n) % 2 == 0, 1, -1)
    X = rng.normal(size=(n, 2)) + 2.0 * y[:, None] * [1, 0]
    X[n_target:] += [0, 1.5]
    target = np.arange(n) < n_target
    train = np.concatenate([np.arange(6), np.arange(n_target, n)])  # 6 labeled target videos + all auxiliary
    D = cdist(X, X)
    kernels = {("toy", l): base_kernels(D, train, (-1, 0, 1)) for l in (0, 1)}  # identical levels: only exercises fusion
    return kernels, y, train, target


def test_svm_t_ignores_auxiliary_videos():
    kernels, y, train, target = two_domain_problem()
    flipped = y.copy()
    flipped[~target] *= -1
    np.testing.assert_allclose(svm_t(kernels, flipped, train, target), svm_t(kernels, y, train, target))


def test_svm_at_learns_from_auxiliary_videos():
    kernels, y, train, target = two_domain_problem()
    test = np.flatnonzero(target)[6:]
    flipped = y.copy()
    flipped[~target] *= -1  # 200 auxiliary labels outvote the 6 target ones
    assert average_precision_score(y[test] == 1, svm_at(kernels, y, train, target)[test]) > 0.9
    assert average_precision_score(y[test] == 1, svm_at(kernels, flipped, train, target)[test]) < 0.5
