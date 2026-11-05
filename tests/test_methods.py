import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from ver.amkl import T_MAX, TAU, amkl_solve
from ver.kernels import base_kernels
from ver.methods import amkl, dtsvm, mkl, prelearned_classifiers, svm_at, svm_t


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


def test_amkl_objective_never_increases_and_converges():
    kernels, y, train, target = two_domain_problem()
    result = amkl_solve(kernels, prelearned_classifiers(kernels, y, train), y, train, target)
    G = np.array(result.objective)
    assert np.all(np.diff(G) <= 0)
    assert 1 < len(G) <= T_MAX
    assert len(G) < T_MAX or abs(G[-1] - G[-2]) <= TAU


def test_amkl_kernel_weights_stay_on_the_simplex():
    kernels, y, train, target = two_domain_problem()
    d = amkl_solve(kernels, prelearned_classifiers(kernels, y, train), y, train, target).d
    assert np.all(d >= 0)
    assert d.sum() == pytest.approx(1)
    assert not np.allclose(d, d[0])  # moved away from the uniform start


def test_amkl_beats_svm_at_with_noisy_auxiliary_labels_and_an_uninformative_feature():
    """Equal-weight fusion trusts both features; A-MKL learns to rely on the informative one."""
    svm_at_aps, amkl_aps = [], []
    for seed in range(5):
        kernels, y, train, target = two_domain_problem(seed)
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(len(y), 2))
        kernels[("toy", 1)] = base_kernels(cdist(Z, Z), train, (-1, 0, 1))  # pure noise feature
        noisy = y.copy()
        aux = np.flatnonzero(~target)
        noisy[rng.choice(aux, len(aux) // 5, replace=False)] *= -1
        test = np.flatnonzero(target)[6:]
        svm_at_aps.append(average_precision_score(y[test] == 1, svm_at(kernels, noisy, train, target)[test]))
        amkl_aps.append(average_precision_score(y[test] == 1, amkl(kernels, noisy, train, target)[test]))
    assert np.mean(amkl_aps) > np.mean(svm_at_aps) + 0.03


def test_dtsvm_is_amkl_without_prelearned_classifiers():
    kernels, y, train, target = two_domain_problem()
    no_classifiers = np.empty((len(y), 0))
    expected = amkl_solve(kernels, no_classifiers, y, train, target).scores
    np.testing.assert_allclose(dtsvm(kernels, y, train, target), expected)
    assert not np.allclose(expected, amkl(kernels, y, train, target))


def test_mkl_objective_never_increases_and_weights_stay_on_the_simplex():
    kernels, y, train, target = two_domain_problem()
    result = amkl_solve(kernels, np.empty((len(y), 0)), y, train, target, mismatch=False)
    np.testing.assert_allclose(mkl(kernels, y, train, target), result.scores)
    assert 1 < len(result.objective) <= T_MAX
    assert np.all(np.diff(result.objective) <= 0)
    assert np.all(result.d >= 0)
    assert result.d.sum() == pytest.approx(1)
    assert not np.allclose(result.d, result.d[0])  # moved away from the uniform start
