import numpy as np

from ver.data import N_TARGET, load_labels, load_splits, proportion_splits


def test_paper_splits_are_3_per_event_and_disjoint():
    labels = load_labels()
    for train, test in load_splits():
        labeled = train[train < N_TARGET]
        assert (labels[labeled] == 1).sum(0).tolist() == [3] * 6
        assert len(np.intersect1d(train, test)) == 0
        assert len(labeled) + len(test) == N_TARGET


def test_proportion_splits_label_at_least_one_per_event():
    labels = load_labels()
    for train, test in proportion_splits(labels, 0.05):
        labeled = train[train < N_TARGET]
        assert ((labels[labeled] == 1).sum(0) >= 1).all()
        assert sorted(np.concatenate([labeled, test])) == list(range(N_TARGET))
