"""Loads the authors' archived distance matrices, labels and train/test splits."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "distance_matrix"

# Column order of labels.mat (verified from per-event target counts 16/14/6/57/75/27);
# it differs from the order listed in the archive's README.
EVENTS = ("birthday", "parade", "picnic", "show", "sports", "wedding")
N_TARGET = 195  # rows 0..194 are target-domain videos, rows 195..1100 auxiliary-domain videos
FEATURES = ("sift", "st")
LEVELS = (0, 1)
_FILE_FEATURE = {"sift": "SIFT", "st": "STIP"}


def load_distances(data_dir=DATA_DIR):
    """{(feature, level): (N, N) ASTPM distance matrix} for all 1101 videos."""
    return {
        (f, l): loadmat(data_dir / f"dist_{_FILE_FEATURE[f]}_L{l}.mat")["distMat"]
        for f in FEATURES
        for l in LEVELS
    }


def load_labels(data_dir=DATA_DIR):
    """(N, 6) matrix of +1/-1, one event per video, columns ordered as EVENTS."""
    return loadmat(data_dir / "labels.mat")["labels"].astype(int)


def load_splits(data_dir=DATA_DIR):
    """The paper's 5 rounds as 0-based (train_idx, test_idx).

    train = all 906 auxiliary videos + 3 labeled target videos per event; test = remaining 177 target videos.
    """
    splits = []
    for r in range(1, 6):
        m = loadmat(data_dir / "TRAINING_IND" / f"{r}.mat")
        splits.append((m["training_ind"].ravel().astype(int) - 1, m["test_ind"].ravel().astype(int) - 1))
    return splits


def proportion_splits(labels, r, rounds=5, seed=0):
    """Fig. 7 splits: a proportion r of each event's target videos is labeled (at least 1), the rest is test."""
    rng = np.random.default_rng(seed)
    aux = np.arange(N_TARGET, len(labels))
    target_event = labels[:N_TARGET].argmax(1)
    splits = []
    for _ in range(rounds):
        labeled = np.concatenate([
            rng.choice(idx, max(1, round(r * len(idx))), replace=False)
            for idx in (np.flatnonzero(target_event == e) for e in range(len(EVENTS)))
        ])
        test = np.setdiff1d(np.arange(N_TARGET), labeled)
        splits.append((np.concatenate([aux, np.sort(labeled)]), test))
    return splits
