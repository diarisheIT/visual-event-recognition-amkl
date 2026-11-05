"""Aligned Space-Time Pyramid Matching (Section 3): video-to-video distances at pyramid levels 0 and 1.

A video is (words, xyt): visual-word indices and their normalised (x, y, t) positions in [0, 1).
Words sharing a t value come from the same frame."""

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _histogram(words, n_words):
    h = np.bincount(words, minlength=n_words).astype(float)
    return h / max(h.sum(), 1)


def _chi2(h, g):
    s = h + g
    return 0.5 * np.sum(np.divide((h - g) ** 2, s, out=np.zeros_like(s), where=s > 0))


def _volumes(video, level):
    """Words of each space-time volume: level 0 is the whole video, level 1 the 2×2×2 split."""
    words, xyt = video
    volume = (xyt >= 0.5).astype(int) @ [4, 2, 1] if level else np.zeros(len(words), int)
    return [(words[volume == v], xyt[volume == v, 2]) for v in range(8 ** level)]


def _volume_distance(a, b, n_words, feature):
    (wa, ta), (wb, tb) = a, b
    if feature == "st":
        return _chi2(_histogram(wa, n_words), _histogram(wb, n_words))
    # SIFT: EMD between the sets of image-block histograms (one block per frame), uniform weights, Euclidean ground
    # distance; an empty volume counts as one empty block
    blocks = [np.array([_histogram(w[t == f], n_words) for f in np.unique(t)]) if len(t) else np.zeros((1, n_words))
              for w, t in ((wa, ta), (wb, tb))]
    return ot.emd2(ot.unif(len(blocks[0])), ot.unif(len(blocks[1])), cdist(*blocks))


def astpm(a, b, n_words, feature):
    """Level-0, USTPM (same-position volumes) and ASTPM (optimal one-to-one volumes) distances between videos."""
    level0 = _volume_distance(*_volumes(a, 0), *_volumes(b, 0), n_words, feature)
    D = np.array([[_volume_distance(va, vb, n_words, feature) for vb in _volumes(b, 1)] for va in _volumes(a, 1)])
    return level0, np.trace(D) / len(D), D[linear_sum_assignment(D)].mean()
