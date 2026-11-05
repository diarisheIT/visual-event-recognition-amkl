import numpy as np
import pytest

from ver.astpm import astpm
from ver.data import FEATURES

N_WORDS = 20


def random_video(rng, n_frames=10, words_per_frame=30):
    """Visual words with normalised (x, y, t) positions; each frame shares one t value."""
    n = n_frames * words_per_frame
    xyt = np.column_stack([rng.random((n, 2)), np.repeat(np.arange(n_frames) / n_frames, words_per_frame)])
    return rng.integers(N_WORDS, size=n), xyt


@pytest.mark.parametrize("feature", FEATURES)
def test_video_is_at_zero_distance_from_itself(feature):
    video = random_video(np.random.default_rng(0))
    np.testing.assert_allclose(astpm(video, video, N_WORDS, feature), 0, atol=1e-12)


@pytest.mark.parametrize("feature", FEATURES)
def test_distances_are_symmetric(feature):
    rng = np.random.default_rng(1)
    a, b = random_video(rng), random_video(rng, n_frames=7)
    np.testing.assert_allclose(astpm(a, b, N_WORDS, feature), astpm(b, a, N_WORDS, feature))


@pytest.mark.parametrize("feature", FEATURES)
def test_aligned_distance_never_exceeds_unaligned(feature):
    rng = np.random.default_rng(2)
    for _ in range(10):
        _, unaligned, aligned = astpm(random_video(rng), random_video(rng), N_WORDS, feature)
        assert aligned <= unaligned + 1e-12


def test_alignment_recovers_shuffled_volumes():
    """A video whose space-time volumes are swapped around matches its original perfectly only when aligned."""
    a = random_video(np.random.default_rng(3))
    words, xyt = a
    b = words, np.column_stack([(xyt[:, 0] + 0.5) % 1, xyt[:, 1], (xyt[:, 2] + 0.5) % 1])  # swap left/right, first/second half
    level0, unaligned, aligned = astpm(a, b, N_WORDS, "st")
    assert unaligned > 0.1
    np.testing.assert_allclose([level0, aligned], 0, atol=1e-12)
