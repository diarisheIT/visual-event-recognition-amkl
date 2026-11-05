"""Turns video-to-video distances into the paper's four base kernel types (Section 5.2)."""

import numpy as np

KERNEL_TYPES = ("gaussian", "laplacian", "isd", "id")


def default_gamma(D, train_idx):
    """gamma0 = 1 / A, with A the mean squared distance between training samples."""
    return 1.0 / np.mean(D[np.ix_(train_idx, train_idx)] ** 2)


def kernel(D, kind, gamma):
    if kind == "gaussian":
        return np.exp(-gamma * D**2)
    if kind == "laplacian":
        return np.exp(-np.sqrt(gamma) * D)
    if kind == "isd":
        return 1.0 / (gamma * D**2 + 1)
    if kind == "id":
        return 1.0 / (np.sqrt(gamma) * D + 1)
    raise ValueError(f"unknown kernel type {kind!r}")


def base_kernels(D, train_idx, exponents):
    """4 * len(exponents) kernels on all videos, with gamma = 2**l * gamma0 for l in exponents."""
    g0 = default_gamma(D, train_idx)
    return [kernel(D, kind, 2.0**l * g0) for kind in KERNEL_TYPES for l in exponents]
