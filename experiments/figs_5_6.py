"""Figs. 5 and 6: A-MKL's pre-learned classifier coefficients beta and objective history, case (c), round 1."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ver.amkl import amkl_solve
from ver.data import EVENTS, FEATURES, LEVELS, N_TARGET, load_distances, load_labels, load_splits
from ver.kernels import base_kernels
from ver.methods import prelearned_classifiers

BASE_EXPONENTS = range(-3, 2)  # L = {-3, ..., 1}
OUT = Path("results")


def run():
    """Solve A-MKL per event on round 1 with SIFT + ST; returns {event: {"beta": {...}, "objective": [...]}}."""
    dists, labels, splits = load_distances(), load_labels(), load_splits()
    target = np.arange(len(labels)) < N_TARGET
    train, _ = splits[0]
    kernels = {(f, l): base_kernels(dists[(f, l)], train, BASE_EXPONENTS) for f in FEATURES for l in LEVELS}
    results = {}
    for e, event in enumerate(EVENTS):
        y = labels[:, e]
        res = amkl_solve(kernels, prelearned_classifiers(kernels, y, train), y, train, target)
        results[event] = {
            "beta": {f"{f.upper()} L{l}": float(b) for (f, l), b in zip(kernels, res.beta)},
            "objective": [float(g) for g in res.objective],
        }
    return results


def plot(results):
    """Write results/fig_5.png (β bars) and results/fig_6.png (objective curves), one panel per event."""
    fig5, axes5 = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    fig6, axes6 = plt.subplots(2, 3, figsize=(12, 6))
    for ax5, ax6, (event, r) in zip(axes5.flat, axes6.flat, results.items()):
        ax5.bar(list(r["beta"]), list(r["beta"].values()))
        ax5.axhline(0, color="black", linewidth=0.5)
        ax5.set_title(event)
        ax5.tick_params(axis="x", labelsize=8)
        ax6.plot(range(1, len(r["objective"]) + 1), r["objective"], marker="o")
        ax6.set_title(event)
        ax6.set_xlabel("iteration")
        ax6.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax6.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    for ax5, ax6 in zip(axes5[:, 0], axes6[:, 0]):
        ax5.set_ylabel("β")
        ax6.set_ylabel("objective G(d)")
    fig5.suptitle("Fig. 5: pre-learned classifier coefficients β, case (c), round 1")
    fig6.suptitle("Fig. 6: A-MKL objective vs iteration, case (c), round 1")
    fig5.tight_layout()
    fig6.tight_layout()
    fig5.savefig(OUT / "fig_5.png", dpi=150)
    fig6.savefig(OUT / "fig_6.png", dpi=150)


def main():
    OUT.mkdir(exist_ok=True)
    saved = OUT / "figs_5_6.json"
    saved.write_text(json.dumps(run(), indent=1))
    results = json.loads(saved.read_text())
    plot(results)

    print("\nFig. 5: β per event (paper: SIFT L1 > SIFT L0 for every event, ST near 0)")
    for event, r in results.items():
        b = r["beta"]
        print(f"{event:10}" + "".join(f"{k}={v:7.2f}  " for k, v in b.items())
              + ("SIFT L1 > L0" if b["SIFT L1"] > b["SIFT L0"] else "SIFT L1 < L0 (reversed)"))
    print("\nFig. 6: objective (paper: ~5-7e-3, monotone, flat after ~6-8 iterations)")
    for event, r in results.items():
        g = r["objective"]
        print(f"{event:10}{len(g):3} iters  G: {g[0]:.3e} -> {g[-1]:.3e}")


if __name__ == "__main__":
    main()
