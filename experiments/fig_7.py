"""Fig. 7: MAP vs proportion r of labeled target videos per event, case (c), five seeded rounds per r.
SVM_T, DTSVM, A-MKL_4 (H = {-3..1}) and A-MKL_24 (H = {-6..1}); base kernels L = {-3..1}."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ver.amkl import amkl_solve
from ver.data import EVENTS, FEATURES, LEVELS, N_TARGET, load_distances, load_labels, proportion_splits
from ver.evaluate import average_precisions, mean_std
from ver.kernels import base_kernels
from ver.methods import METHODS, prelearned_classifiers

BASE_EXPONENTS = range(-3, 2)  # L = {-3, ..., 1}
H_24 = range(-6, 2)  # H = {-6, ..., 1} for A-MKL_24
VARIANTS = ("SVM_T", "DTSVM", "A-MKL_4", "A-MKL_24")
PAPER = {  # r -> approximate MAP % per variant, read from the paper's Fig. 7 chart
    r: dict(zip(VARIANTS, maps)) for r, maps in {
        "5%": (39, 48, 51, 53), "10%": (44, 49, 51.5, 53.5), "20%": (48, 60, 61, 61.5), "30%": (52.5, 59.5, 60.5, 62),
    }.items()
}
OUT = Path("results")


def run():
    dists, labels = load_distances(), load_labels()
    target = np.arange(len(labels)) < N_TARGET
    keys = [(f, l) for f in FEATURES for l in LEVELS]
    results = {}
    for r in PAPER:
        maps = {v: [] for v in VARIANTS}  # -> per-round MAP
        for train, test in proportion_splits(labels, int(r[:-1]) / 100):
            kernels = {k: base_kernels(dists[k], train, BASE_EXPONENTS) for k in keys}
            h_kernels = {k: base_kernels(dists[k], train, H_24) for k in keys}
            F24 = np.hstack([prelearned_classifiers(h_kernels, labels[:, e], train) for e in range(len(EVENTS))])
            per_event = {
                "SVM_T": lambda y: METHODS["SVM_T"](kernels, y, train, target),
                "DTSVM": lambda y: METHODS["DTSVM"](kernels, y, train, target),
                "A-MKL_4": lambda y: METHODS["A-MKL"](kernels, y, train, target),
                "A-MKL_24": lambda y: amkl_solve(kernels, F24, y, train, target).scores,
            }
            for v, method in per_event.items():
                scores = np.column_stack([method(labels[:, e]) for e in range(len(EVENTS))])
                maps[v].append(average_precisions(scores, labels, test).mean())
        results[r] = {v: [round(float(x), 2) for x in mean_std(maps[v])] for v in VARIANTS}
    OUT.mkdir(exist_ok=True)
    (OUT / "fig_7.json").write_text(json.dumps(results, indent=1))


def plot():
    results = json.loads((OUT / "fig_7.json").read_text())
    x = np.arange(len(results))
    width = 0.8 / len(VARIANTS)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, v in enumerate(VARIANTS):
        mean, std = np.array([results[r][v] for r in results]).T
        ax.bar(x + (i - (len(VARIANTS) - 1) / 2) * width, mean, width, yerr=std, capsize=3, label=v)
    ax.set_xticks(x, [f"r = {r}" for r in results])
    ax.set_ylabel("MAP (%)")
    ax.set_ylim(0, 80)
    ax.legend(ncol=len(VARIANTS), loc="upper left")
    ax.set_title("Fig. 7: MAP vs proportion of labeled target videos, case (c), mean ± std over 5 rounds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_7.png", dpi=150)

    print("\nFig. 7: MAP % ours (paper, approximate), case (c)")
    print(f"{'':6}" + "".join(f"{v:>22}" for v in VARIANTS))
    for r, paper in PAPER.items():
        cells = ["{:5.2f} ± {:4.2f} (~{:4.1f})".format(*results[r][v], paper[v]) for v in VARIANTS]
        print(f"{r:6}" + "".join(f"{c:>22}" for c in cells))


def main():
    if "--plot-only" not in sys.argv:  # --plot-only redraws the figure from the saved results/fig_7.json
        run()
    plot()


if __name__ == "__main__":
    main()
