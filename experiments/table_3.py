"""Table 3: MAP of every method in three feature cases, (a) SIFT, (b) ST, (c) SIFT + ST, over the paper's 5 rounds."""

import json
from pathlib import Path

import numpy as np

from ver.data import EVENTS, FEATURES, LEVELS, N_TARGET, load_distances, load_labels, load_splits
from ver.evaluate import average_precisions, mean_std
from ver.kernels import base_kernels
from ver.methods import METHODS

BASE_EXPONENTS = range(-3, 2)  # L = {-3, ..., 1}
CASES = {"a": ("sift",), "b": ("st",), "c": ("sift", "st")}
PAPER = {  # (case, method) -> MAP % (mean, std), Table 3
    ("a", "SVM_T"): (42.32, 5.50), ("a", "SVM_AT"): (53.93, 5.58), ("a", "FR"): (49.98, 5.63),
    ("a", "A-SVM"): (38.42, 7.93),
    ("a", "MKL"): (47.19, 2.59), ("a", "DTSVM"): (52.36, 1.88), ("a", "A-MKL"): (57.14, 2.34),
    ("b", "SVM_T"): (32.56, 2.08), ("b", "SVM_AT"): (24.73, 2.22), ("b", "FR"): (28.44, 2.61),
    ("b", "A-SVM"): (24.95, 1.25),
    ("b", "MKL"): (35.34, 1.55), ("b", "DTSVM"): (31.07, 2.60), ("b", "A-MKL"): (37.24, 1.58),
    ("c", "SVM_T"): (42.00, 4.94), ("c", "SVM_AT"): (36.23, 3.37), ("c", "FR"): (44.11, 3.57),
    ("c", "A-SVM"): (32.40, 4.99),
    ("c", "MKL"): (46.92, 2.53), ("c", "DTSVM"): (53.78, 2.99), ("c", "A-MKL"): (58.20, 1.87),
}


def main():
    dists, labels, splits = load_distances(), load_labels(), load_splits()
    target = np.arange(len(labels)) < N_TARGET
    aps = {(case, m): [] for case in CASES for m in METHODS}  # -> per-round (n_events,) AP
    for train, test in splits:
        all_kernels = {(f, l): base_kernels(dists[(f, l)], train, BASE_EXPONENTS) for f in FEATURES for l in LEVELS}
        for case, feats in CASES.items():
            kernels = {(f, l): ks for (f, l), ks in all_kernels.items() if f in feats}
            for name, method in METHODS.items():
                scores = np.column_stack([
                    method(kernels, labels[:, e], train, target) for e in range(len(EVENTS))
                ])
                aps[(case, name)].append(average_precisions(scores, labels, test))

    results = {}
    for (case, m), per_round in aps.items():
        map_mean, map_std = mean_std([ap.mean() for ap in per_round])
        ap_mean, ap_std = mean_std(per_round)
        results.setdefault(case, {})[m] = {
            "map": [round(float(map_mean), 2), round(float(map_std), 2)],
            "ap": {e: [round(float(a), 2), round(float(s), 2)] for e, a, s in zip(EVENTS, ap_mean, ap_std)},
        }

    print("\nTable 3: MAP % ours (paper)")
    print(f"{'':6}" + "".join(f"{m:>30}" for m in METHODS))
    for case in CASES:
        cells = [
            "{:5.2f} ± {:4.2f} ({:5.2f} ± {:4.2f})".format(*results[case][m]["map"], *PAPER[(case, m)])
            for m in METHODS
        ]
        print(f"({case}){'':3}" + "".join(f"{c:>30}" for c in cells))

    out = Path("results") / "table_3.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
