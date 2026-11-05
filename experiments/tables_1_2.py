"""Tables 1 and 2: SVM_AT with the default kernel parameter, per feature, pyramid level and kernel type.

Only the level-0 and level-1 (aligned) rows can be reproduced: the archive has no unaligned (USTPM) distances.
"""

import json
from pathlib import Path

import numpy as np

from ver.data import EVENTS, FEATURES, LEVELS, load_distances, load_labels, load_splits
from ver.evaluate import average_precisions, mean_std
from ver.kernels import KERNEL_TYPES, default_gamma, kernel
from ver.svm import svm_scores

PAPER = {  # (feature, level) -> MAP % per kernel type, Tables 1 and 2
    ("sift", 0): (41.4, 44.2, 45.0, 46.2),
    ("sift", 1): (50.4, 53.8, 52.9, 51.0),
    ("st", 0): (22.2, 36.1, 22.0, 35.6),
    ("st", 1): (20.6, 35.8, 22.3, 35.9),
}


def main():
    dists, labels, splits = load_distances(), load_labels(), load_splits()
    results = {}
    for (f, l), D in dists.items():
        for kind in KERNEL_TYPES:
            maps = []
            for train, test in splits:
                K = kernel(D, kind, default_gamma(D, train))
                scores = np.column_stack([svm_scores(K, labels[:, e], train) for e in range(len(EVENTS))])
                maps.append(average_precisions(scores, labels, test).mean())
            results[f"{f}_L{l}_{kind}"] = mean_std(maps)

    for f, table in zip(FEATURES, ("Table 1 (SIFT)", "Table 2 (ST)")):
        print(f"\n{table}: MAP % ours (paper)")
        print(f"{'':10}" + "".join(f"{k:>24}" for k in KERNEL_TYPES))
        for l in LEVELS:
            cells = [
                f"{m:5.1f} ± {s:3.1f} ({p:4.1f})"
                for (m, s), p in zip((results[f"{f}_L{l}_{k}"] for k in KERNEL_TYPES), PAPER[(f, l)])
            ]
            print(f"{'Level-' + str(l):10}" + "".join(f"{c:>24}" for c in cells))

    out = Path("results") / "tables_1_2.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({k: [round(float(m), 2), round(float(s), 2)] for k, (m, s) in results.items()}, indent=1))


if __name__ == "__main__":
    main()
