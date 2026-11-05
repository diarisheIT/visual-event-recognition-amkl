"""Table 4: A-MKL with the event's 4 pre-learned classifiers (A-MKL_4) vs all 24 from the six events (A-MKL_24),
for four bandwidth sets H of the pre-learned classifiers; case (c), base kernels L = {-3..1}, over the paper's 5 rounds."""

import json
from pathlib import Path

import numpy as np

from ver.amkl import amkl_solve
from ver.data import EVENTS, FEATURES, LEVELS, N_TARGET, load_distances, load_labels, load_splits
from ver.evaluate import average_precisions, mean_std
from ver.kernels import base_kernels
from ver.methods import prelearned_classifiers

BASE_EXPONENTS = range(-3, 2)  # L = {-3, ..., 1}
H_SETS = {f"{{{lo}..1}}": range(lo, 2) for lo in (-3, -4, -5, -6)}
VARIANTS = ("A-MKL_4", "A-MKL_24")
PAPER = {  # (variant, H) -> MAP % (mean, std), Table 4
    ("A-MKL_4", "{-3..1}"): (58.20, 1.87), ("A-MKL_4", "{-4..1}"): (58.33, 2.33),
    ("A-MKL_4", "{-5..1}"): (58.56, 2.53), ("A-MKL_4", "{-6..1}"): (57.16, 3.02),
    ("A-MKL_24", "{-3..1}"): (58.74, 3.30), ("A-MKL_24", "{-4..1}"): (59.04, 3.53),
    ("A-MKL_24", "{-5..1}"): (59.25, 3.73), ("A-MKL_24", "{-6..1}"): (59.28, 2.57),
}
OUT = Path("results")


def main():
    dists, labels, splits = load_distances(), load_labels(), load_splits()
    target = np.arange(len(labels)) < N_TARGET
    keys = [(f, l) for f in FEATURES for l in LEVELS]
    maps = {(v, h): [] for v in VARIANTS for h in H_SETS}  # -> per-round MAP
    for train, test in splits:
        kernels = {k: base_kernels(dists[k], train, BASE_EXPONENTS) for k in keys}
        for h, exponents in H_SETS.items():
            h_kernels = {k: base_kernels(dists[k], train, exponents) for k in keys}
            F = [prelearned_classifiers(h_kernels, labels[:, e], train) for e in range(len(EVENTS))]  # 6 x (N, 4)
            for v in VARIANTS:
                scores = np.column_stack([
                    amkl_solve(kernels, F[e] if v == "A-MKL_4" else np.hstack(F), labels[:, e], train, target).scores
                    for e in range(len(EVENTS))
                ])
                maps[(v, h)].append(average_precisions(scores, labels, test).mean())

    results = {v: {h: [round(float(x), 2) for x in mean_std(maps[(v, h)])] for h in H_SETS} for v in VARIANTS}

    print("\nTable 4: MAP % ours (paper), case (c), L = {-3..1}")
    print(f"{'':10}" + "".join(f"{'H = ' + h:>30}" for h in H_SETS))
    for v in VARIANTS:
        cells = ["{:5.2f} ± {:4.2f} ({:5.2f} ± {:4.2f})".format(*results[v][h], *PAPER[(v, h)]) for h in H_SETS]
        print(f"{v:10}" + "".join(f"{c:>30}" for c in cells))

    OUT.mkdir(exist_ok=True)
    (OUT / "table_4.json").write_text(json.dumps(results, indent=1))

    # A-MKL_4 with H = L is the same run as Table 3 (c) A-MKL
    expected = json.loads((OUT / "table_3.json").read_text())["c"]["A-MKL"]["map"]  # written by experiments/table_3.py
    assert results["A-MKL_4"]["{-3..1}"] == expected, f"A-MKL_4, H = {{-3..1}} differs from Table 3 (c) A-MKL {expected}"
    print(f"\nConsistency: A-MKL_4, H = {{-3..1}} matches Table 3 (c) A-MKL {expected}")


if __name__ == "__main__":
    main()
