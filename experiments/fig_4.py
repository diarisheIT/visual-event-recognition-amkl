"""Fig. 4: per-event AP (mean ± std over 5 rounds) of every method in the three feature cases, from saved Table 3."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ver.data import EVENTS
from ver.methods import METHODS

OUT = Path("results")
CASES = {"a": "SIFT", "b": "ST", "c": "SIFT + ST"}


def main():
    results = json.loads((OUT / "table_3.json").read_text())  # written by experiments/table_3.py
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharey=True)
    for ax, (case, name) in zip(axes, CASES.items()):
        x = np.arange(len(EVENTS))
        width = 0.8 / len(METHODS)
        for i, m in enumerate(METHODS):
            mean, std = np.array([results[case][m]["ap"][e] for e in EVENTS]).T
            ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, mean, width, yerr=std, capsize=2, label=m)
        ax.set_xticks(x, EVENTS)
        ax.set_ylabel("AP (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"({case}) {name}")
    axes[0].legend(ncol=len(METHODS), loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=9)
    fig.suptitle("Fig. 4: per-event AP, mean ± std over 5 rounds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_4.png", dpi=150)

    print("\nFig. 4: per-event AP % (paper: sports 80-90, parade 60-75, show 50-70, wedding 25-50, "
          "birthday 10-25, picnic 0-50)")
    for case in CASES:
        print(f"\n({case})       " + "".join(f"{m:>9}" for m in METHODS))
        for e in EVENTS:
            print(f"{e:12}" + "".join(f"{results[case][m]['ap'][e][0]:9.2f}" for m in METHODS))


if __name__ == "__main__":
    main()
