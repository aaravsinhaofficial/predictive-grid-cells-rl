"""Plot the complete, separately analyzed raw-input architecture cohort."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ring_heading/raw_sensitivity_evaluation"


def panel(ax, arrays, labels, colors, title):
    arrays = np.asarray(arrays)
    x = np.arange(len(labels))
    for row in arrays.T:
        ax.plot(x, row, color="#cbd0d4", lw=.7, zorder=1)
    for j, (values, color) in enumerate(zip(arrays, colors)):
        ci = t.ppf(.975, 7) * values.std(ddof=1) / np.sqrt(8)
        ax.scatter(np.full(8, j), values, color=color, s=21, alpha=.8, zorder=3)
        ax.errorbar(j+.13, values.mean(), yerr=ci, color=color, fmt="_", ms=13,
                    capsize=3, lw=1.8, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Heading error (degrees)")
    ax.set_title(title, loc="left", fontsize=11, pad=13)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=.2)


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    rows = summary["per_seed"]
    if sorted(r["seed"] for r in rows) != list(range(3001, 3009)):
        raise ValueError("Require every one of the eight fixed seed triplets")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    panel(axes[0], [[r[c]["same_region_native_mae_deg"] for r in rows]
                   for c in ("grid", "predictive", "grid_heading")],
          ["Grid", "PGC + grid", "Explicit\ndirection"],
          ["#2878B5", "#E66A32", "#338569"], "A  Native inputs")
    panel(axes[1], [[r["predictive"][k] for r in rows]
                   for k in ("same_region_native_mae_deg", "same_region_ordinary_mae_deg")],
          ["Predictive", "Ordinary\nreplacement"],
          ["#E66A32", "#7e8791"], "B  Same frozen PGC networks")
    fig.suptitle("Without the nonlinear sensory encoder", fontsize=14, y=.99)
    fig.text(.5, .012, "Eight paired training seeds; means and 95% seed-level intervals", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .055, 1, .94))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"raw_architecture.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
