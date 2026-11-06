"""Show the complete cohort's long-delay memory, including unfavorable outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/confirmation_evaluation"))
    args = parser.parse_args()
    report = json.loads((args.root / "memory_summary.json").read_text())
    if not report["complete"] or report["models_loaded"] != 48:
        raise ValueError("Plot requires all48 encoder models and their untrained controls")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
    times = np.array([2, 7, 32])
    for condition, label, color in (("grid", "Grid", "#2878B5"), ("predictive", "PGC", "#E66A32"),
                                    ("grid_heading", "Explicit heading", "#338569")):
        group = report["groups"][condition]["trained"]["in_region"]
        for axis, metric, scale in ((axes[0], "heading_mae_deg", 1.), (axes[1], "coverage", 100.)):
            values = [group[f"blank_{time}s_{metric}"] for time in times]
            mean = np.array([v["mean"] for v in values]) * scale
            low = np.array([v["ci95"][0] for v in values]) * scale
            high = np.array([v["ci95"][1] for v in values]) * scale
            axis.plot(times, mean, "o-", color=color, lw=2, ms=4, label=label)
            axis.fill_between(times, low, high, color=color, alpha=.12, linewidth=0)
    axes[0].set(title="Heading drifts without cues", ylabel="Direction error (°)", ylim=(0, 90))
    axes[1].set(title="Angular coverage also falls", ylabel="Decoded angular coverage (%)", ylim=(0, 102))
    for axis in axes:
        axis.set(xlabel="Time without input (s)", xticks=times, xlim=(0, 33))
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.root / f"blank_memory.{extension}", dpi=220)
    plt.close(fig)
    (args.root / "blank_memory_figure_metadata.json").write_text(json.dumps({"models": 48, "seeds": list(range(2001, 2017)),
        "times": "total seconds of all-zero input following four native initialization cues",
        "contexts": "mean of(0,0) and(0.8,-0.8) within each training seed",
        "bands": "descriptive95%training-seed tintervals; no primary tests",
        "coverage": "one minus largest decoded angular gap divided by2pi; not full-state topology",
        "interpretation": "PGC benefit with ongoing directional information is distinct from stronger autonomous long-delay memory"}, indent=2) + "\n")
    print(json.dumps({"figure": str(args.root / "blank_memory.png"), "models": 48}))


if __name__ == "__main__":
    main()
