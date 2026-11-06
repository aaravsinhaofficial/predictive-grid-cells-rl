"""Plot all paired models from the completed independent geometry replication."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def interval(values):
    values = np.asarray(values)
    mean = values.mean()
    width = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - width, mean + width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/geometry_confirmation"))
    args = parser.parse_args()
    report = json.loads((args.root / "summary.json").read_text())
    if not report["complete"] or report["models_loaded"] != 32:
        raise ValueError("Independent replication must include all32 models before plotting")
    rows = {(r["seed"], r["condition"]): r for r in report["per_seed"]}
    seeds = range(5001, 5017)
    colors = {"grid": "#2878B5", "predictive": "#E66A32"}
    labels = {"grid": "Grid", "predictive": "PGC"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    jitter = np.linspace(-.075, .075, 16)
    for i, seed in enumerate(seeds):
        values = [rows[seed, c]["primary"] for c in ("grid", "predictive")]
        axes[0].plot(np.array([0., 1.]) + jitter[i], values, color=".8", lw=.7, zorder=1)
        for column, condition in enumerate(("grid", "predictive")):
            axes[0].scatter(column + jitter[i], values[column], color=colors[condition], s=22, zorder=3)
    for column, condition in enumerate(("grid", "predictive")):
        mean, low, high = interval([rows[seed, condition]["primary"] for seed in seeds])
        axes[0].errorbar(column + .18, mean, yerr=[[mean - low], [high - mean]], fmt="_", color="black", ms=10, capsize=3, lw=1.6)
    axes[0].set(xticks=[0, 1], xticklabels=["Grid", "PGC"], xlim=(-.25, 1.35), title="Sensitivity of the learned memory map")
    variants = ("native_blank0", "native_blank40", "ordinary_blank0", "ordinary_blank40")
    for condition, offset in (("grid", -.06), ("predictive", .06)):
        means, lows, highs = [], [], []
        for variant in variants:
            mean, low, high = interval([rows[seed, condition]["maps"][variant]["log10_peak_over_median"] for seed in seeds])
            means.append(mean); lows.append(low); highs.append(high)
        means, lows, highs = map(np.asarray, (means, lows, highs))
        for pair in ([0, 1], [2, 3]):
            axes[1].plot(np.asarray(pair) + offset, means[pair], color=colors[condition], lw=1.6)
        axes[1].errorbar(np.arange(4) + offset, means, yerr=[means - lows, highs - means], fmt="o", ms=5,
                         color=colors[condition], capsize=3, lw=1.5, label=labels[condition])
    axes[1].axvline(1.5, color=".85", lw=.8)
    axes[1].set(xticks=range(4), xticklabels=["Cue end", "+2 s blank", "Cue end", "+2 s blank"],
                title="Input initialization and blank recurrence")
    axes[1].text(.25, -.23, "Native initialization", ha="center", transform=axes[1].transAxes, fontsize=10)
    axes[1].text(.77, -.23, "Ordinary initialization", ha="center", transform=axes[1].transAxes, fontsize=10)
    axes[1].legend(frameon=False, fontsize=10)
    for axis in axes:
        axis.set_ylabel("Peak / median state sensitivity")
        _, high = axis.get_ylim()
        ratios = np.array([1., 3., 10., 30., 100., 300., 1000., 3000., 10000.])
        ratios = ratios[np.log10(ratios) <= high]
        axis.set_ylim(0, high)
        axis.set_yticks(np.log10(ratios), [f"{value:g}" for value in ratios])
        axis.grid(axis="y", color=".92", lw=.6)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.root / f"memory_map_sharpness.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (args.root / "figure_metadata.json").write_text(json.dumps({"all32_models_included": True,
        "training_seeds": list(seeds), "left": "single primary: native4cue then40blank; all paired seed values",
        "right": "prespecified descriptive cue-end and commonordinary initialization controls",
        "y_coordinate": "log10 peak/median exact state angular sensitivity, tick labels give original ratio",
        "bars": "mean log sensitivity and descriptive95%training-seed tintervals",
        "interpretation": "learned memory-initialization map, not isolated recurrent weights or a ring pass/fail test"}, indent=2) + "\n")
    print(json.dumps({"figure": str(args.root / "memory_map_sharpness.png"), "models": 32}))


if __name__ == "__main__":
    main()
