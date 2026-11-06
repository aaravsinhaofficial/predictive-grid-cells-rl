"""Complete eight-pair raw-input memory and anchoring diagnostics."""
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
    half = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/raw_memory_evaluation"))
    args = parser.parse_args()
    memory = json.loads((args.root / "memory_summary.json").read_text())
    anchor = json.loads((args.root / "anchor_summary.json").read_text())
    if not all(r["complete"] and r["models_loaded"] == 16 for r in (memory, anchor)):
        raise ValueError("Wait for every raw-input Grid/PGC model and memory/anchor evaluation")
    seeds = list(range(3001, 3009))
    lookup = {(r["seed"], r["condition"]): r for r in anchor["per_seed"]}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    times = np.array([2, 7, 32])
    for condition, label, color in (("grid", "Grid", "#2878B5"), ("predictive", "PGC", "#E66A32")):
        group = memory["groups"][condition]["trained"]["in_region"]
        values = [group[f"blank_{time}s_heading_mae_deg"] for time in times]
        means = [v["mean"] for v in values]
        axes[0].plot(times, means, "o-", color=color, label=label, lw=2, ms=4)
        axes[0].fill_between(times, [v["ci95"][0] for v in values], [v["ci95"][1] for v in values], color=color, alpha=.12, lw=0)
    axes[0].set(xlabel="Time without input (s)", ylabel="Direction error (°)", xticks=times,
                xlim=(0, 33), ylim=(0, None), title="A  Memory without input")
    axes[0].legend(frameon=False, fontsize=9)
    branches = (("grid", "native", "Grid", "#2878B5"), ("predictive", "native", "PGC", "#E66A32"),
                ("predictive", "ordinary", "PGC cue\nreplaced", "#858585"))
    for axis, metric in ((axes[1], "baseline_mae_deg"), (axes[2], "phase_local_gain_median")):
        groups = [[lookup[s, condition]["in_region"][variant][metric] for s in seeds] for condition, variant, _, _ in branches]
        jitter = np.linspace(-.055, .055, len(seeds))
        for i in range(len(seeds)):
            axis.plot(np.arange(3) + jitter[i], [group[i] for group in groups], color=".84", lw=.6, zorder=1)
        for index, (values, branch) in enumerate(zip(groups, branches)):
            axis.scatter(index + jitter, values, color=branch[3], s=17, zorder=2)
            mean, low, high = interval(values)
            axis.errorbar(index + .15, mean, yerr=[[mean - low], [high - mean]], fmt="_", color="black", ms=9, capsize=3, lw=1.5)
        axis.set_xticks(range(3), [branch[2] for branch in branches])
        axis.set_xlim(-.2, 2.35)
    axes[1].set(ylabel="Direction error (°)", title="B  After the cue returns", ylim=(0, None))
    axes[2].set(ylabel="Phase disturbance retained", title="C  Perturbations shrink")
    axes[2].axhline(1, color=".5", ls=":", lw=.8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.root / f"raw_memory_and_anchor.{extension}", dpi=220)
    plt.close(fig)
    (args.root / "figure_metadata.json").write_text(json.dumps({"complete": True, "models": 16,
        "training_seeds": seeds, "architecture": "unconstrained64RNN with linear sensory input; no nonlinear encoder",
        "memory": "mean of two in-region positions per seed; total blank time after4cue initialization",
        "anchoring": "5seconds after cue returns, following2seconds blank; mean of5in-region contexts per seed",
        "phase_gain": "median local decoded displacement retention; read with direction accuracy and coverage",
        "dots": "all eight paired training seeds; no model excluded", "intervals": "descriptive95%training-seed tintervals",
        "scope": "separate raw architecture cohort; not pooled with encoder results",
        "not_a_ring_classifier": True}, indent=2) + "\n")
    print(json.dumps({"figure": str(args.root / "raw_memory_and_anchor.png"), "models": 16}))


if __name__ == "__main__":
    main()
