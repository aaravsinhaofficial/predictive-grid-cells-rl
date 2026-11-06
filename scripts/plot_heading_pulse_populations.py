"""Compare repair of matched direction-conditioned state substitutions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_heading_mechanistic_confirmation import interval, line, save
from ring_dynamics import wrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/mechanistic_confirmation"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.root / "population_pulse_summary.json").read_text())
    if not summary["complete"] and not args.allow_partial:
        raise ValueError("Wait for all16matched population triplets")
    seeds = summary["training_seeds"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    conditions = (("grid", "Grid", "#2878B5"), ("predictive", "PGC", "#E66A32"),
                  ("grid_heading", "Explicit heading", "#338569"))
    for c, (condition, label, color) in enumerate(conditions):
        phases, baseline = [], []
        for seed in seeds:
            prefix = args.root / f"memory_pulse_{condition}_s{seed}"
            report = json.loads(prefix.with_suffix(".json").read_text())
            v = report["protocol"]["variant_order"].index("native")
            with np.load(prefix.with_suffix(".npz")) as arrays:
                for state, target in (("phase_balanced20", phases), ("baseline", baseline)):
                    rows = []
                    for repeat in range(len(report["evaluation_seeds"])):
                        key = f"repeat{repeat}_{state}_trace_"
                        rows.append(np.abs(wrap(arrays[key + "decoded_heading"][:, v]
                                               - arrays[key + "heading"][:, v])) * 180 / np.pi)
                    target.append(np.concatenate(rows, 1).mean(1))
        time = np.arange(len(phases[0])) * report["protocol"]["trace_interval_seconds"]
        line(axes[0], time, phases, color, label)
        line(axes[0], time, baseline, color, None, ":", band=False)
        for state, offset, filled in (("phase_balanced20", -.10, True), ("normal_matched20", .10, False)):
            values = [row["populations"][condition][f"{state}_first5s_path_deviation_m"] * 100
                      for row in summary["per_seed"]]
            mean, low, high = interval(values)
            axes[1].errorbar(c + offset, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=7,
                color=color, markerfacecolor=color if filled else "white", capsize=3, lw=1.8)
    axes[0].plot([], [], ":", color=".5", lw=2, label="Unperturbed memory")
    axes[0].set(xlabel="Time after navigation starts (s)", ylabel="Mean direction error (°)",
                xlim=(0, 2), ylim=(0, 45), title="Directional cues repair memory")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot([], [], "o", color=".4", label="Direction-state swap")
    axes[1].plot([], [], "o", color=".4", markerfacecolor="white", label="Equal-size normal pulse")
    axes[1].set(xticks=range(3), xticklabels=[row[1] for row in conditions],
                ylabel="Mean path change, first 5 s (cm)", xlim=(-.5, 2.5), ylim=(0, None),
                title="Repair protects the next movements")
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, args.root, "memory_pulse_populations")
    (args.root / "population_pulse_figure_metadata.json").write_text(json.dumps({"complete": summary["complete"],
        "training_seeds": seeds, "bands_and_error_bars": "unadjusted95%tintervals across16paired training seeds",
        "phase_pulse": "balanced+/-20degree cue-conditioned full-state substitutions; actual decoded offsets archived",
        "geometry_caution": "substitution can cross a sharp hidden-state transition; not universally a pure phase-coordinate change",
        "normal_pulse": "same full-state norm, orthogonal to empirical local tangent",
        "all_conditions_native_sensory_input": True}, indent=2) + "\n")
    print(json.dumps({"figure": "memory_pulse_populations", "training_seeds": seeds}))


if __name__ == "__main__":
    main()
