"""Show how the same memory pulse affects heading and the subsequent path."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ring_dynamics import wrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    arrays = np.load(args.result.with_suffix(".npz"))
    repeats = len(report["evaluation_seeds"])
    interval = report["protocol"]["trace_interval_seconds"]
    variants = report["protocol"]["variant_order"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    largest_shown_deviation = 0
    for variant, color, label in (("native", "#E66A32", "PGC cue present"),
                                  ("ordinary", "#858585", "PGC cue removed")):
        v = variants.index(variant)
        for pulse, style in (("phase_balanced20", "-"), ("baseline", ":")):
            errors, deviations = [], []
            for repeat in range(repeats):
                prefix = f"repeat{repeat}_{pulse}_trace_"
                angle = arrays[prefix + "decoded_heading"][:, v]
                heading = arrays[prefix + "heading"][:, v]
                errors.append(np.abs(wrap(angle - heading)) * 180 / np.pi)
                positions = arrays[prefix + "position"][:, v]
                baseline = arrays[f"repeat{repeat}_baseline_trace_position"][:, v]
                deviations.append(np.linalg.norm(positions - baseline, axis=-1))
            error = np.concatenate(errors, 1).mean(1)
            deviation = np.concatenate(deviations, 1).mean(1) * 100
            time = np.arange(len(error)) * interval
            axes[0].plot(time, error, style, color=color, lw=2, label=label if pulse != "baseline" else None)
            if pulse != "baseline":
                axes[1].plot(time, deviation, color=color, lw=2, label=label)
                largest_shown_deviation = max(largest_shown_deviation, float(deviation[time <= 5].max()))
    axes[0].plot([], [], ":", color=".5", lw=2, label="Unperturbed memory")
    axes[0].set(xlabel="Time after navigation starts (s)", ylabel="Mean direction error (°)",
                xlim=(0, 2), ylim=(0, 90), title="PGCs correct a memory error")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set(xlabel="Time after navigation starts (s)", ylabel="Change from unperturbed path (cm)",
                xlim=(0, 5), ylim=(0, largest_shown_deviation * 1.12), title="Correction limits the path change")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.out.with_suffix("." + suffix), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
