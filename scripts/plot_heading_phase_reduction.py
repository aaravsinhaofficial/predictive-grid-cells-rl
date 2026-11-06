"""Minimal held-out field and recovery figure for a frozen RNN phase reduction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_heading_phase_reduction import basis
from ring_dynamics import wrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    archive = np.load(args.result.with_suffix(".npz"))
    coefficients = result["coefficients"]
    phi = np.concatenate([archive[f"context{i}_field_phi"] for i in (2, 3, 4)])
    cue = np.concatenate([archive[f"context{i}_field_cue"] for i in (2, 3, 4)])
    actual = np.concatenate([archive[f"context{i}_field_native"] for i in (2, 3, 4)])
    error = wrap(phi - cue) * 180 / np.pi
    correcting = (actual - basis(phi) @ np.array(coefficients["f0_coefficients"])) * 180 / np.pi
    bins = np.linspace(-180, 180, 25)
    centers = (bins[1:] + bins[:-1]) / 2
    grouped = [correcting[(error >= lo) & (error < hi)] for lo, hi in zip(bins[:-1], bins[1:])]
    mean, low, high = [np.array([fn(values) for values in grouped]) for fn in
                       (np.mean, lambda v: np.quantile(v, .1), lambda v: np.quantile(v, .9))]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    measured_color, model_color = "#E66A32", "#2878B5"
    axes[0].fill_between(centers, low, high, color=measured_color, alpha=.12, linewidth=0)
    axes[0].plot(centers, mean, "o", ms=3.5, color=measured_color, label="Trained RNN")
    fine = np.linspace(-180, 180, 361)
    axes[0].plot(fine, -coefficients["k_per_second"] * np.sin(np.deg2rad(fine)) * 180 / np.pi,
                 color=model_color, lw=2, ls="--", label="One sine term")
    axes[0].axhline(0, color=".7", lw=.7)
    axes[0].set(xlabel="Memory error from cue (°)", ylabel="Cue correction (°/s)",
                xticks=[-180, -90, 0, 90, 180], xlim=(-180, 180), title="The cue pushes memory back")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    trajectories = {}
    for label in ("actual", "sine", "no_anchor"):
        rows = []
        for context in (2, 3, 4):
            prefix = f"context{context}_recovery_"
            cue_angle = archive[prefix + "cue"]
            initial_offset = archive[prefix + "nominal_initial_offsets_deg"]
            mask = (np.abs(initial_offset) > 0) & (np.abs(initial_offset) <= 45)
            trajectory = archive[prefix + "native_" + label]
            rows.append(np.abs(wrap(trajectory[:, mask] - cue_angle[mask])) * 180 / np.pi)
        trajectories[label] = np.concatenate(rows, axis=1).mean(1)
    time = np.arange(len(trajectories["actual"])) * result["config"]["dt"]
    for name, color, style, label in (("actual", measured_color, "-", "Trained RNN"),
                                      ("sine", model_color, "--", "Phase-model prediction"),
                                      ("no_anchor", ".6", ":", "No-anchor prediction")):
        axes[1].plot(time, trajectories[name], style, lw=2, color=color, label=label)
    axes[1].set(xlabel="Time after cue returns (s)", ylabel="Mean direction error (°)",
                xlim=(0, 5), ylim=(0, max(35, np.max(trajectories["no_anchor"])) + 2),
                title="It predicts recovery at new places")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.out.with_suffix("." + extension), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
