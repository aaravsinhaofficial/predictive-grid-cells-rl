"""Explicitly exploratory full-state loop example; original branch case retained."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ring_dynamics import wrap


def main():
    out = Path("results/ring_heading/geometry_audit")
    report = json.loads((out / "dense_ring_predictive_s2002.json").read_text())
    arrays = np.load(out / "dense_ring_predictive_s2002.npz")
    state = arrays["curve"]
    centered = state - state.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T
    variance = float(np.sum(singular[:3] ** 2) / np.sum(singular ** 2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(11, 3.7), layout="constrained")
    first = fig.add_subplot(131, projection="3d")
    first.plot(*projected.T, color="#E66A32", lw=1.8)
    for index, label in zip((0, len(state)//4, len(state)//2, 3*len(state)//4), ("0°", "90°", "180°", "270°")):
        first.scatter(*projected[index], color="#E66A32", s=12)
        first.text(*projected[index], label, fontsize=8)
    first.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title="A  A smooth full-state loop")
    first.set_xticks([]); first.set_yticks([]); first.set_zticks([])
    first.text2D(.01, .55, "PC3", transform=first.transAxes, rotation=90, fontsize=10)
    first.set_box_aspect((1, 1, .7))
    first.view_init(elev=20, azim=35)
    second = fig.add_subplot(132)
    times = [0, 1, 5]
    normal = [report["perturbations"][str(step)]["normal_remaining_mean_distance_ratio"] for step in (0, 20, 100)]
    phase = [report["perturbations"][str(step)]["tangent_phase_gain_per_phase"]["median"] for step in (0, 20, 100)]
    second.plot(times, normal, "o:", color="#222222", lw=2, label="Across the curve")
    second.plot(times, phase, "o-", color="#E66A32", lw=2, label="Around the curve")
    second.set(xlabel="Time after disturbance (s)", ylabel="Disturbance remaining", ylim=(-.03, 1.1),
               yticks=[0, .5, 1], yticklabels=["0%", "50%", "100%"], title="B  Memory changes more slowly")
    second.legend(frameon=False, fontsize=8)
    third = fig.add_subplot(133)
    source = Path("results/ring_heading/mechanistic_confirmation/phase_reduction_predictive_s2002.npz")
    recovery = np.load(source)
    for variant, color, label in (("native", "#E66A32", "Predictive code"), ("ordinary", "#858585", "Ordinary replacement")):
        rows = []
        for context in (2, 3, 4):
            prefix = f"context{context}_recovery_"
            selected = np.abs(recovery[prefix + "nominal_initial_offsets_deg"]) == 10
            trace = recovery[prefix + variant + "_actual"][:, selected]
            cue = recovery[prefix + "cue"][selected]
            rows.append(np.abs(wrap(trace - cue)) * 180 / np.pi)
        error = np.concatenate(rows, 1).mean(1)
        third.plot(np.arange(len(error)) * .05, error, color=color, lw=2, label=label)
    third.set(xlabel="Time after cue returns (s)", ylabel="Direction error (°)", xlim=(0, 5), ylim=(0, 105),
              title="C  Prediction corrects memory")
    third.legend(frameon=False, fontsize=8, loc="upper left")
    fig.supxlabel("Exploratory example: seed 2002, first subsequent candidate after the seed 2001 branch audit • All cohort models retained", fontsize=8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"ring_example_s2002.{extension}", dpi=220)
    plt.close(fig)
    (out / "ring_example_s2002_metadata.json").write_text(json.dumps({
        "selection": "exploratory first subsequent candidate after the prespecified firstseed2001 branch finding",
        "seed2001_report_retained": "dense_predictive_s2001.json", "state_curve": "4096initialangles,full64dimensionalstate after4cue+40blanksteps",
        "display_projection": "first3PCs, visual only", "displayed_variance_fraction": variance,
        "perturbations": "10%radius, exact angular tangent, dense full-state normal distances; normal mean versus phase median",
        "phase_anchor_source": str(source), "phase_anchor_selection": "all +/-10degree memory substitutions, all3heldoutcontexts, actualRNNtrajectories",
        "scope": "one audited model; not a claim that every training seed has the same topology"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
