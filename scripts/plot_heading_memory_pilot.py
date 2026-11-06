"""Simple, data-based illustration of memory and predictive phase correction."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path("results/ring_heading/encoder_pilot")
    memory = json.loads((folder/"memory_predictive_s1006_final.json").read_text())
    arrays = np.load(folder/"memory_predictive_s1006_final.npz")
    anchor = json.loads((folder/"predictive_s1006_6000_anchor.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.5), constrained_layout=True)
    blue, orange, gray, purple = "#2878B5", "#E66A32", "#777777", "#8054A1"
    ax = axs[0, 0]
    curve = arrays["trained_context0_base"][0]
    centered = curve-curve.mean(0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    embedding = centered@vectors[:2].T
    ax.plot(*np.vstack((embedding, embedding[0])).T, color="#cccccc", lw=1)
    ax.scatter(*embedding.T, c=arrays["phase"], cmap="hsv", s=14, zorder=3)
    ax.set(xlabel="Activity PC 1", ylabel="Activity PC 2", title="A  Learned direction-memory ring")
    ax.set_aspect("equal", adjustable="datalim")
    ax.text(.03, .03, "Input off for 2 seconds", transform=ax.transAxes, fontsize=9, color="#555555")
    ax = axs[0, 1]
    rows = [r for r in memory["models"]["trained"][0]["perturbations"]["0.1"] if r["step"] <= 100]
    t = [r["seconds_after_settling"] for r in rows]
    normal0 = rows[0]["normal_distance_radius"]
    ax.plot(t, [r["normal_distance_radius"]/normal0 for r in rows], color=blue, lw=2.4, label="Across the ring")
    ax.plot(t, [r["decoded_tangent_retention"]["median"] for r in rows], color=orange, lw=2.4, label="Around the ring")
    ax.set(xlabel="Seconds without input", ylabel="Disturbance remaining", ylim=(-.03, 1.08),
           title="B  Return across; remember around")
    ax.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1, 0]
    labels = {"native": "Predictive input", "blank": "Input off", "self_generated": "Self-generated preview"}
    for variant, color in (("native", orange), ("blank", gray), ("self_generated", purple)):
        ts = sorted({r["step"] for r in anchor["rows"] if r["step"] <= 100})
        values = []
        for step in ts:
            selected = [r for r in anchor["rows"] if r["context"] < 5 and r["variant"] == variant and r["step"] == step]
            values.append(np.mean([r["phase_local_gain_median"] for r in selected]))
        ax.plot(np.asarray(ts)*.05, values, color=color, lw=2.4, label=labels[variant])
    ax.set(xlabel="Seconds after phase displacement", ylabel="Phase disturbance remaining", ylim=(-.03, 1.3),
           title="C  Phase disturbances shrink")
    ax.set_yticks([0, .5, 1], ["0%", "50%", "100%"])
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1, 1]
    variants = ["native", "ordinary", "reverse", "self_generated", "restore_20"]
    means = [np.mean([r["baseline_mae_deg"] for r in anchor["rows"]
                      if r["context"] < 5 and r["step"] == 100 and r["variant"] == v]) for v in variants]
    colors = [orange, gray, purple, purple, orange]
    bars = ax.bar(range(5), means, color=colors, width=.62)
    ax.set_xticks(range(5), ["Native", "Removed", "Reversed", "Self\ngenerated", "Restored"], fontsize=9)
    ax.set(ylabel="Heading error (degrees)", ylim=(0, 200), title="D  Correct input selects correct direction")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, value+4, f"{value:.0f}°", ha="center", fontsize=10)
    for ax in axs.flat:
        ax.grid(axis="y", color="#e9e9e9", lw=.7)
        ax.set_axisbelow(True)
    fig.suptitle("How predictive input corrects a learned memory ring", fontsize=15)
    fig.supxlabel("Exploratory model, seed 1006 • Full-state dynamics; PCA is a display only", fontsize=9, color="#555555")
    out = Path("results/ring_heading/figures")
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out/("memory_mechanism_pilot."+extension))
    metadata = {"training_seed": 1006, "stage": "exploratory", "activity_pc2_fraction": float(np.sum(singular[:2]**2)/np.sum(singular**2)),
                "sources": [str(folder/"memory_predictive_s1006_final.json"), str(folder/"predictive_s1006_6000_anchor.json")],
                "note": "Panel B central context; panels C/D mean across five prescribed in-region positions. Self-generated preview follows each replica's own current decoded heading."}
    (out/"memory_mechanism_pilot.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
