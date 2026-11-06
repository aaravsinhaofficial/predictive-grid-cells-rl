"""Simple seed-averaged figures for the frozen phase and navigation pulse analyses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from evaluate_heading_phase_reduction import basis
from ring_dynamics import wrap


def interval(values):
    values = np.asarray(values)
    mean = values.mean(0)
    width = t.ppf(.975, len(values) - 1) * values.std(0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
    return mean, mean - width, mean + width


def line(axis, x, values, color, label, style="-", band=True):
    mean, low, high = interval(values)
    if band:
        axis.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
    axis.plot(x, mean, style, color=color, lw=2, label=label)


def save(fig, root, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / f"{stem}.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/mechanistic_confirmation"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    seeds = [seed for seed in range(2001, 2017)
             if (args.root / f"phase_reduction_predictive_s{seed}.json").exists()
             and (args.root / f"memory_pulse_predictive_s{seed}.json").exists()]
    if len(seeds) != 16 and not args.allow_partial:
        raise ValueError(f"Wait for all16predeclared seeds; currently{len(seeds)}")
    if not seeds:
        raise ValueError("No paired results")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    grid_color, pgc_color, ordinary_color, normal_color = "#2878B5", "#E66A32", "#858585", "#222222"
    field_values, fitted_k = [], []
    recovery = {key: [] for key in ("actual", "sine", "no_anchor")}
    pulse_heading = {key: [] for key in ("native", "ordinary", "native_baseline", "ordinary_baseline")}
    pulse_path = {key: [] for key in ("native", "ordinary", "normal_native")}
    bins = np.linspace(-180, 180, 25)
    centers = (bins[1:] + bins[:-1]) / 2
    for seed in seeds:
        prefix = args.root / f"phase_reduction_predictive_s{seed}"
        report = json.loads(prefix.with_suffix(".json").read_text())
        with np.load(prefix.with_suffix(".npz")) as arrays:
            phi = np.concatenate([arrays[f"context{i}_field_phi"] for i in (2, 3, 4)])
            cue = np.concatenate([arrays[f"context{i}_field_cue"] for i in (2, 3, 4)])
            actual = np.concatenate([arrays[f"context{i}_field_native"] for i in (2, 3, 4)])
            error = wrap(phi - cue) * 180 / np.pi
            correction = (actual - basis(phi) @ np.array(report["coefficients"]["f0_coefficients"])) * 180 / np.pi
            field_values.append([correction[(error >= lo) & (error < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])])
            fitted_k.append(report["coefficients"]["k_per_second"])
            for key in recovery:
                rows = []
                for context in (2, 3, 4):
                    key_prefix = f"context{context}_recovery_"
                    cue = arrays[key_prefix + "cue"]
                    offset = arrays[key_prefix + "nominal_initial_offsets_deg"]
                    mask = (np.abs(offset) > 0) & (np.abs(offset) <= 45)
                    state = arrays[key_prefix + "native_" + key]
                    rows.append(np.abs(wrap(state[:, mask] - cue[mask])) * 180 / np.pi)
                recovery[key].append(np.concatenate(rows, 1).mean(1))
        prefix = args.root / f"memory_pulse_predictive_s{seed}"
        pulse = json.loads(prefix.with_suffix(".json").read_text())
        with np.load(prefix.with_suffix(".npz")) as arrays:
            for key in pulse_heading:
                variant = "ordinary" if key.startswith("ordinary") else "native"
                state = "baseline" if key.endswith("baseline") else "phase_balanced20"
                v = pulse["protocol"]["variant_order"].index(variant)
                rows = []
                for repeat in range(len(pulse["evaluation_seeds"])):
                    key_prefix = f"repeat{repeat}_{state}_trace_"
                    rows.append(np.abs(wrap(arrays[key_prefix + "decoded_heading"][:, v]
                                           - arrays[key_prefix + "heading"][:, v])) * 180 / np.pi)
                pulse_heading[key].append(np.concatenate(rows, 1).mean(1))
            for key in pulse_path:
                state = "normal_matched20" if key == "normal_native" else "phase_balanced20"
                v = pulse["protocol"]["variant_order"].index("ordinary" if key == "ordinary" else "native")
                rows = []
                for repeat in range(len(pulse["evaluation_seeds"])):
                    position = arrays[f"repeat{repeat}_{state}_trace_position"][:, v]
                    baseline = arrays[f"repeat{repeat}_baseline_trace_position"][:, v]
                    rows.append(np.linalg.norm(position - baseline, axis=-1) * 100)
                pulse_path[key].append(np.concatenate(rows, 1).mean(1))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    line(axes[0], centers, field_values, pgc_color, "Trained RNN", "o-")
    fine = np.linspace(-180, 180, 361)
    fitted = -np.array(fitted_k)[:, None] * np.sin(np.deg2rad(fine)) * 180 / np.pi
    line(axes[0], fine, fitted, grid_color, "One sine term", "--", band=False)
    axes[0].axhline(0, color=".7", lw=.7)
    axes[0].set(xlabel="Memory error from cue (°)", ylabel="Cue correction (°/s)",
                xticks=[-180, -90, 0, 90, 180], xlim=(-180, 180), title="The cue pushes memory back")
    axes[0].legend(frameon=False, fontsize=9)
    time = np.arange(len(recovery["actual"][0])) * report["config"]["dt"]
    for key, color, style, label in (("actual", pgc_color, "-", "Trained RNN"),
                                    ("sine", grid_color, "--", "Phase-model prediction"),
                                    ("no_anchor", ordinary_color, ":", "No-anchor prediction")):
        line(axes[1], time, recovery[key], color, label, style)
    axes[1].set(xlabel="Time after cue returns (s)", ylabel="Mean direction error (°)",
                xlim=(0, 5), ylim=(0, 40), title="It predicts recovery at new places")
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, args.root, "phase_reduction")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    time = np.arange(len(pulse_heading["native"][0])) * pulse["protocol"]["trace_interval_seconds"]
    for key, color, label in (("native", pgc_color, "PGC cue present"), ("ordinary", ordinary_color, "PGC cue removed")):
        line(axes[0], time, pulse_heading[key], color, label)
        line(axes[0], time, pulse_heading[key + "_baseline"], color, None, ":", band=False)
        line(axes[1], time, pulse_path[key], color, label)
    axes[0].plot([], [], ":", color=normal_color, lw=2, label="Unperturbed memory")
    line(axes[1], time, pulse_path["normal_native"], normal_color, "Equal-size normal pulse", ":")
    axes[0].set(xlabel="Time after navigation starts (s)", ylabel="Mean direction error (°)",
                xlim=(0, 2), ylim=(0, 100), title="PGCs correct a memory error")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    maximum = max(interval(value)[2][time <= 5].max() for value in pulse_path.values())
    axes[1].set(xlabel="Time after navigation starts (s)", ylabel="Change from unperturbed path (cm)",
                xlim=(0, 5), ylim=(0, maximum * 1.12), title="Correction limits the path change")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, args.root, "memory_pulse")
    (args.root / "figure_metadata.json").write_text(json.dumps({"complete": len(seeds) == 16,
        "training_seeds": seeds, "bands": "unadjusted95%tintervals across independent training seeds",
        "phase_field": "heldoutpositions/angles;15degree bins; independently fitted intrinsic drift subtracted",
        "recovery": "heldoutpositions;initialmemory substitutions+/-10and+/-45degrees; equal weighting of training seeds",
        "pulse": "128pairedtrials per training seed;balanced+/-20degree conditional memory-state substitutions",
        "pulse_geometry_caution": "full-state substitution can cross a sharp transition; not universally a pure phase-coordinate change",
        "no_anchor": "prediction from fitted blank drift; not an actual sensory intervention"}, indent=2) + "\n")
    print(json.dumps({"figures_written": ["phase_reduction", "memory_pulse"], "n_seeds": len(seeds)}))


if __name__ == "__main__":
    main()
