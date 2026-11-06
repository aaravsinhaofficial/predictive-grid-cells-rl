"""Paired-seed analysis of the engineered sensorimotor-ring experiment."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate(values, confidence=.95):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    half = float(t.ppf((1+confidence)/2, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean, ci=[mean-half, mean+half], n=len(x))


def paired(a, b):
    diff = np.asarray(b)-np.asarray(a)
    flips = np.asarray(list(itertools.product((-1, 1), repeat=len(diff))))
    observed = abs(diff.mean())
    probability = float(np.mean(np.abs((flips*diff).mean(1)) >= observed-1e-14))
    return dict(**estimate(diff), exact_p=probability, paired_differences=diff.tolist())


def holm(comparisons):
    keys = sorted(comparisons, key=lambda key: comparisons[key]["exact_p"])
    highest = 0.
    for index, key in enumerate(keys):
        highest = max(highest, min(1., comparisons[key]["exact_p"]*(len(keys)-index)))
        comparisons[key]["holm_exact_p"] = highest


def summarize(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    runs = [json.loads((folder/f"seed_{seed}.json").read_text()) for seed in range(2101, 2109)]
    conditions = ("grid", "predictive", "velocity", "removed", "restored", "reversed")
    def branch(run, condition):
        return run["models"][condition] if condition in run["models"] else run["interventions"][condition]
    metrics = {c: {key: np.array([branch(r, c)["metrics"][key] for r in runs])
                    for key in branch(runs[0], c)["metrics"]} for c in conditions}
    primary = {
        "predictive_minus_grid": paired(metrics["grid"]["objective"], metrics["predictive"]["objective"]),
        "removal_minus_predictive": paired(metrics["predictive"]["objective"], metrics["removed"]["objective"]),
        "reverse_minus_predictive": paired(metrics["predictive"]["objective"], metrics["reversed"]["objective"])}
    holm(primary)
    equivalence = estimate(metrics["predictive"]["objective"]-metrics["velocity"]["objective"], .90)
    equivalence.update(margin=.001, equivalent=bool(equivalence["ci"][0]>-.001 and equivalence["ci"][1]<.001))
    secondary = {key: paired(metrics["grid"][key], metrics["predictive"][key])
                 for key in metrics["grid"] if key != "objective"}
    result = dict(seeds=[r["seed"] for r in runs],
                  means={c: {key: estimate(values) for key, values in metrics[c].items()} for c in conditions},
                  primary=primary, explicit_velocity_equivalence=equivalence,
                  secondary_uncorrected=secondary,
                  percent_objective_reduction=estimate(100*(1-metrics["predictive"]["objective"]/metrics["grid"]["objective"])),
                  decoder_max_compilation_error=max(r[k]["compilation_error"] for r in runs for k in ("decoder_ordinary", "decoder_forecast")),
                  full_jacobian_max_spectrum_error=max(m["full_jacobian_validation"]["full_state_jacobian_spectrum_error"]
                                                      for r in runs for m in r["models"].values()),
                  exact_restoration=all(r["models"]["predictive"]["metrics"] == r["interventions"]["restored"]["metrics"] for r in runs),
                  interpretation="Prediction supports a faster stable high-gain sensorimotor controller. Intrinsic neural goal-memory dynamics are identical.")
    (folder/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    colors = dict(grid="#2c7bb6", predictive="#e76f36", velocity="#47926c", removed="#8664a7", restored="#e76f36", reversed="#777777")
    labels = dict(grid="Grid", predictive="PGC", velocity="Grid +\nvelocity", removed="Forecast\nremoved", restored="Restored", reversed="Reversed")
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(folder/f"{name}.{ext}", dpi=180)
        plt.close(fig)
    def paired_plot(ax, keys, field, title, ylabel):
        values = np.stack([metrics[c][field] for c in keys], axis=1)
        for row in values:
            ax.plot(range(len(keys)), row, color="#cccccc", alpha=.7, lw=.8, zorder=1)
        for index, key in enumerate(keys):
            ax.scatter(np.full(8, index), values[:, index], color=colors[key], s=21, zorder=3)
            ax.plot([index-.16, index+.16], [values[:, index].mean()]*2, color="#222222", lw=2, zorder=4)
        ax.set(xticks=range(len(keys)), xticklabels=[labels[c] for c in keys], title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.15)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    paired_plot(axes[0], ("grid", "predictive", "velocity"), "first_entry_seconds", "Earlier arrival", "First goal entry (seconds)")
    paired_plot(axes[1], ("grid", "predictive", "velocity"), "max_overshoot_deg", "Less overshoot", "Maximum overshoot (degrees)")
    paired_plot(axes[2], ("grid", "predictive", "velocity"), "settling_seconds", "Settling depends on the learned gain", "Permanent settling (seconds)")
    fig.supxlabel("Engineered agent · independently trained controllers · 8 paired seeds · 512 held-out episodes/seed", fontsize=9)
    save(fig, "trained_navigation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    paired_plot(axes[0], ("predictive", "removed", "restored", "reversed"), "objective", "Forecast removal changes behavior", "Navigation error + effort")
    keys = ("predictive", "removed", "restored", "reversed")
    values = np.array([[branch(r, c)["spectrum"]["spectral_radius"] for c in keys] for r in runs])
    for row in values:
        axes[1].plot(range(4), row, color="#cccccc", alpha=.7, lw=.8)
    for i, c in enumerate(keys):
        axes[1].scatter([i]*8, values[:, i], color=colors[c], s=21)
    axes[1].axhline(1, color="#555555", ls="--", lw=1)
    axes[1].set(xticks=range(4), xticklabels=[labels[c] for c in keys],
                ylabel="Transverse multiplier per step", title="Prediction changes loop stability")
    axes[1].grid(axis="y", alpha=.15)
    fig.supxlabel("Same trained PGC controller · only the forecast code changes · goal-memory ring unchanged", fontsize=9)
    save(fig, "forecast_ablation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for c in ("grid", "predictive", "removed"):
        rows = [[v for v in r["delay_sweep"] if v["condition"] == c] for r in runs]
        delay = [v["delay"] for v in rows[0]]
        rho = np.array([[v["spectral_radius"] for v in row] for row in rows])
        axes[0].plot(delay, rho.mean(0), color=colors[c], lw=2.3, label=labels[c].replace("\n", " "))
        axes[0].fill_between(delay, rho.min(0), rho.max(0), color=colors[c], alpha=.10)
        behavior = [[v for v in r["behavior_delay_sweep"] if v["condition"] == c] for r in runs]
        d = [v["delay"] for v in behavior[0]]
        error = np.array([[v["metrics"]["objective"] for v in row] for row in behavior])
        axes[1].plot(d, error.mean(0), marker="o", ms=4, color=colors[c], lw=2.3)
    axes[0].axhline(1, color="#555555", lw=1, ls="--")
    axes[0].set(xlabel="Sensory delay (seconds)", ylabel="Transverse multiplier per step",
                title="Boundary predicted without refitting", ylim=(.90, 1.035))
    axes[1].set(xlabel="Sensory delay (seconds)", ylabel="Navigation error + effort", title="Test at new delays")
    axes[0].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.axvline(.3, color="#aaaaaa", ls=":", lw=1)
        ax.grid(alpha=.15)
    fig.supxlabel("Training delay: 0.3 s · every decoder and controller is frozen throughout the sweep", fontsize=9)
    save(fig, "delay_boundary")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    factor = [v["factor"] for v in runs[0]["horizon_sweep"]]
    rho = np.array([[v["spectral_radius"] for v in r["horizon_sweep"]] for r in runs])
    error = np.array([[v["metrics"]["objective"] for v in r["horizon_sweep"]] for r in runs])
    for ax, value in zip(axes, (rho, error)):
        ax.plot(factor, value.mean(0), color=colors["predictive"], marker="o", lw=2.3)
        ax.fill_between(factor, value.min(0), value.max(0), color=colors["predictive"], alpha=.12)
        ax.axvline(1, color="#aaaaaa", ls=":", lw=1)
        ax.set(xlabel="Forecast / trained forecast")
        ax.grid(alpha=.15)
    axes[0].axhline(1, color="#555555", ls="--", lw=1)
    axes[0].set(ylabel="Transverse multiplier per step", title="Too little or too much prediction")
    axes[1].set(ylabel="Navigation error + effort", title="Behavior follows the change")
    fig.supxlabel("Same trained controller · 0 removes prediction · 1 restores it · negative values reverse it", fontsize=9)
    save(fig, "forecast_horizon")
    print(json.dumps({k: result[k] for k in ("primary", "explicit_velocity_equivalence", "percent_objective_reduction",
                                            "decoder_max_compilation_error", "full_jacobian_max_spectrum_error")}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default="results/ring_attractor/sensorimotor_confirmatory")
    summarize(p.parse_args().folder)
