"""Complete-cohort figures/statistics for the explicitly designed ring."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

from train_designed_predictive_ring import DEFAULT, protocol, source_manifest, sha, VARIANTS


def statistics(values):
    values = np.asarray(values, float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("All eight finite model-seed contrasts are required")
    mean = values.mean()
    null = (np.array(list(itertools.product((-1., 1.), repeat=8)))*values).mean(1)
    half = student_t.ppf(.975, 7)*values.std(ddof=1)/np.sqrt(8)
    return {"mean": float(mean), "ci95": [float(mean-half), float(mean+half)],
            "exact_p_two_sided": float(np.mean(np.abs(null) >= abs(mean)-1e-14)),
            "positive_seeds": int((values > 0).sum()), "seed_values": values.tolist()}


def load(directory):
    frozen = json.loads((directory/"frozen_protocol.json").read_text())
    if frozen["sources"] != source_manifest() or frozen["protocol"] != protocol():
        raise ValueError("Current sources differ from the frozen constructive protocol")
    results = []
    for seed in range(4001, 4009):
        d = json.loads((directory/f"seed{seed}/summary.json").read_text())
        if d["seed"] != seed or d["sources"] != frozen["sources"] or d["protocol"] != frozen["protocol"]:
            raise ValueError("Seed/source/protocol mismatch")
        with np.load(directory/f"seed{seed}/arrays.npz", allow_pickle=False) as a:
            phase = a["phase_error"]
            if not np.array_equal(phase[:40, 3], phase[:40, 6]):
                raise ValueError("Ordinary and restored histories differ before restoration")
            grid_gain = next(x["selected_gain_per_second"] for x in d["training"] if x["condition"] == "grid")
            if grid_gain == 0 and not np.array_equal(phase[:, 0], phase[:, 4]):
                raise ValueError("Zero-gain and blank-cue trajectories should coincide exactly")
        results.append(d)
    return results


def summarize(directory):
    data = load(directory)
    contrasts = []
    for name in protocol()["primary_family"]:
        a, b = name.split("-")
        values = [d["navigation"][a]["success_fraction"]-d["navigation"][b]["success_fraction"] for d in data]
        contrasts.append({"comparison": name, **statistics(values)})
    ps = np.array([x["exact_p_two_sided"] for x in contrasts])
    order = np.argsort(ps)
    adjusted = np.maximum.accumulate(ps[order]*(4-np.arange(4)))
    for i, p in zip(order, adjusted):
        contrasts[i]["holm_p_four_primary"] = float(min(p, 1.))
    navigation = {}
    for scenario in ("navigation", "navigation_nominal"):
        navigation[scenario] = {}
        for name, _, _ in VARIANTS:
            navigation[scenario][name] = {k: float(np.mean([d[scenario][name][k] for d in data])) for k in
                ("success_fraction", "heading_mae_degrees", "mean_capped_arrival_seconds",
                 "mean_final_goal_distance_m", "mean_self_position_error_m", "terminal_goal_occupancy_fraction")}
    result = {"seeds": list(range(4001, 4009)), "frozen_protocol_sha256": sha(directory/"frozen_protocol.json"),
              "design_boundary": "ring geometry, phase cancellation, geometry inverse and correction law designed; spatial phase readouts and gain learned",
              "primary": contrasts, "averages": navigation,
              "gains": {kind: [next(x["selected_gain_per_second"] for x in d["training"] if x["condition"] == kind) for d in data]
                        for kind in ("grid", "predictive", "grid_heading")},
              "max_blank_state_drift": max(d["dynamics"]["max_full_state_drift_over10s_blank"] for d in data),
              "max_spectrum_prediction_error": max(max(d["dynamics"]["max_spectral_prediction_error"].values()) for d in data),
              "phase_error_after5s_degrees": {k: float(np.mean([d["dynamics"]["phase_error_after5s_degrees"][k] for d in data])) for k in ("blank", "native", "ordinary", "reverse")},
              "total_evaluation_seconds": sum(d["elapsed_seconds"] for d in data),
              "artifacts": {str(p.relative_to(directory)): sha(p) for d in data for p in
                  [directory/f"seed{d['seed']}/summary.json", directory/f"seed{d['seed']}/model.pt"]}}
    (directory/"aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    plot(directory, data)
    print(json.dumps(result, indent=2))
    return result


def plot(directory, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = {"grid": "#2878B5", "pgc": "#E66A32", "explicit": "#338569"}
    arrays = [np.load(directory/f"seed{d['seed']}/arrays.npz", allow_pickle=False) for d in data]
    fig, ax = plt.subplots(1, 3, figsize=(11.8, 3.4))
    # All64 eigenvalues are shown;63 normal modes coincide at rho by design.
    for cue, color, label in (("blank", "#747b85", "Cue off"), ("actual_decoded_cue", "#E66A32", "PGC cue on")):
        values = []
        for d in data:
            spectrum = d["dynamics"]["spectra"][cue]
            measured = np.asarray(spectrum["real"])
            phase_index = int(np.argmin(np.abs(measured-spectrum["predicted"][0])))
            values.append(np.r_[measured[phase_index], np.delete(measured, phase_index)])
        values = np.asarray(values)
        ax[0].scatter(np.arange(64), values.mean(0), s=17, color=color, label=label)
    ax[0].axhline(1, color="black", lw=.7, ls=":")
    ax[0].set(xticks=[0, 33], xticklabels=["Phase", "Other 63 directions"], ylim=(0, 1.08),
              ylabel="Perturbation multiplier", title="Mean 64D ring spectrum")
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    for condition, color, label in (("ordinary", "#747b85", "Ordinary replacement"), ("native", "#E66A32", "Predictive cue")):
        values = np.array([np.abs(a["phase_"+condition]).mean(1)*180/np.pi for a in arrays])
        values = np.column_stack((np.full(8, 10.), values))
        ax[1].plot(np.arange(101)*.05, values.mean(0), color=color, lw=2, label=label)
    ax[1].set(xlim=(0, 1), ylim=(-.3, 11), xlabel="Time (s)", ylabel="Phase error (°)", title="Noiseless spatial-rate assay")
    ax[1].legend(frameon=False, fontsize=9)
    for i, (name, label) in enumerate((("grid", "Grid"), ("pgc", "PGC"), ("explicit", "Explicit\ndirection"))):
        values = np.array([d["navigation"][name]["success_fraction"]*100 for d in data])
        half = student_t.ppf(.975, 7)*values.std(ddof=1)/np.sqrt(8)
        ax[2].bar(i, values.mean(), color=colors[name], width=.62)
        ax[2].errorbar(i, values.mean(), yerr=half, color="black", capsize=3, fmt="none")
        ax[2].scatter(i+np.linspace(-.13, .13, 8), values, color="black", s=10, alpha=.65)
    ax[2].set(xticks=[0, 1, 2], xticklabels=["Grid", "PGC", "Explicit\ndirection"], ylim=(0, 105),
              ylabel="Reached goal (%)", title="Held-out navigation")
    fig.suptitle("Designed ring · learned spatial readouts and cue gain", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory/f"designed_ring_mechanism.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.7))
    variants = [x[0] for x in VARIANTS]
    for name, color, label in (("pgc", "#E66A32", "Predictive"), ("pgc_ordinary", "#747b85", "Ordinary replacement"), ("pgc_restored", "#338569", "Restored at 2 s")):
        index = variants.index(name)
        values = np.array([np.abs(a["phase_error"][:, index]).mean(1)*180/np.pi for a in arrays])
        ax[0].plot(np.arange(400)*.05, values.mean(0), color=color, label=label, lw=1.8)
    ax[0].axvspan(4, 7, color="#cdd1d7", alpha=.3)
    ax[0].set(xlabel="Time (s)", ylabel="Heading error (°)", title="Shared 4–7 s cue blackout", xlim=(0, 20))
    ax[0].legend(frameon=False, fontsize=9)
    names = ["pgc", "pgc_ordinary", "pgc_blank", "pgc_reverse", "pgc_restored"]
    for i, name in enumerate(names):
        values = np.array([d["navigation"][name]["success_fraction"]*100 for d in data])
        ax[1].bar(i, values.mean(), color=["#E66A32", "#747b85", "#a6abb3", "#a95356", "#338569"][i], width=.65)
        ax[1].scatter(i+np.linspace(-.13, .13, 8), values, color="black", s=10)
    ax[1].set(xticks=np.arange(5), xticklabels=["Native", "Ordinary", "Blank", "Reverse", "Restored"],
              ylabel="Reached goal (%)", ylim=(0, 105), title="Frozen PGC-trained interface")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory/f"designed_ring_interventions.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    for a in arrays:
        a.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    summarize(parser.parse_args().out)
