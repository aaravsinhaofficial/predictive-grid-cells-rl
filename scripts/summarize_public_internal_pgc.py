"""Complete-cohort aggregation of the frozen five-test internal-PGC protocol."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/"results/ring_attractor/public_internal_pgc"
CONTEXTS = ["stationary_zero_input", "moving_physical_displacements"]
MODES = ["zero_output", "mean_replacement"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_summary(values):
    values = np.asarray(values, float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Exactly ten finite seed-level contrasts are required")
    mean = values.mean()
    sem = values.std(ddof=1)/np.sqrt(10)
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=10)))
    null = (signs*values).mean(1)
    p = np.mean(np.abs(null) >= abs(mean)-1e-14)
    half = student_t.ppf(.975, 9)*sem
    return {"mean": float(mean), "ci95": [float(mean-half), float(mean+half)],
            "exact_p_two_sided": float(p), "positive_seeds": int((values > 0).sum()),
            "negative_seeds": int((values < 0).sum()), "seed_values": values.tolist()}


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adjusted = np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps))))
    result = np.empty_like(ps)
    result[order] = np.minimum(adjusted, 1.)
    return result


def select_value(summary, context, mode, dose, steps, metric, group):
    rows = [r for r in summary["rows"] if r["context"] == context and r["mode"] == mode and r["dose"] == dose and r["steps"] == steps]
    if group == "controls":
        selected = [r for r in rows if r["group"].startswith("matched_nonpgc_")]
        if len(selected) != 20 or len({r["group"] for r in selected}) != 20:
            raise ValueError("Missing/duplicate matched-control rows")
        return float(np.mean([r[metric] for r in selected]))
    selected = [r for r in rows if r["group"] == group]
    if len(selected) != 1:
        raise ValueError("Missing/duplicate population row")
    return selected[0][metric]


def load_cohort(directory):
    frozen_path = directory/"frozen_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    for path, expected in frozen["sources"].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f"Frozen source/protocol changed: {path}")
    protocol_path = ROOT/"docs/research/public_internal_pgc_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assets = json.loads((ROOT/"docs/research/public_internal_pgc_sources.json").read_text())
    summaries = []
    if protocol["model_seeds"] != list(range(10)):
        raise ValueError("Unexpected cohort")
    for seed in range(10):
        d = json.loads((directory/f"seed{seed:02d}/summary.json").read_text())
        if d["seed"] != seed or d["protocol_sha256"] != sha(protocol_path) or d["frozen_manifest_sha256"] != sha(frozen_path):
            raise ValueError(f"Seed {seed} result provenance mismatch")
        expected = [x for x in assets["files"] if f"/Seed {seed} weight decay" in x["path"] and x["path"].endswith("/final_model.pth")]
        if len(expected) != 1 or d["checkpoint_sha256"] != expected[0]["sha256"]:
            raise ValueError("Result/checkpoint mismatch")
        expected_class = [x for x in assets["files"] if f"/Seed {seed} weight decay" in x["path"] and x["path"].endswith("/spatial shift/predictive_retrospective/predictive_ids_random_walk.npy")]
        if len(expected_class) != 1 or d["classification_sha256"] != expected_class[0]["sha256"]:
            raise ValueError("Result/classification mismatch")
        summaries.append(d)
    return protocol, summaries


def summarize(directory):
    protocol, data = load_cohort(directory)
    contrasts = []
    primary_differences = {}
    for context in CONTEXTS:
        for mode in MODES:
            pg = [select_value(d, context, mode, .25, 20, "position_error_m", "pgc") for d in data]
            control = [select_value(d, context, mode, .25, 20, "position_error_m", "controls") for d in data]
            values = np.asarray(pg)-control
            primary_differences[context, mode] = values
            contrasts.append({"comparison": context+" "+mode, "pgc_mean_m": float(np.mean(pg)),
                              "control_mean_m": float(np.mean(control)), **paired_summary(values)})
    interaction = primary_differences[CONTEXTS[1], MODES[1]]-primary_differences[CONTEXTS[0], MODES[1]]
    contrasts.append({"comparison": "mean_replacement moving-minus-stationary interaction", **paired_summary(interaction)})
    adjusted = holm([c["exact_p_two_sided"] for c in contrasts])
    for c, p in zip(contrasts, adjusted):
        c["holm_p_five_primary"] = float(p)
    descriptive = []
    for context in CONTEXTS:
        for mode in MODES:
            for dose in (.1, .25, .5):
                for steps in (20, 50, 100, 200):
                    metrics = ["position_error_m", "final_position_error_m", "projective_position_spread_ratio", "median_norm_ratio"]
                    metrics += ["increment_gain", "increment_alignment"] if context == CONTEXTS[1] else ["mean_projective_perturbation_gain", "mean_full_state_perturbation_gain"]
                    for metric in metrics:
                        pg = [select_value(d, context, mode, dose, steps, metric, "pgc") for d in data]
                        control = [select_value(d, context, mode, dose, steps, metric, "controls") for d in data]
                        descriptive.append({"context": context, "mode": mode, "dose": dose, "steps": steps, "metric": metric,
                                            "pgc_mean": float(np.mean(pg)), "control_mean": float(np.mean(control)),
                                            "pgc_seeds": pg, "control_seeds": control,
                                            "difference_seeds": (np.asarray(pg)-control).tolist()})
    balance = [{"seed": d["seed"], **{key: [float(min(x[key] for x in d["matching_balance"])), float(max(x[key] for x in d["matching_balance"]))]
                for key in ("mean_activity_ratio", "rms_activity_ratio", "outgoing_recurrent_column_norm_ratio")}} for d in data]
    result = {"protocol_sha256": sha(ROOT/"docs/research/public_internal_pgc_protocol.json"),
              "frozen_manifest_sha256": sha(directory/"frozen_manifest.json"),
              "seeds": list(range(10)), "predictive_counts": [d["predictive_count"] for d in data],
              "primary": contrasts, "descriptive": descriptive, "matching_balance": balance,
              "validations": [d["validation"] for d in data],
              "total_evaluation_seconds": sum(d["elapsed_seconds"] for d in data)}
    (directory/"aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    plot(directory, data)
    print(json.dumps({"counts": result["predictive_counts"], "primary": contrasts, "seconds": result["total_evaluation_seconds"]}, indent=2))
    return result


def plot(directory, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    def curves(ax, context, mode, steps, metric, scale=1):
        for group, color, label in (("controls", "#747b85", "Matched cells"), ("pgc", "#d97127", "Predictive cells")):
            values = []
            for dose in (0., .1, .25, .5):
                values.append([select_value(d, context, "native" if dose == 0 else mode, dose, steps, metric, "native" if dose == 0 else group)*scale for d in data])
            a = np.asarray(values)
            mean = a.mean(1)
            half = student_t.ppf(.975, 9)*a.std(1, ddof=1)/np.sqrt(10)
            ax.plot([0, .1, .25, .5], mean, color=color, marker="o", lw=2, label=label)
            ax.fill_between([0, .1, .25, .5], mean-half, mean+half, color=color, alpha=.12, linewidth=0)
        ax.set_xticks([0, .1, .25, .5], ["0", "10", "25", "50"])
        ax.grid(axis="y", alpha=.16)
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.2), sharex=True, sharey="row")
    for i, context in enumerate(CONTEXTS):
        for j, mode in enumerate(MODES):
            curves(axes[i, j], context, mode, 20, "position_error_m", 100)
            if i == 0:
                axes[i, j].set_title(["Weaken recurrent output", "Keep mean; remove variation"][j])
            if j == 0:
                axes[i, j].set_ylabel(["Stationary memory\nPosition error (cm)", "Moving animal\nPosition error (cm)"][i])
            if i == 1:
                axes[i, j].set_xlabel("Intervention strength (%)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("First 0.4 seconds · 10 published networks", y=.99)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory/f"internal_pgc_causal_replication.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.1), sharex=True, sharey="col")
    metrics = ["projective_position_spread_ratio", "median_norm_ratio", "mean_projective_perturbation_gain"]
    titles = ["Position-pattern spread", "Activity amplitude", "Perturbation remaining"]
    for i, mode in enumerate(MODES):
        for j, metric in enumerate(metrics):
            curves(axes[i, j], CONTEXTS[0], mode, 200, metric)
            if i == 0:
                axes[i, j].set_title(titles[j])
            if j == 0:
                axes[i, j].set_ylabel(["Weaken output", "Keep mean"][i])
            if i == 1:
                axes[i, j].set_xlabel("Intervention strength (%)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Stationary memory after 4 seconds · relative to initial state", y=.99)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory/f"internal_pgc_memory_geometry.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    summarize(parser.parse_args().out)
