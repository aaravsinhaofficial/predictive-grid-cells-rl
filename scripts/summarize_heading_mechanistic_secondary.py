"""Descriptive seed-level replication summaries for the frozen secondary analyses.

Intervals use training seeds as the independent units. These secondary
intervals are unadjusted; this script does not add confirmatory p-values.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def summarize(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    half = float(t.ppf(.975, n - 1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return {"n_training_seeds": n, "mean": mean, "ci95": [mean - half, mean + half] if half is not None else None,
            "min": float(values.min()), "max": float(values.max()), "positive_seed_count": int((values > 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/mechanistic_confirmation"))
    parser.add_argument("--out", type=Path, default=Path("results/ring_heading/mechanistic_confirmation/summary.json"))
    args = parser.parse_args()
    rows = []
    for seed in range(2001, 2017):
        phase_path = args.root / f"phase_reduction_predictive_s{seed}.json"
        pulse_path = args.root / f"memory_pulse_predictive_s{seed}.json"
        if not phase_path.exists() or not pulse_path.exists():
            continue
        phase, pulse = json.loads(phase_path.read_text()), json.loads(pulse_path.read_text())
        assert phase["seed"] == pulse["training_seed"] == seed
        assert phase["steps_completed"] == pulse["steps_completed"] == 6000
        assert phase["checkpoint_sha256"] == pulse["checkpoint_sha256"]
        early, nav = pulse["early_heading_and_path"], pulse["navigation"]
        row = {"seed": seed, "k_per_second": phase["coefficients"]["k_per_second"],
            "heldout_native_field_variance_explained": phase["pooled_heldout_fields"]["native"]["sine"]["variance_explained"],
            "heldout_blank_field_variance_explained": phase["pooled_heldout_fields"]["blank"]["no_anchor"]["variance_explained"]}
        for variant in ("native", "reverse", "blank", "ordinary", "self_generated"):
            for reduction in ("sine", "no_anchor", "constant"):
                row[f"{variant}_{reduction}_recovery_prediction_mae_deg"] = float(np.mean([
                    context["variants"][variant][reduction]["trajectory_prediction_mae_deg"]
                    for context in phase["heldout_recovery"]]))
        row["native_recovery_mae_improvement_over_no_anchor_deg"] = row["native_no_anchor_recovery_prediction_mae_deg"] - row["native_sine_recovery_prediction_mae_deg"]
        for variant in ("native", "ordinary", "restored"):
            row[f"{variant}_phase_pulse_first5s_path_deviation_m"] = early["phase_balanced20"][variant]["first5s_path_deviation_m"]
            row[f"{variant}_phase_pulse_first2s_heading_mae_deg"] = early["phase_balanced20"][variant]["early_heading_mae_deg"]
            row[f"{variant}_phase_pulse_heading_mae_at1s_deg"] = early["phase_balanced20"][variant]["heading_error_at_1s_deg"]
            row[f"{variant}_phase_pulse_heading_mae_at2s_deg"] = early["phase_balanced20"][variant]["heading_error_at_2s_deg"]
            for state in ("baseline", "phase_balanced20", "normal_matched20"):
                for metric in ("success_fraction", "sustained_dwell_success_fraction", "mean_capped_arrival_seconds"):
                    row[f"{variant}_{state}_{metric}"] = nav[state]["40s"][variant][metric]
        row["phase_pulse_path_deviation_ordinary_minus_native_m"] = row["ordinary_phase_pulse_first5s_path_deviation_m"] - row["native_phase_pulse_first5s_path_deviation_m"]
        row["native_path_deviation_phase_minus_normal_m"] = early["phase_balanced20"]["native"]["first5s_path_deviation_m"] - early["normal_matched20"]["native"]["first5s_path_deviation_m"]
        row["native_arrival_change_phase_minus_baseline"] = nav["phase_balanced20"]["40s"]["native"]["success_fraction"] - nav["baseline"]["40s"]["native"]["success_fraction"]
        row["native_arrival_change_normal_minus_baseline"] = nav["normal_matched20"]["40s"]["native"]["success_fraction"] - nav["baseline"]["40s"]["native"]["success_fraction"]
        rows.append(row)
    if not rows:
        raise ValueError("No completed phase-field and memory-pulse checkpoint pairs")
    summary = {key: summarize([row[key] for row in rows]) for key in rows[0] if key != "seed"}
    report = {"complete": len(rows) == 16, "training_seeds": [row["seed"] for row in rows],
              "expected_training_seeds": list(range(2001, 2017)), "summary": summary, "per_seed": rows,
              "interpretation": "secondary descriptive replication; unadjusted95%tintervals across training seeds; no new primary tests"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "training_seeds": report["training_seeds"],
        "summary": {key: summary[key] for key in ("k_per_second", "heldout_native_field_variance_explained",
            "native_sine_recovery_prediction_mae_deg", "native_recovery_mae_improvement_over_no_anchor_deg",
            "phase_pulse_path_deviation_ordinary_minus_native_m", "native_path_deviation_phase_minus_normal_m")}}, indent=2))


if __name__ == "__main__":
    main()
