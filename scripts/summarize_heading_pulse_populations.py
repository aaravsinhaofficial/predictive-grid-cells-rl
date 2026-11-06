"""Pair training seeds across Grid, PGC and explicit-heading memory-pulse controls."""
import argparse
import json
from pathlib import Path

from summarize_heading_mechanistic_secondary import summarize


CONDITIONS = ("grid", "predictive", "grid_heading")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/mechanistic_confirmation"))
    args = parser.parse_args()
    rows = []
    for seed in range(2001, 2017):
        paths = {c: args.root / f"memory_pulse_{c}_s{seed}.json" for c in CONDITIONS}
        if not all(p.exists() for p in paths.values()):
            continue
        row = {"seed": seed, "populations": {}}
        for condition, path in paths.items():
            report = json.loads(path.read_text())
            assert report["training_seed"] == seed and report["steps_completed"] == 6000
            early, nav = report["early_heading_and_path"], report["navigation"]
            result = {}
            for state in ("baseline", "phase_balanced20", "normal_matched20"):
                for key in ("early_heading_mae_deg", "heading_error_at_1s_deg", "heading_error_at_2s_deg", "first5s_path_deviation_m"):
                    result[f"{state}_{key}"] = early[state]["native"][key]
                for key in ("success_fraction", "sustained_dwell_success_fraction", "mean_capped_arrival_seconds"):
                    result[f"{state}_{key}"] = nav[state]["40s"]["native"][key]
            result["phase_pulse_arrival_cost"] = result["baseline_success_fraction"] - result["phase_balanced20_success_fraction"]
            result["phase_pulse_dwell_cost"] = result["baseline_sustained_dwell_success_fraction"] - result["phase_balanced20_sustained_dwell_success_fraction"]
            result["phase_pulse_time_cost_seconds"] = result["phase_balanced20_mean_capped_arrival_seconds"] - result["baseline_mean_capped_arrival_seconds"]
            result["phase_minus_normal_path_deviation_m"] = result["phase_balanced20_first5s_path_deviation_m"] - result["normal_matched20_first5s_path_deviation_m"]
            result["pulse_induced_heading_error_at2s_deg"] = result["phase_balanced20_heading_error_at_2s_deg"] - result["baseline_heading_error_at_2s_deg"]
            row["populations"][condition] = result
        row["paired_contrasts"] = {}
        for left, right in (("grid", "predictive"), ("predictive", "grid_heading")):
            for metric in ("phase_balanced20_first5s_path_deviation_m", "phase_minus_normal_path_deviation_m",
                           "phase_pulse_arrival_cost", "phase_pulse_time_cost_seconds", "pulse_induced_heading_error_at2s_deg"):
                row["paired_contrasts"][f"{left}_minus_{right}_{metric}"] = row["populations"][left][metric] - row["populations"][right][metric]
        rows.append(row)
    if not rows:
        raise ValueError("No complete matched population triplets")
    populations = {condition: {key: summarize([row["populations"][condition][key] for row in rows])
                               for key in rows[0]["populations"][condition]} for condition in CONDITIONS}
    contrasts = {key: summarize([row["paired_contrasts"][key] for row in rows]) for key in rows[0]["paired_contrasts"]}
    report = {"complete": len(rows) == 16, "training_seeds": [row["seed"] for row in rows],
        "expected_training_seeds": list(range(2001, 2017)), "populations": populations,
        "paired_contrasts": contrasts, "per_seed": rows,
        "protocol": "docs/research/heading_memory_pulse_population_extension.md",
        "inference": "secondary descriptive replication; unadjusted95%tintervals; training seeds are the independent units"}
    out = args.root / "population_pulse_summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "training_seeds": report["training_seeds"],
        "paired_contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
