"""Descriptive anchoring summary for all eight raw-input Grid/PGC seed pairs."""
import argparse
import json
from pathlib import Path
import time

from summarize_heading_confirmation import anchor_endpoint
from summarize_heading_full_state_memory import summarize


def build(root):
    rows = []
    for seed in range(3001, 3009):
        for condition in ("grid", "predictive"):
            path = root / f"anchor_{condition}_s{seed}.json"
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            assert report["seed"] == seed and report["condition"] == condition
            rows.append({"seed": seed, "condition": condition, "checkpoint_sha256": report["checkpoint_sha256"],
                         "in_region": anchor_endpoint(report),
                         "new_position": {r["variant"]: {key: r[key] for key in
                             ("phase_local_gain_median", "baseline_mae_deg", "circle_coverage",
                              "full_state_phase_perturbation_gain_median", "state_radius_retained")}
                              for r in report["rows"] if r["context"] == 5 and r["step"] == 100}})
    groups = {}
    for condition in ("grid", "predictive"):
        chosen = [row for row in rows if row["condition"] == condition]
        groups[condition] = {"training_seeds": [row["seed"] for row in chosen]}
        if not chosen:
            continue
        for context in ("in_region", "new_position"):
            groups[condition][context] = {variant: {key: summarize(
                [row[context][variant][key] for row in chosen], [row["seed"] for row in chosen], 8)
                for key in chosen[0][context][variant]} for variant in chosen[0][context]}
    report = {"complete": len(rows) == 16, "models_loaded": len(rows), "models_expected": 16,
              "groups": groups, "per_seed": rows,
              "scope": "secondary raw architecture cohort; never pooled with16seed encoder cohort",
              "endpoint": "5seconds continuation after2seconds blank settling; local median phase retention with accuracy and coverage",
              "uncertainty": "descriptive unadjusted95%tintervals across training seeds; no additional primary hypothesis tests",
              "reverse": "reverse learned PGC displacement; no-op forGrid",
              "self_generated": "PGCpreview follows each replica own decoded memory; contraction may accompany inaccurate locking"}
    (root / "anchor_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/raw_memory_evaluation"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    start, count = time.monotonic(), -1
    while True:
        report = build(args.root)
        if count != report["models_loaded"]:
            count = report["models_loaded"]
            print(json.dumps({"complete": report["complete"], "models_loaded": count}), flush=True)
        if report["complete"] or not args.watch:
            return
        if time.monotonic() - start > 4 * 3600:
            raise TimeoutError("Raw anchoring cohort remains incomplete")
        time.sleep(30)


if __name__ == "__main__":
    main()
