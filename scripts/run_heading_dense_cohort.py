"""Bounded single-worker evaluation of every exploratory dense geometry model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from summarize_heading_full_state_memory import summarize


OUT = Path("results/ring_heading/dense_geometry_cohort")
CONDITIONS = ("grid", "predictive")
SEEDS = tuple(range(2001, 2017))
SOURCES = ("scripts/diagnose_heading_dense_cohort.py", "scripts/diagnose_heading_dense_ring.py",
           "scripts/diagnose_heading_memory_connectedness.py", "scripts/evaluate_heading_memory.py",
           "scripts/evaluate_ring_dynamics.py", "scripts/ring_dynamics.py", "scripts/ring_heading_model.py",
           "scripts/ring_task_model.py", "docs/research/heading_dense_geometry_protocol.md")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate():
    rows = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            path = OUT / f"{condition}_s{seed}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            row = {"seed": seed, "condition": condition, "checkpoint_sha256": report["checkpoint_sha256"],
                "elapsed_seconds": report["elapsed_seconds"], **report["diagnostic_properties"],
                "maximum_adjacent_distance_radius": report["geometry"]["adjacent_distance_max"] / report["geometry"]["state_radius"],
                "maximum_to_median_adjacent_distance": report["geometry"]["max_to_median"],
                "decoded_heading_mae_deg": report["geometry"]["decoded_heading_mae_deg"],
                "decoded_coverage": report["geometry"]["decoded_coverage"],
                "endpoint_closure_error": report["geometry"]["endpoint_closure_error"],
                "normal_remaining_1s": report["perturbations"]["20"]["normal_remaining_mean_distance_ratio"],
                "normal_remaining_5s": report["perturbations"]["100"]["normal_remaining_mean_distance_ratio"],
                "phase_retained_median_1s": report["perturbations"]["20"]["tangent_phase_gain_per_phase"]["median"],
                "phase_retained_median_5s": report["perturbations"]["100"]["tangent_phase_gain_per_phase"]["median"],
                "finite_difference_max_relative_error": report["tangent_finite_difference_max_relative_error"]}
            rows.append(row)
    groups = {}
    for condition in CONDITIONS:
        chosen = [r for r in rows if r["condition"] == condition]
        if not chosen:
            continue
        groups[condition] = {"training_seeds": [r["seed"] for r in chosen],
            "normal_radius_at_least_one_seeds": [r["seed"] for r in chosen if r["normal_radius_at_least_one_at_any_uniform_or_adaptive_point"]],
            "nonpositive_decoded_derivative_seeds": [r["seed"] for r in chosen if r["nonpositive_decoded_derivative_at_any_uniform_or_adaptive_point"]],
            "metrics": {k: summarize([r[k] for r in chosen], [r["seed"] for r in chosen], 16)
                for k, value in chosen[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool) and k != "seed"}}
    result = {"complete": len(rows) == 32, "models_loaded": len(rows), "models_expected": 32,
        "per_seed": rows, "groups": groups, "protocol": "docs/research/heading_dense_geometry_protocol.md",
        "interpretation": "explicitly exploratory; all32models retained; properties are not a binary ring classifier",
        "uncertainty": "descriptive95%tintervals across training seeds, not across angles"}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    locked = {name: digest(name) for name in SOURCES}
    manifest = OUT / "frozen_sources.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["source_hashes"] != locked:
            raise RuntimeError("Dense-cohort frozen sources differ from previous run")
    else:
        manifest.write_text(json.dumps({"source_hashes": locked, "n_phase": 1024, "threads": 2,
            "workers": 1, "expected_training_seeds": SEEDS, "conditions": CONDITIONS,
            "stage": "exploratory after seed2001/2002 diagnostic observations"}, indent=2) + "\n")
    completed, previous_sizes, start = set(), {}, time.monotonic()
    while time.monotonic() - start < 4 * 3600:
        for path, expected in locked.items():
            if digest(path) != expected:
                raise RuntimeError(f"Frozen dense-cohort source changed:{path}")
        for seed in SEEDS:
            for condition in CONDITIONS:
                key = f"{condition}_s{seed}"
                if key in completed:
                    continue
                files = sorted(Path("results/ring_cloud_artifacts").glob(f"ring-20261105-confirm-*/{key}.pt"))
                if len(files) > 1:
                    raise RuntimeError(f"Duplicate checkpoint:{key}")
                if not files:
                    continue
                checkpoint = files[0]
                size = checkpoint.stat().st_size
                if previous_sizes.get(key) != size:
                    previous_sizes[key] = size
                    continue
                prefix = OUT / key
                path = prefix.with_suffix(".json")
                if not path.exists():
                    print(json.dumps({"starting": key}), flush=True)
                    with prefix.with_suffix(".log").open("w") as log:
                        subprocess.run([sys.executable, "scripts/diagnose_heading_dense_cohort.py", str(checkpoint),
                            "--out", str(prefix), "--n-phase", "1024", "--threads", "2"], stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(path.read_text())
                if report["checkpoint_sha256"] != digest(checkpoint) or report["seed"] != seed or report["condition"] != condition:
                    raise RuntimeError(f"Dense report identity mismatch:{key}")
                if report["steps_completed"] != 6000 or report["n_phase"] != 1024:
                    raise RuntimeError(f"Dense report protocol mismatch:{key}")
                for source, checksum in report["source_hashes"].items():
                    relative = str(Path(source).resolve().relative_to(Path.cwd()))
                    if checksum != locked[relative]:
                        raise RuntimeError(f"Dense report source mismatch:{key}/{source}")
                completed.add(key)
                aggregate()
                print(json.dumps({"completed": key, "completed_count": len(completed),
                                  "model_seconds": report["elapsed_seconds"]}), flush=True)
        if len(completed) == 32:
            return
        time.sleep(20)
    raise TimeoutError("Dense geometry cohort remains incomplete")


if __name__ == "__main__":
    main()
