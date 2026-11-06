"""Aggregate the four locked heading endpoints using training seeds as units.

Incomplete cohorts are rejected. Pilot files, intermediate checkpoints and
episodes cannot silently become independent statistical replicates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = tuple(range(2001, 2017))
CONDITIONS = ("grid", "predictive", "grid_heading")
ROOT = Path(__file__).resolve().parents[1]


def validate_report(report, seed, condition, locked):
    expected_sources = {p: digest for p, digest in locked["sources"].items() if p != "scripts/run_ring_batch.py"}
    provenance = report["provenance"]
    if provenance.get("verified_training_sources") != expected_sources:
        raise ValueError("Evaluation report source hashes differ from locked sources")
    if provenance.get("protocol_sha256") != locked["protocol_sha256"]:
        raise ValueError("Evaluation report protocol hash differs from locked plan")
    protocol = report["protocol"]
    expected = {"evaluation_seeds": [71001,71002,71003,71004], "episodes_per_seed": 512,
                "total_episodes_per_network": 2048, "sequence_length": 256,
                "exclude_first_steps": 4, "recurrent_noise": 0, "position_translation_m": [10.,0.],
                "pilot_override": False, "native_and_ordinary_share_all_underlying_experience": True}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Evaluation setting differs from locked protocol: {key}")
    if report["training_seed"] != seed or report["condition"] != condition or report["steps_completed"] != 6000:
        raise ValueError("Evaluation checkpoint identity differs from expected cohort")


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    corrected = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = corrected
    return out


def paired_summary(differences):
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("Expected all 16 finite paired training-seed differences")
    observed = values.mean()
    signs = 2 * (((np.arange(2**16)[:, None] >> np.arange(16)) & 1).astype(float)) - 1
    null = signs @ values / len(values)
    tolerance = np.finfo(float).eps * 64 * max(np.abs(values).max(), abs(observed), 1e-300)
    p = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    halfwidth = float(stats.t.ppf(.975, 15) * stats.sem(values))
    return dict(mean_difference=float(observed), ci95=[float(observed-halfwidth), float(observed+halfwidth)],
                p_exact_sign_flip=p, n_training_seeds=16,
                positive_seed_differences=int((values > 0).sum()),
                negative_seed_differences=int((values < 0).sum()),
                seed_differences=values.tolist())


def anchor_endpoint(report):
    protocol = report["protocol"]
    expected = {"phase_perturbation_deg": 10, "initial_heading_cue_steps": 4,
                "blank_settling_steps": 40, "n_phase": 72, "continuation_steps": 200,
                "physical_positions": [[-.8,-.8],[-.8,.8],[.8,-.8],[.8,.8],[0.,0.],[10.,10.]]}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Anchor setting differs from locked protocol: {key}")
    rows = [r for r in report["rows"] if r["context"] < 5 and r["step"] == 100]
    means = {}
    for variant in ("native", "ordinary", "blank", "reverse", "restore_20", "self_generated"):
        chosen = [r for r in rows if r["variant"] == variant]
        if not chosen:
            continue
        if len(chosen) != 5 or {r["context"] for r in chosen} != set(range(5)):
            raise ValueError("Each anchoring variant requires exactly five specified contexts")
        means[variant] = {key: float(np.mean([r[key] for r in chosen])) for key in
                          ("phase_local_gain_median", "baseline_mae_deg", "circle_coverage",
                           "full_state_phase_perturbation_gain_median", "state_radius_retained")}
    if not {"native", "ordinary", "blank"} <= means.keys():
        raise ValueError("Missing required anchoring variant")
    return means


def summarize(folder, output):
    folder, output = Path(folder), Path(output)
    locked = json.loads((ROOT/"docs/research/heading_confirmation_sources.json").read_text())
    if hashlib.sha256((ROOT/"docs/research/heading_confirmation_protocol.md").read_bytes()).hexdigest() != locked["protocol_sha256"]:
        raise ValueError("Locked protocol document has changed")
    rows, sources = [], []
    for seed in SEEDS:
        row = {"seed": seed}
        for condition in CONDITIONS:
            stem = f"{condition}_s{seed}"
            path = folder / f"primary_{stem}.json"
            report = json.loads(path.read_text())
            validate_report(report, seed, condition, locked)
            if (report["training_seed"] != seed or report["condition"] != condition or
                    report["steps_completed"] != 6000 or report["protocol"]["pilot_override"]):
                raise ValueError(f"Unexpected confirmation checkpoint: {path}")
            if not report["provenance"].get("verified_training_sources"):
                raise ValueError(f"Unverified training source: {path}")
            row[condition] = dict(report["metrics"])
            sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "checkpoint_sha256": report["checkpoint_sha256"]})
            if condition == "predictive":
                anchor_path = folder / f"anchor_{stem}.json"
                anchor = json.loads(anchor_path.read_text())
                if anchor["checkpoint_sha256"] != report["checkpoint_sha256"]:
                    raise ValueError("Tracking and anchor endpoints use different checkpoint bytes")
                row["anchor"] = anchor_endpoint(anchor)
                row["accurate_anchoring"] = (
                    row["anchor"]["native"]["circle_coverage"] >= .90 and
                    row["anchor"]["native"]["baseline_mae_deg"] < row["anchor"]["ordinary"]["baseline_mae_deg"])
                sources.append({"path": str(anchor_path), "sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest()})
        rows.append(row)
    differences = {
        "tracking_pgc_minus_grid_deg": [r["predictive"]["same_region_native_mae_deg"]-r["grid"]["same_region_native_mae_deg"] for r in rows],
        "removal_minus_native_deg": [r["predictive"]["ordinary_minus_native_mae_deg"] for r in rows],
        "phase_gain_native_minus_ordinary": [r["anchor"]["native"]["phase_local_gain_median"]-r["anchor"]["ordinary"]["phase_local_gain_median"] for r in rows],
        "transfer_pgc_minus_grid_deg": [r["predictive"]["new_places_native_mae_deg"]-r["grid"]["new_places_native_mae_deg"] for r in rows],
    }
    endpoints = {name: paired_summary(values) for name, values in differences.items()}
    corrected = holm([r["p_exact_sign_flip"] for r in endpoints.values()])
    for report, p in zip(endpoints.values(), corrected):
        report["p_holm_four_primary"] = float(p)
    result = {"protocol": "docs/research/heading_confirmation_protocol.md", "seeds": list(SEEDS),
              "primary_endpoints": endpoints, "accurate_anchoring_seeds": sum(r["accurate_anchoring"] for r in rows),
              "per_seed": rows, "source_reports": sources,
              "aggregation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note": "All 16 training seeds included; primary family has four two-sided tests. Anchor qualification is not an exclusion."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"primary_endpoints": endpoints, "accurate_anchoring_seeds": result["accurate_anchoring_seeds"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("results/ring_heading/confirmation_evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/ring_heading/confirmation_summary.json"))
    args = parser.parse_args()
    summarize(args.folder, args.out)
