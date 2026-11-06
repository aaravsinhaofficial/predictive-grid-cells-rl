"""Seed-level full-state memory audit, including every completed and untrained model.

No performance-based exclusion or ring classification is performed. Undefined
geometry stays explicit, and results remain partial until all expected files
arrive. Different physical contexts are summarized within a training seed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import t


def summarize(values, seeds, expected):
    defined = [i for i, value in enumerate(values) if value is not None and np.isfinite(value)]
    finite = np.array([values[i] for i in defined], dtype=float)
    unavailable = [seed for i, seed in enumerate(seeds) if i not in defined]
    result = {"n_expected_training_seeds": expected, "n_loaded_training_seeds": len(seeds),
              "n_defined": len(defined), "undefined_training_seeds": unavailable}
    if not len(finite):
        return dict(result, mean=None, ci95=None, min=None, max=None)
    mean = float(finite.mean())
    half = float(t.ppf(.975, len(finite) - 1) * finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else None
    return dict(result, mean=mean, ci95=[mean - half, mean + half] if half is not None else None,
                min=float(finite.min()), max=float(finite.max()))


def context_scalars(context, dt):
    values = {"cue_end_mae_deg": context["cue_end"]["angular_mae_deg"],
              "cue_end_coverage": context["cue_end"]["circle_coverage"],
              "settling_radius_retained": context["settling_radius_retained"],
              "numerically_resolvable_tangent": float(context["numerically_resolvable_tangent"])}
    for seconds in (2, 7, 32):
        row = next((r for r in context["memory"] if abs(r["absolute_blank_step"] * dt - seconds) < 1e-8), None)
        for source, suffix in (("angular_mae_deg", "heading_mae_deg"), ("circle_coverage", "coverage"),
                               ("state_radius", "state_radius"), ("radius_from_cue_end", "radius_from_cue"),
                               ("radius_from_settled_curve", "radius_from_settled_curve")):
            values[f"blank_{seconds}s_{suffix}"] = row[source] if row else None
    for size in ("0.02", "0.1", "0.3"):
        rows = context["perturbations"].get(size, [])
        initial = rows[0] if rows else None
        prefix = "" if size == "0.1" else f"size{size}_"
        for seconds in (1, 5):
            row = next((r for r in rows if abs(r["seconds_after_settling"] - seconds) < 1e-8), None)
            ratio = (row["normal_distance_radius"] / initial["normal_distance_radius"]
                     if row and initial["normal_distance_radius"] > 1e-12 else None)
            values[f"{prefix}normal_remaining_{seconds}s"] = ratio
            values[f"{prefix}normal_returned_{seconds}s"] = 1 - ratio if ratio is not None else None
            values[f"{prefix}tangent_phase_retained_{seconds}s"] = row["decoded_tangent_retention"]["median"] if row else None
            values[f"{prefix}tangent_phase_retained_q10_{seconds}s"] = row["decoded_tangent_retention"]["q10"] if row else None
            values[f"{prefix}tangent_phase_retained_q90_{seconds}s"] = row["decoded_tangent_retention"]["q90"] if row else None
            values[f"{prefix}tangent_resolvable_fraction_{seconds}s"] = row["tangent_gain_resolvable_fraction"] if row else None
            values[f"{prefix}normal_fullstate_pair_remaining_{seconds}s"] = (row["normal_pair_distance_radius"] / initial["normal_pair_distance_radius"]
                if row and initial["normal_pair_distance_radius"] > 1e-12 else None)
            values[f"{prefix}tangent_fullstate_pair_remaining_{seconds}s"] = (row["tangent_pair_distance_radius"] / initial["tangent_pair_distance_radius"]
                if row and initial["tangent_pair_distance_radius"] > 1e-12 else None)
    for seconds in (2, 5, 7, 15, 17):
        row = next((r for r in context["spectra"] if abs(r["absolute_blank_step"] * dt - seconds) < 1e-8), None)
        distributions = row.get("distributions", {}) if row else {}
        for key in ("slow_eigenvalue_real", "slow_eigenvalue_imag", "slow_distance_to_one",
                    "slow_mode_tangent_alignment", "normal_spectral_radius", "spectral_radius", "tangent_gain",
                    "normal_largest_singular_value", "largest_singular_value"):
            values[f"blank_{seconds}s_{key}"] = distributions.get(key, {}).get("median")
    return values


def mean_contexts(contexts):
    return {key: float(np.mean([c[key] for c in contexts])) if all(c[key] is not None for c in contexts) else None
            for key in contexts[0]}


def build(args):
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    rows, pending, errors = [], [], []
    for seed in seeds:
        for condition in args.conditions:
            path = args.root / f"memory_{condition}_s{seed}.json"
            if not path.exists():
                pending.append({"seed": seed, "condition": condition})
                continue
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError:
                pending.append({"seed": seed, "condition": condition, "reason": "write_in_progress"})
                continue
            assert report["seed"] == seed and report["condition"] == condition
            assert report["steps_completed"] == 6000
            row = {"seed": seed, "condition": condition, "source": str(path),
                   "checkpoint_sha256": report["checkpoint_sha256"], "evaluator_sha256": report["evaluator_sha256"],
                   "models": {}}
            for model_name in ("trained", "untrained"):
                if model_name not in report["models"]:
                    errors.append({"seed": seed, "condition": condition, "missing_model": model_name})
                    continue
                contexts = [context_scalars(c, report["config"]["dt"]) for c in report["models"][model_name]]
                assert len(contexts) == 3
                row["models"][model_name] = {"contexts": contexts,
                    "in_region": mean_contexts(contexts[:2]), "new_position": contexts[2],
                    "all_positions": mean_contexts(contexts)}
            rows.append(row)
    groups = {}
    for condition in args.conditions:
        selected = [row for row in rows if row["condition"] == condition]
        groups[condition] = {"training_seeds": [row["seed"] for row in selected]}
        for model_name in ("trained", "untrained"):
            groups[condition][model_name] = {}
            defined = [row for row in selected if model_name in row["models"]]
            if not defined:
                continue
            for region in ("in_region", "new_position", "all_positions"):
                groups[condition][model_name][region] = {key: summarize(
                    [row["models"][model_name][region][key] for row in defined],
                    [row["seed"] for row in defined], args.seed_count)
                    for key in defined[0]["models"][model_name][region]}
    contrasts = {}
    for left, right in (("predictive", "grid"), ("grid_heading", "predictive")):
        if left not in args.conditions or right not in args.conditions:
            continue
        paired = [(next((r for r in rows if r["seed"] == seed and r["condition"] == left), None),
                   next((r for r in rows if r["seed"] == seed and r["condition"] == right), None)) for seed in seeds]
        paired = [(l, r) for l, r in paired if l and r and "trained" in l["models"] and "trained" in r["models"]]
        if paired:
            contrasts[f"{left}_minus_{right}"] = {key: summarize([
                l["models"]["trained"]["in_region"][key] - r["models"]["trained"]["in_region"][key]
                if l["models"]["trained"]["in_region"][key] is not None and r["models"]["trained"]["in_region"][key] is not None else None
                for l, r in paired], [l["seed"] for l, r in paired], args.seed_count)
                for key in paired[0][0]["models"]["trained"]["in_region"]}
    full_count = args.seed_count * len(args.conditions)
    result = {"complete": len(rows) == full_count and not errors, "models_loaded": len(rows),
        "models_expected": full_count, "expected_training_seeds": seeds, "conditions": args.conditions,
        "pending": pending, "errors": errors, "groups": groups, "paired_contrasts": contrasts, "per_seed": rows,
        "definitions": {"independent_unit": "training seed; contexts and phases summarized within seed",
            "uncertainty": "unadjusted descriptive95%tintervals across all available training seeds; no primary tests",
            "normal_return": "distance to time-matched full-state curve divided by initial distance; return=1-remaining",
            "tangent_retention": "median of per-phase decoded displacement ratios, then mean across stated contexts",
            "main_perturbation_size": "10%of initial full-state ring radius;2%and30%sensitivities retained",
            "perturbation_times": "1/5seconds after perturbing the state prepared by2seconds blank input",
            "blank_memory_times": "2/7/32seconds total blank input after initial cue",
            "spectra": "median over phases, then mean within specified context group; slow mode closest to+1",
            "in_region_positions": [[0, 0], [.8, -.8]], "new_position": [10, 10],
            "untrained_output_caution": "untrained readout is also random; heading accuracy alone does not diagnose hidden memory",
            "undefined_values": "not imputed; model retained, unavailable metric and denominator explicitly reported"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ring_heading/confirmation_evaluation"))
    parser.add_argument("--out", type=Path, default=Path("results/ring_heading/confirmation_evaluation/memory_summary.json"))
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--conditions", nargs="+", default=["grid", "predictive", "grid_heading"])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--max-hours", type=float, default=4.)
    args = parser.parse_args()
    start, previous = time.monotonic(), None
    while True:
        report = build(args)
        count = report["models_loaded"]
        if count != previous:
            print(json.dumps({"complete": report["complete"], "models_loaded": count,
                              "models_expected": report["models_expected"], "errors": report["errors"]}), flush=True)
        previous = count
        if report["complete"] or not args.watch:
            break
        if time.monotonic() - start > args.max_hours * 3600:
            raise TimeoutError("Full-state memory cohort remains incomplete")
        time.sleep(30)


if __name__ == "__main__":
    main()
