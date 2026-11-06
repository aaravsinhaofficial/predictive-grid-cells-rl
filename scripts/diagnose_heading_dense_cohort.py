"""One model from the explicitly exploratory 32-model dense geometry cohort."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from diagnose_heading_dense_ring import angular_curve, describe, perturbations, spectra
from diagnose_heading_memory_connectedness import adaptive_transition, geometry, make_double, prepare
from ring_heading_model import load_checkpoint


def phase_derivative(model, curve, derivative):
    output = model.readout(curve)
    d_output = derivative @ model.readout.weight.T
    return ((output[:, 0] * d_output[:, 1] - output[:, 1] * d_output[:, 0]) / output.square().sum(-1)).cpu().numpy()


def run(args):
    started = time.monotonic()
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    make_double(model, task)
    phase = torch.arange(args.n_phase, dtype=torch.float64, device=task.device) * 2 * math.pi / args.n_phase
    curve, derivative = angular_curve(model, task, phase, checkpoint["condition"])
    decoded_derivative = phase_derivative(model, curve, derivative)
    indices = torch.tensor(np.linspace(0, args.n_phase - 1, 16).astype(int), device=task.device)
    epsilon = 1e-6
    numerical = (prepare(model, task, phase[indices] + epsilon, checkpoint["condition"])
                 - prepare(model, task, phase[indices] - epsilon, checkpoint["condition"])) / (2 * epsilon)
    check_error = float(((numerical - derivative[indices]).norm(dim=-1) / derivative[indices].norm(dim=-1).clamp_min(1e-10)).max())
    if check_error > 1e-4:
        raise ValueError(f"Angular tangent finite difference check failed:{check_error}")
    closed = prepare(model, task, phase.new_tensor([0., 2 * math.pi]), checkpoint["condition"])
    geometry_result = geometry(model, curve, phase.cpu().numpy())
    geometry_result.update(endpoint_closure_error=float((closed[0] - closed[1]).norm()),
        exact_angular_derivative_norm=describe(derivative.norm(dim=-1).cpu().numpy()),
        decoded_angular_derivative=describe(decoded_derivative),
        decoded_angular_derivative_nonpositive_fraction=float((decoded_derivative <= 0).mean()))
    perturbation_result, archive = perturbations(model, curve, derivative)
    spectral_result = spectra(model, curve, derivative)
    distances = (curve.roll(-1, 0) - curve).norm(dim=-1).cpu().numpy()
    chosen = []
    for index in np.argsort(distances)[::-1]:
        if all(min(abs(int(index) - previous), args.n_phase - abs(int(index) - previous)) > 2 for previous in chosen):
            chosen.append(int(index))
            if len(chosen) == 4:
                break
    adaptive, centers = [], []
    for rank, index in enumerate(chosen):
        low, high = float(phase[index]), float(phase[index]) + 2 * math.pi / args.n_phase
        rows, angles, states = adaptive_transition(model, task, checkpoint["condition"], low, high, levels=20)
        center = (rows[-1]["low_rad"] + rows[-1]["high_rad"]) / 2
        centers.append(center)
        adaptive.append({"rank": rank, "original_index": index, "refinement": rows, "center_rad": center})
        archive[f"adaptive{rank}_angles"] = angles
        archive[f"adaptive{rank}_states"] = states
    center_phase = phase.new_tensor(centers)
    center_state, center_derivative = angular_curve(model, task, center_phase, checkpoint["condition"])
    center_spectra = spectra(model, center_state, center_derivative)
    center_decoded = phase_derivative(model, center_state, center_derivative)
    for i, row in enumerate(adaptive):
        row["center_state_derivative_norm"] = float(center_derivative[i].norm())
        row["center_decoded_phase_derivative"] = float(center_decoded[i])
        row["center_spectrum"] = center_spectra["per_phase"][i]
    normal_values = [r["normal_spectral_radius"] for r in spectral_result["per_phase"]] + [r["normal_spectral_radius"] for r in center_spectra["per_phase"]]
    derivative_peak = max(float(derivative.norm(dim=-1).max()), float(center_derivative.norm(dim=-1).max()))
    diagnostic_properties = {"normal_radius_at_least_one_at_any_uniform_or_adaptive_point": bool(max(normal_values) >= 1),
        "maximum_normal_spectral_radius": float(max(normal_values)),
        "nonpositive_decoded_derivative_at_any_uniform_or_adaptive_point": bool((decoded_derivative <= 0).any() or (center_decoded <= 0).any()),
        "maximum_state_derivative_over_uniform_median": derivative_peak / float(derivative.norm(dim=-1).median()),
        "minimum_slow_alignment_uniform_or_adaptive": float(min(
            [r["slow_alignment_to_exact_tangent"] for r in spectral_result["per_phase"]]
            + [r["slow_alignment_to_exact_tangent"] for r in center_spectra["per_phase"]]))}
    source_files = [Path(__file__), Path("scripts/diagnose_heading_dense_ring.py"),
                    Path("scripts/diagnose_heading_memory_connectedness.py"), Path("docs/research/heading_dense_geometry_protocol.md")]
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "condition": checkpoint["condition"], "seed": checkpoint["seed"], "steps_completed": checkpoint["steps_completed"],
        "n_phase": args.n_phase, "geometry": geometry_result, "perturbations": perturbation_result,
        "spectra": spectral_result, "adaptive_intervals": adaptive, "diagnostic_properties": diagnostic_properties,
        "tangent_finite_difference_max_relative_error": check_error, "elapsed_seconds": time.monotonic() - started,
        "protocol": "docs/research/heading_dense_geometry_protocol.md", "cohort": "exploratory_dense_geometry",
        "no_seed_exclusion": True, "not_a_binary_ring_classifier": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    archive.update(phase=phase.cpu().numpy(), curve=curve.cpu().numpy(), exact_derivative=derivative.cpu().numpy(),
                   decoded_phase_derivative=decoded_derivative)
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"condition": checkpoint["condition"], "seed": checkpoint["seed"],
                      "elapsed_seconds": report["elapsed_seconds"], "diagnostic_properties": diagnostic_properties}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-phase", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
