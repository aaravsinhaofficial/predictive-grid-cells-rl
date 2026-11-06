"""Cue-free full-state geometry and dynamics of learned heading memory.

Circular initialization alone is not an attractor test. This audit separately
measures cue-off persistence, transverse return, tangent displacement, output
coverage and local spectra, with an untrained architecture-matched control.
All distances use the complete64-dimensional recurrent state. PCA is only a
description of covariance, never the criterion for identifying a ring.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_ring_dynamics import jacobian
from ring_dynamics import ring_radius, tangents, perturbations, trajectory_measurements, wrap, validate_metrics
from ring_heading_model import HeadingRNN, load_checkpoint


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.1)), "q90": float(np.quantile(values, 0.9))}


@torch.no_grad()
def decoded_angle(model, states):
    value = torch.as_tensor(states, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    output = model.readout(value)
    return torch.atan2(output[..., 1], output[..., 0]).cpu().numpy(), output.norm(dim=-1).cpu().numpy()


def memory_measurements(model, states, phase, cue_radius, settled_radius):
    angle, amplitude = decoded_angle(model, states)
    ordered = np.sort(np.mod(angle, 2 * np.pi))
    largest_gap = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi]).max()
    error = wrap(angle - phase)
    local_phase_gain = wrap(np.roll(angle, -1) - angle) / (2 * np.pi / len(angle))
    centered = states - states.mean(0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / len(states))[::-1].clip(0)
    total = eigen.sum()
    radius = ring_radius(states)
    return {"angular_mae_deg": float(np.abs(error).mean() * 180 / np.pi),
            "angular_rmse_deg": float(np.sqrt(np.mean(error ** 2)) * 180 / np.pi),
            "circle_coverage": float(1 - largest_gap / (2 * np.pi)),
            "largest_uncovered_arc_deg": float(largest_gap * 180 / np.pi),
            "decoded_winding": float(wrap(np.roll(angle, -1) - angle).sum() / (2 * np.pi)),
            "local_decoded_phase_gain": distribution(local_phase_gain),
            "state_radius": radius, "radius_from_cue_end": radius / max(cue_radius, 1e-12),
            "radius_from_settled_curve": radius / max(settled_radius, 1e-12),
            "readout_amplitude": distribution(amplitude),
            "covariance_pc3_fraction": float(eigen[:3].sum() / total) if total else 0.0,
            "covariance_participation_ratio": float(total ** 2 / max(np.sum(eigen ** 2), 1e-30))}


@torch.no_grad()
def initialize(model, task, phase, position, condition, settle_steps=40):
    direction = torch.stack((phase.cos(), phase.sin()), -1)
    zero, one = torch.zeros_like(phase), torch.ones_like(phase)
    position = phase.new_tensor(position).expand(len(phase), 2)
    x = task.inputs(position, direction, zero, zero, one, one, condition, shuffled_direction=-direction)
    state = phase.new_zeros(len(phase), model.hidden_size)
    for _ in range(model.config.initial_heading_steps):
        state = model.step(x, state)
    cue_end = state.clone()
    blank = x * 0
    for _ in range(settle_steps):
        state = model.step(blank, state)
    return cue_end, state


@torch.no_grad()
def continue_all(model, initial_states, steps, save_times):
    """Batch baseline and every perturbation, with exactly identical input."""
    shape = initial_states.shape
    state = initial_states.reshape(-1, shape[-1])
    zero = state.new_zeros(len(state), model.input_size)
    drive = model.input(zero)
    result = {}
    for t in range(steps + 1):
        if t in save_times:
            result[t] = state.reshape(shape).cpu().numpy().copy()
        if t < steps:
            pre = drive + model.recurrent(state)
            state = (1 - model.config.leak) * state + model.config.leak * pre.tanh()
    return result


def eigenspace_alignment(vector, tangent):
    """For complex modes use the real invariant plane, rather than a complex angle."""
    if np.max(np.abs(vector.imag)) < 1e-8:
        axis = vector.real
        return float(abs(axis @ tangent) / max(np.linalg.norm(axis), 1e-15))
    basis, singular, _ = np.linalg.svd(np.column_stack((vector.real, vector.imag)), full_matrices=False)
    rank = int(np.sum(singular > max(singular[0], 1e-15) * 1e-8))
    return float(np.linalg.norm(basis[:, :rank].T @ tangent))


def spectral_measurements(model, states):
    state = torch.as_tensor(states, device=next(model.parameters()).device, dtype=torch.float32)
    matrices = jacobian(model, state).detach().cpu().numpy().astype(np.float64)
    try:
        tangent = tangents(states)
    except ValueError:
        return {"status": "collapsed_curve_no_tangent"}, {"jacobian": matrices}
    rows, values = [], []
    for matrix, tan in zip(matrices, tangent):
        eigen, vectors = np.linalg.eig(matrix)
        slow = int(np.argmin(np.abs(eigen - 1)))
        dominant = int(np.argmax(np.abs(eigen)))
        normal_basis = np.linalg.qr(tan[:, None], mode="complete")[0][:, 1:]
        normal = normal_basis.T @ matrix @ normal_basis
        jtan = matrix @ tan
        rows.append({"spectral_radius": float(np.abs(eigen).max()),
                     "slow_eigenvalue_real": float(eigen[slow].real),
                     "slow_eigenvalue_imag": float(eigen[slow].imag),
                     "slow_distance_to_one": float(abs(eigen[slow] - 1)),
                     "slow_mode_tangent_alignment": eigenspace_alignment(vectors[:, slow], tan),
                     "dominant_mode_tangent_alignment": eigenspace_alignment(vectors[:, dominant], tan),
                     "tangent_gain": float(tan @ jtan),
                     "tangent_shear": float(np.linalg.norm(jtan - tan * (tan @ jtan))),
                     "normal_spectral_radius": float(np.abs(np.linalg.eigvals(normal)).max()),
                     "normal_largest_singular_value": float(np.linalg.svd(normal, compute_uv=False)[0]),
                     "largest_singular_value": float(np.linalg.svd(matrix, compute_uv=False)[0])})
        values.append(eigen)
    result = {"status": "ok", "distributions": {key: distribution([row[key] for row in rows]) for key in rows[0]},
              "per_phase": rows,
              "slow_mode_definition": "eigenvalue closest to+1; dominant-modulus mode reported separately"}
    return result, {"jacobian": matrices, "eigenvalues": np.asarray(values), "tangents": tangent}


def run(checkpoint_path, output_prefix, n_phase=128, steps=600, device="cpu", include_untrained=True):
    trained, task, checkpoint = load_checkpoint(checkpoint_path, device)
    trained.requires_grad_(False)
    models = {"trained": trained}
    if include_untrained:
        torch.manual_seed(checkpoint["seed"])
        untrained = HeadingRNN(trained.config).to(device).eval()
        untrained.requires_grad_(False)
        models["untrained"] = untrained
    phase_tensor = torch.arange(n_phase, device=device) * (2 * math.pi / n_phase)
    phase = phase_tensor.cpu().numpy()
    positions = ((0.0, 0.0), (0.8, -0.8), (10.0, 10.0))
    sample_times = sorted(set(t for t in (0, 1, 5, 10, 20, 60, 100, 300, 600, steps) if t <= steps))
    # Absolute blank times40/100/300 plus continuation times100/300.
    spectral_times = [t for t in (0, 60, 100, 260, 300) if t <= steps]
    save_times = sorted(set(sample_times + spectral_times))
    all_results, archive = {}, {"phase": phase, "sample_times": np.asarray(sample_times)}
    for model_name, model in models.items():
        contexts = []
        for context_index, position in enumerate(positions):
            cue_end, settled = initialize(model, task, phase_tensor, position, checkpoint["condition"])
            cue_curve, curve = cue_end.cpu().numpy(), settled.cpu().numpy()
            cue_radius, settled_radius = ring_radius(cue_curve), ring_radius(curve)
            initial_states, sizes = [settled], []
            for size in (0.02, 0.10, 0.30):
                try:
                    tangent, normal = perturbations(curve, size, seed=419)
                except ValueError:
                    break
                initial_states.extend([settled + settled.new_tensor(tangent), settled + settled.new_tensor(normal)])
                sizes.append(size)
            initial_states = torch.stack(initial_states)
            paths = continue_all(model, initial_states, steps, set(save_times))
            base = np.stack([paths[t][0] for t in sample_times])
            memory = []
            for t, state in zip(sample_times, base):
                row = memory_measurements(model, state, phase, cue_radius, settled_radius)
                row.update(continuation_step=t, absolute_blank_step=40 + t, seconds_after_settling=t * model.config.dt)
                memory.append(row)
            cue_summary = memory_measurements(model, cue_curve, phase, cue_radius, settled_radius)
            perturbation_rows = {}
            prefix = f"{model_name}_context{context_index}"
            archive[f"{prefix}_cue_end"] = cue_curve
            archive[f"{prefix}_base"] = base
            for index, size in enumerate(sizes):
                tangent_path = np.stack([paths[t][1 + 2 * index] for t in sample_times])
                normal_path = np.stack([paths[t][2 + 2 * index] for t in sample_times])
                rows = trajectory_measurements(base, tangent_path, normal_path)
                initial_base_angle, _ = decoded_angle(model, base[0])
                initial_tangent_angle, _ = decoded_angle(model, tangent_path[0])
                initial_delta = wrap(initial_tangent_angle - initial_base_angle)
                valid = np.abs(initial_delta) >= 1e-5
                gains = []
                for t, row, b, tangent_state, normal_state in zip(sample_times, rows, base, tangent_path, normal_path):
                    base_angle, _ = decoded_angle(model, b)
                    tangent_angle, _ = decoded_angle(model, tangent_state)
                    normal_angle, _ = decoded_angle(model, normal_state)
                    delta = wrap(tangent_angle - base_angle)
                    local_gain = np.full(n_phase, np.nan)
                    local_gain[valid] = np.abs(delta[valid]) / np.abs(initial_delta[valid])
                    gains.append(local_gain)
                    row.update(step=t, absolute_blank_step=40 + t, seconds_after_settling=t * model.config.dt,
                               decoded_tangent_displacement_deg=distribution(np.abs(delta) * 180 / np.pi),
                               decoded_tangent_retention=distribution(local_gain),
                               decoded_normal_phase_error_deg=distribution(np.abs(wrap(normal_angle - base_angle)) * 180 / np.pi),
                               tangent_gain_resolvable_fraction=float(valid.mean()))
                perturbation_rows[str(size)] = rows
                archive[f"{prefix}_tangent_{size}"] = tangent_path
                archive[f"{prefix}_normal_{size}"] = normal_path
                archive[f"{prefix}_tangent_phase_gain_{size}"] = np.asarray(gains)
            spectral = []
            for t in spectral_times:
                summary, arrays = spectral_measurements(model, paths[t][0])
                summary.update(continuation_step=t, absolute_blank_step=40 + t)
                spectral.append(summary)
                archive.update({f"{prefix}_blank{40+t}_{key}": value for key, value in arrays.items()})
            # Independently check the optimized common-input update and analytic
            # full-state Jacobian against the public step implementation.
            zero = settled.new_zeros(n_phase, model.input_size)
            expected = model.step(zero, settled).detach().cpu().numpy()
            fixed_error = float(np.abs(expected - paths[1][0]).max()) if 1 in paths else 0.0
            with torch.enable_grad():
                h = settled[0].detach().clone().requires_grad_(True)
                x = zero[0]
                auto = torch.autograd.functional.jacobian(lambda state: model.step(x, state), h)
            analytic = jacobian(model, settled[:1])[0]
            jacobian_error = float((auto - analytic).abs().max())
            assert fixed_error < 2e-6 and jacobian_error < 2e-6
            result = {"position": list(position), "cue_end": cue_summary, "memory": memory,
                      "settling_angular_mae_change_deg": memory[0]["angular_mae_deg"] - cue_summary["angular_mae_deg"],
                      "settling_radius_retained": settled_radius / max(cue_radius, 1e-12),
                      "perturbations": perturbation_rows, "spectra": spectral,
                      "numerically_resolvable_tangent": bool(sizes),
                      "fixed_input_step_max_error": fixed_error, "jacobian_autograd_max_error": jacobian_error}
            contexts.append(result)
            print(json.dumps({"model": model_name, "context": context_index,
                "position": list(position), "radius_retained_after_settling": result["settling_radius_retained"],
                "final_radius_from_cue": memory[-1]["radius_from_cue_end"],
                "final_mae_deg": memory[-1]["angular_mae_deg"], "final_coverage": memory[-1]["circle_coverage"]}), flush=True)
        all_results[model_name] = contexts
    validation = validate_metrics()
    result = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "condition": checkpoint["condition"], "seed": checkpoint["seed"], "config": checkpoint["config"],
              "steps_completed": checkpoint["steps_completed"], "models": all_results,
              "protocol": {"n_phase": n_phase, "full_state_dimensions": trained.hidden_size,
                  "initial_heading_cue_steps": trained.config.initial_heading_steps, "blank_settle_steps": 40,
                  "continuation_steps": steps, "neutral_input": "all input channels zero, learned encoder bias retained",
                  "perturbation_sizes_fraction_of_full_state_radius": [0.02, 0.10, 0.30],
                  "perturbation_reference": "time-matched unperturbed full-state curve",
                  "spectra_absolute_blank_steps": [40 + t for t in spectral_times],
                  "phase_retention": "per-phase absolute displacement ratios, median and quantiles; no reliance on winding-invariant mean",
                  "untrained_control": "same architecture and original training initialization seed",
                  "state_noise": 0, "input_noise": 0, "primary_confirmation_endpoint": False},
              "metric_controls": {"status": validation["status"], "known_ring_final": validation["positive_ring"][-1],
                                  "collapse_final": validation["driven_circle_after_cue_removal"][-1]}}
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output_prefix.with_suffix(".npz"), **archive)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-phase", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-untrained", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args.checkpoint, args.out, args.n_phase, args.steps, args.device, not args.skip_untrained)


if __name__ == "__main__":
    main()
