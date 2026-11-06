"""Fit a simple phase equation on training fields, then predict held-out RNN recovery.

The RNN is frozen. Its cue-free drift and cue response are measured separately.
No held-out one-step field or recovery trajectory contributes to the fit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ring_heading_model import load_checkpoint
from ring_dynamics import wrap


POSITIONS = ((-.8, -.8), (.8, .8), (0., 0.), (.8, -.8), (10., 10.))
VARIANTS = ("native", "blank", "ordinary", "reverse", "self_generated")
REDUCTIONS = ("constant", "no_anchor", "sine", "local_sine", "three_harmonics")


def angle(model, state):
    output = model.readout(state)
    return torch.atan2(output[..., 1], output[..., 0])


def basis(phase, harmonics=3, intercept=True):
    columns = [np.ones_like(phase)] if intercept else []
    for k in range(1, harmonics + 1):
        columns.extend((np.sin(k * phase), np.cos(k * phase)))
    return np.stack(columns, -1)


@torch.no_grad()
def initial_curve(model, task, condition, position, n_phase):
    phase = torch.arange(n_phase, device=task.device) * (2 * math.pi / n_phase)
    direction = torch.stack((phase.cos(), phase.sin()), -1)
    zeros, ones = torch.zeros_like(phase), torch.ones_like(phase)
    pos = torch.tensor(position, device=task.device).expand(n_phase, 2)
    cue = task.inputs(pos, direction, zeros, zeros, ones, ones, condition)
    state = torch.zeros(n_phase, model.hidden_size, device=task.device)
    for _ in range(model.config.initial_heading_steps):
        state = model.step(cue, state)
    for _ in range(40):
        state = model.step(torch.zeros_like(cue), state)
    return phase, state


def cue_input(model, task, condition, position, cue_angle, variant, state_angle):
    if variant == "blank":
        return torch.zeros(len(cue_angle), model.input_size, device=task.device)
    observed = cue_angle
    input_condition = condition
    if variant == "ordinary":
        input_condition = "grid"
    elif variant == "reverse" and condition != "grid":
        # Reverse the learned directional cue: spatial displacement for PGC,
        # explicit heading components for the information-matched control.
        observed = observed + math.pi
    elif variant == "self_generated":
        observed = state_angle
    direction = torch.stack((observed.cos(), observed.sin()), -1)
    zeros, ones = torch.zeros_like(cue_angle), torch.ones_like(cue_angle)
    pos = torch.tensor(position, device=task.device).expand(len(cue_angle), 2)
    return task.inputs(pos, direction, zeros, zeros, ones, zeros, input_condition)


@torch.no_grad()
def measure_field(model, task, condition, position, phase, curve, indices):
    state_index, cue_index = torch.meshgrid(indices, indices, indexing="ij")
    state = curve[state_index.flatten()]
    cue = phase[cue_index.flatten()]
    phi = angle(model, state)
    rows = {"phi": phi.cpu().numpy(), "cue": cue.cpu().numpy()}
    for variant in VARIANTS:
        x = cue_input(model, task, condition, position, cue, variant, phi)
        next_phi = angle(model, model.step(x, state))
        rows[variant] = wrap((next_phi - phi).cpu().numpy()) / model.config.dt
    return rows


def fit_reduction(training_fields):
    phi = np.concatenate([row["phi"] for row in training_fields])
    cue = np.concatenate([row["cue"] for row in training_fields])
    blank = np.concatenate([row["blank"] for row in training_fields])
    native = np.concatenate([row["native"] for row in training_fields])
    f0 = np.linalg.lstsq(basis(phi), blank, rcond=None)[0]
    residual = native - basis(phi) @ f0
    difference = wrap(cue - phi)
    sine = np.sin(difference)
    k = float(np.dot(sine, residual) / np.dot(sine, sine))
    local = np.abs(difference) <= math.pi / 3
    local_k = float(np.dot(sine[local], residual[local]) / np.dot(sine[local], sine[local]))
    richer = np.linalg.lstsq(basis(difference, intercept=False), residual, rcond=None)[0]
    return {"f0_coefficients": f0.tolist(), "k_per_second": k,
            "local_k_per_second": local_k, "cue_harmonic_coefficients": richer.tolist(),
            "fit_rows": len(phi), "local_fit_rows": int(local.sum()),
            "fit_blank_rmse_deg_per_second": float(np.sqrt(np.mean((blank - basis(phi) @ f0) ** 2)) * 180 / np.pi)}


def predict_velocity(phi, cue, variant, reduction, coefficients, condition):
    if reduction == "constant":
        return np.zeros_like(phi)
    velocity = basis(phi) @ np.array(coefficients["f0_coefficients"])
    if reduction == "no_anchor" or variant in ("blank", "ordinary"):
        return velocity
    effective_cue = cue
    if variant == "reverse" and condition != "grid":
        effective_cue = cue + math.pi
    if variant == "self_generated":
        effective_cue = phi
    difference = wrap(effective_cue - phi)
    if reduction in ("sine", "local_sine"):
        key = "k_per_second" if reduction == "sine" else "local_k_per_second"
        return velocity + coefficients[key] * np.sin(difference)
    return velocity + basis(difference, intercept=False) @ np.array(coefficients["cue_harmonic_coefficients"])


def field_metrics(actual, predicted):
    squared = np.mean((actual - predicted) ** 2)
    variance = np.var(actual)
    return {"variance_explained": float(1 - squared / variance) if variance > 1e-12 else None,
            "rmse_deg_per_second": float(np.sqrt(squared) * 180 / np.pi),
            "mae_deg_per_second": float(np.mean(np.abs(actual - predicted)) * 180 / np.pi)}


def evaluate_field(row, coefficients, condition):
    result = {}
    for variant in VARIANTS:
        result[variant] = {reduction: field_metrics(row[variant], predict_velocity(
            row["phi"], row["cue"], variant, reduction, coefficients, condition))
            for reduction in REDUCTIONS}
    return result


@torch.no_grad()
def recovery(model, task, condition, position, phase, curve, coefficients, steps):
    n_phase = len(phase)
    cue_index = torch.arange(1, n_phase, n_phase // 12, device=task.device)
    offsets = torch.tensor([0., -10., 10., -45., 45., -90., 90., -150., 150.], device=task.device)
    offset_index = (offsets / (360 / n_phase)).round().long()
    state_index = (cue_index[:, None] + offset_index[None]) % n_phase
    state0 = curve[state_index.flatten()]
    cue = phase[cue_index][:, None].expand(-1, len(offsets)).flatten()
    phi0 = angle(model, state0).cpu().numpy()
    cue_np = cue.cpu().numpy()
    archive = {"cue": cue_np, "initial_phi": phi0,
               "nominal_initial_offsets_deg": offsets.repeat(len(cue_index)).cpu().numpy()}
    report = {}
    for variant in VARIANTS:
        state = state0.clone()
        actual = [phi0]
        predicted = {reduction: [phi0.copy()] for reduction in REDUCTIONS}
        for _ in range(steps):
            x = cue_input(model, task, condition, position, cue, variant, angle(model, state))
            state = model.step(x, state)
            actual.append(angle(model, state).cpu().numpy())
            for reduction in REDUCTIONS:
                current = predicted[reduction][-1]
                velocity = predict_velocity(current, cue_np, variant, reduction, coefficients, condition)
                predicted[reduction].append(wrap(current + task.config.dt * velocity))
        actual = np.stack(actual)
        archive[f"{variant}_actual"] = actual
        report[variant] = {"actual_final_true_heading_mae_deg": float(np.mean(np.abs(wrap(actual[-1] - cue_np))) * 180 / np.pi)}
        for reduction in REDUCTIONS:
            expected = np.stack(predicted[reduction])
            error = wrap(expected - actual)
            archive[f"{variant}_{reduction}"] = expected
            report[variant][reduction] = {
                "trajectory_prediction_mae_deg": float(np.mean(np.abs(error[1:])) * 180 / np.pi),
                "final_prediction_mae_deg": float(np.mean(np.abs(error[-1])) * 180 / np.pi),
                "predicted_final_true_heading_mae_deg": float(np.mean(np.abs(wrap(expected[-1] - cue_np))) * 180 / np.pi),
                "sampled_prediction_mae_deg": {str(t): float(np.mean(np.abs(error[t])) * 180 / np.pi)
                    for t in (1, 5, 10, 20, 40, 100, 200) if t <= steps}}
    return report, archive


def run(args):
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    if args.n_phase % 72:
        raise ValueError("n_phase must be divisible by72 for five-degree phase substitutions")
    condition = checkpoint["condition"]
    curves, field_rows = [], []
    archive = {}
    for i, position in enumerate(POSITIONS):
        phase, curve = initial_curve(model, task, condition, position, args.n_phase)
        curves.append((phase, curve))
        parity = 0 if i < 2 else 1
        indices = torch.arange(parity, args.n_phase, 2, device=task.device)
        row = measure_field(model, task, condition, position, phase, curve, indices)
        field_rows.append(row)
        archive[f"context{i}_curve"] = curve.cpu().numpy()
        archive.update({f"context{i}_field_{key}": value for key, value in row.items()})
    coefficients = fit_reduction(field_rows[:2])
    fit = [evaluate_field(row, coefficients, condition) for row in field_rows[:2]]
    heldout = [evaluate_field(row, coefficients, condition) for row in field_rows[2:]]
    pooled = {key: np.concatenate([row[key] for row in field_rows[2:]]) for key in field_rows[0]}
    recoveries = []
    for i, position in enumerate(POSITIONS[2:], 2):
        phase, curve = curves[i]
        summary, arrays = recovery(model, task, condition, position, phase, curve, coefficients, args.steps)
        recoveries.append({"position": list(position), "variants": summary})
        archive.update({f"context{i}_recovery_{key}": value for key, value in arrays.items()})
    report = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "condition": condition, "seed": checkpoint["seed"], "config": checkpoint["config"],
              "steps_completed": checkpoint["steps_completed"], "coefficients": coefficients,
              "fitted_fields": fit, "heldout_fields": heldout,
              "pooled_heldout_fields": evaluate_field(pooled, coefficients, condition),
              "heldout_recovery": recoveries,
              "protocol": {"plan": "docs/research/heading_phase_reduction_protocol.md",
                  "equation": "dphi/dt=f0(phi)+k*sin(theta_cue-phi)",
                  "f0": "independently fitted blank-input field; three Fourier harmonics plus intercept",
                  "coefficient_sign_constrained": False, "phase_readout": "trained linear cosine/sine heading output",
                  "fit_positions": POSITIONS[:2], "heldout_positions": POSITIONS[2:],
                  "fit_field_angles": "even indices", "heldout_field_angles": "odd indices", "n_phase": args.n_phase,
                  "heldout_recovery_angles": "new combinations; both parity indices; all recovery positions held out",
                  "initial_heading_steps": model.config.initial_heading_steps, "blank_settle_steps": 40,
                  "recovery_steps": args.steps, "dt": task.config.dt,
                  "main_reduction": "sine", "diagnostics": ["local_sine", "three_harmonics"],
                  "sensory_and_recurrent_noise": 0, "heldout_trajectories_used_for_fit": False,
                  "reversed_cue_refitted": False,
                  "grid_reversal": "no-op", "explicit_heading_reversal": "reverse heading components, ordinary code unchanged"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"condition": condition, "seed": checkpoint["seed"], "coefficients": coefficients,
        "heldout_native": report["pooled_heldout_fields"]["native"],
        "heldout_native_recovery": [r["variants"]["native"] for r in recoveries]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-phase", type=int, default=72)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
