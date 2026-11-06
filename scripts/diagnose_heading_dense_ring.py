"""Dense full-state ring diagnostics using exact propagated angular tangents.

This is an exploratory follow-up to the branch problem in seed2001, not a
replacement of the original cohort or a rule for excluding any seed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from diagnose_heading_memory_connectedness import cue_input, decode, geometry, make_double, prepare
from evaluate_heading_memory import eigenspace_alignment
from evaluate_ring_dynamics import jacobian
from ring_dynamics import project_to_ring, ring_radius, wrap
from ring_heading_model import load_checkpoint


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "q01": float(np.quantile(values, .01)),
            "q10": float(np.quantile(values, .1)), "median": float(np.median(values)),
            "q90": float(np.quantile(values, .9)), "q99": float(np.quantile(values, .99)),
            "max": float(values.max()), "mean": float(values.mean())}


def angular_curve(model, task, phase, condition):
    # One directional JVP yields each independent batch element's exact input
    # derivative; recurrent tangent propagation below uses the chain rule.
    drive, drive_derivative = torch.func.jvp(lambda a: model.input(cue_input(task, a, condition)),
        (phase,), (torch.ones_like(phase),))
    h = drive.new_zeros(len(phase), model.hidden_size)
    derivative = torch.zeros_like(h)
    leak = model.config.leak
    for _ in range(model.config.initial_heading_steps):
        activation = (drive + model.recurrent(h)).tanh()
        derivative = (1 - leak) * derivative + leak * (1 - activation.square()) * (drive_derivative + model.recurrent(derivative))
        h = (1 - leak) * h + leak * activation
    zero = drive.new_zeros(len(phase), model.input_size)
    blank_drive = model.input(zero)
    for _ in range(40):
        activation = (blank_drive + model.recurrent(h)).tanh()
        derivative = (1 - leak) * derivative + leak * (1 - activation.square()) * model.recurrent(derivative)
        h = (1 - leak) * h + leak * activation
    return h.detach(), derivative.detach()


def near_polyline(points, curve):
    """Candidate segments from eight nearest vertices; independently checked below."""
    n = len(curve)
    _, vertices = cKDTree(curve).query(points, k=min(8, n))
    segments = np.concatenate((vertices, (vertices - 1) % n), axis=1)
    start = curve[segments]
    direction = curve[(segments + 1) % n] - start
    delta = points[:, None] - start
    fraction = np.clip((delta * direction).sum(-1) / np.maximum((direction ** 2).sum(-1), 1e-30), 0, 1)
    distance = np.linalg.norm(delta - fraction[..., None] * direction, axis=-1)
    return distance.min(1)


@torch.no_grad()
def perturbations(model, curve, derivative, size=.1):
    initial = curve.cpu().numpy()
    tangent = derivative / derivative.norm(dim=-1, keepdim=True)
    generator = torch.Generator(device=curve.device).manual_seed(419)
    normal = torch.randn(curve.shape, dtype=curve.dtype, device=curve.device, generator=generator)
    normal -= (normal * tangent).sum(-1, keepdim=True) * tangent
    normal /= normal.norm(dim=-1, keepdim=True)
    amplitude = size * ring_radius(initial)
    state = torch.stack((curve, curve + amplitude * tangent, curve + amplitude * normal))
    shape = state.shape
    state = state.reshape(-1, model.hidden_size)
    drive = model.input(torch.zeros(len(state), model.input_size, dtype=state.dtype, device=state.device))
    saved = {}
    for step in range(101):
        if step in (0, 20, 100):
            saved[step] = state.reshape(shape).cpu().numpy().copy()
        if step < 100:
            state = (1 - model.config.leak) * state + model.config.leak * (drive + model.recurrent(state)).tanh()
    output = {}
    initial_base = decode(model, curve).cpu().numpy()
    initial_tangent = decode(model, torch.as_tensor(saved[0][1], dtype=curve.dtype, device=curve.device)).cpu().numpy()
    delta0 = wrap(initial_tangent - initial_base)
    normal0 = near_polyline(saved[0][2], saved[0][0])
    archive = {}
    for step, path in saved.items():
        base, tangent_path, normal_path = path
        distance = near_polyline(normal_path, base)
        selected = np.linspace(0, len(base) - 1, 64).astype(int)
        exact, _ = project_to_ring(normal_path[selected], base)
        error = float(np.max(np.abs(distance[selected] - exact)))
        if error > 1e-8:
            # Preserve exactness if nearest-vertex candidates miss a segment.
            distance, _ = project_to_ring(normal_path, base)
        base_angle = decode(model, torch.as_tensor(base, dtype=curve.dtype, device=curve.device)).cpu().numpy()
        tangent_angle = decode(model, torch.as_tensor(tangent_path, dtype=curve.dtype, device=curve.device)).cpu().numpy()
        gain = np.abs(wrap(tangent_angle - base_angle)) / np.maximum(np.abs(delta0), 1e-10)
        output[str(step)] = {"seconds_after_perturbation": step * model.config.dt,
            "normal_remaining_mean_distance_ratio": float(distance.mean() / normal0.mean()),
            "normal_remaining_per_phase": describe(distance / np.maximum(normal0, 1e-10)),
            "tangent_phase_gain_per_phase": describe(gain), "initial_resolvable_fraction": float((np.abs(delta0) >= 1e-5).mean()),
            "normal_projection_check_max_error": error, "curve_radius_retained": ring_radius(base) / ring_radius(initial),
            "baseline_geometry": geometry(model, torch.as_tensor(base, dtype=curve.dtype, device=curve.device),
                                           np.arange(len(base)) * 2 * np.pi / len(base))}
        archive[f"step{step}_states"] = path
        archive[f"step{step}_normal_remaining"] = distance / np.maximum(normal0, 1e-10)
        archive[f"step{step}_phase_gain"] = gain
    return output, archive


@torch.no_grad()
def spectra(model, curve, derivative):
    rows = []
    tangent = derivative.cpu().numpy().copy()
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    for start in range(0, len(curve), 64):
        matrices = jacobian(model, curve[start:start + 64]).cpu().numpy()
        for offset, matrix in enumerate(matrices):
            tan = tangent[start + offset]
            eigen, vectors = np.linalg.eig(matrix)
            slow, dominant = np.argmin(np.abs(eigen - 1)), np.argmax(np.abs(eigen))
            normal = np.linalg.qr(tan[:, None], mode="complete")[0][:, 1:]
            transverse = normal.T @ matrix @ normal
            rows.append({"spectral_radius": float(np.abs(eigen).max()),
                "slow_eigenvalue_real": float(eigen[slow].real), "slow_eigenvalue_imag": float(eigen[slow].imag),
                "slow_distance_to_one": float(abs(eigen[slow] - 1)),
                "slow_alignment_to_exact_tangent": eigenspace_alignment(vectors[:, slow], tan),
                "dominant_alignment_to_exact_tangent": eigenspace_alignment(vectors[:, dominant], tan),
                "normal_spectral_radius": float(np.abs(np.linalg.eigvals(transverse)).max()),
                "tangent_gain": float(tan @ matrix @ tan)})
    return {"per_phase": rows, "distributions": {key: describe([r[key] for r in rows]) for key in rows[0]}}


def run(args):
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    make_double(model, task)
    phase = torch.arange(args.n_phase, dtype=torch.float64, device=task.device) * 2 * math.pi / args.n_phase
    curve, derivative = angular_curve(model, task, phase, checkpoint["condition"])
    output = model.readout(curve)
    d_output = derivative @ model.readout.weight.T
    phase_derivative = ((output[:, 0] * d_output[:, 1] - output[:, 1] * d_output[:, 0]) / output.square().sum(-1)).cpu().numpy()
    # The propagated tangent is checked against independent finite differences.
    index = torch.tensor(np.linspace(0, args.n_phase - 1, 16).astype(int), device=task.device)
    epsilon = 1e-6
    numerical = (prepare(model, task, phase[index] + epsilon, checkpoint["condition"])
                 - prepare(model, task, phase[index] - epsilon, checkpoint["condition"])) / (2 * epsilon)
    relative_error = float(((numerical - derivative[index]).norm(dim=-1) / derivative[index].norm(dim=-1).clamp_min(1e-10)).max())
    if relative_error > 1e-4:
        raise ValueError(f"Angular tangent finite difference check failed:{relative_error}")
    closed = prepare(model, task, phase.new_tensor([0., 2 * math.pi]), checkpoint["condition"])
    geometry_result = geometry(model, curve, phase.cpu().numpy())
    geometry_result["endpoint_closure_error"] = float((closed[0] - closed[1]).norm())
    geometry_result["exact_angular_derivative_norm"] = describe(derivative.norm(dim=-1).cpu().numpy())
    geometry_result["decoded_angular_derivative"] = describe(phase_derivative)
    geometry_result["decoded_angular_derivative_negative_fraction"] = float((phase_derivative < 0).mean())
    perturbation_result, archive = perturbations(model, curve, derivative)
    spectral_result = spectra(model, curve, derivative)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "condition": checkpoint["condition"], "seed": checkpoint["seed"], "n_phase": args.n_phase,
        "geometry": geometry_result, "perturbations": perturbation_result, "spectra": spectral_result,
        "tangent_finite_difference_max_relative_error": relative_error,
        "protocol": {"exploratory_followup": True, "all_phases_included": True, "state_dimensions": model.hidden_size,
            "phase_coordinate": "initial heading angle; readout monotonicity checked separately",
            "tangent": "exact input angular JVP propagated through recurrent chain rule",
            "initialization": "same four-step cue plus40blank as frozen cohort, float64 diagnostic arithmetic",
            "perturbation_size_radius": .1, "normal_reference": "dense time-matched curve",
            "selection": "first subsequent PGCseed2002 after branch finding in preselectedseed2001; exploratory candidate, no cohort seed excluded",
            "claim_limit": "dense finite-time diagnostic, not a proof of an exact continuous attractor or all-seed ring topology"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    archive.update(phase=phase.cpu().numpy(), curve=curve.cpu().numpy(), exact_derivative=derivative.cpu().numpy(),
                   decoded_phase_derivative=phase_derivative)
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"condition": checkpoint["condition"], "seed": checkpoint["seed"], "geometry": geometry_result,
        "spectra": spectral_result["distributions"], "perturbations": perturbation_result}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-phase", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
