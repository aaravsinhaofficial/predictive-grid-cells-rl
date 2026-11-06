"""Resolve hidden-state transitions concealed by circular heading readouts.

This diagnostic never substitutes for or overwrites the frozen memory cohort.
Finite-time continuity, attraction and a useful circular readout are separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_ring_dynamics import jacobian
from ring_dynamics import ring_radius, wrap
from ring_heading_model import load_checkpoint


def make_double(model, task):
    model.double()
    for key in ("K", "phi", "shift"):
        setattr(task.population, key, getattr(task.population, key).double())
    task._phase_offset = (task.population.K * task.population.phi[:, None, :]).sum(-1)


def cue_input(task, angles, condition, initial=True, position=None, angular=0., speed=0.):
    zero, one = torch.zeros_like(angles), torch.ones_like(angles)
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    if position is None:
        position = torch.zeros(len(angles), 2, dtype=angles.dtype, device=angles.device)
    return task.inputs(position, direction, zero + angular, zero + speed, one,
                       one if initial else zero, condition)


def prepare(model, task, angles, condition, blank_steps=40):
    x = cue_input(task, angles, condition)
    h = x.new_zeros(len(angles), model.hidden_size)
    for _ in range(model.config.initial_heading_steps):
        h = model.step(x, h)
    blank = x * 0
    for _ in range(blank_steps):
        h = model.step(blank, h)
    return h


def decode(model, h):
    y = model.readout(h)
    return torch.atan2(y[..., 1], y[..., 0])


def geometry(model, h, phase):
    state = h.detach().cpu().numpy()
    actual = decode(model, h).detach().cpu().numpy()
    phase = np.asarray(phase)
    differences = np.roll(state, -1, axis=0) - state
    distance = np.linalg.norm(differences, axis=1)
    order = np.argsort(distance)[::-1]
    readout = model.readout.weight.detach().cpu().numpy()
    projector = readout.T @ np.linalg.solve(readout @ readout.T, readout)
    null_fraction = 1 - np.sum((differences @ projector) ** 2, axis=1) / np.maximum(distance ** 2, 1e-30)
    sorted_angle = np.sort(np.mod(actual, 2 * np.pi))
    singular = np.linalg.svd(state - state.mean(0), compute_uv=False)
    return {"n_phase": len(state), "state_radius": ring_radius(state), "adjacent_distance_median": float(np.median(distance)),
        "adjacent_distance_max": float(distance.max()), "max_to_median": float(distance.max() / np.median(distance)),
        "longest_segment_fraction_of_polygon_length": float(distance.max() / distance.sum()),
        "polygon_length": float(distance.sum()), "largest_intervals": [{"index": int(i),
            "initial_angle_deg": float(phase[i] * 180 / np.pi), "distance": float(distance[i]),
            "readout_nullspace_fraction": float(null_fraction[i]),
            "decoded_phase_difference_deg": float(wrap(actual[(i + 1) % len(state)] - actual[i]) * 180 / np.pi)} for i in order[:4]],
        "decoded_heading_mae_deg": float(np.mean(np.abs(wrap(actual - phase))) * 180 / np.pi),
        "decoded_coverage": float(1 - np.diff(np.r_[sorted_angle, sorted_angle[0] + 2 * np.pi]).max() / (2 * np.pi)),
        "decoded_winding": float(wrap(np.roll(actual, -1) - actual).sum() / (2 * np.pi)),
        "pc3_variance_fraction": float(np.sum(singular[:3] ** 2) / np.sum(singular ** 2))}


@torch.no_grad()
def adaptive_transition(model, task, condition, low, high, levels=20):
    bounds = torch.tensor([low, high], dtype=torch.float64, device=task.device)
    left, right = prepare(model, task, bounds, condition)
    rows, angles, states = [], [low, high], [left.cpu().numpy(), right.cpu().numpy()]
    for level in range(levels + 1):
        x = cue_input(task, bounds, condition)
        gap = float((right - left).norm())
        rows.append({"level": level, "low_rad": float(bounds[0]), "high_rad": float(bounds[1]),
                     "width_deg": float((bounds[1] - bounds[0]) * 180 / math.pi), "state_gap": gap,
                     "input_gap": float((x[1] - x[0]).norm()),
                     "state_change_per_radian": gap / float(bounds[1] - bounds[0]),
                     "decoded_angle_difference_deg": float(wrap((decode(model, right) - decode(model, left)).cpu().numpy()) * 180 / np.pi)})
        if level == levels:
            break
        midpoint = bounds.mean().reshape(1)
        middle = prepare(model, task, midpoint, condition)[0]
        angles.append(float(midpoint[0])); states.append(middle.cpu().numpy())
        if (middle - left).norm() > (right - middle).norm():
            bounds[1] = midpoint[0]; right = middle
        else:
            bounds[0] = midpoint[0]; left = middle
    return rows, np.array(angles), np.array(states)


def angular_derivative(model, task, condition, angle):
    a = torch.tensor(angle, dtype=torch.float64, device=task.device, requires_grad=True)
    function = lambda value: prepare(model, task, value.reshape(1), condition)[0]
    state = function(a)
    derivative = torch.autograd.functional.jacobian(function, a, vectorize=True)
    input_derivative = torch.autograd.functional.jacobian(
        lambda value: cue_input(task, value.reshape(1), condition)[0], a, vectorize=True)
    tangent = derivative.detach().cpu().numpy().copy()
    tangent /= np.linalg.norm(tangent)
    matrix = jacobian(model, state[None]).detach().cpu().numpy()[0]
    eigen, vectors = np.linalg.eig(matrix)
    slow, dominant = np.argmin(np.abs(eigen - 1)), np.argmax(np.abs(eigen))
    def alignment(index):
        vector = vectors[:, index]
        return float(np.abs(tangent @ vector) / np.linalg.norm(vector))
    output = model.readout(state)
    d_output = model.readout.weight @ derivative
    d_phase = (output[0] * d_output[1] - output[1] * d_output[0]) / output.square().sum()
    return {"initial_angle_rad": angle, "initial_angle_deg": angle * 180 / np.pi,
        "state_derivative_norm_per_radian": float(derivative.norm()), "input_derivative_norm_per_radian": float(input_derivative.norm()),
        "decoded_phase_derivative": float(d_phase.detach()), "decoded_heading_deg": float(decode(model, state).detach() * 180 / math.pi),
        "spectral_radius": float(np.abs(eigen).max()), "eigenvalues_above_one": int((np.abs(eigen) > 1 + 1e-8).sum()),
        "slow_eigenvalue_real": float(eigen[slow].real), "slow_eigenvalue_imag": float(eigen[slow].imag),
        "slow_alignment_to_autograd_tangent": alignment(slow), "dominant_alignment_to_autograd_tangent": alignment(dominant)}


@torch.no_grad()
def common_history(model, task, condition, n_phase=512, cycles=3, mode="moving_circle", clockwise=False):
    direction_sign = -1 if clockwise else 1
    omega = direction_sign * 2 * math.pi / (n_phase * model.config.dt)
    speed = .15 if mode == "moving_circle" else 0.
    state = prepare(model, task, torch.tensor([0.], dtype=torch.float64, device=task.device), condition, blank_steps=0)
    position = torch.zeros(1, 2, dtype=torch.float64, device=task.device)
    traces, phases, positions = [], [], []
    start_last = None
    for step in range(cycles * n_phase):
        angle = torch.tensor([omega * model.config.dt * (step + 1)], dtype=torch.float64, device=task.device)
        direction = torch.stack((angle.cos(), angle.sin()), -1)
        position += model.config.dt * speed * direction
        x = cue_input(task, angle, condition, initial=mode == "full_heading_cue_sweep", position=position,
                      angular=omega, speed=speed)
        if step == (cycles - 1) * n_phase:
            start_last = state.clone()
        state = model.step(x, state)
        if step >= (cycles - 1) * n_phase:
            traces.append(state[0].cpu().numpy()); phases.append(float(angle[0] % (2 * math.pi)));positions.append(position[0].cpu().numpy().copy())
    before = np.array(traces)
    after = torch.as_tensor(before, dtype=torch.float64, device=task.device)
    blank = torch.zeros(n_phase, model.input_size, dtype=torch.float64, device=task.device)
    for _ in range(40):
        after = model.step(blank, after)
    before_tensor = torch.as_tensor(before, dtype=torch.float64, device=task.device)
    result = {"mode": mode, "clockwise": clockwise, "cycles": cycles, "n_phase": n_phase,
        "omega_rad_per_second": omega, "speed_m_per_second": speed,
        "full_state_cycle_closure_error": float((state - start_last).norm()),
        "full_state_cycle_closure_error_radius": float((state - start_last).norm()) / ring_radius(before),
        "before_blank": geometry(model, before_tensor, phases), "after_blank": geometry(model, after, phases)}
    return result, {"phase": np.array(phases), "position": np.array(positions), "before_blank": before,
                    "after_blank": after.cpu().numpy()}


def run(args):
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    make_double(model, task)
    condition = checkpoint["condition"]
    phase = torch.arange(args.n_phase, dtype=torch.float64, device=task.device) * 2 * math.pi / args.n_phase
    h = prepare(model, task, phase, condition, blank_steps=0)
    blank = torch.zeros(args.n_phase, model.input_size, dtype=torch.float64, device=task.device)
    archive, geometry_rows = {"phase": phase.cpu().numpy()}, []
    saved = {}
    with torch.no_grad():
        for step in range(641):
            if step in (0, 10, 20, 40, 100, 140, 640):
                saved[step] = h.clone()
                row = geometry(model, h, phase.cpu().numpy()); row["blank_steps"] = step
                geometry_rows.append(row); archive[f"direct_blank{step}"] = h.cpu().numpy()
            if step < 640:
                h = model.step(blank, h)
    index = geometry_rows[3]["largest_intervals"][0]["index"]
    low, high = float(phase[index]), float(phase[index]) + 2 * math.pi / args.n_phase
    adaptive, angles, states = adaptive_transition(model, task, condition, low, high)
    archive["adaptive_angles"] = angles; archive["adaptive_states"] = states
    center = (adaptive[-1]["low_rad"] + adaptive[-1]["high_rad"]) / 2
    probe_angles = [0., math.pi / 2, low, high, center]
    derivatives = [angular_derivative(model, task, condition, angle) for angle in probe_angles]
    history_rows = []
    for mode in ("moving_circle", "full_heading_cue_sweep"):
        for clockwise in (False, True):
            result, arrays = common_history(model, task, condition, mode=mode, clockwise=clockwise)
            history_rows.append(result)
            archive.update({f"{mode}_{'cw' if clockwise else 'ccw'}_{key}": value for key, value in arrays.items()})
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "condition": condition, "seed": checkpoint["seed"],
        "direct_initialization": geometry_rows, "adaptive_largest_transition": adaptive,
        "autograd_angular_derivatives": derivatives, "common_history": history_rows,
        "protocol": {"diagnostic_only": True, "frozen_cohort_unchanged": True,
            "precision": "float64 arithmetic from unchanged stored weights; analytical code constants promoted to float64",
            "direct_initialization": "four native heading-cue steps fromzero followed by all-zero input",
            "moving_circle": "three continuous forward circular trajectories with native codes, exact angular velocity/speed, heading cue only atreset",
            "full_heading_cue_sweep": "diagnostic continuation with heading cue heldavailable atfixedposition; outside normal initialization protocol",
            "history_curve": "last fullcycle sampled then each recorded state independently given40blank steps",
            "warning": "finite-time connectedness follows continuity; it doesnot imply a globally attracting ring"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"condition": condition, "transition": adaptive[-1], "derivatives": derivatives,
                      "common_history": history_rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-phase", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
