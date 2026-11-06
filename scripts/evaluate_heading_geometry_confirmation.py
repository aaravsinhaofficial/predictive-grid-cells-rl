"""Frozen single-model endpoint for independent memory-map geometry confirmation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ring_heading_model import HeadingConfig, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/research/heading_geometry_confirmation_protocol.md"
N_PHASE, LEVELS, FLOOR = 1024, 20, 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(checkpoint, allow_exploratory=False):
    expected = json.loads((ROOT / "docs/research/heading_confirmation_sources.json").read_text())["sources"]
    recorded = checkpoint.get("source_sha256", {})
    verified = {}
    for relative in ("scripts/train_ring_heading.py", "scripts/ring_heading_model.py", "scripts/ring_task_model.py", "code/codes.py"):
        matches = [value for name, value in recorded.items() if name == relative or name.endswith("/" + relative)]
        if matches != [expected[relative]] or digest(ROOT / relative) != expected[relative]:
            raise ValueError(f"Frozen training source mismatch: {relative}")
        verified[relative] = matches[0]
    if checkpoint["config"] != HeadingConfig(encoder_size=256, hidden_size=64, code_seed=checkpoint["seed"]).to_dict():
        raise ValueError("Full model/task configuration differs from protocol")
    if checkpoint["condition"] not in ("grid", "predictive") or checkpoint["steps_completed"] != 6000 or checkpoint["stopping_reason"] != "steps_completed":
        raise ValueError("Condition or training completion differs from protocol")
    if not allow_exploratory and checkpoint["seed"] not in range(5001, 5017):
        raise ValueError("Independent confirmation seed must be 5001–5016")
    if checkpoint.get("study_stage") != "confirmatory" or checkpoint.get("exploratory_arm") is not False:
        raise ValueError("Checkpoint stage differs from protocol")
    for key, expected_value in {"steps": 6000, "length": 128, "batch_size": 64, "learning_rate": .001}.items():
        if checkpoint["trainer_args"][key] != expected_value:
            raise ValueError(f"Training argument differs: {key}")
    return verified


def cue(task, angles, condition):
    zero, one = torch.zeros_like(angles), torch.ones_like(angles)
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    position = angles.new_zeros(len(angles), 2)
    return task.inputs(position, direction, zero, zero, one, one, condition)


def state_map(model, task, angles, condition, blank_steps, derivative=False):
    if derivative:
        drive, drive_d = torch.func.jvp(lambda a: model.input(cue(task, a, condition)), (angles,), (torch.ones_like(angles),))
    else:
        drive = model.input(cue(task, angles, condition))
    h = drive.new_zeros(len(angles), model.hidden_size)
    tangent = torch.zeros_like(h)
    leak = model.config.leak
    for _ in range(4):
        activation = (drive + model.recurrent(h)).tanh()
        if derivative:
            tangent = (1 - leak) * tangent + leak * (1 - activation.square()) * (drive_d + model.recurrent(tangent))
        h = (1 - leak) * h + leak * activation
    blank = model.input(drive.new_zeros(len(angles), model.input_size))
    for _ in range(blank_steps):
        activation = (blank + model.recurrent(h)).tanh()
        if derivative:
            tangent = (1 - leak) * tangent + leak * (1 - activation.square()) * model.recurrent(tangent)
        h = (1 - leak) * h + leak * activation
    return (h, tangent) if derivative else h


def distribution(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
            "mean": float(values.mean()), "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9))}


def evaluate_map(model, task, condition, blank_steps):
    phases = torch.arange(N_PHASE, dtype=torch.float64, device=task.device) * 2 * math.pi / N_PHASE
    curve, derivatives = state_map(model, task, phases, condition, blank_steps, True)
    distances = (curve.roll(-1, 0) - curve).norm(dim=-1).cpu().numpy()
    chosen = []
    for index in np.lexsort((np.arange(N_PHASE), -distances)):
        if all(min(abs(int(index) - old), N_PHASE - abs(int(index) - old)) > 2 for old in chosen):
            chosen.append(int(index))
            if len(chosen) == 4:
                break
    refinements, centers = [], []
    with torch.no_grad():
        for index in chosen:
            low, high = float(phases[index]), float(phases[index]) + 2 * math.pi / N_PHASE
            left, right = state_map(model, task, phases.new_tensor([low, high]), condition, blank_steps)
            rows = []
            for level in range(LEVELS + 1):
                rows.append({"level": level, "low_rad": low, "high_rad": high, "state_separation": float((right - left).norm())})
                if level == LEVELS:
                    break
                middle_angle = (low + high) / 2
                middle = state_map(model, task, phases.new_tensor([middle_angle]), condition, blank_steps)[0]
                if float((middle - left).norm()) >= float((right - middle).norm()):
                    high, right = middle_angle, middle
                else:
                    low, left = middle_angle, middle
            centers.append((low + high) / 2)
            refinements.append({"original_interval": index, "levels": rows, "center_rad": centers[-1]})
    center_phase = phases.new_tensor(centers)
    center_state, center_derivative = state_map(model, task, center_phase, condition, blank_steps, True)
    norms = derivatives.norm(dim=-1).cpu().numpy()
    center_norms = center_derivative.norm(dim=-1).cpu().numpy()
    all_norms, all_phase = np.r_[norms, center_norms], np.r_[phases.cpu().numpy(), centers]
    peak_index = int(np.argmax(all_norms))
    peak, median = float(all_norms[peak_index]), float(np.median(norms))
    errors = []
    for angle in [0., math.pi / 2, math.pi, 3 * math.pi / 2, float(all_phase[peak_index])]:
        scalar = phases.new_tensor(angle).requires_grad_()
        direct = torch.autograd.functional.jacobian(lambda a: state_map(model, task, a.reshape(1), condition, blank_steps)[0], scalar, vectorize=True)
        _, propagated = state_map(model, task, scalar.detach().reshape(1), condition, blank_steps, True)
        errors.append(float((direct - propagated[0]).norm() / direct.norm().clamp_min(FLOOR)))
    if max(errors) > 1e-7:
        raise ValueError(f"Independent derivative verification failed: {max(errors)}")
    output = model.readout(curve)
    output_derivative = derivatives @ model.readout.weight.T
    angles = torch.atan2(output[:, 1], output[:, 0]).cpu().numpy()
    target = phases.cpu().numpy()
    sorted_angles = np.sort(angles % (2 * np.pi))
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    radius = float((curve - curve.mean(0)).square().sum(-1).mean().sqrt())
    amplitude = output.norm(dim=-1).cpu().numpy()
    decoded_derivative = ((output[:, 0] * output_derivative[:, 1] - output[:, 1] * output_derivative[:, 0]) / output.square().sum(-1).clamp_min(FLOOR ** 2)).cpu().numpy()
    closed = state_map(model, task, phases.new_tensor([0., 2 * math.pi]), condition, blank_steps)
    report = {"input_condition": condition, "blank_steps": blank_steps, "blank_seconds": blank_steps * model.config.dt,
        "log10_peak_over_median": float(np.log10(max(peak, FLOOR) / max(median, FLOOR))),
        "peak_over_median": max(peak, FLOOR) / max(median, FLOOR), "absolute_peak_sensitivity": peak,
        "absolute_median_sensitivity": median, "peak_angle_rad": float(all_phase[peak_index]),
        "uniform_sensitivity": distribution(norms), "state_radius": radius,
        "peak_over_radius_descriptive": peak / max(radius, FLOOR), "median_over_radius_descriptive": median / max(radius, FLOOR),
        "decoded_heading_mae_deg": float(np.mean(np.abs(wrap(angles - target))) * 180 / np.pi),
        "decoded_coverage": float(1 - np.diff(np.r_[sorted_angles, sorted_angles[0] + 2 * np.pi]).max() / (2 * np.pi)),
        "decoded_winding": float(wrap(np.roll(angles, -1) - angles).sum() / (2 * np.pi)),
        "readout_amplitude": distribution(amplitude), "decoded_angular_derivative": distribution(decoded_derivative),
        "decoded_derivative_nonpositive_fraction": float(np.mean(decoded_derivative <= 0)),
        "endpoint_closure_error": float((closed[0] - closed[1]).norm()),
        "flags": {"state_radius_below_1e_minus8": radius < 1e-8, "median_sensitivity_below_1e_minus8": median < 1e-8,
                  "readout_amplitude_below_1e_minus8": bool(np.any(amplitude < 1e-8)), "primary_floor_used": peak < FLOOR or median < FLOOR},
        "independent_autograd_relative_errors": errors, "adaptive_refinements": refinements}
    archive = {"phase": target, "state": curve.cpu().numpy(), "derivative": derivatives.cpu().numpy(),
               "center_phase": np.asarray(centers), "center_state": center_state.cpu().numpy(),
               "center_derivative": center_derivative.cpu().numpy(), "decoded_phase": angles}
    if not all(np.isfinite(value).all() for value in archive.values()):
        raise FloatingPointError("Nonfinite state or derivative; do not exclude or replace this seed")
    return report, archive


def run(args):
    start = time.monotonic()
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    provenance = verify(checkpoint, args.allow_exploratory)
    model.requires_grad_(False).double()
    for key in ("K", "phi", "shift"):
        setattr(task.population, key, getattr(task.population, key).double())
    task._phase_offset = (task.population.K * task.population.phi[:, None, :]).sum(-1)
    results, archive = {}, {}
    for variant, condition in (("native", checkpoint["condition"]), ("ordinary", "grid")):
        for blank_steps in (0, 40):
            key = f"{variant}_blank{blank_steps}"
            if variant == "ordinary" and checkpoint["condition"] == "grid":
                results[key] = results[f"native_blank{blank_steps}"]
                for name, value in list(archive.items()):
                    if name.startswith(f"native_blank{blank_steps}__"):
                        archive[name.replace("native_", "ordinary_", 1)] = value
            else:
                results[key], values = evaluate_map(model, task, condition, blank_steps)
                archive.update({key + "__" + name: value for name, value in values.items()})
    sources = ("scripts/evaluate_heading_geometry_confirmation.py", PROTOCOL,
               "docs/research/heading_confirmation_sources.json", *provenance.keys())
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
        "seed": checkpoint["seed"], "condition": checkpoint["condition"], "steps_completed": checkpoint["steps_completed"],
        "stage": "exploratory_validation" if args.allow_exploratory else "independent_geometry_confirmation",
        "protocol": PROTOCOL, "source_hashes": {name: digest(ROOT / name) for name in sources},
        "verified_training_sources": provenance, "n_phase": N_PHASE, "refinement_levels": LEVELS,
        "primary": results["native_blank40"]["log10_peak_over_median"], "maps": results,
        "native_recurrence_sharpening_descriptive": results["native_blank40"]["log10_peak_over_median"] - results["native_blank0"]["log10_peak_over_median"],
        "elapsed_seconds": time.monotonic() - start, "all_models_retained": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"seed": report["seed"], "condition": report["condition"], "primary": report["primary"], "seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-exploratory", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
