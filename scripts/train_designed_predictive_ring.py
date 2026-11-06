"""Train and evaluate the explicitly designed-ring constructive mechanism.

Designed: 64D ring memory and fixed navigation controller.
Learned: spatial-phase readouts and cue gain, using independent calibration and
heading trajectories. No navigation evaluation target enters either fit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from designed_predictive_ring import DesignedRing, KINDS, GAIN_GRID, fit_decoders, fit_gains, analytic_spectrum
from evaluate_ring_heading_navigation import (HoldingScores, controller, integrate_position,
                                             observation_inputs, realization)
from ring_task_model import wrap_angle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/"results/ring_attractor/designed_compass"
VARIANTS = [("grid", "grid", "native"), ("pgc", "predictive", "native"),
            ("explicit", "grid_heading", "native"), ("pgc_ordinary", "predictive", "ordinary"),
            ("pgc_blank", "predictive", "blank"), ("pgc_reverse", "predictive", "reverse"),
            ("pgc_restored", "predictive", "restored")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    return {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
        ROOT/"scripts/designed_predictive_ring.py", ROOT/"scripts/decode_predictive_phase_difference.py",
        ROOT/"scripts/ring_heading_model.py", ROOT/"scripts/evaluate_ring_heading_navigation.py",
        ROOT/"scripts/ring_task_model.py", ROOT/"code/codes.py"]}


def protocol():
    return {"architecture": "DESIGNED full64D attracting unit circle: one neutral phase,63 transverse multipliers rho=.90",
            "learned": "linear spatial-phase readouts and a scalar cue gain; no recurrent-ring geometry learned",
            "additional_designed_operations": "complex spatial-phase subtraction, known wave-vector pseudoinverse, and sinusoidal phase-correction law",
            "calibration_positions": 4096, "calibration_seed_rule": "model_seed+95311",
            "gain_training": {"seed_rule": "model_seed+95321", "episodes": 128, "steps": 256,
                              "objective": "mean1-cos(headingerror) after first4 observations", "candidates": list(GAIN_GRID)},
            "cohort_seeds": list(range(4001, 4009)), "pilot_seed": 3999,
            "navigation_episodes_per_seed": 128, "navigation_seconds": 20, "navigation_seed": 95401,
            "navigation_primary": {"position_translation_m": 10., "initial_heading_error_degrees": 20.,
                                   "bias_multiplier": 1., "blackout_seconds": [4, 7],
                                   "restore_seconds": 2., "recurrent_noise_sd": .01},
            "navigation_secondary_nominal": {"position_translation_m": 0., "initial_heading_error_degrees": 0.,
                                             "other_conditions_identical": True},
            "primary_outcome": "success_fraction by20s; capture_radius=.15m; all timeouts included",
            "primary_family": ["pgc-grid", "pgc-pgc_ordinary", "pgc_restored-pgc_ordinary", "pgc-pgc_reverse"],
            "inference": "all8 seedpaired exact2^8 sign flips, two-sided, Holm across4 comparisons;95% paired-t CI descriptive",
            "information_control": "explicit same measured heading used by PGC generator, same visibility/noise/initialization",
            "causal_control": "frozen PGC decoder/gain; ordinary code, zero spatial availability, reversed shift, ordinary then restoration",
            "controller": "existing fixed controller uses decoded heading, integrated measured speed, known reset position and goal only",
            "scope": "instantaneous ordinary-grid phase decoder; no claim against an optimal temporal grid decoder",
            "state_noise": "identical additive64D recurrent noise across all variants within seed",
            "no_exclusions": True}


@torch.no_grad()
def navigation(models, task, seed=95401, episodes=128, seconds=20,
               position_offset=10., initial_error_degrees=20.):
    c = task.config
    length = round(seconds/c.dt)
    noise = realization(task, seed, length, episodes)
    generator = torch.Generator().manual_seed(seed+21)
    recurrent_noise = c.recurrent_noise*torch.randn(length, episodes, 64, generator=generator)
    starts, goals = noise["starts"].clone(), noise["goals"].clone()
    starts[:, 0] += position_offset
    goals[:, 0] += position_offset
    base_heading = noise["initial_heading"]
    zeros, ones = torch.zeros(episodes), torch.ones(episodes)
    # Identical measured initial heading observations for every trained input
    # condition and intervention. No model-specific cue gain changes warmup.
    initial_cues = []
    for warm in range(4):
        x, _ = observation_inputs(task, starts, base_heading, zeros, zeros, ones, ones,
            "grid", noise, warm, 1., initial_error_degrees)
        initial_cues.append(x[:, -3:-1])
    initial_direction = torch.stack(initial_cues).mean(0)
    h0 = models["predictive"].initialize(initial_direction)
    state = h0[None].expand(len(VARIANTS), -1, -1).clone()
    position = starts[None].expand(len(VARIANTS), -1, -1).clone()
    estimated_position = position.clone()
    heading = base_heading[None].expand(len(VARIANTS), -1).clone()
    angular, speed = torch.zeros_like(heading), torch.zeros_like(heading)
    first_arrival = torch.full_like(heading, length+1, dtype=torch.long)
    holding = HoldingScores(heading, c.dt, length, .15)
    phase_errors, state_radius, traces = [], [], []
    for t in range(length):
        measured_speeds = []
        for v, (_, kind, intervention) in enumerate(VARIANTS):
            if intervention == "ordinary" or (intervention == "restored" and t*c.dt < 2.):
                condition = "grid"
            elif intervention == "reverse":
                condition = "reverse"
            else:
                condition = kind
            visible = zeros if intervention == "blank" else noise["visible"][t]
            x, measured_speed = observation_inputs(task, position[v], heading[v], angular[v], speed[v],
                visible, zeros, condition, noise, t+4)
            state[v] = models[kind].step(x, state[v])+recurrent_noise[t]
            measured_speeds.append(measured_speed)
        output = models["predictive"].readout(state)  # Same fixed basis in all models.
        decoded = torch.atan2(output[..., 1], output[..., 0])
        phase_errors.append(wrap_angle(decoded-heading).cpu().numpy())
        state_radius.append(output.norm(dim=-1).cpu().numpy())
        if t:
            estimated_position = integrate_position(estimated_position, decoded, torch.stack(measured_speeds), c.dt)
        angular_command, forward_command = controller(estimated_position, decoded, goals[None])
        before = position.clone()
        stop = forward_command == 0
        angular = (0.8*angular+0.2*angular_command+0.03*noise["motor_angular_noise"][t][None]).masked_fill(stop, 0)
        speed = (0.8*speed+0.2*forward_command+0.005*noise["motor_speed_noise"][t][None]).clamp(.001, .35).masked_fill(stop, 0)
        heading = wrap_angle(heading+c.dt*angular)
        position += c.dt*speed[..., None]*torch.stack((heading.cos(), heading.sin()), -1)
        distance = (position-goals[None]).norm(dim=-1)
        hit = (distance <= .15) & (first_arrival > length)
        first_arrival[hit] = t+1
        holding.update(distance, speed, first_arrival <= t+1)
        if t % 4 == 0:
            traces.append(torch.stack((position[:, :16], estimated_position[:, :16]), -2).numpy())
    rows, arrays = {}, {"phase_error": np.stack(phase_errors), "phase_radius": np.stack(state_radius),
                        "position_traces": np.stack(traces), "starts": starts.numpy(), "goals": goals.numpy(),
                        "initial_state": h0.numpy()}
    for v, (name, _, _) in enumerate(VARIANTS):
        success = first_arrival[v] <= length
        secondary, secondary_arrays = holding.summarize(v, success)
        rows[name] = {"success_fraction": float(success.float().mean()),
            "mean_capped_arrival_seconds": float(first_arrival[v].clamp_max(length).float().mean()*c.dt),
            "mean_final_goal_distance_m": float(distance[v].mean()),
            "mean_self_position_error_m": float((estimated_position[v]-before[v]).norm(dim=-1).mean()),
            "heading_mae_degrees": float(np.mean(np.abs(arrays["phase_error"][:, v]))*180/np.pi), **secondary}
        arrays[name+"_success"] = success.numpy()
        arrays[name+"_capped_arrival"] = first_arrival[v].clamp_max(length).float().numpy()*c.dt
        arrays.update({name+"_"+k: value.numpy() for k, value in secondary_arrays.items()})
    return rows, arrays


def dynamical_validation(model, task, seed):
    """Full-state continuity, actual-code perturbations, and64×64 Jacobians."""
    angle = torch.arange(180)*2*math.pi/180
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    h0 = model.initialize(direction)
    zero = torch.zeros(180)
    q0 = torch.zeros(180, 2)
    with torch.no_grad():
        h = h0.clone()
        for _ in range(200):
            h = model.step_cue(h, zero, q0)
        blank_drift = float((h-h0).norm(dim=-1).max())
        adjacent = (h0.roll(-1, 0)-h0).norm(dim=-1)
        rng = np.random.default_rng(seed+95341)
        pos = torch.tensor(rng.uniform(8.75, 11.25, (180, 2)), dtype=torch.float32)
        native_rates = task.encode(pos, direction, "predictive")
        ordinary_rates = task.encode(pos, direction, "grid")
        reverse_rates = task.encode(pos, direction, "reverse")
        actual_cues = {"blank": q0, "native": model.decoder(native_rates),
                      "ordinary": model.decoder(ordinary_rates), "reverse": model.decoder(reverse_rates)}
        perturbed_direction = torch.stack(((angle+.1745329252).cos(), (angle+.1745329252).sin()), -1)
        initial_perturbed = model.initialize(perturbed_direction)
        phase_traces = {}
        for name, cue in actual_cues.items():
            states = initial_perturbed.clone()
            values = []
            for _ in range(100):
                states = model.step_cue(states, zero, cue)
                output = model.readout(states)
                values.append(wrap_angle(torch.atan2(output[:, 1], output[:, 0])-angle).numpy())
            phase_traces[name] = np.stack(values)
        # Transverse disturbance includes both radius and all62 extra dimensions.
        normal = torch.randn(h0.shape, generator=torch.Generator().manual_seed(seed+95342))
        tangent = torch.stack((-angle.sin(), angle.cos()), -1)@model.basis.T
        normal -= (normal*tangent).sum(-1, keepdim=True)*tangent
        normal *= .1/normal.norm(dim=-1, keepdim=True)
        disturbed = h0+normal
        normal_trace = []
        for _ in range(50):
            disturbed = model.step_cue(disturbed, zero, q0)
            normal_trace.append(((disturbed-h0).norm(dim=-1)/normal.norm(dim=-1)).numpy())
    # Double precision for the independent numerical spectrum check. Recreate
    # an exactly orthonormal basis to avoid accumulating float32 roundoff.
    probe = DesignedRing(seed, None, "predictive", model.gain).double()
    probe.basis.copy_(torch.linalg.qr(probe.basis).Q)
    spectra, max_errors = {}, {}
    actual_q = actual_cues["native"][:1].double()
    for condition, q in (("blank", torch.zeros(1, 2, dtype=torch.float64)),
                         ("anchored", torch.tensor([[1., 0.]], dtype=torch.float64)),
                         ("actual_decoded_cue", actual_q)):
        cue_magnitude = float(q.norm())
        aligned = q if cue_magnitude else torch.tensor([[1., 0.]], dtype=torch.float64)
        state = probe.initialize(aligned)[0]
        jac = torch.autograd.functional.jacobian(lambda x: probe.step_cue(x[None], torch.zeros(1, dtype=torch.float64), q)[0], state).numpy()
        eig = np.linalg.eigvals(jac)
        expected = analytic_spectrum(model.gain, cue_magnitude=cue_magnitude)
        max_errors[condition] = float(np.max(np.abs(np.sort(eig.real)-np.sort(expected))))
        spectra[condition] = {"real": eig.real.tolist(), "imaginary": eig.imag.tolist(), "predicted": expected.tolist()}
        if max_errors[condition] > 1e-9 or np.max(np.abs(eig.imag)) > 1e-9:
            raise ValueError("Full-state spectrum disagrees with analytic design")
    if blank_drift > 5e-5:
        raise ValueError("Designed autonomous ring failed finite-precision memory check")
    result = {"max_full_state_drift_over10s_blank": blank_drift,
              "adjacent_chord_cv": float(adjacent.std()/adjacent.mean()),
              "max_spectral_prediction_error": max_errors, "spectra": spectra,
              "normal_gain_after50steps": float(np.mean(normal_trace[-1])),
              "normal_gain_prediction": model.rho**50,
              "phase_error_after5s_degrees": {k: float(np.abs(v[-1]).mean()*180/np.pi) for k, v in phase_traces.items()},
              "intrinsic_ring_geometry_is_designed": True}
    arrays = {"ring_states": h0.numpy(), "normal_gain_trace": np.stack(normal_trace),
              **{"phase_"+k: v for k, v in phase_traces.items()}}
    return result, arrays


def run(args):
    torch.set_num_threads(2)
    dest = args.out/f"seed{args.seed}"
    if (dest/"summary.json").exists():
        raise FileExistsError("Do not overwrite a completed constructed-ring seed")
    dest.mkdir(parents=True, exist_ok=True)
    if args.seed != 3999:
        frozen = json.loads((args.out/"frozen_protocol.json").read_text())
        if frozen["sources"] != source_manifest() or frozen["protocol"] != protocol():
            raise ValueError("Constructive-ring frozen protocol/source mismatch")
        if args.seed not in protocol()["cohort_seeds"]:
            raise ValueError("Seed outside constructive cohort")
    start = time.monotonic()
    task, decoders, decoder_validation = fit_decoders(args.seed)
    models, training = fit_gains(args.seed, task, decoders)
    print(json.dumps({"seed": args.seed, "training": training, "elapsed": time.monotonic()-start}), flush=True)
    dynamics, dynamics_arrays = dynamical_validation(models["predictive"], task, args.seed)
    rows, nav_arrays = navigation(models, task)
    nominal_rows, nominal_arrays = navigation(models, task, position_offset=0., initial_error_degrees=0.)
    result = {"seed": args.seed, "protocol": protocol(), "sources": source_manifest(),
              "runtime_versions": {"torch": str(torch.__version__), "numpy": np.__version__},
              "configuration": task.config.to_dict(), "training": training,
              "decoder_validation": decoder_validation, "dynamics": dynamics,
              "navigation": rows, "navigation_nominal": nominal_rows,
              "elapsed_seconds": time.monotonic()-start}
    torch.save({"seed": args.seed, "configuration": task.config.to_dict(),
                "states": {k: m.state_dict() for k, m in models.items()},
                "gains": {k: m.gain for k, m in models.items()}, "sources": source_manifest(),
                "designed_ring": True}, dest/"model.pt")
    np.savez_compressed(dest/"arrays.npz", **dynamics_arrays, **nav_arrays,
                        **{"nominal_"+k: v for k, v in nominal_arrays.items()})
    (dest/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"seed": args.seed, "elapsed_seconds": result["elapsed_seconds"], "dynamics": {k: v for k, v in dynamics.items() if k != "spectra"}, "navigation": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    run(parser.parse_args())
