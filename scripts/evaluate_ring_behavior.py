"""Causal input tests and navigation for the controlled trained ring estimator.

The controller sees the learned estimate and the externally specified goal.
True position is used only by the simulator and outcome measurements. Every
intervention uses frozen model weights and retains all spatial input channels.
Closed-loop branches share starts, goals, observation/noise schedules and pulse
draws, while their movements and resulting observations can diverge.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ring_task_model import load_checkpoint, phase_error, wrap_angle


VARIANTS = ("native", "ordinary", "reverse", "restored")


def input_condition(variant, native_condition, step, restore_step):
    if variant == "native":
        return native_condition
    if variant == "ordinary":
        return "grid"
    if variant == "reverse":
        return "reverse"
    if variant == "restored":
        return native_condition if step >= restore_step else "grid"
    raise ValueError(variant)


@torch.no_grad()
def forced_replay(model, task, native_condition, seed=874231, length=320,
                  batch_size=256, restore_step=40):
    generator = torch.Generator(device=task.device).manual_seed(seed)
    _, target, data = task.batch(length, batch_size, generator, native_condition)
    rows, archive = {}, {}
    elapsed = torch.arange(length, device=task.device)[:, None]
    eligible = elapsed >= 8
    masks = {"all": eligible.expand(length, batch_size),
             "moving": (data["velocity"].abs() > 0.1) & eligible,
             "blank": (data["visible"] == 0) & eligible,
             "stationary_blank": data["stopped"] & eligible}
    for variant in VARIANTS:
        state = target.new_zeros(batch_size, model.hidden_size)
        output, states = [], []
        for t in range(length):
            condition = input_condition(variant, native_condition, t, restore_step)
            x = task.inputs(data["delayed_phase"][t], data["observed_velocity"][t],
                            data["visible"][t], condition, data["delayed_velocity"][t],
                            data["shuffled_velocity"][t], data["channel_noise"][t])
            state = model.step(x, state)
            states.append(state)
            output.append(model.readout(state))
        output, states = torch.stack(output), torch.stack(states)
        error = phase_error(output, target)
        rows[variant] = {f"{name}_mae_deg": float(error[mask].abs().mean() * 180 / math.pi)
                         for name, mask in masks.items() if mask.any()}
        v, e = data["velocity"][masks["moving"]], error[masks["moving"]]
        rows[variant]["signed_tracking_lag_seconds"] = float(-(v * e).sum() / v.square().sum())
        rows[variant]["final_mae_deg"] = float(error[-1].abs().mean() * 180 / math.pi)
        archive[f"forced_error_{variant}"] = error.cpu().numpy()
        archive[f"forced_state_final_{variant}"] = states[-1].cpu().numpy()
    archive["forced_theta"] = data["theta"].cpu().numpy()
    archive["forced_velocity"] = data["velocity"].cpu().numpy()
    archive["forced_visible"] = data["visible"].cpu().numpy()
    return rows, archive


@torch.no_grad()
def initial_state(model, task, starts, native_condition, warmup_steps=32):
    """Same native cue history initializes every within-model branch."""
    velocity = torch.zeros_like(starts)
    x = task.inputs(starts, velocity, torch.ones_like(starts), native_condition,
                    shuffled_velocity=velocity)
    state = starts.new_zeros(len(starts), model.hidden_size)
    for _ in range(warmup_steps):
        state = model.step(x, state)
    return state


@torch.no_grad()
def navigate(model, task, native_condition, variant, realization,
             blackout_steps=0, pulse_rms=0, pulse_step=10, restore_step=40,
             controller_gain=2.0, controller_max_speed=0.8,
             inertia=0.2, capture_radius=0.10, goal_hold=False,
             hold_seconds=10.0, hold_speed=0.15, dwell_seconds=0.5,
             motor_noise=0.0, maintenance_blackout=None):
    """One frozen-model intervention on matched closed-loop episodes.

    At t: observe the current position and completed previous movement; process
    the delayed cue; select a target velocity from the decoded phase; execute
    one movement. This matches the training generator's velocity convention.
    """
    c = task.config
    starts, goals = realization["starts"], realization["goals"]
    length, batch_size = realization["visible"].shape
    theta, velocity = starts.clone(), torch.zeros_like(starts)
    state = initial_state(model, task, starts, native_condition)
    acquired_phase, acquired_velocity = [], []
    arrived = torch.zeros(batch_size, dtype=torch.bool, device=starts.device)
    arrival_step = torch.full((batch_size,), length, dtype=torch.long, device=starts.device)
    path_length = torch.zeros_like(starts)
    oscillations = torch.zeros_like(starts)
    previous_command = torch.zeros_like(starts)
    dwell_run = torch.zeros(batch_size, dtype=torch.long, device=starts.device)
    first_dwell = torch.full_like(dwell_run, length)
    dwell_succeeded = torch.zeros_like(arrived)
    hold_fraction = torch.zeros_like(starts)
    hold_goal_error = torch.zeros_like(starts)
    hold_energy = torch.zeros_like(starts)
    holding_start = max(0, length - round(hold_seconds / c.dt))
    dwell_steps = max(1, round(dwell_seconds / c.dt))
    trace = {name: [] for name in ("theta", "estimate", "velocity", "command", "arrived", "visible")}
    for t in range(length):
        observed_velocity = velocity + c.velocity_noise * realization["velocity_noise"][t]
        acquired_velocity.append(observed_velocity)
        acquired_phase.append(wrap_angle(theta + c.position_noise * realization["phase_noise"][t]))
        observation_index = max(0, t - c.delay_steps)
        visible = realization["visible"][t]
        if t < blackout_steps:
            visible = visible * 0
        if maintenance_blackout is not None and maintenance_blackout[0] <= t < maintenance_blackout[1]:
            visible = visible * 0
        condition = input_condition(variant, native_condition, t, restore_step)
        past_velocity = acquired_velocity[observation_index]
        x = task.inputs(acquired_phase[observation_index], observed_velocity, visible,
                        condition, past_velocity, past_velocity.roll(batch_size // 2),
                        c.channel_noise * realization["channel_noise"][t])
        if t == pulse_step and pulse_rms:
            state = state + pulse_rms * realization["pulse"]
        state = model.step(x, state)
        output = model.readout(state)
        estimate = torch.atan2(output[:, 1], output[:, 0])
        error_to_goal = wrap_angle(goals - estimate)
        command = (controller_gain * error_to_goal).clamp(-controller_max_speed, controller_max_speed)
        if not goal_hold:
            command = command.masked_fill(arrived, 0)
        new_velocity = (1 - inertia) * velocity + inertia * command
        if motor_noise:
            new_velocity = new_velocity + motor_noise * realization["motor_noise"][t]
        if not goal_hold:
            new_velocity = new_velocity.masked_fill(arrived, 0)
        new_theta = wrap_angle(theta + c.dt * new_velocity)
        path_length += c.dt * new_velocity.abs()
        oscillations += ((command * previous_command < 0) & (torch.ones_like(arrived) if goal_hold else ~arrived)).float()
        previous_command = command
        hit = (wrap_angle(goals - new_theta).abs() <= capture_radius) & ~arrived
        arrival_step[hit] = t + 1
        arrived |= hit
        in_goal = wrap_angle(goals - new_theta).abs() <= capture_radius
        precise_hold = in_goal & (new_velocity.abs() <= hold_speed)
        dwell_run = torch.where(precise_hold, dwell_run + 1, 0)
        new_dwell = (dwell_run >= dwell_steps) & ~dwell_succeeded
        first_dwell[new_dwell] = t + 1
        dwell_succeeded |= new_dwell
        if t >= holding_start:
            hold_fraction += precise_hold.float()
            hold_goal_error += wrap_angle(goals - new_theta).abs()
            hold_energy += command.square() * c.dt
        for name, value in (("theta", theta), ("estimate", estimate), ("velocity", velocity),
                            ("command", command), ("arrived", arrived), ("visible", visible)):
            trace[name].append(value.clone())
        theta, velocity = new_theta, new_velocity
    capped_time = arrival_step.float() * c.dt
    shortest = wrap_angle(goals - starts).abs()
    # Every timeout remains in all primary outcome means.
    outcome = {"mean_capped_arrival_seconds": float(capped_time.mean()),
               "success_fraction": float(arrived.float().mean()),
               "mean_path_length_rad": float(path_length.mean()),
               "mean_command_reversals": float(oscillations.mean()),
               "mean_final_goal_error_deg": float(wrap_angle(goals - theta).abs().mean() * 180 / math.pi),
               "mean_shortest_distance_rad": float(shortest.mean())}
    per_episode = {"arrival_seconds": capped_time.cpu().numpy(), "success": arrived.cpu().numpy(),
                   "path_length": path_length.cpu().numpy(), "starts": starts.cpu().numpy(),
                   "goals": goals.cpu().numpy(), "command_reversals": oscillations.cpu().numpy()}
    if goal_hold:
        hold_fraction /= length - holding_start
        hold_goal_error /= length - holding_start
        outcome.update(mean_capped_dwell_seconds=float(first_dwell.float().mean() * c.dt),
                       sustained_hold_success_fraction=float(dwell_succeeded.float().mean()),
                       final_window_precise_hold_fraction=float(hold_fraction.mean()),
                       final_window_goal_mae_deg=float(hold_goal_error.mean() * 180 / math.pi),
                       final_window_control_energy=float(hold_energy.mean()))
        per_episode.update(dwell_seconds=(first_dwell.float() * c.dt).cpu().numpy(),
                           hold_success=dwell_succeeded.cpu().numpy(),
                           precise_hold_fraction=hold_fraction.cpu().numpy(),
                           holding_goal_mae=hold_goal_error.cpu().numpy(),
                           control_energy=hold_energy.cpu().numpy())
    # Keep complete first32 examples; outcomes above include every episode.
    for name, values in trace.items():
        per_episode[f"trace_{name}"] = torch.stack(values)[:, :32].cpu().numpy()
    return outcome, per_episode


def navigation_realization(task, seed, length, batch_size):
    generator = torch.Generator(device=task.device).manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, device=task.device, generator=generator)
    normal = lambda *shape: torch.randn(*shape, device=task.device, generator=generator)
    starts = rand(batch_size) * (2 * math.pi) - math.pi
    distance = 0.7 + rand(batch_size) * (math.pi - 0.7)
    sign = (rand(batch_size) < 0.5).float() * 2 - 1
    goals = wrap_angle(starts + sign * distance)
    pulse = normal(batch_size, task.config.hidden_size)
    pulse = pulse / pulse.square().mean(-1, keepdim=True).sqrt()
    data = {"starts": starts, "goals": goals,
            "visible": (rand(length, batch_size) < task.config.visibility).float(),
            "phase_noise": normal(length, batch_size), "velocity_noise": normal(length, batch_size),
            "channel_noise": normal(length, batch_size, task.config.n_code), "pulse": pulse}
    # Draw this last to preserve the original first-entry experiment exactly.
    data["motor_noise"] = normal(length, batch_size)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=874231)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--restore-step", type=int, default=40)
    parser.add_argument("--pulse-rms", type=float, default=0.10)
    parser.add_argument("--blackout-steps", type=int, default=80)
    parser.add_argument("--controller-gain", type=float, default=2.0)
    parser.add_argument("--controller-max-speed", type=float, default=0.8)
    parser.add_argument("--inertia", type=float, default=0.2)
    parser.add_argument("--capture-radius", type=float, default=0.10)
    parser.add_argument("--skip-holding", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    replay, archive = forced_replay(model, task, checkpoint["condition"], args.seed,
                                    batch_size=args.episodes, restore_step=args.restore_step)
    realization = navigation_realization(task, args.seed + 1, args.steps, args.episodes)
    navigation = {}
    for blackout_steps in (0, args.blackout_steps):
        for pulse_rms in (0, args.pulse_rms):
            scenario = f"blank{blackout_steps}_pulse{pulse_rms:g}"
            navigation[scenario] = {}
            for variant in VARIANTS:
                outcome, episodes = navigate(model, task, checkpoint["condition"], variant, realization,
                    blackout_steps=blackout_steps, pulse_rms=pulse_rms, restore_step=args.restore_step,
                    controller_gain=args.controller_gain, controller_max_speed=args.controller_max_speed,
                    inertia=args.inertia, capture_radius=args.capture_radius)
                navigation[scenario][variant] = outcome
                archive.update({f"{scenario}_{variant}_{key}": value for key, value in episodes.items()})
            print(json.dumps({"scenario": scenario, "outcomes": navigation[scenario]}), flush=True)
            if blackout_steps:
                # With no spatial observations, changing their forecast shift
                # must have exactly zero effect, even after an internal pulse.
                for variant in VARIANTS:
                    for metric in ("trace_theta", "trace_estimate", "trace_command"):
                        assert np.array_equal(archive[f"{scenario}_native_{metric}"][:blackout_steps],
                                              archive[f"{scenario}_{variant}_{metric}"][:blackout_steps])
    if checkpoint["condition"] == "grid":
        # These branches are the exact same computation, even after perturbing
        # the state or retaining only delayed observations.
        for scenario in navigation:
            assert navigation[scenario]["native"] == navigation[scenario]["ordinary"]
            assert navigation[scenario]["native"] == navigation[scenario]["restored"]
            for metric in ("arrival_seconds", "path_length", "trace_estimate"):
                assert np.array_equal(archive[f"{scenario}_native_{metric}"], archive[f"{scenario}_ordinary_{metric}"])
    holding = {}
    if not args.skip_holding:
        if args.steps * task.config.dt < 15:
            raise ValueError("Holding evaluation needs >=15 seconds; use --skip-holding for short smoke tests")
        maintenance_window = (round(5 / task.config.dt), round(15 / task.config.dt))
        for outage in (False, True):
            for pulse_rms in (0, args.pulse_rms):
                scenario = f"maintenance_blank{int(outage)}_pulse{pulse_rms:g}"
                holding[scenario] = {}
                for variant in VARIANTS:
                    outcome, episodes = navigate(model, task, checkpoint["condition"], variant, realization,
                        pulse_rms=pulse_rms, pulse_step=maintenance_window[0], restore_step=args.restore_step,
                        controller_gain=args.controller_gain, controller_max_speed=args.controller_max_speed,
                        inertia=args.inertia, capture_radius=args.capture_radius, goal_hold=True,
                        hold_seconds=10.0, hold_speed=0.15, dwell_seconds=0.5, motor_noise=0.03,
                        maintenance_blackout=maintenance_window if outage else None)
                    holding[scenario][variant] = outcome
                    archive.update({f"{scenario}_{variant}_{key}": value for key, value in episodes.items()})
                if checkpoint["condition"] == "grid":
                    assert holding[scenario]["native"] == holding[scenario]["ordinary"]
                    assert holding[scenario]["native"] == holding[scenario]["restored"]
                print(json.dumps({"holding_scenario": scenario, "outcomes": holding[scenario]}), flush=True)
    result = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "condition": checkpoint["condition"], "seed": checkpoint["seed"],
              "config": checkpoint["config"], "evaluation_seed": args.seed, "episodes_per_condition": args.episodes,
              "protocol": {"recurrent_noise": 0, "frozen_weights": True, "controller": "fixed_gain_times_decoded_goal_phase_error",
                 "controller_gain": args.controller_gain, "controller_max_speed": args.controller_max_speed,
                 "inertia": args.inertia,
                 "goal_capture_radius_rad": args.capture_radius, "timeout_seconds": args.steps * task.config.dt,
                 "restore_step": args.restore_step, "pulse_step": 10, "pulse_rms": args.pulse_rms,
                 "blackout_steps": args.blackout_steps, "same_warmup_history_within_model": True,
                 "warmup_steps": 32, "warmup": "stationary noiseless native cue; all branches identical",
                 "same_random_realizations_across_branches": True, "true_phase_used_by_controller": False,
                 "all_timeouts_included": True, "causal_input_changes": "forecast shift only; all channels retained"},
              "holding_protocol": {"secondary_exploratory": True,
                 "motivation": "First crossing permits overshoot; holding tests whether the representation supports stable position control",
                 "goal_fixed_for_entire_trial": True, "continue_after_first_entry": True,
                 "assessment": "last 10 seconds of the fixed-duration trial",
                 "dwell_seconds": 0.5, "holding_speed_threshold_rad_per_second": 0.15,
                 "motor_velocity_noise_sd": 0.03, "maintenance_blackout_seconds": [5, 15],
                 "pulse_seconds": 5.0, "all_trials_included": True},
              "forced_replay": replay, "navigation": navigation, "holding": holding,
              "seconds": time.monotonic() - start}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"event": "complete", "out": str(args.out), "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
