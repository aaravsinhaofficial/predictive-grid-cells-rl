"""Navigation through learned heading, with no true-state controller input.

The fixed controller integrates measured speed along the RNN's decoded heading
to maintain its own position estimate. It knows the reset position and goal.
Only the simulator/observation generator and scoring use actual pose. Grid
observations are generated from actual pose just as in the training task.

Twenty- and forty-second results are nested prefixes of the same episodes.
Input interventions use frozen weights, paired noise and identical warmup.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch

from ring_heading_model import load_checkpoint
from ring_task_model import wrap_angle


VARIANTS = ("native", "ordinary", "reverse", "restored")


class HoldingScores:
    """Outcome-only bookkeeping. No value produced here enters the controller."""

    def __init__(self, reference, dt, episode_steps, capture_radius,
                 terminal_seconds=5.0, dwell_seconds=2.0, dwell_speed=0.05):
        self.dt, self.capture_radius = dt, capture_radius
        self.window_steps = max(1, round(terminal_seconds / dt))
        self.dwell_steps, self.dwell_speed = max(1, round(dwell_seconds / dt)), dwell_speed
        self.max_post_entry_distance = torch.zeros_like(reference)
        self.run = torch.zeros_like(reference, dtype=torch.long)
        self.longest = torch.zeros_like(self.run)
        self.first_dwell = torch.full_like(self.run, episode_steps + 1)
        self.occupancy = []
        self.steps = 0

    def update(self, distance, speed, entered):
        self.steps += 1
        inside = distance <= self.capture_radius
        self.occupancy.append(inside)
        if len(self.occupancy) > self.window_steps:
            self.occupancy.pop(0)
        self.max_post_entry_distance = torch.where(entered,
            torch.maximum(self.max_post_entry_distance, distance), self.max_post_entry_distance)
        holding = inside & (speed <= self.dwell_speed)
        self.run = torch.where(holding, self.run + 1, 0)
        self.longest = torch.maximum(self.longest, self.run)
        reached = (self.run >= self.dwell_steps) & (self.first_dwell > self.steps)
        self.first_dwell[reached] = self.steps

    def summarize(self, variant, entered):
        maximum = self.max_post_entry_distance[variant]
        overshoot = (maximum - self.capture_radius).clamp_min(0)
        terminal = torch.stack(self.occupancy).float().mean(0)[variant]
        dwell_success = self.first_dwell[variant] <= self.steps
        dwell_time = self.first_dwell[variant].clamp_max(self.steps).float() * self.dt
        longest = self.longest[variant].float() * self.dt
        metrics = {"post_entry_n": int(entered.sum()),
            "mean_max_distance_after_first_entry_m": float(maximum[entered].mean()) if entered.any() else None,
            "mean_overshoot_after_first_entry_m": float(overshoot[entered].mean()) if entered.any() else None,
            "terminal_goal_occupancy_fraction": float(terminal.mean()),
            "sustained_dwell_success_fraction": float(dwell_success.float().mean()),
            "mean_capped_sustained_dwell_seconds": float(dwell_time.mean()),
            "mean_longest_sustained_hold_seconds": float(longest.mean())}
        arrays = {"max_distance_after_entry": torch.where(entered, maximum, torch.nan),
                  "overshoot_after_entry": torch.where(entered, overshoot, torch.nan),
                  "terminal_goal_occupancy": terminal, "sustained_dwell_success": dwell_success,
                  "capped_sustained_dwell_seconds": dwell_time, "longest_hold_seconds": longest}
        return metrics, arrays


def integrate_position(estimated_position, decoded_heading, measured_speed, dt):
    direction = torch.stack((decoded_heading.cos(), decoded_heading.sin()), -1)
    return estimated_position + dt * measured_speed[..., None] * direction


def controller(estimated_position, decoded_heading, goal, gain=2.0,
               max_angular_velocity=1.8, travel_speed=0.25, stop_radius=0.08):
    """Pure observed-estimate controller: no actual position/heading arguments."""
    displacement = goal - estimated_position
    bearing = torch.atan2(displacement[..., 1], displacement[..., 0])
    error = wrap_angle(bearing - decoded_heading)
    angular_command = (gain * error).clamp(-max_angular_velocity, max_angular_velocity)
    forward_command = travel_speed * error.cos().clamp(0.24, 1.0)
    stop = displacement.norm(dim=-1) <= stop_radius
    # Turning is forbidden when stopped, preserving heading=travel direction.
    return angular_command.masked_fill(stop, 0), forward_command.masked_fill(stop, 0)


def realization(task, seed, length, episodes):
    generator = torch.Generator(device=task.device).manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, device=task.device, generator=generator)
    normal = lambda *shape: torch.randn(*shape, device=task.device, generator=generator)
    starts = (rand(episodes, 2) * 2 - 1) * 0.5
    initial_heading = rand(episodes) * (2 * math.pi) - math.pi
    goal_bearing = rand(episodes) * (2 * math.pi) - math.pi
    goal_distance = 0.8 + rand(episodes) * 0.7
    goals = starts + goal_distance[:, None] * torch.stack((goal_bearing.cos(), goal_bearing.sin()), -1)
    visible = (rand(length, episodes) < task.config.visibility).float()
    visible[round(4 / task.config.dt):round(7 / task.config.dt)] = 0
    warmup = task.config.initial_heading_steps
    return {"starts": starts, "goals": goals, "initial_heading": initial_heading,
            "initial_error_sign": (rand(episodes) < 0.5).float() * 2 - 1,
            "angular_bias": normal(episodes), "angular_noise": normal(length + warmup, episodes),
            "speed_noise": normal(length + warmup, episodes),
            "direction_noise": normal(length + warmup, episodes),
            "position_noise": normal(length + warmup, episodes, 2),
            "channel_noise": normal(length + warmup, episodes, task.config.n_code),
            "motor_angular_noise": normal(length, episodes), "motor_speed_noise": normal(length, episodes),
            "visible": visible}


def observation_inputs(task, position, heading, angular, speed, visible, initial_cue,
                       condition, noise, index, bias_multiplier=1.0,
                       initial_error_degrees=0.0):
    """Only this sensory generator accesses actual physical pose."""
    c = task.config
    observed_heading = heading + c.direction_noise * noise["direction_noise"][index]
    direction = torch.stack((observed_heading.cos(), observed_heading.sin()), -1)
    observed_position = position + c.position_noise * noise["position_noise"][index]
    measured_angular = angular + bias_multiplier * c.angular_velocity_bias * noise["angular_bias"] + c.angular_velocity_noise * noise["angular_noise"][index]
    measured_speed = (speed + c.speed_noise * noise["speed_noise"][index]).clamp_min(0)
    x = task.inputs(observed_position, direction, measured_angular, measured_speed,
                    visible, initial_cue, condition, c.channel_noise * noise["channel_noise"][index],
                    direction.roll(max(1, len(direction) // 2), 0))
    if initial_error_degrees:
        misleading = observed_heading + math.radians(initial_error_degrees) * noise["initial_error_sign"]
        misleading = torch.stack((misleading.cos(), misleading.sin()), -1)
        x[..., -3:-1] = torch.where(initial_cue[..., None].bool(), misleading, x[..., -3:-1])
    return x, measured_speed


@torch.no_grad()
def simulate(model, task, checkpoint, noise, position_offset=0.0, bias_multiplier=1.0,
             initial_error_degrees=0.0, restore_seconds=2.0, capture_radius=0.15):
    c, device = task.config, task.device
    length, episodes = noise["visible"].shape
    starts, goals = noise["starts"].clone(), noise["goals"].clone()
    starts[:, 0] += position_offset
    goals[:, 0] += position_offset
    base_heading = noise["initial_heading"]
    base_state = torch.zeros(episodes, model.hidden_size, device=device)
    zeros, ones = torch.zeros(episodes, device=device), torch.ones(episodes, device=device)
    for warm in range(c.initial_heading_steps):
        x, _ = observation_inputs(task, starts, base_heading, zeros, zeros, ones, ones,
            checkpoint["condition"], noise, warm, bias_multiplier, initial_error_degrees)
        base_state = model.step(x, base_state)
    initial_output = model.readout(base_state)
    initial_estimate = torch.atan2(initial_output[:, 1], initial_output[:, 0])
    initial_error = wrap_angle(initial_estimate - base_heading)
    n_variants = len(VARIANTS)
    state = base_state[None].expand(n_variants, -1, -1).clone()
    position = starts[None].expand(n_variants, -1, -1).clone()
    estimated_position = position.clone()  # Known reset coordinate, only once.
    heading = base_heading[None].expand(n_variants, -1).clone()
    angular, speed = torch.zeros_like(heading), torch.zeros_like(heading)
    first_arrival = torch.full((n_variants, episodes), length + 1, dtype=torch.long, device=device)
    traveled = torch.zeros_like(heading)
    holding_scores = HoldingScores(heading, c.dt, length, capture_radius)
    snapshots, archive = {}, {"initial_heading_error": initial_error.cpu().numpy(),
                              "starts": starts.cpu().numpy(), "goals": goals.cpu().numpy()}
    trace = {name: [] for name in ("position", "estimated_position", "heading", "decoded_heading")}
    report_steps = [round(seconds / c.dt) for seconds in (20, 40) if round(seconds / c.dt) <= length]
    for t in range(length):
        x_rows, measured_speeds = [], []
        for v, variant in enumerate(VARIANTS):
            if variant == "ordinary" or (variant == "restored" and t * c.dt < restore_seconds):
                condition = "grid"
            elif variant == "reverse":
                condition = "reverse"
            else:
                condition = checkpoint["condition"]
            x, measured_speed = observation_inputs(task, position[v], heading[v], angular[v], speed[v],
                noise["visible"][t], zeros, condition, noise, t + c.initial_heading_steps, bias_multiplier)
            x_rows.append(x)
            measured_speeds.append(measured_speed)
        x = torch.stack(x_rows)
        state = model.step(x.reshape(-1, model.input_size), state.reshape(-1, model.hidden_size)).reshape(n_variants, episodes, -1)
        output = model.readout(state)
        decoded_heading = torch.atan2(output[..., 1], output[..., 0])
        # No actual pose appears in either of the following computations.
        if t:
            estimated_position = integrate_position(estimated_position, decoded_heading, torch.stack(measured_speeds), c.dt)
        angular_command, forward_command = controller(estimated_position, decoded_heading, goals[None])
        position_before_action = position.clone()
        heading_before_action = heading.clone()
        stop = forward_command == 0
        angular = (0.8 * angular + 0.2 * angular_command + 0.03 * noise["motor_angular_noise"][t][None]).masked_fill(stop, 0)
        speed = (0.8 * speed + 0.2 * forward_command + 0.005 * noise["motor_speed_noise"][t][None]).clamp(0.001, 0.35).masked_fill(stop, 0)
        assert not torch.any((speed == 0) & (angular != 0))
        heading = wrap_angle(heading + c.dt * angular)
        direction = torch.stack((heading.cos(), heading.sin()), -1)
        position = position + c.dt * speed[..., None] * direction
        traveled += speed * c.dt
        # Outcome measurements do not modify any controller or physical state.
        goal_distance = (position - goals[None]).norm(dim=-1)
        new_hit = (goal_distance <= capture_radius) & (first_arrival > length)
        first_arrival[new_hit] = t + 1
        holding_scores.update(goal_distance, speed, first_arrival <= t + 1)
        if (t + 1) in report_steps:
            seconds = round((t + 1) * c.dt)
            rows = {}
            for v, variant in enumerate(VARIANTS):
                success = first_arrival[v] <= t + 1
                capped = first_arrival[v].clamp_max(t + 1).float() * c.dt
                rows[variant] = {"success_fraction": float(success.float().mean()),
                    "mean_capped_arrival_seconds": float(capped.mean()),
                    "mean_final_goal_distance_m": float(goal_distance[v].mean()),
                    "mean_self_position_error_m": float((estimated_position[v] - position_before_action[v]).norm(dim=-1).mean()),
                    "mean_path_length_m": float(traveled[v].mean())}
                secondary, secondary_arrays = holding_scores.summarize(v, success)
                rows[variant].update(secondary)
                for key, value in secondary_arrays.items():
                    archive[f"{seconds}s_{variant}_{key}"] = value.cpu().numpy()
                for key, value in (("success", success), ("capped_arrival", capped),
                                   ("final_goal_distance", goal_distance[v]),
                                   ("self_position_error", (estimated_position[v] - position_before_action[v]).norm(dim=-1))):
                    archive[f"{seconds}s_{variant}_{key}"] = value.cpu().numpy()
            snapshots[f"{seconds}s"] = rows
        if t % 2 == 0:
            for name, value in (("position", position_before_action), ("estimated_position", estimated_position),
                                ("heading", heading_before_action), ("decoded_heading", decoded_heading)):
                trace[name].append(value[:, :32].clone())
    for name, values in trace.items():
        archive[f"trace_{name}"] = torch.stack(values).cpu().numpy()
    if checkpoint["condition"] == "grid":
        for duration, rows in snapshots.items():
            assert rows["native"] == rows["ordinary"] == rows["restored"]
    return {"initial_decoded_mae_deg": float(initial_error.abs().mean() * 180 / math.pi),
            "durations": snapshots}, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=836219)
    parser.add_argument("--seconds", type=int, choices=(20, 40), default=40)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    noise = realization(task, args.seed, round(args.seconds / task.config.dt), args.episodes)
    scenarios = {"baseline": (0.0, 1.0, 0.0), "bias_stress": (0.0, 2.0, 0.0),
                 "misleading_initial_cue": (0.0, 1.0, 20.0), "new_places": (10.0, 1.0, 0.0),
                 "new_places_bias_and_initial_error": (10.0, 2.0, 20.0)}
    results, archive = {}, {}
    for name, (offset, bias, initial_error) in scenarios.items():
        results[name], arrays = simulate(model, task, checkpoint, noise, offset, bias, initial_error)
        archive.update({f"{name}_{key}": value for key, value in arrays.items()})
        print(json.dumps({"scenario": name, "outcomes": results[name]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "condition": checkpoint["condition"], "seed": checkpoint["seed"],
              "config": checkpoint["config"], "steps_completed": checkpoint["steps_completed"],
              "evaluation_seed": args.seed, "episodes": args.episodes, "scenarios": results,
              "protocol": {"controller_inputs": list(inspect.signature(controller).parameters),
                  "controller_has_true_position_or_heading": False, "known_reset_position": True,
                  "known_goal": True, "external_position_estimate": "integral of measured speed along RNN-decoded heading",
                  "training": "learned heading estimator; fixed identical navigation controller",
                  "spatial_cue_blackout_seconds": [4, 7], "capture_radius_m": 0.15,
                  "restoration_seconds": 2.0, "initial_heading_error_degrees": [0, -20, 20],
                  "initial_error_applies_to_heading_cue_only": True,
                  "pgc_direction_remains_true_noisy_travel_direction": True,
                  "angular_sensor_bias_sd_rad_per_second": [task.config.angular_velocity_bias, 2 * task.config.angular_velocity_bias],
                  "no_rotation_while_stopped": True, "recurrent_noise": 0,
                  "all_timeouts_included": True, "scoring_does_not_change_trajectory": True,
                  "self_position_error_timing": "estimated and actual position at the same pre-action observation",
                  "duration_comparisons": "nested20/40second prefixes of the same trial",
                  "within_model_warmup_identical_across_interventions": True}}
    result["secondary_navigation_protocol"] = {
        "plan": "docs/research/heading_navigation_secondary_protocol.md",
        "post_entry_distance": "maximum goal-center distance after first entry, conditional on entering; denominator reported",
        "terminal_window_seconds": 5.0, "sustained_dwell_seconds": 2.0,
        "sustained_dwell_speed_limit_m_per_second": 0.05,
        "dwell_timeouts_included": True, "nonentrant_post_entry_values": "missing"}
    result["variant_interpretation"] = {
        "reverse": ("Reverse the learned predictive spatial displacement" if checkpoint["condition"] == "predictive"
                    else "Introduce unfamiliar reversed predictive spatial inputs; OOD diagnostic, not a learned-input reversal"),
        "ordinary": ("Remove explicit heading after common initialization" if checkpoint["condition"] == "grid_heading"
                     else "Replace predictive spatial channels with ordinary grid channels")}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)


if __name__ == "__main__":
    main()
