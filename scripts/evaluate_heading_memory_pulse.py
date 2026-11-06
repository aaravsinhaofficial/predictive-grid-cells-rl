"""Inject measured direction-memory states before otherwise unchanged navigation.

The wrapper changes only the recurrent state at the end of common warmup.
The frozen controller receives exactly the same arguments as before.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import evaluate_ring_heading_navigation as navigation
from ring_dynamics import wrap
from ring_heading_model import load_checkpoint


PULSES = ("baseline", "phase_minus20", "phase_plus20", "normal_matched20")


class StateReplacement:
    """Transparent model proxy replacing only the final warmup state."""

    def __init__(self, model, replacement):
        self.model, self.replacement = model, replacement
        self.hidden_size, self.input_size, self.config = model.hidden_size, model.input_size, model.config
        self.calls, self.replacements = 0, 0
        self.readout = model.readout

    def step(self, x, state):
        self.calls += 1
        if self.calls == self.config.initial_heading_steps:
            self.replacements += 1
            assert self.replacement.shape == state.shape
            return self.replacement.clone()
        return self.model.step(x, state)


def phase(model, state):
    output = model.readout(state)
    return torch.atan2(output[..., 1], output[..., 0])


@torch.no_grad()
def prepare(model, task, checkpoint, noise, seed):
    count = len(noise["starts"])
    zeros, ones = torch.zeros(count, device=task.device), torch.ones(count, device=task.device)
    states = {}
    for offset in (0, -20, 20, -5, 5):
        heading = noise["initial_heading"] + math.radians(offset)
        h = torch.zeros(count, model.hidden_size, device=task.device)
        for t in range(task.config.initial_heading_steps):
            x, _ = navigation.observation_inputs(task, noise["starts"], heading, zeros, zeros,
                ones, ones, checkpoint["condition"], noise, t)
            h = model.step(x, h)
        for _ in range(40):
            h = model.step(torch.zeros(count, model.input_size, device=task.device), h)
        states[offset] = h
    tangent = states[5] - states[-5]
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    generator = torch.Generator(device=task.device).manual_seed(seed + 50000)
    signs = torch.ones(count, device=task.device)
    signs[:count // 2] = -1
    signs = signs[torch.randperm(count, generator=generator, device=task.device)]
    selected = torch.where(signs[:, None] > 0, states[20], states[-20])
    selected_delta = selected - states[0]
    normal = torch.randn(count, model.hidden_size, generator=generator, device=task.device)
    normal = normal - (normal * tangent).sum(-1, keepdim=True) * tangent
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    normal = normal * selected_delta.norm(dim=-1, keepdim=True)
    assert torch.allclose(normal.norm(dim=-1), selected_delta.norm(dim=-1), atol=1e-6, rtol=1e-5)
    assert float((normal * tangent).sum(-1).abs().max()) < 1e-6
    pulses = {"baseline": states[0], "phase_minus20": states[-20], "phase_plus20": states[20],
              "normal_matched20": states[0] + normal}
    baseline_phase = phase(model, states[0])
    archive = {"balanced_phase_sign": signs.cpu().numpy(), "empirical_tangent": tangent.cpu().numpy(),
               "initial_physical_heading": noise["initial_heading"].cpu().numpy()}
    for name, state in pulses.items():
        archive[f"{name}_state"] = state.cpu().numpy()
        archive[f"{name}_initial_error"] = wrap((phase(model, state) - noise["initial_heading"]).cpu().numpy())
        archive[f"{name}_phase_displacement"] = wrap((phase(model, state) - baseline_phase).cpu().numpy())
        archive[f"{name}_state_displacement"] = (state - states[0]).norm(dim=-1).cpu().numpy()
    return pulses, archive


def summarize_trace(arrays, baseline, dt):
    heading_error = np.abs(wrap(arrays["trace_decoded_heading"] - arrays["trace_heading"])) * 180 / np.pi
    deviation = np.linalg.norm(arrays["trace_position"] - baseline["trace_position"], axis=-1)
    interval = 2 * dt
    per_trial = {"early_heading_mae_deg": heading_error[:round(2 / interval)].mean(0),
                 "first5s_path_deviation_m": deviation[:round(5 / interval)].mean(0)}
    for seconds in (.1, .5, 1., 2.):
        per_trial[f"heading_error_at_{seconds:g}s_deg"] = heading_error[round(seconds / interval)]
    for seconds in (1., 2., 5.):
        per_trial[f"path_deviation_at_{seconds:g}s_m"] = deviation[round(seconds / interval)]
    return per_trial


def aggregate_navigation(records):
    combined = {}
    for duration in records[0]["durations"]:
        combined[duration] = {}
        for variant in navigation.VARIANTS:
            rows = [row["durations"][duration][variant] for row in records]
            n_entered = sum(row["post_entry_n"] for row in rows)
            metrics = {}
            for key in rows[0]:
                if key == "post_entry_n":
                    metrics[key] = n_entered
                elif key in ("mean_max_distance_after_first_entry_m", "mean_overshoot_after_first_entry_m"):
                    metrics[key] = (sum(row[key] * row["post_entry_n"] for row in rows if row["post_entry_n"])
                                    / n_entered) if n_entered else None
                else:
                    metrics[key] = float(np.mean([row[key] for row in rows]))
            combined[duration][variant] = metrics
    return combined


def aggregate_balanced_navigation(archive, repeats):
    combined = {}
    for duration in ("20s", "40s"):
        combined[duration] = {}
        for variant in navigation.VARIANTS:
            def values(key):
                return np.concatenate([archive[f"repeat{r}_phase_balanced20_{duration}_{variant}_{key}"]
                                       for r in range(repeats)])
            success = values("success")
            metrics = {"success_fraction": float(success.mean()), "post_entry_n": int(success.sum())}
            for output, key in (("mean_capped_arrival_seconds", "capped_arrival"),
                                ("mean_final_goal_distance_m", "final_goal_distance"),
                                ("mean_self_position_error_m", "self_position_error"),
                                ("terminal_goal_occupancy_fraction", "terminal_goal_occupancy"),
                                ("sustained_dwell_success_fraction", "sustained_dwell_success"),
                                ("mean_capped_sustained_dwell_seconds", "capped_sustained_dwell_seconds"),
                                ("mean_longest_sustained_hold_seconds", "longest_hold_seconds")):
                metrics[output] = float(values(key).mean())
            for output, key in (("mean_max_distance_after_first_entry_m", "max_distance_after_entry"),
                                ("mean_overshoot_after_first_entry_m", "overshoot_after_entry")):
                metrics[output] = float(np.nanmean(values(key))) if success.any() else None
            combined[duration][variant] = metrics
    return combined


@torch.no_grad()
def run(args):
    if args.episodes != 32:
        raise ValueError("Use exactly32episodes per repeat to retain every trace in the unchanged simulator")
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    records = {name: [] for name in PULSES}
    trace_metrics = {name: [] for name in (*PULSES, "phase_balanced20")}
    archive = {}
    state_metrics = {name: [] for name in PULSES}
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        noise = navigation.realization(task, seed, round(40 / task.config.dt), args.episodes)
        pulses, preparation = prepare(model, task, checkpoint, noise, seed)
        arrays_by_pulse = {}
        for name, state in pulses.items():
            proxy = StateReplacement(model, state)
            summary, arrays = navigation.simulate(proxy, task, checkpoint, noise)
            assert proxy.replacements == 1
            assert arrays["trace_position"].shape[2] == args.episodes
            records[name].append(summary)
            arrays_by_pulse[name] = arrays
            archive.update({f"repeat{repeat}_{name}_{key}": value for key, value in arrays.items()})
            state_metrics[name].append({key: preparation[f"{name}_{key}"] for key in
                                       ("initial_error", "phase_displacement", "state_displacement")})
        signs = preparation["balanced_phase_sign"]
        selected = {}
        for key, plus in arrays_by_pulse["phase_plus20"].items():
            minus = arrays_by_pulse["phase_minus20"][key]
            shape = [1] * plus.ndim
            episode_axis = 2 if key.startswith("trace_") else 0
            shape[episode_axis] = args.episodes
            selected[key] = np.where(signs.reshape(shape) > 0, plus, minus)
        arrays_by_pulse["phase_balanced20"] = selected
        archive.update({f"repeat{repeat}_phase_balanced20_{key}": value for key, value in selected.items()})
        for name, arrays in arrays_by_pulse.items():
            per_trial = summarize_trace(arrays, arrays_by_pulse["baseline"], task.config.dt)
            trace_metrics[name].append(per_trial)
            archive.update({f"repeat{repeat}_{name}_{key}": value for key, value in per_trial.items()})
        archive.update({f"repeat{repeat}_preparation_{key}": value for key, value in preparation.items()})
    outcomes = {name: aggregate_navigation(rows) for name, rows in records.items()}
    outcomes["phase_balanced20"] = aggregate_balanced_navigation(archive, args.repeats)
    early = {}
    for name, rows in trace_metrics.items():
        early[name] = {variant: {key: float(np.concatenate([row[key][v] for row in rows]).mean())
                    for key in rows[0]} for v, variant in enumerate(navigation.VARIANTS)}
    preparation_summary = {}
    for name, rows in state_metrics.items():
        displacement = np.concatenate([row["phase_displacement"] for row in rows]) * 180 / np.pi
        preparation_summary[name] = {
            "mean_signed_phase_displacement_deg": float(displacement.mean()),
            "median_absolute_phase_displacement_deg": float(np.median(np.abs(displacement))),
            "phase_displacement_q10_q90_deg": np.quantile(displacement, [.1, .9]).tolist(),
            "mean_initial_heading_mae_deg": float(np.mean(np.abs(np.concatenate([row["initial_error"] for row in rows]))) * 180 / np.pi),
            "mean_state_displacement": float(np.mean(np.concatenate([row["state_displacement"] for row in rows])))}
    penalties = {}
    for name in (*PULSES[1:], "phase_balanced20"):
        penalties[name] = {duration: {variant: {key: row[key] - outcomes["baseline"][duration][variant][key]
             for key in ("success_fraction", "mean_capped_arrival_seconds", "mean_final_goal_distance_m",
                         "terminal_goal_occupancy_fraction", "sustained_dwell_success_fraction")}
             for variant, row in variants.items()} for duration, variants in outcomes[name].items()}
    report = {"checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_navigation_sha256": hashlib.sha256(Path(navigation.__file__).read_bytes()).hexdigest(),
        "condition": checkpoint["condition"], "training_seed": checkpoint["seed"],
        "steps_completed": checkpoint["steps_completed"], "evaluation_seeds": list(range(args.seed, args.seed + args.repeats)),
        "episodes": args.episodes * args.repeats, "preparation": preparation_summary,
        "early_heading_and_path": early, "navigation": outcomes, "pulse_minus_baseline": penalties,
        "protocol": {"plan": "docs/research/heading_memory_pulse_protocol.md", "blank_preparation_steps": 40,
            "target_phase_pulse_degrees": [-20, 20], "state_substitution": "measured conditional memory states",
            "normal_control": "orthogonal to +/-5degree empirical tangent, equal norm to balanced signed phase pulse",
            "controller_unchanged": True, "physical_state_unchanged_at_pulse": True,
            "all_subsequent_sensory_noise_paired": True, "pgc_direction_independent_of_memory_pulse": True,
            "true_direction_used_to_select_experimental_state_only": True, "recurrent_noise": 0,
            "scoring_radius_m": .15, "trace_interval_seconds": 2 * task.config.dt,
            "variant_order": navigation.VARIANTS, "primary_interpretations": ["native", "ordinary", "restored"]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **archive)
    print(json.dumps({"training_seed": checkpoint["seed"], "condition": checkpoint["condition"],
                     "preparation": preparation_summary, "early": early}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=961001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
