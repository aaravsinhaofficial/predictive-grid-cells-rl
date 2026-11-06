"""Held-out heading trajectories, unseen places and input removal/restoration.

The 10m spatial translation places every evaluated physical position beyond
the conservative reach bound of the128-step training episodes. Grid phase
patterns are evaluated by the original analytic population, not extrapolated
from a learned spatial decoder.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ring_heading_model import load_checkpoint
from ring_task_model import phase_error, wrap_angle


def alter_measurements(data, position_offset=0.0, noise_multiplier=1.0):
    result = {key: value.clone() for key, value in data.items()}
    multiplier = noise_multiplier
    result["measured_position"] = data["position"] + multiplier * (data["measured_position"] - data["position"])
    measured_angle = torch.atan2(data["measured_direction"][..., 1], data["measured_direction"][..., 0])
    direction_angle = data["heading"] + multiplier * wrap_angle(measured_angle - data["heading"])
    result["measured_direction"] = torch.stack((direction_angle.cos(), direction_angle.sin()), -1)
    result["measured_angular_velocity"] = data["angular_velocity"] + multiplier * (data["measured_angular_velocity"] - data["angular_velocity"])
    result["measured_speed"] = (data["speed"] + multiplier * (data["measured_speed"] - data["speed"])).clamp_min(0)
    result["channel_noise"] = multiplier * data["channel_noise"]
    result["shuffled_direction"] = result["measured_direction"].roll(max(1, data["heading"].shape[1] // 2), 1)
    result["position"][..., 0] += position_offset
    result["measured_position"][..., 0] += position_offset
    return result


@torch.no_grad()
def evaluate(checkpoint_path, output_prefix, seed=931731, length=256, batch_size=256, restore_step=64, device="cpu"):
    model, task, ckpt = load_checkpoint(checkpoint_path, device)
    model.requires_grad_(False)
    generator = torch.Generator(device=task.device).manual_seed(seed)
    base = task.realization(length, batch_size, generator)
    trained_length = ckpt["trainer_args"]["length"]
    training_reach_bound = 1.25 + (trained_length - 1) * task.config.dt * 0.35
    scenarios = {"same_region": (0.0, 1.0), "new_places": (10.0, 1.0),
                 "double_sensor_noise": (0.0, 2.0), "new_places_double_noise": (10.0, 2.0)}
    rows, archive = {}, {}
    for scenario, (offset, multiplier) in scenarios.items():
        data = alter_measurements(base, offset, multiplier)
        if offset:
            assert float(data["position"][..., 0].min()) > training_reach_bound
        inputs = {}
        for condition in {ckpt["condition"], "grid", "reverse", "shuffled"}:
            inputs[condition], _, _ = task.batch(length, batch_size, generator, condition, data)
        variants = {"native": inputs[ckpt["condition"]], "ordinary": inputs["grid"],
                    "reverse": inputs["reverse"], "shuffled": inputs["shuffled"]}
        restored = inputs[ckpt["condition"]].clone()
        restored[:restore_step] = inputs["grid"][:restore_step]
        variants["restored"] = restored
        eligible = (torch.arange(length, device=task.device)[:, None] >= 8).expand(length, batch_size)
        masks = {"all": eligible, "moving": eligible & ~data["stopped"],
                 "blank": eligible & (data["visible"] == 0),
                 "stationary_blank": eligible & data["stopped"] & (data["visible"] == 0),
                 "stationary_cued": eligible & data["stopped"] & (data["visible"] > 0)}
        rows[scenario] = {}
        for variant, x in variants.items():
            output = model(x)
            error = phase_error(output, data["target"])
            rows[scenario][variant] = {f"{name}_mae_deg": float(error[mask].abs().mean() * 180 / math.pi)
                                      for name, mask in masks.items() if mask.any()}
            rows[scenario][variant]["final_mae_deg"] = float(error[-1].abs().mean() * 180 / math.pi)
            rows[scenario][variant]["late_mae_deg"] = float(error[max(restore_step + 20, 8):].abs().mean() * 180 / math.pi)
            archive[f"{scenario}_{variant}_error"] = error.cpu().numpy()
        if ckpt["condition"] == "grid":
            assert rows[scenario]["native"] == rows[scenario]["ordinary"] == rows[scenario]["restored"]
        archive[f"{scenario}_position"] = data["position"].cpu().numpy()
        print(json.dumps({"scenario": scenario, "outcomes": rows[scenario]}), flush=True)
    archive["heading"] = base["heading"].cpu().numpy()
    archive["visible"] = base["visible"].cpu().numpy()
    archive["stopped"] = base["stopped"].cpu().numpy()
    result = {"checkpoint": str(checkpoint_path),
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "condition": ckpt["condition"], "seed": ckpt["seed"], "config": ckpt["config"],
              "steps_completed": ckpt["steps_completed"], "evaluation_seed": seed,
              "length": length, "batch_size": batch_size, "restore_step": restore_step,
              "protocol": {"frozen_weights": True, "recurrent_noise": 0,
                           "heading_initialization_common": True, "matched_latent_experience": True,
                           "training_position_coordinate_reach_bound_m": training_reach_bound,
                           "new_place_offset_m": [10.0, 0.0], "new_places_strictly_outside_training_reach": True,
                           "sensor_noise_multipliers": [1.0, 2.0],
                           "explicit_heading_control_receives_same_noisy_direction_as_pgc": True,
                           "ordinary_intervention": "all grid channels ordinary; explicit heading also removed after the common initialization"},
              "scenarios": rows}
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output_prefix.with_suffix(".npz"), **archive)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=931731)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--restore-step", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate(args.checkpoint, args.out, args.seed, args.length, args.batch_size, args.restore_step, args.device)


if __name__ == "__main__":
    main()
