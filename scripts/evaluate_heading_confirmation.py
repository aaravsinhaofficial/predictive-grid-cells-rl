"""Locked tracking and spatial-transfer endpoints for the heading study.

Primary errors exclude exactly the four initialization timesteps. Each trained
network is one replicate; episodes are averaged inside that replicate. The
separate anchoring endpoint is evaluated by evaluate_heading_anchor.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ring_heading_model import HeadingConfig, load_checkpoint
from ring_task_model import phase_error


def verify_training_sources(checkpoint, source_manifest=None):
    """Match recorded cloud/local source paths to the locked content hashes."""
    root = Path(__file__).resolve().parents[1]
    manifest_path = Path(source_manifest) if source_manifest else root / "docs/research/heading_confirmation_sources.json"
    manifest = json.loads(manifest_path.read_text())
    protocol_path = root / "docs/research/heading_confirmation_protocol.md"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Locked confirmation protocol bytes have changed")
    recorded = checkpoint.get("source_sha256", {})
    matches = {}
    for relative in ("scripts/train_ring_heading.py", "scripts/ring_heading_model.py",
                     "scripts/ring_task_model.py", "code/codes.py"):
        candidates = [digest for path, digest in recorded.items()
                      if path == relative or path.endswith("/" + relative)]
        if len(candidates) != 1 or candidates[0] != manifest["sources"][relative]:
            raise ValueError(f"Checkpoint source does not match locked protocol: {relative}")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != candidates[0]:
            raise ValueError(f"Evaluation import differs from locked training source: {relative}")
        matches[relative] = candidates[0]
    if checkpoint.get("study_stage") != "confirmatory" or checkpoint.get("exploratory_arm") is not False:
        raise ValueError("Checkpoint is not labeled confirmatory")
    expected_config = HeadingConfig(encoder_size=256, hidden_size=64, code_seed=checkpoint["seed"]).to_dict()
    if checkpoint["config"] != expected_config:
        raise ValueError("Full model/task configuration differs from locked protocol")
    if checkpoint.get("stopping_reason") != "steps_completed" or checkpoint["steps_completed"] != 6000:
        raise ValueError("Confirmation checkpoint did not finish its fixed training budget")
    args = checkpoint["trainer_args"]
    for key, expected in {"steps": 6000, "length": 128, "batch_size": 64, "learning_rate": .001}.items():
        if args[key] != expected:
            raise ValueError(f"Training setting differs from protocol: {key}")
    return {"locked_source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "verified_training_sources": matches, "protocol_sha256": manifest["protocol_sha256"]}


@torch.no_grad()
def evaluate(checkpoint_path, output_prefix, device="cpu", evaluation_seeds=(71001, 71002, 71003, 71004),
             episodes_per_seed=512, length=256, exclude_first=4, allow_pilot=False):
    model, task, checkpoint = load_checkpoint(checkpoint_path, device)
    model.requires_grad_(False)
    if not allow_pilot:
        assert checkpoint["seed"] in range(2001, 2017), "Confirmation training seed must be2001–2016"
        assert checkpoint["steps_completed"] == 6000
        assert task.config.encoder_size == 256 and task.config.hidden_size == 64
        assert tuple(evaluation_seeds) == (71001, 71002, 71003, 71004)
        assert episodes_per_seed == 512 and length == 256 and exclude_first == 4
        provenance = verify_training_sources(checkpoint)
    else:
        provenance = {"pilot_override": True, "verified_training_sources": None}
    assert checkpoint["condition"] in ("grid", "predictive", "grid_heading")
    training_reach = 1.25 + (checkpoint["trainer_args"]["length"] - 1) * task.config.dt * 0.35
    episode_values = {f"{place}_{variant}": [] for place in ("same_region", "new_places")
                      for variant in ("native", "ordinary")}
    evaluation_rows = []
    for evaluation_seed in evaluation_seeds:
        generator = torch.Generator(device=task.device).manual_seed(evaluation_seed)
        base = task.realization(length, episodes_per_seed, generator)
        row = {"evaluation_seed": evaluation_seed, "episodes": episodes_per_seed}
        for place, offset in (("same_region", 0.0), ("new_places", 10.0)):
            data = {key: value.clone() for key, value in base.items()}
            data["position"][..., 0] += offset
            data["measured_position"][..., 0] += offset
            if offset:
                assert float(data["position"][..., 0].min()) > training_reach
            output_cache = {}
            for variant, condition in (("native", checkpoint["condition"]), ("ordinary", "grid")):
                if condition not in output_cache:
                    x, target, _ = task.batch(length, episodes_per_seed, generator, condition, data)
                    output_cache[condition] = model(x)
                error = phase_error(output_cache[condition], data["target"])
                per_episode = error[exclude_first:].abs().mean(0).cpu().numpy().astype(np.float64) * (180 / np.pi)
                key = f"{place}_{variant}"
                episode_values[key].append(per_episode)
                row[f"{key}_mae_deg"] = float(per_episode.mean())
        evaluation_rows.append(row)
        print(json.dumps(row), flush=True)
    arrays = {key: np.concatenate(values) for key, values in episode_values.items()}
    metrics = {f"{key}_mae_deg": float(values.mean()) for key, values in arrays.items()}
    metrics["ordinary_minus_native_mae_deg"] = float((arrays["same_region_ordinary"] - arrays["same_region_native"]).mean())
    metrics["new_places_ordinary_minus_native_mae_deg"] = float((arrays["new_places_ordinary"] - arrays["new_places_native"]).mean())
    if checkpoint["condition"] == "grid":
        assert np.array_equal(arrays["same_region_native"], arrays["same_region_ordinary"])
        assert np.array_equal(arrays["new_places_native"], arrays["new_places_ordinary"])
    result = {"checkpoint": str(checkpoint_path),
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "condition": checkpoint["condition"], "training_seed": checkpoint["seed"],
              "provenance": {**provenance,
                  "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "torch_version": torch.__version__, "device": str(device)},
              "steps_completed": checkpoint["steps_completed"], "config": checkpoint["config"],
              "protocol": {"plan": "docs/research/heading_confirmation_protocol.md",
                  "evaluation_seeds": list(evaluation_seeds), "episodes_per_seed": episodes_per_seed,
                  "total_episodes_per_network": episodes_per_seed * len(evaluation_seeds),
                  "sequence_length": length, "exclude_first_steps": exclude_first,
                  "recurrent_noise": 0, "position_translation_m": [10.0, 0.0],
                  "training_reach_bound_m": training_reach, "new_positions_strictly_outside_training_reach": True,
                  "aggregation": "mean absolute circular error within episode, then equal mean over all episodes",
                  "independent_replication_unit": "training seed", "pilot_override": allow_pilot,
                  "native_and_ordinary_share_all_underlying_experience": True,
                  "anchoring_endpoint": "separate evaluate_heading_anchor.py output"},
              "metrics": metrics, "per_evaluation_seed": evaluation_rows}
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    arrays["evaluation_seed"] = np.repeat(np.asarray(evaluation_seeds), episodes_per_seed)
    np.savez_compressed(output_prefix.with_suffix(".npz"), **arrays)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[71001, 71002, 71003, 71004])
    parser.add_argument("--episodes-per-seed", type=int, default=512)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--exclude-first", type=int, default=4)
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = evaluate(args.checkpoint, args.out, args.device, args.evaluation_seeds,
                      args.episodes_per_seed, args.length, args.exclude_first, args.allow_pilot)
    print(json.dumps({"event": "complete", "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
