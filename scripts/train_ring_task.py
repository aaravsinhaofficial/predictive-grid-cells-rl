"""Train the controlled circular-memory estimator used by a navigation agent.

Example: python scripts/train_ring_task.py --condition predictive --seed 1001 \
  --steps 1000 --out results/ring_attractor/pilot/predictive_s1001.pt
"""
import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from ring_task_model import CONDITIONS, RingConfig, RingRNN, RingTask, phase_error


@torch.no_grad()
def evaluate(model, task, condition, length, batch_size=256, seed=982451653):
    generator = torch.Generator(device=task.device).manual_seed(seed)
    x, target, data = task.batch(length, batch_size, generator, condition)
    model.eval()
    output = model(x)
    error = phase_error(output, target).abs() * (180 / math.pi)
    eligible = torch.arange(length, device=task.device)[:, None] >= 8
    masks = {"all": eligible.expand_as(error),
             "blank": (data["visible"] == 0) & eligible,
             "stationary_blank": data["stopped"] & eligible}
    result = {f"{name}_mae_deg": float(error[mask].mean())
              for name, mask in masks.items() if mask.any()}
    result["final_mae_deg"] = float(error[-1].mean())
    result["mse"] = float((output - target).square().mean())
    result["evaluation_seed"] = seed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-seconds", type=float, default=0)
    for field in fields(RingConfig):
        # Code seed follows paired training seed unless explicitly overridden.
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    values = {f.name: getattr(args, f.name) for f in fields(RingConfig)
              if getattr(args, f.name) is not None}
    values.setdefault("code_seed", args.seed)
    config = RingConfig(**values)
    model, task = RingRNN(config).to(device), RingTask(config, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps,
                                                          eta_min=args.learning_rate * 0.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".jsonl")
    if args.out.exists() or log_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.out}")
    start, history = time.monotonic(), []
    stopping_reason = "steps_completed"
    config_json = json.dumps(config.to_dict(), sort_keys=True)
    source_hash = hashlib.sha256(Path(__file__).with_name("ring_task_model.py").read_bytes()).hexdigest()
    metadata = {"condition": args.condition, "seed": args.seed, "config": config.to_dict(),
                "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
                "model_source_sha256": source_hash, "task": "controlled_1d_periodic_code_navigation_memory",
                "objective": "supervised_cos_sin_position_estimation",
                "ring_structure_imposed": False, "trainer_args": {k: str(v) if isinstance(v, Path) else v
                                                                                for k, v in vars(args).items()}}
    print(json.dumps({"event": "start", **metadata}), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        x, target, data = task.batch(args.length, args.batch_size, generator, args.condition)
        output, states = model(x, noise=data["recurrent_noise"], return_states=True)
        # Representation is supervised only through the task's 2-D output.
        # This generic activity cost neither specifies nor rewards a ring.
        task_loss = (output - target).square().mean()
        loss = task_loss + 1e-4 * states.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        elapsed = time.monotonic() - start
        should_stop = args.max_seconds > 0 and elapsed >= args.max_seconds
        if step == 1 or step % args.log_every == 0 or step == args.steps or should_stop:
            row = {"step": step, "loss": float(task_loss.detach()), "grad_norm": float(grad_norm),
                   "seconds": elapsed, "steps_per_second": step / elapsed}
            if step % args.eval_every == 0 or step == args.steps or should_stop:
                row["validation"] = evaluate(model, task, args.condition, args.length * 2)
            history.append(row)
            with log_path.open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if should_stop:
            stopping_reason = "time_limit"
            break
    final_validation = evaluate(model, task, args.condition, args.length * 2)
    checkpoint = {**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "steps_completed": step, "seconds": time.monotonic() - start,
                  "stopping_reason": stopping_reason, "validation": final_validation, "history": history}
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.out)
    summary = {k: v for k, v in checkpoint.items() if k not in ("model", "optimizer")}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "checkpoint": str(args.out), "seconds": checkpoint["seconds"],
                      "steps_completed": step, "validation": final_validation}), flush=True)


if __name__ == "__main__":
    main()
