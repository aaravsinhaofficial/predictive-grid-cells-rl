"""Train a new heading integrator using original hexagonal grid/PGC inputs.

Fixed intermediate checkpoints are saved for every condition; the final
training step is not selected according to a favorable population comparison.
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

from ring_heading_model import CONDITIONS, HeadingConfig, HeadingRNN, HeadingTask
from ring_task_model import phase_error


@torch.no_grad()
def evaluate(model, task, condition, length=256, batch_size=128, seed=31872941):
    generator = torch.Generator(device=task.device).manual_seed(seed)
    x, target, data = task.batch(length, batch_size, generator, condition)
    model.eval()
    output = model(x)
    error = phase_error(output, target).abs() * 180 / math.pi
    eligible = (torch.arange(length, device=task.device)[:, None] >= 8).expand_as(error)
    masks = {"all": eligible, "moving": eligible & ~data["stopped"],
             "blank": eligible & (data["visible"] == 0),
             "stationary_blank": eligible & data["stopped"] & (data["visible"] == 0),
             "stationary_cued": eligible & data["stopped"] & (data["visible"] > 0)}
    result = {f"{name}_mae_deg": float(error[mask].mean()) for name, mask in masks.items() if mask.any()}
    result.update(final_mae_deg=float(error[-1].mean()), mse=float((output - target).square().mean()), evaluation_seed=seed)
    return result


def save(model, optimizer, path, metadata, step, elapsed, validation, history, stopping_reason):
    checkpoint = {**metadata, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "steps_completed": step, "seconds": elapsed, "validation": validation,
                  "history": history, "stopping_reason": stopping_reason}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(json.dumps({k: v for k, v in checkpoint.items()
                                                    if k not in ("model", "optimizer")}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-at", type=int, nargs="*", default=[3000])
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument('--study-stage', choices=('exploratory','confirmatory'), default='exploratory')
    for field in fields(HeadingConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(field.default), default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    values = {f.name: getattr(args, f.name) for f in fields(HeadingConfig) if getattr(args, f.name) is not None}
    values.setdefault("code_seed", args.seed)
    config = HeadingConfig(**values)
    model, task = HeadingRNN(config).to(device), HeadingTask(config, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.learning_rate * 0.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".jsonl")
    if args.out.exists() or log_path.exists():
        raise FileExistsError(args.out)
    sources = [Path(__file__), Path(__file__).with_name("ring_heading_model.py"),
               Path(__file__).with_name("ring_task_model.py"), Path(__file__).resolve().parents[1] / "code/codes.py"]
    metadata = {"condition": args.condition, "seed": args.seed, "config": config.to_dict(),
        "task": "hexagonal_grid_code_heading_memory", "exploratory_arm": args.study_stage=='exploratory',
        "study_stage": args.study_stage,
        "objective": "supervised_cos_sin_heading_estimation", "ring_structure_imposed": False,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "population": task.population.describe(),
        "trainer_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    print(json.dumps({"event": "start", **metadata}), flush=True)
    start, history, stopping_reason = time.monotonic(), [], "steps_completed"
    for step in range(1, args.steps + 1):
        model.train()
        x, target, data = task.batch(args.length, args.batch_size, generator, args.condition)
        output, states = model(x, noise=data["recurrent_noise"], return_states=True)
        task_loss = (output - target).square().mean()
        loss = task_loss + 1e-4 * states.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        elapsed = time.monotonic() - start
        should_stop = args.max_seconds > 0 and elapsed >= args.max_seconds
        should_evaluate = step % args.eval_every == 0 or step == args.steps or step in args.save_at or should_stop
        validation = evaluate(model, task, args.condition, args.length * 2) if should_evaluate else None
        if step == 1 or step % args.log_every == 0 or should_evaluate:
            row = {"step": step, "loss": float(task_loss.detach()), "grad_norm": float(grad_norm),
                   "seconds": time.monotonic() - start, "steps_per_second": step / (time.monotonic() - start)}
            if validation is not None:
                row["validation"] = validation
            history.append(row)
            with log_path.open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step in args.save_at and step != args.steps:
            checkpoint_path = args.out.with_name(args.out.stem + f"_step{step}" + args.out.suffix)
            save(model, optimizer, checkpoint_path, metadata, step, time.monotonic() - start,
                 validation, history, "fixed_intermediate_checkpoint")
        if should_stop:
            stopping_reason = "time_limit"
            break
    validation = evaluate(model, task, args.condition, args.length * 2)
    save(model, optimizer, args.out, metadata, step, time.monotonic() - start, validation, history, stopping_reason)
    print(json.dumps({"event": "complete", "checkpoint": str(args.out), "steps_completed": step,
                      "seconds": time.monotonic() - start, "validation": validation}), flush=True)


if __name__ == "__main__":
    main()
