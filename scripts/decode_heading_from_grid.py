"""Static diagnostic: can heading be decoded from the spatial population?

Position and heading are independently sampled. No motion history, velocity,
initial heading cue, or recurrent state is available to this decoder.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from ring_heading_model import HeadingConfig, HeadingTask
from ring_task_model import phase_error


def samples(task, generator, batch, condition, offset=0.0):
    position = (torch.rand(batch, 2, generator=generator) * 2 - 1) * 1.25
    position[:, 0] += offset
    angle = torch.rand(batch, generator=generator) * (2 * math.pi) - math.pi
    target = torch.stack((angle.cos(), angle.sin()), -1)
    measured_angle = angle + task.config.direction_noise * torch.randn(batch, generator=generator)
    direction = torch.stack((measured_angle.cos(), measured_angle.sin()), -1)
    position = position + task.config.position_noise * torch.randn(batch, 2, generator=generator)
    code = task.encode(position, direction, condition)
    code = (code + task.config.channel_noise * torch.randn(code.shape, generator=generator)).clamp(0, 1)
    return (code - 0.185185) / 0.22, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("grid", "predictive"), required=True)
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    task = HeadingTask(HeadingConfig(code_seed=args.seed))
    model = nn.Sequential(nn.Linear(128, args.hidden), nn.SiLU(), nn.Linear(args.hidden, args.hidden), nn.SiLU(), nn.Linear(args.hidden, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed + 41193)
    start, history = time.monotonic(), []
    for step in range(1, args.steps + 1):
        x, target = samples(task, generator, args.batch_size, args.condition)
        loss = (model(x) - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss.detach()), "seconds": time.monotonic() - start}
            history.append(row)
            print(json.dumps(row), flush=True)
    scores = {}
    with torch.no_grad():
        for name, offset in (("same_region", 0.0), ("new_places", 10.0)):
            x, target = samples(task, torch.Generator().manual_seed(591391), 8192, args.condition, offset)
            output = model(x)
            scores[name] = {"mae_deg": float(phase_error(output, target).abs().mean() * 180 / math.pi),
                            "mse": float((output - target).square().mean()),
                            "readout_norm": float(output.norm(dim=-1).mean())}
    result = {"condition": args.condition, "seed": args.seed, "steps": args.steps,
              "hidden": args.hidden, "batch_size": args.batch_size, "history": history,
              "evaluation": scores, "seconds": time.monotonic() - start,
              "protocol": "Heading independent of position; only spatial code provided; no memory or motion cue"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({**result, "model": model.state_dict()}, args.out.with_suffix(".pt"))
    print(json.dumps({"event": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
