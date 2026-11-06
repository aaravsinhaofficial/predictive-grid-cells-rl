"""Run unchanged full-state memory and phase anchoring on the raw-input cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/ring_cloud_artifacts"))
    parser.add_argument("--out", type=Path, default=Path("results/ring_heading/raw_memory_evaluation"))
    parser.add_argument("--include-explicit", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=4.)
    args = parser.parse_args()
    conditions = ["grid", "predictive"] + (["grid_heading"] if args.include_explicit else [])
    paths = ["scripts/evaluate_heading_memory.py", "scripts/evaluate_heading_anchor.py",
             "scripts/ring_heading_model.py", "scripts/ring_task_model.py",
             "docs/research/heading_raw_architecture_protocol.md"]
    locked = {name: digest(name) for name in paths}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "frozen_sources.json").write_text(json.dumps({"files": locked, "conditions": conditions,
        "training_seeds": list(range(3001, 3009)), "memory_n_phase": 128, "memory_steps": 600,
        "anchor_n_phase": 72, "anchor_steps": 200, "primary_tracking_evaluation": "handled separately"}, indent=2) + "\n")
    completed, previous_sizes, started = {}, {}, time.monotonic()
    expected = 8 * len(conditions)
    while time.monotonic() - started < args.max_hours * 3600:
        for name, expected_hash in locked.items():
            if digest(name) != expected_hash:
                raise RuntimeError(f"Raw memory source/protocol changed: {name}")
        for seed in range(3001, 3009):
            for condition in conditions:
                key = f"{condition}_s{seed}"
                if key in completed:
                    continue
                checkpoints = sorted(args.checkpoint_root.glob(f"ring-20261105-raw-*/{key}.pt"))
                if len(checkpoints) > 1:
                    raise RuntimeError(f"Duplicate raw checkpoint: {key}")
                if not checkpoints:
                    continue
                checkpoint = checkpoints[0]
                size = checkpoint.stat().st_size
                if previous_sizes.get(key) != size:
                    previous_sizes[key] = size
                    continue
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if (saved["seed"] != seed or saved["condition"] != condition or saved["steps_completed"] != 6000
                        or saved["config"]["encoder_size"] != 0 or saved["config"]["hidden_size"] != 64):
                    raise RuntimeError(f"Unexpected raw checkpoint settings: {checkpoint}")
                del saved
                checksum = digest(checkpoint)
                reports = {}
                for kind, script in (("memory", "evaluate_heading_memory.py"), ("anchor", "evaluate_heading_anchor.py")):
                    out = args.out / f"{kind}_{key}"
                    report_path = out.with_suffix(".json")
                    reports[kind] = str(report_path)
                    if not report_path.exists():
                        command = [sys.executable, f"scripts/{script}"]
                        if kind == "anchor":
                            command.append("--checkpoint")
                        command += [str(checkpoint), "--out", str(out), "--threads", str(args.threads)]
                        print(json.dumps({"starting": kind, "model": key}), flush=True)
                        with out.with_suffix(".log").open("w") as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    report = json.loads(report_path.read_text())
                    if report["checkpoint_sha256"] != checksum:
                        raise RuntimeError(f"Checkpoint mismatch: {report_path}")
                    if kind == "memory" and report["evaluator_sha256"] != locked[f"scripts/{script}"]:
                        raise RuntimeError(f"Memory source mismatch: {report_path}")
                completed[key] = reports
                (args.out / "progress.json").write_text(json.dumps({"complete": len(completed) == expected,
                    "expected_models": expected, "completed": completed, "elapsed_seconds": time.monotonic() - started,
                    "source_hashes": locked}, indent=2) + "\n")
                subprocess.run([sys.executable, "scripts/summarize_heading_full_state_memory.py",
                    "--root", str(args.out), "--out", str(args.out / "memory_summary.json"),
                    "--seed-start", "3001", "--seed-count", "8", "--conditions", *conditions], check=True)
                print(json.dumps({"completed_model": key, "completed_count": len(completed)}), flush=True)
        if len(completed) == expected:
            return
        time.sleep(20)
    raise TimeoutError("Raw full-state memory cohort remains incomplete after4hours")


if __name__ == "__main__":
    main()
