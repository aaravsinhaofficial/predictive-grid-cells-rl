"""Apply the unchanged memory-pulse evaluator to matched confirmation controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/ring_cloud_artifacts"))
    parser.add_argument("--out", type=Path, default=Path("results/ring_heading/mechanistic_confirmation"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=4.)
    args = parser.parse_args()
    locked = json.loads(Path("results/ring_heading/mechanistic_secondary_frozen_sources.json").read_text())["files"]
    extension = Path("docs/research/heading_memory_pulse_population_extension.md")
    extension_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
    completed, previous_sizes, start = {}, {}, time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    while time.monotonic() - start < args.max_hours * 3600:
        for name, expected in locked.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen source changed: {name}")
        if hashlib.sha256(extension.read_bytes()).hexdigest() != extension_sha:
            raise RuntimeError("Population-extension protocol changed")
        for seed in range(2001, 2017):
            for condition in ("grid", "grid_heading"):
                key = f"{condition}_s{seed}"
                if key in completed:
                    continue
                paths = sorted(args.checkpoint_root.glob(f"ring-20261105-confirm-*/{key}.pt"))
                if len(paths) > 1:
                    raise RuntimeError(f"Duplicate checkpoint: {key}")
                if not paths:
                    continue
                checkpoint = paths[0]
                size = checkpoint.stat().st_size
                if previous_sizes.get(key) != size:
                    previous_sizes[key] = size
                    continue
                checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                out = args.out / f"memory_pulse_{key}"
                report_path = out.with_suffix(".json")
                if not report_path.exists():
                    print(json.dumps({"starting": key}), flush=True)
                    with out.with_suffix(".log").open("w") as log:
                        subprocess.run([sys.executable, "scripts/evaluate_heading_memory_pulse.py", str(checkpoint),
                            "--out", str(out), "--threads", str(args.threads)], stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(report_path.read_text())
                if report["checkpoint_sha256"] != checksum or report["evaluator_sha256"] != locked["scripts/evaluate_heading_memory_pulse.py"]:
                    raise RuntimeError(f"Checkpoint/evaluator identity mismatch: {key}")
                if report["steps_completed"] != 6000:
                    raise RuntimeError(f"Unexpected training duration: {key}")
                completed[key] = str(report_path)
                print(json.dumps({"completed": key, "completed_count": len(completed)}), flush=True)
                (args.out / "population_pulse_progress.json").write_text(json.dumps({"expected_models": 32,
                    "completed": completed, "extension_sha256": extension_sha, "elapsed_seconds": time.monotonic() - start,
                    "frozen_sources": locked}, indent=2) + "\n")
        if len(completed) == 32:
            break
        time.sleep(20)
    print(json.dumps({"finished": len(completed) == 32, "completed_count": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
