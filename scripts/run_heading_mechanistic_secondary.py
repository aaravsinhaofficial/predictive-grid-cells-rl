"""Consume completed confirmation PGC checkpoints with frozen secondary analyses."""
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
    parser.add_argument("--freeze", type=Path, default=Path("results/ring_heading/mechanistic_secondary_frozen_sources.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=4.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    locked = json.loads(args.freeze.read_text())["files"]
    started, completed, previous_sizes = time.monotonic(), {}, {}
    args.out.mkdir(parents=True, exist_ok=True)
    while time.monotonic() - started < args.max_hours * 3600:
        for name, expected in locked.items():
            actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Frozen secondary source changed: {name}")
        for seed in range(2001, 2017):
            if seed in completed:
                continue
            paths = sorted(args.checkpoint_root.glob(f"ring-20261105-confirm-*/predictive_s{seed}.pt"))
            if len(paths) > 1:
                raise RuntimeError(f"Duplicate checkpoint for seed{seed}: {paths}")
            if not paths:
                continue
            checkpoint = paths[0]
            size = checkpoint.stat().st_size
            if previous_sizes.get(str(checkpoint)) != size and not args.once:
                previous_sizes[str(checkpoint)] = size
                continue
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            outputs = {}
            for kind, script in (("phase_reduction", "evaluate_heading_phase_reduction.py"),
                                 ("memory_pulse", "evaluate_heading_memory_pulse.py")):
                out = args.out / f"{kind}_predictive_s{seed}"
                report_path = out.with_suffix(".json")
                outputs[kind] = str(report_path)
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    if report["checkpoint_sha256"] != checksum or report["evaluator_sha256"] != locked[f"scripts/{script}"]:
                        raise RuntimeError(f"Existing result has mismatched source/checkpoint: {report_path}")
                    continue
                print(json.dumps({"starting": kind, "seed": seed, "checkpoint": str(checkpoint)}), flush=True)
                with out.with_suffix(".log").open("w") as log:
                    subprocess.run([sys.executable, f"scripts/{script}", str(checkpoint), "--out", str(out),
                                    "--threads", str(args.threads)], stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(report_path.read_text())
                if report["steps_completed"] != 6000:
                    raise RuntimeError(f"Unexpected checkpoint training duration: {checkpoint}")
            completed[seed] = outputs
            print(json.dumps({"completed_seed": seed, "completed_count": len(completed)}), flush=True)
            (args.out / "progress.json").write_text(json.dumps({"expected_training_seeds": list(range(2001, 2017)),
                "completed": completed, "elapsed_seconds": time.monotonic() - started,
                "frozen_sources": locked}, indent=2) + "\n")
        if len(completed) == 16 or args.once:
            break
        time.sleep(20)
    print(json.dumps({"finished": len(completed) == 16, "completed_seeds": sorted(completed)}), flush=True)


if __name__ == "__main__":
    main()
