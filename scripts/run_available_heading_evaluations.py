"""Evaluate completed, downloaded confirmation checkpoints without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    output = ROOT/"results/ring_heading/confirmation_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for folder in sorted((ROOT/"results/ring_cloud_artifacts").glob("ring-20261105-confirm-*")):
        progress = folder/"batch_progress.json"
        if not progress.exists():
            continue
        for job in json.loads(progress.read_text()):
            if job["exit_code"] != 0 or job["steps_completed"] != 6000:
                continue
            name = job["name"]
            path = folder/(name+".pt")
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            definitions = [("primary", "evaluate_heading_confirmation.py", [str(path)])]
            if name.startswith("predictive_") or args.all_anchors:
                definitions.append(("anchor", "evaluate_heading_anchor.py", ["--checkpoint", str(path)]))
            if args.memory:
                definitions.append(("memory", "evaluate_heading_memory.py", [str(path)]))
            for kind, script, arguments in definitions:
                prefix = output/(kind+"_"+name)
                report_path = prefix.with_suffix(".json")
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    if report["checkpoint_sha256"] != digest:
                        raise ValueError(f"Checkpoint bytes changed after evaluation: {path}")
                    evaluated_hash = report.get("evaluator_sha256", report.get("provenance", {}).get("evaluator_sha256"))
                    source_hash = hashlib.sha256((ROOT/"scripts"/script).read_bytes()).hexdigest()
                    if evaluated_hash == source_hash:
                        continue
                command = [sys.executable, str(ROOT/"scripts"/script), *arguments, "--out", str(prefix), "--threads", "2"]
                with prefix.with_suffix(".log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                print(json.dumps({"event": "evaluated", "kind": kind, "name": name}), flush=True)
            count += 1
    print(json.dumps({"event": "pass_complete", "available_checkpoints": count}), flush=True)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--all-anchors", action="store_true")
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.watch_seconds <= 14400:
        raise ValueError("Evaluation watcher must be bounded to at most four hours")
    start = time.monotonic()
    while True:
        if run(args) == 48 or time.monotonic()-start >= args.watch_seconds:
            break
        time.sleep(30)
