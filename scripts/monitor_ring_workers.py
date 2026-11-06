"""Collect this study's workers and delete them after their DONE marker.

Never creates a resource. All deletion is delegated to ring_cloud.py, which
checks the ledger, study label and immutable instance identity first. Provider
DELETE deadlines remain an independent backstop.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--name", action="append", default=[], help="Optional explicit subset of ledger workers")
    args = parser.parse_args()
    if not 60 <= args.max_seconds <= 14400 or not 30 <= args.interval <= 600:
        raise ValueError("Monitor duration/interval outside bounded study limits")
    ledger_path = ROOT/"docs/research/ring_cloud_budget.json"
    ledger = json.loads(ledger_path.read_text())
    targets = {r["name"] for r in ledger["runs"] if "actual_upper_bound_usd" not in r}
    if args.name:
        requested = set(args.name)
        if not requested <= {r["name"] for r in ledger["runs"]}:
            raise ValueError("Monitor targets must already exist in the study ledger")
        targets &= requested
    logs = ROOT/"results/ring_cloud_artifacts/monitor_logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while time.monotonic()-started < args.max_seconds:
        ledger = json.loads(ledger_path.read_text())
        active = [r for r in ledger["runs"] if r["name"] in targets and "actual_upper_bound_usd" not in r]
        if not active:
            print(json.dumps({"event": "all_workers_collected_and_absent", "targets": sorted(targets)}), flush=True)
            return
        for record in active:
            done = subprocess.run(["gcloud", "storage", "cat", record["gcs_prefix"]+"/DONE.json"],
                                  capture_output=True, text=True, timeout=60)
            if done.returncode == 0:
                result = json.loads(done.stdout)
                action = "delete"  # Includes final artifact collection.
            else:
                result, action = None, "collect"
            with (logs/(record["name"]+".log")).open("a") as log:
                run = subprocess.run([sys.executable, str(ROOT/"scripts/ring_cloud.py"), action, record["name"]],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            print(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              "name": record["name"], "action": action, "returncode": run.returncode,
                              "worker_result": result}), flush=True)
        deadline = min(started+args.max_seconds, time.monotonic()+args.interval)
        while time.monotonic() < deadline:
            time.sleep(min(5, deadline-time.monotonic()))
    print(json.dumps({"event": "monitor_time_limit", "targets": sorted(targets)}), flush=True)


if __name__ == "__main__":
    main()
