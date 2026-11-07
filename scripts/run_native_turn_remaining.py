"""Operational scheduler for the already frozen four-arm native-turn study.

No experiment parameters, endpoints, or deadlines are changed here. Finish the
two clean evaluations before starting the two noise acquisitions, then run the
four endpoint geometry jobs in two pairs. Each subprocess runs once only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/native_turn_distillation"
GEOMETRY = ROOT / "results/native_turn_geometry"
PROTOCOL = ROOT / "docs/research/native_turn_distillation_protocol.json"
EXPECTED = "ac37ba336f79299ed285c4bb0ef126296e5f8d46aee795e431d58544aab06c42"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def completed_clean_pair():
    for condition in ("gc100", "pgc33"):
        folder = BASE / condition / "clean"
        if (folder / "incomplete.json").exists():
            raise RuntimeError(f"Incomplete clean acquisition: {condition}")
        summary = folder / "summary.json"
        execution = BASE / f"{condition}_clean_execution.json"
        if not summary.exists() or not execution.exists():
            return False
        result = json.loads(summary.read_text())
        record = json.loads(execution.read_text())
        if not record.get("complete"):
            return False
        assert record["exit_code"] == 0 and record["no_cloud"]
        assert result["complete"] and result["updates"] == 180
        assert result["protocol_sha256"] == EXPECTED
    return True


def run_pair(kind, endpoint):
    jobs = []
    try:
        for condition in ("gc100", "pgc33"):
            if kind == "training":
                folder = BASE / condition / endpoint
                assert not folder.exists() or not any(folder.iterdir()), folder
                command = [sys.executable, "scripts/run_native_turn_distillation.py",
                           "--condition", condition, "--arm", endpoint]
                record_path = BASE / f"{condition}_{endpoint}_execution.json"
                log_path = BASE / f"{condition}_{endpoint}.log"
                cap = 2500
            else:
                training = json.loads((BASE / condition / endpoint / "summary.json").read_text())
                assert training["complete"] and training["updates"] == 180
                folder = GEOMETRY / condition / endpoint
                assert not folder.exists() or not any(folder.iterdir()), folder
                command = [sys.executable, "scripts/native_turn_geometry.py",
                           "--condition", condition, "--endpoint", endpoint]
                record_path = GEOMETRY / f"{condition}_{endpoint}_geometry_execution.json"
                log_path = GEOMETRY / f"{condition}_{endpoint}_geometry.log"
                cap = 7200
            assert not record_path.exists() and not log_path.exists()
            record = dict(condition=condition, arm=endpoint, endpoint=endpoint,
                          kind=kind, source_commit="4ac55dfb", protocol_sha256=EXPECTED,
                          started_utc=now(), no_cloud=True, command=command,
                          external_timeout_seconds=cap, complete=False)
            if kind == "geometry":
                record["geometry_protocol_sha256"] = hashlib.sha256(
                    (ROOT / "docs/research/native_turn_geometry_protocol.json").read_bytes()).hexdigest()
            save(record_path, record)
            output = log_path.open("x")
            process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                       stderr=subprocess.STDOUT,
                                       env={**os.environ, "OPENBLAS_NUM_THREADS": "2"})
            jobs.append(dict(process=process, output=output, start=time.monotonic(),
                             record=record, record_path=record_path, cap=cap, done=False))
            print(json.dumps(dict(event="started", **record)), flush=True)
        while not all(job["done"] for job in jobs):
            for job in jobs:
                if job["done"]:
                    continue
                elapsed = time.monotonic() - job["start"]
                if elapsed > job["cap"] and job["process"].poll() is None:
                    if "terminate_at" not in job:
                        job["record"]["external_timeout"] = True
                        job["terminate_at"] = time.monotonic()
                        job["process"].terminate()
                    elif time.monotonic() - job["terminate_at"] > 10:
                        job["process"].kill()
                code = job["process"].poll()
                if code is None:
                    continue
                job["output"].close()
                job["record"].update(complete=True, exit_code=code,
                                     finished_utc=now(), monotonic_seconds=elapsed)
                save(job["record_path"], job["record"])
                job["done"] = True
                print(json.dumps(dict(event="finished", **job["record"])), flush=True)
            time.sleep(1)
        assert all(job["record"]["exit_code"] == 0 for job in jobs), "A fixed job failed; no retry"
    finally:
        for job in jobs:
            if job["process"].poll() is None:
                job["process"].terminate()
                try:
                    job["process"].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    job["process"].kill()
                    job["process"].wait()
            if not job["output"].closed:
                job["output"].close()


def main():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == EXPECTED
    started = time.monotonic()
    while not completed_clean_pair():
        assert time.monotonic() - started < 900, "Clean completion wait exceeded 15 minutes"
        time.sleep(2)
    run_pair("training", "noise")
    run_pair("geometry", "clean")
    run_pair("geometry", "noise")
    print(json.dumps(dict(event="scheduler_complete", utc=now(), no_cloud=True)), flush=True)


if __name__ == "__main__":
    main()
