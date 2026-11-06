"""Bounded, isolated cloud workers for the ring-attractor study.

This is deliberately separate from gcp_fleet.py. Every VM has a provider-enforced
DELETE deadline, and its full cost envelope is reserved before creation.
`plan` and `status` never create cloud resources. See docs/research/ring_cloud_plan.md.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/research/ring_cloud_budget.json"
PROJECT = "connectome-ai-research"
BUCKET = "connectome-ai-research-popcode"
PREFIX = "ring-20261105"
IMAGE = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20261029"
RATE = 1.25
BUDGET = 500.0
ANCILLARY_RESERVE = 25.0
GIB = 1024 ** 3


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(cmd, *, check=True, timeout=180):
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"Command failed: {shlex.join(cmd)}\n{p.stderr[-3000:]}")
    return p


@contextlib.contextmanager
def locked_ledger():
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(LEDGER.read_text())
        if data["budget_usd"] != BUDGET or data["reserved_rate_usd_per_hour"] != RATE:
            raise RuntimeError("Budget/rate constants disagree with ledger; review rather than bypass.")
        yield data
        tmp = LEDGER.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(LEDGER)


def committed(data):
    return data["ancillary_reserve_usd"] + sum(
        r.get("actual_upper_bound_usd", r["reserved_usd"]) for r in data["runs"])


def record(data, name):
    for r in data["runs"]:
        if r["name"] == name:
            return r
    raise ValueError("Name is not in this study's ledger; refusing to touch it.")


def check_name(name):
    if not re.fullmatch(PREFIX + r"-[a-z0-9][a-z0-9-]{0,35}", name):
        raise ValueError(f"VM name must start {PREFIX}- and use lowercase letters, digits, hyphens.")


def relative_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Path must be within repository: {value}")
    return path.relative_to(ROOT)


def package_files(extra, include_only=False):
    files = []
    for folder in (() if include_only else ("code", "scripts", "tests")):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".sh", ".json")
                     and "__pycache__" not in p.parts)
    for value in extra:
        p = ROOT / relative_path(value)
        if not p.is_file():
            raise ValueError(f"Explicit include must be a file: {value}")
        files.append(p)
    files = sorted(set(files))
    for p in files:
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError(f"Refusing symlink/outside file: {p}")
        if any(s in p.name.lower() for s in ("credential", "secret", ".env", "id_rsa")):
            raise ValueError(f"Potential credential file cannot be packaged: {p.name}")
    size = sum(p.stat().st_size for p in files)
    if size > 2 * GIB:
        raise ValueError("Package exceeds 2 GiB; explicitly select fewer input files.")
    return files, size


WORKER = r'''import datetime, json, os, pathlib, signal, subprocess, sys, threading, time
root = pathlib.Path('/work/ring_repo')
cfg = json.loads((root / '.ring_job.json').read_text())
out = root / cfg['output_dir']
out.mkdir(parents=True, exist_ok=True)
stop = threading.Event()
sync_lock = threading.Lock()
def sync():
    with sync_lock:
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if size > 5 * 1024**3:
            raise RuntimeError('Artifact tree exceeds 5 GiB safety bound')
        subprocess.run(['gsutil', '-m', 'rsync', '-r', str(out), cfg['gcs_output']],
                       check=True, timeout=120)
def periodic():
    while not stop.wait(300):
        try: sync()
        except Exception as exc: print('SYNC_ERROR', str(exc), flush=True)
thread = threading.Thread(target=periodic, daemon=True)
thread.start()
cmd = cfg['command'][:]
if cmd[0] in ('python', 'python3'): cmd[0] = sys.executable
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
started = time.time()
with (out / 'worker.log').open('a', buffering=1) as log:
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        code = proc.wait(timeout=max(30, cfg['duration_seconds'] - 240))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        code = 124
stop.set()
thread.join(timeout=125)
result = dict(exit_code=code, elapsed_seconds=time.time()-started,
              finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              command=cmd, package_sha256=cfg['package_manifest_sha256'])
(out / 'WORKER_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
sync()
subprocess.run(['gsutil', 'cp', str(out / 'WORKER_RESULT.json'),
                cfg['gcs_prefix']+'/DONE.json'], check=True, timeout=60)
# Do not shut down here: a stopped VM may retain its boot disk. Root collects and
# deletes promptly; the independent Compute Engine DELETE deadline remains active.
while True: time.sleep(30)
'''


def describe(name, zone):
    p = run(["gcloud", "compute", "instances", "describe", name, "--project", PROJECT,
             "--zone", zone, "--format=json"], check=False)
    if p.returncode:
        if "was not found" in p.stderr or "notFound" in p.stderr:
            return None
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout)


def validate_identity(r, vm):
    if vm.get("labels", {}).get("study") != PREFIX:
        raise RuntimeError("VM study label mismatch; refusing modification.")
    if r.get("instance_id") and str(vm["id"]) != str(r["instance_id"]):
        raise RuntimeError("VM was replaced since launch; refusing modification.")


def startup_script(r, archive_hash):
    source = shlex.quote(r["gcs_prefix"] + "/package.tar.gz")
    setup = ("#!/bin/bash\nset -euo pipefail\n"
             "exec > >(tee -a /var/log/ring-worker-startup.log) 2>&1\n"
             "mkdir -p /work/ring_repo\ncd /work/ring_repo\n"
             f"gsutil cp {source} /work/package.tar.gz\n"
             f"echo '{archive_hash}  /work/package.tar.gz' | sha256sum --check\n"
             "tar xzf /work/package.tar.gz\n"
             "PYBIN=/opt/conda/bin/python\n"
             "if [ ! -x \"$PYBIN\" ]; then PYBIN=$(command -v python3); fi\n")
    if r["machine"].startswith("g2-") or r['machine']=='n1-standard-8':
        setup += ("for trial in $(seq 1 24); do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done\n"
                  "\"$PYBIN\" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'\n")
    else:
        setup += "\"$PYBIN\" -c 'import torch; print(torch.__version__, \"CPU worker\")'\n"
    return setup + "exec \"$PYBIN\" .ring_worker.py\n"


def plan(args):
    name = args.name or PREFIX + "-" + dt.datetime.now(dt.timezone.utc).strftime("%H%M%S")
    check_name(name)
    if not 0.25 <= args.hours <= 24:
        raise ValueError("Each launch must have a 0.25 to 24 hour provider-enforced deadline.")
    if args.machine not in ("g2-standard-4", "g2-standard-8", "e2-standard-8", "n1-standard-8"):
        raise ValueError("Only price-audited G2, E2-8, or N1-8 plus one T4 are allowed.")
    if args.zone not in ("us-central1-a", "us-central1-b", "us-central1-c"):
        raise ValueError("Only price-audited us-central1-a/b/c zones are allowed.")
    command = shlex.split(args.command)
    if not command:
        raise ValueError("A worker command is required.")
    output = str(relative_path(args.output_dir))
    if not output.startswith("results/ring_"):
        raise ValueError("Output directory must be an isolated results/ring_* path.")
    files, size = package_files(args.include, getattr(args, 'include_only', False))
    manifest = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in files]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return dict(name=name, zone=args.zone, machine=args.machine,
                duration_seconds=int(args.hours * 3600), reserved_usd=round((args.hours + .25) * RATE, 4),
                gcs_prefix=f"gs://{BUCKET}/{PREFIX}/{name}", command=command,
                output_dir=output, package_manifest=manifest, package_manifest_sha256=manifest_hash,
                package_uncompressed_bytes=size, status="reserved", reserved_at=now())


def launch(args):
    r = plan(args)
    # Keep the lock through creation: two launchers cannot spend the same budget.
    with locked_ledger() as data:
        if any(x["name"] == r["name"] for x in data["runs"]):
            raise ValueError("Run name already used; choose a new unique name.")
        active = [x for x in data["runs"] if "actual_upper_bound_usd" not in x]
        if len(active) >= data["max_active_workers"]:
            raise ValueError("Active-worker limit reached; finish/collect/delete the pilot first.")
        if committed(data) + r["reserved_usd"] > BUDGET:
            raise ValueError("Cumulative actual upper bounds + reservations exceed $500.")
        if describe(r["name"], r["zone"]) is not None:
            raise ValueError("VM name already exists in cloud; no changes made.")
        data["runs"].append(r)
        # Persist reservation before the first cloud mutation, even if interrupted.
        LEDGER.write_text(json.dumps(data, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="ring-cloud-") as tmp:
            tmp = Path(tmp)
            cfg = dict(r, gcs_output=r["gcs_prefix"] + "/outputs")
            (tmp / ".ring_job.json").write_text(json.dumps(cfg))
            (tmp / ".ring_worker.py").write_text(WORKER)
            archive = tmp / "package.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for entry in r["package_manifest"]:
                    # The exact bytes checked against the manifest enter the tar;
                    # a concurrent editor cannot change them between hash and add.
                    content = (ROOT / entry["path"]).read_bytes()
                    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                        r.update(status="package_changed", actual_upper_bound_usd=0.0)
                        LEDGER.write_text(json.dumps(data, indent=2) + "\n")
                        raise RuntimeError(f"Source changed while packaging: {entry['path']}; rerun with a new name.")
                    info = tarfile.TarInfo(entry["path"])
                    info.size = len(content)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(content))
                for file in (".ring_job.json", ".ring_worker.py"):
                    tar.add(tmp / file, arcname=file)
            archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            run(["gcloud", "storage", "cp", str(archive), r["gcs_prefix"] + "/package.tar.gz"], timeout=300)
            startup = tmp / "startup.sh"
            startup.write_text(startup_script(r, archive_hash))
            cmd = ["gcloud", "compute", "instances", "create", r["name"], "--project", PROJECT,
                   "--zone", r["zone"], "--machine-type", r["machine"],
                   "--image", IMAGE, "--image-project", "deeplearning-platform-release",
                   "--boot-disk-size=100GB", "--boot-disk-type=pd-balanced", "--boot-disk-auto-delete",
                   "--maintenance-policy=" + ("MIGRATE" if r['machine'].startswith('e2-') else "TERMINATE"),
                   "--no-restart-on-failure", "--reservation-affinity=none",
                   "--provisioning-model=STANDARD", "--instance-termination-action=DELETE",
                   f"--max-run-duration={r['duration_seconds']}s", "--scopes=storage-rw",
                   f"--labels=study={PREFIX},purpose=ring-attractor",
                   f"--metadata-from-file=startup-script={startup}", "--format=json", "--quiet"]
            if r['machine']=='n1-standard-8':
                cmd.append('--accelerator=type=nvidia-tesla-t4,count=1')
            if r["machine"].startswith("g2-") or r['machine']=='n1-standard-8':
                cmd.append("--metadata=install-nvidia-driver=True")
            r["create_requested_at"] = now()
            LEDGER.write_text(json.dumps(data, indent=2) + "\n")
            p = run(cmd, check=False, timeout=300)
            vm = describe(r["name"], r["zone"])
            if vm is None:
                r.update(status="launch_failed", actual_upper_bound_usd=0.0, failure=p.stderr[-2000:])
            else:
                validate_identity(r, vm)
                r.update(status=vm["status"], instance_id=vm["id"], created_at=vm["creationTimestamp"],
                         termination_timestamp=vm.get("scheduling", {}).get("terminationTimestamp"),
                         archive_sha256=archive_hash)
                schedule = vm.get("scheduling", {})
                if schedule.get("instanceTerminationAction") != "DELETE" or not schedule.get("maxRunDuration"):
                    raise RuntimeError("VM scheduling verification failed. Reservation held; delete this exact worker now.")
            print(json.dumps({k: v for k, v in r.items() if k != "package_manifest"}, indent=2))


def refresh(r):
    vm = describe(r["name"], r["zone"])
    if vm is not None:
        validate_identity(r, vm)
        r["status"] = vm["status"]
    else:
        r["status"] = "DELETED"
        start = r.get("created_at", r.get("create_requested_at"))
        if start and "actual_upper_bound_usd" not in r:
            elapsed = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(start)).total_seconds()
            # Provider TTL includes a 15 minute provision/termination allowance.
            r["actual_upper_bound_usd"] = round(min(r["reserved_usd"], max(0, elapsed) / 3600 * RATE), 4)
        elif "actual_upper_bound_usd" not in r:
            r["actual_upper_bound_usd"] = 0.0
        r["absence_verified_at"] = now()
    done = run(["gcloud", "storage", "cat", r["gcs_prefix"] + "/DONE.json"], check=False, timeout=60)
    if done.returncode == 0:
        r["worker_result"] = json.loads(done.stdout)
    return vm


def collect(r):
    dest = ROOT / "results/ring_cloud_artifacts" / r["name"]
    dest.mkdir(parents=True, exist_ok=True)
    run(["gcloud", "storage", "rsync", "--recursive", r["gcs_prefix"] + "/outputs", str(dest)], timeout=600)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    r.update(collected_at=now(), collected_bytes=size, local_artifacts=str(dest.relative_to(ROOT)))
    print(str(dest))


def verified_pilot(r):
    """A completed, collected training job validates a multi-job pilot worker.

    The remaining pilot jobs continue unchanged. Requiring the entire batch to
    finish would delay independent experiments despite verified throughput and
    artifact round-trip. Provider TTL and cumulative reservations still apply.
    """
    if r.get('status') == 'DELETED' and r.get('collected_at') and r.get('worker_result',{}).get('exit_code') == 0:
        return True
    if not r.get('collected_at') or not r.get('local_artifacts'):
        return False
    folder=ROOT/r['local_artifacts']
    progress=folder/'batch_progress.json'
    if not progress.exists():
        return False
    for job in json.loads(progress.read_text()):
        stem=job.get('name','')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',stem):
            continue
        path=folder/(stem+'.pt')
        summary=folder/(stem+'.json')
        if job.get('exit_code')==0 and job.get('steps_completed',0)>=1000 and path.is_file() and summary.is_file():
            report=json.loads(summary.read_text())
            if report.get('stopping_reason')!='steps_completed':
                continue
            if report.get('steps_completed')!=report.get('trainer_args',{}).get('steps'):
                continue
            r['pilot_verification']=dict(job=stem,steps=report['steps_completed'],
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                verified_at=now(),basis='Completed training, held-out validation, and cloud-to-local artifact collection.')
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("plan", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--name")
        p.add_argument("--command", required=True, help="Quoted command parsed into argv; never executed through a shell.")
        p.add_argument("--output-dir", default="results/ring_attractor")
        p.add_argument("--hours", type=float, default=2)
        p.add_argument("--zone", default="us-central1-a")
        p.add_argument("--machine", default="g2-standard-4")
        p.add_argument("--include", action="append", default=[], help="Explicit extra input file, e.g. one checkpoint.")
        p.add_argument("--include-only", action="store_true", help="Package only explicit includes (stable isolated training source).")
    sub.add_parser("ledger")
    sub.add_parser("status")
    p = sub.add_parser("logs")
    p.add_argument("name")
    for name in ("collect", "delete"):
        p = sub.add_parser(name)
        p.add_argument("name")
        if name == "delete":
            p.add_argument("--without-collect", action="store_true", help="Use only for failed/no-artifact workers.")
    p = sub.add_parser("set-parallelism")
    p.add_argument("max_active", type=int)
    args = ap.parse_args()
    if args.action == "plan":
        r = plan(args)
        r["package_files"] = len(r.pop("package_manifest"))
        print(json.dumps(r, indent=2))
        return
    if args.action == "launch":
        launch(args)
        return
    with locked_ledger() as data:
        if args.action == "status":
            for r in data["runs"]:
                refresh(r)
        elif args.action == "logs":
            r = record(data, args.name)
            vm = describe(r["name"], r["zone"])
            if vm is None:
                raise ValueError("VM already deleted; use collected worker.log instead.")
            validate_identity(r, vm)
            print(run(["gcloud", "compute", "instances", "get-serial-port-output", r["name"],
                       "--project", PROJECT, "--zone", r["zone"], "--port=1"]).stdout[-16000:])
        elif args.action in ("collect", "delete"):
            r = record(data, args.name)
            vm = refresh(r)
            if args.action == "collect":
                collect(r)
            elif vm is not None:
                if not args.without_collect:
                    collect(r)
                run(["gcloud", "compute", "instances", "delete", r["name"], "--project", PROJECT,
                     "--zone", r["zone"], "--quiet"], timeout=180)
                refresh(r)
        elif args.action == "set-parallelism":
            if not 1 <= args.max_active <= 8:
                raise ValueError("Parallel workers must be 1..8.")
            if args.max_active > 1 and not any(verified_pilot(r) for r in data['runs']):
                raise ValueError('A successful, collected pilot job is required before scaling.')
            data["max_active_workers"] = args.max_active
        print(json.dumps({"budget_usd": BUDGET, "committed_upper_bound_usd": committed(data),
                          "remaining_uncommitted_usd": BUDGET - committed(data),
                          "runs": [{k: v for k, v in r.items() if k != "package_manifest"}
                                   for r in data["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
