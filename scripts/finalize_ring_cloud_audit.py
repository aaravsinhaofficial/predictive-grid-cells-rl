"""Read-only provider verification and conservative final study-cost record.

This script never creates, changes, or deletes any cloud resource. Existing
identity-checked collectors handle lifecycle operations. It waits for their
ledger records to close, verifies all56 final raw/geometry training jobs, then
independently checks Compute Engine for study VMs and disks.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT/"docs/research/ring_cloud_budget.json"
PREFIX = "ring-20261105-"
PROJECT = "connectome-ai-research"
TARGETS = [PREFIX+f"raw-{i:02d}" for i in (1, 2)]+[PREFIX+f"geometry-{i:02d}" for i in range(1, 7)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ledger():
    with LEDGER.with_suffix(".lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        raw = LEDGER.read_bytes()
    return json.loads(raw), raw


def cloud_list(resource):
    if resource not in ("instances", "disks"):
        raise ValueError("Only read-only instance/disk inventories are permitted")
    p = subprocess.run(["gcloud", "compute", resource, "list", "--project", PROJECT,
                        "--format=json"], capture_output=True, text=True, check=True, timeout=90)
    return json.loads(p.stdout)


def validate_collected():
    evidence, seen = [], {"raw": set(), "geometry": set()}
    for name in TARGETS:
        folder = ROOT/"results/ring_cloud_artifacts"/name
        manifest = json.loads((folder/"manifest.json").read_text())
        batch = json.loads((folder/"batch_result.json").read_text())
        worker = json.loads((folder/"WORKER_RESULT.json").read_text())
        if worker["exit_code"] != 0 or batch["planned_jobs"] != len(manifest["jobs"]) or batch["completed_jobs"] != len(manifest["jobs"]):
            raise ValueError(f"Incomplete or failed worker: {name}")
        group = "geometry" if "-geometry-" in name else "raw"
        completed = []
        for job in manifest["jobs"]:
            path = folder/(job["name"]+".pt")
            summary = json.loads(path.with_suffix(".json").read_text())
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing final checkpoint: {path}")
            expected = job["args"]
            if summary["steps_completed"] != 6000 or summary["stopping_reason"] != "steps_completed":
                raise ValueError("Partial checkpoint is not a completed training job")
            if summary["seed"] != expected["seed"] or summary["condition"] != expected["condition"]:
                raise ValueError("Checkpoint identity differs from worker manifest")
            key = (summary["seed"], summary["condition"])
            if key in seen[group]:
                raise ValueError("Duplicate model in completion audit")
            seen[group].add(key)
            completed.append({"name": job["name"], "checkpoint_sha256": digest(path),
                              "summary_sha256": digest(path.with_suffix(".json")), "steps_completed": 6000})
        evidence.append({"worker": name, "exit_code": 0, "planned": len(manifest["jobs"]),
                         "completed": len(completed), "models": completed})
    if seen["geometry"] != {(s, c) for s in range(5001, 5017) for c in ("grid", "predictive")}:
        raise ValueError("Geometry cohort is incomplete")
    if seen["raw"] != {(s, c) for s in range(3001, 3009) for c in ("grid", "predictive", "grid_heading")}:
        raise ValueError("Raw cohort is incomplete")
    return evidence


def finalize():
    ledger, raw = read_ledger()
    active = [r["name"] for r in ledger["runs"] if "actual_upper_bound_usd" not in r]
    if active:
        raise ValueError(f"Study ledger still has active reservations: {active}")
    completion = validate_collected()
    instances, disks = cloud_list("instances"), cloud_list("disks")
    names = {r["name"] for r in ledger["runs"]}
    study_vms = [v for v in instances if v["name"] in names or v["name"].startswith(PREFIX) or v.get("labels", {}).get("study") == PREFIX.rstrip("-")]
    study_disks = [d for d in disks if d["name"] in names or d["name"].startswith(PREFIX)]
    if study_vms or study_disks:
        raise ValueError("Provider still has study resources; investigate owned identities before further action")
    out = ROOT/"results/ring_cloud_artifacts/final_cleanup"
    out.mkdir(parents=True, exist_ok=True)
    (out/"ledger_snapshot.json").write_bytes(raw)
    compute = sum(r["actual_upper_bound_usd"] for r in ledger["runs"])
    ancillary = ledger["ancillary_reserve_usd"]
    result = {"verified_at_utc": now(), "project": PROJECT,
              "provider_study_instances": [], "provider_study_boot_disks": [],
              "unrelated_resources_modified_by_this_audit": False,
              "ledger_sha256": hashlib.sha256(raw).hexdigest(),
              "ledger_snapshot": str((out/"ledger_snapshot.json").relative_to(ROOT)),
              "reserved_rate_usd_per_vm_hour": ledger["reserved_rate_usd_per_hour"],
              "closed_vm_and_boot_disk_upper_bound_usd": round(compute, 4),
              "ancillary_storage_transfer_allowance_usd": ancillary,
              "total_conservative_bound_usd": round(compute+ancillary, 4),
              "budget_usd": ledger["budget_usd"], "active_reservations": 0,
              "accounting_scope": "Conservative study accounting at resource cleanup, not a billing invoice. Cloud artifacts retained; future storage duration is not an infinite-horizon bound.",
              "runs": [{k: r.get(k) for k in ("name", "machine", "zone", "status", "reserved_usd",
                         "actual_upper_bound_usd", "created_at", "absence_verified_at", "collected_at")} for r in ledger["runs"]],
              "latest_cohorts": completion}
    if result["total_conservative_bound_usd"] > ledger["budget_usd"]:
        raise ValueError("Budget ceiling was exceeded")
    target = ROOT/"docs/research/ring_cloud_final_cost_bound.json"
    target.write_text(json.dumps(result, indent=2)+"\n")
    report = ROOT/"docs/research/ring_cloud_final_cost_bound.md"
    report.write_text(f"""# Final cloud cleanup and conservative cost bound

Verified at **{result['verified_at_utc']}**: **zero study VMs and zero study boot disks remain** in Compute Engine. The provider inventory was queried independently after the identity-checked collectors closed all study reservations. This audit made no changes to unrelated resources.

All **32 geometry-confirmation** checkpoints (16 Grid/PGC pairs, seeds 5001–5016) and **24 raw-architecture** checkpoints (8 seeds ×3 conditions, seeds 3001–3008) were collected. Every final checkpoint reports 6,000 completed updates and normal step completion; all eight workers report successful completion. Exact checkpoint hashes are retained in the [audit record](ring_cloud_final_cost_bound.json).

| Conservative accounting component | USD |
|---|---:|
| Closed VM/boot-disk upper bounds, all study runs | {compute:.4f} |
| Storage and transfer allowance | {ancillary:.2f} |
| **Total conservative envelope** | **{compute+ancillary:.4f}** |
| User-authorized ceiling | {ledger['budget_usd']:.2f} |

**This is not an invoice.** The ledger uses a deliberately conservative $1.25 per VM-hour, including boot-disk costs, with a 15-minute reservation allowance per launch. Closed-run bounds use verified absence and the reserved provider-enforced lifetime. The separate $25 allowance covers ancillary storage/transfer. Source archives and model artifacts remain in the study's isolated storage prefix; any future storage after this snapshot depends on retention duration.

Evidence: [final audit record](ring_cloud_final_cost_bound.json), [live budget ledger](ring_cloud_budget.json), [pricing and lifecycle controls](ring_cloud_plan.md), and [read-only audit implementation](../../scripts/finalize_ring_cloud_audit.py). The complete ledger snapshot is recorded by SHA256 `{result['ledger_sha256']}` and retained locally at `{result['ledger_snapshot']}`.
""")
    print(json.dumps({"event": "final_cleanup_verified", "total_conservative_bound_usd": result["total_conservative_bound_usd"],
                      "completed_latest_models": 56, "report": str(report)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 14400:
        parser.error("Use a finite positive audit deadline no longer than4hours")
    start, previous = time.monotonic(), None
    while True:
        d, _ = read_ledger()
        active = [r["name"] for r in d["runs"] if "actual_upper_bound_usd" not in r]
        if active != previous:
            print(json.dumps({"event": "awaiting_cleanup", "time": now(), "active": active}), flush=True)
            previous = active
        if not active:
            finalize()
            break
        if time.monotonic()-start > args.max_seconds:
            raise TimeoutError("Cleanup audit deadline reached; no final cost/absence claim issued")
        time.sleep(30)
