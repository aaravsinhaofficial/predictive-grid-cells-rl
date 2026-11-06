"""Package the 104 final learned RNNs and eight designed-ring interface sets.

No pilot, intermediate checkpoint, public third-party model, or raw state array
is included. Paths inside the archive are relative to the repository root.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(output):
    rows=[]
    cohorts=(("primary",range(2001,2017),"confirm",("grid","predictive","grid_heading")),
             ("raw",range(3001,3009),"raw",("grid","predictive","grid_heading")),
             ("geometry",range(5001,5017),"geometry",("grid","predictive")))
    for cohort,seeds,worker,conditions in cohorts:
        for seed in seeds:
            for condition in conditions:
                pattern=f"ring-20261105-{worker}-*"
                paths=sorted((ROOT/"results/ring_cloud_artifacts").glob(f"{pattern}/{condition}_s{seed}.pt"))
                if len(paths)!=1:
                    raise ValueError(f"Expected exactly one final checkpoint: {cohort} {condition} {seed}")
                path=paths[0]
                report_path=ROOT/{"primary":f"results/ring_heading/confirmation_evaluation/primary_{condition}_s{seed}.json",
                                  "raw":f"results/ring_heading/raw_sensitivity_evaluation/{condition}_s{seed}.json",
                                  "geometry":f"results/ring_heading/geometry_confirmation/{condition}_s{seed}.json"}[cohort]
                report=json.loads(report_path.read_text())
                sha=digest(path)
                if (report["checkpoint_sha256"]!=sha or report["steps_completed"]!=6000 or
                    report.get("training_seed",report.get("seed"))!=seed or report["condition"]!=condition):
                    raise ValueError(f"Checkpoint does not match its verified evaluation: {path}")
                rows.append({"cohort":cohort,"condition":condition,"seed":seed,
                             "path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,
                             "sha256":sha,"evaluation":str(report_path.relative_to(ROOT))})
    for seed in range(4001,4009):
        path=ROOT/f"results/ring_attractor/designed_compass/seed{seed}/model.pt"
        report_path=path.with_name("summary.json")
        report=json.loads(report_path.read_text())
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        if (checkpoint["seed"]!=seed or report["seed"]!=seed or not checkpoint["designed_ring"] or
            set(checkpoint["states"])!={"grid","predictive","grid_heading"} or
            checkpoint["sources"]!=report["sources"]):
            raise ValueError(f"Designed model does not match its record: {path}")
        for name,expected in report["sources"].items():
            if digest(ROOT/name)!=expected:
                raise ValueError(f"Designed model source changed: {name}")
        rows.append({"cohort":"designed_ring","condition":"three trained interfaces in one file",
                     "seed":seed,"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,
                     "sha256":digest(path),"evaluation":str(report_path.relative_to(ROOT))})
    source_paths=["scripts/ring_task_model.py","scripts/ring_heading_model.py","scripts/train_ring_heading.py",
                  "scripts/run_ring_batch.py","code/codes.py","docs/research/heading_confirmation_protocol.md",
                  "docs/research/heading_confirmation_sources.json","docs/research/heading_raw_architecture_protocol.md",
                  "docs/research/heading_geometry_confirmation_protocol.md","docs/research/heading_geometry_confirmation_sources.json",
                  "scripts/designed_predictive_ring.py","scripts/train_designed_predictive_ring.py",
                  "scripts/decode_predictive_phase_difference.py","scripts/evaluate_ring_heading_navigation.py",
                  "results/ring_attractor/designed_compass/frozen_protocol.json"]
    source_manifest=[{"path":name,"sha256":digest(ROOT/name)} for name in source_paths]
    manifest={"description":"All 48 primary, 24 raw-architecture and 32 independent geometry RNNs, plus eight designed-ring interface sets; no outcome-based exclusions",
              "checkpoint_files":112,"unconstrained_RNNs":104,"designed_ring_interface_sets":8,
              "models":rows,"training_sources":source_manifest,"uncompressed_checkpoint_bytes":sum(r["bytes"] for r in rows)}
    assert len(rows)==112
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(output,"w:gz",compresslevel=6) as archive:
        for name in [r["path"] for r in rows]+source_paths:
            archive.add(ROOT/name,arcname=name,recursive=False)
        payload=(json.dumps(manifest,indent=2)+"\n").encode()
        info=tarfile.TarInfo("heading_checkpoints_manifest.json");info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
    manifest["archive"]={"name":output.name,"bytes":output.stat().st_size,"sha256":digest(output)}
    target=ROOT/"docs/research/heading_checkpoint_archive.json"
    target.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"models":len(rows),"archive":manifest["archive"],"manifest":str(target.relative_to(ROOT))},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    package(parser.parse_args().out)
