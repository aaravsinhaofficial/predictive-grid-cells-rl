"""Evaluate every completed raw-input checkpoint in the separate eight-seed cohort."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import stats
import torch

from evaluate_heading_confirmation import evaluate
from ring_heading_model import HeadingConfig
from summarize_heading_confirmation import holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/ring_heading/raw_sensitivity_evaluation"
SEEDS = tuple(range(3001,3009))
CONDITIONS = ("grid","predictive","grid_heading")


def verify(checkpoint):
    locked = json.loads((ROOT/"docs/research/heading_confirmation_sources.json").read_text())
    expected_sources = {k:v for k,v in locked["sources"].items() if k != "scripts/run_ring_batch.py"}
    for relative,digest in expected_sources.items():
        matching = [v for k,v in checkpoint["source_sha256"].items() if k==relative or k.endswith("/"+relative)]
        if matching != [digest] or hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Raw-cohort source differs from locked trainer/task")
    seed = checkpoint["seed"]
    if seed not in SEEDS or checkpoint["condition"] not in CONDITIONS:
        raise ValueError("Unexpected raw-cohort identity")
    if checkpoint["config"] != HeadingConfig(encoder_size=0,hidden_size=64,code_seed=seed).to_dict():
        raise ValueError("Raw-cohort model/task configuration differs from protocol")
    if checkpoint["steps_completed"] != 6000 or checkpoint["stopping_reason"] != "steps_completed" or checkpoint["study_stage"] != "confirmatory":
        raise ValueError("Incomplete or incorrectly labeled raw-cohort training")
    for key,value in {"steps":6000,"length":128,"batch_size":64,"learning_rate":.001}.items():
        if checkpoint["trainer_args"][key] != value:
            raise ValueError("Raw-cohort training budget differs from protocol")
    return expected_sources


def paired(x):
    x = np.asarray(x,dtype=float)
    if len(x) != 8 or not np.isfinite(x).all():
        raise ValueError("All eight paired seed effects are required")
    mean = float(x.mean())
    half = float(stats.t.ppf(.975,7)*stats.sem(x))
    signs = np.asarray(list(itertools.product((-1.,1.),repeat=8)))
    null = signs@x/8
    p = float(np.mean(np.abs(null)>=abs(mean)-1e-12))
    return dict(mean_difference=mean,ci95=[mean-half,mean+half],p_exact_sign_flip=p,
                positive_seeds=int((x>0).sum()),negative_seeds=int((x<0).sum()),paired_values=x.tolist())


def summarize():
    rows=[]
    for seed in SEEDS:
        row={"seed":seed}
        for condition in CONDITIONS:
            path=OUT/f"{condition}_s{seed}.json"
            if not path.exists():
                return None
            d=json.loads(path.read_text())
            if d["cohort"] != "raw_architecture_sensitivity" or d["training_seed"] != seed or d["condition"] != condition:
                raise ValueError("Wrong raw-cohort report")
            row[condition]=d["metrics"]
        rows.append(row)
    effects={
        "tracking_pgc_minus_grid_deg":[r["predictive"]["same_region_native_mae_deg"]-r["grid"]["same_region_native_mae_deg"] for r in rows],
        "removal_minus_native_deg":[r["predictive"]["ordinary_minus_native_mae_deg"] for r in rows],
        "transfer_pgc_minus_grid_deg":[r["predictive"]["new_places_native_mae_deg"]-r["grid"]["new_places_native_mae_deg"] for r in rows]}
    contrasts={key:paired(value) for key,value in effects.items()}
    for contrast,p in zip(contrasts.values(),holm([v["p_exact_sign_flip"] for v in contrasts.values()])):
        contrast["p_holm_three_secondary"]=float(p)
    result={"cohort":"raw_architecture_sensitivity","n_seeds":8,"per_seed":rows,"contrasts":contrasts,
            "protocol":"docs/research/heading_raw_architecture_protocol.md",
            "note":"Separate secondary architecture cohort; never pooled with the16primary models."}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main(args):
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    while True:
        for folder in sorted((ROOT/"results/ring_cloud_artifacts").glob("ring-20261105-raw-*")):
            progress=folder/"batch_progress.json"
            if not progress.exists(): continue
            for job in json.loads(progress.read_text()):
                if job["exit_code"] or job["steps_completed"] != 6000: continue
                path=folder/(job["name"]+".pt")
                if not path.exists(): continue
                prefix=OUT/job["name"]
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if prefix.with_suffix(".json").exists():
                    prior=json.loads(prefix.with_suffix(".json").read_text())
                    if prior.get("cohort")=="raw_architecture_sensitivity":
                        if prior["checkpoint_sha256"] != digest: raise ValueError("Checkpoint changed")
                        continue
                checkpoint=torch.load(path,map_location="cpu",weights_only=False)
                source_hashes=verify(checkpoint)
                result=evaluate(path,prefix,allow_pilot=True)
                result["cohort"]="raw_architecture_sensitivity"
                result["provenance"].update(verified_training_sources=source_hashes,
                    secondary_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                result["protocol"]["secondary_plan"]="docs/research/heading_raw_architecture_protocol.md"
                result["protocol"]["override_meaning"]="Outside primary cohort by design; not an exploratory pilot. Eight new fixed-budget seeds."
                prefix.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
                print(json.dumps({"event":"raw_evaluated","name":job["name"]}),flush=True)
        result=summarize()
        if result:
            print(json.dumps({"event":"raw_complete","contrasts":result["contrasts"]}),flush=True)
            return
        if time.monotonic()-start>=args.watch_seconds: return
        time.sleep(30)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-seconds",type=float,default=10800)
    args=parser.parse_args()
    if not 0<=args.watch_seconds<=14400: raise ValueError("Watcher must be bounded")
    main(args)
