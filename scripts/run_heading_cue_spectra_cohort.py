"""Frozen secondary cue-Jacobian cohort; all 16 trained PGC networks included."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from evaluate_heading_confirmation import verify_training_sources
from summarize_heading_confirmation import paired_summary, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/ring_heading/cue_spectra_confirmation"


def aggregate():
    rows = []
    for seed in range(2001, 2017):
        path = OUT/f"predictive_s{seed}.json"
        if not path.exists():
            return None
        report = json.loads(path.read_text())
        row = {"seed":seed, "checkpoint_sha256":report["checkpoint_sha256"]}
        for reference in ("cue_free", "native_anchored"):
            for variant in ("blank", "native", "ordinary"):
                selected = [r for r in report["rows"] if r["context"] < 5 and r["reference_state"]==reference and r["input"]==variant]
                if len(selected) != 5 or any(r["phase_gain_resolvable_fraction"] < 1 for r in selected):
                    raise ValueError("Incomplete/undefined phase derivative: report explicitly before aggregation")
                row[reference+"_"+variant] = {key:float(np.mean([r["distributions"][key]["median"] for r in selected])) for key in
                    ("phase_gain", "spectral_radius", "slow_mode_tangent_alignment", "largest_singular_value",
                     "readout_amplitude_before", "readout_amplitude_after")}
        rows.append(row)
    differences = {
        "phase_gain_native_minus_ordinary": [r["cue_free_native"]["phase_gain"]-r["cue_free_ordinary"]["phase_gain"] for r in rows],
        "phase_gain_native_minus_blank": [r["cue_free_native"]["phase_gain"]-r["cue_free_blank"]["phase_gain"] for r in rows],
        "spectral_radius_native_minus_ordinary": [r["cue_free_native"]["spectral_radius"]-r["cue_free_ordinary"]["spectral_radius"] for r in rows],
    }
    contrasts = {key: paired_summary(values) for key,values in differences.items()}
    for report, p in zip(contrasts.values(), holm([r["p_exact_sign_flip"] for r in contrasts.values()])):
        report["p_holm_three_secondary"] = float(p)
    result = {"stage":"mechanistic_secondary", "independent_unit":"training seed", "n":16,
              "protocol":"docs/research/heading_cue_spectra_protocol.md", "per_seed":rows, "contrasts":contrasts}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    files = [ROOT/"scripts/evaluate_heading_cue_spectra.py", ROOT/"docs/research/heading_cue_spectra_protocol.md"]
    frozen = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = OUT/"frozen_sources.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError("Secondary source changed after cohort start")
    manifest.write_text(json.dumps(frozen,indent=2)+"\n")
    start = time.monotonic()
    while True:
        for path in sorted((ROOT/"results/ring_cloud_artifacts").glob("ring-20261105-confirm-*/predictive_s*.pt")):
            if "_step" in path.stem:
                continue
            for file in files:
                if hashlib.sha256(file.read_bytes()).hexdigest() != frozen[str(file.relative_to(ROOT))]:
                    raise ValueError("Source changed during secondary cohort")
            out = OUT/path.stem
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if out.with_suffix(".json").exists():
                if json.loads(out.with_suffix(".json").read_text())["checkpoint_sha256"] != digest:
                    raise ValueError("Checkpoint changed during secondary cohort")
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            verify_training_sources(checkpoint)
            if checkpoint["seed"] not in range(2001,2017) or checkpoint["condition"] != "predictive":
                raise ValueError("Unexpected cohort checkpoint")
            with out.with_suffix(".log").open("w") as log:
                subprocess.run([sys.executable,str(files[0]),str(path),"--out",str(out),"--threads","2"],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            print(json.dumps({"event":"evaluated","seed":checkpoint["seed"]}),flush=True)
        result = aggregate()
        if result:
            print(json.dumps({"event":"complete","contrasts":result["contrasts"]}),flush=True)
            return
        if time.monotonic()-start >= args.watch_seconds:
            print(json.dumps({"event":"time_limit"}),flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-seconds",type=float,default=10800)
    args = parser.parse_args()
    if not 0 <= args.watch_seconds <= 14400:
        raise ValueError("Watcher must be bounded")
    main(args)
