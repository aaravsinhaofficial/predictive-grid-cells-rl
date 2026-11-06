"""Watch the independent geometry cohort; test its one endpoint only when complete."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import t


SEEDS = tuple(range(5001, 5017))
CONDITIONS = ("grid", "predictive")
MANIFEST = Path("docs/research/heading_geometry_confirmation_sources.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return {"n": len(values), "mean": mean, "ci95": [mean - half, mean + half] if half is not None else None,
            "min": float(values.min()), "max": float(values.max())}


def exact_sign_flip(values):
    values = np.asarray(values, dtype=np.float64)
    assert len(values) == 16 and np.isfinite(values).all()
    bits = (np.arange(2 ** len(values), dtype=np.uint32)[:, None] >> np.arange(len(values), dtype=np.uint32)) & 1
    statistics = np.abs(((2 * bits.astype(np.float64) - 1) * values).mean(1))
    observed = float(abs(values.mean()))
    # Rounding allowance only ensures the mathematically identical original
    # and globally reversed sign patterns are included in the exact tail.
    tolerance = 16 * np.finfo(float).eps * max(1., observed)
    return {"p_two_sided_exact": float(np.mean(statistics >= observed - tolerance)),
            "sign_patterns": len(statistics), "statistic_absolute_mean": observed,
            "roundoff_tolerance": tolerance}


def aggregate(output, locked):
    rows = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            path = output / f"{condition}_s{seed}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if report["seed"] != seed or report["condition"] != condition or report["stage"] != "independent_geometry_confirmation":
                raise ValueError(f"Report identity mismatch: {path}")
            if report["source_hashes"] != locked:
                raise ValueError(f"Frozen report sources differ: {path}")
            if report["steps_completed"] != 6000 or report["n_phase"] != 1024 or report["refinement_levels"] != 20:
                raise ValueError(f"Protocol parameters differ: {path}")
            maps = {name: {key: value for key, value in item.items() if key != "adaptive_refinements"}
                    for name, item in report["maps"].items()}
            rows.append({"seed": seed, "condition": condition, "primary": report["primary"], "maps": maps,
                         "native_recurrence_sharpening_descriptive": report["native_recurrence_sharpening_descriptive"],
                         "checkpoint_sha256": report["checkpoint_sha256"], "report_sha256": digest(path)})
    complete = len(rows) == 32
    summary = {"complete": complete, "models_loaded": len(rows), "models_expected": 32,
               "protocol": "docs/research/heading_geometry_confirmation_protocol.md", "per_seed": rows,
               "primary_inference": None, "descriptive_controls": None,
               "no_seed_exclusion": True, "source_manifest_sha256": digest(MANIFEST),
               "aggregator_sha256": digest(__file__)}
    if complete:
        lookup = {(r["seed"], r["condition"]): r for r in rows}
        differences = np.array([lookup[s, "predictive"]["primary"] - lookup[s, "grid"]["primary"] for s in SEEDS])
        summary["primary_inference"] = {**summarize(differences), **exact_sign_flip(differences),
            "contrast": "PGC minus Grid log10 peak/median native memory-map sensitivity",
            "paired_differences": dict(zip(map(str, SEEDS), map(float, differences))),
            "geometric_mean_sharpness_ratio": float(10 ** differences.mean()),
            "multiplicity": "one prespecified primary; no family correction",
            "PGC_lower_pairs_descriptive": int((differences < 0).sum())}
        summary["groups"] = {}
        for condition in CONDITIONS:
            selected = [r for r in rows if r["condition"] == condition]
            summary["groups"][condition] = {}
            for variant in ("native_blank0", "native_blank40", "ordinary_blank0", "ordinary_blank40"):
                summary["groups"][condition][variant] = {key: summarize([r["maps"][variant][key] for r in selected])
                    for key in ("log10_peak_over_median", "absolute_peak_sensitivity", "absolute_median_sensitivity",
                                "state_radius", "decoded_heading_mae_deg", "decoded_coverage")}
                summary["groups"][condition][variant]["flagged_seeds"] = {flag: [r["seed"] for r in selected if r["maps"][variant]["flags"][flag]]
                    for flag in selected[0]["maps"][variant]["flags"]}
        controls = {}
        for name in ("native_blank0", "ordinary_blank0", "ordinary_blank40"):
            controls[name] = summarize([lookup[s, "predictive"]["maps"][name]["log10_peak_over_median"]
                                       - lookup[s, "grid"]["maps"][name]["log10_peak_over_median"] for s in SEEDS])
        controls["native_recurrence_sharpening_PGC_minus_Grid"] = summarize([
            lookup[s, "predictive"]["native_recurrence_sharpening_descriptive"] - lookup[s, "grid"]["native_recurrence_sharpening_descriptive"] for s in SEEDS])
        summary["descriptive_controls"] = controls
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-glob", default="results/ring_cloud_artifacts/ring-20261105-geometry-*/*.pt")
    parser.add_argument("--out", type=Path, default=Path("results/ring_heading/geometry_confirmation"))
    parser.add_argument("--max-hours", type=float, default=3.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    locked = json.loads(MANIFEST.read_text())["source_hashes"]
    prior_sizes, transient_retries, complete = {}, {}, set()
    started = time.monotonic()
    while True:
        for path, expected in locked.items():
            if digest(path) != expected:
                raise RuntimeError(f"Frozen source changed: {path}")
        files = list(Path().glob(args.checkpoint_glob))
        for seed in SEEDS:
            for condition in CONDITIONS:
                key = f"{condition}_s{seed}"
                if key in complete:
                    continue
                matches = [p for p in files if p.name == key + ".pt"]
                if len(matches) > 1:
                    raise RuntimeError(f"Duplicate model: {key}")
                if not matches:
                    continue
                checkpoint = matches[0]
                size = checkpoint.stat().st_size
                if prior_sizes.get(key) != size and not args.once:
                    prior_sizes[key] = size
                    continue
                prefix = args.out / key
                report_path = prefix.with_suffix(".json")
                if not report_path.exists():
                    with prefix.with_suffix(".log").open("w") as log:
                        process = subprocess.run([sys.executable, "scripts/evaluate_heading_geometry_confirmation.py", str(checkpoint),
                                                 "--out", str(prefix), "--threads", "2"], stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        failure = prefix.with_suffix(".log").read_text()
                        download_signatures = ("failed finding central directory", "EOFError", "Ran out of input",
                                               "pickle data was truncated", "not a ZIP archive", "failed reading zip archive")
                        retry_count = transient_retries.get(key, 0)
                        if any(marker in failure for marker in download_signatures) and retry_count < 3 and not args.once:
                            transient_retries[key] = retry_count + 1
                            prior_sizes.pop(key, None)
                            print(json.dumps({"transient_checkpoint_read_retry": key, "attempt": retry_count + 1,
                                              "seed_retained": True}), flush=True)
                            continue
                        raise RuntimeError(f"Evaluation failed for retained model {key}; inspect {prefix.with_suffix('.log')}")
                if json.loads(report_path.read_text())["checkpoint_sha256"] != digest(checkpoint):
                    raise ValueError(f"Checkpoint identity changed: {key}")
                complete.add(key)
                print(json.dumps({"completed": key, "models_completed": len(complete)}), flush=True)
        summary = aggregate(args.out, locked)
        if summary["complete"]:
            print(json.dumps({"complete": True, "primary": summary["primary_inference"]}), flush=True)
            return
        if args.once:
            return
        if time.monotonic() - started > args.max_hours * 3600:
            raise TimeoutError("Independent geometry confirmation remains incomplete")
        time.sleep(20)


if __name__ == "__main__":
    main()
