# Reproduce the nonlinear ring search

This release preserves the complete exploratory study: published learned-network calibration, native LSTM equilibrium searches, one- and two-phase grid-module assays, broad recurrent-state perturbations, a local comparison of two versus eight grid-phase nuisance directions, and native movement interventions with their bounded-pulse follow-up. It includes unsuccessful tests, all selected and missing cases, original inputs, frozen protocols, source hashes, reports, and figures. Packaging requires completed outputs, never positive scientific findings.

## What is included

- Nine result trees: `ring_nonlinear_design`, `lstm_module_phase_ring`, `lstm_module_phase_observable`, `lstm_module_phase_normal_audit`, `lstm_module_phase_torus`, `lstm_module_all_phase_control`, `native_movement_cycles`, `native_movement_pulses`, and `native_movement_bounded_pulses` under `results/`.
- Original 1-billion-frame Grid and PGC checkpoints and configurations, seeds 1–2: four checkpoint/configuration pairs.
- Four prior `context0_states.npz` native histories from `original_goal_bearing_manifold/pilot`.
- Eight `fixed_points` and eight `equilibrium_elimination` arrays from `lstm_endogenous_ring`, used by the bounded equilibrium search.
- Every tracked file under `code/`, `scripts/`, and `tests/`, plus the new study scripts and tests, to preserve local import dependencies. Relevant protocols, literature notes, environment metadata, and synthesis pages are included explicitly.
- The original published Memory-Pro model and inspected numeric data from Marraffa et al., pinned to upstream commit `f6702734490c90001db73cd7b7900998fb9a87cf`, with its MIT license and per-file hashes.

Raw arrays, traces, and checkpoints remain release assets separate from git. The release manifest identifies the exact source commit and gives every member's SHA256 and byte count. Each normalized `.tar.gz` part is below 1,500,000,000 bytes; files are not split internally. Each part also contains the identical complete manifest. `index.json` records the archive hashes and disjoint member lists.

## Verify before extracting

Use the repository commit named in `manifest.json`, and place all parts, `manifest.json`, and `index.json` in one directory. From that checkout:

```sh
python scripts/package_nonlinear_ring_outputs.py --verify-release --out /path/to/downloads
```

This uses only the Python standard library. It checks every compressed part, member path, byte count and hash; rejects links, duplicates and traversal paths; and verifies normalized ownership, permissions and timestamps. After it passes, extract all parts into a clean checkout of the stated source commit, preserving their repository-relative paths. Do not execute downloaded vendor code. The included independent benchmark loads reviewed numeric weights only.

## Check completion or prepare a release

```sh
python scripts/package_nonlinear_ring_outputs.py --plan
python scripts/package_nonlinear_ring_outputs.py --self-test
python scripts/package_nonlinear_ring_outputs.py --inventory-only --source-commit FULL_COMMIT_SHA
```

The first command inventories existing bytes and reports every missing completion record without creating archives. The synthetic self-test writes only tiny temporary fixtures. Inventory mode requires the final commit and all four model runs, torus terminal summaries and report, the completed local all-phase control, and both canonical and bounded pulse execution records and reports. Frozen input/source hashes must agree. None of these commands runs a policy, environment, optimizer, cloud job, or scientific experiment.

Only after publication preparation is explicitly complete, the publisher can use:

```sh
python scripts/package_nonlinear_ring_outputs.py --build --source-commit FULL_COMMIT_SHA --out /fresh/archive/directory
```

The builder rechecks inputs, writes deterministic sorted parts, verifies every archived byte, rechecks inputs after writing, and then writes the manifest and index. It refuses an existing nonempty destination. No upload, commit, or publication is performed by this script.

## Scientific reruns

The local study used Python 3.12.7, PyTorch 2.10, NumPy 1.26.4 and SciPy 1.13.1; see the included environment record and requirements. Exact native replay depends on the recorded CPU arithmetic, original batch sizes, noise states and checkpoint bytes. Use the frozen protocol files for numerical settings and the archived result reports for execution context. A rerun may differ across library or hardware versions; retain it in a separate directory and compare numerical tolerances and input hashes before comparing outcomes.

Execution dependencies are: published numeric weights → independent benchmark; original checkpoint/configuration + archived native histories → equilibrium and module-phase assays; module-phase state screen → fixed observables → broad normal audit; dense two-phase torus assay → local all-phase control; native movement census → canonical movement pulses → bounded pulses. The two-phase torus assay uses the original histories directly. Reporting scripts read those completed outputs. `scripts/plot_nonlinear_ring_search_summary.py` generates the overview from completed summaries.

The local all-phase control compares the same perturbations after accounting for two selected grid-phase tangents versus all eight known grid-phase tangents. It includes effective projection ranks, a generic cell-only disturbance, and a cell-only radial disturbance. Its locality guard bounds all eight implied phase shifts, including compensating shifts in the selected two phases, at both measurement windows. This is a local nuisance-direction analysis; it does not fit or certify a global eight-dimensional surface. Its frozen protocol and completed dense torus inputs are required before running:

```sh
python scripts/lstm_module_all_phase_control.py --out /path/to/fresh/all-phase-control --freeze-only
python scripts/lstm_module_all_phase_control.py --out /path/to/fresh/all-phase-control
```

Never overwrite an archived result directory when rerunning. Inspect each script's `--help` and its frozen protocol; use its output option or a separate checkout for scripts with fixed destinations. The packaging tool verifies existing results and does not regenerate missing inputs or results.

## Historical source provenance

Both the initial 400-second calibration and the later 1000-second calibration are preserved. The initial run did not qualify closure. Its exact executed source was reconstructed after the run by reversing the documented changes to the preserved later source; the reconstructed bytes match the SHA256 recorded at the initial run exactly. See `published_benchmark/source_reconstruction.json`. The 1000-second run has its original executed-source copy. The later corrected reduced-model validation is retained separately under `published_benchmark_corrected_rom`; it does not replace either full-network run. Source matching is checked during inventory rather than assuming the current script is identical to a historical script.

`packaging_plan.json` is a preparation snapshot and is not a final release manifest. It may show pending work. The final manifest and index are generated only after all required completion records and the source commit are available.
