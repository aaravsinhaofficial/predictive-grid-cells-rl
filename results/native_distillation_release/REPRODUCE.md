# Reproduce the native policy-distillation pilot

This separate study starts from the original Grid and PGC seed-1 navigation checkpoints. It trains only their existing LSTM weights to match their full-vision policies during brief cue interruptions. It includes every fixed checkpoint, all baseline/final behavior and fixed-input replay conditions, and the prospectively frozen heading/travel geometry screens. No successful scientific result is required for packaging.

The archive contains both complete result trees, original checkpoint/configuration pairs, the two original native-census traces used only for evaluation, sealed masks, frozen protocols, source and input hashes, reports and figures. All tracked code/scripts/tests are included to preserve the import closure. Large arrays and checkpoints are release assets, separate from Git.

Use the source commit named in the final manifest. Download every archive part, `manifest.json` and `index.json` to one directory. Before extracting into a clean checkout of that commit:

```sh
python scripts/package_native_distillation_outputs.py --verify-release --out /path/to/downloads
```

Verification checks compressed hashes, every member's size/hash/path, and normalized archive metadata; it rejects links, duplicates and unsafe paths. The builder reuses the verified deterministic writer from `package_nonlinear_ring_outputs.py`, names these parts `native-distillation-outputs-part...`, updates the index, and verifies the renamed set again. This release's manifest identifies the study and source commit.

For independent reruns, use fresh output directories and the frozen protocols. The training CLI and protocol specify the fixed 180 updates, 1,036,800 native physics frames per model, original teacher checkpoint, and held-out data. Geometry commands and exact 2-/4-second timing are in `docs/research/native_policy_distillation_memory_commands.md`. Both fixed finals are tested regardless of behavior. Paired histories reconstruct each checkpoint's own recurrent state; own-policy contexts are reported separately.

The local run's program limit uses Python's monotonic clock. A recorded host-time discontinuity separates calendar duration from the program's reported runtime; see `results/native_policy_distillation/runtime_clock_audit.json`. Keep that distinction when comparing timings. Do not overwrite original results or silently substitute a different endpoint.

To inspect release readiness or test the packaging code without running models:

```sh
python scripts/package_native_distillation_outputs.py --plan
python scripts/package_native_distillation_outputs.py --self-test
```

Only after every terminal result and report exists and the exact sources are committed:

```sh
python scripts/package_native_distillation_outputs.py --build --source-commit FULL_COMMIT_SHA --out /fresh/archive/directory
```

The packaging commands do not execute experiments, train models, start cloud workers, upload files or publish a release.
