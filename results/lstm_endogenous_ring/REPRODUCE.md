# Reproducing the original and fixed 25M audits

The compact input manifest is [reproduction_inputs_manifest.json](reproduction_inputs_manifest.json); [reproduction_input_paths.txt](reproduction_input_paths.txt) lists exact repository-relative paths for packaging. It contains **74 files, 60,562,327 uncompressed bytes**: recorded native context inputs, their collection metadata, the two fixed 25.024M checkpoints, and checkpoint provenance/reference evaluations. Regenerated state trajectories are omitted.

The 16 original 1B LSTM checkpoints and configurations are already in the earlier `rl-map-checkpoints.tar.gz` release asset. Their existing hashes are in [the checkpoint manifest](../../docs/research/rl_map_checkpoint_archive.json). The new input bundle adds the 25M checkpoints at:

- `results/ring_cloud_artifacts/ring-20261105-vision-grid-01/final.pt`
- `results/ring_cloud_artifacts/ring-20261105-vision-pgc-01/final.pt`

Download the original models, verify the recorded archive hash, and extract at the repository root:

```bash
gh release download rl-map-memory-2026-11-06 \
  --repo aaravsinhaofficial/predictive-grid-cells-rl \
  --pattern rl-map-checkpoints.tar.gz
shasum -a 256 rl-map-checkpoints.tar.gz
tar -xzf rl-map-checkpoints.tar.gz
```

The new [reproduction-input and numerical-detail assets](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/endogenous-ring-tests-2026-11-06) must likewise be extracted at repository root, preserving their relative paths. [Input-archive size and checksum](../../docs/research/endogenous_ring_inputs_archive.json). Download both with:

```bash
gh release download endogenous-ring-tests-2026-11-06 \
  --repo aaravsinhaofficial/predictive-grid-cells-rl \
  --pattern 'endogenous-*.tar.gz'
shasum -a 256 endogenous-ring-inputs-v1.tar.gz
tar -xzf endogenous-ring-inputs-v1.tar.gz
```

Python dependencies are in [requirements-ring-analysis.txt](../../requirements-ring-analysis.txt). The recorded native inputs avoid recollecting trajectories on a different numerical backend. The analyses used local PyTorch 2.10; the behavioral training/evaluations used CUDA PyTorch 2.9.1. `--device cpu` can replace `mps` for renderer-based reproduction, with possible floating-point differences.

## Inputs and frozen methods

| Analysis | Required native input archives | Source and protocol |
|---|---|---|
| Original goal recovery, local spectra, fixed points | `results/original_goal_bearing_manifold/pilot/{gc100,pgc33}_s{1,2}/native_contexts.npz`, plus `collection.json` | `scripts/lstm_endogenous_curve_recovery.py`, `lstm_endogenous_metric_audit.py`, `lstm_endogenous_fixed_points.py`; their output folders contain frozen `protocol.json` |
| Saturation and validated quotient | `results/lstm_endogenous_ring/saturation_audit/{gc100,pgc33}_s{1,2}/native_4s_contexts.npz` | `lstm_endogenous_saturation_audit.py`, `lstm_endogenous_quotient_points.py`; corresponding frozen protocols |
| Exact equilibrium and native precision | Regenerated full-state search NPZs, the same original checkpoints | `lstm_endogenous_equilibrium_elimination.py`, `lstm_endogenous_precision_audit.py`; `equilibrium_elimination/protocol.json` |
| Original heading/travel development | The original goal native contexts above | `lstm_heading_memory_probe.py`, `lstm_heading_curve_flow.py`, `lstm_heading_current_release.py`, `lstm_travel_memory_probe.py`; frozen output protocols |
| Fixed 25M heading | `results/lstm_heading_memory/continued/{gc100,pgc33}/contexts/context{0,1}.npz` | `lstm_continued_heading_memory.py`; `docs/research/continued_heading_memory_protocol.json` and output implementation protocols |
| Fixed 25M travel | `results/lstm_travel_memory/continued/{gc100,pgc33}/contexts/context{0,1}.npz` | `lstm_travel_memory_probe.py`; per-cohort output protocols |
| Held-out original travel | `results/lstm_travel_memory/heldout_original/{gc100,pgc33}_s{3..8}/contexts/context{0,1}.npz` | `lstm_travel_memory_heldout.py`; frozen held-out protocol |
| Observable ring-factor control | Regenerated heading/travel state archives | `lstm_endogenous_observable_flow.py`, metadata-only travel adapter; `observable_history/protocol.json` |
| Every scheduled blind-update reset | Original seed1 pair and fixed 25M pair; original CUDA baseline/final evaluation JSONs | `lstm_endogenous_every_blind_reset.py`; `every_blind_update_reset/protocol.json` |

Every context archive in this table is enumerated and hashed in the compact input manifest. The native `contexts/` files are inputs; `context*_states.npz` files are regenerated outputs.

## Original goal and equilibrium sequence

These first two commands reconstruct the complete original development archives from the published native caches. The subsequent programs reuse those histories, without new context selection.

```bash
python scripts/lstm_goal_bearing_manifold.py --device cpu \
  --conditions gc100 pgc33 --seeds 1 2 --contexts 4 --angles 192 \
  --prepare 15 --future 15 --out results/original_goal_bearing_manifold/pilot
python scripts/refine_lstm_goal_bearing_manifold.py
python scripts/lstm_endogenous_curve_recovery.py
python scripts/lstm_endogenous_metric_audit.py
python scripts/lstm_endogenous_recovery_report.py
python scripts/lstm_endogenous_fixed_points.py
python scripts/lstm_endogenous_saturation_audit.py
python scripts/lstm_endogenous_quotient_points.py
python scripts/lstm_endogenous_equilibrium_elimination.py
python scripts/lstm_endogenous_precision_audit.py
python scripts/lstm_endogenous_equilibrium_report.py
```

The complete per-initializer numerical records are also available as the separate release asset **`endogenous-fixedpoint-detail.tar.gz`**. Its [manifest](fixedpoint_detail_archive_manifest.json) records every member and archive hash. Extracting at repository root restores `results/lstm_endogenous_ring/{fixed_points,quotient_points,equilibrium_elimination}/...`. The archive is a release download, not a Git-tracked binary. The sequence above regenerates it; `scripts/lstm_endogenous_reproduction_inventory.py` rebuilds the deterministic archive and manifest.

```bash
shasum -a 256 endogenous-fixedpoint-detail.tar.gz
tar -xzf endogenous-fixedpoint-detail.tar.gz
```

## Original heading/travel and observable controls

```bash
python scripts/lstm_heading_memory_probe.py
python scripts/plot_lstm_heading_memory.py
python scripts/lstm_heading_curve_flow.py
python scripts/lstm_heading_current_release.py
python scripts/lstm_travel_memory_probe.py --cohort original
python scripts/lstm_endogenous_observable_flow.py --originals
python scripts/lstm_endogenous_travel_observable.py
python scripts/lstm_endogenous_replay_cached_contexts.py --task heldout_travel \
  --source-root results/lstm_travel_memory/heldout_original
python scripts/lstm_endogenous_travel_observable.py \
  --source-root results/lstm_travel_memory/heldout_original \
  --out results/lstm_endogenous_ring/observable_history/travel_heldout_original
python scripts/lstm_endogenous_heldout_observable_report.py
python scripts/lstm_travel_action_context.py
python scripts/plot_lstm_travel_action_context.py
for action in forward backward strafe_left strafe_right; do
  python scripts/lstm_endogenous_travel_observable.py \
    --source-root "results/lstm_travel_memory/action_context_original/$action" \
    --out "results/lstm_endogenous_ring/observable_history/travel_action_context_original/$action"
done
```

The cache-replay wrapper replaces only the collector with published dictionaries and checks their checkpoint hashes. It calls the unchanged frozen analysis entry points. It does not choose new contexts or change candidate gates.

The four-action sensitivity uses the same original native inputs; it adds no required input archive. The cached replay wrapper was tested on both fixed 25M Grid heading contexts: every regenerated state/input array was bit-identical to the published local-MPS output. [Validation record](cached_context_reproduction_validation.json).

## Fixed 25M pair

Run this block from repository root. It reproduces both contexts in each population, then applies the same frozen observable controls.

```bash
python - <<'PY'
import subprocess
for condition, tag in [('gc100', 'grid'), ('pgc33', 'pgc')]:
    checkpoint = f'results/ring_cloud_artifacts/ring-20261105-vision-{tag}-01/final.pt'
    for task, family in [('continued_heading', 'heading'), ('continued_travel', 'travel')]:
        source = f'results/lstm_{family}_memory/continued/{condition}'
        subprocess.run(['python', 'scripts/lstm_endogenous_replay_cached_contexts.py',
            '--task', task, '--source-root', source, '--checkpoint', checkpoint], check=True)
    for context in (0, 1):
        subprocess.run(['python', 'scripts/lstm_endogenous_observable_flow.py',
            '--checkpoint', checkpoint,
            '--archive', f'results/lstm_heading_memory/continued/{condition}/context{context}_states.npz',
            '--label', f'continued/{condition}'], check=True)
    subprocess.run(['python', 'scripts/lstm_endogenous_travel_observable.py',
        '--source-root', f'results/lstm_travel_memory/continued/{condition}',
        '--out', f'results/lstm_endogenous_ring/observable_history/travel_continued/{condition}',
        '--continued'], check=True)
PY
python scripts/lstm_endogenous_observable_report.py
python -m pytest -q tests/test_lstm_endogenous_*.py
```

For the every-update reset control, run the adapter separately for each original/final checkpoint. A supplied parent JSON is appropriate on the matching CUDA backend. Omitting `--reference` reruns the unchanged parent evaluator on the current backend before the intervention; all four nonreset raw episode-row arrays must match exactly.

```bash
python scripts/lstm_endogenous_every_blind_reset.py \
  --checkpoint results/ring_cloud_artifacts/ring-20261105-vision-grid-01/final.pt \
  --reference results/ring_cloud_artifacts/ring-20261105-vision-grid-01/final_evaluation.json \
  --device cuda --out results/lstm_endogenous_ring/every_blind_update_reset/reproduced_grid25M
```

The input manifest and new 25M checkpoints cover the completed work above. The separately frozen final75M wrapper, `lstm_endogenous_continued_observable_v2.py`, is for future V2/V3 endpoints; it requires completed 75.072M total extra frames, 50.048M extension frames, correct arm provenance, and the exact fixed 25M parent and archive checkpoint hashes. It reuses the unchanged observable probe and gates.
