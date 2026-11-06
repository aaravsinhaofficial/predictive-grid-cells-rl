# Reproduce the fixed 40M blind-reorientation pilot

The [public release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/blind-reorientation-2026-11-06) contains `reorientation40-inputs-v1.tar.gz`. Extract it at the repository root. It preserves both actual worker names, all 14 prescribed checkpoint files, all raw evaluations and four native context inputs. The [archive record](../../docs/research/reorientation40_inputs_archive.json) gives its checksum, and the [manifest](reproduction_inputs_manifest.json) lists the SHA256 of every input. Intermediate outcomes remain available; `final.pt` is the fixed endpoint used for neural tests.

The original seed-1 model/config files are also required for provenance validation. They are available in the [original input release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/rl-map-memory-2026-11-06). The frozen source and training manifests are in this repository.

## Neural tests from exact native inputs

Use Python with NumPy, PyTorch, SciPy, Matplotlib and threadpoolctl. The reported deterministic neural analyses used local CPU PyTorch 2.10, two PyTorch threads and two BLAS threads. Run into a new directory; keep the archived inputs unchanged.

```sh
export OPENBLAS_NUM_THREADS=2
python scripts/lstm_reorientation_cached_replay.py \
  --checkpoint results/ring_cloud_artifacts/ring-20261105-reorient-grid-01/final.pt \
  --inputs results/lstm_reorientation_mechanism/final/gc100/contexts \
  --out /tmp/reorientation-replay-grid

python scripts/lstm_reorientation_cached_replay.py \
  --checkpoint results/ring_cloud_artifacts/ring-20261105-reorient-pgc-02/final.pt \
  --inputs results/lstm_reorientation_mechanism/final/pgc33/contexts \
  --out /tmp/reorientation-replay-pgc
```

The adapter replaces only native context collection with the exact saved contexts. It retains the original final-checkpoint validation, actual-physics replay, full-state geometry and all candidate criteria. The [complete original commands](../../docs/research/reorientation_mechanism_reproduction.md) also show fresh native collection and causal navigation. For a causal replay, pass the newly generated `context0_states.npz` and `context1_states.npz` to `scripts/lstm_reorientation_causal_navigation.py`, with the matching final checkpoint and a separate output directory.

Both Grid contexts, including every array field and every JSON metric, reproduced bit-for-bit in the [cached-input validation](cached_reproduction_validation.json).

The future-hidden-activity control reads the derived state arrays at their canonical locations. Copy only these regenerated arrays, leaving the archived native `contexts/` inputs intact, then run the unchanged adapter:

```sh
cp /tmp/reorientation-replay-grid/context0_states.npz /tmp/reorientation-replay-grid/context1_states.npz \
  results/lstm_reorientation_mechanism/final/gc100/
cp /tmp/reorientation-replay-pgc/context0_states.npz /tmp/reorientation-replay-pgc/context1_states.npz \
  results/lstm_reorientation_mechanism/final/pgc33/
python scripts/lstm_endogenous_reorientation_observable.py --condition gc100
python scripts/lstm_endogenous_reorientation_observable.py --condition pgc33 --require-complete
python scripts/plot_reorientation_observable.py
```

The saved `states[60]` has already processed the first free-choice input. The causal wrapper uses its actor output directly for free action one, then performs 29 further recurrent updates for the remaining 29 actions. It preserves actual angular inertia and uses the same physical and action random draws across interventions.

## Behavioral summaries and training audits

The primary frozen summary historically names the first PGC launch, which failed before an instance existed. The real run is `reorient-pgc-02`; [routing records](../rl_blind_reorientation_review/artifact_routing.json) preserve that distinction. To run the unchanged summary, create the documented local alias if it does not exist:

```sh
ln -s ring-20261105-reorient-pgc-02 \
  results/ring_cloud_artifacts/ring-20261105-reorient-pgc-01
python scripts/summarize_blind_reorientation.py --require-complete
python scripts/report_blind_reorientation_final.py
```

The summary consumes raw episode/window rows and retains all episodes, including zeros. No GPU rerun is needed to reproduce these tables. `scripts/audit_rl_blind_reorientation.py --folder <actual-worker-folder> --condition gc100|pgc33 --out <new-audit.json> --require-complete` checks the full training history, optimizer/runtime state, physics, cue identity controls and final hashes.

The read-only action diagnostic can be reproduced from all archived checkpoints with `scripts/lstm_reorientation_entropy_diagnostic.py`. Its saved CUDA generator blobs are retained, while explicitly prescribed private CPU streams generate its future rollout. This is a fresh diagnostic from a saved state, not an exact replay of the cloud worker’s future.

The original training and exact clean-evaluation references used CUDA PyTorch 2.9.1+cu129. Fresh GPU training may depend on backend details; exact baseline reproduction is explicitly required before the separately frozen entropy continuation begins. The primary report concerns one trained seed per population, with nested episodes and contexts.
