# Reproduce the matched entropy continuations

**All four fixed final checkpoints passed independent completion audits.** The input archive requires all four completed endpoints and their native inputs to pass validation. [Exact saved-input replay](cached_reproduction_validation.json) reproduced every neural array value and primary JSON result for both Grid/original contexts, including identical NPZ file hashes.

The `reorientation90-inputs-v1.tar.gz` preserves all 28 prescribed checkpoint files: incoming, five fixed blocks and final for each population/entropy arm. It also includes raw evaluations, complete audits, native context inputs and explicit actual-worker routing. Extract at the repository root. The original model/config files and exact 40M parents are required by the unchanged provenance checks: [original-model release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/rl-map-memory-2026-11-06), [40M parent release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/blind-reorientation-2026-11-06).

The final models have genuinely received 90,132,480 additional learning frames: 40,058,880 in the parent task and 50,073,600 in this continuation. They are never relabeled as 40M checkpoints. The [frozen method](../../docs/research/rl_blind_reorientation_entropy_protocol.json) changes only the entropy weight, restoring each parent's full policy, optimizer, simulator, recurrent state and private random streams.

## Geometry and causal movement from exact cached inputs

Run from the repository root with CPU PyTorch, NumPy, SciPy, Matplotlib and threadpoolctl installed. The scripts preserve the original numerical helpers and candidate criteria. A separate destination keeps archived native context inputs intact.

```sh
export OPENBLAS_NUM_THREADS=2
for arm in original explore; do
  for condition in gc100 pgc33; do
    python scripts/lstm_reorientation_entropy_cached_replay.py \
      --arm "$arm" --condition "$condition" \
      --inputs "results/lstm_reorientation_entropy_mechanism/$arm/$condition/contexts" \
      --out /tmp/reorientation90-cached
    for context in 0 1; do
      python scripts/lstm_reorientation_entropy_mechanism.py \
        --arm "$arm" --condition "$condition" --context "$context" \
        --mode causal --out /tmp/reorientation90-cached
    done
  done
done
```

Only native context collection is replaced by exact archived inputs. The endpoint, source, parent and producer validators remain intact. Full-state dynamics, one-pulse interventions, actual movement physics and first-hit stopping use the frozen functions. State 60 has already processed the first free-choice input; action one uses its actor output directly, followed by 29 further recurrent updates.

## Future-hidden activity and the learned action layer

The frozen secondary adapters expect the derived arrays at canonical paths. Copy only regenerated state arrays; retain the archived `contexts/`, producer metadata and collection records:

```sh
for arm in original explore; do
  for condition in gc100 pgc33; do
    cp "/tmp/reorientation90-cached/$arm/$condition/context0_states.npz" \
       "/tmp/reorientation90-cached/$arm/$condition/context1_states.npz" \
       "results/lstm_reorientation_entropy_mechanism/$arm/$condition/"
    python scripts/lstm_endogenous_entropy_observable.py \
      --arm "$arm" --condition "$condition"
  done
done
python scripts/lstm_endogenous_entropy_observable.py --summarize-only --require-complete
python scripts/lstm_reorientation_entropy_actor_readout.py
```

The future-activity test keeps every hidden unit and all original perturbation and geometry criteria. The action-layer decomposition uses the exact learned linear weights and probabilities, with no new recurrent simulation or trained readout. All 64 observable comparisons and all 128 readout rows remain in the outputs; cue variants, contexts and angles are nested measurements.

## Behavior and full training history

```sh
python scripts/summarize_reorientation_entropy.py --require-complete
```

The summary reads the explicit `name` and `artifact_subdir` entries in [launch routing](../../docs/research/rl_blind_reorientation_entropy_launch.json), so failed first starts and corrected `experiment/` output folders are not confused. Raw baseline and all five block evaluations remain available. Outcomes are per original episode, including zero-event episodes. Paired uncertainty keeps both episodes of each environment slot together; it is conditional evaluation uncertainty for this original seed pair.

Run `scripts/audit_rl_blind_reorientation_entropy.py --folder <actual-worker/experiment> --condition gc100|pgc33 --arm original|explore --out <new-audit.json> --require-complete` to validate true cumulative frames, all checkpoints/evaluations, exact resume, optimizer/noise updates and source hashes.

Training and its exact baseline references use CUDA PyTorch 2.9.1+cu129. Deterministic neural analyses use CPU with two PyTorch and two BLAS threads. Each cloud run must reproduce every parent baseline episode, event, window and counter before learning. Provider time limits and cleanup are retained in the cloud audit records.
