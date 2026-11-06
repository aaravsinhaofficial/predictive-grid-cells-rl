# Reproduce dynamics in the learned action scores

This follow-up applies the exact trained action layer to recurrent states. It fits no decoder. The [frozen protocol](protocol.json) includes every available original-agent movement context, both fixed 40M models and all four fixed 90M models: 152 nested branch/metric checks. The original “heldout” label identifies the earlier collection cohort; earlier full-state and hidden-activity results were already known when this follow-up was planned.

The complete reproduction uses the three input assets below. Their publication status and checksums are recorded in the release and corresponding archive records.

## Exact input archives

Extract these assets at the root of a repository clone for the release:

- `actor-score-existing-inputs-v1.tar`: all 32 original/40M state NPZs and their context metadata. These preserve the exact byte hashes locked in the protocol.
- `reorientation90-neural-arrays-v1.tar`: all eight genuine 90M state NPZs and their producer metadata.
- `reorientation90-inputs-v1.tar.gz`: the four continued models, all fixed intermediate checkpoints, raw evaluations and native context inputs.

The original 1B models/configurations come from the [original-model release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/rl-map-memory-2026-11-06); the exact 40M parent checkpoints and evaluations come from the [blind-reorientation release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/blind-reorientation-2026-11-06). Keep the source files and protocol manifests from this repository revision unchanged.

The arrays include historical absolute checkpoint paths as metadata. The adapter resolves the actual files from the current repository, so a different checkout directory is supported. Replacing an archive with a regenerated file that has different metadata changes its locked byte hash; the exact released inputs avoid that ambiguity. The NPZs are already compressed, so the two array archives use plain tar.

## Run the fixed analyses

Use CPU PyTorch, NumPy, SciPy, Matplotlib and threadpoolctl. Run from the repository root. A separate output directory preserves the published outcomes:

```sh
set -e
export OPENBLAS_NUM_THREADS=2
python scripts/lstm_actor_logit_observable.py --out /tmp/actor-score-reproduction --freeze-only
cmp results/lstm_actor_logit_observable/protocol.json /tmp/actor-score-reproduction/protocol.json
python scripts/lstm_actor_logit_observable.py --out /tmp/actor-score-reproduction --cohort original1B
python scripts/lstm_actor_logit_observable.py --out /tmp/actor-score-reproduction --cohort forty
python scripts/lstm_actor_logit_observable.py --out /tmp/actor-score-reproduction --cohort ninety
python scripts/lstm_actor_logit_observable.py --out /tmp/actor-score-reproduction --summarize-only --require-complete
```

The copied protocol must exactly match the committed protocol. `--freeze-only` writes that same validated manifest into the separate destination; it does not change the analysis design. All 152 checks are required for the final complete summary. Both forms of geometry centering and every failed criterion remain available.

The observation contains all six learned action scores after subtracting their common offset, measured at offsets 0, 5, 10 and 15 updates. This subtraction leaves action probabilities unchanged. The separate centered-geometry calculation subtracts the activity family's mean across directions; it is an analysis operation, never an intervention on the agent. Recurrent updates, full-state kicks and all candidate thresholds reuse the earlier functions unchanged.

To reproduce the native histories and complete recurrent-state analyses themselves, use the [90M input guide](../rl_blind_reorientation_entropy/REPRODUCE.md) and its earlier-release dependencies. The new action-score assay consumes their exact saved state arrays.
