# Fixed 90M mechanism analyses — prospective protocol

**The primary and causal assays are frozen before these four final checkpoint outcomes. Results are pending.** All original/explore-entropy × Grid/PGC endpoints and both selected native contexts per model will be retained.

The new adapter validates the genuine 90,132,480-frame continuation metadata and calls the existing context collector, full-state geometry and cue/navigation functions unchanged. No angle target, new geometry threshold or checkpoint selection is introduced.

| Input | Location |
|---|---|
| Grid 40M parent | `results/ring_cloud_artifacts/ring-20261105-reorient-grid-01/final.pt` |
| PGC 40M parent | `results/ring_cloud_artifacts/ring-20261105-reorient-pgc-02/final.pt` |
| 90M final checkpoints | `results/ring_cloud_artifacts/{explicit worker name}/{artifact_subdir}/final.pt`, resolved from [launch routing](../../docs/research/rl_blind_reorientation_entropy_launch.json) |
| Analysis destinations | `results/lstm_reorientation_entropy_mechanism/{original,explore}/{gc100,pgc33}/` |

Run the three commands below for each population (`gc100`, `pgc33`) and each entropy arm (`original`, `explore`):

```sh
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --condition gc100 --arm original --mode primary
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --condition gc100 --arm original --mode causal --context 0
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_mechanism.py --condition gc100 --arm original --mode causal --context 1
```

The first command produces `context0/1_states.npz`, primary geometry and all native coverage. The next two produce complete causal results and matched navigation tapes in `causal/context0/1/`. At time 60 the first free input has already entered the LSTM, so the actor uses `pi(h60)` directly before 29 further input updates for 30 actions.

[Frozen protocol](protocol.json) · [Independent review](review.json) · [Exact parent hashes, source manifests, folder layout and timing](../../docs/research/reorientation_entropy_mechanism_reproduction.md) · [Completed 40M parent results](../lstm_reorientation_mechanism/final/README.md)

Protocol SHA256: `18728e4b6b3d6f324f180dfced93ad980bcb462916ec971ef58cc5d030ec6562`. The planned conditional rescue analysis in the same fixed future-hidden observation map remains separate from the unchanged primary full-state outcomes.
