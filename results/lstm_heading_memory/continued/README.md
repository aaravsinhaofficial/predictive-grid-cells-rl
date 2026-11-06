# Final blackout-trained models: heading memory within four seconds

We tested the fixed final Grid and PGC + grid checkpoints after 25.024 million additional learning frames. Their training used one second of visible input followed by randomly chosen two-to-four-second blackouts after goal discovery.

Each model supplied two fresh native contexts from its own continued policy under that actual schedule. We replay-checked the saved recurrent histories, then prepared heading families using fifteen visible updates. The primary curve test began at two seconds and ended at four seconds. The original five- and hundred-second screens were not used to judge this task.

**None of the four heading contexts passes the frozen full-state ring-candidate screen.** The PGC network was tested with its native current code and with only its current predictive slots replaced by ordinary-grid values. The goal snapshot and all other inputs stayed unchanged. Grid replacement was an exact identity control.

The files separate initial-angle retention, updated-body-heading tracking, angular bias, decoded-angle concentration, refitted information, full-state curve invariance, targeted radial recovery and actor output. A fixed readout's failure is a functional observation; geometric candidate decisions use full-state distances and dynamics independently.

See the [prespecified design](../../../docs/research/continued_heading_memory_protocol.json), [implementation hashes](implementation_protocol.json), [Grid results](gc100/summary.json) and [PGC results](pgc33/summary.json). Per-model `contexts/` folders contain actual newly generated histories. State archives keep native and ordinary-current branches separately for independent observable-state analyses.

```bash
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_continued_heading_memory.py --checkpoint results/ring_cloud_artifacts/ring-20261105-vision-grid-01/final.pt --out results/lstm_heading_memory/continued/gc100
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_continued_heading_memory.py --checkpoint results/ring_cloud_artifacts/ring-20261105-vision-pgc-01/final.pt --out results/lstm_heading_memory/continued/pgc33
```

This is one continued trained model per population, with two nested contexts. The stage-1 Grid run in `continued_stage1_shape_smoke` tested collection, replay, shape and input identity only; it supplied no mechanistic outcome metrics.
