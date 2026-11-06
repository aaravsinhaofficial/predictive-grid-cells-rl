# Fixed 90M actor readout — prospective follow-up

**Frozen; outcomes pending all four fixed final checkpoints and their primary state archives.** The analysis retains all 128 combinations: two populations × two entropy arms × two contexts × four cues × four times.

The prediction is that stronger entropy may reduce the average preference for one action, allowing existing direction-related scores to influence choices. The same frozen 40M decomposition will measure actor-sensitive hidden variation, direction-dependent scores, average action margins and exact probabilities separately. The cue/state/navigation interventions determine whether any changed readout supports useful memory.

Once every final model and primary archive is ready, run:

```sh
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_reorientation_entropy_actor_readout.py
```

Final checkpoints resolve from explicit worker names and artifact child directories in [launch routing](../../docs/research/rl_blind_reorientation_entropy_launch.json). The archive sources are:

```text
results/lstm_reorientation_entropy_mechanism/
  original/gc100/context{0,1}_states.npz
  original/pgc33/context{0,1}_states.npz
  explore/gc100/context{0,1}_states.npz
  explore/pgc33/context{0,1}_states.npz
```

The output is `results/lstm_reorientation_entropy_actor_readout/summary.json`, containing all 128 rows and source hashes. Every archive must match its genuine 90,132,480-frame checkpoint and producer metadata. There is no stage selection, metadata relabeling, new fitted representation, recurrent update, environment simulation or actor intervention.

[Frozen protocol](protocol.json) · [Pending status](status.json) · [Independent review](independent_review.json) · [Exact inherited 40M method](../lstm_reorientation_actor_readout/README.md) · [Primary archive reproduction](../../docs/research/reorientation_entropy_mechanism_reproduction.md)

Protocol SHA256: `a0f870a23b6be8ce9da5a3913c12de70fb217ab78e3a8e9897a98e53cbb89ebe`. The unchanged five synthetic tests and the routing/metadata tests passed before freeze. These remain one original training-seed pair with repeated conditions, not 128 independent agents.
