# Fixed 90M actor readout

**Completed at all four fixed final checkpoints.** The analysis retains all 128 combinations: two populations × two entropy arms × two contexts × four cues × four times. The decomposition reconstructs the centered action scores to within 9.77 × 10⁻¹⁵; Grid cue shams are exact.

The prespecified prediction was that stronger entropy might reduce the average preference for one action, allowing existing direction-related scores to influence choices. The same frozen 40M decomposition measures actor-sensitive hidden variation, direction-dependent scores, average action margins and exact probabilities separately. The cue/state/navigation interventions determine whether any changed readout supports useful memory.

The completed analysis can be reproduced with:

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

[Frozen protocol](protocol.json) · [Completion status](status.json) · [Independent prospective review](independent_review.json) · [Completed-output audit](completed_output_audit.json) · [Exact inherited 40M method](../lstm_reorientation_actor_readout/README.md) · [Primary archive reproduction](../../docs/research/reorientation_entropy_mechanism_reproduction.md)

Protocol SHA256: `a0f870a23b6be8ce9da5a3913c12de70fb217ab78e3a8e9897a98e53cbb89ebe`. The unchanged five synthetic tests and the routing/metadata tests passed before freeze. These remain one original training-seed pair with repeated conditions, not 128 independent agents.

The higher-entropy PGC models have much smaller average action margins in these contexts and give translation substantial probability. Their remaining direction-dependent score variation is also small. The standard-entropy PGC model is context-dependent: one context retains substantial direction-dependent scores and changes its preferred action across about 11% of angles; the other remains strongly biased toward turning right. All conditions and times are in [the complete numerical summary](summary.json). These observations separate availability of a direction signal from the average action preference; the frozen cue/state/navigation interventions address its use.

“Score variation” is the vector RMS of the six action-centered logit changes across angles, not an average per-logit RMS. Its probability counterpart uses the same vector convention.
