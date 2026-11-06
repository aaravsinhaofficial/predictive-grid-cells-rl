# Test the mechanism inside the reorientation-trained agent

**Prospective analysis; final trained-model results are pending.** The source and criteria were frozen before the new training evaluations. Original-checkpoint smoke runs test physical replay and array shapes only.

The analysis follows the actual task:

1. Collect a real visible/turn/probe event from each final policy.
2. Replay its actual action sequence across directions, choosing starts so every probe reaches the same endpoint. Reject paths that could hit a wall or goal.
3. Present one actual predictive cue, an ordinary cue, or a rotated cue.
4. Hold the same ordinary input during the left-turn delay. Measure whether a circular activity family persists and repairs disturbances.
5. Separately measure whether cue repair changes the original actor's free navigation.

A rotating ring is allowed. The phase readout is compared both with the initial travel angle and with the body heading expected after the real turn dynamics. The full-state geometry and perturbation tests determine candidates independently of that readout.

Two comparisons remain separate: **intrinsic recurrent stability** and **acute phase repair by a predictive pulse**. Native contexts differ across Grid and PGC models, so their initial between-model comparison is descriptive. Exact common-input causal contrasts apply within each model/context. A separate crossed-context intervention is specified in the review addendum.

Both original-model shape smokes passed: two native contexts and 96 physical angles per model, with no rotated wall/goal hits, endpoint discrepancies below 3.1e−7 m, and travel/body alignment errors below 4e−6 radians. The native-support observer records endpoint and free-start body heading, travel direction and egocentric goal bearing separately from the controlled full-circle family. No candidate scores were computed in these smokes.

The [simple original-transfer coverage figure](original_angular_coverage/README.md) retains all 1,141 recorded observations. Every direction/goal-bearing distribution occupies all twelve 30° bins; its largest empty gaps are 5.3–12.5°. These are original-checkpoint transfer histories, separate from the pending final trained-model coverage. [Exact reproduction commands and action/state indexing](../../docs/research/reorientation_mechanism_reproduction.md) are available for the fixed final analyses.

[Prospective design](../../docs/research/reorientation_mechanism_protocol.json) · [Scope and time-indexing clarification](../../docs/research/reorientation_mechanism_review_addendum.md) · [Frozen implementation hashes](implementation_protocol.json) · [Analysis source](../../scripts/lstm_reorientation_mechanism.py)
