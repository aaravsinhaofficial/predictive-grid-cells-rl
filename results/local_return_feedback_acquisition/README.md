# Can the native agent learn the return movement?

The previous blind-return recipe struggled to learn goalward movement. This control supplies **fresh vision and spatial codes throughout**, then trains the unchanged native CNN/LSTM using the real goal reward.

Four arms compare Grid and PGC, each with its original or reset action readout. Every arm receives exactly 1,024 updates and the same 256-trial baseline/final evaluation bank. All network weights can learn. The fixed fast-steering criterion is **192/256 final successes** with complete physical and input checks. Neural outcomes are pending.

The first cloud attempt stopped during input validation, before a model or learning update ran. It exposed a cross-platform rounding mismatch in an unnecessary bank-regeneration check. The exact saved test inputs passed their checksum checks. All affected workers are deleted. A separate reviewed loader now uses those same saved values directly; the first replacement worker has passed its Linux input check and begun training. [Failure evidence and repair](../local_return_feedback_design/PORTABILITY.md) · [Portable reproduction guide](../local_return_feedback_design/PORTABLE_REPRODUCE.md).

The physical references are complete:

| Controller | Goals reached |
|---|---:|
| Random actions | 0/256 |
| Planner using true position and heading | 256/256 |

All 512 trials passed the independent audit. This establishes physical reachability on the sealed bank; the planner has privileged information and supplies no training targets. [Reference results](references/summary.json) · [Independent audit](../local_return_feedback_design/reference_audit.json).

This experiment tests acquisition of a two-second return movement. A successful endpoint can support a separate memory experiment, followed by state perturbation and predictive-cue repair tests. No circular representation is imposed during training.

[Frozen method](../../docs/research/local_return_feedback_training_protocol.json) · [Simple specification](../../docs/research/local_return_feedback_acquisition_spec_v2.md) · [Implementation review](../../docs/research/local_return_feedback_final_implementation_review.json) · [Larger mechanistic study](../../mechanisms/learned-ring-in-navigation.md).
