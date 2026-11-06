# Testing a learned ring in the navigation LSTM

**Active experiment, 6 November 2026 UTC.** The question is whether predictive grid inputs help the navigation LSTM maintain and repair a circular memory, and whether that repair helps it reach the goal.

The target is inside the trained 256-unit LSTM. Its full recurrent state contains 256 hidden values and 256 cell-memory values. The earlier explicitly constructed ring is a separate experiment.

## What would count as the mechanism?

1. **A learned circular memory.** Different remembered directions occupy a closed family of recurrent states. The family survives when the directional cue is removed.
2. **Recovery toward the circle.** A push inward or outward is corrected faster than an equally sized change along the circle. A slowly shrinking plane can retain circular-looking data without this property.
3. **Predictive correction.** Replacing the 169 current predictive inputs with ordinary grid values changes recovery or directional drift. A brief correct predictive cue repairs the memory; a deliberately rotated cue produces the corresponding directional error.
4. **Useful correction.** Those state changes alter the original actor's choices and navigation in the predicted direction.

Phase correction and attraction toward the circle are measured separately. Predictive input could improve either one, both, or neither. The saved goal code is held unchanged in the current-input intervention.

## Why these methods?

[Low, Giocomo and Williams (2023)](https://elifesciences.org/articles/86943) connect learned navigation representations to fixed points and their local modes. [Ságodi and colleagues (2024)](https://arxiv.org/html/2408.00109v2) show how trained LSTMs can use slow rings rather than an exact continuum of stationary points. [Driscoll and colleagues (2024)](https://www.nature.com/articles/s41593-024-01668-6) motivate testing a memory computation within its particular input and task context. The [literature audit](endogenous_ring_literature.md) records what transfers to this experiment.

## Completed original-agent tests

- **Goal-history families:** corrected perturbation recovery in all 512 recurrent coordinates, with the phase component separated from departure from the evolving family. Both raw and block-scaled distances are retained.
- **Heading-history families:** actual rendered heading histories followed by common-input turns in darkness. Frozen and refitted direction readouts distinguish a stable representation from information that an analyst can recover with a new readout.
- **Curve dynamics:** dense closed-curve geometry, phase coverage, targeted radial disturbances, and longer common-input trajectories. Replacing current PGC input with ordinary input checks whether a fixed directional cue was merely pinning the state.
- **Fixed points and numerical conditioning:** direct residual optimization, saturation controls, and an analytic elimination of equilibrium cell state. Failed optimizations and subsequently recovered solutions are both retained. Native-precision and behavioral-reachability checks accompany numerical equilibria.

The completed tests so far favor slow cell-memory traces over a recovered ring in these original contexts. See [goal-state and equilibrium results](../../results/lstm_endogenous_ring/README.md) and [heading-curve results](../../results/lstm_heading_memory/curve_flow/README.md). These are pilot analyses of seeds 1–2, not a population-wide claim about every possible state or context.

## Reward-only continuation

The [fixed 25.024-million-frame pilot](rl_visual_memory_pilot_protocol.json) starts from the original seed-1 Grid and PGC checkpoints. It retains the square arena, uniformly placed goals, maps, rewards, six actions, actor, CNN and LSTM. The only training-task change is occasional black RGB after goal discovery. Visible intervals last 1 second; blackouts gradually grow to 2–4 seconds. All spatial inputs remain native during training. No ring architecture, angle target or ring loss is supplied.

Every fixed checkpoint is tested with vision, blackouts, memory resets, and current-PGC replacement. Reset controls with vision distinguish general controller disruption from an additional cost of losing history in darkness. The original vision score is reported alongside each new score so that a ratio cannot hide overall deterioration.

The [continued-model assay](continued_heading_memory_protocol.json) collects new histories from each continued policy. Its primary recovery test ends within the trained 4-second blackout; the older 5–100-second original-agent tests are not substituted for this endpoint.

## Candidate variable and interpretation

The supplied predictive displacement follows **last translation direction**. It can differ from body heading during backward or sideways movement and stays unchanged during a turn in place. These variables are kept separate. The completed physically coherent travel-history assay tests this variable with identical final positions and ordinary current inputs after cue removal. Four original pilot models supplied one context each. The held-out replication used seeds 3–8: twelve models, two fresh native contexts each. All 24 held-out contexts fail the frozen full-state ring criteria; the separate future-hidden-activity control also yields no candidates. Direction remains readable at four seconds, while radial disturbances persist about as strongly as phase changes. [Held-out travel results](../../results/lstm_travel_memory/heldout_original/README.md) and [observable-history control](../../results/lstm_endogenous_ring/observable_history/travel_heldout_original/README.md).

A separately frozen action-context sensitivity repeats the original pilot test with forward, backward and both sideways approaches. Within each model, all four actions use the same world-space paths and endpoints; there are sixteen model/action combinations in total. None passes the unchanged full-state criteria. The forward repeat reproduces the original saved states and inputs exactly. [Action-context results](../../results/lstm_travel_memory/action_context_original/README.md).

A correct one-update current-PGC pulse after a ±30° full-state substitution produces little lasting repair: the largest absolute change in the discrepancy after three further seconds is 0.601° across twelve held-out PGC contexts and both metrics. Rotated cues, matched-branch references, off-curve displacement and actor effects are retained. [Held-out cue-pulse results](../../results/lstm_heading_memory/travel_cue_pulse/heldout_original/README.md).

Primary criteria are frozen before their corresponding outcomes. Any secondary analysis keeps the primary result visible. A ring may coexist with other memory processes, so possible reductions of the full state must be validated against future neural activity and actor outputs rather than chosen because a two-dimensional projection looks circular.

## Completed pilot and current continuations

The fixed 25.024-million-frame pilot is complete for both models. Blackout scores improve from 23.59 to 43.59 for Grid and from 14.53 to 47.81 for PGC. Continuous-vision scores change from 83.28 to 61.25 and from 85.00 to 83.75, respectively. These are one preselected training-seed pair with 64 evaluation episodes per condition. Neither model passes the frozen heading/travel screens or the future-hidden-activity control.

Resetting both recurrent variables before every black update gives scores of 44.22 for Grid and 45.78 for PGC. Most blackout improvement therefore survives removal of recurrent carryover during darkness, with the previous action/reward, current grid inputs, external saved goal and physical environment still available. [Every-update reset results](../../results/lstm_endogenous_ring/every_blind_update_reset/README.md).

Two further arms each add a fixed 50.048 million frames to those exact final checkpoints and optimizer states: [noiseless adaptation](rl_visual_memory_extension_protocol.json) and [activity-noise training](rl_visual_memory_noise_protocol.json). The latter adds independent Gaussian noise with standard deviation 0.02 to each hidden and cell coordinate during black updates; exact sampled noise is reused during gradient replay. This is the same absolute disturbance in both state blocks, not the same relative disturbance. No ring or angle loss is added. Both arms are running, with the final 75.072-million-frame analysis wrapper already frozen. Geometry uses clean histories and deterministic state evolution in both arms. A separate [matched-noise evaluation](visual_memory_matched_noise_eval_protocol.json) is frozen for all four fixed final checkpoints on one CUDA backend. It verifies identical unmasked noise draws by hash; each model’s actual blackout mask follows its own reward history. All final continuation conclusions remain pending those endpoints.

## Resources and reproduction

The cumulative original compute limit remains $500. The first two-worker visual-memory pilot has completed. Four bounded T4 workers run the paired noiseless and activity-noise continuations; the [budget ledger](ring_cloud_budget.json) records prior use and reservations. Provider-side deletion deadlines and boot-disk deletion protect against unattended costs. Source hashes, checkpoint hashes, all fixed stages and negative outcomes are retained. Raw state arrays remain local and regenerable; figures, protocols, summaries and analysis code are versioned.
