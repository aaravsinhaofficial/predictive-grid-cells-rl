# Does predictive input stabilize a learned ring in the navigation LSTM?

**The experiments currently point to persistent direction traces, with little recovery toward a ring.** We tested the original agents first, including six held-out seed pairs. Reward-only continuations now test whether the same navigation LSTM learns the proposed mechanism under visual blackouts and disturbances to its activity.

This study is active. The longer training experiments below are running; their results will be added at their fixed endpoints. The earlier explicitly constructed goal-memory ring is a [separate result](goal-memory-in-the-navigation-task.md).

## What we tested

Imagine the network's remembered direction as a point on a circle. A ring mechanism should do two things: **preserve differences around the circle, and correct mistakes away from it**. We measured both in all 512 hidden-plus-cell coordinates. PCA was not used to define the circle.

The predictive spatial displacement follows the last **movement direction**. Body heading is a different variable when the agent walks backward, moves sideways, or turns without translating. We tested heading histories and physically coherent movement histories separately.

For the movement test, the agent approaches one location from many directions using the actual renderer, grid maps and movement equations. We then explicitly stop its movement and give every recurrent state the same input: black vision, ordinary current grid code, the unchanged saved goal and a turn action. This removes the current predictive direction cue. The approach and turn are controlled diagnostic actions, initialized from a native policy history.

## Original agents: information remains, but radial mistakes remain too

![Held-out movement-memory results](../results/lstm_travel_memory/heldout_original/heldout_travel_memory.png)

Each dot is a trained model, averaging two contexts. Thin lines show those contexts' range.

After developing the method on seeds 1–2, we tested seeds 3–8 without changing the criteria:

- **Direction information persists.** At four seconds, a newly fitted analysis readout recovers direction with 0.75–2.31° error. The readout fixed before darkness has much larger errors: the expression of the trace changes over time.
- **The circle does not correct radial errors selectively.** Radial disturbances survive about as strongly as differences along the circle: the ratio is 0.89–1.11 in raw state units across the 24 held-out contexts. Both raw and hidden/cell-scaled metrics are retained.
- **The result survives an additional control for unrelated activity drift.** Using all hidden units' future activity, rather than large cell-memory amplitudes alone, produces no ring candidates. PGC has stronger preferential radial recovery in only three of the six held-out pairs.

The pattern suggests a slowly changing memory trace with several persistent directions, rather than the tested one-dimensional attracting ring. [Full held-out results](../results/lstm_travel_memory/heldout_original/README.md) · [Control using future activity](../results/lstm_endogenous_ring/observable_history/travel_heldout_original/README.md).

## Can one predictive cue repair an incorrect direction?

![Correct and rotated brief cues](../results/lstm_heading_memory/travel_cue_pulse/heldout_original/cue_pulse.png)

We replaced the full recurrent state with the state elicited by movement 30° away. We then restored the 169 current predictive values for one update—1/15 second—or supplied a deliberately rotated predictive cue. Ordinary current code resumed immediately afterward. Every saved-goal value and every other input slot stayed unchanged.

**The brief cue produces a transient response, while the substituted direction largely remains.** Three seconds later, the largest change in the 30° discrepancy is 0.601° across the twelve held-out PGC contexts and both distance metrics. Scaling hidden and cell activity changes the size of the immediate apparent correction, which also moves states off the reference curve. The two panels retain that sensitivity.

Grid-model cue branches are exact identity controls. The same test in the first blackout-trained pair also produces no lasting correction of this state substitution. [All pulse conditions, geometry checks and actor effects](../results/lstm_heading_memory/travel_cue_pulse/heldout_original/README.md).

## Training the same agent to navigate through blackouts

The first continuation changes only visual availability. After goal discovery, one second of vision alternates with blackouts that gradually reach two to four seconds. The square arena, uniformly placed goals, rewards, six actions, 512 spatial cells, CNN, LSTM and learned actor stay the same. Training supplies no angle target or ring loss.

After 25.024 million additional learning frames per model:

| Evaluation score | Grid before | Grid after | PGC + grid before | PGC + grid after |
|---|---:|---:|---:|---:|
| With blackouts | 23.59 | 43.59 | 14.53 | 47.81 |
| With continuous vision | 83.28 | 61.25 | 85.00 | 83.75 |

Both models improve in darkness. The PGC model largely preserves its normal-vision score; Grid loses performance there. These are **one fixed training-seed pair**, evaluated on 64 episodes per condition, so they describe this pilot rather than a population effect.

Neither model acquires a ring candidate in the frozen heading or movement-history assays, including the control using future hidden activity. Resetting state at blackout onset also produces no extra cost compared with the corresponding visible-input reset. Resetting hidden and cell state before **every** black update leaves most of the improvement intact: Grid scores 44.22 versus 43.59 with normal carryover; PGC scores 45.78 versus 47.81. Previous action, previous reward, current code, saved goal and the physical environment remain available. This points toward adaptation using those available inputs and externally carried state. [Every-update reset results](../results/lstm_endogenous_ring/every_blind_update_reset/README.md).

[All fixed learning stages and controls](../results/ring_visual_memory/curriculum_summary.json) · [Training protocol](../docs/research/rl_visual_memory_pilot_protocol.json).

## The next controlled comparison

Two matched continuations start from those exact final checkpoints and optimizer states:

1. **More adaptation:** another fixed 50.048 million frames with the final blackout schedule.
2. **Resistance to disturbances:** the same training, adding small independent noise to hidden and cell activity during blackouts. The sampled disturbances are replayed exactly during gradient computation.

The noise experiment follows the robustness principle used by [Ságodi and colleagues](https://arxiv.org/html/2408.00109v2), whose trained networks developed slow attracting memory manifolds. It prescribes no circular representation. Both arms receive the same reward-only task and the same clean navigation and deterministic geometry tests at their fixed final checkpoints. A separately frozen [matched-noise evaluation](../docs/research/visual_memory_matched_noise_eval_protocol.json) will test all four final models on the same CUDA backend and verify identical underlying noise draws by hash. The training runs themselves produce clean evaluations in both arms and additional noisy evaluations in the noise-trained arm.

An additional action-context check covers forward, backward and both sideways approaches in the four original pilot models. **None of these 16 cases passes the same ring criteria.** Radial errors again persist about as strongly as angular differences, despite direction remaining readable. This covers translation actions frequently used by the original policies. [Action-context figure and methods](../results/lstm_travel_memory/action_context_original/README.md).

## Giving remembered direction a necessary job

The reset result exposed a shortcut: the blackout-trained agents could often keep moving using available spatial inputs, previous actions and physical momentum. The next experiment therefore adds a controlled blind turn, a short actual forward movement and one brief predictive direction cue. It then withdraws the cue, turns the agent further, and gives control back for blind navigation. The same LSTM learns from the original goal reward.

![Why history is useful](../results/lstm_blind_turn_necessity/necessity_ceiling.png)

In the design check, two histories end with **identical current inputs and motor state, but opposite body headings**. The same action distribution produces at most 0.077 mm of average goalward progress on the next update; choosing separately for the two remembered headings permits 5.439 mm. These are simulator-based bounds, not trained-agent results. The original agents do not exceed the common-action bound in this check. [Paired-history audit](../results/lstm_blind_turn_necessity/README.md).

The new training pair has a fixed 40.059-million-frame budget per model, with no ring or angle target. The primary outcome counts goals during freely chosen blind movement, excluding forced-movement rewards. Removing the predictive pulse, rotating it, and resetting the LSTM immediately after it test whether the agent stores and uses the cue. Geometry and state-perturbation tests then identify the computation that carries that memory. [Frozen training method](../docs/research/rl_blind_reorientation_implementation.md) · [Frozen behavior analysis](../results/rl_blind_reorientation/behavior_analysis_protocol.json). Training outcomes are pending.

The [literature audit](../docs/research/endogenous_ring_literature.md), [research log](../docs/research/endogenous_ring_run_log.md), [fixed-point and numerical-precision audit](../results/lstm_endogenous_ring/fixed_points_and_saturation.md), and [compute ledger](../docs/research/ring_cloud_budget.json) retain the full method and outcome history. All cloud work draws from the original cumulative $500 authorization.
