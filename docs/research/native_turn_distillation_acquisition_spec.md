# Native-turn recurrent policy distillation

Prepared after the complete original-memory baseline, before acquisition. The original policies produced negligible useful first-choice influence after the eight-step blind suffix, whereas their fully informed teachers produced about 1 mm of expected goalward movement. This experiment teaches that missing computation. It is a new continuation, separate from all frozen original-model findings.

**Four fixed runs:** original 1B seed 1 Grid and PGC + grid, each crossed with clean and weak recurrent-state noise. Every run begins at its original checkpoint. The original teacher, CNN, actor, critic and population code remain fixed; only the four existing 256-unit LSTM parameter tensors are trainable. The full-vision teacher follows the actual actions, rewards and observations executed by the mixed teacher/student controller. No angle labels, geometric decoder, ring loss, new recurrent units or action controller are introduced.

## Native event and learning

The square maps, six actions, native image/motor noise, pickup rewards, saved native goal code and 1350-update episodes remain. After discovery, at most one complete event is admitted per 90-update unroll and at most eight per episode. Its worst-case 57 updates must fit both the unroll and episode.

| Phase | Actual action | Student input |
|---|---|---|
| 15 refresh decisions | Mixed policy | Native RGB/current code |
|Turn A and B, each 5–10 updates | Independently drawn native left/right | Black RGB; ordinary entry code held |
|Four-step forward probe | Native forward | Same held entry code |
|Suffix of 4, 4–8 or 4–15 updates | Common native left | First input is one native/ordinary endpoint cue; remaining inputs hold ordinary endpoint code |
|Three free choices | Mixed policy, all six actions | Black RGB; held ordinary endpoint code |
|Outside event | Mixed policy | Native RGB/current code |

There are explicit brakes at blind entry, before and after the probe, and immediately before the first free choice. Brakes set all three native motor velocities to zero. The cue uses the measured probe travel direction and velocity; body heading is never substituted for last-travel direction. The subsequent turns deliberately make travel direction stale. Actual pickups interrupt events and update the native goal cache; episode endings clear both recurrent histories, previous action/reward and goal. No reward is optimized.

The first suffix input is the cue. A minimum suffix of 4 therefore includes three ordinary post-cue suffix inputs, followed by the first free-choice input. The two noise times are incoming state immediately before that cue and incoming state before the second ordinary post-cue input. No complete recurrent-state erasure is used during training.

Three fixed 60-update blocks use the suffix ranges above and teacher-action mixture probabilities 0.5, 0.25, 0. All runs use 16 environments, 90-update BPTT, 180 optimizer updates and 1,036,800 native physics frames. All event histories remain in one BPTT graph. Fresh Adam uses learning rate 1e−4 and gradient clipping 1.

The loss is 0.5 times mean teacher-forward KL on the ambiguous free choices plus 0.5 times mean KL on other policy-controlled decisions. Forced actions have zero direct loss, while later choice gradients pass through their recurrent histories. A batch with no choices uses the other-control mean alone and is explicitly counted. Choice, other-control and forced exposures, KL and rewards are recorded separately.

## One fixed noise scale

The noisy arm adds independent attempted uniform impulses with block standard deviations equal to 10% of the original post-discovery one-update state-change RMS. The full-precision calibration is retained in `results/native_turn_distillation_design/original_delta_rms.json`.

| Population | σh | σc |
|---|---:|---:|
| Grid |0.005992685313362686|0.07211062172805452|
| PGC + grid |0.0061064750282969005|0.049457829053854534|

This is an unsupervised scale calibration using existing census states. No alternative amplitude is tried. Attempted h is clipped to [-1, 1]; c is additive. Actual h impulses can consequently be biased. Attempted/realized impulses and clipping counts are retained. Independent private event tapes are indexed by slot, episode and event; clean runs draw the same attempted tapes and ignore them. Actual trajectories can diverge across arms after actions differ.

## Fixed evaluation and limits

The existing eight-branch paired-history assay remains primary: all 16 contexts × eight opposite-heading pairs in both turn-only and probe modes. Every evaluated checkpoint reconstructs its own incoming state by replaying the complete original 32-slot native prefix. No original teacher state is injected into a changed student. The exact-input teacher-JS floor and the saved best-common-action physical ceiling remain descriptive references; no favorable endpoint is selected.

Six live holdout branches retain 16 complete episodes each: visible native cue, visible native cue with every-choice erasure, blind native cue, blind ordinary cue, and the two blind cue conditions with every-choice erasure. All use the same held-code/cue and forced-turn/brake scheduling rules. Visibility changes RGB only. These are modified-task controls, not an estimate of original unmodified-task performance. Exposure clocks can diverge after different actions and hits, so their actual counts are reported.

The exact originals are evaluated once per population and shared across arms. All initial, 60-, 120- and 180-update checkpoints are retained; 180 is the sole primary trained endpoint. All four fixed finals and the originals receive the separately registered geometry assay on controlled native-turn histories, regardless of behavioral outcome. This controlled family is not claimed to be a naturally occurring on-policy cycle.

The acquisition limit is 1800 monotonic seconds per run for the fixed 180 updates. Final read-only evaluation has a separate 600-second cap; shared originals and geometry are separately timed. There is no automatic extension, retry, cloud launch or new budget. Final paths are `results/native_turn_distillation/{gc100,pgc33}/{clean,noise}/final.pt`.

The focused validation uses exact teacher/student input and noise replay, full cue-to-choice gradients, native reset/brake checks, fixed-head identities and exact serialized resume. A separately authorized test optimized discarded original-checkpoint copies in a private synthetic fixture; no study or heldout bank was used. Its record is `results/native_turn_distillation_design/native_validation.json`.

The dense-target rationale follows [Policy Distillation](https://arxiv.org/html/1511.06295); learner-visited histories are motivated by [DAgger](https://proceedings.mlr.press/v15/ross11a.html). Noise is motivated by learned-memory robustness work, including [Ságodi et al.](https://arxiv.org/html/2408.00109v2), without borrowing its absolute scale or assuming a ring will emerge. Different teachers and pretrained populations remain distinct starting points; the within-model cue and state interventions carry the causal comparisons.
