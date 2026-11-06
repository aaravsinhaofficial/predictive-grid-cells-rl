# Blind reorientation: implementation review

This is a **new reward-only training experiment in the original navigation agent**. It starts from the original 1B Grid and PGC seed-1 checkpoints. The original square, uniform goal locations, maps, rewards, CNN, LSTM and six motor actions stay in place. There is no ring target, angle label, decoder loss or change to the navigation reward.

Source review and the independent physical/input review are complete. The [protocol](rl_blind_reorientation_protocol.json) is frozen, and two bounded workers are authorized within the original compute budget. Launch and completion records are maintained in the compute ledger. This document describes the prospective experiment; training outcomes are pending.

## One event

```text
15 visible updates
    → labelled turn,5–10 updates
    → labelled turn,5–10 updates
    → forward probe,4 updates
    → one actual endpoint code pulse + first left-delay update
    → remaining left-delay updates
    →30 free blind navigation updates
```

The two turn labels are independently left or right. The delay is always a left turn. Its duration is uniform from half the stage maximum, rounded up, through the stage maximum. The four stage maxima are4,15,30 and60 updates. The cue shares the first delay update; it is not an extra step. The longest event is129 updates.

The agent receives ordinary visible input until its first actual reward. Thereafter, an event can start only when129 updates fit inside both the current160-update training segment and the remaining episode. It admits at most one event per segment and eight per90-second episode. Waiting uses ordinary native vision and policy control. Recurrent state persists between training segments. Actual rewards interrupt an event and refresh the saved goal; episode end clears it.

## What enters the network

During the blind section, RGB is zero. Current spatial input is an ordinary position-only snapshot, held fixed through the two turns and forward probe. At the actual endpoint of the fourth forward action, the snapshot position updates and the network receives one native current-code pulse. Later blind inputs hold the ordinary endpoint snapshot.

The native saved-goal snapshot remains available. Previous action is always the action actually executed, including forced actions. Previous reward remains available. No body angle, travel angle, true goal coordinate or schedule label is added to the inputs.

All three motor velocities are set to zero at blind entry, immediately before the probe, and after the probe. The pulse is computed from the actual probe observation before the last brake. It describes actual travel direction, which can differ from body heading because of motor noise or walls. Every event retains probe speed, displacement, wall contacts, travel-versus-heading error and pre/post brake telemetry. Unfavorable probes remain in the results.

## What is trained

Each stage contains489 updates ×32 environments ×160 steps ×4 simulation frames =10,014,720 learning frames. Four stages give40,058,880 learning frames per model. Before learning,200 zero-learning-rate updates initialize the new original RMSProp accumulators, adding4,096,000 exposure frames while leaving network weights unchanged.

The training unroll increases from100 to160 updates so the entire event can receive credit from its later reward. All summed losses are multiplied by100/160 to retain the original per-update gradient convention. Forced actions have zero policy-gradient and entropy weight. Their critic loss is retained, and recurrent gradients still pass through their updates.

Independent Gaussian noise with standard deviation0.02 is added to both hidden and cell state after each blind LSTM update and before the actor/critic. Every sampled noise vector is stored and replayed exactly during backpropagation. The bootstrap value receives its own draw. Noise, actions and schedule use separate private randomness. Checkpoints include environment, observation, recurrent state, goal/snapshot, event clock, counters and generator states.

## Fixed clean evaluations

The original checkpoint and every fixed stage receive64 episodes in each of12 branches:

| Code/state intervention | Blind RGB | Continuous RGB |
|---|---:|---:|
| Native endpoint pulse | Yes | Yes |
| Ordinary pulse: replace169 predictive slots | Yes | Yes |
| Pulse direction rotated+60° | Yes | Yes |
| Pulse direction rotated−60° | Yes | Yes |
| Reset hidden/cell state after processing the pulse | Yes | Yes |
| Reset hidden/cell state before every scheduled blind update | Yes | Yes |

The continuously visible controls keep the same held-code schedule and forced movements; they restore RGB only. They are controls for this new task. They are not untouched native-navigation evaluations. The Grid model's ordinary and rotated pulses are exact identities.

Matched branches share per-slot episode layouts, motor/image noise, action-sampling uniforms and the plan for each numbered event. An intervention can change later rewards, so later event start times can differ. Each branch follows its actual reward history. This difference is reported, rather than claiming identical later trajectories.

The evaluation records whole-episode scores and every free window's goal hits, distance progress, path length and duration. It also retains every completed or interrupted event, including forced-action hits, waiting/admission/cap counts and coverage checks. Distance progress is a diagnostic, not a training reward.

The reset interventions remove hidden/cell carryover. Previous actions and rewards, saved goal, the held spatial snapshot and physical history remain available. Useful memory requires a benefit specifically during free blind navigation; increased rewards during forced search are reported separately.

## Verification

Nine tests pass in `tests/test_rl_blind_reorientation.py`. They cover the exact129-update maximum, cue and reset timing, action labels, private event plans, event caps and interruptions, loss masking, native endpoint code and brakes, ordinary/rotated code slots, Grid identity controls, free-window accounting and exact noisy optimizer/runtime restoration.

A full-checkpoint smoke exercised a160-step noisy rollout, replay, gradient calculation and zero-rate optimizer step for each original CNN/LSTM. Both completed all test events without crossing the training boundary and kept checkpoint weights unchanged. Maximum rollout/replay logit differences were2.86×10⁻⁶ for Grid and2.15×10⁻⁶ for PGC. This small CPU fixture used two environments and a synthetic known-goal initialization solely to exercise the event; it is not an experimental training result.

The independent physical/input review found a float32 duration-rounding edge case. Clamping duration bins to their declared maxima fixed it; the maximum-window test now uses the nearest float64 number below1 to exercise that boundary. The same review verified the cue-before-brake ordering, constant-left action history, post-cue reset and full recurrent credit path.

## Files and planned commands

- Core task, rollout/replay and controls: `scripts/rl_blind_reorientation.py`
- Training, frame accounting and source manifest: `scripts/rl_blind_reorientation_train.py`
- Invariants: `tests/test_rl_blind_reorientation.py`
- Full-checkpoint smoke: `results/rl_blind_reorientation_review/full_checkpoint_smoke.json`

```bash
python -m pytest -q tests/test_rl_blind_reorientation.py
# Protocol is already frozen; do not overwrite it.
# Authorized commands on bounded workers:
python scripts/rl_blind_reorientation_train.py --condition gc100 --device cuda --out OUTPUT_GRID --max-seconds 6600
python scripts/rl_blind_reorientation_train.py --condition pgc33 --device cuda --out OUTPUT_PGC --max-seconds 6600
```

The approved reservation is $5.625 for two T4 workers, each with a two-hour DELETE guard and a 6600-second process deadline, inside the existing $500 cumulative budget. All four stages and final models are retained. Behavioral success would identify a newly learned predictive-memory computation; dynamics and perturbation tests then determine its geometry and whether predictive input stabilizes it.

The [behavior analysis](../../results/rl_blind_reorientation/behavior_analysis_protocol.json) is also frozen before training outcomes. Its primary quantity is free-window goal hits per original evaluation episode, retaining episodes with no event. Descriptive uncertainty resamples paired environment slots, keeping both episodes together. Individual windows are not counted as independent training replications.
