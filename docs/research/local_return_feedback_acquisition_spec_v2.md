# Feedback-supported fast local steering: prospective acquisition preflight

**Revision 2, 6 November 2026. Specification only; no implementation, bank, neural evaluation or job was run.** This supersedes, but preserves, the [frozen-feature proposal](local_return_feedback_competence_spec.md). Independent review found that a fixed CNN/LSTM after an unusual blind history would test linear-readout sufficiency, leaving task acquisition unresolved. The owner requested the following bounded, fully trainable control. A decision still follows both fixed current finals and the prepared action-use diagnostic.

The question is **whether these native networks can learn fast local goal return with normal sensory feedback before being asked to retain direction through darkness**. The proposed changes concern feedback, preparation duration, state noise, entropy and trainable parameters together. This is an acquisition preflight, not a retroactive causal explanation of the previous failure, a memory test or a substituted ring model.

## Exact proposed block

| Setting | Fixed proposal |
|---|---|
| Models | Four arms: original seed-1 Grid and PGC, each with retained or reset actor |
| Initialization | Original 1B encoder/LSTM in all arms; retained `pi.*` or the exact archived untrained reset `pi.*`; the same archived untrained `v.*` in every arm; fresh optimizer |
| Trainability | **All original network parameters trainable**, including CNN/LayerNorm/LSTM and both heads; unchanged architecture and inputs |
| Physical task | Native square arena, maps, six actions, physics, genuine saved pickup goal, central endpoints ±.30 m, goal-center distance .58–.62 m, goal radius .30 m |
| Preparation | Existing 15 LEFT updates, two independent 5–10 left/right blocks, four forward updates, then LEFT delay uniformly **2–4** updates (`Stage(.58,.62,4)`) |
| Feedback | Fresh native RGB **and** current population code on every active input throughout preparation and FREE; fixed genuine saved-goal code, actual previous action, no added angle/goal-coordinate input |
| Response | At most 30 FREE actions = 2 seconds; first real goal hit or timeout is terminal; no respawn exposure or later payoff |
| Training size | Exactly 1,024 updates × 32 fresh trials = **32,768 trials per arm**; all four fixed endpoints retained |
| Optimizer | Fresh RMSProp, learning rate 3e−4, alpha .99, epsilon 1e−5; gamma .99, value coefficient .5, gradient norm cap 40 |
| Entropy | **8e−5**, the original successful-navigation setting, matched across arms; no temperature, action holds or action supervision |
| Rewards/credit | Native FREE +10 clipped to 1; complete-trial Monte Carlo returns, zero terminal/timeout bootstrap; actor/entropy on FREE, critic on all active inputs; unchanged fixed 100/160 loss scaling |
| Noise | **h/c noise zero**; native FREE motor noise .03/.15, noiseless preparation and rendered images as before |
| Private streams | Separate `TrialStreams(494801)` instances, identical named streams and batch order in every arm; complete tapes drawn before outcomes; state-noise tape archived but unused |

The original entropy coefficient is chosen prospectively because the recent .01 PGC endpoint was nearly uniform despite sparse reward. It is not an established solution. Keeping the original reward/return estimator avoids adding a shaping hypothesis to this first control.

Compose incoming state dictionaries from the already hashed original checkpoints and exact initialized-policy assets; do not rerun QR initialization. All nonhead bytes initially match their original population checkpoint. Fresh critics match across all arms, while retained actors differ between populations. The actor comparison changes both inherited action mapping **and initial exploration**. Common critic bytes, random tapes and optimizer settings do not isolate readout retention alone. Once learning begins, action histories, visited states and learned recurrent weights may diverge legitimately.

## Minimal separate implementation

Create `scripts/local_return_feedback.py`, `scripts/local_return_feedback_run.py` and focused tests. Preserve every frozen existing file; no monkeypatches. Import unchanged `Stage`, `TrialStreams`, `schedule`, `prepare_batch`, `TrialArena`, `brake`, `make_source`, `returns`, `replay` and `losses` from the existing task core. A separate reviewed collector retains its physical loop, goal setup, braking, endpoint rounding guard, sampling, rewards, terminal masks and bookkeeping. Its observation branch changes explicitly:

- On **every active step**, pass `obs['rgb']` and `source(obs['pos'], obs['dir'], obs['vel'])` to the policy. The observation must come from the latest preceding physical action. Do not use the held ordinary code or hide RGB during preparation.
- At the first FREE input, the pose follows the final LEFT delay action. Process this input once, then sample its action; subsequent inputs follow their actual sampled actions. Keep h/c across the preparation/FREE boundary. All trials still begin with zero h/c, previous action −1 and previous reward zero.
- The former `glimpse` index remains useful for the endpoint brake/rounding event, but it is **not a unique cue pulse** in this observed task: native code is available before and after it. PGC current code uses actual travel direction, which can differ from body heading during turns.
- Save the actual RGB, current/goal code, previous action/reward, state noise (zero), actions, active/terminal masks and selected logits. Reuse the full recurrent `replay` and losses to backpropagate through all active history. No feature detachment or head-only optimizer is permitted in revision 2.

The runner needs its own honest version/protocol and validators: original/initialized asset hashes, new source hashes, changed config, four arm identities, fresh-output guard, exact complete-trial resume, optimizer/private/global RNG records, active/padding/setup counters. Do not reuse an old validator asserting four stages, entropy .01 or 4,096 updates. Save incoming, latest and the fixed 1,024-update final. Record metrics at fixed intervals without selecting intermediate checkpoints. Evaluation must restore training RNG/state exactly and never consume its streams.

Required implementation checks before any neural outcome execution: (1) old clean collector equivalence when the new observation branch is disabled on a synthetic fixture; (2) every active input's RGB/code matches its contemporaneous native observation, including last-delay/first-FREE timing; (3) no feedback pulse or unintended h/c reset; (4) frozen original sources/initial tensor provenance and matched heads; (5) all intended parameters enter the optimizer and receive the appropriate replay graph, with masked forced actor/entropy losses; (6) stored rollout/replay logits match within the existing 3e−4 float32 tolerance; (7) first-hit stopping, zero bootstrap, input/plan preservation, paired streams and exact boundary resume. No trained policy is used to tune these tests.

## Sealed evaluation and endpoint criterion

Before policy execution, seal `TrialStreams(491801)` at `Stage(.58,.62,4)`: **16 new physical contexts × 16 evenly spaced headings = 256 trials**, with full tensor/source/file hashes. This is a new development bank, not an independent model-seed cohort or a confirmation bank. Perform **zero-update and fixed-final** evaluations on this exact fully observed bank for all four arms, clean and with matched tapes. This directly tests what each incoming actor can do under the proposed inputs; prior blind or RGB-only results are not substituted.

Run the existing uniform-six-action and physical analysis-planner references once on the same bank, outside learning. Retain every failure and invalid trial; do not replace a context. Save actual goals, FREE poses/motors/actions/positions/current codes, first-action probabilities, entropy, whole-response path/progress and all setup/forced/FREE counts. Invalid trials count as failures within 256 and are reported separately.

The fixed descriptive criterion is **at least 192/256 successes (75%) at the final, with complete collection and passing physical/input/provenance audits, including zero forced rewards**. It measures **fast local steering within this exact two-second task**. A reasonable turn-then-forward strategy for a rear goal can take longer; backward/sideways native actions make the geometry physically reachable but do not guarantee the network learns that strategy. Do not search over response durations or substitute a slower navigation test.

Report every arm, each physical context, incoming-to-final change and actual action/movement distributions. If an incoming arm already exceeds the criterion, label it existing ability; it still receives the same fixed block. If no final qualifies, stop this proposed sequence. If some qualify, the owner may review a separate memory curriculum; this document authorizes neither that curriculum nor any geometry analysis. A qualifying Grid and failing PGC remains a valid outcome. Existing blind-memory gates stay untouched and are not applied to this feedback-rich bank. The old 50% common-information certificate does not apply when fresh observations are available.

## Finite cost and interpretation

Training cap: **131,072 trials across four arms**. Maximum 73 active decisions per trial gives at most **9,568,256 active decisions / 38,273,024 native physics frames** across the pilot, excluding setup, planning and padding. Evaluation cap: 2,048 policy trials (four arms × two endpoints × 256) plus 512 physical-reference trials. Report actual counts; 160-slot padding is never experienced learning data.

Full feedback increases CNN work, so the earlier head-only 90-minute estimate is not retained. Proposed maximum, subject to owner infrastructure review: four T4-equivalent workers, **3.5-hour process cutoff / 4-hour provider deletion limit each**, total reservation **at most $25** after checking current prices and the cumulative ledger. No automatic retries, duration extensions or extra arms. A synthetic shape/timing check may inform whether this cap is feasible; an infeasible plan is reviewed before launch, not silently enlarged. A timeout is incomplete, never a substitute fixed endpoint.

Failure would show that this bounded recipe did not acquire fast steering, leaving optimization, representation learning and action strategy unresolved. Success would establish an attainable control prerequisite, while directional retention, acute PGC repair and ring attraction remain separate later tests. The short-delay, fully observed trial retains native maps, actions, real goal reward and LSTM; it adds no angle labels, readout targets, attractor loss, planner in training or new controller.
