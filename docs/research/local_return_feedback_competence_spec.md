# Feedback-supported competence block: implementation proposal

**Prospective specification only, 6 November 2026. No code, banks, policy evaluations or jobs have been executed for this proposal.** The owner will decide after both current finals and the prepared action-use diagnostic. This is a bounded control-learning prerequisite, not a new ring model or an extension of the current frozen experiment.

The question is: **can the original fixed CNN/LSTM features support reliable local goal return when both visual and spatial feedback are available?** The current seen condition restores RGB but holds the endpoint spatial code. It does not answer this question. Comparing retained and reset actors tests readout initialization within this new block; changing feedback, noise and entropy together prevents attributing a difference from the previous experiment to any one of those changes.

## Fixed arms and settings

| Item | Proposed value and reason |
|---|---|
| Arms | `gc100` and `pgc33`, original seed 1, each with `retained_actor` and `reset_actor`; four arms, all completed and reported |
| Feature weights | Exact original 1B `encoder.*` and `lstm.*`, including LayerNorm parameters and all buffers; frozen throughout |
| Actor | Retained arm: original `pi.*`; reset arm: exact stored untrained `pi.*` from the existing initialized-policy asset |
| Critic | Both arms/populations use the exact same stored untrained `v.*` from that asset; no new QR initialization and no original optimizer |
| Geometry | `Stage(.58, .62, 60)`: endpoint in central ±.30 m, independent goal bearing, 0.58–0.62 m goal-center distance; goal radius remains .30 m; delay uniformly 30–60 updates |
| Task | Existing genuine saved-goal pickup, 15 visible LEFT actions, two 5–10 turn blocks, four forward probe actions, one endpoint cue, LEFT delay, then at most 30 FREE actions; first actual FREE hit or timeout terminates |
| Training | Exactly 1,024 batches × 32 fresh trials = 32,768 trials per arm; no warmup, difficulty schedule, early success stop, checkpoint selection or automatic continuation |
| Learning | Fresh RMSProp on the four head tensors only: learning rate 3e−4, alpha .99, epsilon 1e−5; gamma .99, value coefficient .5, gradient norm cap 40; unchanged complete-trial returns and fixed 100/160 loss scaling |
| Entropy | **8e−5**, the original successful-navigation configuration, fixed for all four arms. The current .01 run ended with PGC near-uniform actions; lowering this coefficient is a disclosed design choice, not a demonstrated cure |
| Rewards | Actual FREE pickup +10, clipped to 1 as before; no setup, forced-step, progress, angle or imitation reward; terminal/timeout bootstrap zero |
| Noise | Clean recurrent state, sigma=0 throughout this diagnostic block; native FREE motor noise .03/.15 retained, noiseless preparation/images unchanged. This assesses clean control sufficiency, not noise robustness |
| Randomness | New private `TrialStreams(494801)` in every arm, drawn in the same batch order; all plan/motor/action/state tapes pre-drawn before behavior. State tapes remain archived but are multiplied by zero |

Use the original and initialized-policy asset hashes already listed in the frozen training protocol. Compose incoming state dictionaries explicitly: original nonhead tensors; selected actor; common stored fresh critic. Validate every tensor before any update. Original actors differ between populations, so only the reset-actor arm has a common actor across populations. Separate RNG instances produce identical draws; they must never share a mutable generator object.

## Minimal implementation boundary

Create **`scripts/local_return_feedback.py`** and **`scripts/local_return_feedback_run.py`**, plus focused tests. Do not edit or monkeypatch frozen sources. Import unchanged `Stage`, `TrialStreams`, `schedule`, `prepare_batch`, `TrialArena`, `brake`, `make_source`, `returns` and `losses` from `local_goal_return_training.py`. A new explicitly reviewed collector copies the existing event/terminal/accounting loop; its scientific differences are restricted to FREE input selection and saving detached hidden features. A disabled-feedback test path must reproduce the original clean collector on a synthetic policy/native-physics fixture.

At each input, calculate `native_current = source(obs['pos'], obs['dir'], obs['vel'])` from the observation produced by the immediately preceding physical action. Preserve the existing held ordinary snapshot and cue selection during preparation. Then, **for active FREE rows only**, overwrite the selected code with `native_current` and make RGB visible. The first FREE input therefore uses the actual pose after the last LEFT delay action; every subsequent FREE input uses the newest pose after the previous sampled action. Never reuse the endpoint snapshot in this branch, process a free input twice, or reset h/c at feedback onset. The actual previous action, zero preterminal previous reward and fixed native saved-goal code stay unchanged.

“Native” means the original population-specific code: ordinary Grid for Grid; ordinary plus predictive features for PGC. The direction in a PGC code is the actual last travel direction, not the changing body heading during a stationary turn. This pilot permits continuous sensory correction during FREE and does not isolate the single endpoint pulse.

Set `requires_grad=False` on every nonhead parameter and use an optimizer containing **only** `pi.weight`, `pi.bias`, `v.weight`, `v.bias` (1,799 parameters). Keep encoder/LSTM in evaluation mode. Save the post-LSTM, pre-head `h` at every active input under `no_grad`; retain c internally for the next input. Recompute only `pi(h.detach())` and `v(h.detach())` for learning, using the existing losses: actor/entropy on FREE only, critic on all active inputs, padding masked. This is exact for a fixed feature network and discrete sampled-action feedback; no recurrent BPTT is needed or claimed. Verify saved versus recomputed logits/values before each update. Hash all nonhead tensors at initialization, each saved checkpoint and the final; reject any difference. Head-only gradients, optimizer membership and post-update frozen-byte identity require tests.

The runner needs a separate honest protocol/version, parent/asset/source hashes, fresh-output guard, complete-trial resume, head-only optimizer state, private/global RNG records and active/padded/setup accounting. It must not reuse old validators that require all parameters trainable, entropy .01, four stages or 4,096 updates. Save incoming, latest and fixed-final checkpoints; fixed reporting intervals may record metrics but do not become selectable endpoints.

## Evaluation and decision

Before any neural execution, seal a **new** `TrialStreams(491801)` bank using the same hard stage: 16 physical contexts × 16 evenly spaced headings = 256 trials, all tensors and hashes preserved. It is a development control bank, not a confirmation cohort. At incoming and fixed final evaluate every arm on two paired conditions, clean and with identical tapes:

1. **Full feedback:** fresh RGB and native current code throughout FREE; this alone determines the control-competence check.
2. **RGB only:** fresh RGB but the held ordinary endpoint code throughout FREE, matching the spatial-feedback limitation of the earlier seen control. This is an endpoint sensitivity, not a trained feedback-ablation arm.

Run uniform-six-action and the existing analysis planner once on the same bank as physical references, outside learning. Save every hit/failure, action, position, FREE input/current code, entropy, path/progress and physical audit. Retain invalid trials in the 256 denominator as failures and separately report their causes; no replacement. The gate also requires complete collection, zero forced rewards and passing physical/input/weight-integrity audits.

**Competence criterion: at least 192/256 full-feedback successes at the fixed final.** This prospective 75% threshold is a descriptive prerequisite above the existing 65% blind-memory target, not a p-value or a theorem. Report each arm and context, baseline-to-final change, and the paired RGB-only contrast. If an incoming arm already qualifies, that is existing competence; training is still the same fixed block for all arms. If no arm qualifies, stop this proposed sequence. If some qualify, the owner decides whether a separately frozen memory-learning study is justified; there is **no automatic blind curriculum or geometry run**. The existing 167-hit/39-erasure-loss memory gate is not changed or applied to these feedback-rich trials.

## Limits and finite budget

- A failure tests **fixed original features plus linear readout learning under this recipe**. It does not show that the fully trainable LSTM cannot learn local navigation. Forced preparation remains outside the original policy's normal history, so the recurrent state can be unsuitable even with fresh FREE inputs.
- Success shows usable feedback-supported control, not retained direction, predictive-cue necessity or a ring. The old 50% common-information bound does not apply once fresh observations are supplied. No new attractor or decoder target is introduced.
- Frozen features prevent their loss during this block, but their adequacy is the experiment. Retained versus reset actors isolate initialization only within this shared block. A global no-memory or head-reset explanation for earlier failures would exceed this comparison.

Hard training cap: **131,072 total trials across all four arms**, at most 16,908,288 active decision updates / 67,633,152 native physics frames; report actual counts, excluding setup, planning and padding. Evaluation is capped at 4,096 policy trials plus 512 physical-reference trials. Proposed infrastructure ceiling, only if separately authorized: four T4-equivalent workers, 90-minute process cutoff and 105-minute provider deletion limit each, no automatic retry or extension, and a total dollar reservation capped at $15 after the owner verifies current prices and the remaining ledger. Timeout is an incomplete pilot, not a successful fixed endpoint. A small synthetic timing check precedes freezing infrastructure; it must not inspect learned-policy performance or increase this scientific budget.

The [scientific review](local_return_next_study_review.md) contains the evidence and primary literature. This specification adds no shaping arm: terminal-zero potential shaping telescopes into a state baseline under the existing full-return estimator. No learned outcomes, new banks or implementation files were created while writing this plan.
