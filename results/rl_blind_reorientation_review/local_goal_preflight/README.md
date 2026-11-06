# Local goal-return physical preflight

**The proposed local goals are physically reachable in this bounded sample. No training was run.** An analysis-only greedy planner reaches all 128 goals; the best of six constant-action baselines reaches 26. The latter is a restricted baseline, not a ceiling on all memoryless policies.

| Context | Goal-center distance | Delay | Total updates | Planner hits /16 | Best constant /16 | Incoming Grid /16 | Incoming PGC /16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.512m | 54 | 121 | 16 | 3 | 0 | 0 |
| 2 | 0.459m | 55 | 117 | 16 | 3 | 3 | 0 |
| 3 | 0.504m | 48 | 112 | 16 | 3 | 0 | 0 |
| 4 | 0.472m | 40 | 108 | 16 | 3 | 0 | 0 |
| 5 | 0.521m | 48 | 113 | 16 | 3 | 0 | 0 |
| 6 | 0.461m | 57 | 122 | 16 | 4 | 0 | 0 |
| 7 | 0.528m | 57 | 118 | 16 | 3 | 0 | 0 |
| 8 | 0.463m | 45 | 112 | 16 | 4 | 0 | 0 |

The planner uses actual position, body heading and motor state to choose the action with the best expected next-step goal distance. Its predictor matches 24 actual native action/state transitions in the unit test. It receives no future motor-noise draw. The planner also reaches 128/128 goals without motor noise. It is a feasible analysis controller, not an optimal oracle or the trained policy. No planner actions enter learning.

**Startup and code provenance:** initialization places the agent inside its goal disk, 0.02 m from the center, then executes one actual native forward step that yields the original +10 pickup reward. This uses the native pickup event and code; its starting position differs from an ordinary first crossing from outside the disk. The exact hit position, travel direction and velocity generate a saved code through the unchanged `CodeSource.goal` and `NativeGoal.advance` functions. That one goal snapshot stays fixed across all 16 heading variants. Setup pickup reward is recorded outside the scored return trial. We then explicitly initialize nearby positions, body headings, zero motors, zero h/c and previous-action/reward −1/0. Native respawn convention sets the initial last-travel direction to body heading.

The setup uses 15 visible forced LEFT actions, the original labelled random-turn blocks, four actual forward-probe actions, a single native code glimpse, 30–60 LEFT-delay actions and 30 free actions. All accepted examples fit 108–122 updates, inside 160-step BPTT. No forced-phase goal reward or wall collision occurred. Ten physical candidates were drawn; two were rejected by the stated preparation wall/goal-clearance rule and remain in `initialization_attempts.json`. No behavior or neural outcome selected a context.

Preparatory motor/image noise is disabled for the exact alias audit. Starting positions compensate for the real probe displacement so all headings finish at one endpoint; the largest float32 endpoint correction is 2.41e−7 m. At the first free input, position, motors, previous action/reward, ordinary current code and saved goal code are byte-identical across headings. The single predictive pulse uses actual travel direction. No new brake is applied at free onset.

Free navigation uses actual native physics and fixed private motor-noise tapes, shared across tested policies. The tapes differ across heading variants, so this run makes no mirrored-noise pairing claim. First-hit scoring stops each trial at its actual pickup position before respawn. All zero-hit trials and failed actor paths are saved.

**Information requirement:** for the first noiseless free action, the best common action has zero mean goalward progress across equally spaced headings; the heading-aware action has about 4.9 mm progress. This is an exact one-decision separation. The six constant 30-action baselines reach 3–4/16 goals per context. They do not exhaust arbitrary history-free action sequences; a stronger full-interval necessity bound is a separate prerequisite before training.

**Incoming policies:** the original 1B Grid/PGC checkpoints were not trained on these local trials. Grid gets 3/128 hits with native state and 3 after post-glimpse erasure; PGC gets 0/128 and 1 after erasure. Their per-context free action counts, entropy, movement and complete failed paths are retained. This is a transfer diagnostic, not an estimate of how newly trained local-trial agents would perform.

[All context outcomes and interpretation](summary.json) · [Every initialization attempt](initialization_attempts.json) · [Source](../../../scripts/preflight_local_goal_return.py) · [Conditional task proposal](../../../docs/research/reorientation_task_incentives_proposal.md)

Reproduce with `OPENBLAS_NUM_THREADS=2 python scripts/preflight_local_goal_return.py`. Raw NPZ setup/state/navigation arrays are ignored by Git; summary JSON is retained. This is a local feasibility check, not a final-settings freeze or a training authorization.
