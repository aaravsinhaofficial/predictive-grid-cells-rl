# Task-linked goal-memory study: experiment inventory

The [simple explanation](../../mechanisms/goal-memory-in-the-navigation-task.md) leads with the original policies and the goal-memory model. This inventory retains all tests, including negative results. The older supervised compass experiments are separate and are not evidence that the original navigation LSTMs contain a ring.

| Experiment | Why run it? | What was changed? | Evidence unit |
|---|---|---|---|
| Original-policy steering map | Does the saved predictive map change where the real actor aims? | Only predictive goal-input channels; controlled positions and headings | Eight original PGC policies; seeds 1–2 used for method development, 3–8 held out |
| Goal-message state family | Does goal information persist after its input disappears? | Valid goal-message family, then withdrawal during matched physical replay | Two original seed pairs, four real contexts each; descriptive pilot |
| Natural false-goal state pulse | Which internal state carries a lasting goal influence on actions? | State elicited by a shifted goal message, then correct input restored; h-only/c-only controls | All eight original Grid/PGC pairs |
| Linear-decoder state pulse | Can a linear angle readout identify a functional memory direction? | Small c-only edit chosen by a fitted readout | One PGC pilot; failed nonlinear/behavioral validation, retained as negative |
| Goal-circle cue geometry | Does the actual predictive point better locate a hidden goal on the circle? | No policy change; compare estimates from the same real first pickups | Original checkpoint-pair pilot; geometry, not a memory intervention |
| Designed goal-location ring | Can a simple recurrent memory connect these rates to real RL navigation? | Replace saved-goal storage with a two-state ring; same frozen Grid actor across cue conditions | Seed 1 feasibility; separately frozen held-out actor cohort |
| Ordinary-grid history control | Is the arrival cue also available from ordinary observations over time? | Decode the recent ordinary positions and reconstruct the ahead point; no true direction supplied | 64 first pickups from one original Grid policy; exact input-equivalence check |
| Goal-flash RL transfer | Do unconstrained LSTMs learn useful memory when the external cue disappears? | Circular goal prior and four-update goal flash; 5,004,800 extra learning frames | One original Grid/PGC pair; no claim of a population-wide advantage |

The ring model changes the goal-location prior and the saved-goal memory interface; it does not change the square movement geometry. The fine-tuning experiment additionally removes the external cue after four policy updates. Neither task gives the policy the true hidden goal coordinate.

The ring interface decodes a point, projects it onto the known circle of possible goals, and regenerates a coherent ordinary goal code. This is different from shifting the entire saved goal snapshot ahead in the original unrestricted-goal task. That earlier intervention hurt the frozen Grid policies, as retained in the [original mechanism report](../../mechanisms/README.md). The new model therefore demonstrates a possible useful memory interface under its explicit goal prior; it does not replace the original input-format findings.

## Sources and complete results

- [Steering-map protocol](original_rl_goal_policy_map_protocol.md), [source identities](original_rl_goal_policy_map_sources.json), and [results](../../results/rl_map_mechanism/policy_map_confirmation/README.md).
- [Goal-information state-family methods and results](../../results/original_goal_bearing_manifold/pilot/README.md).
- [Natural state-pulse protocol](original_rl_goal_state_pulse_protocol.md) and [results](original_rl_goal_state_pulse_results.md).
- [Goal-flash RL protocol](rl_goal_memory_pilot_protocol.json), [methods](rl_goal_memory_pilot.md), [results and figure](../../results/ring_rl_goal_memory/README.md), and [cloud audit](rl_goal_memory_cloud_audit.md).
- [Designed ring results and figure](../../results/original_rl_goal_ring/README.md), [seed-1 implementation](../../scripts/original_rl_goal_ring.py), and [held-out cohort implementation](../../scripts/original_rl_goal_ring_cohort.py). Their output directories contain frozen protocols and numerical results.
- [Ordinary-grid history equivalence control](../../results/original_rl_goal_ring/history_control/README.md), including the compact observed-rate archive.

Original checkpoint files remain unchanged. Raw state arrays are omitted from Git and can be regenerated. The release archive includes all 16 original 1B-frame LSTMs used here and both completed fine-tuned pilot models; the designed ring adds no trained weights.

The [checkpoint archive manifest](rl_map_checkpoint_archive.json) supplies per-file and archive SHA-256 hashes. Only extract the released archive after verifying its hash. It restores the existing relative model paths at the repository root.

```bash
gh release download rl-map-memory-2026-11-06 \
  --repo aaravsinhaofficial/predictive-grid-cells-rl \
  --pattern rl-map-checkpoints.tar.gz
shasum -a 256 rl-map-checkpoints.tar.gz
tar -xzf rl-map-checkpoints.tar.gz
```

Pinned analysis dependencies are listed in the existing [analysis requirements](../../requirements-ring-analysis.txt); individual experiments record their device and source identities. Cloud training used PyTorch 2.9.1 and local analysis used 2.10. Exact sampled trajectories can differ across numerical backends. The checkpoint hashes identify the analyzed trained weights.

```bash
python -m pytest -q tests/test_lstm_goal_bearing_manifold.py \
  tests/test_lstm_goal_policy_map.py tests/test_rl_goal_memory.py \
  tests/test_original_rl_goal_ring.py tests/test_original_rl_goal_ring_cohort.py \
  tests/test_original_goal_ring_history_control.py
python scripts/lstm_goal_policy_map.py --help
python scripts/original_rl_goal_ring.py --help
python scripts/original_rl_goal_ring_cohort.py --help
python scripts/evaluate_rl_goal_memory_checkpoint.py --help
```

The original $500 compute authorization is cumulative. The [cloud ledger](ring_cloud_budget.json) includes the earlier study and this pilot. The cumulative conservative envelope is **$44.1913**, including the earlier allowance; this is not an invoice. The new pilot contributes $0.4549. Independent provider checks confirmed zero study VMs and boot disks after collection. No additional cloud budget was assumed for the redo.

All 25 focused tests above passed. Additional live checks verified every analyzed original checkpoint, all frozen cohort sources, matching first-discovery events, map inversion accuracy, completed training budgets, and the packaged checkpoint bytes. Figures were visually inspected and the new Markdown links checked.
