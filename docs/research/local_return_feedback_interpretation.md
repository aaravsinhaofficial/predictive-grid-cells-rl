# What the completed feedback-acquisition study suggests

**Stop ring-directed continuation from these four endpoints.** Every final misses the unchanged fast-steering criterion of 192/256 returns within two seconds. All four strict artifact audits passed, and every endpoint retains 256 valid trials. These are four adaptations of one original seed pair.

| Population | Starting actor | Before → fixed final |
|---|---|---:|
| [Grid](../../results/local_return_feedback_design/portable/cloud_audit/gc100_retained_actor.json) | Retained | 69 → 97 / 256 |
| [Grid](../../results/local_return_feedback_design/portable/cloud_audit/gc100_reset_actor.json) | Reset | 0 → 0 / 256 |
| [PGC + grid](../../results/local_return_feedback_design/portable/cloud_audit/pgc33_retained_actor.json) | Retained | 79 → 87 / 256 |
| [PGC + grid](../../results/local_return_feedback_design/portable/cloud_audit/pgc33_reset_actor.json) | Reset | 0 → 0 / 256 |

The retained policies improved modestly while preserving a narrow movement strategy. Grid's final sampled actions were entirely right turns and right strafes; PGC used those almost exclusively. A [separate baseline calculation](../../results/local_return_feedback_design/portable/restricted_action_bound/README.md) bounds **any controller restricted to that pair** at 139/256 potentially reachable headings, or 54.3%, on the saved starts and noise tapes. All 16 contexts passed its physical checks. Reaching 75% therefore requires a broader repertoire. The six-action networks can choose other actions, so the bound diagnoses the restricted strategy's limitation.

Reset actors explored broadly but learned little directed movement: final mean goal progress was approximately −0.15 cm for Grid and −0.25 cm for PGC, with action entropy near its six-action maximum. Retained policies made 17.2 cm and 14.8 cm progress. This points toward a control-acquisition bottleneck: preserving useful steering helps, while broad exploration alone did not produce the required rapid return.

The [initialization audit](../../results/local_return_feedback_design/portable/cloud_audit/baseline_actor_isolation.json) isolates the starting actor change. Within each population, retained/reset arms had identical first-free recurrent states, physical observations and critic outputs; only actor parameters differed. Their later training allowed every network weight to change.

The task also changes what constitutes useful navigation. It demands return within two seconds from randomized local orientations, with fresh vision and spatial codes throughout. Original navigation allowed longer trajectories and repeated natural pickups. Here the [setup snapshot](local_return_setup_goal_geometry.md) places ordinary goal markers only 1.45 cm from the center, while predictive markers lie 18.55 cm beyond it. Natural boundary pickups instead gave predictive shifts an inward advantage. This controlled geometry removes that particular benefit from the population comparison.

The literature supports reward-trained recurrent memory control ([Heess et al.](https://arxiv.org/abs/1512.04455)) and slow attracting manifolds in supervised analog-memory networks ([Ságodi et al.](https://arxiv.org/html/2408.00109v2)). Those results motivate behavior-first testing and selective perturbation recovery. Our acquisition failure leaves the organization of a successfully learned local directional memory unresolved.

**The strongest explanation for the original PGC advantage remains useful goal geometry and learned recurrent use of predictive inputs.** Eight-seed removal/restoration and geometry tests support the inward goal cue; candidate-update/output-gate interventions connect predictive influence to navigation. [Original mechanism evidence](../../mechanisms/navigation-explanation.md). Prior ring screens and these acquisition failures favor concentrating the present paper there. Any future ring study should begin with independently demonstrated reward-based directional memory and preserve its stopping rules, rather than extending this failed recipe.
