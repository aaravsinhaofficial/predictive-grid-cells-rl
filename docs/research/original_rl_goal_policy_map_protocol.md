# Original RL goal-code steering-field confirmation

This protocol follows descriptive pilots in original PGC+grid seeds1–2. It freezes the revised measurement before evaluating seeds3–8. The old dominant-action/no-angular-offset pilots and their source remain archived. Revised seed1–2 results are descriptive and never included in the confirmatory six-seed test.

## Task and intervention

Use the original1B-frame reward-trained PGC+grid LSTMs, original square-room renderer, original512-cell hexagonal spatial maps and original six-action actor. No weights are trained. At fixed observer positions, sweep facing direction through64 equally spaced headings. Keep current code, physical position, room appearance, and previous action the same between variants. Current movement direction is the context's arrival direction and is retained at zero velocity, matching the original space-mode convention. Clamp previous action to the dominant turning action identified independently from earlier native-policy recordings. Start h,c at zero and repeat the same complete input for15 and60 recurrent updates. This is an input-clamped response diagnostic, not an actual path, an autonomous-attractor assay, or a natural60-step behavioral history.

Each goal snapshot has ordinary slots at pickup=`goal_center−0.29×arrival_direction`. Predictive slots encode pickup plus0.20m along arrival. Change only those169 predictive goal slots: ordinary/no-shift, native, +90°, −90° and180° shift directions. All343 ordinary slots and every other input remain identical. The primary comparison is native versus ordinary. Rotated variants constrain and visualize the geometry; all are retained.

Use the fixed four contexts: goal(0,0), arrival0rad; goal(0.25,−0.20), arrival1.2rad; goal(−0.20,0.25), arrival−1.0rad; goal(0.20,0.20), arrival2.6rad. Room seeds are989131+context index. Observer positions are the fixed7×7 lattice spanning[−0.95,0.95]m, excluding points within0.4m of the true goal. Checkerboard parity, fixed before inspecting responses, splits training and heldout positions. The primary horizon is60 updates (chosen from the pilots);15 updates are a timing robustness check. No context or trained seed is excluded.

## Direction field and competing explanations

The primary field is the first circular moment of **all four translation actions**, summed across uniformly sampled headings: probability of each action times its physical world direction. Physical offsets from facing are0 for forward,π for backward,+π/2 for strafe-left and−π/2 for strafe-right. This avoids ignoring a secondary translation action. It is a preferred translation field under heading sampling, not the instantaneous movement at a particular heading.

Fit two models using only training positions:

1. A separate effective aiming point for each goal-code variant and one common angular offset.
2. One common aiming point and a separate angular offset for each variant.

The first describes a target displacement; the second describes a global heading-response rotation. Both allow a frame offset, so known physical action offsets are not confused with the policy's heading-response bias. Fit wrapped angular residuals by bounded least squares. Point coordinates are bounded to±2.5m; report optimizer failures and bound hits. Initialization uses training-position ray fits and their circular residual offsets. All variants use the **same** position weights: ordinary-condition all-translation tuning strength squared, floored at1e−9. This prevents variant-dependent weight changes from moving a fitted point without changing the direction field. Evaluate angular prediction error on heldout positions using the same fixed weights. Repeat with uniform weights; retain dominant-translation-only fields as robustness.

## Primary endpoint and interpretation

For each seed, average the four contexts' native-minus-ordinary fitted-point displacement projected onto the actual arrival direction, at60 updates, using all actions and common weights. Test the six unseen seed means with a two-sided exact sign-flip test; report the signed mean and descriptive95% Student-t interval. Seeds1–2 are displayed separately. The intended-direction claim requires a positive mean and p≤.05. All seeded models, contexts, timing variants and weighting checks are reported, with no post-result tuning.

An effective-target interpretation additionally requires that the variable-target model have lower mean heldout angular error than the variable-offset alternative in at least five of the six unseen seeds, after averaging contexts, and a positive overall mean improvement. Report the improvement interval and every context regardless. Any optimizer failure or point-bound hit blocks a clean target interpretation and is retained in the results. If these checks fail, describe the causal **steering-field change** without claiming an inferred goal coordinate. A successful fit still summarizes policy behavior; it does not demonstrate a neuronally encoded target or a ring attractor.

Synthetic tests verify each action-frame offset, recovery of known displaced targets despite a common angular bias, and rejection of a target-shift explanation for purely rotated fields. Freeze source and protocol hashes before the unseen models run. The summary must validate these hashes and all checkpoint identities.
