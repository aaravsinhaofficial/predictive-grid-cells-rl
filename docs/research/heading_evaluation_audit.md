# Independent heading-evaluation audit

Audited 2026-11-05 before confirmation-cohort analysis. This audit did not edit the frozen training/model files or access cloud resources.

The primary tracking wrapper and navigation timing passed independent checks. Seven tests in `tests/test_heading_navigation_audit.py` passed with `/opt/anaconda3/bin/python -m unittest discover -s tests -p test_heading_navigation_audit.py -v`.

## What was checked

- The training generator first completes movement at time `t`, then supplies the corresponding location, direction and angular/speed measurements. The navigation evaluator observes the previous completed action, decodes its current heading, and integrates its measured speed along that current direction. It skips integration at reset. An exact angular-sensor test instrument, with nonzero varying motor noise and turns, recovered the physical pre-action position within 3e-6 m using only supplied sensor channels.
- Navigation's controller takes estimated position, decoded heading, and known goal. True physical position and heading only generate observations and scores. The initial coordinate is known once at reset. Reaching the true goal does not silently terminate or alter a trajectory.
- All input-intervention branches clone one identical warmup state and physical reset. Restoration changes supplied input after two seconds without resetting state/history. Ordinary replacement is an exact no-op for the ordinary-code model. Exogenous noise is paired; physical trajectories may diverge because control depends on the model.
- A misleading initial heading cue changes only the two explicit heading-cue components. It leaves predictive spatial rates, measured angular velocity, speed and availability unchanged. Thus it tests repairing a cue-induced memory error while predictive code continues to encode the correct noisy travel direction.
- Predictive and ordinary models receive exactly the same noisy angular/speed measurements. The explicit-heading model receives the same measured direction used to construct the predictive shift. Ordinary and explicit-heading spatial rates match. After initialization, spatial dropout removes both predictive spatial information and explicit heading information while preserving the common motion sensors.
- The primary wrapper uses evaluation seeds 71001–71004, exactly 512 episodes each, and excludes exactly steps 0–3. A test with deliberately different error at step 4 detects accidentally using the older eight-step exclusion. Episodes average inside each network; the declared independent replicate is its training seed.
- The fixed 10 m transfer shifts actual and measured positions together. Actual test locations are checked against the training physical reach bound. Gaussian observation noise is unbounded mathematically; this is a check on physical locations, not a claim of disjoint sensor-distribution support.
- Anchor full-state perturbation gains use the actual initial state distances. Local phase-gain quantiles accompany the mean. A counterexample test shows that mean absolute phase displacement can remain exactly 10 degrees under a strongly nonuniform degree-one circular deformation.

## Interpretation and remaining reporting details

The navigation policy is fixed; the heading estimator is trained. The model moves forward and cannot turn while stopped, so retained travel direction equals heading. That assumption differs from an arena allowing backward or sideways motion.

In `evaluate_heading_anchor.py`, `neutral_slow_mode_tangent_alignment` selects the largest-magnitude eigenvalue at a state obtained after 40 blank steps. It is a dominant instantaneous mode alignment. It is not necessarily the neutral tangent eigenmode of an autonomous fixed ring. Long cue-free persistence, normal recovery and state geometry remain separate tests.

Navigation's `reverse` variant currently injects reversed predictive code even into ordinary and explicit-heading models. In the anchor evaluator it is a no-op in those arms. Use the navigation reversal result for the predictive arm or label the other interventions as distinct out-of-distribution code insertions.

The reviewed navigation evaluator reports first arrival, capped arrival time, final distance and path length. The locked protocol additionally requests overshoot and sustained holding; these require separate metrics before navigation results can satisfy that full secondary reporting requirement. First arrival can count a brief goal crossing. This does not affect the primary heading endpoints.

## Reviewed source fingerprints

These identify the reviewed versions; later evaluator amendments need their own tests/fingerprint.

| File | SHA256 |
| --- | --- |
| scripts/evaluate_ring_heading_navigation.py | c4602c35dc8a7ac9b631fd04cf93b69456693f2577b59d70be2132a818b0dc13 |
| scripts/evaluate_heading_confirmation.py | 69e9ce8327946c78ba99e8b964f479cc15fc684d3edc39a72cdf2d2953710ef2 |
| scripts/evaluate_heading_anchor.py | 623528cbf0db4d4441c0d1b85e5d15fb6c63625bbdded46efa8f89b542ebc3fc |
| scripts/train_ring_heading.py | 19d283beccd289e38cf516fb4b760bfe2ff1e45bd5fcc471a82a89308526d3fa |
| scripts/ring_heading_model.py | 6972766bfb029e5e97049123434ff75a235fec92c76c89986782dda12a1582f1 |
| tests/test_heading_navigation_audit.py | 4f1bdf88d03b06b3e454e5ddee28a23d14ba23905616fc8b9cde5cac6a0edf2e |
