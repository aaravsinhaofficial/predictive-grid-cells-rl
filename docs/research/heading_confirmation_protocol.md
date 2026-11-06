# Locked confirmation: predictive input anchors learned direction memory

Written 2026-11-05, before training any confirmation seed. This is a locally time-stamped analysis plan, not an external preregistration.

## Model and training

Use the existing `HeadingTask` fixed20cm hexagonal-code task, with the current default noise, visibility, movement and stopping distributions. The model has a learned256/256 sensory encoder and64 unconstrained leaky recurrent units. Every population has the same121,026 trainable parameters. No ring connectivity, hidden-state geometry loss or attractor regularization is imposed. Training supervises only sine/cosine heading, plus the already specified generic activity cost.

Train16 independent paired seeds **2001–2016**, each with Grid, PGC+grid and Grid+explicit-heading inputs. Use6,000 updates, batch64, sequence length128, AdamW1e-3 with the existing cosine schedule, fixed checkpoint at6,000. Preserve3,000-update checkpoints as descriptive learning-curve data. Do not select a checkpoint by which condition performs better. All completed seeds remain in the analysis; unsuccessful learning is an outcome. Incomplete jobs are resumed/repeated to their originally specified budget and recorded.

The explicit-heading control receives the identical direction measurement used to generate PGC shifts. Thus it tests the contribution of direction information and how easily that information is extracted. Ordinary and predictive conditions receive identical noisy angular-velocity/speed streams. The planned raw-input64-unit comparison is a secondary architecture sensitivity, kept separate from this confirmation family.

## Primary family: four paired effects

1. **Tracking benefit:** PGC+grid minus Grid heading error on a fresh256-step test distribution, excluding the first4 initialization steps. Lower error is the predicted effect.
2. **Causal input contribution:** in each frozen PGC network, ordinary replacement minus native heading error on exactly the same physical trajectories/noise. Positive means the predictive signal supplies useful correction.
3. **Phase anchoring:** after a ±10degree shift along the measured conditional memory curve, compare the median local retained phase displacement under native versus ordinary-replacement input at100steps. Lower native retention indicates stronger correction. A network qualifies as accurate anchoring only when native decoded-phase coverage is at least0.90 (largest unsampled arc at most36degrees) and its true-heading error is lower than ordinary replacement. Report the effect for all seeds and the number meeting these independent safeguards; failed qualification is not an exclusion.
4. **Spatial transfer:** PGC+grid minus Grid heading error after translating test trajectories10m beyond the training region. Lower error is the predicted effect.

The first, second and fourth endpoints use the same fresh underlying trajectories within seed and condition, with2,048 total episodes split over four fixed evaluation seeds71001–71004 (512 per seed), keeping evaluation seeds distinct from pilot/validation seeds. Aggregate episodes within each trained seed first. The anchoring endpoint uses the five specified in-region physical positions and72 initial phases; the10m position is a separate sensitivity check. Initial cue generation,40-step cue-free settling, exact phase-state substitution and100-step endpoint are fixed in `evaluate_heading_anchor.py`.

Use paired mean differences,95%t confidence intervals, exact two-sided seed sign-flip tests over all2^16 assignments, and Holm correction over these four endpoints. Also report individual seed effects. Do not use episodes or phase samples as independent training replicates. Retain both mean-error and phase-correction evidence rather than equating them.

## Required mechanistic checks and secondary analyses

- Cue-free full-state manifold persistence, normal perturbation recovery, phase-memory heterogeneity and Jacobian spectrum, including slow-mode/tangent alignment. A mean phase displacement around a complete circle can be fixed by winding; use local distributions and coverage as well.
- Reversed shifts, shuffled directions, and restoration in frozen networks.
- Grid+explicit-heading control, doubled sensor noise, longer cue gaps, and learned-encoder removal/projection where meaningful.
- Navigation with the same fixed controller for every model. The controller receives learned heading and measured speed, known reset position and goal, and integrates its own position estimate. Ground-truth position/heading may generate observations and scores, but may not enter the controller. Report failures, arrival, overshoot and sustained holding separately.
- A fixed initial-heading cue error tests whether predictive spatial input repairs memory and helps navigation.
- Compare raw-input architecture as a separate sensitivity cohort. Do not pool its seeds with the primary architecture.

Confirmatory figures and prose must distinguish a learned recurrent memory ring, externally driven phase correction, and intrinsic transverse stability. The original square-arena LSTMs are a separate evidence source; the new task does not by itself identify their internal representation as a ring.
