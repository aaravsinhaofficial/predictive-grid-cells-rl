# Methods: predictive input, learned heading memory, and navigation

This appendix details two distinct experiments. The **new heading model** is trained to estimate heading from simulated movement and spatial observations; a fixed controller then uses its estimate to navigate. The **original navigation agents** are previously trained square-arena LSTMs examined with input and gate interventions. Their architectures, tasks, training objectives, and seed cohorts are different and are not pooled. The separate published-network experiment on **internally classified predictive neurons** has its own [report](internal-pgc-recurrent-outputs.md) and [frozen protocol](../docs/research/public_internal_pgc_protocol.json); its ten networks are not part of either cohort below.

**Completed cohort:** all 16 paired training seeds, 2001–2016, in all three conditions have completed their fixed training and primary evaluations: 48 models. All 16 predictive models have also completed the direction-subspace and fixed-mean follow-ups. Pilot observations remain separate. Completion is recorded in the [primary summary](../results/ring_heading/confirmation_summary.json), [subspace summary](../results/ring_heading/subspace_confirmation/summary.json), and [fixed-mean summary](../results/ring_heading/tonic_direction_confirmation/summary.json). The original eight-seed gate experiments are complete. The 24-model raw-input and 32-model independent geometry cohorts also completed every planned model and evaluation; the latter's single primary test did not reach significance (exact p = 0.110). Separate later geometry, published-network, and explicitly designed-ring cohorts have their own protocols and status; they are not counted in these 48 models.

## 1. New heading task and its information sources

The model and simulator are defined in [ring_heading_model.py](../scripts/ring_heading_model.py). The spatial observations use the repository's analytic [GridPopulation](../code/codes.py), rather than learned biological-cell recordings. Each cell has a fixed hexagonal map,

`r_j(x) = clip((Σ_k cos(K_jk · (x − φ_j)) + 1.5) / 4.5, 0, 1)²`.

There are 128 cells in four equally sized modules. Module spacings are `0.40 × 1.42^m` metres for `m = 0,1,2,3`. Each module's orientation is sampled uniformly over 0–60°, and cell phases are uniform over the corresponding lattice unit cell. Within a paired training seed, the conditions have identical maps and cell identities. The population seed equals the training seed, so populations vary across the 16 seed pairs.

The three training conditions are:

| Condition | Spatial observation | Continuing direction information |
|---|---|---|
| Grid | All 128 maps evaluated at measured current position | No direct direction cue after initialization |
| PGC+grid | 43 maps evaluated 20 cm ahead along measured travel direction; the other 85 at measured current position | Direction is encoded indirectly in the shifted spatial pattern |
| Grid+explicit heading | Same ordinary maps as Grid | The same measured cosine/sine direction used to construct the PGC shift, available when the spatial observation is available |

PGCs are assigned across modules in counts 11, 11, 11, and 10. Predictive displacement is a fixed **spatial** shift, not a speed-dependent look-ahead time. The task supplies an independent noisy direction measurement to the observation generator. This is the information source that can correct integrated heading; it is not generated from the RNN's own estimate. The explicit-heading condition controls for access to that information in a more directly readable form.

All conditions receive measured angular velocity, measured forward speed, a spatial-availability flag, two initial-heading components, and a heading-availability flag. The complete input has 134 components. During the first four updates, every condition receives the same measured heading cue. After those updates, Grid and PGC+grid lose that explicit cue. When spatial observations are absent, all spatial code components are zero; continuing angular-velocity and speed measurements remain available. In dedicated cue-free dynamical tests, **all 134 input components are zero**, while learned encoder biases remain active.

The agent moves in an unbounded two-dimensional plane with 50 ms updates. Initial positions are uniform in `[−1.25,1.25]²` m and initial headings are uniform on the circle. Angular velocity follows a clipped autoregression, `ω_t = clip(0.92 ω_(t−1) + 0.30 ε_t, −2, 2)` rad/s. Forward speed follows `v_t = clip(0.98 v_(t−1) + 0.02×0.22 + 0.015 η_t, 0.10, 0.35)` m/s, except at the initial observation and stationary tails. Half the trajectories receive a stationary tail whose start is randomized by the source implementation. Turning is zero when stopped, so retained travel direction equals heading in this task.

At index `t`, position and heading already include movement from `ω_t` and `v_t`. The angular/speed measurement delivered at `t` refers to that same completed movement. Thus the observation and heading target are temporally aligned. The source is authoritative for exact random-number order and stopping/blackout sampling; the [independent navigation tests](../tests/test_heading_navigation_audit.py) check timing and information isolation.

| Noise or availability term | Setting |
|---|---:|
| Angular-velocity measurement noise | Gaussian SD 0.30 rad/s per update |
| Angular-velocity sensor bias | Gaussian SD 0.10 rad/s, fixed within a trajectory |
| Forward-speed measurement noise | Gaussian SD 0.01 m/s; measured speed clipped below at zero |
| Direction measurement noise | Gaussian angular SD 0.05 rad |
| Position measurement noise | Gaussian SD 0.01 m per coordinate |
| Rate-channel measurement noise | Gaussian SD 0.03, followed by clipping to [0,1] |
| Spatial availability | Independent probability 0.50, plus a continuous randomized blackout |
| Training recurrent preactivation noise | Gaussian SD 0.01 per unit per update |

The continuous training blackout begins between one-quarter and one-half of the sequence and lasts between approximately one-eighth and three-eighths of it. Some stopped tails remain cued and others become blank. The first four observations are always visible. The same trajectory, sensor noise, channel noise, visibility, and recurrent-noise realizations are paired across conditions within each training seed.

## 2. Architecture, objective, and fixed training budget

The input encoder is `134 → 256 → 256 → 64`, with SiLU between its linear layers. The recurrent state has 64 unconstrained units:

`h_(t+1) = 0.75 h_t + 0.25 tanh(b(x_t) + W_rec h_t + ξ_t)`.

An affine two-output readout is trained against cosine and sine of the current true heading. Every condition has **121,026 trainable parameters**. Recurrent weights begin with an orthogonal initialization; encoder weights use Xavier initialization. The architecture has no circular connectivity constraint, state-geometry loss, ring target, or attractor regularizer. Circular heading is the supervised variable, so successful training may organize memory around that variable; hidden-state ring geometry is measured afterward.

[train_ring_heading.py](../scripts/train_ring_heading.py) minimizes mean squared readout error plus `10⁻⁴ × mean(h²)`. AdamW uses learning rate `10⁻³`, weight decay `10⁻⁵`, global gradient clipping at 1, and a cosine schedule ending at learning rate `2×10⁻⁴`. Each model receives exactly 6,000 updates, batch size 64, sequence length 128. Initial hidden state is zero. The condition-independent task generator uses seed `training_seed + 1,000,003`; model initialization uses the training seed. The intermediate 3,000-update and final 6,000-update checkpoints are fixed in advance. Validation does not select the final checkpoint.

The [confirmation protocol](../docs/research/heading_confirmation_protocol.md) was locally timestamped before training seeds 2001–2016. It is not an external preregistration. The [source manifest](../docs/research/heading_confirmation_sources.json) records the protocol and training-source hashes. Checkpoints record the complete configuration, source hashes, seed, optimizer, training history, completed updates, and stopping reason. All planned seeds remain in the analysis, including unsuccessful training; interrupted jobs must complete their original budgets. Earlier raw-input and other exploratory models are separate sensitivity cohorts.

## 3. Heading evaluation and statistical units

The locked [primary evaluator](../scripts/evaluate_heading_confirmation.py) uses four fresh evaluation seeds, **71001–71004**, with 512 trajectories per seed: **2,048 trajectories per trained network**, each 256 steps long. Evaluation includes sensor noise but no added recurrent noise. Native and ordinary-replacement inputs use exactly the same underlying physical trajectories and measurement noise. Primary errors exclude exactly the first **four** initialization updates, unlike some pilot and subspace assays that exclude eight.

Heading error is the absolute wrapped angular difference between the affine readout's `atan2` angle and the true heading. Errors are averaged over eligible times within each trajectory, then over trajectories within each trained seed. Translation tests add `[10,0]` m to both true and measured positions while retaining movement, direction, noise, and timing. An explicit bound verifies that translated true positions lie beyond the training task's reachable x range.

The four primary paired endpoints are PGC-versus-Grid tracking error, ordinary-replacement-versus-native error within frozen PGC networks, native-versus-ordinary phase anchoring, and PGC-versus-Grid translated-position error. The [aggregator](../scripts/summarize_heading_confirmation.py) requires all 16 planned seeds and validates protocol, source, and checkpoint identity. It reports mean paired differences, nominal 95% paired-t intervals, exact two-sided sign-flip tests over all `2^16` assignments, and Holm correction across these four endpoints. Training seeds are the independent units; trajectories, contexts, and phase samples are not additional training replicates.

The [transfer evaluator](../scripts/evaluate_ring_heading_transfer.py) supplies separately labeled secondary reversal, direction-shuffling, restoration, and doubled-noise tests. Its defaults are seed 931731, 256 trajectories × 256 steps, exclusion of eight initial steps, and restoration at step 64. These results do not replace the primary endpoints.

## 4. What qualifies as a memory ring or as phase anchoring

A circular plot or a low-dimensional PCA projection is not the operational criterion. [evaluate_heading_memory.py](../scripts/evaluate_heading_memory.py) initializes 128 evenly spaced headings at three physical positions, supplies the four-step training cue, then applies 40 all-zero-input settling steps followed by 600 further blank steps. This tests 30 seconds of continuation after settling. It evaluates the trained model and an architecture-matched untrained model initialized with the same seed.

The analysis uses the full 64-dimensional recurrent state. It measures state radius, decoded-heading coverage and accuracy, winding, readout amplitude, and the distribution of local phase gains. Tangents are cyclic central differences along the measured state curve. Random normal perturbations (seed 419) are projected into the tangent's orthogonal complement. Tangent and normal disturbances have equal norms of 2%, 10%, or 30% of the initial full-state curve radius. Perturbed paths are compared with the **time-matched** unperturbed curve, using nearest projection onto its closed piecewise-linear interpolation as well as direct paired-state distance.

A finite-time ring interpretation requires a persisting, noncollapsed family of states, relatively preserved displacement along that family, and recovery toward it after transverse displacement. Report the entire trajectory and phase distribution; a circle can deform or gradually drift toward a few preferred states. The [geometry routines](../scripts/ring_dynamics.py) include a known ring as a positive control and a collapsing input-driven circle as a negative control. Both contract disturbances, but only the first preserves a finite memory manifold.

A subsequent descriptive [curve-bridge audit](../scripts/audit_heading_curve_bridges.py) checks an additional limitation: uniform sampling of initialization headings need not sample full-state arc length uniformly. A few adjacent states can differ sharply in hidden dimensions while their decoded directions remain close. Near such transitions, a coarse central difference is a secant rather than a validated local manifold tangent, and a straight interpolated segment may cross unoccupied states. The audit records maximum adjacent jump relative to full-state radius, arc length concentrated in segments longer than five times the median, slow-mode alignment near those segments, and recovery-distance sensitivity after removing only their interpolated interiors while retaining every measured vertex. This threshold is an exploratory numerical diagnostic, not a new confirmatory classification rule. Local recovery and useful direction correction can remain strong even when an unqualified smooth global-ring interpretation fails. Dense-angle and common-history checks are therefore considered separately from angular readout coverage. [Audit data](../results/ring_heading/geometry_audit/curve_bridge_sensitivity_all_available.json).

The separate [anchoring assay](../scripts/evaluate_heading_anchor.py) uses 72 initial headings at five in-region positions and `[10,10]` m. After four cue steps and 40 blank steps, a ±10° phase disturbance substitutes actual neighboring full states from that measured curve. Baseline and disturbed replicas then receive the same physical input. Conditions include native, all-blank, ordinary replacement, reversed predictive shift, and ordinary input followed by restoration after 20 updates. For PGC networks, a self-generated-preview branch constructs each replica's predictive direction from its own previous readout, removing the independent external direction reference.

The locked endpoint is the median per-state retained phase displacement at update 100, averaged across the five in-region positions. The complete continuation is 200 updates. Accuracy, full-state error, state radius, and coverage are reported alongside correction. Accurate anchoring additionally requires native coverage at least 0.90 and lower true-heading error than ordinary replacement. Failure of that criterion remains an outcome, not an exclusion. Convergence into one wrong state counts as collapse. A fixed directional cue pins one phase of the memory ring; it is not expected to preserve a continuum of neutral phases while that same cue remains fixed.

## 5. Encoder direction-plane intervention

The [subspace follow-up protocol](../results/ring_heading/subspace_confirmation/PROTOCOL.md) and its [frozen source manifest](../results/ring_heading/subspace_confirmation/frozen_manifest.json) were fixed after the pilot and before confirmation subspace outcomes. All 16 PGC models are included, regardless of their geometry or performance. Validation-only and local-phase-summary amendments are recorded in that manifest.

[heading_direction_subspace.py](../scripts/heading_direction_subspace.py) computes the frozen encoder-drive difference `Δb = b(x_PGC) − b(x_ordinary)` for paired inputs generated at the same measured position and direction. For fitting, speed/angular velocity are zero, spatial availability is one, and the initial-heading cue is absent. Calibration uses seed **517913**, 1,024 independent positions in `[−1.25,1.25]²`, and 32 uniformly spaced directions with independent position-specific phase offsets. A regression on an intercept, cosine, and sine supplies two coefficient vectors; their orthonormal span defines the fixed rank-2 projector `P`. The intercept is excluded from this plane.

Held-out sets contain 256 new positions × 32 directions, both in-region and shifted 10 m along x. Report harmonic regression variance explained, within-position directional variance explained, projected drive energy, and held-out circular direction readout error. Rank two is imposed by the first-harmonic question; the complete encoder difference is not assumed intrinsically two-dimensional. Harmonics 1–3 are a predeclared fit-only secondary diagnostic.

Frozen recurrent networks receive native drive, ordinary drive, `b_PGC − PΔb`, or `b_ordinary + PΔb`. The residual-only expression `b_ordinary + (I−P)Δb` provides an algebraic check on the deletion branch. Three rank-2 random planes, generated with seed **517914** and orthogonal to the fitted plane, provide deletion and restoration controls. Their components are scaled to match the fitted component's norm **for each input in encoder preactivation space**. This does not guarantee matched post-tanh state change or readout sensitivity.

Only calibration and scoring use direction labels. The intervention receives two observed encoder drives and a frozen projector, not true heading, position, or a target answer. The experiment supplies the counterfactual ordinary input; it does not assume the unmodified agent naturally receives both vectors as separate channels.

The plane is tested in the 72-phase anchoring assay, 256 matched 256-step forced trajectories with evaluation seed **931731**, and closed-loop navigation. Forced-trajectory comparisons share four native initialization steps and exclude the first eight steps from error summaries. Subspace navigation uses all 16 models, 128 episodes each, seed **836219**, 40 seconds, and native-region and 10 m transfer scenarios. Each navigation branch controls its own physical trajectory. The primary subspace heading and navigation outcomes are at transferred places. Their four paired contrasts—native benefit, deletion cost, restoration benefit, and restoration over the mean random control—have separate prespecified four-test Holm families. The [runner and aggregator](../scripts/run_heading_subspace_confirmation.py) wait for every seed before producing complete-cohort tests.

Validation checks untouched native/ordinary full-state trajectories, deletion/residual equality, projector idempotence, matched sham norms, zero intervention during blackouts, and unchanged checkpoint hashes. Projector archives are bound to their originating checkpoint SHA and model seed. Independent navigation reference runs require identical success and arrival, with continuous-distance tolerance 10⁻⁴ m. Exact sources and episode-level data are archived.

A [validation-only numerical amendment](../results/ring_heading/subspace_confirmation/NUMERICAL_VALIDATION_AMENDMENT.md) was needed for the final model, seed 2012. Different float32 arithmetic order in the algebraically equivalent deletion and residual-only expressions accumulated 1.79×10⁻⁵ maximum hidden-coordinate discrepancy over 256 updates. Float64 reduced it below 6.2×10⁻¹⁵; the float32 decoded discrepancy was below 0.002°. A new documented wrapper changes only this unused alias's assertion tolerance from 10⁻⁵ to 10⁻⁴. Original evaluator bytes, scientific computations, primary branches, other checks, statistics, and all 16 included seeds remain unchanged. The original failing log, effective-function hashes, and independent numerical audit are retained. Shared native, ordinary, and full-plane restoration outcomes match exactly between the two complete subspace studies.

An additional [fixed-mean diagnostic](../results/ring_heading/tonic_direction_confirmation/PROTOCOL.md), specified after the pilot and after six confirmation models had their original subspace analyses computed, separates a tonic offset from the varying plane component. It uses the original independent calibration mean `μ=mean(Δb)` and the observed availability flag `a`. The tonic component is `aPμ`; the varying component is `PΔb−aPμ`. New branches restore only the tonic component to ordinary input or remove only the varying component from native input. They preserve exact no-op behavior during blackouts and receive no direction label at inference. All 16 PGC models enter this additional diagnostic, with the same trial seeds, sample sizes, and assays. Three exploratory contrasts per endpoint have their own correction and leave the earlier four-plus-four families unchanged. This control addresses a global fixed offset, not every context-dependent average or nonlinear response to drive fluctuations. [Diagnostic implementation](../scripts/heading_tonic_direction_diagnostic.py) · [Runner](../scripts/run_heading_tonic_diagnostic.py).

## 6. Jacobian eigenvalues, eigenvectors, and phase gain

For fixed input, the full-state update Jacobian is

`J(x,h) = 0.75 I + 0.25 diag(1 − tanh²[b(x)+W_rec h]) W_rec`.

The cue-free memory analysis samples spectra at absolute blank updates 40, 100, 140, 300, and 340. It reports the eigenvalue closest to +1 separately from the largest-modulus eigenvalue, their alignment with the empirical tangent, full singular values, and projected normal quantities. For a complex eigenmode, tangent alignment uses its real invariant plane rather than a phase-dependent complex eigenvector angle. A normal-to-normal projected singular value excludes leakage into the tangent and must not be interpreted as total growth of all transverse disturbances.

[evaluate_heading_cue_spectra.py](../scripts/evaluate_heading_cue_spectra.py) compares blank, native, and ordinary inputs at **identical** cue-free states and, separately, identical native-anchored states. It uses 72 phases and the six anchoring positions. Native-anchored states follow 200 native updates from the cue-free curve. Changing input while holding state and recurrent weights fixed isolates the change in the activation slopes that multiply recurrent feedback.

For affine readout `y = Wh+b`, the phase gradient is `(y₀W₁ − y₁W₀)/||y||²`. One-step phase gain is

`[grad phase(h_next) · J(x,h) · tangent] / [grad phase(h) · tangent]`.

This is gain in the trained phase coordinate, not an eigenvalue. The denominator accounts for local conversion from state displacement to phase displacement. Report resolvability and output amplitude, and distinguish instantaneous operating-point spectra from finite-time recovery. The [independent audit](../docs/research/heading_cue_spectra_audit.md) checks the derivative against autograd and central differences, invariance to tangent scale/sign and readout rotation/scaling, and a system with known radial and phase contraction. It also checks the full-state Jacobian identity through the tanh slopes.

## 7. Navigation controller and behavioral outcomes

Navigation is a downstream test of the trained heading estimate, not a further reinforcement-learning phase. [evaluate_ring_heading_navigation.py](../scripts/evaluate_ring_heading_navigation.py) supplies the same fixed controller to every model. The controller knows the reset coordinate and goal. It integrates measured forward speed along decoded heading to maintain its own position estimate; it never receives current true position or true heading. The simulator uses true pose only for physical evolution, sensory generation, and scoring. The subspace [navigation wrapper](../scripts/heading_subspace_navigation.py) retains that controller and simulator.

Starts are uniform in `[−0.5,0.5]²` m. Goals are 0.8–1.5 m away at a uniform random bearing. Goal-bearing error is computed from estimated position and decoded heading. Angular command is `clip(2 × error, ±1.8)` rad/s. Forward command is `0.25 × clip(cos(error), 0.24, 1)` m/s. The controller stops within 0.08 m of its **estimated** goal. Physical angular velocity and speed use smoothing coefficients 0.8/0.2 and motor-noise SDs 0.03 rad/s and 0.005 m/s per update. Moving speed is clipped to 0.001–0.35 m/s; both speed and rotation are zeroed when stopped. No ground-truth arrival signal resets or stops the controller.

Navigation supplies the standard noisy observations with spatial visibility probability 0.5 and a shared blackout from 4–7 s. Four native initialization steps precede the intervention. The current observation describes the preceding completed movement; the position estimate is updated before choosing the next action, with the initial no-movement observation handled separately. Conditions share underlying per-step observation and motor noise, but their actions produce different physical paths.

First-arrival success means entering a 0.15 m goal radius by 20 or 40 s. The 20 s outcome is a prefix of the same 40 s episode. Also report capped arrival time, final goal distance, traveled distance, and position-estimate error. The [secondary navigation protocol](../docs/research/heading_navigation_secondary_protocol.md) distinguishes passing through from holding at the goal: maximum post-entry excursion among entrants, occupancy during the final five seconds across all episodes, and two consecutive seconds inside the goal with speed at or below 0.05 m/s. These are score-only measurements. Scenarios additionally include doubled angular bias, a ±20° misleading initial heading cue, and their combinations with spatial translation. Subspace confirmation fixes its scope to baseline and translated places.

## 8. Independent geometry replication and architecture sensitivity

The [independent geometry protocol](../docs/research/heading_geometry_confirmation_protocol.md) was fixed after exploring the primary cohort's geometry and before training new seeds 5001–5016. It trains 16 new Grid/PGC pairs with the unchanged encoder, task and 6,000-update budget. At position (0,0), each model receives four deterministic initialization cues followed by 40 all-zero-input steps. The analysis propagates exact derivatives of the full state with respect to initial heading in float64, using 1,024 uniform angles and 20 adaptive bisections in each of four large-separation intervals. The primary endpoint is log10 of the estimated peak derivative norm divided by its uniform-angle median. All 32 models are retained; fixed floors and degeneration flags are recorded.

This measures unevenness of the initialization-to-memory map. It is not a count of attractors, a topological classification, or an assay of transverse contraction. Absolute peak and median sensitivities, state radius, readout coverage and accuracy accompany the ratio. Cue-end and matched ordinary-initialization conditions are descriptive controls. The single paired contrast uses all 16 seeds, a nominal 95% paired-t interval, and an exact two-sided sign-flip test over all 65,536 sign patterns. Aggregation waits for all pairs. The [frozen evaluator](../scripts/evaluate_heading_geometry_confirmation.py) checks independently propagated derivatives against automatic differentiation in each model, and the [final-cohort audit](../scripts/audit_heading_final_cohorts.py) reconstructs the endpoint from saved arrays.

The separate [raw-input protocol](../docs/research/heading_raw_architecture_protocol.md) uses eight new seeds, 3001–3008, with all three input conditions. It replaces the nonlinear encoder with a linear sensory map while retaining 64 recurrent units, the same task, and the same 6,000-update budget. This reduces parameter count: it is an architecture and capacity sensitivity, not an isolated manipulation of nonlinearity. Parameter count remains matched across input conditions within the cohort. Tracking, predictive removal, and spatial transfer use the unchanged 2,048-trajectory evaluation and a separate three-test Holm family over eight paired seeds. Raw-input memory and anchoring diagnostics are descriptive. These 24 models are never pooled with the primary 48 or geometry 32.

The explicitly designed ring uses another eight-seed cohort, 4001–4008, with its own [fixed protocol and report](designed-predictive-ring.md). Its sensory phase readouts and scalar cue gain are trained; its ring geometry and correction law are imposed. Its exact mathematical derivation appears after these methods in the PDF and is also available [separately](../docs/research/predictive_ring_derivation.md).

## 9. Separate original-agent LSTM gate experiment

The original trained navigation LSTMs use seeds **1–8**, distinct from the new heading cohort. Their completed input and gate experiments are documented in [predictive-memory-gates.md](predictive-memory-gates.md), with source/data links and all statistical families.

The targeted stability assay takes 32 native replay states per PGC model, computes the leading right singular vector of the full `(h,c)` update under native or ordinary-goal input, and tests each fixed direction under both inputs. Perturbations have raw-coordinate RMS amplitudes 10⁻⁴ and 10⁻², equivalent to Euclidean norms of those values times √512 for the concatenated 256-unit hidden and 256-unit cell states. Replicas follow 60 actual subsequent input updates and are compared with their own condition-matched unperturbed branch. One-update and 60-update amplification form four planned paired contrasts. These selected vulnerable directions complement a separate random-perturbation experiment; they are not a claim about average noise sensitivity.

Gate attribution restores selected **input preactivation rows** while each branch retains its own evolving state and unchanged recurrent weights. It decomposes the cell update into direct carried-state, forget-feedback, input-gate, and candidate terms. Twelve exploratory gate comparisons are corrected separately. This analysis identifies memory sensitivity pathways; it does not by itself identify a ring.

Matched behavioral tests use four conditions: intact predictive goal input; ordinary goal replacement; ordinary goal with forget-gate restoration; and ordinary goal with candidate-plus-output restoration. Current predictive spatial input remains intact throughout. Only the 169 predictive goal slots are converted to ordinary code. There is no added state noise in this behavioral assay. Evaluation seed **737373**, eight models, four conditions, and 128 matched episodes per model-condition produce **4,096 completed episodes**. Layouts, physical noise, and action random uniforms are shared; trajectories and recurrent states evolve separately. Four planned paired score contrasts use the eight training seeds as units, with separate Holm correction of paired-t and exact sign-flip tests. Memory-protection and navigation-recovery percentages refer to different assays and are not additive gate contributions. A behavioral disturbance test would be needed to link the measured protection directly to navigation benefit.

## 10. Reproduction and provenance

Training, evaluation, and aggregation entry points are linked above. A single example model can be reproduced with:

```bash
python scripts/train_ring_heading.py --condition predictive --seed 2001 \
  --encoder-size 256 --hidden-size 64 --steps 6000 --length 128 \
  --batch-size 64 --learning-rate 0.001 --study-stage confirmatory \
  --device cuda --out reproduction/predictive_s2001.pt
python scripts/evaluate_heading_confirmation.py reproduction/predictive_s2001.pt \
  --out reproduction/primary_predictive_s2001
python scripts/evaluate_heading_anchor.py --checkpoint reproduction/predictive_s2001.pt \
  --out reproduction/anchor_predictive_s2001
python scripts/evaluate_heading_memory.py reproduction/predictive_s2001.pt \
  --out reproduction/memory_predictive_s2001
python scripts/heading_direction_subspace.py --checkpoint reproduction/predictive_s2001.pt \
  --out reproduction/subspace_predictive_s2001
python scripts/heading_subspace_navigation.py --checkpoint reproduction/predictive_s2001.pt \
  --subspaces reproduction/subspace_predictive_s2001/fitted_subspaces.npz \
  --out reproduction/subspace_predictive_s2001/navigation --validate-reference
```

Use all three training conditions and all seeds 2001–2016 for the complete paired study. CPU is also supported. Training refuses to overwrite an existing checkpoint/log. The locked evaluator rejects mismatched source hashes, configurations, incomplete training, and pilot identities. Checkpoint bytes and exact source snapshots, rather than a claimed hardware-independent bitwise retraining guarantee, identify the analyzed models. Frozen protocols, intermediate checkpoints, numerical controls, per-episode data, and per-seed summaries make the reported comparisons traceable. The [reproduction guide](../docs/research/ring_reproduction.md) documents the public final-checkpoint archive, dependencies, tests, and regenerated figures.
