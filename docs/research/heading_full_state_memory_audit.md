# Full-state memory aggregation and interpretation

The reproducible aggregate is `results/ring_heading/confirmation_evaluation/memory_summary.json`, produced by `scripts/summarize_heading_full_state_memory.py`. **All 48 final models (16 training seeds × three input populations) and their untrained controls are now included.** Every seed is retained; undefined geometry is reported with the unavailable seed IDs and denominators. The separately trained raw-input cohort is tracked independently.

Each trained or untrained model is assessed at three fixed physical positions. The main within-region summary averages (0,0) and (0.8,-0.8) within a training seed. The held-out position (10,10) is reported separately. The all-position summary is additional. Confidence intervals use training seeds, never phases, positions or timesteps as independent replicates.

## What the measures mean

- **Normal return:** how much a displacement away from the full 64-dimensional state curve decreases. Remaining distance is divided by the initial distance; return is one minus that ratio. The reference is the unperturbed curve at the same time. Full paired-state distance and curve-radius retention are also reported, because approaching a shrinking curve alone is not sufficient evidence of an attractor.
- **Tangent retention:** how much a disturbance along the remembered-direction curve persists. First calculate an absolute phase-displacement ratio for each resolvable initial phase, then take the median. Report the distribution's lower/upper quantiles and the resolvable fraction as well. A winding-invariant mean is not used as the retention endpoint.
- **Blank memory:** heading error, coverage and full-state radius after 2, 7 and 32 seconds without input. These are total blank durations after the initial cue. The perturbation outcomes at 1 and 5 seconds instead count from perturbing a state that has already undergone two seconds of blank settling.
- **Slow spectrum:** the eigenvalue closest to +1 and its alignment with the local remembered-direction tangent. Median statistics across phases are averaged within the stated context group. Normal-subspace spectral radii and singular values are retained to distinguish asymptotic modes from transient amplification.

The main perturbation has a norm equal to 10% of the initial full-state curve radius; the unchanged 2% and 30% sensitivity analyses are also included.

## How to describe the finding

Fast return across the sampled state curve, slower decoded-phase retention, heading coverage and finite state radius support structured direction memory. They are insufficient on their own to identify one smooth ring in the complete recurrent state. Dense inspection of the first seed found sharp transitions largely hidden from the output readout; sparse polyline segments can cross regions scarcely occupied by states, and median tangent alignment can miss the transition sector. The original aggregate therefore preserves these as operational measurements, not an all-model ring classification.

The separate [dense geometry protocol](heading_dense_geometry_protocol.md) evaluates every Grid/PGC seed with exact angular tangents, local spectral extrema, adaptive resolution and dense full-state perturbation measurements. It retains the original seed2001 branch case and the explicitly exploratory seed2002 continuous-loop example. Describing an **approximate full-state ring** requires this additional geometric and dynamical evidence for the particular model. Across the full population, “direction memory” or “circularly decoded memory” remains the broader description until model-specific geometry is stated. Even a smooth example should not imply an exactly neutral continuum or indefinite memory; long-delay drift and coverage loss remain visible.

The untrained recurrent initialization can itself have eigenvalues near +1. Therefore, a near-one eigenvalue alone is not evidence that training produced a ring. Alignment with the task-relevant tangent and accurate task memory are essential. The untrained readout is also random, so poor untrained heading accuracy alone does not determine its hidden-state geometry.

These are secondary descriptive diagnostics. They supplement the locked primary comparisons and do not create a new seed-selection rule or a new corrected significance family. The raw-input cohort is summarized separately with the same definitions under `results/ring_heading/raw_memory_evaluation`; it is never pooled with the encoder cohort.

## Final encoder-cohort results

The following are means over all 16 training seeds, averaging the two in-region contexts inside each seed. These preserve the original 128-phase evaluator; the dense geometry audit is reported separately because of the sparse-curve caveat above.

| Measure | Grid | PGC | Explicit heading |
| --- | ---: | ---: | ---: |
| Normal distance remaining after 1 s | 4.93% | 11.70% | 11.45% |
| Normal distance remaining after 5 s | 1.35% | 0.80% | 0.88% |
| Median phase displacement retained after 1 s | 96.20% | 99.12% | 98.80% |
| Median phase displacement retained after 5 s | 68.64% | 82.80% | 80.37% |
| Heading error after 2 s blank | 3.97° | 4.04° | 3.53° |
| Heading error after 7 s blank | 12.82° | 14.61° | 21.20° |
| Heading error after 32 s blank | 31.62° | 44.49° | 63.79° |
| Angular coverage after 32 s blank | 72.09% | 65.50% | 64.17% |
| Median slow eigenvalue at 2 s blank | 0.99793 | 0.99917 | 0.99940 |
| Median slow-mode/tangent alignment at 2 s blank | 0.99888 | 0.99737 | 0.99185 |

For orientation, the descriptive 95% training-seed intervals for 32-second blank error are 24.92–38.32° (Grid), 35.48–53.51° (PGC) and 52.54–75.04° (explicit heading). Untrained counterparts have approximately 90° heading error and far poorer angular coverage, yet their slow eigenvalues are also near one (~0.99965) and their median tangent alignment is much lower (~0.38–0.42). Their readouts are random, so these controls distinguish the joint learned representation rather than isolating recurrent dynamics from decoding.

The pattern supports fast correction of many off-curve disturbances and a slower direction-like coordinate. PGC preserves small direction displacements better over the first few seconds, but Grid returns normal disturbances faster at one second and has less long-delay heading drift. Thus the behaviorally useful PGC mechanism is **input-dependent direction correction**, not uniformly stronger autonomous memory.

[Complete-cohort blank-memory figure](../../results/ring_heading/confirmation_evaluation/blank_memory.png)

The completed exploratory dense cohort contains all 32 Grid/PGC models. PGC has lower peak-to-median angular state sensitivity in 12/16 pairs, with a mean paired log10 ratio of −0.683. This is a post-discovery observation about initialization-to-memory geometry, not a primary test or a count of valid rings. The independent seeds 5001–5016 and sole endpoint in [the new confirmation protocol](heading_geometry_confirmation_protocol.md) were specified before those new models were trained.

That independent replication is now complete: all 32 models are retained. The paired log10 contrast is −0.448 [−1.013,0.117], exact two-sided p=0.11014, with lower PGC sharpness in 10/16 pairs. The common-ordinary initialization control is similarly uncertain. This leaves the smoother-map tendency as heterogeneous rather than independently confirmed at the specified threshold. No model triggered a degeneracy or numerical-floor flag. The useful cue-anchoring mechanism does not require a reliable geometry advantage across every trained population. [Complete replication and figure](../../results/ring_heading/geometry_confirmation/README.md)

The raw-input memory and anchoring cohort is also complete (all eight Grid/PGC pairs). Native PGC input strongly contracts phase disturbances but leaves 25.22° stationary direction error and sampled coverage 0.843, versus 64.11° and 0.328 after ordinary replacement. The reduced architecture therefore uses the cue without reproducing the encoder cohort's precise anchoring. Removing the encoder also changes capacity, so the architectural comparison is not an isolated test of nonlinearity. [Raw-input figure and all results](../../results/ring_heading/raw_memory_evaluation/README.md)
