# Independent audit of the main mechanistic report

Audit date: 2026-11-05. Target: [main report](../../mechanisms/predictive-ring-attractor.md).
Initial reviewed SHA256: `bde840ca667efffeeff91993009913530bdceecee49f28d524520b9f48ed4fd5`.
The report is still being finalized. The completion check below now covers the
independent geometry and raw-input cohorts as well. No main-report or
frozen-evaluator source was edited by this audit.

**Finding:** the completed-cohort numbers, units, and statistical families in
the reviewed report agree with their source summaries. The report correctly
separates designed recurrence, learned direction estimation, original
reinforcement-trained navigation agents, and published internal predictive
neurons. No numerical result requires correction. Minor interpretation
clarifications are recorded below.

## Numerical traceability

| Report item | Source check | Result |
|---|---|---|
| Primary 48 models, 16 paired seeds | [Primary summary](../../results/ring_heading/confirmation_summary.json) | Grid/PGC/explicit errors 28.143/7.418/4.801°; translated errors 28.132/7.977/4.802°. PGC–Grid difference −20.7258°, 95% paired interval [−21.1912, −20.2603]. Ordinary replacement costs 66.0323° [64.5809, 67.4837]. All four exact Holm p values are 0.0001220703; 15/16 pass the additional anchoring criterion, without exclusions. |
| Direction-plane intervention, all 16 PGC models | [Subspace summary](../../results/ring_heading/subspace_confirmation/summary.json) | Native/ordinary/remove/restore/random heading errors 8.225/75.426/71.292/10.047/75.900°. Navigation success 77.393/6.787/8.154/69.775/7.161%. Each of the separate four-heading and four-navigation families has exact Holm p = 0.0001220703 throughout. Random controls are averaged within model. |
| Fixed mean, all 16 models | [Tonic summary](../../results/ring_heading/tonic_direction_confirmation/summary.json) | Mean-only 75.215°/6.592%; preserve-mean/remove-variation 71.042°/8.545%. Mean-only benefits are not significant: heading p = 0.338623, navigation p = 0.838867. Other contrasts have p = 0.000091553 within the separate exploratory three-test families. |
| Reduced phase equation, all 16 models | [Mechanistic summary](../../results/ring_heading/mechanistic_confirmation/summary.json) | Mean κ = 4.79670 s⁻¹; held-out native field variance explained 0.785774; recovery prediction MAE 6.88835° versus 66.88508° without anchoring. These are descriptive model-prediction metrics, not behavioral error or new primary tests. |
| Memory-state replacement, all 16 pairs | [Population pulse summary](../../results/ring_heading/mechanistic_confirmation/population_pulse_summary.json) | Total 2 s heading errors Grid 21.69145°, PGC 4.93734°. First-5 s mean path deviations 0.155796/0.023580 m = 15.58/2.36 cm. Arrival-cost difference 0.0234375 [−0.002044, 0.048919] includes zero. Intervals are descriptive and unadjusted. |
| Long cue-free memory | [Memory summary](../../results/ring_heading/confirmation_evaluation/memory_summary.json) | Mean in-region error after 32 s total blank input is Grid 31.61933°, PGC 44.49306°. Correctly distinguished from input-supported correction. |
| Selected learned example, seed 2002 | [Dense audit](../../results/ring_heading/geometry_audit/dense_ring_predictive_s2002.json) | 4,096 directions; endpoint closure error 1.89×10⁻¹⁵; decoded angular derivative minimum 0.461 > 0; projected normal spectral radii 0.88348–0.98883. At 1 s, mean normal-distance retention 0.115685 and median angular-displacement retention 0.976267. Selected exploratory example, not a cohort-wide topology result. |
| Designed ring, eight new seeds | [Designed aggregate](../../results/ring_attractor/designed_compass/aggregate.json) | Primary navigation PGC/Grid/ordinary/restored/explicit 97.7539/22.7539/21.6797/58.9844/97.7539%. Four tests: raw exact p = 0.0078125; Holm p = 0.03125. PGC gains 12 or 16 s⁻¹ give aligned unit-cue phase multipliers 0.4 or 0.2 at dt = 0.05 s; 63 transverse modes remain 0.9. |
| Original-agent PCA, eight pairs | [Variance-source summary](../../results/lstm_variance_sources_1B/summary.json) | Grid/PGC top-ten shares 46.4167/52.3240%; after episode demeaning 40.9437/42.8377%. These are covariance-distribution results, not dynamics or causal memory attribution. |
| Original-agent gate recovery | [Recovery intervals](../../results/ring_attractor/goal_gate_behavior/recovery_intervals.json), [gate tests](../../results/ring_attractor/gate_stability/summary.json) | Forget restoration recovers 92.3029% of mean memory protection; candidate/output restoration recovers 83.7524% of mean score loss. Separate assays and denominators. Four-second forget attribution exact Holm p = 0.09375 across 12 exploratory tests, correctly disclosed. Behavioral inference uses its separate four-test family. |
| Published internal PGCs, ten models | [Public aggregate](../../results/ring_attractor/public_internal_pgc/aggregate.json) | First-0.4 s, 25%-attenuation population differences are +0.08371, −0.01978, +0.07100, −0.00453 cm. All four paired intervals include zero; all five primary Holm p values equal 1.0. This is nonreplication at the fixed endpoint, not evidence of equivalence or dispensability. |

All sixteen subspace checkpoint hashes match the corresponding primary and
tonic evaluations. Shared native, ordinary, and full-plane restoration metrics
match exactly between the two subspace studies. The report's methods link
discloses the validation-only float32 alias amendment; it changed no scientific
branch, metric, contrast, or seed inclusion.

## Interpretation clarifications sent to the report author

1. **Designed fixed points require zero angular-motion input.** Without a
   spatial cue but with nonzero measured angular velocity, the designed phase
   still integrates motion. “With no cue and zero angular motion” states the
   fixed-ring condition exactly. The aligned-cue eigenvalue formula in the
   reviewed draft is correct; away from alignment its phase derivative also
   contains cosine of cue–memory disagreement.
2. **Use the repository's grid-code generator wording.** The new 512-channel
   designed model uses the same analytic code family, with new population
   seeds. It does not reuse the exact original RL agents' cell-map identities.
3. **Label the selected example's aggregations.** “About 12% mean normal
   displacement and 98% median angular displacement remain” avoids implying
   uniform recovery across every phase. Projected normal spectral radius
   below one is a local linear measurement, alongside—not a replacement for—
   finite-time nonlinear recovery.
4. **Keep total pulse error distinct from the induced penalty.** The Grid/PGC
   no-pulse errors at 2 s are 17.53834°/4.96193°. Thus the displayed
   21.69°/4.94° values include baseline differences. The between-population
   difference in the pulse-induced increment is 4.17769° [3.54785, 4.80753].
   Path deviation is already measured against matched no-pulse paths and
   directly supports the steering-disturbance conclusion.
5. **Make the phase-equation time unit explicit.** The reduction fits
   `(next angle − current angle)/dt`, and κ is in s⁻¹. Its verbal equation should
   say “rate of change” or “change per second,” not an ambiguous per-update
   change. The assay holds angular velocity and speed at zero; it describes
   stationary-cue correction, with motion integration tested elsewhere.

The remaining boundaries are appropriate: no unconditional global-ring claim
for all learned networks; no unique identification of literal phase subtraction
in the learned encoder; no claim that a rank-2 regression proves the entire
input is rank two; no generalization from position translation to arbitrary
environments; no inference that original RL performance is mediated by the new
compass mechanism; and no mixing of primary and exploratory test families.
The measured direction reference is available to the synthetic generator;
the experiment explains its downstream use rather than its biological origin.

## Renderer and completion checks

The [renderer](../../scripts/render_ring_report.py) was inspected and its real
flowable construction executed using the existing `/tmp/ring-pdf-deps`
dependencies. The paragraph ending “summarized by” has `keepWithNext=True`, and
the following flowable is the phase equation. This fixes the identified
introduction/equation widow structurally. No PDF was generated by this audit;
final page placement still needs the author's final render. Image/caption
groups are kept together, and unfinished status prevents an accidental final
render unless explicitly overridden.

Every local link in the reviewed main report, methods, and completed subspace
reports resolves. The original gate figure path is correct. Figure coverage is
labeled decoded heading coverage rather than full-state ring topology.

At the initial review, independent geometry confirmation had 0/32 models
evaluated and raw-input memory/anchoring had 8/16. No partial-cohort inference
was calculated by this audit. Both cohorts have now completed; the independent
follow-up below supersedes that pending status.

## Exact designed-map derivation follow-up

The added [derivation](predictive_ring_derivation.md) was independently checked
against `DesignedRing.step_cue`, its input decoder, the spatial-phase decoder,
and the archived numerical checks. No equation or timing defect was found.

- At zero angular motion and fixed external cue aligned with a unit-radius
  state, `J = ρI + (1−α−ρ)ttᵀ`. The cue changes only the tangent contribution,
  by `−αttᵀ`. One radial plus 62 orthogonal directions yield 63 transverse modes.
- For `0 < |e| < π` and `0 < α < 2`,
  `0 < α sin|e| < 2|e|`, so the absolute deterministic phase error decreases.
  The antipodal exception and the domain with nonzero phase-plane radius are
  correctly identified.
- The small-error bias/noise recurrence matches predict-completed-motion then
  assimilate-current-cue timing. Independent availability and linearization
  are explicitly assumptions, rather than claims about every blackout or
  nonlinear navigation trajectory.
- Re-orthonormalized float64 spectrum discrepancy is 2.1094×10⁻¹⁵. The separate
  operational float32 maximum ten-second blank-state drift is 7.7706×10⁻⁷.
  The appendix correctly does not present these as the same precision test.

No additional simulation was needed or run. Minor missing spaces were reported
to the author as typography only.

## Complete independent geometry and raw-input audit

The [independent audit script](../../scripts/audit_heading_final_cohorts.py)
passed after all **32 geometry models**, **24 raw-input tracking models**, and
**16 raw-input memory/anchoring models** were available. It reads raw saved
derivative arrays and per-model summaries, not the descriptive report writer.
[Machine-readable audit record](ring_final_cohort_audit.json).

Checks passed:

- Every geometry report has the frozen evaluator/protocol/training hashes,
  correct independent seed, complete 6,000-update budget, and exact checkpoint
  identity. Every uniform/refined derivative-array peak and uniform median was
  independently recomputed, giving the recorded primary log ratio.
- All 65,536 geometry sign patterns, the paired-t interval, geometric-mean
  ratio, and complete inclusion of all 16 pairs were independently checked.
- Every raw-input checkpoint matches the locked training sources, raw
  architecture, and corresponding tracking, memory, and anchoring reports.
  The three paired effects were reconstructed from the eight model pairs;
  all 256 sign patterns, intervals, and Holm adjustment were checked.

**Geometry primary outcome:** PGC minus Grid log₁₀ sharpness is **−0.448028**,
95% paired interval **[−1.012604, +0.116549]**, exact two-sided **p = 0.110138**.
PGC is lower in 10/16 pairs. The geometric-mean PGC/Grid ratio is 0.356428,
but its apparent reduction is **not a confirmed population difference** under
the fixed primary test. No primary model used the numerical floor or triggered
the radius, median-derivative, or readout-amplitude degeneracy flag. The sole
primary test needs no multiplicity correction. The common-ordinary-initiation
control also has a descriptive interval crossing zero. The final report must
not upgrade the earlier exploratory geometry tendency into a confirmed
mechanism of the functional performance advantage.

**Raw-input architecture/capacity outcome:** Grid/PGC/explicit tracking errors
are **29.07785/21.38372/5.28378°**, with translated-location errors
**29.10426/21.51960/5.28824°**. PGC improves tracking by **7.69413°**
[6.81788, 8.57037]; ordinary replacement increases PGC error by **9.73548°**
[8.53415, 10.93682]; the translated advantage is **7.58466°**
[6.68122, 8.48811]. All three prespecified secondary contrasts have
Holm-adjusted **p = 0.0234375**, with the expected sign in all eight pairs.
The raw models therefore retain a useful predictive-input benefit. Comparing
this cohort with the encoder cohort changes both architecture and parameter
count; it does not isolate nonlinearity or prove a particular module necessary.

The raw static anchoring summaries also preserve the accuracy check. Mean PGC
native local phase gain is 0.09397, but heading error is 25.21590° and coverage
0.84334; Grid native error is 20.20333° and coverage 0.91115. PGC ordinary
replacement gives 64.10575° and coverage 0.32752. These values support partial
input-dependent correction, not uniformly accurate static anchoring merely
because disturbances contract.

The main report's prior precision edits have been adopted: zero angular motion
for fixed designed states, the repository-generator description, explicit
mean/median aggregation for the selected loop, pulse-induced rather than
baseline-confounded heading comparisons, and “rate of change” in the phase
equation. No completed numerical finding requires correction.

### Final integrated-report check

The completed raw and geometry paragraphs were checked in main-report version
`fdc7295259d16e551f223f5ecadbadcccae33d62884ec31cec09115d9891989c`.
The 104-model count is 48 primary + 24 raw-input + 32 independent geometry
models. The added effect sizes, intervals, p values, and architecture/capacity
boundary match the independent audit above. Main-report, methods, and
derivation links all resolve. The completed report is accepted by the renderer
without the draft override, and the revised rate-of-change equation remains
attached to its introduction in the real flowable construction.

One nonblocking threshold nuance was reported: the raw anchoring qualification
count is exactly 0/8 under the fixed coverage threshold of 0.90, but seed 3004
has coverage 0.899988735 (89.9989%), essentially at that boundary. The mean error
and coverage are more informative than treating this near miss as a qualitative
failure. No cutoff or outcome was altered. No final numerical or provenance
defect remains.
