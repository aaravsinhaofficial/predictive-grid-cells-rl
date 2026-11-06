# Completed phase-field and memory-substitution analyses

All 16 PGC phase-field analyses and all 16 matched Grid/PGC/explicit-heading memory-substitution triplets are complete. Every model is retained. These analyses were specified as mechanistic secondary studies; intervals below are descriptive 95% Student-t intervals across training seeds and do not belong to the main corrected primary family.

## A simple restoring field

We fit the blank-input drift independently, then fit one coefficient for the cue-dependent term:

`dφ/dt = f₀(φ) + k sin(cue direction − φ)`.

The fitting positions and phases are separate from the held-out field measurements. Recovery trajectories test held-out positions and initial-memory/cue combinations; their angles are not all held out from field fitting. The coefficient is positive in all 16 seeds, with mean 4.80/s [4.35, 5.25]. The predicted sign is restorative: a memory ahead of the cue is pushed back and a memory behind it is pushed forward.

The one-term model explains 78.6% [74.6, 82.6] of held-out native field variance. Its recovery prediction error is 6.89° [5.57, 8.21], compared with 66.89° [66.35, 67.42] for the independently fitted blank drift alone and 64.92° for a constant prediction. Reversing the cue is predicted by rotating the same cue term by π, without refitting; trajectory prediction error is 9.38° [7.83, 10.94]. The reversal field itself is a permutation of uniformly sampled cue angles, so its matching fit quality is not an independent field validation.

The equation is a useful reduced description, with a clear limit: ordinary replacement is a continuing sensory drive and is not equivalent to blank input. Applying the blank-only model to that intervention gives poor predictions (~82° error). Likewise, self-generated cues need not follow the simple independent-reference model. These failures are retained in the summary rather than hidden by relabeling either intervention as blank input.

[Simple field and held-out recovery figure](../../results/ring_heading/mechanistic_confirmation/phase_reduction.png) · [All-seed numerical summary](../../results/ring_heading/mechanistic_confirmation/summary.json)

## Does correcting memory help the next movement?

We substituted recurrent states elicited by direction cues displaced by ±20°, after the same four initial cues and 40 blank steps. The physical agent, goal, future sensory code and noise were unchanged. The baseline also received 40 blank steps. An equal-norm normal disturbance supplied a comparison. Each model used the same 128 paired episodes and the same fixed navigation controller. The substitution acts on all 64 hidden-state coordinates; where the memory map has a sharp transition, it may also change readout-null components. It should therefore be described as a direction-conditioned state substitution, not universally as an isolated phase-coordinate manipulation.

| Result | Grid | PGC | Explicit heading |
| --- | ---: | ---: | ---: |
| Direction error 2 s after substitution | 21.69° | 4.94° | 2.21° |
| Mean path deviation over first 5 s | 15.58 cm | 2.36 cm | 0.90 cm |
| Equal-norm normal control path deviation | 3.01 cm | 0.56 cm | 0.33 cm |
| Arrival, baseline | 25.05% | 79.83% | 93.31% |
| Arrival, after substitution | 21.39% | 78.52% | 93.36% |
| Added capped arrival time | 1.73 s | 0.52 s | 0.01 s |

The paired Grid-minus-PGC difference in early path deviation is 13.22 cm [12.87, 13.57], and the extra arrival-time penalty is 1.21 s [0.33, 2.09]. The difference between their arrival-rate penalties is less decisive: 2.34 percentage points [−0.20, 4.89]. Thus heading recovery and early steering provide the clearest behavioral bridge. Explicit heading performs best, supporting the value of independent direction information rather than a uniquely superior predictive format.

With the same PGC-trained weights, replacing the predictive spatial rates by ordinary rates raises the post-substitution two-second direction error to 73.82° and early path deviation to 12.31 cm. Native input gives 4.94° and 2.36 cm. Ordinary replacement already impairs the unperturbed baseline, so a small additional pulse penalty under replacement must not be interpreted as successful repair. Restoring predictive input after two seconds improves navigation but does not undo position-estimation error accumulated while the cue was absent.

[Matched population figure](../../results/ring_heading/mechanistic_confirmation/memory_pulse_populations.png) · [Same-network cue intervention figure](../../results/ring_heading/mechanistic_confirmation/memory_pulse.png) · [All population results](../../results/ring_heading/mechanistic_confirmation/population_pulse_summary.json)

## What the mechanism means

The independently supplied spatial direction signal corrects an internal direction estimate, and that correction reduces subsequent steering error. A smooth, transversely attracting full-state loop is demonstrated in the explicitly selected seed2002 example; other models have sharp hidden-state transitions despite circular readouts. Neither the phase-field fit nor these substitutions alone certify a smooth ring in every model. These are supervised heading estimators driving a fixed controller, and the result is kept separate from the original reinforcement-trained LSTM gate experiments.

[Exploratory full-state ring example](../../results/ring_heading/geometry_audit/ring_example_s2002.png) · [Full-state memory audit](heading_full_state_memory_audit.md)
