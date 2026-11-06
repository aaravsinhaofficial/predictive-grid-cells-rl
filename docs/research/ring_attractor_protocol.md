# Predictive inputs and a navigation ring: research protocol

Status: exploratory pilots; written before new training results. Date: 2026-11-05.

## Question

Can predictive spatial input help a trained recurrent navigation system maintain and recover a circular state representation? If so, which part of the input and recurrent computation produces the improvement?

The working mechanism is that a shifted spatial code supplies a movement-dependent change of the spatial population pattern. For a small shift, `g(x + tau*v) - g(x)` approximates `tau * Dg(x) * v`. This makes the direction of movement through the population representation explicit. It could simplify movement along a learned memory manifold, change feedback gains, or compensate sensory delay. These are different hypotheses and will be measured separately.

## Two connected experiments

1. **Existing agents.** Screen the original eight Grid/PGC LSTM pairs for circular variables and slow recurrent dynamics, using full hidden and cell state. Use shared inputs during memory tests so the input itself does not draw a circle. Follow promising findings with controlled trajectories and behavioral perturbations.
2. **Controlled trained model.** Train unconstrained recurrent networks on circular location estimation and use their estimate to navigate to goals. Ordinary and predictive groups receive the same number of periodic spatial channels, the same noisy velocity measurements, identical training examples, and paired initial weights. No ring connectivity, ring geometry loss, or attractor regularizer is supplied. Random cue gaps require recurrent memory. This is a new one-dimensional task, distinct from the original square arena.

## What counts as stabilization

Report these measurements separately:

- **Circular memory:** a continuous range of initial phases remains distinguishable under identical cue-free, zero-motion input. Report angular error, coverage, and drift over time.
- **Recovery toward the ring:** perturb full recurrent state perpendicular to the measured manifold; quantify return at several sizes and times, relative to an unperturbed trajectory.
- **Motion along the ring:** perturb tangent to the manifold; test whether the phase change persists while deviations away from the manifold decay.
- **Noise robustness:** measure phase diffusion and state departure under matched recurrent noise, normalized for representation scale.
- **Navigation:** measure goal success and time under cue gaps, state disturbances, and previously unseen movement statistics.

A lower tracking error or a circular PCA plot alone is not counted as greater attractor stability. Spectra distinguish slow tangent modes, contracting transverse modes, and transient amplification.

## Controls and causal tests

- Same architecture, channel count, spatial basis, input noise, motion information, training budget, and paired seeds.
- Ordinary, predictive, reversed, and temporally shuffled predictive codes. Record the exact distribution changed by each intervention.
- Zero sensory delay and a matched nonzero delay. Separate prediction-assisted delay correction from robustness of autonomous memory.
- Within a frozen predictive network, replace predictive channels by their ordinary counterparts; restore them; reverse their shift; sweep shift magnitude.
- Remove or preserve the component of predictive input drive aligned with the measured ring tangent. Compare equal-norm random-subspace interventions.
- Perturb tangent and normal state directions at matched norms; connect recovery with subsequent navigation.
- Validate dynamical diagnostics on a known ring model and a driven circular representation that collapses after cue removal.

## Analysis discipline

Pilot seeds are separate from confirmatory seeds. Pilot settings may change; every change and pilot result will remain recorded. Final conditions, endpoints, seed count, exclusions, and statistical family will be locked before confirmatory training. Seeds, not timesteps or individual episodes, are the unit of replication. Report paired seed effects, confidence intervals, exact sign-permutation tests, and correction within explicitly named primary comparison families. Keep all valid runs, including unsuccessful training seeds; report learning reliability separately from dynamics conditional on successful training.

Provenance includes source hashes, configurations, checkpoint hashes, software versions, input generation seeds, and evaluation seeds. Independent checks cover code equality at zero shift, information matching, phase/time alignment, Jacobians, perturbation normalization, and held-out evaluation.

## Compute

New GCP spending is capped at $500. Start with a single bounded pilot VM. Reserve a conservative maximum cost before each launch; use cloud-enforced deletion deadlines and auto-deleted disks. Collect outputs and delete completed instances promptly. Keep a dedicated resource prefix and budget ledger; leave unrelated resources untouched. Scale only after measuring actual throughput.

## Interpretation target

The strongest outcome is a reproducible chain: predictive input changes a specific recurrent computation; that computation strengthens a measured ring-memory property; targeted removal weakens both the property and navigation; restoration rescues both. The controlled model provides an explanation of how this can happen. Evidence from the existing square-arena agents determines how broadly that explanation applies to the original performance difference.

## Exploratory amendments, before confirmation

The first circular-position pilot learns a persistent approximate ring in both groups. Within the predictive network, ordinary-code replacement worsens tracking and precise goal holding; it does not weaken average state-disturbance contraction. This motivates separating **accurate anchoring** from **stronger contraction**, rather than treating them as interchangeable. A reduced mixed-cue phase model and a cue-gain sweep test this distinction. Every original pilot outcome is retained.

A small-capacity pilot (16 and 32 units, seed 1003, zero/four-step delay, both input groups, fixed 3,000 updates) tests whether explicit spatial/movement conjunctions reduce recurrent computational demands. These are exploratory architecture comparisons, not confirmatory evidence selected by their sign.

A second controlled arm follows the original spatial rule more closely: hexagonal codes at `x + 0.20 * travel_direction` and a recurrent memory of travel direction. Ordinary grid inputs can infer movement from their changing position code; predictive inputs also contain instantaneous direction through their spatial shift. All models receive the same noisy angular-velocity/speed measurements and initial direction cue. A Grid-plus-explicit-direction control receives the same direction measurement used to construct PGC input. This tests the proposed **spatially indexed compass** mechanism and identifies the contribution of additional direction information. No ring connectivity or hidden-geometry loss is imposed. Stationary periods retain travel direction and contain no in-place rotation, keeping travel direction and facing unambiguous. Pilot seed 1005 is separate from any later confirmation cohort.
