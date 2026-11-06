# Exploratory mechanism: predictive spatial code as a compass

Design fixed before the first successful heading-arm training run, 5 November 2026. This is a separate experiment from the earlier circular-position model. All earlier results remain part of the study.

**Question:** can a predictive grid code help a learned heading integrator correct its accumulated errors, and can that correction stabilize a useful circular memory?

An ordinary grid code supplies current location. A mixed code also contains activity at a point ahead along the travel direction. The relation between those two spatial patterns can supply a direction signal. This is a hypothesis about how an RNN uses the input information; it does not assume that the RNN itself generates predictions.

The fixed-distance rule follows the direction-dependent spatial displacement reported for predictive grid cells. [Ouchi and Fujisawa, 2024](https://fujisawalab.riken.jp/publications/Ouchi2024.pdf). Grid-derived movement direction also has prior theoretical motivation: Zutshi and colleagues proposed that spatial or theta-compressed grid-cell sequences could supply a direction signal to downstream populations. The distinctive question here is whether comparing current and predictive population patterns can provide such a signal and causally anchor a separately learned recurrent memory. [Zutshi, Leutgeb and Leutgeb, 2017](https://pubmed.ncbi.nlm.nih.gov/29333481/).

## Task and controls

The agent moves forward on an open two-dimensional plane. Its heading is its direction of travel. Speed and angular velocity vary stochastically. When it stops, it does not turn in place, so retained travel direction remains equal to heading. This removes the heading-versus-strafing ambiguity in the original six-action navigation task.

All groups receive the same initial four-step heading cue, noisy angular-velocity stream, noisy forward-speed stream, and intermittent spatial observations. Later observation gaps and stationary blank periods require internal memory.

The spatial population uses the original repository's analytic hexagonal GridPopulation formula, with 128 cells and four modules. For the mixed condition, 43 cells use the original fixed-distance rule `g(x + 0.20*u)`; the other 85 remain ordinary. The shift persists during stopping because `u` is retained travel direction. Cell basis, phases, predictive subset, architecture, random initialization, physical trajectories, and random-noise draws are paired within a seed.

| Group | Spatial input | Additional heading information after initialization |
| --- | --- | --- |
| Grid | All 128 ordinary cells | None |
| PGC | 85 ordinary +43 shifted cells | Direction is embedded in the mixed spatial code |
| Grid + explicit heading | All 128 ordinary cells | The exact same measured direction used to generate PGCs, whenever a spatial cue is available |
| Reverse diagnostic | Same 43 cells shifted backward | Reversed spatial direction |
| Shuffled diagnostic | Same 43 cells shifted using another trajectory's direction | Matched marginal direction statistics with broken correspondence |

All groups have the same input width, including two heading channels and their availability flag. These channels supply the initial cue for everyone and are subsequently zero for groups without explicit heading. Thus the explicit control changes available information without adding network capacity.

The PGC population and explicit-heading control both use a direction measurement with independent Gaussian angular noise, SD0.05rad. Neither receives a noiseless compass unavailable to its information-matched comparator. The common angular-velocity sensor has a per-episode bias with SD0.10rad/s and independent white noise with SD0.30rad/s. Forward-speed measurement noise has SD0.01m/s, position measurement noise SD0.01m, and each spatial channel has additive noise SD0.03 before clipping to its firing-rate range.

## Model and training

The network is a 64-unit unconstrained leaky tanh RNN, with a learned linear sine/cosine heading readout. The leak is0.25 per0.05s timestep. The training objective is heading estimation, plus a generic small activity penalty. Hidden-state noise SD0.01 is applied during training. No ring connectivity, ring coordinates, neuron tuning, eigenvalue target, or attractor regularizer is supplied.

The first pilot seed is1005. Grid, PGC, and Grid+explicit-heading are all run for the same fixed6000 updates, with every group's3000-update checkpoint also retained. These settings are exploratory feasibility choices; checkpoints will not be selected by whether the PGC comparison is favorable. Subsequent confirmation must use independent training seeds and frozen endpoints.

Source files: [model and generator](../../scripts/ring_heading_model.py), [trainer](../../scripts/train_ring_heading.py).

## Required mechanistic distinctions

1. **A learned ring during missing cues.** Initialize different headings, remove heading and spatial observations, set angular velocity to zero, and test whether the full state retains a slow circular family. Normal perturbations should relax while tangent perturbations preserve their displaced phase.
2. **Cue-based correction of ring phase.** Disturb the represented heading while holding physical experience fixed, then return spatial cues. PGCs should provide a phase correction if they function as a compass. Ordinary grids can additionally infer direction from subsequent movement, so compare stationary and moving recovery separately.
3. **Information versus encoding.** Grid+explicit-heading asks how much of any benefit comes from providing a direction signal. Similar or stronger rescue with this control would support the compass-information mechanism. It would not imply that predictive coding uniquely supplies it.
4. **Input versus learned recurrence.** Removing the predictive shift from a frozen PGC model keeps all channels and weights. Restoration, reversal, and shuffled direction test whether the correctly oriented signal drives recovery. Cue-free comparisons between separately trained groups test persistent learned dynamics.
5. **Stability versus accuracy.** Measure normal recovery, phase drift/diffusion, and task accuracy separately. A compass cue can correct phase on an already-stable ring without strengthening its autonomous normal attraction.

The original models use different movement actions and a different behavioral objective. Therefore this arm demonstrates a possible, controlled downstream mechanism using the same spatial-code rule; a positive result alone would not identify that mechanism in every original navigation LSTM.

## Sensory-decoder diagnostic and a matched encoded arm

A separate static diagnostic asks whether the proposed compass information is accessible at all. Position and heading are sampled independently; a feedforward decoder receives only the spatial code. It receives no velocity, prior state, or initial heading. Both Grid and PGC decoders have two hidden layers of 256 units and train for 4000 updates with the same sampling and initialization protocol.

For exploratory seed 1005, the PGC decoder achieves 3.11° mean angular error on new samples and 4.69° at positions translated 10 m away. The ordinary Grid decoder remains at chance, 89.57° and 89.93°. This result identifies an instantaneous directional signal in the mixed spatial code. It does not by itself show recurrent stabilization.

Because the raw-input RNN extracts this signal much less accurately, a new, matched architecture arm adds two 256-wide SiLU layers before the 64-unit recurrent input drive. The encoder is learned jointly from the same final heading objective; the static decoder's weights are **not** reused, and no additional supervision is supplied to PGC models. Every condition has the same 121,026 parameters. The initial pilot seed is 1006, distinct from the raw-input pilot. Every group retains fixed 3000- and 6000-update checkpoints.

The implementation uses `encoder_size=0` by default to preserve the raw-input architecture and execution path. `--encoder-size 256` selects the new arm. Encoding is batched over all times and episodes before the recurrent loop. The ordinary `model.step` interface remains available for causal state perturbations, and it reproduces the batched forward calculation exactly in the initial validation.

The encoded arm tests whether an adequate learned sensory transformation lets the same recurrent memory use the compass signal. Architecture comparisons across different pilot seeds remain exploratory; confirmatory population comparisons must use paired seeds within the chosen architecture.

## Transfer evaluation

[The transfer evaluator](../../scripts/evaluate_ring_heading_transfer.py) uses new 256-step physical trajectories, twice the training episode length. It tests the original spatial region, a 10 m translation, doubled sensor noise, and their combination. The translated physical trajectories remain outside the conservative maximum position extent of the 128-step training generator. Thus this is a concrete spatial transfer test, not merely a new random sample from the same starting region.

Native, ordinary, reversed, shuffled, and restored inputs use the same underlying experiences and noise. Restoration occurs at step 64. Whole-sequence errors include the earlier removal period; later errors are also reported separately. No recurrent noise is added during these evaluations, so sensor uncertainty and input interventions can be assessed without a second changing factor.

## Navigation bridge without an oracle controller

[The navigation evaluator](../../scripts/evaluate_ring_heading_navigation.py) combines the trained heading estimator with a fixed controller. The controller knows the initial reset position and the goal. It maintains its own position estimate by integrating **measured speed along the RNN's decoded heading**. It steers toward the goal using that position estimate. The same gains and speed policy apply to every population.

The actual position and heading are used only to simulate motion, generate sensory observations, and score outcomes. Reaching the real goal does not modify the controller or stop the simulation. An independent test changes the scoring radius from 1 mm to 100 m and confirms that every simulated trajectory is unchanged. Thus scoring cannot silently supply the controller with information about its true position.

Episodes last 40 seconds, with both 20- and 40-second outcomes reported as nested prefixes. Timeouts remain in mean arrival times. The goal-capture radius is 15 cm. Every group experiences the same three-second spatial-cue outage from 4 to 7 seconds and the same motor and sensor noise. The controller either moves forward or stops with angular velocity zero, preserving the task's direction-of-travel interpretation.

Five exploratory scenarios test baseline performance, doubled angular-sensor bias, a paired ±20° misleading initial heading cue, physical start and goal positions translated 10 m, and their combination. The misleading cue changes only the initial-heading channels. Predictive spatial cells continue to encode the true, noisy direction of travel; the resulting decoded initial error is recorded because those truthful cells may already help correct the conflicting cue.

Native, ordinary-input, reversed-input, and restored-input branches begin from identical native warmup within a frozen model. Restoration occurs after two seconds. A late direction correction need not fully repair navigation: the controller may already have accumulated an error in its external position estimate. Direction recovery and position recovery are therefore reported as separate mechanisms.
