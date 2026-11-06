# Predictive input and a learned ring: mechanistic study design

Status: proposed analysis protocol and implemented pilot task, 5 November 2026. This file contains hypotheses and design decisions, not experimental conclusions.

The useful question is whether predictive spatial input helps a learned recurrent network maintain and update a circular state, and whether this improves navigation. A circular-looking PCA plot is insufficient: the network should recover from disturbances back toward the circular state set, while preserving different positions around that set.

## Two connected experiments

1. **Existing navigation LSTMs.** Test the frozen eight paired 1B-frame models for a conditional heading or goal-bearing ring. Hold other physical and task variables fixed, then identify candidate state families. Continue every family member under the same neutral input and perturb the full `(h,c)` state. This asks whether the earlier navigation advantage involves a comparable recurrent mechanism. If a ring exists only while angle-specific inputs are continuously supplied, describe the observed object as an input-driven representation.
2. **Controlled trained model.** Train a small, unconstrained recurrent estimator on circular navigation. Give it noisy, intermittent periodic location codes and noisy velocity. Compare ordinary codes with a mixed population whose predictive channels encode a short extrapolation. The network's learned position estimate drives a goal-seeking controller. This allows controlled experiments on input timing, recurrent dynamics, and behavior with no imposed circular connectivity or hidden-state ring loss.

The controlled model is a one-dimensional periodic grid-code analogue. It is distinct from the original two-dimensional hexagonal-grid task. Supervised position estimation followed by a fixed goal-seeking controller is a trained navigation system, but should not be described as reward-only reinforcement learning.

## Why this is a plausible but nontrivial hypothesis

Optimizing RNNs for angular integration can yield compass-like and velocity-sensitive units with interpretable connectivity. [Cueva et al., 2020](https://arxiv.org/abs/1912.10189).

An approximate continuous attractor can retain a slow circular manifold even when exact continuous fixed points are broken into discrete equilibria and connecting flows. The operational signature is fast recovery across the manifold together with slow flow along it. [Ságodi et al., 2024](https://arxiv.org/html/2408.00109v2).

Perturbation and persistence are central experimental tests: in flies, an experimentally imposed heading pattern persisted and followed naturalistic dynamics. [Kim et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28473639/).

Recent work already relates prospective movement signals to attractor stability. A January 2026 preprint reports recovery of head-direction population dynamics before locomotion in restrained mice and models the effect through lateral connectivity changes. This is closely related background, not a discovery attributable to the present project. [Pak et al., 2026, preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC12871166/).

A separate reviewed preprint models the generation of predictive grid activity through HCN-dependent intrinsic dynamics and circuit asymmetry. Our proposed question is downstream: how an independently trained recurrent consumer uses a predictive spatial code. [Shaikh and Assisi, 2025, reviewed preprint](https://elifesciences.org/reviewed-preprints/108142v1).

## Implemented controlled task

Files: [model and generator](../../scripts/ring_task_model.py), [trainer](../../scripts/train_ring_task.py).

| Component | Initial pilot setting |
| --- | --- |
| Physical variable | Circular position, radians |
| Time step | 0.05 seconds |
| State update | 64-unit unconstrained leaky tanh RNN; leak 0.25 |
| Location input | 64 cosine channels in frequency modules 2, 3, 5, 7 |
| Predictive subset | 21 of 64 channels, distributed across modules |
| Predictive rule | Measured phase + 0.20 seconds × measured velocity |
| Input timing | Both populations receive observations delayed by four steps; zero-delay replication is planned |
| Other inputs | Identical noisy current velocity and cue-availability flag |
| Sensory uncertainty | Shared phase error SD 0.10 rad; per-channel error SD 0.04; velocity error SD 0.05 rad/s |
| Sensory availability | 25% nominal availability, contiguous moving occlusions, and blank stationary tails |
| Training task | Predict sine and cosine of current physical position |
| Training loss | Output MSE plus generic 0.0001 hidden-activity cost |
| Recurrent noise | Gaussian preactivation noise, SD 0.01 |
| Initial state | Zero |

At timestep `t`, physical position is the position **after** the movement represented by `velocity[t]`. Thus velocity integration and the target are aligned. An observation acquired at `max(0,t-delay_steps)` contains that time's noisy phase and velocity. Prediction uses only those measurements. It never uses an actual future position, future velocity, goal, or future action. Every population receives the same complete noisy velocity stream.

The four input conditions are:

- **Grid:** all channels encode the observed current/past position.
- **Predictive:** the designated subset encodes a forward extrapolation.
- **Reverse:** the same subset extrapolates backward by the same amount.
- **Shuffled:** the subset uses another concurrently simulated trajectory's velocity, preserving velocity distribution and temporal structure while breaking its relationship to position.

Channel count, periodic basis, phases, architecture, parameter count, initialization, trajectories, visibility, and random noise are matched within a training seed. The Grid group already contains explicit velocity; a predictive benefit therefore cannot be attributed simply to giving one group access to movement information. The extra structure is the representation of position and velocity together in the input code.

Stationary blank intervals require the system to retain a continuous variable internally. The target specifies only the output variable; it does not specify hidden coordinates, neuron tuning, recurrence, eigenvalues, or topology. There is no ring regularizer and no hand-wired bump.

## Ring tests before any claim of stabilization

Use a dense set of 128–256 initial phases, preferably reached through several different histories. Initialize from valid observations and continue without location cues, with zero movement and identical other inputs. Analyze both the full hidden state and the task readout.

1. **Coverage and memory.** The states retain an ordered, closed family covering the circle over training-length delays and longer held-out delays. A single stable point cannot pass by having a circular readout before the cue disappears.
2. **Autonomous geometry.** Measure nearest-neighbor ordering and topology in the full state space after the shared blank continuation. PCA is a visualization; topology and distances should not depend on choosing two favorable components.
3. **Normal recovery.** Perturb states in directions perpendicular to the local ring tangent. Measure distance to the nearest point anywhere on the reference ring. Recovery toward the original point would mix normal stability with phase restoration and is not the right metric.
4. **Tangent persistence.** Perturb along the ring and observe whether the new phase persists. A stable ring should suppress departures across the ring while allowing displacement around it. A permanent cue can instead pin a unique phase.
5. **Slow flow.** Quantify phase drift over 1, 2, 5, and 10 seconds. Approximate rings may have discrete slow equilibria; the relevant question is how much analog information survives on behavioral timescales.
6. **Spectra and finite-time dynamics.** Compute full state-update Jacobians, tangent alignment of the slow mode, and normal contraction. Confirm with finite perturbations and products of Jacobians along the trajectory. A single near-unit eigenvalue, largest singular value, or PCA spectrum is insufficient by itself.

The fixed/slow-point search can minimize `||F(h,0)-h||²` from many initial conditions, followed by local linearization. This follows a standard trained-RNN analysis strategy. [Sussillo and Barak, 2013](https://barak.net.technion.ac.il/files/2012/11/sussillo_barak-neco.pdf).

## Distinguish three meanings of “more stable”

**Manifold stability:** faster return after a perturbation across the ring, a wider basin of return, or fewer collapses. This concerns the shape of the recurrent representation.

**Phase accuracy:** less random diffusion, drift, bias, or tracking lag around the ring. This concerns the variable represented by the ring. A ring can be structurally stable while its phase is inaccurate.

**Closed-loop behavior:** better goal capture and recovery after disturbances, with fewer oscillations or overshoots. This concerns the coupled agent–environment system.

These endpoints must be reported separately. Forecasting can improve one without improving the others.

For example, consider the reduced phase equation

`dφ/dt = aω + K sin(θ + δ − φ)`.

At constant speed, the equilibrium phase error is `e* = asin((1−a)ω/K) − δ`. A forward offset `δ` can remove the lag. But the local phase contraction is `K cos(e* + δ)`, which is unchanged by `δ` when the other parameters and speed are fixed. This elementary derivation predicts a useful forecasting effect without stronger intrinsic contraction. It is a diagnostic control, not a fitted explanation of the trained networks.

## Causal interventions

Freeze every trained model and intervene both before and after it reaches the candidate ring.

- Remove the predictive shift while retaining every channel and its basis. Compare with sign reversal and matched velocity shuffling.
- Vary forecast horizon continuously, including negative, zero, training, and excessive values. Test whether an optimum shifts when sensory delay or speed changes.
- Freeze the current input and perturb only the recurrent state. Repeat after a common history and at matched represented phases.
- Swap predictive input for ordinary input during a cue-free delay: there should be no immediate effect because no code is present. Any persistent training difference then arises from the learned network and its preceding state.
- Project predictive input effects into local tangent and normal components. Reinject the measured component, with matched-norm random directions as controls. A selective loss and rescue of recovery provides stronger evidence than a weight-norm association.
- Perturb the inferred slow mode and normal modes separately, then test predicted behavioral errors. A phase perturbation should produce a directional navigation error; a normal perturbation should produce a transient impairment that decays at the measured recovery rate.

Avoid directly swapping recurrent weight blocks across independently trained networks: hidden coordinates can differ, making failure uninterpretable. Use within-network interventions or align and validate the coordinate map first.

## Behavioral evaluation

Use the learned sine/cosine readout to estimate circular position and drive a fixed, identical goal-seeking controller. Select unseen start–goal pairs and random movement/noise realizations. Include ordinary conditions, cue outages longer than training, movement reversals, altered velocity gain, and matched internal perturbations. Report all trials, including timeouts.

Primary behavioral endpoint: mean goal-arrival time capped at a fixed timeout. Report success rate separately. The controller must use decoded phase, not true phase, except when the simulator calculates reward, termination, or an explicitly labeled oracle benchmark.

A second-stage learned policy or transfer to the existing 2-D navigation task would strengthen the connection to RL. It is not needed to test whether the first learned state estimator has a ring and whether perturbing that ring changes a navigation agent's behavior.

## Confirmation and statistical plan

Pilot seeds 1001–1002 are for feasibility and selecting fixed task settings. Exclude them from confirmation. Use at least 12 paired, independently trained confirmatory seeds per retained condition, preferably 16 if runtime is as small as the first CPU benchmark indicates. Use fresh evaluation seeds and many trajectories within each network; networks are the independent units.

Predefine three primary paired comparisons for Predictive versus Grid at the chosen training setting:

1. Normal perturbation recovery area under the curve over the first second.
2. Circular phase error after a fixed two-second cue-free delay with matched internal noise.
3. Capped goal-arrival time under the corresponding perturbation/occlusion condition.

Use paired differences and seed-level 95% intervals, plus exact paired sign-flip tests with Holm correction for these three endpoints. Report effect sizes and all seed points. Do not count repeated timesteps or angles as independent samples. Run a separate planned no-delay versus delay interaction to distinguish a delay-compensation benefit from a general change in memory dynamics.

Architecture sizes, longer delays, alternate sensor noise, and additional forecast horizons are generalization tests. Report all tested settings and whether their selection was pilot-informed. Preserve failed training runs and their exclusion rules; convergence should be assessed by a common task-performance rule, not by whether a ring or favorable population effect appears.

## What would make the mechanism convincing

The strongest package would combine: learned circular memory in an unconstrained network; a predictable change in its tangent/normal dynamics from predictive input; a targeted loss and rescue of that change; a quantitative reduced model that predicts held-out perturbations; improved navigation under those same conditions; and a bridge to the original 2-D agents. A ring plot or a collection of significant performance differences would be substantially weaker.

The resulting explanation can be precise even if it is narrower than “prediction stabilizes everything”: for example, prediction may reduce delay-induced phase errors while learned recurrence supplies normal attraction. If actual normal attraction improves as well, the mode-specific perturbations can identify how the two effects are coupled.
