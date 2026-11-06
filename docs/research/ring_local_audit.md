# Ring-attractor study: original-model audit

The existing agents offer a strong test of **learned directional stabilization**, but their architecture does not contain a designed ring. The most direct circular quantity supplied by predictive grid cells is travel direction. A new circular navigation task can provide a controlled mechanistic demonstration; evidence from the original square arena should be reported separately.

## What the original networks receive

`code/codes.py` implements 512 analytic spatial rate maps. Ordinary cells receive position `x`. Predictive cells receive `x + 0.20 u`, where `u` is the last displacement direction above 0.05 m/s. Direction persists while nearly stationary and resets to facing at respawn. It is not simply heading: backward and sideways motion can differ from facing. The 169 predictive cells and 343 ordinary cells share the underlying spatial map bank with the paired ordinary-grid agent.

`code/nets.py` concatenates a 256-dimensional visual embedding, current code (512), remembered goal snapshot (512), previous action (6), and previous reward (1). All 256 LSTM units receive this 1287-dimensional input through all four components. The complete recurrent state is `(h,c)`, 512 dimensions. A ring found only in `h` can omit slow cell-state directions.

The goal code is an external snapshot updated on pickup. It is available every step. A circular goal-related pattern under continuous goal-code presentation can therefore be input-driven, even with no internal memory.

The original task is a square arena, with fixed colored walls giving facing information and random episode textures. Position has no circular topology. Candidate circular quantities are travel direction, facing direction, allocentric goal bearing, and egocentric goal bearing. Test them separately. The first two are especially correlated with preferred turning and translation actions; within-state comparisons are essential.

## What is available without new training

- Sixteen frozen 1B-frame networks, eight paired seeds; original policy checkpoints should remain immutable and be hashed.
- Shared replay: `results/lstm_variable_1B/replay.npz`, 16 episodes and 21,600 decisions. Activity files contain 3,393 selected post-input states, each with full h/c. These give identical sensory/motor histories across networks, but trajectories are controlled externally.
- Native snapshots: `results/lstm_navigation_phases_1B/{condition}_s{seed}_action_states.npz`. These contain pre-input h/c, embedded image, current and goal codes, physical pose/velocity, actual goal center, and previous action/reward. Their inputs can be reconstructed exactly without rerendering images. Snapshot counts differ between models because goal-known states are selected.
- `scripts/lstm_local_dynamics_spectra.py` contains an analytic 512-dimensional LSTM Jacobian checked against autograd. This is reusable for slow-point and directional stability analyses. Prior Jacobians were taken along trajectories, so eigenvalues near one there are not by themselves evidence for a ring of equilibria.
- `scripts/lstm_history_input_factorial.py` provides common-history input interventions. `scripts/lstm_code_swap.py` provides paired environments and exact predictive-to-ordinary replacement.

## Fast original-agent pilot

`scripts/lstm_ring_probe.py` runs a deliberately exploratory screen on native snapshot states:

1. Fit cosine/sine readouts of facing, travel direction, and egocentric goal bearing. Report held-out-episode error and one shuffled-label reference. Continuation uses readouts refit to all snapshot states; this distinction is explicit in the output.
2. Continue every snapshot under an identical per-model mean input, and separately under all-zero input. Neither retains a snapshot-specific direction or goal cue. Both create artificial inputs, so these are screens for persistent dynamics under cue withdrawal. Each snapshot's own frozen input provides a driven reference.
3. Perturb full h/c along a decoder-defined phase-changing direction and along a random direction orthogonal to both circular readout axes. Match the initial RMS perturbation after per-coordinate scaling. These are **readout directions**, not established manifold tangents and normals.
4. Measure decoded phase drift, between-state collapse, and perturbation recovery over 1, 5, 15, 30, 60, and 150 recurrent steps (150 steps = 10 s at the trained decision rate).

The comparison is within each frozen model. Native snapshots come from different policies and are not matched physical states between populations. Seed 1 is a runtime and signal pilot, not the unit of a population inference.

## Strong follow-up on original models

Construct actual conditional trajectories covering 24–48 directions at the same location, goal snapshot, room, and motion speed. Change travel direction while holding facing fixed where kinematically possible. Repeat across held-out locations and goals. Fit periodic conditional state curves in a subset of contexts and test held-out directions; include a nonperiodic encoder and shuffled labels as baselines. Varying PGC direction at fixed position already traces a periodic curve in the supplied code, so a loop under that sweep is expected even in a feedforward network.

Use those states to initialize a **common, direction-neutral** delay input. Search for slow states with the complete LSTM update. A candidate ring should cover the represented angle, have low update speed, one weak tangent mode, contraction in transverse directions, and retain different phases after transverse perturbation. Test explicit input-supported persistence separately from cue-free persistence.

Then restore native closed-loop navigation after equal-size state perturbations. A causal account links loss of phase precision/recovery to navigation error, and rescues the effect by restoring predictive influence to specific gates or input projections. Phase-shifting and transverse perturbations should have distinct behavioral effects predicted by the fitted dynamical model.

## Minimal controlled new training study

Use a one-dimensional circular navigation track embedded in 2D. Train an unstructured recurrent agent to reach remembered targets and traverse variable-speed trajectories. Use exactly the same analytic code bank and number of channels for ordinary and predictive conditions. Give both agents the same motor/efference signal, observation masks, noise, target sequence, and training budget. Prefer a dense RNN/GRU/LSTM with no ring-connectivity constraint and no hidden-state circle loss; otherwise the designed architecture must be stated explicitly.

The simplest hypothesis is that predictive-minus-current information approximates a spatial derivative in the direction of movement: `g(x + d u) - g(x) ≈ d J_g(x) u`. That can supply the tangent direction needed to advance an internal position estimate. It predicts better alignment of recurrent updates to the position manifold and less transverse wandering. It also predicts a useful scale range rather than an unrestricted benefit from larger shifts.

Essential controls are ordinary grids plus explicit velocity, a matched extra current-code copy, wrong-sign prediction, phase-scrambled prediction, and predictive-to-ordinary replacement after training. A delayed-sensory task can reveal phase-lead compensation, but that mechanism should be distinguished from attractor stabilization. If velocity and exact current position are already supplied, the predictive code adds a representation of existing information; the interesting finding is how learning uses that representation, not a new source of information.

Train enough independent seeds to estimate the variation between networks. Evaluate novel noise levels, observation dropouts, and speeds with all seeds retained. Geometry, local eigenvalues, perturbation recovery, and closed-loop behavior should agree before calling the learned object a stabilized ring attractor.

## Scope and implementation notes

The current full visual A2C trainer is `code/train.py`; it uses 32 environments and 100-step BPTT in the original 1B runs. Full visual retraining is likely inefficient for a $500 mechanistic study. A vectorized no-render circular task can isolate the mechanism cheaply, leaving budget for seeds and causal interventions. A ring imposed by physical task topology is acceptable; a ring silently imposed on hidden activity and then presented as discovered is not.

The trainer selects CUDA when available and otherwise CPU, so it does not automatically exploit local MPS. New pilot code can choose its device explicitly. Original-state analyses can be CPU-only and reuse saved visual embeddings.
