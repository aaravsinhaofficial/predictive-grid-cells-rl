# LSTM replay with varying movement and remembered goals

This extends the earlier fixed-direction, fixed-goal PCA probe. All eight
matched seed pairs use their frozen 1-billion-frame checkpoints. No policy
training or perturbations are performed.

## What changes

- Movement and facing vary continuously through real ArenaEnv kinematics,
  including the original motor and image noise.
- Sixteen separate episodes have different physical goals and room layouts.
  Each lasts the training-standard 1,350 decisions. The physical goal remains
  fixed within its episode, as in the actual task.
- Goal memory starts empty and updates at every real pickup from its position,
  movement direction, and velocity. The update becomes available on the next
  policy step. The hit position is inside the goal radius, not necessarily at
  the goal center.
- Hidden and cell states start at zero in each episode and persist through
  pickup respawns. Previous rewards use the training reward clip.

Every model receives the same images, positions, movement, actions, and rewards.
Each uses its own population code and corresponding goal snapshots. A scripted
controller visits the goal and two random waypoints in alternation, turning and
moving forward. Its privileged access to the goal ensures repeated memory
updates. Predicted policy actions do not control the replay: this is an
off-policy representation probe, not a behavioral performance evaluation.

## Recorded data

The replay has 21,600 decisions, 155 pickups (8–17 per episode), and 16 distinct
goal centers. We save every fifth decision after step 100 with movement speed
above 0.05 m/s: 3,393 population states per model. All eight movement-angle
sectors are represented, with 359–512 samples each. Saved states cover 153
distinct remembered pickup positions.

There are 3,392 known-goal samples and only one unknown-goal sample. This is
effectively a post-discovery analysis; it cannot compare discovery phases.

- `replay.npz`: full time-major replay metadata and `sample_index`, which maps
  each saved activity row to its observation. `hit` describes the transition
  after that observation; `memory_*` describes the snapshot before it.
- `*_activity.npz`: sampled 256-dimensional `h` and `c`, policy logits/value,
  PCA scores, and the fitted PCA mean, components, and variance spectrum.
- `summary.json`: settings, checkpoint/replay/image hashes, validation errors,
  and per-model PCA statistics. Exact float32 images are shared in RAM and
  reproducible from the script and seed; they are not saved to disk.

## Figures and readouts

[Movement-colored PCA](all_seeds_direction_pca.png) ·
[Remembered-goal-colored PCA](all_seeds_goal_pca.png) ·
[Replay coverage](replay_coverage.png) ·
[Held-out decoding](heldout_decoding.png)

PCA is fitted separately to each model using all 256 hidden coordinates,
mean-centered without neuron-wise scaling. The same PCA coordinates are
colored by movement angle and remembered pickup x position in separate
figures. No UMAP is used. PC axes are not aligned across models.

The first three PCs explain 23.81% of hidden-state variance on average for
grid-only models and 29.42% for PGC+grid models (21.2–33.1% across all models).
These projections display a minority of the variance.

Diagnostic ridge readouts predict current position, remembered pickup position,
and movement direction from `h`. Four outer folds hold out entire episodes;
three inner episode folds choose the regularization strength. All centering
and scaling use training data only. Direction is fitted as cosine/sine and
reported as angular error; position errors are Euclidean RMSE. Goal readouts
exclude the single unknown-goal sample. The reference predicts the outer
training targets' mean. Metrics are descriptive across eight paired training
seeds; no reward-based causal claim is made.

| Held-out readout (lower is better) | Grid only | PGC + grid | Training-mean baseline | PGC lower error |
|---|---:|---:|---:|---:|
| Current position RMSE | 49.46 cm | 48.18 cm | 72.20 cm | 6/8 seeds |
| Remembered pickup position RMSE | 63.19 cm | 56.58 cm | 86.94 cm | 6/8 seeds |
| Movement direction angular MAE | 22.64° | 25.11° | 96.83° | 2/8 seeds |

Numbers are means across the eight training seeds. Paired PGC-minus-grid
differences (mean ± SEM) are −1.28 ± 1.11 cm for current position,
−6.61 ± 3.26 cm for remembered pickup position, and +2.47 ± 1.31° for
direction. The larger descriptive difference is in the goal-snapshot readout;
these exploratory results do not establish a robust or causal explanation of
the PGC reward advantage. Individual-seed values and fold membership are in
`decoding_summary.json`; held-out predictions are in `decoding_predictions.npz`.

Goal code is supplied externally every step, so decoding a goal does not
demonstrate internal memory storage or use. The controller couples facing and
movement direction, and visual inputs can support direction decoding. The
episode split prevents direct leakage of each test episode's room and goal
into training, but does not isolate every source of information.

Pooling goals and directions can combine conditional representations and
increase dimensionality. A single 3D projection does not establish or exclude
a torus. No new persistent-homology or recurrent-attractor claim is made here.

## Reproduction and verification

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/lstm_variable_replay.py
python scripts/plot_lstm_variable.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/decode_lstm_variable.py
```

The recorder checks snapshot updates against the evaluator's online rule,
episode boundaries, motion coverage, goal visits, finite state values, and
checkpoint hashes before/after inference. It independently checks the original
LSTMCell recurrence against nn.LSTM over the sampled sequence. The decoder
checks disjoint whole-episode train/test folds and exactly one held-out
prediction per sample. Earlier fixed-condition outputs remain unchanged.
