# LSTM hidden activity at 1 billion frames

All eight matched seed pairs were analyzed: eight `gc100` and eight `pgc33`
LSTM + vision checkpoints. No weights were trained or changed.

Open [all seed pairs](all_seeds_pca.png), the [rotatable 3-D viewer](interactive_pca.html),
or individual `seed_01_pca.png` through `seed_08_pca.png` figures. Each individual
figure shows the same PCA coordinates twice: colored by physical x above and
physical y below. PDF copies are provided alongside the PNGs.

The top three PCs capture 50.2–63.5% of hidden-state variance, averaging 56.5%
for grid and 59.6% for PGC + grid. A consistent, clear torus is not apparent in
these views. This is a visual observation, not evidence that a torus is absent:
substantial activity lies outside the displayed projection. No topology or
attractor test was performed. The plotted axes are fitted separately per model
and are not aligned between networks.

## What was replayed

- One 3,079-decision raster sweep using the repository's environment and its
  smoothed movement dynamics. All models receive identical rendered images,
  positions, motion, previous actions, and rewards. Policy actions are ignored.
- Facing is fixed east. Motor noise is disabled; the original image noise
  (0.02) is retained, with the same noisy images replayed to every model.
  One sampled room layout stays fixed throughout.
- The physical goal is fixed at (1.10, 1.10) m. Each model receives its own
  preloaded goal snapshot for that point, with an eastward arrival direction
  and speed 0.35 m/s. The snapshot is held fixed. The path avoids the goal;
  there are no rewards, goal hits, or respawns.
- The full hidden and cell states evolve throughout the replay, initialized
  to zero only once at its start. This diagnostic session intentionally exceeds
  the training episode duration and contains no episode resets.
- PCA uses the same 1,185 samples for every model: eastward motion, within 5°,
  speed 0.30–0.37 m/s, at least 15 decisions into a horizontal leg and at least
  100 decisions after initialization. The actual selected motion directions
  span 0–0.254°. See [spatial coverage](replay_coverage.png).
- PCA is fitted to the 256-dimensional **h** vector, after each current input
  has been processed. Each neuron is mean-centered; neurons are not rescaled.
  Cell state **c** is not included in this PCA.

This is a controlled, off-policy representation probe with a single goal,
room, heading, and path. It samples a rectangle away from the goal, rather than
the whole arena. Results do not establish autonomous navigation dynamics or
generalize to every task condition.

## Saved data and reproduction

`replay.npz` contains the shared states, actions, selection mask, and room/goal
metadata. Each `{condition}_s{seed}_activity.npz` contains all 3,079 hidden vectors,
the selected 3-D scores, the full PCA basis and variance spectrum, and the fixed
goal code. `summary.json` records checkpoint hashes, settings, and per-model
statistics. [Variance figures](pca_variance.png) show the spectrum and seed pairs.

From the repository root:

```sh
python scripts/lstm_pca.py
# Redraw figures from saved activity without repeating inference:
python scripts/lstm_pca.py --figures-only
```

Checks passed for all 16 models: checkpoint metadata at exactly 1B frames,
strict weight loading, finite hidden vectors, correct sample counts, PCA scores
reconstructed from the saved basis, and unchanged checkpoint hashes. Efficient
sequence evaluation was checked against the original LSTMCell for the first
64 steps of each model (maximum absolute difference below 2e-5).
