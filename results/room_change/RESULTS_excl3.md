# Room-change experiment: tables

**Sensitivity analysis: seed(s) 3 left out (both codes).**

grid = 100 % grid cells; PGC = grid + 33 % predictive grid cells; difference = PGC − grid, mean ± SEM over seeds, two-sided paired t-test; "k/n" = seeds in which PGC is higher.

## Before fine-tuning (1 B agents, frozen)

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 96.6 | 105.8 | +9.2 ± 3.8 | 0.054 | 5/7 |
| changed rooms (all 21) | 22.5 | 25.1 | +2.7 ± 3.2 | 0.44 | 5/7 |
| door | 16.7 | 19.0 | +2.3 ± 2.9 | 0.46 | 5/7 |
| obstacle | 45.7 | 51.8 | +6.1 ± 6.4 | 0.38 | 5/7 |
| door + obstacle | 12.1 | 13.6 | +1.4 ± 3.0 | 0.64 | 4/7 |
| goal moved (open room) | 52.7 | 56.5 | +3.9 ± 1.6 | 0.050 | 6/7 |
| door / obstacle and goal moved | 17.9 | 19.9 | +2.0 ± 2.8 | 0.50 | 5/7 |
| new rooms (never trained) | 29.0 | 33.0 | +4.0 ± 3.6 | 0.30 | 5/7 |
| new rooms, goal moved | 20.5 | 22.8 | +2.4 ± 2.7 | 0.42 | 5/7 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.563 | 0.569 | +0.006 ± 0.059 | 0.92 | 3/7 |
| steps to the first goal, episodes that found it | 376 | 350 | -26 ± 37 | 0.51 | 3/7 |
| steps to the first goal (1350 if never found) | 798 | 778 | -19 ± 74 | 0.80 | 3/7 |
| goals per 1000 steps after the first | 2.94 | 3.40 | +0.45 ± 0.21 | 0.072 | 5/7 |
| first find : return, ratio of steps | 2.00 | 2.09 | +0.09 ± 0.12 | 0.50 | 4/7 |
| path efficiency of completed returns | 0.586 | 0.654 | +0.069 ± 0.038 | 0.12 | 6/7 |
| share of steps against a wall | 0.374 | 0.300 | -0.075 ± 0.109 | 0.52 | 2/7 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.878 | 0.859 | -0.019 ± 0.027 | 0.51 | 2/7 |
| steps to the first goal, episodes that found it | 290 | 288 | -2 ± 10 | 0.85 | 2/7 |
| steps to the first goal (1350 if never found) | 420 | 439 | +19 ± 29 | 0.53 | 4/7 |
| goals per 1000 steps after the first | 9.25 | 10.59 | +1.35 ± 0.36 | 0.010 | 7/7 |
| first find : return, ratio of steps | 2.86 | 3.12 | +0.26 ± 0.13 | 0.10 | 6/7 |
| path efficiency of completed returns | 0.698 | 0.776 | +0.078 ± 0.013 | 0.001 | 7/7 |
| share of steps against a wall | 0.006 | 0.005 | -0.001 ± 0.004 | 0.75 | 3/7 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 4.15 | 4.49 | +0.34 ± 0.15 | 0.058 | 5/7 |
| goals after the move, all episodes | 1.12 | 1.16 | +0.05 ± 0.08 | 0.57 | 3/7 |
| episodes that had found the goal before it moved | 0.794 | 0.776 | -0.018 ± 0.027 | 0.54 | 3/7 |
| goals after the move, in those episodes | 0.37 | 0.27 | -0.10 ± 0.14 | 0.50 | 3/7 |
| share of those episodes that found the moved goal | 0.057 | 0.043 | -0.014 ± 0.021 | 0.53 | 3/7 |
| steps from the move to the moved goal, those episodes (675 if never) | 644 | 653 | +8 ± 11 | 0.50 | 4/7 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.90 | 2.96 | +0.06 ± 0.10 | 0.55 | 4/7 |
| time near the old goal until the moved goal is found, share | 0.596 | 0.727 | +0.131 ± 0.041 | 0.018 | 6/7 |
| same for the mirror-image control zone | 0.029 | 0.026 | -0.003 ± 0.006 | 0.60 | 3/7 |
| perseveration index (old-goal zone − control zone) | 0.567 | 0.701 | +0.135 ± 0.039 | 0.014 | 6/7 |
| mean distance to the old goal in that time (m) | 0.496 | 0.421 | -0.074 ± 0.043 | 0.14 | 2/7 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.24 | 1.39 | +0.15 ± 0.25 | 0.57 | 5/7 |
| goals after the move, all episodes | 0.55 | 0.60 | +0.05 ± 0.04 | 0.25 | 4/7 |
| episodes that had found the goal before it moved | 0.464 | 0.479 | +0.016 ± 0.065 | 0.82 | 4/7 |
| goals after the move, in those episodes | 0.21 | 0.20 | -0.01 ± 0.06 | 0.88 | 3/7 |
| share of those episodes that found the moved goal | 0.084 | 0.076 | -0.008 ± 0.024 | 0.76 | 3/7 |
| steps from the move to the moved goal, those episodes (675 if never) | 635 | 639 | +4 ± 11 | 0.73 | 4/7 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.67 | 2.82 | +0.14 ± 0.20 | 0.51 | 5/7 |
| time near the old goal until the moved goal is found, share | 0.280 | 0.326 | +0.046 ± 0.024 | 0.11 | 6/7 |
| same for the mirror-image control zone | 0.051 | 0.051 | +0.000 ± 0.004 | 0.95 | 3/7 |
| perseveration index (old-goal zone − control zone) | 0.229 | 0.275 | +0.046 ± 0.024 | 0.11 | 5/7 |
| mean distance to the old goal in that time (m) | 0.863 | 0.830 | -0.033 ± 0.028 | 0.28 | 2/7 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.51 | 1.67 | +0.17 ± 0.23 | 0.49 | 5/7 |
| goals after the move, all episodes | 0.60 | 0.65 | +0.05 ± 0.03 | 0.16 | 6/7 |
| episodes that had found the goal before it moved | 0.494 | 0.506 | +0.012 ± 0.060 | 0.84 | 4/7 |
| goals after the move, in those episodes | 0.23 | 0.21 | -0.02 ± 0.06 | 0.77 | 3/7 |
| share of those episodes that found the moved goal | 0.081 | 0.073 | -0.008 ± 0.022 | 0.73 | 3/7 |
| steps from the move to the moved goal, those episodes (675 if never) | 636 | 640 | +4 ± 10 | 0.68 | 4/7 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.69 | 2.83 | +0.13 ± 0.19 | 0.51 | 5/7 |
| time near the old goal until the moved goal is found, share | 0.309 | 0.363 | +0.054 ± 0.025 | 0.078 | 6/7 |
| same for the mirror-image control zone | 0.049 | 0.049 | -0.000 ± 0.003 | 0.98 | 2/7 |
| perseveration index (old-goal zone − control zone) | 0.260 | 0.314 | +0.054 ± 0.025 | 0.075 | 6/7 |
| mean distance to the old goal in that time (m) | 0.829 | 0.793 | -0.037 ± 0.028 | 0.24 | 2/7 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.411 | 0.466 | +0.055 ± 0.008 | < 0.001 | 7/7 |
| variance in first 10 components, all steps, familiar room | 0.465 | 0.502 | +0.037 ± 0.009 | 0.007 | 7/7 |
| share of steps with the goal known, familiar room | 0.688 | 0.674 | -0.014 ± 0.021 | 0.53 | 3/7 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.449 | 0.487 | +0.037 ± 0.011 | 0.017 | 6/7 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.543 | 0.555 | +0.013 ± 0.012 | 0.34 | 5/7 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.409 | 0.423 | +0.014 ± 0.055 | 0.80 | 4/7 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 96.6 | 105.8 | +9.2 ± 3.8 | 0.054 | 5/7 |
| D1 | 20.2 | 23.8 | +3.6 ± 3.7 | 0.37 | 5/7 |
| D1+R | 17.1 | 19.4 | +2.3 ± 2.8 | 0.43 | 5/7 |
| D2 | 14.4 | 14.6 | +0.2 ± 2.1 | 0.91 | 4/7 |
| D2+R | 13.5 | 15.0 | +1.5 ± 2.4 | 0.55 | 5/7 |
| D3 | 11.4 | 12.0 | +0.6 ± 2.7 | 0.85 | 5/7 |
| D3+R | 11.8 | 13.1 | +1.3 ± 2.9 | 0.68 | 5/7 |
| D4 | 20.9 | 25.6 | +4.7 ± 3.8 | 0.27 | 5/7 |
| D4+R | 16.3 | 19.6 | +3.3 ± 2.8 | 0.28 | 5/7 |
| O1 | 34.4 | 38.2 | +3.7 ± 9.4 | 0.71 | 3/7 |
| O1+R | 23.6 | 25.0 | +1.4 ± 4.9 | 0.78 | 3/7 |
| O2 | 50.1 | 60.5 | +10.4 ± 6.9 | 0.18 | 5/7 |
| O2+R | 32.0 | 36.1 | +4.0 ± 3.8 | 0.32 | 5/7 |
| O3 | 52.5 | 56.6 | +4.1 ± 5.9 | 0.51 | 5/7 |
| O3+R | 32.6 | 36.5 | +4.0 ± 3.0 | 0.23 | 5/7 |
| DO1 | 12.4 | 14.4 | +2.0 ± 2.6 | 0.47 | 3/7 |
| DO1+R | 11.1 | 12.1 | +0.9 ± 2.7 | 0.74 | 4/7 |
| DO2 | 10.9 | 12.5 | +1.7 ± 3.8 | 0.68 | 5/7 |
| DO2+R | 9.7 | 10.4 | +0.7 ± 3.1 | 0.84 | 4/7 |
| DO3 | 13.1 | 13.8 | +0.7 ± 3.1 | 0.83 | 4/7 |
| DO3+R | 11.4 | 12.0 | +0.6 ± 3.1 | 0.85 | 4/7 |
| open+R | 52.7 | 56.5 | +3.9 ± 1.6 | 0.050 | 6/7 |
| H1 | 19.7 | 19.1 | -0.6 ± 3.3 | 0.86 | 3/7 |
| H1+R | 16.2 | 16.0 | -0.1 ± 3.0 | 0.96 | 4/7 |
| H2 | 53.6 | 63.6 | +10.0 ± 4.7 | 0.080 | 6/7 |
| H2+R | 33.8 | 38.6 | +4.7 ± 2.6 | 0.12 | 5/7 |
| H3 | 13.6 | 16.4 | +2.8 ± 4.4 | 0.55 | 4/7 |
| H3+R | 11.4 | 13.8 | +2.5 ± 3.7 | 0.53 | 4/7 |

## After 250 M frames in the changed rooms

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 62.7 | 73.0 | +10.3 ± 8.7 | 0.28 | 4/7 |
| changed rooms (all 21) | 27.4 | 33.3 | +5.9 ± 5.7 | 0.34 | 5/7 |
| door | 22.7 | 29.2 | +6.6 ± 6.0 | 0.31 | 5/7 |
| obstacle | 48.2 | 58.8 | +10.6 ± 10.2 | 0.34 | 5/7 |
| door + obstacle | 20.0 | 24.8 | +4.7 ± 7.1 | 0.53 | 5/7 |
| goal moved (open room) | 42.8 | 46.9 | +4.0 ± 3.0 | 0.23 | 5/7 |
| door / obstacle and goal moved | 23.8 | 28.5 | +4.7 ± 4.2 | 0.31 | 5/7 |
| new rooms (never trained) | 28.9 | 34.9 | +6.0 ± 6.9 | 0.42 | 6/7 |
| new rooms, goal moved | 22.6 | 25.8 | +3.3 ± 3.9 | 0.44 | 4/7 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.602 | 0.621 | +0.019 ± 0.043 | 0.67 | 3/7 |
| steps to the first goal, episodes that found it | 307 | 284 | -23 ± 27 | 0.43 | 2/7 |
| steps to the first goal (1350 if never found) | 721 | 687 | -34 ± 57 | 0.57 | 2/7 |
| goals per 1000 steps after the first | 3.37 | 4.17 | +0.81 ± 0.71 | 0.30 | 6/7 |
| first find : return, ratio of steps | 1.46 | 1.59 | +0.13 ± 0.07 | 0.13 | 6/7 |
| path efficiency of completed returns | 0.527 | 0.585 | +0.058 ± 0.040 | 0.19 | 5/7 |
| share of steps against a wall | 0.214 | 0.155 | -0.060 ± 0.071 | 0.43 | 1/7 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.797 | 0.793 | -0.004 ± 0.035 | 0.91 | 3/7 |
| steps to the first goal, episodes that found it | 277 | 262 | -15 ± 13 | 0.30 | 3/7 |
| steps to the first goal (1350 if never found) | 495 | 487 | -8 ± 40 | 0.84 | 3/7 |
| goals per 1000 steps after the first | 6.24 | 7.32 | +1.08 ± 0.80 | 0.23 | 4/7 |
| first find : return, ratio of steps | 1.89 | 2.02 | +0.13 ± 0.14 | 0.39 | 4/7 |
| path efficiency of completed returns | 0.546 | 0.606 | +0.060 ± 0.035 | 0.14 | 5/7 |
| share of steps against a wall | 0.007 | 0.007 | -0.000 ± 0.005 | 0.92 | 3/7 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 2.91 | 3.32 | +0.41 ± 0.34 | 0.28 | 5/7 |
| goals after the move, all episodes | 1.37 | 1.37 | -0.00 ± 0.12 | 0.99 | 4/7 |
| episodes that had found the goal before it moved | 0.724 | 0.733 | +0.009 ± 0.038 | 0.83 | 4/7 |
| goals after the move, in those episodes | 0.87 | 0.73 | -0.15 ± 0.16 | 0.39 | 3/7 |
| share of those episodes that found the moved goal | 0.231 | 0.174 | -0.057 ± 0.046 | 0.26 | 3/7 |
| steps from the move to the moved goal, those episodes (675 if never) | 565 | 593 | +29 ± 23 | 0.25 | 3/7 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.65 | 2.88 | +0.23 ± 0.15 | 0.18 | 4/7 |
| time near the old goal until the moved goal is found, share | 0.336 | 0.411 | +0.075 ± 0.050 | 0.18 | 5/7 |
| same for the mirror-image control zone | 0.054 | 0.040 | -0.014 ± 0.013 | 0.33 | 3/7 |
| perseveration index (old-goal zone − control zone) | 0.283 | 0.371 | +0.088 ± 0.058 | 0.18 | 5/7 |
| mean distance to the old goal in that time (m) | 0.733 | 0.662 | -0.071 ± 0.057 | 0.26 | 2/7 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.49 | 1.82 | +0.33 ± 0.34 | 0.37 | 5/7 |
| goals after the move, all episodes | 0.89 | 1.03 | +0.14 ± 0.10 | 0.20 | 5/7 |
| episodes that had found the goal before it moved | 0.532 | 0.560 | +0.028 ± 0.051 | 0.61 | 5/7 |
| goals after the move, in those episodes | 0.52 | 0.55 | +0.04 ± 0.08 | 0.66 | 4/7 |
| share of those episodes that found the moved goal | 0.198 | 0.191 | -0.007 ± 0.040 | 0.87 | 3/7 |
| steps from the move to the moved goal, those episodes (675 if never) | 582 | 585 | +3 ± 17 | 0.86 | 4/7 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.70 | 2.83 | +0.13 ± 0.22 | 0.57 | 4/7 |
| time near the old goal until the moved goal is found, share | 0.228 | 0.276 | +0.048 ± 0.045 | 0.32 | 5/7 |
| same for the mirror-image control zone | 0.069 | 0.059 | -0.009 ± 0.011 | 0.44 | 3/7 |
| perseveration index (old-goal zone − control zone) | 0.160 | 0.217 | +0.057 ± 0.052 | 0.31 | 5/7 |
| mean distance to the old goal in that time (m) | 0.934 | 0.894 | -0.040 ± 0.054 | 0.48 | 2/7 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.62 | 1.96 | +0.34 ± 0.34 | 0.36 | 5/7 |
| goals after the move, all episodes | 0.93 | 1.06 | +0.13 ± 0.09 | 0.20 | 4/7 |
| episodes that had found the goal before it moved | 0.550 | 0.576 | +0.026 ± 0.050 | 0.62 | 5/7 |
| goals after the move, in those episodes | 0.55 | 0.57 | +0.02 ± 0.08 | 0.82 | 4/7 |
| share of those episodes that found the moved goal | 0.201 | 0.189 | -0.012 ± 0.040 | 0.78 | 3/7 |
| steps from the move to the moved goal, those episodes (675 if never) | 581 | 586 | +5 ± 17 | 0.77 | 4/7 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.70 | 2.84 | +0.14 ± 0.20 | 0.52 | 4/7 |
| time near the old goal until the moved goal is found, share | 0.238 | 0.289 | +0.051 ± 0.044 | 0.30 | 5/7 |
| same for the mirror-image control zone | 0.067 | 0.058 | -0.010 ± 0.011 | 0.42 | 3/7 |
| perseveration index (old-goal zone − control zone) | 0.171 | 0.231 | +0.060 ± 0.051 | 0.29 | 5/7 |
| mean distance to the old goal in that time (m) | 0.916 | 0.873 | -0.043 ± 0.053 | 0.45 | 2/7 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.403 | 0.436 | +0.033 ± 0.009 | 0.010 | 6/7 |
| variance in first 10 components, all steps, familiar room | 0.464 | 0.480 | +0.017 ± 0.009 | 0.12 | 6/7 |
| share of steps with the goal known, familiar room | 0.633 | 0.639 | +0.006 ± 0.030 | 0.84 | 4/7 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.442 | 0.466 | +0.025 ± 0.010 | 0.044 | 5/7 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.525 | 0.529 | +0.004 ± 0.014 | 0.79 | 4/7 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.466 | 0.491 | +0.025 ± 0.042 | 0.57 | 5/7 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 62.7 | 73.0 | +10.3 ± 8.7 | 0.28 | 4/7 |
| D1 | 19.9 | 25.5 | +5.6 ± 3.6 | 0.17 | 5/7 |
| D1+R | 20.2 | 25.2 | +5.0 ± 3.5 | 0.21 | 5/7 |
| D2 | 24.5 | 30.0 | +5.6 ± 7.3 | 0.47 | 4/7 |
| D2+R | 21.5 | 27.1 | +5.6 ± 4.0 | 0.22 | 5/7 |
| D3 | 22.2 | 29.5 | +7.2 ± 7.1 | 0.35 | 4/7 |
| D3+R | 21.6 | 26.8 | +5.2 ± 5.1 | 0.35 | 4/7 |
| D4 | 24.0 | 32.0 | +8.0 ± 7.8 | 0.35 | 4/7 |
| D4+R | 20.7 | 26.2 | +5.5 ± 5.5 | 0.36 | 4/7 |
| O1 | 47.8 | 58.5 | +10.7 ± 12.0 | 0.41 | 5/7 |
| O1+R | 35.1 | 39.6 | +4.4 ± 5.2 | 0.43 | 5/7 |
| O2 | 49.9 | 60.9 | +11.0 ± 9.5 | 0.29 | 4/7 |
| O2+R | 34.6 | 41.3 | +6.7 ± 4.2 | 0.16 | 4/7 |
| O3 | 46.8 | 56.9 | +10.1 ± 10.7 | 0.38 | 5/7 |
| O3+R | 33.4 | 37.4 | +4.0 ± 4.5 | 0.41 | 5/7 |
| DO1 | 19.5 | 24.2 | +4.7 ± 6.7 | 0.51 | 4/7 |
| DO1+R | 17.2 | 20.5 | +3.3 ± 4.5 | 0.49 | 4/7 |
| DO2 | 20.6 | 25.8 | +5.3 ± 7.7 | 0.52 | 4/7 |
| DO2+R | 17.6 | 20.6 | +3.1 ± 4.7 | 0.54 | 4/7 |
| DO3 | 20.1 | 24.3 | +4.2 ± 8.3 | 0.63 | 4/7 |
| DO3+R | 16.2 | 20.6 | +4.5 ± 5.4 | 0.44 | 5/7 |
| open+R | 42.8 | 46.9 | +4.0 ± 3.0 | 0.23 | 5/7 |
| H1 | 22.8 | 22.0 | -0.8 ± 6.4 | 0.90 | 5/7 |
| H1+R | 18.6 | 18.8 | +0.2 ± 4.4 | 0.96 | 4/7 |
| H2 | 45.5 | 56.3 | +10.8 ± 8.2 | 0.24 | 6/7 |
| H2+R | 32.5 | 37.7 | +5.2 ± 4.1 | 0.25 | 5/7 |
| H3 | 18.3 | 26.4 | +8.1 ± 7.5 | 0.32 | 5/7 |
| H3+R | 16.5 | 21.0 | +4.4 ± 4.4 | 0.36 | 5/7 |

## After 250 M more frames in the familiar room (seeds 1–2)

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 103.1 | 105.3 | +2.2 ± 3.1 | 0.61 | 1/2 |
| changed rooms (all 21) | 28.5 | 24.6 | -3.8 ± 9.8 | 0.76 | 1/2 |
| door | 21.4 | 16.5 | -4.9 ± 8.7 | 0.67 | 1/2 |
| obstacle | 56.1 | 53.5 | -2.6 ± 16.6 | 0.90 | 1/2 |
| door + obstacle | 18.9 | 13.5 | -5.5 ± 9.1 | 0.66 | 1/2 |
| goal moved (open room) | 54.8 | 58.6 | +3.9 ± 4.0 | 0.51 | 1/2 |
| door / obstacle and goal moved | 23.2 | 19.1 | -4.1 ± 8.9 | 0.73 | 1/2 |
| new rooms (never trained) | 35.1 | 29.7 | -5.4 ± 9.6 | 0.67 | 1/2 |
| new rooms, goal moved | 24.9 | 21.0 | -3.9 ± 7.3 | 0.68 | 1/2 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.604 | 0.544 | -0.060 ± 0.104 | 0.67 | 1/2 |
| steps to the first goal, episodes that found it | 294 | 390 | +96 ± 87 | 0.47 | 2/2 |
| steps to the first goal (1350 if never found) | 712 | 825 | +112 ± 156 | 0.60 | 1/2 |
| goals per 1000 steps after the first | 3.47 | 3.48 | +0.01 ± 0.76 | 1.00 | 1/2 |
| first find : return, ratio of steps | 1.73 | 2.29 | +0.56 ± 0.30 | 0.31 | 2/2 |
| path efficiency of completed returns | 0.652 | 0.626 | -0.026 ± 0.059 | 0.73 | 1/2 |
| share of steps against a wall | 0.191 | 0.453 | +0.262 ± 0.185 | 0.39 | 2/2 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.845 | 0.807 | -0.038 ± 0.003 | 0.042 | 0/2 |
| steps to the first goal, episodes that found it | 277 | 288 | +11 ± 1 | 0.045 | 2/2 |
| steps to the first goal (1350 if never found) | 443 | 492 | +49 ± 4 | 0.051 | 2/2 |
| goals per 1000 steps after the first | 10.14 | 11.13 | +0.98 ± 0.55 | 0.32 | 2/2 |
| first find : return, ratio of steps | 2.99 | 3.31 | +0.32 ± 0.11 | 0.21 | 2/2 |
| path efficiency of completed returns | 0.749 | 0.784 | +0.035 ± 0.003 | 0.057 | 2/2 |
| share of steps against a wall | 0.002 | 0.005 | +0.003 ± 0.003 | 0.45 | 2/2 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 4.45 | 4.44 | -0.00 ± 0.04 | 0.96 | 1/2 |
| goals after the move, all episodes | 1.03 | 1.42 | +0.39 ± 0.35 | 0.47 | 2/2 |
| episodes that had found the goal before it moved | 0.775 | 0.723 | -0.052 ± 0.008 | 0.090 | 0/2 |
| goals after the move, in those episodes | 0.14 | 0.24 | +0.10 ± 0.02 | 0.11 | 2/2 |
| share of those episodes that found the moved goal | 0.025 | 0.038 | +0.013 ± 0.010 | 0.42 | 2/2 |
| steps from the move to the moved goal, those episodes (675 if never) | 662 | 658 | -4 ± 4 | 0.53 | 1/2 |
| reversal cost: steps to the moved goal / steps to the first goal | 3.11 | 3.13 | +0.02 ± 0.07 | 0.81 | 1/2 |
| time near the old goal until the moved goal is found, share | 0.696 | 0.751 | +0.055 ± 0.043 | 0.42 | 2/2 |
| same for the mirror-image control zone | 0.014 | 0.033 | +0.020 ± 0.004 | 0.14 | 2/2 |
| perseveration index (old-goal zone − control zone) | 0.682 | 0.718 | +0.036 ± 0.047 | 0.59 | 1/2 |
| mean distance to the old goal in that time (m) | 0.413 | 0.396 | -0.017 ± 0.023 | 0.60 | 1/2 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.69 | 1.31 | -0.38 ± 0.70 | 0.68 | 1/2 |
| goals after the move, all episodes | 0.63 | 0.60 | -0.02 ± 0.19 | 0.93 | 1/2 |
| episodes that had found the goal before it moved | 0.540 | 0.443 | -0.097 ± 0.139 | 0.61 | 1/2 |
| goals after the move, in those episodes | 0.12 | 0.21 | +0.09 ± 0.04 | 0.26 | 2/2 |
| share of those episodes that found the moved goal | 0.040 | 0.079 | +0.039 ± 0.012 | 0.19 | 2/2 |
| steps from the move to the moved goal, those episodes (675 if never) | 656 | 637 | -19 ± 5 | 0.17 | 0/2 |
| reversal cost: steps to the moved goal / steps to the first goal | 3.07 | 2.57 | -0.50 ± 0.36 | 0.40 | 0/2 |
| time near the old goal until the moved goal is found, share | 0.329 | 0.345 | +0.016 ± 0.024 | 0.63 | 1/2 |
| same for the mirror-image control zone | 0.041 | 0.052 | +0.010 ± 0.006 | 0.32 | 2/2 |
| perseveration index (old-goal zone − control zone) | 0.288 | 0.294 | +0.005 ± 0.018 | 0.82 | 1/2 |
| mean distance to the old goal in that time (m) | 0.791 | 0.811 | +0.020 ± 0.032 | 0.65 | 1/2 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.94 | 1.59 | -0.35 ± 0.64 | 0.68 | 1/2 |
| goals after the move, all episodes | 0.66 | 0.68 | +0.02 ± 0.21 | 0.95 | 1/2 |
| episodes that had found the goal before it moved | 0.562 | 0.469 | -0.093 ± 0.127 | 0.60 | 1/2 |
| goals after the move, in those episodes | 0.12 | 0.21 | +0.09 ± 0.03 | 0.22 | 2/2 |
| share of those episodes that found the moved goal | 0.039 | 0.076 | +0.037 ± 0.010 | 0.17 | 2/2 |
| steps from the move to the moved goal, those episodes (675 if never) | 656 | 639 | -18 ± 4 | 0.15 | 0/2 |
| reversal cost: steps to the moved goal / steps to the first goal | 3.07 | 2.62 | -0.45 ± 0.32 | 0.39 | 0/2 |
| time near the old goal until the moved goal is found, share | 0.363 | 0.382 | +0.019 ± 0.018 | 0.47 | 2/2 |
| same for the mirror-image control zone | 0.039 | 0.050 | +0.011 ± 0.006 | 0.29 | 2/2 |
| perseveration index (old-goal zone − control zone) | 0.324 | 0.332 | +0.008 ± 0.012 | 0.63 | 1/2 |
| mean distance to the old goal in that time (m) | 0.756 | 0.773 | +0.017 ± 0.027 | 0.65 | 1/2 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.428 | 0.478 | +0.049 ± 0.013 | 0.16 | 2/2 |
| variance in first 10 components, all steps, familiar room | 0.483 | 0.514 | +0.032 ± 0.011 | 0.21 | 2/2 |
| share of steps with the goal known, familiar room | 0.671 | 0.635 | -0.036 ± 0.003 | 0.051 | 0/2 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.470 | 0.486 | +0.016 ± 0.013 | 0.44 | 2/2 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.554 | 0.557 | +0.003 ± 0.001 | 0.13 | 2/2 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.472 | 0.389 | -0.083 ± 0.116 | 0.60 | 1/2 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 103.1 | 105.3 | +2.2 ± 3.1 | 0.61 | 1/2 |
| D1 | 26.5 | 21.8 | -4.8 ± 6.9 | 0.62 | 1/2 |
| D1+R | 21.3 | 17.1 | -4.1 ± 6.8 | 0.65 | 1/2 |
| D2 | 18.8 | 16.2 | -2.6 ± 12.4 | 0.87 | 1/2 |
| D2+R | 17.8 | 15.4 | -2.4 ± 11.6 | 0.87 | 1/2 |
| D3 | 15.7 | 8.4 | -7.4 ± 9.0 | 0.56 | 1/2 |
| D3+R | 15.3 | 9.1 | -6.2 ± 6.4 | 0.51 | 1/2 |
| D4 | 24.7 | 19.6 | -5.0 ± 6.5 | 0.58 | 1/2 |
| D4+R | 20.9 | 16.5 | -4.4 ± 4.2 | 0.49 | 0/2 |
| O1 | 47.5 | 32.6 | -14.9 ± 20.1 | 0.59 | 1/2 |
| O1+R | 33.1 | 23.3 | -9.8 ± 11.9 | 0.56 | 1/2 |
| O2 | 59.6 | 66.6 | +7.0 ± 14.7 | 0.72 | 1/2 |
| O2+R | 36.9 | 41.0 | +4.1 ± 12.2 | 0.80 | 1/2 |
| O3 | 61.1 | 61.1 | +0.0 ± 15.1 | 1.00 | 1/2 |
| O3+R | 37.6 | 37.5 | -0.0 ± 10.0 | 1.00 | 1/2 |
| DO1 | 19.8 | 13.9 | -5.9 ± 6.6 | 0.54 | 1/2 |
| DO1+R | 17.4 | 11.6 | -5.8 ± 7.7 | 0.59 | 1/2 |
| DO2 | 17.3 | 10.8 | -6.5 ± 8.3 | 0.58 | 1/2 |
| DO2+R | 15.4 | 8.3 | -7.1 ± 7.6 | 0.53 | 1/2 |
| DO3 | 19.7 | 15.7 | -4.0 ± 12.5 | 0.80 | 1/2 |
| DO3+R | 16.5 | 11.7 | -4.8 ± 10.1 | 0.72 | 1/2 |
| open+R | 54.8 | 58.6 | +3.9 ± 4.0 | 0.51 | 1/2 |
| H1 | 25.5 | 18.4 | -7.1 ± 7.5 | 0.52 | 1/2 |
| H1+R | 20.6 | 14.5 | -6.1 ± 6.0 | 0.50 | 0/2 |
| H2 | 64.7 | 56.0 | -8.7 ± 11.3 | 0.58 | 1/2 |
| H2+R | 39.0 | 36.2 | -2.7 ± 5.4 | 0.70 | 1/2 |
| H3 | 15.2 | 14.6 | -0.6 ± 9.8 | 0.96 | 1/2 |
| H3+R | 15.2 | 12.2 | -3.0 ± 10.4 | 0.82 | 1/2 |

## PCA with the earlier protocol (one scripted trajectory replayed through every network)

| agents | replay in | quantity | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|---|---|
| 1 B | familiar room | variance in first 10 components | 0.4653 | 0.5235 | +0.0582 ± 0.0044 | < 0.001 | 7/7 |
| 1 B | familiar room | participation ratio | 31.12 | 23.98 | -7.14 ± 1.27 | 0.001 | 0/7 |
| 1 B | familiar room | total variance of the 256 outputs | 19.68 | 24.81 | +5.13 ± 0.99 | 0.002 | 6/7 |
| 1 B | changed rooms | variance in first 10 components | 0.4697 | 0.5250 | +0.0553 ± 0.0062 | < 0.001 | 7/7 |
| 1 B | changed rooms | participation ratio | 30.29 | 24.60 | -5.69 ± 1.25 | 0.004 | 0/7 |
| 1 B | changed rooms | total variance of the 256 outputs | 19.25 | 24.07 | +4.83 ± 1.07 | 0.004 | 6/7 |
| 1.25 B, changed rooms | familiar room | variance in first 10 components | 0.4613 | 0.4923 | +0.0310 ± 0.0091 | 0.014 | 7/7 |
| 1.25 B, changed rooms | familiar room | participation ratio | 31.86 | 28.12 | -3.75 ± 1.52 | 0.049 | 0/7 |
| 1.25 B, changed rooms | familiar room | total variance of the 256 outputs | 17.81 | 22.64 | +4.83 ± 2.01 | 0.053 | 6/7 |
| 1.25 B, changed rooms | changed rooms | variance in first 10 components | 0.4740 | 0.5024 | +0.0284 ± 0.0112 | 0.045 | 6/7 |
| 1.25 B, changed rooms | changed rooms | participation ratio | 29.97 | 26.38 | -3.58 ± 2.22 | 0.16 | 1/7 |
| 1.25 B, changed rooms | changed rooms | total variance of the 256 outputs | 17.67 | 22.07 | +4.40 ± 2.04 | 0.074 | 6/7 |
| 1.25 B, familiar room (seeds 1–2) | familiar room | variance in first 10 components | 0.4635 | 0.5298 | +0.0662 ± 0.0140 | 0.13 | 2/2 |
| 1.25 B, familiar room (seeds 1–2) | familiar room | participation ratio | 30.52 | 23.42 | -7.10 ± 1.02 | 0.090 | 0/2 |
| 1.25 B, familiar room (seeds 1–2) | familiar room | total variance of the 256 outputs | 20.25 | 23.52 | +3.27 ± 2.60 | 0.43 | 2/2 |
| 1.25 B, familiar room (seeds 1–2) | changed rooms | variance in first 10 components | 0.4780 | 0.5315 | +0.0535 ± 0.0081 | 0.095 | 2/2 |
| 1.25 B, familiar room (seeds 1–2) | changed rooms | participation ratio | 28.86 | 24.04 | -4.82 ± 1.64 | 0.21 | 0/2 |
| 1.25 B, familiar room (seeds 1–2) | changed rooms | total variance of the 256 outputs | 19.99 | 23.29 | +3.30 ± 2.33 | 0.39 | 2/2 |

Change of each agent's first-10 share against its own 1 B value:

| agents | replay in | grid | PGC | PGC − grid | p |
|---|---|---|---|---|---|
| 1.25 B, changed rooms | familiar | -0.0040 ± 0.0086 | -0.0312 ± 0.0052 | -0.0272 ± 0.0070 | 0.008 |
| 1.25 B, changed rooms | changed | +0.0044 ± 0.0112 | -0.0225 ± 0.0073 | -0.0269 ± 0.0108 | 0.046 |
| 1.25 B, familiar room (seeds 1–2) | familiar | +0.0031 ± 0.0028 | +0.0244 ± 0.0087 | +0.0213 ± 0.0115 | 0.32 |
| 1.25 B, familiar room (seeds 1–2) | changed | +0.0059 ± 0.0041 | +0.0165 ± 0.0018 | +0.0105 ± 0.0023 | 0.14 |

## Change over the first 250 M frames in the changed rooms (after − before), per code

| rooms | statistic | grid | PGC | PGC − grid | p |
|---|---|---|---|---|---|
| familiar room | score | -34.0 ± 2.5 | -32.8 ± 8.2 | +1.1 ± 8.3 | 0.90 |
| familiar room | path efficiency of completed returns | -0.152 ± 0.020 | -0.170 ± 0.024 | -0.018 ± 0.026 | 0.53 |
| familiar room | goals per 1000 steps after the first | -3.00 ± 0.42 | -3.27 ± 0.84 | -0.27 ± 0.87 | 0.77 |
| changed rooms (all 21) | score | +4.9 ± 1.6 | +8.2 ± 3.2 | +3.2 ± 3.3 | 0.36 |
| changed rooms, goal fixed (10) | score | +5.5 ± 2.2 | +9.5 ± 4.1 | +4.1 ± 4.4 | 0.39 |
| changed rooms, goal fixed (10) | path efficiency of completed returns | -0.059 ± 0.010 | -0.069 ± 0.012 | -0.010 ± 0.015 | 0.53 |
| changed rooms, goal fixed (10) | goals per 1000 steps after the first | +0.43 ± 0.32 | +0.78 ± 0.53 | +0.35 ± 0.63 | 0.60 |
| changed rooms, goal fixed (10) | share of steps against a wall | -0.160 ± 0.041 | -0.145 ± 0.039 | +0.015 ± 0.051 | 0.78 |
| changed rooms, goal fixed (10) | steps to the first goal (1350 if never found) | -77 ± 32 | -91 ± 32 | -14 ± 57 | 0.81 |
| door | score | +5.9 ± 1.6 | +10.2 ± 3.3 | +4.3 ± 3.9 | 0.31 |
| obstacle | score | +2.5 ± 2.8 | +7.0 ± 6.0 | +4.5 ± 5.8 | 0.47 |
| door + obstacle | score | +7.9 ± 3.2 | +11.2 ± 4.0 | +3.3 ± 4.7 | 0.51 |
| goal moved (open room) | goals after the move, in those episodes | +0.50 ± 0.18 | +0.46 ± 0.13 | -0.04 ± 0.25 | 0.87 |
| goal moved (open room) | share of those episodes that found the moved goal | +0.174 ± 0.053 | +0.131 ± 0.034 | -0.043 ± 0.059 | 0.49 |
| goal moved (open room) | perseveration index (old-goal zone − control zone) | -0.284 ± 0.029 | -0.330 ± 0.047 | -0.046 ± 0.052 | 0.40 |
| rooms with a goal move (11) | goals after the move, in those episodes | +0.32 ± 0.06 | +0.36 ± 0.05 | +0.04 ± 0.07 | 0.58 |
| rooms with a goal move (11) | share of those episodes that found the moved goal | +0.120 ± 0.023 | +0.116 ± 0.016 | -0.004 ± 0.028 | 0.90 |
| rooms with a goal move (11) | perseveration index (old-goal zone − control zone) | -0.089 ± 0.017 | -0.083 ± 0.034 | +0.006 ± 0.037 | 0.87 |
| new rooms (never trained) | score | -0.1 ± 2.1 | +1.9 ± 3.3 | +2.0 ± 4.0 | 0.64 |

## Confirmatory contrasts (pre-declared; PGC − grid; exact sign-flip p over 8 seeds, Holm-adjusted)

| contrast | estimate | 95 % CI | p (t) | p (sign-flip) | p (Holm) | k/n |
|---|---|---|---|---|---|---|
| score gap, changed rooms, before (frozen 1 B agents) | +2.7 | ± 7.8 | 0.44 | 0.42 | 1.00 | 5/7 |
| score gap, changed rooms, after 250 M frames | +5.9 | ± 13.9 | 0.34 | 0.31 | 1.00 | 5/7 |
| change of the score gap in the changed rooms with fine-tuning | +3.2 | ± 8.0 | 0.36 | 0.36 | 1.00 | 5/7 |
| change of the familiar-room score with fine-tuning (after − before) | +1.1 | ± 20.3 | 0.90 | 0.92 | 1.00 | 4/7 |
| first-10 variance share, familiar-room replay, after 250 M frames | +0.0310 | ± 0.0222 | 0.014 | 0.016 | 0.094 | 7/7 |
| first-10 variance share, changed-room replay, after 250 M frames | +0.0284 | ± 0.0275 | 0.045 | 0.031 | 0.16 | 6/7 |

Training score in the changed rooms, last 25 M window of the planned 250 M minus the first: grid +4.4, PGC +8.2, difference +3.7 ± 1.9 (p 0.098, 6/7 seeds)

PGC − grid lead in the familiar room minus the lead in the changed rooms: before +6.5 ± 3.4 (p 0.10); after +4.4 ± 5.2 (p 0.43); same +6.1 ± 6.6 (p 0.53)

## Transfer table: After 250 M frames in the changed rooms

| quantity | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| jumpstart: changed-room score before fine-tuning / familiar-room score | 0.232 | 0.238 | +0.007 ± 0.026 | 0.81 | 4/7 |
| gain: changed-room score, later − before | 4.9 | 8.2 | +3.2 ± 3.3 | 0.36 | 5/7 |
| backward transfer: familiar-room score, later − before | -34.0 | -32.8 | +1.1 ± 8.3 | 0.90 | 4/7 |
| changed-room score after / familiar-room score before | 0.281 | 0.314 | +0.033 ± 0.051 | 0.54 | 4/7 |

Gain in the changed rooms against the change in the familiar room, Spearman over agents: gc100 rho -0.25 (p 0.59, n = 7); pgc33 rho +0.43 (p 0.34, n = 7); pooled rho +0.22 (p 0.45, n = 14)

Control agents (After 250 M more frames in the familiar room (seeds 1–2)): gc100_s1 familiar +0.0, changed -4.1; gc100_s2 familiar -0.5, changed +3.7; pgc33_s1 familiar +8.0, changed +4.6; pgc33_s2 familiar +1.6, changed +2.9

## Time to threshold during fine-tuning (frames, M; training score in the changed rooms, 25 M running mean)

| run | familiar-room training score before | best changed-room score | 50 % reached at | 90 % reached at |
|---|---|---|---|---|
| gc100room_lstm_s1 | 94.1 | 22.0 | not reached | not reached |
| gc100room_lstm_s2 | 108.4 | 45.0 | not reached | not reached |
| gc100room_lstm_s4 | 103.2 | 28.3 | not reached | not reached |
| gc100room_lstm_s5 | 91.7 | 23.2 | not reached | not reached |
| gc100room_lstm_s6 | 100.9 | 23.5 | not reached | not reached |
| gc100room_lstm_s7 | 85.3 | 21.9 | not reached | not reached |
| gc100room_lstm_s8 | 92.4 | 23.6 | not reached | not reached |
| pgc33room_lstm_s1 | 104.6 | 24.0 | not reached | not reached |
| pgc33room_lstm_s2 | 108.2 | 29.4 | not reached | not reached |
| pgc33room_lstm_s4 | 118.8 | 53.0 | not reached | not reached |
| pgc33room_lstm_s5 | 108.9 | 28.5 | not reached | not reached |
| pgc33room_lstm_s6 | 119.0 | 37.1 | not reached | not reached |
| pgc33room_lstm_s7 | 114.5 | 47.4 | not reached | not reached |
| pgc33room_lstm_s8 | 98.2 | 23.0 | not reached | not reached |

## Alignment index of top-10 subspaces (scripted replay; 1 = same subspace)

| agents | comparison | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|---|
| before | familiar vs changed | 0.855 | 0.866 | +0.010 ± 0.009 | 0.30 | 5/7 |
| after | familiar vs changed | 0.855 | 0.859 | +0.004 ± 0.006 | 0.56 | 5/7 |
| after | before vs after familiar | 0.526 | 0.564 | +0.039 ± 0.031 | 0.26 | 6/7 |
| after | before vs after changed | 0.520 | 0.548 | +0.028 ± 0.032 | 0.42 | 5/7 |
| same | familiar vs changed | 0.857 | 0.844 | -0.014 ± 0.022 | 0.66 | 1/2 |
| same | before vs same familiar | 0.709 | 0.742 | +0.032 ± 0.006 | 0.11 | 2/2 |
| same | before vs same changed | 0.709 | 0.707 | -0.002 ± 0.019 | 0.93 | 1/2 |

## Variance concentration at 1 B against transfer (Spearman, description only)

| fine-tuning | agents | n | first-10 share vs jumpstart | first-10 share vs gain |
|---|---|---|---|---|
| after | gc100 | 7 | rho -0.46 (p 0.29) | rho +0.79 (p 0.036) |
| after | pgc33 | 7 | rho +0.39 (p 0.38) | rho +0.64 (p 0.12) |
| after | pooled | 14 | rho +0.02 (p 0.93) | rho +0.47 (p 0.091) |

## Training score in the changed rooms during fine-tuning, per 25 M frames

| frames (M) | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| 1000–1025 | 21.7 | 25.2 | +3.5 ± 4.9 | 0.50 | 5/7 |
| 1025–1050 | 22.1 | 26.5 | +4.4 ± 4.7 | 0.39 | 4/7 |
| 1050–1075 | 22.4 | 28.1 | +5.7 ± 4.8 | 0.28 | 5/7 |
| 1075–1100 | 23.8 | 29.4 | +5.6 ± 5.5 | 0.35 | 4/7 |
| 1100–1125 | 23.1 | 30.7 | +7.6 ± 5.4 | 0.21 | 6/7 |
| 1125–1150 | 23.4 | 32.0 | +8.5 ± 5.0 | 0.14 | 6/7 |
| 1150–1175 | 23.1 | 31.5 | +8.5 ± 5.3 | 0.16 | 6/7 |
| 1175–1200 | 25.1 | 32.5 | +7.4 ± 5.3 | 0.21 | 5/7 |
| 1200–1225 | 25.8 | 33.2 | +7.3 ± 5.5 | 0.23 | 5/7 |
| 1225–1250 | 26.1 | 33.4 | +7.3 ± 5.7 | 0.25 | 5/7 |

## Score as a share of the same agent's familiar-room score at 1 B

| state | rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|---|
| before | changed rooms (all 21) | 0.232 | 0.238 | +0.007 ± 0.026 | 0.81 | 4/7 |
| before | door | 0.171 | 0.180 | +0.009 ± 0.023 | 0.72 | 5/7 |
| before | obstacle | 0.471 | 0.489 | +0.018 ± 0.055 | 0.75 | 4/7 |
| before | door + obstacle | 0.123 | 0.128 | +0.005 ± 0.026 | 0.85 | 3/7 |
| before | goal moved (open room) | 0.548 | 0.536 | -0.012 ± 0.013 | 0.41 | 3/7 |
| before | new rooms (never trained) | 0.299 | 0.312 | +0.013 ± 0.030 | 0.68 | 4/7 |
| after | changed rooms (all 21) | 0.281 | 0.314 | +0.033 ± 0.051 | 0.54 | 4/7 |
| after | door | 0.232 | 0.276 | +0.044 ± 0.054 | 0.45 | 4/7 |
| after | obstacle | 0.491 | 0.555 | +0.064 ± 0.090 | 0.51 | 4/7 |
| after | door + obstacle | 0.201 | 0.231 | +0.030 ± 0.065 | 0.66 | 4/7 |
| after | goal moved (open room) | 0.445 | 0.446 | +0.001 ± 0.029 | 0.98 | 4/7 |
| after | new rooms (never trained) | 0.294 | 0.330 | +0.037 ± 0.060 | 0.56 | 3/7 |
| after | familiar room | 0.642 | 0.695 | +0.053 ± 0.079 | 0.53 | 4/7 |

## Lesions in the grid + PGC agents (score)

| state | rooms | intact | 169 PGCs silenced | 169 random grid cells silenced | PGC-specific cost (random − PGC) | p | n |
|---|---|---|---|---|---|---|---|
| before | familiar room | 105.8 | 80.6 | 100.5 | +20.0 ± 3.7 | 0.002 | 7 |
| before | changed rooms (all 21) | 25.1 | 22.6 | 25.9 | +3.2 ± 0.6 | 0.002 | 7 |
| after | familiar room | 73.0 | 60.2 | 68.7 | +8.5 ± 1.7 | 0.003 | 7 |
| after | changed rooms (all 21) | 33.3 | 31.1 | 33.0 | +1.9 ± 0.5 | 0.013 | 7 |

Change of the PGC-specific cost, after − before: familiar room -11.4 ± 3.9 (p 0.026, lower in 6/7 agents); changed rooms (all 21) -1.3 ± 0.6 (p 0.080, lower in 5/7 agents)

## Matched-budget reference (seeds 1–2): score

| agent | state | familiar room | changed rooms | new rooms |
|---|---|---|---|---|
| gc100_s1 | 1 B | 94.0 | 23.3 | 28.9 |
| gc100_s1 | 1.25 B, all in the familiar room | 94.1 | 19.2 | 24.5 |
| gc100_s1 | 1.25 B, last 250 M in the changed rooms | 64.5 | 21.2 | 23.0 |
| gc100_s2 | 1 B | 112.5 | 34.0 | 41.5 |
| gc100_s2 | 1.25 B, all in the familiar room | 112.0 | 37.7 | 45.7 |
| gc100_s2 | 1.25 B, last 250 M in the changed rooms | 82.6 | 45.5 | 51.7 |
| pgc33_s1 | 1 B | 91.5 | 20.6 | 24.7 |
| pgc33_s1 | 1.25 B, all in the familiar room | 99.5 | 25.1 | 28.6 |
| pgc33_s1 | 1.25 B, last 250 M in the changed rooms | 73.7 | 23.7 | 28.1 |
| pgc33_s2 | 1 B | 109.6 | 21.2 | 31.7 |
| pgc33_s2 | 1.25 B, all in the familiar room | 111.2 | 24.1 | 30.7 |
| pgc33_s2 | 1.25 B, last 250 M in the changed rooms | 61.0 | 28.1 | 25.0 |
