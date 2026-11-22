# Room-change experiment: tables

grid = 100 % grid cells; PGC = grid + 33 % predictive grid cells; difference = PGC − grid, mean ± SEM over seeds, two-sided paired t-test; "k/n" = seeds in which PGC is higher.

## Before fine-tuning (1 B agents, frozen)

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 96.9 | 106.1 | +9.3 ± 3.3 | 0.027 | 6/8 |
| changed rooms (all 21) | 21.5 | 24.4 | +3.0 ± 2.8 | 0.32 | 6/8 |
| door | 15.7 | 17.9 | +2.2 ± 2.5 | 0.41 | 6/8 |
| obstacle | 43.8 | 51.0 | +7.2 ± 5.7 | 0.25 | 6/8 |
| door + obstacle | 11.5 | 13.2 | +1.7 ± 2.6 | 0.52 | 5/8 |
| goal moved (open room) | 52.6 | 56.7 | +4.1 ± 1.4 | 0.021 | 7/8 |
| door / obstacle and goal moved | 16.9 | 19.2 | +2.3 ± 2.4 | 0.38 | 6/8 |
| new rooms (never trained) | 27.8 | 32.3 | +4.6 ± 3.1 | 0.19 | 6/8 |
| new rooms, goal moved | 19.5 | 22.1 | +2.6 ± 2.4 | 0.30 | 6/8 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.548 | 0.557 | +0.009 ± 0.051 | 0.86 | 4/8 |
| steps to the first goal, episodes that found it | 391 | 358 | -33 ± 33 | 0.35 | 3/8 |
| steps to the first goal (1350 if never found) | 820 | 794 | -26 ± 64 | 0.70 | 3/8 |
| goals per 1000 steps after the first | 2.86 | 3.35 | +0.49 ± 0.18 | 0.033 | 6/8 |
| first find : return, ratio of steps | 2.06 | 2.14 | +0.08 ± 0.11 | 0.50 | 4/8 |
| path efficiency of completed returns | 0.585 | 0.651 | +0.066 ± 0.033 | 0.082 | 7/8 |
| share of steps against a wall | 0.415 | 0.341 | -0.073 ± 0.094 | 0.46 | 2/8 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.871 | 0.857 | -0.014 ± 0.024 | 0.59 | 3/8 |
| steps to the first goal, episodes that found it | 286 | 286 | -1 ± 9 | 0.93 | 3/8 |
| steps to the first goal (1350 if never found) | 424 | 438 | +14 ± 25 | 0.60 | 4/8 |
| goals per 1000 steps after the first | 9.34 | 10.63 | +1.29 ± 0.32 | 0.005 | 8/8 |
| first find : return, ratio of steps | 2.83 | 3.11 | +0.28 ± 0.12 | 0.048 | 7/8 |
| path efficiency of completed returns | 0.701 | 0.778 | +0.078 ± 0.012 | < 0.001 | 8/8 |
| share of steps against a wall | 0.006 | 0.005 | -0.001 ± 0.003 | 0.81 | 4/8 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 4.17 | 4.50 | +0.34 ± 0.13 | 0.031 | 6/8 |
| goals after the move, all episodes | 1.09 | 1.16 | +0.07 ± 0.07 | 0.35 | 4/8 |
| episodes that had found the goal before it moved | 0.791 | 0.779 | -0.011 ± 0.024 | 0.66 | 4/8 |
| goals after the move, in those episodes | 0.33 | 0.28 | -0.06 ± 0.13 | 0.67 | 4/8 |
| share of those episodes that found the moved goal | 0.051 | 0.043 | -0.008 ± 0.019 | 0.68 | 4/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 647 | 652 | +5 ± 10 | 0.64 | 4/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.93 | 2.96 | +0.03 ± 0.09 | 0.76 | 4/8 |
| time near the old goal until the moved goal is found, share | 0.611 | 0.737 | +0.125 ± 0.036 | 0.010 | 7/8 |
| same for the mirror-image control zone | 0.029 | 0.025 | -0.005 ± 0.006 | 0.42 | 3/8 |
| perseveration index (old-goal zone − control zone) | 0.582 | 0.712 | +0.130 ± 0.034 | 0.007 | 7/8 |
| mean distance to the old goal in that time (m) | 0.484 | 0.418 | -0.066 ± 0.038 | 0.13 | 2/8 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.17 | 1.34 | +0.17 ± 0.22 | 0.46 | 6/8 |
| goals after the move, all episodes | 0.52 | 0.58 | +0.06 ± 0.04 | 0.13 | 5/8 |
| episodes that had found the goal before it moved | 0.443 | 0.465 | +0.022 ± 0.057 | 0.71 | 5/8 |
| goals after the move, in those episodes | 0.20 | 0.21 | +0.00 ± 0.05 | 0.94 | 4/8 |
| share of those episodes that found the moved goal | 0.083 | 0.078 | -0.005 ± 0.021 | 0.81 | 4/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 635 | 638 | +3 ± 9 | 0.74 | 4/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.69 | 2.80 | +0.12 ± 0.18 | 0.54 | 5/8 |
| time near the old goal until the moved goal is found, share | 0.277 | 0.327 | +0.051 ± 0.022 | 0.052 | 7/8 |
| same for the mirror-image control zone | 0.050 | 0.050 | -0.000 ± 0.003 | 0.93 | 3/8 |
| perseveration index (old-goal zone − control zone) | 0.226 | 0.277 | +0.051 ± 0.022 | 0.052 | 6/8 |
| mean distance to the old goal in that time (m) | 0.873 | 0.832 | -0.041 ± 0.025 | 0.15 | 2/8 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.44 | 1.63 | +0.18 ± 0.20 | 0.39 | 6/8 |
| goals after the move, all episodes | 0.57 | 0.63 | +0.06 ± 0.03 | 0.074 | 7/8 |
| episodes that had found the goal before it moved | 0.474 | 0.493 | +0.019 ± 0.052 | 0.73 | 5/8 |
| goals after the move, in those episodes | 0.22 | 0.21 | -0.00 ± 0.05 | 0.98 | 4/8 |
| share of those episodes that found the moved goal | 0.080 | 0.074 | -0.006 ± 0.019 | 0.78 | 4/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 636 | 640 | +3 ± 9 | 0.71 | 4/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.71 | 2.82 | +0.11 ± 0.17 | 0.54 | 5/8 |
| time near the old goal until the moved goal is found, share | 0.307 | 0.365 | +0.057 ± 0.022 | 0.036 | 7/8 |
| same for the mirror-image control zone | 0.048 | 0.048 | -0.001 ± 0.003 | 0.82 | 2/8 |
| perseveration index (old-goal zone − control zone) | 0.259 | 0.317 | +0.058 ± 0.022 | 0.034 | 7/8 |
| mean distance to the old goal in that time (m) | 0.838 | 0.794 | -0.044 ± 0.025 | 0.13 | 2/8 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.411 | 0.460 | +0.048 ± 0.010 | 0.002 | 7/8 |
| variance in first 10 components, all steps, familiar room | 0.467 | 0.499 | +0.032 ± 0.009 | 0.011 | 7/8 |
| share of steps with the goal known, familiar room | 0.686 | 0.675 | -0.010 ± 0.019 | 0.60 | 4/8 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.448 | 0.483 | +0.035 ± 0.010 | 0.010 | 7/8 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.544 | 0.555 | +0.011 ± 0.011 | 0.32 | 6/8 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.392 | 0.411 | +0.019 ± 0.048 | 0.70 | 5/8 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 96.9 | 106.1 | +9.3 ± 3.3 | 0.027 | 6/8 |
| D1 | 18.9 | 22.5 | +3.6 ± 3.2 | 0.30 | 6/8 |
| D1+R | 16.0 | 18.2 | +2.2 ± 2.4 | 0.39 | 6/8 |
| D2 | 13.5 | 13.7 | +0.2 ± 1.8 | 0.91 | 4/8 |
| D2+R | 12.7 | 14.0 | +1.3 ± 2.1 | 0.55 | 5/8 |
| D3 | 10.7 | 11.6 | +0.8 ± 2.4 | 0.73 | 6/8 |
| D3+R | 11.0 | 12.6 | +1.6 ± 2.6 | 0.54 | 6/8 |
| D4 | 19.8 | 23.9 | +4.0 ± 3.4 | 0.27 | 5/8 |
| D4+R | 15.4 | 18.4 | +3.0 ± 2.5 | 0.26 | 6/8 |
| O1 | 32.8 | 37.7 | +4.9 ± 8.2 | 0.57 | 4/8 |
| O1+R | 22.5 | 25.0 | +2.5 ± 4.3 | 0.59 | 4/8 |
| O2 | 47.6 | 60.0 | +12.4 ± 6.3 | 0.088 | 6/8 |
| O2+R | 30.2 | 35.6 | +5.4 ± 3.5 | 0.17 | 6/8 |
| O3 | 51.0 | 55.2 | +4.3 ± 5.1 | 0.43 | 6/8 |
| O3+R | 31.7 | 35.3 | +3.6 ± 2.6 | 0.22 | 6/8 |
| DO1 | 11.6 | 14.1 | +2.5 ± 2.3 | 0.32 | 4/8 |
| DO1+R | 10.3 | 11.6 | +1.4 ± 2.3 | 0.57 | 5/8 |
| DO2 | 10.2 | 12.3 | +2.1 ± 3.3 | 0.55 | 6/8 |
| DO2+R | 9.0 | 10.3 | +1.2 ± 2.8 | 0.67 | 5/8 |
| DO3 | 12.6 | 13.2 | +0.7 ± 2.7 | 0.82 | 5/8 |
| DO3+R | 10.7 | 11.4 | +0.7 ± 2.7 | 0.80 | 5/8 |
| open+R | 52.6 | 56.7 | +4.1 ± 1.4 | 0.021 | 7/8 |
| H1 | 18.6 | 19.1 | +0.5 ± 3.1 | 0.87 | 4/8 |
| H1+R | 15.2 | 15.6 | +0.4 ± 2.6 | 0.88 | 5/8 |
| H2 | 51.8 | 62.0 | +10.2 ± 4.1 | 0.042 | 7/8 |
| H2+R | 32.6 | 37.7 | +5.0 ± 2.3 | 0.063 | 6/8 |
| H3 | 12.8 | 15.8 | +3.0 ± 3.8 | 0.46 | 5/8 |
| H3+R | 10.6 | 13.1 | +2.5 ± 3.2 | 0.46 | 5/8 |

## After 250 M frames in the changed rooms

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 60.7 | 69.4 | +8.7 ± 7.7 | 0.29 | 4/8 |
| changed rooms (all 21) | 26.5 | 31.3 | +4.8 ± 5.0 | 0.37 | 5/8 |
| door | 21.6 | 27.1 | +5.5 ± 5.3 | 0.33 | 5/8 |
| obstacle | 46.6 | 55.6 | +9.0 ± 9.0 | 0.35 | 5/8 |
| door + obstacle | 18.9 | 23.0 | +4.1 ± 6.2 | 0.53 | 5/8 |
| goal moved (open room) | 42.6 | 45.1 | +2.5 ± 3.0 | 0.43 | 5/8 |
| door / obstacle and goal moved | 23.1 | 26.8 | +3.7 ± 3.8 | 0.36 | 5/8 |
| new rooms (never trained) | 27.2 | 32.9 | +5.7 ± 6.0 | 0.38 | 7/8 |
| new rooms, goal moved | 21.5 | 24.6 | +3.1 ± 3.4 | 0.40 | 5/8 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.602 | 0.617 | +0.015 ± 0.037 | 0.70 | 3/8 |
| steps to the first goal, episodes that found it | 312 | 300 | -12 ± 25 | 0.64 | 3/8 |
| steps to the first goal (1350 if never found) | 724 | 700 | -24 ± 51 | 0.65 | 3/8 |
| goals per 1000 steps after the first | 3.19 | 3.89 | +0.70 ± 0.62 | 0.30 | 6/8 |
| first find : return, ratio of steps | 1.43 | 1.57 | +0.15 ± 0.07 | 0.064 | 7/8 |
| path efficiency of completed returns | 0.519 | 0.573 | +0.054 ± 0.035 | 0.17 | 6/8 |
| share of steps against a wall | 0.243 | 0.204 | -0.039 ± 0.065 | 0.56 | 2/8 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.809 | 0.793 | -0.016 ± 0.033 | 0.65 | 3/8 |
| steps to the first goal, episodes that found it | 278 | 268 | -10 ± 13 | 0.46 | 4/8 |
| steps to the first goal (1350 if never found) | 483 | 491 | +8 ± 38 | 0.84 | 4/8 |
| goals per 1000 steps after the first | 5.95 | 6.96 | +1.01 ± 0.69 | 0.19 | 5/8 |
| first find : return, ratio of steps | 1.82 | 1.97 | +0.15 ± 0.12 | 0.26 | 5/8 |
| path efficiency of completed returns | 0.532 | 0.589 | +0.058 ± 0.030 | 0.100 | 6/8 |
| share of steps against a wall | 0.009 | 0.026 | +0.017 ± 0.018 | 0.38 | 4/8 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 2.84 | 3.15 | +0.31 ± 0.31 | 0.35 | 5/8 |
| goals after the move, all episodes | 1.42 | 1.36 | -0.06 ± 0.12 | 0.65 | 4/8 |
| episodes that had found the goal before it moved | 0.735 | 0.733 | -0.003 ± 0.035 | 0.94 | 4/8 |
| goals after the move, in those episodes | 0.98 | 0.78 | -0.19 ± 0.14 | 0.22 | 3/8 |
| share of those episodes that found the moved goal | 0.278 | 0.201 | -0.077 ± 0.044 | 0.13 | 3/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 543 | 581 | +38 ± 22 | 0.12 | 4/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.54 | 2.77 | +0.23 ± 0.13 | 0.13 | 5/8 |
| time near the old goal until the moved goal is found, share | 0.310 | 0.388 | +0.078 ± 0.043 | 0.11 | 6/8 |
| same for the mirror-image control zone | 0.060 | 0.041 | -0.019 ± 0.012 | 0.17 | 3/8 |
| perseveration index (old-goal zone − control zone) | 0.250 | 0.347 | +0.097 ± 0.051 | 0.098 | 6/8 |
| mean distance to the old goal in that time (m) | 0.780 | 0.703 | -0.076 ± 0.049 | 0.17 | 2/8 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.44 | 1.71 | +0.28 ± 0.30 | 0.39 | 5/8 |
| goals after the move, all episodes | 0.87 | 0.97 | +0.10 ± 0.09 | 0.32 | 5/8 |
| episodes that had found the goal before it moved | 0.529 | 0.550 | +0.020 ± 0.045 | 0.66 | 5/8 |
| goals after the move, in those episodes | 0.53 | 0.54 | +0.01 ± 0.07 | 0.93 | 4/8 |
| share of those episodes that found the moved goal | 0.214 | 0.198 | -0.016 ± 0.036 | 0.66 | 3/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 577 | 584 | +7 ± 15 | 0.67 | 5/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.66 | 2.76 | +0.10 ± 0.19 | 0.63 | 4/8 |
| time near the old goal until the moved goal is found, share | 0.212 | 0.258 | +0.046 ± 0.039 | 0.28 | 6/8 |
| same for the mirror-image control zone | 0.072 | 0.058 | -0.013 ± 0.010 | 0.25 | 3/8 |
| perseveration index (old-goal zone − control zone) | 0.140 | 0.200 | +0.059 ± 0.045 | 0.23 | 6/8 |
| mean distance to the old goal in that time (m) | 0.960 | 0.922 | -0.038 ± 0.046 | 0.44 | 2/8 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.56 | 1.84 | +0.28 ± 0.30 | 0.38 | 5/8 |
| goals after the move, all episodes | 0.92 | 1.01 | +0.08 ± 0.09 | 0.35 | 4/8 |
| episodes that had found the goal before it moved | 0.548 | 0.566 | +0.018 ± 0.044 | 0.69 | 5/8 |
| goals after the move, in those episodes | 0.57 | 0.56 | -0.01 ± 0.08 | 0.88 | 4/8 |
| share of those episodes that found the moved goal | 0.220 | 0.198 | -0.022 ± 0.036 | 0.56 | 3/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 574 | 583 | +10 ± 16 | 0.55 | 5/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.65 | 2.76 | +0.11 ± 0.18 | 0.56 | 4/8 |
| time near the old goal until the moved goal is found, share | 0.221 | 0.270 | +0.049 ± 0.038 | 0.24 | 6/8 |
| same for the mirror-image control zone | 0.071 | 0.057 | -0.014 ± 0.010 | 0.23 | 3/8 |
| perseveration index (old-goal zone − control zone) | 0.150 | 0.213 | +0.063 ± 0.045 | 0.20 | 6/8 |
| mean distance to the old goal in that time (m) | 0.944 | 0.902 | -0.041 ± 0.046 | 0.40 | 2/8 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.410 | 0.432 | +0.023 ± 0.012 | 0.11 | 6/8 |
| variance in first 10 components, all steps, familiar room | 0.470 | 0.481 | +0.012 ± 0.010 | 0.27 | 6/8 |
| share of steps with the goal known, familiar room | 0.641 | 0.636 | -0.006 ± 0.028 | 0.84 | 4/8 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.446 | 0.464 | +0.017 ± 0.011 | 0.18 | 5/8 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.532 | 0.530 | -0.002 ± 0.013 | 0.86 | 4/8 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.463 | 0.481 | +0.018 ± 0.037 | 0.65 | 5/8 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 60.7 | 69.4 | +8.7 ± 7.7 | 0.29 | 4/8 |
| D1 | 18.3 | 23.3 | +5.0 ± 3.1 | 0.16 | 6/8 |
| D1+R | 18.8 | 23.2 | +4.3 ± 3.1 | 0.20 | 6/8 |
| D2 | 24.2 | 27.8 | +3.7 ± 6.6 | 0.60 | 4/8 |
| D2+R | 21.2 | 25.3 | +4.1 ± 3.8 | 0.31 | 5/8 |
| D3 | 21.1 | 27.8 | +6.8 ± 6.1 | 0.31 | 5/8 |
| D3+R | 20.8 | 25.0 | +4.1 ± 4.6 | 0.40 | 4/8 |
| D4 | 22.7 | 29.3 | +6.6 ± 6.9 | 0.37 | 4/8 |
| D4+R | 19.8 | 24.3 | +4.5 ± 4.9 | 0.39 | 4/8 |
| O1 | 47.3 | 55.4 | +8.1 ± 10.7 | 0.47 | 5/8 |
| O1+R | 35.4 | 37.8 | +2.4 ± 4.9 | 0.64 | 5/8 |
| O2 | 47.2 | 58.2 | +11.0 ± 8.2 | 0.22 | 5/8 |
| O2+R | 33.2 | 39.5 | +6.2 ± 3.6 | 0.13 | 5/8 |
| O3 | 45.3 | 53.2 | +7.9 ± 9.6 | 0.44 | 5/8 |
| O3+R | 32.9 | 35.5 | +2.6 ± 4.2 | 0.55 | 5/8 |
| DO1 | 18.2 | 22.7 | +4.5 ± 5.8 | 0.47 | 5/8 |
| DO1+R | 16.4 | 19.2 | +2.8 ± 3.9 | 0.49 | 4/8 |
| DO2 | 19.3 | 23.8 | +4.5 ± 6.7 | 0.52 | 4/8 |
| DO2+R | 16.7 | 19.2 | +2.5 ± 4.1 | 0.56 | 4/8 |
| DO3 | 19.2 | 22.5 | +3.3 ± 7.2 | 0.66 | 4/8 |
| DO3+R | 15.6 | 19.4 | +3.7 ± 4.7 | 0.45 | 5/8 |
| open+R | 42.6 | 45.1 | +2.5 ± 3.0 | 0.43 | 5/8 |
| H1 | 21.5 | 20.9 | -0.6 ± 5.6 | 0.92 | 6/8 |
| H1+R | 17.6 | 18.0 | +0.4 ± 3.8 | 0.93 | 5/8 |
| H2 | 42.4 | 53.1 | +10.7 ± 7.1 | 0.18 | 7/8 |
| H2+R | 30.7 | 36.0 | +5.3 ± 3.5 | 0.18 | 6/8 |
| H3 | 17.8 | 24.7 | +6.9 ± 6.6 | 0.33 | 5/8 |
| H3+R | 16.3 | 19.8 | +3.5 ± 3.9 | 0.40 | 5/8 |

## After 500 M frames in the changed rooms

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 58.3 | 75.1 | +16.8 ± 8.6 | 0.093 | 5/8 |
| changed rooms (all 21) | 28.3 | 36.8 | +8.5 ± 4.6 | 0.11 | 7/8 |
| door | 24.5 | 32.7 | +8.2 ± 5.3 | 0.17 | 7/8 |
| obstacle | 49.6 | 64.1 | +14.5 ± 8.0 | 0.11 | 7/8 |
| door + obstacle | 20.9 | 29.4 | +8.5 ± 5.3 | 0.15 | 7/8 |
| goal moved (open room) | 40.8 | 48.2 | +7.4 ± 3.5 | 0.075 | 6/8 |
| door / obstacle and goal moved | 24.4 | 31.4 | +6.9 ± 3.4 | 0.084 | 7/8 |
| new rooms (never trained) | 28.6 | 36.1 | +7.4 ± 7.0 | 0.33 | 7/8 |
| new rooms, goal moved | 22.3 | 27.2 | +4.8 ± 4.5 | 0.32 | 5/8 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.588 | 0.648 | +0.060 ± 0.035 | 0.13 | 6/8 |
| steps to the first goal, episodes that found it | 297 | 284 | -13 ± 16 | 0.44 | 4/8 |
| steps to the first goal (1350 if never found) | 732 | 660 | -72 ± 42 | 0.13 | 2/8 |
| goals per 1000 steps after the first | 3.65 | 4.66 | +1.00 ± 0.53 | 0.099 | 7/8 |
| first find : return, ratio of steps | 1.57 | 1.69 | +0.13 ± 0.08 | 0.16 | 6/8 |
| path efficiency of completed returns | 0.557 | 0.611 | +0.054 ± 0.022 | 0.046 | 7/8 |
| share of steps against a wall | 0.180 | 0.152 | -0.028 ± 0.045 | 0.55 | 4/8 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.761 | 0.799 | +0.039 ± 0.070 | 0.60 | 5/8 |
| steps to the first goal, episodes that found it | 274 | 261 | -13 ± 22 | 0.56 | 3/8 |
| steps to the first goal (1350 if never found) | 535 | 480 | -55 ± 66 | 0.43 | 3/8 |
| goals per 1000 steps after the first | 6.00 | 7.55 | +1.55 ± 0.55 | 0.025 | 7/8 |
| first find : return, ratio of steps | 1.91 | 2.08 | +0.17 ± 0.22 | 0.45 | 4/8 |
| path efficiency of completed returns | 0.572 | 0.642 | +0.070 ± 0.024 | 0.022 | 6/8 |
| share of steps against a wall | 0.007 | 0.011 | +0.004 ± 0.010 | 0.67 | 4/8 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 2.78 | 3.47 | +0.69 ± 0.35 | 0.095 | 5/8 |
| goals after the move, all episodes | 1.29 | 1.35 | +0.05 ± 0.19 | 0.79 | 4/8 |
| episodes that had found the goal before it moved | 0.686 | 0.743 | +0.058 ± 0.062 | 0.39 | 5/8 |
| goals after the move, in those episodes | 0.69 | 0.64 | -0.04 ± 0.17 | 0.80 | 2/8 |
| share of those episodes that found the moved goal | 0.164 | 0.171 | +0.007 ± 0.041 | 0.87 | 3/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 596 | 603 | +7 ± 16 | 0.67 | 6/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 3.01 | 2.94 | -0.07 ± 0.24 | 0.77 | 5/8 |
| time near the old goal until the moved goal is found, share | 0.363 | 0.435 | +0.072 ± 0.025 | 0.025 | 6/8 |
| same for the mirror-image control zone | 0.057 | 0.039 | -0.018 ± 0.013 | 0.21 | 3/8 |
| perseveration index (old-goal zone − control zone) | 0.306 | 0.396 | +0.090 ± 0.030 | 0.019 | 6/8 |
| mean distance to the old goal in that time (m) | 0.679 | 0.627 | -0.052 ± 0.027 | 0.10 | 2/8 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.58 | 2.00 | +0.42 ± 0.27 | 0.16 | 7/8 |
| goals after the move, all episodes | 0.86 | 1.13 | +0.27 ± 0.09 | 0.017 | 7/8 |
| episodes that had found the goal before it moved | 0.521 | 0.586 | +0.064 ± 0.036 | 0.12 | 6/8 |
| goals after the move, in those episodes | 0.41 | 0.54 | +0.13 ± 0.07 | 0.12 | 5/8 |
| share of those episodes that found the moved goal | 0.150 | 0.169 | +0.019 ± 0.023 | 0.44 | 5/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 603 | 595 | -8 ± 10 | 0.41 | 3/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.86 | 2.79 | -0.07 ± 0.10 | 0.50 | 3/8 |
| time near the old goal until the moved goal is found, share | 0.274 | 0.325 | +0.051 ± 0.020 | 0.040 | 7/8 |
| same for the mirror-image control zone | 0.065 | 0.051 | -0.014 ± 0.008 | 0.11 | 2/8 |
| perseveration index (old-goal zone − control zone) | 0.209 | 0.274 | +0.065 ± 0.025 | 0.035 | 7/8 |
| mean distance to the old goal in that time (m) | 0.862 | 0.818 | -0.044 ± 0.032 | 0.21 | 3/8 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.69 | 2.14 | +0.45 ± 0.27 | 0.14 | 7/8 |
| goals after the move, all episodes | 0.90 | 1.15 | +0.25 ± 0.08 | 0.020 | 7/8 |
| episodes that had found the goal before it moved | 0.536 | 0.600 | +0.064 ± 0.037 | 0.13 | 6/8 |
| goals after the move, in those episodes | 0.44 | 0.55 | +0.11 ± 0.07 | 0.16 | 5/8 |
| share of those episodes that found the moved goal | 0.151 | 0.170 | +0.018 ± 0.023 | 0.46 | 4/8 |
| steps from the move to the moved goal, those episodes (675 if never) | 603 | 596 | -7 ± 9 | 0.46 | 4/8 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.87 | 2.80 | -0.07 ± 0.11 | 0.51 | 3/8 |
| time near the old goal until the moved goal is found, share | 0.282 | 0.335 | +0.053 ± 0.020 | 0.031 | 7/8 |
| same for the mirror-image control zone | 0.064 | 0.050 | -0.014 ± 0.008 | 0.11 | 2/8 |
| perseveration index (old-goal zone − control zone) | 0.218 | 0.285 | +0.067 ± 0.024 | 0.027 | 7/8 |
| mean distance to the old goal in that time (m) | 0.845 | 0.801 | -0.045 ± 0.031 | 0.19 | 3/8 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.399 | 0.424 | +0.024 ± 0.009 | 0.028 | 7/8 |
| variance in first 10 components, all steps, familiar room | 0.467 | 0.462 | -0.005 ± 0.010 | 0.67 | 3/8 |
| share of steps with the goal known, familiar room | 0.603 | 0.644 | +0.041 ± 0.049 | 0.43 | 5/8 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.434 | 0.452 | +0.018 ± 0.009 | 0.084 | 6/8 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.516 | 0.502 | -0.014 ± 0.008 | 0.11 | 2/8 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.458 | 0.511 | +0.053 ± 0.031 | 0.13 | 6/8 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 58.3 | 75.1 | +16.8 ± 8.6 | 0.093 | 5/8 |
| D1 | 22.3 | 30.8 | +8.5 ± 4.5 | 0.10 | 6/8 |
| D1+R | 21.4 | 28.3 | +7.0 ± 2.5 | 0.026 | 7/8 |
| D2 | 25.9 | 34.0 | +8.1 ± 6.0 | 0.21 | 6/8 |
| D2+R | 22.9 | 29.7 | +6.8 ± 3.9 | 0.13 | 6/8 |
| D3 | 23.4 | 31.4 | +8.0 ± 4.0 | 0.087 | 7/8 |
| D3+R | 21.7 | 28.3 | +6.6 ± 3.4 | 0.089 | 7/8 |
| D4 | 26.3 | 34.5 | +8.2 ± 8.3 | 0.36 | 5/8 |
| D4+R | 22.9 | 28.5 | +5.6 ± 5.4 | 0.34 | 6/8 |
| O1 | 49.4 | 62.0 | +12.5 ± 9.7 | 0.24 | 4/8 |
| O1+R | 34.0 | 41.0 | +7.0 ± 4.1 | 0.13 | 5/8 |
| O2 | 50.9 | 70.9 | +20.0 ± 6.9 | 0.023 | 7/8 |
| O2+R | 35.2 | 46.5 | +11.3 ± 3.4 | 0.012 | 7/8 |
| O3 | 48.4 | 59.3 | +11.0 ± 8.6 | 0.24 | 5/8 |
| O3+R | 33.4 | 38.4 | +5.0 ± 4.2 | 0.27 | 5/8 |
| DO1 | 18.4 | 28.6 | +10.2 ± 4.8 | 0.071 | 7/8 |
| DO1+R | 16.7 | 24.2 | +7.6 ± 3.8 | 0.086 | 6/8 |
| DO2 | 22.3 | 31.9 | +9.6 ± 4.9 | 0.093 | 7/8 |
| DO2+R | 18.8 | 25.2 | +6.4 ± 3.7 | 0.13 | 6/8 |
| DO3 | 22.0 | 27.7 | +5.7 ± 7.4 | 0.46 | 5/8 |
| DO3+R | 17.5 | 23.5 | +6.0 ± 4.5 | 0.22 | 6/8 |
| open+R | 40.8 | 48.2 | +7.4 ± 3.5 | 0.075 | 6/8 |
| H1 | 19.7 | 21.1 | +1.4 ± 5.3 | 0.81 | 4/8 |
| H1+R | 17.1 | 18.6 | +1.5 ± 4.2 | 0.73 | 5/8 |
| H2 | 44.2 | 56.8 | +12.6 ± 9.2 | 0.21 | 7/8 |
| H2+R | 30.7 | 37.9 | +7.2 ± 5.3 | 0.21 | 4/8 |
| H3 | 22.0 | 30.3 | +8.3 ± 7.3 | 0.29 | 6/8 |
| H3+R | 19.3 | 25.1 | +5.8 ± 4.8 | 0.27 | 6/8 |

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

## After 500 M more frames in the familiar room (seeds 1–2)

### Score by room type

| rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| familiar room | 109.8 | 111.6 | +1.8 ± 15.2 | 0.92 | 1/2 |
| changed rooms (all 21) | 29.2 | 23.6 | -5.6 ± 5.5 | 0.49 | 0/2 |
| door | 21.0 | 17.4 | -3.6 ± 5.0 | 0.60 | 1/2 |
| obstacle | 61.3 | 46.3 | -15.0 ± 10.0 | 0.37 | 0/2 |
| door + obstacle | 18.4 | 12.4 | -6.0 ± 4.1 | 0.38 | 0/2 |
| goal moved (open room) | 55.2 | 61.4 | +6.2 ± 5.1 | 0.44 | 2/2 |
| door / obstacle and goal moved | 23.4 | 18.9 | -4.5 ± 4.7 | 0.51 | 1/2 |
| new rooms (never trained) | 38.2 | 28.0 | -10.2 ± 3.0 | 0.18 | 0/2 |
| new rooms, goal moved | 26.4 | 19.7 | -6.6 ± 1.5 | 0.14 | 0/2 |

### Navigation, changed rooms, goal fixed (10)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.641 | 0.501 | -0.139 ± 0.066 | 0.28 | 0/2 |
| steps to the first goal, episodes that found it | 317 | 350 | +33 ± 114 | 0.82 | 1/2 |
| steps to the first goal (1350 if never found) | 681 | 848 | +167 ± 130 | 0.42 | 2/2 |
| goals per 1000 steps after the first | 3.51 | 3.54 | +0.03 ± 0.10 | 0.81 | 1/2 |
| first find : return, ratio of steps | 1.92 | 2.24 | +0.31 ± 0.61 | 0.70 | 1/2 |
| path efficiency of completed returns | 0.671 | 0.651 | -0.020 ± 0.057 | 0.78 | 1/2 |
| share of steps against a wall | 0.219 | 0.372 | +0.153 ± 0.288 | 0.69 | 1/2 |

### Navigation, familiar room

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| episodes that found the goal | 0.917 | 0.790 | -0.128 ± 0.077 | 0.35 | 0/2 |
| steps to the first goal, episodes that found it | 287 | 279 | -8 ± 5 | 0.33 | 0/2 |
| steps to the first goal (1350 if never found) | 375 | 504 | +129 ± 79 | 0.35 | 2/2 |
| goals per 1000 steps after the first | 10.20 | 12.04 | +1.84 ± 0.49 | 0.16 | 2/2 |
| first find : return, ratio of steps | 3.09 | 3.49 | +0.40 ± 0.31 | 0.42 | 2/2 |
| path efficiency of completed returns | 0.753 | 0.815 | +0.062 ± 0.022 | 0.22 | 2/2 |
| share of steps against a wall | 0.003 | 0.001 | -0.002 ± 0.002 | 0.56 | 1/2 |

### Goal relocation, goal moved (open room)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 4.69 | 4.72 | +0.03 ± 0.60 | 0.96 | 1/2 |
| goals after the move, all episodes | 0.83 | 1.42 | +0.59 ± 0.09 | 0.099 | 2/2 |
| episodes that had found the goal before it moved | 0.830 | 0.723 | -0.107 ± 0.072 | 0.38 | 0/2 |
| goals after the move, in those episodes | 0.25 | 0.37 | +0.12 ± 0.01 | 0.059 | 2/2 |
| share of those episodes that found the moved goal | 0.030 | 0.044 | +0.014 ± 0.001 | 0.023 | 2/2 |
| steps from the move to the moved goal, those episodes (675 if never) | 656 | 647 | -9 ± 2 | 0.13 | 0/2 |
| reversal cost: steps to the moved goal / steps to the first goal | 2.97 | 3.01 | +0.05 ± 0.07 | 0.62 | 1/2 |
| time near the old goal until the moved goal is found, share | 0.679 | 0.815 | +0.136 ± 0.054 | 0.24 | 2/2 |
| same for the mirror-image control zone | 0.024 | 0.037 | +0.013 ± 0.001 | 0.044 | 2/2 |
| perseveration index (old-goal zone − control zone) | 0.655 | 0.778 | +0.123 ± 0.055 | 0.27 | 2/2 |
| mean distance to the old goal in that time (m) | 0.438 | 0.369 | -0.069 ± 0.023 | 0.20 | 0/2 |

### Goal relocation, door / obstacle and goal moved

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 1.76 | 1.29 | -0.47 ± 0.49 | 0.52 | 1/2 |
| goals after the move, all episodes | 0.59 | 0.60 | +0.02 ± 0.02 | 0.61 | 1/2 |
| episodes that had found the goal before it moved | 0.565 | 0.421 | -0.143 ± 0.114 | 0.43 | 0/2 |
| goals after the move, in those episodes | 0.11 | 0.17 | +0.06 ± 0.07 | 0.56 | 1/2 |
| share of those episodes that found the moved goal | 0.045 | 0.063 | +0.017 ± 0.041 | 0.75 | 1/2 |
| steps from the move to the moved goal, those episodes (675 if never) | 653 | 644 | -9 ± 19 | 0.71 | 1/2 |
| reversal cost: steps to the moved goal / steps to the first goal | 3.09 | 2.84 | -0.26 ± 0.61 | 0.75 | 1/2 |
| time near the old goal until the moved goal is found, share | 0.338 | 0.366 | +0.028 ± 0.016 | 0.33 | 2/2 |
| same for the mirror-image control zone | 0.049 | 0.050 | +0.001 ± 0.001 | 0.61 | 1/2 |
| perseveration index (old-goal zone − control zone) | 0.289 | 0.316 | +0.027 ± 0.014 | 0.31 | 2/2 |
| mean distance to the old goal in that time (m) | 0.812 | 0.781 | -0.031 ± 0.041 | 0.58 | 1/2 |

### Goal relocation, rooms with a goal move (11)

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| goals before the move | 2.02 | 1.60 | -0.42 ± 0.50 | 0.55 | 1/2 |
| goals after the move, all episodes | 0.61 | 0.68 | +0.07 ± 0.03 | 0.25 | 2/2 |
| episodes that had found the goal before it moved | 0.589 | 0.449 | -0.140 ± 0.110 | 0.42 | 0/2 |
| goals after the move, in those episodes | 0.12 | 0.19 | +0.07 ± 0.06 | 0.49 | 2/2 |
| share of those episodes that found the moved goal | 0.044 | 0.061 | +0.017 ± 0.037 | 0.73 | 1/2 |
| steps from the move to the moved goal, those episodes (675 if never) | 654 | 644 | -9 ± 17 | 0.68 | 1/2 |
| reversal cost: steps to the moved goal / steps to the first goal | 3.08 | 2.85 | -0.23 ± 0.55 | 0.75 | 1/2 |
| time near the old goal until the moved goal is found, share | 0.369 | 0.407 | +0.038 ± 0.019 | 0.30 | 2/2 |
| same for the mirror-image control zone | 0.047 | 0.049 | +0.002 ± 0.001 | 0.34 | 2/2 |
| perseveration index (old-goal zone − control zone) | 0.323 | 0.358 | +0.036 ± 0.018 | 0.30 | 2/2 |
| mean distance to the old goal in that time (m) | 0.778 | 0.743 | -0.035 ± 0.039 | 0.54 | 1/2 |

### PCA of the LSTM output during the agent's own behaviour

| statistic | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| variance in first 10 components, goal known, familiar room | 0.405 | 0.479 | +0.074 ± 0.006 | 0.048 | 2/2 |
| variance in first 10 components, all steps, familiar room | 0.444 | 0.525 | +0.081 ± 0.004 | 0.028 | 2/2 |
| share of steps with the goal known, familiar room | 0.722 | 0.626 | -0.096 ± 0.059 | 0.35 | 0/2 |
| variance in first 10 components, goal known, changed rooms, goal fixed (10) | 0.452 | 0.485 | +0.033 ± 0.024 | 0.40 | 2/2 |
| variance in first 10 components, all steps, changed rooms, goal fixed (10) | 0.521 | 0.563 | +0.042 ± 0.015 | 0.23 | 2/2 |
| share of steps with the goal known, changed rooms, goal fixed (10) | 0.495 | 0.372 | -0.124 ± 0.096 | 0.42 | 0/2 |

### Every room (score)

| room | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| open | 109.8 | 111.6 | +1.8 ± 15.2 | 0.92 | 1/2 |
| D1 | 23.0 | 22.2 | -0.8 ± 9.7 | 0.95 | 1/2 |
| D1+R | 19.1 | 16.5 | -2.6 ± 9.6 | 0.83 | 1/2 |
| D2 | 22.4 | 15.7 | -6.6 ± 4.3 | 0.37 | 0/2 |
| D2+R | 19.6 | 14.5 | -5.1 ± 5.7 | 0.54 | 1/2 |
| D3 | 15.3 | 11.0 | -4.3 ± 2.5 | 0.33 | 0/2 |
| D3+R | 17.1 | 12.1 | -5.0 ± 2.5 | 0.30 | 0/2 |
| D4 | 23.4 | 20.8 | -2.6 ± 3.6 | 0.60 | 1/2 |
| D4+R | 19.6 | 18.3 | -1.2 ± 4.2 | 0.82 | 1/2 |
| O1 | 50.2 | 29.6 | -20.7 ± 16.0 | 0.42 | 0/2 |
| O1+R | 31.4 | 22.0 | -9.4 ± 8.1 | 0.45 | 0/2 |
| O2 | 64.1 | 53.0 | -11.1 ± 10.5 | 0.48 | 0/2 |
| O2+R | 37.5 | 38.4 | +0.9 ± 4.6 | 0.88 | 1/2 |
| O3 | 69.6 | 56.4 | -13.2 ± 3.5 | 0.16 | 0/2 |
| O3+R | 40.6 | 35.3 | -5.3 ± 0.7 | 0.089 | 0/2 |
| DO1 | 16.8 | 13.5 | -3.3 ± 1.4 | 0.26 | 0/2 |
| DO1+R | 16.9 | 11.4 | -5.5 ± 2.8 | 0.31 | 0/2 |
| DO2 | 17.6 | 8.9 | -8.7 ± 5.6 | 0.37 | 0/2 |
| DO2+R | 14.6 | 8.6 | -6.0 ± 5.1 | 0.45 | 0/2 |
| DO3 | 20.9 | 14.7 | -6.2 ± 5.3 | 0.45 | 0/2 |
| DO3+R | 18.1 | 11.9 | -6.2 ± 3.7 | 0.35 | 0/2 |
| open+R | 55.2 | 61.4 | +6.2 ± 5.1 | 0.44 | 2/2 |
| H1 | 27.9 | 16.8 | -11.1 ± 7.8 | 0.39 | 0/2 |
| H1+R | 22.4 | 14.3 | -8.1 ± 4.0 | 0.29 | 0/2 |
| H2 | 69.0 | 52.0 | -17.0 ± 2.9 | 0.11 | 0/2 |
| H2+R | 40.3 | 31.9 | -8.4 ± 3.1 | 0.22 | 0/2 |
| H3 | 17.7 | 15.2 | -2.6 ± 3.9 | 0.63 | 1/2 |
| H3+R | 16.4 | 13.0 | -3.4 ± 3.6 | 0.52 | 1/2 |

## PCA with the earlier protocol (one scripted trajectory replayed through every network)

| agents | replay in | quantity | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|---|---|
| 1 B | familiar room | variance in first 10 components | 0.4642 | 0.5232 | +0.0591 ± 0.0039 | < 0.001 | 8/8 |
| 1 B | familiar room | participation ratio | 31.33 | 23.84 | -7.49 ± 1.15 | < 0.001 | 0/8 |
| 1 B | familiar room | total variance of the 256 outputs | 19.57 | 24.28 | +4.71 ± 0.96 | 0.002 | 7/8 |
| 1 B | changed rooms | variance in first 10 components | 0.4679 | 0.5249 | +0.0570 ± 0.0056 | < 0.001 | 8/8 |
| 1 B | changed rooms | participation ratio | 30.79 | 24.57 | -6.22 ± 1.20 | 0.001 | 0/8 |
| 1 B | changed rooms | total variance of the 256 outputs | 19.15 | 23.59 | +4.44 ± 1.01 | 0.003 | 7/8 |
| 1.25 B, changed rooms | familiar room | variance in first 10 components | 0.4676 | 0.4908 | +0.0232 ± 0.0111 | 0.075 | 7/8 |
| 1.25 B, changed rooms | familiar room | participation ratio | 30.43 | 28.55 | -1.88 ± 2.28 | 0.44 | 1/8 |
| 1.25 B, changed rooms | familiar room | total variance of the 256 outputs | 17.31 | 21.96 | +4.65 ± 1.75 | 0.032 | 7/8 |
| 1.25 B, changed rooms | changed rooms | variance in first 10 components | 0.4804 | 0.5011 | +0.0207 ± 0.0124 | 0.14 | 6/8 |
| 1.25 B, changed rooms | changed rooms | participation ratio | 29.09 | 26.82 | -2.26 ± 2.34 | 0.36 | 2/8 |
| 1.25 B, changed rooms | changed rooms | total variance of the 256 outputs | 17.20 | 21.45 | +4.25 ± 1.77 | 0.048 | 7/8 |
| 1.5 B, changed rooms | familiar room | variance in first 10 components | 0.4565 | 0.4892 | +0.0327 ± 0.0094 | 0.010 | 8/8 |
| 1.5 B, changed rooms | familiar room | participation ratio | 32.56 | 29.06 | -3.50 ± 1.63 | 0.069 | 2/8 |
| 1.5 B, changed rooms | familiar room | total variance of the 256 outputs | 17.34 | 22.88 | +5.55 ± 1.73 | 0.015 | 7/8 |
| 1.5 B, changed rooms | changed rooms | variance in first 10 components | 0.4701 | 0.4920 | +0.0219 ± 0.0082 | 0.032 | 7/8 |
| 1.5 B, changed rooms | changed rooms | participation ratio | 30.68 | 29.30 | -1.38 ± 1.30 | 0.32 | 4/8 |
| 1.5 B, changed rooms | changed rooms | total variance of the 256 outputs | 17.40 | 22.15 | +4.75 ± 1.65 | 0.023 | 7/8 |
| 1.25 B, familiar room (seeds 1–2) | familiar room | variance in first 10 components | 0.4635 | 0.5298 | +0.0662 ± 0.0140 | 0.13 | 2/2 |
| 1.25 B, familiar room (seeds 1–2) | familiar room | participation ratio | 30.52 | 23.42 | -7.10 ± 1.02 | 0.090 | 0/2 |
| 1.25 B, familiar room (seeds 1–2) | familiar room | total variance of the 256 outputs | 20.25 | 23.52 | +3.27 ± 2.60 | 0.43 | 2/2 |
| 1.25 B, familiar room (seeds 1–2) | changed rooms | variance in first 10 components | 0.4780 | 0.5315 | +0.0535 ± 0.0081 | 0.095 | 2/2 |
| 1.25 B, familiar room (seeds 1–2) | changed rooms | participation ratio | 28.86 | 24.04 | -4.82 ± 1.64 | 0.21 | 0/2 |
| 1.25 B, familiar room (seeds 1–2) | changed rooms | total variance of the 256 outputs | 19.99 | 23.29 | +3.30 ± 2.33 | 0.39 | 2/2 |
| 1.5 B, familiar room (seeds 1–2) | familiar room | variance in first 10 components | 0.4467 | 0.5233 | +0.0766 ± 0.0199 | 0.16 | 2/2 |
| 1.5 B, familiar room (seeds 1–2) | familiar room | participation ratio | 33.67 | 23.08 | -10.59 ± 0.29 | 0.017 | 0/2 |
| 1.5 B, familiar room (seeds 1–2) | familiar room | total variance of the 256 outputs | 20.99 | 24.53 | +3.54 ± 5.04 | 0.61 | 1/2 |
| 1.5 B, familiar room (seeds 1–2) | changed rooms | variance in first 10 components | 0.4631 | 0.5180 | +0.0549 ± 0.0252 | 0.27 | 2/2 |
| 1.5 B, familiar room (seeds 1–2) | changed rooms | participation ratio | 31.77 | 24.06 | -7.72 ± 2.78 | 0.22 | 0/2 |
| 1.5 B, familiar room (seeds 1–2) | changed rooms | total variance of the 256 outputs | 20.80 | 24.49 | +3.69 ± 4.40 | 0.56 | 1/2 |

Change of each agent's first-10 share against its own 1 B value:

| agents | replay in | grid | PGC | PGC − grid | p |
|---|---|---|---|---|---|
| 1.25 B, changed rooms | familiar | +0.0035 ± 0.0106 | -0.0324 ± 0.0047 | -0.0359 ± 0.0106 | 0.012 |
| 1.25 B, changed rooms | changed | +0.0125 ± 0.0127 | -0.0238 ± 0.0065 | -0.0363 ± 0.0132 | 0.029 |
| 1.5 B, changed rooms | familiar | -0.0076 ± 0.0056 | -0.0340 ± 0.0051 | -0.0264 ± 0.0080 | 0.013 |
| 1.5 B, changed rooms | changed | +0.0023 ± 0.0070 | -0.0329 ± 0.0042 | -0.0351 ± 0.0076 | 0.002 |
| 1.25 B, familiar room (seeds 1–2) | familiar | +0.0031 ± 0.0028 | +0.0244 ± 0.0087 | +0.0213 ± 0.0115 | 0.32 |
| 1.25 B, familiar room (seeds 1–2) | changed | +0.0059 ± 0.0041 | +0.0165 ± 0.0018 | +0.0105 ± 0.0023 | 0.14 |
| 1.5 B, familiar room (seeds 1–2) | familiar | -0.0138 ± 0.0016 | +0.0179 ± 0.0190 | +0.0316 ± 0.0174 | 0.32 |
| 1.5 B, familiar room (seeds 1–2) | changed | -0.0089 ± 0.0009 | +0.0030 ± 0.0158 | +0.0120 ± 0.0149 | 0.57 |

## Change over the first 250 M frames in the changed rooms (after − before), per code

| rooms | statistic | grid | PGC | PGC − grid | p |
|---|---|---|---|---|---|
| familiar room | score | -36.2 ± 3.1 | -36.7 ± 8.1 | -0.5 ± 7.4 | 0.95 |
| familiar room | path efficiency of completed returns | -0.169 ± 0.024 | -0.189 ± 0.028 | -0.020 ± 0.023 | 0.41 |
| familiar room | goals per 1000 steps after the first | -3.39 ± 0.53 | -3.67 ± 0.83 | -0.29 ± 0.76 | 0.72 |
| changed rooms (all 21) | score | +5.0 ± 1.4 | +6.9 ± 3.1 | +1.9 ± 3.2 | 0.58 |
| changed rooms, goal fixed (10) | score | +5.4 ± 1.9 | +8.0 ± 3.9 | +2.6 ± 4.1 | 0.55 |
| changed rooms, goal fixed (10) | path efficiency of completed returns | -0.066 ± 0.011 | -0.078 ± 0.014 | -0.013 ± 0.013 | 0.38 |
| changed rooms, goal fixed (10) | goals per 1000 steps after the first | +0.33 ± 0.29 | +0.54 ± 0.52 | +0.21 ± 0.56 | 0.72 |
| changed rooms, goal fixed (10) | share of steps against a wall | -0.172 ± 0.038 | -0.138 ± 0.035 | +0.034 ± 0.048 | 0.51 |
| changed rooms, goal fixed (10) | steps to the first goal (1350 if never found) | -95 ± 33 | -94 ± 28 | +1 ± 52 | 0.98 |
| door | score | +5.8 ± 1.4 | +9.2 ± 3.1 | +3.3 ± 3.5 | 0.37 |
| obstacle | score | +2.8 ± 2.5 | +4.6 ± 5.7 | +1.8 ± 5.7 | 0.76 |
| door + obstacle | score | +7.4 ± 2.9 | +9.8 ± 3.7 | +2.4 ± 4.1 | 0.58 |
| goal moved (open room) | goals after the move, in those episodes | +0.64 ± 0.21 | +0.51 ± 0.12 | -0.13 ± 0.24 | 0.59 |
| goal moved (open room) | share of those episodes that found the moved goal | +0.226 ± 0.070 | +0.158 ± 0.040 | -0.068 ± 0.057 | 0.27 |
| goal moved (open room) | perseveration index (old-goal zone − control zone) | -0.332 ± 0.054 | -0.365 ± 0.053 | -0.033 ± 0.047 | 0.50 |
| rooms with a goal move (11) | goals after the move, in those episodes | +0.36 ± 0.06 | +0.35 ± 0.04 | -0.01 ± 0.07 | 0.89 |
| rooms with a goal move (11) | share of those episodes that found the moved goal | +0.140 ± 0.028 | +0.124 ± 0.016 | -0.016 ± 0.027 | 0.57 |
| rooms with a goal move (11) | perseveration index (old-goal zone − control zone) | -0.108 ± 0.024 | -0.104 ± 0.037 | +0.005 ± 0.032 | 0.89 |
| new rooms (never trained) | score | -0.5 ± 1.9 | +0.6 ± 3.1 | +1.1 ± 3.6 | 0.77 |

## Change over all 500 M frames (long − before), per code

| rooms | statistic | grid | PGC | PGC − grid | p |
|---|---|---|---|---|---|
| familiar room | score | -38.5 ± 3.7 | -31.0 ± 6.9 | +7.5 ± 8.9 | 0.43 |
| familiar room | path efficiency of completed returns | -0.129 ± 0.011 | -0.137 ± 0.019 | -0.007 ± 0.019 | 0.71 |
| familiar room | goals per 1000 steps after the first | -3.34 ± 0.24 | -3.09 ± 0.67 | +0.26 ± 0.64 | 0.70 |
| changed rooms (all 21) | score | +6.9 ± 2.1 | +12.4 ± 2.7 | +5.5 ± 2.6 | 0.072 |
| changed rooms, goal fixed (10) | score | +8.1 ± 2.7 | +14.7 ± 3.4 | +6.6 ± 3.5 | 0.098 |
| changed rooms, goal fixed (10) | path efficiency of completed returns | -0.027 ± 0.013 | -0.039 ± 0.015 | -0.012 ± 0.017 | 0.51 |
| changed rooms, goal fixed (10) | goals per 1000 steps after the first | +0.79 ± 0.30 | +1.31 ± 0.46 | +0.51 ± 0.49 | 0.32 |
| changed rooms, goal fixed (10) | share of steps against a wall | -0.234 ± 0.054 | -0.189 ± 0.055 | +0.045 ± 0.072 | 0.55 |
| changed rooms, goal fixed (10) | steps to the first goal (1350 if never found) | -88 ± 31 | -134 ± 27 | -46 ± 33 | 0.20 |
| door | score | +8.7 ± 2.3 | +14.8 ± 3.3 | +6.0 ± 3.4 | 0.12 |
| obstacle | score | +5.8 ± 4.2 | +13.1 ± 4.5 | +7.3 ± 5.0 | 0.19 |
| door + obstacle | score | +9.4 ± 3.2 | +16.2 ± 3.1 | +6.8 ± 3.3 | 0.079 |
| goal moved (open room) | goals after the move, in those episodes | +0.35 ± 0.15 | +0.37 ± 0.10 | +0.01 ± 0.21 | 0.95 |
| goal moved (open room) | share of those episodes that found the moved goal | +0.113 ± 0.025 | +0.128 ± 0.029 | +0.015 ± 0.039 | 0.71 |
| goal moved (open room) | perseveration index (old-goal zone − control zone) | -0.276 ± 0.030 | -0.316 ± 0.037 | -0.040 ± 0.040 | 0.35 |
| rooms with a goal move (11) | goals after the move, in those episodes | +0.22 ± 0.07 | +0.33 ± 0.10 | +0.12 ± 0.10 | 0.28 |
| rooms with a goal move (11) | share of those episodes that found the moved goal | +0.071 ± 0.019 | +0.095 ± 0.029 | +0.024 ± 0.030 | 0.46 |
| rooms with a goal move (11) | perseveration index (old-goal zone − control zone) | -0.041 ± 0.016 | -0.032 ± 0.023 | +0.009 ± 0.022 | 0.70 |
| new rooms (never trained) | score | +0.9 ± 3.1 | +3.7 ± 3.0 | +2.9 ± 4.5 | 0.55 |

## Change from 250 M to 500 M frames, per code

| rooms | statistic | grid | PGC | PGC − grid | p |
|---|---|---|---|---|---|
| familiar room | score | -2.3 ± 4.1 | +5.7 ± 3.1 | +8.0 ± 2.8 | 0.026 |
| familiar room | path efficiency of completed returns | +0.040 ± 0.021 | +0.052 ± 0.024 | +0.013 ± 0.023 | 0.60 |
| familiar room | goals per 1000 steps after the first | +0.05 ± 0.35 | +0.59 ± 0.44 | +0.54 ± 0.49 | 0.30 |
| changed rooms (all 21) | score | +1.8 ± 1.2 | +5.5 ± 1.3 | +3.7 ± 0.9 | 0.004 |
| changed rooms, goal fixed (10) | score | +2.7 ± 1.6 | +6.7 ± 1.6 | +4.0 ± 0.9 | 0.002 |
| changed rooms, goal fixed (10) | path efficiency of completed returns | +0.039 ± 0.005 | +0.039 ± 0.018 | +0.000 ± 0.016 | 0.98 |
| changed rooms, goal fixed (10) | goals per 1000 steps after the first | +0.46 ± 0.19 | +0.76 ± 0.21 | +0.30 ± 0.21 | 0.20 |
| changed rooms, goal fixed (10) | share of steps against a wall | -0.063 ± 0.031 | -0.051 ± 0.031 | +0.011 ± 0.035 | 0.76 |
| changed rooms, goal fixed (10) | steps to the first goal (1350 if never found) | +7 ± 26 | -40 ± 16 | -47 ± 34 | 0.20 |
| door | score | +2.9 ± 1.5 | +5.6 ± 1.1 | +2.7 ± 0.8 | 0.013 |
| obstacle | score | +3.0 ± 2.9 | +8.5 ± 2.8 | +5.5 ± 1.9 | 0.022 |
| door + obstacle | score | +2.0 ± 1.4 | +6.4 ± 1.8 | +4.4 ± 1.7 | 0.037 |
| goal moved (open room) | goals after the move, in those episodes | -0.29 ± 0.17 | -0.14 ± 0.14 | +0.15 ± 0.21 | 0.51 |
| goal moved (open room) | share of those episodes that found the moved goal | -0.113 ± 0.067 | -0.030 ± 0.050 | +0.083 ± 0.069 | 0.26 |
| goal moved (open room) | perseveration index (old-goal zone − control zone) | +0.056 ± 0.037 | +0.048 ± 0.051 | -0.007 ± 0.044 | 0.87 |
| rooms with a goal move (11) | goals after the move, in those episodes | -0.14 ± 0.07 | -0.01 ± 0.10 | +0.13 ± 0.11 | 0.29 |
| rooms with a goal move (11) | share of those episodes that found the moved goal | -0.069 ± 0.033 | -0.029 ± 0.038 | +0.040 ± 0.045 | 0.40 |
| rooms with a goal move (11) | perseveration index (old-goal zone − control zone) | +0.068 ± 0.023 | +0.072 ± 0.041 | +0.005 ± 0.036 | 0.90 |
| new rooms (never trained) | score | +1.4 ± 2.3 | +3.2 ± 1.7 | +1.8 ± 1.6 | 0.32 |

## Confirmatory contrasts (pre-declared; PGC − grid; exact sign-flip p over 8 seeds, Holm-adjusted)

| contrast | estimate | 95 % CI | p (t) | p (sign-flip) | p (Holm) | k/n |
|---|---|---|---|---|---|---|
| score gap, changed rooms, before (frozen 1 B agents) | +3.0 | ± 6.6 | 0.32 | 0.31 | 1.00 | 6/8 |
| score gap, changed rooms, after 250 M frames | +4.8 | ± 11.9 | 0.37 | 0.34 | 1.00 | 5/8 |
| change of the score gap in the changed rooms with fine-tuning | +1.9 | ± 7.5 | 0.58 | 0.59 | 1.00 | 5/8 |
| change of the familiar-room score with fine-tuning (after − before) | -0.5 | ± 17.4 | 0.95 | 0.93 | 1.00 | 4/8 |
| first-10 variance share, familiar-room replay, after 250 M frames | +0.0232 | ± 0.0262 | 0.075 | 0.062 | 0.38 | 7/8 |
| first-10 variance share, changed-room replay, after 250 M frames | +0.0207 | ± 0.0293 | 0.14 | 0.12 | 0.62 | 6/8 |

Training score in the changed rooms, last 25 M window of the planned 250 M minus the first: grid +4.1, PGC +8.0, difference +3.9 ± 1.7 (p 0.051, 7/8 seeds)

PGC − grid lead in the familiar room minus the lead in the changed rooms: before +6.3 ± 2.9 (p 0.069); after +3.9 ± 4.5 (p 0.41); same +6.1 ± 6.6 (p 0.53); long +8.3 ± 5.7 (p 0.19); same_long +7.4 ± 9.7 (p 0.59)

## Transfer table: After 250 M frames in the changed rooms

| quantity | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| jumpstart: changed-room score before fine-tuning / familiar-room score | 0.221 | 0.231 | +0.010 ± 0.023 | 0.67 | 5/8 |
| gain: changed-room score, later − before | 5.0 | 6.9 | +1.9 ± 3.2 | 0.58 | 5/8 |
| backward transfer: familiar-room score, later − before | -36.2 | -36.7 | -0.5 ± 7.4 | 0.95 | 4/8 |
| changed-room score after / familiar-room score before | 0.271 | 0.295 | +0.024 ± 0.045 | 0.61 | 4/8 |

Gain in the changed rooms against the change in the familiar room, Spearman over agents: gc100 rho -0.26 (p 0.53, n = 8); pgc33 rho +0.52 (p 0.18, n = 8); pooled rho +0.27 (p 0.31, n = 16)

## Transfer table: After 500 M frames in the changed rooms

| quantity | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| jumpstart: changed-room score before fine-tuning / familiar-room score | 0.221 | 0.231 | +0.010 ± 0.023 | 0.67 | 5/8 |
| gain: changed-room score, later − before | 6.9 | 12.4 | +5.5 ± 2.6 | 0.072 | 5/8 |
| backward transfer: familiar-room score, later − before | -38.5 | -31.0 | +7.5 ± 8.9 | 0.43 | 4/8 |
| changed-room score after / familiar-room score before | 0.288 | 0.347 | +0.058 ± 0.040 | 0.18 | 7/8 |

Gain in the changed rooms against the change in the familiar room, Spearman over agents: gc100 rho +0.05 (p 0.91, n = 8); pgc33 rho +0.55 (p 0.16, n = 8); pooled rho +0.38 (p 0.14, n = 16)

Control agents (After 250 M more frames in the familiar room (seeds 1–2)): gc100_s1 familiar +0.0, changed -4.1; gc100_s2 familiar -0.5, changed +3.7; pgc33_s1 familiar +8.0, changed +4.6; pgc33_s2 familiar +1.6, changed +2.9

Control agents (After 500 M more frames in the familiar room (seeds 1–2)): gc100_s1 familiar +4.9, changed -0.6; gc100_s2 familiar +8.1, changed +1.6; pgc33_s1 familiar +24.5, changed +2.1; pgc33_s2 familiar -2.3, changed +3.5

## Time to threshold during fine-tuning (frames, M; training score in the changed rooms, 25 M running mean)

| run | familiar-room training score before | best changed-room score | 50 % reached at | 90 % reached at |
|---|---|---|---|---|
| gc100room_lstm_s1 | 94.1 | 22.7 | not reached | not reached |
| gc100room_lstm_s2 | 108.4 | 47.4 | not reached | not reached |
| gc100room_lstm_s3 | 90.8 | 21.6 | not reached | not reached |
| gc100room_lstm_s4 | 103.2 | 34.2 | not reached | not reached |
| gc100room_lstm_s5 | 91.7 | 24.8 | not reached | not reached |
| gc100room_lstm_s6 | 100.9 | 28.3 | not reached | not reached |
| gc100room_lstm_s7 | 85.3 | 24.5 | not reached | not reached |
| gc100room_lstm_s8 | 92.4 | 27.6 | not reached | not reached |
| pgc33room_lstm_s1 | 104.6 | 26.6 | not reached | not reached |
| pgc33room_lstm_s2 | 108.2 | 34.7 | not reached | not reached |
| pgc33room_lstm_s3 | 110.4 | 28.6 | not reached | not reached |
| pgc33room_lstm_s4 | 118.8 | 57.6 | not reached | not reached |
| pgc33room_lstm_s5 | 108.9 | 33.2 | not reached | not reached |
| pgc33room_lstm_s6 | 119.0 | 39.7 | not reached | not reached |
| pgc33room_lstm_s7 | 114.5 | 52.3 | not reached | not reached |
| pgc33room_lstm_s8 | 98.2 | 25.9 | not reached | not reached |

## Alignment index of top-10 subspaces (scripted replay; 1 = same subspace)

| agents | comparison | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|---|
| before | familiar vs changed | 0.853 | 0.861 | +0.008 ± 0.008 | 0.34 | 5/8 |
| after | familiar vs changed | 0.861 | 0.859 | -0.002 ± 0.008 | 0.80 | 5/8 |
| after | before vs after familiar | 0.502 | 0.550 | +0.048 ± 0.029 | 0.14 | 7/8 |
| after | before vs after changed | 0.498 | 0.535 | +0.036 ± 0.029 | 0.25 | 6/8 |
| same | familiar vs changed | 0.857 | 0.844 | -0.014 ± 0.022 | 0.66 | 1/2 |
| same | before vs same familiar | 0.709 | 0.742 | +0.032 ± 0.006 | 0.11 | 2/2 |
| same | before vs same changed | 0.709 | 0.707 | -0.002 ± 0.019 | 0.93 | 1/2 |
| long | familiar vs changed | 0.857 | 0.848 | -0.009 ± 0.007 | 0.23 | 3/8 |
| long | before vs long familiar | 0.476 | 0.535 | +0.059 ± 0.037 | 0.16 | 6/8 |
| long | before vs long changed | 0.474 | 0.517 | +0.043 ± 0.038 | 0.29 | 6/8 |
| same_long | familiar vs changed | 0.852 | 0.850 | -0.002 ± 0.010 | 0.90 | 1/2 |
| same_long | before vs same long familiar | 0.653 | 0.627 | -0.026 ± 0.040 | 0.63 | 1/2 |
| same_long | before vs same long changed | 0.652 | 0.585 | -0.067 ± 0.036 | 0.31 | 0/2 |

## Variance concentration at 1 B against transfer (Spearman, description only)

| fine-tuning | agents | n | first-10 share vs jumpstart | first-10 share vs gain |
|---|---|---|---|---|
| after | gc100 | 8 | rho -0.24 (p 0.57) | rho +0.71 (p 0.047) |
| after | pgc33 | 8 | rho +0.29 (p 0.49) | rho +0.55 (p 0.16) |
| after | pooled | 16 | rho +0.06 (p 0.84) | rho +0.35 (p 0.18) |
| long | gc100 | 8 | rho -0.24 (p 0.57) | rho +0.62 (p 0.10) |
| long | pgc33 | 8 | rho +0.29 (p 0.49) | rho +0.64 (p 0.086) |
| long | pooled | 16 | rho +0.06 (p 0.84) | rho +0.54 (p 0.031) |

## Training score in the changed rooms during fine-tuning, per 25 M frames

| frames (M) | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|
| 1000–1025 | 20.8 | 24.1 | +3.3 ± 4.2 | 0.46 | 6/8 |
| 1025–1050 | 21.2 | 25.3 | +4.1 ± 4.1 | 0.35 | 5/8 |
| 1050–1075 | 21.5 | 26.8 | +5.3 ± 4.1 | 0.24 | 6/8 |
| 1075–1100 | 22.4 | 28.0 | +5.6 ± 4.7 | 0.28 | 5/8 |
| 1100–1125 | 20.4 | 29.4 | +9.0 ± 4.8 | 0.11 | 7/8 |
| 1125–1150 | 20.7 | 30.4 | +9.7 ± 4.5 | 0.067 | 7/8 |
| 1150–1175 | 21.2 | 30.3 | +9.1 ± 4.6 | 0.091 | 7/8 |
| 1175–1200 | 23.5 | 31.1 | +7.6 ± 4.6 | 0.14 | 6/8 |
| 1200–1225 | 24.5 | 31.7 | +7.2 ± 4.8 | 0.17 | 6/8 |
| 1225–1250 | 24.8 | 32.1 | +7.2 ± 5.0 | 0.19 | 6/8 |
| 1250–1275 | 25.6 | 33.2 | +7.7 ± 4.5 | 0.13 | 7/8 |
| 1275–1300 | 26.5 | 34.0 | +7.5 ± 4.8 | 0.16 | 6/8 |
| 1300–1325 | 25.8 | 35.2 | +9.4 ± 4.4 | 0.067 | 6/8 |
| 1325–1350 | 24.0 | 35.2 | +11.2 ± 3.6 | 0.016 | 7/8 |
| 1350–1375 | 26.4 | 35.4 | +9.0 ± 4.7 | 0.100 | 6/8 |
| 1375–1400 | 27.1 | 35.2 | +8.1 ± 4.7 | 0.13 | 6/8 |
| 1400–1425 | 27.7 | 36.2 | +8.5 ± 4.6 | 0.10 | 6/8 |
| 1425–1450 | 26.9 | 36.0 | +9.1 ± 4.6 | 0.088 | 6/8 |
| 1450–1475 | 27.1 | 35.5 | +8.4 ± 5.1 | 0.14 | 6/8 |
| 1475–1500 | 27.8 | 35.4 | +7.6 ± 4.6 | 0.14 | 6/8 |

## Score as a share of the same agent's familiar-room score at 1 B

| state | rooms | grid | PGC | difference | p | k/n |
|---|---|---|---|---|---|---|
| before | changed rooms (all 21) | 0.221 | 0.231 | +0.010 ± 0.023 | 0.67 | 5/8 |
| before | door | 0.161 | 0.169 | +0.008 ± 0.020 | 0.70 | 6/8 |
| before | obstacle | 0.450 | 0.480 | +0.030 ± 0.049 | 0.56 | 5/8 |
| before | door + obstacle | 0.116 | 0.124 | +0.008 ± 0.023 | 0.74 | 4/8 |
| before | goal moved (open room) | 0.545 | 0.536 | -0.010 ± 0.012 | 0.43 | 4/8 |
| before | new rooms (never trained) | 0.286 | 0.305 | +0.019 ± 0.027 | 0.50 | 5/8 |
| after | changed rooms (all 21) | 0.271 | 0.295 | +0.024 ± 0.045 | 0.61 | 4/8 |
| after | door | 0.221 | 0.255 | +0.034 ± 0.048 | 0.50 | 4/8 |
| after | obstacle | 0.475 | 0.524 | +0.049 ± 0.079 | 0.56 | 4/8 |
| after | door + obstacle | 0.189 | 0.214 | +0.025 ± 0.056 | 0.67 | 4/8 |
| after | goal moved (open room) | 0.441 | 0.428 | -0.013 ± 0.029 | 0.66 | 4/8 |
| after | new rooms (never trained) | 0.277 | 0.311 | +0.034 ± 0.052 | 0.54 | 4/8 |
| after | familiar room | 0.621 | 0.659 | +0.039 ± 0.070 | 0.60 | 4/8 |
| long | changed rooms (all 21) | 0.288 | 0.347 | +0.058 ± 0.040 | 0.18 | 7/8 |
| long | door | 0.250 | 0.308 | +0.058 ± 0.047 | 0.25 | 5/8 |
| long | obstacle | 0.504 | 0.604 | +0.100 ± 0.071 | 0.20 | 6/8 |
| long | door + obstacle | 0.208 | 0.274 | +0.067 ± 0.047 | 0.20 | 6/8 |
| long | goal moved (open room) | 0.420 | 0.457 | +0.036 ± 0.034 | 0.32 | 5/8 |
| long | new rooms (never trained) | 0.290 | 0.340 | +0.049 ± 0.062 | 0.46 | 5/8 |
| long | familiar room | 0.596 | 0.713 | +0.117 ± 0.083 | 0.20 | 4/8 |

## Lesions in the grid + PGC agents (score)

| state | rooms | intact | 169 PGCs silenced | 169 random grid cells silenced | PGC-specific cost (random − PGC) | p | n |
|---|---|---|---|---|---|---|---|
| before | familiar room | 106.1 | 79.2 | 98.4 | +19.2 ± 3.3 | < 0.001 | 8 |
| before | changed rooms (all 21) | 24.4 | 21.6 | 24.7 | +3.1 ± 0.6 | < 0.001 | 8 |
| after | familiar room | 69.4 | 57.1 | 65.7 | +8.6 ± 1.5 | < 0.001 | 8 |
| after | changed rooms (all 21) | 31.3 | 29.1 | 31.0 | +1.9 ± 0.5 | 0.004 | 8 |
| long | familiar room | 75.1 | 53.3 | 67.7 | +14.4 ± 1.8 | < 0.001 | 8 |
| long | changed rooms (all 21) | 36.8 | 30.7 | 35.0 | +4.3 ± 0.7 | < 0.001 | 8 |

Change of the PGC-specific cost, after − before: familiar room -10.6 ± 3.5 (p 0.018, lower in 7/8 agents); changed rooms (all 21) -1.2 ± 0.6 (p 0.077, lower in 6/8 agents)

Change of the PGC-specific cost, long − before: familiar room -4.7 ± 3.8 (p 0.25, lower in 6/8 agents); changed rooms (all 21) +1.2 ± 1.0 (p 0.28, lower in 2/8 agents)

## Matched-budget reference (seeds 1–2): score

| agent | state | familiar room | changed rooms | new rooms |
|---|---|---|---|---|
| gc100_s1 | 1 B | 94.0 | 23.3 | 28.9 |
| gc100_s1 | 1.25 B, all in the familiar room | 94.1 | 19.2 | 24.5 |
| gc100_s1 | 1.5 B, all in the familiar room | 99.0 | 22.7 | 31.9 |
| gc100_s1 | 1.25 B, last 250 M in the changed rooms | 64.5 | 21.2 | 23.0 |
| gc100_s1 | 1.5 B, last 500 M in the changed rooms | 55.8 | 22.6 | 25.2 |
| gc100_s2 | 1 B | 112.5 | 34.0 | 41.5 |
| gc100_s2 | 1.25 B, all in the familiar room | 112.0 | 37.7 | 45.7 |
| gc100_s2 | 1.5 B, all in the familiar room | 120.7 | 35.6 | 44.5 |
| gc100_s2 | 1.25 B, last 250 M in the changed rooms | 82.6 | 45.5 | 51.7 |
| gc100_s2 | 1.5 B, last 500 M in the changed rooms | 91.0 | 49.4 | 58.2 |
| pgc33_s1 | 1 B | 91.5 | 20.6 | 24.7 |
| pgc33_s1 | 1.25 B, all in the familiar room | 99.5 | 25.1 | 28.6 |
| pgc33_s1 | 1.5 B, all in the familiar room | 116.0 | 22.6 | 24.7 |
| pgc33_s1 | 1.25 B, last 250 M in the changed rooms | 73.7 | 23.7 | 28.1 |
| pgc33_s1 | 1.5 B, last 500 M in the changed rooms | 76.5 | 28.2 | 25.4 |
| pgc33_s2 | 1 B | 109.6 | 21.2 | 31.7 |
| pgc33_s2 | 1.25 B, all in the familiar room | 111.2 | 24.1 | 30.7 |
| pgc33_s2 | 1.5 B, all in the familiar room | 107.2 | 24.6 | 31.3 |
| pgc33_s2 | 1.25 B, last 250 M in the changed rooms | 61.0 | 28.1 | 25.0 |
| pgc33_s2 | 1.5 B, last 500 M in the changed rooms | 76.0 | 35.8 | 32.4 |
