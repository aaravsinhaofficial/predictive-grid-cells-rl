# Remembering an actual travel direction

**A trace of travel direction survives in the original networks. Disturbances toward and away from its curve persist almost as strongly as differences around the curve.**

This test targets what the PGC input actually encodes: the direction of the last movement. We used the real arena, renderer, spatial maps, CNN and trained LSTM. For each of 768 directions, the agent moved forward toward the same endpoint during fifteen visible observations. The starts were chosen using the original smoothed-velocity physics. Every path avoided the real goal and walls.

We then explicitly stopped translation, made vision black and gave the same turn command. Movement direction stayed fixed while body heading changed. During the memory test, only the current PGC slots were replaced with ordinary grid values. That made the complete input identical across directions. The real saved goal code stayed available.

| Model | Radial / tangential gain, raw state | Radial / tangential gain, rescaled state |
| --- | ---: | ---: |
| Grid, seed 1 | 1.023 | 1.348 |
| Grid, seed 2 | 1.007 | 1.014 |
| PGC + grid, seed 1 | 1.011 | 0.984 |
| PGC + grid, seed 2 | 1.005 | 1.045 |

The table measures two seconds of recovery after two seconds without the predictive current cue. A ring's selective attraction would give smaller radial than tangential gains. Here, radial mistakes mostly persist with the angular trace. The released families also move several radii away from their earlier fixed curves. None of the four original models passes the frozen released-ring screen.

Information remains measurable: an analysis readout refitted at four seconds recovers travel angle with **1.0–1.95° error** on held-out angles. The original fixed readout has **80–89° error**. These readouts never supply inputs or actions to the agent. Native predictive-cue trajectories are saved separately and labelled input-driven.

All endpoints agreed within **1.7–1.9e-7 metres** before a bounded numerical snap to the exact shared position. Velocity before and after the explicit stop, physical phase uniformity, input identity and all candidate outcomes are saved. The stop belongs to this controlled dynamical experiment; later navigation tests must use unrestricted original physics.

The [protocol](protocol.json) was frozen before these results. Model directories contain the full 512-coordinate state sequences and the exact ordinary/native current-input vectors needed for independent single-cue-pulse tests. Two physical controls passed: the calculated forward distance matches the native environment, and the all-angle clearance rule excludes wall or goal intersections.

```bash
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python -m pytest tests/test_lstm_travel_memory_probe.py -q
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_travel_memory_probe.py
```

The continued models use freshly collected histories from their own trained policies and the same primary four-second horizon. Their outcomes belong in a separate directory.
