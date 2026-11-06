# Where rewards and opportunities occur

**Read-only audit of every saved fixed evaluation endpoint.** All original episodes are retained, including those with no discovery or no admitted event. Clean evaluation always uses the final task’s 30–60-update turn delay; early-stage training has shorter delays. These are separate measurements.

In the saved blind evaluations with at least one goal, waiting/visible navigation accounts for **87.2–98.6% of goals**, while occupying 65.9–72.7% of updates. Whole-episode reward can therefore remain substantial while free blind navigation is weak.

| Model / endpoint / vision | Total goals/episode | Free | Forced | Waiting + visible | Free goals/100 free updates |
|---|---:|---:|---:|---:|---:|
| Grid / base / blind | 4.672 | 0.188 | 0.078 | 4.406 | 0.136 |
| Grid / base / seen | 5.984 | 0.672 | 0.062 | 5.250 | 0.514 |
| Grid / S1 / blind | 5.109 | 0.578 | 0.062 | 4.469 | 0.447 |
| Grid / S1 / seen | 6.312 | 0.844 | 0.047 | 5.422 | 0.652 |
| Grid / S2 / blind | 5.078 | 0.359 | 0.109 | 4.609 | 0.287 |
| Grid / S2 / seen | 6.000 | 0.625 | 0.078 | 5.297 | 0.485 |
| Grid / S3 / blind | 4.984 | 0.297 | 0.031 | 4.656 | 0.225 |
| Grid / S3 / seen | 6.531 | 0.688 | 0.078 | 5.766 | 0.558 |
| Grid / S4 / blind | 4.891 | 0.516 | 0.109 | 4.266 | 0.388 |
| Grid / S4 / seen | 5.938 | 0.750 | 0.125 | 5.062 | 0.562 |
| PGC / base / blind | 5.188 | 0.031 | 0.078 | 5.078 | 0.027 |
| PGC / base / seen | 6.328 | 0.594 | 0.078 | 5.656 | 0.529 |
| PGC / S1 / blind | 5.594 | 0.016 | 0.094 | 5.484 | 0.012 |
| PGC / S1 / seen | 6.922 | 0.688 | 0.141 | 6.094 | 0.563 |
| PGC / S2 / blind | 5.172 | 0.016 | 0.062 | 5.094 | 0.013 |
| PGC / S2 / seen | 6.719 | 0.812 | 0.047 | 5.859 | 0.683 |
| PGC / S3 / blind | 6.234 | 0.000 | 0.141 | 6.094 | 0.000 |
| PGC / S3 / seen | 7.578 | 0.719 | 0.094 | 6.766 | 0.596 |
| PGC / S4 / blind | 5.766 | 0.000 | 0.078 | 5.688 | 0.000 |
| PGC / S4 / seen | 7.297 | 0.625 | 0.094 | 6.578 | 0.457 |

Free means the agent-controlled 30-update test window. Forced includes both labelled turns, the forward probe, and the left-turn delay. Waiting + visible includes initial goal discovery, native visible navigation between events, and each event’s visible refresh. The JSON separates waiting and visible rewards.

| Model / endpoint / vision | Free updates % | Forced updates % | Waiting + visible % | Admitted | Completed without interruption | Reward-ended | Episode-ended |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grid / base / blind | 10.2 | 22.5 | 67.3 | 359 | 289 | 70 | 0 |
| Grid / base / seen | 9.7 | 22.0 | 68.3 | 359 | 253 | 106 | 0 |
| Grid / S1 / blind | 9.6 | 22.1 | 68.3 | 356 | 259 | 97 | 0 |
| Grid / S1 / seen | 9.6 | 22.2 | 68.2 | 356 | 245 | 111 | 0 |
| Grid / S2 / blind | 9.3 | 20.7 | 70.0 | 341 | 255 | 86 | 0 |
| Grid / S2 / seen | 9.6 | 21.7 | 68.8 | 341 | 251 | 90 | 0 |
| Grid / S3 / blind | 9.8 | 21.5 | 68.7 | 345 | 271 | 74 | 0 |
| Grid / S3 / seen | 9.1 | 20.8 | 70.0 | 345 | 239 | 106 | 0 |
| Grid / S4 / blind | 9.8 | 22.3 | 67.9 | 355 | 265 | 90 | 0 |
| Grid / S4 / seen | 9.9 | 22.6 | 67.5 | 355 | 254 | 101 | 0 |
| PGC / base / blind | 8.6 | 18.7 | 72.7 | 312 | 247 | 65 | 0 |
| PGC / base / seen | 8.3 | 18.9 | 72.8 | 312 | 215 | 97 | 0 |
| PGC / S1 / blind | 9.5 | 20.4 | 70.1 | 336 | 273 | 63 | 0 |
| PGC / S1 / seen | 9.0 | 20.6 | 70.4 | 336 | 232 | 104 | 0 |
| PGC / S2 / blind | 9.2 | 19.9 | 70.9 | 323 | 265 | 58 | 0 |
| PGC / S2 / seen | 8.8 | 20.2 | 71.0 | 323 | 219 | 104 | 0 |
| PGC / S3 / blind | 9.3 | 20.2 | 70.5 | 349 | 268 | 81 | 0 |
| PGC / S3 / seen | 8.9 | 20.6 | 70.5 | 349 | 227 | 122 | 0 |
| PGC / S4 / blind | 10.8 | 23.2 | 65.9 | 374 | 312 | 62 | 0 |
| PGC / S4 / seen | 10.1 | 22.9 | 66.9 | 374 | 267 | 107 | 0 |

A reward ends its event, including a successful free-navigation hit. “Completed without interruption” means the full event ended without a goal hit; it is not a success count. Exact counts/shares for all seven phases, free-window completion, waiting reasons, and every episode are retained in [the JSON](summary.json).

**Separate training counter differences, with warmup excluded:**

| Model / stage | Training delay range | Free goals | Free updates | Free goals/100 free updates | Forced goals/100 forced updates |
|---|---:|---:|---:|---:|---:|
| Grid / S1 | 2–4 | 363 | 233385 | 0.156 | 0.068 |
| Grid / S2 | 8–15 | 380 | 225706 | 0.168 | 0.049 |
| Grid / S3 | 15–30 | 481 | 219574 | 0.219 | 0.036 |
| Grid / S4 | 30–60 | 624 | 210091 | 0.297 | 0.031 |
| PGC / S1 | 2–4 | 33 | 219792 | 0.015 | 0.079 |
| PGC / S2 | 8–15 | 38 | 219090 | 0.017 | 0.063 |
| PGC / S3 | 15–30 | 20 | 218048 | 0.009 | 0.046 |
| PGC / S4 | 30–60 | 15 | 216204 | 0.007 | 0.036 |

These training rates come from exact cumulative hit/phase-counter differences at fixed checkpoints, starting after warmup. They include training state noise and the stage’s own delay. Total training goal counts and their waiting/visible allocation are unavailable from these counters; recent score averages are not substituted. Training and clean evaluation also differ in policy evolution, state noise and environment samples, so their difference is not a controlled delay-only effect.

Reproduce with `OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/summarize_reorientation_reward_allocation.py`; add `--require-complete` after both four-stage runs finish. Only baseline and the four prescribed stage endpoints are considered; latest/intermediate checkpoints are ignored. No training settings or frozen mechanism sources are changed.
