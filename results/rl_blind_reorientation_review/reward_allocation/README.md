# Where rewards and opportunities occur

**Read-only audit of every saved fixed evaluation endpoint.** All original episodes are retained, including those with no discovery or no admitted event. Clean evaluation always uses the final task’s 30–60-update turn delay; early-stage training has shorter delays. These are separate measurements.

In the saved blind evaluations with at least one goal, waiting/visible navigation accounts for **87.5–98.0% of goals**, while occupying 67.3–72.7% of updates. Whole-episode reward can therefore remain substantial while free blind navigation is weak.

| Model / endpoint / vision | Total goals/episode | Free | Forced | Waiting + visible | Free goals/100 free updates |
|---|---:|---:|---:|---:|---:|
| Grid / base / blind | 4.672 | 0.188 | 0.078 | 4.406 | 0.136 |
| Grid / base / seen | 5.984 | 0.672 | 0.062 | 5.250 | 0.514 |
| Grid / S1 / blind | 5.109 | 0.578 | 0.062 | 4.469 | 0.447 |
| Grid / S1 / seen | 6.312 | 0.844 | 0.047 | 5.422 | 0.652 |
| Grid / S2 / blind | 5.078 | 0.359 | 0.109 | 4.609 | 0.287 |
| Grid / S2 / seen | 6.000 | 0.625 | 0.078 | 5.297 | 0.485 |
| PGC / base / blind | 5.188 | 0.031 | 0.078 | 5.078 | 0.027 |
| PGC / base / seen | 6.328 | 0.594 | 0.078 | 5.656 | 0.529 |
| PGC / S1 / blind | 5.594 | 0.016 | 0.094 | 5.484 | 0.012 |
| PGC / S1 / seen | 6.922 | 0.688 | 0.141 | 6.094 | 0.563 |

Free means the agent-controlled 30-update test window. Forced includes both labelled turns, the forward probe, and the left-turn delay. Waiting + visible includes initial goal discovery, native visible navigation between events, and each event’s visible refresh. The JSON separates waiting and visible rewards.

| Model / endpoint / vision | Free updates % | Forced updates % | Waiting + visible % | Admitted | Completed without interruption | Reward-ended | Episode-ended |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grid / base / blind | 10.2 | 22.5 | 67.3 | 359 | 289 | 70 | 0 |
| Grid / base / seen | 9.7 | 22.0 | 68.3 | 359 | 253 | 106 | 0 |
| Grid / S1 / blind | 9.6 | 22.1 | 68.3 | 356 | 259 | 97 | 0 |
| Grid / S1 / seen | 9.6 | 22.2 | 68.2 | 356 | 245 | 111 | 0 |
| Grid / S2 / blind | 9.3 | 20.7 | 70.0 | 341 | 255 | 86 | 0 |
| Grid / S2 / seen | 9.6 | 21.7 | 68.8 | 341 | 251 | 90 | 0 |
| PGC / base / blind | 8.6 | 18.7 | 72.7 | 312 | 247 | 65 | 0 |
| PGC / base / seen | 8.3 | 18.9 | 72.8 | 312 | 215 | 97 | 0 |
| PGC / S1 / blind | 9.5 | 20.4 | 70.1 | 336 | 273 | 63 | 0 |
| PGC / S1 / seen | 9.0 | 20.6 | 70.4 | 336 | 232 | 104 | 0 |

A reward ends its event, including a successful free-navigation hit. “Completed without interruption” means the full event ended without a goal hit; it is not a success count. Exact counts/shares for all seven phases, free-window completion, waiting reasons, and every episode are retained in [the JSON](summary.json).

**Separate training counter differences, with warmup excluded:**

| Model / stage | Training delay range | Free goals | Free updates | Free goals/100 free updates | Forced goals/100 forced updates |
|---|---:|---:|---:|---:|---:|
| Grid / S1 | 2–4 | 363 | 233385 | 0.156 | 0.068 |
| Grid / S2 | 8–15 | 380 | 225706 | 0.168 | 0.049 |
| PGC / S1 | 2–4 | 33 | 219792 | 0.015 | 0.079 |

These training rates come from exact cumulative hit/phase-counter differences at fixed checkpoints, starting after warmup. They include training state noise and the stage’s own delay. Total training goal counts and their waiting/visible allocation are unavailable from these counters; recent score averages are not substituted. Training and clean evaluation also differ in policy evolution, state noise and environment samples, so their difference is not a controlled delay-only effect.

Reproduce with `OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/summarize_reorientation_reward_allocation.py`; add `--require-complete` after both four-stage runs finish. Only baseline and the four prescribed stage endpoints are considered; latest/intermediate checkpoints are ignored. No training settings or frozen mechanism sources are changed.
