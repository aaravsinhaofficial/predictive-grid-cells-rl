# Physical opportunity in the fixed40M task

All original episodes and every free window are retained. Near/far strata are descriptive checks; the bounds below do not incorporate the actual native motor-noise tape or turning trajectory.

| Model / stage / vision | Free windows | Start ≤0.5m: hits/windows | Start ≤0.7m: hits/windows | Start >1.0m | Mean free path |
|---|---:|---:|---:|---:|---:|
| gc100 / 0 / blind_native | 301 | 10/74 | 12/132 | 99 | 0.155m |
| gc100 / 0 / seen_native | 296 | 27/71 | 39/128 | 107 | 0.319m |
| gc100 / 1 / blind_native | 296 | 26/70 | 35/127 | 106 | 0.453m |
| gc100 / 1 / seen_native | 299 | 39/77 | 52/141 | 98 | 0.292m |
| gc100 / 2 / blind_native | 278 | 15/66 | 19/124 | 79 | 0.340m |
| gc100 / 2 / seen_native | 291 | 23/64 | 34/114 | 103 | 0.289m |
| gc100 / 3 / blind_native | 290 | 12/67 | 16/134 | 90 | 0.330m |
| gc100 / 3 / seen_native | 283 | 27/74 | 40/134 | 94 | 0.282m |
| gc100 / 4 / blind_native | 298 | 13/49 | 27/133 | 81 | 0.494m |
| gc100 / 4 / seen_native | 302 | 28/65 | 44/138 | 96 | 0.319m |
| pgc33 / 0 / blind_native | 249 | 2/54 | 2/112 | 84 | 0.052m |
| pgc33 / 0 / seen_native | 253 | 22/62 | 34/106 | 81 | 0.288m |
| pgc33 / 1 / blind_native | 274 | 1/57 | 1/116 | 101 | 0.032m |
| pgc33 / 1 / seen_native | 276 | 26/51 | 42/103 | 101 | 0.277m |
| pgc33 / 2 / blind_native | 266 | 1/60 | 1/117 | 85 | 0.027m |
| pgc33 / 2 / seen_native | 271 | 31/70 | 45/110 | 92 | 0.225m |
| pgc33 / 3 / blind_native | 268 | 0/62 | 0/111 | 95 | 0.028m |
| pgc33 / 3 / seen_native | 273 | 26/55 | 42/111 | 94 | 0.279m |
| pgc33 / 4 / blind_native | 312 | 0/61 | 0/125 | 113 | 0.028m |
| pgc33 / 4 / seen_native | 307 | 29/72 | 37/123 | 117 | 0.290m |

The goal radius is 0.3 m. A 0.5 m starting distance leaves 0.2 m to the goal disk; 0.7 m leaves 0.4 m. At 0.35 m/s, a 2 s free window has a generous 0.7 m noiseless path budget before acceleration and turning costs. Native Gaussian motor noise prevents using this nominal budget as a strict impossibility proof.

At the fixed final PGC endpoint, 125 blind windows begin within 0.7 m and 61 within 0.5 m, yet none hits the goal; the near-window mean path is about 0.027 m. Distance alone therefore does not explain the failed behavior. The seen PGC condition has 37 hits in 123 windows within 0.7 m, although later seen/blind paths and window starts can differ.

Conversely, 113 of 312 final PGC blind windows begin beyond 1 m, exceeding the generous noiseless 2 s budget. This reduces the fraction of trials that can teach successful local navigation. Three of the four separate controlled mechanism contexts are also rigorously opportunity-limited under their saved noise tapes; that exact bound applies to those controlled contexts only.

The maximum 129-step reorientation chain fits inside the 160-step BPTT segment. Discounting at 0.99 gives 0.4088 weight to a reward 89 updates after the cue and 0.2763 at 128 updates; credit is present but attenuated. These checks favor an incentive/exploration bottleneck over a claim that the LSTM architecture cannot hold useful direction.

[Every original episode/window and source hash](summary.json) · [Reward allocation and training-stage exposures](../reward_allocation/README.md) · [Exact controlled-context opportunity audit](../../lstm_reorientation_mechanism/final/independent_artifact_audit.json)

Reproduce with `python scripts/summarize_reorientation_task_opportunity.py`. No experiment or frozen training/analysis source is changed.
