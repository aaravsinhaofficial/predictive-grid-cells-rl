# Final blackout-trained models: coherent travel memory

The final Grid and PGC + grid models each supplied two new, physically eligible contexts from actual policy rollouts. Real forward trajectories established last-movement direction and ended at a common position. After an explicit stop, replacing the current predictive slots with ordinary-grid values gave every phase the same input. The saved native goal code remained unchanged.

The primary test covered two-to-four seconds, inside the trained blackout duration.

| Final model | Context | Departure from fixed curve, in radii | Radial / tangential gain |
| --- | ---: | ---: | ---: |
| Grid | 0 | 4.60 | 1.06 |
| Grid | 1 | 2.64 | 1.09 |
| PGC + grid | 0 | 4.03 | 1.29 |
| PGC + grid | 1 | 2.33 | 0.97 |

The table uses raw full-state coordinates. Every case was also checked after separate hidden/cell RMS scaling. **All four released families fail the frozen ring-candidate screen.** Radial mistakes persist roughly as strongly as changes around the family, while the full-state family changes location.

Travel information survives: at four seconds, a refitted readout estimates the original direction with **1.16–1.51° error** on held-out angles. The readout fixed before darkness has **70–83° error**. Native predictive-cue trajectories are saved as an explicitly input-driven comparison.

The [Grid protocol/results](gc100/summary.json) and [PGC protocol/results](pgc33/summary.json) preserve both contexts. Exact phase-labelled states and input vectors are available in each model's `context*_states.npz` for the independent cue-pulse tests. The contexts were collected from the final weights; no original-model recurrent state was reused.

```bash
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_travel_memory_probe.py --cohort continued --checkpoint results/ring_cloud_artifacts/ring-20261105-vision-grid-01/final.pt --out results/lstm_travel_memory/continued/gc100
OPENBLAS_NUM_THREADS=2 /opt/anaconda3/bin/python scripts/lstm_travel_memory_probe.py --cohort continued --checkpoint results/ring_cloud_artifacts/ring-20261105-vision-pgc-01/final.pt --out results/lstm_travel_memory/continued/pgc33
```

This pilot contains one trained model per population and two nested contexts. Navigation usefulness is assessed separately by the fixed blind/visible state-reset and current-cue replacement evaluations.
