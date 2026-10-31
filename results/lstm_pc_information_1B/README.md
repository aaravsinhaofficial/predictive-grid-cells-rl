# Do leading PCs contain more useful goal or movement information?

This simple diagnostic reuses the saved activity from all eight matched pairs
of frozen 1B LSTMs. It fits linear readouts of remembered pickup position and
movement direction from the first 3, 10, 20, 50, 100, or all 256 PCs. It does
not rerun, retrain, or perturb the policies.

[Readout curves](pc_information.png) · [PDF](pc_information.pdf)

## Result

The specific claim of greater goal/direction information concentration in PGC
models' leading PCs is **not established by this test**. PGC models have lower
average goal-position readout error at both 10 and 256 PCs, but the error added
by restricting the readout to 10 PCs is similar. Their movement-direction
readout is worse on average at both sizes.

| Held-out error, mean across 8 seeds | Grid | PGC + grid |
|---|---:|---:|
| Goal position, 10 PCs | 73.06 cm | 63.28 cm |
| Goal position, all 256 PCs | 67.04 cm | 58.02 cm |
| Goal error added by using only 10 PCs | 6.02 cm | 5.26 cm |
| Movement direction, 10 PCs | 34.22° | 39.31° |
| Movement direction, all 256 PCs | 22.58° | 25.08° |
| Direction error added by using only 10 PCs | 11.63° | 14.22° |

Lower errors are better. The mean PGC-minus-grid difference in the loss from
compression is −0.75 ± 2.15 cm (SEM) for goal position and +2.59 ± 1.22° for
direction. These do not demonstrate that PGC leading PCs preserve more of the
information available to the full linear readout.

## Simple test, with episode holdouts

- Both tasks use the same 3,392 known-goal samples, spanning 16 episodes. The
  one unknown-goal sample is excluded. Every model sees the same sampled states.
- The goal target is the remembered pickup x/y position, not the physical goal
  center. Direction is predicted as cosine/sine and evaluated with wrapped
  angular error. Goal error is Euclidean RMSE.
- Four outer folds hold out entire episodes, including their goals and room
  layouts. Three inner episode folds choose ridge regularization separately
  for each task and PC count.
- PCA, centering, and one scalar feature scale are fitted using training data
  only in every inner and outer fold. Neurons and PCs are not individually
  rescaled, and PCA scores are not whitened.
- Ten PCs was selected as the primary comparison before computing the new
  readouts, following the preceding PCA discussion. All six planned counts are
  shown, and the 256-PC reference is recomputed in the identical pipeline.
- Ridge strengths are 0.01, 0.1, 1, 10, 100, 1,000, 10,000, and 100,000. The
  range extends the earlier decoder's upper bound because it was often chosen.
  The full-state numbers here therefore differ from that earlier analysis.
- The baseline predicts each outer training set's mean target on its test set.

## Primary statistical comparisons

Four contrasts were specified before obtaining results: 10-PC error and the
10-PC-minus-256-PC error penalty, each for goal and direction. Independent units
are the eight matched training seeds. All tests are two-sided; Holm correction
is applied over these four contrasts, separately for paired t tests and exact
paired sign-flip tests. These remain exploratory analyses on one shared replay.

| Contrast, PGC minus grid | Mean difference | Holm paired-t p | Holm exact p |
|---|---:|---:|---:|
| Goal error at 10 PCs | −9.77 cm | 0.0581 | 0.0938 |
| Goal compression penalty | −0.75 cm | 0.7372 | 0.7344 |
| Direction error at 10 PCs | +5.09° | 0.0681 | 0.1406 |
| Direction compression penalty | +2.59° | 0.1449 | 0.1563 |

None of these four comparisons passes 0.05 after correction with either test.
The full curves and other component counts are descriptive, not extra selected
significance tests. Per-seed errors, selected regularization, episode folds,
marginal confidence intervals, and all raw test results are in `summary.json`.

## Interpretation limits

This tests linear accessibility of two specific variables. A better readout
does not establish policy use or explain the behavioral advantage. Goal codes
are supplied externally, and movement direction is correlated with facing and
visual input in the scripted replay. Other variables or nonlinear readouts
could behave differently; they were not searched here.

The compression penalty is a practical decoding loss, not an information-
theoretic quantity. Negative penalties are allowed: discarding components can
improve finite-sample generalization. Higher variance in early PCs alone does
not establish more task-relevant information, a torus, or recurrent stability.

## Reproduce and verify

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/lstm_pc_information.py
python scripts/plot_lstm_pc_information.py
```

`predictions.npz` stores held-out predictions at each PC count, baselines,
targets, and sample indices. Source activity and replay files are hashed and
unchanged. Each sample is predicted exactly once, with disjoint episode folds.
An independent check matched the prefix-SVD computation against scikit-learn
PCA plus Ridge at 10 PCs, and ordinary Ridge at 256 PCs, to numerical precision.
