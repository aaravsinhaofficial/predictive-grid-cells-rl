# What distinguishes grid-only and grid + PGC models at 1B frames?

Follow-up: [exact code swaps and causal RNN interventions](../lstm_code_swap_1B/README.md) now separate current-code from goal-code effects and test [behavioral rescue through LSTM components](../lstm_gate_rescue_1B/README.md). These provide more specific causal evidence than the exploratory weight/PCA comparisons below.

The strongest supported explanation is **learned use of a joint position-and-movement-direction code during goal-directed navigation**. The mixed models preferentially weight their predictive inputs, and existing interventions show that their behavior depends on those inputs and their shifts. We did not find a reliable improvement in linear position decoding or a reliable change in PCA dimensionality.

This is an exploratory investigation of eight matched seed pairs. New computations use the frozen checkpoints and saved hidden activity; behavioral intervention scores are reanalyzed from the existing 200-episode evaluations in [`results_ext/extension.json`](../../results_ext/extension.json). No new behavioral evaluations or policy training were performed.

[Comparison figure](mechanism_comparison.png) · [PDF](mechanism_comparison.pdf) · [All numerical results](summary.json) · [Analysis script](../../scripts/lstm_mechanism_analysis.py)

## 1. The input code differs in a way the original PCA probe largely held fixed

The grid-only code consists of 512 rates `g_i(x)`. The mixed code preserves the same phases, orientations, spacings, and number of cells, but evaluates 169 cells at `g_i(x + 0.20 u)`, where `u` is the unit direction of movement. The other 343 cells still encode current position. Thus, the mixed population combines information about current position and a point 20 cm along the current movement direction.

The shift is spatial. At 0.35 m/s it corresponds to about 0.57 seconds of straight continuation, but it is not a prediction of the actual future trajectory through a turn. It also does not directly encode speed. The environment retains the last valid movement direction when nearly stationary, and movement direction can differ from facing direction.

At the same physical position, the ordinary code is identical for opposite movement directions. The mixed code changes: across 128 shared random position/direction probes per seed, the RMS population-rate difference between opposite directions averages **0.211** on the 0–1 rate scale, versus exactly **0** for grid-only.

A numerical local derivative check gives rank **2** for the grid-only input as a function of `(x, y, movement angle)` and rank **3** for the mixed input, at the median sampled state in every seed. On average, 81.6% of the norm of the mixed code's direction derivative remains after projecting out the two position derivatives. These are measurements of the analytic input representation, not of the LSTM's intrinsic dimensionality or topology.

In the earlier PCA plot, movement was nearly fixed eastward. With fixed `u`, the predictive shift becomes a constant change of each affected cell's spatial phase. The experiment therefore largely removed the movement-dependent variation that distinguishes the codes. A fixed goal snapshot also prevents it from measuring how hidden activity varies across remembered goals. Neither future-position decoding nor goal-vector decoding can be isolated from ordinary position decoding with that single fixed-direction, fixed-goal replay.

## 2. The trained LSTMs give predictive cells consistently larger input weights

For each input cell, we measured the Euclidean norm of its weights into all four LSTM gates. We compared the mean norm of the 169 predictive slots with that of the 343 ordinary slots. The grid-only counterpart uses exactly the same designated indices within each seed, controlling for cell identity and module allocation.

| Input pathway | Ratio in grid-only | Ratio in mixed models | Mixed-versus-grid paired p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Current population code | 0.997 | **1.168** | 2.30e-9 | 1.38e-8 |
| Remembered goal code | 0.997 | **1.155** | 1.76e-7 | 8.80e-7 |

The direction of both effects is consistent across all eight seed pairs. In mixed models, predictive cells have approximately **17% larger weights per cell for current input** and **16% larger weights per cell for goal memory**. This is a clear learned parameter difference, although weight magnitude alone does not measure functional importance: activation statistics, recurrent state, and gate nonlinearities also matter. The interventions below provide the behavioral evidence.

## 3. Interventions show specific dependence on predictive features

These scores are averages over the eight frozen 1B models per condition. The ordinary-cell lesion is the mean of five module-stratified draws of 169 **ordinary grid cells**, not arbitrary cells that could include PGCs.

| Existing evaluation | Grid only | Grid + 33% PGC |
|---|---:|---:|
| Intact | 96.78 | **107.12** |
| Goal memory removed | 39.17 | 39.59 |
| Current population code removed | 42.93 | 45.32 |
| Predictive shift set to zero in current and goal codes | 96.78 | 91.38 |
| All 169 PGCs silenced in current and goal codes | 96.78 | 76.54 |
| 169 ordinary cells silenced in current and goal codes | 88.34 | 94.72 |

Grid-only has no PGCs, so its de-shift and PGC-lesion evaluations are unchanged by definition.

- **Removing the shift costs the mixed model 15.74 points**, 95% paired t interval **[10.90, 20.58]**. All eight seeds lose performance. Paired t p = 0.000117; exact two-sided sign-flip p = 0.0078125; Holm-adjusted t p = 0.000705.
- **Silencing PGCs costs 30.58 points**, whereas silencing the same number of ordinary cells costs 12.40. The **PGC-specific excess cost is 18.18 points**, 95% interval **[13.43, 22.93]**, positive in all eight seeds. Paired t p = 0.0000411; exact sign-flip p = 0.0078125; Holm-adjusted t p = 0.000287.
- The intact between-group gap is **10.34 points**, p = 0.0262. Without goal memory, its point estimate becomes **0.42 points**, p = 0.806. The difference in goal-lesion costs is 9.92 points, p = 0.0186 before correction and **0.0744 after correction** within the exploratory behavioral family. This supports a homing interpretation descriptively, but the nonsignificant residual gap does not prove equivalence or formally establish mediation.

These effects show that the trained mixed agents use the predictive cells and their shifted code format. De-shifting or silencing also changes the inputs relative to training, so these interventions do not by themselves prove that forward prediction is the unique cause of the between-model performance gain. Both current and remembered-goal codes are changed together in these PGC-specific probes.

The repository's previous behavioral summary independently reports that at 1B, mixed LSTMs have **+0.070 path efficiency**, **+1.22 goals per 1,000 post-discovery steps**, and approximately the same first-goal time, **467 steps** in both conditions. Those numbers are taken from [README, extended-training results](../../README.md#37-longer-training-widens-the-gap-which-then-holds-with-vision), not recomputed from raw episode records here. They place the advantage primarily in returning to a known goal, rather than discovering it initially.

## 4. We did not establish a better general position representation

We fitted linear ridge readouts of `(x, y)` from all 256 hidden units, using the existing matched replay. Four outer folds hold out contiguous bands of sweep lanes, with one additional sampled lane excluded on either side as a guard. Ridge penalties are selected using three inner spatial folds, and all centering, scalar feature normalization, and PCA are fitted exclusively to training data. Adjacent samples along a lane never appear in both train and test.

| Hidden-state measure | Grid only | Mixed | Paired p |
|---|---:|---:|---:|
| Position RMSE, all 256 units | 40.3 cm | 36.7 cm | 0.521 |
| Position RMSE, first 3 training-fitted PCs | 53.5 cm | 51.1 cm | 0.494 |
| Variance captured by first 3 PCs | 56.5% | 59.6% | 0.167 |
| Participation ratio of variance spectrum | 6.92 | 6.43 | 0.429 |

The participation ratio measures concentration of variance, not the number of physical variables encoded. None of these four comparisons is significant, even before multiple-comparison correction. Position RMSE is Euclidean error on held-out spatial bands, including extrapolation at the edges; it is not a direct estimate of the agent's localization error during behavior.

Across seeds, the change in first-three-PC variance correlates with the performance gap at r = 0.49, p = 0.219; the position-RMSE change correlates at r = −0.61, p = 0.112. These eight-seed correlations are descriptive and uncorrected. They do not establish that compression or improved position decoding explains the performance advantage. No representation-to-performance correlation examined here reached nominal significance.

The existing topology screen also found no torus-compatible signature in either set of LSTM clouds under this sampling protocol; see [that analysis and its limits](../lstm_topology_1B/README.md). It does not supply a population distinction explaining the gap.

## Interpretation and the remaining mechanistic question

A plausible account is that the mixed code gives the policy a useful position-and-movement relationship directly in its spatial features. Both its current input and its goal snapshot contain predictive features, and the trained networks give those features larger weights. The behavioral benefit appears when using a remembered goal, and performance falls when the learned predictive format is disrupted. This is stronger evidence than the small PCA trend.

What remains unresolved is the actual computation: whether the gain comes mainly from instantaneous movement information, a more useful current-to-goal comparison, goal-entry snapshot geometry, or easier learning of the policy. The saved fixed-goal activity cannot separate these possibilities. Earlier 300M controls with explicit velocity, position-only PGC goal memory, and shifted ordinary-grid goal memory were inconclusive; they also do not directly settle the mechanism at 1B. The long retrospective-code runs make a universal claim that only forward shifts can help premature.

The most discriminating next behavioral experiment is a **current-shift × goal-shift factorial intervention** at 1B: intact, current code de-shifted only, goal snapshot de-shifted only, and both de-shifted, with matched episodes across all eight seeds. It should record path efficiency, post-discovery goal rate, and first-discovery time. Matching shifts on/off still changes the trained input distribution; a stronger causal account would combine those interventions with trained controls. A complementary shared replay should independently vary current position, movement direction, goal position, and goal-arrival direction before testing held-out decoding of goal-relative direction and action-relevant variables.

## Reproduction and statistical scope

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_mechanism_analysis.py
```

The script saves all seed-level outcomes, chosen decoder penalties and folds, out-of-fold predictions, exact sign-flip statistics, confidence intervals, and checkpoint hashes. Every checkpoint hash matched the source PCA analysis. Statistical replication is over **eight training seeds**, not 1,185 temporally correlated hidden states or 200 episodes. The intact score is the project's original endpoint; the new behavioral comparisons form one nine-test exploratory family and the representation comparisons form one six-test family, with Holm correction within each. The much smaller parametric p-values for consistent effects should be read alongside the finite-resolution exact sign-flip results.
