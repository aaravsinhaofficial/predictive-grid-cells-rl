# What contributes to the PCA variance difference?

The PGC models have stronger leading activity patterns in absolute magnitude,
and substantially more variance associated with shifts in average activity
between episodes. Removing those episode averages reduces, but does not
eliminate, the difference in PCA concentration. This gives the difference a
concrete representational interpretation; it does not by itself identify a
change in recurrent dynamics or the cause of improved navigation.

[Simple figure](variance_sources.png) · [PDF](variance_sources.pdf) ·
[Numerical results](summary.json) · [Current-state sensitivity](current_state_sensitivity.json)

All results reuse the same 3,393 saved moving states from 16 shared episodes,
for each of the eight grid and eight PGC+grid 1B LSTMs. No new policy rollouts,
weight changes, or policy training were performed.

## Main result

| Quantity, mean across eight seeds | Grid | PGC + grid |
|---|---:|---:|
| Original variance captured by first 10 PCs | 46.42% | 52.32% |
| Total variance associated with episode means | 20.96% | 30.20% |
| First 10 PCs after removing episode means and refitting PCA | 40.94% | 42.84% |
| First 10 PCs after equalizing individual neuron variance | 31.00% | 37.11% |
| Total absolute population variance | 19.566 | 24.276 |
| Absolute variance in first 10 PCs | 9.088 | 12.713 |
| Absolute variance in remaining 246 PCs | 10.478 | 11.562 |

Absolute population variance is the sum of the 256 neuron variances, in
squared hidden-activation units, with divisor n. Higher leading-PC fractions
are not produced by a simple reduction of the remaining variance: PGC leading
variance is larger in all eight pairs, and its remaining variance is also
larger on average. The equal-neuron-variance result shows that heterogeneous
neuron amplitudes alone do not account for the PCA difference.

The original first-10-PC gap is 5.91 percentage points; after episode mean
removal it is 1.89 points. This smaller residual is positive in seven of eight
pairs and remains significant in this exploratory five-comparison family.
The reduction is descriptive, not a causal percentage explained: both the
fitted PCA axes and the variance denominator change after demeaning.

## Exact variance accounting

For every model, subtract its overall mean from h. For each row, let B be its
episode's mean of the centered activity, and W be its deviation from that
episode mean. Because the within-episode deviations sum to zero,

`Cov(h) = Cov(B) + Cov(W)`.

Episode means are weighted by their actual sample counts. This identity holds
for the full covariance and for its projection onto any fixed PCA subspace.
Projecting onto each model's **original** first ten PCs gives:

| Contribution to original first-10-PC variance, as percent of total | Grid | PGC + grid |
|---|---:|---:|
| Differences between episode means | 16.76% | 25.75% |
| Deviations within episodes | 29.66% | 26.58% |
| Sum | 46.42% | 52.32% |

Thus the exact difference in the original PCA fractions is +8.99 percentage
points from the between-episode contribution and −3.08 points from the
within-episode contribution, adding to +5.91. A lower normalized within term
does not mean lower absolute within-episode activity: its denominator is also
different between models.

In practical terms, each episode's hidden-state cloud has an average location.
Those average locations differ more, relative to total activity variation,
in the PGC models. This is a statement about activity within each network's
own coordinates; neuron identities and PCA directions are not equated across
separately trained networks.

## Does ordinary position/movement variation account for the episode shifts?

A subsequent descriptive check fitted one common, modest quadratic basis of
current x/y, movement cosine/sine, facing cosine/sine, and speed, plus previous
action indicators. These 35 independent centered columns were projected out
of each model's activity by ordinary least squares. The basis includes no
goal, room, episode identity, actor output, or future variable.

The fit accounts for 19.34% of grid activity variance and 17.76% of PGC
activity variance. About 79% of the original between-episode mean-offset
energy remains in both conditions. Remaining between-episode variance, still
using the **original total variance** as denominator, is 16.52% for grid and
23.98% for PGC. Thus this simple current-state fit does not remove the larger
PGC offsets.

This is an in-sample sensitivity analysis, not a held-out prediction score or
complete occupancy adjustment. It cannot remove arbitrary nonlinear or
periodic tuning, visual image effects, or all sampled-history differences.
Its covariates also correlate with episode context, so it cannot uniquely
separate context from position/movement information.

## Statistical scope

Five contrasts were specified before running this decomposition: absolute
first-10-PC variance, absolute remaining variance, between-episode fraction,
first-10-PC fraction after episode demeaning, and first-10-PC fraction after
neuron standardization. Tests are paired over eight training seeds, with
two-sided t and exhaustive 256-sign tests, Holm-adjusted separately across
these five comparisons. The adjustment is within this exploratory analysis,
not global across the project's preceding investigations.

| Contrast | Holm paired-t p | Holm exact sign-flip p |
|---|---:|---:|
| Absolute first-10-PC variance | 0.000835 | 0.0391 |
| Absolute remaining variance | 0.0442 | 0.0547 |
| Between-episode fraction | 0.000381 | 0.0391 |
| First-10-PC fraction after episode demeaning | 0.0442 | 0.0391 |
| First-10-PC fraction after neuron standardization | 0.0000332 | 0.0391 |

No samples or episodes are counted as extra independent training seeds.
The sign-flip test assumes the null supports sign exchangeability of paired
differences. Confidence intervals in the JSON are marginal, not simultaneous.
Additional descriptive quantities and the later covariate check are not used
for further significance claims.

## What this establishes, and what remains unresolved

The PGC advantage in PCA concentration is associated substantially with
episode-dependent mean activity, plus a smaller within-episode difference.
It is not merely an unequal-neuron-amplitude effect. However, every episode
also has its own goal, room appearance, sampled positions/directions, and
history. **Episode-associated offsets are not uniquely goal memory.** Nor is
the supplied goal snapshot evidence of memory maintained internally without
external input. The previous held-out readout analysis did not establish
greater concentration of goal/direction information in the leading PCs.

PCA covariance discards temporal order. Randomly permuting the activity rows
leaves the spectrum unchanged; this was also checked numerically. A reliable
PCA difference therefore does not establish different transition dynamics,
attractor topology, stability, or computational usefulness. Existing separate
[input-swap](../lstm_code_swap_1B/README.md) and
[gate-rescue](../lstm_gate_rescue_1B/README.md) experiments provide evidence
about input-driven state updates and behavior, but do not show that this PCA
difference mediates the behavioral benefit.

## Reproduction and checks

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_variance_sources.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_variance_covariates.py
python scripts/plot_lstm_variance_sources.py
```

The core covariance decomposition's maximum relative error is 1.90e-15;
the maximum eigenvalue difference after row shuffling is 3.56e-15. An
independent SVD/direct-sums calculation reproduced all 16 decompositions
within 2.14e-14 and verified all five test results and adjustments. Source
recordings and replay were hashed before and after analysis and were
unchanged. The covariate projection was checked for residual orthogonality
and conservation of fitted-plus-residual squared activity.

Neuron standardization uses each unit's standard deviation and would exclude
units at or below 1e-10; all 256 units were retained in the primary check.
This changes the activity metric and is a sensitivity analysis, not a
replacement for the original PCA in raw actor-input coordinates.
