# Significance of the seven reported PCA differences

Eight matched training seeds are the independent units. All tests are two-sided and the seven selected rows form one exploratory family. Differences are PGC minus grid. Holm correction is applied separately to both test methods; neither method is selected according to its result.

| Metric | Mean difference | Marginal 95% t CI | Paired t p | Holm t p | Exact sign-flip p | Holm exact p |
|---|---:|---|---:|---:|---:|---:|
| Variance in first 3 PCs | +5.603 percentage points | [3.661, 7.546] | 0.000248155 | 0.000992621 | 0.0078125 | 0.0546875 |
| Variance in first 10 PCs | +5.907 percentage points | [4.989, 6.826] | 1.28036e-06 | 8.96252e-06 | 0.0078125 | 0.0546875 |
| PCs for 50% variance | -2.875 PCs | [-3.573, -2.177] | 2.53737e-05 | 0.000152242 | 0.0078125 | 0.0546875 |
| PCs for 80% variance | -6.125 PCs | [-7.763, -4.487] | 4.7876e-05 | 0.00023938 | 0.0078125 | 0.0546875 |
| PCs for 90% variance | -4.500 PCs | [-7.874, -1.126] | 0.0160634 | 0.0481902 | 0.03125 | 0.09375 |
| PCs for 95% variance | -3.250 PCs | [-7.758, 1.258] | 0.131988 | 0.263976 | 0.132812 | 0.265625 |
| PCs for 99% variance | +0.625 PCs | [-5.218, 6.468] | 0.807572 | 0.807572 | 0.84375 | 0.84375 |

The paired-t analysis rejects equality for the first five rows after Holm correction. The exact sign-flip analysis rejects none after correction; its four smallest adjusted p-values are 0.0546875. This reflects the coarse resolution of only eight pairs: the smallest attainable two-sided exact p is 2/256, which becomes 7 × 2/256 = 0.0546875 at the first Holm step. This is not evidence that the effects are zero.

Paired t tests rely on independent, approximately normal seed differences for small-sample calibration. Exact sign-flip tests instead require within-pair exchangeability/sign symmetry under the null; they are not assumption-free. The confidence intervals above are unadjusted and should not be read as simultaneous intervals for all seven quantities.

These tests were requested after the spectra were inspected. They are exploratory, conditional on one shared replay, and do not establish toroidal structure, grid-cell tuning, dynamical stability, or why rewards differ. The correction does not cover every exploratory analysis performed elsewhere in the project.

Methods: [SciPy paired t test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [SciPy paired permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [Holm adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).

Reproduce: `python scripts/lstm_pca_significance.py`
