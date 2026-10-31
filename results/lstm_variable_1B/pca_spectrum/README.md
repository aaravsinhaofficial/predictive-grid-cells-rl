# PCA spectrum of all eight seed pairs

PCA was refitted from the saved 256-dimensional hidden vectors `h`, using float64 full SVD, mean centering, and no neuron-wise scaling. The variable replay uses 3,393 matched moving states per model; the earlier fixed replay uses its original 1,185 selected states. Policies were not rerun or retrained. No UMAP is used.

[Per-seed cumulative curves](all_seeds_cumulative_variance.png) · [Spectrum summary](pca_spectrum_summary.png) · [Exact component counts](components_by_seed.png)

The companion projection script writes [all 16 PC1–3 clouds colored by arena x](all_seeds_pca.png), [arena y](all_seeds_pc1_3_y.png), and [movement direction](all_seeds_pc1_3_direction.png), plus [PC4–6](all_seeds_pc4_6_x.png) and [PC7–9](all_seeds_pc7_9_x.png). These are fixed consecutive component blocks, not projections chosen to resemble a particular shape.

## Variable replay: summary across eight seeds

| Quantity | Grid only, mean ± SEM | PGC + grid, mean ± SEM | PGC − grid, paired mean ± SEM |
|---|---:|---:|---:|
| Variance captured by PC1 (%) | 10.37 ± 0.58 | 12.87 ± 0.45 | 2.50 ± 0.83 |
| Variance captured by first 3 PCs (%) | 23.81 ± 0.46 | 29.42 ± 0.75 | 5.60 ± 0.82 |
| Variance captured by first 6 PCs (%) | 36.08 ± 0.38 | 42.03 ± 0.66 | 5.95 ± 0.54 |
| Variance captured by first 10 PCs (%) | 46.42 ± 0.38 | 52.32 ± 0.57 | 5.91 ± 0.39 |
| Variance captured by first 20 PCs (%) | 60.82 ± 0.37 | 65.87 ± 0.37 | 5.05 ± 0.22 |
| PCs for 50% variance | 12.38 ± 0.26 | 9.50 ± 0.33 | -2.88 ± 0.30 |
| PCs for 80% variance | 49.38 ± 0.60 | 43.25 ± 0.56 | -6.12 ± 0.69 |
| PCs for 90% variance | 82.00 ± 0.85 | 77.50 ± 0.98 | -4.50 ± 1.43 |
| PCs for 95% variance | 112.88 ± 1.03 | 109.62 ± 1.65 | -3.25 ± 1.91 |
| PCs for 99% variance | 164.88 ± 1.98 | 165.50 ± 2.90 | 0.62 ± 2.47 |
| Participation ratio | 31.33 ± 0.69 | 23.84 ± 0.78 | -7.49 ± 1.15 |

SEM is across the eight independent training seeds, not across time samples. Comparisons are descriptive, without p-value screening.

## Every model

| Seed | Condition | PC1–3 (%) | PC1–6 (%) | PC1–10 (%) | PCs: 50% | 80% | 90% | 95% | 99% |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Grid only | 24.86 | 35.14 | 44.88 | 13 | 52 | 84 | 112 | 158 |
| 1 | Grid + 33% PGC | 26.53 | 39.39 | 49.62 | 11 | 44 | 76 | 105 | 155 |
| 2 | Grid only | 24.30 | 36.81 | 47.21 | 12 | 49 | 81 | 112 | 165 |
| 2 | Grid + 33% PGC | 27.11 | 40.55 | 51.46 | 10 | 42 | 74 | 105 | 159 |
| 3 | Grid only | 22.63 | 34.69 | 45.63 | 13 | 48 | 79 | 109 | 157 |
| 3 | Grid + 33% PGC | 29.54 | 41.87 | 52.12 | 10 | 42 | 76 | 107 | 159 |
| 4 | Grid only | 25.21 | 37.29 | 48.08 | 11 | 48 | 81 | 113 | 168 |
| 4 | Grid + 33% PGC | 30.90 | 43.96 | 53.95 | 9 | 43 | 79 | 112 | 168 |
| 5 | Grid only | 23.59 | 35.69 | 45.59 | 13 | 52 | 86 | 117 | 169 |
| 5 | Grid + 33% PGC | 30.25 | 41.79 | 51.90 | 10 | 44 | 77 | 108 | 165 |
| 6 | Grid only | 24.49 | 37.28 | 47.27 | 12 | 49 | 84 | 117 | 172 |
| 6 | Grid + 33% PGC | 33.14 | 45.39 | 55.06 | 8 | 41 | 76 | 109 | 166 |
| 7 | Grid only | 21.21 | 35.00 | 46.09 | 13 | 49 | 81 | 113 | 169 |
| 7 | Grid + 33% PGC | 28.34 | 41.67 | 52.38 | 9 | 46 | 83 | 119 | 181 |
| 8 | Grid only | 24.23 | 36.71 | 46.59 | 12 | 48 | 80 | 110 | 161 |
| 8 | Grid + 33% PGC | 29.51 | 41.62 | 52.11 | 9 | 44 | 79 | 112 | 171 |

## Interpretation and limitations

The inspected 3D projections do not show a consistently clear toroidal surface or regular lattice across the seed pairs. PGC models concentrate more variance in the leading PCs: all eight need fewer PCs to reach 50% and 80% variance. This difference weakens at higher thresholds; at 99%, the group means are nearly identical and the seed-wise direction is mixed. Thus the supported result is greater concentration of linear variance, not greater gridness.

A principal component is a linear axis of variation, ordered from largest variance to smallest. Cumulative variance is the fraction of total squared centered activity retained by the first k axes. For example, 90% variance means that projecting onto those axes discards 10% of the sample variance; it does not mean 90% task accuracy or 90% biological explanation.

Participation ratio is 1 / sum(p_i²), where p_i is each component’s variance fraction. It is a spectrum-weighted measure of linear dimensionality, not an estimate of a manifold’s intrinsic topological dimension.

Greater variance in early PCs means a more concentrated linear spectrum. It does not establish grid-cell tuning, a torus, a recurrent attractor, stability, or better task performance. A curved manifold can require many linear PCs. Mixing directions, goals, vision, and histories can also broaden the spectrum.

The fixed-to-variable contrast is descriptive: the probes also differ in sampled trajectory, room layouts, sample count, episode duration/reset structure, and direction/goal coverage. It cannot attribute a change to one factor alone. These fits summarize the sampled activity, not held-out prediction performance.

The variable replay is scripted and effectively post-discovery. Movement and facing are correlated, goal snapshots are externally supplied, and multiple goals and rooms are pooled. These conditions limit mechanistic conclusions.

## Reproduce and audit

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_pca_spectrum.py
```

`summary.json` stores all 256 variance fractions, per-model counts and metrics, paired comparisons, input hashes, and numerical checks. Each variable-model NPZ stores the first 12 scores and the full fitted PCA basis/mean/spectrum. Row order matches `../replay.npz` → `sample_index`.

Validation checks finite values, sorted/nonnegative eigenvalues, variance conservation, basis orthonormality, full-rank reconstruction, and agreement with the earlier stored variance spectra. Source files are read-only and hashed again after the run.
