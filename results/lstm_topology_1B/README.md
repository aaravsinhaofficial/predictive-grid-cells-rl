# Persistent homology of 1B LSTM activity

Analyzed all 16 frozen LSTMs (8 matched seed pairs), using the same saved eastward activity as the PCA analysis. No UMAP, neuron rescaling, PCA projection, retraining, or new model rollouts were used.

**Result: 0 of 16 LSTM activity clouds show the coexisting torus-compatible signature.** The stricter global-count screen also finds 0/16. This result is conditional on the sampled states and analysis below; it does not establish that a model has no toroidal representation.

At 512 landmarks, 0/16 LSTM clouds have a common interval for the dominant features with raw Betti counts (1,2,1). This is an observed-data sensitivity check, not a second null-calibrated test.

[All-seed barcodes](all_seeds_barcodes.png) · [Signature summary](torus_signature.png) · [Controls](controls_barcodes.png)

| Dataset | Significant H1 / H2 bars | p: second H1 | p: strongest H2 | Coexisting signature | Strict global count |
|---|---:|---:|---:|---|---|
| Seed 1 · Grid only | 0 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Seed 1 · Grid + 33% PGC | 0 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Seed 2 · Grid only | 0 / 0 | 0.99 | 1.00 | Not detected | Does not pass |
| Seed 2 · Grid + 33% PGC | 0 / 0 | 1.00 | 0.71 | Not detected | Does not pass |
| Seed 3 · Grid only | 0 / 0 | 0.97 | 1.00 | Not detected | Does not pass |
| Seed 3 · Grid + 33% PGC | 0 / 0 | 1.00 | 0.68 | Not detected | Does not pass |
| Seed 4 · Grid only | 1 / 0 | 1.00 | 0.86 | Not detected | Does not pass |
| Seed 4 · Grid + 33% PGC | 0 / 0 | 1.00 | 0.76 | Not detected | Does not pass |
| Seed 5 · Grid only | 0 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Seed 5 · Grid + 33% PGC | 0 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Seed 6 · Grid only | 0 / 0 | 1.00 | 0.99 | Not detected | Does not pass |
| Seed 6 · Grid + 33% PGC | 0 / 0 | 0.99 | 1.00 | Not detected | Does not pass |
| Seed 7 · Grid only | 0 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Seed 7 · Grid + 33% PGC | 1 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Seed 8 · Grid only | 0 / 0 | 1.00 | 0.97 | Not detected | Does not pass |
| Seed 8 · Grid + 33% PGC | 0 / 0 | 1.00 | 1.00 | Not detected | Does not pass |
| Analytic grid module | 2 / 1 | 0.01 | 0.01 | Detected | Pass |
| Analytic PGC + grid module | 2 / 1 | 0.01 | 0.01 | Detected | Pass |
| Known synthetic torus | 3 / 1 | 0.01 | 0.01 | Detected | Does not pass |
| Physical position sheet | 0 / 0 | 0.46 | 1.00 | Not detected | Does not pass |

## What the test measures

Each population vector is a point. A Vietoris–Rips complex connects points closer than an increasing distance threshold and fills all corresponding cliques with triangles, tetrahedra, and higher simplices. H0 counts connected components, H1 counts independent loops, and H2 counts two-dimensional cycles (closed surfaces). A barcode records each feature’s birth and death as the distance threshold changes. Persistence is a range of distance scales, not a duration in time.

A torus has Betti numbers (1, 2, 1): one connected component, two independent loops, and one closed surface. We look for two strong H1 bars and one strong H2 bar existing together. These counts are compatible with a torus but do not uniquely characterize one; a positive result would need further validation such as circular-coordinate decoding.

## Fixed analysis protocol

- Data: the same 1,185 selected states, retaining every one of the 256 hidden-vector coordinates. Cell state c is not analyzed.
- Metric: Euclidean distance, divided by the median pairwise distance of each cloud. This scalar normalization makes lifetimes relative to the cloud size; it does not change topology.
- Sampling: 256 farthest-point landmarks, always starting at row 0, selected afresh with the same rule for observed and shuffled clouds. This approximates the original sample; landmark indices and covering radii are saved.
- Ripser computes H0, H1, and H2 over the full distance filtration with coefficient field Z47. The odd prime avoids the particular orientability ambiguity of characteristic 2, but does not make the torus signature unique.
- Null: 99 independent circular time shifts per neuron, applied to the ordered selected-sample traces. This preserves each neuron’s marginal values and cyclic serial structure while disturbing alignment between neurons. The selected series contains gaps between sweep lanes; the shifts preserve selected-index structure, not exact physical-time lag structure. A common shuffle of whole population vectors would leave topology unchanged and is not used.
- The per-dimension reference is the 95th percentile (higher order statistic) of each shuffle’s longest bar. Every observed bar is compared to the same maximum-bar threshold for its dimension. These are exploratory per-dataset thresholds, without correction across 16 models.
- Empirical p = (1 + number of null maxima at least as large as the observed lifetime) / (1 + number of shuffles). With 99 shuffles the smallest value is 0.01. These are null-tail statistics for bar lifetimes, not probabilities that a cloud is a torus.
- Original strict global-count screen (`torus_screen`): exactly two H1 and one H2 bar exceed their null thresholds across the entire filtration, and the top two H1 and top H2 overlap at a distance interval with raw Betti counts (1, 2, 1). This flag is retained unchanged.
- Coexisting signature (`core_torus_signature`): the second-longest H1 bar and longest H2 bar exceed their null thresholds, and the top two H1 bars and top H2 share an interval with raw Betti counts (1, 2, 1). Extra bars outside this interval are recorded rather than used as an automatic veto.
- Transparent refinement: the known synthetic torus produced one additional above-threshold loop that died before its main surface appeared. The original global-count rule therefore rejected a known torus, despite a broad clean (1,2,1) interval. The scale-local signature was added after that control check and applied uniformly to every dataset; distances, sampling, nulls, thresholds, and the negative LSTM conclusions did not change. Both outcomes are reported.
- Sensitivity check: repeat the observed-cloud computation at 512 landmarks. Those results have no matching shuffled test and are not compared to the smaller-sample thresholds.

## Controls and limits

Positive controls are a known Clifford torus and the ordinary / mixed predictive grid populations from one module (128 cells, spacing 0.568 m, seed 1), evaluated at the identical positions and directions. The physical position sheet is a negative reference. Controls use the identical analysis and shuffle rules. Detecting their topology establishes basic sensitivity here, but cannot guarantee sensitivity to a noisy or partially sampled torus in the LSTM.

The recordings use one fixed goal snapshot, one room, one facing direction, and a forced raster path. Only a rectangle away from the goal is sampled. The diagnostic sequence exceeds the training episode duration and has no resets. The full policy mixes multiple grid modules, visual information, and history; it is not necessarily one grid module or one two-dimensional torus. Negative findings apply only to this representation, sampling, metric, and scale.

**A recurrent attractor is a different claim:** states must be dynamically drawn toward an invariant set. Persistent homology examines an activity cloud and discards time ordering; it does not measure attraction, stability, eigenvalues, or recurrent maintenance.

The biological grid-cell study used persistent cohomology with six-PC and neighborhood-distance preprocessing, independent circular-shift controls, and additional circular-coordinate decoding. This analysis uses a simpler direct Euclidean pipeline and is not an exact replication. [Gardner et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8810387/).

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_topology.py
python scripts/plot_lstm_topology.py
```

Individual JSON files retain observed and sensitivity diagrams, landmark indices, all null maxima and circular offsets, input hashes, thresholds, and metrics. `summary.json` contains the compact results.
