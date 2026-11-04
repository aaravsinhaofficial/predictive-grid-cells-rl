# Transfer of the frozen 1B LSTMs to larger rooms

The [fixed-room PCA](../results/lstm_pca_1B/README.md) found 56.5% vs 59.6% of hidden variance in PCs 1 to 3. The [variable replay analysis](../results/lstm_variance_sources_1B/README.md) examined 16 episodes, and [native-policy interventions](../mechanisms/navigation-explanation.md) tested how predictive cues affect homing. Those experiments used the training arena size. The [4 m pilot](pilot_4m_arena.md) evaluated feed-forward agents trained there. This experiment transfers the original recurrent policies and relates their activity to their navigation in the changed rooms.

## Proposed protocol

| Setting | Frozen transfer | Optional adaptation |
| --- | --- | --- |
| Parents | 1B LSTM + raycast, gc100 vs pgc33, matched training seeds 1 through 8 | Independent copies of those same parents for each target width |
| Square side | 2.5 m reference, 3.25 m intermediate shift, 4 m primary shift | 3.25 m and 4 m |
| Evaluation | 64 stochastic episodes per model and width, 1,350 decisions each, 16 slots, seed 730031 | Identical evaluation after adaptation |
| Training | None | 100 original A2C updates, 32 slots × 100 decisions × 4 frames per update, fresh RMSProp at 0.0003 |
| Training randomness | Original code seed preserved | Adaptation seed 9000 + parent seed, paired layouts, separate test stream |

Only room width changes. Movement, physical grid spacing and shift, capture radius, visual generator, and time budget stay at training values. Resizing also widens the distributions of goals, spawns and landmarks. This tests arena-scale transfer within the same navigation task. The reference uses new draws from the training distribution. Both policies get the same episode and respawn draws; their action histories can diverge. Each episode begins with zero recurrent state and unknown goal. Goal hits update memory and respawn the agent normally.

**Primary outcome.** For each seed compute the pgc33 minus gc100 difference in autonomous goals per episode. Compare that gap in the 4 m room with the same seed's gap in the 2.5 m room. Report all eight differences and a percentile interval obtained by resampling seed pairs. The intermediate room gives a secondary size check. Each per-model file also includes raw scores and goal counts, discovery probability, censored first-goal latency, post-discovery goal rate, and path efficiency with its contributing episode count.

**Representation comparison.** Each policy drives its own episodes. For the first 16, replay its actions through its partner from the same starting conditions and random stream. Check sampled physical states, returns, and discovery times. Pool both drivers equally, taking every eighth decision including step zero. This gives 5,408 rows per network at the defaults. Fit centered PCA independently on the current 512-cell code, the remembered 512-cell code, the full 1,287-dimensional LSTM input, and the 256-dimensional post-input hidden state. The output includes the PC1–3 and PC1–10 variance fractions, dimensions for 90% variance, total variance and participation ratio. The own-policy hidden PCA is also saved. The full input includes a learned visual embedding, so the current-code PCA is the direct test of spatial input dimensionality.

The figure `pc10_vs_navigation.png` plots the fraction of matched-path hidden variance explained by the first ten PCs against autonomous goals per episode, connecting each training-seed pair within each width. Compute Spearman correlations across the eight seed pairs between the PGC-minus-grid differences in top-ten variance and goals. Repeat with the first three PCs for continuity with the fixed-room result. A constant measure or fewer than three valid pairs yields a null association. Treat this as exploratory. Neither the correlation nor the PCA identifies the cause of better navigation. Frozen and adapted policies generate different path banks. Adaptation gains compare each policy against its frozen score on the identical test protocol.

## Running

From the repository root, the default command prints the plan and checkpoint inventory and exits.

```sh
python scripts/lstm_transfer.py
```

Restore the 16 original files under `models/{condition}/{condition}_lstm_s{seed}_1000M.pt` or point `--models` to an equivalent directory. Configs and expected SHA-256 hashes are tracked. Each weight file must match the [original manifest](../results/lstm_pca_1B/summary.json). The runner checks complete pairs before starting. It requires Python, NumPy, SciPy, Matplotlib and PyTorch. The plan only requires NumPy. A single CUDA device is suitable for sequential jobs. CPU is supported but the full sweep includes millions of rendered observations.

For an approved pilot, use both conditions for seed 1 and a separate output directory. This measures runtime on the intended hardware.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/lstm_transfer.py frozen \
  --device cuda:0 --seeds 1 --episodes 16 --out runs_transfer_pilot
```

Full commands, to run after reviewing the pilot:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/lstm_transfer.py frozen --device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python scripts/lstm_transfer.py finetune --device cuda:0
```

The frozen sweep uses 3,072 autonomous episodes and 768 partner replays, totaling 5,184,000 policy decisions. Fine-tuning adds 32 adaptation jobs and 40.96M training frames, plus evaluation and replay at the two target widths. GPU time depends on local throughput. Results and parent/evaluated checkpoint hashes go in separate `runs_transfer/frozen` and `runs_transfer/finetune` directories. Complete pair results are saved as each finishes. Each mode requires a new output directory, and adaptation requires a completed frozen run under the same root with the same protocol and device. All time budgets for adaptation count additional frames from zero.
