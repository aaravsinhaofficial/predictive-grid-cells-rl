# Relative benefit of predictive grid cells (final evaluations)

Source: 250 runs with canonical 200-episode evaluations (250 read from `runs/`, 0 from `data/evals/`). All comparisons are paired within seed (seed s of one condition against seed s of the other; the two share one population); the condition means printed in the tables are over the paired seeds, so the difference and the ratio of means follow from the two printed means (results/RESULTS.md prints the mean over all seeds of each condition instead, which differs only where one condition has fewer seeds, as pgc100 with 3 of the 8 gc100 seeds). Difference statistics follow code/analyze.py (paired t-test, bootstrap 95 % CI over seeds with 10000 resamples and a fixed seed, Wilcoxon signed-rank when n >= 6). Ratios are per seed, A / gc100, with mean +/- SEM and bootstrap CI over seeds; the log-ratio is tested against 0 with a paired t-test. Ratios are left blank when a value is not positive in every seed. For the first-goal step a lower value is better.

## 1. Relative benefit per arm: grid + 33 % PGC (pgc33) vs grid (gc100)

Final intact score per arm, 8 paired seeds.

| arm | n | gc100 | pgc33 | diff +/- SEM | 95 % CI (diff) | paired t p | Wilcoxon p | ratio of means | mean ratio +/- SEM | 95 % CI (ratio) | log-ratio +/- SEM | p (log-ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code only | 8 | 27.77 | 31.16 | 3.38 +/- 0.92 | [1.64, 4.98] | 0.008 | 0.039 | 1.122 | 1.129 +/- 0.036 | [1.062, 1.193] | 0.117 +/- 0.033 | 0.009 |
| code + vision | 8 | 68.18 | 71.75 | 3.57 +/- 5.49 | [-6.00, 13.91] | 0.535 | 0.742 | 1.052 | 1.083 +/- 0.098 | [0.919, 1.273] | 0.053 +/- 0.086 | 0.552 |
| LSTM + vision | 8 | 66.35 | 66.56 | 0.21 +/- 6.44 | [-10.96, 12.58] | 0.975 | 0.945 | 1.003 | 1.047 +/- 0.115 | [0.866, 1.280] | 0.010 +/- 0.098 | 0.924 |

Other metrics (intact evaluation unless the metric says otherwise; the PGC-specific lesion cost is sham - no_pgc with sham the mean of the five count-matched draws).

| arm, metric | n | gc100 | pgc33 | diff +/- SEM | 95 % CI (diff) | paired t p | Wilcoxon p | ratio of means | mean ratio +/- SEM | 95 % CI (ratio) | log-ratio +/- SEM | p (log-ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code only, path efficiency | 8 | 0.311 | 0.337 | 0.025 +/- 0.012 | [0.003, 0.048] | 0.076 | 0.109 | 1.081 | 1.089 +/- 0.044 | [1.012, 1.173] | 0.079 +/- 0.039 | 0.084 |
| code + vision, path efficiency | 8 | 0.568 | 0.611 | 0.043 +/- 0.014 | [0.018, 0.070] | 0.018 | 0.016 | 1.076 | 1.082 +/- 0.029 | [1.033, 1.140] | 0.076 +/- 0.026 | 0.022 |
| LSTM + vision, path efficiency | 8 | 0.532 | 0.589 | 0.057 +/- 0.030 | [0.004, 0.114] | 0.102 | 0.195 | 1.108 | 1.120 +/- 0.062 | [1.013, 1.237] | 0.103 +/- 0.054 | 0.098 |
| code only, post-discovery goal rate | 8 | 2.12 | 2.46 | 0.34 +/- 0.12 | [0.13, 0.57] | 0.025 | 0.023 | 1.161 | 1.188 +/- 0.070 | [1.066, 1.323] | 0.160 +/- 0.058 | 0.028 |
| code + vision, post-discovery goal rate | 8 | 6.97 | 7.48 | 0.51 +/- 0.49 | [-0.43, 1.36] | 0.337 | 0.250 | 1.072 | 1.098 +/- 0.081 | [0.954, 1.250] | 0.074 +/- 0.074 | 0.345 |
| LSTM + vision, post-discovery goal rate | 8 | 6.49 | 6.42 | -0.07 +/- 0.61 | [-1.08, 1.16] | 0.912 | 0.641 | 0.989 | 1.036 +/- 0.126 | [0.855, 1.296] | -0.006 +/- 0.102 | 0.958 |
| code only, first-goal step | 8 | 495.6 | 478.2 | -17.5 +/- 12.6 | [-41.9, 4.0] | 0.209 | 0.109 | 0.965 | 0.967 +/- 0.024 | [0.920, 1.009] | -0.036 +/- 0.026 | 0.203 |
| code + vision, first-goal step | 8 | 503.9 | 527.7 | 23.8 +/- 24.1 | [-21.8, 65.4] | 0.356 | 0.312 | 1.047 | 1.053 +/- 0.047 | [0.965, 1.136] | 0.044 +/- 0.046 | 0.371 |
| LSTM + vision, first-goal step | 8 | 481.1 | 458.5 | -22.5 +/- 24.7 | [-67.6, 23.1] | 0.391 | 0.461 | 0.953 | 0.964 +/- 0.049 | [0.878, 1.057] | -0.046 +/- 0.051 | 0.399 |
| code only, episodes with >= 2 goals | 8 | 0.714 | 0.729 | 0.016 +/- 0.023 | [-0.026, 0.059] | 0.521 | 0.742 | 1.022 | 1.026 +/- 0.033 | [0.967, 1.089] | 0.022 +/- 0.032 | 0.519 |
| code + vision, episodes with >= 2 goals | 8 | 0.729 | 0.713 | -0.017 +/- 0.029 | [-0.068, 0.038] | 0.581 | 0.641 | 0.977 | 0.983 +/- 0.043 | [0.908, 1.066] | -0.024 +/- 0.043 | 0.590 |
| LSTM + vision, episodes with >= 2 goals | 8 | 0.744 | 0.771 | 0.026 +/- 0.039 | [-0.044, 0.099] | 0.525 | 0.617 | 1.035 | 1.053 +/- 0.059 | [0.951, 1.168] | 0.041 +/- 0.055 | 0.477 |
| code only, homing (intact - no_goal) | 8 | 0.01 | -0.18 | -0.19 +/- 0.84 | [-1.67, 1.31] | 0.829 | 0.742 | - | - | - | - | - |
| code + vision, homing (intact - no_goal) | 8 | 30.40 | 35.36 | 4.96 +/- 3.99 | [-1.86, 12.56] | 0.254 | 0.461 | 1.163 | 2.137 +/- 1.063 | [0.968, 4.296] | 0.331 +/- 0.283 | 0.282 |
| LSTM + vision, homing (intact - no_goal) | 8 | 27.22 | 27.03 | -0.19 +/- 5.76 | [-10.03, 10.89] | 0.974 | 0.945 | 0.993 | 1.306 +/- 0.356 | [0.746, 2.026] | 0.047 +/- 0.242 | 0.851 |
| code only, learned gain (intact - untrained) | 8 | 26.70 | 30.09 | 3.39 +/- 0.92 | [1.65, 4.99] | 0.008 | 0.039 | 1.127 | 1.135 +/- 0.038 | [1.065, 1.201] | 0.122 +/- 0.034 | 0.009 |
| code + vision, learned gain (intact - untrained) | 8 | 67.12 | 70.69 | 3.57 +/- 5.48 | [-5.98, 13.91] | 0.535 | 0.742 | 1.053 | 1.085 +/- 0.100 | [0.918, 1.279] | 0.054 +/- 0.087 | 0.552 |
| LSTM + vision, learned gain (intact - untrained) | 8 | 65.33 | 65.55 | 0.22 +/- 6.45 | [-10.95, 12.60] | 0.973 | 0.945 | 1.003 | 1.048 +/- 0.117 | [0.865, 1.287] | 0.010 +/- 0.100 | 0.922 |
| code only, PGC-specific lesion cost (sham - no_pgc) | 8 | -2.95 | 6.30 | 9.24 +/- 1.62 | [6.41, 12.27] | <0.001 | 0.008 | - | - | - | - | - |
| code + vision, PGC-specific lesion cost (sham - no_pgc) | 8 | -3.14 | 6.54 | 9.68 +/- 2.95 | [4.36, 14.92] | 0.013 | 0.008 | - | - | - | - | - |
| LSTM + vision, PGC-specific lesion cost (sham - no_pgc) | 8 | -7.23 | 3.95 | 11.18 +/- 2.93 | [6.17, 16.98] | 0.007 | 0.008 | - | - | - | - | - |

### Interaction: is the relative benefit larger without vision?

Per-seed log-ratio of the score, log(pgc33 / gc100), differenced between arms with the same seed index (same population; the policies of the two arms are trained independently, so the pairing removes only the variance shared through the population). Paired t-test, bootstrap CI, sign-flip permutation test with 10000 random flips and the exact test over all 2^n sign patterns; for comparison, Welch's unpaired t-test on the two sets of per-seed log-ratios and their correlation across seeds.

| contrast | n | mean difference of log-ratios +/- SEM | 95 % CI | ratio of ratios | paired t p | Wilcoxon p | sign-flip p (random) | sign-flip p (exact, 2^n patterns) | Welch p (unpaired) | r across seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| code only - code + vision | 8 | 0.064 +/- 0.081 | [-0.084, 0.209] | 1.066 | 0.455 | 0.461 | 0.422 | 0.422 | 0.503 | 0.33 |
| code only - LSTM + vision | 8 | 0.108 +/- 0.102 | [-0.083, 0.289] | 1.114 | 0.326 | 0.383 | 0.324 | 0.328 | 0.325 | 0.04 |
| code + vision - LSTM + vision | 8 | 0.044 +/- 0.100 | [-0.130, 0.235] | 1.045 | 0.675 | 0.844 | 0.682 | 0.688 | 0.741 | 0.41 |

The mean difference of log-ratios between the code-only arm and the two vision arms is +0.064 +/- 0.081 and +0.108 +/- 0.102; the bootstrap CIs include 0 and the smallest p-value of the two contrasts is 0.32.

With n = 8 seeds the SEM of each between-arm difference is as large as the difference itself, and the ratio-of-ratios CIs ([0.92, 1.23], [0.92, 1.33]) span both a smaller and a clearly larger relative benefit without vision. The point estimates favour a larger relative benefit without vision, but these data neither establish nor exclude an arm dependence of the relative benefit; the unpaired Welch test agrees (p = 0.503, 0.325).

Mean log-ratios per arm: code only +0.117 +/- 0.033 (geometric-mean ratio 1.125, CI [1.058, 1.191]), code + vision +0.053 +/- 0.086 (geometric-mean ratio 1.055, CI [0.910, 1.244]), LSTM + vision +0.010 +/- 0.098 (geometric-mean ratio 1.010, CI [0.854, 1.221]).

## 2. 100 % PGC vs grid, and the code-only arrangements and concentrations vs grid

pgc100 vs gc100, 3 paired seeds per arm (no Wilcoxon below n = 6; the bootstrap CI over 3 seeds takes only 10 distinct values).

| arm, metric | n | gc100 | pgc100 | diff +/- SEM | 95 % CI (diff) | paired t p | Wilcoxon p | ratio of means | mean ratio +/- SEM | 95 % CI (ratio) | log-ratio +/- SEM | p (log-ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code only, score | 3 | 25.15 | 35.95 | 10.80 +/- 0.95 | [8.90, 11.75] | 0.008 | - | 1.429 | 1.434 +/- 0.054 | [1.327, 1.500] | 0.359 +/- 0.039 | 0.011 |
| code + vision, score | 3 | 68.83 | 64.33 | -4.50 +/- 13.17 | [-19.50, 21.75] | 0.765 | - | 0.935 | 0.951 +/- 0.198 | [0.725, 1.346] | -0.090 +/- 0.195 | 0.690 |
| LSTM + vision, score | 3 | 77.08 | 79.62 | 2.53 +/- 10.89 | [-18.05, 19.00] | 0.838 | - | 1.033 | 1.040 +/- 0.134 | [0.784, 1.239] | 0.021 +/- 0.137 | 0.892 |
| code only, path efficiency | 3 | 0.310 | 0.370 | 0.060 +/- 0.025 | [0.022, 0.106] | 0.134 | - | 1.195 | 1.203 +/- 0.090 | [1.066, 1.374] | 0.179 +/- 0.074 | 0.136 |
| code + vision, path efficiency | 3 | 0.564 | 0.580 | 0.016 +/- 0.078 | [-0.118, 0.152] | 0.859 | - | 1.028 | 1.044 +/- 0.141 | [0.811, 1.297] | 0.025 +/- 0.135 | 0.872 |
| LSTM + vision, path efficiency | 3 | 0.567 | 0.623 | 0.056 +/- 0.058 | [-0.057, 0.136] | 0.436 | - | 1.100 | 1.105 +/- 0.103 | [0.904, 1.243] | 0.091 +/- 0.097 | 0.449 |
| code only, post-discovery goal rate | 3 | 1.90 | 3.04 | 1.14 +/- 0.13 | [0.89, 1.30] | 0.012 | - | 1.601 | 1.607 +/- 0.084 | [1.440, 1.707] | 0.472 +/- 0.054 | 0.013 |
| code + vision, post-discovery goal rate | 3 | 6.85 | 6.48 | -0.37 +/- 1.19 | [-1.58, 2.00] | 0.785 | - | 0.946 | 0.964 +/- 0.181 | [0.778, 1.325] | -0.069 +/- 0.175 | 0.732 |
| LSTM + vision, post-discovery goal rate | 3 | 7.44 | 8.17 | 0.73 +/- 1.29 | [-1.85, 2.03] | 0.629 | - | 1.098 | 1.111 +/- 0.172 | [0.768, 1.302] | 0.078 +/- 0.171 | 0.693 |
| code only, first-goal step | 3 | 514.2 | 467.8 | -46.3 +/- 8.0 | [-56.2, -30.4] | 0.029 | - | 0.910 | 0.910 +/- 0.016 | [0.892, 0.941] | -0.095 +/- 0.017 | 0.031 |
| code + vision, first-goal step | 3 | 476.9 | 502.7 | 25.7 +/- 38.0 | [-49.6, 71.9] | 0.568 | - | 1.054 | 1.054 +/- 0.079 | [0.896, 1.142] | 0.046 +/- 0.078 | 0.615 |
| LSTM + vision, first-goal step | 3 | 451.7 | 497.6 | 46.0 +/- 32.3 | [-2.4, 107.3] | 0.291 | - | 1.102 | 1.100 +/- 0.067 | [0.995, 1.225] | 0.091 +/- 0.061 | 0.270 |
| code only, episodes with >= 2 goals | 3 | 0.695 | 0.760 | 0.065 +/- 0.023 | [0.025, 0.105] | 0.106 | - | 1.094 | 1.096 +/- 0.037 | [1.034, 1.160] | 0.091 +/- 0.033 | 0.113 |
| code + vision, episodes with >= 2 goals | 3 | 0.750 | 0.698 | -0.052 +/- 0.071 | [-0.180, 0.065] | 0.542 | - | 0.931 | 0.936 +/- 0.095 | [0.766, 1.094] | -0.076 +/- 0.103 | 0.537 |
| LSTM + vision, episodes with >= 2 goals | 3 | 0.807 | 0.717 | -0.090 +/- 0.036 | [-0.140, -0.020] | 0.130 | - | 0.888 | 0.889 +/- 0.043 | [0.836, 0.975] | -0.119 +/- 0.048 | 0.128 |
| code only, homing (intact - no_goal) | 3 | -1.38 | 4.02 | 5.40 +/- 1.19 | [3.15, 7.20] | 0.045 | - | - | - | - | - | - |
| code + vision, homing (intact - no_goal) | 3 | 31.53 | 29.17 | -2.37 +/- 11.80 | [-21.65, 19.05] | 0.860 | - | 0.925 | 1.002 +/- 0.381 | [0.431, 1.724] | -0.153 +/- 0.400 | 0.740 |
| LSTM + vision, homing (intact - no_goal) | 3 | 36.47 | 39.77 | 3.30 +/- 9.66 | [-13.55, 19.90] | 0.765 | - | 1.090 | 1.100 +/- 0.243 | [0.670, 1.511] | 0.042 +/- 0.238 | 0.877 |
| code only, learned gain (intact - untrained) | 3 | 24.11 | 34.88 | 10.77 +/- 0.94 | [8.88, 11.75] | 0.008 | - | 1.447 | 1.451 +/- 0.056 | [1.340, 1.522] | 0.371 +/- 0.040 | 0.011 |
| code + vision, learned gain (intact - untrained) | 3 | 67.80 | 63.27 | -4.53 +/- 13.20 | [-19.58, 21.78] | 0.764 | - | 0.933 | 0.950 +/- 0.201 | [0.720, 1.352] | -0.092 +/- 0.198 | 0.688 |
| LSTM + vision, learned gain (intact - untrained) | 3 | 76.07 | 78.58 | 2.51 +/- 10.87 | [-18.03, 18.95] | 0.839 | - | 1.033 | 1.040 +/- 0.136 | [0.781, 1.241] | 0.021 +/- 0.139 | 0.895 |
| code only, PGC-specific lesion cost (sham - no_pgc) | 3 | -2.24 | 17.54 | 19.78 +/- 4.42 | [13.98, 28.46] | 0.047 | - | - | - | - | - | - |
| code + vision, PGC-specific lesion cost (sham - no_pgc) | 3 | -5.92 | 20.92 | 26.84 +/- 3.20 | [20.45, 30.06] | 0.014 | - | - | - | - | - | - |
| LSTM + vision, PGC-specific lesion cost (sham - no_pgc) | 3 | -7.19 | 22.49 | 29.68 +/- 11.72 | [17.00, 53.09] | 0.127 | - | - | - | - | - | - |

Code-only arm, final intact score of every PGC arrangement (33 % PGC) and of the concentration series (quarter-spacing shift; its 33 % point is the quarter-spacing arrangement) vs gc100, 8 paired seeds each.

| condition | n | gc100 | condition | diff +/- SEM | 95 % CI (diff) | paired t p | Wilcoxon p | ratio of means | mean ratio +/- SEM | 95 % CI (ratio) | log-ratio +/- SEM | p (log-ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform, 20 cm (pgc33) | 8 | 27.77 | 31.16 | 3.38 +/- 0.92 | [1.64, 4.98] | 0.008 | 0.039 | 1.122 | 1.129 +/- 0.036 | [1.062, 1.193] | 0.117 +/- 0.033 | 0.009 |
| quarter-spacing shift (pgc33phase27) | 8 | 27.77 | 32.76 | 4.98 +/- 1.75 | [1.84, 8.15] | 0.025 | 0.039 | 1.179 | 1.197 +/- 0.073 | [1.068, 1.331] | 0.167 +/- 0.061 | 0.029 |
| MEC-weighted modules (pgc33mec) | 8 | 27.77 | 32.58 | 4.80 +/- 2.28 | [0.58, 8.93] | 0.074 | 0.109 | 1.173 | 1.184 +/- 0.082 | [1.031, 1.333] | 0.151 +/- 0.073 | 0.078 |
| quarter-spacing + heterogeneous (pgc33bio) | 8 | 27.77 | 31.43 | 3.66 +/- 0.58 | [2.59, 4.67] | <0.001 | 0.008 | 1.132 | 1.137 +/- 0.023 | [1.094, 1.178] | 0.127 +/- 0.021 | <0.001 |
| heterogeneous 20 cm (pgc33jitter) | 8 | 27.77 | 31.10 | 3.32 +/- 1.13 | [1.40, 5.50] | 0.022 | 0.016 | 1.120 | 1.125 +/- 0.044 | [1.052, 1.213] | 0.113 +/- 0.038 | 0.021 |
| coarse modules only (pgc33coarse) | 8 | 27.77 | 30.48 | 2.71 +/- 1.22 | [0.48, 4.96] | 0.062 | 0.078 | 1.097 | 1.109 +/- 0.046 | [1.029, 1.197] | 0.098 +/- 0.041 | 0.050 |
| middle modules only (pgc33middle) | 8 | 27.77 | 29.74 | 1.96 +/- 1.61 | [-0.89, 4.93] | 0.261 | 0.312 | 1.071 | 1.082 +/- 0.063 | [0.972, 1.200] | 0.067 +/- 0.058 | 0.284 |
| retrospective (20 cm behind) (pgc33retro) | 8 | 27.77 | 28.23 | 0.46 +/- 1.17 | [-1.56, 2.71] | 0.709 | 0.844 | 1.016 | 1.019 +/- 0.044 | [0.943, 1.103] | 0.012 +/- 0.042 | 0.779 |
| half-spacing shift (pgc33phase) | 8 | 27.77 | 27.62 | -0.15 +/- 1.94 | [-3.78, 3.26] | 0.941 | 1.000 | 0.995 | 1.016 +/- 0.069 | [0.894, 1.144] | -0.000 +/- 0.068 | 0.998 |
| fine modules only (pgc33fine) | 8 | 27.77 | 27.33 | -0.44 +/- 0.94 | [-2.11, 1.29] | 0.652 | 0.742 | 0.984 | 0.991 +/- 0.036 | [0.927, 1.057] | -0.014 +/- 0.036 | 0.714 |
| 10 % PGC (pgc10) | 8 | 27.77 | 27.96 | 0.19 +/- 1.21 | [-1.88, 2.56] | 0.881 | 0.945 | 1.007 | 1.019 +/- 0.046 | [0.943, 1.109] | 0.012 +/- 0.043 | 0.784 |
| 25 % PGC (pgc25) | 8 | 27.77 | 32.71 | 4.94 +/- 1.49 | [2.30, 7.67] | 0.013 | 0.016 | 1.178 | 1.194 +/- 0.063 | [1.083, 1.309] | 0.167 +/- 0.053 | 0.015 |
| 30 % PGC (pgc30) | 8 | 27.77 | 29.92 | 2.14 +/- 1.81 | [-0.94, 5.64] | 0.274 | 0.383 | 1.077 | 1.094 +/- 0.072 | [0.974, 1.235] | 0.076 +/- 0.063 | 0.268 |
| 35 % PGC (pgc35) | 8 | 27.77 | 31.28 | 3.51 +/- 1.47 | [0.91, 6.27] | 0.049 | 0.039 | 1.126 | 1.142 +/- 0.060 | [1.040, 1.258] | 0.124 +/- 0.051 | 0.045 |
| 40 % PGC (pgc40) | 8 | 27.77 | 30.62 | 2.84 +/- 1.87 | [-0.48, 6.41] | 0.171 | 0.109 | 1.102 | 1.102 +/- 0.064 | [0.985, 1.220] | 0.085 +/- 0.060 | 0.203 |

## 3. Lesion cost per silenced cell

Cost = intact score - score with the cells zeroed at test time (final evaluation, 200 episodes); sham is the mean of the five count-matched draws of random grid cells. Per-cell numbers are cost / cells silenced and assume that costs add up linearly over cells. pgc33: 8 seeds per arm; gc100 and pgc100 are reference rows (in gc100 no_gc is the whole code and no_pgc silences nothing; in pgc100 no_pgc is the whole code and sham draws from the PGCs).

| arm | condition | lesion | cells | cost +/- SEM | cost per cell +/- SEM |
|---|---|---|---|---|---|
| code only | pgc33 (n = 8) | PGC lesion (no_pgc) | 169 | 9.86 +/- 1.35 | 0.0584 +/- 0.0080 |
| code only | pgc33 (n = 8) | grid-cell lesion (no_gc) | 343 | 12.39 +/- 2.04 | 0.0361 +/- 0.0059 |
| code only | pgc33 (n = 8) | sham lesion (sham) | 169 | 3.57 +/- 1.28 | 0.0211 +/- 0.0076 |
| code only | gc100 (n = 8) | PGC lesion (no_pgc) | 0 | 0.00 +/- 0.00 | - |
| code only | gc100 (n = 8) | grid-cell lesion (no_gc) | 512 | 15.76 +/- 2.80 | 0.0308 +/- 0.0055 |
| code only | gc100 (n = 8) | sham lesion (sham) | 169 | 2.95 +/- 0.80 | 0.0174 +/- 0.0047 |
| code only | pgc100 (n = 3) | PGC lesion (no_pgc) | 512 | 20.87 +/- 4.96 | 0.0408 +/- 0.0097 |
| code only | pgc100 (n = 3) | grid-cell lesion (no_gc) | 0 | 0.00 +/- 0.00 | - |
| code only | pgc100 (n = 3) | sham lesion (sham) | 169 | 3.33 +/- 1.04 | 0.0197 +/- 0.0062 |
| code + vision | pgc33 (n = 8) | PGC lesion (no_pgc) | 169 | 11.62 +/- 2.48 | 0.0688 +/- 0.0147 |
| code + vision | pgc33 (n = 8) | grid-cell lesion (no_gc) | 343 | 24.38 +/- 6.10 | 0.0711 +/- 0.0178 |
| code + vision | pgc33 (n = 8) | sham lesion (sham) | 169 | 5.08 +/- 2.51 | 0.0301 +/- 0.0149 |
| code + vision | gc100 (n = 8) | PGC lesion (no_pgc) | 0 | 0.00 +/- 0.00 | - |
| code + vision | gc100 (n = 8) | grid-cell lesion (no_gc) | 512 | 25.90 +/- 5.22 | 0.0506 +/- 0.0102 |
| code + vision | gc100 (n = 8) | sham lesion (sham) | 169 | 3.14 +/- 2.44 | 0.0186 +/- 0.0144 |
| code + vision | pgc100 (n = 3) | PGC lesion (no_pgc) | 512 | 24.12 +/- 9.56 | 0.0471 +/- 0.0187 |
| code + vision | pgc100 (n = 3) | grid-cell lesion (no_gc) | 0 | 0.00 +/- 0.00 | - |
| code + vision | pgc100 (n = 3) | sham lesion (sham) | 169 | 3.20 +/- 5.39 | 0.0189 +/- 0.0319 |
| LSTM + vision | pgc33 (n = 8) | PGC lesion (no_pgc) | 169 | 13.07 +/- 2.69 | 0.0773 +/- 0.0159 |
| LSTM + vision | pgc33 (n = 8) | grid-cell lesion (no_gc) | 343 | 23.69 +/- 3.88 | 0.0691 +/- 0.0113 |
| LSTM + vision | pgc33 (n = 8) | sham lesion (sham) | 169 | 9.12 +/- 2.35 | 0.0539 +/- 0.0139 |
| LSTM + vision | gc100 (n = 8) | PGC lesion (no_pgc) | 0 | 0.00 +/- 0.00 | - |
| LSTM + vision | gc100 (n = 8) | grid-cell lesion (no_gc) | 512 | 22.69 +/- 5.74 | 0.0443 +/- 0.0112 |
| LSTM + vision | gc100 (n = 8) | sham lesion (sham) | 169 | 7.23 +/- 2.40 | 0.0428 +/- 0.0142 |
| LSTM + vision | pgc100 (n = 3) | PGC lesion (no_pgc) | 512 | 35.00 +/- 11.37 | 0.0684 +/- 0.0222 |
| LSTM + vision | pgc100 (n = 3) | grid-cell lesion (no_gc) | 0 | 0.00 +/- 0.00 | - |
| LSTM + vision | pgc100 (n = 3) | sham lesion (sham) | 169 | 12.51 +/- 3.56 | 0.0740 +/- 0.0211 |

Paired tests (pgc33): PGC lesion cost minus sham cost (absolute; equals the PGC-specific lesion cost sham - no_pgc) and per-cell PGC lesion cost minus per-cell grid-cell lesion cost.

| arm | test | mean +/- SEM | 95 % CI | paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| code only | cost(no_pgc) - cost(sham) | 6.30 +/- 1.05 | [4.46, 8.30] | <0.001 | 0.008 |
| code only | per cell: no_pgc - no_gc | 0.0222 +/- 0.0054 | [0.0117, 0.0312] | 0.004 | 0.023 |
| code + vision | cost(no_pgc) - cost(sham) | 6.54 +/- 1.61 | [3.61, 9.30] | 0.005 | 0.008 |
| code + vision | per cell: no_pgc - no_gc | -0.0023 +/- 0.0114 | [-0.0243, 0.0166] | 0.846 | 1.000 |
| LSTM + vision | cost(no_pgc) - cost(sham) | 3.95 +/- 1.51 | [1.51, 6.95] | 0.034 | 0.023 |
| LSTM + vision | per cell: no_pgc - no_gc | 0.0083 +/- 0.0148 | [-0.0186, 0.0353] | 0.593 | 0.547 |

## 4. Figure 3 check (figures/fig3_probes.png)

make_figures.fig_probes reads, through make_figures.load_runs, the files runs/<run>/eval/<probe>.json whose name has no "@" and whose "canonical" flag is true (written only for ckpt_final.pt, 200 episodes, stochastic policy, evaluation seed 424242); the sham bar is the mean of sham0-4. Recomputed from the same files (mean +/- SEM over seeds): 720 probe files, 720 canonical, 720 with 200 episodes, trained-policy frames [300006400].

| arm | condition | n | intact | no_code | no_goal | sham | no_pgc |
|---|---|---|---|---|---|---|---|
| code only | gc100 | 8 | 27.77 +/- 1.63 | 12.02 +/- 1.97 | 27.77 +/- 0.79 | 24.83 +/- 1.19 | 27.77 +/- 1.63 |
| code only | pgc33 | 8 | 31.16 +/- 1.59 | 12.41 +/- 1.20 | 31.34 +/- 0.72 | 27.59 +/- 0.81 | 21.29 +/- 0.99 |
| code + vision | gc100 | 8 | 68.18 +/- 5.21 | 42.27 +/- 0.89 | 37.77 +/- 0.98 | 65.03 +/- 4.06 | 68.18 +/- 5.21 |
| code + vision | pgc33 | 8 | 71.75 +/- 5.17 | 39.24 +/- 1.05 | 36.39 +/- 1.93 | 66.67 +/- 3.16 | 60.13 +/- 3.40 |
| LSTM + vision | gc100 | 8 | 66.35 +/- 5.36 | 43.66 +/- 1.60 | 39.13 +/- 1.30 | 59.12 +/- 4.01 | 66.35 +/- 5.36 |
| LSTM + vision | pgc33 | 8 | 66.56 +/- 4.57 | 41.88 +/- 0.95 | 39.54 +/- 1.00 | 57.45 +/- 4.61 | 53.49 +/- 4.85 |

Intact means to three decimals (gc100 / pgc33): code only 27.775 / 31.156, code + vision 68.175 / 71.750, LSTM + vision 66.350 / 66.562.

Bar heights recomputed with make_figures.load_runs/score agree with the values above to 0.0e+00: the figure is drawn from the canonical final evaluations.

## 5. Secondary endpoints: grid + 33 % PGC vs grid, per arm

Intact final evaluation, 8 paired seeds. Path efficiency is averaged over goal returns; the post-discovery goal rate counts goals after the first; the first-goal step is lower when the goal is found sooner.

| arm | endpoint | gc100 | pgc33 | diff +/- SEM | 95 % CI | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| code only | path efficiency | 0.311 | 0.337 | 0.025 +/- 0.012 | [0.003, 0.048] | 0.076 | 0.109 |
| code only | post-discovery goal rate | 2.12 | 2.46 | 0.34 +/- 0.12 | [0.13, 0.57] | 0.025 | 0.023 |
| code only | first-goal step | 495.6 | 478.2 | -17.5 +/- 12.6 | [-41.9, 4.0] | 0.209 | 0.109 |
| code only | episodes with >= 2 goals | 0.714 | 0.729 | 0.016 +/- 0.023 | [-0.026, 0.059] | 0.521 | 0.742 |
| code + vision | path efficiency | 0.568 | 0.611 | 0.043 +/- 0.014 | [0.018, 0.070] | 0.018 | 0.016 |
| code + vision | post-discovery goal rate | 6.97 | 7.48 | 0.51 +/- 0.49 | [-0.43, 1.36] | 0.337 | 0.250 |
| code + vision | first-goal step | 503.9 | 527.7 | 23.8 +/- 24.1 | [-21.8, 65.4] | 0.356 | 0.312 |
| code + vision | episodes with >= 2 goals | 0.729 | 0.713 | -0.017 +/- 0.029 | [-0.068, 0.038] | 0.581 | 0.641 |
| LSTM + vision | path efficiency | 0.532 | 0.589 | 0.057 +/- 0.030 | [0.004, 0.114] | 0.102 | 0.195 |
| LSTM + vision | post-discovery goal rate | 6.49 | 6.42 | -0.07 +/- 0.61 | [-1.08, 1.16] | 0.912 | 0.641 |
| LSTM + vision | first-goal step | 481.1 | 458.5 | -22.5 +/- 24.7 | [-67.6, 23.1] | 0.391 | 0.461 |
| LSTM + vision | episodes with >= 2 goals | 0.744 | 0.771 | 0.026 +/- 0.039 | [-0.044, 0.099] | 0.525 | 0.617 |

## Consistency check

1360 numbers of results/RESULTS.md (final-evaluation means, derived quantities and paired comparisons vs gc100 for the conditions used here) were recomputed from the evaluation JSONs; 0 differ beyond the rounding of the printed value.

## Figure 10 (figures/fig10_relative_benefit.png)

A: score ratio grid + 33 % PGC / grid per seed (dots) with mean +/- SEM, per arm; the line marks a ratio of 1; the labels give the mean ratio and the paired t p-value of the log-ratio against 0. B: score lost per silenced cell in the grid + 33 % PGC agent, per arm, for the PGC lesion (169 cells), the grid-cell lesion (343 cells) and the sham lesion (169 random grid cells, mean of 5 draws); bars are means over 8 seeds with SEM, dots are seeds. C: code-only arm, score ratio vs grid only for the PGC arrangements at 33 % and for the concentration series (quarter-spacing shift), dots per seed and mean +/- SEM; the dashed line is the grid-only reference (ratio 1) and the solid line the uniform 20 cm arrangement.
