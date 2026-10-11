# Results: idealised grid vs predictive-grid population codes

Runs found: 6 (6 finished). Reference gc100 final score per arm: ffb: 9.3, ff: 13.1

## Final evaluation (200 paired episodes, stochastic policy; mean +/- SEM over seeds)

| arm | condition | n (done) | intact | untrained | no_code | no_current | no_goal | sham | no_pgc | pgc_as_gc | fake_goal | goals/ep | found | first goal | post-disc. rate | path eff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ffb | 100% grid cells | 1 (1) | 9.35 | 0.58 | 8.05 | 8.60 | 9.15 | 9.23 | 9.35 | 9.35 | 9.00 | 0.94 | 0.58 | 1197 | 0.53 | 0.316 |
| ffb | 67% grid + 33% PGC | 1 (1) | 8.50 | 0.62 | 5.35 | 5.30 | 8.55 | 7.60 | 7.30 | 8.30 | 7.85 | 0.85 | 0.56 | 1203 | 0.61 | 0.298 |
| ffb | no spatial code | 1 (1) | 9.00 | 0.60 | 9.00 | 9.00 | 9.00 | 9.00 | 9.00 | 9.00 | 8.85 | 0.90 | 0.61 | 1171 | 0.48 | 0.305 |
| ff | 100% grid cells | 1 (1) | 13.05 | 0.63 | 12.50 | 12.15 | 12.00 | 12.76 | 13.05 | 13.05 | 11.50 | 1.30 | 0.62 | 1127 | 1.13 | 0.356 |
| ff | 67% grid + 33% PGC | 1 (1) | 9.00 | 0.62 | 5.15 | 4.45 | 9.35 | 8.20 | 9.05 | 8.45 | 9.40 | 0.90 | 0.58 | 1182 | 0.49 | 0.329 |
| ff | no spatial code | 1 (1) | 13.00 | 0.63 | 13.00 | 13.00 | 13.00 | 13.00 | 13.00 | 13.00 | 12.90 | 1.30 | 0.40 | 1284 | 1.67 | 0.381 |

## Derived per-group quantities (mean +/- SEM over seeds)

| arm | condition | learned gain | reliance (no_code) | reliance (no_current) | homing (no_goal) | PGC-specific (sham - no_pgc) | train final | early AUC (<=100M) | frames to 50% (M) | frames to 80% (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| ffb | 100% grid cells | 8.77 | 1.30 | 0.75 | 0.20 | -0.12 | 7.68 | 6.32 | 12 | 52 |
| ffb | 67% grid + 33% PGC | 7.88 | 3.15 | 3.20 | -0.05 | 0.30 | 7.22 | 6.45 | 18 | 32 |
| ffb | no spatial code | 8.40 | 0.00 | 0.00 | 0.00 | 0.00 | 6.35 | 5.35 | 28 | 68 |
| ff | 100% grid cells | 12.42 | 0.55 | 0.90 | 1.05 | -0.29 | 10.76 | 8.16 | 32 | 62 |
| ff | 67% grid + 33% PGC | 8.38 | 3.85 | 4.55 | -0.35 | -0.85 | 7.12 | 6.86 | 22 | 48 |
| ff | no spatial code | 12.37 | 0.00 | 0.00 | 0.00 | 0.00 | 13.89 | 7.24 | 18 | 52 |

## Comparisons (paired within seed where both runs exist; a - b)

| comparison | metric | pairs | mean a | mean b | mean diff +/- SEM | bootstrap 95% CI | paired t p | Welch p | Wilcoxon p (n>=6) | TOST p (equiv. within 10%) |
|---|---|---|---|---|---|---|---|---|---|---|
| ffb: pgc33 - gc100 | eval_intact_score | 1 | 8.50 | 9.35 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | train_final_return | 1 | 7.22 | 7.68 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | train_early_auc | 1 | 6.45 | 6.32 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | frames_to_80 | 1 | 32500000.00 | 52500000.00 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | homing | 1 | -0.05 | 0.20 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | eval_intact_path_eff | 1 | 0.30 | 0.32 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | eval_intact_post_rate | 1 | 0.61 | 0.53 | - | - | - | - | - | - |
| ffb: pgc33 - gc100 | eval_intact_first_goal | 1 | 1202.61 | 1197.19 | - | - | - | - | - | - |
| ffb: gc100 - none | eval_intact_score | 1 | 9.35 | 9.00 | - | - | - | - | - | - |
| ffb: gc100 - none | train_final_return | 1 | 7.68 | 6.35 | - | - | - | - | - | - |
| ffb: gc100 - none | train_early_auc | 1 | 6.32 | 5.35 | - | - | - | - | - | - |
| ffb: gc100 - none | frames_to_80 | 1 | 52500000.00 | 67500000.00 | - | - | - | - | - | - |
| ffb: gc100 - none | homing | 1 | 0.20 | 0.00 | - | - | - | - | - | - |
| ffb: gc100 - none | eval_intact_path_eff | 1 | 0.32 | 0.30 | - | - | - | - | - | - |
| ffb: gc100 - none | eval_intact_post_rate | 1 | 0.53 | 0.48 | - | - | - | - | - | - |
| ffb: gc100 - none | eval_intact_first_goal | 1 | 1197.19 | 1170.81 | - | - | - | - | - | - |
| ffb: pgc33 - none | eval_intact_score | 1 | 8.50 | 9.00 | - | - | - | - | - | - |
| ffb: pgc33 - none | train_final_return | 1 | 7.22 | 6.35 | - | - | - | - | - | - |
| ffb: pgc33 - none | train_early_auc | 1 | 6.45 | 5.35 | - | - | - | - | - | - |
| ffb: pgc33 - none | frames_to_80 | 1 | 32500000.00 | 67500000.00 | - | - | - | - | - | - |
| ffb: pgc33 - none | homing | 1 | -0.05 | 0.00 | - | - | - | - | - | - |
| ffb: pgc33 - none | eval_intact_path_eff | 1 | 0.30 | 0.30 | - | - | - | - | - | - |
| ffb: pgc33 - none | eval_intact_post_rate | 1 | 0.61 | 0.48 | - | - | - | - | - | - |
| ffb: pgc33 - none | eval_intact_first_goal | 1 | 1202.61 | 1170.81 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | eval_intact_score | 1 | 9.00 | 13.05 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | train_final_return | 1 | 7.12 | 10.76 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | train_early_auc | 1 | 6.86 | 8.16 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | frames_to_80 | 1 | 47500000.00 | 62500000.00 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | homing | 1 | -0.35 | 1.05 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | eval_intact_path_eff | 1 | 0.33 | 0.36 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | eval_intact_post_rate | 1 | 0.49 | 1.13 | - | - | - | - | - | - |
| ff: pgc33 - gc100 | eval_intact_first_goal | 1 | 1181.72 | 1127.07 | - | - | - | - | - | - |
| ff: gc100 - none | eval_intact_score | 1 | 13.05 | 13.00 | - | - | - | - | - | - |
| ff: gc100 - none | train_final_return | 1 | 10.76 | 13.89 | - | - | - | - | - | - |
| ff: gc100 - none | train_early_auc | 1 | 8.16 | 7.24 | - | - | - | - | - | - |
| ff: gc100 - none | frames_to_80 | 1 | 62500000.00 | 52500000.00 | - | - | - | - | - | - |
| ff: gc100 - none | homing | 1 | 1.05 | 0.00 | - | - | - | - | - | - |
| ff: gc100 - none | eval_intact_path_eff | 1 | 0.36 | 0.38 | - | - | - | - | - | - |
| ff: gc100 - none | eval_intact_post_rate | 1 | 1.13 | 1.67 | - | - | - | - | - | - |
| ff: gc100 - none | eval_intact_first_goal | 1 | 1127.07 | 1283.74 | - | - | - | - | - | - |
| ff: pgc33 - none | eval_intact_score | 1 | 9.00 | 13.00 | - | - | - | - | - | - |
| ff: pgc33 - none | train_final_return | 1 | 7.12 | 13.89 | - | - | - | - | - | - |
| ff: pgc33 - none | train_early_auc | 1 | 6.86 | 7.24 | - | - | - | - | - | - |
| ff: pgc33 - none | frames_to_80 | 1 | 47500000.00 | 52500000.00 | - | - | - | - | - | - |
| ff: pgc33 - none | homing | 1 | -0.35 | 0.00 | - | - | - | - | - | - |
| ff: pgc33 - none | eval_intact_path_eff | 1 | 0.33 | 0.38 | - | - | - | - | - | - |
| ff: pgc33 - none | eval_intact_post_rate | 1 | 0.49 | 1.67 | - | - | - | - | - | - |
| ff: pgc33 - none | eval_intact_first_goal | 1 | 1181.72 | 1283.74 | - | - | - | - | - | - |

## Per-run

| run | frames (M) | episodes | train final | intact | untrained | no_code | no_current | no_goal |
|---|---|---|---|---|---|---|---|---|
| gc100_ff_s1 | 100 | 13857 | 10.76 | 13.05 | 0.63 | 12.50 | 12.15 | 12.00 |
| gc100_ffb_s1 | 100 | 13857 | 7.68 | 9.35 | 0.58 | 8.05 | 8.60 | 9.15 |
| none_ff_s1 | 100 | 13857 | 13.89 | 13.00 | 0.63 | 13.00 | 13.00 | 13.00 |
| none_ffb_s1 | 100 | 13857 | 6.35 | 9.00 | 0.60 | 9.00 | 9.00 | 9.00 |
| pgc33_ff_s1 | 100 | 13857 | 7.12 | 9.00 | 0.62 | 5.15 | 4.45 | 9.35 |
| pgc33_ffb_s1 | 100 | 13857 | 7.22 | 8.50 | 0.62 | 5.35 | 5.30 | 8.55 |

Figures: learning_curves.png, final_scores.png, probes.png, reliance_over_training.png
