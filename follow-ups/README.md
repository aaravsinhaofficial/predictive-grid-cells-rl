# Follow-up experiments

Each section answers one follow-up question about the main experiment ([main README](../README.md)).

**Setup in brief.** An RL agent learns to find a hidden goal in a 2.5 m room. Its spatial input is a perfect population code of 512 grid cells. In the **grid + 33 % PGC** code, 169 of those cells become predictive grid cells that read their lattice 20 cm ahead of the agent; nothing else changes. There are three agent types:

- **code only:** the code is the agent's only source of position.
- **code + vision:** a feed-forward policy that also has first-person vision.
- **LSTM + vision:** the setting of Banino et al. 2018.

The score is 10 points per goal reached in a 90 s episode. The two codes of a seed share the same cells, and every comparison is paired within seed. Unless stated otherwise, results use 8 seeds and 300 M training frames.

| # | question | short answer |
|---|---|---|
| 1 | Is the gain just from having fewer inputs? | No. Removing a third of the grid cells barely hurts; making them predictive helps (+4.6, p = 0.028). |
| 2 | Do grid cells only look less important because there are more of them? | No. At equal numbers, silencing predictive cells still costs more, in every agent type. |
| 3 | Why are the LSTM agents tied in Figure 3? | No bug. They really were tied at 300 M; the gap appears with longer training. |
| 4 | Is the benefit relatively bigger without vision? | It points that way at 300 M (not significant). With longer training the benefit is significant in all three agent types. |
| 5 | How does the predictive shift compare with grid spacing? | About a fifth to a quarter of the spacing, in both the recordings and our RNNs. |
| 6 | Uniform vs quarter-spacing vs MEC-weighted vs backward? | The forward-looking codes are about equal; the backward shift is worse at 300 M. |
| 7 | Does the gap grow with training, and when do runs plateau? | It grows up to 600 M in every agent type, then holds at about 10–16 points with vision. |
| 8 | Path efficiency and time to the goal? | Predictive agents return on straighter paths and revisit the goal more often; they don't find it sooner. |

---

## 1. Is the gain just from having fewer inputs?

If the agents ignored the predictive cells, swapping a third of the grid cells for predictive cells would simply leave them with fewer useful inputs, which might be easier to learn from.

**Run:** a **67 % grid** code. This is the grid + 33 % PGC population of each seed with its 169 predictive cells switched off, leaving 343 grid cells. All three agent types, 8 seeds.

**Result:** no. Taking a third of the grid cells away barely hurts, and putting predictive cells in those slots helps.

| code-only agent, 8 seeds | score |
|---|---|
| grid, 512 cells | 27.8 |
| 67 % grid, 343 cells | 26.6 (−1.2 ± 1.5 vs grid) |
| grid + 33 % PGC | 31.2 (**+4.6 ± 1.7 vs 67 % grid, p = 0.028**) |

Compared with the 67 % grid code, the predictive agents also return on straighter paths (path efficiency +0.034 ± 0.010, p = 0.014). They also reach the goal more often after finding it (+0.41 ± 0.15 goals per 1000 steps, p = 0.025). The two vision agents show the same order (67 % grid < grid < grid + PGC), but they vary four times more between seeds, and none of their differences is significant.

## 2. Do grid cells only look less important because there are more of them?

In the 33 % code there are twice as many grid cells as predictive cells, so silencing some grid cells might cost little only because the others cover for them.

**Run:** a **50 / 50** code of 256 grid cells and 256 predictive cells. At test time, 128 cells of one kind were silenced (three random halves, averaged), so both lesions remove the same number of cells.

**Result:** silencing predictive cells costs more in every agent type.

| agent, 8 seeds | cost of silencing 128 predictive cells | cost of silencing 128 grid cells | difference |
|---|---|---|---|
| code only | 5.0 | 2.3 | **+2.7 ± 1.0, p = 0.037** |
| code + vision | 5.5 | 3.0 | +2.5 ± 1.2, p = 0.08 (trend) |
| LSTM + vision | 8.9 | 3.0 | **+5.9 ± 1.7, p = 0.010** |

Even at equal numbers, the agents rely on the predictive cells more than on the grid cells. Without vision, the 50 / 50 code also scores 34.7: above every 33 % code and 6.9 ± 2.3 above grid (p = 0.021).

## 3. Why are the LSTM agents tied in Figure 3?

**Checked:** Figure 3 of the main README is drawn from the final 200-episode tests of the 300 M agents, the same tests behind every other score. Recomputing its bars from the raw evaluation files gives identical numbers, so there is no bug.

At 300 M the LSTM agents really were tied: 66.4 (grid) vs 66.6 (grid + PGC). The gap appears with longer training: +10 points at 600 M and at 1 B (Section 7).

## 4. Is the benefit relatively bigger without vision?

**Run:** the score ratio (grid + 33 % PGC) ÷ grid for each agent type at each training length. Each p-value comes from the paired score difference.

| agent | 300 M | 600 M | 1 B |
|---|---|---|---|
| code only | **+12 %** (p = 0.008) | **+33 %** (p < 0.001) | |
| code + vision | +5 % (p = 0.54) | **+21 %** (p = 0.003) | |
| LSTM + vision | 0 % (p = 0.97) | **+12 %** (p = 0.047) | **+11 %** (p = 0.026) |

**Result:**
- **At 300 M:** the relative benefit is largest without vision, but with 8 seeds the difference between agent types is not significant (p ≥ 0.32).
- **With more training:** the benefit is significant in all three agent types.

So predictive cells are not useful only when the code is the sole source of position, as when navigating in the dark. The benefit holds with vision and memory: it is smaller in relative terms and takes longer to appear.

## 5. How does the predictive shift compare with grid spacing?

**Checked:** the MEC recordings of Ouchi & Fujisawa (2024) and our 30-seed path-integration RNN cohort.

| source | preferred shift | grid spacing | shift ÷ spacing |
|---|---|---|---|
| MEC recordings | 22.5 ± 10.2 cm | ≈ 84 cm (median) | 0.27 ± 0.11 |
| path-integration RNNs | 18–21 cm (median) | ≈ 91 cm (median) | 0.21–0.24 (median) |

**Result:** both put the shift at about a fifth to a quarter of the grid spacing.

One difference: in the RNNs the shift stays roughly the same in centimetres across cells and does not grow with spacing. The recordings instead point to a constant fraction of each cell's spacing.

Our quarter-spacing code uses the recorded fraction, 0.27. In our four modules that gives shifts of 11 / 15 / 22 / 31 cm. Details and sources: [`docs/shift_vs_spacing.md`](../docs/shift_vs_spacing.md).

## 6. Uniform vs quarter-spacing vs MEC-weighted vs backward

**Run:** the four ways of placing the predictive cells, side by side in the code-only agent:

- **uniform:** 20 cm ahead.
- **quarter-spacing:** 0.27 × each module's spacing.
- **MEC-weighted:** twice as many predictive cells in the two middle modules.
- **backward:** 20 cm behind.

Each was run at 33 %, in the 50 / 50 code, and in the long runs to 2 B frames. The 2 B values are averaged over the last 300 M frames (Section 7).

| code-only agent | 33 % predictive, 300 M (8 seeds) | 50 / 50, 300 M (8 seeds) | 33 % predictive, 2 B (seed 1, seed 2) |
|---|---|---|---|
| grid only | 27.8 | 27.8 | 63, 32 |
| uniform (20 cm ahead) | 31.2 | 34.7 | 75, 74 |
| quarter-spacing | 32.8 | 34.9 | 76, 85 |
| MEC-weighted | 32.6 | 34.7 | 85, 68 |
| backward (20 cm behind) | 28.2 | 25.4 | 75, 63 |

**Result:**
- **Forward-looking codes:** all three do about equally well.
- **Backward shift at 300 M:**
  - At 33 % it scores at grid level (+0.5 ± 1.2 vs grid).
  - In the 50 / 50 code it falls 9.2 ± 2.4 below the forward version (p = 0.006) and below plain grid.
  - Its agents barely use the shifted cells.
- **Backward shift at 2 B:** it ended at 75 and 63. Those two seeds were already unusually good at 300 M, level with the forward shift, so this does not show that a backward shift catches up. That would need more seeds.

## 7. Does the gap grow with training, and when do runs plateau?

**Runs (54):**
- grid and grid + 33 % PGC, all 8 seeds of every agent type, trained on to 600 M frames;
- the LSTM runs of all 8 seeds, trained on to 1 B;
- seeds 1 and 2, trained on to 1.5 B in the code + vision agent and to 2 B in the code-only agent (with the quarter-spacing, MEC-weighted and backward codes alongside).

Checkpoints were tested on the same 200 episodes as the 300 M agents.

![training length](../figures/fig11_training_length.png)

*Training score of grid (blue) and grid + 33 % PGC (orange) for each agent type (line style), averaged over seeds in 50 M-frame bins. Each curve averages the seeds trained to its end: seeds 1 and 2 for code only and code + vision (the code-only runs went on to 2 B; shown to 1.5 B), all 8 seeds for LSTM + vision. These are training-episode scores; the tables below use the separate 200-episode tests.*

**Gap between the two codes** (grid + 33 % PGC − grid, mean ± SEM over seeds):

| agent | 300 M | 600 M | 1 B | 1.5 B | 2 B |
|---|---|---|---|---|---|
| code only | **+3.4 ± 0.9** (p = 0.008) | **+11.5 ± 1.7** (p < 0.001) | +24.9 (2 seeds) | +25.0 (2 seeds) | +38.1 (2 seeds) |
| code + vision | +3.6 ± 5.5 (p = 0.54) | **+16.7 ± 3.7** (p = 0.003) | +18.9 (2 seeds) | +23.2 (2 seeds) | |
| LSTM + vision | +0.2 ± 6.4 (p = 0.97) | **+10.0 ± 4.2** (p = 0.047) | **+10.3 ± 3.7** (p = 0.026) | | |

A single checkpoint can score up to 16 points away from its neighbours, so the end of training is also averaged over the checkpoints saved every 25 M frames: the last 200 M frames for the LSTM agent, the last 300 M for the others.

| end of training, averaged | grid | grid + 33 % PGC | gap |
|---|---|---|---|
| LSTM + vision, 1 B (8 seeds) | 91.7 | 105.0 | **+13.3 ± 3.1**, higher in all 8 seeds, p = 0.004 |
| code + vision, 1.5 B (seed 1, seed 2) | 78.0, 90.2 | 88.2, 103.5 | +10.1, +13.3 |
| code only, 2 B (seed 1, seed 2) | 62.8, 31.9 | 75.4, 74.2 | +12.6, +42.3 |

**Result:**
- **The gap grows from 300 M to 600 M in every agent type.** Within seed it widened by:
  - code only: +8.1 ± 1.7 (wider in all 8 seeds, p = 0.002)
  - code + vision: +13.1 ± 6.0 (p = 0.065)
  - LSTM: +9.8 ± 3.1 (p = 0.016)
- **With vision it then holds at about 10–16 points.**
  - **LSTM:** the gap at 1 B (+10.3) equals the one at 600 M (+10.0).
  - **Code + vision:** grid levels off at about 450–600 M frames and grid + PGC at about 650–800 M. Their fitted plateaus are 14 and 16 points apart.
- **When the runs plateau:**
  - **Code + vision:** reaches its plateau by about 800 M frames.
  - **LSTM:** close at 1 B. In 7 of 8 runs of each code, the curve is at 85 % or more of its fitted plateau.
  - **Code only:** the predictive codes slow down by 2 B, climbing under 2 points per 100 M over the last 300 M frames. Even so, the fits place them at only 68–88 % of their plateau.
- **Without vision the grid code learns more slowly.**
  - Over 500–600 M it climbed 2.3 points per 100 M against 5.4 for grid + PGC (slower in all 8 seeds, p = 0.001).
  - In the two 2 B seeds, one grid run stayed low (32). The other began to climb after 600 M and reached 63, 13 below grid + PGC.
  - Two seeds cannot tell whether the grid code eventually catches up.

Full statistics and per-seed values: [`results_ext/EXTENSION.md`](../results_ext/EXTENSION.md).

## 8. Path efficiency and time to the goal

**Path efficiency** is the straight-line distance to the goal ÷ the distance actually walked on the way back to it. The **post-discovery goal rate** counts goals reached per 1000 steps after the first one. Values below are grid + 33 % PGC − grid, 8 paired seeds.

| measure | code only | code + vision | LSTM + vision |
|---|---|---|---|
| path efficiency, 300 M | +0.025 (p = 0.08) | **+0.043** (p = 0.018) | +0.057 (p = 0.10) |
| path efficiency, 600 M | **+0.079** (p = 0.005) | **+0.078** (p = 0.009) | **+0.048** (p = 0.028) |
| goals per 1000 steps after the first, 300 M | **+0.34** (p = 0.025) | +0.51 (p = 0.34) | −0.07 (p = 0.91) |
| goals per 1000 steps after the first, 600 M | **+1.44** (p < 0.001) | **+1.95** (p = 0.003) | **+1.13** (p = 0.048) |
| time to the first goal | no significant difference | no significant difference | no significant difference |

The LSTM agents at 1 B show the same pattern:
- Path efficiency is +0.070 ± 0.009 and the post-discovery goal rate +1.22 ± 0.33, both higher in all 8 seeds (p < 0.001 and p = 0.008).
- The time to the first goal is the same for both codes (467 steps).

**Result:** predictive cells do not help the agent find the goal the first time. They help it get back to the goal once it knows where the goal is.
