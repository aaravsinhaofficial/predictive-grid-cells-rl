# Do predictive grid cells help an agent navigate?

**Aarav Sinha & William Redman — first experiment of the project _Behavioral optimization reveals principles of neural coding_.**

We give a reinforcement-learning agent a *perfect* spatial population code, the way Banino et al. (2018) gave their agent a grid-cell code, and ask one question: does a code in which a third of the grid cells are **predictive grid cells** (PGCs; they fire for where the animal is *about to be*) help the agent more than a code of ordinary grid cells?

**Short answer.** A little, and only in some ways. With 8 paired training seeds per condition:

- When the code is the agent's **only** source of position, swapping 33 % of the grid cells for PGCs raises the score by **12 %** (27.8 → 31.2, p = 0.008). A code of 100 % PGCs is better still (+39 %).
- When the agent also has vision, PGCs do **not** change the score, but the agent takes **straighter paths** back to the goal (path efficiency +7–10 %) and learns a bit faster.
- The PGC advantage is not explained by the velocity information PGCs carry (giving the grid agent an explicit velocity input does nothing), nor by the better-centred goal memory PGCs produce.

![main result](figures/fig1_main_result.png)

*Figure 1. Each dot is one training seed; a grey line joins the grid-only and grid+PGC runs that share the same cells. Thick bars are means, thin bars SEM. Dashed line: an agent with no spatial code. Top: score (10 points per goal reached in a 90 s episode). Bottom: how straight the agent's return trips to the goal are.*

---

## 1. Background

**Grid cells** fire in a hexagonal lattice of locations. Banino et al. (*Nature* 2018) trained a deep RL agent whose policy received a grid-like code (from a separately trained path-integration network) plus a *goal grid code* (the code the last time it reached the goal), and showed this agent navigated better than baselines.

**Predictive grid cells.** Ouchi & Fujisawa (2024) found grid cells in the medial entorhinal cortex whose hexagonal firing is best aligned with where the animal will be a few centimetres further along its path. In our earlier work, path-integration RNNs develop the same cells (median forward shift ≈ 16–21 cm), and ablating them disrupts the network's toroidal dynamics. Whether such cells are *useful* for behaviour was open.

**The idea tested here** (Redman, meeting of 9 Oct 2026): don't learn the code at all. Define a perfect grid cell and a perfect predictive grid cell analytically, read the whole population's activity off the agent's true state at every step, feed it to a Banino-style agent, and compare populations. This removes every confound about how well the code was learned and lets us test exactly one thing: the format of the code.

## 2. What was built

| piece | what it is |
|---|---|
| **Population code** (`code/codes.py`) | 512 analytic cells in 4 modules (spacings 0.40, 0.57, 0.81, 1.15 m), random orientation per module, random phase per cell. A **PGC** is the same hexagonal pattern read out **20 cm ahead of the agent along its direction of travel** (the definition used by the PGC analysis pipeline). Validated with the paper's own gridness-vs-shift scorer: grid cells peak at 0 cm, PGCs at 19.5 ± 1.2 cm (Fig. 5). |
| **Task** (`code/env.py`) | Banino's goal task in a 2.5 m square arena: hidden goal, +10 and random respawn on reaching it, 90 s episodes, six actions, first-person vision (walls, stripes and landmarks re-drawn every episode). Vectorised on the GPU. |
| **Agent** (`code/nets.py`) | The paper's architecture: 4-layer conv vision net + 256-unit layer, policy LSTM (256) fed `[vision, code, goal code, previous action, previous reward]`, 6-way actor and critic. A feed-forward variant (same inputs, no memory) is used where the code must be the only position source. |
| **Training** (`code/train.py`) | A2C with the paper's hyperparameters (BPTT 100, entropy 8e-5, γ 0.99, reward clip 1), 300 M frames per run. |
| **Evaluation** (`code/evaluate.py`) | 200 episodes with identical layouts across runs, plus probes that zero or alter parts of the code at test time (whole code, current code only, goal memory, the 169 PGCs, 169 random cells, …) and an untrained chance line. |

![how the code is built](figures/fig0_grid_modules.png)

*How the population code is built.* **(A)** One example cell from each of the four modules; the 128 cells of a module share one spacing (arrow) and one orientation. **(B)** Cells within a module differ only in phase, i.e. where the lattice sits (inset: the 128 phases of module 2, spread uniformly over one unit cell). **(C)** All 512 cells as the agent walks a straight line: each module repeats with its own period, and the four periods read together make position unambiguous across the 2.5 m arena. **(D)** A predictive grid cell is the same lattice read 20 cm ahead along the direction of travel. **(E, F)** The follow-up experiment varies how the 169 predictive cells are allotted to modules and how far ahead they look (a fixed 20 cm, or a fixed fraction of each module's spacing).

## 3. Design

Three **arms** (what the agent is):

| arm | policy | vision | why |
|---|---|---|---|
| code only | feed-forward | heading only | the code is the *only* way to know where you are |
| code + vision | feed-forward | first-person | vision + code, no memory |
| LSTM + vision | LSTM | first-person | the paper's agent |

**Conditions** (same 512 cells per seed; only the PGC shift differs, so runs are paired):

| condition | seeds | population |
|---|---|---|
| grid cells | 8 | 512 grid cells |
| **grid + 33 % PGC** | 8 | 343 grid cells + 169 PGCs (stratified over modules), 20 cm shift |
| 33 % PGC, position-only goal memory | 8 | as above, but the goal snapshot uses zero shift (control for the goal-memory geometry) |
| no code | 3 | code zeroed |
| 100 % PGC | 3 | all 512 cells shifted |
| grid + velocity input | 3 | grid cells plus the agent's velocity as an extra input (is PGC = velocity information?) |
| grid, look-ahead goal memory | 3 | grid cells whose goal snapshot is taken 20 cm past the goal (the geometry PGCs produce) |
| 33 % PGC, 10 cm / 40 cm shift | 3 + 3 | dose series |

126 runs × 300 M frames, trained on 42 cloud GPUs in ~7 h. Primary endpoint, fixed in advance: the within-seed paired difference (grid + PGC) − (grid) in the final 200-episode score, in arms where the code is actually used.

## 4. Results

### The code is used, and goal-homing emerges late

![learning curves](figures/fig2_learning_curves.png)

*Figure 2. Training score (mean ± SEM over seeds). Agents with a code pull away from the no-code agent, mostly after 100 M frames, when they start returning to the remembered goal.*

![reliance](figures/fig6_reliance_over_training.png)

*Figure 6. Score of intermediate checkpoints with the code intact (solid) or zeroed at test time (dashed). In every arm the trained agents depend on the code; zeroing the goal memory costs 30+ points in the two vision arms (Fig. 3).*

### Grid vs grid + PGC

| arm | grid | grid + 33 % PGC | paired Δ (8 seeds) | p (paired t) | path efficiency Δ | p |
|---|---|---|---|---|---|---|
| code only | 27.8 ± 1.6 | 31.2 ± 1.6 | **+3.4 ± 0.9** [95 % CI 1.6, 5.0] | **0.008** | +0.025 ± 0.012 | 0.08 |
| code + vision | 68.2 ± 5.2 | 71.8 ± 5.2 | +3.6 ± 5.5 | 0.53 | **+0.043 ± 0.014** | **0.018** |
| LSTM + vision | 66.4 ± 5.4 | 66.6 ± 4.6 | +0.2 ± 6.4 | 0.98 | +0.057 ± 0.030 | 0.10 |

(no-code agents: 21.4, 41.0, 41.9; untrained ≈ 1.)

**Do PGCs help when the agent also has vision?** Not on the score, which varies by ~15 points from seed to seed in these arms (an effect of 3 points would need over a hundred seeds to detect). But the trained agents do read the predictive cells specifically, and their return trips are straighter:

| code + vision, 8 paired seeds | grid | grid + 33 % PGC | paired Δ |
|---|---|---|---|
| final score (feed-forward) | 68.2 | 71.8 | +3.6 ± 5.5, p = 0.53 |
| final score (LSTM) | 66.4 | 66.6 | +0.2 ± 6.4, p = 0.98 |
| path efficiency (feed-forward) | | | **+0.043 ± 0.014, p = 0.018** |
| path efficiency (LSTM) | | | +0.057 ± 0.030, p = 0.10 |
| frames to 80 % of final score (feed-forward) | | | 35 M fewer, p = 0.09 |
| PGC-specific lesion cost (feed-forward) | | | **6.5 ± 1.6 points, p = 0.005** |
| PGC-specific lesion cost (LSTM) | | | **4.0 ± 1.5 points, p = 0.034** |

(PGC-specific lesion cost = score lost when the 169 predictive cells are silenced at test time, minus the score lost when 169 random cells are silenced. Path efficiency = straight-line distance to the goal over distance walked.)

### Controls and probes

![probes](figures/fig3_probes.png)

*Figure 3. Test-time lesions on the same trained agents. Removing the 169 PGCs costs the PGC agent 4–6 points more than removing 169 random grid cells (count-matched sham) in every arm: the policies specifically exploit the predictive cells.*

![all conditions](figures/fig4_all_conditions.png)

*Figure 4. All conditions (mean ± SEM, faint dots = seeds). In the code-only arm 100 % PGC is the best code (36.0, +39 % over grid cells, p = 0.008); giving grid cells an explicit velocity input does not help; the dose series (10/40 cm) sits between grid-only and the 20 cm code.*

![codes](figures/fig5_codes.png)

*Figure 5. Left: an example grid cell and a PGC from the same module; the PGC's lattice is displaced along the direction of travel. Right: gridness of the two cell types on the agent's own trajectories as a function of how far their activity is shifted along the path (mean ± SEM over 24 cells each), scored with the predictive-grid-cell pipeline.*

### How to read this

- **In the code-only arm the PGC gain is real but modest (+12 %)**, and those agents never learned to use the goal memory (zeroing it costs nothing), so the gain is in *search coverage*, not in vector navigation.
- **In the vision arms agents do home** (goal memory is worth 30+ points), and there PGCs leave the total score unchanged but make the return trips straighter and learning somewhat faster (code + vision: 35 M fewer frames to 80 % of final score, p = 0.09).
- A grid-cell agent given the *geometry* of a PGC goal snapshot (look-ahead goal memory) gains in the vision arm (+18, n = 3, p = 0.2), which suggests part of what a predictive code could offer downstream is a better-placed goal memory; the PGC agents themselves do not get that benefit because their current code and their goal code are displaced together.
- Caveats: one arena difficulty (a harder 4 m arena was piloted and shelved: no homing by 100 M frames, see `docs/pilot_4m_arena.md`); 300 M frames per run (the paper used 10⁹); one PGC definition (spatial shift along the path; a time-lag definition is implemented but not run at scale); 3 seeds for the controls. In the two vision arms the seed-to-seed spread is large and the biggest paired differences come from seeds whose grid-only partner learned to use the code less, so those arms are honest nulls rather than hints. LSTM runs show a benign ≤ 0.4 % of saturated gates (an earlier optimizer setting saturated 87 % and made the code invisible; see `docs/LOG.md`).
- Every statistic above was recomputed independently from the raw evaluation files by three separate checks (agreement to machine precision; audit of configs, checkpoints, population identity and episode pairing for all 72 primary runs).

## 5. Follow-up: arrangement, number and type of predictive cells

### Arranging the predictive cells

The uniform code scatters its 169 PGCs at random over the four modules, all with the same 20 cm look-ahead. Nine other arrangements were run in the code-only arm (8 seeds each, paired with the uniform and grid-only runs of the same seed, 72 runs): three module layouts (all PGCs in the finest, coarsest or middle modules), an MEC-like allotment (1:2:2:1 across modules), look-aheads proportional to each module's spacing (a quarter, the value measured by Ouchi & Fujisawa, or a half, the predicted worst case), cell-to-cell spread of the look-ahead, the quarter-spacing rule with spread (the closest match to the recorded population), and a retrospective control (20 cm *behind* the agent).

![arrangements](figures/fig7_arrangements.png)

*Figure 7. **(A)** Final score per arrangement (dots = seeds, circle = mean ± SEM; reference lines = grid-only and uniform means; right column = paired difference from uniform). **(B)** Training score. **(C)** Frames to reach a training score of 25, relative to uniform (negative = faster). **(D)** PGC-specific lesion cost: extra loss from silencing the 169 predictive cells versus 169 random cells.*

| arrangement | final score | Δ vs uniform | Δ vs grid-only |
|---|---|---|---|
| quarter-spacing shift (11 / 15 / 22 / 31 cm) | 32.8 | +1.6 ± 1.8, ns | +5.0 ± 1.8, p = 0.025 |
| MEC-weighted modules (28 / 57 / 56 / 28 PGCs) | 32.6 | +1.4 ± 2.4, ns | +4.8 ± 2.3, p = 0.07 |
| quarter-spacing + heterogeneous | 31.4 | +0.3 ± 1.0, ns | +3.7 ± 0.6, p < 0.001 |
| uniform (reference) | 31.2 | | +3.4 ± 0.9, p = 0.008 |
| heterogeneous 20 cm | 31.1 | −0.1 ± 1.3, ns | +3.3 ± 1.1, p = 0.022 |
| coarse / middle modules only | 30.5 / 29.7 | −0.7 / −1.4, ns | ns |
| retrospective (−20 cm) | 28.2 | −2.9 ± 1.7, ns | 0 |
| half-spacing shift (20 / 28 / 40 / 57 cm) | 27.6 | −3.5 ± 1.9, p = 0.11 | 0 |
| fine modules only | 27.3 | −3.8 ± 1.1, p = 0.012 | 0 |

- **Where the PGCs sit across modules barely matters**: every forward-looking arrangement with a look-ahead below half a period lands within ±1.6 points of uniform, none significantly.
- **The shift is what matters**: backward, half-period, or concentrated in the finest module (where 20 cm *is* half a period) all fall back to grid-only level, and those agents stop relying on their predictive cells (lesion cost 1–3 points instead of 6–12).
- **Secondary effects**: the MEC-weighted code learns fastest (20 M fewer frames to criterion, p = 0.016) but is the least consistent across seeds; the biology-matched code is the most consistent at the same mean as uniform.

One figure per arrangement, with what each one changes and its numbers, is in [`docs/arrangements.md`](docs/arrangements.md).

### How many predictive cells?

The quarter-spacing code (the arrangement with the highest mean) was run with 10, 25, 30, 35 and 40 % predictive cells, 8 seeds each, in the code-only arm, and compared within seed with the 33 % runs.

![concentration](figures/fig8_concentration.png)

*Figure 8. Final score vs. the fraction of predictive cells (dots = seeds, circles = mean ± SEM). The filled series is the quarter-spacing code; the hollow markers are the 20 cm-look-ahead codes of the main experiment (33 % and 100 %). 0 % = grid only.*

| predictive cells | final score | paired Δ vs 33 % |
|---|---|---|
| 0 % (grid only) | 27.8 | −5.0 ± 1.8, p = 0.025 |
| 10 % | 28.0 | **−4.8 ± 1.3, p = 0.009** |
| 25 % | 32.7 | −0.0 ± 1.1, p = 0.97 |
| 30 % | 29.9 | −2.8 ± 2.0, p = 0.20 |
| **33 %** | **32.8** | reference |
| 35 % | 31.3 | −1.5 ± 1.3, p = 0.28 |
| 40 % | 30.6 | −2.1 ± 1.9, p = 0.29 |
| 100 % (20 cm look-ahead, 3 seeds) | 36.0 | |

Ten percent is not enough: it scores like the grid-only code. From 25 % to 40 % the score sits on a plateau around 31–33, where no two points differ by more than the seed noise, and 33 % is the best tested value with 25 % tied. The 100 % code from the main experiment is above the whole plateau, so the gain does not peak at a third. Because 33 % stayed the best, the planned rerun of all arrangements at another percentage (automated, conditional on a significant gain over 33 %) was not triggered.

### Band cells

A pilot with **band cells**: each cell is a single cosine along one of its module's three lattice directions (three orientations per module, same spacings and phases), so a cell is periodic along one axis instead of a hexagonal lattice. A predictive band cell is shifted along the direction of travel exactly like a PGC. Code-only arm, 4 seeds, run locally.

![bands](figures/fig9_bands.png)

*Figure 9. Band codes next to the grid codes of the main experiment (dots = seeds, bar = mean ± SEM).*

| code (code-only arm, 4 paired seeds) | final score | paired Δ |
|---|---|---|
| grid (seeds 1–4) | 26.6 | |
| band | 25.0 | vs grid −1.7 ± 1.3, p = 0.29 |
| band + 33 % predictive | 30.4 | **vs band +5.4 ± 1.6, p = 0.041**, 4/4 seeds |
| 100 % predictive band | 32.4 | **vs band +7.4 ± 1.2, p = 0.009**, 4/4 seeds |
| grid + 33 % PGC (seeds 1–4) | 29.7 | band + 33 % vs this: +0.6 ± 2.0, p = 0.77 |
| 100 % PGC (3 seeds) | 35.9 | 100 % predictive band vs this: −4.6 ± 1.1, p = 0.054 |

A band code carries position about as well as a grid code for this agent, and predictive band cells help it as much as PGCs help the grid code: +5 points at 33 % and +7 at 100 %, in every seed, with the trained agents relying on the predictive cells specifically (PGC-specific lesion cost 7 and 22 points). The benefit of prediction is therefore not tied to the hexagonal structure; it comes from shifting a periodic spatial code forward along the path. At 100 % the predictive band code trails the 100 % PGC code by about 5 points (p = 0.05), so the hexagonal code remains the better substrate.

**Cost of the follow-up:** 124 cloud runs (arrangements 72, concentrations 40, plus retries) ≈ $200 of the $300 budget; the band pilot ran locally.

## 6. Reproduce

```bash
PY=python            # needs torch (CUDA), numpy, scipy, matplotlib
$PY -m pytest -q tests/                                                   # unit + smoke tests
$PY -m code.validate_codes --cond pgc33 --seed 1 --out runs/validation_pgc33_s1
$PY -m code.train --cond pgc33 --policy ff --vision heading_only --seed 1 \
    --frames 300000000 --encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5 --out runs/pgc33_ffb_s1
$PY -m code.evaluate --run runs/pgc33_ffb_s1 --probes intact,no_code,no_goal,untrained
$PY scripts/launch_matrix.py --wave 2            # the whole matrix on one machine (slow)
$PY scripts/gcp_fleet.py plan --wave 2 --vms 32 && $PY scripts/gcp_fleet.py launch --vms 32   # or on GCP
$PY -m code.analyze --runs_dir runs --out results && $PY scripts/make_figures.py
```

Every run's configuration, final evaluations and checkpoint probes are in `data/evals/`, the binned training curves in `data/learning_curves.csv`, and the aggregate statistics in `data/summary.json`, so the tables and most figures can be regenerated without retraining (`python scripts/make_figures.py --runs_dir data/evals`).

**Two things worth knowing before re-running.** With the paper's RMSProp ε = 0.1 the policy never leaves chance in 25 M frames (sparse reward); with a normalised step and no LayerNorm on the vision embedding, the LSTM saturates and silently ignores the code. The settings above avoid both; every log line carries saturation diagnostics. Details in `docs/LOG.md`.

## 7. Layout

```
code/         codes.py (population code) · env.py (arena + vision) · nets.py (agent) · train.py · evaluate.py · analyze.py · validate_codes.py
scripts/      launch_matrix.py (scheduler) · gcp_fleet.py (cloud fleet) · make_figures.py · package_data.py · status.py
figures/      fig0–fig9 (png + svg) · arrangements/ (one figure per PGC arrangement)
results/      RESULTS.md (all tables) · summary.json
data/         per-run configs + evaluations (250 runs) · learning_curves.csv · validation_pgc33_s1.json
docs/         LOG.md (day log, decisions, incidents) · arrangements.md (follow-up, one figure per arrangement) · pilot_4m_arena.md
tests/
```
