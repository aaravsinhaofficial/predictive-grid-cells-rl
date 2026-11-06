# Native LSTM equilibrium branch search

**512 bounded Newton starts recovered the previously known activity branch in every case. No additional stable or unstable activity branch was found.**

This search specifically addressed whether the earlier residual minimization missed unstable fixed points. It used the original frozen 1B-step agents, the same saved context0 inputs, two seeds per population, and goal-withheld / actual-goal-held conditions. No network was retrained.

| Population | Seed | Goal cue | Qualified starts | Prior / new activity branches | Unstable branches | Equilibrium cell / native cell scale |
|---|---:|---|---:|---:|---:|---:|
| Grid | 1 | Held | 40/64 | 1 / 0 | 0 | 35472.3× |
| Grid | 1 | Withheld | 40/64 | 1 / 0 | 0 | 4913.3× |
| Grid | 2 | Held | 40/64 | 1 / 0 | 0 | 2474306.1× |
| Grid | 2 | Withheld | 40/64 | 1 / 0 | 0 | 35919.5× |
| PGC + grid | 1 | Held | 37/64 | 1 / 0 | 0 | 616.0× |
| PGC + grid | 1 | Withheld | 40/64 | 1 / 0 | 0 | 94.8× |
| PGC + grid | 2 | Held | 32/64 | 1 / 0 | 0 | 834721.7× |
| PGC + grid | 2 | Withheld | 40/64 | 1 / 0 | 0 | 104748.7× |

## What changed in the search

Each case used 16 native hidden states, zero, 23 uniformly random states within the legal hidden-state bounds, and 24 fixed-scale perturbations of the known root. The exact equilibrium identity `c = i*g/sigmoid(-forget_logit)` reduced the problem to 256 hidden coordinates. An analytic-Jacobian hybrid Newton solver replaced Adam/LBFGS minimization. A preassigned 24 starts per case used deflation against the known branch to discourage returning to it. Each start had a strict 200-residual-evaluation limit.

## What was checked

A returned state qualified only when both hidden consistency and the numerically stable original 512-dimensional residual had RMS at most 1e-8, all coordinates were finite, and hidden states respected their bounds. Every prior qualified root was revalidated. The full recurrent Jacobian used stable derivatives of saturated gates, and four finite-difference direction checks were retained for every branch. Eigenvalues, hidden-state fractions, and actor-logit gains are recorded.

Across all cases, 309/512 starts qualified. The largest directional Jacobian error was 2.41e-10. Four independent unit tests pass, including a synthetic LSTM with a known unstable root that the deflated solver recovers.

## What the result means

The extra search gives no new unstable equilibrium from which to construct a local two-dimensional unstable SSM in these eight input settings. The recovered equilibria have extremely large cell-state magnitudes relative to native activity. Separate native full-state rollouts for 100 seconds quantify their distance in hidden state, cell state, and action logits; those measurements are in the aggregate JSON. These roots describe mathematical long-term behavior under fixed inputs and need a reachability argument before being used to explain navigation.

Branches are grouped by **hidden-state RMS distance of 1e-5**. This is an activity grouping, not a claim that cell states coincide. All 512 full candidate states, all residual evaluations, and the previous source states remain available; the aggregate JSON reports the maximum cell spread within each activity group. The finite search does not enumerate every possible equilibrium.

## Reproduction and files

Run `/opt/anaconda3/bin/python scripts/native_lstm_equilibrium_branches.py` and `/opt/anaconda3/bin/python -m pytest -q tests/test_native_lstm_equilibrium_branches.py` from the repository root. The protocol and each case record the exact script and checkpoint SHA256 hashes. No cloud compute was used.

- `aggregate_summary.json`: compact results, cell scales, derivative checks, and 100-second accessibility.
- Each case: `inputs.npz`, `all_starts.npz`, `outcomes.json`, `roots.npz`, and `summary.json`.
- Existing results and trained weights were not modified.
