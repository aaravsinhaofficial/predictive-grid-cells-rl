# Plan for the code-variation experiment

Written before any of these runs was started (2026-11-27). It fixes what is compared, how, and for how long, so the write-up cannot pick results after the fact.

**Question.** Each follow-up from the 22 November meeting changes one property of the spatial code. Does the change make the code better to navigate with?

**Agents.** Unless stated, the code-only agent: a feed-forward policy whose image shows only its heading, so the code is its only source of position. Eight seeds per code. All codes of a seed share the same cells, so every comparison is paired within seed. Code names are defined in `scripts/codes/matrix.py`.

**Rooms.** The original open room, and a room with a wall, one door and an obstacle (layout DO1 for every seed). The agent cannot see the walls. It has to learn the room from the code.

**Outcome.** The training score (10 points per goal) averaged over the last 25 million frames of training. The 200-episode test of the final network is shown beside it. Earlier runs showed that the test is noisier (SD of a paired difference 4.7 against 3.1).

**Tests.** Paired t-test over the 8 seeds, Holm-corrected within each list of named comparisons below. The exact sign-flip p is given uncorrected as a check. No seed is left out. Training lengths are fixed here. Anything not named below is reported as a difference with a 95 % confidence interval and called descriptive. "No difference" is always written as "within ± X points".

## Open room

Core codes run for 600 M frames, with a readout at 300 M on the way: `gc100`, `u20`, `prop10`, `prop27`, `mec`, `gard`, `rand`, `side`.

Named comparisons at 600 M (4 tests):

1. `prop10` − `u20`: one look-ahead per module at 10 % of its spacing, against 20 cm everywhere.
2. `mec` − `u20`: the Ouchi & Fujisawa shift distribution.
3. `gard` − `u20`: the shifts recorded in the lab's own reanalysis.
4. `side` − `u20`: the same shift pointing sideways.

Named in advance, descriptive: many distances against one, as the mean of `rand` − `u20`, `mec` − `prop27` and `gard` − `prop10`. Also `prop27` − `u20`, `prop27` − `rand`, and every code against `gc100`.

Screen codes run for 300 M frames: `two`, `lr30`, `p100`, `gcvar`, `gcvar3`, `gcshear`, `gcjit`, `mecreal`, `m1`, `m1var`.

Named comparisons at 300 M (4 tests): `gcvar` − `gc100`, `gcshear` − `gc100`, `gcjit` − `gc100`, `mecreal` − `mec`. Descriptive: `gcvar3`, `two`, `lr30`, `p100`, and `m1var` − `m1` (one module, the case in which Redman et al. 2025 predict that variability helps. With four modules they predict little effect.)

## Door room

`gc100`, `u20`, `prop10`, `mec`, `gard`, `rand` for 600 M frames. No code (`none`) for 3 seeds and 300 M frames as the floor.

Check at 100 M frames: the room counts as learnable from the code only if `gc100` is above `none`. If it is not, the door-room runs stop and the write-up says so.

Named comparisons at 600 M (3 tests): `u20` − `gc100`, `prop10` − `u20`, `mec` − `u20`. Descriptive: `rand` − `u20`, `gard` − `u20`, and whether `prop10` − `u20` differs between the two rooms.

## Trainable look-ahead (exploratory, no test)

One forward look-ahead per module, shared by that module's predictive cells and trained together with the policy. Two starts, 0 cm and 20 cm. Open room and door room, 4 seeds each, 600 M frames. Reported: the four look-aheads in centimetres and as a fraction of spacing, and whether the two starts end at the same values. Control: the same runs with the sign of the look-ahead gradient scrambled, which shows how far noise alone moves them.

## The paper's agent (LSTM with first-person vision)

Trained from scratch on the mixture of 21 changed rooms for 500 M frames, 8 seeds: `gc100`, `u20`, `prop10`, `mec`. One outcome: the training score over the last 50 M frames. `u20` − `gc100`, `prop10` − `u20` and `mec` − `u20` are reported as estimates with 95 % confidence intervals, not as tests. Earlier from-scratch runs in changed rooms showed no gap between the grid and the predictive code at 450 M frames, and this agent is about three times noisier than the code-only agent. So this part can only show differences above roughly 12 points.

## What the design can and cannot see

With 8 seeds, the smallest difference found four times out of five is about 3.6 points at 300 M frames and about 5 points at 600 M. The benefit of predictive cells itself is about 3.5 points at 300 M and about 11.5 at 600 M. A code that loses half of that benefit should be visible at 600 M. Differences below about 3 points between codes that all work cannot be resolved.

Two properties of the task matter when reading the results. The look-ahead point is the position plus the shift along the direction of travel, with no check for walls, so it can lie outside the room or behind a wall. And the remembered goal code is taken on the rim of the 30 cm goal zone, which ties the useful look-ahead to that size.

## Added after the launch

Added on 27 November, about an hour after the runs above had started and before any result was looked at.

**Trainable look-ahead in the paper's agent (exploratory, no test).** The same four trainable look-aheads, one per module, in the LSTM agent with first-person vision, trained from scratch on the mixture of 21 changed rooms for 500 M frames. Two starts (0 cm and 20 cm), 4 seeds each, plus one control run per start with the gradient sign scrambled. Reported as for the code-only agent. The spread of the control runs is the yardstick for what counts as movement. If the two starts do not end at the same values, that is reported as not resolved, not as evidence for two optima.

**Reading of all trainable runs.** Each look-ahead moves by at most 1 mm per update. Over 600 M frames noise alone moves it by about 2 cm. Differences of a few centimetres between runs or starts are therefore noise.
