# RL map benchmark: implementation and compute audit

This is a proposal, not a launched experiment. The previous study's files and cloud accounting remain unchanged.

The original authorization is **$500 total**. The completed study's conservative envelope is **$43.7364**, leaving **$456.2636**. The previous amount includes its storage allowance and is carried forward in full; it is not reset or credited back. See the [completed audit](ring_cloud_final_cost_bound.md).

## Recommended first experiment

Run a small **original-square A2C pilot**, using the existing `code/train.py`, before deciding whether a circular-track reduction is needed. This is the shortest implementation path to genuine reward-trained navigation with the repository's actual maps and goal-memory semantics. It provides a direct bridge to the existing sixteen 1B-frame LSTM agents.

The existing setup is already vectorized in PyTorch. Its 3.03M-parameter policy has a four-layer image encoder, a 256-unit LSTM, six action logits, and a value head. Preserve the existing successful optimizer settings: LayerNorm, learning rate `3e-4`, RMSProp epsilon `1e-5`, 32 environments, 100-step rollouts, and action repeat four. `--vision none` alone does **not** remove the convolutional encoder or its training cost.

The primary square configuration is the actual 2.5m arena: `arena_half=1.25`, goal radius 0.30m, 1,350 decisions per episode. Preserve the original 512-cell, four-module hexagonal population and 20cm predictive displacement. Paired Grid/PGC conditions use the same population seed, map parameters, initialization seed, and frame budget.

## What must remain faithful

- The goal is hidden and fixed within an episode. Reaching its radius gives **+10 raw reward**, clipped to +1 only for the learning update, and respawns the agent at a new pose while keeping the goal.
- The supplied goal code is **zero before the first reward**. At a hit it becomes the population activity at the actual entry position and arrival direction, not the true goal center. Predictive slots consequently shift inward relative to this entry point. The original goal is not handed to the policy as coordinates or a bearing.
- Current code uses actual position and the last sufficiently large movement direction; this direction is held while nearly stationary. Position is used to generate observations, not additionally supplied to the policy unless a declared control does so.
- The actor chooses actions from reward learning. There is no supervised heading target, hand-built controller, or imposed recurrent ring in this benchmark.

The original maps also give first-person wall cues. Historical results show that early reward improvement can come from more efficient searching without using goal memory. Therefore the pilot must evaluate intact, no-goal, ordinary replacement, and an untrained policy, with paired episode layouts. A short pilot can validate throughput and learning health; it cannot establish a mature PGC advantage. The original LSTM difference became clear much later, around 600M–1B frames.

## Throughput and cost

Repository history reports approximately **13,000–19,000 environment frames/s** on L4 for these full agents ([training log](../LOG.md)). These are historical measurements, not a new benchmark. One frame is one of the four action-repeat substeps; 25M frames are 6.25M agent decisions. Single-run projections using the existing conservative **$1.25/VM-hour** envelope are:

| Budget per model | Time at historical L4 rates | VM/boot-disk envelope per model |
|---|---:|---:|
| 5M frames | 4.4–6.4 min | $0.09–0.13 |
| 25M frames | 22–32 min | $0.46–0.67 |
| 100M frames | 1.46–2.14 h | $1.83–2.67 |
| 300M frames | 4.39–6.41 h | $5.48–8.01 |
| 1B frames | 14.62–21.37 h | $18.27–26.71 |

These projections exclude startup, evaluation, collection, and any new storage allowance. They do not assume a speedup from concurrent processes. At 13k frames/s, 24 models ×300M frames would consume about $192.31 of the conservative VM envelope. Sixteen models ×1B would consume about $427.35 before overhead, too close to the remaining limit to commit before measurement. A slower 5k frames/s stress case costs $20.83 per 300M model; a full 24-model cohort would exceed the remaining budget.

Official on-demand Iowa prices remain $0.706832276/h for G2-standard-4 with L4 and $0.853624312/h for G2-standard-8 with L4. A T4 attachment is $0.35/h plus its host; our prior host-price audit totals $0.729998/h for N1-standard-8 plus T4, before the small boot-disk charge. The $1.25 reservation is deliberately higher. [Official GPU pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [prior price audit](ring_cloud_plan.md).

## Circular-track alternative

A circular track can evaluate the **same original 2D hexagonal maps** at `x = R(cos θ, sin θ)`, using the actual tangent movement direction for `g(x + 0.20 u)`. Its goal snapshot and reward/respawn rules can remain the same. However, its constrained movement, action space, and geometry would be a new task. The 20cm displacement should remain the original straight tangent displacement, not silently become an angular phase shift or an oracle direction input.

Removing image convolution and using a small reward-trained LSTM should reduce compute, but there is no measured RL throughput or convergence budget for this proposed task. For planning only, 20M frames at 10k–50k frames/s would take 7–33min, and 100M frames 0.56–2.78h. These are throughput scenarios, not promises that learning succeeds in that budget. A fixed two-hour pilot is safer than a large advance allocation.

Two scientific pitfalls matter more than speed: (1) simply circling can repeatedly find the goal, so success alone need not show goal-memory use; and (2) an informative current position code is present every step, so a recurrent spatial attractor is not required for navigation. No-goal controls and predetermined cue blackouts can test memory use, with the blackouts explicitly identified as a task modification. A ring-shaped physical track does not by itself establish a neural ring attractor.

## Bounded pilot setup

After the experiment is selected, use one on-demand GPU, a two-hour provider-enforced **DELETE** deadline, the already pinned PyTorch image, and an automatically deleted 100GB boot disk. Reserve **$2.8125** for this worker, including the existing 15-minute allowance. Prefer L4 for the CNN workload; if capacity is unavailable, the tested N1-standard-8/T4 setup is an appropriate pilot whose speed must be measured. Do not assume the tiny-compass training rate transfers to this larger RL model.

Start with two matched 5M-frame Grid/PGC jobs, then use the measured rate to choose a fixed 25M-frame continuation or a later cohort. Save resumable optimizer/RNG checkpoints and intermediate models before the deadline. A two-hour pilot cannot safely promise two 25M jobs on an unbenchmarked slow GPU. Package only the selected original code, new pilot wrapper/manifest, and its protocol; do not upload unrelated model trees.

Use a **new RL namespace and new accounting ledger** with a permanent $43.7364 prior-study charge and the same $500 total ceiling. A new $25 storage/transfer allowance, if reserved, reduces the remaining VM allocation to $431.2636. Keep the old final ledger snapshot and reports immutable. Reuse the bounded worker's identity checks, periodic artifact uploads, collect-before-delete logic, and independent boot-disk audit. The older `gcp_fleet.py` scheduler is useful for reading historical commands but should not be used for this pilot: its safety controls lack the newer cumulative cost reservation and provider-enforced deletion deadline.

No VM, bucket, or other cloud resource was created or changed during this audit.
