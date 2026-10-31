# Causal rescue of navigation through LSTM components

**Restoring the PGC-driven candidate cell update and output gate recovers about 81% of the mean score lost when predictive population inputs are replaced with ordinary grid code.** Score rises from **90.24 to 101.34**, versus **103.99 intact**. Restoring the other two components together reaches only **95.18**. The candidate/output pair gives a larger recovery than that matched pair control, including after correction for the tested score comparisons.

This is a causal intervention inside each frozen PGC-trained LSTM, followed by actual navigation. It tests a computational route from predictive input to behavior, beyond correlations between weights, PCA, and score. The experiment identifies an important route used by the trained circuit; it does not establish an exclusive minimal circuit or explain how training produced the weights.

[Gate-rescue figure](gate_rescue.png) · [PDF](gate_rescue.pdf) · [Numerical results](summary.json) · [Input-swap and state-dynamics experiments](../lstm_code_swap_1B/README.md)

## Intervention

The LSTM computes four components from its input and preceding recurrent state:

`c_new = f * c_old + i * candidate`

`h_new = o * tanh(c_new)`

Here `i` is the input gate, `f` is the forget gate, and `o` is the output gate. The actor reads `h_new`.

At every decision, for **the same actual preceding h/c of that branch**, we computed the four components twice: once with its ordinary-grid current and goal inputs, and once with the original mixed predictive inputs. We then selected which components received the predictive-code values:

- **Intact:** all four use mixed-code values.
- **Grid substitution:** all four use ordinary-grid values.
- **Single-component rescue:** only the named component uses its mixed-code value.
- **Candidate + output rescue:** candidate and output use mixed-code values; input and forget gates use grid-code values.
- **Input + forget control:** input and forget use mixed-code values; candidate and output use grid-code values.

Each branch retains its own evolving recurrent state, chooses its own stochastic actions, and moves through the environment. Goal snapshots are saved from that branch's actual goal-hit states. We do not transplant hidden states from a different trajectory. All weights stay fixed.

A rescue branch **still has access to predictive information through the restored components**. This is not a claim that a pure grid-input policy reaches the rescued score without PGC information. It is a way to isolate where that information enters the computation.

## Results across eight training seeds

Each condition uses 128 paired evaluation episodes per seed, with the same evaluation seed, episode layouts, environmental noise, and action-sampling uniforms as the input-swap experiment. The intact and grid-substitution baseline episode records reproduce exactly across the experiments. The supplemental input/forget pair control also reproduces every grid-substitution baseline episode record exactly.

| PGC-trained evaluation | Score | Mean score loss recovered | Return-path efficiency | Goals / 1,000 post-discovery steps |
|---|---:|---:|---:|---:|
| Intact mixed code | **103.99** | Reference | **0.7815** | **10.516** |
| Ordinary grid substitution | **90.24** | 0% | **0.7293** | **9.058** |
| Restore input gate | 93.23 | 22% | 0.7462 | 9.371 |
| Restore forget gate | 93.52 | 24% | 0.7413 | 9.458 |
| Restore candidate update | 97.16 | 50% | 0.7528 | 9.755 |
| Restore output gate | 94.95 | 34% | 0.7490 | 9.544 |
| **Restore candidate + output** | **101.34** | **81%** | **0.7702** | **10.215** |
| Restore input + forget | 95.18 | 36% | 0.7497 | 9.583 |

Recovery is the ratio of group-mean differences, `(rescue − grid) / (intact − grid)`. It is descriptive, not a formal percentage of causal mediation, and effects of different restorations are not additive.

The candidate/output pair:

- Improves score over grid substitution in **all eight seeds**, by **11.09 points**, 95% paired t interval **[7.32, 14.87]**; paired t p = **0.000223**, Holm-adjusted p = **0.00223**, exact two-sided sign-flip p = **0.0078125**.
- Beats the input/forget pair by **6.16 points**, 95% interval **[2.42, 9.90]**; p = **0.00595**, Holm-adjusted p = **0.0357**, exact sign-flip p = **0.015625**.
- Restores **78% of the mean return-path-efficiency loss** and **79% of the post-discovery goal-rate loss**. Its path-efficiency advantage over the input/forget pair is **0.0205**, Holm-adjusted p = **0.0103**; its post-discovery goal-rate advantage is **0.632**, Holm-adjusted p = **0.0142**.

The candidate update alone improves score by **6.91 points**, Holm-adjusted p = **0.0446**. Its score advantage over output-gate restoration alone is uncertain (difference 2.21, p = 0.120), so the behavioral results do not establish a clean ranking of every single component. Forget-gate restoration also gives a consistent smaller score improvement, **3.27 points**, Holm-adjusted p = **0.0337**, despite having a small effect in the immediate action-distribution probe. Immediate sensitivity and long-run behavioral importance need not rank components identically.

This is substantial **partial** recovery. Candidate/output rescue still has a mean score **2.66 points below intact**, with interval **[−6.22, +0.91]** for rescue minus intact. The nonsignificant score difference is not an equivalence test. Post-discovery goal rate remains **0.301 below intact**, Holm-adjusted p = **0.0215**. Nor does behavioral recovery imply that the complete hidden/cell states are restored.

## What this says about the mechanism

The [input-swap experiment](../lstm_code_swap_1B/README.md) showed a clear cost of replacing the **supplied predictive goal snapshot**, and a smaller, less certain score effect of replacing the live current code. Its matched-state probes showed that predictive-code removal changes the candidate cell update and output gate enough to alter policy probabilities before the agent experiences a different trajectory.

The closed-loop experiment here connects that circuit effect to navigation: routing predictive information through the **candidate update and output gate** recovers most of the lost performance, while routing it through the equally sized **input/forget pair** recovers substantially less.

The resulting working mechanism is:

**Predictive spatial/goal features → content proposed for cell-state updates and exposure of cell state to the actor → more effective homing.**

This is a more specific account than a difference in PCA variance or input-weight magnitude. It is also a qualified account: predictive information is distributed across interacting components, and the data do not show that only the candidate/output pair matters. The gate rescues restore contributions of both current and goal inputs; they do not separately localize every goal-specific versus current-specific effect.

The subsequent [goal-geometry experiment](../lstm_goal_geometry_1B/goal_geometry.png) narrows what the goal message contributes. Reflecting its sideways arrival component while preserving the 20-cm offset and distance to the goal center retains score (103.99 to 103.13); reversing the offset reduces it to 58.10. The reflected point differs by about 18 cm from the original, so this is a substantial perturbation. The mirror-minus-original 90% paired interval is [−2.45, +0.73], within the pre-run coarse tolerance of ±5 points. Reflection retains radial arrival information; this does not show that all direction information is unnecessary.

Simply moving the entire ordinary-grid goal snapshot ahead hurts every grid-trained seed (95.01 to 91.28), and supplying the exact goal center gives no clear score gain (grid 95.45; PGC 101.00). These results support a working account in which the PGC-trained network learned to use the relation between the pickup location and the ahead/inward spatial signal. They do not establish a particular decoded spatial variable or show that closer-to-center coordinates alone explain the advantage. Center and reflection controls use privileged goal coordinates. All 3,072 repeated baseline episode records match exactly; all goal-only branches have identical first-discovery records. [Full statistics and limitations](../lstm_goal_geometry_1B/summary.json) and [rollout implementation](../../scripts/lstm_goal_geometry.py) are retained.

The goal code remains an external snapshot supplied every step, not an independently maintained internal goal memory. These evaluation interventions reveal how the trained networks use predictive features; they do not by themselves explain the learning process that created their parameters. The gate rescues identify a route within PGC-trained networks, not a different gate architecture in grid-trained networks.

## Controls, provenance, and scope

- All checkpoints are frozen at exactly 1B frames; hashes are saved in the per-seed files.
- Ordinary-grid substitution is identical to the paired grid-only encoding rule, with no cells silenced.
- All branches use the same explicit LSTM update. It was checked against the original LSTMCell; the observed maximum reconstruction error in these full runs was **0**.
- The supplied environmental noise and stochastic-action uniforms are shared across intervention branches, while each branch's behavior and recurrent state evolve normally.
- The input/forget pair control was added after inspecting early gate-rescue results. This is an exploratory mechanism study, not a preregistered confirmatory mediation analysis. All tested single components and both tested pairs are reported.
- The eight training-seed means are the statistical replicates. Episodes are not treated as independent model replicates. Holm correction is applied within each ten-comparison outcome family, not globally across outcome families. Raw and corrected statistics are saved.
- Path-efficiency means condition on episodes with successful returns; score includes every episode. Figures use SEM across seeds, while reported difference intervals use paired t intervals.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_gate_rescue.py --device mps --episodes 128 --n-envs 64
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_gate_rescue.py --device mps --episodes 128 --n-envs 64 --variants both_grid restore_input_forget --out results/lstm_gate_rescue_1B/pair_control
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/summarize_lstm_gate_rescue.py
```

Implementation: [`lstm_gate_rescue.py`](../../scripts/lstm_gate_rescue.py), [`summarize_lstm_gate_rescue.py`](../../scripts/summarize_lstm_gate_rescue.py). The protocol and every episode-level result are retained here and under `pair_control/`.
