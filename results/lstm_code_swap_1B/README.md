# Exact population-code substitution and causal RNN responses

Replacing the predictive code with the paired ordinary grid code reduces the frozen PGC-trained models' score from **103.99 to 90.24**, a **13.75-point loss in all eight training seeds**. Replacing the supplied goal code alone costs 10.10 points; replacing the current code alone costs 5.60 points, with greater uncertainty. The substitutions directly alter LSTM cell updates and action distributions even when the preceding state and external experience are held fixed.

[Results figure](code_swap_and_dynamics.png) · [PDF](code_swap_and_dynamics.pdf) · [Numerical summary](summary.json) · [Protocol](protocol.json)

The subsequent [closed-loop gate-rescue experiment](../lstm_gate_rescue_1B/README.md) links these state effects to behavior: restoring the predictive-code-driven candidate update and output gate raises score from **90.24 to 101.34**, recovering **81%** of the mean loss. Restoring the input and forget gates together reaches **95.18**; the candidate/output pair gives the larger recovery after correction for the tested score comparisons.

The later [goal-geometry controls](../lstm_goal_geometry_1B/goal_geometry.png) test the content of the saved goal message: reflecting the sideways arrival component preserves score within a pre-run ±5-point tolerance (103.99 to 103.13), whereas reversing the predictive shift reduces score to 58.10. Moving a grid-trained model's entire goal snapshot ahead reduces score from 95.01 to 91.28. Thus, the exact sideways arrival detail appears dispensable at this tolerance, but a simple forward shift is not sufficient to confer the PGC advantage on an already grid-trained network. Reflection retains radial direction information and uses the hidden true center to construct the control. See the [updated mechanism discussion](../lstm_gate_rescue_1B/README.md#what-this-says-about-the-mechanism) and [complete new statistics](../lstm_goal_geometry_1B/summary.json).

## What was replaced

The mixed input has 343 ordinary cells and 169 predictive cells. We replaced each predictive response `g_i(x + 0.20 u)` with `g_i(x)`. Every slot remains active, and the frozen policy weights remain unchanged. This is the **same 512-cell encoding rule received by the paired grid-trained model**, evaluated at the state visited by the tested policy.

For each of the eight paired populations, we independently constructed the grid-only population and checked 512 random states. Current and goal outputs of the substituted mixed population were **bit-for-bit equal** to the grid-only population, with identical phases and wavevectors. The repository's existing `pgc_as_gc` probe is also exactly this substitution in both current and goal codes; it is not a cell-silencing probe.

We ran a fresh, independent evaluation set: **128 episodes per variant per training seed**, evaluation seed 737373, stochastic actions, standard 1,350-step episodes, 1B frozen checkpoints, and no retraining. Within a seed, variants share layouts, motor/image noise, and action-sampling uniforms. All model comparisons use the same episode scheme. There are 6,144 episodes across the four PGC-trained and two grid-trained evaluation variants.

The goal code is the population snapshot at the most recent goal hit. It is an **external memory vector supplied to the LSTM at every step**, not a goal that the LSTM must maintain without input. Before the first hit it is zero. For the goal-only intervention, we save the ordinary grid response at the same physical hit; we do not zero goal memory or change the true goal.

## Behavioral result

| Training condition | Evaluation code | Score | Return-path efficiency | Goals / 1,000 post-discovery steps |
|---|---|---:|---:|---:|
| PGC + grid | Intact mixed current and goal | **103.99** | **0.7815** | **10.516** |
| PGC + grid | Current code replaced only | 98.40 | 0.7724 | 10.009 |
| PGC + grid | Goal code replaced only | 93.90 | 0.7431 | 9.411 |
| PGC + grid | Both replaced by ordinary grid | **90.24** | **0.7293** | **9.058** |
| Grid only | Intact ordinary grid | 95.01 | 0.6993 | 9.118 |
| Grid only | Reciprocal mixed-code substitution | 93.27 | 0.7042 | 9.091 |

Each entry averages eight training-seed means. Path efficiency is conditional on episodes with successful goal returns, as in the repository's existing evaluation; it should not be read as an unconditional success metric.

| Within-PGC intervention | Score change | 95% paired t interval | Paired t p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Current code only | −5.60 | [−11.59, +0.40] | 0.0631 | 0.315 |
| Goal code only | **−10.10** | **[−13.45, −6.75]** | **0.000188** | **0.00151** |
| Both codes | **−13.75** | **[−18.99, −8.51]** | **0.000445** | **0.00312** |

Goal-only and both-code substitutions hurt all eight seeds; their exact two-sided sign-flip p-values are both 0.0078125. Current-code replacement hurts six seeds and improves two. A nonsignificant current-code score result does not establish that current PGCs are unnecessary.

The goal-only score loss is larger in its point estimate, but its direct comparison with the current-only score loss is uncertain (difference −4.50, p = 0.0977). The goal-only effect on return-path efficiency is more clearly larger: an additional −0.0293 versus current-only replacement, p = 0.00152, Holm-adjusted p = 0.00761. Goal-only replacement also reduces post-discovery goal rate by 1.106, p = 0.000095, Holm-adjusted p = 0.000664.

First-goal discovery is exactly unchanged by the goal-only intervention in every episode. This is a useful implementation control, not an independent behavioral discovery: the goal code is zero before discovery, so that branch receives identical inputs and random draws until the first hit. The current-code and both-code branches similarly have the same first-discovery times.

Giving the grid-trained model mixed inputs does not reliably improve its score (95.01 to 93.27; paired p = 0.360). The advantage is therefore not reproduced simply by adding predictive inputs to an already grid-trained network. Conversely, a PGC-trained network supplied with grid inputs does not become identical to a grid-trained network: its weights and prior learning remain different. Its mean score is 4.77 points lower here, but that comparison is uncertain (p = 0.165); its conditional return paths remain more efficient by 0.0301 (p = 0.00964, Holm-adjusted p = 0.0385).

The earlier canonical 200-episode evaluations gave the same directional substitution result: **107.12 to 91.38**, a 15.74-point loss. Those absolute scores differ because the fresh evaluation uses a different random seed and episode set; the two evaluations are not pooled.

## What changes inside the RNN, before behavior can diverge?

We recorded the intact PGC policy's natural experience and evaluated counterfactual branches with the same visual embedding, previous action, previous reward, and goal-hit history. Only the population-code input changed. Diagnostic samples are taken after first discovery, every 20 steps after step 40.

**Immediate response:** start both branches from the exact same preceding hidden state `h` and cell state `c`, then replace the input for one step. Thus, any immediate state or policy difference is caused by the input substitution, not by visiting a different position or choosing a different previous action.

**Sustained response:** run additional recurrent branches throughout the same intact experience, maintaining each branch's own recurrent state under its altered code. These branches are forced to see intact observations/actions/rewards; they do not control the environment. This isolates repeated input and recurrent effects from behavioral feedback.

| Intervention | Immediate action-distribution change | Sustained action-distribution change |
|---|---:|---:|
| Current only | 0.0342 | 0.0641 |
| Goal only | 0.0431 | 0.1107 |
| Both | **0.0529** | **0.1223** |

The measure is total variation distance, `0.5 * sum(abs(p_action - q_action))`, on a 0–1 scale. It is not the percentage of sampled actions that differ. For both-code substitution, the preferred action changes in **5.14%** of immediate probes and **12.33%** of sustained probes. The mean per-unit hidden-state RMS change is 0.1177 immediately and 0.1755 under sustained substitution. Cell-state RMS change is 0.3953 immediately and 13.1126 under sustained substitution; cell states are unbounded, so their numerical scale is not comparable to bounded hidden output.

An unchanged-input recurrent fork, evaluated with the same branch batch size, showed **zero** hidden, cell, and policy drift in every seed. This controls for numerical differences being mistaken for an intervention effect. The explicit four-component LSTM update also matched the original LSTMCell in the full experiment.

## Which LSTM computation transmits the input effect?

For the same preceding state, we reconstructed the LSTM update:

`c_new = forget_gate * c_old + input_gate * candidate_update`

`h_new = output_gate * tanh(c_new)`

After substituting the code, we restored one component to the value it would have had with the intact predictive input, leaving the other three altered. We then measured how much of the immediate action-distribution change disappeared.

| Component restored | Current-only swap | Goal-only swap | Both-code swap |
|---|---:|---:|---:|
| Input gate | 11.2% | 12.0% | 11.5% |
| Forget gate | 2.4% | 3.0% | 2.9% |
| **Candidate cell update** | **38.4%** | **35.8%** | **35.1%** |
| Output gate | 22.2% | 21.5% | 21.8% |

The largest local effect is through the **candidate cell update**, followed by the **output gate**. This identifies a causal route for the immediate policy response: predictive input affects the content proposed for the cell state and the exposure of that state to the actor. It goes beyond observing larger input weights or different PCA clouds.

These restorations are nonlinear, interacting interventions. Their percentages need not sum to 100%, and they are percentages of immediate policy-distribution change, **not percentages of navigation score explained**. The separate [closed-loop gate-restoration experiment](../lstm_gate_rescue_1B/README.md) tests whether these pathways also recover performance and finds substantial partial recovery through the candidate/output pair.

## Does a brief substitution permanently change the recurrent state?

Every 100 steps, after goal discovery, we forked the intact preceding state, substituted the code for **one decision step**, then restored intact input. The fork and control then received identical intact future experience. All pulse cohorts have complete measurements at all horizons, and matched no-op pulse controls were included.

For a pulse affecting both codes:

| Time after altered step | Policy total variation | Hidden-state RMS difference | Cell-state RMS difference |
|---|---:|---:|---:|
| Immediate | 0.05285 | 0.11775 | 0.39485 |
| 0.067 s | 0.03705 | 0.04045 | 0.32686 |
| 0.333 s | 0.00675 | 0.00758 | 0.24171 |
| 1 s | 0.00172 | 0.00243 | 0.17844 |
| 2 s | 0.00082 | 0.00126 | 0.14482 |
| 4 s | 0.00049 | 0.00075 | 0.11534 |

The policy perturbation falls to approximately **3.25% of its initial value after one second**, and **0.93% after four seconds**, averaging within-seed ratios. A measurable cell-state trace remains longer. Thus, under matched future input, an isolated substitution has a largely transient policy effect, while sustained substitution continually drives different updates. This is not a test of attractor topology or of autonomous recovery after the policy takes a different action.

## Interpretation and limits

These experiments narrow the earlier hypothesis. The **supplied predictive goal representation** has a clear causal effect on homing, and it can affect navigation even when the live population code is ordinary grid code. Current PGC input also changes the circuit, but its standalone behavioral contribution is less certain in the score endpoint. The state interventions show direct changes in candidate updates, output gating, recurrent cell state, and action probabilities.

They do not yet identify exactly what information the goal code contributes. A predictive snapshot includes position displaced along the goal-arrival direction. It could provide an arrival-direction cue, a more useful comparison target, or a location closer to the goal center than the first physical contact point. Distinguishing those explanations requires additional geometry-matched controls. The goal snapshot is sampled at the hit position inside the 30-cm goal region, not necessarily its center.

All interventions are at evaluation time. They change the input distribution relative to training, and causal effects within a trained model do not by themselves isolate why training with PGCs produced better weights. No neurons are compared by raw coordinate across separately trained networks. The appropriate causal comparisons here are forks of the **same frozen RNN**.

## Reproduction and statistical scope

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_code_swap.py --device mps --episodes 128 --n-envs 64
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/summarize_lstm_code_swap.py
```

The implementation is in [`scripts/lstm_code_swap.py`](../../scripts/lstm_code_swap.py), with summaries in [`scripts/summarize_lstm_code_swap.py`](../../scripts/summarize_lstm_code_swap.py). Per-seed JSON files retain all episode scores, first-goal times, return metrics, diagnostic sample counts, state/gate effects, no-op controls, and checkpoint hashes.

Training seeds are the eight statistical replicates. Episodes and recurrent-state samples are not treated as independent training runs. Behavioral summaries include paired t intervals, exact two-sided sign-flip p-values, and Holm correction within each eight-comparison outcome family. Correction is not global across the different outcome families; secondary endpoints and circuit analyses are exploratory. Diagnostic uncertainty is summarized across per-seed means. Figures use SEM, while the tabled score intervals use the paired t distribution.
