# Finding a learned ring inside the navigation LSTM

Literature checked 5 November 2026; state-noise methods rechecked against version 2, and action-dependent gating examined, on 6 November 2026 UTC. This note guides the next experiments; it reports no new experimental result. The target is a mechanism inside the original reinforcement-trained navigation agent: predictive spatial input corrects a persistent circular state, and that correction improves the agent's navigation.

The most useful precedent is a **slow ring**: activity quickly returns toward a closed curve after an off-curve disturbance, while its position around the curve changes much more slowly. This allows a finite-time memory mechanism even when long simulations eventually reach a few preferred states.

## Five primary precedents

### 1. Navigation, context, fixed points and eigenvectors

[Low, Giocomo & Williams (2023), *Remapping in a recurrent neural network model of navigation and context inference*, eLife](https://elifesciences.org/articles/86943).

The authors trained a randomly initialized 248-unit ReLU RNN to integrate velocity on a circular track and remember a brief context cue. Their supervised objective combined angular-position and context errors. Training sequences grew from one step by one step every 50 updates; gradient clipping and this curriculum helped training. Under zero velocity and no context cue, fixed-point optimization and Jacobian eigenvectors revealed context-specific rings and intervening saddles. Slow directions along a ring represented position; unstable directions between rings represented context changes. A two-dimensional extension produced toroidal structure.

**Use here:** search full-state fixed/slow points and measure eigenvector alignment with the proposed circular variable. Treat a two-dimensional location memory as potentially two-dimensional. Their [analysis code](https://github.com/ahwillia/rnn_remapping_paper) supplies a reproducible implementation reference. Our navigation objective remains reinforcement learning.

### 2. A ring can be one computation inside a larger network

[Driscoll, Shenoy & Sussillo (2024), *Flexible multitask computation in recurrent networks utilizes shared dynamical motifs*, Nature Neuroscience](https://www.nature.com/articles/s41593-024-01668-6).

The authors trained recurrent networks on multiple cognitive tasks, including remembering a stimulus direction during a delay. They optimized the residual between a state and its next state under a specified input, then studied local stability and how fixed-point arrangements changed across task epochs. Memory rings could participate in larger task-dependent computations. Changing context inputs changed the dynamical structure.

**Use here:** analyze the actual navigation inputs and task epochs separately. A ring may be present during a memory interval even if pooled trajectories have complicated geometry. Interpolating predictive input strength can reveal how that input changes a conditional landscape. The [author code](https://github.com/lauradriscoll/flexible_multitask) and [fixed-point finder](https://github.com/mattgolub/fixed-point-finder) are methodological starting points; the navigation LSTM needs its complete hidden and cell state.

### 3. The closest architecture precedent: slow rings in LSTMs

[Ságodi, Martín-Sánchez, Sokół & Park (2024), *Back to the Continuous Attractor*, arXiv version 2](https://arxiv.org/html/2408.00109v2).

Appendix S7.7 finds slow rings in trained LSTMs and GRUs, with different numbers of stable points and saddles. The analysis follows autonomous trajectories, fits a periodic curve to slow portions, and measures flow and stability. Discrete gated-network flow is approximated by next-state differences. Their task output supplies an angular coordinate; ours requires independent validation.

**Use here:** test transverse recovery and phase drift over task-relevant and longer durations. Retain phase coverage, fixed-point topology and drift. Calibrate speed thresholds and state scaling to this LSTM.

Section 4.1 explicitly includes state noise in training all listed architectures, including LSTMs. Section 6 and the limitations connect that training to robust fast–slow solutions. Appendix S7.6.3 analyzes trajectories with noise switched off. The variance of 1/100 in S7.3 belongs to the vanilla-network formulation; we have not inferred an identical LSTM injection rule or scale. Our original agents and first 25M pilot had no injected recurrent-state noise. The separately frozen [activity-noise RL arm](rl_visual_memory_noise_protocol.json) now tests this robustness principle against a contemporaneous noiseless continuation, using the same navigation task.

### 4. Learned directional computation

[Cueva, Wang, Chin & Wei (ICLR 2020), *Emergence of functional and structural properties of the head direction system by optimization of recurrent neural networks*](https://arxiv.org/abs/1912.10189).

Training recurrent networks to integrate angular velocity produced head-direction representations and functionally distinct compass and shifting populations. The authors connected learned activity and connectivity to directional computation through model analysis and in-silico interventions.

**Use here:** ask which state directions preserve an angle and which move that angle. This is a precedent for interpreting a learned mechanism. The experiment required now retains the navigation LSTM and its learned actor; the previous standalone supervised compass model is already a separate result.

### 5. Reward learning can produce stable spatial memory

[Wang, Wang, Xu & Pan (2024), *A working memory model based on recurrent neural networks using reinforcement learning*, Cognitive Neurodynamics](https://link.springer.com/article/10.1007/s11571-024-10137-6).

A recurrent actor and reward-predicting baseline learned a spatial working-memory task. Population activity maintained stimulus information during the delay despite varied single-unit activity. Memory stability, stimulus separability and choice accuracy improved during learning. The initially available publisher abstract was supplemented on6November by the [full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564475/). It uses GRUs, eight possible stimulus locations, a1500ms delay and a brief final choice. Correct choices receive positive reward; premature, wrong or missing choices receive negative reward. Its stable encoding subspace is evidence for learned memory, not a demonstration of an attracting continuous ring.

**Use here:** reward-only memory training has a concrete precedent. Its result motivates testing memory in our actor, with ring geometry and attracting dynamics established by the additional assays above.

### 6. Navigation attractors can depend on behavioral state

[A split attractor design for rapidly writing a navigational goal, Nature (2026)](https://www.nature.com/articles/s41586-026-11144-9) combines fly navigation recordings, circuit interventions and modeling. The h∆K and PFG populations share persistent goal-related activity during straight runs, while their activity differs during turns. A model with gated recurrent coupling explains how a heading input can be written into persistent goal activity. This is a biological circuit with an explicit model, not a reward-trained LSTM.

**Use here:** test whether the original navigation LSTM's recurrent dynamics depend on its previous-action input. Our earlier action-context experiment changed the approach action but still released the state under a stationary-turn input. The prospective sensitivity sweep instead holds the exact initial state, ordinary spatial code, saved goal and black visual embedding fixed while changing only the previous-action entries across all six actions. It reuses the same full-state and future-activity criteria. Translational messages with frozen location are labeled clamped-input interventions; any candidate then needs physically consistent and behavioral follow-up. The paper motivates this test without implying that the LSTM contains the fly circuit.

## Initial task proposal and the frozen implementation

The following was the initial proposal inferred from these papers. The implemented [pilot protocol](rl_visual_memory_pilot_protocol.json) subsequently fixed five training stages in advance, with maximum blackout gaps of 4, 8, 15, 30 and 60 updates at 15 updates per second. Its schedule was not selected adaptively from observed performance.

Keep the original square arena, uniformly distributed goals, six actions, grid maps, goal snapshots, rewards, 256-unit LSTM and learned actor. Continue training matched original checkpoints with short, randomly timed **visual blackouts**, gradually increasing the blackout duration. Visible intervals refresh orientation; the following blind interval makes remembered orientation useful for selecting actions. Start with gaps of approximately 0.25–0.5 seconds and increase only when performance supports it. Training sequences must span the relevant memory interval. Use the existing reinforcement objective throughout.

The natural circular candidate is body heading or movement direction. These must be tested separately: this repository's predictive displacement follows the last actual translation, which can differ from body heading during backward motion or strafing and stays stale during rotation without translation. Decode both variables on held-out positions and trajectories. The goal remains a two-dimensional variable; the experiment does not assume that its whole representation is a ring.

Before expensive manifold analysis, demonstrate that the trained agent benefits from recurrent history during blackout. Compare preserved state with a state reset at blackout onset, plus the same reset when vision remains available. This separates useful memory from generic disruption of the controller. Ordinary grid histories can themselves supply movement information, so include the matched grid-only agents and history-based controls.

## Connected mechanistic test

1. **Find the persistent circular state.** Initialize full `(h,c)` states from actual trajectories covering all candidate angles. Give them the same meaningful native input and follow their evolution. Test phase coverage, closed geometry, intrinsic dimension and slower tangential than transverse dynamics. A sweep of different sustained inputs is a separate input-response measurement.
2. **Identify what the predictive input changes.** From identical states and physical trajectories, compare native PGC input, ordinary-grid replacement, rotated predictive displacement, and restored native input. Measure phase drift and transverse recovery separately. An informative cue can anchor phase without changing autonomous transverse attraction.
3. **Connect the state to choices.** Perturb along the inferred circular state and across it, with matched-size random controls. Test the predicted change in action preference and navigation. A phase rotation should produce the corresponding behavioral error; restoring the correct predictive cue should correct both state and behavior.
4. **Localize the recurrent mechanism.** Measure complete-state Jacobians, leading eigenvectors and finite-time perturbation gains at the same states and inputs. Relate the changed modes to the candidate-update and output-gate pathways already implicated by the original-agent interventions. Test targeted pathway removal and restoration in the actual actor.
5. **Confirm across agents.** Choose criteria and analysis settings on pilot seeds, freeze them, then evaluate held-out matched seeds. Retain unsuccessful ring candidates and negative interventions in the report. Use trained agents as the replication unit.

Report three distinct quantities: **return toward the ring, accuracy of the remembered angle, and navigation performance**. Their linked response to removal, rotation and restoration is the desired mechanistic chain.

## Practical analysis checks

- Use all 512 hidden-plus-cell coordinates. PCA can display a result after full-state analysis; it does not define the mechanism.
- Hold the input condition fixed when comparing local dynamics. All-zero inputs are meaningful only when they correspond to the intended condition.
- Measure recovery relative to the unperturbed trajectory at the same time, and report both raw-state and scale-normalized sensitivity.
- Check additional slow dimensions before calling every direction orthogonal to a fitted curve “transverse.”
- A finite set of fixed points can lie on a useful slow ring. Report both task-duration behavior and eventual convergence.
- Use enough phase samples to distinguish a closed curve from separated clusters and to avoid nearest-sample distance artifacts.

Biological motivation for the supplied predictive code remains [Ouchi & Fujisawa, *Predictive grid coding in the medial entorhinal cortex*, Science (2024)](https://doi.org/10.1126/science.ado4166). The new contribution sought here is the downstream causal chain inside the navigation LSTM.
