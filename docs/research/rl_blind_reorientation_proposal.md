# Proposed next task: remember a turn, then navigate

**Design only, 6 November 2026 UTC. No experiment has been launched or added to the current training arms.** The 25M pilot and the fixed noiseless/noise continuations keep their existing protocols and outcomes.

The smallest useful change is a **short blind reorientation interval** inside the same navigation task: turn the agent using labelled native actions, temporarily stop refreshing its current spatial observations, then let its existing actor navigate before those observations return. First test whether successful navigation actually depends on recurrent memory. Only then analyze a ring.

## Why the present blackout task permits a shortcut

After choosing an action with vision, an agent can keep doing it during darkness. Its previous-action input, physical heading and inertia, and continuously refreshed spatial code all carry information forward. Resetting `h` and `c` does not remove those channels. In the completed 25M pilot, every-blind-step resets preserve most of the score improvement.

Adding turns alone leaves another shortcut: successive ordinary position codes reveal displacement once translation resumes. A predictive code held during a stationary turn carries **old travel direction**, not the updated body heading. It cannot be described as a fresh heading anchor.

## Behavioral necessity comes first

Before a training launch, implement and validate a small **paired-history audit** in the original simulator. It needs no learned decoder or new controller:

- Start two branches at the same interior position, with the same goal, room, initial heading, native recurrent context and saved-goal code. Explicitly brake translation.
- Execute different earlier turn sequences, then an identical final action sequence. Current spatial observations are held at the same ordinary-grid values and RGB is black. Zero physical velocities at the same specified boundary, so residual inertia cannot distinguish the branches.
- At the first free decision, verify byte-identical complete RNN inputs and the same last three input/action observations. The true body headings differ. Evaluate the best common action directly from those two actual headings. Add an exactly antipodal control by varying the initial visible heading as needed; discrete turn durations alone need not produce exactly 180°. Training would sample varied turns rather than just two cases.
- From zero recurrent state, the two branches must produce the same action distribution. In the exactly antipodal control, with zero initial velocity, the same translation action moves toward the goal in one branch and away in the other. Average signed movement toward the goal is zero across that pair; turn actions have zero immediate translation. For other heading separations, calculate the best memoryless common-action score using the actual six-action physics, then ask whether the intact recurrent policy exceeds it.

This establishes an information requirement for **successful direction-sensitive behavior**: the current inputs alone cannot select the appropriate action in both branches. The trained agent must then demonstrate that it uses the earlier history.

The first trained-model endpoint is therefore behavioral: intact recurrent state must produce positive goal-directed movement across these ambiguous pairs and outperform state reset at the end of the turn sequence. The effect must also appear in goal-reaching performance during the modified task, with normal-vision reset controls and held-out turn durations/positions. Report actual first-choice motion and full episode score separately. A model that waits for vision, searches blindly, or shows no recurrent benefit fails this gate; further ring analysis would not rescue that task design.

## Minimal training variation

Keep the square arena, uniformly distributed goals, maps, original rewards and episode clock, six-action actor, CNN, 256-unit LSTM and external saved-goal code. Use the original reinforcement objective. No angle target, angular readout loss, ring architecture or navigation controller is added.

A concrete first schedule to review before freezing is:

| Interval | Physical actions | Inputs entering the same LSTM |
|---|---|---|
| Visible refresh: 15 updates / 1 s | Existing actor | Native RGB and current code; saved goal unchanged |
| Random blind turns: two blocks of 5–10 updates each | Independently chosen left/right native actions; brake at entry | Black RGB; current code held at its **ordinary-grid value at interval onset**; actual executed previous action/reward |
| Actual forward probe: 4 updates / 0.267 s | Brake, then four native forward actions | Held code for the first three observations; **one native-code glimpse at the fourth endpoint** |
| Common blind suffix: 6 updates / 0.4 s | Brake, then left/right/left/right/left/right, identically across cue branches | Black RGB; ordinary current code at the probe endpoint held throughout; actual executed previous action/reward |
| Free blind navigation: 30 updates / 2 s | Existing actor chooses every action | Same held ordinary endpoint code, black RGB and native saved goal |
| Refresh | Existing actor | Resume native vision and current code |

This event lasts 65–75 updates including visible refresh, with 3.33–4 seconds from blackout onset through blind navigation. The separate diagnostic uses a minimal three-update common suffix to verify identical-input necessity; the proposed training suffix is six updates. The full event must lie inside one 100-update BPTT segment. This is a proposed schedule, not an authorized or launched training run.

The frozen implementation would specify event frequency, timing and braking exactly. Use existing post-discovery refresh boundaries and geometry-only eligibility, with a fixed cap on interventions per episode. Never select events using neural geometry or model success. Reorientation steps count toward the original episode time. An actual goal pickup or episode ending invokes normal goal/reset logic; its event cannot silently continue with an obsolete saved code.

At the start of the forced turn and immediately before its common suffix, explicitly set `vf`, `vs` and `vr` to zero. This is a declared environmental perturbation, not an assertion that a turn action naturally stops movement. The subsequent turn integration uses the original native physics. In the deterministic audit, verify stationary position exactly. In noisy training, retain and log actual physical drift; no fresh position observation is supplied during the interval.

The action input must name the **executed** action. Forced steps have neither an actor policy-gradient nor an entropy-loss term because the actor did not choose them; the value loss and ordinary return calculation still use their genuine transitions. Keep gradients through the recurrent updates during forced steps. Save this mask and actual action sequence for exact BPTT replay. The actor resumes full control during navigation. Intervention randomness needs its own generator. Record total frames and actor-controlled frames, since imposed steps reduce the number of policy decisions. Schedule each complete visual-refresh/turn/cue/suffix/blind-choice chain inside one existing 100-update BPTT unroll; simply making it shorter than 100 updates would not prevent it from crossing a truncation boundary.

Holding position observations through the free-navigation interval also creates a short path-integration demand. A learned heading ring would therefore be a computation within a larger navigation state, assessed at matched position/goal contexts; the entire state need not be one ring.

## A physically valid brief PGC repair cue

The table includes a prespecified cue opportunity. Validate the turn-only necessity diagnostic first, then its complete probe/glimpse version before any launch. If trained behavior later passes the recurrent-necessity gate, use the matched cue interventions below. This is a new task design, not a reinterpretation of the old stationary-turn pulse.

1. During the blind interval, execute a short known translation using one native action. Brake first so prior rotation/sideways velocity cannot mislabel the motion. Four forward updates from rest take about 0.267 s and move about 4.3 cm with the original deterministic physics. The actual displacement must exceed the map’s movement threshold; record measured `mDir` and body heading.
2. At its endpoint, present one update of the native current code. In the matched ordinary-cue branch, replace only the 169 predictive slots by their ordinary values **at that same actual endpoint**. The other 343 current slots, saved-goal code, RGB, reward and executed-action input are identical. The correct cue is computed from actual movement, not an artificial heading substitution. Rotated ±60° cues are explicit counterfactual controls.
3. Immediately withhold the direction cue again. Hold the endpoint’s ordinary current code; explicitly stop motion and use a common forced turn suffix before the actor’s free choice. No policy-selected action during cue delivery can store the answer in the previous-action channel. Include varied labelled turns after the cue so the agent must retain and update its information.
4. Compare correct, ordinary and wrong cues following the same pre-cue recurrent disturbance. Initially use a single state reset just before the pulse to test recovery of useful information. If a circular state is subsequently detected, use the existing phase and radial disturbances to test the specific mechanism. A reset after cue withdrawal tests whether the later benefit is carried in recurrent state.

Ordinary endpoint observations can also reveal travel direction through comparison with earlier position. That is a valid competing computation and must remain in the matched ordinary-cue control. The specific PGC claim requires an additional, persistent benefit from its predictive slots after withdrawal, plus appropriately signed effects of rotated cues.

## Shortcuts and their checks

| Possible shortcut | Required check |
|---|---|
| Repeat the visually selected action | Forced turns overwrite the earlier action; evaluate matched final inputs and last-action suffixes |
| Use body inertia as memory | Explicit common braking boundaries; audit position and all velocities |
| Infer heading from refreshed positions | Hold current spatial observations until after the measured blind choice; track every refresh |
| Use a continuously supplied PGC as a compass | Ordinary current input throughout the autonomous interval; native cue is brief and withdrawn |
| Store cue information in the previous action | Cue-period actions and final suffix imposed identically; the actor cannot choose an encoding action |
| Exploit walls or goal contact | Interior, geometry-selected audit; first-choice measure precedes collision/reward; retain all native navigation events in the score |
| Wait for the next visual refresh | Measure progress and goals during blind intervals, as well as whole-episode score; require a recurrent advantage |
| Memorize two turn categories | Varied signed durations, initial headings and contexts; hold out durations and combinations |
| Infer heading from position, goal snapshot or event timing | Match position, complete saved goal including pickup-direction content, previous reward and event timing across audit branches; vary contexts during training and held-out tests |
| Confuse body heading with last movement direction | Measure both; cue translation and executed action establish their actual relation |

A successful task could still use a counter, Cartesian memory or another recurrent computation. The existing full-state/observable geometry, radial-versus-phase recovery, cue withdrawal and actor/navigation interventions determine which computation was learned.

## Literature basis and scope

[Cueva and colleagues](https://arxiv.org/abs/1912.10189) trained unconstrained RNNs to integrate angular velocity and found directional representations and distinct functional populations. The relevant ingredient here is an actual need to update direction from movement history; their supervised angle objective is not copied.

[Low, Giocomo and Williams](https://elifesciences.org/articles/86943) combined position integration with transient context information, then analyzed fixed points and local modes. This motivates separating remembered variables and using complete-state dynamics rather than projecting a circle onto a convenient plane.

[Ajabi and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060160/) manipulated visual landmarks while recording head-direction populations. They measured reorientation, subsequent drift and effects persisting after cue removal, with an additional population-gain dimension. This motivates correct/rotated brief cues, withdrawal tests and allowing an angular computation to coexist with other neural state.

[Ságodi and colleagues](https://arxiv.org/html/2408.00109v2) trained with state noise and assessed slow attracting manifolds in deterministic dynamics. Their work motivates testing robust recovery after functional memory has been demonstrated. It does not specify this navigation protocol.

**Scope of a positive result:** a newly adapted version of the same navigation agent learns a useful angular-memory mechanism that native predictive input stabilizes. The original checkpoints’ mechanism remains determined by the already completed original-agent tests. This proposal cannot retroactively turn their persistent, weakly repairing traces into an attracting ring.
