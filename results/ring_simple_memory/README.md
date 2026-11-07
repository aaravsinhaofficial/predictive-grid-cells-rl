# Give the agent a reason to remember

The agent plays the same navigation task as before. The one change is simple: after finding the goal, its view and current grid code sometimes disappear for **2–4 seconds**. Remembering where it was can help it navigate between views.

![Example input schedule; design illustration, not a result](task.png)

| Training task | Grid only | PGC + grid |
|---|---|---|
| Inputs always available | 4 seeds × 1B frames | 4 seeds × 1B frames |
| Brief sensory blackouts | 4 seeds × 1B frames | 4 seeds × 1B frames |

Each seed starts with identical network weights across the four conditions. The full network learns from navigation reward. The maps, actions, movement noise, reward, and stored goal code stay the same.

After a goal pickup, the agent sees normally for one second. It then alternates brief views with blackouts. Blackouts grow gradually over the first 200M frames. Before the first goal discovery, everything stays visible.

**During a blackout:** the image and all 512 current grid-code entries are zero. The LSTM still receives the stored goal, its previous action, and its previous reward. No extra movement or direction information is supplied.

We will ask three simple questions:

1. **Does it learn to remember?** Erase the LSTM's hidden and cell states and measure the navigation cost, with the same erasure in a visible control.
2. **Do predictive cells help that memory?** Replace current predictive activity with ordinary grid activity, then compare retention, recovery, and navigation.
3. **What form does the memory take?** Examine activity and perturb the complete recurrent state. A ring needs a continuous family of memories that survives without cues and recovers from disturbances.

The run target is 16 models, each trained for exactly 1B environment frames. The cumulative cloud ceiling is **$600**, including previous recorded research spending. The GPU test passed speed and exact restart checks. Every worker has an enforced deletion deadline.

**Training started November 7, 2026. Eight models are running; eight more are queued.** Each run takes roughly one day, so the two waves should take about two days if throughput holds. The first four passed numerical checks and started from exactly identical network weights. See the [launch audit](../ring_simple_memory_supervision/first_four_audit.json).

There are no completed 1B results yet. The experiment tests whether memory pressure produces useful recurrent structure and whether predictive cues help preserve or repair it. A ring is a hypothesis to test.

The analysis methods are now fixed, with **68 private control tests passing**. We will measure PCA activity variance, remembered direction, recovery after a state disturbance, and navigation after a single correct or misleading predictive cue. These checks have been tested on private fixtures; trained-model findings come after the runs finish.

The follow-up plan waits for all sixteen verified 1B checkpoints, then evaluates eight Grid/PGC pairs. Each evaluation worker has a one-hour deletion deadline. The conservative combined budget is **$599.20**, including earlier recorded research spending. Its launch and artifact checks passed another **29 private tests**.

[Training protocol](../../docs/research/simple_memory_training_protocol.json) · [Analysis protocol](../../docs/research/simple_memory_analysis_protocol.json) · [Training cloud plan](../../docs/research/simple_memory_cloud_plan.json) · [Evaluation cloud plan](../../docs/research/simple_memory_followup_plan.json)
