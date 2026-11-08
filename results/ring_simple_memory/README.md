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

**Training started November 7, 2026. Eight models were launched; eight more are queued.** The first four passed numerical checks and started from exactly identical network weights. See the [launch audit](../ring_simple_memory_supervision/first_four_audit.json).

**One run needs a technical diagnosis.** PGC + grid, blackout, seed 1 stopped at 348M frames when two calculations of the same LSTM outputs differed by about 0.00005. Its complete restart state at 345.6M frames is preserved. We are reproducing the exact failed update before continuing. See the [failure record](../ring_simple_memory_design/replay_failure_20261107.json).

Cloud monitoring now checks whether each running job can finish before its original deletion deadline. The [operational amendment](../../docs/research/operational_amendments/20261107_monitor_recovery/amendment.json) preserves the earlier monitoring files and documents the change. Scientific training and analysis methods remain frozen.

There are no completed 1B results yet. The experiment tests whether memory pressure produces useful recurrent structure and whether predictive cues help preserve or repair it. A ring is a hypothesis to test.

The analysis methods are now fixed, with **68 private control tests passing**. We will measure PCA activity variance, remembered direction, recovery after a state disturbance, and navigation after a single correct or misleading predictive cue. These checks have been tested on private fixtures; trained-model findings come after the runs finish.

The follow-up plan waits for all sixteen verified 1B checkpoints, then evaluates eight Grid/PGC pairs. Each evaluation worker has a one-hour deletion deadline. Every diagnostic or continuation must fit the live **$600 cumulative ceiling**, including earlier research spending. The orchestration and recovery code passed **71 private tests**.

[Training protocol](../../docs/research/simple_memory_training_protocol.json) · [Analysis protocol](../../docs/research/simple_memory_analysis_protocol.json) · [Training cloud plan](../../docs/research/simple_memory_cloud_plan.json) · [Evaluation cloud plan](../../docs/research/simple_memory_followup_plan.json)
