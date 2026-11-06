# Original-agent goal-memory pilot

This exploratory pilot asks whether the repository's trained navigation RNNs can retain a briefly supplied goal message. It starts from the original **1B-frame Grid and PGC LSTMs, seed 1**, and keeps free movement through the same square room, the same renderer, six actions, 512-cell maps, actual-hit goal snapshots, +10 rewards, and random respawns.

There are exactly two task changes. Goal centers are sampled uniformly on a **0.65m-radius circle** inside the room. After a reward, the actual saved goal code is supplied for **four subsequent policy updates** (about 0.267 seconds), then its input slots become zero. Current spatial code remains available. The goal code is still zero before discovery and refreshed from the actual pickup position and arrival direction after each reward. No true goal coordinate or angle reaches the policy.

Both copied networks receive **5M additional environment frames** of the original reward-only A2C update, with their original optimizer settings. Fresh RMSProp statistics are initialized by the original 200-update, zero-learning-rate warmup. This warmup supplies 2.56M additional exposure frames per network but does not alter policy weights or count as learning frames. Full 100-step updates round the requested 5M learning frames up to 5,004,800. All network weights remain trainable. Original checkpoint files remain unchanged. There is no heading target, auxiliary loss, designed controller, or ring constraint.

Before and after fine-tuning, each network is evaluated on 64 paired episodes per condition. The five branches share goal layouts, spawn draws, motor/image noise, and action-sampling uniforms:

| Evaluation branch | Goal message | Internal-state intervention |
|---|---|---|
| Flash | Four updates after each pickup | None |
| Flash + reset | Same four-update message | Clear both h and c immediately before the first blank update |
| Continuous | Saved goal remains available | None |
| Continuous + reset | Saved goal remains available | Same reset timing as the flash branch |
| No goal message | Always zero | None |

The reset leaves position, motion, previous action, and previous reward intact. It removes all recurrent information, so it is not a selective lesion of a proven goal subspace. The continuous+reset branch helps distinguish a loss of retained goal information from general controller disruption. A flash advantage over no-goal, together with a larger reset cost when the cue disappears, would support useful internal retention. Failed returns remain in the primary descriptive return count; conditional path efficiency is reported separately.

The goal-location circle is a simple one-dimensional family of targets, not a ring-shaped maze. Agents still move freely in two dimensions. The actual message also contains pickup and arrival information, and the recurrent network still controls movement and vision, so a one-dimensional target family does not guarantee a one-dimensional hidden-state manifold. Any later neural-ring claim requires separate dynamical evidence. Inward predictive information still means movement **into the reward disk**, not toward the center of the circle of possible goals.

This is one preselected checkpoint pair and a feasibility experiment, not a new multi-seed confirmation. The original agents have different training histories and initial performance. The pilot can show whether this transfer works and whether the resulting policy uses retained goal information; it cannot estimate a population-wide PGC advantage by treating episodes as independent trained networks.

[Frozen protocol and hashes](rl_goal_memory_pilot_protocol.json) · [implementation](../../scripts/rl_goal_memory.py) · [focused timing/environment tests](../../tests/test_rl_goal_memory.py) · [compute plan](rl_map_compute_plan.md).

Seventeen focused and cloud-guard tests passed before launch, and an independent review found no timing or information-leakage blocker. A short forward test with the actual 1B PGC checkpoint also passed without changing its weights. The cloud pilot uses one T4 worker with a two-hour provider-enforced deletion deadline and a $2.8125 conservative reservation. The original $500 authorization is cumulative; the previous study's $43.7364 envelope remains counted.
