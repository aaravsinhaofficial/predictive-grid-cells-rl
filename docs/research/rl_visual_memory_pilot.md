# Reward-trained visual memory in the original navigation agent

This pilot starts from the original 1-billion-frame Grid and PGC LSTMs, seed 1. It preserves the original square arena, uniformly sampled hidden goals, +10 rewards, 30 cm reward disks, respawns, six actions, 512 spatial maps, saved goal-code semantics, network architecture, and A2C optimizer settings. Original checkpoints and previous experiment sources stay unchanged.

The training rationale builds on [Low, Giocomo & Williams (2023)](https://elifesciences.org/articles/86943), who used a sequence-length curriculum to learn circular memory, and [Ságodi et al. (2024)](https://arxiv.org/html/2408.00109v2), who identified slow rings in trained LSTMs. Their memory tasks used supervised objectives. Here, the only objective is the original navigation reward. The [literature note](endogenous_ring_literature.md) separates these precedents from the new experiments and sets out the subsequent dynamical tests.

**Only visual availability changes.** Before the first reward, vision is continuously available. After each reward, the agent sees 15 normal images (one second), followed by a randomly sampled interval of black images. Vision then returns for 15 updates and the cycle repeats. Every subsequent reward renews the visible period. Episode end clears the schedule. The saved goal code remains continuously available after discovery; it is never flashed or blanked in this study.

The curriculum has five fixed stages of five million learning frames each. The maximum blackout lengths are 4, 8, 15, 30, and 60 decision updates (about 0.27, 0.53, 1, 2, and 4 seconds). Each new gap is uniformly sampled from ceil(maximum/2) through the maximum. An already-started gap finishes normally at a curriculum boundary. The schedule uses an RNG separate from physics and action sampling.

Both models receive the full fixed curriculum regardless of intermediate performance. Each stage rounds to 391 complete original 100-step ×32-environment updates: 5,004,800 learning frames, or 25,024,000 total. A fresh RMSProp optimizer first runs 200 updates at zero learning rate, adding 2,560,000 exposure frames without changing policy weights. All weights remain trainable. There is no angular supervision, auxiliary representation loss, prescribed ring wiring, map decoder, or external controller.

At baseline and after every stage, evaluate the same six branches on 64 paired episodes. All evaluations use the final 30–60-update blackout distribution, so the curve measures progress on one fixed task.

| Branch | Image availability | Additional intervention |
|---|---|---|
| Blind | Scheduled blackouts | None |
| Blind + reset | Same blackouts | Clear both h and c immediately before each blackout |
| Seen | Continuous images | None |
| Seen + reset | Continuous images | Clear h and c at the corresponding schedule transitions |
| Blind + ordinary current code | Scheduled blackouts | During scheduled blackout windows, replace only the 169 current predictive values by their ordinary values |
| Seen + ordinary current code | Continuous images | Apply the same current-code replacement during the schedule windows |

The ordinary goal channels, saved goal code, physical state, previous action, and previous reward are not changed by these interventions. Each branch follows the actual learned policy after the intervention. Layouts, motor/image noise, action uniforms, and schedule-duration uniforms are coupled across branches. Later trajectories and reward times can diverge. All interventions begin after discovery, so first-discovery events must match exactly. For the Grid model, the current-code replacements must be bitwise no-ops.

The pilot reports all scores, failure-retaining return counts, actual blackout exposure, and reset counts. The memory contrast is the reset cost during darkness minus the corresponding reset cost with images available. The prediction contrast compares the current-code replacement cost in darkness with its cost under continuous vision. Positive contrasts motivate mechanistic analysis; they are not themselves evidence of a neural ring. A blind/seen return-rate ratio of 0.70 is a practical health reference, and five score points is a practical reference for a nontrivial additional reset cost. These references do not select checkpoints or terminate stages.

The circular variable of interest is heading or movement direction during real navigation. These must be analyzed separately: predictive maps encode the last sufficiently large movement direction, which can be stale during a pure turn. Many original policies use a sideways or backward translation action, so movement direction can differ systematically from body heading. The pilot records this distinction. Full-state ring geometry, transverse versus phase perturbations, and prediction removal/restoration are separate analyses of the trained copies.

This is one preselected checkpoint pair. The two original agents have different 1B-frame training histories; the continuation does not make their weights or prior experiences identical. Episode trials are not independent trained-network replicates. The visual schedule is an explicit change in the training distribution.

Focused tests check the exact image schedule, reward/reset precedence, current-gap carry across stage changes, independent coupled RNG, preservation of source images and environment dynamics, unchanged goal snapshots, rollout/replay/bootstrap equality, state-reset timing, and Grid substitution identity. A short evaluation using both actual original checkpoints additionally verified paired discovery events and unchanged policy hashes.

Two isolated T4 workers have a two-hour provider-enforced DELETE deadline and automatic boot-disk deletion. Their total conservative reservation is $5.625. The earlier $44.1913 cumulative envelope remains counted against the original $500 authorization. Expected runtime is about 35 minutes per worker using the prior measured 15,000 environment frames/s. No additional worker is authorized by this pilot protocol.
