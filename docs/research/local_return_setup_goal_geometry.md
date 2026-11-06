# The setup goal snapshot differs from a natural pickup

This methods note changes no running task, bank or criterion. It uses source inspection and arithmetic, without a rollout.

[Setup](../../scripts/local_goal_return_training.py) places the agent **2 cm before the goal center**, already inside the 30 cm reward disk, then takes one noiseless native forward step from rest. Native acceleration moves it 0.546 cm: `0.35 × (1 − exp(−(4/60)/0.25)) × (4/60)`. The actual pickup is therefore **1.454 cm before the center**. Its reward creates the saved code but is excluded from learning.

Ordinary goal slots encode that nearly central pickup. The 169 predictive slots evaluate their maps 20 cm ahead along the pickup direction: **18.546 cm beyond the center**, still inside the reward disk. This follows the unchanged [population-code definition](../../code/codes.py); it is a mixed snapshot, not one common marker for all 512 cells.

One pickup angle and snapshot are shared across every heading history within a context. The angle is sampled independently of those headings, so the saved goal cannot reveal which heading variant is present.

[Natural first pickups](../../results/lstm_goal_cue_mechanism_1B/README.md) were usually near the boundary: the predictive shift moved their marker inward, closer to the center. Here the ordinary marker is already almost central, removing that specific geometric advantage. The local tasks therefore study adaptation under this controlled snapshot distribution. Keep this geometry explicit when relating their population differences to the original inward-cue mechanism.
