Both fixed local-return models completed 4,096 updates and 131,072 trials. Grid and PGC + grid each finish with **0/256 successful returns in all 12 evaluation conditions**. Every earlier stage is retained. The conditional ring analysis stops at its prespecified behavioral gate because neither model learned the required blind return.

The saved-action analysis identifies a steering-acquisition bottleneck in this new task: stage-3 paths average about 33 cm, while final distance to the goal increases by about 4.5 cm. An independent calculation reconciles every one of 38,912 saved records and all six counterfactual first steps per record.

- [Training results and simple figures](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/blob/main/results/ring_local_goal_return/report/README.md)
- [First-action analysis](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/blob/main/results/local_return_action_use/RESULTS.md)
- [Three-asset reproduction guide](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/blob/main/results/ring_local_goal_return/REPRODUCE.md)

These are new task adaptations of one original seed pair, with identical reset action/value heads at initialization. All native CNN/LSTM weights then trained from goal reward. These results do not replace the original navigation study.

Use source commit `4f73dd99f1ce94773a7495c2da93aef01672913c` with the first three assets. The fourth contains optional derived tables, figures and the independent checker. An isolated checkout reproduced both complete audits, matched private streams, primary report and action diagnostic. Scientific tables and PNGs match byte-for-byte; timestamp and checkout-path differences are recorded separately.

| Asset | Bytes | SHA256 |
|---|---:|---|
| `local-return-behavior-banks-v1.tar.gz` | 10,954,659 | `d7c42a9294020b389034cd2d01f0441227b8aae0cd0b285e48c5c8b7ff7f3123` |
| `local-return-training-inputs-v1.tar.gz` | 81,090,261 | `9f0c3c5c0bbd62b5a16e2f9bf2c78080f8185885036ae4b2c067b5ae63401b4f` |
| `local-return-final-outputs-v1.tar.gz` | 536,866,504 | `fcaea6c6da11252fa4263dfc5f8c43fc34b383f8d4ed7484413508aa4cc1e058` |
| `local-return-action-use-v1.tar.gz` | 22,022,807 | `879ef05df799053d54857fd17368b0df1be128ad904dbebd52cb6dcd6d768206` |

The final archive preserves all 22 checkpoints, optimizer and RNG states, raw evaluations at all four stages, independent audits and provider cleanup evidence. Both workers and boot disks are verified absent. The pair's conservative compute cost bound is $6.1044 within the original cumulative $500 authorization.
