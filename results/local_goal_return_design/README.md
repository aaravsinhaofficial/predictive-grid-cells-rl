# Local goal-return: sealed behavioral inputs

**The inputs and decision rules were frozen before evaluating the new readouts or training a model.** Each bank contains 16 physical contexts × 16 headings = 256 trials, with 128 opposite-heading pairs. Generation used only private random draws; no environment or policy was run.

| Bank | Seed | Goal-center distance | Delay updates | Role |
|---|---:|---:|---:|---|
| Near baseline | 391801 | 0.36–0.40 m | 2–4 | Check that initial rewards are discoverable |
| Final hard | 391901 | 0.58–0.62 m | 30–60 | Test learned memory at the fixed endpoint |

The near-baseline gate requires at least **13/256 successes in each population**, zero forced-phase rewards and passing physical checks. The final descriptive gate requires at least **167/256 intact successes** and a post-cue erasure cost of at least **39 successes** in one population before expensive dynamics. Both models and every trial remain reported. The special 50% common-information bound requires all 128 physical pairs to pass the independent certificate; failed pairs remain in the data and contribute the conservative bound 1 instead of 0.5.

Both populations receive the same physical contexts and body-frame motor/action tapes. The native action and value heads use a common dedicated initialization seed (291801 for source seed 1); the original CNN/LSTM tensors are retained exactly. The actual baseline action probabilities are measured, rather than assumed to be uniform. Clean evaluation has zero added recurrent noise; state-noise draws are nevertheless archived for an explicitly separate matched-noise analysis.

The full plans and 160-update tapes are in ignored binary files `banks/near_baseline.pt` and `banks/final_hard.pt`. The protocol records their file hashes, every tensor's shape/dtype/hash, the pure generator class and stage hashes, Python source provenance and the Torch version. The training module was still undergoing review; its whole-file hash is historical provenance, while the sealed bank bytes and generator definitions are fixed. Evaluators must consume these banks directly or verify exact tensor identity after regeneration.

Protocol SHA256: `4c7ba90dd79c8c0b5f8ac13ba89e4a7d6ed5bf0287556982ade07f911fc16e48`.

[Complete frozen protocol](behavior_protocol.json) · [Scientific design review and primary literature](../../docs/research/local_goal_return_study_review.md) · [Pure bank-generation source](../../scripts/freeze_local_goal_return_banks.py) · [Existing information-bound proof](../rl_blind_reorientation_review/local_goal_preflight_v2/README.md)

The generator refuses to overwrite these published banks. Reproduction in a fresh checkout can compare tensor hashes independently of the new freeze timestamp. Training source, final checkpoint validators and mechanistic analysis remain subject to their own separate review and freeze.
