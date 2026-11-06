This release preserves the completed exploratory ring search in the original 1B-frame navigation LSTMs, using Grid and PGC seeds 1–2.

The clearest positive result is a causal influence of current predictive input on action selection through the candidate-update and output pathways. The circular-manifold tests separate rapid activity adjustment from persistent cell-memory errors. A published trained memory RNN provides a successful ring-detection control. Unsuccessful candidates and numerical corrections are retained.

Included studies cover unstable equilibria, individual grid-module phases, both module phases together, local phase directions across all modules, and matched disturbances during native navigation. The bounded-hidden-state follow-up preserves all 29 selected movement segments. These nested probes are not independent trained-seed replications.

Source commit: `4c90a93ae9d29bd5de9960484d5fea8d0d165f5d`.

- [Simple figures and findings](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/blob/4c90a93ae9d29bd5de9960484d5fea8d0d165f5d/results/ring_nonlinear_design/README.md)
- [Archive verification and reproduction](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/blob/4c90a93ae9d29bd5de9960484d5fea8d0d165f5d/results/nonlinear_ring_release/REPRODUCE.md)

Download every archive part plus `manifest.json` and `index.json`. Verify with `python scripts/package_nonlinear_ring_outputs.py --verify-release --out /path/to/downloads` before extracting into a clean checkout of the source commit. The manifest records every member's size and SHA256; the index records each archive's hash. Original checkpoints, native traces, exact source snapshots, all outcomes, and the MIT-licensed published numeric model are included.

All 36 focused scientific/report tests passed. The packager also passed seven synthetic integrity checks and verifies every archived byte. This batch used local computation only. The separate new policy-distillation continuation is excluded.

An independent clean-checkout check verified all 844 payload files and 486 committed source files, then regenerated four numerical reports and four PNG figures exactly. SVG/PDF differences were limited to documented metadata and generated IDs. The additional `reproduction_proof.json` asset records the full check.
