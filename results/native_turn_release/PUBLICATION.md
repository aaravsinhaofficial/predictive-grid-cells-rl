# Verified data and reproduction

[GitHub release: native turn memory](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/native-turn-memory-2026-11-06)

All **29 published assets** match their recorded sizes and SHA256 digests. [Publication verification](publication_verification.json) binds the remote files to the successful upload records and frozen archive indexes. Across the three archive sets, all **71 saved-report outputs** reproduced with only documented metadata differences.

The release contains separate archives with explicit dependencies. Keep each archive set in its own directory. The download names below avoid collisions; the verification tools expect `index.json` and `manifest.json` inside each archive directory.

| Archive | Downloaded index / manifest | Files to verify | Scientific scope |
| --- | --- | --- | --- |
| Primary native-turn study | `index.json`, `manifest.json` | `native-turn-outputs-part*-of014.tar.gz` | Original pair, four continuations, 128-heading geometry and saved-policy decompositions. Source `aa7ee742bd8aeb2e664471226c286fa0ed1495c2`. |
| Counterbalanced supplement | Copy `counterbalanced-index.json` to `index.json`; copy `counterbalanced-manifest.json` to `manifest.json`. | `counterbalanced-incomplete-equivalence-part001-of001.tar.gz` | Complete descriptive readout; scientific equivalence remains rejected, with 12 input and six cell-state failures. Requires the exact primary archive. |
| Six-action observer supplement | Copy `observer-index.json` to `index.json`; copy `observer-manifest.json` to `manifest.json`. | `native-input-observer-*.tar.gz` | Actual-action queries of the original pair; all geometry screens retained, zero conditional perturbation/held-out tests triggered. Source `4f8f7dd7ee4139ed448325f217caf8d93f5a371e`; exact shared primary dependencies are bound in its manifest. |

Use each accompanying README and reproduction proof for the exact extraction, source checkout and saved-data report commands. Do not combine manifests or substitute the main branch for a pinned source revision. The primary guide is [REPRODUCE.md](REPRODUCE.md); [primary proof](reproduction_proof.json), [counterbalanced proof](../native_turn_counterbalanced_release/reproduction_proof.json), and [observer proof](../native_input_observability_release/reproduction_proof.json) record independent checks and permitted metadata-only differences.

The later 1,024-heading study's code, frozen protocols, reports, independent reviews and execution records are on the repository's main branch. Its 33 GB of dense raw outputs are retained locally and are outside these release archives. The observer archive includes only the four dense terminal records required by its frozen protocol, not those trajectories.

These analyses use one original training seed per population and four continuations. Contexts, headings and disturbances are repeated observations within those models. No qualifying ring was found. [Plain-language findings and figure](../native_turn_ring_followup/README.md).
