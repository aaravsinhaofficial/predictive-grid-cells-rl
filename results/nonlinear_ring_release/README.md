# Standalone release reproduction

[Download the public release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/nonlinear-ring-search-2026-11-06). All eight uploaded assets match their local SHA256 digests; [publication verification](publication.json) records the server-reported hashes.

**All five archive parts verified, and four reports regenerated successfully in a fresh detached checkout of source commit `4c90a93ae9d29bd5de9960484d5fea8d0d165f5d`.**

The check independently extracted and verified all **844 payload files** by path, type, size and SHA256, then read every extracted file back. All **486 included source files** matched the committed source. The original workspace's archived inputs remained unchanged.

| Regenerated artifact | Result |
| --- | --- |
| Four numerical JSON reports | Exact byte matches |
| Four PNG figures | Exact byte matches |
| Native-census README | Exact byte match |
| Two SVG figures | Identical after removing date metadata and consistently renaming generated element IDs and references |
| Four PDF figures | Identical after normalizing creation/modification dates only |

The SVG and PDF differences are recorded explicitly; they are not described as matching original file hashes. Coordinates, drawing commands and remaining content matched after those narrow normalizations.

These commands ran in the isolated checkout using Python 3.12.7, NumPy 1.26.4 and Matplotlib 3.9.2:

```sh
python scripts/plot_nonlinear_ring_search_summary.py
python scripts/report_native_movement_cycles.py
python scripts/lstm_module_phase_torus_report.py
python scripts/lstm_module_all_phase_control_report.py
```

All eleven report data inputs were present in the release manifest and matched their archived hashes. Every non-report input in the extracted checkout also remained unchanged. This check regenerated steering, native-turn, two-phase, and local eight-phase reports from existing results; it ran no policies, training, cloud jobs, or scientific experiments.

[Detailed reproduction proof](reproduction_proof.json) records the exact commands, extraction procedure, source and archive integrity, environment, per-artifact hashes, and metadata comparisons. The proof and this note were created after archive generation and are separate release supplements; the frozen archive inputs were not edited.

[Full reproduction guide](REPRODUCE.md)
