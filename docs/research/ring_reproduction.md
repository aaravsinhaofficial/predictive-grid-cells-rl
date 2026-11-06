# Reproducing the predictive-ring study

Start with the [main report](../../mechanisms/predictive-ring-attractor.md), [methods](../../mechanisms/predictive-ring-methods.md), and [experiment inventory](ring_experiment_inventory.md). The designed ring, unconstrained heading RNNs, original navigation LSTMs, and published path integrators are separate experiments.

## Exact trained models

The [research release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/ring-mechanism-2026-11-05) provides `predictive-ring-checkpoints.tar.gz`. It contains all 104 final unconstrained RNNs (48 primary, 24 raw-input, and 32 independent geometry models), plus eight files containing the three trained interfaces for each designed-ring seed. Pilots, intermediate checkpoints, and third-party models are excluded. No final seed is selected by its result.

The [archive manifest](heading_checkpoint_archive.json) records every checkpoint's byte hash, source hashes, evaluation record, and relative path. Download the archive from the release, verify its SHA-256 against that manifest, and extract it at the repository root. This restores the paths expected by the recorded commands. The archive does not require access to the researcher's cloud account.

For example, with the GitHub CLI installed:

```bash
gh release download ring-mechanism-2026-11-05 \
  --repo aaravsinhaofficial/predictive-grid-cells-rl \
  --pattern predictive-ring-checkpoints.tar.gz
shasum -a 256 predictive-ring-checkpoints.tar.gz
tar -xzf predictive-ring-checkpoints.tar.gz
```

Only extract the archive after comparing the displayed hash to the published manifest. Training checkpoints include optimizer and configuration records and should be treated as trusted project artifacts; public third-party state dictionaries are loaded with `weights_only=True` by their separate evaluator.

## Environment and focused checks

Use a separate Python environment with [requirements-ring-analysis.txt](../../requirements-ring-analysis.txt). The recorded local numerical analysis used Python 3.12 and PyTorch 2.10. Cloud training used the pinned image and PyTorch 2.9.1 recorded in each checkpoint and the [cloud ledger](ring_cloud_budget.json). Exact retraining bytes can vary with hardware; the released checkpoint hashes identify the analyzed models.

```bash
python -m pytest -q tests/test_heading_confirmation_statistics.py \
  tests/test_heading_cue_spectra_audit.py tests/test_heading_mechanistic_audit.py \
  tests/test_heading_navigation_audit.py tests/test_predictive_fraction_information.py \
  tests/test_predictive_phase_decoder.py tests/test_public_internal_pgc.py \
  tests/test_ring_cloud.py tests/test_ring_dynamics.py tests/test_sensorimotor_ring.py \
  tests/test_designed_predictive_ring.py
```

The final focused run passed 85 tests. Independently propagated geometry derivatives are additionally checked against automatic differentiation within every geometry evaluation. Report audits compare inclusion counts, source hashes, checkpoint identities, and numerical claims.

## Re-run one model before a cohort

From the repository root, the primary checkpoint can be evaluated with:

```bash
python scripts/evaluate_heading_confirmation.py \
  results/ring_cloud_artifacts/ring-20261105-confirm-01/predictive_s2001.pt \
  --out reproduction/primary_predictive_s2001
python scripts/evaluate_heading_memory.py \
  results/ring_cloud_artifacts/ring-20261105-confirm-01/predictive_s2001.pt \
  --out reproduction/memory_predictive_s2001
```

The [methods appendix](../../mechanisms/predictive-ring-methods.md) gives training and direction-subspace commands. Full primary aggregation requires all 48 models; the [geometry protocol](heading_geometry_confirmation_protocol.md) requires all 32 fresh models. Do not pool those cohorts or substitute a visually appealing example for the complete seed-level analysis.

The inexpensive designed-ring experiment can be retrained independently with `scripts/train_designed_predictive_ring.py`; its [report](../../mechanisms/designed-predictive-ring.md) specifies the complete command, fixed protocol, and what is learned versus imposed. Its exact [mathematical derivation](predictive_ring_derivation.md) is distinct from the empirical phase model fitted to unconstrained RNNs.

## Published internal-PGC comparison and original agents

Run `python scripts/download_internal_pgc_assets.py` to retrieve the separately pinned published model assets. The [source manifest](public_internal_pgc_sources.json) verifies the commit, Git blob hashes, sizes, and SHA-256 values. The [ten-network report](../../mechanisms/internal-pgc-recurrent-outputs.md) records its null primary result and reproduction procedure. These external models are not redistributed in our checkpoint archive.

The original 1B-frame navigation LSTM analyses use the project's existing checkpoints under `models/`, not the new heading-model archive. Their [gate report](../../mechanisms/predictive-memory-gates.md) and [cloud job record](goal_gate_behavior_job.md) specify the input files and intervention. This release supplies the new code and derived results, not a second copy of the earlier large navigation checkpoints.

## Figures and PDF

Compact per-seed numerical records, protocols, plot scripts, PNGs, and vector figures are committed. Large intermediate state arrays are omitted from Git and can be regenerated from the released models. Figures use trained seeds as replicates; the many simulated time points are not independent samples.

For the PDF renderer, additionally install `reportlab==5.1.0`, `Markdown==3.4.1`, `beautifulsoup4==4.12.3`, and Pillow. Then run:

```bash
python scripts/render_ring_report.py --methods --derivation
```

The editable Markdown and figure files are the canonical sources. The PDF is a computational research report, not an accepted journal publication.
