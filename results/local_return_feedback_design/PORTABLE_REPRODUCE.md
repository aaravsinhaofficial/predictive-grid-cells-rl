# Reproduce the portable feedback-acquisition study

Use source commit **`f1e5e1725c4449ee6c8bd39cac022b9e94b0139f`** and the exact 49-file input archive:

`portable/cloud_package/feedback_acquisition_portable_inputs.tar.gz`  
**96,310,897 bytes** · SHA256 **`797981f1e1792588ee0facf3d82125dc508e3d607d02f6aaf7f0a857514c77ae`**

The archive contains every source and input needed for the frozen training/evaluation, including the four incoming policies, original checkpoints, unchanged test bank and its creation record. [Exact members](portable/cloud_package/manifest.json) · [Protocol](../../docs/research/local_return_feedback_portable_protocol.json) · [Independent compatibility review](../../docs/research/local_return_feedback_portable_review.json).

This revision authenticates the saved test inputs directly. The original implementation and [pre-learning cloud failure](PORTABILITY.md) remain preserved. Training computations and settings are unchanged.

In an isolated checkout, verify the archive digest and extract it with `tarfile.extractall(..., filter='data')`. Then validate all inputs without evaluating a policy:

```bash
OPENBLAS_NUM_THREADS=2 python - <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path.cwd().resolve()
sys.path[:0] = [str(root), str(root / 'scripts')]
import local_return_feedback_portable_run as runner
path = root / 'docs/research/local_return_feedback_portable_protocol.json'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '4d5afa22d93e9f3a238768049dddd522e2db158fa3dc067ba0b605702f4a1f19'
frozen = json.loads(path.read_text())
frozen.pop('frozen_utc')
assert runner.protocol() == frozen
for name, expected in {**frozen['source_sha256'], **frozen['input_assets']}.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('Exact saved bank, source and input checks passed.')
PY
```

The local environment is recorded in [requirements-ring-analysis.txt](../../requirements-ring-analysis.txt); the cloud uses the image and actual runtime recorded in the [launch manifest](../../docs/research/local_return_feedback_portable_launch.json). Each worker saves a pure input-validation record before model initialization. The sealed inputs are byte-identical; rerunning learning on another numerical stack may produce different trajectories.

Run an arm in a fresh destination:

```bash
OPENBLAS_NUM_THREADS=2 python scripts/local_return_feedback_portable_run.py \
  --condition gc100 --actor-mode retained_actor --device cuda \
  --out /absolute/path/to/new-experiment --max-seconds 12600
```

The four combinations are `gc100`/`pgc33` × `retained_actor`/`reset_actor`. Each receives the same prescribed baseline, 1,024 updates × 32 trials, and fixed-final evaluation. No intermediate score selects an endpoint. The runner retains policy, optimizer, RNG, counters and lossless evaluation inputs. A timeout remains incomplete.

Audit an existing collected folder without running a new policy:

```bash
OPENBLAS_NUM_THREADS=2 python scripts/audit_local_return_feedback.py \
  --protocol docs/research/local_return_feedback_portable_protocol.json \
  --folder /absolute/path/to/experiment \
  --condition gc100 --actor-mode retained_actor \
  --out /absolute/path/to/audit.json --require-complete
```

Omit `--require-complete` when inspecting a partial run. A passing partial audit is explicitly incomplete. The frozen auditor verifies the saved physical trajectory, inputs, sampled actions, first-FREE action/value readout and learning/evaluation provenance. Its full scope and numerical tolerances are declared in the source.

The once-only CPU physical references are reused from the original bank: uniform actions reached 0/256 goals and the privileged planner 256/256. Their original raw tensors, source and audit identity remain unchanged. They supply no training targets. [Reference audit](reference_audit.json).

## Reproduce the saved results

Download all four assets from the [public release](https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/feedback-acquisition-pilot-2026-11-06). The [isolated verification](portable/cloud_audit/isolated_extraction_validation.json) reproduces all audits and figures while preserving all 817 archived files.

The output tooling is pinned at **`8175f0fd5df07db72ccc4a80d8321938d207bbbe`**. Check out that commit in a fresh directory and extract these four assets at its root:

1. `feedback_acquisition_portable_inputs.tar.gz` — the exact inputs described above.
2. `feedback-acquisition-grid-outputs-v1.tar.gz` — both Grid arms and their earlier failed-start evidence.
3. `feedback-acquisition-pgc-outputs-v1.tar.gz` — both PGC arms and their earlier failed-start evidence.
4. `feedback-acquisition-common-outputs-v1.tar.gz` — protocols, audits, physical references, figures and diagnostic results.

The [archive record](../../docs/research/feedback_acquisition_outputs_archive.json) gives every asset's size and SHA256. Each output asset embeds a separate `*_outputs_manifest.json` under `results/local_return_feedback_acquisition/portable/`, listing the exact size and SHA256 of every member. Verify those before running analysis. The input member list is `portable/cloud_package/manifest.json` → `package_manifest`. Extract with `tarfile.extractall(..., filter='data')` and retain the archived files unchanged.

Create a new `reproduced/` folder. Run the audit command above for all four worker folders listed in `docs/research/local_return_feedback_portable_launch.json`, writing the results to `reproduced/audits/{condition}_{actor_mode}.json`. Then run:

```bash
OPENBLAS_NUM_THREADS=2 python scripts/plot_local_return_feedback.py \
  --audit-root reproduced/audits --out reproduced/report
OPENBLAS_NUM_THREADS=2 python scripts/compare_feedback_training_streams.py \
  --out "$PWD/reproduced/streams"
OPENBLAS_NUM_THREADS=2 python scripts/check_feedback_baseline_actor_isolation.py \
  --out reproduced/actor_isolation.json
OPENBLAS_NUM_THREADS=2 python scripts/check_feedback_completed_finals.py \
  --out reproduced/final_review.json
```

These commands read the saved experiments; they perform no new navigation or learning. The strict audits replay saved physics and inspect recorded inputs/readouts. The stream comparison authenticates the private generator states saved in the final checkpoints. Historical timestamps and absolute audit paths remain recorded honestly; newly generated audit paths and dates will differ. Figure data, counts and unchanged raw artifacts must agree. PDF creation dates may differ.

The guide embedded in the frozen common archive used a relative stream-output path. That command must use the absolute path shown above: the original helper records outputs relative to its inferred checkout root. This documentation correction leaves the numerical source and all archived evidence unchanged.

The separate [two-action bound](portable/restricted_action_bound/README.md) reads pretraining snapshots only. Its original numerical source and the serialization-only wrapper are both retained, together with all 512 row calculations and an independent review.

[Study overview](../local_return_feedback_acquisition/README.md) · [Fixed endpoint report](../local_return_feedback_acquisition/portable/report/README.md).
