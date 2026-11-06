# Reproduce the frozen feedback-acquisition inputs

This separate four-arm control asks whether the native agent can learn fast local return with continuous sensory feedback. The [protocol](../../docs/research/local_return_feedback_training_protocol.json) was frozen before any study policy evaluation or learning. Neural outcomes are pending; this guide currently reproduces the inputs and physical references.

Use source commit **`016ec248d066476428ae356a5bac75d7db556f06`** in an isolated checkout. The exact input archive is `cloud_package/feedback_acquisition_inputs.tar.gz`, **96,303,489 bytes**, SHA256 **`1057cf8b55d15a5424136861942fafdeebee972448ffef9f0a35163daa917c05`**. Its 44 files contain all frozen sources, the original checkpoints/configurations, archived heads, four composed incoming policies, sealed evaluation bank and protocol. [Member manifest](cloud_package/manifest.json).

The local analysis environment is recorded in [requirements-ring-analysis.txt](../../requirements-ring-analysis.txt). Cloud training uses the separately recorded Google image and each worker's actual runtime metadata. A different numerical backend need not reproduce sampled learning trajectories bit for bit.

After checking the archive digest, extract it into that isolated checkout with Python's safe data filter:

```python
import tarfile
with tarfile.open('/absolute/path/to/feedback_acquisition_inputs.tar.gz') as archive:
    archive.extractall('.', filter='data')
```

Validate the exact inputs without running a policy:

```bash
OPENBLAS_NUM_THREADS=2 python - <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path.cwd().resolve()
sys.path[:0] = [str(root), str(root / 'scripts')]
import local_return_feedback_run as runner
path = root / 'docs/research/local_return_feedback_training_protocol.json'
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ad969e782ec2943de2e59b2d72629002dea537f7c9c0fcf2ad3905ce051038c7'
frozen = json.loads(path.read_text())
frozen.pop('frozen_utc')
assert runner.protocol() == frozen
for name, expected in {**frozen['source_sha256'], **frozen['input_assets']}.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('All frozen sources, assets and regenerated bank agree.')
PY
```

The isolated package preflight already performed this validation successfully, with no imports from the original checkout, policy execution, environment step or optimization. [Record](isolated_package_preflight.json) · [Final packaged-input validation](cloud_package/manifest.json).

To rerun the physical references, choose fresh output paths:

```bash
OPENBLAS_NUM_THREADS=2 python scripts/local_return_feedback_evaluate.py \
  --references --device cpu --out /tmp/feedback-references
OPENBLAS_NUM_THREADS=2 python scripts/audit_local_return_feedback.py \
  --references --folder /tmp/feedback-references \
  --out /tmp/feedback-reference-audit.json --require-complete
```

The recorded CPU results are **0/256** successes for uniform actions and **256/256** for the privileged physical planner. All 512 trials pass the independent audit. The planner supplies no training data. Reference summaries contain timestamps and elapsed times, so their whole-file hashes change when rerun; compare the physical trial results and audit checks.

To reproduce an individual learning arm, use the same frozen inputs and a fresh destination:

```bash
OPENBLAS_NUM_THREADS=2 python scripts/local_return_feedback_run.py \
  --condition gc100 --actor-mode retained_actor --device cuda \
  --out /absolute/path/to/new-experiment --max-seconds 12600
```

The four prescribed combinations are `gc100`/`pgc33` × `retained_actor`/`reset_actor`. Each run performs its zero-update evaluation, all 1,024 updates and fixed-final evaluation. The saved raw inputs are lossless. The runner refuses to overwrite an existing experiment or change the frozen design. A timeout is incomplete; this study has no automatic retry or extension. [Exact cloud routing and resource bounds](../../docs/research/local_return_feedback_launch.json).

Run the independent auditor on each collected folder using the matching condition and actor mode, adding `--require-complete` for a complete endpoint. Its scope includes saved physical motion, inputs, sampled actions, first-FREE action/value readout, checkpoint provenance, optimizer counters and evaluation preservation. It does not independently replay every CNN/LSTM state. The full raw tensors retain the data needed for that additional computation. [Auditor scope](../../scripts/audit_local_return_feedback.py).
