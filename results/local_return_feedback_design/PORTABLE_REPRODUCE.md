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

Study outputs and the final public archive will be added after collection and independent validation. Current status and any available results appear in [the acquisition overview](../local_return_feedback_acquisition/README.md).
