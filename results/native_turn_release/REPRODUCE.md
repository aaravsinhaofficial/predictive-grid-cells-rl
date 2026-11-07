# Reproduce the native-turn study

This package preserves one original 1B-frame checkpoint per population, every fixed preflight and memory-control case, both clean/noisy continuations per population, all six 128-angle geometry endpoints, and the exploratory saved-probability decomposition and prefix comparison. Each continuation uses the frozen 180-update acquisition. Scientific positives are not required for inclusion. The separate higher-resolution geometry and counterbalanced-readout studies are outside this package.

The final source commit, every member's SHA256 and byte count, and the normalized multipart inventory are recorded in the release manifest and index. Raw tensors, checkpoints and dense JSON results belong in the release archives rather than ordinary Git blobs. Only explicit study trees and frozen upstream assets are included; the original seed-1 checkpoints/configs and 32-slot census input traces are included through the protocol inventory.

Use a fresh detached checkout of the manifest's full source commit and the Python environment recorded in `docs/research/ring_local_environment.json` (dependencies in `requirements-ring-analysis.txt`). Run from that checkout's root, with the root explicitly on `PYTHONPATH`: the local `code` package must take precedence over Python's standard-library `code` module. Download all parts together with `index.json` and `manifest.json`, then verify before extracting into the checkout:

```sh
cd /path/to/fresh-detached-checkout
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
python scripts/package_native_turn_outputs.py --verify-release --out /path/to/archive-directory
```

The verifier checks every archive/member hash and size, rejects unsafe member paths and symlinks, and checks part completeness. Extract only the verified normalized regular files. Keep the original release copies unchanged. Report regeneration reads archived inputs and creates no policies, training rollouts, or optimization; static checkpoint tensors may be loaded for integrity and actor-output calculations.

To regenerate the saved-data reports in a disposable checkout, first move the existing report directories outside that checkout. The dense geometry reporter refuses to overwrite a nonempty output directory. Preserve the complete archived report directories, including all independent validation JSONs, artifact manifests, and the geometry presentation addendum. These audits describe the archived artifacts and must not be copied onto regenerated artifacts as though they audited a new run.

```sh
export NATIVE_TURN_SAVED_REPORTS="$(mktemp -d /tmp/native-turn-reports.XXXXXX)"
python - <<'PY'
from pathlib import Path
import os
saved = Path(os.environ['NATIVE_TURN_SAVED_REPORTS'])
for relative in (
    'results/native_turn_teacher_preflight/report',
    'results/native_turn_original_memory/report',
    'results/native_turn_distillation/report',
    'results/native_turn_geometry/report_original',
    'results/native_turn_geometry/report_all',
):
    source = Path(relative)
    target = saved / relative
    assert source.is_dir() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
print(saved)
PY
```

Run only these saved-data reporters:

```sh
python scripts/report_native_turn_teacher_preflight.py
python scripts/report_native_turn_original_memory.py
python scripts/report_native_turn_distillation.py
python scripts/report_native_turn_geometry.py --endpoints original
python scripts/report_native_turn_geometry_failures.py --report-directory report_original
python scripts/report_native_turn_geometry.py --endpoints original clean noise
python scripts/report_native_turn_geometry_failures.py --report-directory report_all
```

The archived `report_all/README.md` includes a documented presentation-only correction made after the frozen reporter ran. Reapply its exact replacement, checking both recorded SHA256 values. Do not modify either reporting source:

```sh
python - <<'PY'
from pathlib import Path
import hashlib, json, os
relative = Path('results/native_turn_geometry/report_all')
saved = Path(os.environ['NATIVE_TURN_SAVED_REPORTS']) / relative
change = json.loads((saved / 'presentation_addendum.json').read_text())
target = relative / 'README.md'
before = target.read_bytes()
assert change['complete'] and change['file'] == 'README.md'
assert hashlib.sha256(before).hexdigest() == change['previous_sha256']
old = change['previous_text'].encode('utf-8')
new = change['updated_text'].encode('utf-8')
assert before.count(old) == 1
after = before.replace(old, new)
assert hashlib.sha256(after).hexdigest() == change['updated_sha256']
assert after == (saved / 'README.md').read_bytes()
target.write_bytes(after)
print(change['updated_sha256'])
PY
```

This restores the precise wording that geometry screening completed and no broad disturbance test was triggered. The archived addendum, audits and artifact manifest remain in the preserved directory. Any manifest generated before this replacement describes the pre-addendum README; a new reproduction proof must bind the final README separately.

The saved-probability decomposition requires the archived, independently audited training report. In a separate clean checkout, move only `results/native_turn_imitation_decomposition` outside the checkout and run `python scripts/decompose_native_turn_imitation.py`. Its exact information identity separates changes in average-policy mismatch from changes in within-input-group history gain. This calculation is descriptive; it executes no state-erasure intervention.

The separate prefix comparison requires the archived main decomposition and its audit. Use another clean checkout, move only `results/native_turn_imitation_prefix` outside it, and run `python scripts/decompose_native_turn_prefix.py`. This compares saved policies with their average within the actual LEFT/RIGHT preparation prefix and their average across the complete identical-input group. It does not change any recurrent state. Initial presentation attempts and their exact source snapshots remain in the two decomposition design trees.

The geometry report recomputes saved-array gates and finite-perturbation arithmetic; the complete six-endpoint report took about 17 minutes on the recorded host. It does not execute the LSTM or the environment. Compare every scientific JSON value, CSV, PNG and final README exactly. Only explicitly recorded creation timestamps and PDF `/CreationDate` and `/ModDate` metadata may differ. PDF comparison must normalize those date fields only and require every remaining byte to match. A report's figure-hash fields may consequently differ for PDFs; verify each regenerated hash against its actual PDF and the date-normalized equality before accounting for that derived difference. Similarly, `failure_patterns.json` binds `validation.json`, whose only permitted non-scientific difference is `created_utc`. Never ignore source, checkpoint, raw-input, protocol or numerical differences.

The planned standalone proof will independently extract every member into the detached checkout, reject traversal/links/duplicates, stream-check and reread all hashes, and verify all source bytes against the pinned commit. Each report will run through a Python file-access audit guard installed before project imports. The guard resolves paths, rejects reads or directory enumeration inside the original workspace, logs checkout reads, and first proves rejection with an intentional blocked read. Its wrapper explicitly puts the detached root and `scripts` directory first on `sys.path`, even when invoked with isolated Python (`-I`). Every successful project-data read must belong to the extracted manifest. The guard covers the reviewed Python reporters' file access; it is not an operating-system sandbox. The proof will preserve exact commands, read sets, outputs and all permitted metadata differences, and verify archived inputs and untouched checkout files remain unchanged. This is a report-reproduction check, not a new scientific experiment.

The frozen experiment protocols and acquisition specification document the exact original experiments. Repeating acquisition is a new experiment with its own costs and outputs, not a report-reproduction step. Calendar duration, monotonic acquisition/evaluation time and external subprocess caps are recorded separately. The operational scheduler has an explicitly documented v1 peer-abort metadata limitation; its exact source and successful individual process records are retained.

Package planning is safe while results are incomplete:

```sh
python scripts/package_native_turn_outputs.py --plan
python scripts/package_native_turn_outputs.py --self-test
```

A build is allowed only after every prespecified terminal summary, exit record, report and independent audit matches the final source commit. The guard accepts negative and inconclusive findings. Archives are not created by the default command.
