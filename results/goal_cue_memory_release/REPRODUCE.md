# Reproduce the original-agent goal-memory experiments

This archive contains the continuous-goal qualification and the separate exploratory memory diagnostic. Both use the original seed-1 actors, CNNs and LSTMs, with the same archived fresh critic. No learning or cloud run occurred. The failed paired training prerequisite remains part of the record.

The diagnostic was registered after the qualification scores were known and before its own outcomes. Its protocol is `docs/research/goal_cue_memory_original_diagnostic_protocol.json`, SHA256 `2d9386fad85f9774d80ced37d7d6220d6bb75c682879684511a0da4d4bb04125`. All 11 conditions, both models and every sealed-bank trial are retained. The numerical source was committed at `fbb34e60d96ccd9bc1ab4412a34ff43060bfc3ec` before execution. Use the later **reproduction commit recorded in `index.json` and `manifest.json`**, which also contains the reporting and packaging tools.

## Download and verify

Download every asset from the `original-goal-memory-diagnostic-2026-11-06` GitHub release into a directory called `downloaded`. The release includes `index.json`, `manifest.json` and every numbered archive part. A missing part is an incomplete dataset.

```sh
gh release download original-goal-memory-diagnostic-2026-11-06 \
  --repo aaravsinhaofficial/predictive-grid-cells-rl --dir downloaded
git clone https://github.com/aaravsinhaofficial/predictive-grid-cells-rl.git goal-memory-reproduction
cd goal-memory-reproduction
python - <<'PY'
import json, subprocess
from pathlib import Path
record = json.loads(Path('../downloaded/index.json').read_text())
subprocess.run(['git', 'checkout', '--detach', record['source_commit']], check=True)
PY
```

The following verifies the archive hashes and every member before extraction. Existing checkout files must already match; they are not overwritten. All paths are relative to this fresh checkout.

```sh
python - <<'PY'
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile
from scripts.package_goal_cue_memory_outputs import sha, verify_archive, MANIFEST_NAME
root = Path.cwd().resolve()
assets = root.parent / 'downloaded'
body = (assets / 'manifest.json').read_bytes()
manifest = json.loads(body)
index = json.loads((assets / 'index.json').read_text())
assert hashlib.sha256(body).hexdigest() == index['manifest_sha256']
assert index['source_commit'] == manifest['source_commit']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == manifest['source_commit']
rows = {row['path']: row for row in manifest['files']}
seen = set()
for part in index['parts']:
    path = assets / part['path']
    assert path.stat().st_size == part['bytes'] and sha(path) == part['sha256']
    names = part['members']
    assert len(names) == len(set(names)) and not seen.intersection(names)
    verify_archive(path, [rows[name] for name in names], body)
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            target = root / member.name
            assert target.resolve().is_relative_to(root)
            check = root
            for component in Path(member.name).parts:
                check /= component
                assert not check.is_symlink()
            expected = index['manifest_sha256'] if member.name == MANIFEST_NAME else rows[member.name]['sha256']
            if target.exists():
                assert target.is_file() and sha(target) == expected
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as destination:
                    shutil.copyfileobj(source, destination)
                assert sha(target) == expected
    seen.update(names)
assert seen == set(rows)
for name, row in rows.items():
    assert (root/name).stat().st_size == row['bytes'] and sha(root/name) == row['sha256']
print('Every archived file verified:', len(rows))
PY
```

Original execution used Python 3.12.7, PyTorch 2.10.0 on CPU, and two Torch threads per process. NumPy 1.26.4, SciPy 1.13.1 and Matplotlib 3.9.2 are pinned in `requirements-ring-analysis.txt`. The saved protocol and evaluation records include the Torch runtime. Authentication uses the **saved bank bytes**; do not recreate either bank from its seed.

```sh
python -m pip install -r requirements-ring-analysis.txt
```

## Replay the independent checks

These commands read the saved trials. They reconstruct the stationary neural sequence and first movement transition, verify physical trajectories and input codes, and check all interventions, action sampling, denominators and unchanged model snapshots. They do not run training or generate replacement evaluation trials. The audit records state the exact numerical scope, including the distinction from recomputing the entire moving-phase neural history.

```sh
mkdir -p reproduced/audits
for study_condition in gc100 pgc33; do
  OPENBLAS_NUM_THREADS=2 python scripts/audit_goal_cue_memory.py \
    --phase qualification --condition "$study_condition" \
    --folder "results/goal_cue_memory/qualification/$study_condition" \
    --out "reproduced/audits/qualification_$study_condition.json" --require-complete
  OPENBLAS_NUM_THREADS=2 python scripts/audit_goal_cue_memory_original_diagnostic.py \
    --condition "$study_condition" \
    --folder "results/goal_cue_memory/original_diagnostic/$study_condition" \
    --out "reproduced/audits/original_diagnostic_$study_condition.json" --require-complete
done
```

All four audits must finish with `passed: true`, `complete: true` and no failures or pending records. Passing data-integrity checks does not change a failed scientific screen.

The additional deterministic consistency check compares wrong-goal cues with the corresponding genuine donor-state substitutions. It authenticates the saved tensor hashes; it adds no trial or fitted model.

```sh
python scripts/check_goal_cue_donor_equivalence.py --out reproduced/donor_equivalence.json
```

## Rebuild the simple figures and tables

The reporting modules permit redirecting only their output directory. This leaves every archived result and original audit untouched.

```sh
OPENBLAS_NUM_THREADS=2 python - <<'PY'
from pathlib import Path
import scripts.report_goal_cue_qualification as qualification
import scripts.report_goal_original_diagnostic as memory
qualification.OUT = Path('reproduced/qualification_report')
qualification.main()
memory.OUT = Path('reproduced/original_diagnostic_report')
memory.main()
PY
```

Compare the regenerated CSV, JSON, Markdown and PNG files with the corresponding archived reports. PDF creation timestamps may differ. Fresh independent audit records also contain their new timestamps and checkout paths; compare their scientific checks and values separately from those fields. The release's extraction-validation record reports the checks actually completed by the authors.

A fresh policy rollout is optional and separate from saved-data reproduction. The fixed diagnostic CLI can evaluate **both** populations into new empty directories using the same frozen protocol. Preserve any platform-dependent numerical differences rather than replacing the archived runs. The qualification failure continues to prohibit the training phase of the stopped proposal.
