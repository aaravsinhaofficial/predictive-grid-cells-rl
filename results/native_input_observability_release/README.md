# Fixed six-action observer supplement

This separate archive preserves the complete observer study: **6,656 named cases from two original networks**, including every raw profile, physical record, execution/report log and independent audit. No case qualified for conditional perturbation or held-out testing, and zero candidates were nominated. These counts describe this fixed screen; they do not mean the RNNs contain no angular information.

The single `native-input-observer-part001-of001.tar.gz` is **274,577,872 bytes** and contains 5,762 files (757,631,940 uncompressed bytes). Source is pinned to `4f8f7dd7ee4139ed448325f217caf8d93f5a371e`. Only the four required original 1,024-angle summary/execution JSONs are included as provenance dependencies; the large dense-array dataset is excluded.

The supplement requires the **primary 128-angle native-turn archives**, which supply the original checkpoints, saved neural states and native input banks:

- Primary source: `aa7ee742bd8aeb2e664471226c286fa0ed1495c2`.
- Primary manifest SHA256: `f8d847cbead359ba1dbee8bc05b1d3e68494054b14bdb31ef0dcb25c6e8608ea`.
- Primary index SHA256: `40f3944db68493f514fcfd8ee03d6cbd44fa95dce671f1c37cb606cd55639a2d`.

**Download filenames differ from the internal verifier names.** The release exposes `observer-index.json` and `observer-manifest.json` to avoid colliding with the primary archive files. Put them in a separate observer archive directory and copy/rename them to **`index.json` and `manifest.json`** there. Keep the part filename unchanged. Do not overwrite the primary index/manifest.

```sh
mkdir -p observer-archives
cp observer-index.json observer-archives/index.json
cp observer-manifest.json observer-archives/manifest.json
cp native-input-observer-part001-of001.tar.gz observer-archives/
```

Use a fresh detached checkout of the observer source commit. Populate it with the already verified primary archive data; after primary extraction, restore tracked files from the observer commit in that disposable checkout so its source takes precedence. Raw untracked data remain. Verify this supplement before extracting its regular files:

```sh
export PYTHONPATH="$PWD:$PWD/scripts"
python - <<'PYCODE'
from pathlib import Path
import package_nonlinear_ring_outputs as package
package.MANIFEST_NAME = 'results/native_input_observability_release/manifest.json'
print(package.verify_release(Path('/path/to/observer-archives')))
PYCODE
```

Preserve `results/native_input_observability/report` outside the disposable checkout, including its original independent audits, then regenerate only the saved-data report:

```sh
python scripts/report_native_input_observability.py
```

Independent reproduction completed in a fresh detached checkout. Every supplement member passed streamed and readback hash checks, and its source matched the pinned commit. The report reproduced all seven compared artifacts: summary JSON, README and both PNGs byte-identically; both PDFs differed only in explicit creation/modification dates, and validation JSON only in `created_utc`. Zero conditional tests and zero candidates were preserved.

The reproduction guard blocked access to both the original workspace and the previous checkout, and protected shared inputs from writes. Every read project input belonged to the verified primary/supplement inventory or the newly generated report. Shared data and all archived report files were restored unchanged. No policy, environment, recurrent transition, optimization or new scientific query was run. The exact member checks, commands, access guard and comparisons are in [reproduction_proof.json](reproduction_proof.json). Existing primary and counterbalanced archives remain unchanged.
