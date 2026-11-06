# Reproduce the fixed local-return training results

**Both prescribed endpoints are complete and independently audited.** Each model scores 0/256 on every final condition. Every associated VM and boot disk is verified absent. The final-output archive preserves all four stages and the failed endpoints. Archive creation and isolated extraction are recorded separately in `docs/research/local_return_outputs_archive.json` and the release verification records; this guide supplies the reproduction procedure.

This is adaptation of the original seed-1 Grid/PGC spatial, visual and recurrent weights with **explicitly reset, identical new action/value heads**. Training has four fixed stages and 131,072 trials per model. Every stage and failure remains part of the evidence; the last prescribed checkpoint is the final endpoint.

## Three required assets and source checkout

Use tooling commit `4f73dd99f1ce94773a7495c2da93aef01672913c`, also recorded in `docs/research/local_return_outputs_archive.json`. The final asset records its exact source commit and includes the runtime/audit/report source files with hashes.

```bash
git clone https://github.com/aaravsinhaofficial/predictive-grid-cells-rl.git
cd predictive-grid-cells-rl
git checkout 4f73dd99f1ce94773a7495c2da93aef01672913c
```

Download all three assets from the study release and verify the published SHA256/byte counts before extracting into this fresh checkout:

| Asset | Role |
|---|---|
| `local-return-behavior-banks-v1.tar.gz` | Exact physical contexts and complete random tapes for both sealed banks |
| `local-return-training-inputs-v1.tar.gz` | Original parent models/configs, exact new-head initial policies, full baseline evidence, protocols and validation |
| `local-return-final-outputs-v1.tar.gz` | Complete final training outputs, all four stage evaluations, saved optimizer/RNG checkpoints, raw evidence, audits, report, explicit routing and cleanup proof |

The first two assets remain required for unchanged source, parent, baseline and bank validation. The final asset does not duplicate their large data. Their existing hashes are:

- Banks: `d7c42a9294020b389034cd2d01f0441227b8aae0cd0b285e48c5c8b7ff7f3123` (10,954,659 bytes).
- Training inputs: `9f0c3c5c0bbd62b5a16e2f9bf2c78080f8185885036ae4b2c067b5ae63401b4f` (81,090,261 bytes).
- Final outputs: use the archive SHA256 and byte count in the completed archive record accompanying the release. Its own digest is kept outside the archive to avoid a circular hash dependency.

```bash
tar -xzf /path/to/downloads/local-return-behavior-banks-v1.tar.gz
tar -xzf /path/to/downloads/local-return-training-inputs-v1.tar.gz
tar -xzf /path/to/downloads/local-return-final-outputs-v1.tar.gz
```

Do this in a fresh checkout so extraction cannot replace your own revised results. Paths remain repository-relative, including the explicitly routed worker/experiment folders. The routing file may retain historical launch status; the separately hashed owner cleanup verification supplies the final resource-absence evidence. No cloud credentials or unrelated worker trees are included.

Install the recorded local audit environment with `python -m pip install -r requirements-ring-analysis.txt`, preferably in a dedicated virtual environment. The original CPU baselines used Torch 2.10.0. Each routed worker's actual CUDA environment is recorded in `results/local_goal_return_design/cloud_audit/{worker}_environment.json` and included as operational evidence; the frozen training protocol records the prescribed setup. Read-only audits load tensors on CPU and do not resume learning.

## Verify every extracted output

Run the input checks in [the separate input guide](../local_goal_return_design/REPRODUCE.md) first. Then verify the full final-output manifest, including each checkpoint, raw evaluation, prior-attempt file and source:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

manifest = json.loads(Path('results/ring_local_goal_return/final_outputs_manifest.json').read_text())
for row in manifest['files']:
    p = Path(row['path'])
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
print('Every extracted output and source hash matches the final inventory.')
PY
```

## Re-run the complete audits without overwriting published outcomes

These commands use saved tensors, actions, motor tapes, stage boundaries and counters. They do not execute new navigation trials or optimizer updates. The fresh temporary folder receives the new audit records.

```bash
LOCAL_RETURN_AUDIT=$(mktemp -d)
export LOCAL_RETURN_AUDIT
OPENBLAS_NUM_THREADS=2 python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
routes = json.loads(Path('docs/research/local_goal_return_training_launch.json').read_text())['workers']
assert len(routes) == 2 and {r['condition'] for r in routes} == {'gc100', 'pgc33'}
for row in routes:
    folder = Path(row['local_artifact_root']) / row['artifact_subdir']
    subprocess.run([sys.executable, 'scripts/audit_local_goal_return_training.py',
        '--condition', row['condition'], '--folder', str(folder), '--require-complete',
        '--out', str(Path(os.environ['LOCAL_RETURN_AUDIT']) / (row['condition'] + '.json'))], check=True)
print('Both fixed-final audits passed; published audit files were preserved.')
PY
```

Successful integrity audits mean that both finals have exactly 4,096 optimizer updates, 131,072 trials and four completed stage evaluations, with consistent policy/optimizer/RNG state and raw results. They can pass when the behavioral memory gate fails. The primary gate and every negative result stay visible.

The final asset also requires a passing comparison of the two models' saved private trial streams at all four stages and the final checkpoint. To reproduce that comparison into a fresh directory:

```bash
LOCAL_RETURN_STREAMS=$(mktemp -d)
python scripts/compare_local_goal_return_private_streams.py --out "$LOCAL_RETURN_STREAMS"
```

This reads saved seeds, counters and every named generator state; it does not draw new random numbers or run models. Global worker RNG is preserved separately in the original checkpoints. Earlier partial comparisons and raw snapshots remain archived as history; the stage-1 comparison includes its exact historical producer source and hash record.

To rebuild the figures and table in another fresh output directory:

```bash
LOCAL_RETURN_REPORT=$(mktemp -d)
OPENBLAS_NUM_THREADS=2 python scripts/plot_local_goal_return_training.py \
  --out "$LOCAL_RETURN_REPORT"
```

This plotter verifies the archived complete audits and all input hashes. It retains the reset-head baseline, all four stages and all 12 fixed final variants. Training curves use the accepted history across any documented resume; rolled-back updates remain archived evidence rather than extra accepted training.

## What the full output tree contains

Each experiment preserves `incoming.pt`, `latest.pt`, all four `stageN_before_evaluation.pt` / `stageN.pt` pairs, and `final.pt`. Checkpoints include weights, RMSProp state, private/global RNG, configuration, initialization and counters. Each stage retains all 16 contexts × 16 headings × 12 variants, its summaries, raw tensors and evaluation-integrity record. Invalid contexts remain explicit failure records. Metrics, worker completion, and any failure/resume ancestry are retained rather than reduced to final averages.

The `latest.pt` checkpoint at a stage boundary can precede that stage's evaluation; this is intentional. The final checkpoint includes all four evaluations. Resume lineage records distinguish copied prior evidence from the accepted optimizer-update prefix. The full source/hash and file manifests specify exactly which bytes are included.

Any preserved stage-time audit snapshots are included as historical evidence. Their hashes for mutable uploads such as `latest.pt` and `metrics.jsonl` describe those earlier observations, not the final files. The two complete final audits validate the completed output trees.

The final descriptive memory gate requires at least 167/256 blind/native successes and at least 39 fewer successes after post-cue h/c erasure. All 128 physical-pair certificates must pass for the special certified-memory gate. Failed certificates remain in the conservative bound. One seed pair and repeated headings are not independent trained-network replications, and these behavior results do not presuppose a ring mechanism.

## Optional training or conditional dynamics

Re-running training spends substantial compute. Follow the separate [input/training guide](../local_goal_return_design/REPRODUCE.md), use fresh output folders, and preserve the fixed budget. The audits and plot commands above do not train. Reproducing trajectories on another backend is a new numerical execution and should be reported as such; exact stored checkpoint bytes are available for inspection.

Conditional ring analysis is separately specified in [the mechanism design](../../docs/research/local_return_mechanism_design.md). Its wrapper requires both audited finals and the prespecified useful-memory gate. It records a failed gate and stops automatic geometry if neither model qualifies. Use the fixed final stage and frozen behavior gate for this analysis.

## Packaging and extraction status

The packager is `scripts/package_local_goal_return_outputs.py`. It refuses incomplete audits, missing stages, unresolved cleanup, unsafe paths and symlinks; includes whole experiment trees; hashes every archived byte; and writes a deterministic tar.gz plus the final manifest/record. It performs no cloud, training or publication action. A final-source pin can be supplied with `--source-commit`; that commit must contain the exact runtime/audit/report source bytes.

Extraction of **all three actual assets** is checked separately in an isolated source checkout by reproducing the audits, private-stream comparison and report. Consult the release's extraction-validation record for that execution and its outcome. The earlier input-only extraction check concerns the two earlier assets and is retained as history.
