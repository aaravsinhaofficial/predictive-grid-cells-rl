# Reproduce the local goal-return pilot

This study adapts the original seed-1 Grid and PGC networks to a reward-only local navigation task. It retains the original CNN/LSTM/spatial weights and **explicitly replaces the action and value readouts with identical new heads**. All weights can then learn. The starting policies are stored exactly; reproduction never needs to regenerate their head initialization.

The input assets contain the untrained local-task baselines. Final trained checkpoints will be supplied separately. The experiment tests whether useful directional memory is learned before testing a ring mechanism.

## Source and both required assets

Use a fresh checkout of commit `6f316f4e4f5df23f40b177c6ed88ae11bbf2e795`:

```bash
git clone https://github.com/aaravsinhaofficial/predictive-grid-cells-rl.git
cd predictive-grid-cells-rl
git checkout 6f316f4e4f5df23f40b177c6ed88ae11bbf2e795
```

Download these two release assets to a directory outside the checkout. Release URLs are provided with the published study; this guide does not create or download a release.

| Required asset | Contents |
|---|---|
| `local-return-training-inputs-v1.tar.gz` | Original parent checkpoints/configs, both exact initialized policies, every baseline JSON/raw/control file, protocols and validation records |
| `local-return-behavior-banks-v1.tar.gz` | The two sealed 256-trial physical/random-input banks, original bank README and manifest |

The bank asset is **10,954,659 bytes**, SHA256 `d7c42a9294020b389034cd2d01f0441227b8aae0cd0b285e48c5c8b7ff7f3123`. The training-input asset's size and SHA256 are recorded in [`local_return_inputs_archive.json`](../../docs/research/local_return_inputs_archive.json) in the reporting commit. Verify the downloaded assets against these records before extraction. The asset records are outside their own archives to avoid recursive hashes.

From the repository root, substitute the real download directory:

```bash
tar -xzf /path/to/downloads/local-return-behavior-banks-v1.tar.gz
tar -xzf /path/to/downloads/local-return-training-inputs-v1.tar.gz
```

The assets restore repository-relative paths. In a fresh checkout, overlapping tracked protocol/baseline JSON files have the same bytes. Do not extract into a working directory containing your own revised outcomes. No source code, bank bytes, or original checkpoint needs editing. The input package omits final trained models, active worker routing, and duplicate report images.

For the recorded CPU analysis environment:

```bash
python -m venv .venv-local-return
source .venv-local-return/bin/activate
python -m pip install -r requirements-ring-analysis.txt
```

The local baseline used Torch 2.10.0. CUDA training requires a working CUDA-enabled Torch installation; the worker's actual backend/version is recorded in its output provenance. Loading the stored heads avoids backend-dependent initialization differences, but a different numerical backend is not a guarantee of bit-identical subsequent learning.

## Verify inputs and load exact starting policies

This is a read-only check: it loads weights and checks their identities, without running a policy rollout or an optimizer. It verifies every manifest entry, every frozen runtime source, both banks and the complete training protocol.

```bash
OPENBLAS_NUM_THREADS=2 python - <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
torch.set_num_threads(2)
sys.path.insert(0, 'scripts')
import local_goal_return_training_run as run
from rl_goal_memory import state_digest

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

manifest = json.loads(Path('results/local_goal_return_design/training_inputs_manifest.json').read_text())
for row in manifest['files']:
    path = Path(row['path'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
for name, digest in manifest['source_sha256'].items():
    assert sha(name) == digest, name
frozen = json.loads(run.PROTOCOL.read_text())
assert frozen.pop('frozen_utc')
assert frozen == run.protocol()
assert run.baseline_assets() == frozen['passed_baseline_assets']
for condition in ('gc100', 'pgc33'):
    net, cfg, initialization = run.load_baseline_policy(condition, 'cpu')
    assert state_digest(net) == initialization['full_initialized_policy_hash']
    print(condition, state_digest(net))
print('All packaged inputs and frozen training inputs verified; no rollout or update.')
PY
```

Expected initialized-policy hashes:

- Grid: `6a87cb72efd4107adc597b107772c9a5e3c44d8eaedc9f6260eb8c8ca5a8c015`
- PGC: `afd78fac6437dd870c18b4c4d683bf53b3b4d1c1f67e606415c9cd1506d1004a`

The frozen training protocol SHA256 is `0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038`. Historical implementation-review records retain their original source snapshots; the frozen protocol supplies the authoritative runtime hashes.

## Recheck saved baseline evidence and rebuild the figure

The raw audit reconciles all saved trials, actions, native motor trajectories, rewards, masks, Grid shams, initial head identities and the information-bound certificate. It executes no recurrent policy or new environment trial. Write its result to a fresh temporary location:

```bash
LOCAL_RETURN_AUDIT=$(mktemp -d)
OPENBLAS_NUM_THREADS=2 python scripts/audit_local_goal_return_baseline.py \
  --out "$LOCAL_RETURN_AUDIT/completed_baseline_audit.json"
OPENBLAS_NUM_THREADS=2 python scripts/plot_local_goal_return_baseline.py
```

The plot command reads the archived published audit and baseline records, and rebuilds only `results/local_goal_return_baseline/report/`. It refuses incomplete evidence. The expected baseline is **31/256 near successes for Grid and 32/256 for PGC; 0/256 hard successes for each**. The uniform control achieves 31/256 near and 0/256 hard; the heading-aware analysis-only planner achieves 256/256 in both banks. These are initial reward-discovery checks, not a learned PGC advantage.

Both populations pass the prespecified near-start requirement: at least 13/256 successes, zero forced rewards and passing physical checks. The hard bank's 128 opposite-heading pairs all pass the common-information certificate. Its 50% ceiling is conditional on the stored tapes and held-observation rule; it does not apply to an agent retaining heading information.

## Optional: regenerate baseline rollouts separately

This executes new environment/policy evaluations, but no optimization. Each output directory must be new and empty. The exact stored initialized policy is loaded via `--checkpoint`; no published outcome is overwritten.

```bash
LOCAL_RETURN_REPLAY=$(mktemp -d)
for condition in gc100 pgc33; do
  OPENBLAS_NUM_THREADS=2 python scripts/local_goal_return_training_evaluate.py \
    --condition "$condition" --seed 1 --device cpu --controls \
    --checkpoint "results/local_goal_return_baseline/$condition/initialized_policy.pt" \
    --out "$LOCAL_RETURN_REPLAY/$condition"
done
```

Compare each regenerated bank's trial rows and variant/control summaries with the archived counterpart. Execution metadata intentionally identifies the supplied checkpoint, unlike the original initializer run. The original baseline auditor checks that original initializer provenance, so its command above targets the archived baseline, not this checkpoint-mode replay. Cross-backend numerical differences should be recorded rather than silently rewriting the reference results.

## Optional: run the fixed training experiment

**These commands spend substantial compute.** They do not provision a VM. Run on a suitable CUDA machine only when you intend to reproduce training, after the read-only checks pass. Each population uses 4 stages × 1,024 batches × 32 trials = **131,072 trials**, with 4,096 optimizer updates. Actual experienced physics frames are active updates × 4; setup, padding and analysis-only planning are excluded.

Choose a new output parent. Each `--out` child must be empty; keep logs outside it until the runner starts.

```bash
LOCAL_RETURN_RUN=$(mktemp -d)
OPENBLAS_NUM_THREADS=2 python scripts/local_goal_return_training_run.py \
  --condition gc100 --seed 1 --device cuda --out "$LOCAL_RETURN_RUN/gc100"
OPENBLAS_NUM_THREADS=2 python scripts/local_goal_return_training_run.py \
  --condition pgc33 --seed 1 --device cuda --out "$LOCAL_RETURN_RUN/pgc33"
```

The sources load the exact archived starting policies, validate the frozen protocol, use fresh native RMSProp, and retain all four prescribed stage evaluations. The process cap is 12,600 seconds per run; an incomplete run remains incomplete. A compatible saved boundary checkpoint can resume with `--resume /path/to/checkpoint.pt` into another empty output directory. Resume requires the original condition/backend/version/provenance, retains prior-attempt evidence, and finishes the same fixed budget rather than extending it. Do not use `--freeze-only` when reproducing this already frozen study.

The last prescribed checkpoint is the endpoint; do not select a better stage. The final descriptive behavior gate is at least **167/256 intact hard successes** and at least **39 fewer successes after post-cue state erasure**. The special certified-memory gate also requires all 128 pair certificates; failed pairs remain with conservative bound 1 rather than 0.5. These are pilot decisions for one original seed pair, not significance across independent trained seeds. Passing them motivates subsequent causal dynamics tests; it does not predetermine a ring result.

[Frozen training protocol](../../docs/research/local_goal_return_training_protocol.json) · [Frozen banks and gates](behavior_protocol.json) · [Complete baseline audit](completed_baseline_audit.json) · [Task figure](design/README.md)
