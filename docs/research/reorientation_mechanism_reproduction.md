# Reproduce the fixed-final reorientation mechanism tests

Run from the repository root after both **audited final** checkpoints arrive. Grid uses `ring-20261105-reorient-grid-01`; PGC uses the actual `ring-20261105-reorient-pgc-02` artifact, not the routing alias. The wrappers require the completed 40,058,880-frame continuation, exact training provenance, 200 warmup updates, and 2,156 total updates. Do not select a stage checkpoint.

The commands first verify both frozen source manifests, then collect fresh native histories from each final network, apply the frozen primary geometry tests, then run every prescribed causal/navigation context regardless of the geometry outcome. The original shape-check directories remain separate.

```sh
cd /Users/aaravsinha/predictive-grid-cells-rl
export OPENBLAS_NUM_THREADS=2
REORIENT_PYTHON=/opt/anaconda3/bin/python
REORIENT_OUT=results/lstm_reorientation_mechanism/final

"$REORIENT_PYTHON" - <<'PY'
from pathlib import Path
import hashlib, json
for manifest in (
    'results/lstm_reorientation_mechanism/implementation_protocol.json',
    'results/lstm_reorientation_mechanism/causal_navigation/protocol.json',
):
    for name, expected in json.loads(Path(manifest).read_text())['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Frozen source hashes match')
PY

run_reorientation () {
  "$REORIENT_PYTHON" scripts/lstm_reorientation_mechanism.py \
    --checkpoint "$2" --out "$REORIENT_OUT/$1" || return 1
  for context in 0 1; do
    "$REORIENT_PYTHON" scripts/lstm_reorientation_causal_navigation.py \
      --checkpoint "$2" \
      --states "$REORIENT_OUT/$1/context${context}_states.npz" \
      --out "$REORIENT_OUT/$1/causal/context${context}" || return 1
  done
}

run_reorientation gc100 \
  results/ring_cloud_artifacts/ring-20261105-reorient-grid-01/final.pt
run_reorientation pgc33 \
  results/ring_cloud_artifacts/ring-20261105-reorient-pgc-02/final.pt
```


The main outputs are `context0.json`/`context1.json` for geometry, `native_phase_coverage.json` for all recorded native event directions, and `causal/contextN/result.json` plus `navigation_paths.npz` for interventions and actual behavior. All native context caches, incomplete/invalid statuses, and unsuccessful navigation windows must be retained. Angular samples and two contexts are nested measurements of one trained seed pair.

The crucial time indexing is:

| Saved state | Input already processed | Next physical action |
|---|---|---|
| `pre_cue_state` | Last pre-glimpse history | Process the endpoint cue first |
| `states[0]` | The one endpoint cue; previous action is forward | First imposed LEFT |
| `states[15]` | Ordinary input after 15 LEFT actions | LEFT action 16 |
| `states[60]` | First free-choice input, after 60 LEFT actions | **Use `pi(h)` directly for free action 1** |

After free action 1, advance the real environment and process the next held observation with that actual previous action. Thirty free actions require **29 further recurrent updates**. Carry the angular velocity left by the sixty imposed turns; do not brake again at free onset. Stop each window at its first real goal hit, before respawn or goal-message refresh. The causal wrapper enforces these rules and uses identical phase/time-indexed motor and action draws across branches.

The controlled forced delay is noiseless to match the saved neural flow; free physics uses native motor noise. Whole-task behavioral evaluation retains its native noisy forced turns and is a separate measurement. No analysis decoder or ideal action is supplied to the policy. The crossed-context/common-encoder comparison in the [review addendum](reorientation_mechanism_review_addendum.md) remains a separate prospective sensitivity analysis, outside these commands.

[Primary design](reorientation_mechanism_protocol.json) · [Causal test explanation](../../results/lstm_reorientation_mechanism/causal_navigation/README.md) · [Original transfer coverage](../../results/lstm_reorientation_mechanism/original_angular_coverage/README.md)
