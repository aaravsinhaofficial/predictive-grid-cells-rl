#!/bin/bash
# Waits for the arrangement wave (tag v2a) to be collected, picks the best PGC
# arrangement (highest mean final score over 8 seeds among uniform / fine /
# coarse / middle / phase / jitter, code-only arm), launches the concentration
# series on it (tag v2b, 10/25/30/35/40 %, 8 seeds), monitors it, and runs the
# final analysis + figures + data packaging when it is collected.
export PATH=$PATH:/home/ec2-user/google-cloud-sdk/bin
PY=/home/ec2-user/banino/.venv/bin/python
cd /home/ec2-user/optimization
LOG=runs/auto_wave_b.log
log() { echo "$(TZ=America/Chicago date '+%Y-%m-%d %H:%M CST') $*" | tee -a $LOG; }
log "auto driver start"
until grep -q "v2a: collected" runs/fleet_monitor_v2a.log 2>/dev/null && grep -q "v2a2: collected" runs/fleet_monitor_v2a2.log 2>/dev/null; do sleep 300; done
log "arrangement waves (v2a + v2a2) collected; analysing"
$PY -m popcode.analyze --runs_dir runs --out results >> $LOG 2>&1
BEST=$($PY - <<'PYEOF'
import json
s = json.load(open('results/summary.json'))['groups']
cands = {'pgc33': '', 'pgc33fine': '--pgc_layout fine', 'pgc33coarse': '--pgc_layout coarse',
         'pgc33middle': '--pgc_layout middle',
         'pgc33phase': '--pgc_shift_rule spacing --pgc_shift_frac 0.5',
         'pgc33jitter': '--pgc_shift_jitter 0.5',
         'pgc33phase27': '--pgc_shift_rule spacing --pgc_shift_frac 0.27',
         'pgc33bio': '--pgc_shift_rule spacing --pgc_shift_frac 0.27 --pgc_shift_jitter 0.4',
         'pgc33mec': '--pgc_layout weights --pgc_module_weights 1,2,2,1'}
scores = {}
for c in cands:
    g = s.get(f'ffb/{c}')
    if g and g.get('eval_intact') and g['n'] >= 6:
        scores[c] = g['eval_intact']['mean']
best = max(scores, key=scores.get)
with open('results/best_arrangement.json', 'w') as f:
    json.dump(dict(best=best, extra_args=cands[best], mean_final_score=scores), f, indent=1)
print(best)
PYEOF
)
EXTRA=$($PY -c "import json; print(json.load(open('results/best_arrangement.json'))['extra_args'])")
log "best arrangement: $BEST (extra args: '$EXTRA')"
$PY scripts/make_matrix_v2.py concentrations --layout "$EXTRA" --seeds 8 --out /tmp/waveB.json >> $LOG 2>&1
$PY scripts/gcp_fleet.py package >> $LOG 2>&1
$PY scripts/gcp_fleet.py plan --matrix_json /tmp/waveB.json --vms 14 --tag v2b >> $LOG 2>&1
$PY scripts/gcp_fleet.py launch --tag v2b --vms 14 --max_concurrent 3 \
   --zones us-central1-a,us-central1-b,us-central1-c,us-central1-f,us-east1-b,us-east1-c,us-east1-d,us-west1-a,us-west1-b,us-east4-a,us-east4-b >> $LOG 2>&1
N=$(gcloud compute instances list --filter="name~^popcode-v2b-" --format="value(name)" 2>/dev/null | wc -l)
log "concentration wave launched: $N/14 VMs"
if [ "$N" -lt 14 ]; then
  sleep 600
  $PY scripts/gcp_fleet.py launch --tag v2b --vms 14 --max_concurrent 3 \
     --zones us-central1-c,us-central1-f,us-east1-b,us-east1-c,us-east1-d,us-west1-a,us-west1-b,us-east4-a,us-east4-b,us-west4-a,us-south1-a >> $LOG 2>&1
  N=$(gcloud compute instances list --filter="name~^popcode-v2b-" --format="value(name)" 2>/dev/null | wc -l)
  log "after retry: $N/14 VMs"
fi
FLEET_LOG=runs/fleet_monitor_v2b.log ./scripts/fleet_monitor.sh v2b:$N
log "concentration wave collected; final analysis"
$PY -m popcode.analyze --runs_dir runs --out results >> $LOG 2>&1
$PY scripts/make_figures.py --runs_dir runs --out figures >> $LOG 2>&1
$PY scripts/package_data.py --runs_dir runs --out data >> $LOG 2>&1
cp results/summary.json data/summary.json
log "DONE: results/RESULTS.md, results/best_arrangement.json, figures/, data/ updated"
