#!/bin/bash
# Wave C. When the concentration series (fleet v2b) has been collected and analysed by
# auto_wave_b.sh, pick the best PGC percentage (scripts/pick_concentration.py). If it is not
# 33 %, rerun every arrangement at that percentage at the same scale (9 codes x 8 seeds =
# 72 runs; the quarter-spacing code at that percentage already exists from the concentration
# series): 48 runs on GCP (tag v2c, 16 L4 VMs, 3 per VM) and 24 locally (tmux popcode_wavec,
# 12 concurrent, started once the band pilot is evaluated). Then analyse, make the figures,
# write docs/arrangements_<P>pct.md and package the data.
export PATH=$PATH:/home/ec2-user/google-cloud-sdk/bin
PY=/home/ec2-user/banino/.venv/bin/python
cd /home/ec2-user/optimization
LOG=runs/auto_wave_c.log
W=runs/wavec
mkdir -p $W
log() { echo "$(TZ=America/Chicago date '+%Y-%m-%d %H:%M CST') $*" | tee -a $LOG; }
log "wave C driver start"
until grep -q "DONE:" runs/auto_wave_b.log 2>/dev/null; do sleep 300; done
log "wave B done; picking the best concentration"
$PY scripts/pick_concentration.py --runs_dir runs --out results/best_concentration.json > $W/pick.txt 2>&1
cat $W/pick.txt >> $LOG
P=$(tail -n 1 $W/pick.txt)
if ! [[ "$P" =~ ^[0-9]+$ ]]; then log "picker failed (see $W/pick.txt)"; exit 1; fi
if [ "$P" = "33" ]; then log "best concentration is 33 %: no rerun needed"; exit 0; fi
log "best concentration: $P % -> rerunning all arrangements at $P % (9 codes x 8 seeds)"
$PY scripts/make_matrix_v2.py rerun --frac $P --seeds 8 --out $W/waveC.json >> $LOG 2>&1
$PY - "$W" <<'PYEOF' >> $LOG 2>&1
import json, sys
w = sys.argv[1]
runs = json.load(open(f'{w}/waveC.json'))
runs.sort(key=lambda r: (r['seed'], r['name']))
local, cloud = runs[-24:], runs[:-24]
json.dump(cloud, open(f'{w}/cloud.json', 'w'), indent=1)
json.dump(local, open(f'{w}/local.json', 'w'), indent=1)
print(f'{len(cloud)} runs on GCP, {len(local)} local: {[r["name"] for r in local]}')
PYEOF
VMS=16
log "cloud part: $VMS VMs x 3 runs, estimated ~\$$($PY -c "print(round($VMS*6.2*0.85))") (fleets so far ~\$200 of the \$300 budget)"
until gcloud auth print-access-token >/dev/null 2>&1; do
  log "gcloud token expired: the user must run  gcloud auth login --no-launch-browser  (retrying in 10 min)"; sleep 600
done
$PY scripts/gcp_fleet.py package >> $LOG 2>&1
$PY scripts/gcp_fleet.py plan --matrix_json $W/cloud.json --vms $VMS --tag v2c >> $LOG 2>&1
$PY scripts/gcp_fleet.py launch --tag v2c --vms $VMS --max_concurrent 3 \
   --zones us-central1-a,us-central1-b,us-central1-c,us-central1-f,us-east1-b,us-east1-c,us-east1-d,us-west1-a,us-west1-b,us-east4-a,us-east4-b >> $LOG 2>&1
N=$(gcloud compute instances list --filter="name~^popcode-v2c-" --format="value(name)" 2>/dev/null | wc -l)
log "cloud part launched: $N/$VMS VMs"
for attempt in 1 2 3; do
  if [ "$N" -ge "$VMS" ]; then break; fi
  sleep 600
  $PY scripts/gcp_fleet.py launch --tag v2c --vms $VMS --max_concurrent 3 \
     --zones us-central1-c,us-central1-f,us-east1-b,us-east1-c,us-east1-d,us-west1-a,us-west1-b,us-east4-a,us-east4-b,us-west4-a,us-south1-a >> $LOG 2>&1
  N=$(gcloud compute instances list --filter="name~^popcode-v2c-" --format="value(name)" 2>/dev/null | wc -l)
  log "after retry $attempt: $N/$VMS VMs"
done
if [ "$N" -eq 0 ]; then log "no VMs could be created; stopping (local part not started)"; exit 1; fi

band_done() { $PY - <<'PYEOF'
import json, os, sys
ok = 0
for c in ('band', 'pband33', 'pband100'):
    for s in range(1, 5):
        p = f'runs/{c}_ffb_s{s}/eval/intact.json'
        if os.path.exists(p) and json.load(open(p)).get('episodes', 0) >= 200:
            ok += 1
sys.exit(0 if ok >= 12 else 1)
PYEOF
}
# local part in the background so that the cloud monitor starts right away
(
  until band_done; do sleep 300; done
  log "band pilot evaluated; starting the local part (24 runs, 12 concurrent) in tmux popcode_wavec"
  tmux new-session -d -s popcode_wavec "cd /home/ec2-user/optimization && $PY scripts/launch_matrix.py --matrix_json $W/local.json --max_concurrent 12 >> $W/local_scheduler.log 2>&1"
) &
FLEET_LOG=runs/fleet_monitor_v2c.log ./scripts/fleet_monitor.sh v2c:$N
log "cloud part collected"
wait
local_done() { $PY - "$W" <<'PYEOF'
import json, os, sys
w = sys.argv[1]
runs = json.load(open(f'{w}/local.json'))
ok = sum(1 for r in runs if os.path.exists(f"runs/{r['name']}/eval/intact.json")
         and json.load(open(f"runs/{r['name']}/eval/intact.json")).get('episodes', 0) >= 200)
print(f'{ok}/{len(runs)} local runs evaluated')
sys.exit(0 if ok >= len(runs) else 1)
PYEOF
}
until local_done >> $LOG 2>&1; do sleep 900; done
log "local part finished; final analysis"
$PY -m popcode.analyze --runs_dir runs --out results >> $LOG 2>&1
$PY scripts/make_figures.py --runs_dir runs --out figures >> $LOG 2>&1
$PY scripts/arrangement_report.py --pct $P --runs_dir runs --out docs/arrangements_${P}pct.md >> $LOG 2>&1
$PY scripts/package_data.py --runs_dir runs --out data >> $LOG 2>&1
cp results/summary.json data/summary.json
log "DONE: wave C at $P % -> figures/fig7_arrangements_${P}pct.png, figures/arrangements_${P}pct/, docs/arrangements_${P}pct.md"
