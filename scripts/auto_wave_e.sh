#!/bin/bash
# Driver for the control / extension wave that runs partly on GCP and partly
# on the local GPU.
#
#   ./scripts/auto_wave_e.sh ctl:24:runs ext6:8:runs_ext long:5:runs_ext
#
# Each argument is <fleet tag>:<number of VMs>:<local destination>. Every
# 10 minutes the driver counts the DONE markers of each fleet in the bucket;
# when a fleet's count reaches its VM number, the fleet is collected into its
# destination (scripts/gcp_fleet.py collect) and its VMs are deleted. When
# every fleet has been collected and the local scheduler (scripts/launch_matrix.py
# on runs_ext) has exited, it runs scripts/auto_wave_d.sh --now (milestone
# evaluations, analysis, extension report, figures, data package). An expired
# gcloud token is logged and retried; collection resumes once the token is
# renewed (gcloud auth login --no-launch-browser). The log also carries a
# running estimate of the fleet cost (alive VMs x poll interval x $0.85/h).
# Logs to runs/auto_wave_e.log with US Central timestamps.
export PATH=$PATH:/home/ec2-user/google-cloud-sdk/bin
PY=/home/ec2-user/banino/.venv/bin/python
BUCKET=connectome-ai-research-popcode
PRICE=0.85
POLL=${POLL:-600}
cd /home/ec2-user/optimization || exit 1
LOG=runs/auto_wave_e.log
log() { echo "$(TZ=America/Chicago date '+%Y-%m-%d %H:%M CST') $*" | tee -a "$LOG"; }
declare -A EXPECT=() DEST=() COLLECTED=()
for spec in "$@"; do
  IFS=: read -r tag n dest <<< "$spec"
  EXPECT[$tag]=$n; DEST[$tag]=$dest
done
[ ${#EXPECT[@]} -gt 0 ] || { echo "usage: $0 tag:vms:dest [tag:vms:dest ...]"; exit 2; }
log "driver start: fleets ${!EXPECT[*]}"
COST=0
while true; do
  if ! gcloud auth print-access-token > /dev/null 2>&1; then
    log "gcloud token expired: run  gcloud auth login --no-launch-browser  (retrying in $POLL s)"
    sleep "$POLL"; continue
  fi
  all=1
  alive_total=0
  for tag in "${!EXPECT[@]}"; do
    [ -n "${COLLECTED[$tag]}" ] && continue
    n=$(gsutil ls gs://$BUCKET/runs_$tag/popcode-$tag-*/DONE 2>/dev/null | wc -l)
    alive=$(gcloud compute instances list --filter="name~^popcode-$tag-" --format="value(status)" 2>/dev/null | grep -cE "RUNNING|PROVISIONING|STAGING|REPAIRING")
    total=$(gcloud compute instances list --filter="name~^popcode-$tag-" --format="value(name)" 2>/dev/null | wc -l)
    alive_total=$((alive_total + alive))
    log "$tag: done=$n/${EXPECT[$tag]} vms alive=$alive/$total"
    if [ "$n" -ge "${EXPECT[$tag]}" ]; then
      log "$tag: collecting into ${DEST[$tag]}"
      if $PY scripts/gcp_fleet.py collect --tag "$tag" --dest "${DEST[$tag]}" >> "$LOG" 2>&1; then
        COLLECTED[$tag]=1; log "$tag: collected"
        $PY scripts/gcp_fleet.py cleanup --tag "$tag" >> "$LOG" 2>&1 && log "$tag: VMs deleted"
      else
        log "$tag: collect FAILED (will retry)"; all=0
      fi
    else
      all=0
    fi
  done
  COST=$(awk -v c="$COST" -v a="$alive_total" -v p="$POLL" -v r="$PRICE" 'BEGIN{printf "%.2f", c + a*p/3600*r}')
  log "fleet cost so far ~\$$COST (alive VMs $alive_total)"
  if [ "$all" -eq 1 ]; then
    if pgrep -f '^[^ ]*python scripts/launch_matrix.py --matrix_json' > /dev/null; then
      log "all fleets collected; waiting for the local scheduler"
    else
      log "all fleets collected and the local scheduler has exited; post-processing"
      ./scripts/auto_wave_d.sh --now >> "$LOG" 2>&1
      log "DONE"
      break
    fi
  fi
  sleep "$POLL"
done
