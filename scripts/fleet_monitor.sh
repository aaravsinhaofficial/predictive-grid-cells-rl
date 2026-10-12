#!/bin/bash
# Detached fleet monitor: every 10 min, for each fleet tag, count DONE markers;
# when a fleet is fully done (or has no running VMs), collect it into runs/,
# delete its VMs; when all fleets are collected, run the aggregate analysis.
export PATH=$PATH:/home/ec2-user/google-cloud-sdk/bin
PY=/home/ec2-user/banino/.venv/bin/python
cd /home/ec2-user/optimization
LOG=runs/fleet_monitor.log
LOG=${FLEET_LOG:-runs/fleet_monitor.log}
# usage: fleet_monitor.sh tag:count [tag:count ...]   (default: main:32 dose:6 ext:4)
declare -A EXPECT=()
if [ $# -gt 0 ]; then for kv in "$@"; do EXPECT[${kv%%:*}]=${kv##*:}; done
else EXPECT=( [main]=32 [dose]=6 [ext]=4 ); fi
TAGS="${!EXPECT[@]}"
declare -A COLLECTED=()
log() { echo "$(TZ=America/Chicago date '+%Y-%m-%d %H:%M CST') $*" | tee -a $LOG; }
log "monitor start"
while true; do
  all_done=1
  for tag in $TAGS; do
    [ -n "${COLLECTED[$tag]}" ] && continue
    prefix=runs; [ "$tag" != main ] && prefix=runs_$tag
    n=$(gsutil ls gs://connectome-ai-research-popcode/$prefix/popcode-$tag-*/DONE 2>/dev/null | wc -l)
    alive=$(gcloud compute instances list --filter="name~^popcode-$tag-" --format="value(status)" 2>/dev/null | grep -cE "RUNNING|PROVISIONING|STAGING|REPAIRING")
    total=$(gcloud compute instances list --filter="name~^popcode-$tag-" --format="value(name)" 2>/dev/null | wc -l)
    log "$tag: done=$n/${EXPECT[$tag]} vms alive=$alive/$total"
    # Collect + delete ONLY when every expected VM has written its DONE marker.
    if [ "$n" -ge "${EXPECT[$tag]}" ]; then
      log "$tag: collecting"
      $PY scripts/gcp_fleet.py collect --tag $tag --dest runs >> $LOG 2>&1 && COLLECTED[$tag]=1 && log "$tag: collected" \
        && $PY scripts/gcp_fleet.py cleanup --tag $tag >> $LOG 2>&1 && log "$tag: VMs deleted"
    else
      all_done=0
    fi
  done
  if [ "$all_done" -eq 1 ]; then
    log "all fleets collected; running analysis"
    $PY -m code.analyze --runs_dir runs --out results >> $LOG 2>&1
    log "analysis written to results/RESULTS.md"
    break
  fi
  sleep 600
done
