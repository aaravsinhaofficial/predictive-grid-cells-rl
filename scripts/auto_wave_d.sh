#!/bin/bash
# Local post-processing driver for the control wave (runs/) and the extension
# wave (runs_ext/). No cloud resources are involved.
#
# Waits until no scripts/launch_matrix.py process is running and both
# runs/scheduler.log and runs_ext/scheduler.log end with a 'scheduler finished'
# line written after this driver started; then
#   (i)   evaluates, with the full probe set at 200 episodes, the checkpoint at
#         each milestone (600M, 1B, 1.5B) of every run in runs_ext whose budget
#         exceeds that milestone: the first checkpoint at or after the milestone
#         and less than 25M frames past it (ckpt_0600000000.pt,
#         ckpt_1000000000.pt; ckpt_1500006400.pt for a run continued from 1B);
#         probes whose evaluation file already exists are skipped (files
#         eval/<probe>@<frames>f_200ep.json, <frames> = the checkpoint's)
#   (ii)  python -m code.analyze --runs_dir runs --out results
#   (iii) python scripts/extension_report.py --runs_dir runs_ext --out results_ext --figures figures
#   (iv)  python scripts/make_figures.py --runs_dir runs --out figures
#   (v)   python scripts/package_data.py --runs_dir runs --out data; cp results/summary.json data/summary.json
#   (vi)  logs DONE
# The probe set is read from scripts/launch_matrix.py (FULL_PROBES) at run time.
# Logs to runs/auto_wave_d.log with US Central timestamps.
#
#   nohup ./scripts/auto_wave_d.sh > /dev/null 2>&1 &   # wait for the schedulers, then post-process
#   ./scripts/auto_wave_d.sh --now                       # post-process immediately (re-runnable)
#   ./scripts/auto_wave_d.sh --now --dry_run             # only print the commands
# Directories can be overridden through the environment (RUNS, RUNS_EXT,
# RESULTS, RESULTS_EXT, FIGURES, DATA) and the milestone episode count through
# EPISODES (default 200), e.g. for a smoke test outside the repository.
PY=/home/ec2-user/banino/.venv/bin/python
cd /home/ec2-user/optimization || exit 1
RUNS=${RUNS:-runs}
RUNS_EXT=${RUNS_EXT:-runs_ext}
RESULTS=${RESULTS:-results}
RESULTS_EXT=${RESULTS_EXT:-results_ext}
FIGURES=${FIGURES:-figures}
DATA=${DATA:-data}
EPISODES=${EPISODES:-200}
POLL=${POLL:-300}
MILESTONES="600000000 1000000000 1500000000"
mkdir -p "$RUNS"
LOG=$RUNS/auto_wave_d.log
log() { echo "$(TZ=America/Chicago date '+%Y-%m-%d %H:%M CST') $*" | tee -a "$LOG"; }

NOW=0
DRY=0
for a in "$@"; do
  case $a in
    --now) NOW=1 ;;
    --dry_run) DRY=1 ;;
    *) echo "unknown argument $a (use --now and/or --dry_run)"; exit 2 ;;
  esac
done
run() {   # log + run a command, its output appended to the log
  log "+ $*"
  if [ "$DRY" = 0 ]; then
    "$@" >> "$LOG" 2>&1 || log "FAILED (rc=$?): $*"
  fi
}
START=$(date +%s)

# True when $1 ends with a 'scheduler finished' line whose timestamp (box clock,
# the same clock as `date`) is not older than this driver's start.
finished_after_start() {
  [ -f "$1" ] || return 1
  local last ts
  last=$(tail -n 1 "$1")
  case "$last" in *"scheduler finished"*) ;; *) return 1 ;; esac
  ts=$(date -d "${last:0:19}" +%s 2>/dev/null) || return 1
  [ "$ts" -ge "$START" ]
}

if [ "$NOW" = 0 ]; then
  log "auto driver start: waiting for the schedulers ($RUNS/scheduler.log, $RUNS_EXT/scheduler.log)"
  until ! pgrep -f "launch_matrix.py" > /dev/null \
        && finished_after_start "$RUNS/scheduler.log" \
        && finished_after_start "$RUNS_EXT/scheduler.log"; do
    sleep "$POLL"
  done
  log "schedulers finished; post-processing"
else
  log "post-processing now (--now)"
fi

# (i) milestone evaluations of the long runs
FULL_PROBES=$($PY -c "import sys; sys.path.insert(0, 'scripts'); import launch_matrix; print(launch_matrix.FULL_PROBES)")
log "full probe set: $FULL_PROBES"
for d in "$RUNS_EXT"/*_s[0-9]*/; do
  d=${d%/}
  [ -f "$d/config.json" ] || continue
  TARGET=$($PY -c "import json, sys; print(int(json.load(open(sys.argv[1]))['frames']))" "$d/config.json")
  for M in $MILESTONES; do
    [ "$TARGET" -gt "$M" ] || continue
    CK=$($PY - "$d" "$M" <<'PYEOF'
import glob, os, sys
d, m = sys.argv[1], int(sys.argv[2])
cks = sorted((int(os.path.basename(f)[5:15]), f) for f in glob.glob(os.path.join(d, 'ckpt_[0-9]*.pt')))
print(next((f for fr, f in cks if m <= fr < m + 25_000_000), ''))
PYEOF
)
    if [ -z "$CK" ]; then
      log "$(basename "$d"): no checkpoint within 25M frames of $M yet"
      continue
    fi
    F=$((10#$(basename "$CK" .pt | cut -c6-15)))
    MISSING=$($PY - "$d" "$F" "$EPISODES" "$FULL_PROBES" <<'PYEOF'
import os, sys
d, m, ep, probes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(',')
print(','.join(p for p in probes
               if not os.path.exists(os.path.join(d, 'eval', f'{p}@{m}f_{ep}ep.json'))))
PYEOF
)
    if [ -z "$MISSING" ]; then
      log "$(basename "$d") @ $F: all probes evaluated"
      continue
    fi
    run $PY -m code.evaluate --run "$d" --ckpt "$CK" --probes "$MISSING" --episodes "$EPISODES"
  done
done

# (ii) - (v) aggregate results, extension report, figures, data package
run $PY -m code.analyze --runs_dir "$RUNS" --out "$RESULTS"
run $PY scripts/extension_report.py --runs_dir "$RUNS_EXT" --out "$RESULTS_EXT" --figures "$FIGURES"
run $PY scripts/make_figures.py --runs_dir "$RUNS" --out "$FIGURES"
run $PY scripts/package_data.py --runs_dir "$RUNS" --out "$DATA"
run cp "$RESULTS/summary.json" "$DATA/summary.json"
log "DONE: $RESULTS/RESULTS.md, $RESULTS_EXT/EXTENSION.md, $FIGURES/, $DATA/ updated"
