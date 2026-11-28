"""GCP workers for the code-variation experiment.

Each VM (one L4 GPU) gets a slice of a run matrix and runs
scripts/launch_matrix.py on it: several training runs side by side, each
resumable, and the final test of every finished run. The run directories are
copied to gs://<bucket>/codes/<tag>/runs/<vm>/ every five minutes. VMs are
spot instances by default: when one is preempted it stops, `watch` starts it
again and the scheduler resumes every run from its last saved state.

`watch` is the driver (run it in tmux). On every pass it
  - starts stopped VMs that are not finished (or recreates them elsewhere when
    their zone has no capacity),
  - deletes VMs whose work is finished (DONE marker in the bucket),
  - adds the cost of the running VMs to <dest>/cost.json (list prices),
  - deletes every VM of the tag when the cost reaches --budget.

  python -m scripts.codes.fleet package --tag t1
  python -m scripts.codes.fleet plan    --tag t1 --matrix runs_codes/t1/matrix.json --per_vm 5
  python -m scripts.codes.fleet launch  --tag t1
  python -m scripts.codes.fleet watch   --tag t1 --budget 400        # tmux
  python -m scripts.codes.fleet status  --tag t1
  python -m scripts.codes.fleet collect --tag t1
  python -m scripts.codes.fleet cleanup --tag t1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil  # noqa: E402
GCLOUD = os.path.expanduser('~/google-cloud-sdk/bin/gcloud')
GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
if not os.path.exists(GCLOUD):          # e.g. on the controller VM
  GCLOUD, GSUTIL = shutil.which('gcloud') or 'gcloud', shutil.which('gsutil') or 'gsutil'
PROJECT = 'connectome-ai-research'
BUCKET = 'connectome-ai-research-popcode'
IMAGE_FAMILY, IMAGE_PROJECT = 'pytorch-2-9-cu129-ubuntu-2204-nvidia-580', 'deeplearning-platform-release'
# List prices (USD per hour) used for the cost estimate: g2-standard-8 with one L4.
PRICE = {'SPOT': 0.40, 'STANDARD': 0.90}
DISK_GB, DISK_PRICE = 100, 0.10 / 730.0          # pd-balanced, USD per GB-hour
# Spot quota is 16 L4 per region, so the fleet is spread over regions: the five US regions
# first (the bucket is in us-central1; transfers inside North America are the cheapest).
N_FIRST = 5
ZONES = ['us-central1-a', 'us-east4-a', 'us-west4-a', 'us-east1-b', 'us-west1-b',
         'us-central1-b', 'us-east4-c', 'us-west4-c', 'us-east1-c', 'us-west1-a',
         'us-central1-c', 'us-east1-d', 'us-west1-c', 'us-west4-b', 'us-east4-b',
         'europe-west4-a', 'europe-west1-b', 'europe-west2-a', 'europe-west4-b', 'europe-west1-c',
         'europe-west2-b', 'europe-west4-c', 'europe-west3-b', 'northamerica-northeast1-b',
         'asia-southeast1-a', 'asia-northeast1-a', 'asia-southeast1-b', 'asia-northeast1-c',
         'asia-southeast1-c', 'asia-east1-a', 'asia-east1-c', 'asia-south1-a', 'us-south1-a']
PROBES = 'intact,no_code,no_goal,no_pgc,pgc_as_gc,sham0,sham1,sham2'

STARTUP = r'''#!/bin/bash
# code-variation worker (runs at every boot: a restarted spot VM resumes its runs)
exec > >(tee -a /var/log/codes-startup.log) 2>&1
set -x
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
md() { curl -s -H 'Metadata-Flavor: Google' "$MD/$1"; }
BUCKET=$(md bucket); TAG=$(md tag); MAXC=$(md max_concurrent); DEADMAN=$(md deadman)
PROBES=$(md probes); TRAIN_ARGS=$(md train_args)
VM=$(hostname)
shutdown -h +$DEADMAN                      # dead-man: bounds the cost of a forgotten VM
mkdir -p /work && cd /work
for i in $(seq 1 45); do nvidia-smi && break; sleep 20; done
gsutil cp "gs://$BUCKET/codes/$TAG/code.tar.gz" code.tar.gz && tar xzf code.tar.gz || exit 1
gsutil cp "gs://$BUCKET/codes/$TAG/assign/$VM.json" assign.json || exit 1
PYBIN=$(ls /opt/conda/bin/python 2>/dev/null || which python3)
$PYBIN -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" || exit 1
if [ ! -d runs ]; then                     # new disk: take whatever state the bucket has
  mkdir -p runs
  gsutil -m rsync -r "gs://$BUCKET/codes/$TAG/runs/$VM" runs || true
fi
if gsutil -q stat "gs://$BUCKET/codes/$TAG/runs/$VM/DONE"; then shutdown -h now; exit 0; fi
# Logs, the resume states and the final networks go to the bucket every 15 minutes (a VM
# rebuilt in another zone then loses at most that much); numbered checkpoints go hourly.
( sleep 240; gsutil -m rsync -r -x '.*ckpt_[0-9].*\.pt$' runs "gs://$BUCKET/codes/$TAG/runs/$VM" >/dev/null 2>&1
  k=0
  while true; do
    sleep 900; k=$((k + 1))
    if [ $((k % 4)) = 0 ]; then
      gsutil -m rsync -r runs "gs://$BUCKET/codes/$TAG/runs/$VM" >/dev/null 2>&1
    else
      gsutil -m rsync -r -x '.*ckpt_[0-9].*\.pt$' runs "gs://$BUCKET/codes/$TAG/runs/$VM" >/dev/null 2>&1
    fi
  done ) &
SYNC=$!
sed -i "s#^PY = .*#PY = '$PYBIN'#" scripts/launch_matrix.py
$PYBIN scripts/launch_matrix.py --matrix_json assign.json --runs_dir /work/runs --max_concurrent "$MAXC" \
    --no_ckpt_probes --probes "$PROBES" --train_args "$TRAIN_ARGS" 2>&1 | tee -a runs/scheduler.stdout
kill $SYNC
N_OK=$(ls runs/*/eval/intact.json 2>/dev/null | wc -l)
N_ALL=$($PYBIN -c "import json; print(len(json.load(open('assign.json'))))")
for i in 1 2 3 4 5 6; do
  gsutil -m rsync -r runs "gs://$BUCKET/codes/$TAG/runs/$VM" && break
  sleep 30
done
if [ "$N_OK" = "$N_ALL" ]; then
  touch DONE && gsutil cp DONE "gs://$BUCKET/codes/$TAG/runs/$VM/DONE"
else
  echo "$N_OK of $N_ALL runs evaluated" > FAILED && gsutil cp FAILED "gs://$BUCKET/codes/$TAG/runs/$VM/FAILED"
fi
shutdown -h now
'''


def sh(cmd, check=True):
  r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
  if check and r.returncode != 0:
    raise SystemExit(f'command failed ({r.returncode}): {cmd}\n{r.stderr}')
  return r.stdout


def now():
  return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + 'Z'


class Fleet:
  def __init__(self, tag, dest=None):
    self.tag = tag
    self.prefix = f'cv-{tag}'
    self.gs = f'gs://{BUCKET}/codes/{tag}'
    self.dest = dest or os.path.join(ROOT, 'runs_codes', tag)
    os.makedirs(self.dest, exist_ok=True)
    self.plan_path = os.path.join(self.dest, 'plan.json')

  def log(self, msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(os.path.join(self.dest, 'fleet.log'), 'a') as f:
      f.write(line + '\n')

  def plan(self):
    return json.load(open(self.plan_path))

  def instances(self):
    out = sh([GCLOUD, 'compute', 'instances', 'list', '--project', PROJECT, '--filter',
              f'name~^{self.prefix}-[0-9]', '--format',
              'value(name,zone.basename(),status,scheduling.provisioningModel)'])
    return {l.split()[0]: l.split()[1:] for l in out.strip().splitlines() if l.strip()}

  def markers(self):
    out = sh(f'{GSUTIL} ls {self.gs}/runs/*/DONE {self.gs}/runs/*/FAILED 2>/dev/null', check=False)
    done, failed = set(), set()
    for u in out.split():
      (done if u.endswith('/DONE') else failed).add(u.split('/')[-2])
    return done, failed

  def create(self, name, zones, spot, meta):
    """Try the zones in turn; returns (zone or None, message)."""
    path = os.path.join(self.dest, 'startup.sh')
    with open(path, 'w') as f:
      f.write(STARTUP)
    err = ''
    for zone in zones:
      cmd = [GCLOUD, 'compute', 'instances', 'create', name, '--project', PROJECT, '--zone', zone,
             '--machine-type', meta['machine'], '--image-family', IMAGE_FAMILY,
             '--image-project', IMAGE_PROJECT, '--boot-disk-size', f'{DISK_GB}GB',
             '--boot-disk-type', 'pd-balanced', '--maintenance-policy', 'TERMINATE',
             '--scopes', 'storage-rw',
             '--metadata', (f'bucket={BUCKET},tag={self.tag},max_concurrent={meta["max_concurrent"]},'
                            f'deadman={meta["deadman"]},install-nvidia-driver=True'),
             '--metadata-from-file',
             f'startup-script={path},probes={meta["probes_file"]},train_args={meta["train_args_file"]}']
      if spot:
        cmd += ['--provisioning-model', 'SPOT', '--instance-termination-action', 'STOP']
      r = subprocess.run(cmd, capture_output=True, text=True)
      if r.returncode == 0:
        return zone, 'OK'
      if 'already exists' in r.stderr:
        return zone, 'exists'
      e = r.stderr.strip().replace('\n', ' ')
      err = err or e[-260:]          # keep the first zone's reason
    return None, f'FAILED everywhere: {err}'


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('cmd', choices=['package', 'plan', 'launch', 'watch', 'status', 'collect', 'cleanup'])
  ap.add_argument('--tag', required=True)
  ap.add_argument('--matrix', default=None)
  ap.add_argument('--per_vm', type=int, default=5, help='runs per VM (all run side by side)')
  ap.add_argument('--first', type=int, default=0, help='plan: index of the first VM of this plan')
  ap.add_argument('--machine', default='g2-standard-8')
  ap.add_argument('--on_demand', action='store_true', help='launch: standard instead of spot VMs')
  ap.add_argument('--deadman', type=int, default=2100, help='minutes until a VM powers itself off')
  ap.add_argument('--probes', default=PROBES)
  ap.add_argument('--train_args', default='--encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5')
  ap.add_argument('--only', default=None, help='comma list of VM names')
  ap.add_argument('--budget', type=float, default=None, help='watch: USD at which the fleet is deleted')
  ap.add_argument('--poll', type=int, default=150)
  ap.add_argument('--mirror', action='store_true',
                  help='watch: copy fleet.log and cost.json to gs://<bucket>/codes/_controller/state/<tag>/ '
                       'after every pass (for a watcher that runs on a cloud VM)')
  ap.add_argument('--dest', default=None)
  ap.add_argument('--with_ckpts', action='store_true', help='collect: also the intermediate checkpoints')
  ap.add_argument('--logs_only', action='store_true', help='collect: no network states at all (interim look)')
  args = ap.parse_args()
  F = Fleet(args.tag, args.dest)
  only = set(args.only.split(',')) if args.only else None

  if args.cmd == 'package':
    tar = os.path.join(F.dest, 'code.tar.gz')
    sh(f'cd {ROOT} && tar czf {tar} --exclude=__pycache__ code scripts/launch_matrix.py scripts/codes')
    sh(f'{GSUTIL} -q cp {tar} {F.gs}/code.tar.gz')
    print('code ->', f'{F.gs}/code.tar.gz')

  elif args.cmd == 'plan':
    runs = json.load(open(args.matrix))
    n_vm = (len(runs) + args.per_vm - 1) // args.per_vm
    plan = F.plan() if os.path.exists(F.plan_path) else {}
    adir = os.path.join(F.dest, 'assign')
    os.makedirs(adir, exist_ok=True)
    for i in range(n_vm):
      vm = f'{F.prefix}-{args.first + i:02d}'
      if vm in plan:
        raise SystemExit(f'{vm} is already planned; use --first')
      part = runs[i * args.per_vm:(i + 1) * args.per_vm]
      plan[vm] = dict(runs=[r['name'] for r in part], max_concurrent=args.per_vm, machine=args.machine,
                      deadman=args.deadman)
      json.dump(part, open(os.path.join(adir, f'{vm}.json'), 'w'), indent=1)
    json.dump(plan, open(F.plan_path, 'w'), indent=1)
    sh(f'{GSUTIL} -q -m rsync -r {adir} {F.gs}/assign')
    for k in ('probes', 'train_args'):
      with open(os.path.join(F.dest, f'{k}.txt'), 'w') as f:
        f.write(getattr(args, k))
    print(f'{len(runs)} runs over {n_vm} VMs ({args.per_vm} per VM) -> {F.plan_path}')

  elif args.cmd == 'launch':
    plan, have = F.plan(), F.instances()
    todo = [vm for vm in sorted(plan) if vm not in have and (only is None or vm in only)]
    F.log(f'launch: {len(have)} exist, creating {len(todo)} ({"on demand" if args.on_demand else "spot"})')

    def one(i_vm):
      i, vm = i_vm
      meta = dict(plan[vm], probes_file=os.path.join(F.dest, 'probes.txt'),
                  train_args_file=os.path.join(F.dest, 'train_args.txt'))
      k = i % N_FIRST
      zones = ZONES[k:N_FIRST] + ZONES[:k] + ZONES[N_FIRST:]
      return (vm,) + F.create(vm, zones, not args.on_demand, meta)
    with ThreadPoolExecutor(8) as ex:
      for vm, zone, msg in ex.map(one, list(enumerate(todo))):
        F.log(f'  {vm} {zone} {msg}')

  elif args.cmd == 'status':
    have = F.instances()
    done, failed = F.markers()
    plan = F.plan() if os.path.exists(F.plan_path) else {}
    for vm in sorted(set(plan) | set(have)):
      st = have.get(vm, ['-', 'absent', '-'])
      print(vm, *st, 'DONE' if vm in done else ('FAILED' if vm in failed else ''))
    cost = json.load(open(os.path.join(F.dest, 'cost.json'))) if os.path.exists(os.path.join(F.dest, 'cost.json')) else {}
    print(f'{len(done)} of {len(plan)} VMs done; estimated cost so far USD {cost.get("usd", 0):.2f}')

  elif args.cmd == 'watch':
    cost_path = os.path.join(F.dest, 'cost.json')
    cost = json.load(open(cost_path)) if os.path.exists(cost_path) else dict(usd=0.0, vm_hours=0.0)
    down = {}                      # vm -> consecutive passes it could not be started
    kept = set()                   # finished VMs kept because the bucket copy looked incomplete
    last = time.time()
    F.log(f'watch start: budget {args.budget}, cost so far {cost["usd"]:.2f}')
    while True:
      try:
        plan, have = F.plan(), F.instances()
        done, failed = F.markers()
      except SystemExit as e:
        F.log(f'watch: listing failed ({str(e)[:120]}); retrying')
        time.sleep(args.poll)
        continue
      dt = (time.time() - last) / 3600.0
      last = time.time()
      for vm, (zone, status, model) in have.items():
        if status in ('RUNNING', 'PROVISIONING', 'STAGING', 'STOPPING'):
          cost['usd'] += dt * PRICE.get(model, PRICE['STANDARD'])
          cost['vm_hours'] += dt
        cost['usd'] += dt * DISK_GB * DISK_PRICE
      cost['time'] = now()
      json.dump(cost, open(cost_path, 'w'))
      if args.mirror:
        subprocess.run(f'{GSUTIL} -q cp {cost_path} {os.path.join(F.dest, "fleet.log")} '
                       f'gs://{BUCKET}/codes/_controller/state/{F.tag}/', shell=True, capture_output=True)
      if args.budget is not None and cost['usd'] >= args.budget:
        F.log(f'BUDGET REACHED (USD {cost["usd"]:.2f} >= {args.budget}): deleting the fleet')
        for vm, (zone, *_) in have.items():
          subprocess.run([GCLOUD, 'compute', 'instances', 'delete', vm, '--zone', zone, '--project', PROJECT,
                          '--quiet'], capture_output=True)
        break
      for vm, (zone, status, model) in sorted(have.items()):
        if vm in done or vm in failed:
          if status in ('TERMINATED', 'RUNNING'):
            # The disk is the only other copy: delete it only when the bucket holds the final
            # network and the test of every run of this VM.
            have_gs = sh(f'{GSUTIL} ls "{F.gs}/runs/{vm}/*/ckpt_final.pt" "{F.gs}/runs/{vm}/*/eval/intact.json" '
                         f'2>/dev/null', check=False).split()
            names = {u.split('/')[-2] for u in have_gs if u.endswith('ckpt_final.pt')} & \
                    {u.split('/')[-3] for u in have_gs if u.endswith('intact.json')}
            lacking = [r for r in plan.get(vm, {}).get('runs', []) if r not in names]
            if vm in done and lacking:
              if vm not in kept:
                F.log(f'{vm} done but the bucket lacks {lacking}: VM kept (stopped) for inspection')
                kept.add(vm)
              continue
            F.log(f'{vm} {"done" if vm in done else "FAILED"}: deleting')
            subprocess.run([GCLOUD, 'compute', 'instances', 'delete', vm, '--zone', zone, '--project', PROJECT,
                            '--quiet'], capture_output=True)
          continue
        if status == 'TERMINATED':       # preempted (or dead-man): start it again
          r = subprocess.run([GCLOUD, 'compute', 'instances', 'start', vm, '--zone', zone, '--project', PROJECT],
                             capture_output=True, text=True)
          if r.returncode == 0:
            F.log(f'{vm} was stopped ({model}): started again in {zone}')
            down.pop(vm, None)
          else:
            down[vm] = down.get(vm, 0) + 1
            F.log(f'{vm} could not be started in {zone} (pass {down[vm]}): {r.stderr.strip()[:120]}')
            if down[vm] >= 6:            # no capacity there: move it (state comes from the bucket)
              subprocess.run([GCLOUD, 'compute', 'instances', 'delete', vm, '--zone', zone, '--project', PROJECT,
                              '--quiet'], capture_output=True)
              meta = dict(plan[vm], probes_file=os.path.join(F.dest, 'probes.txt'),
                          train_args_file=os.path.join(F.dest, 'train_args.txt'))
              zones = [z for z in ZONES if z != zone]
              k = hash(vm) % len(zones)
              z2, msg = F.create(vm, zones[k:] + zones[:k], model == 'SPOT', meta)
              F.log(f'{vm} recreated in {z2}: {msg}')
              down.pop(vm, None)
      # A planned VM that is not listed and not finished (its recreation failed everywhere): try again.
      for vm in sorted(plan):
        if vm not in have and vm not in done and vm not in failed:
          meta = dict(plan[vm], probes_file=os.path.join(F.dest, 'probes.txt'),
                      train_args_file=os.path.join(F.dest, 'train_args.txt'))
          k = hash(vm) % len(ZONES)
          z2, msg = F.create(vm, ZONES[k:] + ZONES[:k], True, meta)
          F.log(f'{vm} was missing: created in {z2}: {msg}')
      left = [vm for vm in plan if vm not in done and vm not in failed]
      if not left:
        F.log(f'all {len(plan)} VMs finished ({len(failed)} failed); estimated cost USD {cost["usd"]:.2f}')
        break
      time.sleep(args.poll)

  elif args.cmd == 'collect':
    excl = '' if args.with_ckpts else '-x ".*ckpt_[0-9].*\\.pt$|.*resume\\.pt$" '
    if args.logs_only:
      excl = '-x ".*\\.pt$" '
    tmp = os.path.join(F.dest, '_bucket')
    os.makedirs(tmp, exist_ok=True)
    rc = subprocess.run(f'{GSUTIL} -q -m rsync -r {excl}{F.gs}/runs {tmp}', shell=True).returncode
    if rc != 0:
      raise SystemExit(f'collect {F.tag}: gsutil failed (exit {rc}); nothing copied')
    n = 0
    for vm in sorted(os.listdir(tmp)):
      vdir = os.path.join(tmp, vm)
      if not os.path.isdir(vdir):
        continue
      for run in sorted(os.listdir(vdir)):
        src = os.path.join(vdir, run)
        if os.path.isdir(src) and os.path.exists(os.path.join(src, 'config.json')):
          sh(f'mkdir -p {F.dest}/{run} && rsync -a -u {src}/ {F.dest}/{run}/')
          n += 1
    print(f'collected {n} run directories into {F.dest}')

  elif args.cmd == 'cleanup':
    for vm, (zone, *_) in sorted(F.instances().items()):
      if only is not None and vm not in only:
        continue
      print('deleting', vm, zone, flush=True)
      subprocess.run([GCLOUD, 'compute', 'instances', 'delete', vm, '--zone', zone, '--project', PROJECT,
                      '--quiet'], capture_output=True)


if __name__ == '__main__':
  main()
