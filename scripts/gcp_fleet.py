"""Run a slice of the experiment matrix on a fleet of GCP GPU VMs.

Each VM boots from a Deep Learning VM image (CUDA + PyTorch preinstalled),
downloads the code tarball and its assigned matrix from GCS, runs
scripts/launch_matrix.py on it (training + checkpoint probes + final
evaluation), rsyncs its runs/ directory to GCS every 10 minutes and shuts
itself down when the scheduler finishes. Results are pulled back into the
local runs/ tree with `collect`, where analyze.py treats them like local runs.

  python scripts/gcp_fleet.py package                      # code -> GCS
  python scripts/gcp_fleet.py plan --wave 2 --vms 12       # split matrix into per-VM JSON files
  python scripts/gcp_fleet.py launch --vms 12 [--zone ..]  # create VMs
  python scripts/gcp_fleet.py status                       # VM states + progress from GCS
  python scripts/gcp_fleet.py collect                      # GCS -> runs/ (idempotent)
  python scripts/gcp_fleet.py cleanup                      # delete fleet VMs

Cost guard: `launch` refuses to create more than --max_vms and prints the
estimated hourly cost before creating anything.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GCLOUD = os.path.expanduser('~/google-cloud-sdk/bin/gcloud')
GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
PY = '/home/ec2-user/banino/.venv/bin/python'
PRICES = {'g2-standard-8': 0.85, 'g2-standard-4': 0.71, 'a2-highgpu-1g': 3.67,
          'n1-standard-8+t4': 0.73}
NAME_PREFIX = 'popcode'

STARTUP = r'''#!/bin/bash
# popcode fleet worker startup script
set -x
exec > >(tee -a /var/log/popcode-startup.log) 2>&1
BUCKET="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)"
ASSIGN="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/assign)"
RUNS_PREFIX="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/runs_prefix)"
TRAIN_ARGS="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/train_args)"
MAXC="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/max_concurrent)"
VM="$(hostname)"
mkdir -p /work && cd /work
for i in 1 2 3 4 5 6; do nvidia-smi && break; sleep 20; done
gsutil cp "gs://$BUCKET/popcode/code.tar.gz" code.tar.gz && tar xzf code.tar.gz
gsutil cp "gs://$BUCKET/popcode/$ASSIGN" assign.json
PYBIN=$(ls /opt/conda/bin/python 2>/dev/null || which python3)
$PYBIN -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
$PYBIN -m pip install -q scipy matplotlib 2>/dev/null
# Resume any previous state for this VM (preempted / restarted).
mkdir -p runs && gsutil -m rsync -r "gs://$BUCKET/$RUNS_PREFIX/$VM" runs 2>/dev/null || true
( while true; do sleep 600; gsutil -m rsync -r runs "gs://$BUCKET/$RUNS_PREFIX/$VM" >/dev/null 2>&1; done ) &
SYNC=$!
sed -i "s#^PY = .*#PY = '$PYBIN'#" scripts/launch_matrix.py
$PYBIN scripts/launch_matrix.py --matrix_json assign.json --runs_dir /work/runs --max_concurrent "$MAXC" --train_args "$TRAIN_ARGS" 2>&1 | tee -a runs/scheduler.stdout
kill $SYNC
gsutil -m rsync -r runs "gs://$BUCKET/$RUNS_PREFIX/$VM"
touch DONE && gsutil cp DONE "gs://$BUCKET/$RUNS_PREFIX/$VM/DONE"
sudo shutdown -h now
'''


def sh(cmd, check=True, capture=True):
  r = subprocess.run(cmd, shell=isinstance(cmd, str), check=False, capture_output=capture, text=True)
  if check and r.returncode != 0:
    raise SystemExit(f'command failed ({r.returncode}): {cmd}\n{r.stderr if capture else ""}')
  return r.stdout if capture else ''


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', choices=['package', 'plan', 'launch', 'status', 'collect', 'cleanup'])
  ap.add_argument('--project', default='connectome-ai-research')
  ap.add_argument('--bucket', default='connectome-ai-research-popcode')
  ap.add_argument('--zone', default='us-central1-a')
  ap.add_argument('--zones', default=None,
                  help='comma list for round-robin placement (capacity stockouts), e.g. us-central1-a,us-central1-b,us-central1-c')
  ap.add_argument('--machine', default='g2-standard-8')
  ap.add_argument('--gpu', default='nvidia-l4', help='accelerator type (ignored for g2/a2 machines)')
  ap.add_argument('--image_family', default='pytorch-2-9-cu129-ubuntu-2204-nvidia-580')
  ap.add_argument('--image_project', default='deeplearning-platform-release')
  ap.add_argument('--vms', type=int, default=12)
  ap.add_argument('--max_vms', type=int, default=24)
  ap.add_argument('--max_concurrent', type=int, default=4, help='runs per VM')
  ap.add_argument('--spot', action='store_true')
  ap.add_argument('--wave', default='2')
  ap.add_argument('--frames', type=int, default=300_000_000)
  ap.add_argument('--only', default=None)
  ap.add_argument('--train_args',
                  default='--encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5')
  ap.add_argument('--matrix_json', default=None, help='use this matrix instead of --wave')
  ap.add_argument('--tag', default='main',
                  help='namespace for this fleet (VM names popcode-<tag>-NN, bucket paths '
                       'popcode/assign_<tag>/, runs_<tag>/); use one tag per arena/config')
  ap.add_argument('--extra_args', default='',
                  help='appended to EVERY run of the plan, e.g. "--arena_half 2.0 --goal_radius 0.25 --episode_len 1800"')
  ap.add_argument('--dest', default=None, help='collect: local runs dir (default runs_<tag>, or runs for tag main)')
  args = ap.parse_args()
  tag = args.tag
  prefix = f'popcode-{tag}'
  runs_prefix = 'runs' if tag == 'main' else f'runs_{tag}'
  dest = args.dest or os.path.join(ROOT, runs_prefix)
  env = dict(os.environ, CLOUDSDK_CORE_PROJECT=args.project)
  g = lambda c: sh([GCLOUD] + c + ['--project', args.project])

  if args.cmd == 'package':
    sh(f'cd {ROOT} && tar czf /tmp/popcode_code.tar.gz code scripts tests README.md')
    sh(f'{GSUTIL} ls gs://{args.bucket} >/dev/null 2>&1 || {GSUTIL} mb -p {args.project} -l us-central1 gs://{args.bucket}')
    sh(f'{GSUTIL} cp /tmp/popcode_code.tar.gz gs://{args.bucket}/popcode/code.tar.gz')
    print(f'code uploaded to gs://{args.bucket}/popcode/code.tar.gz')
  elif args.cmd == 'plan':
    from scripts.launch_matrix import matrix
    if args.matrix_json:
      runs = json.load(open(args.matrix_json))
    else:
      wave = args.wave if args.wave == 'all' else int(args.wave)
      runs = matrix(wave, args.frames, args.frames)
    if args.only:
      import re
      runs = [r for r in runs if re.search(args.only, r['name'])]
    # Skip runs already finished locally; append per-fleet extra args (arena etc.).
    runs = [r for r in runs if not os.path.exists(os.path.join(dest, r['name'], 'ckpt_final.pt'))]
    for r in runs:
      r['extra'] = r.get('extra', []) + args.extra_args.split()
    adir = f'/tmp/popcode_assign_{tag}'
    sh(f'rm -rf {adir} && mkdir -p {adir}')
    shards = [[] for _ in range(args.vms)]
    for i, r in enumerate(runs):
      shards[i % args.vms].append(r)
    for i, s in enumerate(shards):
      json.dump(s, open(f'{adir}/vm{i:02d}.json', 'w'), indent=1)
    sh(f'{GSUTIL} -m cp {adir}/*.json gs://{args.bucket}/popcode/assign_{tag}/')
    print(f'{len(runs)} runs over {args.vms} VMs ({len(shards[0])} per VM); assignments uploaded')
    for i, s in enumerate(shards):
      print(f'  vm{i:02d}: {[r["name"] for r in s]}')
  elif args.cmd == 'launch':
    if args.vms > args.max_vms:
      raise SystemExit(f'refusing to launch {args.vms} VMs (> --max_vms {args.max_vms})')
    price = PRICES.get(args.machine, 1.0) * (0.35 if args.spot else 1.0)
    print(f'launching {args.vms} x {args.machine} in {args.zone}: ~${price * args.vms:.2f}/h '
          f'(~${price * args.vms * 12:.0f} for 12 h)')
    with open('/tmp/popcode_startup.sh', 'w') as f:
      f.write(STARTUP)
    zones = args.zones.split(',') if args.zones else [args.zone]
    existing = set(g(['compute', 'instances', 'list', '--filter', f'name~^{prefix}-',
                      '--format', 'value(name)']).split())
    for i in range(args.vms):
      name = f'{prefix}-{i:02d}'
      if name in existing:        # VM names are unique per zone only: never recreate
        print(name, 'exists, skipping')
        continue
      zone = zones[i % len(zones)]
      cmd = [GCLOUD, 'compute', 'instances', 'create', name, '--project', args.project,
             '--zone', zone, '--machine-type', args.machine,
             '--image-family', args.image_family, '--image-project', args.image_project,
             '--boot-disk-size', '100GB', '--maintenance-policy', 'TERMINATE',
             '--scopes', 'storage-rw,compute-rw',
             '--metadata', f'bucket={args.bucket},assign=assign_{tag}/vm{i:02d}.json,'
                           f'runs_prefix={runs_prefix},train_args={args.train_args},'
                           f'max_concurrent={args.max_concurrent},install-nvidia-driver=True',
             '--metadata-from-file', 'startup-script=/tmp/popcode_startup.sh']
      if not args.machine.startswith(('g2-', 'a2-', 'a3-')):
        cmd += ['--accelerator', f'type={args.gpu},count=1']
      if args.spot:
        cmd += ['--provisioning-model', 'SPOT', '--instance-termination-action', 'STOP']
      r = subprocess.run(cmd, capture_output=True, text=True)
      if r.returncode != 0 and len(zones) > 1 and 'already exists' not in r.stderr:   # stockout: try other zones
        for z in zones:
          if z == zone:
            continue
          cmd[cmd.index('--zone') + 1] = z
          r = subprocess.run(cmd, capture_output=True, text=True)
          if r.returncode == 0:
            zone = z
            break
      print(name, zone, 'OK' if r.returncode == 0 else f'FAILED: {r.stderr.strip()[:300]}')
  elif args.cmd == 'status':
    print(g(['compute', 'instances', 'list', '--filter', f'name~^{prefix}-',
             '--format', 'table(name,zone.basename(),machineType.basename(),status)']))
    out = sh(f'{GSUTIL} ls gs://{args.bucket}/{runs_prefix}/ 2>/dev/null', check=False)
    for vm in [x.strip('/').split('/')[-1] for x in out.split() if x.endswith('/')]:
      done = sh(f'{GSUTIL} ls gs://{args.bucket}/{runs_prefix}/{vm}/DONE 2>/dev/null', check=False).strip()
      logs = sh(f'{GSUTIL} cat gs://{args.bucket}/{runs_prefix}/{vm}/scheduler.log 2>/dev/null', check=False).strip().splitlines()
      print(f'{vm}: {"DONE" if done else "running"} | {logs[-1][:110] if logs else "(no log yet)"}')
  elif args.cmd == 'collect':
    tmp = f'/tmp/popcode_collect_{tag}'
    os.makedirs(tmp, exist_ok=True)
    sh(f'{GSUTIL} -m rsync -r -x ".*resume\\.pt.*" gs://{args.bucket}/{runs_prefix} {tmp}', capture=False)
    n = 0
    os.makedirs(dest, exist_ok=True)
    for vm in sorted(os.listdir(tmp)):
      vdir = os.path.join(tmp, vm)
      if not os.path.isdir(vdir):
        continue
      for run in sorted(os.listdir(vdir)):
        src = os.path.join(vdir, run)
        if not os.path.isdir(src):
          continue
        dst = os.path.join(dest, run)
        sh(f'mkdir -p {dst} && rsync -a {src}/ {dst}/')
        n += 1
    print(f'collected {n} run directories into {dest}')
  elif args.cmd == 'cleanup':
    out = g(['compute', 'instances', 'list', '--filter', f'name~^{prefix}-',
             '--format', 'value(name,zone.basename())'])
    for line in out.strip().splitlines():
      name, zone = line.split()
      print('deleting', name, zone)
      subprocess.run([GCLOUD, 'compute', 'instances', 'delete', name, '--zone', zone,
                      '--project', args.project, '--quiet'])


if __name__ == '__main__':
  main()
