"""GCP workers for the room-change experiment: one training run per L4 VM.

A VM boots, waits for its job script gs://<bucket>/room/jobs/<vm>.sh and runs
it, so the fleet can be provisioned (stock-outs take a while) before the jobs
are final. The job script (written by `jobs`) pulls the code and the run's
starting state, trains with --resume in a retry loop to 1.25 B frames, copies
the run directory to gs://<bucket>/runs_room/<run>/ every 5 minutes and writes
DONE; it then keeps training the same run (larger frame budget) until the VM
is deleted with `cleanup`. A dead-man power-off bounds the cost of a
forgotten VM. (The VMs of the first launch ran an earlier template that shut
down at 1.25 B; runs_room/extend.py switched them to continue.)

  python -m scripts.room.fleet launch --first 0 --count 20
  python -m scripts.room.fleet jobs --matrix runs_room/matrix.json
  python -m scripts.room.fleet status
  python -m scripts.room.fleet collect
  python -m scripts.room.fleet cleanup
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GCLOUD = os.path.expanduser('~/google-cloud-sdk/bin/gcloud')
GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
PROJECT = 'connectome-ai-research'
BUCKET = 'connectome-ai-research-popcode'
PREFIX = 'room'
ZONES = ['us-central1-a', 'us-central1-b', 'us-central1-c', 'us-east4-a', 'us-west4-a',
         'us-east1-b', 'us-west1-b', 'us-east4-c', 'us-east1-c', 'us-west1-a', 'us-west4-c',
         'us-east1-d', 'us-west1-c', 'europe-west4-a', 'europe-west4-b', 'europe-west4-c',
         'europe-west1-b', 'europe-west1-c', 'europe-west2-a', 'europe-west2-b',
         'europe-west3-b', 'asia-southeast1-a', 'asia-southeast1-b', 'asia-southeast1-c',
         'asia-northeast1-a', 'asia-northeast1-c', 'asia-east1-a', 'asia-east1-c',
         'northamerica-northeast1-b', 'southamerica-east1-c', 'asia-south1-a', 'asia-south1-b',
         'australia-southeast1-a', 'europe-west6-b', 'me-west1-b', 'us-south1-a', 'us-south1-b']

STARTUP = r'''#!/bin/bash
exec > >(tee -a /var/log/room-startup.log) 2>&1
set -x
BUCKET=__BUCKET__
VM="$(hostname)"
shutdown -h +__DEADMAN__
mkdir -p /work && cd /work
for i in $(seq 1 45); do nvidia-smi && break; sleep 20; done
until gsutil -q stat "gs://$BUCKET/room/jobs/$VM.sh"; do sleep 20; done
gsutil cp "gs://$BUCKET/room/jobs/$VM.sh" job.sh
bash job.sh
'''

JOB = r'''#!/bin/bash
# room-change job for __VM__: __RUN__
# Phase 1 trains to 1.25 B and writes the DONE marker; phase 2 keeps training the same run until the
# VM is deleted from outside. Safe to re-run after a reboot: the bucket copy is the only state that counts.
set -x
BUCKET=__BUCKET__
RUN=__RUN__
cd /work
systemd-run --on-active=__DEADMAN__m systemctl poweroff 2>/dev/null || true      # dead-man
gsutil cp "gs://$BUCKET/room/code.tar.gz" code.tar.gz && tar xzf code.tar.gz
PYBIN=$(ls /opt/conda/bin/python 2>/dev/null || which python3)
$PYBIN -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" || exit 1
rm -rf runs/$RUN && mkdir -p runs/$RUN
gsutil -m rsync -r "gs://$BUCKET/room/init/$RUN" runs/$RUN || exit 1
if gsutil -q stat "gs://$BUCKET/runs_room/$RUN/resume.pt"; then
  gsutil -m rsync -r "gs://$BUCKET/runs_room/$RUN" runs/$RUN || exit 1
fi
[ -f runs/$RUN/resume.pt ] || exit 1
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
( while true; do sleep 300; gsutil -m rsync -r runs/$RUN "gs://$BUCKET/runs_room/$RUN" >/dev/null 2>&1; done ) &
if ! gsutil -q stat "gs://$BUCKET/runs_room/$RUN/DONE"; then
  for attempt in 1 2 3 4 5 6; do
    $PYBIN -m code.train __ARGS__ --out runs/$RUN --resume >> runs/$RUN/train.log 2>&1
    [ -f runs/$RUN/ckpt_final.pt ] && break
    echo "attempt $attempt failed" >> runs/$RUN/train.log
    sleep 20
  done
  gsutil -m rsync -r runs/$RUN "gs://$BUCKET/runs_room/$RUN"
  if [ -f runs/$RUN/ckpt_final.pt ]; then touch DONE; else touch FAILED; fi
  for i in 1 2 3; do for f in DONE FAILED; do [ -f $f ] && gsutil cp $f "gs://$BUCKET/runs_room/$RUN/$f"; done; sleep 2; done
  [ -f FAILED ] && exit 1
fi
for attempt in 1 2 3 4 5; do
  $PYBIN -m code.train __ARGS2__ --out runs/$RUN --resume >> runs/$RUN/train.log 2>&1
  echo "extension attempt $attempt ended" >> runs/$RUN/train.log
  sleep 20
done
'''


def sh(cmd, check=True):
  r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
  if check and r.returncode != 0:
    raise SystemExit(f'command failed ({r.returncode}): {cmd}\n{r.stderr}')
  return r.stdout


def instances():
  out = sh([GCLOUD, 'compute', 'instances', 'list', '--project', PROJECT, '--filter',
            f'name~^{PREFIX}-[0-9]', '--format', 'value(name,zone.basename(),status,machineType.basename())'])
  return {l.split()[0]: l.split()[1:] for l in out.strip().splitlines() if l.strip()}


def create(name, zones, machine, gpu, deadman, startup_path):
  for zone in zones:
    cmd = [GCLOUD, 'compute', 'instances', 'create', name, '--project', PROJECT, '--zone', zone,
           '--machine-type', machine, '--image-family', 'pytorch-2-9-cu129-ubuntu-2204-nvidia-580',
           '--image-project', 'deeplearning-platform-release', '--boot-disk-size', '100GB',
           '--maintenance-policy', 'TERMINATE', '--scopes', 'storage-rw',
           '--metadata', 'install-nvidia-driver=True',
           '--metadata-from-file', f'startup-script={startup_path}']
    if gpu:
      cmd += ['--accelerator', f'type={gpu},count=1']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 or 'already exists' in r.stderr:
      return name, zone, 'OK'
    err = r.stderr.strip().replace('\n', ' ')[:160]
    if 'GPUS_ALL_REGIONS' in r.stderr:
      return name, zone, f'GLOBAL QUOTA: {err}'
  return name, None, f'FAILED everywhere: {err}'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', choices=['launch', 'jobs', 'status', 'collect', 'cleanup'])
  ap.add_argument('--first', type=int, default=0)
  ap.add_argument('--count', type=int, default=20)
  ap.add_argument('--machine', default='g2-standard-8')
  ap.add_argument('--gpu', default='', help="e.g. nvidia-tesla-t4 with --machine n1-standard-8")
  ap.add_argument('--deadman', type=int, default=720, help='minutes until a VM shuts itself down')
  ap.add_argument('--matrix', default=os.path.join(ROOT, 'runs_room', 'matrix.json'))
  ap.add_argument('--dest', default=os.path.join(ROOT, 'runs_room'))
  ap.add_argument('--only', default=None, help='cleanup/jobs: comma list of VM names')
  args = ap.parse_args()

  if args.cmd == 'launch':
    path = '/tmp/room_startup.sh'
    with open(path, 'w') as f:
      f.write(STARTUP.replace('__BUCKET__', BUCKET).replace('__DEADMAN__', str(args.deadman)))
    have = instances()
    names = [f'{PREFIX}-{i:02d}' for i in range(args.first, args.first + args.count)]
    todo = [n for n in names if n not in have]
    print(f'{len(have)} exist; creating {len(todo)}', flush=True)
    with ThreadPoolExecutor(8) as ex:
      futs = [ex.submit(create, n, ZONES[i % 13:13] + ZONES[:i % 13] + ZONES[13:], args.machine,
                        args.gpu, args.deadman, path) for i, n in enumerate(todo)]
      for f in futs:
        print(*f.result(), flush=True)
  elif args.cmd == 'jobs':
    runs = json.load(open(args.matrix))
    for r in runs:
      if args.only and r['vm'] not in args.only.split(','):
        continue
      job = (JOB.replace('__BUCKET__', BUCKET).replace('__RUN__', r['name']).replace('__VM__', r['vm'])
             .replace('__ARGS2__', r['args'].replace('--frames 1250000000', '--frames 2000000000'))
             .replace('__ARGS__', r['args']).replace('__DEADMAN__', str(r.get('deadman', 720))))
      p = f'/tmp/room_job_{r["vm"]}.sh'
      with open(p, 'w') as f:
        f.write(job)
      sh(f'{GSUTIL} cp {p} gs://{BUCKET}/room/jobs/{r["vm"]}.sh')
      print('job', r['vm'], r['name'])
  elif args.cmd == 'status':
    have = instances()
    for n in sorted(have):
      print(n, *have[n])
    out = sh(f'{GSUTIL} ls gs://{BUCKET}/runs_room/*/DONE gs://{BUCKET}/runs_room/*/FAILED 2>/dev/null', check=False)
    print('markers:', len(out.split()), out.replace(f'gs://{BUCKET}/runs_room/', '').split())
  elif args.cmd == 'collect':
    os.makedirs(args.dest, exist_ok=True)
    subprocess.run(f'{GSUTIL} -m rsync -r gs://{BUCKET}/runs_room {args.dest}', shell=True)
  elif args.cmd == 'cleanup':
    have = instances()
    for n, (zone, *_) in sorted(have.items()):
      if args.only and n not in args.only.split(','):
        continue
      print('deleting', n, zone, flush=True)
      subprocess.run([GCLOUD, 'compute', 'instances', 'delete', n, '--zone', zone,
                      '--project', PROJECT, '--quiet'], capture_output=True)


if __name__ == '__main__':
  main()
