"""From-scratch reference runs for the room-change experiment (one run per L4 VM).

  <cond>scr<ENV>_lstm_s<seed>   ENV = one changed layout: the agent only ever lives in that room
                                (rooms '<L>,<L>+R': goal fixed or moved, half of the episodes each);
                                ENV = R: the open room with a moving goal; one grid and one PGC agent
                                per room, seeds rotating over 1-8
  <cond>scrmix_lstm_s<seed>     the mixture the fine-tuned agents train on (--rooms changed), seeds 1-2

Same hyperparameters as the original agents. Each VM also tests its own checkpoints (all 28 rooms, the
evaluation of scripts/room/eval_rooms.py) at the frame marks below and copies the results to
gs://<bucket>/runs_scratch_eval/; the run directory goes to gs://<bucket>/runs_scratch/<run>/.

  python -m scripts.room.scratch prepare      # matrix, code tarball and job scripts -> bucket
  python -m scripts.room.fleet launch --first 20 --count 26
  python -m scripts.room.scratch collect      # evaluations + training logs -> results_room/, runs_scratch/
"""
import argparse
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
BUCKET = 'connectome-ai-research-popcode'
FRAMES = 1_500_000_000
MARKS = '250 400 450 475 500 750 1000 1250 1500'
ENVS = ['D1', 'D2', 'D3', 'D4', 'O1', 'O2', 'O3', 'DO1', 'DO2', 'DO3', 'R']
BASE_ARGS = ('--cond {cond} --seed {seed} --frames {frames} --policy lstm --vision raycast '
             '--encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5 --rooms {rooms}')

JOB = r'''#!/bin/bash
# from-scratch job for __VM__: __RUN__  (re-runnable after a reboot: the bucket copy is the state)
set -x
BUCKET=__BUCKET__
RUN=__RUN__
cd /work
systemd-run --on-active=__DEADMAN__m systemctl poweroff 2>/dev/null || true      # dead-man
gsutil cp "gs://$BUCKET/room/code_scratch.tar.gz" code.tar.gz && tar xzf code.tar.gz
PYBIN=$(ls /opt/conda/bin/python 2>/dev/null || which python3)
$PYBIN -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" || exit 1
rm -rf runs/$RUN && mkdir -p runs/$RUN eval_out
if gsutil -q stat "gs://$BUCKET/runs_scratch/$RUN/resume.pt"; then
  gsutil -m rsync -r "gs://$BUCKET/runs_scratch/$RUN" runs/$RUN || exit 1
fi
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
( while true; do sleep 300; gsutil -m rsync -r runs/$RUN "gs://$BUCKET/runs_scratch/$RUN" >/dev/null 2>&1; done ) &
( while true; do
    for M in __MARKS__; do
      CK=$(ls runs/$RUN/ckpt_$(printf '%04d' $M)*.pt 2>/dev/null | head -1)
      if [ -n "$CK" ] && ! gsutil -q stat "gs://$BUCKET/runs_scratch_eval/$RUN@${M}M.json"; then
        $PYBIN -m scripts.room.eval_rooms --run runs/$RUN --ckpt $CK --tag ${M}M --out eval_out >> eval.log 2>&1
        gsutil -m rsync -r eval_out "gs://$BUCKET/runs_scratch_eval" >/dev/null 2>&1
        gsutil cp eval.log "gs://$BUCKET/runs_scratch/$RUN/eval.log" >/dev/null 2>&1
      fi
    done
    sleep 120
  done ) &
for attempt in 1 2 3 4 5 6; do
  $PYBIN -m code.train __ARGS__ --out runs/$RUN --resume >> runs/$RUN/train.log 2>&1
  [ -f runs/$RUN/ckpt_final.pt ] && break
  echo "attempt $attempt failed" >> runs/$RUN/train.log
  sleep 20
done
gsutil -m rsync -r runs/$RUN "gs://$BUCKET/runs_scratch/$RUN"
sleep 900      # let the last evaluation finish
gsutil -m rsync -r eval_out "gs://$BUCKET/runs_scratch_eval"
touch DONE && gsutil cp DONE "gs://$BUCKET/runs_scratch/$RUN/DONE"
systemctl poweroff
'''


def matrix():
  runs = []
  for i, env in enumerate(ENVS):
    seed = i % 8 + 1
    rooms = 'open+R' if env == 'R' else f'{env},{env}+R'
    for cond in ('gc100', 'pgc33'):
      runs.append(dict(name=f'{cond}scr{env}_lstm_s{seed}', cond=cond, seed=seed, env=env, rooms=rooms, deadman=720))
  for seed in (1, 2):
    for cond in ('gc100', 'pgc33'):
      runs.append(dict(name=f'{cond}scrmix_lstm_s{seed}', cond=cond, seed=seed, env='mix', rooms='changed', deadman=1560))
  for i, r in enumerate(runs):
    r['vm'] = f'room-{20 + i:02d}'
    r['args'] = BASE_ARGS.format(cond=r['cond'], seed=r['seed'], frames=FRAMES, rooms=r['rooms'])
  return runs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cmd', choices=['prepare', 'collect'])
  args = ap.parse_args()
  runs = matrix()
  os.makedirs(os.path.join(ROOT, 'runs_scratch'), exist_ok=True)
  if args.cmd == 'prepare':
    json.dump(runs, open(os.path.join(ROOT, 'runs_scratch', 'matrix.json'), 'w'), indent=1)
    tar = '/tmp/room_code_scratch.tar.gz'
    subprocess.run(f'cd {ROOT} && tar czf {tar} --exclude=__pycache__ code scripts/room', shell=True, check=True)
    subprocess.run(f'{GSUTIL} -q cp {tar} gs://{BUCKET}/room/code_scratch.tar.gz', shell=True, check=True)
    for r in runs:
      job = (JOB.replace('__BUCKET__', BUCKET).replace('__RUN__', r['name']).replace('__VM__', r['vm'])
             .replace('__ARGS__', r['args']).replace('__DEADMAN__', str(r['deadman'])).replace('__MARKS__', MARKS))
      p = f'/tmp/room_job_{r["vm"]}.sh'
      open(p, 'w').write(job)
    subprocess.run(f'{GSUTIL} -q -m cp /tmp/room_job_room-[2-4][0-9].sh gs://{BUCKET}/room/jobs_tmp/', shell=True, check=True)
    for r in runs:      # the VMs look for room/jobs/<vm>.sh
      subprocess.run(f'{GSUTIL} -q cp gs://{BUCKET}/room/jobs_tmp/room_job_{r["vm"]}.sh gs://{BUCKET}/room/jobs/{r["vm"]}.sh',
                     shell=True, check=True)
    print(len(runs), 'from-scratch runs prepared:', ', '.join(r['name'] for r in runs))
  else:
    os.makedirs(os.path.join(ROOT, 'results_room', 'eval'), exist_ok=True)
    subprocess.run(f'{GSUTIL} -q -m rsync -r gs://{BUCKET}/runs_scratch_eval {ROOT}/results_room/eval', shell=True)
    subprocess.run(f'{GSUTIL} -q -m rsync -r -x ".*\\.pt$" gs://{BUCKET}/runs_scratch {ROOT}/runs_scratch', shell=True)
    print('collected')


if __name__ == '__main__':
  main()
