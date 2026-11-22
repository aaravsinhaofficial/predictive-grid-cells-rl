"""Starting states and run matrix of the room-change experiment.

Every run continues one of the sixteen 1 B LSTM agents (runs_ext/<cond>_lstm_s<seed>:
weights, RMSProp statistics and random-number state from resume.pt) for another
250 M frames:

  <cond>room_lstm_s<seed>   seeds 1-8   in the changed rooms (--rooms changed)
  <cond>same_lstm_s<seed>   seeds 1-2   in the familiar room (matched-budget reference)

Writes runs_room/init/<run>/{config.json,resume.pt} and runs_room/matrix.json
(run -> VM, train arguments); --upload copies the starting states and the code
to the bucket.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
BUCKET = 'connectome-ai-research-popcode'
FRAMES = 1_250_000_000
BASE_ARGS = ('--cond {cond} --seed {seed} --frames {frames} --policy lstm --vision raycast '
             '--encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--upload', action='store_true')
  ap.add_argument('--src', default=os.path.join(ROOT, 'runs_ext'))
  ap.add_argument('--dest', default=os.path.join(ROOT, 'runs_room'))
  args = ap.parse_args()
  runs = []
  for seed in range(1, 9):
    for cond in ('gc100', 'pgc33'):
      runs.append(dict(name=f'{cond}room_lstm_s{seed}', cond=cond, seed=seed, rooms='changed'))
  for seed in (1, 2):
    for cond in ('gc100', 'pgc33'):
      runs.append(dict(name=f'{cond}same_lstm_s{seed}', cond=cond, seed=seed, rooms='none'))
  for i, r in enumerate(runs):
    src = os.path.join(args.src, f'{r["cond"]}_lstm_s{r["seed"]}')
    d = os.path.join(args.dest, 'init', r['name'])
    os.makedirs(d, exist_ok=True)
    cfg = json.load(open(os.path.join(src, 'config.json')))
    assert cfg['frames'] == 1_000_000_000 and cfg['policy'] == 'lstm' and cfg['vision'] == 'raycast', cfg
    cfg['rooms'] = r['rooms']
    cfg['extensions'] = list(cfg.get('extensions') or []) + [
        dict(from_frames=1_000_000_000, to_frames=FRAMES, mode='resume', rooms=r['rooms'],
             source=f'runs_ext/{r["cond"]}_lstm_s{r["seed"]}')]
    json.dump(cfg, open(os.path.join(d, 'config.json'), 'w'), indent=1)
    # The starting state must be the 1 B agent itself: the resume state of a run
    # whose last extension ran in the cloud is not collected with it (seeds 5-8:
    # fetched from gs://<bucket>/runs_lstm1b/*/<run>/resume.pt), so check it
    # against the final checkpoint instead of trusting the file on disk.
    import torch
    st = torch.load(os.path.join(src, 'resume.pt'), map_location='cpu', weights_only=False)
    fin = torch.load(os.path.join(src, 'ckpt_final.pt'), map_location='cpu')
    assert st['frames_done'] == fin['frames'] == 1_000_000_000, (r['name'], st['frames_done'], fin['frames'])
    assert all(torch.equal(st['policy'][k].cpu(), fin['policy'][k].cpu()) for k in fin['policy']), r['name']
    shutil.copyfile(os.path.join(src, 'resume.pt'), os.path.join(d, 'resume.pt'))
    r['vm'] = f'room-{i:02d}'
    r['args'] = BASE_ARGS.format(cond=r['cond'], seed=r['seed'], frames=FRAMES) + \
        (' --rooms changed' if r['rooms'] == 'changed' else '')
    r['deadman'] = 720
  json.dump(runs, open(os.path.join(args.dest, 'matrix.json'), 'w'), indent=1)
  print(len(runs), 'runs;', os.path.join(args.dest, 'matrix.json'))
  if args.upload:
    tar = '/tmp/room_code.tar.gz'
    subprocess.run(f'cd {ROOT} && tar czf {tar} --exclude=__pycache__ code scripts/__init__.py scripts/room 2>/dev/null || '
                   f'(cd {ROOT} && tar czf {tar} --exclude=__pycache__ code scripts/room)', shell=True, check=True)
    subprocess.run(f'{GSUTIL} cp {tar} gs://{BUCKET}/room/code.tar.gz', shell=True, check=True)
    subprocess.run(f'{GSUTIL} -m rsync -r {os.path.join(args.dest, "init")} gs://{BUCKET}/room/init',
                   shell=True, check=True)
    print('uploaded code and starting states')


if __name__ == '__main__':
  main()
