"""Staged run scheduler for the population-code experiment.

Keeps up to --max_concurrent training processes alive on the GPU, resumes any
run that has a resume.pt, extends finished runs whose budget was raised, skips
finished runs, retries failures, probes every intermediate checkpoint
(ckpt_*.pt, every 25M frames) with the reliance probes, and evaluates every
finished run with the full probe set. Idempotent: relaunch it any time.

  tmux new -d -s popcode "cd /home/ec2-user/optimization && \
     /home/ec2-user/banino/.venv/bin/python scripts/launch_matrix.py --wave 1"

Arms (policy x vision):
  ff    feed-forward policy, raycast vision   (memoryless; vision + code)
  ffb   feed-forward policy, heading-only vision (memoryless; the code is the
        ONLY position source -- the code-dependent arm)
  lstm  policy LSTM, raycast vision           (paper-faithful)

Waves:
  1  pilot / interpretability guards: gc100, pgc33, none x 3 arms, seed 1,
     150M frames, checkpoint probes every 25M (decides which arms are
     code-dependent, when goal-homing emerges, and the budget)
  2  primary contrast: gc100 vs pgc33 (goal code intact) and pgc33 with a
     position-only goal code, seeds 1-8, all arms (wave-1 runs are extended)
  3  controls (3 seeds): none, pgc100, gc100 + velocity input, gc100 with the
     look-ahead goal snapshot, PGC shift dose series (10 / 40 cm)
Run names: <cond>[<variant>]_<arm>_s<seed>.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/ec2-user/banino/.venv/bin/python'
FULL_PROBES = ('intact,no_code,no_current,no_goal,no_pgc,no_gc,pgc_as_gc,fake_goal,'
               'shuffled_current,sham0,sham1,sham2,sham3,sham4,untrained')
CKPT_PROBES = 'intact,no_code,no_current,no_goal,untrained'
CKPT_EPISODES = 64
ARMS = {
    'ff': ['--policy', 'ff', '--vision', 'raycast'],
    'ffb': ['--policy', 'ff', '--vision', 'heading_only'],
    'lstm': ['--policy', 'lstm', '--vision', 'raycast'],
}


def run_spec(cond, arm, seed, frames, prio, variant='', extra=()):
  return dict(name=f'{cond}{variant}_{arm}_s{seed}', cond=cond, arm=arm, seed=seed,
              frames=frames, prio=prio, extra=list(extra))


def matrix(wave, frames_pilot, frames):
  runs = []
  if wave in (1, 'all'):
    for arm in ('ffb', 'ff', 'lstm'):
      for cond in ('gc100', 'pgc33', 'none'):
        runs.append(run_spec(cond, arm, 1, frames_pilot, prio=0))
  if wave in (2, 'all'):
    for seed in range(1, 9):
      for arm in ('ffb', 'ff', 'lstm'):
        runs.append(run_spec('gc100', arm, seed, frames, prio=1 + (seed > 4)))
        runs.append(run_spec('pgc33', arm, seed, frames, prio=1 + (seed > 4)))
        runs.append(run_spec('pgc33', arm, seed, frames, prio=1 + (seed > 4),
                             variant='goalzero', extra=['--goal_code_shift', 'zero']))
  if wave in (3, 'all'):
    for seed in (1, 2, 3):
      for arm in ('ffb', 'ff', 'lstm'):
        runs.append(run_spec('none', arm, seed, frames, prio=3))
        runs.append(run_spec('pgc100', arm, seed, frames, prio=4))
        runs.append(run_spec('gc100', arm, seed, frames, prio=4, variant='vel',
                             extra=['--extra_input', 'vel']))
        runs.append(run_spec('gc100', arm, seed, frames, prio=4, variant='lookahead',
                             extra=['--goal_code_shift', 'lookahead']))
        runs.append(run_spec('pgc33', arm, seed, frames, prio=5, variant='shift10',
                             extra=['--pgc_shift', '0.10']))
        runs.append(run_spec('pgc33', arm, seed, frames, prio=5, variant='shift40',
                             extra=['--pgc_shift', '0.40']))
  # de-duplicate by name (a later wave may re-list a run with a larger budget)
  by_name = {}
  for r in runs:
    if r['name'] in by_name:
      by_name[r['name']]['frames'] = max(by_name[r['name']]['frames'], r['frames'])
    else:
      by_name[r['name']] = r
  return sorted(by_name.values(), key=lambda r: (r['prio'], r['seed'], r['arm'], r['cond']))


def train_cmd(r, out, train_args):
  return ([PY, '-m', 'code.train', '--cond', r['cond'], '--seed', str(r['seed']),
           '--frames', str(r['frames']), '--out', out, '--resume']
          + ARMS[r['arm']] + r['extra'] + train_args)


def eval_cmd(out, episodes, probes, ckpt='final'):
  return [PY, '-m', 'code.evaluate', '--run', out, '--probes', probes,
          '--episodes', str(episodes), '--ckpt', ckpt]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--wave', default='1', help='1 | 2 | 3 | all')
  ap.add_argument('--max_concurrent', type=int, default=9)
  ap.add_argument('--frames', type=int, default=300_000_000)
  ap.add_argument('--frames_pilot', type=int, default=150_000_000)
  ap.add_argument('--eval_episodes', type=int, default=200)
  ap.add_argument('--runs_dir', default=os.path.join(ROOT, 'runs'))
  ap.add_argument('--only', default=None, help='regex on run name')
  ap.add_argument('--matrix_json', default=None, help='override the matrix')
  ap.add_argument('--train_args',
                  default='--encoder_norm layernorm --lr 3e-4 --rms_eps 1e-5',
                  help='extra arguments appended to every train command')
  ap.add_argument('--no_ckpt_probes', action='store_true')
  ap.add_argument('--dry_run', action='store_true')
  ap.add_argument('--poll', type=float, default=30.0)
  args = ap.parse_args()

  wave = args.wave if args.wave == 'all' else int(args.wave)
  runs = json.load(open(args.matrix_json)) if args.matrix_json else \
      matrix(wave, args.frames_pilot, args.frames)
  if args.only:
    runs = [r for r in runs if re.search(args.only, r['name'])]
  train_args = args.train_args.split()
  os.makedirs(args.runs_dir, exist_ok=True)
  log_f = open(os.path.join(args.runs_dir, 'scheduler.log'), 'a')

  def log(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    log_f.write(line + '\n')
    log_f.flush()

  log(f'scheduler start: wave={wave} {len(runs)} runs, max_concurrent={args.max_concurrent}')
  if args.dry_run:
    for r in runs:
      print(' '.join(train_cmd(r, os.path.join(args.runs_dir, r['name']), train_args)))
    return

  env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
  pending = list(runs)
  running = {}     # name -> dict(proc, stage, run, out, logf)
  attempts = {}
  probe_proc = None   # at most one checkpoint-probe process at a time
  probed = set()

  def start(r, stage):
    out = os.path.join(args.runs_dir, r['name'])
    os.makedirs(out, exist_ok=True)
    cmd = train_cmd(r, out, train_args) if stage == 'train' else \
        eval_cmd(out, args.eval_episodes, FULL_PROBES)
    logf = open(os.path.join(out, f'{stage}.log'), 'a')
    logf.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(cmd)}\n')
    logf.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    running[r['name']] = dict(proc=p, stage=stage, run=r, out=out, logf=logf)
    log(f'started {stage} {r["name"]} (pid {p.pid})')

  def final_frames(out):
    p = os.path.join(out, 'ckpt_final.pt')
    if not os.path.exists(p):
      return -1
    try:
      import torch
      return int(torch.load(p, map_location='cpu').get('frames', 0))
    except Exception:
      return -1

  def finished(r, out):
    return final_frames(out) >= r['frames']

  def evaluated(out):
    return all(os.path.exists(os.path.join(out, 'eval', f'{p}.json'))
               for p in FULL_PROBES.split(','))

  def next_ckpt_to_probe():
    """(run dir, ckpt path) of an intermediate checkpoint lacking probe evals."""
    for d in sorted(glob.glob(os.path.join(args.runs_dir, '*_s[0-9]*'))):
      for ck in sorted(glob.glob(os.path.join(d, 'ckpt_0*.pt'))):
        if ck in probed:
          continue
        frames = int(os.path.basename(ck)[5:15])
        done = all(glob.glob(os.path.join(d, 'eval', f'{p}@{frames}f_{CKPT_EPISODES}ep.json'))
                   for p in CKPT_PROBES.split(','))
        if done:
          probed.add(ck)
          continue
        return d, ck
    return None

  while True:
    # Reap training / final-eval processes.
    for name in list(running):
      st = running[name]
      rc = st['proc'].poll()
      if rc is None:
        continue
      st['logf'].close()
      del running[name]
      r, out = st['run'], st['out']
      if st['stage'] == 'train':
        if finished(r, out):
          log(f'train done {name}')
          if not evaluated(out):
            start(r, 'eval')
        else:
          attempts[name] = attempts.get(name, 0) + 1
          log(f'train FAILED {name} rc={rc} attempt {attempts[name]}')
          if attempts[name] <= 3:
            pending.insert(0, r)
      else:
        log(f'eval {"done" if evaluated(out) else "FAILED rc=%d" % rc} {name}')
    # Launch training.
    n_train = sum(1 for s in running.values() if s['stage'] == 'train')
    while pending and n_train < args.max_concurrent:
      r = pending.pop(0)
      out = os.path.join(args.runs_dir, r['name'])
      if r['name'] in running:
        pending.append(r)
        break
      if finished(r, out):
        if evaluated(out):
          log(f'skip {r["name"]} (finished + evaluated)')
        else:
          start(r, 'eval')
        continue
      start(r, 'train')
      n_train += 1
    # Checkpoint probes (one at a time, low priority).
    if probe_proc is not None and probe_proc[0].poll() is not None:
      rc = probe_proc[0].returncode
      log(f'ckpt probe {"done" if rc == 0 else "FAILED rc=%d" % rc} {probe_proc[1]}')
      probed.add(probe_proc[1])
      probe_proc[2].close()
      probe_proc = None
    if probe_proc is None and not args.no_ckpt_probes:
      nxt = next_ckpt_to_probe()
      if nxt is not None:
        d, ck = nxt
        logf = open(os.path.join(d, 'probe.log'), 'a')
        cmd = eval_cmd(d, CKPT_EPISODES, CKPT_PROBES, ckpt=ck)
        logf.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(cmd)}\n')
        logf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        probe_proc = (p, ck, logf)
        log(f'started ckpt probe {os.path.relpath(ck, args.runs_dir)} (pid {p.pid})')
    if not pending and not running and probe_proc is None:
      break
    time.sleep(args.poll)
  log('scheduler finished')


if __name__ == '__main__':
  main()
