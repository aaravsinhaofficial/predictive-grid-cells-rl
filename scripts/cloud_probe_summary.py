"""Pull the fleet runs' small files (config.json, eval/*.json, metrics.jsonl)
from the bucket into a scratch tree and summarize the checkpoint probes by
(arm, condition, frames): mean +/- SEM over seeds.

  python scripts/cloud_probe_summary.py [--tags main,ext,dose] [--probe intact]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import collections

import numpy as np

GSUTIL = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
BUCKET = 'connectome-ai-research-popcode'
NAME_RE = re.compile(r'^(gc100|pgc33|pgc100|none|band|pband|pgc)([a-z0-9]*)_(ffb|ff|lstm)_s(\d+)$')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--tags', default='main,ext,dose')
  ap.add_argument('--dest', default='/tmp/claude-1000/-home-ec2-user-optimization/417ab975-bbb5-4ebd-8437-dec7451094cf/scratchpad/cloud_probes')
  ap.add_argument('--no_pull', action='store_true')
  ap.add_argument('--local_dirs', default='runs', help='comma list of local run dirs to include')
  args = ap.parse_args()
  os.makedirs(args.dest, exist_ok=True)
  if not args.no_pull:
    for tag in args.tags.split(','):
      prefix = 'runs' if tag == 'main' else f'runs_{tag}'
      d = os.path.join(args.dest, tag)
      os.makedirs(d, exist_ok=True)
      subprocess.run([GSUTIL, '-m', '-q', 'rsync', '-r', '-x', r'.*\.pt$|.*\.csv$|.*\.log$|.*\.stdout$',
                      f'gs://{BUCKET}/{prefix}', d], check=False)
  rows = collections.defaultdict(lambda: collections.defaultdict(dict))   # (arm,cond,frames) -> probe -> {seed: score}
  latest = {}
  run_dirs = glob.glob(os.path.join(args.dest, '*', '*', '*')) + \
      [p for ld in args.local_dirs.split(',') for p in glob.glob(os.path.join(ld, '*'))]
  for rd in run_dirs:
    name = os.path.basename(rd)
    m = NAME_RE.match(name)
    if not m or not os.path.isdir(rd):
      continue
    cond, arm, seed = m.group(1) + m.group(2), m.group(3), int(m.group(4))
    mp = os.path.join(rd, 'metrics.jsonl')
    if os.path.exists(mp):
      last = None
      for line in open(mp):
        if line.strip():
          last = line
      if last:
        r = json.loads(last)
        latest[name] = (r['frames'] / 1e6, r.get('avg_return_100'), r.get('frac_saturated'))
    for f in glob.glob(os.path.join(rd, 'eval', '*.json')):
      base = os.path.basename(f)[:-5]
      try:
        v = json.load(open(f))
      except Exception:
        continue
      if '@' in base:
        probe, rest = base.split('@', 1)
        fr = int(round(int(rest.split('f_')[0]) / 25e6) * 25)
      else:
        probe, fr = base, 'final'
      rows[(arm, cond, fr)][probe][seed] = v['mean_score']
  # progress
  if latest:
    by = collections.defaultdict(list)
    for n, (fr, ret, sat) in latest.items():
      by[NAME_RE.match(n).group(3)].append(fr)
    print('progress (M frames): ' + ', '.join(f'{a}: min {min(v):.0f} / median {np.median(v):.0f} / n {len(v)}' for a, v in sorted(by.items())))
    sat = [(n, s) for n, (_, _, s) in latest.items() if s]
    if sat:
      print('WARNING saturated gates in:', sat)
  probes = ['intact', 'no_code', 'no_current', 'no_goal', 'untrained']
  print(f"{'arm':5s} {'cond':16s} {'M':>5s} " + ' '.join(f'{p:>17s}' for p in probes))
  for (arm, cond, fr) in sorted(rows, key=lambda k: (k[0], k[1], str(k[2]))):
    pr = rows[(arm, cond, fr)]
    cells = []
    for p in probes:
      vals = np.array(list(pr.get(p, {}).values()), float)
      if len(vals) == 0:
        cells.append(f'{"-":>17s}')
      elif len(vals) == 1:
        cells.append(f'{vals[0]:8.2f} (n=1)   ')
      else:
        cells.append(f'{vals.mean():8.2f} +/- {vals.std(ddof=1)/np.sqrt(len(vals)):4.2f} n{len(vals):<2d}')
    print(f'{arm:5s} {cond:16s} {str(fr):>5s} ' + ' '.join(cells))


if __name__ == '__main__':
  main()
