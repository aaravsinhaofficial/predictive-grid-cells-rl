"""Package the small, shareable artefacts of every run into data/:

  data/evals/<run>/config.json             run configuration (population, arm, seed, ...)
  data/evals/<run>/eval/<probe>.json       final 200-episode paired evaluations (all probes)
  data/evals/<run>/eval/<probe>@<frames>f_64ep.json   checkpoint probes every 25M frames
  data/learning_curves.csv                 training return per run, binned by 5M frames
  data/summary.json                        the aggregate statistics behind results/RESULTS.md

Checkpoints, resume files, episode logs and metrics are NOT included (~10 GB).

  python scripts/package_data.py --runs_dir runs --out data
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil

import numpy as np

NAME_RE = re.compile(r'^(gc100|pgc33|pgc100|none|band|pband|pgc)([a-z0-9]*)_(ffb|ff|lstm)_s(\d+)$')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default='data')
  ap.add_argument('--bin_frames', type=float, default=5e6)
  args = ap.parse_args()
  os.makedirs(os.path.join(args.out, 'evals'), exist_ok=True)
  rows = []
  n = 0
  for d in sorted(glob.glob(os.path.join(args.runs_dir, '*'))):
    name = os.path.basename(d)
    if not NAME_RE.match(name) or not os.path.exists(os.path.join(d, 'config.json')):
      continue
    dst = os.path.join(args.out, 'evals', name)
    os.makedirs(os.path.join(dst, 'eval'), exist_ok=True)
    shutil.copy(os.path.join(d, 'config.json'), dst)
    for f in glob.glob(os.path.join(d, 'eval', '*.json')):
      shutil.copy(f, os.path.join(dst, 'eval'))
    ep_p = os.path.join(d, 'episodes.csv')
    if os.path.exists(ep_p):
      ep = np.genfromtxt(ep_p, delimiter=',', names=True)
      key = 'ret' if 'ret' in ep.dtype.names else 'return_'
      edges = np.arange(0, ep['frames'].max() + args.bin_frames, args.bin_frames)
      idx = np.digitize(ep['frames'], edges) - 1
      for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 5:
          rows.append((name, int(edges[b]), int(edges[b + 1]), int(m.sum()),
                       round(float(ep[key][m].mean()), 3), round(float(ep['n_goals'][m].mean()), 4)))
    n += 1
  with open(os.path.join(args.out, 'learning_curves.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'frames_lo', 'frames_hi', 'n_episodes', 'mean_return', 'mean_goals'])
    w.writerows(rows)
  if os.path.exists('results/summary.json'):
    shutil.copy('results/summary.json', os.path.join(args.out, 'summary.json'))
  print(f'packaged {n} runs, {len(rows)} curve bins into {args.out}')


if __name__ == '__main__':
  main()
