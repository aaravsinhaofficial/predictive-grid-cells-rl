"""Figure 12: the two control populations run in every arm.

  (A-C) final score per seed for the grid code (512 cells), the 67 % grid code
        (the grid + 33 % PGC population with its 169 predictive slots held at
        zero), grid + 33 % PGC and the 50 / 50 code (256 + 256), one panel per
        arm; seeds are joined by lines, bars are mean +/- SEM
  (D)   class-size-matched lesions in the 50 / 50 code: cost of silencing a
        random half of the predictive cells (128) against a random half of
        the grid cells (128), mean of three draws, paired within seed

Reads runs/<cond>_<arm>_s<seed>/eval/*.json; writes
<out>/fig12_controls.png and .svg.

  python scripts/controls_figure.py --runs_dir runs --out figures
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import make_figures as mf  # noqa: E402  (colours, rc style, panel letters)

CONDS = [('gc100', 'grid\n(512)', mf.BLUE),
         ('gc67', '67 % grid\n(343)', mf.GRAY),
         ('pgc33', 'grid +\n33 % PGC', mf.ORANGE),
         ('pgc50', '50 / 50\n(256 + 256)', mf.DARK_ORANGE)]
NAME_RE = re.compile(r'^(gc100|gc67|pgc33|pgc50)_(ffb|ff|lstm)_s(\d+)$')


def read_score(path):
  try:
    with open(path) as f:
      return float(json.load(f)['mean_score'])
  except Exception:
    return None


def load(runs_dir):
  """{(cond, arm, seed): {'intact': s, 'pgc_half': mean of draws, 'gc_half': ...}}"""
  out = {}
  for d in sorted(glob.glob(os.path.join(runs_dir, '*'))):
    m = NAME_RE.match(os.path.basename(d))
    if not m or not os.path.isdir(d):
      continue
    key = (m.group(1), m.group(2), int(m.group(3)))
    rec = dict(intact=read_score(os.path.join(d, 'eval', 'intact.json')))
    for cls in ('pgc', 'gc'):
      draws = [read_score(os.path.join(d, 'eval', f'{cls}_half{k}.json')) for k in range(3)]
      draws = [v for v in draws if v is not None]
      rec[f'{cls}_half'] = float(np.mean(draws)) if draws else None
    if rec['intact'] is not None:
      out[key] = rec
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default='figures')
  args = ap.parse_args(argv)
  data = load(args.runs_dir)
  os.makedirs(args.out, exist_ok=True)

  fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
  for j, arm in enumerate(mf.ARMS):
    ax = axes[j]
    seeds = sorted({s for (c, a, s) in data if a == arm and c == 'gc100'})
    for s in seeds:
      pts = [(k, data[(c, arm, s)]['intact']) for k, (c, _, _) in enumerate(CONDS)
             if (c, arm, s) in data]
      if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=mf.GRAY, lw=0.7, alpha=0.55, zorder=1)
    for k, (c, lab, col) in enumerate(CONDS):
      v = np.array([data[(c, arm, s)]['intact'] for s in seeds if (c, arm, s) in data])
      if not len(v):
        continue
      ax.scatter(np.full(len(v), k), v, s=26, color=col, zorder=3, edgecolor='white', lw=0.6)
      ax.hlines(v.mean(), k - 0.26, k + 0.26, color=mf.INK, lw=2.6, zorder=4)
      if len(v) > 1:
        ax.vlines(k + 0.33, v.mean() - mf.sem(v), v.mean() + mf.sem(v), color=mf.INK, lw=1.2, zorder=4)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([lab for _, lab, _ in CONDS], fontsize=8)
    ax.set_xlim(-0.6, len(CONDS) - 0.4)
    ax.set_title(mf.ARM_TITLE[arm], color=mf.INK, fontsize=11)
    if j == 0:
      ax.set_ylabel('final score (200 episodes)')
    mf.letter_ax(ax, 'ABC'[j])

  # (D) class-matched lesions in the 50 / 50 code
  ax = axes[3]
  xt, xl = [], []
  for j, arm in enumerate(mf.ARMS):
    keys = sorted(k for k in data if k[0] == 'pgc50' and k[1] == arm
                  and data[k]['pgc_half'] is not None and data[k]['gc_half'] is not None)
    if not keys:
      continue
    cp = np.array([data[k]['intact'] - data[k]['pgc_half'] for k in keys])
    cg = np.array([data[k]['intact'] - data[k]['gc_half'] for k in keys])
    x0, x1 = 3 * j, 3 * j + 1
    for a, b in zip(cp, cg):
      ax.plot([x0, x1], [a, b], color=mf.GRAY, lw=0.7, alpha=0.55, zorder=1)
    ax.scatter(np.full(len(cp), x0), cp, s=26, color=mf.DARK_ORANGE, zorder=3, edgecolor='white', lw=0.6)
    ax.scatter(np.full(len(cg), x1), cg, s=26, color=mf.BLUE, zorder=3, edgecolor='white', lw=0.6)
    for x, v in ((x0, cp), (x1, cg)):
      ax.hlines(v.mean(), x - 0.26, x + 0.26, color=mf.INK, lw=2.6, zorder=4)
      if len(v) > 1:
        ax.vlines(x + 0.33, v.mean() - mf.sem(v), v.mean() + mf.sem(v), color=mf.INK, lw=1.2, zorder=4)
    xt += [x0, x1]
    xl += ['128\nPGC', '128\ngrid']
    ax.text((x0 + x1) / 2, 1.0, mf.ARM_TITLE[arm], transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8.5, color=mf.INK2)
  ax.axhline(0, color=mf.GRAY, ls=(0, (3, 3)), lw=1.0, zorder=0)
  ax.set_xticks(xt)
  ax.set_xticklabels(xl, fontsize=8.3)
  ax.set_xlim(-0.7, 3 * len(mf.ARMS) - 1.3)
  ax.set_ylabel('lesion cost in the 50 / 50 code\n(intact − half of a class silenced)')
  mf.letter_ax(ax, 'D')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'fig12_controls.png'))
  fig.savefig(os.path.join(args.out, 'fig12_controls.svg'))
  plt.close(fig)
  print('wrote', os.path.join(args.out, 'fig12_controls.png'))


if __name__ == '__main__':
  main()
