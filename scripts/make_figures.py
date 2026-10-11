"""Publication-style figures for the grid-cell vs predictive-grid-cell experiment.

Reads runs/*/config.json, runs/*/eval/*.json, runs/*/episodes.csv and
runs/validation_pgc33_s1/validation.json; writes figures/fig*.png (+ .svg).

  python scripts/make_figures.py --runs_dir runs --out figures
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NAME_RE = re.compile(r'^(gc100|pgc33|pgc100|none)([a-z0-9]*)_(ffb|ff|lstm)_s(\d+)$')
ARMS = ['ffb', 'ff', 'lstm']
ARM_TITLE = {'ffb': 'code only', 'ff': 'code + vision', 'lstm': 'LSTM + vision'}
# Validated categorical palette (dataviz reference instance): blue, orange, aqua + gray.
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9891'
INK, INK2 = '#0b0b0b', '#52514e'
COND_STYLE = {'gc100': (BLUE, 'grid cells'), 'pgc33': (ORANGE, 'grid + 33% PGC'),
              'none': (GRAY, 'no code')}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': False,
    'savefig.dpi': 200, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


# ------------------------------------------------------------------ data
def load_runs(runs_dir):
  runs = {}
  for d in sorted(glob.glob(os.path.join(runs_dir, '*'))):
    m = NAME_RE.match(os.path.basename(d))
    if not m or not os.path.exists(os.path.join(d, 'config.json')):
      continue
    cond, arm, seed = m.group(1) + m.group(2), m.group(3), int(m.group(4))
    evals = {}
    for f in glob.glob(os.path.join(d, 'eval', '*.json')):
      base = os.path.basename(f)[:-5]
      if '@' in base:
        continue
      v = json.load(open(f))
      if v.get('canonical'):
        evals[base] = v
    shams = [evals[k]['mean_score'] for k in evals if k.startswith('sham')]
    if shams:
      evals['sham'] = {'mean_score': float(np.mean(shams))}
    ep = None
    p = os.path.join(d, 'episodes.csv')
    if os.path.exists(p):
      try:
        ep = np.genfromtxt(p, delimiter=',', names=True)
      except Exception:
        ep = None
    runs[(cond, arm, seed)] = dict(evals=evals, ep=ep, cfg=json.load(open(os.path.join(d, 'config.json'))))
  return runs


def score(runs, cond, arm, seed, probe='intact', key='mean_score'):
  r = runs.get((cond, arm, seed))
  if r is None or probe not in r['evals']:
    return None
  return r['evals'][probe].get(key)


def seeds_of(runs, cond, arm):
  return sorted(s for (c, a, s) in runs if c == cond and a == arm)


def sem(x):
  x = np.asarray(x, float)
  return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def binned(ep, edges, key='ret'):
  if ep is None:
    return np.full(len(edges) - 1, np.nan)
  names = ep.dtype.names
  k = key if key in names else ('return_' if 'return_' in names else key)
  f, v = ep['frames'], ep[k].astype(float)
  idx = np.digitize(f, edges) - 1
  out = np.full(len(edges) - 1, np.nan)
  for b in range(len(edges) - 1):
    mm = idx == b
    if mm.sum() >= 5:
      out[b] = v[mm].mean()
  return out


# --------------------------------------------------------------- figures
def paired_panel(ax, runs, arm, key='mean_score', probe='intact', ylabel=None, show_none=True):
  """Slopegraph: grid vs grid+PGC per seed, with group means."""
  seeds = sorted(set(seeds_of(runs, 'gc100', arm)) & set(seeds_of(runs, 'pgc33', arm)))
  a = np.array([score(runs, 'gc100', arm, s, probe, key) for s in seeds], float)
  b = np.array([score(runs, 'pgc33', arm, s, probe, key) for s in seeds], float)
  ok = np.isfinite(a) & np.isfinite(b)
  a, b = a[ok], b[ok]
  x0, x1 = 0, 1
  for ai, bi in zip(a, b):
    ax.plot([x0, x1], [ai, bi], color=GRAY, lw=0.8, alpha=0.6, zorder=1)
  ax.scatter(np.full(len(a), x0), a, s=28, color=BLUE, zorder=3, edgecolor='white', lw=0.6)
  ax.scatter(np.full(len(b), x1), b, s=28, color=ORANGE, zorder=3, edgecolor='white', lw=0.6)
  for x, v, c in ((x0, a, BLUE), (x1, b, ORANGE)):
    ax.hlines(v.mean(), x - 0.22, x + 0.22, color=c, lw=3, zorder=4)
    ax.vlines(x + (0.3 if x == x0 else -0.3), v.mean() - sem(v), v.mean() + sem(v), color=c, lw=1.4, zorder=4)
  if show_none:
    n = [score(runs, 'none', arm, s, probe, key) for s in seeds_of(runs, 'none', arm)]
    n = [v for v in n if v is not None and np.isfinite(v)]
    if n:
      ax.axhline(np.mean(n), color=GRAY, ls=(0, (3, 3)), lw=1.2, zorder=0)
      ax.text(1.85, np.mean(n), 'no code', color=INK2, fontsize=8, va='bottom', ha='right')
  d = b - a
  ax.set_xticks([x0, x1])
  ax.set_xticklabels(['grid', 'grid +\n33% PGC'])
  ax.set_xlim(-0.6, 1.9)
  if ylabel:
    ax.set_ylabel(ylabel)
  ax.set_title(ARM_TITLE[arm], color=INK, fontsize=11, pad=8)
  return d


def fig_main(runs, out):
  fig, axes = plt.subplots(2, 3, figsize=(9.2, 7.0))
  for j, arm in enumerate(ARMS):
    d = paired_panel(axes[0, j], runs, arm, ylabel='score (10 per goal, 90 s episode)' if j == 0 else None)
    try:
      from scipy import stats
      p = stats.ttest_rel(d, np.zeros_like(d)).pvalue
    except Exception:
      p = float('nan')
    axes[0, j].text(0.5, 0.02, f'Δ = {d.mean():+.1f} ± {sem(d):.1f}   p = {p:.2g}', transform=axes[0, j].transAxes,
                    ha='center', va='bottom', fontsize=8.5, color=INK2)
    d2 = paired_panel(axes[1, j], runs, arm, key='mean_path_eff',
                      ylabel='path efficiency of goal returns' if j == 0 else None)
    try:
      p2 = stats.ttest_rel(d2, np.zeros_like(d2)).pvalue
    except Exception:
      p2 = float('nan')
    axes[1, j].text(0.5, 0.02, f'Δ = {d2.mean():+.3f} ± {sem(d2):.3f}   p = {p2:.2g}', transform=axes[1, j].transAxes,
                    ha='center', va='bottom', fontsize=8.5, color=INK2)
    axes[1, j].set_title('')
  for ax in axes[0]:
    ax.set_ylim(bottom=0)
  for ax in axes[1]:
    ax.set_ylim(0, 1)
  fig.tight_layout(h_pad=2.0)
  fig.savefig(os.path.join(out, 'fig1_main_result.png'))
  fig.savefig(os.path.join(out, 'fig1_main_result.svg'))
  plt.close(fig)


def fig_curves(runs, out, bin_frames=10e6):
  budget = max(r['cfg']['frames'] for r in runs.values())
  edges = np.arange(0, budget + bin_frames, bin_frames)
  x = (edges[:-1] + edges[1:]) / 2 / 1e6
  fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
  for j, arm in enumerate(ARMS):
    ax = axes[j]
    for cond in ('none', 'gc100', 'pgc33'):
      cs = np.array([binned(runs[(c, a, s)]['ep'], edges) for (c, a, s) in runs if c == cond and a == arm])
      if not len(cs):
        continue
      mu = np.nanmean(cs, 0)
      se = np.nanstd(cs, 0) / np.sqrt(np.maximum((~np.isnan(cs)).sum(0), 1))
      col, label = COND_STYLE[cond]
      ax.plot(x, mu, color=col, lw=2)
      ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.18, lw=0)
      last = np.where(np.isfinite(mu))[0][-1]
      ax.text(x[last] + 3, mu[last], label, color=col, fontsize=8.5, va='center')
    ax.set_title(ARM_TITLE[arm], color=INK, fontsize=11)
    ax.set_xlabel('training frames (millions)')
    ax.set_xlim(0, 360)
    ax.spines['right'].set_visible(False)
  axes[0].set_ylabel('episode score (training)')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig2_learning_curves.png'))
  fig.savefig(os.path.join(out, 'fig2_learning_curves.svg'))
  plt.close(fig)


def fig_probes(runs, out):
  probes = [('intact', 'intact'), ('no_code', 'no code'), ('no_goal', 'no goal\nmemory'),
            ('sham', 'sham lesion\n(169 random)'), ('no_pgc', 'PGC lesion\n(169 PGCs)')]
  fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
  w = 0.36
  for j, arm in enumerate(ARMS):
    ax = axes[j]
    for k, (cond, off) in enumerate((('gc100', -w / 2), ('pgc33', w / 2))):
      col, label = COND_STYLE[cond]
      means, errs = [], []
      for p, _ in probes:
        vals = [score(runs, cond, arm, s, p) for s in seeds_of(runs, cond, arm)]
        vals = [v for v in vals if v is not None]
        means.append(np.mean(vals) if vals else np.nan)
        errs.append(sem(vals) if len(vals) > 1 else 0)
      xs = np.arange(len(probes)) + off
      ax.bar(xs, means, w - 0.04, color=col, label=label, zorder=2)
      ax.errorbar(xs, means, yerr=errs, fmt='none', ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
    ax.set_xticks(np.arange(len(probes)))
    ax.set_xticklabels([l for _, l in probes], fontsize=7.5)
    ax.set_title(ARM_TITLE[arm], color=INK, fontsize=11)
    ax.grid(axis='y', color='#e8e7e3', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
  axes[0].set_ylabel('score at evaluation')
  axes[0].legend(loc='upper right', fontsize=8.5)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig3_probes.png'))
  fig.savefig(os.path.join(out, 'fig3_probes.svg'))
  plt.close(fig)


def fig_conditions(runs, out):
  conds = [('none', 'no code', GRAY), ('gc100', 'grid cells', BLUE),
           ('gc100vel', 'grid cells + velocity input', BLUE),
           ('gc100lookahead', 'grid cells, look-ahead goal memory', BLUE),
           ('pgc33shift10', '33% PGC, 10 cm shift', ORANGE),
           ('pgc33', '33% PGC, 20 cm shift', ORANGE),
           ('pgc33shift40', '33% PGC, 40 cm shift', ORANGE),
           ('pgc33goalzero', '33% PGC, position-only goal memory', AQUA),
           ('pgc100', '100% PGC, 20 cm shift', ORANGE)]
  fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
  for j, arm in enumerate(ARMS):
    ax = axes[j]
    for i, (cond, label, col) in enumerate(conds):
      vals = [score(runs, cond, arm, s) for s in seeds_of(runs, cond, arm)]
      vals = np.array([v for v in vals if v is not None], float)
      if not len(vals):
        continue
      y = len(conds) - 1 - i
      ax.scatter(vals, np.full(len(vals), y), s=14, color=col, alpha=0.45, zorder=2, edgecolor='none')
      ax.errorbar(vals.mean(), y, xerr=sem(vals), fmt='o', color=col, ms=6, capsize=3,
                  elinewidth=1.2, zorder=3, markeredgecolor='white')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([l for _, l, _ in conds][::-1], fontsize=8.5)
    ax.set_title(ARM_TITLE[arm], color=INK, fontsize=11)
    ax.set_xlabel('score at evaluation')
    ax.grid(axis='x', color='#e8e7e3', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig4_all_conditions.png'))
  fig.savefig(os.path.join(out, 'fig4_all_conditions.svg'))
  plt.close(fig)


def fig_codes(runs_dir, out):
  from popcode.codes import GridPopulation
  vpath = os.path.join(runs_dir, 'validation_pgc33_s1', 'validation.json')
  val = json.load(open(vpath)) if os.path.exists(vpath) else None
  pop = GridPopulation(n_cells=512, pgc_frac=0.33, pgc_shift=0.20, seed=1, device='cpu')
  half = 1.25
  maps = pop.ratemaps(res=100, half=half, dir=(1.0, 0.0))
  gc = [i for i in pop.gc_idx if pop.module[i] == 1][0]
  pgc = [i for i in pop.pgc_idx if pop.module[i] == 1][0]
  fig = plt.figure(figsize=(10.5, 3.4))
  gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.08, 1.6], wspace=0.35)
  for k, (idx, title, col) in enumerate(((gc, 'grid cell', BLUE), (pgc, 'predictive grid cell', ORANGE))):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(maps[idx].T, origin='lower', extent=[-half, half, -half, half], cmap='Blues' if k == 0 else 'Oranges',
              vmin=0, vmax=1)
    ax.set_title(title, color=col, fontsize=11)
    ax.set_xlabel('x (m)')
    if k == 0:
      ax.set_ylabel('y (m)')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    for s in ax.spines.values():
      s.set_visible(False)
    if k == 1:
      ax.annotate('', xy=(0.55, -1.05), xytext=(0.05, -1.05),
                  arrowprops=dict(arrowstyle='->', color=INK, lw=1.4))
      ax.text(0.3, -0.98, 'travel', ha='center', fontsize=8, color=INK)
  ax = fig.add_subplot(gs[0, 3])
  if val:
    for cls, col in (('gc', BLUE), ('pgc', ORANGE)):
      s = val['summary'].get(f'space_{cls}')
      if not s:
        continue
      x = np.array(s['axis_cm'])
      mu, se = np.array(s['curve_mean']), np.array(s['curve_sd']) / np.sqrt(s['n'])
      ax.plot(x, mu, color=col, lw=2, label='grid cells' if cls == 'gc' else 'predictive grid cells')
      ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.18, lw=0)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.axvline(20, color=ORANGE, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel('position shift along direction of travel (cm)')
    ax.set_ylabel('gridness')
    ax.legend(loc='lower center', fontsize=8.5)
  fig.savefig(os.path.join(out, 'fig5_codes.png'), bbox_inches='tight')
  fig.savefig(os.path.join(out, 'fig5_codes.svg'), bbox_inches='tight')
  plt.close(fig)


def fig_reliance_over_training(runs_dir, out):
  """Intact vs code-zeroed score through training (checkpoint probes), per arm, gc100 vs pgc33."""
  rows = {}
  for d in sorted(glob.glob(os.path.join(runs_dir, '*'))):
    m = NAME_RE.match(os.path.basename(d))
    if not m:
      continue
    cond, arm, seed = m.group(1) + m.group(2), m.group(3), int(m.group(4))
    if cond not in ('gc100', 'pgc33'):
      continue
    for f in glob.glob(os.path.join(d, 'eval', '*@*f_64ep.json')):
      base = os.path.basename(f)[:-5]
      probe, rest = base.split('@', 1)
      if probe not in ('intact', 'no_code'):
        continue
      fr = int(round(int(rest.split('f_')[0]) / 25e6) * 25)
      v = json.load(open(f))
      rows.setdefault((arm, cond, probe), {}).setdefault(fr, []).append(v['mean_score'])
  fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
  for j, arm in enumerate(ARMS):
    ax = axes[j]
    for cond in ('gc100', 'pgc33'):
      col, label = COND_STYLE[cond]
      for probe, ls in (('intact', '-'), ('no_code', (0, (3, 2)))):
        ser = rows.get((arm, cond, probe), {})
        if not ser:
          continue
        xs = sorted(ser)
        mu = [np.mean(ser[x]) for x in xs]
        ax.plot(xs, mu, color=col, lw=2, ls=ls,
                label=f'{label}' + ('' if probe == 'intact' else ', code zeroed'))
    ax.set_title(ARM_TITLE[arm], color=INK, fontsize=11)
    ax.set_xlabel('training frames (millions)')
  axes[0].set_ylabel('score (64-episode probe)')
  axes[0].legend(fontsize=7.5, loc='upper left')
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig6_reliance_over_training.png'))
  fig.savefig(os.path.join(out, 'fig6_reliance_over_training.svg'))
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default='figures')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  runs = load_runs(args.runs_dir)
  print(f'{len(runs)} runs')
  fig_main(runs, args.out)
  fig_curves(runs, args.out)
  fig_probes(runs, args.out)
  fig_conditions(runs, args.out)
  fig_codes(args.runs_dir, args.out)
  fig_reliance_over_training(args.runs_dir, args.out)
  print('figures written to', args.out)


if __name__ == '__main__':
  main()
