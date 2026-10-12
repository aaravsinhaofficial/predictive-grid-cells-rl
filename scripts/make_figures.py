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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NAME_RE = re.compile(r'^(gc100|pgc33|pgc100|none|band|pband|pgc)([a-z0-9]*)_(ffb|ff|lstm)_s(\d+)$')
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


def fig_modules(out):
  """How the population code is built (Fig. 0): four modules of increasing
  spacing, random phases within a module, the code as a function of position,
  a grid cell next to its predictive counterpart, and how the PGCs are allotted
  and shifted per module in the arrangement experiment."""
  from popcode.codes import GridPopulation
  pop = GridPopulation(n_cells=512, pgc_frac=0.33, pgc_shift=0.20, seed=1, device='cpu')
  half, res = 1.25, 160
  maps = pop.ratemaps(res=res, half=half, dir=(1.0, 0.0), shift_scale=0.0)   # every cell as a grid cell
  maps_pgc = pop.ratemaps(res=res, half=half, dir=(1.0, 0.0), shift_scale=1.0)
  phi = pop.phi.cpu().numpy()
  lam, th, M = pop.spacings, pop.orientation, pop.n_modules
  ext = [-half, half, -half, half]
  PURPLE = '#8b5cf6'

  def lattice(m):
    a1 = lam[m] * np.array([np.cos(th[m] + np.pi / 6), np.sin(th[m] + np.pi / 6)])
    a2 = lam[m] * np.array([np.cos(th[m] + np.pi / 2), np.sin(th[m] + np.pi / 2)])
    return a1, a2

  def nearest_field(i, target):
    a1, a2 = lattice(pop.module[i])
    A = np.stack([a1, a2], 1)
    return phi[i] + A @ np.round(np.linalg.solve(A, target - phi[i])), a1

  def clean(ax):
    for s in ax.spines.values():
      s.set_visible(False)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

  def letter(ax, s):
    ax.text(-0.16, 1.1, s, transform=ax.transAxes, fontsize=12, fontweight='bold', color=INK)

  fig = plt.figure(figsize=(11, 10))
  gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.75], hspace=0.5, wspace=0.4)

  # A-D: one example grid cell per module, arrow = spacing between neighbouring fields
  ref = np.array([-0.45, -0.55])
  for m in range(M):
    idx = [i for i in pop.gc_idx if pop.module[i] == m]
    c0 = [nearest_field(i, ref)[0] for i in idx]
    i = idx[int(np.argmin([np.linalg.norm(c - ref) for c in c0]))]
    c, a1 = nearest_field(i, ref)
    ax = fig.add_subplot(gs[0, m])
    ax.imshow(maps[i].T, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=1)
    ax.annotate('', xy=c + a1, xytext=c, arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3, shrinkA=0, shrinkB=0))
    mid = c + 0.5 * a1
    ax.text(mid[0] + 0.12, mid[1] - 0.05, f'{lam[m]:.2f} m', fontsize=9, color=INK, ha='left', va='center')
    ax.set_title(f'module {m + 1}', fontsize=11, color=INK)
    ax.set_xlabel('x (m)')
    if m == 0:
      ax.set_ylabel('y (m)')
      letter(ax, 'A')
    clean(ax)

  # E: three cells of module 2 differ only in phase (+ inset: all 128 phases in the unit cell)
  ax = fig.add_subplot(gs[1, 0])
  m = 1
  idx = [i for i in pop.gc_idx if pop.module[i] == m][:3]
  xs = np.linspace(-half, half, res)
  for i, col in zip(idx, (BLUE, AQUA, PURPLE)):
    ax.contourf(xs, xs, maps[i].T, levels=[0.45, 1.01], colors=[col], alpha=0.55)
  ax.set_xlim(-half, half)
  ax.set_ylim(-half, half)
  ax.set_aspect('equal')
  ax.set_title('three cells, one module', fontsize=11, color=INK)
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  clean(ax)
  letter(ax, 'B')
  ins = ax.inset_axes([0.63, 0.03, 0.34, 0.34])
  ins.set_facecolor('white')
  ins.set_zorder(5)
  a1, a2 = lattice(m)
  A = np.stack([a1, a2], 1)
  uv = np.linalg.solve(A, phi[pop.module == m].T).T
  poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
  ins.plot(poly[:, 0], poly[:, 1], color=INK2, lw=0.8)
  ins.scatter(uv[:, 0], uv[:, 1], s=3, color=GRAY, lw=0)
  for i, col in zip(idx, (BLUE, AQUA, PURPLE)):
    u = np.linalg.solve(A, phi[i])
    ins.scatter(u[0], u[1], s=22, color=col, lw=0, zorder=3)
  ins.set_xticks([])
  ins.set_yticks([])
  ins.set_aspect('equal')
  for s in ins.spines.values():
    s.set_visible(False)
  ins.set_title('phases', fontsize=8, color=INK2, pad=1)

  # F: the whole code along a straight line through the arena
  ax = fig.add_subplot(gs[1, 1:3])
  line = np.linspace(-half, half, 400)
  pos = torch.tensor(np.stack([line, np.zeros_like(line)], 1), dtype=torch.float32)
  d = torch.tensor([[1.0, 0.0]]).expand(len(line), 2)
  with torch.no_grad():
    act = pop.activity(pos, dir=d, shift_scale=0.0).numpy()                 # [P, N]
  order = []
  for mm in range(M):
    ids = np.where(pop.module == mm)[0]
    order += list(ids[np.argsort(np.argmax(act[:, ids], 0))])
  ax.imshow(act[:, order].T, aspect='auto', cmap='Blues', vmin=0, vmax=1,
            extent=[-half, half, pop.n_cells, 0], interpolation='nearest')
  for mm in range(1, M):
    ax.axhline(mm * pop.n_cells / M, color='white', lw=1.2)
  ax.set_yticks([])
  for mm in range(M):
    ax.text(-half + 0.03, (mm + 0.5) * pop.n_cells / M, f'module {mm + 1}', ha='left', va='center', fontsize=9,
            color=INK, bbox=dict(facecolor='white', alpha=0.85, lw=0, pad=1.5))
  ax.set_xlabel('position along a straight line through the arena (m)')
  ax.set_title('the 512-cell code vs position', fontsize=11, color=INK)
  for s in ax.spines.values():
    s.set_visible(False)
  letter(ax, 'C')

  # G: a grid cell and the predictive grid cell made from it
  ax = fig.add_subplot(gs[1, 3])
  i = [j for j in pop.pgc_idx if pop.module[j] == 1][0]
  ax.imshow(maps[i].T, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=1)
  ax.contour(xs, xs, maps_pgc[i].T, levels=[0.45], colors=[ORANGE], linewidths=1.6)
  ax.annotate('', xy=(0.55, -1.08), xytext=(0.05, -1.08), arrowprops=dict(arrowstyle='->', color=INK, lw=1.4))
  ax.text(0.3, -0.99, 'travel', ha='center', fontsize=8, color=INK)
  ax.text(0.0, 1.14, 'grid cell', transform=ax.transAxes, fontsize=11, color=BLUE, ha='left', va='baseline')
  ax.text(0.0, 1.03, 'predictive, +20 cm', transform=ax.transAxes, fontsize=11, color=ORANGE, ha='left', va='baseline')
  ax.set_xlabel('x (m)')
  clean(ax)
  letter(ax, 'D')

  # H: how the PGCs are allotted to modules, and how far ahead they look, per arrangement
  mods = np.arange(1, M + 1)
  w = 0.36
  ax = fig.add_subplot(gs[2, 0:2])
  ax.bar(mods - w / 2, [43, 42, 42, 42], w, color=ORANGE, label='uniform')
  ax.bar(mods + w / 2, [28, 57, 56, 28], w, color='#9c3f14', label='MEC-weighted (1:2:2:1)')
  ax.set_xticks(mods)
  ax.set_xticklabels([f'module {mm}\n{lam[mm - 1]:.2f} m' for mm in mods])
  ax.set_ylabel('predictive cells (of 128)')
  ax.set_ylim(0, 78)
  ax.legend(fontsize=8.5, loc='upper right')
  letter(ax, 'E')
  ax = fig.add_subplot(gs[2, 2:4])
  ax.bar(mods - w, [20] * M, w * 0.95, color=ORANGE, label='20 cm everywhere')
  ax.bar(mods, 100 * 0.27 * lam, w * 0.95, color='#9c3f14', label='quarter spacing (0.27 λ)')
  ax.bar(mods + w, 100 * 0.5 * lam, w * 0.95, color=GRAY, label='half spacing (0.5 λ)')
  ax.set_xticks(mods)
  ax.set_xticklabels([f'module {mm}\n{lam[mm - 1]:.2f} m' for mm in mods])
  ax.set_ylabel('look-ahead (cm)')
  ax.set_ylim(0, 72)
  ax.legend(fontsize=8.5, loc='upper left')
  letter(ax, 'F')

  fig.savefig(os.path.join(out, 'fig0_grid_modules.png'), bbox_inches='tight')
  fig.savefig(os.path.join(out, 'fig0_grid_modules.svg'), bbox_inches='tight')
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


# ----------------------------------------------- PGC arrangement follow-up
DARK_ORANGE, PURPLE, DARK = '#9c3f14', '#8b5cf6', '#5f5e5a'
ARR_LABEL = {
    'pgc33': 'uniform (reference)',
    'pgc33phase27': 'quarter-spacing shift',
    'pgc33mec': 'MEC-weighted modules',
    'pgc33bio': 'quarter-spacing + heterogeneous',
    'pgc33jitter': 'heterogeneous 20 cm',
    'pgc33coarse': 'coarse modules only',
    'pgc33middle': 'middle modules only',
    'pgc33retro': 'retrospective (20 cm behind)',
    'pgc33phase': 'half-spacing shift',
    'pgc33fine': 'fine modules only',
}
ARR_COLOR = {'pgc33': ORANGE, 'pgc33mec': AQUA, 'pgc33phase27': DARK_ORANGE, 'pgc33bio': PURPLE, 'gc100': BLUE}


def arr_color(cond):
  return ARR_COLOR.get(cond, DARK)


def paired_vals(runs, cond, ref, arm, fn):
  """Per-seed (variant, reference) values for the seeds present in both."""
  seeds = sorted(set(seeds_of(runs, cond, arm)) & set(seeds_of(runs, ref, arm)))
  a = np.array([fn(runs, cond, arm, s) for s in seeds], float)
  b = np.array([fn(runs, ref, arm, s) for s in seeds], float)
  ok = np.isfinite(a) & np.isfinite(b)
  return a[ok], b[ok]


def frames_to(runs, cond, arm, seed, thr=25.0, bin_frames=5e6, budget=300e6):
  """Training frames (millions) until the 5M-frame binned training score first reaches thr."""
  r = runs.get((cond, arm, seed))
  if r is None or r['ep'] is None:
    return np.nan
  edges = np.arange(0, budget + bin_frames, bin_frames)
  b = binned(r['ep'], edges)
  i = np.where(b >= thr)[0]
  return edges[i[0] + 1] / 1e6 if len(i) else budget / 1e6


def pgc_reliance(runs, cond, arm, seed):
  """PGC-specific lesion cost: (intact - no_pgc) - (intact - count-matched sham) = sham - no_pgc."""
  r = runs.get((cond, arm, seed))
  if r is None:
    return np.nan
  e = r['evals']
  if 'no_pgc' not in e or 'sham' not in e:
    return np.nan
  return e['sham']['mean_score'] - e['no_pgc']['mean_score']


def letter_ax(ax, s, x=-0.14, y=1.08):
  ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight='bold', color=INK)


def dot_rows(ax, y, vals_by_row, colors, rng):
  for yi, vals, col in zip(y, vals_by_row, colors):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
      continue
    ax.scatter(vals, np.full(len(vals), yi) + rng.uniform(-0.16, 0.16, len(vals)), s=15, color=col,
               alpha=0.4, edgecolor='none', zorder=2)
    ax.errorbar(vals.mean(), yi, xerr=sem(vals), fmt='o', color=col, ms=6.5, lw=1.6, capsize=0,
                markeredgecolor='white', markeredgewidth=0.8, zorder=4)


def curves(ax, runs, arm, conds, edges, labels=True, lw=2):
  x = (edges[:-1] + edges[1:]) / 2 / 1e6
  handles = []
  for cond, col, label in conds:
    cs = np.array([binned(runs[(cond, arm, s)]['ep'], edges) for s in seeds_of(runs, cond, arm)])
    if not len(cs):
      continue
    mu = np.nanmean(cs, 0)
    se = np.nanstd(cs, 0) / np.sqrt(np.maximum((~np.isnan(cs)).sum(0), 1))
    h, = ax.plot(x, mu, color=col, lw=lw, zorder=3, label=label)
    ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.14, lw=0)
    handles.append(h)
  ax.set_xlabel('training frames (millions)')
  ax.set_xlim(0, 305)
  return handles


def slopegraph(ax, a, b, labels, colors, ylabel):
  """Per-seed slopegraph reference (a) -> variant (b) with group means and the paired difference."""
  for ai, bi in zip(a, b):
    ax.plot([0, 1], [ai, bi], color=GRAY, lw=0.8, alpha=0.6, zorder=1)
  ax.scatter(np.zeros(len(a)), a, s=28, color=colors[0], zorder=3, edgecolor='white', lw=0.6)
  ax.scatter(np.ones(len(b)), b, s=28, color=colors[1], zorder=3, edgecolor='white', lw=0.6)
  for xx, v, c in ((0, a, colors[0]), (1, b, colors[1])):
    ax.hlines(v.mean(), xx - 0.22, xx + 0.22, color=c, lw=3, zorder=4)
    ax.vlines(xx + (0.3 if xx == 0 else -0.3), v.mean() - sem(v), v.mean() + sem(v), color=c, lw=1.4, zorder=4)
  d = b - a
  ax.set_xticks([0, 1])
  ax.set_xticklabels(labels, fontsize=9)
  ax.set_xlim(-0.6, 1.6)
  ax.set_ylabel(ylabel)
  if len(d):
    fmt = '{:+.1f} ± {:.1f}' if abs(d.mean()) >= 0.5 or sem(d) >= 0.5 else '{:+.3f} ± {:.3f}'
    ax.set_title('Δ ' + fmt.format(d.mean(), sem(d)), fontsize=9, color=INK2, pad=6)
  return d


def fig_arrangements(runs, out):
  """Code-only arm: the nine PGC arrangements vs the uniform 33% code, paired by seed.
  A final score, B learning curves, C learning speed, D PGC-specific lesion cost."""
  arm = 'ffb'
  present = [c for c in ARR_LABEL if seeds_of(runs, c, arm)]
  if len(present) < 2:
    return
  means = {c: np.nanmean([score(runs, c, arm, s) for s in seeds_of(runs, c, arm)]) for c in present}
  order = sorted(present, key=lambda c: -means[c])
  y = np.arange(len(order))[::-1]
  rng = np.random.default_rng(7)
  fig = plt.figure(figsize=(12, 8.4))
  gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.42, wspace=0.42)

  # A: final score
  ax = fig.add_subplot(gs[0, 0])
  gc = np.array([score(runs, 'gc100', arm, s) for s in seeds_of(runs, 'gc100', arm)], float)
  gc = gc[np.isfinite(gc)]
  if len(gc):
    ax.axvline(gc.mean(), color=BLUE, ls=(0, (3, 3)), lw=1.2, zorder=0)
    ax.text(gc.mean(), 1.01, 'grid only', color=BLUE, fontsize=8, ha='center', va='bottom',
            transform=ax.get_xaxis_transform())
  if 'pgc33' in means:
    ax.axvline(means['pgc33'], color=ORANGE, lw=1.0, alpha=0.8, zorder=0)
    ax.text(means['pgc33'], 1.01, 'uniform', color=ORANGE, fontsize=8, ha='center', va='bottom',
            transform=ax.get_xaxis_transform())
  dot_rows(ax, y, [[score(runs, c, arm, s) for s in seeds_of(runs, c, arm)] for c in order],
           [arr_color(c) for c in order], rng)
  for yi, c in zip(y, order):
    if c == 'pgc33':
      continue
    a, b = paired_vals(runs, c, 'pgc33', arm, score)
    if len(a):
      ax.text(1.02, yi, f'{(a - b).mean():+.1f}', transform=ax.get_yaxis_transform(), fontsize=8,
              color=INK2, va='center', ha='left')
  ax.text(1.02, 1.01, 'Δ vs uniform', transform=ax.transAxes, fontsize=8, color=INK2, ha='left', va='bottom')
  ax.set_yticks(y)
  ax.set_yticklabels([ARR_LABEL[c] for c in order], fontsize=9)
  ax.set_ylim(-0.7, len(order) - 0.3)
  ax.set_xlabel('final score (200 episodes)')
  letter_ax(ax, 'A', x=-0.42)

  # B: learning curves
  ax = fig.add_subplot(gs[0, 1])
  edges = np.arange(0, 300e6 + 10e6, 10e6)
  x = (edges[:-1] + edges[1:]) / 2 / 1e6
  for c in order:
    if c in ARR_COLOR:
      continue
    cs = np.array([binned(runs[(c, arm, s)]['ep'], edges) for s in seeds_of(runs, c, arm)])
    ax.plot(x, np.nanmean(cs, 0), color=GRAY, lw=0.8, alpha=0.7, zorder=1)
  shown = [(c, ARR_COLOR[c], ARR_LABEL[c] if c != 'pgc33' else 'uniform')
           for c in ('pgc33phase27', 'pgc33mec', 'pgc33bio', 'pgc33') if c in present]
  shown.append(('gc100', BLUE, 'grid only'))
  curves(ax, runs, arm, shown, edges)
  ax.plot([], [], color=GRAY, lw=0.8, label='other arrangements')
  ax.legend(fontsize=8, loc='upper left', handlelength=1.6)
  ax.set_ylabel('episode score (training)')
  letter_ax(ax, 'B')

  # C: learning speed (paired vs uniform)
  ax = fig.add_subplot(gs[1, 0])
  ax.axvline(0, color=ORANGE, lw=1.0, alpha=0.8, zorder=0)
  rows, cols = [], []
  for c in order:
    if c == 'pgc33':
      rows.append([0.0])
    else:
      a, b = paired_vals(runs, c, 'pgc33', arm, frames_to)
      rows.append(a - b)
    cols.append(arr_color(c))
  dot_rows(ax, y, rows, cols, rng)
  ax.set_yticks(y)
  ax.set_yticklabels([ARR_LABEL[c] for c in order], fontsize=9)
  ax.set_ylim(-0.7, len(order) - 0.3)
  ax.set_xlabel('frames to reach a training score of 25, relative to uniform (millions)')
  ax.text(0.02, 0.985, 'faster  ◄', transform=ax.transAxes, fontsize=8, color=INK2, ha='left', va='top')
  ax.text(0.98, 0.985, '►  slower', transform=ax.transAxes, fontsize=8, color=INK2, ha='right', va='top')
  letter_ax(ax, 'C', x=-0.42)

  # D: PGC-specific lesion cost
  ax = fig.add_subplot(gs[1, 1])
  ax.axvline(0, color=GRAY, lw=0.8, zorder=0)
  dot_rows(ax, y, [[pgc_reliance(runs, c, arm, s) for s in seeds_of(runs, c, arm)] for c in order],
           [arr_color(c) for c in order], rng)
  ax.set_yticks(y)
  ax.set_yticklabels([])
  ax.set_ylim(-0.7, len(order) - 0.3)
  ax.set_xlabel('PGC-specific lesion cost (score)')
  letter_ax(ax, 'D')

  fig.savefig(os.path.join(out, 'fig7_arrangements.png'), bbox_inches='tight')
  fig.savefig(os.path.join(out, 'fig7_arrangements.svg'), bbox_inches='tight')
  plt.close(fig)


def fig_arrangement_details(runs, out):
  """One figure per PGC arrangement (figures/arrangements/<cond>.png): the population
  (cells per module, look-ahead per predictive cell), learning curve, paired final score
  and path efficiency against the reference, and test-time lesions."""
  from popcode.common import build_population
  arm = 'ffb'
  sub = os.path.join(out, 'arrangements')
  os.makedirs(sub, exist_ok=True)
  edges = np.arange(0, 300e6 + 10e6, 10e6)
  rng = np.random.default_rng(3)
  lesions = [('intact', 'nothing'), ('sham', '169\nrandom'), ('no_pgc', 'predictive\n(169)'),
             ('no_gc', 'grid\n(343)'), ('no_code', 'whole\ncode')]
  for cond, label in ARR_LABEL.items():
    seeds = seeds_of(runs, cond, arm)
    if not seeds:
      continue
    ref = 'gc100' if cond == 'pgc33' else 'pgc33'
    ref_label = 'grid only' if ref == 'gc100' else 'uniform'
    col, rcol = arr_color(cond), arr_color(ref)
    cfg = runs[(cond, arm, seeds[0])]['cfg']
    pop = build_population(cfg, 'cpu')
    shifts, is_pgc, module, lam = pop._np['shifts'], pop._np['is_pgc'], pop.module, pop.spacings
    M = pop.n_modules
    mods = np.arange(1, M + 1)
    xt = [f'module {m}\n{lam[m - 1]:.2f} m' for m in mods]

    fig = plt.figure(figsize=(12.5, 6.8))
    fig.text(0.01, 0.985, label if cond != 'pgc33' else 'uniform 33 % (reference)', fontsize=13,
             color=col, ha='left', va='top', fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.38, top=0.9)

    # A: cells per module
    ax = fig.add_subplot(gs[0, 0])
    n_pgc = np.array([int((is_pgc & (module == m)).sum()) for m in range(M)])
    n_gc = np.array([int((~is_pgc & (module == m)).sum()) for m in range(M)])
    ax.bar(mods, n_gc, 0.62, color=BLUE, label='grid cells')
    ax.bar(mods, n_pgc, 0.62, bottom=n_gc, color=ORANGE, label='predictive cells')
    for m in range(M):
      if n_pgc[m]:
        ax.text(mods[m], n_gc[m] + n_pgc[m] / 2, str(n_pgc[m]), ha='center', va='center', fontsize=8, color='white')
    ax.set_xticks(mods)
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_ylabel('cells')
    ax.set_ylim(0, 168)
    ax.legend(fontsize=8, loc='upper left', ncol=2, columnspacing=0.8, handlelength=1.2, bbox_to_anchor=(0, 1.02))
    letter_ax(ax, 'A')

    # B: look-ahead per predictive cell
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color=GRAY, lw=0.8, zorder=0)
    ax.scatter(mods, 50 * lam, marker='_', s=260, color=GRAY, lw=1.2, zorder=1, label='half spacing')
    for m in range(M):
      v = 100 * shifts[is_pgc & (module == m)]
      if not len(v):
        continue
      ax.scatter(np.full(len(v), m + 1) + rng.uniform(-0.2, 0.2, len(v)), v, s=9, color=ORANGE, alpha=0.45,
                 edgecolor='none', zorder=2)
      ax.hlines(v.mean(), m + 1 - 0.3, m + 1 + 0.3, color=DARK_ORANGE, lw=2.4, zorder=3)
    ax.set_xticks(mods)
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_xlim(0.4, M + 0.6)
    ax.set_ylabel('look-ahead per predictive cell (cm)')
    ax.legend(fontsize=8, loc='upper left', handlelength=1.2)
    letter_ax(ax, 'B')

    # C: learning curve
    ax = fig.add_subplot(gs[0, 2])
    conds = [(cond, col, label if cond != 'pgc33' else 'uniform')]
    if cond != 'pgc33':
      conds.append(('pgc33', ORANGE, 'uniform'))
    conds.append(('gc100', BLUE, 'grid only'))
    curves(ax, runs, arm, conds, edges)
    ax.legend(fontsize=8, loc='upper left', handlelength=1.6)
    ax.set_ylabel('episode score (training)')
    letter_ax(ax, 'C')

    # D: final score, paired
    ax = fig.add_subplot(gs[1, 0])
    a, b = paired_vals(runs, cond, ref, arm, score)
    slopegraph(ax, b, a, [ref_label, 'this\narrangement'], [rcol, col], 'final score (200 episodes)')
    letter_ax(ax, 'D')

    # E: path efficiency, paired
    ax = fig.add_subplot(gs[1, 1])
    fn = lambda r, c, ar, s: score(r, c, ar, s, key='mean_path_eff')
    a, b = paired_vals(runs, cond, ref, arm, fn)
    slopegraph(ax, b, a, [ref_label, 'this\narrangement'], [rcol, col], 'path efficiency')
    letter_ax(ax, 'E')

    # F: test-time lesions (bars: this arrangement; hollow: reference)
    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(len(lesions))
    for k, (probe, _) in enumerate(lesions):
      v = np.array([score(runs, cond, arm, s, probe) for s in seeds], float)
      v = v[np.isfinite(v)]
      if len(v):
        ax.bar(k, v.mean(), 0.62, color=col, alpha=0.9 if probe == 'intact' else 0.6, zorder=2)
        ax.errorbar(k, v.mean(), yerr=sem(v), fmt='none', ecolor=INK2, lw=1, capsize=2, zorder=3)
      w = np.array([score(runs, ref, arm, s, probe) for s in seeds_of(runs, ref, arm)], float)
      w = w[np.isfinite(w)]
      if len(w):
        ax.scatter(k, w.mean(), s=36, facecolor='white', edgecolor=rcol, lw=1.5, zorder=4)
    ax.scatter([], [], s=36, facecolor='white', edgecolor=rcol, lw=1.5, label=ref_label)
    ax.legend(fontsize=8, loc='upper right', handlelength=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in lesions], fontsize=8)
    ax.set_xlabel('cells removed at test time')
    ax.set_ylabel('score at test time')
    ax.set_ylim(0, None)
    letter_ax(ax, 'F')

    fig.savefig(os.path.join(sub, f'{cond}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(sub, f'{cond}.svg'), bbox_inches='tight')
    plt.close(fig)


def fig_concentration(runs, out):
  """Code-only arm: final score vs fraction of predictive cells."""
  arm = 'ffb'
  series = [('gc100', 0.0), ('pgc10', 0.10), ('pgc25', 0.25), ('pgc30', 0.30), ('pgc33', 0.33),
            ('pgc35', 0.35), ('pgc40', 0.40), ('pgc100', 1.0)]
  pts = []
  for cond, frac in series:
    vals = np.array([score(runs, cond, arm, s) for s in seeds_of(runs, cond, arm)], float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
      pts.append((frac, vals))
  if len(pts) < 3:
    return
  fig, ax = plt.subplots(figsize=(5.2, 3.8))
  for frac, vals in pts:
    ax.scatter(np.full(len(vals), frac * 100), vals, s=16, color=ORANGE, alpha=0.4, edgecolor='none', zorder=2)
  x = [f * 100 for f, _ in pts]
  m = [v.mean() for _, v in pts]
  e = [sem(v) for _, v in pts]
  ax.errorbar(x, m, yerr=e, fmt='-o', color=ORANGE, lw=2, ms=6, capsize=3, markeredgecolor='white', zorder=3)
  n = [score(runs, 'none', arm, s) for s in seeds_of(runs, 'none', arm)]
  n = [v for v in n if v is not None]
  if n:
    ax.axhline(np.mean(n), color=GRAY, ls=(0, (3, 3)), lw=1.2, zorder=0)
    ax.text(99, np.mean(n), 'no code', color=INK2, fontsize=8, va='bottom', ha='right')
  ax.set_xlabel('predictive grid cells (% of the 512-cell code)')
  ax.set_ylabel('score (code-only agent)')
  ax.set_xlim(-3, 103)
  ax.set_ylim(bottom=0)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig8_concentration.png'))
  fig.savefig(os.path.join(out, 'fig8_concentration.svg'))
  plt.close(fig)


def fig_bands(runs, out):
  """Code-only arm: band-cell codes vs grid-cell codes."""
  arm = 'ffb'
  conds = [('gc100', 'grid', BLUE), ('pgc33', 'grid +\n33% PGC', ORANGE), ('pgc100', '100% PGC', ORANGE),
           ('band', 'band', AQUA), ('pband33', 'band +\n33% predictive', AQUA), ('pband100', '100%\npredictive band', AQUA)]
  present = [(c, l, col) for c, l, col in conds if seeds_of(runs, c, arm)]
  if not any(c.startswith('band') or c.startswith('pband') for c, _, _ in present):
    return
  fig, ax = plt.subplots(figsize=(1.3 * len(present) + 2, 4.0))
  for i, (cond, label, col) in enumerate(present):
    vals = np.array([score(runs, cond, arm, s) for s in seeds_of(runs, cond, arm)], float)
    vals = vals[np.isfinite(vals)]
    ax.scatter(np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals)),
               vals, s=22, color=col, alpha=0.6, edgecolor='white', lw=0.5, zorder=2)
    ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=col, lw=3, zorder=3)
    ax.vlines(i + 0.36, vals.mean() - sem(vals), vals.mean() + sem(vals), color=col, lw=1.4, zorder=3)
  n = [score(runs, 'none', arm, s) for s in seeds_of(runs, 'none', arm)]
  n = [v for v in n if v is not None]
  if n:
    ax.axhline(np.mean(n), color=GRAY, ls=(0, (3, 3)), lw=1.2, zorder=0)
    ax.text(len(present) - 0.5, np.mean(n), 'no code', color=INK2, fontsize=8, va='bottom', ha='right')
  ax.set_xticks(range(len(present)))
  ax.set_xticklabels([l for _, l, _ in present], fontsize=8.5)
  ax.set_ylabel('score (code-only agent)')
  ax.set_ylim(bottom=0)
  fig.tight_layout()
  fig.savefig(os.path.join(out, 'fig9_bands.png'))
  fig.savefig(os.path.join(out, 'fig9_bands.svg'))
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default='figures')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  runs = load_runs(args.runs_dir)
  print(f'{len(runs)} runs')
  fig_modules(args.out)
  fig_main(runs, args.out)
  fig_curves(runs, args.out)
  fig_probes(runs, args.out)
  fig_conditions(runs, args.out)
  fig_codes(args.runs_dir, args.out)
  fig_reliance_over_training(args.runs_dir, args.out)
  fig_arrangements(runs, args.out)
  fig_arrangement_details(runs, args.out)
  fig_concentration(runs, args.out)
  fig_bands(runs, args.out)
  print('figures written to', args.out)


if __name__ == '__main__':
  main()
