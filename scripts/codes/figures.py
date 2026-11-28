"""Figures of the code-variation experiment.

Reads results/code_variations/summary.json and data/*.csv (written by
scripts/codes/analyse.py) and writes results/code_variations/figures/*.png and .svg.

  python -m scripts.codes.figures [--only codes,recorded,rooms,inputs,open,door,lattice,curves,learned,lstm,free]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.codes_var import GridPopulation  # noqa: E402
from code.codes_data import GARDNER_PGC_SHIFT_CM, GARDNER_SHIFT_FRAC, GARDNER_SHIFT_M  # noqa: E402
from code.env import BOX_COLOR, PART_COLOR, WALL_COLORS, room_spec  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'code_variations')
for f in glob.glob(os.path.expanduser('~/.local/share/fonts/Times*.TTF')):
  fm.fontManager.addfont(f)
HAVE_TNR = any(f.name == 'Times New Roman' for f in fm.fontManager.ttflist)
# Colours (checked with the palette validator: colour-blind separation and contrast).
BLUE, ORANGE, TEAL, VIOLET = '#2a78d6', '#eb6834', '#00a888', '#8a4fd0'
INK, INK2, GRID, GREY = '#0b0b0b', '#52514e', '#e4e2dc', '#9a9893'
RAMP = ['#84b9f3', '#519ced', '#2370cd', '#0f4a96']        # modules, fine to coarse
GOAL = '#1baf7a'
plt.rcParams.update({
    'font.family': 'Times New Roman' if HAVE_TNR else 'TeX Gyre Termes', 'mathtext.fontset': 'stix',
    'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False, 'savefig.dpi': 220, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'svg.fonttype': 'none'})
HALF = 1.25
SPACING = [40.0, 56.8, 80.7, 114.5]
SHORT = {
    'gc100': 'grid only', 'u20': '20 cm', 'prop10': '10 % of\nspacing', 'prop27': '27 % of\nspacing',
    'gard': 'lab\nrecordings', 'mec': 'Ouchi &\nFujisawa', 'rand': 'random\n0 to 40 cm', 'two': '10 and\n30 cm',
    'side': '20 cm\nsideways', 'lr30': '30° left\nor right', 'p100': 'all cells\n20 cm',
    'gcvar': 'spacing +\norientation\nspread', 'gcvar3': 'spread\n× 3', 'gcshear': 'sheared', 'gcjit': 'jittered\nfields',
    'mecreal': 'Ouchi &\nFujisawa\n+ all three', 'm1': 'one\nmodule', 'm1var': 'one module\n+ spread', 'none': 'no code'}
HAS_PGC = {'u20', 'prop10', 'prop27', 'gard', 'mec', 'rand', 'two', 'side', 'lr30', 'p100', 'mecreal'}


def colour(code):
  return GREY if code == 'none' else ORANGE if code in HAS_PGC else BLUE


def save(fig, name):
  out = os.path.join(OUT, 'figures')
  os.makedirs(out, exist_ok=True)
  for ext in ('png', 'svg'):
    fig.savefig(os.path.join(out, f'{name}.{ext}'), bbox_inches='tight')
  plt.close(fig)
  print('wrote', os.path.join(out, name + '.png'))


def summary():
  return json.load(open(os.path.join(OUT, 'summary.json')))


def read_csv(name):
  with open(os.path.join(OUT, 'data', name)) as f:
    return list(csv.DictReader(f))


# ------------------------------------------------------------- 1. the codes
def population(code, seed=1):
  kw = dict(seed=seed, pgc_frac=0.33)
  if code == 'prop10':
    kw.update(pgc_shift_rule='spacing', pgc_shift_frac=0.10)
  elif code == 'prop27':
    kw.update(pgc_shift_rule='spacing', pgc_shift_frac=0.27)
  elif code == 'mec':
    kw.update(pgc_shift_dist='tnorm', pgc_shift_dist_args=[0.27, 0.11, 0.01, 0.5])
  elif code == 'gard':
    kw.update(pgc_shift_dist='empm', pgc_shift_dist_args=list(GARDNER_SHIFT_M))
  elif code == 'rand':
    kw.update(pgc_shift_dist='unif', pgc_shift_dist_args=[0.0, 0.4])
  elif code == 'two':
    kw.update(pgc_shift_dist='set', pgc_shift_dist_args=[0.1, 0.3])
  return GridPopulation(**kw)


def fig_codes():
  """What each code changes: look-aheads per module (top), direction and lattice (bottom)."""
  codes = ['u20', 'prop10', 'prop27', 'gard', 'mec', 'rand', 'two']
  titles = {'u20': '20 cm', 'prop10': '10 % of spacing', 'prop27': '27 % of spacing', 'gard': 'lab recordings',
            'mec': 'Ouchi & Fujisawa', 'rand': 'random 0 to 40 cm', 'two': '10 and 30 cm'}
  fig = plt.figure(figsize=(11.0, 5.2))
  gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.28)
  rng = np.random.default_rng(0)
  for j, code in enumerate(codes):
    ax = fig.add_subplot(gs[0, j])
    p = population(code)
    sh = p.shift.numpy() * 100
    pg = p._np['is_pgc']
    for m in range(4):
      v = sh[(p.module == m) & pg]
      ax.scatter(m + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color=ORANGE, alpha=0.55, lw=0)
      ax.plot([m - 0.33, m + 0.33], [SPACING[m] / 2] * 2, color=GREY, lw=0.8)
    ax.set_ylim(-2, 60)
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['40', '57', '81', '115'], fontsize=8.5)
    ax.set_title(titles[code], fontsize=10.5, color=INK)
    if j == 0:
      ax.set_ylabel('look-ahead (cm)')
    else:
      ax.set_yticklabels([])
    if j == 3:
      ax.set_xlabel('grid spacing of the module (cm)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)

  # direction of the look-ahead
  ax = fig.add_subplot(gs[1, 0:3])
  ax.set_xlim(-0.55, 4.0)
  ax.set_ylim(-0.95, 0.85)
  ax.set_aspect('equal')
  ax.axis('off')
  for k, (name, vecs) in enumerate([('ahead', [(0, 0.55)]), ('sideways', [(-0.55, 0)]),
                                    ('30° left or right', [(-0.275, 0.476), (0.275, 0.476)])]):
    x0 = 0.25 + 1.55 * k
    ax.plot([x0, x0], [-0.42, 0.72], color=GRID, lw=1.2)
    ax.plot([x0], [0], 'o', ms=7, color=INK2)
    for dx, dy in vecs:
      ax.annotate('', xy=(x0 + dx, dy), xytext=(x0, 0),
                  arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=2.0, shrinkA=3, shrinkB=0))
    ax.text(x0, -0.6, name, ha='center', va='top', fontsize=10.5, color=INK)
  ax.text(0.33, 0.6, 'direction\nof travel', ha='left', va='center', fontsize=8.5, color=INK2)

  # lattice variations: firing fields of example cells (module 3, spacing 81 cm)
  from scipy import ndimage

  def fields(pop, cell, res=260):
    r = pop.ratemaps(res=res, half=HALF, shift_scale=0.0)[cell]
    mx = ndimage.maximum_filter(r, size=9)
    ii, jj = np.where((r == mx) & (r > 0.55))
    xs = np.linspace(-HALF, HALF, res)
    return xs[ii], xs[jj]

  base = GridPopulation(seed=1, pgc_frac=0.0)
  cells = [i for i in range(512) if base.module[i] == 2][:3]
  panels = [('ideal lattice', base, None),
            ('spacing + orientation\nspread', GridPopulation(seed=1, pgc_frac=0.0, spacing_sd=0.06, orient_sd_deg=1.0), 'spread'),
            ('sheared', GridPopulation(seed=1, pgc_frac=0.0, lattice_shear=0.13), None),
            ('jittered fields', GridPopulation(seed=1, pgc_frac=0.0, field_jitter=0.06), None)]
  for k, (title, pop, kind) in enumerate(panels):
    ax = fig.add_subplot(gs[1, 3 + k])
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(True)
      s.set_color(INK2)
    c = cells[0]
    bx, by = fields(base, c)
    ax.scatter(bx, by, s=46, facecolors='none', edgecolors=GREY, lw=0.9)
    if kind == 'spread':
      # three cells of the module, each moved so that its field nearest the centre sits on the centre
      for cc, col in zip(cells, RAMP[1:]):
        x, y = fields(pop, cc)
        k0 = np.argmin(x ** 2 + y ** 2)
        b0 = np.argmin(bx ** 2 + by ** 2)
        ax.scatter(x - x[k0] + bx[b0], y - y[k0] + by[b0], s=13, color=col, lw=0)
    elif k > 0:
      x, y = fields(pop, c)
      ax.scatter(x, y, s=15, color=BLUE, lw=0)
    else:
      ax.scatter(bx, by, s=15, color=BLUE, lw=0)
    ax.set_title(title, fontsize=10.5, color=INK)
  save(fig, 'codes')


# ------------------------------------------------- 2. recorded look-aheads
def fig_recorded():
  from scipy import stats
  fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), gridspec_kw=dict(wspace=0.28))
  cm = np.array([s for v in GARDNER_PGC_SHIFT_CM.values() for s in v], float)
  ax = axes[0]
  bins = np.arange(0, 58, 2) + 1.0
  ax.hist(cm, bins=bins, color=TEAL, alpha=0.9, rwidth=0.86, label='lab analysis of the Gardner\nrecordings, 108 cells')
  x = np.linspace(0, 56, 300)
  scale = len(cm) * 2.0
  ax.plot(x, scale * stats.norm.pdf(x, 22.5, 10.2), color=VIOLET, lw=2.0,
          label='Ouchi & Fujisawa (bell curve with\ntheir average and spread)')
  ax.set_xlabel('look-ahead of a predictive grid cell (cm)')
  ax.set_ylabel('cells')
  ax.legend(fontsize=9, loc='upper right')
  ax.set_xlim(0, 56)
  ax = axes[1]
  fr = np.array(GARDNER_SHIFT_FRAC)
  b = np.arange(0, 0.56, 0.025)
  ax.hist(fr, bins=b, color=TEAL, alpha=0.9, rwidth=0.86)
  x = np.linspace(0, 0.55, 300)
  ax.plot(x, len(fr) * 0.025 * stats.norm.pdf(x, 0.27, 0.11), color=VIOLET, lw=2.0)
  ymax = ax.get_ylim()[1]
  for v, t in ((0.10, '10 %'), (0.27, '27 %')):
    ax.plot([v, v], [0, ymax * 0.93], color=INK2, lw=0.9)
    ax.text(v, ymax * 0.95, t, ha='center', va='bottom', fontsize=9.5, color=INK)
  ax.set_xlabel('look-ahead as a fraction of grid spacing')
  ax.set_xlim(0, 0.55)
  save(fig, 'recorded_shifts')


# ---------------------------------------------------------------- 3. rooms
def draw_room(ax, layout, lw=2.4):
  ax.set_xlim(-HALF - 0.06, HALF + 0.06)
  ax.set_ylim(-HALF - 0.06, HALF + 0.06)
  ax.set_aspect('equal')
  ax.axis('off')
  ends = [((HALF, -HALF), (HALF, HALF)), ((-HALF, HALF), (HALF, HALF)),
          ((-HALF, -HALF), (-HALF, HALF)), ((-HALF, -HALF), (HALF, -HALF))]
  for (p, q), c in zip(ends, WALL_COLORS):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, solid_capstyle='butt')
  for k, b in enumerate(room_spec(layout)['boxes']):
    if abs(b[0]) < 50:
      ax.add_patch(plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], lw=0,
                                 color=[x * 0.8 for x in PART_COLOR] if k < 2 else BOX_COLOR))


def fig_rooms():
  fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
  for ax, layout, title, goal, agent, hd in ((axes[0], 'open', 'open room', (0.55, -0.45), (-0.5, 0.35), -0.6),
                                              (axes[1], 'DO1', 'door room', (-0.85, 0.55), (0.1, -0.75), 2.6)):
    draw_room(ax, layout)
    ax.add_patch(plt.Circle(goal, 0.30, color=GOAL, alpha=0.30, lw=0))
    ax.add_patch(plt.Circle(goal, 0.30, fill=False, color=GOAL, lw=1.3))
    ax.plot(*agent, 'o', ms=6, color=INK)
    ax.annotate('', xy=(agent[0] + 0.2 * np.cos(hd), agent[1] + 0.2 * np.sin(hd)), xytext=agent,
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.8, shrinkA=0, shrinkB=0))
    ax.set_title(title, fontsize=11, color=INK)
  save(fig, 'rooms')


# --------------------------------------------------------------- 4. inputs
def fig_inputs():
  parts = [('vision', 256, '#a9a7a1'), ('code, now', 512, '#77756f'), ('code at the goal', 512, '#4f4e4a'),
           ('last action', 6, INK), ('last reward', 1, INK)]
  fig, ax = plt.subplots(figsize=(8.2, 1.25))
  x = 0
  for name, n, c in parts:
    ax.barh(0, n - 4, left=x + 2, height=0.5, color=c, lw=0)
    if n > 50:
      ax.text(x + n / 2, 0, f'{name}\n{n}', ha='center', va='center', fontsize=10, color='white')
    x += n
  ax.annotate('last action 6\nlast reward 1', xy=(1284, 0.26), xytext=(1284, 0.62), ha='right', fontsize=9.5,
              color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7))
  ax.text(-12, 0, '1,287\ninputs', ha='right', va='center', fontsize=10.5, color=INK)
  ax.annotate('', xy=(1420, 0), xytext=(1296, 0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.2))
  ax.text(1430, 0, 'LSTM\n256 units', ha='left', va='center', fontsize=10.5, color=INK)
  ax.set_xlim(-130, 1560)
  ax.set_ylim(-0.4, 0.9)
  ax.axis('off')
  save(fig, 'lstm_inputs')


# ----------------------------------------------------- 5. scores by code
def score_panel(fig, gs_top, gs_bot, blk, codes, ref, ylabel, show_ref_lines=True):
  ax = fig.add_subplot(gs_top)
  codes = [c for c in codes if c in blk['codes'] and blk['codes'][c]['n']]
  for i, c in enumerate(codes):
    g = blk['codes'][c]
    v = np.array(list(g['seeds'].values()))
    ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v, s=13, color=colour(c), alpha=0.55, lw=0)
    ax.plot([i - 0.27, i + 0.27], [g['mean']] * 2, color=colour(c), lw=2.6, solid_capstyle='round')
    if g['sem'] is not None:
      ax.plot([i, i], [g['mean'] - g['sem'], g['mean'] + g['sem']], color=colour(c), lw=1.1)
  if show_ref_lines:
    for r, col in (('gc100', BLUE), ('u20', ORANGE)):
      if r in blk['codes'] and blk['codes'][r]['mean'] is not None:
        ax.axhline(blk['codes'][r]['mean'], color=col, lw=0.7, alpha=0.6, zorder=0)
  ax.set_xticks(range(len(codes)))
  ax.set_xticklabels([])
  ax.set_xlim(-0.6, len(codes) - 0.4)
  ax.set_ylabel(ylabel)
  ax.grid(axis='y', color=GRID, lw=0.6)
  ax.set_axisbelow(True)
  axb = fig.add_subplot(gs_bot, sharex=ax)
  axb.axhline(0, color=INK2, lw=0.8)
  for i, c in enumerate(codes):
    t = blk['vs'].get(ref, {}).get(c)
    if c in (ref, 'none') or t is None or t.get('diff') is None:
      continue
    d = np.array(t['seeds'])
    axb.scatter(np.full(len(d), i) + np.linspace(-0.16, 0.16, len(d)), d, s=11, color=colour(c), alpha=0.5, lw=0)
    axb.plot([i - 0.27, i + 0.27], [t['diff']] * 2, color=colour(c), lw=2.6, solid_capstyle='round')
    if t.get('ci95') is not None:
      axb.plot([i, i], [t['diff'] - t['ci95'], t['diff'] + t['ci95']], color=INK, lw=1.1)
  axb.set_xticks(range(len(codes)))
  axb.set_xticklabels([SHORT.get(c, c) for c in codes], fontsize=9.5)
  axb.tick_params(axis='x', length=0)
  axb.set_ylabel(f'difference to\n"{SHORT[ref].replace(chr(10), " ")}"')
  axb.grid(axis='y', color=GRID, lw=0.6)
  axb.set_axisbelow(True)
  plt.setp(ax.get_xticklabels(), visible=False)
  return ax, axb


def fig_scores(S, env, mark, codes, name, ref='u20', width=None):
  blk = S[env].get(str(mark))
  if not blk:
    print(f'no data for {env} at {mark} M')
    return
  codes = [c for c in codes if c in blk['codes']]
  fig = plt.figure(figsize=(width or (1.6 + 0.92 * len(codes)), 4.6))
  gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.12)
  score_panel(fig, gs[0], gs[1], blk, codes, ref, 'score')
  save(fig, name)


# ---------------------------------------------------- 6. learning curves
def curves_by(rows, env, arm, codes):
  out = {}
  for r in rows:
    code, e, a, seed = r['run'].rsplit('_', 3)[0], r['run'].split('_')[-3], r['run'].split('_')[-2], r['run'].split('_')[-1]
    if e != env or a != arm or code not in codes:
      continue
    out.setdefault(code, {}).setdefault(seed, {})[int(r['frames_hi_M'])] = float(r['training_score'])
  return out


def draw_curves(ax, data, codes, cols, labels, min_agents=3):
  for c in codes:
    if c not in data:
      continue
    xs = sorted(set.union(*[set(v) for v in data[c].values()]))
    x, m, s = [], [], []
    for xx in xs:
      v = [d[xx] for d in data[c].values() if xx in d]
      if len(v) >= min(min_agents, len(data[c])):
        x.append(xx - 5)
        m.append(np.mean(v))
        s.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    if not x:
      continue
    x, m, s = np.array(x), np.array(m), np.array(s)
    ax.fill_between(x, m - s, m + s, color=cols[c], alpha=0.12, lw=0)
    ax.plot(x, m, color=cols[c], lw=2.0, solid_capstyle='round')
    ax.plot([x[-1]], [m[-1]], 'o', ms=4.5, color=cols[c])
  ax.grid(axis='y', color=GRID, lw=0.6)
  ax.set_axisbelow(True)


def fig_curves(S):
  rows = read_csv('curves.csv')
  cols = dict(gc100=BLUE, u20=ORANGE, prop10=TEAL, mec=VIOLET, none=GREY)
  labels = dict(gc100='grid only', u20='20 cm', prop10='10 % of spacing', mec='Ouchi & Fujisawa', none='no code')
  fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw=dict(wspace=0.42))
  for ax, env, title, codes in ((axes[0], 'open', 'open room', ['gc100', 'u20', 'prop10', 'mec']),
                                (axes[1], 'door', 'door room', ['none', 'gc100', 'u20', 'prop10', 'mec'])):
    data = curves_by(rows, env, 'ffb', codes)
    xmax = max([max(d) for v in data.values() for d in v.values()] or [600])
    ax.set_xlim(0, xmax)
    ymax = max([max(d.values()) for v in data.values() for d in v.values()] or [1])
    ax.set_ylim(0, ymax * 0.95)
    draw_curves(ax, data, codes, cols, labels)
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    ax.set_xlabel('training frames (million)')
    ax.set_ylabel('score')
    handles = [plt.Line2D([], [], color=cols[c], lw=2.0, label=labels[c]) for c in codes if c in data]
    ax.legend(handles=handles, fontsize=9, loc='lower right', handlelength=1.4)
  save(fig, 'curves')


def fig_spread_curves(S):
  """Score over training for the codes with and without spread: three small panels with one y axis.
  Blue: no predictive cells, orange: predictive cells; darker = more spread. Lines are named at their right end."""
  rows = read_csv('curves.csv')
  DARK, DARKER, BROWN = '#0f4a96', '#082a57', '#9a3412'
  panels = [('four modules', [('gc100', BLUE, 'grid only'), ('gcvar', DARK, 'spread'), ('gcvar3', DARKER, 'spread × 3')]),
            ('one module', [('m1', BLUE, 'no spread'), ('m1var', DARK, 'spread')]),
            ('Ouchi & Fujisawa look-aheads', [('mec', ORANGE, 'ideal grid'), ('mecreal', BROWN, 'spread, shear and\nmoved fields')])]
  fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=True, gridspec_kw=dict(wspace=0.10))
  ymax, xmax = 1.0, 300
  datas = []
  for _, lines in panels:
    d = curves_by(rows, 'open', 'ffb', [c for c, _, _ in lines])
    datas.append(d)
    for v in d.values():
      for dd in v.values():
        ymax, xmax = max(ymax, max(dd.values())), max(xmax, max(dd))
  for ax, (title, lines), data in zip(axes, panels, datas):
    cols = {c: col for c, col, _ in lines}
    draw_curves(ax, data, [c for c, _, _ in lines], cols, None)
    ends = []
    for c, col, lab in lines:
      if c not in data:
        continue
      xs = sorted(set.union(*[set(v) for v in data[c].values()]))
      full = [x for x in xs if sum(x in d for d in data[c].values()) >= min(3, len(data[c]))]
      if full:
        ends.append([np.mean([d[full[-1]] for d in data[c].values() if full[-1] in d]), full[-1] - 5, lab, col])
    ends.sort(key=lambda e: e[0])
    gap = 0.085 * ymax                       # keep the names apart when two lines end close together
    for i in range(1, len(ends)):
      ends[i].append(max(ends[i][0], ends[i - 1][-1] + gap) if len(ends[i - 1]) > 4 else max(ends[i][0], ends[i - 1][0] + gap))
    for i, e in enumerate(ends):
      ypos = e[4] if len(e) > 4 else e[0]
      ax.text(e[1] + 0.02 * xmax, ypos, e[2], ha='left', va='center', fontsize=9.5, color=INK, linespacing=0.95)
    ax.set_xlim(0, xmax * 1.36)
    ax.set_xticks([x for x in (0, 100, 200, 300, 400, 500, 600) if x <= xmax])
    ax.set_ylim(0, ymax * 1.02)
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    ax.set_xlabel('training frames (million)')
    if xmax > 320:
      ax.axvline(300, color=GREY, lw=0.8, ls=(0, (2, 2)), zorder=0)
  axes[0].set_ylabel('score')
  save(fig, 'spread_curves')


# ------------------------------------------------ 7. trainable look-aheads
def fig_learned(S):
  rows = read_csv('learned_lookaheads.csv')
  if not rows:
    print('no learned look-aheads yet')
    return
  data = {}
  for r in rows:
    key = (r['start'], r['room'])
    data.setdefault(key, {}).setdefault(r['seed'], []).append(
        [float(r['frames_M'])] + [float(r[f'module{m}_cm']) for m in range(1, 5)])
  fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.0), sharex=True, gridspec_kw=dict(hspace=0.28, wspace=0.26))
  for i, room in enumerate(('open', 'door')):
    for m in range(4):
      ax = axes[i, m]
      for frac, ls in ((0.10, (0, (1, 2))), (0.27, (0, (4, 2)))):
        ax.axhline(frac * SPACING[m], color=GREY, lw=0.9, ls=ls, zorder=0)
      for start, col, lab in (('lzero', BLUE, 'start at 0 cm'), ('lu20', ORANGE, 'start at 20 cm'),
                              ('lzeroX', GREY, None), ('lu20X', GREY, None)):
        runs = data.get((start, room), {})
        for seed, tr in runs.items():
          a = np.array(tr)
          ax.plot(a[:, 0], a[:, 1 + m], color=col, lw=0.9 if start.endswith('X') else 1.5,
                  alpha=0.9 if start.endswith('X') else 0.75)
      ax.grid(axis='y', color=GRID, lw=0.6)
      ax.set_axisbelow(True)
      if i == 0:
        ax.set_title(f'module with {SPACING[m]:.0f} cm spacing', fontsize=10.5, color=INK)
      if m == 0:
        ax.set_ylabel(f'{room} room\nlook-ahead (cm)')
      if i == 1:
        ax.set_xlabel('training frames (million)')
  handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label='start at 0 cm'),
             plt.Line2D([], [], color=ORANGE, lw=1.6, label='start at 20 cm'),
             plt.Line2D([], [], color=GREY, lw=1.0, label='control: gradient sign scrambled'),
             plt.Line2D([], [], color=GREY, lw=0.9, ls=(0, (1, 2)), label='10 % of spacing'),
             plt.Line2D([], [], color=GREY, lw=0.9, ls=(0, (4, 2)), label='27 % of spacing')]
  fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.045))
  save(fig, 'learned_lookaheads')


# ------------------------------------------------------- 8. LSTM agents
def fig_lstm(S):
  rows = read_csv('curves.csv')
  cols = dict(gc100=BLUE, u20=ORANGE, prop10=TEAL, mec=VIOLET, gcvar3='#082a57', mecreal='#9a3412')
  labels = dict(gc100='grid only', u20='20 cm', prop10='10 % of spacing', mec='Ouchi & Fujisawa',
                gcvar3='spread × 3', mecreal='Ouchi & Fujisawa + all three')
  codes = ['gc100', 'u20', 'prop10', 'mec']
  last = S['lstm'].get(str(max(S['lstm_marks'])), {}).get('codes', {})
  codes += [c for c in ('gcvar3', 'mecreal') if last.get(c, {}).get('n', 0) >= 8]      # follow-on block lx, once complete
  data = curves_by(rows, 'mix', 'lstm', codes)
  if not data:
    print('no LSTM curves yet')
    return
  codes = [c for c in codes if c in data]
  fig = plt.figure(figsize=(9.6 + 0.6 * max(0, len(codes) - 4), 3.6))
  gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0 + 0.2 * max(0, len(codes) - 4)], wspace=0.42)
  ax = fig.add_subplot(gs[0])
  xmax = max(max(d) for v in data.values() for d in v.values())
  ymax = max(max(d.values()) for v in data.values() for d in v.values())
  ax.set_xlim(0, xmax)
  ax.set_ylim(0, ymax * 0.95)
  draw_curves(ax, data, codes, cols, labels)
  ax.set_xlabel('training frames (million)')
  ax.set_ylabel('score in the mix of changed rooms')
  handles = [plt.Line2D([], [], color=cols[c], lw=2.0, label=labels[c]) for c in codes if c in data]
  ax.legend(handles=handles, fontsize=9, loc='lower right', handlelength=1.4)
  ax = fig.add_subplot(gs[1])
  marks = sorted([k for k in S['lstm'] if k.isdigit()], key=int)
  if marks:
    blk = S['lstm'][marks[-1]]
    cs = [c for c in codes if c in blk['codes']]
    for i, c in enumerate(cs):
      g = blk['codes'][c]
      v = np.array(list(g['seeds'].values()))
      ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v, s=15, color=cols[c], alpha=0.6, lw=0)
      ax.plot([i - 0.27, i + 0.27], [g['mean']] * 2, color=cols[c], lw=2.6, solid_capstyle='round')
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([SHORT[c] for c in cs], fontsize=9.5)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(cs) - 0.4)
    ax.set_ylabel(f'score at {marks[-1]} M frames')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
  save(fig, 'lstm_rooms')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--only', default='codes,recorded,rooms,inputs,open,door,lattice,curves,learned,lstm')
  ap.add_argument('--open_mark', type=int, default=None)
  ap.add_argument('--screen_mark', type=int, default=None)
  ap.add_argument('--dir', default=None, help='read summary.json and data/ from here and write figures/ here')
  args = ap.parse_args()
  if args.dir:
    global OUT
    OUT = os.path.abspath(args.dir)
  only = set(args.only.split(','))
  if 'codes' in only:
    fig_codes()
  if 'recorded' in only:
    fig_recorded()
  if 'rooms' in only:
    fig_rooms()
  if 'inputs' in only:
    fig_inputs()
  if not os.path.exists(os.path.join(OUT, 'summary.json')):
    return
  S = summary()
  top = args.open_mark or max(S['marks'])
  low = args.screen_mark or min(S['marks'])
  if 'open' in only:
    fig_scores(S, 'open', top, ['gc100', 'u20', 'prop10', 'prop27', 'gard', 'mec', 'rand', 'side'], 'open_shifts')
  if 'lattice' in only:
    fig_scores(S, 'open', low, ['gc100', 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'mec', 'mecreal', 'm1', 'm1var'],
               'open_lattice', ref='gc100')
    fig_scores(S, 'open', low, ['gc100', 'u20', 'two', 'lr30', 'p100'], 'open_more_shifts')
    if 'gcvar3' in S['open'].get(str(top), {}).get('codes', {}):      # the screen codes continued (block ox)
      fig_scores(S, 'open', top, ['gc100', 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'mec', 'mecreal', 'm1', 'm1var'],
                 'open_lattice_600', ref='gc100')
  if 'door' in only:
    dx = [c for c in ('gcvar3', 'mecreal')       # follow-on block dx, once complete
          if S['door'].get(str(top), {}).get('codes', {}).get(c, {}).get('n', 0) >= 8]
    fig_scores(S, 'door', top, ['none', 'gc100', 'u20', 'prop10', 'gard', 'mec', 'rand'] + dx, 'door_shifts')
  if 'curves' in only:
    fig_curves(S)
    fig_spread_curves(S)
  if 'learned' in only:
    fig_learned(S)
  if 'lstm' in only:
    fig_lstm(S)


if __name__ == '__main__':
  main()
