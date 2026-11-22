"""Figures of the room-change experiment (reads results/room_change/summary.json and
results_room/eval/*.npz; writes results/room_change/figures/*.png and .svg).

  python -m scripts.room.figures [--only layouts,scores,finetune,pca,gap,paths]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.env import (BOX_COLOR, HELDOUT_LAYOUTS, PART_COLOR, TRAIN_LAYOUTS, WALL_COLORS,  # noqa: E402
                      room_spec)

for f in glob.glob(os.path.expanduser('~/.local/share/fonts/Times*.TTF')):
  fm.fontManager.addfont(f)
HAVE_TNR = any(f.name == 'Times New Roman' for f in fm.fontManager.ttflist)
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e2dc'
COL = {'gc100': BLUE, 'pgc33': ORANGE}
LAB = {'gc100': 'grid', 'pgc33': 'grid + PGC'}
plt.rcParams.update({
    'font.family': 'Times New Roman' if HAVE_TNR else 'TeX Gyre Termes', 'mathtext.fontset': 'stix',
    'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False, 'savefig.dpi': 220, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'svg.fonttype': 'none'})
HALF = 1.25


def save(fig, out, name):
  os.makedirs(out, exist_ok=True)
  for ext in ('png', 'svg'):
    fig.savefig(os.path.join(out, f'{name}.{ext}'), bbox_inches='tight')
  plt.close(fig)
  print('wrote', os.path.join(out, name + '.png'))


def draw_room(ax, layout, lw=2.2):
  ax.set_xlim(-HALF - 0.06, HALF + 0.06)
  ax.set_ylim(-HALF - 0.06, HALF + 0.06)
  ax.set_aspect('equal')
  ax.axis('off')
  ends = [((HALF, -HALF), (HALF, HALF)), ((-HALF, HALF), (HALF, HALF)),
          ((-HALF, -HALF), (-HALF, HALF)), ((-HALF, -HALF), (HALF, -HALF))]     # E, N, W, S
  for (p, q), c in zip(ends, WALL_COLORS):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, solid_capstyle='butt')
  for k, b in enumerate(room_spec(layout)['boxes']):
    if abs(b[0]) < 50:
      ax.add_patch(plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], lw=0,
                                 color=[x * 0.8 for x in PART_COLOR] if k < 2 else BOX_COLOR))


def fig_layouts(out):
  import torch
  from code.env import ArenaEnv
  names = ['open'] + list(TRAIN_LAYOUTS) + list(HELDOUT_LAYOUTS)
  fig = plt.figure(figsize=(10.5, 5.6))
  gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 1.5], hspace=0.28, wspace=0.12)
  for i, n in enumerate(names):
    ax = fig.add_subplot(gs[i // 8, i % 8])
    draw_room(ax, n)
    ax.set_title(n, fontsize=10, color=INK, pad=3)
  ax = fig.add_subplot(gs[1, 6])          # goal relocation
  draw_room(ax, 'open')
  ax.add_patch(plt.Circle((-0.55, 0.45), 0.30, fill=False, ec=INK2, lw=1.0, ls=(0, (3, 2))))
  ax.add_patch(plt.Circle((0.55, -0.5), 0.30, color='#1baf7a', lw=0))
  ax.annotate('', xy=(0.33, -0.3), xytext=(-0.33, 0.25), arrowprops=dict(arrowstyle='->', color=INK2, lw=1.0))
  ax.set_title('+R', fontsize=10, color=INK, pad=3)
  dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  views = [('open', (-0.5, 0.0), 0), ('D1', (-0.8, -0.6), 20), ('O1', (-0.9, -0.9), 45), ('DO2', (-0.8, -0.8), 60)]
  for j, (n, pos, hd) in enumerate(views):
    env = ArenaEnv(1, device=dev, seed=1, rooms=None if n == 'open' else [n], img_noise=0.0)
    env.reset_all()
    env.pos[0] = torch.tensor(pos, device=dev)
    env.hd[0] = math.radians(hd)
    img = env.render()[0].permute(1, 2, 0).cpu().numpy()
    ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
    ax.imshow(img, interpolation='nearest')
    ax.axis('off')
    ax.set_title(n, fontsize=10, color=INK, pad=3)
  save(fig, out, 'rooms_layouts')


def bars(ax, st_by_group, groups, ticks, ylabel=None, legend=False):
  w = 0.34
  for gi, g in enumerate(groups):
    st = st_by_group[g]
    for c, off in (('gc100', -w / 2 - 0.02), ('pgc33', w / 2 + 0.02)):
      key = 'gc' if c == 'gc100' else 'pgc'
      ax.bar(gi + off, st[key], width=w, color=COL[c], lw=0, label=LAB[c] if legend and gi == 0 else None, zorder=2)
      ax.errorbar(gi + off, st[key], yerr=st[key + '_sem'], color=INK, lw=0.9, capsize=0, zorder=4)
      seeds = np.array(st[key + '_seeds'])
      ax.scatter(np.full(len(seeds), gi + off) + np.linspace(-0.09, 0.09, len(seeds)), seeds, s=7, color='white',
                 edgecolor=INK2, lw=0.5, zorder=3)
  ax.set_xticks(range(len(groups)))
  ax.set_xticklabels(ticks, fontsize=10)
  ax.tick_params(axis='x', length=0)
  ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
  ax.set_axisbelow(True)
  if ylabel:
    ax.set_ylabel(ylabel)
  if legend:
    ax.legend(loc='upper right', fontsize=10, handlelength=1.0)


def state_titles(S):
  extra = (S.get('long_m') or 0) - 1000
  return {'before': 'before', 'after': 'after 250 M frames', 'long': f'after {extra} M frames'}


def fig_scores(S, out):
  groups = ['familiar', 'door', 'obstacle', 'door_obstacle', 'goal_moved', 'layout_and_goal_moved']
  ticks = ['familiar', 'door', 'obstacle', 'door +\nobstacle', 'goal\nmoved', 'door / obstacle\n+ goal moved']
  states = [s for s in ('before', 'after', 'long') if s in S['stats']]
  tt = state_titles(S)
  fig, axs = plt.subplots(1, len(states), figsize=(5.4 * len(states), 3.5), sharey=True, squeeze=False)
  for ax, state in zip(axs[0], states):
    bars(ax, {g: S['stats'][state][g]['score'] for g in groups}, groups, ticks,
         ylabel='score' if state == states[0] else None, legend=state == states[0])
    ax.set_title(tt[state], fontsize=11, color=INK)
    ax.set_ylim(0, None)
  fig.subplots_adjust(wspace=0.06)
  save(fig, out, 'rooms_scores')


def band(ax, cv, lw=2.0, alpha=0.14, **kw):
  x = np.array(cv['x']) / 1000.0
  m = np.array(cv['mean'])
  ax.plot(x, m, lw=lw, solid_capstyle='round', **kw)
  if cv.get('sem') is not None:
    s = np.array(cv['sem'])
    ax.fill_between(x, m - s, m + s, color=kw.get('color'), alpha=alpha, lw=0)


def fig_finetune(S, out):
  C = S['curves']
  if 'room/gc100/ret_all' not in C:
    return
  fig = plt.figure(figsize=(10.5, 5.6))
  gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.42, wspace=0.32)
  ax = fig.add_subplot(gs[0, 0:3])
  for c in ('gc100', 'pgc33'):
    band(ax, C[f'room/{c}/ret_all'], color=COL[c], label=LAB[c])
  ax.set_title('changed rooms', fontsize=11)
  ax.set_ylabel('score')
  ax.legend(loc='lower right', fontsize=10, handlelength=1.2)
  ax2 = fig.add_subplot(gs[0, 3])
  for c in ('gc100', 'pgc33'):
    k = f'same/{c}/ret_all'
    if k in C:
      band(ax2, C[k], color=COL[c])
  ax2.set_title('familiar room', fontsize=11)
  for j, (key, t) in enumerate((('ret_door', 'door'), ('ret_obstacle', 'obstacle'),
                                ('ret_door_obstacle', 'door + obstacle'), ('ret_relocation', 'goal moved'))):
    a = fig.add_subplot(gs[1, j])
    for c in ('gc100', 'pgc33'):
      band(a, C[f'room/{c}/{key}'], color=COL[c], lw=1.6)
    a.set_title(t, fontsize=11)
    a.set_xlabel('frames (billions)')
    if j == 0:
      a.set_ylabel('score')
    a.set_ylim(0, None)
    a.yaxis.grid(True, color=GRID, lw=0.6)
  for a in (ax, ax2):
    a.set_ylim(0, None)
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_xlabel('frames (billions)')
  long_m = S.get('long_m')
  ticks = [1.0, 1.25, 1.5] if long_m and long_m > 1300 else [1.0, 1.1, 1.2]
  for a in fig.axes:
    a.set_xticks(ticks)
    a.set_xlim(1.0, max(1.25, (long_m or 0) / 1000.0))
    if long_m and long_m > 1300:
      a.axvline(1.25, color=INK2, lw=0.7)
  save(fig, out, 'rooms_finetune')


def fig_pca(S, out):
  """Scripted-replay protocol (the same inputs to every network): cumulative
  variance at 1 B and after the fine-tuning, and the first-10 share in every state."""
  P = S.get('replay_pca') or {}
  if 'before' not in P:
    return
  last = 'long' if 'long' in P else 'after' if 'after' in P else None
  tt = state_titles(S)
  panels = [('before', 'familiar', 'familiar room, before')]
  if last:
    panels += [(last, 'familiar', f'familiar room, {tt[last]}'), (last, 'changed', f'changed rooms, {tt[last]}')]
  fig, axs = plt.subplots(1, len(panels) + 1, figsize=(3.3 * len(panels) + 4.2, 3.3), squeeze=False,
                          gridspec_kw=dict(width_ratios=[1] * len(panels) + [1.35]))
  for ax, (state, proto, t) in zip(axs[0], panels):
    for c in ('gc100', 'pgc33'):
      cu = P[state][proto][f'cumulative_{c}']
      m, sd = np.array(cu['mean'][:30]), np.array(cu['sem'][:30])
      x = np.arange(1, len(m) + 1)
      ax.plot(x, m, color=COL[c], lw=2.0, label=LAB[c], solid_capstyle='round')
      ax.fill_between(x, m - sd, m + sd, color=COL[c], alpha=0.14, lw=0)
    ax.axvline(10, color=INK2, lw=0.7)
    ax.set_xlabel('components')
    ax.set_title(t, fontsize=10.5)
    ax.set_ylim(0, 0.85)
    ax.set_xlim(1, 30)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
  axs[0][0].set_ylabel('share of variance')
  axs[0][0].legend(loc='lower right', fontsize=10, handlelength=1.2)
  ax = axs[0][-1]
  short = {'before': 'before', 'after': '+250 M', 'long': f'+{(S.get("long_m") or 0) - 1000} M'}
  cells = [(s_, pr, f'{pr}\n{short[s_]}') for pr in ('familiar', 'changed') for s_ in ('before', 'after', 'long') if s_ in P]
  for i, (state, proto, _) in enumerate(cells):
    st = P[state][proto]['var10']
    for c, off in (('gc100', -0.14), ('pgc33', 0.14)):
      key = 'gc' if c == 'gc100' else 'pgc'
      seeds = np.array(st[key + '_seeds'])
      ax.scatter(np.full(len(seeds), i + off), seeds, s=9, color='white', edgecolor=COL[c], lw=0.7, zorder=2)
      ax.errorbar(i + off, st[key], yerr=st[key + '_sem'], color=INK, lw=0.9, zorder=3)
      ax.scatter([i + off], [st[key]], s=42, color=COL[c], edgecolor='white', lw=1.2, zorder=4)
  ax.set_xticks(range(len(cells)))
  ax.set_xticklabels([c[2] for c in cells], fontsize=9.5)
  ax.tick_params(axis='x', length=0)
  ax.set_xlim(-0.5, len(cells) - 0.5)
  ax.set_title('first 10 components', fontsize=10.5)
  ax.yaxis.grid(True, color=GRID, lw=0.6)
  fig.subplots_adjust(wspace=0.3)
  save(fig, out, 'rooms_pca')


def fig_gap(S, out):
  layouts = ['open'] + list(TRAIN_LAYOUTS) + list(HELDOUT_LAYOUTS)
  states = [s for s in ('before', 'after', 'long') if s in S['per_condition']]
  tt = state_titles(S)
  offs = {1: (0.0,), 2: (-0.16, 0.16), 3: (-0.22, 0.0, 0.22)}[len(states)]
  fill = {'before': 'white', 'after': '#9a9891', 'long': INK}
  fig, axs = plt.subplots(1, 2, figsize=(7.4, 5.4), sharey=True, sharex=True)
  for ax, suf, t in zip(axs, ('', '+R'), ('goal fixed', 'goal moved')):
    for i, l in enumerate(layouts):
      for state, dy in zip(states, offs):
        st = S['per_condition'][state][l + suf]['score']
        ax.errorbar(st['diff'], i + dy, xerr=st['sem'], color=INK2, lw=0.9, zorder=2)
        ax.scatter([st['diff']], [i + dy], s=28, color=fill[state], edgecolor=INK, lw=0.9, zorder=3,
                   label=tt[state] if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(t, fontsize=11)
    ax.set_yticks(range(len(layouts)))
    ax.set_yticklabels(layouts)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel('score, grid + PGC − grid')
    for y in (0.5, 4.5, 7.5, 10.5):
      ax.axhline(y, color=GRID, lw=0.6)
  axs[0].invert_yaxis()          # shared axis: 'open' on top
  h, l = axs[0].get_legend_handles_labels()
  fig.legend(h, l, loc='lower center', ncol=len(states), fontsize=10, handletextpad=0.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
  fig.subplots_adjust(wspace=0.08, bottom=0.16)
  save(fig, out, 'rooms_gap_by_room')


def fig_paths(eval_dir, out, seed=1, rooms=('D1', 'O1'), env=0, steps=1350, long_m=None):
  pat = {'before': '{c}_lstm_s{s}@1000M.npz', 'after': '{c}room_lstm_s{s}@1250M.npz'}
  if long_m:
    pat['long'] = '{c}room_lstm_s{s}@%dM.npz' % long_m
  have = [s for s in ('before', 'after', 'long') if s in pat and
          all(os.path.exists(os.path.join(eval_dir, pat[s].format(c=c, s=seed))) for c in COL)]
  have = [have[0], have[-1]] if len(have) > 2 else have
  cols = [(r, s) for r in rooms for s in have]
  fig, axs = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 4.9), squeeze=False)
  for j, (room, state) in enumerate(cols):
    for i, c in enumerate(('gc100', 'pgc33')):
      z = np.load(os.path.join(eval_dir, pat[state].format(c=c, s=seed)))
      tr = z[f'{room}/traj'][:steps, env]
      g = z[f'{room}/traj_goal'][env]
      ax = axs[i][j]
      draw_room(ax, room, lw=1.6)
      ax.add_patch(plt.Circle(g, 0.30, color='#1baf7a', alpha=0.35, lw=0))
      jump = np.linalg.norm(np.diff(tr, axis=0), axis=1) > 0.1
      seg = np.split(np.arange(len(tr)), np.where(jump)[0] + 1)
      for sidx in seg:
        ax.plot(tr[sidx, 0], tr[sidx, 1], color=COL[c], lw=0.7, alpha=0.8)
      if i == 0:
        ax.set_title(f'{room}, ' + {'before': 'before', 'after': '+250 M', 'long': f'+{(long_m or 0) - 1000} M'}[state], fontsize=10)
  for i, c in enumerate(('gc100', 'pgc33')):
    axs[i][0].text(-0.08, 0.5, LAB[c], transform=axs[i][0].transAxes, rotation=90, va='center', ha='right', fontsize=10, color=INK)
  fig.subplots_adjust(wspace=0.06, hspace=0.06)
  save(fig, out, 'rooms_paths')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--summary', default=os.path.join(ROOT, 'results', 'room_change', 'summary.json'))
  ap.add_argument('--eval_dir', default=os.path.join(ROOT, 'results_room', 'eval'))
  ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'room_change', 'figures'))
  ap.add_argument('--only', default='layouts,scores,finetune,pca,gap,paths')
  ap.add_argument('--path_seed', type=int, default=1)
  ap.add_argument('--path_env', type=int, default=0)
  args = ap.parse_args()
  only = args.only.split(',')
  print('font:', plt.rcParams['font.family'], '(Times New Roman found)' if HAVE_TNR else '(fallback)')
  S = json.load(open(args.summary)) if os.path.exists(args.summary) else None
  if 'layouts' in only:
    fig_layouts(args.out)
  if S and 'scores' in only:
    fig_scores(S, args.out)
  if S and 'finetune' in only:
    fig_finetune(S, args.out)
  if S and 'pca' in only:
    fig_pca(S, args.out)
  if S and 'gap' in only:
    fig_gap(S, args.out)
  if 'paths' in only:
    fig_paths(args.eval_dir, args.out, seed=args.path_seed, env=args.path_env, long_m=(S or {}).get('long_m'))


if __name__ == '__main__':
  main()
