"""From-scratch reference runs of the room-change experiment: curves, tests, figure.

  single rooms   <cond>scr<ROOM>_lstm_s<seed>: an agent that has only ever lived in one changed room
                 (goal fixed or moved, half of the episodes each; ROOM = R is the open room with a
                 moving goal); one grid and one PGC agent per room, seeds rotating over the rooms
  mixture        <cond>scrmix_lstm_s<seed>: trained from scratch on the 21 changed-room conditions

compared, at equal numbers of frames spent in the changed rooms, with the agents that were first
trained 1 B frames in the familiar room and then fine-tuned (runs_room/<cond>room_lstm_s<seed>).

Reads runs_scratch/<run>/episodes.csv, runs_room/<run>/episodes.csv, runs_ext/<run>/episodes.csv and
the frozen tests results_room/eval/<run>@<M>M.json (made on the training VMs with eval_rooms.py);
writes results/room_change/scratch.json, data/scratch_curves.csv, data/scratch_eval.csv and
figures/rooms_scratch.png.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.env import CHANGED, TRAIN_LAYOUTS  # noqa: E402
from scripts.room.analyse import paired  # noqa: E402

CONDS = ('gc100', 'pgc33')
ENVS = list(TRAIN_LAYOUTS) + ['R']
BIN = 10e6


def env_rooms(env):
  return ['open+R'] if env == 'R' else [env, env + '+R']


def read_episodes(path, offset=0.0):
  fr, ret, room = [], [], []
  with open(path) as f:
    for r in csv.DictReader(f):
      fr.append(float(r['frames']) - offset), ret.append(float(r['ret'])), room.append(r.get('room') or 'open')
  fr, ret, room = np.array(fr), np.array(ret), np.array(room)
  keep = np.ones(len(fr), bool)
  for i in np.flatnonzero(np.diff(fr) < 0) + 1:        # a restarted run repeats a stretch of its log
    keep[:i] &= fr[:i] < fr[i]
  return fr[keep], ret[keep], room[keep]


def binned(fr, ret, last, sel=None):
  """Mean return per BIN of frames; NaN where a bin has under 30 episodes."""
  edges = np.arange(0, last + BIN, BIN)
  out = np.full(len(edges) - 1, np.nan)
  if sel is not None:
    fr, ret = fr[sel], ret[sel]
  idx = np.digitize(fr, edges) - 1
  for b in range(len(out)):
    m = idx == b
    if m.sum() >= 30:
      out[b] = ret[m].mean()
  return out


def summarize(M):
  M = np.array(M, float)
  ok = ~np.isnan(M).any(0) if len(M) else np.zeros(0, bool)
  n = len(M)
  return dict(n=n, mean=[float(x) if o else None for x, o in zip(np.nanmean(M, 0) if n else [], ok)],
              sem=[float(x) if o and n > 1 else None for x, o in zip(np.nanstd(M, 0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(M.shape[1] if n else 0), ok)],
              rows=[[None if v != v else float(v) for v in row] for row in M])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'room_change'))
  ap.add_argument('--eval_dir', default=os.path.join(ROOT, 'results_room', 'eval'))
  ap.add_argument('--no_figure', action='store_true')
  args = ap.parse_args()
  runs = json.load(open(os.path.join(ROOT, 'runs_scratch', 'matrix.json')))
  S = dict(bin_m=BIN / 1e6, curves={}, tests={}, runs={})

  # ---------------- training curves, x = frames spent in the changed rooms
  ep = {}
  for r in runs:
    p = os.path.join(ROOT, 'runs_scratch', r['name'], 'episodes.csv')
    if os.path.exists(p):
      ep[r['name']] = read_episodes(p)
      S['runs'][r['name']] = dict(env=r['env'], cond=r['cond'], seed=r['seed'], frames_m=float(ep[r['name']][0].max() / 1e6))
  if not ep:
    print('no from-scratch logs yet')
    return
  last = min(v[0].max() for v in ep.values())
  last = np.floor(last / BIN) * BIN
  S['common_frames_m'] = float(last / 1e6)
  ft = {}
  for c in CONDS:
    for s in range(1, 9):
      p = os.path.join(ROOT, 'runs_room', f'{c}room_lstm_s{s}', 'episodes.csv')
      if os.path.exists(p):
        ft[(c, s)] = read_episodes(p, offset=1000e6)
  orig = {}
  for c in CONDS:
    for s in range(1, 9):
      p = os.path.join(ROOT, 'runs_ext', f'{c}_lstm_s{s}', 'episodes.csv')
      if os.path.exists(p):
        orig[(c, s)] = read_episodes(p)
  ft_last = min([v[0].max() for v in ft.values()] + [last]) if ft else 0
  ft_last = np.floor(ft_last / BIN) * BIN
  for c in CONDS:
    # single rooms from scratch: one curve per room, then mean and SEM over rooms
    S['curves'][f'scratch_room/{c}'] = summarize([binned(*ep[r['name']][:2], last) for r in runs
                                                  if r['cond'] == c and r['env'] != 'mix' and r['name'] in ep])
    S['curves'][f'scratch_mix/{c}'] = summarize([binned(*ep[r['name']][:2], last) for r in runs
                                                 if r['cond'] == c and r['env'] == 'mix' and r['name'] in ep])
    # pretrained + fine-tuned agents in the same rooms (their mixture gives each room 2 of 21 episodes)
    rows_all, rows_env = [], []
    for s in range(1, 9):
      if (c, s) not in ft:
        continue
      fr, ret, room = ft[(c, s)]
      rows_all.append(binned(fr, ret, ft_last))
      rows_env.append(np.nanmean([binned(fr, ret, ft_last, np.isin(room, env_rooms(e))) for e in ENVS], 0))
    S['curves'][f'finetune_all/{c}'] = summarize(rows_all)
    S['curves'][f'finetune_rooms/{c}'] = summarize(rows_env)
    S['curves'][f'familiar_original/{c}'] = summarize([binned(*orig[(c, s)][:2], last) for s in range(1, 9) if (c, s) in orig])
  S['x_m'] = [float((i + 0.5) * BIN / 1e6) for i in range(int(last / BIN))]
  S['x_ft_m'] = [float((i + 0.5) * BIN / 1e6) for i in range(int(ft_last / BIN))]
  # PGC - grid on the training score of the last 50 M common frames, paired by room
  def tail(name, lo, hi):
    fr, ret, _ = ep[name]
    return float(ret[(fr > lo) & (fr <= hi)].mean())
  rooms_done = [e for e in ENVS if all(f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}' in ep for c in CONDS)]
  if rooms_done and last >= 50e6:
    g = [tail(f'gc100scr{e}_lstm_s{ENVS.index(e) % 8 + 1}', last - 50e6, last) for e in rooms_done]
    p_ = [tail(f'pgc33scr{e}_lstm_s{ENVS.index(e) % 8 + 1}', last - 50e6, last) for e in rooms_done]
    S['train_tail'] = dict(rooms=rooms_done, window_m=[float((last - 50e6) / 1e6), float(last / 1e6)], stat=paired(g, p_))

  # ---------------- frozen tests at the frame marks every from-scratch agent has reached
  ev = {}
  for f in glob.glob(os.path.join(args.eval_dir, '*scr*_lstm_s*@*M.json')):
    m = re.match(r'(.+)@(\d+)M\.json$', os.path.basename(f))
    ev.setdefault(int(m.group(2)), {})[m.group(1)] = f
  def score(path, conds):
    d = json.load(open(path))['conditions']
    return float(np.mean([d[c]['score'] for c in conds]))
  def peff(path, conds):
    d = json.load(open(path))['conditions']
    v = [d[c]['path_eff'] for c in conds if d[c]['path_eff'] == d[c]['path_eff']]
    return float(np.mean(v)) if v else float('nan')
  for mark in sorted(ev):
    names = ev[mark]
    T = dict(n_runs=len(names))
    rooms_ok = [e for e in ENVS if all(f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}' in names for c in CONDS)]
    if len(rooms_ok) >= 2:
      own = {c: [score(names[f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}'], env_rooms(e)) for e in rooms_ok] for c in CONDS}
      fam = {c: [score(names[f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}'], ['open']) for e in rooms_ok] for c in CONDS}
      eff = {c: [peff(names[f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}'], [x for x in env_rooms(e) if not x.endswith('+R')] or ['open+R']) for e in rooms_ok] for c in CONDS}
      T['single_rooms'] = dict(rooms=rooms_ok, own_room=paired(own['gc100'], own['pgc33']), familiar_room=paired(fam['gc100'], fam['pgc33']),
                               path_eff=paired(eff['gc100'], eff['pgc33']),
                               per_room={e: dict(gc=own['gc100'][i], pgc=own['pgc33'][i], seed=ENVS.index(e) % 8 + 1) for i, e in enumerate(rooms_ok)})
    mix = {}
    for c in CONDS:
      for s in (1, 2):
        n = f'{c}scrmix_lstm_s{s}'
        if n in names:
          mix[f'{c}_s{s}'] = dict(changed=score(names[n], CHANGED), familiar=score(names[n], ['open']))
    T['mixture'] = mix
    # the original training in the familiar room at the same number of frames (training score, last 25 M frames; 8 seeds)
    def win(v):
      fr, ret, _ = v
      m_ = (fr > (mark - 25) * 1e6) & (fr <= mark * 1e6)
      return float(ret[m_].mean())
    seeds8 = [s for s in range(1, 9) if all((c, s) in orig for c in CONDS)]
    if seeds8:
      T['familiar_original'] = paired([win(orig[('gc100', s)]) for s in seeds8], [win(orig[('pgc33', s)]) for s in seeds8])
    # the same window of the from-scratch training scores, paired by room
    rooms_tr = [e for e in ENVS if all(f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}' in ep and
                                       ep[f'{c}scr{e}_lstm_s{ENVS.index(e) % 8 + 1}'][0].max() >= mark * 1e6 for c in CONDS)]
    if len(rooms_tr) >= 2:
      T['scratch_training'] = dict(rooms=rooms_tr, stat=paired([win(ep[f'gc100scr{e}_lstm_s{ENVS.index(e) % 8 + 1}']) for e in rooms_tr],
                                                               [win(ep[f'pgc33scr{e}_lstm_s{ENVS.index(e) % 8 + 1}']) for e in rooms_tr]))
    S['tests'][str(mark)] = T
  # the fine-tuned agents in the same rooms (8 seeds), for the reference columns
  ref = {}
  for tag in ('1000M', '1250M', '1500M'):
    r = {}
    for c in CONDS:
      per_seed = []
      for s in range(1, 9):
        f = os.path.join(args.eval_dir, (f'{c}_lstm_s{s}' if tag == '1000M' else f'{c}room_lstm_s{s}') + f'@{tag}.json')
        if os.path.exists(f):
          per_seed.append(dict(seed=s, rooms=float(np.mean([score(f, env_rooms(e)) for e in ENVS])), changed=score(f, CHANGED),
                               familiar=score(f, ['open']), per_room={e: score(f, env_rooms(e)) for e in ENVS}))
      if per_seed:
        r[c] = dict(n=len(per_seed), rooms=float(np.mean([x['rooms'] for x in per_seed])), changed=float(np.mean([x['changed'] for x in per_seed])),
                    familiar=float(np.mean([x['familiar'] for x in per_seed])),
                    per_room={e: float(np.mean([x['per_room'][e] for x in per_seed])) for e in ENVS}, per_seed=per_seed)
    if len(r) == 2 and [x['seed'] for x in r['gc100']['per_seed']] == [x['seed'] for x in r['pgc33']['per_seed']]:
      for k in ('rooms', 'changed', 'familiar'):      # PGC - grid, paired within seed
        r['stat_' + k] = paired([x[k] for x in r['gc100']['per_seed']], [x[k] for x in r['pgc33']['per_seed']])
    if r:
      ref[tag] = r
  S['finetuned_reference'] = ref
  json.dump(S, open(os.path.join(args.out, 'scratch.json'), 'w'))

  # ---------------- data files
  os.makedirs(os.path.join(args.out, 'data'), exist_ok=True)
  with open(os.path.join(args.out, 'data', 'scratch_curves.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'code', 'seed', 'room', 'frames_m', 'training_score'])
    for r in runs:
      if r['name'] in ep:
        for x, v in zip(S['x_m'], binned(*ep[r['name']][:2], last)):
          if v == v:
            w.writerow([r['name'], r['cond'], r['seed'], r['env'], x, f'{v:.4g}'])
  with open(os.path.join(args.out, 'data', 'scratch_eval.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'frames_m', 'room', 'score', 'path_eff', 'first_goal_step', 'wall_contact'])
    for mark in sorted(ev):
      for name, path in sorted(ev[mark].items()):
        for rc, v in json.load(open(path))['conditions'].items():
          w.writerow([name, mark, rc, f'{v["score"]:.4g}', f'{v["path_eff"]:.4g}', f'{v["first_goal_step"]:.5g}', f'{v["wall_contact"]:.4g}'])

  if args.no_figure:
    return
  # ---------------- figure
  from scripts.room import figures as F
  plt = F.plt
  fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
  def line(ax, key, x, **kw):
    cu = S['curves'].get(key)
    if not cu or not cu['n']:
      return
    m = np.array([np.nan if v is None else v for v in cu['mean']])
    ax.plot(x[:len(m)], m, solid_capstyle='round', **kw)
    if cu['n'] > 1 and kw.get('lw', 2) >= 2:
      s = np.array([np.nan if v is None else v for v in cu['sem']])
      ax.fill_between(x[:len(m)], m - s, m + s, color=kw['color'], alpha=0.12, lw=0)
  x, xf = np.array(S['x_m']), np.array(S['x_ft_m'])
  for c in CONDS:
    line(axs[0], f'finetune_all/{c}', xf, color=F.COL[c], lw=2.0, label=F.LAB[c] + ', pretrained')
    line(axs[0], f'scratch_mix/{c}', x, color=F.COL[c], lw=2.0, ls=(0, (4, 2)), label=F.LAB[c] + ', from scratch')
    line(axs[1], f'finetune_rooms/{c}', xf, color=F.COL[c], lw=2.0)
    line(axs[1], f'scratch_room/{c}', x, color=F.COL[c], lw=2.0, ls=(0, (4, 2)))
    line(axs[2], f'familiar_original/{c}', x, color=F.COL[c], lw=2.0, ls=(0, (4, 2)))
  for ax, t in zip(axs, ('all changed rooms', 'one room each', 'familiar room, original training')):
    ax.set_title(t, fontsize=11)
    ax.set_xlabel('frames in these rooms (millions)')
    ax.yaxis.grid(True, color=F.GRID, lw=0.6)
    ax.set_xlim(0, max(float(x.max()) if len(x) else 1, float(xf.max()) if len(xf) else 1))
    ax.set_ylim(0, None)
  axs[0].set_ylabel('score')
  axs[0].legend(loc='upper left', fontsize=9, handlelength=2.2)
  fig.subplots_adjust(wspace=0.08)
  F.save(fig, os.path.join(args.out, 'figures'), 'rooms_scratch')


if __name__ == '__main__':
  main()
