"""Further summaries of the code-variation experiment (run after scripts/codes/analyse.py).

  data/door_eval_summary.json     door-room trips of the trained agents (from data/door_eval.json,
                                  written by scripts/codes/door_eval.py)
  data/lstm_rooms_summary.json    the LSTM agents tested in the open room, the 21 rooms they trained
                                  in and 6 held-out room conditions (from results_room/eval_codes,
                                  written by scripts.room.eval_rooms)
  data/free_lookaheads.json       the per-cell look-ahead vectors learned by the six local runs with a
                                  free (forward, sideways) vector per cell (runs_codes/free)

  python -m scripts.codes.extras
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.env import CHANGED, HELDOUT_LAYOUTS  # noqa: E402
from scripts.codes.analyse import group_mean, paired, read_metrics  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'code_variations', 'data')
SPACINGS_CM = [40.0, 56.8, 80.656, 114.53]


def door():
  path = os.path.join(OUT, 'door_eval.json')
  if not os.path.exists(path):
    return
  D = json.load(open(path))
  metrics = dict(score=lambda r: r['score'], wall_contact=lambda r: r['wall_contact'],
                 reach_same=lambda r: r['known_same']['reach'], reach_door=lambda r: r['known_door']['reach'],
                 steps_same=lambda r: r['known_same']['steps'], steps_door=lambda r: r['known_door']['steps'],
                 search_same=lambda r: r['search_same']['reach'], search_door=lambda r: r['search_door']['reach'])
  by = {}
  for name, r in D.items():
    m = re.match(r'^([A-Za-z0-9]+)_(open|door)_ffb_s(\d+)$', name)
    if not m or m[2] != 'door':
      continue
    for k, fn in metrics.items():
      v = fn(r)
      if v is not None:
        by.setdefault(k, {}).setdefault(m[1], {})[int(m[3])] = float(v)
  S = {}
  for k, codes in by.items():
    S[k] = dict(codes={c: group_mean(v) for c, v in codes.items()},
                vs={ref: {c: paired(codes[c], codes[ref]) for c in codes if c != ref}
                    for ref in ('u20', 'gc100') if ref in codes})
  json.dump(S, open(os.path.join(OUT, 'door_eval_summary.json'), 'w'), indent=1)
  print('door:', {c: round(g['mean'], 3) for c, g in S.get('reach_door', {}).get('codes', {}).items()})


def lstm_rooms():
  files = sorted(glob.glob(os.path.join(ROOT, 'results_room', 'eval_codes', '*_mix_lstm_s*@500M.json')))
  if not files:
    return
  new = [h + r for h in HELDOUT_LAYOUTS for r in ('', '+R')]
  groups = dict(open=['open'], trained=list(CHANGED), new=new,
                door=[c for c in CHANGED if c.startswith('D') and not c.startswith('DO') and not c.endswith('+R')],
                obstacle=[c for c in CHANGED if c.startswith('O') and not c.endswith('+R')],
                both=[c for c in CHANGED if c.startswith('DO') and not c.endswith('+R')],
                goal_moved=[c for c in CHANGED if c.endswith('+R')])
  by = {}
  rows = []
  for f in files:
    j = json.load(open(f))
    m = re.match(r'^([A-Za-z0-9]+)_mix_lstm_s(\d+)$', j['name'])
    if not m:
      continue
    for g, conds in groups.items():
      v = [j['conditions'][c]['score'] for c in conds if c in j['conditions']]
      if v:
        by.setdefault(g, {}).setdefault(m[1], {})[int(m[2])] = float(np.mean(v))
    for c, v in j['conditions'].items():
      rows.append((j['name'], m[1], int(m[2]), c, v['score'], v.get('first_goal_step'), v.get('path_eff'),
                   v.get('wall_contact')))
  S = {}
  for g, codes in by.items():
    S[g] = dict(codes={c: group_mean(v) for c, v in codes.items()},
                vs={ref: {c: paired(codes[c], codes[ref]) for c in codes if c != ref}
                    for ref in ('u20', 'gc100') if ref in codes})
  json.dump(S, open(os.path.join(OUT, 'lstm_rooms_summary.json'), 'w'), indent=1)
  with open(os.path.join(OUT, 'lstm_rooms.csv'), 'w') as f:
    f.write('agent,code,seed,room,score,first_goal_step,path_efficiency,wall_contact\n')
    for r in rows:
      f.write(','.join('' if x is None else (f'{x:.4f}' if isinstance(x, float) else str(x)) for x in r) + '\n')
  print('lstm rooms:', {g: {c: round(v['mean'], 1) for c, v in S[g]['codes'].items()} for g in ('open', 'trained', 'new') if g in S})


def free():
  import torch
  out = {}
  for d in sorted(glob.glob(os.path.join(ROOT, 'runs_codes', 'free', 'cfree*_ffb_s*'))):
    name = os.path.basename(d)
    keys = ['shift_fwd_mean', 'shift_fwd_sd', 'shift_lat_mean', 'shift_lat_sd'] + [f'shift_fwd_m{m}' for m in range(4)]
    rows = read_metrics(d, keys)
    if not rows:
      continue
    rec = dict(frames=rows[-1][0], traj=[[float(x) for x in r] for r in rows[::max(1, len(rows) // 120)]], keys=keys)
    ck = os.path.join(d, 'ckpt_final.pt')
    if os.path.exists(ck):
      sh = torch.load(ck, map_location='cpu')['shift'].numpy() * 100.0      # cm, [512, 2]
      mod = np.arange(512) % 4
      rec.update(final=True, fwd_mean=float(sh[:, 0].mean()), fwd_sd=float(sh[:, 0].std()),
                 lat_mean=float(sh[:, 1].mean()), lat_sd=float(sh[:, 1].std()),
                 fwd_by_module=[float(sh[mod == m, 0].mean()) for m in range(4)],
                 fwd_sd_by_module=[float(sh[mod == m, 0].std()) for m in range(4)],
                 frac_fwd_gt5=float((sh[:, 0] > 5).mean()), frac_fwd_lt5=float((sh[:, 0] < -5).mean()),
                 fwd=[round(float(x), 2) for x in sh[:, 0]], lat=[round(float(x), 2) for x in sh[:, 1]])
    else:
      last = rows[-1]
      rec.update(final=False, fwd_mean=last[1], fwd_sd=last[2], lat_mean=last[3], lat_sd=last[4],
                 fwd_by_module=last[5:9])
    out[name] = rec
  if out:
    json.dump(out, open(os.path.join(OUT, 'free_lookaheads.json'), 'w'))
    print('free:', {k: (round(v['fwd_mean'], 2), round(v['fwd_sd'], 2), round(v['lat_mean'], 2), round(v['lat_sd'], 2))
                    for k, v in out.items()})


if __name__ == '__main__':
  os.makedirs(OUT, exist_ok=True)
  door()
  lstm_rooms()
  free()
