"""Room-change experiment: statistics, tables and data files.

Reads results_room/eval/<agent>@<tag>.{json,npz} (scripts/room/eval_rooms.py),
results_room/replay_pca/<agent>@<tag>.json (scripts/room/replay_pca.py) and the
training logs runs_room/<run>/episodes.csv; writes

  results/room_change/summary.json              everything the figures and the README use
  results/room_change/RESULTS.md                tables
  results/room_change/data/room_eval.csv        one row per agent state x room condition
  results/room_change/data/pca_spectra.csv      eigenvalue spectra of the LSTM output covariance (own behaviour)
  results/room_change/data/pca_replay.csv       spectra with the scripted-replay protocol
  results/room_change/data/finetune_curves.csv  training return in 5 M-frame bins per run

Agent states:
  before     <cond>_lstm_s<seed>@1000M       the 1 B agents (familiar room only)
  after      <cond>room_lstm_s<seed>@1250M   + 250 M frames in the changed rooms
  same       <cond>same_lstm_s<seed>@1250M   + 250 M frames in the familiar room (seeds 1-2)
  long       <cond>room_lstm_s<seed>@<L>M    the changed-room runs continued to L M frames
  same_long  <cond>same_lstm_s<seed>@<L>M    the familiar-room runs continued likewise
Comparisons are paired within seed (the two codes of a seed share cells and
evaluation episodes); p values are two-sided paired t-tests over seeds, with
the exact sign-flip permutation p (smallest possible value 2 / 2^n) alongside
in summary.json.

Statistics that need care (see README, methods):
  * first goal, goal rate, time per return and wall contact are taken from the
    rooms WITHOUT a goal move (in a room with a move the first arrival can be
    at the moved goal);
  * time per return = steps with the goal known / completed returns, so the
    time spent on returns that never finish is counted;
  * relocation statistics are restricted to the episodes in which the agent
    had reached the goal before it moved.
"""
import argparse
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.env import CHANGED, HELDOUT_LAYOUTS, TEST_CONDITIONS, TRAIN_LAYOUTS  # noqa: E402

CONDS = ('gc100', 'pgc33')
T_EP, T_MOVE = 1350, 675
GROUPS = {
    'familiar': ['open'],
    'changed': list(CHANGED),
    'door': [l for l in TRAIN_LAYOUTS if l.startswith('D') and not l.startswith('DO')],
    'obstacle': [l for l in TRAIN_LAYOUTS if l.startswith('O')],
    'door_obstacle': [l for l in TRAIN_LAYOUTS if l.startswith('DO')],
    'goal_moved': ['open+R'],
    'layout_and_goal_moved': [l + '+R' for l in TRAIN_LAYOUTS],
    'no_relocation': list(TRAIN_LAYOUTS),
    'relocation': [l + '+R' for l in TRAIN_LAYOUTS] + ['open+R'],
    'new_rooms': list(HELDOUT_LAYOUTS),
    'new_rooms_goal_moved': [l + '+R' for l in HELDOUT_LAYOUTS],
}
METRICS = ('score', 'first_goal_step', 'frac_found', 'first_found_step', 'post_rate', 'return_rate',
           'steps_per_completed_return', 'latency_ratio', 'path_eff', 'wall_contact', 'var10',
           'participation_ratio', 'n90', 'total_var', 'var10_search', 'var10_return', 'frac_return')
RELOC_METRICS = ('goals_pre', 'goals_post', 'knew_share', 'goals_post_knew', 'refound_knew',
                 'relearn_steps_knew', 'reversal_cost', 'old_goal_dwell_knew', 'control_zone_dwell_knew',
                 'perseveration_knew', 'old_goal_dist_knew')
NAMES = {'familiar': 'familiar room', 'changed': 'changed rooms (all 21)', 'door': 'door', 'obstacle': 'obstacle',
         'door_obstacle': 'door + obstacle', 'goal_moved': 'goal moved (open room)',
         'layout_and_goal_moved': 'door / obstacle and goal moved', 'no_relocation': 'changed rooms, goal fixed (10)',
         'relocation': 'rooms with a goal move (11)', 'new_rooms': 'new rooms (never trained)',
         'new_rooms_goal_moved': 'new rooms, goal moved'}
MNAME = {'score': ('score', 1), 'first_goal_step': ('steps to the first goal (1350 if never found)', 0),
         'frac_found': ('episodes that found the goal', 3), 'post_rate': ('goals per 1000 steps after the first', 2),
         'first_found_step': ('steps to the first goal, episodes that found it', 0),
         'return_rate': ('completed returns per 1000 steps with the goal known', 2),
         'steps_per_completed_return': ('steps per completed return', 1),
         'latency_ratio': ('first find : return, ratio of steps', 2),
         'path_eff': ('path efficiency of completed returns', 3), 'wall_contact': ('share of steps against a wall', 3),
         'var10': ('variance in first 10 components, all steps', 3), 'var10_return': ('variance in first 10 components, goal known', 3),
         'var10_search': ('variance in first 10 components, goal not yet found', 3),
         'frac_return': ('share of steps with the goal known', 3), 'participation_ratio': ('participation ratio', 1),
         'goals_pre': ('goals before the move', 2), 'goals_post': ('goals after the move, all episodes', 2),
         'knew_share': ('episodes that had found the goal before it moved', 3),
         'goals_post_knew': ('goals after the move, in those episodes', 2),
         'refound_knew': ('share of those episodes that found the moved goal', 3),
         'relearn_steps_knew': ('steps from the move to the moved goal, those episodes (675 if never)', 0),
         'reversal_cost': ('reversal cost: steps to the moved goal / steps to the first goal', 2),
         'old_goal_dwell_knew': ('time near the old goal until the moved goal is found, share', 3),
         'control_zone_dwell_knew': ('same for the mirror-image control zone', 3),
         'perseveration_knew': ('perseveration index (old-goal zone − control zone)', 3),
         'old_goal_dist_knew': ('mean distance to the old goal in that time (m)', 3)}


def states_for(long_m):
  st = {'before': ('{c}_lstm_s{s}', '1000M'), 'after': ('{c}room_lstm_s{s}', '1250M'),
        'same': ('{c}same_lstm_s{s}', '1250M')}
  if long_m:
    st['long'] = ('{c}room_lstm_s{s}', f'{long_m}M')
    st['same_long'] = ('{c}same_lstm_s{s}', f'{long_m}M')
  return st


def condition_row(rc, v, events):
  r = {k: v.get(k) for k in ('score', 'first_goal_step', 'frac_found', 'post_rate', 'path_eff', 'wall_contact',
                             'frac_return', 'sem_score')}
  r['steps_per_completed_return'] = v.get('steps_per_return')
  r.update(var10=v['pca_h']['var10'], participation_ratio=v['pca_h']['participation_ratio'],
           n90=v['pca_h']['n90'], total_var=v['pca_h']['total_var'], eig_h=v['eig_h'])
  for ph in ('search', 'return'):
    r[f'var10_{ph}'] = v[f'pca_h_{ph}']['var10'] if f'pca_h_{ph}' in v else None
  if v.get('frac_return'):
    r['return_rate'] = 1000.0 * v['n_returns'] / (v['frac_return'] * v['episodes'] * T_EP)
  fs = np.array(v['first_goal_steps'])
  if (fs < T_EP).any() and not rc.endswith('+R'):
    r['first_found_step'] = float(fs[fs < T_EP].mean())
    if v.get('steps_per_return') == v.get('steps_per_return') and v.get('steps_per_return'):
      r['latency_ratio'] = r['first_found_step'] / v['steps_per_return']     # Mirowski et al. 2017, 'Latency 1:>1'
  if rc.endswith('+R'):
    gp = np.array(v['goals_post_list'])
    pre = np.rint(np.array(v['scores']) / 10).astype(int) - gp
    knew = pre >= 1
    r.update(goals_pre=float(pre.mean()), goals_post=float(gp.mean()), knew_share=float(knew.mean()))
    if knew.sum() >= 5:
      first = np.full(len(gp), np.inf)
      if events is not None and len(events):
        post = events[events[:, 3] > 0.5]
        np.minimum.at(first, post[:, 0].astype(int), post[:, 1])
      relearn = np.where(np.isfinite(first), first - T_MOVE, T_EP - T_MOVE)
      r.update(goals_post_knew=float(gp[knew].mean()), refound_knew=float((gp[knew] >= 1).mean()),
               relearn_steps_knew=float(relearn[knew].mean()),
               reversal_cost=float(relearn[knew].mean() / max(fs[knew].mean(), 1.0)))
      if v.get('old_goal_dwell') is not None:      # perseveration (Maei et al. 2009; Gallagher et al. 1993)
        dw, ct = np.array(v['old_goal_dwell'])[knew], np.array(v['control_zone_dwell'])[knew]
        r.update(old_goal_dwell_knew=float(dw.mean()), control_zone_dwell_knew=float(ct.mean()),
                 perseveration_knew=float((dw - ct).mean()),
                 old_goal_dist_knew=float(np.array(v['old_goal_dist'])[knew].mean()))
  return r


def load(eval_dir, states):
  """{state: {cond: {seed: {rows: {room condition: metrics}, frames}}}}"""
  out = {}
  for state, (pat, tag) in states.items():
    for c in CONDS:
      for s in range(1, 9):
        p = os.path.join(eval_dir, pat.format(c=c, s=s) + f'@{tag}')
        if not os.path.exists(p + '.json'):
          continue
        d = json.load(open(p + '.json'))
        z = np.load(p + '.npz') if os.path.exists(p + '.npz') else None
        rows = {rc: condition_row(rc, v, z[f'{rc}/events'] if z is not None and f'{rc}/events' in z else None)
                for rc, v in d['conditions'].items()}
        out.setdefault(state, {}).setdefault(c, {})[s] = dict(rows=rows, frames=d['frames'])
  return out


def group_value(rows, conds, metric):
  vals = [rows[c][metric] for c in conds if c in rows and rows[c].get(metric) is not None]
  vals = [v for v in vals if v == v]
  return float(np.mean(vals)) if vals else float('nan')


def perm_p(d):
  """Exact two-sided sign-flip permutation p of the mean of paired differences."""
  d = np.asarray(d, float)
  if len(d) < 2 or len(d) > 16:
    return None
  obs = abs(d.mean())
  signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
  return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())


def paired(a, b):
  """a = grid, b = grid + PGC, same seeds -> difference b - a."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  ok = ~(np.isnan(a) | np.isnan(b))
  a, b = a[ok], b[ok]
  d = b - a
  n = len(d)
  if n < 2:
    return dict(n=n, gc=float(a.mean()) if n else None, pgc=float(b.mean()) if n else None,
                diff=float(d.mean()) if n else None, sem=None, p=None, pos=int((d > 0).sum()))
  sd = d.std(ddof=1)
  return dict(n=n, gc=float(a.mean()), pgc=float(b.mean()), diff=float(d.mean()),
              sem=float(sd / np.sqrt(n)), ci95=float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)),
              p=float(stats.ttest_rel(b, a).pvalue) if sd > 0 else None,
              p_perm=perm_p(d), dz=float(d.mean() / sd) if sd > 0 else None, pos=int((d > 0).sum()),
              gc_sem=float(a.std(ddof=1) / np.sqrt(n)), pgc_sem=float(b.std(ddof=1) / np.sqrt(n)),
              gc_seeds=[float(x) for x in a], pgc_seeds=[float(x) for x in b])


def one_sample(x):
  x = np.asarray(x, float)
  x = x[~np.isnan(x)]
  if len(x) < 2:
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None, sem=None, p=None, pos=int((x > 0).sum()))
  sd = x.std(ddof=1)
  return dict(n=len(x), mean=float(x.mean()), sem=float(sd / np.sqrt(len(x))),
              ci95=float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))),
              dz=float(x.mean() / sd) if sd > 0 else None,
              p=float(stats.ttest_1samp(x, 0.0).pvalue) if sd > 0 else None, p_perm=perm_p(x),
              pos=int((x > 0).sum()), seeds=[float(v) for v in x])


def curves(runs_dir, last_m, bin_m=5):
  """Training return per 5 M-frame bin of every changed-room / same-room run. A
  restarted run repeats a stretch of its log: only the rows written after the
  last backward step in the frame counter are kept for that stretch."""
  rows = []
  edges = np.arange(1000, last_m + bin_m, bin_m) * 1e6
  for d in sorted(glob.glob(os.path.join(runs_dir, '*_lstm_s[0-9]'))):
    name = os.path.basename(d)
    kind = 'room' if 'room_' in name else 'same' if 'same_' in name else None
    p = os.path.join(d, 'episodes.csv')
    if kind is None or not os.path.exists(p):
      continue
    cond, seed = name.split('_')[0][:-4], int(name.rsplit('_s', 1)[1])
    fr, ret, room = [], [], []
    with open(p) as f:
      for r in csv.DictReader(f):
        fr.append(float(r['frames'])), ret.append(float(r['ret'])), room.append(r.get('room') or 'open')
    fr, ret, room = np.array(fr), np.array(ret), np.array(room)
    keep = np.ones(len(fr), bool)
    for i in np.flatnonzero(np.diff(fr) < 0) + 1:
      keep[:i] &= fr[:i] < fr[i]
    fr, ret, room = fr[keep], ret[keep], room[keep]
    reloc = np.array([x.endswith('+R') for x in room])
    lay = np.array([x[:-2] if x.endswith('+R') else x for x in room])
    sel = {'all': np.ones(len(fr), bool), 'door': np.isin(lay, GROUPS['door']) & ~reloc,
           'obstacle': np.isin(lay, GROUPS['obstacle']) & ~reloc,
           'door_obstacle': np.isin(lay, GROUPS['door_obstacle']) & ~reloc,
           'relocation': reloc}
    for i in range(len(edges) - 1):
      m = (fr > edges[i]) & (fr <= edges[i + 1])
      if m.sum() < 50:
        continue
      row = dict(run=name, cond=cond, seed=seed, kind=kind, frames_m=float((edges[i] + edges[i + 1]) / 2e6),
                 episodes=int(m.sum()))
      for k, s in sel.items():
        mm = m & s
        row[f'ret_{k}'] = float(ret[mm].mean()) if mm.any() else float('nan')
      rows.append(row)
  return rows


def fmt(x, nd=1):
  return '–' if x is None or x != x else f'{x:.{nd}f}'


def fmt_p(p):
  if p is None or p != p:
    return '–'
  return '< 0.001' if p < 0.001 else f'{p:.3f}' if p < 0.1 else f'{p:.2f}'


def row_md(label, st, nd=1):
  if st is None or st.get('n', 0) == 0:
    return f'| {label} | – | – | – | – | – |'
  return (f'| {label} | {fmt(st["gc"], nd)} | {fmt(st["pgc"], nd)} | {st["diff"]:+.{nd}f} ± {fmt(st["sem"], nd)} | '
          f'{fmt_p(st["p"])} | {st["pos"]}/{st["n"]} |')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--eval_dir', default=os.path.join(ROOT, 'results_room', 'eval'))
  ap.add_argument('--runs_dir', default=os.path.join(ROOT, 'runs_room'))
  ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'room_change'))
  ap.add_argument('--data', default=os.path.join(ROOT, 'results', 'room_change', 'data'))
  ap.add_argument('--replay_dir', default=os.path.join(ROOT, 'results_room', 'replay_pca'))
  ap.add_argument('--long', type=int, default=None,
                  help='frames (M) of the longer fine-tuning checkpoint (default: runs_room/LONG_FRAMES if present)')
  ap.add_argument('--exclude', default='',
                  help='comma list of seeds to leave out (both codes), for a sensitivity analysis; the outputs get the '
                       'suffix _excl<seeds> and no data files are written')
  args = ap.parse_args()
  excl = [int(x) for x in args.exclude.split(',') if x.strip()]
  suffix = ('_excl' + '-'.join(str(x) for x in excl)) if excl else ''
  lf = os.path.join(args.runs_dir, 'LONG_FRAMES')
  long_m = args.long or (int(open(lf).read().split()[0]) if os.path.exists(lf) else None)
  STATES = states_for(long_m)
  D = load(args.eval_dir, STATES)
  for state in D:
    for c in D[state]:
      for s_ in excl:
        D[state][c].pop(s_, None)
  os.makedirs(args.data, exist_ok=True)
  S = dict(states={k: {c: sorted(D[k][c]) for c in D[k]} for k in D}, groups=GROUPS, long_m=long_m, stats={},
           per_condition={}, pca={}, change={}, reference={})

  def vals(state, cond, seeds, conds, metric):
    return [group_value(D[state][cond][s]['rows'], conds, metric) for s in seeds]

  def common(state):
    if state not in D or any(c not in D[state] for c in CONDS):
      return []
    return sorted(set(D[state]['gc100']) & set(D[state]['pgc33']))

  # ---- grid + PGC vs grid, per state, room group and metric
  for state in STATES:
    seeds = common(state)
    if not seeds:
      continue
    S['stats'][state] = {}
    for g, conds in GROUPS.items():
      mets = METRICS + (RELOC_METRICS if all(c.endswith('+R') for c in conds) else ())
      S['stats'][state][g] = {m: paired(vals(state, 'gc100', seeds, conds, m), vals(state, 'pgc33', seeds, conds, m))
                              for m in mets}
    S['per_condition'][state] = {
        rc: {m: paired(vals(state, 'gc100', seeds, [rc], m), vals(state, 'pgc33', seeds, [rc], m))
             for m in ('score', 'path_eff', 'post_rate', 'first_goal_step', 'wall_contact', 'var10')}
        for rc in TEST_CONDITIONS}
  # ---- change with training (later - earlier state), per code, and the difference between the codes
  for a, b in (('before', 'after'), ('before', 'long'), ('after', 'long')):
    seeds = sorted(set(common(a)) & set(common(b)))
    if not seeds:
      continue
    S['change'][f'{a}->{b}'] = {}
    for g, conds in GROUPS.items():
      S['change'][f'{a}->{b}'][g] = {}
      for m in METRICS + (RELOC_METRICS if all(c.endswith('+R') for c in conds) else ()):
        dg = np.array(vals(b, 'gc100', seeds, conds, m)) - np.array(vals(a, 'gc100', seeds, conds, m))
        dp = np.array(vals(b, 'pgc33', seeds, conds, m)) - np.array(vals(a, 'pgc33', seeds, conds, m))
        S['change'][f'{a}->{b}'][g][m] = dict(gc=one_sample(dg), pgc=one_sample(dp), diff=one_sample(dp - dg))
  # ---- own-behaviour PCA: cumulative variance curves per room group
  for state in STATES:
    seeds = common(state)
    if not seeds:
      continue
    cum = {}
    for c in CONDS:
      for g in ('familiar', 'no_relocation'):
        cs = np.array([[np.cumsum(D[state][c][s]['rows'][rc]['eig_h']) / np.sum(D[state][c][s]['rows'][rc]['eig_h'])
                        for rc in GROUPS[g]] for s in seeds]).mean(1)
        cum[f'{c}/{g}'] = dict(mean=[float(x) for x in cs.mean(0)[:40]],
                               sem=[float(x) for x in (cs.std(0, ddof=1) / np.sqrt(len(seeds)))[:40]] if len(seeds) > 1 else None)
    S['pca'][state] = dict(cumulative=cum)
  # ---- the earlier PCA protocol: one scripted trajectory replayed through every network
  S['replay_pca'] = {}
  replay_rows = []
  for state, (pat, tag) in STATES.items():
    R = {c: {s: json.load(open(os.path.join(args.replay_dir, pat.format(c=c, s=s) + f'@{tag}.json')))
             for s in range(1, 9) if s not in excl and os.path.exists(os.path.join(args.replay_dir, pat.format(c=c, s=s) + f'@{tag}.json'))}
         for c in CONDS}
    seeds = sorted(set(R['gc100']) & set(R['pgc33']))
    if not seeds:
      continue
    S['replay_pca'][state] = {}
    for proto in ('familiar', 'changed'):
      for c in CONDS:
        for s_ in sorted(R[c]):
          r = R[c][s_][proto]
          replay_rows.append([state, c, s_, R[c][s_]['frames'], proto, r['samples'], r['var10'], r['participation_ratio'],
                              r['n50'], r['n90'], r['total_var']] + r['ratios'][:40])
      st = {m: paired([R['gc100'][s][proto][m] for s in seeds], [R['pgc33'][s][proto][m] for s in seeds])
            for m in ('var3', 'var10', 'var20', 'n50', 'n90', 'participation_ratio', 'total_var')}
      for c in CONDS:
        cs = np.array([np.cumsum(R[c][s][proto]['ratios']) for s in seeds])
        st[f'cumulative_{c}'] = dict(mean=[float(x) for x in cs.mean(0)[:40]],
                                     sem=[float(x) for x in (cs.std(0, ddof=1) / np.sqrt(len(seeds)))[:40]] if len(seeds) > 1 else None)
      st['seeds'] = seeds
      S['replay_pca'][state][proto] = st
    # PGC - grid difference of the first-10 share: changed-room replay minus familiar-room replay (difference of differences)
    S['replay_pca'][state]['changed']['var10_vs_familiar'] = one_sample(np.array(
        [(R['pgc33'][s]['changed']['var10'] - R['gc100'][s]['changed']['var10'])
         - (R['pgc33'][s]['familiar']['var10'] - R['gc100'][s]['familiar']['var10']) for s in seeds]))
    # change of each agent's own value against its 1 B state
    if state != 'before' and 'before' in STATES:
      pat0, tag0 = STATES['before']
      R0 = {c: {s: json.load(open(os.path.join(args.replay_dir, pat0.format(c=c, s=s) + f'@{tag0}.json')))
                for s in seeds if os.path.exists(os.path.join(args.replay_dir, pat0.format(c=c, s=s) + f'@{tag0}.json'))} for c in CONDS}
      if all(len(R0[c]) == len(seeds) for c in CONDS):
        for proto in ('familiar', 'changed'):
          dg = [R['gc100'][s][proto]['var10'] - R0['gc100'][s][proto]['var10'] for s in seeds]
          dp = [R['pgc33'][s][proto]['var10'] - R0['pgc33'][s][proto]['var10'] for s in seeds]
          S['replay_pca'][state][proto]['var10_change'] = dict(gc=one_sample(dg), pgc=one_sample(dp),
                                                               diff=one_sample(np.array(dp) - np.array(dg)))
  # ---- how much of the familiar-room score (1 B) is kept in the changed rooms
  S['retention'] = {}
  for state in STATES:
    seeds = sorted(set(common(state)) & set(common('before')))
    if not seeds or state.startswith('same'):
      continue
    S['retention'][state] = {}
    for g in ('changed', 'door', 'obstacle', 'door_obstacle', 'goal_moved', 'new_rooms', 'familiar'):
      r = {c: [group_value(D[state][c][s]['rows'], GROUPS[g], 'score') /
               group_value(D['before'][c][s]['rows'], GROUPS['familiar'], 'score') for s in seeds] for c in CONDS}
      S['retention'][state][g] = paired(r['gc100'], r['pgc33'])
  # ---- lesions in the grid + PGC agents: the 169 predictive cells silenced, or 169 random grid cells
  S['lesion'] = {}
  for state, (pat, tag) in STATES.items():
    rows = {}
    for s in [x for x in range(1, 9) if x not in excl]:
      base = os.path.join(args.eval_dir, pat.format(c='pgc33', s=s) + f'@{tag}')
      if not all(os.path.exists(f'{base}+{pr}.json') for pr in ('no_pgc', 'sham0', 'sham1')) or s not in D.get(state, {}).get('pgc33', {}):
        continue
      les = {pr: json.load(open(f'{base}+{pr}.json'))['conditions'] for pr in ('no_pgc', 'sham0', 'sham1')}
      rows[s] = {}
      for g in ('familiar', 'changed'):
        sc = lambda cd: float(np.mean([cd[rc]['score'] for rc in GROUPS[g]]))
        rows[s][g] = dict(intact=group_value(D[state]['pgc33'][s]['rows'], GROUPS[g], 'score'), no_pgc=sc(les['no_pgc']),
                          sham=(sc(les['sham0']) + sc(les['sham1'])) / 2)
    if len(rows) >= 2:
      S['lesion'][state] = {}
      for g in ('familiar', 'changed'):
        I = np.array([rows[s][g]['intact'] for s in rows])
        Pz = np.array([rows[s][g]['no_pgc'] for s in rows])
        H = np.array([rows[s][g]['sham'] for s in rows])
        S['lesion'][state][g] = dict(intact=float(I.mean()), no_pgc=float(Pz.mean()), sham=float(H.mean()),
                                     pgc_cost=one_sample(I - Pz), sham_cost=one_sample(I - H), specific=one_sample(H - Pz),
                                     seeds=sorted(rows))
  # change of the PGC-specific cost with fine-tuning (the same agents, later - before)
  S['lesion_change'] = {}
  for later in ('after', 'long'):
    if later in S['lesion'] and 'before' in S['lesion']:
      S['lesion_change'][later] = {}
      for g in ('familiar', 'changed'):
        b, a = S['lesion']['before'][g], S['lesion'][later][g]
        d = [a['specific']['seeds'][a['seeds'].index(s)] - b['specific']['seeds'][b['seeds'].index(s)] for s in a['seeds'] if s in b['seeds']]
        if len(d) >= 2:
          S['lesion_change'][later][g] = one_sample(np.array(d))
  # ---- the matched-budget reference (seeds 1-2)
  if 'same' in D:
    ref = {}
    for c in CONDS:
      for s in sorted(D['same'].get(c, {})):
        ref[f'{c}_s{s}'] = {st: {g: group_value(D[st][c][s]['rows'], GROUPS[g], 'score') for g in ('familiar', 'changed', 'new_rooms')}
                            for st in STATES if st in D and s in D[st].get(c, {})}
    S['reference'] = ref
    ref_seeds = sorted(set(D['same'].get('gc100', {})) & set(D['same'].get('pgc33', {})))
    S['reference_stats'] = {st: {g: paired(vals(st, 'gc100', ref_seeds, GROUPS[g], 'score'), vals(st, 'pgc33', ref_seeds, GROUPS[g], 'score'))
                                 for g in ('familiar', 'changed')}
                            for st in STATES if st in D and all(s in D[st].get(c, {}) for c in CONDS for s in ref_seeds)}
  # ---- transfer table (Taylor & Stone 2009; Lopez-Paz & Ranzato 2017): jumpstart, gain, backward transfer
  S['transfer'] = {}
  per_agent = {}
  for later in ('after', 'long'):
    seeds = sorted(set(common('before')) & set(common(later)))
    if not seeds:
      continue
    T = {}
    for c in CONDS:
      b_open = np.array(vals('before', c, seeds, GROUPS['familiar'], 'score'))
      b_ch = np.array(vals('before', c, seeds, GROUPS['changed'], 'score'))
      a_open = np.array(vals(later, c, seeds, GROUPS['familiar'], 'score'))
      a_ch = np.array(vals(later, c, seeds, GROUPS['changed'], 'score'))
      T[c] = dict(jumpstart=b_ch / b_open, gain=a_ch - b_ch, backward=a_open - b_open, recovered=a_ch / b_open)
      per_agent[(later, c)] = dict(seeds=seeds, **{k: v.tolist() for k, v in T[c].items()})
    S['transfer'][later] = {k: paired(T['gc100'][k], T['pgc33'][k]) for k in ('jumpstart', 'gain', 'backward', 'recovered')}
    # is the familiar-room loss the price of the changed-room gain? (description: Spearman over agents)
    S.setdefault('gain_vs_forgetting', {})[later] = {}
    for key, cs in (('gc100', ['gc100']), ('pgc33', ['pgc33']), ('pooled', CONDS)):
      r = stats.spearmanr(np.concatenate([T[c]['gain'] for c in cs]), np.concatenate([T[c]['backward'] for c in cs]))
      S['gain_vs_forgetting'][later][key] = dict(n=len(seeds) * len(cs), rho=float(r.statistic), p=float(r.pvalue))
  for st in ('same', 'same_long'):       # the control agents: change in the familiar room and in the changed rooms
    if st in D:
      S['transfer'][st] = {f'{c}_s{s}': dict(
          backward=group_value(D[st][c][s]['rows'], GROUPS['familiar'], 'score') - group_value(D['before'][c][s]['rows'], GROUPS['familiar'], 'score'),
          gain=group_value(D[st][c][s]['rows'], GROUPS['changed'], 'score') - group_value(D['before'][c][s]['rows'], GROUPS['changed'], 'score'))
          for c in CONDS for s in sorted(D[st].get(c, {}))}
  # ---- alignment index (Elsayed et al. 2016) between a network's own top-10 subspaces, scripted replay
  def top10(C):
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :10], float(w[::-1][:10].sum())

  def align(C_ref, C_x):
    Vx, _ = top10(C_x)
    return float(np.trace(Vx.T @ C_ref @ Vx) / top10(C_ref)[1])

  S['alignment'] = {}
  covs = {}
  for state, (pat, tag) in STATES.items():
    for c in CONDS:
      for s in range(1, 9):
        f = os.path.join(args.replay_dir, pat.format(c=c, s=s) + f'@{tag}.npz')
        if os.path.exists(f) and s not in excl:
          z = np.load(f)
          covs[(state, c, s)] = dict(familiar=z['cov_familiar'], changed=z['cov_changed'])
  for state in STATES:
    seeds = [s for s in range(1, 9) if all((state, c, s) in covs for c in CONDS)]
    if len(seeds) < 2:
      continue
    S['alignment'][state] = {}
    # familiar-room variance captured by the top-10 subspace of the changed-room replay (same network)
    S['alignment'][state]['familiar_vs_changed'] = paired(
        [align(covs[(state, 'gc100', s)]['familiar'], covs[(state, 'gc100', s)]['changed']) for s in seeds],
        [align(covs[(state, 'pgc33', s)]['familiar'], covs[(state, 'pgc33', s)]['changed']) for s in seeds])
    if state != 'before':
      seeds0 = [s for s in seeds if all(('before', c, s) in covs for c in CONDS)]
      for proto in ('familiar', 'changed'):   # 1 B variance captured by the later network's top-10 subspace (same inputs)
        if len(seeds0) >= 2:
          S['alignment'][state][f'before_vs_{state}_{proto}'] = paired(
              [align(covs[('before', 'gc100', s)][proto], covs[(state, 'gc100', s)][proto]) for s in seeds0],
              [align(covs[('before', 'pgc33', s)][proto], covs[(state, 'pgc33', s)][proto]) for s in seeds0])
  # ---- does the concentration of variance at 1 B go with transfer or adaptation? (description only)
  S['rank_vs_adaptation'] = {}
  for later in ('after', 'long'):
    if ('after' if later == 'after' else 'long', 'gc100') not in per_agent:
      continue
    out = {}
    xs, js, gs = [], [], []
    for c in CONDS:
      pa = per_agent[(later, c)]
      pat0, tag0 = STATES['before']
      v10 = []
      for s in pa['seeds']:
        f = os.path.join(args.replay_dir, pat0.format(c=c, s=s) + f'@{tag0}.json')
        v10.append(json.load(open(f))['familiar']['var10'] if os.path.exists(f) else np.nan)
      v10 = np.array(v10)
      ok = ~np.isnan(v10)
      if ok.sum() >= 4:
        rj = stats.spearmanr(v10[ok], np.array(pa['jumpstart'])[ok])
        rg = stats.spearmanr(v10[ok], np.array(pa['gain'])[ok])
        out[c] = dict(n=int(ok.sum()), rho_jumpstart=float(rj.statistic), p_jumpstart=float(rj.pvalue),
                      rho_gain=float(rg.statistic), p_gain=float(rg.pvalue), var10=v10.tolist(),
                      jumpstart=pa['jumpstart'], gain=pa['gain'])
        xs += v10[ok].tolist(); js += np.array(pa['jumpstart'])[ok].tolist(); gs += np.array(pa['gain'])[ok].tolist()
    if len(xs) >= 8:
      rj, rg = stats.spearmanr(xs, js), stats.spearmanr(xs, gs)
      out['pooled'] = dict(n=len(xs), rho_jumpstart=float(rj.statistic), p_jumpstart=float(rj.pvalue),
                           rho_gain=float(rg.statistic), p_gain=float(rg.pvalue))
    S['rank_vs_adaptation'][later] = out
  # ---- training curves
  cv = curves(args.runs_dir, long_m or 1250)
  cv_all = cv
  cv = [r for r in cv if r['seed'] not in excl]
  S['curves'] = {}
  for kind in ('room', 'same'):
    for c in CONDS:
      runs = sorted({r['run'] for r in cv if r['kind'] == kind and r['cond'] == c})
      if not runs:
        continue
      xs = sorted({r['frames_m'] for r in cv if r['kind'] == kind and r['cond'] == c})
      xs = [x for x in xs if all(any(r['run'] == run and r['frames_m'] == x for r in cv) for run in runs)]
      for key in ('ret_all', 'ret_door', 'ret_obstacle', 'ret_door_obstacle', 'ret_relocation'):
        M = np.array([[next(r[key] for r in cv if r['run'] == run and r['frames_m'] == x) for x in xs] for run in runs])
        S['curves'][f'{kind}/{c}/{key}'] = dict(x=xs, mean=[float(v) for v in np.nanmean(M, 0)],
                                                sem=[float(v) for v in (np.nanstd(M, 0, ddof=1) / np.sqrt(len(runs)))] if len(runs) > 1 else None,
                                                runs=runs, per_run=[[float(v) for v in row] for row in M])
  # paired gap of the training return, per 25 M frames (same seeds in both codes)
  if 'room/gc100/ret_all' in S['curves'] and 'room/pgc33/ret_all' in S['curves']:
    cg, cp = S['curves']['room/gc100/ret_all'], S['curves']['room/pgc33/ret_all']
    sg = {int(r.rsplit('_s', 1)[1]): i for i, r in enumerate(cg['runs'])}
    sp = {int(r.rsplit('_s', 1)[1]): i for i, r in enumerate(cp['runs'])}
    seeds = sorted(set(sg) & set(sp))
    xs = [x for x in cg['x'] if x in cp['x']]
    gap = []
    for lo in range(1000, int(max(xs)) + 1, 25):
      sel = [x for x in xs if lo < x <= lo + 25]
      if len(sel) < 5:
        continue
      g = np.array([np.mean([cg['per_run'][sg[s]][cg['x'].index(x)] for x in sel]) for s in seeds])
      p = np.array([np.mean([cp['per_run'][sp[s]][cp['x'].index(x)] for x in sel]) for s in seeds])
      st = paired(g, p)
      st['frames_m'] = lo + 25
      gap.append(st)
    S['curve_gap'] = gap
    # growth of the lead over the planned 250 M frames: gain of each code from the first window to 225-250 M, paired
    w = {st_['frames_m']: st_ for st_ in gap}
    if 1025 in w and 1250 in w:
      S['curve_growth'] = paired(np.array(w[1250]['gc_seeds']) - np.array(w[1025]['gc_seeds']),
                                 np.array(w[1250]['pgc_seeds']) - np.array(w[1025]['pgc_seeds']))
  # time to threshold (Taylor & Stone 2009): frames of fine-tuning until the 25 M-frame running mean of the
  # training score in the changed rooms reaches 50 % / 90 % of the agent's own familiar-room training score
  # over its last 25 M frames before the change; None = not reached
  S['time_to_threshold'] = {}
  for c in CONDS:
    key = f'room/{c}/ret_all'
    if key not in S['curves']:
      continue
    cu = S['curves'][key]
    res = {}
    for run, row in zip(cu['runs'], cu['per_run']):
      seed = int(run.rsplit('_s', 1)[1])
      base_csv = os.path.join(ROOT, 'runs_ext', f'{c}_lstm_s{seed}', 'episodes.csv')
      if not os.path.exists(base_csv):
        continue
      fr, ret = [], []
      with open(base_csv) as f:
        for r in csv.DictReader(f):
          if float(r['frames']) > 975e6:
            fr.append(float(r['frames'])), ret.append(float(r['ret']))
      base = float(np.mean(ret))
      sm = np.convolve(np.array(row), np.ones(5) / 5, mode='valid')          # 5 bins of 5 M frames
      xs_ = np.array(cu['x'][4:]) + 2.5 - 1000.0                              # frames of fine-tuning at the window end (M)
      hit = {f'{int(q * 100)}': (float(xs_[np.argmax(sm >= q * base)]) if (sm >= q * base).any() else None) for q in (0.5, 0.9)}
      res[run] = dict(familiar_training_score=base, best=float(sm.max()), **hit)
    S['time_to_threshold'][c] = res
  # ---- the small pre-declared family of confirmatory contrasts, Holm-adjusted (exact sign-flip p)
  fam = []
  def add(label, st):
    if st and st.get('p_perm') is not None:
      fam.append(dict(label=label, estimate=st.get('diff', st.get('mean')), ci95=st.get('ci95'), p_t=st.get('p'),
                      p_perm=st['p_perm'], n=st['n'], pos=st.get('pos')))
  last = 'after'
  if last in S['stats']:
    add('score gap, changed rooms, before (frozen 1 B agents)', S['stats']['before']['changed']['score'])
    add('score gap, changed rooms, after 250 M frames', S['stats'][last]['changed']['score'])
    add('change of the score gap in the changed rooms with fine-tuning', S['change']['before->after']['changed']['score']['diff'])
    add('change of the familiar-room score with fine-tuning (after − before)', S['change']['before->after']['familiar']['score']['diff'])
    if last in S['replay_pca']:
      add('first-10 variance share, familiar-room replay, after 250 M frames', S['replay_pca'][last]['familiar']['var10'])
      add('first-10 variance share, changed-room replay, after 250 M frames', S['replay_pca'][last]['changed']['var10'])
    order = np.argsort([f['p_perm'] for f in fam])
    running = 0.0
    for rank, i in enumerate(order):
      running = max(running, min(1.0, fam[i]['p_perm'] * (len(fam) - rank)))
      fam[i]['p_holm'] = float(running)
  S['confirmatory'] = fam
  # the PGC - grid lead in the familiar room minus the lead in the changed rooms (difference of differences), per state
  S['gap_drop'] = {}
  for state in S['stats']:
    f_, c_ = S['stats'][state]['familiar']['score'], S['stats'][state]['changed']['score']
    if f_.get('n') == c_.get('n') and f_.get('n', 0) >= 2:
      S['gap_drop'][state] = one_sample((np.array(f_['pgc_seeds']) - np.array(f_['gc_seeds']))
                                        - (np.array(c_['pgc_seeds']) - np.array(c_['gc_seeds'])))
  S['excluded_seeds'] = excl
  json.dump(S, open(os.path.join(args.out, f'summary{suffix}.json'), 'w'))

  # ---- data files (all seeds only)
  cv = cv_all
  if excl:
    args.data = os.path.join('/tmp', 'room_change_data' + suffix)
    os.makedirs(args.data, exist_ok=True)
  with open(os.path.join(args.data, 'room_eval.csv'), 'w', newline='') as f:
    cols = ['state', 'code', 'seed', 'frames', 'room', 'layout', 'goal_moved'] + list(METRICS) + list(RELOC_METRICS) + ['sem_score']
    w = csv.writer(f)
    w.writerow(cols)
    for state in D:
      for c in D[state]:
        for s in sorted(D[state][c]):
          for rc, r in D[state][c][s]['rows'].items():
            w.writerow([state, c, s, D[state][c][s]['frames'], rc, rc[:-2] if rc.endswith('+R') else rc, int(rc.endswith('+R'))]
                       + [('' if r.get(m) is None else f'{r[m]:.6g}') for m in METRICS + RELOC_METRICS] + [f'{r["sem_score"]:.4g}'])
  with open(os.path.join(args.data, 'pca_spectra.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['state', 'code', 'seed', 'room'] + [f'ev{i + 1}' for i in range(40)] + ['total_var'])
    for state in D:
      for c in D[state]:
        for s in sorted(D[state][c]):
          for rc, r in D[state][c][s]['rows'].items():
            w.writerow([state, c, s, rc] + [f'{x:.6g}' for x in r['eig_h'][:40]] + [f'{sum(r["eig_h"]):.6g}'])
  if replay_rows:
    with open(os.path.join(args.data, 'pca_replay.csv'), 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['state', 'code', 'seed', 'frames', 'replay_rooms', 'samples', 'var10', 'participation_ratio', 'n50', 'n90',
                  'total_var'] + [f'ratio{i + 1}' for i in range(40)])
      for r in replay_rows:
        w.writerow([(f'{x:.6g}' if isinstance(x, float) else x) for x in r])
  if cv:
    with open(os.path.join(args.data, 'finetune_curves.csv'), 'w', newline='') as f:
      w = csv.DictWriter(f, fieldnames=list(cv[0].keys()))
      w.writeheader()
      for r in cv:
        w.writerow({k: (f'{v:.6g}' if k == 'frames_m' else f'{v:.4g}') if isinstance(v, float) else v for k, v in r.items()})

  # ---- tables
  extra = (long_m - 1000) if long_m else None
  title = {'before': 'Before fine-tuning (1 B agents, frozen)', 'after': 'After 250 M frames in the changed rooms',
           'long': f'After {extra} M frames in the changed rooms', 'same': 'After 250 M more frames in the familiar room (seeds 1–2)',
           'same_long': f'After {extra} M more frames in the familiar room (seeds 1–2)'}
  hdr = ['| statistic | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|']
  L = ['# Room-change experiment: tables', '',
       'grid = 100 % grid cells; PGC = grid + 33 % predictive grid cells; difference = PGC − grid, mean ± SEM over seeds, '
       'two-sided paired t-test; "k/n" = seeds in which PGC is higher.', '']
  for state in ('before', 'after', 'long', 'same', 'same_long'):
    if state not in S['stats']:
      continue
    St = S['stats'][state]
    L += [f'## {title[state]}', '', '### Score by room type', '', '| rooms | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|']
    L += [row_md(NAMES[g], St[g]['score']) for g in ('familiar', 'changed', 'door', 'obstacle', 'door_obstacle', 'goal_moved',
                                                    'layout_and_goal_moved', 'new_rooms', 'new_rooms_goal_moved')]
    for g in ('no_relocation', 'familiar'):
      L += ['', f'### Navigation, {NAMES[g]}', ''] + hdr
      L += [row_md(MNAME[m][0], St[g][m], MNAME[m][1]) for m in
            ('frac_found', 'first_found_step', 'first_goal_step', 'post_rate', 'latency_ratio', 'path_eff', 'wall_contact')]
    for g in ('goal_moved', 'layout_and_goal_moved', 'relocation'):
      L += ['', f'### Goal relocation, {NAMES[g]}', ''] + hdr
      L += [row_md(MNAME[m][0], St[g][m], MNAME[m][1]) for m in RELOC_METRICS if St[g][m].get('n', 0)]
    L += ['', '### PCA of the LSTM output during the agent\'s own behaviour', ''] + hdr
    for g in ('familiar', 'no_relocation'):
      for m in ('var10_return', 'var10', 'frac_return'):
        L.append(row_md(f'{MNAME[m][0]}, {NAMES[g]}', St[g][m], 3))
    L += ['', '### Every room (score)', '', '| room | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|']
    L += [row_md(rc, S['per_condition'][state][rc]['score']) for rc in TEST_CONDITIONS]
    L.append('')
  if S['replay_pca']:
    L += ['## PCA with the earlier protocol (one scripted trajectory replayed through every network)', '',
          '| agents | replay in | quantity | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|---|---|']
    lab = {'before': '1 B', 'after': '1.25 B, changed rooms', 'long': f'{(long_m or 0) / 1000:g} B, changed rooms',
           'same': '1.25 B, familiar room (seeds 1–2)', 'same_long': f'{(long_m or 0) / 1000:g} B, familiar room (seeds 1–2)'}
    for state in ('before', 'after', 'long', 'same', 'same_long'):
      for proto in ('familiar', 'changed'):
        if state in S['replay_pca']:
          st = S['replay_pca'][state][proto]
          pre = f'| {lab[state]} | {proto} room' + ('s ' if proto == 'changed' else ' ')
          L.append(pre + row_md('variance in first 10 components', st['var10'], 4))
          L.append(pre + row_md('participation ratio', st['participation_ratio'], 2))
          L.append(pre + row_md('total variance of the 256 outputs', st['total_var'], 2))
    L.append('')
    L += ['Change of each agent\'s first-10 share against its own 1 B value:', '',
          '| agents | replay in | grid | PGC | PGC − grid | p |', '|---|---|---|---|---|---|']
    for state in ('after', 'long', 'same', 'same_long'):
      for proto in ('familiar', 'changed'):
        ch = S['replay_pca'].get(state, {}).get(proto, {}).get('var10_change')
        if ch:
          L.append(f'| {lab[state]} | {proto} | {ch["gc"]["mean"]:+.4f} ± {fmt(ch["gc"]["sem"], 4)} | {ch["pgc"]["mean"]:+.4f} ± {fmt(ch["pgc"]["sem"], 4)} | '
                   f'{ch["diff"]["mean"]:+.4f} ± {fmt(ch["diff"]["sem"], 4)} | {fmt_p(ch["diff"]["p"])} |')
    L.append('')
  for key, ttl in (('before->after', 'Change over the first 250 M frames in the changed rooms (after − before)'),
                   ('before->long', f'Change over all {extra} M frames (long − before)'),
                   ('after->long', f'Change from 250 M to {extra} M frames')):
    if key not in S['change']:
      continue
    L += [f'## {ttl}, per code', '', '| rooms | statistic | grid | PGC | PGC − grid | p |', '|---|---|---|---|---|---|']
    for g, mets in (('familiar', ('score', 'path_eff', 'post_rate')), ('changed', ('score',)),
                    ('no_relocation', ('score', 'path_eff', 'post_rate', 'wall_contact', 'first_goal_step')),
                    ('door', ('score',)), ('obstacle', ('score',)), ('door_obstacle', ('score',)),
                    ('goal_moved', ('goals_post_knew', 'refound_knew', 'perseveration_knew')),
                    ('relocation', ('goals_post_knew', 'refound_knew', 'perseveration_knew')), ('new_rooms', ('score',))):
      for m in mets:
        c = S['change'][key][g][m]
        if c['gc'].get('mean') is None:
          continue
        nd = MNAME[m][1]
        L.append(f'| {NAMES[g]} | {MNAME[m][0]} | {c["gc"]["mean"]:+.{nd}f} ± {fmt(c["gc"]["sem"], nd)} | {c["pgc"]["mean"]:+.{nd}f} ± {fmt(c["pgc"]["sem"], nd)} | '
                 f'{c["diff"]["mean"]:+.{nd}f} ± {fmt(c["diff"]["sem"], nd)} | {fmt_p(c["diff"]["p"])} |')
    L.append('')
  if S.get('confirmatory'):
    L += ['## Confirmatory contrasts (pre-declared; PGC − grid; exact sign-flip p over 8 seeds, Holm-adjusted)', '',
          '| contrast | estimate | 95 % CI | p (t) | p (sign-flip) | p (Holm) | k/n |', '|---|---|---|---|---|---|---|']
    for f in S['confirmatory']:
      nd = 4 if 'share' in f['label'] else 1
      L.append(f'| {f["label"]} | {f["estimate"]:+.{nd}f} | ± {f["ci95"]:.{nd}f} | {fmt_p(f["p_t"])} | {fmt_p(f["p_perm"])} | {fmt_p(f["p_holm"])} | {f["pos"]}/{f["n"]} |')
    L.append('')
  if S.get('curve_growth'):
    g_ = S['curve_growth']
    L += ['Training score in the changed rooms, last 25 M window of the planned 250 M minus the first: '
          f'grid {g_["gc"]:+.1f}, PGC {g_["pgc"]:+.1f}, difference {g_["diff"]:+.1f} ± {fmt(g_["sem"])} (p {fmt_p(g_["p"])}, {g_["pos"]}/{g_["n"]} seeds)', '']
  if S.get('gap_drop'):
    L += ['PGC − grid lead in the familiar room minus the lead in the changed rooms: '
          + '; '.join(f'{k_} {v_["mean"]:+.1f} ± {fmt(v_["sem"])} (p {fmt_p(v_["p"])})' for k_, v_ in S['gap_drop'].items()), '']
  if S.get('transfer'):
    tl = {'jumpstart': ('jumpstart: changed-room score before fine-tuning / familiar-room score', 3),
          'gain': ('gain: changed-room score, later − before', 1),
          'backward': ('backward transfer: familiar-room score, later − before', 1),
          'recovered': ('changed-room score after / familiar-room score before', 3)}
    for later in ('after', 'long'):
      if later in S['transfer']:
        L += [f'## Transfer table: {title[later]}', '', '| quantity | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|']
        L += [row_md(tl[k][0], S['transfer'][later][k], tl[k][1]) for k in ('jumpstart', 'gain', 'backward', 'recovered')]
        L.append('')
        if later in S.get('gain_vs_forgetting', {}):
          L += ['Gain in the changed rooms against the change in the familiar room, Spearman over agents: '
                + '; '.join(f'{k} rho {v["rho"]:+.2f} (p {fmt_p(v["p"])}, n = {v["n"]})' for k, v in S['gain_vs_forgetting'][later].items()), '']
    for st in ('same', 'same_long'):
      if st in S['transfer']:
        L += [f'Control agents ({title[st]}): ' + '; '.join(f'{a} familiar {v["backward"]:+.1f}, changed {v["gain"]:+.1f}' for a, v in S['transfer'][st].items()), '']
  if S.get('time_to_threshold'):
    L += ['## Time to threshold during fine-tuning (frames, M; training score in the changed rooms, 25 M running mean)', '',
          '| run | familiar-room training score before | best changed-room score | 50 % reached at | 90 % reached at |', '|---|---|---|---|---|']
    for c in CONDS:
      for run, v in S['time_to_threshold'].get(c, {}).items():
        L.append(f'| {run} | {v["familiar_training_score"]:.1f} | {v["best"]:.1f} | {v["50"] if v["50"] is not None else "not reached"} | {v["90"] if v["90"] is not None else "not reached"} |')
    L.append('')
  if S.get('alignment'):
    L += ['## Alignment index of top-10 subspaces (scripted replay; 1 = same subspace)', '',
          '| agents | comparison | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|---|']
    for state in S['alignment']:
      for k, st in S['alignment'][state].items():
        L.append(f'| {state} ' + row_md(k.replace('_', ' '), st, 3))
    L.append('')
  if S.get('rank_vs_adaptation'):
    L += ['## Variance concentration at 1 B against transfer (Spearman, description only)', '',
          '| fine-tuning | agents | n | first-10 share vs jumpstart | first-10 share vs gain |', '|---|---|---|---|---|']
    for later, o in S['rank_vs_adaptation'].items():
      for c, v in o.items():
        L.append(f'| {later} | {c} | {v["n"]} | rho {v["rho_jumpstart"]:+.2f} (p {fmt_p(v["p_jumpstart"])}) | rho {v["rho_gain"]:+.2f} (p {fmt_p(v["p_gain"])}) |')
    L.append('')
  if S.get('curve_gap'):
    L += ['## Training score in the changed rooms during fine-tuning, per 25 M frames', '',
          '| frames (M) | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|']
    L += [row_md(f'{st["frames_m"] - 25}–{st["frames_m"]}', st) for st in S['curve_gap']]
    L.append('')
  if S['retention']:
    L += ['## Score as a share of the same agent\'s familiar-room score at 1 B', '',
          '| state | rooms | grid | PGC | difference | p | k/n |', '|---|---|---|---|---|---|---|']
    for state in S['retention']:
      for g, st in S['retention'][state].items():
        if not (state == 'before' and g == 'familiar'):
          L.append(f'| {state} ' + row_md(NAMES[g], st, 3))
    L.append('')
  if S['lesion']:
    L += ['## Lesions in the grid + PGC agents (score)', '',
          '| state | rooms | intact | 169 PGCs silenced | 169 random grid cells silenced | PGC-specific cost (random − PGC) | p | n |',
          '|---|---|---|---|---|---|---|---|']
    for state in S['lesion']:
      for g, st in S['lesion'][state].items():
        sp = st['specific']
        L.append(f'| {state} | {NAMES[g]} | {st["intact"]:.1f} | {st["no_pgc"]:.1f} | {st["sham"]:.1f} | '
                 f'{sp["mean"]:+.1f} ± {fmt(sp["sem"])} | {fmt_p(sp["p"])} | {sp["n"]} |')
    L.append('')
    for later, o in S.get('lesion_change', {}).items():
      L += [f'Change of the PGC-specific cost, {later} − before: '
            + '; '.join(f'{NAMES[g]} {v["mean"]:+.1f} ± {fmt(v["sem"])} (p {fmt_p(v["p"])}, lower in {v["n"] - v["pos"]}/{v["n"]} agents)' for g, v in o.items()), '']
  if S['reference']:
    L += ['## Matched-budget reference (seeds 1–2): score', '', '| agent | state | familiar room | changed rooms | new rooms |', '|---|---|---|---|---|']
    lab = {'before': '1 B', 'after': '1.25 B, last 250 M in the changed rooms', 'same': '1.25 B, all in the familiar room',
           'long': f'{(long_m or 0) / 1000:g} B, last {extra} M in the changed rooms', 'same_long': f'{(long_m or 0) / 1000:g} B, all in the familiar room'}
    for a, v in S['reference'].items():
      for st in ('before', 'same', 'same_long', 'after', 'long'):
        if st in v:
          L.append(f'| {a} | {lab[st]} | {v[st]["familiar"]:.1f} | {v[st]["changed"]:.1f} | {v[st]["new_rooms"]:.1f} |')
    L.append('')
  if excl:
    L.insert(2, f'**Sensitivity analysis: seed(s) {", ".join(str(x) for x in excl)} left out (both codes).**')
    L.insert(3, '')
  open(os.path.join(args.out, f'RESULTS{suffix}.md'), 'w').write('\n'.join(L))
  print('\n'.join(L[:40]))


if __name__ == '__main__':
  main()
