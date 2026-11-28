"""Analysis of the code-variation experiment (plan: results/code_variations/PLAN.md).

Reads the collected run directories runs_codes/<block>/<run>/ (episodes.csv,
metrics.jsonl, train.log, config.json, eval/*.json) and writes

  results/code_variations/summary.json     every number used in the write-up
  results/code_variations/RESULTS.md       all tables
  results/code_variations/data/*.csv       scores per run, learning curves, final tests, learned look-aheads

Outcome (fixed in the plan): the training score, i.e. the mean reward per
episode over the last 25 M frames before a mark (300 M and 600 M frames; the
last 50 M before 250 M and 500 M for the LSTM agents). Comparisons are paired
within seed: paired t-test, 95 % confidence interval, exact sign-flip p, and
Holm correction inside each list of named comparisons.

  python -m scripts.codes.analyse                # all blocks
  python -m scripts.codes.analyse --marks 20,40  # early look while the runs are going (not the planned marks)
"""
import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scripts.codes.matrix import DOOR_CORE, LABEL, LSTM_CODES, OPEN_CORE, OPEN_SCREEN  # noqa: E402

BASE = os.path.join(ROOT, 'runs_codes')
OUT = os.path.join(ROOT, 'results', 'code_variations')
BLOCKS = ('oc', 'os', 'dr', 'tr', 'lstm', 'trl', 'free')
M = 1_000_000
SPACINGS_CM = [40.0, 56.8, 80.656, 114.53]


# ------------------------------------------------------------------ reading
def read_episodes(run_dir):
  """(frames, reward) of every logged episode, in training order. Episodes
  logged after a saved state and before an interruption are logged again when
  the run resumes from that state; the first copy is dropped."""
  path = os.path.join(run_dir, 'episodes.csv')
  fr, ret = [], []
  if not os.path.exists(path):
    return np.zeros(0, np.int64), np.zeros(0)
  resumes = []
  log = os.path.join(run_dir, 'train.log')
  if os.path.exists(log):
    with open(log, errors='replace') as f:
      resumes = [int(x) for x in re.findall(r'resumed at frames=(\d+)', f.read())]
  with open(path, errors='replace') as f:
    next(f, None)
    for line in f:
      p = line.split(',')
      try:
        a, b = int(p[0]), float(p[3])
      except (ValueError, IndexError):
        continue          # a line cut off by an interruption
      if fr and a < fr[-1]:
        # the run went back to a saved state: drop what was logged after it
        start = max([r for r in resumes if r <= a] or [a - 12800])
        while fr and fr[-1] > start:
          fr.pop()
          ret.pop()
      fr.append(a)
      ret.append(b)
  return np.asarray(fr, np.int64), np.asarray(ret, float)


def window(fr, ret, lo, hi):
  m = (fr > lo) & (fr <= hi)
  return (float(ret[m].mean()), int(m.sum())) if m.any() else (float('nan'), 0)


def read_metrics(run_dir, keys):
  out = []
  path = os.path.join(run_dir, 'metrics.jsonl')
  if not os.path.exists(path):
    return out
  with open(path, errors='replace') as f:
    for line in f:
      try:
        j = json.loads(line)
      except ValueError:
        continue
      if all(k in j for k in keys):
        while out and out[-1][0] >= j['frames']:      # the run went back to a saved state
          out.pop()
        out.append([j['frames']] + [j[k] for k in keys])
  return out


def read_evals(run_dir):
  ev = {}
  for f in glob.glob(os.path.join(run_dir, 'eval', '*.json')):
    name = os.path.basename(f)[:-5]
    if '@' in name:
      continue
    try:
      j = json.load(open(f))
      ev[name] = dict(score=j['mean_score'], frames=j.get('frames'), path_eff=j.get('mean_path_eff'),
                      first_goal=j.get('mean_first_goal_step'), post_rate=j.get('post_discovery_goal_rate'))
    except Exception:
      pass
  return ev


def load(blocks, base=BASE):
  runs = {}
  for tag in blocks:
    for cfg_path in sorted(glob.glob(os.path.join(base, tag, '*', 'config.json'))):
      d = os.path.dirname(cfg_path)
      name = os.path.basename(d)
      m = re.match(r'^(?P<code>[A-Za-z0-9]+)_(?P<env>open|door|mix)_(?P<arm>ffb|lstm)_s(?P<seed>\d+)$', name)
      if not m:
        continue
      try:
        cfg = json.load(open(cfg_path))
      except ValueError:
        continue
      fr, ret = read_episodes(d)
      runs[name] = dict(name=name, tag=tag, dir=d, code=m['code'], env=m['env'], arm=m['arm'],
                        seed=int(m['seed']), budget=int(cfg.get('frames', 0)), fr=fr, ret=ret,
                        frames_done=int(fr.max()) if len(fr) else 0, evals=read_evals(d),
                        learn=int(cfg.get('learn_shift') or 0))
  return runs


# -------------------------------------------------------------------- stats
def flip_p(d):
  d = np.asarray(d, float)
  if len(d) < 2 or len(d) > 16:
    return None
  signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
  return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def one_sample(d):
  d = np.asarray([x for x in d if x == x], float)
  n = len(d)
  if n < 2:
    return dict(n=n, diff=float(d.mean()) if n else None, sem=None, ci95=None, p=None, p_flip=None,
                pos=int((d > 0).sum()), seeds=[float(x) for x in d])
  sd = d.std(ddof=1)
  return dict(n=n, diff=float(d.mean()), sem=float(sd / np.sqrt(n)), sd=float(sd),
              ci95=float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)),
              p=float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else None, p_flip=flip_p(d),
              pos=int((d > 0).sum()), seeds=[float(x) for x in d])


def paired(sa, sb):
  """sa, sb: {seed: score}. Statistics of a - b over the seeds both have."""
  seeds = sorted(set(sa) & set(sb))
  seeds = [s for s in seeds if sa[s] == sa[s] and sb[s] == sb[s]]
  st = one_sample([sa[s] - sb[s] for s in seeds])
  st.update(a=float(np.mean([sa[s] for s in seeds])) if seeds else None,
            b=float(np.mean([sb[s] for s in seeds])) if seeds else None, seed_ids=seeds)
  return st


def holm(tests):
  """Adds p_holm to a list of statistics (those with a p value)."""
  idx = [i for i, t in enumerate(tests) if t.get('p') is not None]
  order = sorted(idx, key=lambda i: tests[i]['p'])
  m, running = len(order), 0.0
  for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * tests[i]['p']))
    tests[i]['p_holm'] = running
  return tests


def group_mean(sc):
  v = [x for x in sc.values() if x == x]
  n = len(v)
  return dict(n=n, mean=float(np.mean(v)) if n else None,
              sem=float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else None,
              seeds={int(k): float(x) for k, x in sc.items()})


# ----------------------------------------------------------------- analysis
def scores_at(runs, env, arm, mark, win):
  """{code: {seed: training score}} of the runs that reached the mark."""
  out = {}
  for r in runs.values():
    if r['env'] != env or r['arm'] != arm or r['learn']:
      continue
    if r['frames_done'] < mark * M - 1.5 * M or r['budget'] < mark * M - M:
      continue
    s, n = window(r['fr'], r['ret'], (mark - win) * M, mark * M)
    if n:
      out.setdefault(r['code'], {})[r['seed']] = s
  return out


def final_tests(runs, env, arm, budget_m):
  """{code: {seed: {probe: score}}} for finished runs of that budget."""
  out = {}
  for r in runs.values():
    if r['env'] != env or r['arm'] != arm or r['learn'] or 'intact' not in r['evals']:
      continue
    if abs(r['budget'] - budget_m * M) > M:
      continue
    out.setdefault(r['code'], {})[r['seed']] = {k: v['score'] for k, v in r['evals'].items()}
  return out


def contrast_block(sc, named, descriptive, ref_all=('u20', 'gc100')):
  res = dict(codes={c: group_mean(v) for c, v in sc.items()}, named=[], descriptive=[], vs={})
  for a, b in named:
    if a in sc and b in sc:
      res['named'].append(dict(paired(sc[a], sc[b]), name=f'{a} - {b}', a_code=a, b_code=b))
  holm(res['named'])
  for a, b in descriptive:
    if a in sc and b in sc:
      res['descriptive'].append(dict(paired(sc[a], sc[b]), name=f'{a} - {b}', a_code=a, b_code=b))
  for ref in ref_all:
    if ref in sc:
      res['vs'][ref] = {c: paired(sc[c], sc[ref]) for c in sc if c != ref}
  return res


def many_vs_one(sc):
  pairs = [('rand', 'u20'), ('mec', 'prop27'), ('gard', 'prop10')]
  if not all(a in sc and b in sc for a, b in pairs):
    return None
  seeds = sorted(set.intersection(*[set(sc[c]) for p in pairs for c in p]))
  d = [np.mean([sc[a][s] - sc[b][s] for a, b in pairs]) for s in seeds]
  return dict(one_sample(d), name='many distances - one distance (mean of three matched pairs)', seed_ids=seeds)


# Codes with 169 predictive cells (the lesion of the predictive cells is count-matched to the
# random lesions only for these) and codes with any look-ahead at all.
PGC169 = ('u20', 'prop10', 'prop27', 'mec', 'gard', 'rand', 'two', 'side', 'lr30', 'mecreal')
SHIFTED = PGC169 + ('p100',)


def lesions(ft):
  """Costs of switching cells off at test time, per code: mean over seeds."""
  out = {}
  for code, by_seed in ft.items():
    rows = dict(pgc_specific=[], pgc_as_gc=[], no_code=[], no_goal=[], intact=[])
    for s, ev in by_seed.items():
      if 'intact' not in ev:
        continue
      rows['intact'].append(ev['intact'])
      sham = [ev[k] for k in ('sham0', 'sham1', 'sham2') if k in ev]
      if 'no_pgc' in ev and sham and code in PGC169:
        rows['pgc_specific'].append(float(np.mean(sham)) - ev['no_pgc'])
      for k in ('pgc_as_gc', 'no_code', 'no_goal'):
        if k in ev and (k != 'pgc_as_gc' or code in SHIFTED) and code != 'none':
          rows[k].append(ev['intact'] - ev[k])
    out[code] = {k: one_sample(v) for k, v in rows.items() if v}
  return out


def learned(runs):
  """Trainable look-aheads: trajectory and end value per run, cm per module."""
  keys = [f'shift_fwd_m{m}' for m in range(4)]
  out = {}
  for r in runs.values():
    if r['learn'] != 2:
      continue
    rows = read_metrics(r['dir'], keys)
    if not rows:
      continue
    a = np.asarray(rows, float)
    last = a[a[:, 0] >= a[-1, 0] - 25 * M]
    out[r['name']] = dict(code=r['code'], env=r['env'], seed=r['seed'], frames=float(a[-1, 0]),
                          end_cm=[float(x) for x in last[:, 1:].mean(0)],
                          traj=[[float(x) for x in row] for row in a[::max(1, len(a) // 150)]])
  groups = {}
  for v in out.values():
    groups.setdefault(f'{v["code"]}_{v["env"]}', []).append(v['end_cm'])
  summ = {}
  for k, v in groups.items():
    v = np.asarray(v)
    summ[k] = dict(n=len(v), mean_cm=[float(x) for x in v.mean(0)],
                   sem_cm=[float(x) for x in (v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.full(4, np.nan))],
                   frac=[float(x) for x in v.mean(0) / np.asarray(SPACINGS_CM)],
                   per_seed=[[float(x) for x in row] for row in v])
  return dict(runs=out, groups=summ)


def old_reference(mark):
  """Training score of the earlier gc100 / u20 code-only runs (same seeds, same
  code path) at a mark: the check that a rerun reproduces a run."""
  out = {}
  for code, old in (('gc100', 'gc100'), ('u20', 'pgc33')):
    for seed in range(1, 9):
      for base in ('runs_ext', 'runs'):
        d = os.path.join(ROOT, base, f'{old}_ffb_s{seed}')
        if os.path.exists(os.path.join(d, 'episodes.csv')):
          fr, ret = read_episodes(d)
          if len(fr) and fr.max() >= mark * M - 1.5 * M:
            s, n = window(fr, ret, (mark - 25) * M, mark * M)
            if n:
              out.setdefault(code, {})[seed] = s
              break
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--blocks', default=','.join(BLOCKS))
  ap.add_argument('--marks', default='300,600', help='frame marks (M) for the code-only agents')
  ap.add_argument('--lstm_marks', default='250,500')
  ap.add_argument('--out', default=OUT)
  ap.add_argument('--base', default=BASE)
  args = ap.parse_args()
  marks = [int(x) for x in args.marks.split(',')]
  lstm_marks = [int(x) for x in args.lstm_marks.split(',')]
  runs = load(args.blocks.split(','), args.base)
  os.makedirs(os.path.join(args.out, 'data'), exist_ok=True)
  S = dict(marks=marks, lstm_marks=lstm_marks, labels=LABEL, n_runs=len(runs),
           progress={t: dict(n=sum(r['tag'] == t for r in runs.values()),
                             frames_min=min([r['frames_done'] for r in runs.values() if r['tag'] == t] or [0]) / M,
                             frames_max=max([r['frames_done'] for r in runs.values() if r['tag'] == t] or [0]) / M,
                             evaluated=sum(r['tag'] == t and 'intact' in r['evals'] for r in runs.values()))
                     for t in args.blocks.split(',')},
           open={}, door={}, lstm={})

  top = max(marks)
  for mark in marks:
    # ----- open room
    sc = scores_at(runs, 'open', 'ffb', mark, 25)
    if sc:
      if mark == top:
        named = [('prop10', 'u20'), ('mec', 'u20'), ('gard', 'u20'), ('side', 'u20')]
        desc = [('prop27', 'u20'), ('prop27', 'rand'), ('rand', 'u20'), ('mec', 'prop27'), ('gard', 'prop10'),
                ('u20', 'gc100')]
      else:
        named = [('gcvar', 'gc100'), ('gcshear', 'gc100'), ('gcjit', 'gc100'), ('mecreal', 'mec')]
        desc = [('prop10', 'u20'), ('mec', 'u20'), ('gard', 'u20'), ('side', 'u20'), ('prop27', 'u20'),
                ('rand', 'u20'), ('two', 'u20'), ('lr30', 'u20'), ('p100', 'u20'), ('gcvar3', 'gc100'),
                ('m1var', 'm1'), ('m1', 'gc100'), ('u20', 'gc100'), ('two', 'rand'), ('prop27', 'rand')]
      blk = contrast_block(sc, named, desc)
      blk['many_vs_one'] = many_vs_one(sc)
      old = old_reference(mark)
      blk['rerun_check'] = {c: paired(sc[c], old[c]) for c in old if c in sc}
      S['open'][str(mark)] = blk
    # ----- door room
    sc = scores_at(runs, 'door', 'ffb', mark, 25)
    if sc:
      named = [('u20', 'gc100'), ('prop10', 'u20'), ('mec', 'u20')]
      desc = [('rand', 'u20'), ('gard', 'u20'), ('gc100', 'none'), ('u20', 'none')]
      blk = contrast_block(sc, named if mark == top else [], desc if mark == top else named + desc)
      blk['many_vs_one'] = None
      so = scores_at(runs, 'open', 'ffb', mark, 25)
      if all(c in so for c in ('prop10', 'u20')) and all(c in sc for c in ('prop10', 'u20')):
        seeds = sorted(set(sc['prop10']) & set(sc['u20']) & set(so['prop10']) & set(so['u20']))
        blk['room_interaction'] = dict(one_sample(
            [(sc['prop10'][s] - sc['u20'][s]) - (so['prop10'][s] - so['u20'][s]) for s in seeds]),
            name='(prop10 - u20) door - open', seed_ids=seeds)
      S['door'][str(mark)] = blk
  # the door room's go / no-go mark
  sc = scores_at(runs, 'door', 'ffb', 100, 25)
  if sc:
    S['door']['100'] = contrast_block(sc, [], [('gc100', 'none'), ('u20', 'none'), ('u20', 'gc100')])

  # ----- final 200-episode tests and lesions
  for env, key in (('open', 'open'), ('door', 'door')):
    for budget in sorted({r['budget'] // M for r in runs.values() if r['env'] == env and r['arm'] == 'ffb'}):
      ft = final_tests(runs, env, 'ffb', budget)
      if ft:
        intact = {c: {s: ev['intact'] for s, ev in v.items()} for c, v in ft.items()}
        S[key].setdefault('final_test', {})[str(budget)] = dict(
            contrast_block(intact, [], [(c, 'u20') for c in intact if c != 'u20']), lesions=lesions(ft))

  # ----- LSTM agents on the room mixture
  for mark in lstm_marks:
    sc = scores_at(runs, 'mix', 'lstm', mark, 50)
    if sc:
      S['lstm'][str(mark)] = contrast_block(sc, [], [('u20', 'gc100'), ('prop10', 'u20'), ('mec', 'u20'),
                                                      ('prop10', 'gc100'), ('mec', 'gc100')])
  ft = final_tests(runs, 'mix', 'lstm', max(lstm_marks))
  if ft:
    intact = {c: {s: ev['intact'] for s, ev in v.items()} for c, v in ft.items()}
    S['lstm']['final_test'] = dict(contrast_block(intact, [], [('u20', 'gc100'), ('prop10', 'u20'), ('mec', 'u20')]),
                                   lesions=lesions(ft))

  S['learned'] = learned(runs)

  # ----- learned-look-ahead agents: score against the fixed codes
  for env in ('open', 'door'):
    for mark in marks:
      sc = {}
      for r in runs.values():
        if r['learn'] == 2 and r['env'] == env and r['frames_done'] >= mark * M - 1.5 * M:
          s, n = window(r['fr'], r['ret'], (mark - 25) * M, mark * M)
          if n:
            sc.setdefault(r['code'], {})[r['seed']] = s
      fixed = scores_at(runs, env, 'ffb', mark, 25)
      if sc:
        S['learned'].setdefault('scores', {}).setdefault(env, {})[str(mark)] = dict(
            codes={c: group_mean(v) for c, v in sc.items()},
            vs={ref: {c: paired(sc[c], fixed[ref]) for c in sc} for ref in ('gc100', 'u20') if ref in fixed})

  json.dump(S, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)

  # ----- data files
  with open(os.path.join(args.out, 'data', 'scores.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'block', 'code', 'room', 'agent', 'seed', 'frames_planned_M', 'frames_done_M', 'mark_M',
                'window_M', 'training_score', 'episodes_in_window'])
    for r in sorted(runs.values(), key=lambda r: (r['tag'], r['name'])):
      for mark in ([100] + marks if r['arm'] == 'ffb' else lstm_marks):
        win = 25 if r['arm'] == 'ffb' else 50
        if r['frames_done'] >= mark * M - 1.5 * M:
          s, n = window(r['fr'], r['ret'], (mark - win) * M, mark * M)
          w.writerow([r['name'], r['tag'], r['code'], r['env'], r['arm'], r['seed'], r['budget'] // M,
                      round(r['frames_done'] / M, 1), mark, win, round(s, 3), n])
  with open(os.path.join(args.out, 'data', 'curves.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'frames_lo_M', 'frames_hi_M', 'episodes', 'training_score'])
    for r in sorted(runs.values(), key=lambda r: (r['tag'], r['name'])):
      if not len(r['fr']):
        continue
      edges = np.arange(0, r['fr'].max() + 10 * M, 10 * M)
      idx = np.digitize(r['fr'], edges, right=True)
      for k in range(1, len(edges)):
        m = idx == k
        if m.sum() >= 20:
          w.writerow([r['name'], int(edges[k - 1] // M), int(edges[k] // M), int(m.sum()), round(float(r['ret'][m].mean()), 3)])
  with open(os.path.join(args.out, 'data', 'final_tests.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'code', 'room', 'agent', 'seed', 'probe', 'score', 'path_efficiency', 'first_goal_step',
                'returns_per_1000_steps'])
    for r in sorted(runs.values(), key=lambda r: (r['tag'], r['name'])):
      for probe, v in sorted(r['evals'].items()):
        w.writerow([r['name'], r['code'], r['env'], r['arm'], r['seed'], probe, round(v['score'], 3),
                    None if v['path_eff'] is None else round(v['path_eff'], 4),
                    None if v['first_goal'] is None else round(v['first_goal'], 1),
                    None if v['post_rate'] is None else round(v['post_rate'], 4)])
  with open(os.path.join(args.out, 'data', 'learned_lookaheads.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run', 'start', 'room', 'seed', 'frames_M', 'module1_cm', 'module2_cm', 'module3_cm', 'module4_cm'])
    for name, v in sorted(S['learned']['runs'].items()):
      for row in v['traj']:
        w.writerow([name, v['code'], v['env'], v['seed'], round(row[0] / M, 1)] + [round(x, 3) for x in row[1:]])

  write_results_md(S, os.path.join(args.out, 'RESULTS.md'))
  print(f'{len(runs)} runs; wrote', os.path.join(args.out, 'summary.json'))
  for t, p in S['progress'].items():
    print(f'  {t}: {p["n"]} runs, frames {p["frames_min"]:.0f}-{p["frames_max"]:.0f} M, {p["evaluated"]} with final test')


# ------------------------------------------------------------------ tables
def f1(x, nd=1):
  return '–' if x is None or x != x else f'{x:.{nd}f}'


def sg(x, nd=1):
  return '–' if x is None or x != x else f'{x:+.{nd}f}'.replace('-', '−')


def fp(p):
  if p is None or p != p:
    return '–'
  return '< 0.001' if p < 0.001 else f'{p:.3f}' if p < 0.1 else f'{p:.2f}'


def stat_row(label, t):
  ci = '–' if t.get('ci95') is None else f'{sg(t["diff"] - t["ci95"])} to {sg(t["diff"] + t["ci95"])}'
  return (f'| {label} | {f1(t.get("a"))} | {f1(t.get("b"))} | {sg(t["diff"])} ± {f1(t.get("sem"))} | {ci} | '
          f'{fp(t.get("p"))} | {fp(t.get("p_holm"))} | {fp(t.get("p_flip"))} | {t["pos"]}/{t["n"]} |')


HEAD = ('| comparison | first | second | difference ± SEM | 95 % CI | p (paired t) | p (Holm) | p (sign flip) | '
        'seeds higher |\n|---|---|---|---|---|---|---|---|---|')


def lab(code):
  return f'{LABEL.get(code, code)} (`{code}`)'


def code_table(blk, order):
  rows = ['| code | agents | training score ± SEM | vs 20 cm ahead | vs grid only |', '|---|---|---|---|---|']
  for c in order:
    if c not in blk['codes']:
      continue
    g = blk['codes'][c]
    cells = []
    for ref in ('u20', 'gc100'):
      t = blk['vs'].get(ref, {}).get(c)
      cells.append('–' if t is None or t.get('diff') is None else
                   f'{sg(t["diff"])} ± {f1(t.get("sem"))} (p {fp(t.get("p"))}, {t["pos"]}/{t["n"]})')
    rows.append(f'| {lab(c)} | {g["n"]} | {f1(g["mean"])} ± {f1(g["sem"])} | {cells[0]} | {cells[1]} |')
  return '\n'.join(rows)


def write_results_md(S, path):
  L = ['# Code variations: all tables', '',
       'Generated by `python -m scripts.codes.analyse`. Scores are training scores (mean reward per episode over the '
       'last 25 M frames before the mark, 10 points per goal) unless a table says "final test". Differences are '
       'first minus second, paired within seed. Named comparisons are the ones fixed in [PLAN.md](PLAN.md).', '']
  order = list(OPEN_CORE) + list(OPEN_SCREEN) + ['none']
  for env, title in (('open', 'Open room'), ('door', 'Door room')):
    for mark in sorted([k for k in S[env] if k.isdigit()], key=int):
      blk = S[env][mark]
      L += [f'## {title}, {mark} M frames', '', code_table(blk, order), '']
      if blk.get('named'):
        L += ['Named comparisons:', '', HEAD] + [stat_row(t['name'], t) for t in blk['named']] + ['']
      if blk.get('descriptive'):
        L += ['Descriptive:', '', HEAD] + [stat_row(t['name'], t) for t in blk['descriptive']] + ['']
      for k in ('many_vs_one', 'room_interaction'):
        t = blk.get(k)
        if t:
          L += [f'{t["name"]}: {sg(t["diff"])} ± {f1(t.get("sem"))} (95 % CI {sg(t["diff"] - t["ci95"])} to '
                f'{sg(t["diff"] + t["ci95"])}, p {fp(t.get("p"))}, {t["pos"]}/{t["n"]} seeds).', '']
      if blk.get('rerun_check'):
        L += ['Rerun check (these runs minus the runs of October with the same seeds):', '', HEAD]
        L += [stat_row(f'{c}: new - old', t) for c, t in blk['rerun_check'].items()] + ['']
    for budget, ft in sorted(S[env].get('final_test', {}).items(), key=lambda kv: int(kv[0])):
      L += [f'## {title}, final test of the runs trained for {budget} M frames', '', code_table(ft, order), '',
            'Cost of switching inputs off at test time (points lost, mean ± SEM over agents). "predictive cells, '
            'beyond random" = score with 169 random grid cells off minus score with the predictive cells off.', '',
            '| code | predictive cells, beyond random | look-aheads set to zero | whole code off | goal code off |',
            '|---|---|---|---|---|']
      for c in order:
        le = ft['lesions'].get(c)
        if le:
          cell = lambda k: '–' if k not in le or le[k].get('diff') is None else f'{sg(le[k]["diff"])} ± {f1(le[k].get("sem"))}'
          L.append(f'| {lab(c)} | {cell("pgc_specific")} | {cell("pgc_as_gc")} | {cell("no_code")} | {cell("no_goal")} |')
      L.append('')
  if S['lstm']:
    for mark in sorted([k for k in S['lstm'] if k.isdigit()], key=int):
      blk = S['lstm'][mark]
      L += [f'## LSTM agents on the room mixture, {mark} M frames (training score over the last 50 M frames)', '',
            code_table(blk, LSTM_CODES), '', HEAD] + [stat_row(t['name'], t) for t in blk['descriptive']] + ['']
    if 'final_test' in S['lstm']:
      L += ['## LSTM agents, final test on the mixture', '', code_table(S['lstm']['final_test'], LSTM_CODES), '']
  if S['learned']['groups']:
    L += ['## Trainable look-aheads (mean over the last 25 M frames)', '',
          '| start, room | agents | module 1 (40 cm) | module 2 (57 cm) | module 3 (81 cm) | module 4 (115 cm) | as fraction of spacing |',
          '|---|---|---|---|---|---|---|']
    for k, g in sorted(S['learned']['groups'].items()):
      L.append(f'| {k} | {g["n"]} | ' + ' | '.join(f'{f1(m)} ± {f1(s)} cm' for m, s in zip(g['mean_cm'], g['sem_cm']))
               + ' | ' + ', '.join(f1(x, 2) for x in g['frac']) + ' |')
    L.append('')
    for env, d in S['learned'].get('scores', {}).items():
      for mark, v in d.items():
        L += [f'Scores of the agents with trainable look-aheads, {env} room, {mark} M frames: '
              + '; '.join(f'{c} {f1(g["mean"])} ± {f1(g["sem"])} (n {g["n"]})' for c, g in v['codes'].items()) + '.', '']
  open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
  main()
