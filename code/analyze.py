"""Aggregate the run matrix: learning curves, final scores, probes, reliance
over training, and paired statistics.

Reads runs/*/config.json, episodes.csv and eval/*.json (canonical finals and
checkpoint probes); writes results/learning_curves.png, final_scores.png,
probes.png, reliance_over_training.png, summary.json and RESULTS.md.

Run names are <cond><variant>_<arm>_s<seed> (see scripts/launch_matrix.py);
groups are (arm, cond+variant).

Endpoints (fixed before the run matrix was launched):
  primary    within-seed paired difference pgc33 - gc100 in the final 200-episode
             evaluation score, in arms that pass the reliance guard
  learning   frames to 50 % / 80 % of the arm's pooled gc100 final score; early
             AUC (first 100M frames)
  homing     intact - no_goal (goal-memory use), post-discovery goal rate,
             path efficiency (reported with the number of returns)
  reliance   intact - no_code, intact - no_current, learned gain intact - untrained
  PGC-specific lesion effect: sham (count-matched, mean of 5 draws) - no_pgc

  python -m code.analyze --runs_dir runs --out results
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAME_RE = re.compile(r'^(gc100|pgc33|pgc100|none|band|pband|pgc)([a-z0-9]*)_(ffb|ff|lstm)_s(\d+)$')
ARMS = ['ffb', 'ff', 'lstm']
ARM_LABEL = {'ffb': 'FF policy, heading-only vision (code-only)',
             'ff': 'FF policy, raycast vision (vision + code)',
             'lstm': 'LSTM policy, raycast vision (paper)'}
COND_ORDER = ['gc100', 'pgc33', 'pgc33goalzero', 'pgc100', 'none', 'gc100vel',
              'gc100lookahead', 'pgc33shift10', 'pgc33shift40']
class _Default(dict):
  def __missing__(self, k):
    return k if self is COND_LABEL else 'tab:brown'


COND_LABEL = _Default({'gc100': '100% grid cells', 'pgc33': '67% grid + 33% PGC',
              'pgc33goalzero': '33% PGC, position-only goal code', 'pgc100': '100% PGC',
              'none': 'no spatial code', 'gc100vel': 'grid cells + velocity input',
              'gc100lookahead': 'grid cells, look-ahead goal code',
              'pgc33shift10': '33% PGC, shift 10 cm', 'pgc33shift40': '33% PGC, shift 40 cm',
              'pgc33fine': '33% PGC in finest modules', 'pgc33coarse': '33% PGC in coarsest modules',
              'pgc33middle': '33% PGC in middle modules', 'pgc33phase': '33% PGC, shift = half spacing',
              'pgc33jitter': '33% PGC, heterogeneous shifts', 'pgc10': '10% PGC', 'pgc25': '25% PGC',
              'pgc33phase27': '33% PGC, shift = 0.27 spacing', 'pgc33bio': '33% PGC, 0.27 spacing + heterogeneous',
              'pgc33mec': '33% PGC, MEC-weighted modules (1:2:2:1)', 'pgc33retro': '33% retrospective (20 cm behind)',
              'pgc30': '30% PGC', 'pgc35': '35% PGC', 'pgc40': '40% PGC',
              'band': '100% band cells', 'pband33': 'band + 33% predictive bands',
              'pband100': '100% predictive band cells'})
COND_COLOR = _Default({'gc100': 'tab:blue', 'pgc33': 'tab:red', 'pgc33goalzero': 'tab:pink',
                       'pgc100': 'tab:orange', 'none': 'tab:gray', 'gc100vel': 'tab:cyan',
                       'gc100lookahead': 'tab:purple', 'pgc33shift10': 'tab:brown',
                       'pgc33shift40': 'tab:olive'})
PROBES = ['intact', 'no_code', 'no_current', 'no_goal', 'no_pgc', 'no_gc', 'sham',
          'pgc_as_gc', 'fake_goal', 'shuffled_current', 'untrained']
COMPARISONS = [('pgc33', 'gc100'), ('pgc33goalzero', 'gc100'), ('pgc33', 'pgc33goalzero'),
               ('gc100', 'none'), ('pgc33', 'none'), ('pgc100', 'gc100'),
               ('gc100lookahead', 'gc100'), ('gc100vel', 'gc100'),
               ('pgc33shift10', 'gc100'), ('pgc33shift40', 'gc100'),
               ('pgc33fine', 'gc100'), ('pgc33coarse', 'gc100'), ('pgc33middle', 'gc100'),
               ('pgc33phase', 'gc100'), ('pgc33jitter', 'gc100'),
               ('pgc33fine', 'pgc33'), ('pgc33coarse', 'pgc33'), ('pgc33middle', 'pgc33'),
               ('pgc33phase', 'pgc33'), ('pgc33jitter', 'pgc33'),
               ('pgc33phase27', 'gc100'), ('pgc33bio', 'gc100'), ('pgc33mec', 'gc100'), ('pgc33retro', 'gc100'),
               ('pgc33phase27', 'pgc33'), ('pgc33bio', 'pgc33'), ('pgc33mec', 'pgc33'), ('pgc33retro', 'pgc33'),
               ('pgc10', 'gc100'), ('pgc25', 'gc100'), ('pgc30', 'gc100'), ('pgc35', 'gc100'),
               ('pgc40', 'gc100'), ('band', 'gc100'), ('pband33', 'band'), ('pband100', 'band'),
               ('pband33', 'pgc33')]


# ----------------------------------------------------------------- loading
def _load_eval(f):
  try:
    return json.load(open(f))
  except Exception:
    return None


def load_runs(runs_dir, allow_nonfinal=False):
  runs = []
  for d in sorted(x for x in glob.glob(os.path.join(runs_dir, '*')) if os.path.isdir(x)):
    name = os.path.basename(d)
    m = NAME_RE.match(name)
    cfg_p, ep_p = os.path.join(d, 'config.json'), os.path.join(d, 'episodes.csv')
    if not m or not (os.path.exists(cfg_p) and os.path.exists(ep_p)):
      continue
    cfg = json.load(open(cfg_p))
    try:
      ep = np.genfromtxt(ep_p, delimiter=',', names=True)
    except Exception:
      continue
    if ep.size == 0:
      continue
    ep = np.atleast_1d(ep)
    if 'ret' not in ep.dtype.names:
      ep.dtype.names = tuple('ret' if n in ('return', 'return_') else n for n in ep.dtype.names)
    cond = m.group(1) + m.group(2)
    evals, ckpt_evals = {}, {}
    for f in glob.glob(os.path.join(d, 'eval', '*.json')):
      base = os.path.basename(f)[:-5]
      v = _load_eval(f)
      if v is None:
        continue
      if '@' in base:
        probe, rest = base.split('@', 1)
        fr = int(rest.split('f_')[0])
        ckpt_evals.setdefault(probe, {})[fr] = v['mean_score']
      elif v.get('canonical', False) or allow_nonfinal:
        evals[base] = v
    shams = [evals[k] for k in evals if k.startswith('sham')]
    if shams:
      evals['sham'] = dict(shams[0])
      for k in ('mean_score', 'mean_goals', 'mean_first_goal_step', 'mean_path_eff',
                'frac_found_goal', 'post_discovery_goal_rate', 'frac_two_goals'):
        evals['sham'][k] = float(np.nanmean([s.get(k, np.nan) for s in shams]))
      evals['sham']['sem_score'] = float(np.sqrt(np.mean([s['sem_score'] ** 2 for s in shams]) / len(shams)))
      evals['sham']['n_draws'] = len(shams)
    runs.append(dict(name=name, cond=cond, arm=m.group(3), seed=int(m.group(4)),
                     budget=cfg['frames'], ep=ep, evals=evals, ckpt_evals=ckpt_evals,
                     episode_len=cfg.get('episode_len', 1350),
                     final=os.path.exists(os.path.join(d, 'ckpt_final.pt')),
                     frames=float(ep['frames'].max())))
  return runs


# ----------------------------------------------------------------- metrics
def binned_curve(ep, key, edges, episode_len=1350):
  f = ep['frames']
  v = ep[key].astype(float)
  if key == 'first_goal_step':
    v = np.where(v < 0, float(episode_len), v)
  if key == 'path_eff':
    ok = ep['path_eff_n'] > 0
    f, v = f[ok], v[ok]
  idx = np.digitize(f, edges) - 1
  out = np.full(len(edges) - 1, np.nan)
  for b in range(len(edges) - 1):
    m = idx == b
    if m.sum() >= 5:
      out[b] = v[m].mean()
  return out


def sem(x):
  x = np.asarray(x, float)
  x = x[np.isfinite(x)]
  return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')


def frames_to_threshold(curve, centers, thr):
  if not np.isfinite(thr):
    return float('nan')
  for c, v in zip(centers, curve):
    if np.isfinite(v) and v >= thr:
      return float(c)
  return float('nan')


def paired_stats(a, b, eq_bound=None):
  """a, b: dict seed -> value. Paired (a - b) stats; eq_bound: TOST bound."""
  seeds = sorted(set(a) & set(b))
  res = dict(n_pairs=len(seeds), n_a=len(a), n_b=len(b),
             mean_a=float(np.mean(list(a.values()))), mean_b=float(np.mean(list(b.values()))))
  try:
    from scipy import stats
  except Exception:
    stats = None
  if stats is not None and len(a) > 1 and len(b) > 1:
    t2, p2 = stats.ttest_ind(list(a.values()), list(b.values()), equal_var=False)
    res['welch_t'], res['welch_p'] = float(t2), float(p2)
  if len(seeds) < 2:
    return res
  d = np.array([a[s] - b[s] for s in seeds], float)
  res.update(seeds=seeds, diffs=[float(x) for x in d], mean_diff=float(d.mean()),
             sem_diff=sem(d))
  rng = np.random.default_rng(0)
  boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
  res['boot_ci95'] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
  if stats is not None:
    t, p = stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
    res['paired_t'], res['paired_p'] = float(t), float(p)
    if len(seeds) >= 6 and np.any(d != 0):
      res['wilcoxon_p'] = float(stats.wilcoxon(d).pvalue)
    if eq_bound is not None and len(seeds) >= 3 and d.std(ddof=1) > 0:
      se = d.std(ddof=1) / np.sqrt(len(d))
      p_low = 1 - stats.t.cdf((d.mean() + eq_bound) / se, len(d) - 1)
      p_high = stats.t.cdf((d.mean() - eq_bound) / se, len(d) - 1)
      res['tost_bound'] = float(eq_bound)
      res['tost_p'] = float(max(p_low, p_high))
  return res


# -------------------------------------------------------------------- main
def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default='results')
  ap.add_argument('--bin_frames', type=float, default=5e6)
  ap.add_argument('--final_frac', type=float, default=0.1)
  ap.add_argument('--early_frames', type=float, default=100e6)
  ap.add_argument('--eq_frac', type=float, default=0.10,
                  help='TOST equivalence bound as a fraction of the reference mean')
  ap.add_argument('--allow_nonfinal', action='store_true')
  args = ap.parse_args(argv)
  os.makedirs(args.out, exist_ok=True)
  runs = load_runs(args.runs_dir, allow_nonfinal=args.allow_nonfinal)
  if not runs:
    print('no runs found')
    return
  budget = max(r['budget'] for r in runs)
  edges = np.arange(0, budget + args.bin_frames, args.bin_frames)
  centers = (edges[:-1] + edges[1:]) / 2
  groups = {}
  for r in runs:
    groups.setdefault((r['arm'], r['cond']), []).append(r)

  ref = {}
  for arm in ARMS:
    rs = groups.get((arm, 'gc100'), [])
    vals = [r['evals']['intact']['mean_score'] for r in rs if 'intact' in r['evals']]
    if not vals:
      for r in rs:
        ret = r['ep']['ret'].astype(float)
        late = r['ep']['frames'] >= (1 - args.final_frac) * r['budget']
        if late.any():
          vals.append(float(ret[late].mean()))
    ref[arm] = float(np.mean(vals)) if vals else float('nan')

  per_run = {}
  for r in runs:
    ep = r['ep']
    f = ep['frames']
    late = f >= (1 - args.final_frac) * r['budget']
    ret = ep['ret'].astype(float)
    curve = binned_curve(ep, 'ret', edges, r['episode_len'])
    early = centers <= args.early_frames
    e = r['evals']
    ev = {p: dict(score=v['mean_score'], sem=v['sem_score'], goals=v['mean_goals'],
                  first_goal=v['mean_first_goal_step'], found=v['frac_found_goal'],
                  path_eff=v['mean_path_eff'], n_path_eff=v.get('n_path_eff'),
                  post_rate=v.get('post_discovery_goal_rate'), two_goals=v.get('frac_two_goals'))
          for p, v in e.items()}
    g = lambda p: ev[p]['score'] if p in ev else float('nan')
    per_run[r['name']] = dict(
        cond=r['cond'], arm=r['arm'], seed=r['seed'], final=r['final'], frames=r['frames'],
        episodes=int(len(ep)),
        train_final_return=float(ret[late].mean()) if late.any() else float('nan'),
        train_early_auc=float(np.nanmean(curve[early])) if np.isfinite(curve[early]).any() else float('nan'),
        frames_to_50=frames_to_threshold(curve, centers, 0.5 * ref[r['arm']]),
        frames_to_80=frames_to_threshold(curve, centers, 0.8 * ref[r['arm']]),
        eval=ev,
        learned_gain=g('intact') - g('untrained'),
        reliance_code=g('intact') - g('no_code'),
        reliance_current=g('intact') - g('no_current'),
        homing=g('intact') - g('no_goal'),
        pgc_specific=g('sham') - g('no_pgc'),
        ckpt_evals=r['ckpt_evals'])

  summary = dict(reference_gc100_final=ref, groups={}, comparisons={}, per_run=per_run)
  scalar_keys = ['train_final_return', 'train_early_auc', 'frames_to_50', 'frames_to_80',
                 'learned_gain', 'reliance_code', 'reliance_current', 'homing', 'pgc_specific']
  for (arm, cond), rs in sorted(groups.items()):
    gsum = dict(n=len(rs), seeds=[r['seed'] for r in rs], finished=int(sum(r['final'] for r in rs)))
    for key in scalar_keys:
      vals = np.array([per_run[r['name']][key] for r in rs], float)
      gsum[key] = dict(mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan'),
                       sem=sem(vals), values=[float(v) for v in vals])
    for p in PROBES:
      vals = [r['evals'][p]['mean_score'] for r in rs if p in r['evals']]
      if vals:
        gsum[f'eval_{p}'] = dict(mean=float(np.mean(vals)), sem=sem(vals), values=vals)
    for key in ('goals', 'first_goal', 'found', 'path_eff', 'post_rate', 'two_goals'):
      vals = [per_run[r['name']]['eval']['intact'][key] for r in rs if 'intact' in r['evals']]
      vals = [v for v in vals if v is not None]
      if vals:
        gsum[f'eval_intact_{key}'] = dict(mean=float(np.nanmean(vals)), sem=sem(vals), values=vals)
    summary['groups'][f'{arm}/{cond}'] = gsum
  metric_getters = {
      'eval_intact_score': lambda r: r['evals']['intact']['mean_score'] if 'intact' in r['evals'] else None,
      'train_final_return': lambda r: per_run[r['name']]['train_final_return'],
      'train_early_auc': lambda r: per_run[r['name']]['train_early_auc'],
      'frames_to_80': lambda r: per_run[r['name']]['frames_to_80'],
      'homing': lambda r: per_run[r['name']]['homing'],
      'eval_intact_path_eff': lambda r: r['evals']['intact']['mean_path_eff'] if 'intact' in r['evals'] else None,
      'eval_intact_post_rate': lambda r: r['evals']['intact'].get('post_discovery_goal_rate') if 'intact' in r['evals'] else None,
      'eval_intact_first_goal': lambda r: r['evals']['intact']['mean_first_goal_step'] if 'intact' in r['evals'] else None,
  }
  for arm in ARMS:
    for ca, cb in COMPARISONS:
      ra, rb = groups.get((arm, ca), []), groups.get((arm, cb), [])
      if not ra or not rb:
        continue
      comp = {}
      for key, getter in metric_getters.items():
        a = {r['seed']: getter(r) for r in ra if getter(r) is not None and np.isfinite(getter(r))}
        b = {r['seed']: getter(r) for r in rb if getter(r) is not None and np.isfinite(getter(r))}
        if a and b:
          bound = args.eq_frac * abs(np.mean(list(b.values()))) if key == 'eval_intact_score' else None
          comp[key] = paired_stats(a, b, eq_bound=bound)
      summary['comparisons'][f'{arm}: {ca} - {cb}'] = comp
  with open(os.path.join(args.out, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)

  # ---------------------------------------------------------------- figures
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  arms_present = [a for a in ARMS if any(k[0] == a for k in groups)]
  conds_present = [c for c in COND_ORDER if any(k[1] == c for k in groups)] + \
      sorted({k[1] for k in groups} - set(COND_ORDER))
  na = max(len(arms_present), 1)
  metrics = [('ret', 'episode return (10 / goal)'), ('n_goals', 'goals / episode'),
             ('first_goal_step', 'step of first goal (censored)'), ('path_eff', 'path efficiency')]
  fig, axes = plt.subplots(len(metrics), na, figsize=(6 * na, 3.2 * len(metrics)),
                           sharex=True, squeeze=False)
  for j, arm in enumerate(arms_present):
    for i, (key, label) in enumerate(metrics):
      ax = axes[i, j]
      for cond in conds_present:
        rs = groups.get((arm, cond), [])
        if not rs:
          continue
        cs = np.array([binned_curve(r['ep'], key, edges, r['episode_len']) for r in rs])
        mu = np.nanmean(cs, 0)
        se = np.nanstd(cs, 0) / np.sqrt(np.maximum((~np.isnan(cs)).sum(0), 1))
        x = centers / 1e6
        ax.plot(x, mu, color=COND_COLOR[cond], label=f'{COND_LABEL[cond]} (n={len(rs)})')
        ax.fill_between(x, mu - se, mu + se, color=COND_COLOR[cond], alpha=0.2)
      ax.set_ylabel(label)
      if i == 0:
        ax.set_title(ARM_LABEL[arm], fontsize=10)
        ax.legend(fontsize=7)
      if i == len(metrics) - 1:
        ax.set_xlabel('environment frames (millions)')
  fig.suptitle('Learning curves (mean +/- SEM over seeds; episodes binned by frames)')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'learning_curves.png'), dpi=110)
  plt.close(fig)

  fig, axes = plt.subplots(1, na, figsize=(6 * na, 4.2), squeeze=False)
  for j, arm in enumerate(arms_present):
    ax = axes[0, j]
    ticks = []
    for k, cond in enumerate(conds_present):
      rs = groups.get((arm, cond), [])
      vals = [r['evals']['intact']['mean_score'] for r in rs if 'intact' in r['evals']]
      if not vals:
        continue
      ax.bar(k, np.mean(vals), yerr=sem(vals) if len(vals) > 1 else 0, color=COND_COLOR[cond],
             alpha=0.7, capsize=4)
      ax.scatter([k] * len(vals), vals, color='k', zorder=3, s=18)
      ticks.append((k, COND_LABEL[cond]))
      ch = [r['evals']['untrained']['mean_score'] for r in rs if 'untrained' in r['evals']]
      if ch:
        ax.hlines(np.mean(ch), k - 0.4, k + 0.4, colors='k', linestyles=':', lw=1)
    ax.set_xticks([x for x, _ in ticks])
    ax.set_xticklabels([l for _, l in ticks], rotation=25, fontsize=7, ha='right')
    ax.set_ylabel('eval score (200 paired episodes)')
    ax.set_title(ARM_LABEL[arm] + '\n(dotted = untrained chance line)', fontsize=9)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'final_scores.png'), dpi=110)
  plt.close(fig)

  fig, axes = plt.subplots(1, na, figsize=(7 * na, 4.2), squeeze=False)
  for j, arm in enumerate(arms_present):
    ax = axes[0, j]
    cs = [c for c in conds_present if (arm, c) in groups]
    w = 0.8 / max(len(cs), 1)
    for k, cond in enumerate(cs):
      rs = groups[(arm, cond)]
      means, sems = [], []
      for p in PROBES:
        vals = [r['evals'][p]['mean_score'] for r in rs if p in r['evals']]
        means.append(np.mean(vals) if vals else np.nan)
        sems.append(sem(vals) if len(vals) > 1 else 0)
      x = np.arange(len(PROBES)) + k * w
      ax.bar(x, means, w, yerr=sems, color=COND_COLOR[cond], alpha=0.75, capsize=2,
             label=COND_LABEL[cond])
    ax.set_xticks(np.arange(len(PROBES)) + 0.4 - w / 2)
    ax.set_xticklabels(PROBES, rotation=30, fontsize=7, ha='right')
    ax.set_ylabel('eval score')
    ax.set_title('test-time probes: ' + ARM_LABEL[arm], fontsize=9)
    ax.legend(fontsize=6)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'probes.png'), dpi=110)
  plt.close(fig)

  nc = max(len(conds_present), 1)
  fig, axes = plt.subplots(nc, na, figsize=(6 * na, 3.0 * nc), sharex=True, squeeze=False)
  pstyle = {'intact': ('k', '-'), 'no_code': ('tab:red', '--'), 'no_current': ('tab:orange', '--'),
            'no_goal': ('tab:green', ':'), 'untrained': ('tab:gray', ':')}
  for j, arm in enumerate(arms_present):
    for i, cond in enumerate(conds_present):
      ax = axes[i, j]
      rs = groups.get((arm, cond), [])
      for p, (c, ls) in pstyle.items():
        series = {}
        for r in rs:
          for fr, sc in r['ckpt_evals'].get(p, {}).items():
            series.setdefault(fr, []).append(sc)
          if p in r['evals']:
            series.setdefault(int(r['frames']), []).append(r['evals'][p]['mean_score'])
        if series:
          xs = sorted(series)
          ax.plot(np.array(xs) / 1e6, [np.mean(series[x]) for x in xs], color=c, ls=ls,
                  marker='o', ms=3, label=p)
      ax.set_title(f'{ARM_LABEL[arm]}\n{COND_LABEL[cond]}', fontsize=8)
      ax.set_ylabel('score (64-ep probes)')
      if i == 0 and j == 0:
        ax.legend(fontsize=7)
      if i == nc - 1:
        ax.set_xlabel('frames (millions)')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'reliance_over_training.png'), dpi=110)
  plt.close(fig)

  # ----------------------------------------------------------------- report
  def f1(d, p=2):
    if not d or not np.isfinite(d.get('mean', np.nan)):
      return '-'
    if np.isfinite(d.get('sem', np.nan)):
      return f'{d["mean"]:.{p}f} +/- {d["sem"]:.{p}f}'
    return f'{d["mean"]:.{p}f}'
  L = ['# Results: idealised grid vs predictive-grid population codes', '',
       f'Runs found: {len(runs)} ({sum(r["final"] for r in runs)} finished). '
       f'Reference gc100 final score per arm: ' +
       ', '.join(f'{a}: {v:.1f}' for a, v in ref.items() if np.isfinite(v)), '',
       '## Final evaluation (200 paired episodes, stochastic policy; mean +/- SEM over seeds)', '',
       '| arm | condition | n (done) | intact | untrained | no_code | no_current | no_goal | sham | no_pgc | pgc_as_gc | fake_goal | goals/ep | found | first goal | post-disc. rate | path eff |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for arm in arms_present:
    for cond in conds_present:
      g = summary['groups'].get(f'{arm}/{cond}')
      if not g:
        continue
      L.append('| ' + ' | '.join([
          arm, COND_LABEL[cond], f'{g["n"]} ({g["finished"]})',
          f1(g.get('eval_intact')), f1(g.get('eval_untrained')), f1(g.get('eval_no_code')),
          f1(g.get('eval_no_current')), f1(g.get('eval_no_goal')), f1(g.get('eval_sham')),
          f1(g.get('eval_no_pgc')), f1(g.get('eval_pgc_as_gc')), f1(g.get('eval_fake_goal')),
          f1(g.get('eval_intact_goals')), f1(g.get('eval_intact_found')),
          f1(g.get('eval_intact_first_goal'), 0), f1(g.get('eval_intact_post_rate')),
          f1(g.get('eval_intact_path_eff'), 3)]) + ' |')
  L += ['', '## Derived per-group quantities (mean +/- SEM over seeds)', '',
        '| arm | condition | learned gain | reliance (no_code) | reliance (no_current) | homing (no_goal) | PGC-specific (sham - no_pgc) | train final | early AUC (<=100M) | frames to 50% (M) | frames to 80% (M) |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
  for arm in arms_present:
    for cond in conds_present:
      g = summary['groups'].get(f'{arm}/{cond}')
      if not g:
        continue
      fm = lambda d: ('-' if not np.isfinite(d['mean']) else f'{d["mean"]/1e6:.0f}')
      L.append('| ' + ' | '.join([arm, COND_LABEL[cond], f1(g['learned_gain']), f1(g['reliance_code']),
                                   f1(g['reliance_current']), f1(g['homing']), f1(g['pgc_specific']),
                                   f1(g['train_final_return']), f1(g['train_early_auc']),
                                   fm(g['frames_to_50']), fm(g['frames_to_80'])]) + ' |')
  L += ['', '## Comparisons (paired within seed where both runs exist; a - b)', '',
        '| comparison | metric | pairs | mean a | mean b | mean diff +/- SEM | bootstrap 95% CI | paired t p | Welch p | Wilcoxon p (n>=6) | TOST p (equiv. within 10%) |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
  for name, comp in summary['comparisons'].items():
    for key, st in comp.items():
      fmt = lambda k, p='.3f': (f'{st[k]:{p}}' if k in st and np.isfinite(st[k]) else '-')
      ci = (f'[{st["boot_ci95"][0]:.2f}, {st["boot_ci95"][1]:.2f}]' if 'boot_ci95' in st else '-')
      md = (f'{st["mean_diff"]:.3f} +/- {st["sem_diff"]:.3f}' if 'mean_diff' in st else '-')
      L.append(f'| {name} | {key} | {st["n_pairs"]} | {st["mean_a"]:.2f} | {st["mean_b"]:.2f} | {md} | {ci} | '
               f'{fmt("paired_p")} | {fmt("welch_p")} | {fmt("wilcoxon_p")} | {fmt("tost_p")} |')
  L += ['', '## Per-run', '', '| run | frames (M) | episodes | train final | intact | untrained | no_code | no_current | no_goal |',
        '|---|---|---|---|---|---|---|---|---|']
  for name, pr in sorted(per_run.items()):
    e = pr['eval']
    g = lambda p: f'{e[p]["score"]:.2f}' if p in e else '-'
    L.append(f'| {name} | {pr["frames"]/1e6:.0f} | {pr["episodes"]} | {pr["train_final_return"]:.2f} | '
             f'{g("intact")} | {g("untrained")} | {g("no_code")} | {g("no_current")} | {g("no_goal")} |')
  L += ['', 'Figures: learning_curves.png, final_scores.png, probes.png, reliance_over_training.png']
  with open(os.path.join(args.out, 'RESULTS.md'), 'w') as f:
    f.write('\n'.join(L) + '\n')
  print('\n'.join(L))


if __name__ == '__main__':
  main()
