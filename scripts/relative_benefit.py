"""Relative benefit of predictive grid cells (PGCs) in the final evaluations.

Reads the canonical 200-episode evaluations runs/<run>/eval/<probe>.json and
runs/<run>/config.json (falling back to data/evals/<run>/ when a run directory
lacks them) and writes results/relative_benefit.md, results/relative_benefit.json
and figures/fig10_relative_benefit.png (+ .svg).

Everything is paired within seed: seed s of condition A against seed s of
condition B, which share one population because code_seed = seed.
  1. relative benefit pgc33 vs gc100 per arm (difference, ratio, log-ratio of
     the intact score and the secondary endpoints; bootstrap CIs over seeds,
     paired t and Wilcoxon tests), and whether the log-ratio of the score is
     larger in the code-only arm than in the vision arms (pairing the arms by
     seed index; paired t and sign-flip permutation test)
  2. the same for pgc100 vs gc100 and, in the code-only arm, for every PGC
     arrangement and concentration vs gc100
  3. lesion cost per silenced cell: intact - no_pgc, intact - no_gc and
     intact - sham (mean of the five count-matched draws), per arm
  4. the inputs of figures/fig3_probes.png (make_figures.fig_probes)
  5. secondary endpoints gc100 vs pgc33 per arm
  6. a consistency check of every number that also appears in results/RESULTS.md

  python scripts/relative_benefit.py --runs_dir runs --fallback_dir data/evals \
      --results results --figures figures
"""
import argparse
import glob
import importlib.util
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from code.analyze import COND_LABEL, NAME_RE, paired_stats, sem  # noqa: E402
from code.common import SHAM_FRAC  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ['ffb', 'ff', 'lstm']
ARM_TITLE = {'ffb': 'code only', 'ff': 'code + vision', 'lstm': 'LSTM + vision'}
ARM_TICK = {'ffb': 'code\nonly', 'ff': 'code +\nvision', 'lstm': 'LSTM +\nvision'}
# Validated categorical palette (dataviz reference instance): blue, orange, aqua + gray.
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9891'
INK, INK2 = '#0b0b0b', '#52514e'
DARK_ORANGE, PURPLE, DARK = '#9c3f14', '#8b5cf6', '#5f5e5a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': False,
    'savefig.dpi': 200, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

N_BOOT, BOOT_SEED, N_PERM = 10000, 0, 10000
SHAM_KEYS = ('mean_score', 'mean_goals', 'mean_first_goal_step', 'mean_path_eff',
             'frac_found_goal', 'post_discovery_goal_rate', 'frac_two_goals')

# metric key -> (label, decimals, lower is better)
METRICS = [('score', 'score', 2, False),
           ('path_eff', 'path efficiency', 3, False),
           ('post_rate', 'post-discovery goal rate', 2, False),
           ('first_goal', 'first-goal step', 1, True),
           ('two_goals', 'episodes with >= 2 goals', 3, False),
           ('homing', 'homing (intact - no_goal)', 2, False),
           ('learned_gain', 'learned gain (intact - untrained)', 2, False),
           ('pgc_specific', 'PGC-specific lesion cost (sham - no_pgc)', 2, False)]
METRIC_LABEL = {k: l for k, l, _, _ in METRICS}
METRIC_DEC = {k: d for k, _, d, _ in METRICS}
LOWER_BETTER = {k for k, _, _, lb in METRICS if lb}
SECONDARY = ['path_eff', 'post_rate', 'first_goal', 'two_goals']

# code-only arm: PGC arrangements at 33 % and the concentration series (quarter-spacing shift)
ARRANGEMENTS = [('pgc33', 'uniform, 20 cm'), ('pgc33phase27', 'quarter-spacing shift'),
                ('pgc33mec', 'MEC-weighted modules'), ('pgc33bio', 'quarter-spacing + heterogeneous'),
                ('pgc33jitter', 'heterogeneous 20 cm'), ('pgc33coarse', 'coarse modules only'),
                ('pgc33middle', 'middle modules only'), ('pgc33retro', 'retrospective (20 cm behind)'),
                ('pgc33phase', 'half-spacing shift'), ('pgc33fine', 'fine modules only')]
CONCENTRATION = [('pgc10', '10 % PGC'), ('pgc25', '25 % PGC'), ('pgc30', '30 % PGC'),
                 ('pgc35', '35 % PGC'), ('pgc40', '40 % PGC')]
COND_COLOR = {'pgc33': ORANGE, 'pgc33phase27': DARK_ORANGE, 'pgc33mec': AQUA, 'pgc33bio': PURPLE}
LESIONS = [('no_pgc', 'PGC lesion'), ('no_gc', 'grid-cell lesion'), ('sham', 'sham lesion')]
LESION_COLOR = {'no_pgc': ORANGE, 'no_gc': BLUE, 'sham': GRAY}
FIG3_PROBES = ['intact', 'no_code', 'no_goal', 'sham', 'no_pgc']


# ------------------------------------------------------------------ loading
def load_eval_dir(d):
  """Canonical final probes in one eval/ directory -> {probe: dict}; 'sham' is the
  mean over the sham0..4 draws of every summary key (as in code.analyze)."""
  evals = {}
  for f in sorted(glob.glob(os.path.join(d, '*.json'))):
    base = os.path.basename(f)[:-5]
    if '@' in base:
      continue
    try:
      v = json.load(open(f))
    except Exception:
      continue
    if v.get('canonical', False):
      evals[base] = v
  shams = [evals[k] for k in sorted(evals) if re.fullmatch(r'sham\d+', k)]
  if shams:
    evals['sham'] = {k: float(np.mean([s[k] for s in shams])) for k in SHAM_KEYS
                     if all(s.get(k) is not None for s in shams)}
    evals['sham']['n_draws'] = len(shams)
  return evals


def load_runs(runs_dir, fallback_dir=None):
  """{(cond, arm, seed): run}; the eval JSONs come from runs_dir, or from
  fallback_dir when the run directory has no canonical intact.json."""
  names = set()
  for base in (runs_dir, fallback_dir):
    if base and os.path.isdir(base):
      names |= {os.path.basename(d) for d in glob.glob(os.path.join(base, '*')) if os.path.isdir(d)}
  runs = {}
  for name in sorted(names):
    m = NAME_RE.match(name)
    if not m:
      continue
    cfg, evals, src = None, {}, None
    for base in (runs_dir, fallback_dir):
      if not base:
        continue
      d = os.path.join(base, name)
      p = os.path.join(d, 'config.json')
      if cfg is None and os.path.exists(p):
        cfg = json.load(open(p))
      ev = load_eval_dir(os.path.join(d, 'eval'))
      if 'intact' in ev:
        evals, src = ev, base
        break
    if cfg is None or 'intact' not in evals:
      continue
    runs[(m.group(1) + m.group(2), m.group(3), int(m.group(4)))] = dict(
        name=name, cond=m.group(1) + m.group(2), arm=m.group(3), seed=int(m.group(4)),
        cfg=cfg, evals=evals, source=src)
  return runs


def runs_of(runs, cond, arm):
  return [runs[k] for k in sorted(runs) if k[0] == cond and k[1] == arm]


def probe_value(run, probe, key='mean_score'):
  v = run['evals'].get(probe, {}).get(key)
  return float(v) if v is not None else float('nan')


def metric_value(run, key):
  g = lambda p, k='mean_score': probe_value(run, p, k)
  if key == 'score':
    return g('intact')
  if key == 'path_eff':
    return g('intact', 'mean_path_eff')
  if key == 'post_rate':
    return g('intact', 'post_discovery_goal_rate')
  if key == 'first_goal':
    return g('intact', 'mean_first_goal_step')
  if key == 'two_goals':
    return g('intact', 'frac_two_goals')
  if key == 'homing':
    return g('intact') - g('no_goal')
  if key == 'learned_gain':
    return g('intact') - g('untrained')
  if key == 'pgc_specific':
    return g('sham') - g('no_pgc')
  if key == 'reliance_code':
    return g('intact') - g('no_code')
  if key == 'reliance_current':
    return g('intact') - g('no_current')
  raise KeyError(key)


def seed_values(runs, cond, arm, key):
  out = {}
  for r in runs_of(runs, cond, arm):
    v = metric_value(r, key)
    if np.isfinite(v):
      out[r['seed']] = v
  return out


# --------------------------------------------------------------- statistics
def boot_ci(x, rng):
  x = np.asarray(x, float)
  b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
  return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def one_sample(d):
  """Paired differences d (one per seed): mean +/- SEM, bootstrap 95 % CI, paired t, Wilcoxon."""
  d = np.asarray(d, float)
  d = d[np.isfinite(d)]
  out = dict(n=int(len(d)), values=d.tolist())
  if len(d) < 2:
    return out
  out.update(mean=float(d.mean()), sem=sem(d), boot_ci95=boot_ci(d, np.random.default_rng(BOOT_SEED)))
  t, p = stats.ttest_1samp(d, 0.0)
  out['paired_t'], out['paired_p'] = float(t), float(p)
  if len(d) >= 6 and np.any(d != 0):
    out['wilcoxon_p'] = float(stats.wilcoxon(d).pvalue)
  return out


def paired_relative(a, b, eq_bound=None):
  """a, b: dict seed -> value (a = PGC condition, b = reference). The paired
  difference a - b comes from code.analyze.paired_stats (the convention of
  RESULTS.md); added here: the ratio of means, the per-seed ratio a/b and
  log-ratio with mean +/- SEM, bootstrap 95 % CI over seeds, paired t on the
  log-ratio (against 0) and Wilcoxon on the log-ratio. Ratios (of means and per
  seed) are only defined when every value of both conditions is positive.
  mean_a / mean_b (from paired_stats) average over all seeds of each condition,
  as RESULTS.md prints them; paired_mean_a / paired_mean_b and the ratio of
  means use the paired seeds only (the two differ when one condition has fewer
  seeds, as pgc100 with 3 of the 8 gc100 seeds)."""
  res = paired_stats(a, b, eq_bound=eq_bound)
  seeds = res.get('seeds')
  if not seeds:
    return res
  av = np.array([a[s] for s in seeds], float)
  bv = np.array([b[s] for s in seeds], float)
  res['values_a'], res['values_b'] = av.tolist(), bv.tolist()
  res['paired_mean_a'], res['paired_mean_b'] = float(av.mean()), float(bv.mean())
  if np.all(av > 0) and np.all(bv > 0):
    res['ratio_of_means'] = float(av.mean() / bv.mean())
    r = av / bv
    lr = np.log(r)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    rb, lb = r[idx].mean(1), lr[idx].mean(1)
    res.update(ratios=r.tolist(), mean_ratio=float(r.mean()), sem_ratio=sem(r),
               ratio_ci95=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
               log_ratios=lr.tolist(), mean_log_ratio=float(lr.mean()), sem_log_ratio=sem(lr),
               log_ratio_ci95=[float(np.percentile(lb, 2.5)), float(np.percentile(lb, 97.5))],
               geo_mean_ratio=float(np.exp(lr.mean())),
               geo_mean_ratio_ci95=[float(np.exp(np.percentile(lb, 2.5))), float(np.exp(np.percentile(lb, 97.5)))])
    t, p = stats.ttest_1samp(lr, 0.0)
    res['log_ratio_t'], res['log_ratio_p'] = float(t), float(p)
    if len(seeds) >= 6 and np.any(lr != 0):
      res['log_ratio_wilcoxon_p'] = float(stats.wilcoxon(lr).pvalue)
  return res


_COMPARE_CACHE = {}


def compare(runs, cond_a, cond_b, arm, key, tost=False):
  """paired_relative of cond_a vs cond_b in one arm (cached: the consistency
  check asks for the same comparisons again)."""
  k = (id(runs), cond_a, cond_b, arm, key, tost)
  if k not in _COMPARE_CACHE:
    a, b = seed_values(runs, cond_a, arm, key), seed_values(runs, cond_b, arm, key)
    bound = 0.10 * abs(np.mean(list(b.values()))) if (tost and b) else None   # the TOST bound of code.analyze
    _COMPARE_CACHE[k] = paired_relative(a, b, eq_bound=bound) if (a and b) else None
  return _COMPARE_CACHE[k]


def interaction(lr1, lr2):
  """Per-seed differences of the score log-ratio between two arms (same seed
  index = same population): paired t, bootstrap CI, random sign-flip permutation
  test (N_PERM flips) and the exact sign-flip test over all 2^n patterns; for
  comparison, Welch's unpaired t-test and the correlation across seeds."""
  seeds = sorted(set(lr1) & set(lr2))
  d = np.array([lr1[s] - lr2[s] for s in seeds], float)
  n = len(d)
  out = one_sample(d)
  out.update(seeds=seeds, ratio_of_ratios=float(np.exp(d.mean())))
  rng = np.random.default_rng(BOOT_SEED)
  obs = abs(d.mean())
  flips = rng.choice([-1.0, 1.0], size=(N_PERM, n))
  out['perm_p'] = float(np.mean(np.abs((flips * d).mean(1)) >= obs - 1e-12))
  out['n_perm'] = N_PERM
  patterns = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
  out['exact_sign_flip_p'] = float(np.mean(np.abs((patterns * d).mean(1)) >= obs - 1e-12))
  out['n_sign_patterns'] = int(2 ** n)
  # for comparison: the unpaired test on the two sets of log-ratios (all seeds of each arm) and the
  # correlation of the paired log-ratios across seeds (how much variance the pairing can remove)
  v1, v2 = np.array(list(lr1.values()), float), np.array(list(lr2.values()), float)
  if len(v1) > 2 and len(v2) > 2:
    out['welch_p'] = float(stats.ttest_ind(v1, v2, equal_var=False).pvalue)
  if n > 2:
    out['corr_log_ratios'] = float(np.corrcoef([lr1[s] for s in seeds], [lr2[s] for s in seeds])[0, 1])
  return out


# ------------------------------------------------------------------ lesions
def lesion_cells(cfg):
  """Cells silenced by each probe, from the run's population config: no_pgc = all
  PGCs, no_gc = all grid cells, sham = round(SHAM_FRAC * n) cells drawn
  module-stratified from the grid cells (from all cells when there are none),
  as in code.common.sham_indices."""
  pop = cfg.get('population', {})
  n, M = int(cfg.get('n_cells', 512)), int(cfg.get('n_modules', 4))
  n_pgc = int(pop.get('n_pgc', len(pop.get('pgc_idx', []))))
  n_gc = int(pop.get('n_gc', n - n_pgc))
  per_mod_pgc = pop.get('pgc_per_module') or [n_pgc // M] * M
  k = int(round(SHAM_FRAC * n))
  per = [k // M + (1 if m < k % M else 0) for m in range(M)]
  pool = [(n // M - int(per_mod_pgc[m])) if n_gc else n // M for m in range(M)]
  return dict(no_pgc=n_pgc, no_gc=n_gc, sham=int(sum(min(per[m], pool[m]) for m in range(M))), no_code=n)


def lesion_costs(runs, cond, arm):
  """Per seed: intact - probe for no_pgc, no_gc, sham, absolute and per silenced
  cell; group means; paired tests no_pgc vs sham (absolute) and per-cell no_pgc
  vs per-cell no_gc."""
  rows = []
  for r in runs_of(runs, cond, arm):
    if not all(p in r['evals'] for p in ('intact',) + tuple(p for p, _ in LESIONS)):
      continue
    cells = lesion_cells(r['cfg'])
    row = dict(seed=r['seed'], cells=cells)
    for p, _ in LESIONS:
      c = probe_value(r, 'intact') - probe_value(r, p)
      row[f'cost_{p}'] = c
      row[f'percell_{p}'] = c / cells[p] if cells[p] else float('nan')
    rows.append(row)
  if not rows:
    return None
  out = dict(n=len(rows), seeds=[r['seed'] for r in rows], cells=rows[0]['cells'], per_seed=rows)
  for p, _ in LESIONS:
    for q in ('cost', 'percell'):
      v = np.array([r[f'{q}_{p}'] for r in rows], float)
      out[f'{q}_{p}'] = dict(mean=float(np.nanmean(v)) if np.isfinite(v).any() else float('nan'),
                             sem=sem(v), values=v.tolist())
  out['test_abs_no_pgc_vs_sham'] = one_sample([r['cost_no_pgc'] - r['cost_sham'] for r in rows])
  out['test_percell_no_pgc_vs_no_gc'] = one_sample([r['percell_no_pgc'] - r['percell_no_gc'] for r in rows])
  return out


# --------------------------------------------------------------- fig3 check
def fig3_check(runs, runs_dir):
  """What figures/fig3_probes.png is built from. make_figures.fig_probes draws,
  per arm and for gc100 / pgc33, the mean over seeds of the probes intact,
  no_code, no_goal, sham (mean of sham0-4) and no_pgc, read by
  make_figures.load_runs: runs/*/eval/<probe>.json without '@' in the name and
  with canonical = True. Recomputed here from the same files; make_figures is
  also imported and its values compared."""
  out = dict(probes=FIG3_PROBES, bars={}, files={})
  n_files, n_canon, n_200, frames = 0, 0, 0, set()
  for arm in ARMS:
    for cond in ('gc100', 'pgc33'):
      rs = runs_of(runs, cond, arm)
      for p in FIG3_PROBES:
        vals = [probe_value(r, p) for r in rs if p in r['evals']]
        out['bars'][f'{arm}/{cond}/{p}'] = dict(mean=float(np.mean(vals)), sem=sem(vals), n=len(vals))
      for r in rs:
        for k, v in r['evals'].items():
          if k == 'sham':
            continue
          n_files += 1
          n_canon += bool(v.get('canonical'))
          n_200 += v.get('episodes') == 200
          if k != 'untrained':
            frames.add(int(v.get('frames', -1)))
  out['files'] = dict(n=n_files, canonical=n_canon, episodes_200=n_200, frames=sorted(frames))
  try:
    spec = importlib.util.spec_from_file_location('make_figures', os.path.join(REPO, 'scripts', 'make_figures.py'))
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    mfr = mf.load_runs(runs_dir)
    dev = 0.0
    for key, v in out['bars'].items():
      arm, cond, p = key.split('/')
      vals = [mf.score(mfr, cond, arm, s, p) for s in mf.seeds_of(mfr, cond, arm)]
      vals = [x for x in vals if x is not None]
      dev = max(dev, abs(float(np.mean(vals)) - v['mean']))
    out['make_figures_max_abs_dev'] = dev
    out['match'] = bool(dev < 1e-9)
  except Exception as e:   # make_figures needs torch; the recomputation above stands on its own
    out['make_figures_error'] = repr(e)
  return out


# --------------------------------------------------------- RESULTS.md check
def parse_results_md(path):
  """Tables of RESULTS.md as {section: [row dict]} (cells kept as strings)."""
  tables, section, header = {}, None, None
  for line in open(path):
    line = line.rstrip('\n')
    if line.startswith('## '):
      section, header = line[3:].strip(), None
      continue
    if not line.startswith('|') or section is None:
      continue
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    if header is None:
      header = cells
      tables[section] = []
      continue
    if all(set(c) <= set('-:') for c in cells):
      continue
    tables[section].append(dict(zip(header, cells)))
  return tables


def parse_cell(s):
  """Numbers in a table cell with their printed decimals: '27.77 +/- 1.63' ->
  [(27.77, 2), (1.63, 2)]; '[1.64, 4.98]' -> [(1.64, 2), (4.98, 2)]; '-' -> []."""
  return [(float(m.group(0)), len(m.group(1)) if m.group(1) else 0)
          for m in re.finditer(r'[-+]?\d+(?:\.(\d+))?', s)]


class Checker:
  """Compares numbers printed in RESULTS.md with recomputed values; a mismatch is
  a difference beyond the rounding of the printed value."""

  def __init__(self):
    self.n, self.mismatches = 0, []

  def add(self, where, field, cell, values):
    printed = parse_cell(cell) if cell is not None else None
    values = [v for v in (values if isinstance(values, (list, tuple)) else [values])]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if printed is None:
      return
    self.n += 1
    if not printed and not finite:
      return
    if len(printed) != len(finite):
      self.mismatches.append(dict(where=where, field=field, results_md=cell,
                                  recomputed=[None if v is None else float(v) for v in values]))
      return
    for (p, d), v in zip(printed, finite):
      if abs(v - p) > 0.5 * 10 ** -d + 1e-6:
        self.mismatches.append(dict(where=where, field=field, results_md=cell, recomputed=float(v)))
        return


def consistency_check(runs, md_path, scope):
  """scope: [(arm, cond)] whose numbers are compared with RESULTS.md."""
  chk = Checker()
  if not os.path.exists(md_path):
    return dict(n_checked=0, mismatches=[], note=f'{md_path} not found')
  tables = parse_results_md(md_path)
  final = {(r['arm'], r['condition']): r for s, rows in tables.items() if s.startswith('Final evaluation') for r in rows}
  derived = {(r['arm'], r['condition']): r for s, rows in tables.items() if s.startswith('Derived') for r in rows}
  comps = {(r['comparison'], r['metric']): r for s, rows in tables.items() if s.startswith('Comparisons') for r in rows}
  probe_cols = {'intact': 'intact', 'untrained': 'untrained', 'no_code': 'no_code', 'no_current': 'no_current',
                'no_goal': 'no_goal', 'sham': 'sham', 'no_pgc': 'no_pgc', 'pgc_as_gc': 'pgc_as_gc',
                'fake_goal': 'fake_goal'}
  intact_cols = {'goals/ep': 'mean_goals', 'found': 'frac_found_goal', 'first goal': 'mean_first_goal_step',
                 'post-disc. rate': 'post_discovery_goal_rate', 'path eff': 'mean_path_eff'}
  derived_cols = {'learned gain': 'learned_gain', 'reliance (no_code)': 'reliance_code',
                  'reliance (no_current)': 'reliance_current', 'homing (no_goal)': 'homing',
                  'PGC-specific (sham - no_pgc)': 'pgc_specific'}
  comp_metrics = {'eval_intact_score': 'score', 'homing': 'homing', 'eval_intact_path_eff': 'path_eff',
                  'eval_intact_post_rate': 'post_rate', 'eval_intact_first_goal': 'first_goal'}
  for arm, cond in scope:
    rs = runs_of(runs, cond, arm)
    if not rs:
      continue
    key = (arm, COND_LABEL[cond])
    row = final.get(key)
    if row:
      where = f'final evaluation {arm}/{cond}'
      chk.add(where, 'n', row.get('n (done)'), [len(rs), len(rs)])
      for col, probe in probe_cols.items():
        v = [probe_value(r, probe) for r in rs if probe in r['evals']]
        chk.add(where, col, row.get(col), [np.mean(v), sem(v)] if v else [])
      for col, k in intact_cols.items():
        v = [probe_value(r, 'intact', k) for r in rs]
        chk.add(where, col, row.get(col), [np.mean(v), sem(v)])
    row = derived.get(key)
    if row:
      where = f'derived {arm}/{cond}'
      for col, k in derived_cols.items():
        v = [metric_value(r, k) for r in rs]
        chk.add(where, col, row.get(col), [np.nanmean(v), sem(v)])
    if cond == 'gc100':
      continue
    for metric, k in comp_metrics.items():
      row = comps.get((f'{arm}: {cond} - gc100', metric))
      if not row:
        continue
      st = compare(runs, cond, 'gc100', arm, k, tost=(k == 'score'))
      if not st:
        continue
      where = f'comparison {arm}: {cond} - gc100 / {metric}'
      chk.add(where, 'pairs', row.get('pairs'), st['n_pairs'])
      chk.add(where, 'mean a', row.get('mean a'), st['mean_a'])
      chk.add(where, 'mean b', row.get('mean b'), st['mean_b'])
      chk.add(where, 'mean diff +/- SEM', row.get('mean diff +/- SEM'), [st.get('mean_diff'), st.get('sem_diff')])
      chk.add(where, 'bootstrap 95% CI', row.get('bootstrap 95% CI'), st.get('boot_ci95', []))
      chk.add(where, 'paired t p', row.get('paired t p'), st.get('paired_p'))
      chk.add(where, 'Welch p', row.get('Welch p'), st.get('welch_p'))
      chk.add(where, 'Wilcoxon p (n>=6)', row.get('Wilcoxon p (n>=6)'), st.get('wilcoxon_p'))
      chk.add(where, 'TOST p (equiv. within 10%)', row.get('TOST p (equiv. within 10%)'), st.get('tost_p'))
  return dict(n_checked=chk.n, n_mismatch=len(chk.mismatches), mismatches=chk.mismatches)


# ------------------------------------------------------------------- figure
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


def make_figure(rel, lesion, ffb_rows, out_base):
  """A per-seed score ratio pgc33/gc100 per arm; B lesion cost per silenced cell
  per arm (pgc33); C score ratio vs gc100 of the arrangements and concentrations
  in the code-only arm."""
  rng = np.random.default_rng(7)
  fig = plt.figure(figsize=(11.5, 7.6))
  gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], hspace=0.5, wspace=0.42)

  # A: paired ratio per arm
  ax_a = ax = fig.add_subplot(gs[0, 0])
  ax.axhline(1.0, color=GRAY, lw=0.8, zorder=0)
  for j, arm in enumerate(ARMS):
    st = rel[arm]['score']
    r = np.array(st['ratios'], float)
    ax.scatter(np.full(len(r), j) + rng.uniform(-0.14, 0.14, len(r)), r, s=22, color=ORANGE, alpha=0.45,
               edgecolor='none', zorder=2)
    ax.errorbar(j, r.mean(), yerr=sem(r), fmt='o', color=ORANGE, ms=7, lw=1.6, capsize=0,
                markeredgecolor='white', markeredgewidth=0.8, zorder=4)
    ax.text(j + 0.22, r.mean(), f'{r.mean():.2f}\np = {st["log_ratio_p"]:.2g}', fontsize=8, color=INK2,
            ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5), zorder=5)
  ax.set_xticks(range(len(ARMS)))
  ax.set_xticklabels([ARM_TICK[a] for a in ARMS])
  ax.set_xlim(-0.5, len(ARMS) - 0.05)
  ax.set_ylabel('score ratio, grid + 33% PGC / grid\n(one dot per seed)')
  letter_ax(ax, 'A')

  # B: lesion cost per silenced cell
  ax = fig.add_subplot(gs[1, 0])
  ax.axhline(0, color=GRAY, lw=0.8, zorder=0)
  w = 0.25
  for k, (p, label) in enumerate(LESIONS):
    off = (k - 1) * (w + 0.03)
    for j, arm in enumerate(ARMS):
      les = lesion[arm]
      v = np.array(les[f'percell_{p}']['values'], float)
      ax.bar(j + off, v.mean(), w, color=LESION_COLOR[p], zorder=2,
             label=f'{label} ({les["cells"][p]} cells)' if j == 0 else None)
      ax.errorbar(j + off, v.mean(), yerr=sem(v), fmt='none', ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
      ax.scatter(np.full(len(v), j + off) + rng.uniform(-0.05, 0.05, len(v)), v, s=11, color=INK2, alpha=0.55,
                 edgecolor='white', lw=0.4, zorder=4)
  ax.set_xticks(range(len(ARMS)))
  ax.set_xticklabels([ARM_TICK[a] for a in ARMS])
  ax.set_ylabel('score lost per silenced cell\n(grid + 33% PGC agent)')
  ax.legend(fontsize=8, loc='upper left', handlelength=1.2)
  letter_ax(ax, 'B')

  # C: ratio vs gc100 in the code-only arm
  ax = fig.add_subplot(gs[:, 1])
  arr = [(c, l) for c, l in ARRANGEMENTS if c in ffb_rows]
  arr.sort(key=lambda cl: -ffb_rows[cl[0]]['mean_ratio'])
  conc = [(c, l) for c, l in CONCENTRATION if c in ffb_rows]
  blocks = [('arrangements of 33 % PGC', arr, lambda c: COND_COLOR.get(c, DARK)),
            ('concentration, quarter-spacing shift', conc, lambda c: DARK_ORANGE)]
  ys, labels, rows, cols, heads = [], [], [], [], []
  y = 0
  for title, items, color in blocks[::-1]:
    for c, l in items[::-1]:
      ys.append(y)
      labels.append(l)
      rows.append(ffb_rows[c]['ratios'])
      cols.append(color(c))
      y += 1
    heads.append((y - 0.35, title))
    y += 1.2
  ax.axvline(1.0, color=BLUE, ls=(0, (3, 3)), lw=1.2, zorder=0)
  ax.text(1.0, 1.005, 'grid only', color=BLUE, fontsize=8, ha='center', va='bottom', transform=ax.get_xaxis_transform())
  if 'pgc33' in ffb_rows:
    ax.axvline(ffb_rows['pgc33']['mean_ratio'], color=ORANGE, lw=1.0, alpha=0.8, zorder=0)
    ax.text(ffb_rows['pgc33']['mean_ratio'], 1.005, 'uniform', color=ORANGE, fontsize=8, ha='center', va='bottom',
            transform=ax.get_xaxis_transform())
  dot_rows(ax, ys, rows, cols, rng)
  for yi, c in zip(ys, [c for _, items, _ in blocks[::-1] for c, _ in items[::-1]]):
    ax.text(1.02, yi, f'{ffb_rows[c]["mean_ratio"]:.2f}', transform=ax.get_yaxis_transform(), fontsize=8,
            color=INK2, va='center', ha='left')
  ax.text(1.02, 1.005, 'mean ratio', transform=ax.transAxes, fontsize=8, color=INK2, ha='left', va='bottom')
  for yh, title in heads:
    ax.text(0.0, yh, title, transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK, ha='left', va='bottom',
            fontstyle='italic')
  ax.set_yticks(ys)
  ax.set_yticklabels(labels, fontsize=9)
  ax.set_ylim(-0.7, y - 0.4)
  ax.set_xlabel('score ratio vs grid only, code-only arm (one dot per seed)')
  # letter at the height of panel A's letter (C spans both rows)
  letter_ax(ax, 'C', x=-0.5, y=1.0 + 0.08 * ax_a.get_position().height / ax.get_position().height)

  fig.savefig(out_base + '.png', bbox_inches='tight')
  fig.savefig(out_base + '.svg', bbox_inches='tight')
  plt.close(fig)


# ------------------------------------------------------------------- report
def f(x, p=2):
  return '-' if x is None or not np.isfinite(x) else f'{x:.{p}f}'


def pm(d, k_mean, k_sem, p=2):
  if not d or d.get(k_mean) is None or not np.isfinite(d[k_mean]):
    return '-'
  return f'{f(d[k_mean], p)} +/- {f(d.get(k_sem), p)}'


def ci(c, p=2):
  return '-' if not c else f'[{f(c[0], p)}, {f(c[1], p)}]'


def pv(p):
  if p is None or not np.isfinite(p):
    return '-'
  return '<0.001' if p < 0.001 else f'{p:.3f}'


def rel_row(label, st, dec):
  """One table row of the relative-benefit tables."""
  return '| ' + ' | '.join([
      label, str(st['n_pairs']), f(st.get('paired_mean_b', st['mean_b']), dec),
      f(st.get('paired_mean_a', st['mean_a']), dec),
      pm(st, 'mean_diff', 'sem_diff', dec), ci(st.get('boot_ci95'), dec), pv(st.get('paired_p')),
      pv(st.get('wilcoxon_p')), f(st.get('ratio_of_means'), 3), pm(st, 'mean_ratio', 'sem_ratio', 3),
      ci(st.get('ratio_ci95'), 3), pm(st, 'mean_log_ratio', 'sem_log_ratio', 3), pv(st.get('log_ratio_p'))]) + ' |'


REL_HEADER = ['| {} | n | gc100 | {} | diff +/- SEM | 95 % CI (diff) | paired t p | Wilcoxon p | ratio of means | '
              'mean ratio +/- SEM | 95 % CI (ratio) | log-ratio +/- SEM | p (log-ratio) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']


def write_md(path, R, n_sources):
  L = ['# Relative benefit of predictive grid cells (final evaluations)', '',
       f'Source: {R["inputs"]["n_runs"]} runs with canonical 200-episode evaluations '
       f'({n_sources[0]} read from `{R["inputs"]["runs_dir"]}/`, {n_sources[1]} from `{R["inputs"]["fallback_dir"]}/`). '
       'All comparisons are paired within seed (seed s of one condition against seed s of the other; the two share one '
       'population); the condition means printed in the tables are over the paired seeds, so the difference and the '
       'ratio of means follow from the two printed means (results/RESULTS.md prints the mean over all seeds of each '
       'condition instead, which differs only where one condition has fewer seeds, as pgc100 with 3 of the 8 gc100 '
       'seeds). Difference statistics follow code/analyze.py (paired t-test, bootstrap 95 % CI over seeds with '
       f'{N_BOOT} resamples and a fixed seed, Wilcoxon signed-rank when n >= 6). Ratios are per seed, A / gc100, with '
       'mean +/- SEM and bootstrap CI over seeds; the log-ratio is tested against 0 with a paired t-test. Ratios are left '
       'blank when a value is not positive in every seed. For the first-goal step a lower value is better.', '']

  # 1 ---------------------------------------------------------------
  L += ['## 1. Relative benefit per arm: grid + 33 % PGC (pgc33) vs grid (gc100)', '',
        'Final intact score per arm, 8 paired seeds.', '']
  L += [REL_HEADER[0].format('arm', 'pgc33'), REL_HEADER[1]]
  for arm in ARMS:
    L.append(rel_row(ARM_TITLE[arm], R['relative_benefit'][arm]['score'], METRIC_DEC['score']))
  L += ['', 'Other metrics (intact evaluation unless the metric says otherwise; the PGC-specific lesion cost is '
        'sham - no_pgc with sham the mean of the five count-matched draws).', '']
  L += [REL_HEADER[0].format('arm, metric', 'pgc33'), REL_HEADER[1]]
  for key, label, dec, _ in METRICS:
    if key == 'score':
      continue
    for arm in ARMS:
      st = R['relative_benefit'][arm].get(key)
      if st:
        L.append(rel_row(f'{ARM_TITLE[arm]}, {label}', st, dec))
  L += ['', '### Interaction: is the relative benefit larger without vision?', '',
        'Per-seed log-ratio of the score, log(pgc33 / gc100), differenced between arms with the same seed index '
        '(same population; the policies of the two arms are trained independently, so the pairing removes only the '
        'variance shared through the population). Paired t-test, bootstrap CI, sign-flip permutation test with '
        f'{N_PERM} random flips and the exact test over all 2^n sign patterns; for comparison, Welch\'s unpaired t-test '
        'on the two sets of per-seed log-ratios and their correlation across seeds.', '',
        '| contrast | n | mean difference of log-ratios +/- SEM | 95 % CI | ratio of ratios | paired t p | Wilcoxon p | '
        'sign-flip p (random) | sign-flip p (exact, 2^n patterns) | Welch p (unpaired) | r across seeds |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
  for name, st in R['interaction'].items():
    L.append('| ' + ' | '.join([name, str(st['n']), pm(st, 'mean', 'sem', 3), ci(st.get('boot_ci95'), 3),
                                f(st['ratio_of_ratios'], 3), pv(st.get('paired_p')), pv(st.get('wilcoxon_p')),
                                pv(st['perm_p']), pv(st['exact_sign_flip_p']), pv(st.get('welch_p')),
                                f(st.get('corr_log_ratios'), 2)]) + ' |')
  inter = [st for name, st in R['interaction'].items() if name.startswith(ARM_TITLE['ffb'])]
  if inter:
    inc = all(st['boot_ci95'][0] <= 0 <= st['boot_ci95'][1] for st in inter)
    L += ['', 'The mean difference of log-ratios between the code-only arm and the two vision arms is ' +
          ' and '.join(f'{st["mean"]:+.3f} +/- {st["sem"]:.3f}' for st in inter) +
          f'; the bootstrap CIs {"include" if inc else "do not all include"} 0 and the smallest p-value of the two '
          f'contrasts is {min(min(st["paired_p"], st["perm_p"], st["exact_sign_flip_p"]) for st in inter):.2f}.']
    L += ['', f'With n = {inter[0]["n"]} seeds the SEM of each between-arm difference is as large as the difference '
          'itself, and the ratio-of-ratios CIs (' +
          ', '.join(f'[{np.exp(st["boot_ci95"][0]):.2f}, {np.exp(st["boot_ci95"][1]):.2f}]' for st in inter) +
          ') span both a smaller and a clearly larger relative benefit without vision. The point estimates favour a '
          'larger relative benefit without vision, but these data neither establish nor exclude an arm dependence of '
          'the relative benefit; the unpaired Welch test agrees (p = ' +
          ', '.join(pv(st.get('welch_p')) for st in inter) + ').']
  lr = {arm: R['relative_benefit'][arm]['score'] for arm in ARMS}
  L += ['', 'Mean log-ratios per arm: ' + ', '.join(
      f'{ARM_TITLE[a]} {lr[a]["mean_log_ratio"]:+.3f} +/- {lr[a]["sem_log_ratio"]:.3f} (geometric-mean ratio '
      f'{lr[a]["geo_mean_ratio"]:.3f}, CI {ci(lr[a]["geo_mean_ratio_ci95"], 3)})' for a in ARMS) + '.', '']

  # 2 ---------------------------------------------------------------
  L += ['## 2. 100 % PGC vs grid, and the code-only arrangements and concentrations vs grid', '',
        'pgc100 vs gc100, 3 paired seeds per arm (no Wilcoxon below n = 6; the bootstrap CI over 3 seeds takes '
        'only 10 distinct values).', '']
  L += [REL_HEADER[0].format('arm, metric', 'pgc100'), REL_HEADER[1]]
  for key, label, dec, _ in METRICS:
    for arm in ARMS:
      st = R['pgc100_vs_gc100'][arm].get(key)
      if st:
        L.append(rel_row(f'{ARM_TITLE[arm]}, {label}', st, dec))
  L += ['', 'Code-only arm, final intact score of every PGC arrangement (33 % PGC) and of the concentration series '
        '(quarter-spacing shift; its 33 % point is the quarter-spacing arrangement) vs gc100, 8 paired seeds each.', '']
  L += [REL_HEADER[0].format('condition', 'condition'), REL_HEADER[1]]
  for cond, label in ARRANGEMENTS + CONCENTRATION:
    st = R['ffb_vs_gc100'].get(cond)
    if st:
      L.append(rel_row(f'{label} ({cond})', st, 2))

  # 3 ---------------------------------------------------------------
  L += ['', '## 3. Lesion cost per silenced cell', '',
        'Cost = intact score - score with the cells zeroed at test time (final evaluation, 200 episodes); sham is the mean '
        'of the five count-matched draws of random grid cells. Per-cell numbers are cost / cells silenced and assume '
        'that costs add up linearly over cells. pgc33: 8 seeds per arm; gc100 and pgc100 are reference rows '
        '(in gc100 no_gc is the whole code and no_pgc silences nothing; in pgc100 no_pgc is the whole code and sham '
        'draws from the PGCs).', '',
        '| arm | condition | lesion | cells | cost +/- SEM | cost per cell +/- SEM |', '|---|---|---|---|---|---|']
  for arm in ARMS:
    for cond in ('pgc33', 'gc100', 'pgc100'):
      les = R['lesion_per_cell'][arm].get(cond)
      if not les:
        continue
      for p, label in LESIONS:
        L.append('| ' + ' | '.join([ARM_TITLE[arm], f'{cond} (n = {les["n"]})', f'{label} ({p})', str(les['cells'][p]),
                                    pm(les[f'cost_{p}'], 'mean', 'sem', 2),
                                    pm(les[f'percell_{p}'], 'mean', 'sem', 4)]) + ' |')
  L += ['', 'Paired tests (pgc33): PGC lesion cost minus sham cost (absolute; equals the PGC-specific lesion cost '
        'sham - no_pgc) and per-cell PGC lesion cost minus per-cell grid-cell lesion cost.', '',
        '| arm | test | mean +/- SEM | 95 % CI | paired t p | Wilcoxon p |', '|---|---|---|---|---|---|']
  for arm in ARMS:
    les = R['lesion_per_cell'][arm].get('pgc33')
    if not les:
      continue
    for key, label, dec in (('test_abs_no_pgc_vs_sham', 'cost(no_pgc) - cost(sham)', 2),
                            ('test_percell_no_pgc_vs_no_gc', 'per cell: no_pgc - no_gc', 4)):
      st = les[key]
      L.append('| ' + ' | '.join([ARM_TITLE[arm], label, pm(st, 'mean', 'sem', dec), ci(st.get('boot_ci95'), dec),
                                  pv(st.get('paired_p')), pv(st.get('wilcoxon_p'))]) + ' |')

  # 4 ---------------------------------------------------------------
  c = R['fig3_check']
  L += ['', '## 4. Figure 3 check (figures/fig3_probes.png)', '',
        'make_figures.fig_probes reads, through make_figures.load_runs, the files runs/<run>/eval/<probe>.json whose '
        'name has no "@" and whose "canonical" flag is true (written only for ckpt_final.pt, 200 episodes, stochastic '
        'policy, evaluation seed 424242); the sham bar is the mean of sham0-4. Recomputed from the same files '
        f'(mean +/- SEM over seeds): {c["files"]["n"]} probe files, {c["files"]["canonical"]} canonical, '
        f'{c["files"]["episodes_200"]} with 200 episodes, trained-policy frames {c["files"]["frames"]}.', '',
        '| arm | condition | n | ' + ' | '.join(FIG3_PROBES) + ' |', '|---|---|---|' + '---|' * len(FIG3_PROBES)]
  for arm in ARMS:
    for cond in ('gc100', 'pgc33'):
      b = [c['bars'][f'{arm}/{cond}/{p}'] for p in FIG3_PROBES]
      L.append('| ' + ' | '.join([ARM_TITLE[arm], cond, str(b[0]['n'])] + [pm(x, 'mean', 'sem', 2) for x in b]) + ' |')
  L += ['', 'Intact means to three decimals (gc100 / pgc33): ' + ', '.join(
      f'{ARM_TITLE[arm]} {c["bars"][f"{arm}/gc100/intact"]["mean"]:.3f} / {c["bars"][f"{arm}/pgc33/intact"]["mean"]:.3f}'
      for arm in ARMS) + '.']
  if 'match' in c:
    L.append('')
    L.append(f'Bar heights recomputed with make_figures.load_runs/score agree with the values above to '
             f'{c["make_figures_max_abs_dev"]:.1e}: the figure is drawn from the canonical final evaluations.'
             if c['match'] else
             f'Bar heights recomputed with make_figures.load_runs/score deviate by up to {c["make_figures_max_abs_dev"]:.3g}.')
  else:
    L += ['', f'make_figures could not be imported for a direct comparison ({c.get("make_figures_error")}).']

  # 5 ---------------------------------------------------------------
  L += ['', '## 5. Secondary endpoints: grid + 33 % PGC vs grid, per arm', '',
        'Intact final evaluation, 8 paired seeds. Path efficiency is averaged over goal returns; the post-discovery goal '
        'rate counts goals after the first; the first-goal step is lower when the goal is found sooner.', '',
        '| arm | endpoint | gc100 | pgc33 | diff +/- SEM | 95 % CI | paired t p | Wilcoxon p |',
        '|---|---|---|---|---|---|---|---|']
  for arm in ARMS:
    for key in SECONDARY:
      st = R['relative_benefit'][arm].get(key)
      if st:
        dec = METRIC_DEC[key]
        L.append('| ' + ' | '.join([ARM_TITLE[arm], METRIC_LABEL[key], f(st.get('paired_mean_b', st['mean_b']), dec),
                                    f(st.get('paired_mean_a', st['mean_a']), dec),
                                    pm(st, 'mean_diff', 'sem_diff', dec), ci(st.get('boot_ci95'), dec),
                                    pv(st.get('paired_p')), pv(st.get('wilcoxon_p'))]) + ' |')

  # consistency -----------------------------------------------------
  cc = R['consistency']
  L += ['', '## Consistency check', '',
        f'{cc["n_checked"]} numbers of results/RESULTS.md (final-evaluation means, derived quantities and paired '
        'comparisons vs gc100 for the conditions used here) were recomputed from the evaluation JSONs; '
        f'{cc["n_mismatch"]} differ beyond the rounding of the printed value.']
  if cc['mismatches']:
    L += ['', '| where | field | RESULTS.md | recomputed |', '|---|---|---|---|']
    for m in cc['mismatches']:
      L.append(f'| {m["where"]} | {m["field"]} | {m["results_md"]} | {m["recomputed"]} |')

  # figure caption --------------------------------------------------
  les = R['lesion_per_cell']['ffb']['pgc33']['cells']
  L += ['', '## Figure 10 (figures/fig10_relative_benefit.png)', '',
        'A: score ratio grid + 33 % PGC / grid per seed (dots) with mean +/- SEM, per arm; the line marks a ratio of 1; '
        'the labels give the mean ratio and the paired t p-value of the log-ratio against 0. '
        f'B: score lost per silenced cell in the grid + 33 % PGC agent, per arm, for the PGC lesion ({les["no_pgc"]} cells), '
        f'the grid-cell lesion ({les["no_gc"]} cells) and the sham lesion ({les["sham"]} random grid cells, mean of 5 draws); '
        'bars are means over 8 seeds with SEM, dots are seeds. '
        'C: code-only arm, score ratio vs grid only for the PGC arrangements at 33 % and for the concentration series '
        '(quarter-spacing shift), dots per seed and mean +/- SEM; the dashed line is the grid-only reference (ratio 1) '
        'and the solid line the uniform 20 cm arrangement.']
  with open(path, 'w') as fh:
    fh.write('\n'.join(L) + '\n')


# --------------------------------------------------------------------- main
def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--fallback_dir', default='data/evals', help="used for a run whose runs_dir entry lacks the eval JSONs ('' to disable)")
  ap.add_argument('--results', default='results')
  ap.add_argument('--figures', default='figures')
  ap.add_argument('--results_md', default=None, help='RESULTS.md to cross-check (default <results>/RESULTS.md)')
  args = ap.parse_args(argv)
  os.makedirs(args.results, exist_ok=True)
  os.makedirs(args.figures, exist_ok=True)
  runs = load_runs(args.runs_dir, args.fallback_dir or None)
  if not runs:
    print('no runs with canonical evaluations found')
    return
  n_sources = (sum(r['source'] == args.runs_dir for r in runs.values()),
               sum(r['source'] != args.runs_dir for r in runs.values()))
  R = dict(inputs=dict(runs_dir=args.runs_dir, fallback_dir=args.fallback_dir, n_runs=len(runs),
                       n_from_runs_dir=n_sources[0], n_from_fallback=n_sources[1],
                       n_boot=N_BOOT, boot_seed=BOOT_SEED, n_perm=N_PERM))

  # 1 relative benefit pgc33 vs gc100, all metrics, per arm; interaction across arms
  R['relative_benefit'] = {arm: {} for arm in ARMS}
  for arm in ARMS:
    for key, _, _, _ in METRICS:
      st = compare(runs, 'pgc33', 'gc100', arm, key, tost=(key == 'score'))
      if st:
        R['relative_benefit'][arm][key] = st
  lr = {arm: dict(zip(R['relative_benefit'][arm]['score']['seeds'], R['relative_benefit'][arm]['score']['log_ratios']))
        for arm in ARMS if 'log_ratios' in R['relative_benefit'][arm].get('score', {})}
  R['interaction'] = {}
  for a1, a2 in (('ffb', 'ff'), ('ffb', 'lstm'), ('ff', 'lstm')):
    if a1 in lr and a2 in lr:
      R['interaction'][f'{ARM_TITLE[a1]} - {ARM_TITLE[a2]}'] = interaction(lr[a1], lr[a2])

  # 2 pgc100 vs gc100; code-only arrangements and concentrations vs gc100
  R['pgc100_vs_gc100'] = {arm: {} for arm in ARMS}
  for arm in ARMS:
    for key, _, _, _ in METRICS:
      st = compare(runs, 'pgc100', 'gc100', arm, key, tost=(key == 'score'))
      if st:
        R['pgc100_vs_gc100'][arm][key] = st
  R['ffb_vs_gc100'] = {}
  for cond, _ in ARRANGEMENTS + CONCENTRATION:
    st = compare(runs, cond, 'gc100', 'ffb', 'score', tost=True)
    if st and 'ratios' in st:
      R['ffb_vs_gc100'][cond] = st

  # 3 lesion costs per cell
  R['lesion_per_cell'] = {arm: {} for arm in ARMS}
  for arm in ARMS:
    for cond in ('pgc33', 'gc100', 'pgc100'):
      les = lesion_costs(runs, cond, arm)
      if les:
        R['lesion_per_cell'][arm][cond] = les

  # 4 figure 3 inputs
  R['fig3_check'] = fig3_check(runs, args.runs_dir)

  # 6 consistency with RESULTS.md
  scope = [(arm, cond) for arm in ARMS for cond in ('gc100', 'pgc33', 'pgc100')]
  scope += [('ffb', cond) for cond, _ in ARRANGEMENTS + CONCENTRATION if cond != 'pgc33']
  R['consistency'] = consistency_check(runs, args.results_md or os.path.join(args.results, 'RESULTS.md'), scope)

  make_figure(R['relative_benefit'], {a: R['lesion_per_cell'][a]['pgc33'] for a in ARMS}, R['ffb_vs_gc100'],
              os.path.join(args.figures, 'fig10_relative_benefit'))
  write_md(os.path.join(args.results, 'relative_benefit.md'), R, n_sources)
  with open(os.path.join(args.results, 'relative_benefit.json'), 'w') as fh:
    json.dump(R, fh, indent=1, default=float)

  for arm in ARMS:
    st = R['relative_benefit'][arm]['score']
    print(f'{arm:4s} gc100 {st["mean_b"]:.2f}  pgc33 {st["mean_a"]:.2f}  diff {st["mean_diff"]:+.2f} +/- {st["sem_diff"]:.2f} '
          f'p={st["paired_p"]:.3f}  ratio {st["mean_ratio"]:.3f} +/- {st["sem_ratio"]:.3f} CI {ci(st["ratio_ci95"], 3)} '
          f'log-ratio p={st["log_ratio_p"]:.3f}')
  for name, st in R['interaction'].items():
    print(f'interaction {name}: d log-ratio {st["mean"]:+.3f} +/- {st["sem"]:.3f}  t p={st["paired_p"]:.3f}  '
          f'perm p={st["perm_p"]:.3f}  exact p={st["exact_sign_flip_p"]:.3f}')
  cc = R['consistency']
  print(f'consistency: {cc["n_checked"]} checked, {cc["n_mismatch"]} mismatches')
  print('written:', os.path.join(args.results, 'relative_benefit.md'), os.path.join(args.results, 'relative_benefit.json'),
        os.path.join(args.figures, 'fig10_relative_benefit.png'))


if __name__ == '__main__':
  main()
