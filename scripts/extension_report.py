"""Report on the extended runs: training beyond the 300M-frame budget.

Reads runs_ext/<run>/{config.json, metrics.jsonl, episodes.csv, eval/*.json}.
An extended run keeps its 300M-frame final evaluations as checkpoint
evaluations (eval/<probe>@300006400f_200ep.json, demoted when training
continued), gets 64-episode checkpoint probes every 25M frames
(eval/<probe>@<frames>f_64ep.json), 200-episode milestone evaluations
(eval/<probe>@<frames>f_200ep.json, e.g. at 600M and 1B) and, once finished,
canonical finals (eval/<probe>.json).

Writes <out>/EXTENSION.md and <out>/extension.json with
  (a) per run, the 200-episode intact score at every milestone (300M, 600M,
      1B, ..., final) and the reliance intact - no_code at the same milestones
  (b) per arm, the within-seed gap pgc33 - gc100 at 300M and at the latest
      milestone both conditions reached, with code.analyze.paired_stats, and
      the paired change of the gap between the two
  (c) per run, a plateau estimate from the training return (metrics.jsonl
      avg_return_100 vs frames): a - b exp(-frames / tau) fitted to frames
      >= 100M, the frame count where the fit reaches 95 % of its asymptote,
      and the mean return over the last 50M frames
and <figures>/fig11_training_length.png (+ .svg): left, the paired gap per
seed at each milestone, one panel per arm; right, the training curves of the
runs (binned 10M) with the 300M budget marked and the fitted plateaus.

Runs without error on partial data (only the 300M evaluations) and prints a
message when <runs_dir> does not exist yet.

  python scripts/extension_report.py --runs_dir runs_ext --out results_ext --figures figures
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from code import analyze as an  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

try:   # colour constants and rc style of the publication figures
  import make_figures as mf
  BLUE, ORANGE, AQUA, GRAY, INK, INK2 = mf.BLUE, mf.ORANGE, mf.AQUA, mf.GRAY, mf.INK, mf.INK2
  ARM_TITLE = mf.ARM_TITLE
except Exception:
  BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9891'
  INK, INK2 = '#0b0b0b', '#52514e'
  ARM_TITLE = {'ffb': 'code only', 'ff': 'code + vision', 'lstm': 'LSTM + vision'}
  plt.rcParams.update({
      'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
      'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
      'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': False,
      'savefig.dpi': 200, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

COND_COLOR = {'gc100': BLUE, 'pgc33': ORANGE}
MILESTONE_EPISODES = 200
BUDGET_300M = 300e6


# ----------------------------------------------------------------- loading
def milestone_label(frames):
  """300006400 -> '300M', 600000000 -> '600M', 1000000000 -> '1B'."""
  if frames >= 1e9:
    return f'{frames / 1e9:g}B'
  return f'{int(round(frames / 1e6))}M'


def label_frames(label):
  return float(label[:-1]) * (1e9 if label.endswith('B') else 1e6)


def read_metrics(path):
  frames, ret = [], []
  if not os.path.exists(path):
    return np.zeros(0), np.zeros(0)
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
      except Exception:
        continue
      if rec.get('avg_return_100') is None:
        continue
      frames.append(float(rec['frames']))
      ret.append(float(rec['avg_return_100']))
  return np.array(frames), np.array(ret)


def read_episodes(path):
  if not os.path.exists(path):
    return None
  try:
    ep = np.genfromtxt(path, delimiter=',', names=True)
  except Exception:
    return None
  if ep.size == 0:
    return None
  ep = np.atleast_1d(ep)
  if 'ret' not in ep.dtype.names:
    ep.dtype.names = tuple('ret' if n in ('return', 'return_') else n for n in ep.dtype.names)
  return ep


def final_frames(run_dir):
  p = os.path.join(run_dir, 'ckpt_final.pt')
  if not os.path.exists(p):
    return None
  try:
    import torch
    return int(torch.load(p, map_location='cpu').get('frames', 0))
  except Exception:
    return None


def load_runs(runs_dir):
  runs = []
  for d in sorted(glob.glob(os.path.join(runs_dir, '*'))):
    name = os.path.basename(d)
    m = an.NAME_RE.match(name)
    cfg_p = os.path.join(d, 'config.json')
    if not m or not os.path.isdir(d) or not os.path.exists(cfg_p):
      continue
    cfg = json.load(open(cfg_p))
    r = dict(name=name, cond=m.group(1) + m.group(2), arm=m.group(3), seed=int(m.group(4)),
             target=int(cfg['frames']), episode_len=int(cfg.get('episode_len', 1350)),
             milestones={}, ckpt_probes={}, final_frames=final_frames(d), final_label=None)
    for f in glob.glob(os.path.join(d, 'eval', '*.json')):
      base = os.path.basename(f)[:-5]
      try:
        ev = json.load(open(f))
      except Exception:
        continue
      if '@' in base:
        probe, rest = base.split('@', 1)
        mm = re.match(r'(\d+)f_(\d+)ep$', rest)
        if not mm:
          continue
        frames, episodes = int(mm.group(1)), int(mm.group(2))
        if episodes == MILESTONE_EPISODES:
          r['milestones'].setdefault(milestone_label(frames), {})[probe] = ev
        else:
          r['ckpt_probes'].setdefault(probe, {})[frames] = ev.get('mean_score')
      elif ev.get('canonical', True) and r['final_frames'] is not None:
        lab = milestone_label(r['final_frames'])
        r['final_label'] = lab
        r['milestones'].setdefault(lab, {})[probe] = ev
    r['metrics'] = read_metrics(os.path.join(d, 'metrics.jsonl'))
    r['ep'] = read_episodes(os.path.join(d, 'episodes.csv'))
    r['frames'] = float(r['metrics'][0].max()) if len(r['metrics'][0]) else \
        (float(r['ep']['frames'].max()) if r['ep'] is not None else 0.0)
    runs.append(r)
  return runs


# ----------------------------------------------------------------- analysis
def score(r, label, probe='intact'):
  ev = r['milestones'].get(label, {}).get(probe)
  return float(ev['mean_score']) if ev and ev.get('mean_score') is not None else None


def all_labels(runs):
  return sorted({lab for r in runs for lab in r['milestones']}, key=label_frames)


def gap_analysis(runs):
  """Per arm: the paired gap pgc33 - gc100 at every milestone with pairs, and the
  paired change between 300M and the latest milestone with >= 2 pairs."""
  out = {}
  for arm in an.ARMS:
    by = {(r['cond'], r['seed']): r for r in runs if r['arm'] == arm and r['cond'] in ('gc100', 'pgc33')}
    seeds = sorted({s for (_, s) in by})
    per_label = {}
    for lab in all_labels([r for r in by.values()]):
      pairs = {}
      for s in seeds:
        g, p = by.get(('gc100', s)), by.get(('pgc33', s))
        if g is None or p is None:
          continue
        a, b = score(g, lab), score(p, lab)
        if a is not None and b is not None:
          pairs[s] = dict(gc100=a, pgc33=b, gap=b - a)
      if pairs:
        per_label[lab] = dict(pairs=pairs,
                              stats=an.paired_stats({s: v['pgc33'] for s, v in pairs.items()},
                                                    {s: v['gc100'] for s, v in pairs.items()}))
    if not per_label:
      continue
    labels = sorted(per_label, key=label_frames)
    base = '300M' if '300M' in per_label else labels[0]
    later = [lab for lab in labels if label_frames(lab) > label_frames(base)
             and len(per_label[lab]['pairs']) >= 2]
    res = dict(labels=labels, per_label=per_label, base=base, latest=None, change=None)
    if later:
      latest = later[-1]
      common = sorted(set(per_label[base]['pairs']) & set(per_label[latest]['pairs']))
      if common:
        res['latest'] = latest
        res['change'] = an.paired_stats({s: per_label[latest]['pairs'][s]['gap'] for s in common},
                                        {s: per_label[base]['pairs'][s]['gap'] for s in common})
    out[arm] = res
  return out


def saturating(x, a, b, tau):
  return a - b * np.exp(-x / tau)


TAU_MAX_M = 1e4     # upper bound of tau (millions of frames); a fit at the bound has not saturated


def plateau_fit(frames, ret, fit_from=100e6, last=50e6, slope_window=100e6):
  """Fit a - b exp(-frames/tau) (frames in millions) to the training return
  for frames >= fit_from. Returns the fit, the frame count at 95 % of the
  asymptote, the mean return over the last `last` frames and the linear slope
  over the last `slope_window` frames (score per 100M frames). The fit is
  flagged unreliable when tau sits at its bound or the asymptote lies far
  above the observed returns (the curve has not started to saturate)."""
  res = dict(n_points=0, frames_max=float(frames.max()) if len(frames) else 0.0,
             last_mean=float('nan'), slope_last_100M=float('nan'), asymptote=float('nan'),
             b=float('nan'), tau_M=float('nan'), frames_95pct_M=float('nan'),
             fit_at_300M=float('nan'), fit_at_max=float('nan'),
             frac_at_300M=float('nan'), frac_at_max=float('nan'), reliable=False)
  if not len(frames):
    return res
  tail = frames >= frames.max() - last
  res['last_mean'] = float(ret[tail].mean()) if tail.any() else float('nan')
  win = frames >= frames.max() - slope_window
  if win.sum() >= 10 and np.ptp(frames[win]) > 0:
    res['slope_last_100M'] = float(np.polyfit(frames[win] / 1e8, ret[win], 1)[0])
  ok = frames >= fit_from
  res['n_points'] = int(ok.sum())
  if ok.sum() < 10:
    res['fit_error'] = f'only {int(ok.sum())} points beyond {fit_from / 1e6:.0f}M'
    return res
  x, y = frames[ok] / 1e6, ret[ok]
  try:
    from scipy.optimize import curve_fit
    a0 = float(y[-max(1, len(y) // 10):].mean())
    b0 = max(a0 - float(y[: max(1, len(y) // 10)].mean()), 1e-3) * np.e
    popt, _ = curve_fit(saturating, x, y, p0=[a0, b0, 100.0],
                        bounds=([-np.inf, -np.inf, 1.0], [np.inf, np.inf, TAU_MAX_M]), maxfev=20000)
  except Exception as e:  # noqa: BLE001
    res['fit_error'] = str(e)
    return res
  a, b, tau = (float(v) for v in popt)
  res.update(asymptote=a, b=b, tau_M=tau, fit_at_300M=float(saturating(300.0, a, b, tau)),
             fit_at_max=float(saturating(x.max(), a, b, tau)))
  if a > 0:
    res['frac_at_300M'] = res['fit_at_300M'] / a
    res['frac_at_max'] = res['fit_at_max'] / a
  if a > 0 and b > 0:
    ratio = 0.05 * a / b
    res['frames_95pct_M'] = float(-tau * np.log(ratio)) if ratio < 1 else 0.0
  res['reliable'] = bool(b > 0 and tau < 0.9 * TAU_MAX_M and a <= 1.5 * float(y.max()))
  return res


# ------------------------------------------------------------------- figure
def fig_training_length(runs, gaps, fits, fig_dir, bin_frames=10e6):
  arms = [a for a in an.ARMS if a in gaps]
  n_left = max(len(arms), 1)
  fig, axes = plt.subplots(1, n_left + 1, figsize=(3.1 * n_left + 6.0, 3.8),
                           gridspec_kw=dict(width_ratios=[1] * n_left + [1.9]))
  axes = np.atleast_1d(axes)
  # Left: the paired gap per seed at each milestone.
  for j in range(n_left):
    ax = axes[j]
    if not arms:
      ax.text(0.5, 0.5, 'no gc100 / pgc33 pairs\nwith 200-episode evaluations', ha='center',
              va='center', fontsize=9, color=INK2, transform=ax.transAxes)
      ax.set_axis_off()
      continue
    arm = arms[j]
    g = gaps[arm]
    labels = g['labels']
    x_of = {lab: k for k, lab in enumerate(labels)}
    seeds = sorted({s for lab in labels for s in g['per_label'][lab]['pairs']})
    for s in seeds:
      pts = [(x_of[lab], g['per_label'][lab]['pairs'][s]['gap']) for lab in labels
             if s in g['per_label'][lab]['pairs']]
      if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GRAY, lw=0.8, alpha=0.6, zorder=1)
      ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, color=ORANGE, zorder=3,
                 edgecolor='white', lw=0.6)
    for lab in labels:
      d = np.array([v['gap'] for v in g['per_label'][lab]['pairs'].values()], float)
      ax.hlines(d.mean(), x_of[lab] - 0.25, x_of[lab] + 0.25, color=INK, lw=2.5, zorder=4)
      if len(d) > 1:
        ax.vlines(x_of[lab] + 0.32, d.mean() - an.sem(d), d.mean() + an.sem(d), color=INK,
                  lw=1.2, zorder=4)
    ax.axhline(0, color=GRAY, ls=(0, (3, 3)), lw=1.0, zorder=0)
    ax.set_xticks(list(x_of.values()))
    ax.set_xticklabels([f'{lab}\n(n={len(g["per_label"][lab]["pairs"])})' for lab in labels],
                       fontsize=8.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_title(ARM_TITLE.get(arm, arm), color=INK, fontsize=11)
    if j == 0:
      ax.set_ylabel('score gap per seed\n(grid + 33% PGC) − grid')
  # Right: training curves with the 300M budget marked and the fitted plateaus.
  ax = axes[-1]
  budget = max([r['frames'] for r in runs] + [BUDGET_300M])
  f95 = [fits[r['name']]['frames_95pct_M'] * 1e6 for r in runs
         if fits[r['name']].get('reliable') and np.isfinite(fits[r['name']].get('frames_95pct_M', np.nan))]
  # fits are extrapolated to their 95 % point, but at most to twice the data range
  x_max = max(budget, min(max(f95), 2 * budget) if f95 else 0) * 1.04
  edges = np.arange(0, budget + bin_frames, bin_frames)
  centers = (edges[:-1] + edges[1:]) / 2 / 1e6
  legend = {'gc100': 'grid cells', 'pgc33': 'grid + 33% PGC'}
  seen = set()
  for r in runs:
    col = COND_COLOR.get(r['cond'], GRAY)
    label = legend.get(r['cond'], 'other conditions')
    if r['ep'] is not None:
      mu = an.binned_curve(r['ep'], 'ret', edges, r['episode_len'])
      ax.plot(centers, mu, color=col, lw=1.3, alpha=0.85, zorder=2,
              label=None if label in seen else label)
      seen.add(label)
    ft = fits[r['name']]
    if ft.get('reliable') and np.isfinite(ft.get('asymptote', np.nan)):
      xs = np.linspace(100.0, x_max / 1e6, 200)
      ax.plot(xs, saturating(xs, ft['asymptote'], ft['b'], ft['tau_M']), color=col, lw=0.9,
              ls=(0, (1.5, 2)), alpha=0.8, zorder=1)
      if np.isfinite(ft.get('frames_95pct_M', np.nan)):
        ax.scatter([ft['frames_95pct_M']], [0.95 * ft['asymptote']], marker='v', s=22, color=col,
                   zorder=5, edgecolor='white', lw=0.5)
  ax.axvline(BUDGET_300M / 1e6, color=GRAY, ls=(0, (3, 3)), lw=1.2, zorder=0)
  ax.text(BUDGET_300M / 1e6, 1.0, ' 300M budget', transform=ax.get_xaxis_transform(), fontsize=8,
          color=INK2, va='top', ha='left')
  ax.set_xlim(0, x_max / 1e6)
  if seen:
    ax.legend(loc='lower right', fontsize=8.5)
  ax.set_xlabel('training frames (millions)')
  ax.set_ylabel('episode score (training, 10M bins)')
  ax.set_title('training beyond 300M (dotted: saturating fit, ▾: 95 % of asymptote)',
               color=INK, fontsize=9.5)
  fig.tight_layout()
  os.makedirs(fig_dir, exist_ok=True)
  fig.savefig(os.path.join(fig_dir, 'fig11_training_length.png'))
  fig.savefig(os.path.join(fig_dir, 'fig11_training_length.svg'))
  plt.close(fig)


# ------------------------------------------------------------------- report
def fmt(v, p=2):
  return '-' if v is None or not np.isfinite(v) else f'{v:.{p}f}'


def fmt_stats(st):
  if not st or 'mean_diff' not in st:
    return ['-'] * 4
  ci = f'[{st["boot_ci95"][0]:.2f}, {st["boot_ci95"][1]:.2f}]' if 'boot_ci95' in st else '-'
  return [f'{st["mean_diff"]:+.2f} +/- {st["sem_diff"]:.2f}', ci,
          fmt(st.get('paired_p'), 3), fmt(st.get('wilcoxon_p'), 3)]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--runs_dir', default='runs_ext')
  ap.add_argument('--out', default='results_ext')
  ap.add_argument('--figures', default='figures')
  ap.add_argument('--fit_from', type=float, default=100e6, help='fit frames >= this')
  ap.add_argument('--last', type=float, default=50e6, help='window of the final mean return')
  ap.add_argument('--bin_frames', type=float, default=10e6)
  args = ap.parse_args(argv)
  if not os.path.isdir(args.runs_dir):
    print(f'{args.runs_dir} does not exist yet: nothing to report')
    return
  runs = load_runs(args.runs_dir)
  if not runs:
    print(f'no runs with a config.json in {args.runs_dir}: nothing to report')
    return
  os.makedirs(args.out, exist_ok=True)
  labels = all_labels(runs)
  gaps = gap_analysis(runs)
  fits = {r['name']: plateau_fit(*r['metrics'], fit_from=args.fit_from, last=args.last) for r in runs}
  fig_training_length(runs, gaps, fits, args.figures, bin_frames=args.bin_frames)

  L = ['# Extended runs', '',
       f'Runs in {args.runs_dir}: {len(runs)} ({sum(r["final_frames"] is not None for r in runs)} '
       f'with a final checkpoint). Milestones with {MILESTONE_EPISODES}-episode evaluations: '
       + (', '.join(labels) if labels else 'none yet') + '.', '',
       '## (a) Intact score at each milestone (200 paired episodes, stochastic policy)', '',
       '| run | target (M) | frames so far (M) | ' + ' | '.join(labels) + ' |',
       '|---|---|---|' + '---|' * len(labels)]
  for r in runs:
    cells = []
    for lab in labels:
      v = score(r, lab)
      cells.append('-' if v is None else f'{v:.2f}' + (' *' if lab == r['final_label'] else ''))
    L.append(f'| {r["name"]} | {r["target"] / 1e6:.0f} | {r["frames"] / 1e6:.0f} | ' + ' | '.join(cells) + ' |')
  L += ['', '\\* canonical final evaluation of the run (eval/<probe>.json)', '',
        '### Reliance at each milestone (intact - no_code)', '',
        '| run | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
  for r in runs:
    cells = []
    for lab in labels:
      a, b = score(r, lab), score(r, lab, 'no_code')
      cells.append('-' if a is None or b is None else f'{a - b:.2f}')
    L.append(f'| {r["name"]} | ' + ' | '.join(cells) + ' |')

  L += ['', '## (b) Paired gap pgc33 - gc100 per arm (within seed)', '',
        '| arm | milestone | pairs | gc100 | pgc33 | gap +/- SEM | bootstrap 95% CI | paired t p | Wilcoxon p |',
        '|---|---|---|---|---|---|---|---|---|']
  for arm, g in gaps.items():
    for lab in g['labels']:
      st = g['per_label'][lab]['stats']
      L.append(f'| {arm} | {lab} | {st["n_pairs"]} | {st["mean_b"]:.2f} | {st["mean_a"]:.2f} | '
               + ' | '.join(fmt_stats(st)) + ' |')
  L += ['', '### Change of the gap from 300M to the latest common milestone', '',
        '| arm | from | to | pairs | gap at from | gap at to | change +/- SEM | bootstrap 95% CI | paired t p | Wilcoxon p |',
        '|---|---|---|---|---|---|---|---|---|---|']
  for arm, g in gaps.items():
    if g['change'] is None:
      base_gap = np.mean([v['gap'] for v in g['per_label'][g['base']]['pairs'].values()])
      L.append(f'| {arm} | {g["base"]} | - | {len(g["per_label"][g["base"]]["pairs"])} | {base_gap:+.2f} '
               '| - | no later milestone with >= 2 pairs yet | - | - | - |')
      continue
    st = g['change']
    L.append(f'| {arm} | {g["base"]} | {g["latest"]} | {st["n_pairs"]} | {st["mean_b"]:+.2f} | '
             f'{st["mean_a"]:+.2f} | ' + ' | '.join(fmt_stats(st)) + ' |')
  L += ['', '### Per seed', '', '| arm | seed | milestone | gc100 | pgc33 | gap |', '|---|---|---|---|---|---|']
  for arm, g in gaps.items():
    for lab in g['labels']:
      for s, v in sorted(g['per_label'][lab]['pairs'].items()):
        L.append(f'| {arm} | {s} | {lab} | {v["gc100"]:.2f} | {v["pgc33"]:.2f} | {v["gap"]:+.2f} |')

  L += ['', f'## (c) Training plateau (fit a - b exp(-frames/tau) to avg_return_100, frames >= {args.fit_from / 1e6:.0f}M)', '',
        '| run | target (M) | frames so far (M) | points | fit | asymptote a | tau (M) | 95% of a reached at (M) | fit at 300M (fraction of a) | fit at current (fraction of a) | mean return, last 50M | slope, last 100M (score / 100M) |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
  for r in runs:
    ft = fits[r['name']]
    if 'fit_error' in ft:
      flag = f'failed ({ft["fit_error"]})'
    else:
      flag = 'saturating' if ft['reliable'] else 'not saturating in the data'
    L.append(f'| {r["name"]} | {r["target"] / 1e6:.0f} | {ft["frames_max"] / 1e6:.0f} | {ft["n_points"]} | {flag} | '
             f'{fmt(ft["asymptote"])} | {fmt(ft["tau_M"], 0)} | {fmt(ft["frames_95pct_M"], 0)} | '
             f'{fmt(ft["fit_at_300M"])} ({fmt(ft["frac_at_300M"], 3)}) | '
             f'{fmt(ft["fit_at_max"])} ({fmt(ft["frac_at_max"], 3)}) | {fmt(ft["last_mean"])} | '
             f'{fmt(ft["slope_last_100M"])} |')
  L += ['', 'A fit counts as saturating when tau is below its bound and the asymptote is at most '
        '1.5 x the highest observed return; otherwise the 95 % frame count is an extrapolation '
        'from a curve that is still rising (see the slope column).']
  L += ['', 'Figure: fig11_training_length.png / .svg in ' + args.figures]
  with open(os.path.join(args.out, 'EXTENSION.md'), 'w') as f:
    f.write('\n'.join(L) + '\n')
  summary = dict(
      runs_dir=args.runs_dir, milestones=labels,
      runs={r['name']: dict(cond=r['cond'], arm=r['arm'], seed=r['seed'], target=r['target'],
                            frames=r['frames'], final_frames=r['final_frames'],
                            final_label=r['final_label'],
                            milestones={lab: {p: ev['mean_score'] for p, ev in ms.items()}
                                        for lab, ms in r['milestones'].items()},
                            plateau=fits[r['name']]) for r in runs},
      gaps=gaps)
  with open(os.path.join(args.out, 'extension.json'), 'w') as f:
    json.dump(summary, f, indent=1, default=float)
  print('\n'.join(L))


if __name__ == '__main__':
  main()
