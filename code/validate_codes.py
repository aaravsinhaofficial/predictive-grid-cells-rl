"""Validate the idealised codes against the predictive-grid-cell pipeline's own
definition: on agent trajectories in the RL arena, a grid cell's gridness
must peak at zero shift and a PGC's gridness must peak at a FORWARD shift of
about the configured displacement, in both the time-lag and the
heading-projected spatial-shift analyses (scores.py from
~/predictive-grid-cell-analysis, the paper's GridScorer, unchanged).

Caveats built into the summary: with 20 rate-map bins over 2.5 m (12.5 cm
bins) the gridness-vs-shift curve of an ideal cell is flat within ~ +/-8 cm of
its true shift, so per-unit argmax positions wander over that plateau; the
summary therefore also reports the argmax of the class-mean curve and each
unit's plateau centre (mean of shifts within 2 % of its maximum). Time lags
are converted to cm with the RUNNING speed (steps faster than v_min, no
teleports), not the all-step mean, which is dominated by turning-in-place.

Outputs <out>/validation.json and <out>/validation.png.

  python -m code.validate_codes --cond pgc33 --seed 1 --out runs/validation
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/ec2-user/predictive-grid-cell-analysis/code')
from code.common import (add_shared_args, build_env, build_population,  # noqa: E402
                            resolve_cond)
import scores as pgc_scores  # noqa: E402  (paper pipeline's GridScorer)


def make_scorer(half, nbins=20):
  starts = [0.2] * 10
  ends = np.linspace(0.4, 1.0, num=10)
  return pgc_scores.GridScorer(nbins, ((-half, half), (-half, half)),
                               list(zip(starts, ends.tolist())))


def random_walk_actions(n, T, gen, device, persist=0.85):
  """Random actions with persistence (geometric run lengths), biased to
  forward motion so the arena is covered: a crude rodent-like walk."""
  probs = torch.tensor([0.15, 0.15, 0.5, 0.05, 0.075, 0.075], device=device)
  a = torch.multinomial(probs.expand(n, -1), 1, generator=gen).squeeze(1)
  out = []
  for _ in range(T):
    switch = torch.rand(n, generator=gen, device=device) > persist
    new = torch.multinomial(probs.expand(n, -1), 1, generator=gen).squeeze(1)
    a = torch.where(switch, new, a)
    out.append(a)
  return out


def gridness_vs_shift(scorer, xs, ys, act, shifts, mode, projection='heading'):
  """act [T,B] for one unit -> gridness60 per shift (NaN where undefined)."""
  out = []
  for sh in shifts:
    fx, fy, fa = scorer.aligned_samples(xs, ys, act, sh, shift_mode=mode,
                                        space_projection=projection)
    if fx.size == 0:
      out.append(np.nan)
      continue
    rm = scorer.calculate_ratemap(fx, fy, fa)
    rm = np.nan_to_num(rm, nan=float(np.nanmean(rm)))
    out.append(float(scorer.get_scores(rm)[0]))
  return np.array(out)


def plateau_centre(curve, axis, tol=0.02):
  m = np.nanmax(curve)
  sel = np.isfinite(curve) & (curve >= m * (1 - tol) if m > 0 else curve >= m)
  return float(np.mean(axis[sel]))


def main(argv=None):
  ap = argparse.ArgumentParser()
  add_shared_args(ap)
  ap.add_argument('--seed', type=int, default=1)
  ap.add_argument('--n_envs', type=int, default=48)
  ap.add_argument('--steps', type=int, default=1350)
  ap.add_argument('--units_per_class', type=int, default=32)
  ap.add_argument('--max_lag', type=int, default=25)
  ap.add_argument('--max_shift_cm', type=float, default=50.0)
  ap.add_argument('--shift_step_cm', type=float, default=2.0)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--out', required=True)
  args = ap.parse_args(argv)
  cfg = resolve_cond(vars(args))
  os.makedirs(cfg['out'], exist_ok=True)
  dev = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
  pop = build_population(cfg, dev)
  pop.noise = 0.0                      # analytic maps for validation
  cfg_env = dict(cfg, vision='none')   # rendering not needed
  env = build_env(cfg_env, cfg['n_envs'], seed=cfg['seed'] * 7 + 3, device=dev, stagger=False)
  gen = torch.Generator(device=dev)
  gen.manual_seed(cfg['seed'])
  rng = np.random.default_rng(cfg['seed'])
  gc_u = rng.choice(pop.gc_idx, min(cfg['units_per_class'], len(pop.gc_idx)), replace=False) \
      if len(pop.gc_idx) else np.zeros(0, int)
  pgc_u = rng.choice(pop.pgc_idx, min(cfg['units_per_class'], len(pop.pgc_idx)), replace=False) \
      if len(pop.pgc_idx) else np.zeros(0, int)
  classes = [('gc', gc_u), ('pgc', pgc_u)]
  units = np.concatenate([gc_u, pgc_u]).astype(int)
  cls_of = ['gc'] * len(gc_u) + ['pgc'] * len(pgc_u)

  # Roll out: positions [T,B], motion dirs, and the code of the chosen units.
  obs = env.reset_all()
  acts = random_walk_actions(cfg['n_envs'], cfg['steps'] - 1, gen, dev)
  xs, ys, A, hits = [], [], [], []
  with torch.no_grad():
    for t in range(cfg['steps'] - 1):
      g = pop.activity(obs['pos'], dir=obs['dir'], vel=obs['vel'])[:, units]
      xs.append(obs['pos'][:, 0].cpu().numpy())
      ys.append(obs['pos'][:, 1].cpu().numpy())
      A.append(g.cpu().numpy())
      obs, _, done, info = env.step(acts[t])
      hits.append(info['hit'].cpu().numpy())
      assert not bool(done.any()), 'rollout must stay within one episode'
  xs, ys, A = np.stack(xs), np.stack(ys), np.stack(A)           # [T,B], [T,B,U]
  hits = np.stack(hits)                                          # [T-1,B] teleport after step t
  step = np.sqrt(np.diff(xs, axis=0) ** 2 + np.diff(ys, axis=0) ** 2)   # [T-1,B]
  real = ~hits[:-1] if hits.shape[0] == step.shape[0] + 1 else ~hits[:step.shape[0]]
  cm_all = float(np.nanmean(step) * 100.0)
  running = (step > env.v_min * env.dt) & real
  cm_run = float(np.nanmean(step[running]) * 100.0) if running.any() else cm_all

  scorer = make_scorer(cfg['arena_half'])
  lags = np.arange(-cfg['max_lag'], cfg['max_lag'] + 1)
  cm_shifts = np.arange(-cfg['max_shift_cm'], cfg['max_shift_cm'] + 1e-9, cfg['shift_step_cm'])
  curves = {'time': {}, 'space': {}}
  for ci, u in enumerate(units):
    cls = cls_of[ci]
    curves['time'].setdefault(cls, []).append(
        gridness_vs_shift(scorer, xs, ys, A[:, :, ci], lags, 'time'))
    curves['space'].setdefault(cls, []).append(
        gridness_vs_shift(scorer, xs, ys, A[:, :, ci], cm_shifts, 'space', 'heading'))
  summary = {}
  for mode, axis_cm in (('time', lags * cm_run), ('space', cm_shifts)):
    for cls, cs in curves[mode].items():
      cs = np.array(cs)
      best = np.nanargmax(cs, axis=1)
      best_cm = axis_cm[best]
      mean_curve = np.nanmean(cs, 0)
      zero_i = int(np.argmin(np.abs(axis_cm)))
      summary[f'{mode}_{cls}'] = dict(
          n=int(len(cs)),
          gridness_at_zero_mean=float(np.nanmean(cs[:, zero_i])),
          gridness_best_mean=float(np.nanmean(cs[np.arange(len(cs)), best])),
          best_shift_cm_mean=float(np.mean(best_cm)),
          best_shift_cm_sd=float(np.std(best_cm)),
          best_shift_cm_sem=float(np.std(best_cm) / np.sqrt(len(best_cm))),
          best_shift_cm_all=[float(b) for b in best_cm],
          mean_curve_argmax_cm=float(axis_cm[int(np.nanargmax(mean_curve))]),
          plateau_centre_cm_mean=float(np.mean([plateau_centre(c, axis_cm) for c in cs])),
          plateau_centre_cm_sem=float(np.std([plateau_centre(c, axis_cm) for c in cs]) / np.sqrt(len(cs))),
          axis_cm=[float(v) for v in axis_cm],
          curve_mean=[float(v) for v in mean_curve],
          curve_sd=[float(v) for v in np.nanstd(cs, 0)])
  res = dict(cm_per_step_all=cm_all, cm_per_step_running=cm_run,
             lags_steps=lags.tolist(), cm_shifts=cm_shifts.tolist(),
             n_teleports=int(hits.sum()),
             units=dict(gc=gc_u.tolist(), pgc=pgc_u.tolist()), summary=summary,
             population=pop.describe(), bin_cm=100.0 * 2 * cfg['arena_half'] / 20)
  with open(os.path.join(cfg['out'], 'validation.json'), 'w') as f:
    json.dump(res, f, indent=1)

  # Figure.
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  half = cfg['arena_half']
  maps = pop.ratemaps(res=60, half=half, dir=(1.0, 0.0))
  expected_cm = (cfg['pgc_shift'] * 100.0 if cfg['pgc_shift_mode'] == 'space'
                 else cfg['pgc_tau'] * cm_run * 15.0)   # tau * running speed (cm/s)
  fig, axes = plt.subplots(3, 4, figsize=(15, 11))
  for ax in axes.ravel():
    ax.set_visible(False)
  col = 0
  for cls, uu in classes:
    for k in range(min(2, len(uu))):
      ax = axes[0, col]
      ax.set_visible(True)
      ax.imshow(maps[uu[k]].T, origin='lower', extent=[-half, half, -half, half], cmap='jet')
      ax.set_title(f'{cls} unit {int(uu[k])} (module {pop.module[uu[k]]})'
                   + (', travel +x' if cls == 'pgc' else ''))
      col += 1
  col = 0
  for cls, uu in classes:
    if not len(uu):
      continue
    ci = cls_of.index(cls)
    for sh in (0.0, expected_cm):
      fx, fy, fa = scorer.aligned_samples(xs, ys, A[:, :, ci], sh, shift_mode='space',
                                          space_projection='heading')
      rm = np.nan_to_num(scorer.calculate_ratemap(fx, fy, fa))
      ax = axes[1, col]
      ax.set_visible(True)
      ax.imshow(rm.T, origin='lower', extent=[-half, half, -half, half], cmap='jet')
      ax.set_title(f'{cls} unit {int(units[ci])}: shift {sh:+.0f} cm  g60={scorer.get_scores(rm)[0]:.2f}')
      col += 1
  for m, (mode, xl) in enumerate((('time', f'time lag (cm at running speed {cm_run:.2f} cm/step)'),
                                  ('space', 'spatial shift along heading (cm)'))):
    ax = axes[2, 2 * m]
    ax.set_visible(True)
    for cls, c in (('gc', 'tab:blue'), ('pgc', 'tab:red')):
      key = f'{mode}_{cls}'
      if key not in summary:
        continue
      s = summary[key]
      axis_cm, mu, sd = np.array(s['axis_cm']), np.array(s['curve_mean']), np.array(s['curve_sd'])
      ax.plot(axis_cm, mu, color=c, label=f'{cls} (n={s["n"]})')
      ax.fill_between(axis_cm, mu - sd, mu + sd, color=c, alpha=0.2)
    ax.axvline(0, color='k', lw=0.5)
    ax.axvline(expected_cm, color='tab:red', lw=0.5, ls='--')
    ax.set_xlabel(xl)
    ax.set_ylabel('gridness (60 deg)')
    ax.set_title(f'{mode} mode: gridness vs shift')
    ax.legend()
    ax = axes[2, 2 * m + 1]
    ax.set_visible(True)
    for cls, c in (('gc', 'tab:blue'), ('pgc', 'tab:red')):
      key = f'{mode}_{cls}'
      if key in summary:
        ax.hist(summary[key]['best_shift_cm_all'], bins=15, color=c, alpha=0.5, label=cls)
    ax.axvline(expected_cm, color='tab:red', lw=0.5, ls='--')
    ax.set_xlabel('preferred shift (cm)')
    ax.set_title(f'{mode} mode: per-unit argmax (plateau +/- {res["bin_cm"]/2:.0f} cm)')
    ax.legend()
  fig.suptitle(f'{cfg["cond"]} seed {cfg["seed"]}: {pop.n_pgc} PGC / {pop.n_cells} cells, '
               f'expected shift {expected_cm:.0f} cm ({cfg["pgc_shift_mode"]}); '
               f'running {cm_run:.2f} cm/step, all-step {cm_all:.2f} cm/step', fontsize=11)
  fig.tight_layout()
  fig.savefig(os.path.join(cfg['out'], 'validation.png'), dpi=110)
  print(json.dumps({k: {kk: vv for kk, vv in v.items()
                        if kk not in ('curve_mean', 'curve_sd', 'axis_cm', 'best_shift_cm_all')}
                    for k, v in summary.items()}, indent=1))
  print(f'cm_per_step running={cm_run:.3f} all={cm_all:.3f}; teleports={int(hits.sum())}')


if __name__ == '__main__':
  main()
