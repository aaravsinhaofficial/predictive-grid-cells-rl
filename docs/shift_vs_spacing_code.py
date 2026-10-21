"""Predictive shift relative to grid spacing in the path-integration RNN cohort.

Companion to docs/shift_vs_spacing.md. Four steps:

  compute    per cohort seed: reload the checkpoint, regenerate the classifier's
             trajectories, and for every classified unit build the rate map at zero
             lag and at the unit's stored preferred lag (time mode, as in the
             classifier) with 5 cm bins; the spatial autocorrelogram (SAC) comes from
             the pipeline's own GridScorer. Two spacing estimates per SAC:
               peaks  = mean distance of the six innermost local maxima outside the
                        central field (centre-of-mass sub-bin refinement; distances
                        more than 30 % off their median dropped)
               radial = first maximum of the radially averaged SAC after its first
                        trough (parabolic sub-bin interpolation)
             -> <out_dir>/seed_<s>.npz
  aggregate  pool the seeds. A spacing estimate counts as valid when both estimates
             exist, agree within 25 % and at least four ring peaks were found (plus a
             gridness floor of 0.15 for the zero-lag grid cells and for the maps at
             the preferred lag). The primary per-unit spacing of a predictive unit is
             measured at ZERO lag: maps shifted by the large preferred lags (> 35 cm,
             ~20 of 40 time steps) report a compressed lattice (about 0.8 of the
             zero-lag spacing), so the spacing at the preferred lag is kept only as a
             robustness check. -> <out_dir>/summary.json and summary_units.npz
  figure     figures/shift_vs_spacing.png and .svg
  selftest   recover known spacings from synthetic hexagonal maps

  python docs/shift_vs_spacing_code.py compute --seeds 0 1 2 --out_dir /tmp/svs
  python docs/shift_vs_spacing_code.py aggregate --out_dir /tmp/svs
  python docs/shift_vs_spacing_code.py figure --out_dir /tmp/svs --fig_dir figures

`compute` and `selftest` need the analysis repository (--analysis_root, default
/home/ec2-user/predictive-grid-cell-analysis) and must run with its virtualenv
(torch + its code/ modules); nothing is written into that repository.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage, stats

_G = {}
OUCHI = dict(shift_cm=22.5, shift_sd=10.2, ratio=0.27, ratio_sd=0.11, n=305,
             spacing_median=84.0, spacing_q25=70.0, spacing_q75=96.0)  # Fig. 3E-G
ARENA_SPACINGS = [40.0, 57.0, 81.0, 115.0]        # our four modules (cm)
ARENA_FRAC, ARENA_UNIFORM = 0.27, 20.0            # quarter-spacing rule; uniform rule (cm)


# ------------------------------------------------------------------ spacing
def make_scorer(res, bw):
  from scores import GridScorer  # analysis repo
  ends = np.linspace(0.4, 1.0, num=10)
  return GridScorer(res, ((-bw / 2, bw / 2), (-bw / 2, bw / 2)), zip([0.2] * 10, ends.tolist()))


def radial_profile(sac):
  S = sac.shape[0]
  c = (S - 1) / 2.0
  yy, xx = np.mgrid[0:S, 0:S]
  rbin = np.round(np.hypot(xx - c, yy - c)).astype(int)
  return np.asarray(ndimage.mean(sac, labels=rbin, index=np.arange(rbin.max() + 1)))


def first_trough(prof):
  i = 1
  while i < len(prof) - 1 and not (prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]):
    i += 1
  return i


def spacing_radial(prof):
  n = len(prof)
  i = first_trough(prof)
  if i >= n - 2:
    return np.nan
  j = i + 1
  while j < n - 1 and not (prof[j] >= prof[j - 1] and prof[j] >= prof[j + 1] and prof[j] > 0):
    j += 1
  if j >= n - 1:
    return np.nan
  a, b, c = prof[j - 1], prof[j], prof[j + 1]
  denom = a - 2 * b + c
  return float(j + (0.5 * (a - c) / denom if denom < 0 else 0.0))


def spacing_peaks(sac, prof, n_peaks=6):
  S = sac.shape[0]
  c = (S - 1) / 2.0
  r_excl = max(first_trough(prof), 2)
  mx = ndimage.maximum_filter(sac, size=3, mode='constant', cval=-np.inf)
  pk = np.argwhere((sac == mx) & (sac > 0.0))
  if pk.size == 0:
    return np.nan, 0
  d = np.hypot(pk[:, 0] - c, pk[:, 1] - c)
  pk, d = pk[d >= r_excl], d[d >= r_excl]
  if len(d) < 3:
    return np.nan, int(len(d))
  dd = []
  for iy, ix in pk[np.argsort(d)[:n_peaks]]:
    y0, y1, x0, x1 = max(iy - 1, 0), min(iy + 2, S), max(ix - 1, 0), min(ix + 2, S)
    w = np.clip(sac[y0:y1, x0:x1] - sac[y0:y1, x0:x1].min(), 0, None)
    if w.sum() <= 0:
      dd.append(np.hypot(iy - c, ix - c))
      continue
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dd.append(np.hypot((w * yy).sum() / w.sum() - c, (w * xx).sum() / w.sum() - c))
  dd = np.asarray(dd)
  ok = np.abs(dd - np.median(dd)) <= 0.3 * np.median(dd)
  return float(dd[ok].mean()), int(ok.sum())


def measure(sac, bin_m):
  prof = radial_profile(sac)
  sp_pk, npk = spacing_peaks(sac, prof)
  return sp_pk * bin_m, npk, spacing_radial(prof) * bin_m


def unit_job(u):
  sc, xs, ys, act = _G['sc'], _G['xs'], _G['ys'], _G['act']
  bin_m = _G['bw'] / _G['res']
  out = []
  for k in (0, _G['kbest'][u]):
    if not np.isfinite(k):
      out.append((np.nan, np.nan, 0, np.nan, 0))
      continue
    fx, fy, fa = sc.aligned_samples(xs, ys, act[:, :, u], int(k), shift_mode='time')
    rm = sc.calculate_ratemap(fx, fy, fa)
    s60, _, _, _, sac, _ = sc.get_scores(rm)
    sp_pk, npk, sp_rad = measure(sac, bin_m)
    out.append((float(s60), sp_pk, npk, sp_rad, int(fa.size)))
  return u, out


def selftest(res, bw=2.2, seed=0):
  rng = np.random.default_rng(seed)
  sc = make_scorer(res, bw)
  edges = np.linspace(-bw / 2, bw / 2, res + 1)
  cent = 0.5 * (edges[:-1] + edges[1:])
  X, Y = np.meshgrid(cent, cent, indexing='ij')
  print(f'selftest res={res} bin={bw / res * 100:.1f} cm')
  for lam in (0.40, 0.57, 0.81, 1.15):
    rec = []
    for _ in range(20):
      th, ph = rng.uniform(0, np.pi / 3), rng.uniform(0, lam, size=2)
      rm = np.zeros_like(X)
      for a in (0, np.pi / 3, 2 * np.pi / 3):
        kx, ky = np.cos(th + a), np.sin(th + a)
        rm += np.cos(2 * np.pi / lam * (kx * (X - ph[0]) + ky * (Y - ph[1])) * 2 / np.sqrt(3))
      rm = np.clip(rm, 0, None) ** 2
      rm += rng.normal(0, 0.5 * rm.std(), size=rm.shape)
      _, _, _, _, sac, _ = sc.get_scores(rm)
      rec.append(measure(sac, bw / res))
    rec = np.asarray(rec)
    print(f'  lambda={lam:.2f} m: peaks {np.nanmedian(rec[:, 0]):.3f} (sd {np.nanstd(rec[:, 0]):.3f}) '
          f'radial {np.nanmedian(rec[:, 2]):.3f} (sd {np.nanstd(rec[:, 2]):.3f})')


def compute(args):
  sys.path.insert(0, os.path.join(args.analysis_root, 'code'))
  import pgc_common as C  # noqa: E402  (analysis repo, read only)
  os.makedirs(args.out_dir, exist_ok=True)
  for seed in args.seeds:
    t0 = time.time()
    ckpt = glob.glob(f'{args.analysis_root}/Models/canonical_cohort_v1/seed_{seed}/*/most_recent_model.pth')[0]
    rig = glob.glob(f'{args.analysis_root}/analysis_outputs/canonical_cohort_v1/seed_{seed}/*/pgc_rigor')[0]
    cls, cov = np.load(f'{rig}/pgc_classification.npz'), np.load(f'{rig}/pgc_covariates.npz')
    Ng = int(len(cls['labels']))
    lm = C.load_model(ckpt, device=args.device)
    b = C.collect_bundle(lm, n_batches=args.n_batches, Ng_use=Ng, res=20,
                         collection_seed=1234, with_ratemaps=False)
    cmps = C.cm_per_step(b.xs, b.ys)
    step_cm = float(cls['lag_cm'][1] - cls['lag_cm'][0])
    kbest = np.round(cls['best_cm'] / step_cm)
    bw = float(lm.options.box_width)
    _G.update(sc=make_scorer(args.res, bw), xs=b.xs, ys=b.ys, act=b.activations,
              kbest=kbest, bw=bw, res=args.res)
    del lm
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as ex:
      res = dict(ex.map(unit_job, range(Ng), chunksize=4))
    A = np.array([[*res[u][0], *res[u][1]] for u in range(Ng)], dtype=float)
    out = os.path.join(args.out_dir, f'seed_{seed}.npz')
    np.savez_compressed(
        out, seed=seed, labels=cls['labels'], best_cm=cls['best_cm'],
        best_gridness20=cls['best_gridness'], spacing_cov_m=cov['spacing_m'],
        module_cov=cov['module'], is_grid_cov=cov['is_grid'],
        s60_zero=A[:, 0], spacing_zero_pk=A[:, 1], npk_zero=A[:, 2], spacing_zero_rad=A[:, 3],
        s60_best=A[:, 5], spacing_best_pk=A[:, 6], npk_best=A[:, 7], spacing_best_rad=A[:, 8],
        n_samples_zero=A[:, 4], n_samples_best=A[:, 9], kbest=kbest, cm_per_step=cmps,
        step_cm=step_cm, res=args.res, box_m=bw, n_batches=args.n_batches)
    pred = cls['labels'] == 1
    print(f'seed {seed}: Ng={Ng} cm/step={cmps:.3f} predictive n={pred.sum()} '
          f'median spacing at preferred shift {np.nanmedian(A[pred, 6]) * 100:.1f} cm '
          f'[{time.time() - t0:.0f}s] -> {out}', flush=True)


# ------------------------------------------------------------------ aggregate
def summ(x):
  x = np.asarray(x, float)
  x = x[np.isfinite(x)]
  return dict(n=int(x.size), median=float(np.median(x)), q25=float(np.percentile(x, 25)),
              q75=float(np.percentile(x, 75)), mean=float(x.mean()), sd=float(x.std(ddof=1)),
              cv=float(x.std(ddof=1) / x.mean()))


def fmt(s, nd=1):
  f = lambda v: f'{v:.{nd}f}'
  return (f"median {f(s['median'])} (IQR {f(s['q25'])}-{f(s['q75'])}), mean {f(s['mean'])} +/- "
          f"{f(s['sd'])} (CV {s['cv']:.2f}), n = {s['n']}")


def aggregate(args):
  files = sorted(glob.glob(f'{args.out_dir}/seed_*.npz'), key=lambda f: int(f.split('seed_')[-1][:-4]))
  cols = {}
  for f in files:
    d = np.load(f)
    n = len(d['labels'])
    rec = dict(seed=np.full(n, int(d['seed'])), label=d['labels'], shift=d['best_cm'],
               sp_cov=d['spacing_cov_m'] * 100, sp0=d['spacing_zero_pk'] * 100,
               sp0_rad=d['spacing_zero_rad'] * 100, g0=d['s60_zero'], npk0=d['npk_zero'],
               spb=d['spacing_best_pk'] * 100, spb_rad=d['spacing_best_rad'] * 100,
               gb=d['s60_best'], npkb=d['npk_best'])
    for k, v in rec.items():
      cols.setdefault(k, []).append(np.asarray(v, float))
  R = {k: np.concatenate(v) for k, v in cols.items()}
  seeds = np.unique(R['seed'])

  def valid(sp, sp_rad, npk, g=None):
    ok = (np.isfinite(sp) & np.isfinite(sp_rad) & (npk >= 4)
          & (np.abs(sp - sp_rad) <= args.agree * 0.5 * (sp + sp_rad)))
    return ok if g is None else ok & (g >= args.g_floor)

  def per_seed(mask, values, min_n=3):
    ps = [np.median(values[mask & (R['seed'] == s)]) for s in seeds if np.sum(mask & (R['seed'] == s)) >= min_n]
    return dict(mean=float(np.mean(ps)), sem=float(np.std(ps, ddof=1) / np.sqrt(len(ps))),
                min=float(np.min(ps)), max=float(np.max(ps)), n_seeds=len(ps))

  def terciles(sp, sh):
    t = np.percentile(sp, [33.3, 66.7])
    return [dict(tercile=nm, spacing_median=float(np.median(sp[m])), shift_median=float(np.median(sh[m])),
                 ratio_median=float(np.median(sh[m] / sp[m])), n=int(m.sum()))
            for nm, m in (('low', sp < t[0]), ('mid', (sp >= t[0]) & (sp < t[1])), ('high', sp >= t[1]))]

  out = dict(n_seeds=int(seeds.size), n_units=int(R['label'].size), g_floor=args.g_floor, agree=args.agree)
  nrm, pr, rt = R['label'] == 0, R['label'] == 1, R['label'] == 2
  # population spacing from the zero-lag grid units
  okn = nrm & valid(R['sp0'], R['sp0_rad'], R['npk0'], R['g0'])
  out['normal_spacing'] = summ(R['sp0'][okn])
  out['normal_spacing_stored'] = summ(R['sp_cov'][nrm])
  out['normal_spacing_per_seed_median'] = per_seed(okn, R['sp0'])
  # preferred shift of the predictive units
  out['n_predictive'] = int(pr.sum())
  out['predictive_shift'] = summ(R['shift'][pr])
  out['predictive_shift_per_seed_median'] = per_seed(pr, R['shift'])
  out['predictive_frac_le10'] = float(np.mean(R['shift'][pr] <= 10))
  out['predictive_frac_ge40'] = float(np.mean(R['shift'][pr] >= 40))
  out['predictive_zero_lag_gridness_median'] = float(np.nanmedian(R['g0'][pr]))
  out['predictive_frac_zero_lag_gridness_ge_floor'] = float(np.nanmean(R['g0'][pr] >= args.g_floor))
  # primary: shift / zero-lag spacing
  ok0 = pr & valid(R['sp0'], R['sp0_rad'], R['npk0'])
  ratio0 = R['shift'][ok0] / R['sp0'][ok0]
  out['n_predictive_spacing0'] = int(ok0.sum())
  out['predictive_spacing0'] = summ(R['sp0'][ok0])
  out['ratio0'] = summ(ratio0)
  out['ratio0_frac_in_ouchi_sd'] = float(np.mean((ratio0 >= 0.16) & (ratio0 <= 0.38)))
  out['ratio0_frac_gt_045'] = float(np.mean(ratio0 > 0.45))
  rho, pv = stats.spearmanr(R['shift'][ok0], R['sp0'][ok0])
  out['spearman0'] = dict(rho=float(rho), p=float(pv), n=int(ok0.sum()))
  out['ratio0_per_seed_median'] = per_seed(ok0, np.where(ok0, R['shift'] / np.where(ok0, R['sp0'], 1.0), np.nan))
  out['terciles0'] = terciles(R['sp0'][ok0], R['shift'][ok0])
  hist, edges = np.histogram(ratio0, bins=np.arange(0, 0.85, 0.05))
  out['ratio0_hist'] = dict(counts=hist.tolist(), edges=edges.tolist())
  # robustness: spacing measured at the preferred lag
  okb = pr & valid(R['spb'], R['spb_rad'], R['npkb'], R['gb'])
  out['n_predictive_spacing_best'] = int(okb.sum())
  out['predictive_spacing_best'] = summ(R['spb'][okb])
  out['ratio_best'] = summ(R['shift'][okb] / R['spb'][okb])
  rho_b, p_b = stats.spearmanr(R['shift'][okb], R['spb'][okb])
  out['spearman_best'] = dict(rho=float(rho_b), p=float(p_b), n=int(okb.sum()))
  both = okb & ok0
  out['spacing_best_over_zero_by_shift_band'] = []
  for lo, hi in ((5, 15), (15, 25), (25, 35), (35, 50)):
    m = both & (R['shift'] >= lo) & (R['shift'] < hi)
    q = R['spb'][m] / R['sp0'][m]
    out['spacing_best_over_zero_by_shift_band'].append(dict(
        band=f'{lo}-{hi} cm', n=int(m.sum()), median=float(np.median(q)), q25=float(np.percentile(q, 25)),
        q75=float(np.percentile(q, 75)), frac_lt_07=float(np.mean(q < 0.7))))
  out['ratio_stored_spacing'] = summ(R['shift'][pr] / R['sp_cov'][pr])
  # retrospective units, same measures (signed shift)
  okr = rt & valid(R['sp0'], R['sp0_rad'], R['npk0'])
  out['n_retrospective'] = int(rt.sum())
  out['retro_shift'] = summ(R['shift'][rt])
  out['retro_ratio0'] = summ(R['shift'][okr] / R['sp0'][okr])

  print(f"{out['n_seeds']} seeds, {out['n_units']} units; predictive {out['n_predictive']}, retrospective {out['n_retrospective']}")
  print('zero-lag grid units, spacing (cm):', fmt(out['normal_spacing']))
  print('  stored 11-cm-binned spacing:', fmt(out['normal_spacing_stored']))
  print('  per-seed medians: mean %.1f sem %.1f range %.1f-%.1f (%d seeds)' % tuple(out['normal_spacing_per_seed_median'].values()))
  print('predictive units, preferred shift (cm):', fmt(out['predictive_shift']))
  print('  per-seed median shift: mean %.1f sem %.1f range %.1f-%.1f (%d seeds)' % tuple(out['predictive_shift_per_seed_median'].values()))
  print('  fraction <= 10 cm %.2f, >= 40 cm %.2f; zero-lag gridness median %.2f, fraction >= floor %.2f'
        % (out['predictive_frac_le10'], out['predictive_frac_ge40'], out['predictive_zero_lag_gridness_median'],
           out['predictive_frac_zero_lag_gridness_ge_floor']))
  print(f"primary: zero-lag spacing valid for {out['n_predictive_spacing0']} / {out['n_predictive']} predictive units")
  print('  spacing (cm):', fmt(out['predictive_spacing0']))
  print('  ratio shift / spacing:', fmt(out['ratio0'], 3))
  print('  fraction within 0.27 +/- 0.11: %.2f; fraction > 0.45: %.2f' % (out['ratio0_frac_in_ouchi_sd'], out['ratio0_frac_gt_045']))
  print('  spearman(shift, spacing) rho = %.3f, p = %.2g, n = %d' % tuple(out['spearman0'].values()))
  print('  per-seed median ratio: mean %.3f sem %.3f range %.3f-%.3f (%d seeds)' % tuple(out['ratio0_per_seed_median'].values()))
  for r in out['terciles0']:
    print('  spacing tercile %-4s (median %.0f cm): shift median %.1f cm, ratio %.3f, n %d' % tuple(r.values()))
  print('  ratio histogram (0.05 bins from 0):', out['ratio0_hist']['counts'])
  print(f"robustness: spacing at the preferred lag valid for {out['n_predictive_spacing_best']} units")
  print('  spacing (cm):', fmt(out['predictive_spacing_best']))
  print('  ratio:', fmt(out['ratio_best'], 3), '; spearman rho = %.3f' % rho_b)
  for r in out['spacing_best_over_zero_by_shift_band']:
    print('  shift %-9s n %3d: spacing(preferred lag) / spacing(zero lag) median %.2f (IQR %.2f-%.2f), fraction < 0.7: %.2f'
          % tuple(r.values()))
  print('  ratio with the stored 11-cm-binned spacing:', fmt(out['ratio_stored_spacing'], 3))
  print('retrospective units: shift', fmt(out['retro_shift']), '; ratio', fmt(out['retro_ratio0'], 3))
  np.savez_compressed(f'{args.out_dir}/summary_units.npz', **R, ok_zero_pred=ok0, ok_best=okb, ok_zero_normal=okn)
  with open(f'{args.out_dir}/summary.json', 'w') as f:
    json.dump(out, f, indent=1)
  print('->', f'{args.out_dir}/summary.json')


# ------------------------------------------------------------------ figure
def figure(args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
  INK, INK2 = '#0b0b0b', '#52514e'
  plt.rcParams.update({
      'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
      'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
      'legend.frameon': False, 'savefig.dpi': 200, 'figure.facecolor': 'white'})
  U = np.load(f'{args.out_dir}/summary_units.npz')
  S = json.load(open(f'{args.out_dir}/summary.json'))
  ok = U['ok_zero_pred']
  sp, sh = U['sp0'][ok], U['shift'][ok]
  ratio = sh / sp
  fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2))

  x = np.linspace(30, 150, 2)
  ax.fill_between(x, (OUCHI['ratio'] - OUCHI['ratio_sd']) * x, (OUCHI['ratio'] + OUCHI['ratio_sd']) * x,
                  color=ORANGE, alpha=0.10, lw=0, zorder=0)
  ax.plot(x, OUCHI['ratio'] * x, color=ORANGE, lw=2, zorder=1)
  ax.text(147, OUCHI['ratio'] * 147 + 1.5, '0.27 x spacing\n(MEC mean, band = +/- SD)', color=INK2,
          fontsize=8, ha='right', va='bottom')
  ax.scatter(sp, sh, s=14, color=BLUE, alpha=0.45, edgecolor='none', zorder=2,
             label=f'RNN predictive units (n = {ok.sum()}, {S["n_seeds"]} seeds)')
  ax.errorbar(OUCHI['spacing_median'], OUCHI['shift_cm'],
              xerr=[[OUCHI['spacing_median'] - OUCHI['spacing_q25']], [OUCHI['spacing_q75'] - OUCHI['spacing_median']]],
              yerr=OUCHI['shift_sd'], fmt='o', color=ORANGE, ms=8, capsize=3, elinewidth=1.2,
              markeredgecolor='white', zorder=4, label='MEC predictive grid cells (n = 305)')
  ax.scatter(ARENA_SPACINGS, [ARENA_FRAC * s for s in ARENA_SPACINGS], s=52, marker='D', color=AQUA,
             edgecolor='white', lw=0.8, zorder=5, label='our arena, quarter-spacing rule')
  ax.scatter(ARENA_SPACINGS, [ARENA_UNIFORM] * 4, s=52, marker='D', facecolor='white', edgecolor=AQUA, lw=1.4,
             zorder=5, label='our arena, uniform 20 cm')
  ax.set_xlim(30, 150)
  ax.set_ylim(0, 64)
  ax.set_xlabel('grid spacing (cm; RNN units: at zero lag)')
  ax.set_ylabel('preferred forward shift (cm)')
  ax.set_title('A  shift against spacing', loc='left', color=INK, fontsize=11)
  ax.legend(loc='upper left', fontsize=8, handletextpad=0.4)

  edges = np.arange(0, 0.85, 0.05)
  bx.axvspan(OUCHI['ratio'] - OUCHI['ratio_sd'], OUCHI['ratio'] + OUCHI['ratio_sd'], color=ORANGE, alpha=0.10, lw=0, zorder=0)
  bx.hist(ratio, bins=edges, color=BLUE, edgecolor='white', lw=1.2, zorder=2)
  bx.axvline(OUCHI['ratio'], color=ORANGE, lw=2, zorder=3)
  ymax = bx.get_ylim()[1] * 1.12
  bx.set_ylim(0, ymax)
  bx.text(OUCHI['ratio'] + 0.012, ymax * 0.97, 'MEC 0.27 +/- 0.11\n= our quarter-spacing rule', color=INK2, fontsize=8, va='top')
  for s in ARENA_SPACINGS:
    bx.plot([ARENA_UNIFORM / s] * 2, [ymax * 0.80, ymax * 0.86], color=AQUA, lw=2, zorder=4)
  bx.text(ARENA_UNIFORM / ARENA_SPACINGS[0] + 0.012, ymax * 0.83, 'uniform 20 cm in our\nfour modules', color=INK2,
          fontsize=8, va='center')
  r = S['ratio0']
  bx.text(0.80, ymax * 0.60, f"RNN median {r['median']:.2f}\nIQR {r['q25']:.2f}-{r['q75']:.2f}",
          color=INK2, fontsize=8, ha='right', va='center')
  bx.set_xlim(0, 0.8)
  bx.set_xlabel('preferred shift / grid spacing')
  bx.set_ylabel('RNN predictive units')
  bx.set_title('B  shift as a fraction of spacing', loc='left', color=INK, fontsize=11)
  fig.tight_layout(w_pad=2.5)
  os.makedirs(args.fig_dir, exist_ok=True)
  fig.savefig(os.path.join(args.fig_dir, 'shift_vs_spacing.png'))
  fig.savefig(os.path.join(args.fig_dir, 'shift_vs_spacing.svg'))
  print('->', os.path.join(args.fig_dir, 'shift_vs_spacing.png'))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('step', choices=['compute', 'aggregate', 'figure', 'selftest'])
  p.add_argument('--out_dir', default='/tmp/shift_vs_spacing')
  p.add_argument('--analysis_root', default='/home/ec2-user/predictive-grid-cell-analysis')
  p.add_argument('--seeds', type=int, nargs='*', default=list(range(30)))
  p.add_argument('--res', type=int, default=44, help='rate-map bins over the 2.2 m box (44 = 5 cm)')
  p.add_argument('--n_batches', type=int, default=32)
  p.add_argument('--workers', type=int, default=12)
  p.add_argument('--device', default='cuda')
  p.add_argument('--g_floor', type=float, default=0.15)
  p.add_argument('--agree', type=float, default=0.25)
  p.add_argument('--fig_dir', default='figures')
  args = p.parse_args()
  if args.step in ('compute', 'selftest'):
    sys.path.insert(0, os.path.join(args.analysis_root, 'code'))
  {'compute': compute, 'aggregate': aggregate, 'figure': figure,
   'selftest': lambda a: selftest(a.res)}[args.step](args)


if __name__ == '__main__':
  main()
