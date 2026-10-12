"""Markdown report (table + figures) for the PGC-arrangement experiment at P % PGCs.

  python scripts/arrangement_report.py --pct 40 --runs_dir runs --out docs/arrangements_40pct.md

Numbers are within-seed paired differences in the code-only arm; p from the paired t-test.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, _argv = ['x'], sys.argv
import make_figures as mf  # noqa: E402
sys.argv = _argv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.common import build_population  # noqa: E402


def paired(runs, cond, ref, fn):
  a, b = mf.paired_vals(runs, cond, ref, 'ffb', fn)
  d = a - b
  if not len(d):
    return None
  p = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue) if len(d) >= 3 and np.any(d != 0) else float('nan')
  return dict(mean=float(d.mean()), sem=float(mf.sem(d)), p=p, pos=int((d > 0).sum()), neg=int((d < 0).sum()), n=len(d))


def fmt(d, digits=1, unit=''):
  if d is None:
    return '—'
  star = '**' if d['p'] < 0.05 else ''
  ptxt = 'p < 0.001' if d['p'] < 0.0005 else f"p = {d['p']:.3f}"
  return f"{star}{d['mean']:+.{digits}f} ± {d['sem']:.{digits}f}{unit}, {ptxt}{star}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--pct', type=int, required=True)
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default=None)
  ap.add_argument('--fig_dir', default='figures')
  args = ap.parse_args()
  pct = args.pct
  out = args.out or f'docs/arrangements_{pct}pct.md'
  runs = mf.load_runs(args.runs_dir)
  arm = 'ffb'
  uni = mf.arr_cond('unif', pct)
  sfx = mf.arr_suffix(pct)
  rel = os.path.relpath(args.fig_dir, os.path.dirname(out) or '.')
  L = []
  L.append(f'# Arranging the predictive cells at {pct} % PGCs\n')
  bc = 'results/best_concentration.json'
  if os.path.exists(bc):
    b = json.load(open(bc))
    row = b['series'].get(str(pct)) or b['series'].get(pct)
    if row:
      pv = row['paired_vs_33']
      L.append(f"Rerun of the arrangement experiment at the best percentage of the concentration series "
               f"({pct} %: {row['mean']:.1f} ± {row['sem']:.1f} on the quarter-spacing code, "
               f"{pv['mean']:+.1f} ± {pv['sem']:.1f} vs 33 %, p = {pv['p'] if pv['p'] is None else round(pv['p'], 3)}, "
               f"{pv['positive']}/{pv['n']} seeds better). Same design as the 33 % experiment "
               f"(`docs/arrangements.md`): code-only arm, 8 seeds paired with the uniform code at {pct} % and with "
               f"the grid-only code of the same seed, 300 M frames, final evaluation on 200 identical episodes. "
               f"The quarter-spacing code at {pct} % is the run from the concentration series.\n")
  L.append(f'![arrangements]({rel}/fig7_arrangements{sfx}.png)\n')
  L.append('| arrangement | PGCs per module | look-ahead per module (cm) | final score | Δ vs uniform | Δ vs grid-only | '
           'frames to score 25, vs uniform | PGC-specific lesion cost | path efficiency Δ vs uniform |')
  L.append('|---|---|---|---|---|---|---|---|---|')
  g25 = np.array([mf.frames_to(runs, 'gc100', arm, s) for s in mf.seeds_of(runs, 'gc100', arm)], float)
  u25 = np.array([mf.frames_to(runs, uni, arm, s) for s in mf.seeds_of(runs, uni, arm)], float)
  sections = []
  for cond, role, label in mf.arr_set(pct):
    seeds = mf.seeds_of(runs, cond, arm)
    if not seeds:
      continue
    cfg = runs[(cond, arm, seeds[0])]['cfg']
    pop = build_population(cfg, 'cpu')
    is_pgc, module, shifts = pop._np['is_pgc'], pop.module, pop._np['shifts']
    per = [int((is_pgc & (module == m)).sum()) for m in range(pop.n_modules)]
    sh = [100 * float(shifts[is_pgc & (module == m)].mean()) if per[m] else float('nan') for m in range(pop.n_modules)]
    fin = np.array([mf.score(runs, cond, arm, s) for s in seeds], float)
    fin = fin[np.isfinite(fin)]
    du = paired(runs, cond, uni, mf.score) if cond != uni else None
    dg = paired(runs, cond, 'gc100', mf.score)
    d25 = paired(runs, cond, uni, mf.frames_to) if cond != uni else None
    rl = np.array([mf.pgc_reliance(runs, cond, arm, s) for s in seeds], float)
    rl = rl[np.isfinite(rl)]
    dpe = paired(runs, cond, uni, lambda r, c, a, s: mf.score(r, c, a, s, key='mean_path_eff')) if cond != uni else None
    per_s = ' / '.join(str(x) for x in per)
    sh_s = ' / '.join('–' if np.isnan(x) else f'{x:.0f}' for x in sh)
    L.append(f"| {label} | {per_s} | {sh_s} | {fin.mean():.1f} (sd {fin.std(ddof=1):.1f}, {fin.min():.1f}–{fin.max():.1f}) | "
             f"{fmt(du) if du else '—'} | {fmt(dg)} | {fmt(d25, 0, ' M') if d25 else '—'} | "
             f"{rl.mean():.1f} ± {mf.sem(rl):.1f} | {fmt(dpe, 3) if dpe else '—'} |")
    sections.append((cond, label, per_s, sh_s, fin, du, dg, d25, rl, dpe, len(seeds)))
  L.append('')
  if len(g25) and len(u25):
    L.append(f'(grid-only agents reach a training score of 25 after {np.nanmean(g25):.0f} M frames on average, '
             f'the uniform {pct} % code after {np.nanmean(u25):.0f} M.)\n')
  L.append('## The arrangements one by one\n')
  L.append('Each figure has the same six panels as in `docs/arrangements.md`: **(A)** cells per module, '
           '**(B)** look-ahead per predictive cell (grey tick = half the module spacing), **(C)** training score vs '
           'uniform and grid-only, **(D)** final score per seed paired with the reference, **(E)** path efficiency, '
           '**(F)** score after silencing cells at test time (hollow circles = reference).\n')
  for cond, label, per_s, sh_s, fin, du, dg, d25, rl, dpe, n in sections:
    L.append(f'### {label}\n')
    L.append(f'![{label}]({rel}/arrangements{sfx}/{cond}.png)\n')
    L.append(f'PGCs per module {per_s}; look-ahead {sh_s} cm. {n} seeds. Final score {fin.mean():.1f} ± {mf.sem(fin):.1f} '
             f'(seeds {fin.min():.1f}–{fin.max():.1f}); vs uniform {fmt(du) if du else "(reference)"}; vs grid-only {fmt(dg)}; '
             f'frames to a training score of 25 vs uniform {fmt(d25, 0, " M") if d25 else "—"}; '
             f'PGC-specific lesion cost {rl.mean():.1f} ± {mf.sem(rl):.1f}; path efficiency vs uniform {fmt(dpe, 3) if dpe else "—"}.\n')
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  open(out, 'w').write('\n'.join(L))
  print(f'wrote {out} ({len(sections)} arrangements)')


if __name__ == '__main__':
  main()
