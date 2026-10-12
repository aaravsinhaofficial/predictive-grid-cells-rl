"""Pick the best PGC percentage from the concentration series (quarter-spacing code,
code-only arm): 10 / 25 / 30 / 33 / 35 / 40 %. Writes results/best_concentration.json
and prints the best percentage.

  python scripts/pick_concentration.py --runs_dir runs --out results/best_concentration.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

SERIES = {10: 'pgc10', 25: 'pgc25', 30: 'pgc30', 33: 'pgc33phase27', 35: 'pgc35', 40: 'pgc40'}


def finals(runs_dir, cond):
  out = {}
  for d in glob.glob(os.path.join(runs_dir, f'{cond}_ffb_s*')):
    p = os.path.join(d, 'eval', 'intact.json')
    if os.path.exists(p):
      v = json.load(open(p))
      if v.get('canonical') or v.get('episodes', 0) >= 200:
        out[int(d.rsplit('_s', 1)[1])] = v['mean_score']
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs_dir', default='runs')
  ap.add_argument('--out', default='results/best_concentration.json')
  ap.add_argument('--min_seeds', type=int, default=6)
  args = ap.parse_args()
  data = {pct: finals(args.runs_dir, cond) for pct, cond in SERIES.items()}
  ref = data[33]
  rows = {}
  for pct, sc in data.items():
    if len(sc) < args.min_seeds:
      continue
    vals = np.array(list(sc.values()))
    common = sorted(set(sc) & set(ref))
    d = np.array([sc[s] - ref[s] for s in common])
    p = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue) if len(d) >= 3 and np.any(d != 0) else None
    rows[pct] = dict(cond=SERIES[pct], n=int(len(vals)), mean=float(vals.mean()),
                     sem=float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                     paired_vs_33=dict(n=int(len(d)), mean=float(d.mean()) if len(d) else 0.0,
                                       sem=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                                       p=p, positive=int((d > 0).sum()), negative=int((d < 0).sum())))
  best = max(rows, key=lambda k: rows[k]['mean'])
  res = dict(best_pct=best, rerun=bool(best != 33), series=rows)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  json.dump(res, open(args.out, 'w'), indent=1)
  for pct in sorted(rows):
    r = rows[pct]
    pv = r['paired_vs_33']
    print(f"{pct:3d}%  n={r['n']}  mean {r['mean']:.2f} ± {r['sem']:.2f}   vs 33%: {pv['mean']:+.2f} ± {pv['sem']:.2f}"
          f"  p={pv['p'] if pv['p'] is None else round(pv['p'], 3)}  (+{pv['positive']}/-{pv['negative']})")
  print(f'best: {best}%  rerun: {res["rerun"]}')
  print(best)


if __name__ == '__main__':
  main()
