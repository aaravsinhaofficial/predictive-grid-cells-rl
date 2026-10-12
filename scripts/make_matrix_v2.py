"""Matrices for the follow-up experiment (code-only arm = ff policy +
heading-only vision, where the code is load-bearing and PGCs helped).

  python scripts/make_matrix_v2.py arrangements --seeds 8 --out /tmp/waveA.json
  python scripts/make_matrix_v2.py concentrations --layout <best> --seeds 8 --out /tmp/waveB.json
  python scripts/make_matrix_v2.py bands --seeds 4 --out /tmp/waveBand.json

Every run: 33% PGC unless stated, 20 cm shift unless stated, 300M frames,
same population per seed as the main experiment (code_seed = seed), so each
variant is paired with gc100_ffb_s<seed> and pgc33_ffb_s<seed> from wave 2.
"""
import argparse
import json

ARM = 'ffb'

# name -> extra CLI args (cond defaults to pgc33 unless given)
ARRANGEMENTS = {
    'pgc33fine': ['--pgc_layout', 'fine'],                      # all PGCs in the smallest-spacing modules
    'pgc33coarse': ['--pgc_layout', 'coarse'],                  # all PGCs in the largest-spacing modules
    'pgc33middle': ['--pgc_layout', 'middle'],                  # PGCs in the two middle modules
    'pgc33phase': ['--pgc_shift_rule', 'spacing', '--pgc_shift_frac', '0.5'],   # shift = half the module spacing
    'pgc33jitter': ['--pgc_shift_jitter', '0.5'],               # heterogeneous shifts (lognormal, sigma 0.5)
}
BANDS = {
    'band': ['--cond', 'gc100', '--cell_type', 'band'],
    'pband33': ['--cond', 'pgc33', '--cell_type', 'band'],
    'pband100': ['--cond', 'pgc100', '--cell_type', 'band'],
}
CONCENTRATIONS = {'pgc10': 0.10, 'pgc25': 0.25, 'pgc30': 0.30, 'pgc35': 0.35, 'pgc40': 0.40}


def spec(name, seed, frames, extra, cond='pgc33', prio=0):
  return dict(name=f'{name}_{ARM}_s{seed}', cond=cond, arm=ARM, seed=seed, frames=frames,
              prio=prio, extra=list(extra))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('kind', choices=['arrangements', 'concentrations', 'bands'])
  ap.add_argument('--seeds', type=int, default=8)
  ap.add_argument('--frames', type=int, default=300_000_000)
  ap.add_argument('--layout', default='', help='concentrations: extra args of the winning arrangement, '
                                               'e.g. "--pgc_layout fine" (empty = uniform)')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  runs = []
  seeds = range(1, args.seeds + 1)
  if args.kind == 'arrangements':
    for name, extra in ARRANGEMENTS.items():
      for s in seeds:
        runs.append(spec(name, s, args.frames, extra))
  elif args.kind == 'concentrations':
    layout = args.layout.split()
    for name, frac in CONCENTRATIONS.items():
      for s in seeds:
        runs.append(spec(name, s, args.frames, ['--pgc_frac', str(frac)] + layout, cond='pgc33'))
  else:
    for name, extra in BANDS.items():
      for s in seeds:
        cond = extra[extra.index('--cond') + 1]
        runs.append(spec(name, s, args.frames, [a for a in extra if a not in ('--cond', cond)], cond=cond))
  json.dump(runs, open(args.out, 'w'), indent=1)
  print(f'{len(runs)} runs -> {args.out}')
  for r in runs[: len(runs) // max(args.seeds, 1) + 1]:
    print('  ', r['name'], r['cond'], ' '.join(r['extra']))


if __name__ == '__main__':
  main()
