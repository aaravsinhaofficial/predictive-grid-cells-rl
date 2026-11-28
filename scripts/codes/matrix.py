"""Run matrix of the code-variation experiment (plan: results/code_variations/PLAN.md).

Every code is the 512-cell population of its seed (same phases, orientations
and, for the codes with predictive cells, the same 169 predictive cells) with
one thing changed. Names:

  shift family (33 % predictive grid cells; only the look-aheads differ)
    gc100    grid cells only
    u20      every predictive cell looks 20 cm ahead (the code used so far)
    prop10   10 % of the module spacing: 4.0 / 5.7 / 8.1 / 11.5 cm
    prop27   27 % of the module spacing: 10.8 / 15.3 / 21.8 / 30.9 cm
    gard     per cell: a shift drawn from the 108 recorded shifts of the lab's
             reanalysis of the Gardner et al. (2022) recordings, in centimetres
             as recorded (mean 8.3 cm, median 6 cm)
    mec      per cell: a fraction of the module spacing drawn from the
             distribution of Ouchi & Fujisawa (2024): normal, mean 0.27,
             SD 0.11, kept between 0.01 and 0.5
    rand     per cell: uniform between 0 and 40 cm, whatever the module
    two      half of each module's predictive cells 10 cm, half 30 cm
    side     20 cm to the left of the direction of travel
    lr30     20 cm, 30 degrees left of ahead for half of the predictive cells
             and 30 degrees right for the other half
    p100     all 512 cells look 20 cm ahead (reference for "more predictive cells")
  lattice family (the pattern of every cell is changed)
    gcvar    grid only; per-cell spacing SD 6 % and orientation SD 1 degree
             within each module (the values Redman et al. 2025 used to model
             a recorded module)
    gcvar3   the same with three times the SDs (18 %, 3 degrees)
    gcshear  grid only; each module's lattice sheared by 0.13 along a wall
             axis (the shear of Stensola et al. 2015; ellipticity 1.14)
    gcjit    grid only; every firing field moved by about 6 % of the spacing
    mecreal  the mec code with the three realistic changes together
    m1       grid only, ONE module (512 cells, spacing 81 cm)
    m1var    m1 with the gcvar spread (the case where Redman et al. predict
             that variability helps)
  none       no spatial code (floor)
  trainable look-aheads (one forward distance per module for the predictive
  cells, trained with the policy)
    lzero    start at 0 cm          lu20    start at 20 cm
    lzeroX / lu20X   the same with the sign of the look-ahead gradient
                     scrambled (how far noise alone moves them)

Blocks (one cloud fleet each):
  oc    open room, core codes, 600 M frames, 8 seeds
  os    open room, screen codes, 300 M frames, 8 seeds
  dr    door + obstacle room (layout DO1), 600 M frames, 8 seeds; no code: 3 seeds, 300 M
  tr    trainable look-aheads, open and door room, 600 M frames, 4 seeds (+ 4 controls)
  lstm  LSTM + first-person vision from scratch on the mixture of 21 changed
        rooms, 500 M frames, 8 seeds
  trl   trainable look-aheads in the LSTM agent on the same mixture, 500 M
        frames, 4 seeds (+ 2 controls)
Agent arm 'ffb' (all blocks but lstm): feed-forward policy whose image shows the
heading only, so the code is its only source of position.

  python -m scripts.codes.matrix oc --out runs_codes/oc/matrix.json
"""
import argparse
import json
import os

MEC = ['--pgc_shift_dist', 'tnorm', '--pgc_shift_dist_args', '0.27,0.11,0.01,0.5']
VAR = ['--spacing_sd', '0.06', '--orient_sd_deg', '1.0']
VAR3 = ['--spacing_sd', '0.18', '--orient_sd_deg', '3.0']
SHEAR = ['--lattice_shear', '0.13']
JIT = ['--field_jitter', '0.06']
ONE = ['--n_modules', '1', '--spacing_min', '0.81']
# name: (cond preset, extra arguments)
CODES = {
    'gc100': ('gc100', []),
    'u20': ('pgc33', []),
    'prop10': ('pgc33', ['--pgc_shift_rule', 'spacing', '--pgc_shift_frac', '0.10']),
    'prop27': ('pgc33', ['--pgc_shift_rule', 'spacing', '--pgc_shift_frac', '0.27']),
    'mec': ('pgc33', MEC),
    'gard': ('pgc33', ['--pgc_shift_dist', 'gardner']),
    'rand': ('pgc33', ['--pgc_shift_dist', 'unif', '--pgc_shift_dist_args', '0.0,0.40']),
    'side': ('pgc33', ['--pgc_shift_angle', '90']),
    'two': ('pgc33', ['--pgc_shift_dist', 'set', '--pgc_shift_dist_args', '0.10,0.30']),
    'lr30': ('pgc33', ['--pgc_shift_angle', '30', '--pgc_shift_angle_alt', '1']),
    'p100': ('pgc100', []),
    'gcvar': ('gc100', VAR),
    'gcvar3': ('gc100', VAR3),
    'gcshear': ('gc100', SHEAR),
    'gcjit': ('gc100', JIT),
    'mecreal': ('pgc33', MEC + VAR + SHEAR + JIT),
    'm1': ('gc100', ONE),
    'm1var': ('gc100', ONE + VAR),
    'none': ('none', []),
    'lzero': ('pgc33', ['--pgc_shift', '0.0', '--learn_shift', '2', '--shift_lr', '1e-4']),
    'lu20': ('pgc33', ['--learn_shift', '2', '--shift_lr', '1e-4']),
    'lzeroX': ('pgc33', ['--pgc_shift', '0.0', '--learn_shift', '2', '--shift_lr', '1e-4', '--shift_scramble', '1']),
    'lu20X': ('pgc33', ['--learn_shift', '2', '--shift_lr', '1e-4', '--shift_scramble', '1']),
}
OPEN_CORE = ('gc100', 'u20', 'prop10', 'prop27', 'mec', 'gard', 'rand', 'side')
OPEN_SCREEN = ('two', 'lr30', 'p100', 'gcvar', 'gcvar3', 'gcshear', 'gcjit', 'mecreal', 'm1', 'm1var')
DOOR_CORE = ('gc100', 'u20', 'prop10', 'mec', 'gard', 'rand')
LSTM_CODES = ('gc100', 'u20', 'prop10', 'mec')
DOOR_LAYOUT = 'DO1'
LABEL = {
    'gc100': 'grid only', 'u20': '20 cm ahead', 'prop10': '10 % of spacing',
    'prop27': '27 % of spacing', 'gard': 'lab recordings (median 6 cm)',
    'mec': 'Ouchi & Fujisawa (mean 27 %)', 'rand': 'random 0 to 40 cm', 'two': '10 and 30 cm',
    'side': '20 cm sideways', 'lr30': '20 cm, 30° left or right', 'p100': 'all cells 20 cm ahead',
    'gcvar': 'spacing and orientation spread', 'gcvar3': 'spread × 3', 'gcshear': 'sheared lattice',
    'gcjit': 'jittered fields', 'mecreal': 'Ouchi & Fujisawa + all three',
    'm1': 'one module', 'm1var': 'one module + spread', 'none': 'no code',
}


def run(code, env, seed, frames, arm='ffb'):
  cond, extra = CODES[code]
  ex = list(extra)
  if env == 'door':
    ex += ['--rooms', DOOR_LAYOUT]
  elif env == 'mix':
    ex += ['--rooms', 'changed']
  return dict(name=f'{code}_{env}_{arm}_s{seed}', code=code, env=env, cond=cond, arm=arm, seed=seed,
              frames=frames, prio=0, extra=ex)


def block(name):
  S8, M = range(1, 9), 1_000_000
  if name == 'oc':
    runs = [run(c, 'open', s, 600 * M) for s in S8 for c in OPEN_CORE]
  elif name == 'os':
    runs = [run(c, 'open', s, 300 * M) for s in S8 for c in OPEN_SCREEN]
  elif name == 'dr':
    runs = [run(c, 'door', s, 600 * M) for s in S8 for c in DOOR_CORE]
    runs += [run('none', 'door', s, 300 * M) for s in (1, 2, 3)]
  elif name == 'tr':
    runs = [run(c, e, s, 600 * M) for e in ('open', 'door') for s in (1, 2, 3, 4) for c in ('lzero', 'lu20')]
    runs += [run(c, e, 1, 600 * M) for e in ('open', 'door') for c in ('lzeroX', 'lu20X')]
  elif name == 'lstm':
    runs = [run(c, 'mix', s, 500 * M, arm='lstm') for s in S8 for c in LSTM_CODES]
  elif name == 'trl':
    runs = [run(c, 'mix', s, 500 * M, arm='lstm') for s in (1, 2, 3, 4) for c in ('lzero', 'lu20')]
    runs += [run(c, 'mix', 1, 500 * M, arm='lstm') for c in ('lzeroX', 'lu20X')]
  else:
    raise ValueError(name)
  return runs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('block', choices=['oc', 'os', 'dr', 'tr', 'lstm', 'trl'])
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  runs = block(args.block)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  json.dump(runs, open(args.out, 'w'), indent=1)
  print(f'{len(runs)} runs -> {args.out}')


if __name__ == '__main__':
  main()
