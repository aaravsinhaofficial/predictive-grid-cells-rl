"""Shared builders so train / evaluate / validate instantiate identical objects
from one config dict (the run's config.json)."""
import argparse

import torch

from code.codes import GridPopulation
from code.env import NUM_ACTIONS, ArenaEnv
from code.nets import PolicyNet

CONDITIONS = {
    # name: (pgc_frac, code_mode)
    'gc100': (0.0, 'full'),      # 100 % grid cells
    'pgc33': (0.33, 'full'),     # 33 % predictive grid cells, 67 % grid cells
    'pgc50': (0.5, 'full'),      # 50 % predictive grid cells, 50 % grid cells (256 / 256:
                                 # equal class sizes for the class-matched lesion probes)
    'pgc100': (1.0, 'full'),     # 100 % predictive grid cells
    'gc67': (0.33, 'full'),      # the pgc33 population with its 169 PGC slots held at
                                 # zero (silence_pgc = 1), so only the 343 grid cells
                                 # carry signal. Same code_seed -> same phases,
                                 # orientations and PGC indices as pgc33 / gc100 of
                                 # that seed: controls for the smaller grid-cell count
                                 # of pgc33 rather than for the predictive cells
    'none': (0.0, 'none'),       # no spatial code at all (A3C-style baseline)
}
# Evaluation probes (see CodeSource). The six *_half<k> probes are appended after
# the original set so existing probe lists keep their order.
PROBES = ('intact', 'no_code', 'no_current', 'no_goal', 'no_pgc', 'no_gc', 'pgc_as_gc',
          'fake_goal', 'shuffled_current', 'sham0', 'sham1', 'sham2', 'sham3', 'sham4',
          'untrained', 'pgc_half0', 'pgc_half1', 'pgc_half2', 'gc_half0', 'gc_half1', 'gc_half2')
SHAM_FRAC = 0.33   # count-matched sham lesions remove round(SHAM_FRAC * n_cells) cells
# sham0..4 remove round(SHAM_FRAC * n_cells) = 169 random grid cells (module-
# stratified, one draw per index) in EVERY condition: the count-matched control
# for no_pgc in pgc33 (169 PGCs). For pgc50 (256 PGCs / 256 grid cells) the
# class-size-matched comparisons are no_pgc vs no_gc (256 vs 256) and
# pgc_half<k> vs gc_half<k> (floor(n_class / 2) cells each: 128 vs 128); in
# pgc33 the half probes remove 84 of 169 PGCs vs 171 of 343 grid cells.

# Keys that define the experiment a checkpoint belongs to; a --resume whose
# arguments differ in any of these from the stored config.json is refused.
EXPERIMENT_KEYS = (
    'cond', 'pgc_frac', 'code_mode', 'code_seed', 'n_cells', 'n_modules',
    'spacing_min', 'spacing_ratio', 'pgc_shift', 'pgc_shift_mode', 'pgc_tau',
    'pgc_shift_jitter', 'sharpness', 'code_noise', 'extra_input', 'goal_code_shift',
    'pgc_layout', 'pgc_module_weights', 'pgc_shift_rule', 'pgc_shift_frac', 'cell_type',
    'silence_pgc',
    'arena_half', 'episode_len', 'action_repeat', 'v_max', 'rot_max', 'goal_radius',
    'vision', 'img_noise', 'randomize_textures', 'policy', 'conv_arch', 'nh',
    'encoder_norm', 'forward_bias', 'seed', 'n_envs', 'rollout', 'gamma', 'lr',
    'rms_alpha', 'rms_eps', 'entropy', 'value_coef', 'grad_clip', 'reward_clip')


def add_shared_args(ap: argparse.ArgumentParser):
  g = ap.add_argument_group('population code')
  g.add_argument('--cond', choices=sorted(CONDITIONS), default='gc100',
                 help='preset for pgc_frac/code_mode (overridable below)')
  g.add_argument('--pgc_frac', type=float, default=None)
  g.add_argument('--code_mode', choices=['full', 'none'], default=None)
  g.add_argument('--n_cells', type=int, default=512)
  g.add_argument('--n_modules', type=int, default=4)
  g.add_argument('--spacing_min', type=float, default=0.40)
  g.add_argument('--spacing_ratio', type=float, default=1.42)
  g.add_argument('--pgc_shift', type=float, default=0.20, help='metres (space mode)')
  g.add_argument('--pgc_shift_mode', choices=['space', 'time'], default='space')
  g.add_argument('--pgc_tau', type=float, default=0.5, help='seconds (time mode)')
  g.add_argument('--pgc_shift_jitter', type=float, default=0.0,
                 help='lognormal sigma of per-cell shift heterogeneity (0 = all equal)')
  g.add_argument('--pgc_layout', choices=['uniform', 'fine', 'coarse', 'middle', 'weights'],
                 default='uniform', help='which modules hold the PGCs (see codes.py)')
  g.add_argument('--pgc_module_weights', default=None,
                 help="comma list, one weight per module, for --pgc_layout weights (e.g. '2,1,1,0')")
  g.add_argument('--pgc_shift_rule', choices=['const', 'spacing'], default='const',
                 help="'spacing': shift = pgc_shift_frac x module spacing (constant phase shift)")
  g.add_argument('--pgc_shift_frac', type=float, default=0.5)
  g.add_argument('--cell_type', choices=['grid', 'band'], default='grid',
                 help="'band': stripe cells (one cosine per cell) instead of hexagonal grid cells")
  g.add_argument('--silence_pgc', type=int, default=0,
                 help='1: hold the PGC slots of the population at zero in the current '
                      'and the goal code under every probe, leaving the grid cells '
                      'unchanged (set automatically for --cond gc67)')
  g.add_argument('--sharpness', type=float, default=2.0)
  g.add_argument('--code_noise', type=float, default=0.0)
  g.add_argument('--code_seed', type=int, default=None,
                 help='population seed (default: the run seed, so conditions '
                      'within a seed share one population)')
  g.add_argument('--extra_input', choices=['none', 'vel'], default='none',
                 help="'vel': also feed the policy the allocentric velocity and "
                      'motion direction (4 numbers) -- control for whether a '
                      'PGC advantage is just velocity information')
  g.add_argument('--goal_code_shift', choices=['intact', 'zero', 'lookahead'], default='intact',
                 help="'intact' (paper): g* is the population's activity when the "
                      "goal was reached, so PGC units encode goal + shift along the "
                      "arrival direction; 'zero': PGC units in g* are evaluated with "
                      'zero shift (position-only goal memory); \'lookahead\': ALL cells '
                      'in g* are evaluated at hit_pos + shift*hit_dir with zero PGC shift '
                      '(gives a grid-cell agent the PGCs\' better-centred goal snapshot). '
                      'Controls that separate the goal-memory geometry from the '
                      'current-code effect of PGCs')
  e = ap.add_argument_group('environment')
  e.add_argument('--arena_half', type=float, default=1.25, help='2.5 m arena')
  e.add_argument('--episode_len', type=int, default=1350,
                 help='decision steps (5400 frames / repeat 4 = 90 s)')
  e.add_argument('--action_repeat', type=int, default=4)
  e.add_argument('--v_max', type=float, default=0.35)
  e.add_argument('--rot_max', type=float, default=2.2)
  e.add_argument('--goal_radius', type=float, default=0.30)
  e.add_argument('--vision', choices=['raycast', 'heading_only', 'sighted', 'none'],
                 default='raycast')
  e.add_argument('--img_noise', type=float, default=0.02)
  e.add_argument('--randomize_textures', type=int, default=1,
                 help='1: landmark/stripe layout re-drawn every episode (paper); '
                      '0: fixed layout (a single frame then reveals position)')
  p = ap.add_argument_group('policy')
  p.add_argument('--policy', choices=['lstm', 'ff'], default='lstm')
  p.add_argument('--conv_arch', choices=['si', 'dqn'], default='si')
  p.add_argument('--nh', type=int, default=256)
  p.add_argument('--encoder_norm', choices=['none', 'layernorm'], default='none',
                 help='LayerNorm on the 256-d vision embedding (bounds its scale; '
                      'see nets.py)')
  p.add_argument('--forward_bias', type=float, default=0.0,
                 help='initial logit bias of the forward action (exploration '
                      'prior: an untrained uniform policy barely moves; applied '
                      'identically to the untrained chance-line policy)')


def resolve_cond(cfg):
  frac, mode = CONDITIONS[cfg['cond']]
  if cfg.get('pgc_frac') is None:
    cfg['pgc_frac'] = frac
  if cfg.get('code_seed') is None:
    cfg['code_seed'] = cfg['seed']
  if cfg.get('code_mode') is None:
    cfg['code_mode'] = mode
  # gc67 keeps pgc_frac 0.33, so build_population draws the pgc33 population of
  # the seed (same PGC indices); CodeSource then holds those slots at zero.
  cfg['silence_pgc'] = 1 if cfg['cond'] == 'gc67' else int(cfg.get('silence_pgc') or 0)
  # Make the label honest when a preset is overridden.
  if cfg['pgc_frac'] != frac or cfg['code_mode'] != mode:
    cfg['cond'] = f"{cfg['cond']}+frac{cfg['pgc_frac']}+{cfg['code_mode']}"
  if cfg['silence_pgc'] and not cfg['cond'].startswith('gc67'):
    cfg['cond'] += '+silenced'
  return cfg


def build_population(cfg, device):
  return GridPopulation(
      n_cells=cfg['n_cells'], n_modules=cfg['n_modules'],
      spacing_min=cfg['spacing_min'], spacing_ratio=cfg['spacing_ratio'],
      pgc_frac=cfg['pgc_frac'], pgc_shift=cfg['pgc_shift'],
      pgc_shift_mode=cfg['pgc_shift_mode'], pgc_tau=cfg['pgc_tau'],
      pgc_shift_jitter=cfg['pgc_shift_jitter'], sharpness=cfg['sharpness'],
      noise=cfg.get('code_noise', 0.0), seed=cfg['code_seed'], device=device,
      pgc_layout=cfg.get('pgc_layout', 'uniform'),
      pgc_module_weights=([float(x) for x in str(cfg['pgc_module_weights']).split(',')]
                          if cfg.get('pgc_module_weights') else None),
      pgc_shift_rule=cfg.get('pgc_shift_rule', 'const'),
      pgc_shift_frac=cfg.get('pgc_shift_frac', 0.5), cell_type=cfg.get('cell_type', 'grid'))


def build_env(cfg, n, seed, device, stagger=True, paired=False):
  return ArenaEnv(n, half=cfg['arena_half'], episode_len=cfg['episode_len'],
                  action_repeat=cfg['action_repeat'], v_max=cfg['v_max'],
                  rot_max=cfg['rot_max'], goal_radius=cfg['goal_radius'],
                  vision=cfg['vision'], img_noise=cfg['img_noise'],
                  randomize_textures=bool(cfg.get('randomize_textures', 1)),
                  stagger=stagger, paired=paired, pair_seed=seed, seed=seed, device=device)


def build_policy(cfg, device):
  net = PolicyNet(n_actions=NUM_ACTIONS, n_code=cfg['n_cells'], nh=cfg['nh'],
                  recurrent=(cfg['policy'] == 'lstm'), conv_arch=cfg['conv_arch'],
                  extra_in=4 if cfg.get('extra_input') == 'vel' else 0,
                  encoder_norm=cfg.get('encoder_norm', 'none')).to(device)
  fb = float(cfg.get('forward_bias', 0.0) or 0.0)
  if fb:
    with torch.no_grad():
      net.pi.bias[2] += fb        # ACTIONS[2] == 'forward'
  return net


def extra_from_obs(cfg, obs):
  if cfg.get('extra_input') == 'vel':
    return torch.cat([obs['vel'], obs['dir']], 1)
  return None


def sham_indices(pop, k_draw, frac=SHAM_FRAC):
  """Count-matched, module-stratified random lesion set: round(frac*n) cells
  drawn (draw k_draw) from the grid cells when the population has any, else
  from all cells -- the control that 'no_pgc' must beat."""
  import numpy as np
  rng = np.random.default_rng(1000 + int(k_draw))
  pool = pop.gc_idx if len(pop.gc_idx) else np.arange(pop.n_cells)
  n = int(round(frac * pop.n_cells))
  per = [n // pop.n_modules + (1 if m < n % pop.n_modules else 0) for m in range(pop.n_modules)]
  out = []
  for m in range(pop.n_modules):
    cand = pool[pop.module[pool] == m]
    out.append(rng.choice(cand, size=min(per[m], len(cand)), replace=False))
  return np.concatenate(out).astype(int)


def half_indices(pop, cls, k_draw):
  """Class-half lesion set: floor(n / 2) of the n cells of one class (cls 'pgc'
  or 'gc'), module-stratified: floor(n_m / 2) from every module m, the odd
  remainders topped up module by module so the total is exactly floor(n / 2).
  Draw k_draw uses rng 2000 + k_draw for the PGC class and 3000 + k_draw for
  the grid-cell class. An empty class gives an empty set (the probe then equals
  intact). Class-size-matched counterpart of sham_indices: pgc_half<k> and
  gc_half<k> remove the same number of cells when the classes are equal in
  size (pgc50)."""
  import numpy as np
  rng = np.random.default_rng((2000 if cls == 'pgc' else 3000) + int(k_draw))
  pool = np.asarray(pop.pgc_idx if cls == 'pgc' else pop.gc_idx, dtype=int)
  n = len(pool) // 2
  sizes = [int((pop.module[pool] == m).sum()) for m in range(pop.n_modules)]
  per = [s // 2 for s in sizes]
  extra = n - sum(per)
  for m in range(pop.n_modules):
    if extra <= 0:
      break
    if sizes[m] % 2:
      per[m] += 1
      extra -= 1
  out = []
  for m in range(pop.n_modules):
    if per[m]:
      cand = pool[pop.module[pool] == m]
      out.append(rng.choice(cand, size=per[m], replace=False))
  return np.concatenate(out).astype(int) if out else np.zeros(0, dtype=int)


class CodeSource:
  """Turns env observations into the policy's population-code inputs, with
  the evaluation probes (masking subsets, de-shifting PGCs, scrambling) in
  one place. Call it for the current code g_t; use `.goal(...)` for the goal
  snapshot g* (it honours --goal_code_shift and the same probe masks).

  Probes: no_code (g_t and g* zero), no_current (g_t zero, g* kept),
  no_goal (g* zero), no_pgc / no_gc (subset zeroed in both), sham0..4
  (count-matched random subset zeroed in both; average them), pgc_half0..2 /
  gc_half0..2 (a random half of the PGC class / of the grid-cell class zeroed
  in both; class-size-matched lesions, see half_indices), pgc_as_gc (PGCs
  de-shifted in both), shuffled_current (g_t from a random position/direction:
  in-distribution but uninformative), fake_goal (g* from a random position:
  the paper's 'fake goal grid code'), untrained (handled by the caller).

  cfg['silence_pgc'] (condition gc67): the population's PGC slots are held at
  zero in g_t and g* under every probe, on top of the probe's own mask, so
  only the grid cells carry signal. intact, no_pgc, pgc_as_gc and pgc_half<k>
  are then the same input (de-shifting a silenced cell changes nothing, so
  pgc_as_gc takes the intact code path and the three are bit-identical), the
  sham draws come from the live grid cells exactly as in pgc33, and no_gc
  zeroes everything. n_masked counts every zeroed slot, silenced PGCs
  included."""

  def __init__(self, cfg, pop, device, probe='intact'):
    if probe not in PROBES:
      raise ValueError(f'unknown probe {probe!r}; choose from {PROBES}')
    self.cfg, self.pop, self.device, self.probe = cfg, pop, device, probe
    self.n_code = cfg['n_cells']
    self.none = cfg['code_mode'] == 'none' or probe == 'no_code'
    self.no_current = probe == 'no_current'
    mask = torch.ones(self.n_code, device=device)
    if probe == 'no_pgc':
      mask[pop.pgc_idx] = 0.0
    elif probe == 'no_gc':
      mask[pop.gc_idx] = 0.0
    elif probe.startswith('sham'):
      mask[sham_indices(pop, int(probe[4:]))] = 0.0
    elif probe.startswith('pgc_half'):
      mask[half_indices(pop, 'pgc', int(probe[8:]))] = 0.0
    elif probe.startswith('gc_half'):
      mask[half_indices(pop, 'gc', int(probe[7:]))] = 0.0
    self.silence_pgc = bool(cfg.get('silence_pgc'))
    if self.silence_pgc:
      mask[pop.pgc_idx] = 0.0
    self.mask = mask
    self.n_masked = int((mask == 0).sum())
    deshift = probe == 'pgc_as_gc' and not self.silence_pgc
    self.shift_scale = 0.0 if deshift else 1.0
    gcs = cfg.get('goal_code_shift', 'intact')
    self.goal_shift_scale = 0.0 if (deshift or gcs in ('zero', 'lookahead')) else 1.0
    self.goal_lookahead = (cfg['pgc_shift'] if cfg['pgc_shift_mode'] == 'space' else None) \
        if gcs == 'lookahead' else None
    self.goal_lookahead_tau = cfg['pgc_tau'] if (gcs == 'lookahead' and
                                                 cfg['pgc_shift_mode'] == 'time') else None
    self.no_goal = probe == 'no_goal'
    self.fake_goal = probe == 'fake_goal'
    self.shuffled_current = probe == 'shuffled_current'
    self.lim = cfg['arena_half'] - 0.08

  def _random_state(self, B):
    pos = (torch.rand(B, 2, device=self.device) * 2 - 1) * self.lim
    ang = torch.rand(B, device=self.device) * 2 * 3.141592653589793
    d = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return pos, d, d * self.cfg['v_max']

  def _code(self, pos, dir, vel, shift_scale):
    if self.none:
      return torch.zeros(pos.shape[0], self.n_code, device=self.device)
    g = self.pop.activity(pos, dir=dir, vel=vel, shift_scale=shift_scale)
    return g * self.mask[None]

  def __call__(self, pos, dir, vel):
    if self.no_current:
      return torch.zeros(pos.shape[0], self.n_code, device=self.device)
    if self.shuffled_current:
      pos, dir, vel = self._random_state(pos.shape[0])
    return self._code(pos, dir, vel, self.shift_scale)

  def goal(self, pos, dir, vel):
    if self.no_goal:
      return torch.zeros(pos.shape[0], self.n_code, device=self.device)
    if self.fake_goal:
      pos, dir, vel = self._random_state(pos.shape[0])
    if self.goal_lookahead is not None:
      pos = pos + self.goal_lookahead * dir
    elif self.goal_lookahead_tau is not None:
      pos = pos + self.goal_lookahead_tau * vel
    return self._code(pos, dir, vel, self.goal_shift_scale)
