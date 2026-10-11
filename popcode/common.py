"""Shared builders so train / evaluate / validate instantiate identical objects
from one config dict (the run's config.json)."""
import argparse

import torch

from popcode.codes import GridPopulation
from popcode.env import NUM_ACTIONS, ArenaEnv
from popcode.nets import PolicyNet

CONDITIONS = {
    # name: (pgc_frac, code_mode)
    'gc100': (0.0, 'full'),      # 100 % grid cells
    'pgc33': (0.33, 'full'),     # 33 % predictive grid cells, 67 % grid cells
    'pgc100': (1.0, 'full'),     # 100 % predictive grid cells
    'none': (0.0, 'none'),       # no spatial code at all (A3C-style baseline)
}
PROBES = ('intact', 'no_code', 'no_current', 'no_goal', 'no_pgc', 'no_gc', 'pgc_as_gc',
          'fake_goal', 'shuffled_current', 'sham0', 'sham1', 'sham2', 'sham3', 'sham4',
          'untrained')
SHAM_FRAC = 0.33   # count-matched sham lesions remove round(SHAM_FRAC * n_cells) cells

# Keys that define the experiment a checkpoint belongs to; a --resume whose
# arguments differ in any of these from the stored config.json is refused.
EXPERIMENT_KEYS = (
    'cond', 'pgc_frac', 'code_mode', 'code_seed', 'n_cells', 'n_modules',
    'spacing_min', 'spacing_ratio', 'pgc_shift', 'pgc_shift_mode', 'pgc_tau',
    'pgc_shift_jitter', 'sharpness', 'code_noise', 'extra_input', 'goal_code_shift',
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
  # Make the label honest when a preset is overridden.
  if cfg['pgc_frac'] != frac or cfg['code_mode'] != mode:
    cfg['cond'] = f"{cfg['cond']}+frac{cfg['pgc_frac']}+{cfg['code_mode']}"
  return cfg


def build_population(cfg, device):
  return GridPopulation(
      n_cells=cfg['n_cells'], n_modules=cfg['n_modules'],
      spacing_min=cfg['spacing_min'], spacing_ratio=cfg['spacing_ratio'],
      pgc_frac=cfg['pgc_frac'], pgc_shift=cfg['pgc_shift'],
      pgc_shift_mode=cfg['pgc_shift_mode'], pgc_tau=cfg['pgc_tau'],
      pgc_shift_jitter=cfg['pgc_shift_jitter'], sharpness=cfg['sharpness'],
      noise=cfg.get('code_noise', 0.0), seed=cfg['code_seed'], device=device)


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


class CodeSource:
  """Turns env observations into the policy's population-code inputs, with
  the evaluation probes (masking subsets, de-shifting PGCs, scrambling) in
  one place. Call it for the current code g_t; use `.goal(...)` for the goal
  snapshot g* (it honours --goal_code_shift and the same probe masks).

  Probes: no_code (g_t and g* zero), no_current (g_t zero, g* kept),
  no_goal (g* zero), no_pgc / no_gc (subset zeroed in both), sham0..4
  (count-matched random subset zeroed in both; average them), pgc_as_gc
  (PGCs de-shifted in both), shuffled_current (g_t from a random
  position/direction: in-distribution but uninformative), fake_goal (g* from a
  random position: the paper's 'fake goal grid code'), untrained (handled by
  the caller)."""

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
    self.mask = mask
    self.n_masked = int((mask == 0).sum())
    self.shift_scale = 0.0 if probe == 'pgc_as_gc' else 1.0
    gcs = cfg.get('goal_code_shift', 'intact')
    self.goal_shift_scale = 0.0 if (probe == 'pgc_as_gc' or gcs in ('zero', 'lookahead')) else 1.0
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
