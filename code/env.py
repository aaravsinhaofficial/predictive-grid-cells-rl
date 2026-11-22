"""Vectorised square-arena goal task with egocentric vision (torch, GPU).

Semantics follow the DeepMind-Lab goal tasks of Banino et al. 2018 as wrapped
in ~/banino/rl (env.py / fake_env.py): an unmarked goal fixed within an
episode, +10 reward on reaching it, the agent is then respawned at a random
pose (the goal stays), episodes last a fixed 5400 environment frames
(= 1350 decision steps at action repeat 4, i.e. 90 s at 60 fps), six discrete
actions (turn left/right, move forward/backward, strafe left/right).

Kinematics are those of banino/rl/fake_env.py (first-order velocity smoothing,
motor noise, wall collisions kill velocity), vectorised over n environments.

Vision ('raycast'): a first-person 84x84 RGB rendering of a square room with
four distinctly coloured walls (fixed across episodes: the paper's consistent
distal cues, so heading is always readable), periodic dark stripes and one
white landmark per wall whose positions are re-drawn every episode (the
paper's intra-maze cues "were different for each episode but consistent
within an episode"), depth shading, a dark ceiling and a floor gradient.
'heading_only' and 'sighted' reproduce fake_env.py's synthetic gradient images
(heading-keyed only / position+heading-keyed) for ablations.

Heading of motion `mdir` (used by space-mode PGCs) is the direction of the
last displacement faster than v_min, held while the agent is slower than that
(turning in place produces sub-millimetre motor-noise jitter whose direction
is random), and reset to the facing direction at (re)spawn.

Episode phases are staggered at reset_all() (stagger=True) so the n
environments do not end their episodes in lock-step; statistics of the
resulting partial first episodes are not reported.

Per-episode statistics emitted on episode end: return, number of goals,
step of the first goal, and the mean path efficiency of goal returns after the
goal location is known (straight-line distance minus goal radius, divided by
the distance actually travelled since the respawn).

Changed rooms (`rooms`, default None = the familiar open room, in which case
nothing below is active and the environment is unchanged): every episode draws
one room condition from the given list. A condition may add one partition
wall with a single doorway, one square obstacle, and a goal relocation (the
goal jumps, unsignalled, to a new place at mid-episode). Walls and the
obstacle block movement (the agent slides along them) and sight (they are
rendered in the first-person view). See ROOMS / room_spec below.
"""
import math

import numpy as np
import torch

ACTIONS = ('turn_left', 'turn_right', 'forward', 'backward',
           'strafe_left', 'strafe_right')
NUM_ACTIONS = len(ACTIONS)
# (forward, leftward, rotation) target multipliers; yaw positive = CCW.
_TARGETS = [[0., 0., 1.], [0., 0., -1.], [1., 0., 0.], [-1., 0., 0.],
            [0., 1., 0.], [0., -1., 0.]]
# wall ids: 0 = +x (east), 1 = +y (north), 2 = -x (west), 3 = -y (south)
WALL_COLORS = [[0.85, 0.25, 0.25], [0.25, 0.75, 0.30],
               [0.30, 0.40, 0.90], [0.90, 0.80, 0.25]]
LANDMARK_POS = [-0.6, 0.3, 0.8, -0.2]   # fixed layout (randomize_textures=False)
STRIPE_PHASE = [0.0, 0.25, 0.5, 0.75]
VISION_MODES = ('raycast', 'heading_only', 'sighted', 'none')

# ----------------------------------------------------------- changed rooms
WALL_T = 0.10        # partition thickness (m)
DOOR_W = 0.60        # doorway width (m)
BOX_SIDE = 0.50      # obstacle side (m)
RELOC_MIN = 1.0      # a relocated goal lands at least this far from the old one (m)
N_BOX = 3            # per env: two partition segments + one obstacle
PART_COLOR = [0.70, 0.70, 0.75]
BOX_COLOR = [0.75, 0.35, 0.80]
# name -> (wall, box). wall = (axis, offset, door centre): axis 'x' is a wall
# along the line x = offset with its doorway centred at y = door centre; axis
# 'y' a wall along y = offset with the doorway centred at x = door centre.
# box = centre (x, y) of the obstacle. Every gap left for the agent is at
# least 0.45 m wide (0.29 m for its centre, which keeps 0.08 m from surfaces).
ROOMS = {
    'open': (None, None),
    'D1': (('x', 0.0, 0.0), None),
    'D2': (('x', 0.0, 0.8), None),
    'D3': (('y', 0.0, -0.8), None),
    'D4': (('y', 0.4, 0.4), None),
    'O1': (None, (0.0, 0.0)),
    'O2': (None, (0.5, 0.5)),
    'O3': (None, (-0.55, -0.4)),
    'DO1': (('x', -0.4, -0.7), (0.45, 0.35)),
    'DO2': (('y', 0.4, -0.6), (0.3, -0.5)),
    'DO3': (('x', 0.4, 0.5), (-0.45, -0.3)),
    # held out: never used for fine-tuning, only for testing
    'H1': (('y', -0.4, 0.7), None),
    'H2': (None, (-0.5, 0.55)),
    'H3': (('y', -0.4, -0.7), (0.45, 0.45)),
}
TRAIN_LAYOUTS = ('D1', 'D2', 'D3', 'D4', 'O1', 'O2', 'O3', 'DO1', 'DO2', 'DO3')
HELDOUT_LAYOUTS = ('H1', 'H2', 'H3')
# The changed-room family used for fine-tuning: every layout with and without
# a goal relocation ('+R'), plus the open room with a relocation.
CHANGED = tuple(n + r for n in TRAIN_LAYOUTS for r in ('', '+R')) + ('open+R',)
TEST_CONDITIONS = (('open',) + CHANGED +
                   tuple(n + r for n in HELDOUT_LAYOUTS for r in ('', '+R')))


def room_spec(name, half=1.25):
  """'D2+R' -> dict(name, layout, reloc, boxes): boxes is a list of N_BOX
  axis-aligned rectangles (xmin, xmax, ymin, ymax); unused slots sit far
  outside the arena."""
  layout, reloc = (name[:-2], True) if name.endswith('+R') else (name, False)
  if layout not in ROOMS:
    raise ValueError(f'unknown room {name!r}; layouts: {sorted(ROOMS)} (append +R for a goal relocation)')
  wall, box = ROOMS[layout]
  far = [100.0, 100.0, 100.0, 100.0]
  boxes = [list(far), list(far), list(far)]
  if wall is not None:
    axis, off, door = wall
    lo, hi = door - DOOR_W / 2.0, door + DOOR_W / 2.0
    segs = [(-half, lo), (hi, half)]
    for k, (a, b) in enumerate(segs):
      if b - a <= 1e-6:
        continue
      if axis == 'x':
        boxes[k] = [off - WALL_T / 2.0, off + WALL_T / 2.0, a, b]
      else:
        boxes[k] = [a, b, off - WALL_T / 2.0, off + WALL_T / 2.0]
  if box is not None:
    cx, cy = box
    boxes[2] = [cx - BOX_SIDE / 2.0, cx + BOX_SIDE / 2.0, cy - BOX_SIDE / 2.0, cy + BOX_SIDE / 2.0]
  return dict(name=name, layout=layout, reloc=reloc, boxes=boxes)


def parse_rooms(rooms):
  """None | 'none' -> None (familiar room); 'changed' -> CHANGED; otherwise a
  comma list (or sequence) of condition names."""
  if rooms is None or rooms == 'none' or rooms == '':
    return None
  if rooms == 'changed':
    return list(CHANGED)
  if isinstance(rooms, str):
    return [r.strip() for r in rooms.split(',') if r.strip()]
  return list(rooms)


def wrap_pi(a):
  return (a + math.pi) % (2.0 * math.pi) - math.pi


class ArenaEnv:
  """n square-arena goal environments stepped in lock-step on one device."""

  def __init__(self, n, half=1.25, episode_len=1350, action_repeat=4, fps=60.0,
               v_max=0.35, rot_max=2.2, tau=0.25, noise_v=0.03, noise_r=0.15,
               wall_margin=0.08, goal_radius=0.30, goal_reward=10.0,
               vision='raycast', img=84, fov_deg=90.0, img_noise=0.02,
               randomize_textures=True, v_min=0.05, stagger=True,
               paired=False, pair_seed=0, seed=0, device='cuda', rooms=None):
    if vision not in VISION_MODES:
      raise ValueError(f'vision must be one of {VISION_MODES}, got {vision!r}')
    self.n = int(n)
    self.half = float(half)
    self.episode_len = int(episode_len)
    self.action_repeat = int(action_repeat)
    self.dt = self.action_repeat / float(fps)
    self.v_max, self.rot_max, self.tau = float(v_max), float(rot_max), float(tau)
    self.noise_v, self.noise_r = float(noise_v), float(noise_r)
    self.wall_margin = float(wall_margin)
    self.goal_radius, self.goal_reward = float(goal_radius), float(goal_reward)
    self.vision = vision
    self.img = int(img)
    self.fov = math.radians(float(fov_deg))
    self.img_noise = float(img_noise)
    self.randomize_textures = bool(randomize_textures)
    self.v_min = float(v_min)
    self.stagger = bool(stagger)
    # paired=True: goal / spawn / texture draws come from a CPU generator keyed
    # by (env slot, episode index, draw index), so every run evaluated with the
    # same pair_seed sees identical episode layouts regardless of its policy
    # (common random numbers for paired comparisons). Motor and image noise
    # still come from the main generator.
    self.paired = bool(paired)
    self.pair_seed = int(pair_seed)
    self.device = torch.device(device)
    self.gen = torch.Generator(device=self.device)
    self.gen.manual_seed(int(seed))
    dev = self.device
    n = self.n

    def z(*s):
      return torch.zeros(*s, device=dev)

    self.pos, self.goal, self.mdir, self.spawn = z(n, 2), z(n, 2), z(n, 2), z(n, 2)
    self.hd, self.vf, self.vs, self.vr = z(n), z(n), z(n), z(n)
    self.t = torch.zeros(n, dtype=torch.long, device=dev)
    self.partial = torch.zeros(n, dtype=torch.bool, device=dev)
    self.ep_idx = torch.full((n,), -1, dtype=torch.long, device=dev)
    self.draw_idx = torch.zeros(n, dtype=torch.long, device=dev)
    self.travelled, self.ep_return = z(n), z(n)
    self.n_goals = torch.zeros(n, dtype=torch.long, device=dev)
    self.first_goal_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    self.path_eff_sum, self.path_eff_n = z(n), z(n)
    self.goal_lost = torch.zeros(n, dtype=torch.bool, device=dev)   # goal moved, not yet found again
    self.episodes_done = 0
    # Changed rooms (None = familiar open room: everything below stays inert).
    names = parse_rooms(rooms)
    self.has_rooms = names is not None
    self.room_names = names or ['open']
    specs = [room_spec(r, self.half) for r in self.room_names]
    self._room_boxes = torch.tensor([sp['boxes'] for sp in specs], device=dev)      # [R,N_BOX,4]
    self._room_reloc = torch.tensor([sp['reloc'] for sp in specs], dtype=torch.bool, device=dev)
    self.room_id = torch.zeros(n, dtype=torch.long, device=dev)
    self.boxes = self._room_boxes[self.room_id].clone()                             # [n,N_BOX,4]
    self.reloc = torch.zeros(n, dtype=torch.bool, device=dev)       # this episode relocates its goal
    self.relocated = torch.zeros(n, dtype=torch.bool, device=dev)   # ... and has done so
    self.reloc_t = self.episode_len // 2
    self.n_goals_post = torch.zeros(n, dtype=torch.long, device=dev)
    self.reloc_first_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    self._box_col = torch.tensor([PART_COLOR, PART_COLOR, BOX_COLOR], device=dev)
    self._targets = torch.tensor(_TARGETS, device=dev)
    # Per-env intra-maze cue layout (re-drawn per episode when randomize_textures).
    self.land_pos = torch.tensor(LANDMARK_POS, device=dev).expand(n, 4).clone()
    self.stripe_phase = torch.tensor(STRIPE_PHASE, device=dev).expand(n, 4).clone()
    # Rendering buffers.
    W = self.img
    self._v = (torch.arange(W, device=dev) + 0.5) / W * 2.0 - 1.0        # rows: -1 top .. +1 bottom
    cols = (torch.arange(W, device=dev) + 0.5) / W
    self._col_ang = self.fov / 2.0 - self.fov * cols                     # left column = left of heading
    g = torch.linspace(0.0, 1.0, W, device=dev)
    self._gv, self._gu = torch.meshgrid(g, g, indexing='ij')             # [H,W]
    self._wall_col = torch.tensor(WALL_COLORS, device=dev)
    self._ceil = torch.tensor([0.12, 0.12, 0.16], device=dev)
    self._floor = torch.tensor([0.50, 0.50, 0.50], device=dev)

  # ------------------------------------------------------------ randomness
  def _rand(self, *shape):
    return torch.rand(*shape, generator=self.gen, device=self.device)

  def _randn(self, *shape):
    return torch.randn(*shape, generator=self.gen, device=self.device)

  def _random_pos(self, k):
    lim = self.half - self.wall_margin
    return (self._rand(k, 2) * 2.0 - 1.0) * lim

  def _in_boxes(self, p, idx, clear):
    """p [k,2] for envs idx -> [k] bool: within `clear` of a wall / obstacle."""
    b = self.boxes if idx is None else self.boxes[idx]
    x, y = p[:, 0:1], p[:, 1:2]
    return ((x > b[..., 0] - clear) & (x < b[..., 1] + clear) &
            (y > b[..., 2] - clear) & (y < b[..., 3] + clear)).any(1)

  @staticmethod
  def _in_boxes_np(q, boxes, clear):
    return bool(((q[0] > boxes[:, 0] - clear) & (q[0] < boxes[:, 1] + clear) &
                 (q[1] > boxes[:, 2] - clear) & (q[1] < boxes[:, 3] + clear)).any())

  def _paired_rngs(self, idx):
    """One numpy generator per env in idx, keyed by (slot, episode, draw)."""
    rngs = []
    for i in idx.tolist():
      seed = (self.pair_seed * 1_000_003 + i * 7_919 + int(self.ep_idx[i]) * 104_729
              + int(self.draw_idx[i]) * 13) & 0x7FFFFFFF
      rngs.append(np.random.default_rng(seed))
      self.draw_idx[i] += 1
    return rngs

  # ---------------------------------------------------------------- resets
  def _respawn(self, mask):
    """Random pose away from the goal; zero velocities; motion dir = facing."""
    idx = mask.nonzero().squeeze(1)
    k = idx.numel()
    if k == 0:
      return
    lim = self.half - self.wall_margin
    if self.paired:
      goals = self.goal[idx].cpu().numpy()
      bxs = self.boxes[idx].cpu().numpy()
      ps, hds = [], []
      for rng, g, bx in zip(self._paired_rngs(idx), goals, bxs):
        for _ in range(100):
          q = rng.uniform(-lim, lim, 2)
          if np.linalg.norm(q - g) >= self.goal_radius + 0.1 and not (
              self.has_rooms and self._in_boxes_np(q, bx, self.wall_margin)):
            break
        ps.append(q)
        hds.append(rng.uniform(-math.pi, math.pi))
      p = torch.tensor(np.array(ps), dtype=torch.float32, device=self.device)
      hd = torch.tensor(np.array(hds), dtype=torch.float32, device=self.device)
    else:
      p = self._random_pos(k)
      for _ in range(100):
        bad = (p - self.goal[idx]).norm(dim=1) < self.goal_radius + 0.1
        if self.has_rooms:
          bad = bad | self._in_boxes(p, idx, self.wall_margin)
        nb = int(bad.sum())
        if nb == 0:
          break
        p[bad] = self._random_pos(nb)
      hd = (self._rand(k) * 2.0 - 1.0) * math.pi
    self.pos[idx] = p
    self.hd[idx] = hd
    self.vf[idx] = 0.0
    self.vs[idx] = 0.0
    self.vr[idx] = 0.0
    self.mdir[idx] = torch.stack([torch.cos(self.hd[idx]), torch.sin(self.hd[idx])], 1)
    self.spawn[idx] = p
    self.travelled[idx] = 0.0

  def _reset_envs(self, mask):
    idx = mask.nonzero().squeeze(1)
    k = idx.numel()
    if k == 0:
      return
    self.ep_idx[idx] += 1
    self.draw_idx[idx] = 0
    lim = self.half - self.wall_margin
    if self.has_rooms:
      # One room condition per episode: uniform over the list (paired runs
      # cycle through it by env slot and episode, so layouts match across runs).
      R = len(self.room_names)
      if self.paired:
        rid = (idx + self.ep_idx[idx]) % R
      else:
        rid = torch.randint(0, R, (k,), generator=self.gen, device=self.device)
      self.room_id[idx] = rid
      self.boxes[idx] = self._room_boxes[rid]
      self.reloc[idx] = self._room_reloc[rid]
    self.relocated[idx] = False
    self.n_goals_post[idx] = 0
    self.reloc_first_t[idx] = -1
    self.goal_lost[idx] = False
    if self.paired:
      rngs = self._paired_rngs(idx)
      if self.has_rooms:
        bxs = self.boxes[idx].cpu().numpy()
        gs = []
        for r, bx in zip(rngs, bxs):
          for _ in range(100):
            q = r.uniform(-lim, lim, 2)
            if not self._in_boxes_np(q, bx, self.goal_radius):
              break
          gs.append(q)
      else:
        gs = [r.uniform(-lim, lim, 2) for r in rngs]
      self.goal[idx] = torch.tensor(np.array(gs),
                                    dtype=torch.float32, device=self.device)
      if self.randomize_textures:
        tl = self.half - 0.2
        self.land_pos[idx] = torch.tensor(np.array([r.uniform(-tl, tl, 4) for r in rngs]),
                                          dtype=torch.float32, device=self.device)
        self.stripe_phase[idx] = torch.tensor(np.array([r.uniform(0, 1, 4) for r in rngs]),
                                              dtype=torch.float32, device=self.device)
    else:
      self.goal[idx] = self._free_goal(idx)
    self.t[idx] = 0
    self.partial[idx] = False
    self.ep_return[idx] = 0.0
    self.n_goals[idx] = 0
    self.first_goal_t[idx] = -1
    self.path_eff_sum[idx] = 0.0
    self.path_eff_n[idx] = 0.0
    if self.randomize_textures and not self.paired:
      tl = self.half - 0.2
      self.land_pos[idx] = (self._rand(k, 4) * 2.0 - 1.0) * tl
      self.stripe_phase[idx] = self._rand(k, 4)
    self._respawn(mask)

  def _free_goal(self, idx, old=None, pos=None):
    """Goal centres for envs idx: uniform, a goal radius clear of walls and
    the obstacle. For a relocation (`old`, `pos` given) also at least
    RELOC_MIN from the old goal and not on top of the agent."""
    k = idx.numel()
    g = self._random_pos(k)
    if not self.has_rooms:
      return g
    for _ in range(100):
      bad = self._in_boxes(g, idx, self.goal_radius)
      if old is not None:
        bad = bad | ((g - old).norm(dim=1) < RELOC_MIN) | \
            ((g - pos).norm(dim=1) < self.goal_radius + 0.1)
      nb = int(bad.sum())
      if nb == 0:
        break
      g[bad] = self._random_pos(nb)
    return g

  def _relocate(self, mask):
    """Move the goal of the envs in mask to a new place (unsignalled)."""
    idx = mask.nonzero().squeeze(1)
    if idx.numel() == 0:
      return
    old, pos = self.goal[idx].clone(), self.pos[idx].clone()
    if self.paired:
      lim = self.half - self.wall_margin
      bxs = self.boxes[idx].cpu().numpy()
      gs = []
      # Candidates come from a generator keyed by (env slot, episode) alone, so
      # every run sees the same sequence whatever its policy did before.
      rngs = [np.random.default_rng((self.pair_seed * 1_000_003 + i * 7_919 +
                                     int(self.ep_idx[i]) * 104_729 + 999_983) & 0x7FFFFFFF)
              for i in idx.tolist()]
      for r, bx, o, p in zip(rngs, bxs, old.cpu().numpy(), pos.cpu().numpy()):
        for _ in range(200):
          q = r.uniform(-lim, lim, 2)
          if (not self._in_boxes_np(q, bx, self.goal_radius)
              and np.linalg.norm(q - o) >= RELOC_MIN
              and np.linalg.norm(q - p) >= self.goal_radius + 0.1):
            break
        gs.append(q)
      new = torch.tensor(np.array(gs), dtype=torch.float32, device=self.device)
    else:
      new = self._free_goal(idx, old=old, pos=pos)
    self.goal[idx] = new
    self.relocated[idx] = True
    self.goal_lost[idx] = True

  @property
  def goal_known(self):
    """The goal has been reached this episode and has not moved since."""
    return (self.n_goals >= 1) & ~self.goal_lost

  def reset_all(self):
    """Reset every env. With stagger=True the episode clocks start at random
    phases so episodes end at different times (statistics of those partial
    first episodes are not emitted)."""
    self._reset_envs(torch.ones(self.n, dtype=torch.bool, device=self.device))
    if self.stagger:
      self.t = torch.randint(0, self.episode_len, (self.n,), generator=self.gen,
                             device=self.device)
      self.partial[:] = True
    return self.observe()

  # ------------------------------------------------------------------ step
  def step(self, actions):
    """actions LongTensor [n] -> (obs, reward [n], done [n] bool, info)."""
    a = actions.to(self.device).long()
    tg = self._targets[a]                                               # [n,3]
    tf = tg[:, 0] * self.v_max + self.noise_v * self._randn(self.n)
    ts = tg[:, 1] * self.v_max + self.noise_v * self._randn(self.n)
    tr = tg[:, 2] * self.rot_max + self.noise_r * self._randn(self.n)
    k = 1.0 - math.exp(-self.dt / self.tau)
    self.vf = self.vf + (tf - self.vf) * k
    self.vs = self.vs + (ts - self.vs) * k
    self.vr = self.vr + (tr - self.vr) * k
    self.hd = wrap_pi(self.hd + self.vr * self.dt)
    c, s = torch.cos(self.hd), torch.sin(self.hd)
    step_xy = torch.stack([self.vf * c - self.vs * s,
                           self.vf * s + self.vs * c], 1) * self.dt
    lim = self.half - self.wall_margin
    target = self.pos + step_xy
    new = target.clamp(-lim, lim)
    hit_wall = (new != target).any(1)
    if self.has_rooms:
      # Walls / obstacle: move along x, then along y; a blocked axis keeps its
      # old coordinate, so the agent slides along the surface.
      bx = self._in_boxes(torch.stack([new[:, 0], self.pos[:, 1]], 1), None, self.wall_margin)
      nx = torch.where(bx, self.pos[:, 0], new[:, 0])
      by = self._in_boxes(torch.stack([nx, new[:, 1]], 1), None, self.wall_margin)
      ny = torch.where(by, self.pos[:, 1], new[:, 1])
      new = torch.stack([nx, ny], 1)
      hit_wall = hit_wall | bx | by
    self.vf = torch.where(hit_wall, self.vf * 0.2, self.vf)
    self.vs = torch.where(hit_wall, self.vs * 0.2, self.vs)
    disp = new - self.pos
    dist = disp.norm(dim=1)
    self.travelled = self.travelled + dist
    self.pos = new
    moving = dist > self.v_min * self.dt
    self.mdir = torch.where(moving[:, None], disp / dist.clamp_min(1e-9)[:, None],
                            self.mdir)
    vel = disp / self.dt

    # Goal pickup.
    at_goal = (self.pos - self.goal).norm(dim=1) < self.goal_radius
    reward = at_goal.float() * self.goal_reward
    hit_pos, hit_dir, hit_vel = self.pos.clone(), self.mdir.clone(), vel.clone()
    hit_goal, hit_spawn, hit_trav = self.goal.clone(), self.spawn.clone(), self.travelled.clone()
    known = self.goal_known              # n_goals >= 1, except after a relocation
    straight = ((self.goal - self.spawn).norm(dim=1) - self.goal_radius).clamp_min(0.0)
    eff = (straight / self.travelled.clamp_min(1e-6)).clamp(0.0, 1.0)
    rec = at_goal & known
    self.path_eff_sum = self.path_eff_sum + torch.where(rec, eff, torch.zeros_like(eff))
    self.path_eff_n = self.path_eff_n + rec.float()
    first = at_goal & (self.first_goal_t < 0)
    self.first_goal_t = torch.where(first, self.t, self.first_goal_t)
    self.n_goals = self.n_goals + at_goal.long()
    self.ep_return = self.ep_return + reward
    self.goal_lost = self.goal_lost & ~at_goal
    post = at_goal & self.relocated
    self.reloc_first_t = torch.where(post & (self.reloc_first_t < 0), self.t, self.reloc_first_t)
    self.n_goals_post = self.n_goals_post + post.long()

    # Episode end (auto-reset: the returned obs is the new episode's first).
    self.t = self.t + 1
    done = self.t >= self.episode_len
    if self.has_rooms:
      self._relocate(self.reloc & ~self.relocated & (self.t >= self.reloc_t) & ~done)
    self._respawn(at_goal & ~done)
    vel = torch.where(at_goal[:, None], torch.zeros_like(vel), vel)
    ep_stats = None
    if bool(done.any()):
      full = done & ~self.partial
      if bool(full.any()):
        idx = full.nonzero().squeeze(1)
        ep_stats = dict(
            env=idx.clone(),
            ret=self.ep_return[idx].clone(),
            n_goals=self.n_goals[idx].clone(),
            first_goal_t=self.first_goal_t[idx].clone(),
            path_eff=(self.path_eff_sum[idx] / self.path_eff_n[idx].clamp_min(1.0)).clone(),
            path_eff_n=self.path_eff_n[idx].clone(),
            room=self.room_id[idx].clone(),
            n_goals_post=self.n_goals_post[idx].clone(),
            reloc_first_t=self.reloc_first_t[idx].clone())
        self.episodes_done += int(idx.numel())
      self._reset_envs(done)
      vel = torch.where(done[:, None], torch.zeros_like(vel), vel)

    obs = self.observe(vel)
    info = dict(hit=at_goal, hit_pos=hit_pos, hit_dir=hit_dir, hit_vel=hit_vel,
                hit_wall=hit_wall, ep_stats=ep_stats, hit_goal=hit_goal,
                hit_spawn=hit_spawn, hit_trav=hit_trav, hit_known=known)
    return obs, reward, done, info

  # ----------------------------------------------------------- observation
  def observe(self, vel=None):
    if vel is None:
      vel = torch.zeros(self.n, 2, device=self.device)
    return dict(rgb=self.render(), pos=self.pos.clone(), hd=self.hd.clone(),
                vel=vel, dir=self.mdir.clone())

  def render(self):
    if self.vision == 'raycast':
      img = self._render_raycast()
    elif self.vision == 'none':
      return torch.zeros(self.n, 3, self.img, self.img, device=self.device)
    else:
      img = self._render_gradient(self.vision == 'sighted')
    if self.img_noise > 0:
      img = img + self.img_noise * self._randn(*img.shape)
    return img.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()        # [n,3,H,W]

  def _render_gradient(self, sighted):
    """fake_env.py's synthetic image: gradients keyed on (u, v, heading)."""
    H = self.half
    if sighted:
      u = 0.5 * (self.pos[:, 0] / H + 1.0)
      v = 0.5 * (self.pos[:, 1] / H + 1.0)
    else:
      u = torch.full((self.n,), 0.5, device=self.device)
      v = torch.full((self.n,), 0.5, device=self.device)
    h = 0.5 * (self.hd / math.pi + 1.0)
    gu, gv = self._gu[None], self._gv[None]
    r = 0.5 * gu + 0.5 * u[:, None, None]
    g = 0.5 * gv + 0.5 * v[:, None, None]
    b = 0.5 * h[:, None, None] + 0.5 * gu * gv
    return torch.stack([r, g, b], -1)                                   # [n,H,W,3]

  def _render_raycast(self):
    H = self.half
    ang = self.hd[:, None] + self._col_ang[None, :]                     # [n,W]
    dx, dy = torch.cos(ang), torch.sin(ang)
    x, y = self.pos[:, :1], self.pos[:, 1:]
    big = torch.full_like(dx, 1e9)
    tx = torch.where(dx > 1e-9, (H - x) / dx, torch.where(dx < -1e-9, (-H - x) / dx, big))
    ty = torch.where(dy > 1e-9, (H - y) / dy, torch.where(dy < -1e-9, (-H - y) / dy, big))
    hit_x = tx < ty
    t = torch.where(hit_x, tx, ty)
    zero = torch.zeros_like(dx, dtype=torch.long)
    one, two, three = zero + 1, zero + 2, zero + 3
    wall = torch.where(hit_x, torch.where(dx > 0, zero, two), torch.where(dy > 0, one, three))
    s_along = torch.where(hit_x, y + t * dy, x + t * dx)                # [-H, H] along the wall
    depth = (t * torch.cos(self._col_ang)[None, :]).clamp_min(0.05)     # perpendicular distance
    wall_h = (0.45 / depth).clamp(0.0, 1.0)                              # projected half-height
    base = self._wall_col[wall]                                          # [n,W,3]
    s_norm = (s_along + H) / (2.0 * H)
    phase = torch.gather(self.stripe_phase, 1, wall)                     # [n,W]
    stripe = ((s_norm * 6.0 + phase) % 1.0) < 0.15
    col = torch.where(stripe[..., None], base * 0.45, base)
    land = (s_along - torch.gather(self.land_pos, 1, wall)).abs() < 0.12
    col = torch.where(land[..., None], torch.ones_like(col), col)
    if self.has_rooms:
      # Nearest partition segment / obstacle face in front of the outer wall
      # (ray-rectangle slab test, [n,N_BOX,W]); plain colour, the faces
      # normal to y a little darker so the block reads as a solid.
      b = self.boxes
      tiny = torch.full_like(dx, 1e-9)
      idx_ = 1.0 / torch.where(dx.abs() < 1e-9, tiny, dx)[:, None, :]
      idy_ = 1.0 / torch.where(dy.abs() < 1e-9, tiny, dy)[:, None, :]
      tx1, tx2 = (b[:, :, 0, None] - x[:, None, :]) * idx_, (b[:, :, 1, None] - x[:, None, :]) * idx_
      ty1, ty2 = (b[:, :, 2, None] - y[:, None, :]) * idy_, (b[:, :, 3, None] - y[:, None, :]) * idy_
      tnx, tny = torch.minimum(tx1, tx2), torch.minimum(ty1, ty2)
      tn = torch.maximum(tnx, tny)
      tf_ = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
      tb = torch.where((tn < tf_) & (tn > 0), tn, torch.full_like(tn, 1e9))
      tbmin, kmin = tb.min(1)                                              # [n,W]
      xface = torch.gather(tnx > tny, 1, kmin[:, None, :]).squeeze(1)
      isbox = tbmin < t
      bcol = self._box_col[kmin]
      bcol = torch.where(xface[..., None], bcol, bcol * 0.8)
      col = torch.where(isbox[..., None], bcol, col)
      t = torch.where(isbox, tbmin, t)
      depth = (t * torch.cos(self._col_ang)[None, :]).clamp_min(0.05)
      wall_h = (0.45 / depth).clamp(0.0, 1.0)
    shade = (1.0 / (1.0 + 0.5 * depth)).clamp(0.25, 1.0)
    col = col * shade[..., None]
    v = self._v[None, :, None]                                           # [1,H,1]
    is_wall = v.abs() < wall_h[:, None, :]                               # [n,H,W]
    floor_shade = 0.25 + 0.45 * v.clamp(0.0, 1.0)                        # [1,H,1]
    bg = torch.where((v < 0)[..., None], self._ceil,
                     self._floor * floor_shade[..., None])               # [1,H,1,3]
    img = torch.where(is_wall[..., None], col[:, None, :, :], bg)        # [n,H,W,3]
    return img
