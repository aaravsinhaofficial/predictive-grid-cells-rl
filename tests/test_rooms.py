"""Changed rooms (code/env.py ROOMS) and the shortest-path helper (scripts/room/geo.py)."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code import env as E  # noqa: E402
from scripts.room.geo import Room  # noqa: E402

DEV = 'cpu'


def inside(p, boxes, clear):
  x, y = p[:, 0:1], p[:, 1:2]
  b = boxes[None]
  return ((x > b[..., 0] - clear + 1e-6) & (x < b[..., 1] + clear - 1e-6) &
          (y > b[..., 2] - clear + 1e-6) & (y < b[..., 3] + clear - 1e-6)).any(1)


@pytest.mark.parametrize('name', E.TEST_CONDITIONS)
def test_walls_block_and_goals_are_clear(name):
  n = 48
  env = E.ArenaEnv(n, device=DEV, seed=3, rooms=[name], episode_len=150, stagger=False, vision='none')
  env.reset_all()
  bx = torch.tensor(E.room_spec(name)['boxes'])
  g = torch.Generator().manual_seed(1)
  for t in range(320):
    fwd = torch.rand(n, generator=g) < 0.7
    a = torch.where(fwd, torch.full((n,), 2), torch.randint(0, 6, (n,), generator=g))
    env.step(a)
    assert not inside(env.pos, bx, env.wall_margin).any()
    assert not inside(env.goal, bx, env.goal_radius).any()
    assert (env.pos.abs() <= env.half - env.wall_margin + 1e-6).all()


def test_default_room_has_no_boxes_and_open_matches_it():
  kw = dict(device=DEV, seed=5, paired=True, pair_seed=424242, stagger=False, episode_len=40)
  a, b = E.ArenaEnv(6, **kw), E.ArenaEnv(6, rooms=['open'], **kw)
  oa, ob = a.reset_all(), b.reset_all()
  assert not a.has_rooms and b.has_rooms
  assert torch.equal(oa['rgb'], ob['rgb'])
  g = torch.Generator().manual_seed(0)
  for t in range(90):
    act = torch.randint(0, 6, (6,), generator=g)
    oa, ra, da, _ = a.step(act)
    ob, rb, db, _ = b.step(act)
    assert torch.equal(oa['rgb'], ob['rgb']) and torch.equal(oa['pos'], ob['pos']) and torch.equal(ra, rb)


def test_goal_relocates_once_at_mid_episode():
  n = 32
  env = E.ArenaEnv(n, device=DEV, seed=2, rooms=['D1+R'], episode_len=100, stagger=False, vision='none')
  env.reset_all()
  g0 = env.goal.clone()
  stay = torch.zeros(n, dtype=torch.long)
  for t in range(49):
    env.step(stay)
  assert not env.relocated.any() and torch.equal(env.goal, g0)
  env.step(stay)
  g1 = env.goal.clone()
  assert env.relocated.all() and ((g1 - g0).norm(dim=1) >= E.RELOC_MIN).all()
  for t in range(40):
    env.step(stay)
  assert torch.equal(env.goal, g1)


def test_paired_rooms_share_goals_and_spawns():
  def start(seed):
    env = E.ArenaEnv(8, device=DEV, seed=seed, paired=True, pair_seed=7, rooms=['DO2'], stagger=False, vision='none')
    env.reset_all()
    return env.goal.clone(), env.pos.clone()
  (ga, pa), (gb, pb) = start(1), start(2)
  assert torch.equal(ga, gb) and torch.equal(pa, pb)


def test_mixture_draws_every_condition():
  env = E.ArenaEnv(2100, device=DEV, seed=2, rooms='changed', vision='none')
  env.reset_all()
  assert len(env.room_names) == 21
  assert torch.bincount(env.room_id, minlength=21).min() > 40
  assert abs(float(env.reloc.float().mean()) - 11 / 21) < 0.05


def test_walls_are_drawn():
  env = E.ArenaEnv(1, device=DEV, seed=1, rooms=['D1'], img_noise=0.0)
  env.reset_all()
  env.pos[0] = torch.tensor([-0.5, 0.8])
  env.hd[0] = 0.0                       # facing the partition, 0.45 m away
  img = env.render()[0]
  centre = img[:, 42, 42]
  shade = 1.0 / (1.0 + 0.5 * 0.45)
  assert torch.allclose(centre, torch.tensor(E.PART_COLOR) * shade, atol=0.02)


def test_shortest_path():
  open_room = Room(E.room_spec('open')['boxes'])
  S, G = np.array([[-1.0, 0.3], [0.2, 0.2]]), np.array([[1.0, -0.5], [0.9, -1.0]])
  assert np.allclose(open_room.shortest(S, G), np.linalg.norm(S - G, axis=1))
  d1 = Room(E.room_spec('D1')['boxes'], margin=0.08, lim=1.17)
  # through the central doorway: corners of the grown wall at x = +-0.13, y = 0.22
  assert math.isclose(d1.shortest([[-1.0, 0.8]], [[1.0, 0.8]])[0], 2 * math.hypot(0.87, 0.58) + 0.26, rel_tol=1e-9)
  assert math.isclose(d1.shortest([[-1.0, 0.0]], [[1.0, 0.0]])[0], 2.0, rel_tol=1e-9)
