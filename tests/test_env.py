import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popcode.env import NUM_ACTIONS, ArenaEnv  # noqa: E402
from popcode.nets import PolicyNet  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_shapes_and_bounds():
  env = ArenaEnv(8, episode_len=20, seed=0, device=DEV, stagger=False)
  obs = env.reset_all()
  assert obs['rgb'].shape == (8, 3, 84, 84)
  assert obs['rgb'].min() >= 0 and obs['rgb'].max() <= 1
  lim = env.half - env.wall_margin
  for _ in range(25):
    a = torch.randint(0, NUM_ACTIONS, (8,), device=DEV)
    obs, r, d, info = env.step(a)
    assert obs['pos'].abs().max() <= lim + 1e-6
    assert obs['hd'].abs().max() <= math.pi + 1e-6
    assert torch.allclose(obs['dir'].norm(dim=1), torch.ones(8, device=DEV), atol=1e-4)
    assert r.shape == (8,) and d.shape == (8,)
  assert env.episodes_done == 8  # one full episode of 20 steps each


def test_stagger_hides_partial_first_episodes():
  env = ArenaEnv(16, episode_len=50, seed=4, device=DEV, stagger=True)
  env.reset_all()
  assert len(set(env.t.tolist())) > 1, 'episode clocks should be staggered'
  emitted = 0
  for step in range(120):
    _, _, d, info = env.step(torch.randint(0, NUM_ACTIONS, (16,), device=DEV))
    if info['ep_stats'] is not None:
      emitted += info['ep_stats']['env'].numel()
      assert step >= 49, 'no statistics may come from a partial first episode'
  # every env has completed at least one FULL episode by step 120 (50 + 50)
  assert emitted >= 16 and env.episodes_done == emitted


def test_forward_moves_along_heading_and_goal_rewards():
  env = ArenaEnv(4, episode_len=1000, seed=1, device=DEV, v_max=0.35, stagger=False)
  env.noise_v = env.noise_r = 0.0
  env.reset_all()
  env.pos[:] = 0.0
  env.hd[:] = 0.0
  env.goal[:] = torch.tensor([0.8, 0.0], device=DEV)
  env.spawn[:] = 0.0
  env.travelled[:] = 0.0
  env.n_goals[:] = 1  # pretend the goal is known so path efficiency is logged
  total = 0.0
  for t in range(60):
    obs, r, d, info = env.step(torch.full((4,), 2, dtype=torch.long, device=DEV))  # forward
    total += float(r.sum())
    if float(r.sum()) > 0:
      break
  assert total == 4 * 10.0, 'all four agents reach the goal straight ahead'
  assert float((env.path_eff_sum / env.path_eff_n).min()) > 0.9
  assert bool(info['hit'].all())
  assert torch.allclose(info['hit_dir'], torch.tensor([[1.0, 0.0]], device=DEV).expand(4, 2), atol=1e-3)
  # respawned away from the goal with zero velocity; obs velocity is zero
  assert float(((env.pos - env.goal).norm(dim=1)).min()) > env.goal_radius
  assert float(env.vf.abs().max()) == 0.0
  assert float(obs['vel'].abs().max()) == 0.0


def test_turn_in_place_keeps_motion_direction():
  """Motor noise jitters the position by < 1 mm per step while turning; the
  motion direction must hold the last real travel direction (not noise)."""
  env = ArenaEnv(32, seed=2, device=DEV, stagger=False)
  env.reset_all()
  env.pos[:] = 0.0
  env.hd[:] = 0.0
  env.goal[:] = torch.tensor([-1.0, -1.0], device=DEV)   # out of the way
  for _ in range(10):
    obs, _, _, _ = env.step(torch.full((32,), 2, dtype=torch.long, device=DEV))  # forward
  d_before = obs['dir'].clone()
  assert float((d_before - torch.tensor([1.0, 0.0], device=DEV)).norm(dim=1).max()) < 0.2
  # The forward velocity decays with tau = 0.25 s while the heading turns, so
  # the motion direction legitimately follows the curved path for ~7 steps;
  # once the speed is below v_min it must hold while the heading keeps turning.
  for _ in range(20):
    obs, _, _, _ = env.step(torch.zeros(32, dtype=torch.long, device=DEV))  # turn_left
  d_hold, hd_hold = obs['dir'].clone(), obs['hd'].clone()
  assert float(env.vf.abs().max()) < env.v_min
  for _ in range(20):
    obs, _, _, _ = env.step(torch.zeros(32, dtype=torch.long, device=DEV))  # turn_left
  assert torch.allclose(obs['dir'], d_hold), 'direction must hold while turning in place'
  from popcode.env import wrap_pi
  assert float(wrap_pi(obs['hd'] - hd_hold).abs().min()) > 0.5, 'heading keeps turning'


def test_textures_randomised_per_episode():
  env = ArenaEnv(6, episode_len=5, seed=3, device=DEV, img_noise=0.0, stagger=False)
  env.reset_all()
  lp0 = env.land_pos.clone()
  assert not torch.allclose(lp0[0], lp0[1]), 'different envs get different layouts'
  for _ in range(5):
    env.step(torch.zeros(6, dtype=torch.long, device=DEV))
  assert not torch.allclose(env.land_pos, lp0), 'layout re-drawn at episode end'
  fixed = ArenaEnv(6, seed=3, device=DEV, randomize_textures=False, stagger=False)
  fixed.reset_all()
  assert torch.allclose(fixed.land_pos[0], fixed.land_pos[5])


def test_raycast_wall_colours_follow_heading():
  """Facing +x the image centre must be the east (red) wall; facing +y green."""
  env = ArenaEnv(2, seed=3, device=DEV, img_noise=0.0, stagger=False)
  env.reset_all()
  env.pos[:] = 0.0
  env.hd[0] = 0.0
  env.hd[1] = math.pi / 2
  env.land_pos[:] = 10.0   # move landmarks out of view for this check
  img = env.render()                            # [2,3,84,84]
  centre = img[:, :, 42, 42]                    # [2,3]
  assert centre[0, 0] > centre[0, 1] and centre[0, 0] > centre[0, 2]   # red
  assert centre[1, 1] > centre[1, 0] and centre[1, 1] > centre[1, 2]   # green
  assert float(img[0, :, 0, :].mean()) < float(img[0, :, 83, :].mean())


def test_determinism():
  outs = []
  for _ in range(2):
    env = ArenaEnv(4, seed=11, device=DEV)
    obs = env.reset_all()
    g = torch.Generator(device=DEV)
    g.manual_seed(0)
    for _ in range(10):
      a = torch.randint(0, NUM_ACTIONS, (4,), device=DEV, generator=g)
      obs, _, _, _ = env.step(a)
    outs.append(obs['pos'].clone())
  assert torch.allclose(outs[0], outs[1])


def test_policy_prev_action_sentinel_and_embed_path():
  torch.manual_seed(0)
  net = PolicyNet(n_actions=6, n_code=16, nh=32, recurrent=True).to(DEV)
  rgb = torch.rand(3, 3, 84, 84, device=DEV)
  code = torch.rand(3, 16, device=DEV)
  st = net.zero_state(3, DEV)
  pa = torch.tensor([-1, 0, 5], device=DEV)
  x = net._concat(net.embed(rgb), code, code, pa, torch.zeros(3, device=DEV), None)
  onehot = x[:, 256 + 32:256 + 32 + 6]
  assert float(onehot[0].sum()) == 0.0 and float(onehot[1, 0]) == 1.0 and float(onehot[2, 5]) == 1.0
  l1, v1, _ = net.step(rgb, code, code, pa, torch.zeros(3, device=DEV), st)
  l2, v2, _ = net.step_from_embedding(net.embed(rgb), code, code, pa, torch.zeros(3, device=DEV), st)
  assert torch.allclose(l1, l2) and torch.allclose(v1, v2)


def test_paired_episode_layouts_are_policy_independent():
  """With paired=True the goal/spawn/texture sequence of each env slot depends
  only on (pair_seed, slot, episode, draw), not on the actions taken."""
  from popcode.env import ArenaEnv as E
  outs = []
  for actions_seed in (0, 1):
    env = E(4, episode_len=30, seed=actions_seed + 7, device=DEV, stagger=False,
            paired=True, pair_seed=99)
    env.reset_all()
    g = torch.Generator(device=DEV)
    g.manual_seed(actions_seed)
    goals, lands = [env.goal.clone()], [env.land_pos.clone()]
    for _ in range(35):
      env.step(torch.randint(0, NUM_ACTIONS, (4,), device=DEV, generator=g))
    goals.append(env.goal.clone())
    lands.append(env.land_pos.clone())
    outs.append((goals, lands))
  for a, b in zip(outs[0][0], outs[1][0]):
    assert torch.allclose(a, b)
  for a, b in zip(outs[0][1], outs[1][1]):
    assert torch.allclose(a, b)
  assert not torch.allclose(outs[0][0][0], outs[0][0][1]), 'new goal on the second episode'


def test_probe_code_sources():
  import numpy as np
  from popcode.common import CodeSource, build_population, resolve_cond, sham_indices
  cfg = resolve_cond(dict(cond='pgc33', pgc_frac=None, code_mode=None, code_seed=None, seed=3,
                          n_cells=128, n_modules=4, spacing_min=0.4, spacing_ratio=1.42,
                          pgc_shift=0.2, pgc_shift_mode='space', pgc_tau=0.5, pgc_shift_jitter=0.0,
                          sharpness=2.0, code_noise=0.0, arena_half=1.25, v_max=0.35,
                          goal_code_shift='intact'))
  pop = build_population(cfg, DEV)
  pos = torch.rand(5, 2, device=DEV) - 0.5
  d = torch.tensor([[1.0, 0.0]], device=DEV).expand(5, 2)
  v = d * 0.3
  intact = CodeSource(cfg, pop, DEV, 'intact')
  g, gstar = intact(pos, d, v), intact.goal(pos, d, v)
  assert torch.allclose(g, gstar)
  assert float(CodeSource(cfg, pop, DEV, 'no_current')(pos, d, v).abs().sum()) == 0
  assert torch.allclose(CodeSource(cfg, pop, DEV, 'no_current').goal(pos, d, v), gstar)
  assert float(CodeSource(cfg, pop, DEV, 'no_goal').goal(pos, d, v).abs().sum()) == 0
  sh = CodeSource(cfg, pop, DEV, 'sham2')
  assert sh.n_masked == round(0.33 * 128) == len(sham_indices(pop, 2))
  assert set(sham_indices(pop, 2)) <= set(pop.gc_idx.tolist()), 'sham drawn from grid cells'
  assert len(set(sham_indices(pop, 1)) ^ set(sham_indices(pop, 2))) > 0, 'draws differ'
  # lookahead goal code: all cells evaluated 20 cm ahead with zero PGC shift
  la = CodeSource(dict(cfg, goal_code_shift='lookahead'), pop, DEV, 'intact')
  expect = pop.activity(pos + 0.2 * d, dir=d, vel=v, shift_scale=0.0)
  assert torch.allclose(la.goal(pos, d, v), expect, atol=1e-5)
  # zero goal shift: PGC units in g* are plain grid cells
  z = CodeSource(dict(cfg, goal_code_shift='zero'), pop, DEV, 'intact')
  assert torch.allclose(z.goal(pos, d, v), pop.activity(pos, dir=d, vel=v, shift_scale=0.0), atol=1e-5)
  # fake_goal / shuffled_current return in-range codes that differ from intact
  fg = CodeSource(cfg, pop, DEV, 'fake_goal').goal(pos, d, v)
  assert fg.shape == gstar.shape and not torch.allclose(fg, gstar)
  sc = CodeSource(cfg, pop, DEV, 'shuffled_current')(pos, d, v)
  assert sc.shape == g.shape and not torch.allclose(sc, g) and sc.min() >= 0 and sc.max() <= 1
