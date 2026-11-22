"""Test one trained agent (frozen weights) in the familiar room and in every
changed room, and record what its LSTM does there.

For each room condition: `--episodes` episodes run in parallel, one per
environment, with layouts paired across agents (same goals, spawns, wall cues
and relocated goals for every agent, evaluation seed 424242). Recorded per
condition:

  score            reward per episode (10 per goal arrival)
  first_goal_step  decision step of the first arrival (episode length if none)
  post_rate        arrivals after the first per 1000 steps left in the episode
  steps_per_return decision steps from a respawn to the arrival, goal known
  path_eff         shortest walkable path (around the wall / obstacle, minus
                   the goal radius) over the distance travelled, goal known;
                   mean per episode, then over episodes
  wall_contact     fraction of steps pushing against a wall or the obstacle
  relocation       arrivals before / after the goal moved, per episode (the
                   analysis restricts them to episodes that had found the
                   goal before it moved), and perseveration at the old goal
                   as in water-maze probe trials (Maei et al. 2009; Gallagher
                   et al. 1993): over the steps from the move until the moved
                   goal is first reached, the share spent within 0.45 m of the
                   old goal, the same for a control zone at the mirror image
                   of the old goal through the room centre, and the mean
                   distance to the old goal
  PCA              eigenvalues of the covariance of the 256 LSTM outputs h_t
                   over all steps of all episodes, and separately over the
                   steps before / after the goal is known; var10 = share of
                   variance in the first 10 components. (The headline PCA uses
                   the scripted replay of scripts/room/replay_pca.py, where
                   every network sees the same inputs.)

  python -m scripts.room.eval_rooms --run runs_ext/pgc33_lstm_s1 --tag 1000M
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.common import CodeSource, build_env, build_policy, build_population, extra_from_obs  # noqa: E402
from code.env import TEST_CONDITIONS, room_spec  # noqa: E402
from scripts.room.geo import Room  # noqa: E402

N_TRAJ = 6       # environments whose paths are kept for the example figure
ZONE = 0.45      # radius (m) of the zone around the old goal used for perseveration (1.5 goal radii)


def spectrum(S1, S2, N):
  mu = S1 / N
  cov = S2 / N - torch.outer(mu, mu)
  ev = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0.0)
  return ev.cpu().numpy(), cov.cpu().numpy(), mu.cpu().numpy()


def pca_summary(ev):
  tot = float(ev.sum())
  cum = np.cumsum(ev) / max(tot, 1e-30)
  return dict(var10=float(cum[9]), var5=float(cum[4]), var20=float(cum[19]),
              n90=int(np.searchsorted(cum, 0.90) + 1),
              participation_ratio=float(tot ** 2 / max(float((ev ** 2).sum()), 1e-30)),
              total_var=tot)


def run_condition(cfg, policy, pop, dev, cond, episodes, seed=424242, probe='intact'):
  torch.manual_seed(seed)
  n = episodes
  env = build_env(cfg, n, seed=seed, device=dev, stagger=False, paired=True,
                  rooms=None if cond == 'open' else [cond])
  code_src = CodeSource(cfg, pop, dev, probe=probe)
  obs = env.reset_all()
  state = policy.zero_state(n, dev)
  goal_code = torch.zeros(n, cfg['n_cells'], device=dev)
  prev_action = torch.full((n,), -1, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)
  nh = cfg['nh']
  z64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
  acc = {k: [z64(nh), z64(nh, nh)] for k in ('h', 'c')}
  # the same for h_t split by whether the goal location is known at that step
  # (searching vs returning), so a difference in the spectrum can be told apart
  # from a difference in how long an agent spends in each phase
  ph = {k: [z64(nh), z64(nh, nh), 0.0] for k in ('search', 'return')}
  leg = torch.zeros(n, dtype=torch.long, device=dev)
  wall = torch.zeros(n, device=dev)
  old_in, ctl_in, old_dist, n_lost = (torch.zeros(n, device=dev) for _ in range(4))   # after a goal move
  ev_rows, traj = [], [obs['pos'][:N_TRAJ].cpu().numpy()]
  goal0 = env.goal.clone()
  goal_moved_to = None
  es = None
  T = cfg['episode_len']
  with torch.no_grad():
    for t in range(T):
      code = code_src(obs['pos'], obs['dir'], obs['vel'])
      logits, _, state = policy.step(obs['rgb'], code, goal_code, prev_action, prev_reward,
                                     state, extra_from_obs(cfg, obs))
      for k, x in (('h', state[0]), ('c', state[1])):
        xd = x.double()
        acc[k][0] += xd.sum(0)
        acc[k][1] += xd.T @ xd
      hd_, kn = state[0].double(), env.goal_known
      for k, m in (('search', ~kn), ('return', kn)):
        xm = hd_ * m[:, None].double()
        ph[k][0] += xm.sum(0)
        ph[k][1] += xm.T @ xm
        ph[k][2] += float(m.sum())
      a = torch.distributions.Categorical(logits=logits).sample()
      moved = env.relocated.clone()
      obs, r_raw, done, info = env.step(a)
      r = r_raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else r_raw
      leg += 1
      wall += info['hit_wall'].float()
      if env.has_rooms and bool(env.goal_lost.any()):
        m = (env.goal_lost & ~done).float()          # goal has moved and has not been reached since
        d_old = (obs['pos'] - goal0).norm(dim=1)
        old_in += m * (d_old < ZONE).float()
        ctl_in += m * ((obs['pos'] + goal0).norm(dim=1) < ZONE).float()
        old_dist += m * d_old
        n_lost += m
      hit = info['hit']
      if bool(hit.any()):
        g_hit = code_src.goal(info['hit_pos'], info['hit_dir'], info['hit_vel'])
        goal_code = torch.where(hit[:, None], g_hit, goal_code)
        i = hit.nonzero().squeeze(1)
        ev_rows.append(torch.cat([i[:, None].float(), torch.full((i.numel(), 1), float(t), device=dev),
                                  info['hit_known'][i, None].float(), moved[i, None].float(),
                                  leg[i, None].float(), info['hit_trav'][i, None],
                                  info['hit_spawn'][i], info['hit_goal'][i]], 1).cpu())
        leg[i] = 0
      prev_action, prev_reward = a, r
      if t < T - 1:
        traj.append(obs['pos'][:N_TRAJ].cpu().numpy())
      if goal_moved_to is None and env.has_rooms and t < T - 1 and bool(env.relocated.all()):
        goal_moved_to = env.goal.clone()
      if info['ep_stats'] is not None:
        es = {k: v.cpu().numpy() for k, v in info['ep_stats'].items()}
  assert es is not None and len(es['ret']) == n, 'every episode must end on the last step'
  order = np.argsort(es['env'])
  es = {k: v[order] for k, v in es.items()}
  E = torch.cat(ev_rows).numpy().astype(np.float64) if ev_rows else np.zeros((0, 10))
  # ---- per-episode navigation statistics
  ret, ng, fg = es['ret'].astype(float), es['n_goals'].astype(int), es['first_goal_t'].astype(int)
  found = fg >= 0
  first = np.where(found, fg, T)
  post = (ng[found] - 1) / np.maximum(T - fg[found], 1) * 1000.0
  sp = room_spec(cond, env.half)
  room = Room(sp['boxes'], margin=env.wall_margin, lim=env.half - env.wall_margin)
  known = E[:, 2] > 0.5
  K = E[known]
  geo = room.shortest(K[:, 6:8], K[:, 8:10]) if len(K) else np.zeros(0)
  assert np.isfinite(geo).all(), 'a return without a walkable path'
  eff = np.clip(np.clip(geo - env.goal_radius, 0, None) / np.maximum(K[:, 5], 1e-6), 0, 1)
  eff_e = np.clip(np.clip(np.linalg.norm(K[:, 6:8] - K[:, 8:10], axis=1) - env.goal_radius, 0, None)
                  / np.maximum(K[:, 5], 1e-6), 0, 1)
  ep_eff, ep_eff_e, ep_steps = [], [], []
  ke = K[:, 0].astype(int)
  for e in np.unique(ke):
    m = ke == e
    ep_eff.append(float(eff[m].mean()))
    ep_eff_e.append(float(eff_e[m].mean()))
    ep_steps.append(float(K[m, 4].mean()))
  sem = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')
  mean = lambda x: float(np.mean(x)) if len(x) else float('nan')
  out = dict(cond=cond, layout=sp['layout'], reloc=sp['reloc'], episodes=n,
             score=mean(ret), sem_score=sem(ret), goals=mean(ng),
             frac_found=mean(found), first_goal_step=mean(first),
             post_rate=mean(post), frac_two_goals=mean(ng >= 2),
             steps_per_return=mean(ep_steps), path_eff=mean(ep_eff), path_eff_straight=mean(ep_eff_e),
             n_returns=int(len(K)), wall_contact=float(wall.sum().item() / (n * T)),
             env_path_eff=mean(es['path_eff'][es['path_eff_n'] > 0]),
             scores=[float(x) for x in ret], first_goal_steps=[int(x) for x in first])
  if sp['reloc']:
    gp = es['n_goals_post'].astype(int)
    rf = es['reloc_first_t'].astype(int)
    left = T - env.reloc_t
    nl = n_lost.clamp_min(1.0)
    out.update(goals_pre=mean(ng - gp), goals_post=mean(gp), frac_refound=mean(rf >= 0),
               relearn_steps=mean(np.where(rf >= 0, rf - env.reloc_t, left)),
               goals_post_list=[int(x) for x in gp],
               old_goal_dwell=[float(x) for x in (old_in / nl).cpu().numpy()],
               control_zone_dwell=[float(x) for x in (ctl_in / nl).cpu().numpy()],
               old_goal_dist=[float(x) for x in (old_dist / nl).cpu().numpy()],
               old_goal_centre_dist=[float(x) for x in goal0.norm(dim=1).cpu().numpy()])
  arrays = dict(events=E.astype(np.float32), traj=np.stack(traj).astype(np.float32),
                traj_goal=goal0[:N_TRAJ].cpu().numpy())
  if goal_moved_to is not None:
    arrays['traj_goal_moved'] = goal_moved_to[:N_TRAJ].cpu().numpy()
  for k in ('h', 'c'):
    ev, cov, mu = spectrum(acc[k][0], acc[k][1], float(n * T))
    out[f'pca_{k}'] = pca_summary(ev)
    out[f'eig_{k}'] = [float(x) for x in ev]
    arrays[f'cov_{k}'] = cov.astype(np.float32)
    arrays[f'mu_{k}'] = mu.astype(np.float32)
  for k in ('search', 'return'):
    out[f'frac_{k}'] = ph[k][2] / float(n * T)
    if ph[k][2] > 1000:
      ev, _, _ = spectrum(ph[k][0], ph[k][1], ph[k][2])
      out[f'pca_h_{k}'] = pca_summary(ev)
  return out, arrays


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--run', required=True)
  ap.add_argument('--ckpt', default='final', help="'final' or a checkpoint path")
  ap.add_argument('--tag', required=True, help="label of the agent state, e.g. 1000M, 1250M_room")
  ap.add_argument('--name', default=None, help='agent name in the output (default: run directory name)')
  ap.add_argument('--conds', default=','.join(TEST_CONDITIONS))
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--probe', default='intact',
                  help="input lesion applied during the test (code/common.py PROBES), e.g. no_pgc, "
                       "sham0, no_code; written to <agent>@<tag>+<probe>.json")
  ap.add_argument('--out', default=os.path.join(ROOT, 'results_room', 'eval'))
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  name = args.name or os.path.basename(os.path.normpath(args.run))
  os.makedirs(args.out, exist_ok=True)
  stem = f'{name}@{args.tag}' + ('' if args.probe == 'intact' else f'+{args.probe}')
  jpath = os.path.join(args.out, stem + '.json')
  if os.path.exists(jpath) and not args.force:
    print('exists', jpath)
    return
  cfg = json.load(open(os.path.join(args.run, 'config.json')))
  dev = torch.device(args.device)
  ck_path = os.path.join(args.run, 'ckpt_final.pt') if args.ckpt == 'final' else args.ckpt
  ck = torch.load(ck_path, map_location='cpu')
  torch.backends.cudnn.benchmark = False       # same convolution algorithm in every process
  torch.backends.cudnn.deterministic = True
  pop = build_population(cfg, dev)
  policy = build_policy(cfg, dev)
  policy.load_state_dict(ck['policy'])
  policy.eval()
  res = dict(name=name, tag=args.tag, run=args.run, ckpt=ck_path, frames=int(ck.get('frames', -1)),
             cond=cfg['cond'], seed=cfg['seed'], rooms_trained=cfg.get('rooms', 'none'),
             episodes=args.episodes, probe=args.probe, conditions={})
  arrays = {}
  t0 = time.time()
  for cond in args.conds.split(','):
    out, arr = run_condition(cfg, policy, pop, dev, cond, args.episodes, probe=args.probe)
    res['conditions'][cond] = out
    for k, v in arr.items():
      arrays[f'{cond}/{k}'] = v
    print(f'{name}@{args.tag} {cond:7s} score {out["score"]:6.1f} first {out["first_goal_step"]:6.0f} '
          f'eff {out["path_eff"]:.3f} wall {out["wall_contact"]:.3f} var10 {out["pca_h"]["var10"]:.3f} '
          f'({time.time() - t0:.0f}s)', flush=True)
  json.dump(res, open(jpath, 'w'))
  np.savez_compressed(os.path.join(args.out, stem + '.npz'), **arrays)


if __name__ == '__main__':
  main()
