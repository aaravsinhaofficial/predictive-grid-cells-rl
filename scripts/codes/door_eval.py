"""Door-room measures for trained code-only agents (frozen weights).

Runs 200 paired test episodes (the layouts of code/evaluate.py: same goals and
start places for every agent) in the agent's own room and splits every trip
to the goal by whether the start and the goal lie on the same side of the
door wall:

  score            reward per episode (10 per goal)
  wall_contact     share of steps spent pushing against a wall or the obstacle
  trips            every stretch from a start place to the goal or to the end
                   of the episode: was the goal known, did the trip need the
                   door, was the goal reached, how many steps it took
  reach_same / reach_door      share of trips with a known goal that reached it
  steps_same / steps_door      mean steps of the trips that reached it

  python -m scripts.codes.door_eval --runs 'runs_codes/dr/*_door_ffb_s*' --out results/code_variations/data/door_eval.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.common import CodeSource, build_env, build_policy, build_population, extra_from_obs  # noqa: E402
from code.env import ROOMS  # noqa: E402


def side_fn(layout):
  """Which side of the door wall a point is on (0 / 1); None without a wall."""
  wall = ROOMS[layout][0] if layout in ROOMS else None
  if wall is None:
    return None
  axis, off, _ = wall
  k = 0 if axis == 'x' else 1
  return lambda p: (p[:, k] > off).long()


def evaluate(run, episodes=200, n_envs=32, seed=424242, device='cuda:0'):
  cfg = json.load(open(os.path.join(run, 'config.json')))
  dev = torch.device(device if torch.cuda.is_available() else 'cpu')
  ck = torch.load(os.path.join(run, 'ckpt_final.pt'), map_location='cpu')
  pop = build_population(cfg, dev)
  stored = (cfg.get('population') or {}).get('lattice_hash')
  if stored and stored != pop.describe()['lattice_hash']:
    raise SystemExit(f'{run}: rebuilt cells differ from the cells the run was trained with')
  if 'shift' in ck:
    pop.set_shift(ck['shift'][:, 0], ck['shift'][:, 1])
  torch.manual_seed(cfg['seed'])
  policy = build_policy(cfg, dev)
  policy.load_state_dict(ck['policy'])
  policy.eval()
  torch.manual_seed(seed)
  env = build_env(cfg, n_envs, seed=seed, device=dev, stagger=False, paired=True)
  layout = (cfg.get('rooms') or 'none').split(',')[0]
  side = side_fn(layout)
  code_src = CodeSource(cfg, pop, dev)
  obs = env.reset_all()
  n = n_envs
  state = policy.zero_state(n, dev)
  goal_code = torch.zeros(n, cfg['n_cells'], device=dev)
  prev_action = torch.full((n,), -1, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)

  def needs_door():
    return (side(env.pos) != side(env.goal)) if side is not None else torch.zeros(n, dtype=torch.bool, device=dev)

  t0 = torch.zeros(n, dtype=torch.long, device=dev)          # step at which the current trip began
  cross, known = needs_door(), torch.zeros(n, dtype=torch.bool, device=dev)
  trips, scores, contact, steps = [], [], 0.0, 0
  ep_done = 0
  with torch.no_grad():
    while ep_done < episodes:
      code = code_src(obs['pos'], obs['dir'], obs['vel'])
      logits, _, state = policy.step(obs['rgb'], code, goal_code, prev_action, prev_reward, state,
                                     extra_from_obs(cfg, obs))
      a = torch.distributions.Categorical(logits=logits).sample()
      t_before = env.t.clone()
      obs, r_raw, done, info = env.step(a)
      r = r_raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else r_raw
      contact += float(info['hit_wall'].float().sum())
      steps += n
      hit = info['hit']
      if bool(hit.any()):
        g_hit = code_src.goal(info['hit_pos'], info['hit_dir'], info['hit_vel'])
        goal_code = torch.where(hit[:, None], g_hit, goal_code)
      ended = hit | done
      if bool(ended.any()):
        for j in ended.nonzero().squeeze(1).tolist():
          trips.append((bool(known[j]), bool(cross[j]), bool(hit[j]), int(t_before[j] + 1 - t0[j])))
        t0 = torch.where(done, torch.zeros_like(t0), torch.where(hit, t_before + 1, t0))
        known = torch.where(done, torch.zeros_like(known), known | hit)
        cross = torch.where(ended, needs_door(), cross)       # new start place (and new goal after an episode end)
      prev_action, prev_reward = a, r
      if bool(done.any()):
        keep = (~done).float()
        goal_code = goal_code * keep[:, None]
        state = (state[0] * keep[:, None], state[1] * keep[:, None])
        prev_action = torch.where(done, torch.full_like(a, -1), a)
        prev_reward = prev_reward * keep
        es = info['ep_stats']
        if es is not None:
          scores += [float(x) for x in es['ret']]
          ep_done = len(scores)
  T = np.array(trips, dtype=float).reshape(-1, 4)

  def part(kn, cr):
    m = (T[:, 0] == kn) & (T[:, 1] == cr)
    reached = m & (T[:, 2] == 1)
    return dict(n=int(m.sum()), reach=float(T[m, 2].mean()) if m.any() else None,
                steps=float(T[reached, 3].mean()) if reached.any() else None)

  return dict(run=os.path.basename(run), cond=cfg['cond'], seed=cfg['seed'], room=layout, frames=int(ck.get('frames', 0)),
              episodes=len(scores[:episodes]), score=float(np.mean(scores[:episodes])),
              wall_contact=contact / max(steps, 1),
              known_same=part(1, 0), known_door=part(1, 1), search_same=part(0, 0), search_door=part(0, 1))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--runs', required=True, help='glob of run directories')
  ap.add_argument('--out', required=True)
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--device', default='cuda:0')
  args = ap.parse_args()
  out = json.load(open(args.out)) if os.path.exists(args.out) else {}
  for run in sorted(glob.glob(args.runs)):
    name = os.path.basename(run)
    if name in out or not os.path.exists(os.path.join(run, 'ckpt_final.pt')):
      continue
    out[name] = evaluate(run, args.episodes, device=args.device)
    r = out[name]
    print(f'{name}: score {r["score"]:.1f} wall {r["wall_contact"]:.3f} reach same {r["known_same"]["reach"]} '
          f'door {r["known_door"]["reach"]}', flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
  main()
