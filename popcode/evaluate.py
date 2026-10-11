"""Frozen-policy evaluation (SI protocol: mean score over N episodes with the
stochastic policy acting exactly as in training) plus input probes.

Probes (applied at evaluation only, same weights, same env seeds):
  intact     reference
  no_code    both the current code and the goal code zeroed  (reliance guard:
             if this costs nothing, the policy ignores the code and any
             gc-vs-pgc difference is uninterpretable)
  no_pgc     only the predictive cells zeroed (in g_t and g*)
  no_gc      only the plain grid cells zeroed
  pgc_as_gc  predictive cells evaluated with zero shift (they become ordinary
             grid cells): tests whether the policy exploits the SHIFT itself
  no_goal    goal code zeroed (the paper's Extended Data Fig. 6c lesion)
  untrained  the policy at its random initialisation: the chance line that
             every score must be judged against (an untrained network is not a
             uniform random walker)

Also reports goals/episode, step of first goal (censored at episode_len when
no goal was found; see frac_found_goal), and path efficiency of goal returns
after the goal is known (the paper's goal-homing metric; only episodes with
>= 2 goals contribute).

Output files carry their provenance: eval/<probe>.json is written only for the
final checkpoint at the canonical episode count; anything else is written as
eval/<probe>@<frames>f_<episodes>ep.json and ignored by analyze.py.

  python -m popcode.evaluate --run runs/pgc33_lstm_s1 --probes intact,no_code
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popcode.common import (PROBES, CodeSource, build_env, build_policy,  # noqa: E402
                            build_population, extra_from_obs)

CANONICAL_EPISODES = 200


def run_eval(cfg, policy, pop, dev, probe='intact', episodes=200, n_envs=32,
             seed=424242, greedy=False, code_noise=None, paired=True):
  torch.manual_seed(seed)
  n = n_envs
  env = build_env(cfg, n, seed=seed, device=dev, stagger=False, paired=paired)
  if code_noise is not None:
    pop.noise = float(code_noise)
  code_src = CodeSource(cfg, pop, dev, probe=probe)
  obs = env.reset_all()
  state = policy.zero_state(n, dev)
  goal_code = torch.zeros(n, cfg['n_cells'], device=dev)
  prev_action = torch.full((n,), -1, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)
  rows = []
  policy.eval()
  with torch.no_grad():
    while len(rows) < episodes:
      code = code_src(obs['pos'], obs['dir'], obs['vel'])
      logits, _, state = policy.step(obs['rgb'], code, goal_code, prev_action,
                                     prev_reward, state, extra_from_obs(cfg, obs))
      if greedy:
        a = logits.argmax(-1)
      else:
        a = torch.distributions.Categorical(logits=logits).sample()
      obs, r_raw, done, info = env.step(a)
      r = r_raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else r_raw
      hit = info['hit']
      if bool(hit.any()):
        g_hit = code_src.goal(info['hit_pos'], info['hit_dir'], info['hit_vel'])
        goal_code = torch.where(hit[:, None], g_hit, goal_code)
      prev_action, prev_reward = a, r
      if bool(done.any()):
        keep = (~done).float()
        goal_code = goal_code * keep[:, None]
        state = (state[0] * keep[:, None], state[1] * keep[:, None])
        prev_action = torch.where(done, torch.full_like(a, -1), a)
        prev_reward = prev_reward * keep
        es = info['ep_stats']
        if es is not None:
          for j in range(es['env'].numel()):
            rows.append(dict(ret=float(es['ret'][j]), n_goals=int(es['n_goals'][j]),
                             first_goal_t=int(es['first_goal_t'][j]),
                             path_eff=float(es['path_eff'][j]),
                             path_eff_n=int(es['path_eff_n'][j])))
  rows = rows[:episodes]
  ret = np.array([r['ret'] for r in rows])
  goals = np.array([r['n_goals'] for r in rows])
  first = np.array([r['first_goal_t'] if r['first_goal_t'] >= 0 else cfg['episode_len']
                    for r in rows])
  found = np.array([r['first_goal_t'] >= 0 for r in rows])
  eff = np.array([r['path_eff'] for r in rows if r['path_eff_n'] > 0])
  # Homing: goals after the first one per 1000 post-discovery steps (episodes
  # that found the goal), and the fraction of episodes with >= 2 goals.
  post = np.array([(r['n_goals'] - 1) / max(cfg['episode_len'] - r['first_goal_t'], 1) * 1000.0
                   for r in rows if r['first_goal_t'] >= 0])
  sem = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')
  return dict(
      probe=probe, episodes=len(rows), greedy=greedy, eval_seed=seed, paired=paired,
      post_discovery_goal_rate=float(post.mean()) if len(post) else float('nan'),
      sem_post_discovery_goal_rate=sem(post) if len(post) > 1 else float('nan'),
      frac_two_goals=float(np.mean(goals >= 2)),
      mean_score=float(ret.mean()), sem_score=sem(ret),
      std_score=float(ret.std(ddof=1)) if len(ret) > 1 else float('nan'),
      mean_goals=float(goals.mean()), sem_goals=sem(goals),
      frac_found_goal=float(found.mean()),
      mean_first_goal_step=float(first.mean()), sem_first_goal_step=sem(first),
      mean_path_eff=float(eff.mean()) if len(eff) else float('nan'),
      sem_path_eff=sem(eff) if len(eff) > 1 else float('nan'),
      n_path_eff=int(len(eff)),
      scores=[float(x) for x in ret], goals=[int(x) for x in goals],
      first_goal_steps=[int(x) for x in first])


def pick_checkpoint(run, spec):
  """'final' | 'latest' (highest-frame checkpoint incl. final and resume.pt)
  | explicit path -> (path, frames)."""
  def frames_of(p):
    ck = torch.load(p, map_location='cpu')
    return int(ck.get('frames', ck.get('frames_done', -1)))
  if spec == 'final':
    p = os.path.join(run, 'ckpt_final.pt')
    if not os.path.exists(p):
      raise FileNotFoundError(f'{p} does not exist (run not finished?)')
    return p, frames_of(p)
  if spec == 'latest':
    cands = glob.glob(os.path.join(run, 'ckpt_*.pt')) + \
        [p for p in (os.path.join(run, 'resume.pt'),) if os.path.exists(p)]
    if not cands:
      raise FileNotFoundError(f'no checkpoint in {run}')
    best = max(cands, key=frames_of)
    return best, frames_of(best)
  return spec, frames_of(spec)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--run', required=True)
  ap.add_argument('--ckpt', default='final', help="'final', 'latest', or a path")
  ap.add_argument('--probe', choices=PROBES, default='intact')
  ap.add_argument('--probes', default=None, help='comma list; overrides --probe')
  ap.add_argument('--episodes', type=int, default=CANONICAL_EPISODES)
  ap.add_argument('--n_envs', type=int, default=32)
  ap.add_argument('--seed', type=int, default=424242)
  ap.add_argument('--greedy', action='store_true')
  ap.add_argument('--code_noise', type=float, default=None)
  ap.add_argument('--paired', type=int, default=1,
                  help='1: episode layouts keyed by (env slot, episode, draw) so '
                       'they are identical across runs (common random numbers)')
  ap.add_argument('--untrained_inits', type=int, default=3,
                  help="'untrained' probe: average over this many random inits")
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--out_dir', default=None)
  args = ap.parse_args(argv)

  probes = [p.strip() for p in args.probes.split(',')] if args.probes else [args.probe]
  bad = [p for p in probes if p not in PROBES]
  if bad:
    ap.error(f'unknown probe(s) {bad}; choose from {PROBES}')
  with open(os.path.join(args.run, 'config.json')) as f:
    cfg = json.load(f)
  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  ckpt_path, ckpt_frames = pick_checkpoint(args.run, args.ckpt)
  ck = torch.load(ckpt_path, map_location='cpu')
  pop = build_population(cfg, dev)
  out_dir = args.out_dir or os.path.join(args.run, 'eval')
  os.makedirs(out_dir, exist_ok=True)
  canonical = (os.path.basename(ckpt_path) == 'ckpt_final.pt' and
               args.episodes == CANONICAL_EPISODES and not args.greedy and
               args.code_noise is None and args.seed == 424242)
  for probe in probes:
    if probe == 'untrained':
      # Chance line: the policy at random initialisation (the training init and
      # further inits), each through the identical protocol, averaged.
      parts = []
      for j in range(max(1, args.untrained_inits)):
        torch.manual_seed(cfg['seed'] + 1000 * j)
        policy = build_policy(cfg, dev)
        parts.append(run_eval(cfg, policy, pop, dev, probe='intact', episodes=args.episodes,
                              n_envs=args.n_envs, seed=args.seed, greedy=args.greedy,
                              code_noise=args.code_noise, paired=bool(args.paired)))
      res = dict(parts[0])
      for k in ('mean_score', 'mean_goals', 'frac_found_goal', 'mean_first_goal_step',
                'mean_path_eff', 'post_discovery_goal_rate', 'frac_two_goals'):
        vals = [p[k] for p in parts]
        res[k] = float(np.nanmean(vals))
      res['sem_score'] = float(np.sqrt(np.mean([p['sem_score'] ** 2 for p in parts]) / len(parts)))
      res['per_init_scores'] = [p['mean_score'] for p in parts]
      res['scores'] = [x for p in parts for x in p['scores']]
      res['episodes'] = len(parts[0]['scores'])     # per init
      res['n_inits'] = len(parts)
      res['probe'] = 'untrained'
    else:
      torch.manual_seed(cfg['seed'])
      policy = build_policy(cfg, dev)
      policy.load_state_dict(ck['policy'])
      res = run_eval(cfg, policy, pop, dev, probe=probe, episodes=args.episodes,
                     n_envs=args.n_envs, seed=args.seed, greedy=args.greedy,
                     code_noise=args.code_noise, paired=bool(args.paired))
    res.update(ckpt=ckpt_path, frames=0 if probe == 'untrained' else ckpt_frames,
               canonical=canonical, run=args.run, cond=cfg['cond'],
               policy=cfg['policy'], vision=cfg.get('vision'), seed=cfg['seed'])
    tag = probe
    if not canonical:
      tag += f'@{ckpt_frames}f_{args.episodes}ep'
      if args.greedy:
        tag += '_greedy'
      if args.code_noise is not None:
        tag += f'_noise{args.code_noise}'
      if args.seed != 424242:
        tag += f'_seed{args.seed}'
    with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
      json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ('scores', 'goals', 'first_goal_steps', 'per_init_scores')}),
          flush=True)


if __name__ == '__main__':
  main()
