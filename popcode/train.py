"""A2C training of the population-code agent (Banino et al. 2018 recipe).

Hyperparameters follow SI Table 2 unless overridden: RMSProp (shared
statistics in the paper's async A3C; here synchronous A2C over n parallel
environments, the same simplification as banino/rl/train_rl.py), learning
rate 1e-4 (paper sampled [1e-6, 2e-4]), RMSProp decay 0.99, baseline cost
0.5, entropy cost 8e-5 (paper sampled [6e-5, 1e-4]), discount 0.99, BPTT 100
steps with n-step returns over the same window, action repeat 4,
gradient-norm clip 40, rewards clipped to [-1, 1] for learning (scores are
logged raw).

RMSProp eps: Mnih et al. 2016 used 0.1, which with this codebase's sparse
rewards leaves every parameter in the eps-dominated (plain-SGD) regime and the
policy uniform for >25M frames; the matrix therefore uses a normalised step
(--rms_eps 1e-5, the standard A2C value) and, to keep the vision embedding
bounded under normalised steps, --encoder_norm layernorm (see nets.py and
STATUS.md for the saturation incident).

Loss reduction: SUM over the 100-step window, MEAN over environments (Mnih et
al. 2016 accumulate per-step gradients along the rollout; the earlier .mean()
over T*n in this codebase's history silently shrank every gradient 100x and
kept policies uniform -- see banino/A3C_REPORT.md).

The population code is a pure lookup from the true state (position + heading
of motion); the goal code g* is the code at the position where the goal was
last reached (zeros before the first goal of an episode), as in the paper.

Example:
  python -m popcode.train --cond pgc33 --policy lstm --seed 1 \
      --frames 300000000 --out runs/pgc33_lstm_s1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from popcode.common import (EXPERIMENT_KEYS, CodeSource, add_shared_args,   # noqa: E402
                            build_env, build_policy, build_population,
                            extra_from_obs, resolve_cond)


def parse_args(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  add_shared_args(ap)
  t = ap.add_argument_group('training')
  t.add_argument('--out', required=True)
  t.add_argument('--seed', type=int, default=1)
  t.add_argument('--frames', type=int, default=300_000_000)
  t.add_argument('--n_envs', type=int, default=32)
  t.add_argument('--rollout', type=int, default=100)
  t.add_argument('--gamma', type=float, default=0.99)
  t.add_argument('--lr', type=float, default=1e-4)
  t.add_argument('--rms_alpha', type=float, default=0.99)
  t.add_argument('--rms_eps', type=float, default=0.1)
  t.add_argument('--entropy', type=float, default=8e-5)
  t.add_argument('--value_coef', type=float, default=0.5)
  t.add_argument('--grad_clip', type=float, default=40.0)
  t.add_argument('--reward_clip', type=float, default=1.0)
  t.add_argument('--log_every', type=int, default=20, help='updates')
  t.add_argument('--ckpt_every', type=int, default=25_000_000, help='frames')
  t.add_argument('--resume_every', type=int, default=200, help='updates')
  t.add_argument('--device', default='cuda:0')
  t.add_argument('--resume', action='store_true')
  t.add_argument('--tf32', type=int, default=1,
                 help='TF32 for cuDNN convolutions (matmuls, incl. the code '
                      'lookup and the LSTM, stay full FP32)')
  return ap.parse_args(argv)


def check_resume_config(cfg, path):
  """Refuse to resume under a different experiment definition."""
  old = json.load(open(path))
  bad = {k: (old.get(k), cfg.get(k)) for k in EXPERIMENT_KEYS
         if k in old and old.get(k) != cfg.get(k)}
  if bad:
    raise SystemExit(f'--resume refused: config.json in {cfg["out"]} differs in '
                     f'{bad} (stored, requested)')


def main(argv=None):
  args = parse_args(argv)
  cfg = resolve_cond(vars(args))
  os.makedirs(cfg['out'], exist_ok=True)
  cfg_path = os.path.join(cfg['out'], 'config.json')
  resume_path = os.path.join(cfg['out'], 'resume.pt')
  final_path = os.path.join(cfg['out'], 'ckpt_final.pt')
  if cfg['resume'] and os.path.exists(final_path):
    done_frames = int(torch.load(final_path, map_location='cpu').get('frames', 0))
    if done_frames >= cfg['frames'] or not os.path.exists(resume_path):
      print(f'run already finished (ckpt_final.pt at {done_frames} frames)', flush=True)
      return
    # Budget was raised: keep the old final checkpoint under its frame count,
    # demote its canonical evaluations to checkpoint evaluations, and continue.
    os.replace(final_path, os.path.join(cfg['out'], f'ckpt_{done_frames:010d}.pt'))
    import glob as _glob
    for f in _glob.glob(os.path.join(cfg['out'], 'eval', '*.json')):
      base = os.path.basename(f)[:-5]
      if '@' not in base:
        try:
          ev = json.load(open(f))
          ev['canonical'] = False
          json.dump(ev, open(f, 'w'), indent=1)
        except Exception:
          pass
        os.replace(f, os.path.join(cfg['out'], 'eval', f'{base}@{done_frames}f_200ep.json'))
    print(f'extending run from {done_frames} to {cfg["frames"]} frames', flush=True)
  resuming = cfg['resume'] and os.path.exists(resume_path)
  if resuming and os.path.exists(cfg_path):
    check_resume_config(cfg, cfg_path)
    old = json.load(open(cfg_path))
    if old.get('frames') != cfg['frames']:
      old['frames'] = cfg['frames']
      with open(cfg_path, 'w') as f:
        json.dump(old, f, indent=1)

  dev = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
  cfg['device_used'] = str(dev)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = bool(cfg['tf32'])
  torch.backends.cudnn.benchmark = True
  torch.manual_seed(cfg['seed'])
  np.random.seed(cfg['seed'])

  pop = build_population(cfg, dev)
  n = cfg['n_envs']
  env = build_env(cfg, n, seed=cfg['seed'] * 1000 + 1, device=dev, stagger=True)
  policy = build_policy(cfg, dev)
  code_src = CodeSource(cfg, pop, dev)
  opt = torch.optim.RMSprop(policy.parameters(), lr=cfg['lr'],
                            alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
  n_params = sum(p.numel() for p in policy.parameters())
  cfg['population'] = pop.describe()
  cfg['n_params'] = n_params
  if not resuming:
    with open(cfg_path, 'w') as f:
      json.dump(cfg, f, indent=1)

  T, gamma = cfg['rollout'], cfg['gamma']
  frames_done, update, episodes_seen = 0, 0, 0
  if resuming:
    st = torch.load(resume_path, map_location='cpu')
    policy.load_state_dict(st['policy'])
    opt.load_state_dict(st['opt'])
    frames_done, update = st['frames_done'], st['update']
    episodes_seen = st.get('episodes_seen', 0)
    torch.set_rng_state(st['cpu_rng'].cpu())
    if dev.type == 'cuda' and 'cuda_rng' in st:
      torch.cuda.set_rng_state(st['cuda_rng'].cpu(), dev)
    env.gen.manual_seed(cfg['seed'] * 1000 + 1 + update)
    print(f'resumed at frames={frames_done} update={update}', flush=True)

  def save_resume():
    tmp = resume_path + '.tmp'
    st = dict(policy=policy.state_dict(), opt=opt.state_dict(),
              frames_done=frames_done, update=update, episodes_seen=episodes_seen,
              cpu_rng=torch.get_rng_state())
    if dev.type == 'cuda':
      st['cuda_rng'] = torch.cuda.get_rng_state(dev)
    torch.save(st, tmp)
    os.replace(tmp, resume_path)

  obs = env.reset_all()
  state = policy.zero_state(n, dev)
  goal_code = torch.zeros(n, cfg['n_cells'], device=dev)
  prev_action = torch.full((n,), -1, dtype=torch.long, device=dev)   # -1 = none
  prev_reward = torch.zeros(n, device=dev)
  recent = dict(ret=[], goals=[], first=[], eff=[])
  ep_f = open(os.path.join(cfg['out'], 'episodes.csv'), 'a')
  if ep_f.tell() == 0:
    ep_f.write('frames,update,env,ret,n_goals,first_goal_step,path_eff,path_eff_n\n')
  metrics_f = open(os.path.join(cfg['out'], 'metrics.jsonl'), 'a')
  print(f'{cfg["cond"]} policy={cfg["policy"]} seed={cfg["seed"]} n_pgc={pop.n_pgc}/'
        f'{pop.n_cells} params={n_params/1e6:.2f}M device={dev} lr={cfg["lr"]} '
        f'eps={cfg["rms_eps"]} encoder_norm={cfg["encoder_norm"]}', flush=True)

  t0, f0 = time.time(), frames_done
  next_ckpt = (frames_done // cfg['ckpt_every'] + 1) * cfg['ckpt_every']
  while frames_done < cfg['frames']:
    # ---------------------------------------------------------- rollout
    roll = {k: [] for k in ('rgb', 'code', 'goal', 'pa', 'pr', 'extra',
                            'a', 'r', 'd')}
    state0 = (state[0].detach(), state[1].detach())
    with torch.no_grad():
      for _ in range(T):
        code = code_src(obs['pos'], obs['dir'], obs['vel'])
        extra = extra_from_obs(cfg, obs)
        logits, _, state = policy.step(obs['rgb'], code, goal_code, prev_action,
                                       prev_reward, state, extra)
        a = torch.distributions.Categorical(logits=logits).sample()
        roll['rgb'].append(obs['rgb'])
        roll['code'].append(code)
        roll['goal'].append(goal_code)
        roll['pa'].append(prev_action)
        roll['pr'].append(prev_reward)
        roll['extra'].append(extra)
        roll['a'].append(a)

        obs, r_raw, done, info = env.step(a)
        frames_done += n * cfg['action_repeat']
        r = r_raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) \
            if cfg['reward_clip'] > 0 else r_raw
        roll['r'].append(r)
        roll['d'].append(done.float())
        # Goal code: the population's activity at the position where the goal
        # was reached (its own motion direction at that moment for PGCs).
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
              ret, ng = float(es['ret'][j]), int(es['n_goals'][j])
              fg, pe, pen = int(es['first_goal_t'][j]), float(es['path_eff'][j]), int(es['path_eff_n'][j])
              ep_f.write(f'{frames_done},{update},{int(es["env"][j])},{ret:.1f},{ng},{fg},{pe:.4f},{pen}\n')
              recent['ret'].append(ret)
              recent['goals'].append(ng)
              recent['first'].append(fg if fg >= 0 else cfg['episode_len'])
              if pen > 0:
                recent['eff'].append(pe)
              episodes_seen += 1
            for k in recent:
              if len(recent[k]) > 100:
                del recent[k][:-100]

      # Bootstrap value for the state after the window.
      code_b = code_src(obs['pos'], obs['dir'], obs['vel'])
      _, v_boot, _ = policy.step(obs['rgb'], code_b, goal_code, prev_action,
                                 prev_reward, (state[0].detach(), state[1].detach()),
                                 extra_from_obs(cfg, obs))
    returns = torch.zeros(T, n, device=dev)
    R = v_boot
    for t in reversed(range(T)):
      R = roll['r'][t] + gamma * (1.0 - roll['d'][t]) * R
      returns[t] = R

    # ------------------------------------------------------ A2C update
    # Encode all T*n frames in one pass (the encoder is not recurrent), then
    # run the recurrent core step by step from the window's initial state.
    E = policy.embed(torch.cat(roll['rgb'], 0)).view(T, n, -1)
    st = state0
    logits_l, v_l = [], []
    for t in range(T):
      logits, v, st = policy.step_from_embedding(
          E[t], roll['code'][t], roll['goal'][t], roll['pa'][t], roll['pr'][t], st,
          roll['extra'][t])
      logits_l.append(logits)
      v_l.append(v)
      keep = (1.0 - roll['d'][t]).unsqueeze(1)
      st = (st[0] * keep, st[1] * keep)
    logits_all = torch.stack(logits_l)            # [T,n,A]
    v_all = torch.stack(v_l)                      # [T,n]
    a_all = torch.stack(roll['a'])                # [T,n]
    dist = torch.distributions.Categorical(logits=logits_all)
    adv = (returns - v_all).detach()
    pol_loss = -(dist.log_prob(a_all) * adv).sum(0).mean()
    val_loss = 0.5 * (returns - v_all).pow(2).sum(0).mean()
    ent_sum = dist.entropy().sum(0).mean()
    loss = pol_loss + cfg['value_coef'] * val_loss - cfg['entropy'] * ent_sum
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg['grad_clip'])
    opt.step()
    state = (state[0].detach(), state[1].detach())
    update += 1

    # ---------------------------------------------------------- logging
    if update % cfg['log_every'] == 0:
      el = time.time() - t0
      fps = (frames_done - f0) / max(el, 1e-6)
      rec = dict(update=update, frames=frames_done, fps=round(fps),
                 episodes=episodes_seen,
                 avg_return_100=round(float(np.mean(recent['ret'])), 3) if recent['ret'] else None,
                 avg_goals_100=round(float(np.mean(recent['goals'])), 3) if recent['goals'] else None,
                 first_goal_step_100=round(float(np.mean(recent['first'])), 1) if recent['first'] else None,
                 path_eff_100=round(float(np.mean(recent['eff'])), 4) if recent['eff'] else None,
                 entropy=round(float(ent_sum.item()) / T, 4),
                 pol_loss=round(float(pol_loss.item()), 4),
                 val_loss=round(float(val_loss.item()), 4),
                 grad_norm=round(float(gn), 3),
                 value_mean=round(float(v_all.mean().item()), 4),
                 hours=round(el / 3600, 3))
      diag = policy.diagnostics(obs['rgb'], code_b, goal_code, prev_action, prev_reward,
                                state, extra_from_obs(cfg, obs))
      rec.update({k: round(v, 4) for k, v in diag.items()})
      print(json.dumps(rec), flush=True)
      metrics_f.write(json.dumps(rec) + '\n')
      metrics_f.flush()
      ep_f.flush()
    if update % cfg['resume_every'] == 0:
      save_resume()
    if frames_done >= next_ckpt:
      torch.save({'policy': policy.state_dict(), 'frames': frames_done,
                  'update': update}, os.path.join(cfg['out'], f'ckpt_{frames_done:010d}.pt'))
      next_ckpt += cfg['ckpt_every']

  save_resume()
  torch.save({'policy': policy.state_dict(), 'frames': frames_done, 'update': update},
             final_path)
  ep_f.close()
  metrics_f.close()
  print('done', flush=True)


if __name__ == '__main__':
  main()
