"""A2C training of the population-code agent (Banino et al. 2018 recipe).

Hyperparameters follow SI Table 2 unless overridden: RMSProp (shared
statistics in the paper's async A3C; here synchronous A2C over n parallel
environments, the same simplification as banino/rl/train_rl.py), learning
rate 1e-4 (paper sampled [1e-6, 2e-4]), RMSProp decay 0.99, baseline cost
0.5, entropy cost 8e-5 (paper sampled [6e-5, 1e-4]), discount 0.99, BPTT 100
steps with n-step returns over the same window, action repeat 4,
gradient-norm clip 40, rewards clipped to [-1, 1] for learning (scores are
logged raw).

RMSProp eps: Mnih et al. 2016 used 0.1, which with this task's sparse rewards
leaves every parameter in the eps-dominated (plain-SGD) regime and the policy
uniform for >25M frames; the run matrix therefore uses a normalised step
(--rms_eps 1e-5, the standard A2C value) and, to keep the vision embedding
bounded under normalised steps, --encoder_norm layernorm (see nets.py for the
saturation mechanism).

Loss reduction: SUM over the 100-step window, MEAN over environments (Mnih et
al. 2016 accumulate per-step gradients along the rollout; a mean over T*n
would shrink every gradient 100x and keep the policy uniform).

The population code is a pure lookup from the true state (position + heading
of motion); the goal code g* is the code at the position where the goal was
last reached (zeros before the first goal of an episode), as in the paper.

Resuming and extending (--resume): an interrupted run continues from
resume.pt (policy, optimizer and RNG state). A finished run whose
ckpt_final.pt lies below the requested --frames is extended: the final
checkpoint is kept as ckpt_<frames>.pt, its canonical evaluations become
checkpoint evaluations (eval/<probe>@<frames>f_200ep.json), config.json gets
the new budget and an 'extensions' record {from_frames, to_frames, mode}, and
training continues from resume.pt when it exists (mode 'resume') or, when it
does not, from the checkpoint's weights alone (mode 'weights_only'): a fresh
RMSProp optimizer runs --warmup_updates updates at learning rate 0 (no frames
counted, no episodes logged) so its squared-gradient statistics are
estimated on the frozen policy before any parameter changes; resume.pt is
written when the warm-up ends. A run directory with intermediate checkpoints
but neither ckpt_final.pt nor resume.pt (an extension interrupted during its
warm-up) continues the same way from its latest checkpoint (mode
'recover_weights_only') rather than restarting from zero.

Learnable look-aheads (--learn_shift 1 or 2): look-ahead distances in metres,
initialised from the population and trained by the same A2C loss as the
policy: the rollout stores the agent's states, the update recomputes the
current code and the goal code from them with the look-aheads as parameters,
and the gradient reaches them through the policy's input layer.
  1  every cell has its own (forward, sideways) vector: 2 x n_cells numbers
  2  one forward distance per module, shared by that module's predictive
     cells; the other cells stay plain grid cells: n_modules numbers
They have their own learning rate (--shift_lr), are not gradient-clipped, and
every checkpoint stores the resulting per-cell vectors ('shift', [n_cells, 2])
and the raw parameters ('shift_param'). --shift_scramble 1 multiplies the
look-ahead gradient by a random sign per parameter at every update: the same
step sizes with no signal, the control for how far noise alone moves them.

Example:
  python -m code.train --cond pgc33 --policy lstm --seed 1 \
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
from code.common import (EXPERIMENT_KEYS, CodeSource, add_shared_args,   # noqa: E402
                            build_env, build_policy, build_population,
                            extra_from_obs, resolve_cond)

# Entries of the stored population description (config.json 'population') that
# pin down the cells a run was trained with; compared on --resume when present.
POPULATION_KEYS = ('n_cells', 'n_modules', 'spacings_m', 'orientation_deg', 'pgc_idx',
                   'pgc_shift_mode', 'pgc_shift_m_per_module', 'pgc_shift_jitter',
                   'sharpness', 'noise', 'seed', 'cell_type',
                   'pgc_shift_dist', 'pgc_shift_angle', 'pgc_shift_angle_alt', 'spacing_sd',
                   'orient_sd_deg',
                   'lattice_ellip', 'lattice_shear', 'field_jitter', 'lattice_hash')


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
  t.add_argument('--warmup_updates', type=int, default=200,
                 help='weights-only extension of a finished run (ckpt_final.pt below '
                      '--frames and no resume.pt): updates run at learning rate 0 so '
                      'the fresh RMSProp statistics are estimated on the frozen policy '
                      'before training continues (no frames or episodes are counted)')
  t.add_argument('--learn_shift', type=int, default=0,
                 help='1: train a (forward, sideways) look-ahead vector per cell; 2: train one '
                      'forward look-ahead per module for its predictive cells (module docstring)')
  t.add_argument('--shift_scramble', type=int, default=0,
                 help='1: random sign on the look-ahead gradient at every update (noise control)')
  t.add_argument('--shift_lr', type=float, default=1e-4,
                 help='learning rate of the look-ahead vectors (metres per normalised step)')
  t.add_argument('--tf32', type=int, default=1,
                 help='TF32 for cuDNN convolutions (matmuls, incl. the code '
                      'lookup and the LSTM, stay full FP32)')
  return ap.parse_args(argv)


def check_resume_config(cfg, path):
  """Refuse to resume under a different experiment definition. A stored None
  stands for a key that did not exist when the config was written and is not
  compared; every other stored experiment key must equal the requested value.
  When the stored config carries the population description, the population
  rebuilt from the requested arguments must reproduce it (same cells, PGC
  indices and shifts)."""
  old = json.load(open(path))
  req = dict(cfg)
  # Keys that did not exist when older configs were written take their default
  # on both sides, so an old run cannot be resumed with a later option switched on.
  for k, v in (('rooms', 'none'), ('pgc_shift_dist', 'rule'), ('pgc_shift_angle', 0.0),
               ('pgc_shift_angle_alt', 0), ('spacing_sd', 0.0), ('orient_sd_deg', 0.0),
               ('lattice_ellip', 1.0), ('lattice_shear', 0.0), ('field_jitter', 0.0),
               ('learn_shift', 0)):
    old.setdefault(k, v)
    req.setdefault(k, v)
  bad = {k: (old.get(k), req.get(k)) for k in EXPERIMENT_KEYS
         if old.get(k) is not None and old.get(k) != req.get(k)}
  if bad:
    raise SystemExit(f'--resume refused: config.json in {cfg["out"]} differs in '
                     f'{bad} (stored, requested)')
  stored = old.get('population') or {}
  if stored:
    now = build_population(cfg, 'cpu').describe()
    for k in POPULATION_KEYS:
      if k not in stored or k not in now:
        continue
      a, b = stored[k], now[k]
      if isinstance(a, list) or isinstance(b, list):
        same = (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and
                np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                            rtol=1e-6, atol=1e-6))
      else:
        same = a == b
      if not same:
        raise SystemExit(f'--resume refused: the population rebuilt for {cfg["out"]} differs '
                         f'from the stored one in {k!r} (stored {str(a)[:80]}, '
                         f'rebuilt {str(b)[:80]})')


def main(argv=None):
  args = parse_args(argv)
  cfg = resolve_cond(vars(args))
  os.makedirs(cfg['out'], exist_ok=True)
  cfg_path = os.path.join(cfg['out'], 'config.json')
  resume_path = os.path.join(cfg['out'], 'resume.pt')
  final_path = os.path.join(cfg['out'], 'ckpt_final.pt')
  extension = None       # provenance record when a finished run is continued
  weights_ckpt = None    # weights-only extension: the checkpoint to start from
  if cfg['resume'] and os.path.exists(final_path):
    done_frames = int(torch.load(final_path, map_location='cpu').get('frames', 0))
    if done_frames >= cfg['frames']:
      print(f'run already finished (ckpt_final.pt at {done_frames} frames)', flush=True)
      return
    if os.path.exists(cfg_path):
      check_resume_config(cfg, cfg_path)       # refuse before anything is renamed
    mode = 'resume' if os.path.exists(resume_path) else 'weights_only'
    # Budget was raised: keep the old final checkpoint under its frame count,
    # demote its canonical evaluations to checkpoint evaluations, and continue.
    demoted = os.path.join(cfg['out'], f'ckpt_{done_frames:010d}.pt')
    os.replace(final_path, demoted)
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
    extension = dict(from_frames=done_frames, to_frames=cfg['frames'], mode=mode)
    if mode == 'weights_only':
      weights_ckpt = demoted
    print(f'extending run from {done_frames} to {cfg["frames"]} frames ({mode})', flush=True)
  elif cfg['resume'] and not os.path.exists(resume_path):
    # Neither ckpt_final.pt nor resume.pt, but intermediate checkpoints: a run
    # interrupted before its first resume.pt was written (e.g. a weights-only
    # extension killed during its warm-up, then relaunched by the scheduler).
    # Continue from the latest checkpoint's weights instead of restarting from
    # zero in a directory that already holds the run's history.
    import glob as _glob
    cks = sorted(_glob.glob(os.path.join(cfg['out'], 'ckpt_[0-9]*.pt')))
    if cks:
      if os.path.exists(cfg_path):
        check_resume_config(cfg, cfg_path)
      weights_ckpt = cks[-1]
      done_frames = int(torch.load(weights_ckpt, map_location='cpu').get('frames', 0))
      extension = dict(from_frames=done_frames, to_frames=cfg['frames'], mode='recover_weights_only')
      print(f'no resume.pt or ckpt_final.pt: continuing from {os.path.basename(weights_ckpt)} '
            f'({done_frames} frames) with a fresh optimizer (recover_weights_only)', flush=True)
  resuming = cfg['resume'] and os.path.exists(resume_path)
  weights_only = weights_ckpt is not None
  stored_cfg = None
  if (resuming or weights_only) and os.path.exists(cfg_path):
    if extension is None:
      check_resume_config(cfg, cfg_path)
    stored_cfg = json.load(open(cfg_path))
    changed = False
    if stored_cfg.get('frames') != cfg['frames']:
      stored_cfg['frames'] = cfg['frames']
      changed = True
    if extension is not None:
      stored_cfg['extensions'] = list(stored_cfg.get('extensions') or []) + [extension]
      changed = True
    if changed:
      with open(cfg_path, 'w') as f:
        json.dump(stored_cfg, f, indent=1)

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
  shift_param = None     # learnable look-aheads: [n_cells, 2] (forward, leftward) or [n_modules]
  mod_idx = torch.as_tensor(pop.module, device=dev)
  pgc_f = pop.is_pgc.float()

  def cell_shift(p):
    """Parameters -> per-cell (forward, leftward) look-aheads [n_cells, 2]."""
    if cfg['learn_shift'] == 2:
      fwd = p[mod_idx] * pgc_f
      return torch.stack([fwd, torch.zeros_like(fwd)], 1)
    return p

  if cfg.get('learn_shift'):
    if cfg['learn_shift'] == 2:      # start: mean look-ahead of each module's predictive cells
      init = torch.stack([pop.shift[(mod_idx == m) & pop.is_pgc].mean()
                          for m in range(pop.n_modules)])
    else:
      init = torch.stack([pop.shift, pop.shift_lat], 1)
    shift_param = torch.nn.Parameter(init.clone())
    opt = torch.optim.RMSprop([{'params': list(policy.parameters())},
                               {'params': [shift_param], 'lr': cfg['shift_lr']}],
                              lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
  else:
    opt = torch.optim.RMSprop(policy.parameters(), lr=cfg['lr'],
                              alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
  n_params = sum(p.numel() for p in policy.parameters())
  cfg['population'] = pop.describe()
  cfg['n_params'] = n_params
  if not resuming and stored_cfg is None:
    if extension is not None:
      cfg['extensions'] = [extension]
    with open(cfg_path, 'w') as f:
      json.dump(cfg, f, indent=1)

  T, gamma = cfg['rollout'], cfg['gamma']
  frames_done, update, episodes_seen = 0, 0, 0
  warm_left = 0        # weights-only extension: updates left at learning rate 0
  if resuming:
    st = torch.load(resume_path, map_location='cpu')
    policy.load_state_dict(st['policy'])
    opt.load_state_dict(st['opt'])
    if shift_param is not None:
      shift_param.data.copy_(st['shift_param'].to(dev))
    frames_done, update = st['frames_done'], st['update']
    episodes_seen = st.get('episodes_seen', 0)
    torch.set_rng_state(st['cpu_rng'].cpu())
    if dev.type == 'cuda' and 'cuda_rng' in st:
      torch.cuda.set_rng_state(st['cuda_rng'].cpu(), dev)
    env.gen.manual_seed(cfg['seed'] * 1000 + 1 + update)
    print(f'resumed at frames={frames_done} update={update}', flush=True)
  elif weights_only:
    ck = torch.load(weights_ckpt, map_location='cpu')
    policy.load_state_dict(ck['policy'])
    if shift_param is not None and 'shift_param' in ck:
      shift_param.data.copy_(ck['shift_param'].to(dev))
    frames_done, update = int(ck['frames']), int(ck.get('update', 0))
    episodes_seen = ck.get('episodes_seen')
    if episodes_seen is None:       # older checkpoints: count the logged episodes
      ep_path = os.path.join(cfg['out'], 'episodes.csv')
      episodes_seen = 0
      if os.path.exists(ep_path):
        with open(ep_path) as f:
          episodes_seen = max(0, sum(1 for _ in f) - 1)
    episodes_seen = int(episodes_seen)
    # Fresh random streams that do not replay the start of training.
    torch.manual_seed(cfg['seed'] * 1000 + 1 + update)
    env.gen.manual_seed(cfg['seed'] * 1000 + 1 + update)
    warm_left = max(0, int(cfg['warmup_updates']))
    print(f'weights-only extension from {os.path.basename(weights_ckpt)}: frames={frames_done} '
          f'update={update} episodes={episodes_seen}; fresh optimizer, {warm_left} warm-up '
          f'updates at lr 0', flush=True)

  def save_resume():
    tmp = resume_path + '.tmp'
    st = dict(policy=policy.state_dict(), opt=opt.state_dict(),
              frames_done=frames_done, update=update, episodes_seen=episodes_seen,
              cpu_rng=torch.get_rng_state())
    if dev.type == 'cuda':
      st['cuda_rng'] = torch.cuda.get_rng_state(dev)
    if shift_param is not None:
      st['shift_param'] = shift_param.detach().cpu().clone()
    torch.save(st, tmp)
    os.replace(tmp, resume_path)

  def ckpt_dict():
    d = {'policy': policy.state_dict(), 'frames': frames_done, 'update': update,
         'episodes_seen': episodes_seen}
    if shift_param is not None:
      d['shift'] = cell_shift(shift_param.detach()).cpu().clone()
      d['shift_param'] = shift_param.detach().cpu().clone()
    return d

  def shift_stats():
    """Summary of the learned look-ahead vectors (cm)."""
    if cfg['learn_shift'] == 2:
      return {f'shift_fwd_m{m}': round(float(shift_param.detach()[m]) * 100.0, 4)
              for m in range(pop.n_modules)}
    sf, sl = shift_param.detach()[:, 0] * 100.0, shift_param.detach()[:, 1] * 100.0
    mod = mod_idx
    out = dict(shift_fwd_mean=float(sf.mean()), shift_fwd_sd=float(sf.std()),
               shift_lat_mean=float(sl.mean()), shift_lat_sd=float(sl.std()),
               shift_fwd_gt2=float((sf > 2.0).float().mean()),
               shift_fwd_lt2=float((sf < -2.0).float().mean()))
    for m in range(pop.n_modules):
      out[f'shift_fwd_m{m}'] = float(sf[mod == m].mean())
    return {k: round(v, 4) for k, v in out.items()}

  if shift_param is not None:
    cs = cell_shift(shift_param.data)
    pop.set_shift(cs[:, 0], cs[:, 1])
  obs = env.reset_all()
  state = policy.zero_state(n, dev)
  goal_code = torch.zeros(n, cfg['n_cells'], device=dev)
  # State at the last goal hit (learnable look-aheads recompute g* from it).
  g_pos, g_dir, g_vel = (torch.zeros(n, 2, device=dev) for _ in range(3))
  g_val = torch.zeros(n, device=dev)
  prev_action = torch.full((n,), -1, dtype=torch.long, device=dev)   # -1 = none
  prev_reward = torch.zeros(n, device=dev)
  recent = dict(ret=[], goals=[], first=[], eff=[])
  ep_path = os.path.join(cfg['out'], 'episodes.csv')
  ep_f = open(ep_path, 'a')
  if ep_f.tell() == 0:
    ep_f.write('frames,update,env,ret,n_goals,first_goal_step,path_eff,path_eff_n,room,n_goals_post\n')
    room_cols = True       # new files also log the room condition of each episode
  else:
    with open(ep_path) as f:
      room_cols = 'room' in f.readline()
  metrics_f = open(os.path.join(cfg['out'], 'metrics.jsonl'), 'a')
  print(f'{cfg["cond"]} policy={cfg["policy"]} seed={cfg["seed"]} n_pgc={pop.n_pgc}'
        f'{" (silenced)" if cfg.get("silence_pgc") else ""}/'
        f'{pop.n_cells} params={n_params/1e6:.2f}M device={dev} lr={cfg["lr"]} '
        f'eps={cfg["rms_eps"]} encoder_norm={cfg["encoder_norm"]}', flush=True)

  def set_lr(lr):
    for i, g in enumerate(opt.param_groups):
      g['lr'] = lr if i == 0 else (cfg['shift_lr'] if lr > 0 else 0.0)

  def param_checksum():
    with torch.no_grad():
      return round(float(sum(v.double().abs().sum().item() for v in policy.state_dict().values())), 4)

  def log_warmup(event):
    rec = dict(warmup=True, event=event, update=update, frames=frames_done,
               episodes=episodes_seen, lr=opt.param_groups[0]['lr'], warmup_left=warm_left,
               param_checksum=param_checksum())
    print(json.dumps(rec), flush=True)
    metrics_f.write(json.dumps(rec) + '\n')
    metrics_f.flush()

  if warm_left:
    set_lr(0.0)
    log_warmup('warmup_start')

  t0, f0 = time.time(), frames_done
  next_ckpt = (frames_done // cfg['ckpt_every'] + 1) * cfg['ckpt_every']
  while frames_done < cfg['frames']:
    warm = warm_left > 0      # warm-up update: gradients feed RMSProp only (lr 0)
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
        if shift_param is not None:
          for k_, v_ in (('pos', obs['pos']), ('dir', obs['dir']), ('vel', obs['vel']),
                         ('gpos', g_pos), ('gdir', g_dir), ('gvel', g_vel), ('gval', g_val)):
            roll.setdefault(k_, []).append(v_)

        obs, r_raw, done, info = env.step(a)
        if not warm:
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
          if shift_param is not None:
            g_pos = torch.where(hit[:, None], info['hit_pos'], g_pos)
            g_dir = torch.where(hit[:, None], info['hit_dir'], g_dir)
            g_vel = torch.where(hit[:, None], info['hit_vel'], g_vel)
            g_val = torch.where(hit, torch.ones_like(g_val), g_val)
        prev_action, prev_reward = a, r
        if bool(done.any()):
          keep = (~done).float()
          goal_code = goal_code * keep[:, None]
          g_val = g_val * keep
          state = (state[0] * keep[:, None], state[1] * keep[:, None])
          prev_action = torch.where(done, torch.full_like(a, -1), a)
          prev_reward = prev_reward * keep
          es = info['ep_stats']
          if es is not None and not warm:
            for j in range(es['env'].numel()):
              ret, ng = float(es['ret'][j]), int(es['n_goals'][j])
              fg, pe, pen = int(es['first_goal_t'][j]), float(es['path_eff'][j]), int(es['path_eff_n'][j])
              tail = f',{env.room_names[int(es["room"][j])]},{int(es["n_goals_post"][j])}' if room_cols else ''
              ep_f.write(f'{frames_done},{update},{int(es["env"][j])},{ret:.1f},{ng},{fg},{pe:.4f},{pen}{tail}\n')
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
    codes, goals = roll['code'], roll['goal']
    if shift_param is not None:
      # Recompute both codes from the stored states with the look-ahead vectors
      # as parameters, so the loss gradient reaches them.
      cs = cell_shift(shift_param)
      codes = code_src(torch.cat(roll['pos']), torch.cat(roll['dir']), torch.cat(roll['vel']),
                       shift=cs).view(T, n, -1)
      goals = (code_src.goal(torch.cat(roll['gpos']), torch.cat(roll['gdir']),
                             torch.cat(roll['gvel']), shift=cs)
               * torch.cat(roll['gval'])[:, None]).view(T, n, -1)
      if (update + 1) % cfg['log_every'] == 0:
        with torch.no_grad():
          recode_err = max(float((codes - torch.stack(roll['code'])).abs().max()),
                           float((goals - torch.stack(roll['goal'])).abs().max()))
    st = state0
    logits_l, v_l = [], []
    for t in range(T):
      logits, v, st = policy.step_from_embedding(
          E[t], codes[t], goals[t], roll['pa'][t], roll['pr'][t], st,
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
    if shift_param is not None:
      shift_gn = float(shift_param.grad.norm())
      if cfg.get('shift_scramble'):
        shift_param.grad.mul_(torch.randint(0, 2, shift_param.shape, device=dev).float() * 2 - 1)
    opt.step()
    if shift_param is not None:
      # The population reads the new vectors from now on; the remembered goal
      # code is re-read with them too, so g* always is the code at the stored
      # goal state under the current look-aheads.
      cs = cell_shift(shift_param.data)
      pop.set_shift(cs[:, 0], cs[:, 1])
      with torch.no_grad():
        goal_code = code_src.goal(g_pos, g_dir, g_vel) * g_val[:, None]
    state = (state[0].detach(), state[1].detach())
    if warm:
      # Warm-up: the step above only updated RMSProp's square_avg (lr 0). It
      # counts neither as a learning update nor towards the frame budget.
      warm_left -= 1
      if warm_left == 0:
        set_lr(cfg['lr'])
        log_warmup('warmup_end')
        save_resume()      # a relaunch now resumes with the warmed-up optimizer
        t0, f0 = time.time(), frames_done
      continue
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
      if shift_param is not None:
        rec.update(shift_stats())
        rec['shift_grad_norm'] = round(shift_gn, 5)
        rec['recode_err'] = round(recode_err, 7)
      print(json.dumps(rec), flush=True)
      metrics_f.write(json.dumps(rec) + '\n')
      metrics_f.flush()
      ep_f.flush()
    if update % cfg['resume_every'] == 0:
      save_resume()
    if frames_done >= next_ckpt:
      torch.save(ckpt_dict(), os.path.join(cfg['out'], f'ckpt_{frames_done:010d}.pt'))
      next_ckpt += cfg['ckpt_every']

  save_resume()
  torch.save(ckpt_dict(), final_path)
  ep_f.close()
  metrics_f.close()
  print('done', flush=True)


if __name__ == '__main__':
  main()
