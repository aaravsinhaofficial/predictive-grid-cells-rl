"""The earlier PCA protocol (scripts/lstm_variable_replay.py, results/lstm_variable_1B)
applied to any checkpoint, in the familiar room and in the changed rooms.

One scripted trajectory set is replayed through every network: a controller
with access to the true goal walks to the goal, then to two random waypoints,
and back (16 episodes of 1,350 decisions, the environment's own motor and image
noise). Every network therefore sees the same images, positions, actions and
rewards; each uses its own population code and its own goal snapshots. PCA of
the 256 LSTM outputs h_t over every fifth decision after step 100 with speed
above 0.05 m/s; centred, no scaling of units. The networks' own action choices
play no role.

  familiar  scripts.lstm_variable_replay.make_replay, unchanged (replay seed
            271828): for the 1 B checkpoints this reproduces
            results/lstm_variable_1B/summary.json.
  changed   the same controller in the ten changed layouts (episode e is in
            layout e mod 10, goal fixed), steered around the wall and the
            obstacle along the shortest path.

  python -m scripts.room.replay_pca --jobs before            # the 1 B agents
  python -m scripts.room.replay_pca --jobs after,same        # after the 250 M frames
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from code.common import CodeSource, build_env, build_policy, build_population  # noqa: E402
from code.env import TRAIN_LAYOUTS  # noqa: E402
from scripts.lstm_variable_replay import make_replay  # noqa: E402
from scripts.room.geo import Room, blocked  # noqa: E402

JOBS = {  # state -> (run directory pattern, tag, seeds)
    'before': ('runs_ext/{c}_lstm_s{s}', '1000M', range(1, 9)),
    'after': ('runs_room/{c}room_lstm_s{s}', '1250M', range(1, 9)),
    'same': ('runs_room/{c}same_lstm_s{s}', '1250M', (1, 2)),
}
PAD = 0.05     # route corners sit this far outside the walls' collision margin (m)
_SHARED = {}


class Router:
  """Next point to steer for on the shortest walkable path to a target."""

  def __init__(self, boxes, margin, lim):
    self.true = Room(boxes, margin=margin, lim=lim)
    self.c = Room(boxes, margin=margin + PAD, lim=lim).corners
    C = len(self.c)
    self.D = np.zeros((0, 0))
    if C:
      vis = ~blocked(self.c[:, None], self.c[None], self.true.rects)
      D = np.where(vis, np.linalg.norm(self.c[:, None] - self.c[None], axis=-1), np.inf)
      for k in range(C):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
      self.D = D

  def aim(self, p, q):
    if not len(self.c) or not blocked(p[None], q[None], self.true.rects)[0]:
      return q
    dp = np.linalg.norm(p[None] - self.c, axis=1)
    ds = np.where(blocked(p[None], self.c, self.true.rects) | (dp < 0.06), np.inf, dp)
    dg = np.where(blocked(q[None], self.c, self.true.rects), np.inf, np.linalg.norm(q[None] - self.c, axis=1))
    tot = ds[:, None] + self.D + dg[None, :]
    if not np.isfinite(tot).any():
      return q
    return self.c[np.unravel_index(np.argmin(tot), tot.shape)[0]]


@torch.inference_mode()
def make_replay_rooms(cfg, episodes=16, replay_seed=271828, stride=5):
  """make_replay of scripts/lstm_variable_replay.py with walls: the same
  controller and bookkeeping; targets are reached along the shortest path and
  random waypoints are drawn in free space."""
  env = build_env(cfg, episodes, replay_seed, 'cpu', stagger=False, paired=True, rooms=list(TRAIN_LAYOUTS))
  obs = env.reset_all()
  boxes = env.boxes.numpy().copy()
  room_id = env.room_id.numpy().copy()      # layout of each episode (TRAIN_LAYOUTS index)
  routers = [Router(boxes[e], env.wall_margin, env.half - env.wall_margin) for e in range(episodes)]
  rng = np.random.default_rng(replay_seed + 1)

  def waypoint(e):
    b = boxes[e]
    for _ in range(1000):
      q = rng.uniform(-1.05, 1.05, 2)
      if not ((q[0] > b[:, 0] - 0.25) & (q[0] < b[:, 1] + 0.25) & (q[1] > b[:, 2] - 0.25) & (q[1] < b[:, 3] + 0.25)).any():
        return torch.tensor(q, dtype=torch.float32)
    raise RuntimeError('no free waypoint')

  target = env.goal.clone()
  mode = np.zeros(episodes, dtype=int)
  known = np.zeros(episodes, dtype=bool)
  memory = {k: np.zeros((episodes, 2), np.float32) for k in ('pos', 'dir', 'vel')}
  pa = np.full(episodes, -1, dtype=np.int64)
  pr = np.zeros(episodes, dtype=np.float32)
  fields = {k: [] for k in ('pos', 'dir', 'vel', 'hd', 'goal_pos', 'goal_known', 'memory_pos', 'memory_dir',
                            'memory_vel', 'prev_action', 'prev_reward', 'episode', 'step', 'action', 'hit',
                            'hit_pos', 'hit_dir', 'hit_vel', 'done', 'hit_wall')}
  rgb = torch.empty((cfg['episode_len'], episodes, 3, 84, 84), dtype=torch.float32)
  episode_goals = env.goal.numpy().copy()
  for t in range(cfg['episode_len']):
    reached = (target - obs['pos']).norm(dim=1).numpy() < .15
    for e in np.flatnonzero(reached & (mode > 0)):
      if mode[e] == 1:
        mode[e] = 2
        target[e] = waypoint(e)
      else:
        mode[e] = 0
        target[e] = env.goal[e]
    pos = obs['pos'].numpy().astype(np.float64)
    aim = torch.tensor(np.array([routers[e].aim(pos[e], target[e].numpy().astype(np.float64))
                                 for e in range(episodes)]), dtype=torch.float32)
    delta = aim - obs['pos']
    angle = torch.atan2(delta[:, 1], delta[:, 0]) - obs['hd']
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    error = angle - .20 * env.vr
    action = torch.where(error.abs() < .22, 2, torch.where(error > 0, 0, 1))
    for k in ('pos', 'dir', 'vel', 'hd'):
      fields[k].append(obs[k].numpy().copy())
    for k, value in dict(goal_pos=env.goal.numpy(), goal_known=known, memory_pos=memory['pos'],
                         memory_dir=memory['dir'], memory_vel=memory['vel'], prev_action=pa, prev_reward=pr,
                         episode=np.arange(episodes), step=np.full(episodes, t), action=action.numpy()).items():
      fields[k].append(value.copy())
    rgb[t] = obs['rgb']
    obs, reward, done, info = env.step(action)
    for k in ('hit', 'hit_pos', 'hit_dir', 'hit_vel', 'hit_wall'):
      fields[k].append(info[k].numpy().copy())
    fields['done'].append(done.numpy().copy())
    hit = info['hit'].numpy()
    for k in memory:
      memory[k][hit] = info[f'hit_{k}'].numpy()[hit]
    known[hit] = True
    for e in np.flatnonzero(hit):
      mode[e] = 1
      target[e] = waypoint(e)
    pa = action.numpy().copy()
    pr = (reward.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else reward).numpy().copy()
  replay = {k: np.stack(v).reshape((-1,) + np.asarray(v[0]).shape[1:]) for k, v in fields.items()}
  speed = np.linalg.norm(replay['vel'], axis=1)
  selected = (replay['step'] >= 100) & (replay['step'] % stride == 0) & (speed > .05)
  replay.update(selected=selected, sample_index=np.flatnonzero(selected), episode_goals=episode_goals,
                episodes=np.array(episodes), episode_len=np.array(cfg['episode_len']), boxes=boxes, room_id=room_id)
  hits = replay['hit'].reshape(-1, episodes)
  assert np.all(hits.sum(0) >= 2), f'every episode needs repeated goal arrivals: {hits.sum(0).tolist()}'
  print(f'changed-room replay: {selected.sum():,} saved states; hits/episode {hits.sum(0).tolist()}; '
        f'wall contact {replay["hit_wall"].mean():.3f}', flush=True)
  return replay, rgb.flatten(0, 1)


@torch.inference_mode()
def record(cfg, policy_state, replay, rgb, batch_size=256):
  """h_t of one network on the replay (as record_model of the original script)."""
  net = build_policy(cfg, 'cpu').eval().requires_grad_(False)
  net.load_state_dict(policy_state, strict=True)
  source = CodeSource(cfg, build_population(cfg, 'cpu'), 'cpu')
  n, steps = int(replay['episodes']), int(replay['episode_len'])
  emb = torch.cat([net.embed(rgb[i:i + batch_size]) for i in range(0, len(rgb), batch_size)]).reshape(steps, n, -1)
  code = source(*(torch.from_numpy(replay[k]) for k in ('pos', 'dir', 'vel'))).reshape(steps, n, -1)
  goal = source.goal(*(torch.from_numpy(replay[f'memory_{k}']) for k in ('pos', 'dir', 'vel')))
  goal *= torch.from_numpy(replay['goal_known'])[:, None]
  goal = goal.reshape(steps, n, -1)
  pa = torch.from_numpy(replay['prev_action']).reshape(steps, n)
  pr = torch.from_numpy(replay['prev_reward']).reshape(steps, n)
  selected = replay['selected'].reshape(steps, n)
  state = net.zero_state(n, 'cpu')
  hs = []
  for t in range(steps):
    _, _, state = net.step_from_embedding(emb[t], code[t], goal[t], pa[t], pr[t], state)
    if selected[t].any():
      hs.append(state[0][selected[t]].numpy().copy())
  return np.concatenate(hs)


def spectrum(h):
  r = PCA(svd_solver='full').fit(h).explained_variance_ratio_
  cum = r.cumsum()
  return dict(samples=int(len(h)), var3=float(cum[2]), var10=float(cum[9]), var20=float(cum[19]),
              n50=int(np.searchsorted(cum, .5) + 1), n90=int(np.searchsorted(cum, .9) + 1),
              participation_ratio=float(1.0 / (r ** 2).sum()), total_var=float(h.var(0, ddof=1).sum()),
              ratios=[float(x) for x in r[:60]])


def one(job):
  name, tag, run, out, ckpt = job
  path = os.path.join(out, f'{name}@{tag}.json')
  cfg = json.load(open(os.path.join(ROOT, run, 'config.json')))
  ck = torch.load(ckpt, map_location='cpu', weights_only=True)
  res = dict(name=name, tag=tag, run=run, ckpt=os.path.relpath(ckpt, ROOT), frames=int(ck['frames']), cond=cfg['cond'], seed=cfg['seed'])
  arrays = {}
  for proto in ('familiar', 'changed'):
    replay, rgb = _SHARED[proto]
    h = record(cfg, ck['policy'], replay, rgb)
    res[proto] = spectrum(h)
    arrays[f'cov_{proto}'] = np.cov(h.astype(np.float64), rowvar=False)     # for the alignment index
    arrays[f'mean_{proto}'] = h.astype(np.float64).mean(0)
  json.dump(res, open(path, 'w'))
  np.savez_compressed(path[:-5] + '.npz', **arrays)
  return f'{name}@{tag}: familiar {res["familiar"]["var10"]:.4f} changed {res["changed"]["var10"]:.4f}'


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--jobs', default='before')
  ap.add_argument('--out', default=os.path.join(ROOT, 'results_room', 'replay_pca'))
  ap.add_argument('--seeds', default='1,2,3,4,5,6,7,8', help='run several processes with disjoint seeds to go faster')
  ap.add_argument('--threads', type=int, default=6)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  jobs = []
  import glob
  for state in args.jobs.split(','):
    if ':' in state:      # 'long:1500' / 'same_long:1500': the checkpoint at 1500 M frames of the continued runs
      state, F = state.split(':')
      pat, tag, seeds = JOBS['after' if state == 'long' else 'same'][0], f'{F}M', JOBS['after' if state == 'long' else 'same'][2]
      ckname = f'ckpt_{F}*.pt'
    else:
      pat, tag, seeds = JOBS[state]
      ckname = 'ckpt_final.pt'
    for s in [x for x in seeds if x in [int(v) for v in args.seeds.split(',')]]:
      for c in ('gc100', 'pgc33'):
        run = pat.format(c=c, s=s)
        name = os.path.basename(run)
        cks = sorted(glob.glob(os.path.join(ROOT, run, ckname)))
        if not cks:
          print('missing', run, ckname)
          continue
        if args.force or not os.path.exists(os.path.join(args.out, f'{name}@{tag}.json')):
          jobs.append((name, tag, run, args.out, cks[0]))
  if not jobs:
    return
  torch.set_num_threads(args.threads)
  torch.manual_seed(271828)
  cfg = json.load(open(os.path.join(ROOT, 'runs_ext', 'gc100_lstm_s1', 'config.json')))
  cfg['rooms'] = 'none'
  _SHARED['familiar'] = make_replay(cfg, 16, 271828, 5)
  _SHARED['changed'] = make_replay_rooms(cfg, 16, 271828, 5)
  if not os.path.exists(os.path.join(args.out, 'replay_changed.npz')):
    np.savez_compressed(os.path.join(args.out, 'replay_changed.npz'), **_SHARED['changed'][0])
  for job in jobs:
    print(one(job), flush=True)


if __name__ == '__main__':
  main()
