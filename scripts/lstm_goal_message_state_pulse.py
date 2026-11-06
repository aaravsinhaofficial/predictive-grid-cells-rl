"""Natural goal-message state substitutions in the original frozen RL agents.

Replay actual recent inputs with a rotated, valid pickup/direction goal snapshot.
Insert the elicited state at the actual physical endpoint, restore the real goal
message, and let the original reward-trained actor control the original arena.
No decoder, heading-supervised agent, or presumed ring manifold is used.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena
from lstm_goal_state_pulse import ENV_FIELDS, INPUTS, sha, write_json, wrap

PULSES = ('baseline', 'false_goal_state', 'false_goal_c_only', 'false_goal_h_only')


def rotate(v, angle):
    cs, sn = angle.cos(), angle.sin()
    return torch.stack((v[..., 0] * cs - v[..., 1] * sn, v[..., 0] * sn + v[..., 1] * cs), -1)


@torch.inference_mode()
def collect(net, pop, cfg, args):
    n, device = args.episodes, args.device
    env = MatchedArena(cfg, n, 1, args.eval_seed, device)
    obs = env.reset_all()
    h, c = net.zero_state(n, device)
    goal = torch.zeros(n, cfg['n_cells'], device=device)
    plain_goal = torch.zeros_like(goal)
    memories = {k: torch.zeros(n, 2, device=device) for k in ('pos', 'dir', 'vel')}
    pa = torch.full((n,), -1, dtype=torch.long, device=device)
    pr = torch.zeros(n, device=device)
    age = torch.zeros(n, dtype=torch.long, device=device)
    records, snapshots = deque(maxlen=args.prepare), {}
    captured = np.full(n, -1, int)
    rng = torch.Generator(device=device).manual_seed(args.eval_seed + 1)
    signs = torch.tensor(np.where(np.arange(n) % 2, -1., 1.), dtype=torch.float32, device=device)
    future = {name: np.zeros((args.horizon, n, net.lstm.input_size), np.float32) for name in INPUTS}
    for step in range(cfg['episode_len']):
        cur = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        cur_plain = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0)
        e = net.embed(obs['rgb'])
        x = net._concat(e, cur, goal, pa, pr, None)
        before_h, before_c = h, c
        h, c = net.lstm(x, (h, c))
        records.append(dict(x=x.cpu().numpy(), h=before_h.cpu().numpy(), c=before_c.cpu().numpy(),
                            pos=obs['pos'].cpu().numpy()))
        ages = step - captured
        active = np.flatnonzero((captured >= 0) & (ages >= 1) & (ages <= args.horizon))
        if len(active):
            xs = (x, net._concat(e, cur, plain_goal, pa, pr, None), net._concat(e, cur_plain, goal, pa, pr, None))
            for name, xx in zip(INPUTS, xs):
                future[name][ages[active]-1, active] = xx[active].cpu().numpy()
        if len(records) == args.prepare and step + args.horizon < cfg['episode_len']:
            distance = (env.goal - env.pos).norm(dim=1)
            angle = np.deg2rad(args.angle_degrees) * signs
            false_center = env.pos + rotate(env.goal - env.pos, angle)
            false_pickup = env.pos + rotate(memories['pos'] - env.pos, angle)
            false_dir = rotate(memories['dir'], angle)
            false_vel = rotate(memories['vel'], angle)
            reverse_center = env.pos + rotate(env.goal - env.pos, -angle)
            reverse_pickup = env.pos + rotate(memories['pos'] - env.pos, -angle)
            inbounds = torch.stack((false_center.abs().amax(1), false_pickup.abs().amax(1),
                                    reverse_center.abs().amax(1), reverse_pickup.abs().amax(1)), 1).amax(1) < 1.17
            eligible = (env.n_goals > 0) & (age >= args.prepare + 5) & (distance >= .9) & (distance <= 1.8) & inbounds
            for slot in np.flatnonzero(eligible.cpu().numpy() & (captured < 0)):
                history = list(records)
                # No fictitious reward conflict anywhere in the replayed history.
                prefix_pos = np.stack([r['pos'][slot] for r in history])
                fc = false_center[slot].cpu().numpy()
                if np.linalg.norm(prefix_pos - fc, axis=1).min() <= cfg['goal_radius']:
                    continue
                snapshots[int(slot)] = {**{k: getattr(env, k)[slot].cpu().numpy().copy() for k in ENV_FIELDS},
                    'h': h[slot].cpu().numpy().copy(), 'c': c[slot].cpu().numpy().copy(),
                    'goal_native': goal[slot].cpu().numpy().copy(), 'goal_plain': plain_goal[slot].cpu().numpy().copy(),
                    'history_x': np.stack([r['x'][slot] for r in history]),
                    'history_h': history[0]['h'][slot].copy(), 'history_c': history[0]['c'][slot].copy(),
                    'false_center': fc, 'false_pickup': false_pickup[slot].cpu().numpy(),
                    'false_dir': false_dir[slot].cpu().numpy(), 'false_vel': false_vel[slot].cpu().numpy(),
                    'memory_pos': memories['pos'][slot].cpu().numpy(), 'memory_dir': memories['dir'][slot].cpu().numpy(),
                    'memory_vel': memories['vel'][slot].cpu().numpy(),
                    'sign': signs[slot].cpu().numpy(), 'slot': np.array(slot), 'capture_step': np.array(step)}
                captured[slot] = step
        prob = net.pi(h).softmax(-1)
        action = (torch.rand(n, generator=rng, device=device)[:, None] > prob.cumsum(1)).sum(1).clamp_max(5)
        obs, reward, done, info = env.step(action)
        if info['hit'].any():
            goal = torch.where(info['hit'][:, None], pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel']), goal)
            plain_goal = torch.where(info['hit'][:, None], pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0), plain_goal)
            for key in memories:
                memories[key] = torch.where(info['hit'][:, None], info['hit_' + key], memories[key])
        age = torch.where(info['hit'] | done, torch.zeros_like(age), age + 1)
        pa, pr = action, reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if done.any(): assert step == cfg['episode_len']-1
    slots = sorted(snapshots)
    if not slots: raise RuntimeError('No eligible real return trajectories')
    saved = {k: np.stack([snapshots[s][k] for s in slots]) for k in snapshots[slots[0]]}
    saved.update({'replay_' + name: x[:, slots] for name, x in future.items()})
    return saved


@torch.inference_mode()
def condition_states(net, pop, snapshot, device):
    n = len(snapshot['h'])
    h = torch.tensor(snapshot['history_h'], device=device).repeat(2, 1)
    c = torch.tensor(snapshot['history_c'], device=device).repeat(2, 1)
    goal = pop.activity(*[torch.tensor(snapshot['false_' + key], device=device) for key in ('pickup', 'dir', 'vel')])
    for x in snapshot['history_x'].transpose(1, 0, 2):
        xx = torch.tensor(x, device=device).repeat(2, 1)
        xx[n:, 768:1280] = goal
        h, c = net.lstm(xx, (h, c))
    nh, nc = torch.tensor(snapshot['h'], device=device), torch.tensor(snapshot['c'], device=device)
    reconstruction = max(float((nh-h[:n]).abs().max()), float((nc-c[:n]).abs().max()))
    assert reconstruction < 4e-4, reconstruction
    states = torch.stack((torch.cat((nh, nc), 1), torch.cat((h[n:], c[n:]), 1),
                          torch.cat((nh, c[n:]), 1), torch.cat((h[n:], nc), 1)))
    return states.cpu().numpy(), {'native_replay_max_error': reconstruction,
        'state_change_norm_mean': (states[1:] - states[:1]).norm(dim=-1).mean(-1).cpu().tolist(),
        'natural_full_state_only': 'Full false-goal state is input-elicited. h-only and c-only hybrids are component interventions.'}


@torch.inference_mode()
def same_observation(net, snapshot, states, device):
    n, width = len(snapshot['h']), states.shape[-1] // 2
    z = torch.tensor(np.tile(states, (len(INPUTS), 1, 1)).reshape(-1, width*2), device=device)
    traces = {k: [] for k in ('state_error_norm', 'h_error_norm', 'c_error_norm', 'action_tv')}
    for step in range(len(snapshot['replay_native']) + 1):
        zz = z.reshape(len(INPUTS), len(PULSES), n, width*2)
        diff = zz[:, 1:] - zz[:, :1]
        p = net.pi(z[:, :width]).softmax(-1).reshape(len(INPUTS), len(PULSES), n, 6)
        for k, value in [('state_error_norm', diff.norm(dim=-1)), ('h_error_norm', diff[..., :width].norm(dim=-1)),
                         ('c_error_norm', diff[..., width:].norm(dim=-1)), ('action_tv', .5*(p[:, 1:]-p[:, :1]).abs().sum(-1))]:
            traces[k].append(value.cpu().numpy())
        if step == len(snapshot['replay_native']): break
        x = torch.tensor(np.stack([snapshot['replay_' + key][step] for key in INPUTS]), device=device)
        x = x[:, None].expand(-1, len(PULSES), -1, -1).reshape(len(INPUTS)*len(PULSES)*n, -1)
        h, c = net.lstm(x, (z[:, :width], z[:, width:]))
        z = torch.cat((h, c), 1)
    return {k: np.stack(v) for k, v in traces.items()}


@torch.inference_mode()
def behavior(net, pop, cfg, snapshot, states, args):
    n, groups = len(snapshot['h']), len(INPUTS)*len(PULSES)
    env = MatchedArena(cfg, n, groups, args.eval_seed + 300, args.device)
    for key in ENV_FIELDS:
        value = torch.tensor(snapshot[key], device=args.device)
        getattr(env, key).copy_(value.repeat(groups, *([1]*(value.ndim-1))))
    z = torch.tensor(np.tile(states, (len(INPUTS), 1, 1)).reshape(-1, 512), device=args.device)
    h, c = z[:, :256], z[:, 256:]
    gn = torch.tensor(np.tile(snapshot['goal_native'], (groups, 1)), device=args.device)
    gp = torch.tensor(np.tile(snapshot['goal_plain'], (groups, 1)), device=args.device)
    input_index = torch.arange(len(INPUTS), device=args.device).repeat_interleave(len(PULSES)*n)
    rng = torch.Generator(device=args.device).manual_seed(args.eval_seed + 301)
    first = torch.full((groups*n,), args.horizon, dtype=torch.long, device=args.device)
    traces = {k: [] for k in ('position', 'distance', 'heading', 'action_probability', 'hit')}
    for step in range(args.horizon + 1):
        for key, value in [('position', env.pos), ('distance', (env.goal-env.pos).norm(dim=1)), ('heading', env.hd)]:
            traces[key].append(value.cpu().numpy().reshape(len(INPUTS), len(PULSES), n, *value.shape[1:]).copy())
        if step == args.horizon: break
        p = net.pi(h).softmax(-1)
        uniform = torch.rand(n, generator=rng, device=args.device).repeat(groups)
        action = (uniform[:, None] > p.cumsum(1)).sum(1).clamp_max(5)
        if step == 0:
            aa = action.reshape(len(INPUTS), len(PULSES), n)
            assert torch.equal(aa[:, 0], aa[:, 2])  # c-only leaves immediate policy unchanged.
            assert torch.equal(aa[:, 1], aa[:, 3])
        obs, reward, done, info = env.step(action)
        first = torch.where(info['hit'] & (first == args.horizon), torch.full_like(first, step), first)
        traces['action_probability'].append(p.cpu().numpy().reshape(len(INPUTS), len(PULSES), n, 6))
        traces['hit'].append(info['hit'].cpu().numpy().reshape(len(INPUTS), len(PULSES), n))
        assert not bool(done.any())
        gn = torch.where(info['hit'][:, None], pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel']), gn)
        gp = torch.where(info['hit'][:, None], pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0), gp)
        current = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        plain = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0)
        current = torch.where((input_index == 2)[:, None], plain, current)
        goal = torch.where((input_index == 1)[:, None], gp, gn)
        xx = net._concat(net.embed(obs['rgb']), current, goal, action, reward.clamp(-cfg['reward_clip'], cfg['reward_clip']), None)
        h, c = net.lstm(xx, (h, c))
    arrays = {k: np.stack(v) for k, v in traces.items()}
    arrays['first_hit_step'] = first.cpu().numpy().reshape(len(INPUTS), len(PULSES), n)
    return arrays


def summarize(snapshot, replay, trajectories, args):
    result = {}
    theta = np.arctan2((snapshot['goal']-snapshot['pos'])[:, 1], (snapshot['goal']-snapshot['pos'])[:, 0])
    transverse = np.stack((-np.sin(theta), np.cos(theta)), 1) * snapshot['sign'][:, None]
    for i, name in enumerate(INPUTS):
        result[name] = {}
        for j, pulse in enumerate(PULSES):
            first = trajectories['first_hit_step'][i, j]
            path = trajectories['position'][:, i, j] - trajectories['position'][:, i, 0]
            row = {'goal_hit_fraction': float((first < args.horizon).mean()),
                   'capped_goal_time_seconds': float(np.minimum(first+1, args.horizon).mean()/15),
                   'first_second_path_change_m': float(np.linalg.norm(path[:16], axis=-1).mean()),
                   'first_second_signed_cross_goal_change_m': float(np.sum(path[15]*transverse, axis=-1).mean()),
                   'first_second_distance_progress_m': float((trajectories['distance'][0, i, j]-trajectories['distance'][15, i, j]).mean())}
            p = trajectories['action_probability'][0, i, j]
            base = trajectories['action_probability'][0, i, 0]
            # The first decision's movement tendency, expressed in world axes.
            local = np.stack(((p-base)[:, 2]-(p-base)[:, 3], (p-base)[:, 4]-(p-base)[:, 5]), 1)
            hd = snapshot['hd']; cs, sn = np.cos(hd), np.sin(hd)
            world = np.stack((local[:, 0]*cs-local[:, 1]*sn, local[:, 0]*sn+local[:, 1]*cs), 1)
            row['initial_movement_probability_shift_toward_false_goal'] = float(np.sum(world*transverse, 1).mean())
            if j:
                initial = replay['state_error_norm'][0, i, j-1]
                for step in (0, 1, 5, 15, 30, 60):
                    if step < len(replay['action_tv']):
                        row[f'replay_{step}_action_tv'] = float(replay['action_tv'][step, i, j-1].mean())
                        row[f'replay_{step}_state_error_ratio'] = float((replay['state_error_norm'][step, i, j-1]/np.maximum(initial, 1e-12)).mean())
            result[name][pulse] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('pgc33', 'gc100'), default='pgc33')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--eval-seed', type=int, default=872601)
    parser.add_argument('--prepare', type=int, default=15)
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--angle-degrees', type=float, default=15.)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    cfgpath = ROOT/f'models/{args.condition}/{args.condition}_lstm_s{args.seed}_1000M.json'
    cfg = json.loads(cfgpath.read_text()); checkpointpath = cfgpath.with_suffix('.pt')
    checksum = sha(checkpointpath); checkpoint = torch.load(checkpointpath, map_location='cpu', weights_only=True)
    assert checkpoint['frames'] == 1_000_000_000
    net = build_policy(cfg, args.device).eval().requires_grad_(False); net.load_state_dict(checkpoint['policy'])
    pop = build_population(cfg, args.device)
    protocol = {'stage': 'exploratory_original_RL_goal_message_state_pilot',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'checkpoint_sha256': checksum, 'source_hashes': {p: sha(ROOT/p) for p in
        ('scripts/lstm_goal_message_state_pulse.py', 'scripts/lstm_goal_state_pulse.py', 'scripts/lstm_code_swap.py', 'code/nets.py', 'code/env.py', 'code/common.py', 'code/codes.py')},
        'inputs': INPUTS, 'pulses': PULSES,
        'state_intervention': 'Replay15 real recent inputs with only goal snapshot rotated around current observer; restore actual goal message after intervention.',
        'interpretation': 'Input-elicited goal-conditioned state and h/c component interventions. No ring or autonomous goal storage assumed.',
        'physical_environment_unchanged': True, 'original_reward_trained_actor_frozen': True}
    protocolpath = args.out/'protocol.json'
    if protocolpath.exists(): assert json.loads(protocolpath.read_text()) == json.loads(json.dumps(protocol))
    write_json(protocolpath, protocol); started = time.monotonic()
    path = args.out/'native_windows.npz'
    if not path.exists(): np.savez_compressed(path, **collect(net, pop, cfg, args))
    snapshot = dict(np.load(path)); states, validity = condition_states(net, pop, snapshot, args.device)
    np.savez_compressed(args.out/'conditioned_states.npz', states=states)
    replay = same_observation(net, snapshot, states, args.device)
    np.savez_compressed(args.out/'same_observation_replay.npz', **replay)
    trajectories = behavior(net, pop, cfg, snapshot, states, args)
    np.savez_compressed(args.out/'closed_loop.npz', **trajectories)
    summary = {'protocol': protocol, 'eligible_trials': len(snapshot['h']), 'attempted_episodes': args.episodes,
               'validity': validity, 'results': summarize(snapshot, replay, trajectories, args),
               'elapsed_seconds': time.monotonic()-started, 'checkpoint_unchanged': sha(checkpointpath) == checksum}
    assert summary['checkpoint_unchanged']; write_json(args.out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
