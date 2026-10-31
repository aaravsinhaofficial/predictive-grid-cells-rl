"""Frozen 1B LSTM probe with varying motion, facing, and remembered goals.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_variable_replay.py
The scripted trajectories are identical for every model. Policies are evaluated,
not trained; their predicted actions do not control this off-policy replay.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.decomposition import PCA
from code.common import CodeSource, build_env, build_policy, build_population

CONDITIONS = ('gc100', 'pgc33')
ENV_KEYS = ('arena_half', 'episode_len', 'action_repeat', 'v_max', 'rot_max',
            'goal_radius', 'vision', 'img_noise', 'randomize_textures', 'reward_clip')


def load_config(models, condition, seed):
    stem = models / condition / f'{condition}_lstm_s{seed}_1000M'
    return stem, json.loads(stem.with_suffix('.json').read_text())


@torch.inference_mode()
def make_replay(cfg, episodes=16, replay_seed=271828, stride=5):
    """Generate separate standard-length episodes in parallel, no manual resets.

    The controller visits a goal, then two random waypoints, then the goal again.
    It uses privileged positions to ensure coverage and actual goal encounters;
    it is deliberately not a behavioral evaluation of either trained policy.
    """
    env = build_env(cfg, episodes, replay_seed, 'cpu', stagger=False, paired=True)
    obs = env.reset_all()
    rng = np.random.default_rng(replay_seed + 1)
    target = env.goal.clone()
    mode = np.zeros(episodes, dtype=int)  # 0: goal; 1/2: random waypoints
    known = np.zeros(episodes, dtype=bool)
    memory = {k: np.zeros((episodes, 2), np.float32) for k in ('pos', 'dir', 'vel')}
    snapshot_id = np.zeros(episodes, dtype=np.int64)
    pa = np.full(episodes, -1, dtype=np.int64)
    pr = np.zeros(episodes, dtype=np.float32)
    fields = {k: [] for k in ('pos', 'dir', 'vel', 'hd', 'goal_pos', 'goal_known',
        'memory_pos', 'memory_dir', 'memory_vel', 'snapshot_id', 'prev_action',
        'prev_reward', 'episode', 'step', 'action', 'hit', 'hit_pos', 'hit_dir',
        'hit_vel', 'done')}
    # 1.83 GB for the default 16 episodes; retain exact float32 images in RAM
    # for common input to all models, without creating a large disk artifact.
    rgb = torch.empty((cfg['episode_len'], episodes, 3, 84, 84), dtype=torch.float32)
    episode_goals = env.goal.numpy().copy()
    landmarks = env.land_pos.numpy().copy()
    stripe_phase = env.stripe_phase.numpy().copy()
    for t in range(cfg['episode_len']):
        reached = (target - obs['pos']).norm(dim=1).numpy() < .15
        for e in np.flatnonzero(reached & (mode > 0)):
            if mode[e] == 1:
                mode[e] = 2
                target[e] = torch.tensor(rng.uniform(-1.05, 1.05, 2), dtype=torch.float32)
            else:
                mode[e] = 0
                target[e] = env.goal[e]
        delta = target - obs['pos']
        angle = torch.atan2(delta[:, 1], delta[:, 0]) - obs['hd']
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        # Anticipate residual rotational momentum to reduce controller overshoot.
        error = angle - .20 * env.vr
        action = torch.where(error.abs() < .22, 2, torch.where(error > 0, 0, 1))
        for k in ('pos', 'dir', 'vel', 'hd'):
            fields[k].append(obs[k].numpy().copy())
        for k, value in dict(goal_pos=env.goal.numpy(), goal_known=known,
            memory_pos=memory['pos'], memory_dir=memory['dir'], memory_vel=memory['vel'],
            snapshot_id=snapshot_id, prev_action=pa, prev_reward=pr,
            episode=np.arange(episodes), step=np.full(episodes, t), action=action.numpy()).items():
            fields[k].append(value.copy())
        rgb[t] = obs['rgb']
        obs, reward, done, info = env.step(action)
        for k in ('hit', 'hit_pos', 'hit_dir', 'hit_vel'):
            fields[k].append(info[k].numpy().copy())
        fields['done'].append(done.numpy().copy())
        hit = info['hit'].numpy()
        # Snapshot is from pickup BEFORE respawn, available at the next step.
        for k in memory:
            memory[k][hit] = info[f'hit_{k}'].numpy()[hit]
        known[hit] = True
        snapshot_id[hit] += 1
        for e in np.flatnonzero(hit):
            mode[e] = 1
            target[e] = torch.tensor(rng.uniform(-1.05, 1.05, 2), dtype=torch.float32)
        pa = action.numpy().copy()
        pr = reward.clamp(-cfg['reward_clip'], cfg['reward_clip']).numpy().copy() if cfg['reward_clip'] > 0 else reward.numpy().copy()
        if t < cfg['episode_len'] - 1:
            assert not done.any()
        else:
            assert done.all()
    replay = {k: np.stack(v).reshape((-1,) + np.asarray(v[0]).shape[1:])
              for k, v in fields.items()}
    speed = np.linalg.norm(replay['vel'], axis=1)
    selected = (replay['step'] >= 100) & (replay['step'] % stride == 0) & (speed > .05)
    replay['selected'] = selected
    replay['sample_index'] = np.flatnonzero(selected)
    replay['episode_goals'] = episode_goals
    replay['landmark_pos'] = landmarks
    replay['stripe_phase'] = stripe_phase
    replay['replay_seed'] = np.array(replay_seed)
    replay['episodes'] = np.array(episodes)
    replay['episode_len'] = np.array(cfg['episode_len'])
    assert len(np.unique(episode_goals, axis=0)) == episodes
    hits = replay['hit'].reshape(-1, episodes)
    assert np.all(hits.sum(0) >= 2), 'Every episode should have repeated actual discoveries'
    assert not replay['goal_known'][:episodes].any()
    assert (replay['prev_action'][:episodes] == -1).all()
    assert (replay['prev_reward'][:episodes] == 0).all()
    # Memories change exactly at actual hits; no goal is preloaded or invented.
    for k in ('pos', 'dir', 'vel'):
        saved = replay[f'memory_{k}'].reshape(-1, episodes, 2)
        pickup = replay[f'hit_{k}'].reshape(-1, episodes, 2)
        assert np.array_equal(saved[1:][hits[:-1]], pickup[:-1][hits[:-1]])
        assert np.array_equal(saved[1:][~hits[:-1]], saved[:-1][~hits[:-1]])
    assert np.allclose(replay['goal_pos'].reshape(-1, episodes, 2), episode_goals[None])
    angles = np.arctan2(replay['dir'][selected, 1], replay['dir'][selected, 0])
    counts, _ = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, 9))
    assert (counts > 0).all(), 'Selected replay must cover every motion octant'
    print(f'Replay: {episodes} episodes, {len(selected):,} decisions; {selected.sum():,} saved moving states; hits/episode {hits.sum(0).tolist()}', flush=True)
    return replay, rgb.flatten(0, 1)


@torch.inference_mode()
def record_model(stem, cfg, replay, rgb, batch_size=256):
    path = stem.with_suffix('.pt')
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000 and cfg['policy'] == 'lstm'
    assert cfg.get('extra_input', 'none') == 'none' and cfg.get('code_noise', 0) == 0
    net = build_policy(cfg, 'cpu').eval().requires_grad_(False)
    net.load_state_dict(ckpt['policy'], strict=True)
    pop = build_population(cfg, 'cpu')
    source = CodeSource(cfg, pop, 'cpu')
    n = int(replay['episodes'])
    steps = int(replay['episode_len'])
    embedding = torch.cat([net.embed(rgb[i:i + batch_size]) for i in range(0, len(rgb), batch_size)]).reshape(steps, n, -1)
    code = source(*(torch.from_numpy(replay[k]) for k in ('pos', 'dir', 'vel'))).reshape(steps, n, -1)
    goal = source.goal(*(torch.from_numpy(replay[f'memory_{k}']) for k in ('pos', 'dir', 'vel')))
    goal *= torch.from_numpy(replay['goal_known'])[:, None]
    goal = goal.reshape(steps, n, -1)
    assert torch.equal(goal[0], torch.zeros_like(goal[0]))
    # Verify reconstructed goal input exactly follows the online evaluator.
    online_goal = torch.zeros(n, cfg['n_cells'])
    hit = replay['hit'].reshape(steps, n)
    for t in range(steps):
        assert torch.equal(goal[t], online_goal)
        if hit[t].any():
            updated = source.goal(*(torch.from_numpy(replay[f'hit_{k}'].reshape(steps, n, 2)[t]) for k in ('pos', 'dir', 'vel')))
            online_goal = torch.where(torch.from_numpy(hit[t])[:, None], updated, online_goal)
    pa = torch.from_numpy(replay['prev_action']).reshape(steps, n)
    pr = torch.from_numpy(replay['prev_reward']).reshape(steps, n)
    selected = replay['selected'].reshape(steps, n)
    state = net.zero_state(n, 'cpu')
    recorded = {k: [] for k in ('h', 'c', 'logits', 'value')}
    for t in range(steps):
        logits, value, state = net.step_from_embedding(embedding[t], code[t], goal[t], pa[t], pr[t], state)
        if selected[t].any():
            for k, arr in zip(recorded, (*state, logits, value)):
                recorded[k].append(arr[selected[t]].numpy().copy())
    data = {k: np.concatenate(v) for k, v in recorded.items()}
    assert all(np.isfinite(v).all() for v in data.values())
    assert data['h'].shape == data['c'].shape == (len(replay['sample_index']), 256)
    pca = PCA(svd_solver='full').fit(data['h'])
    data.update(scores=pca.transform(data['h'])[:, :3], pca_mean=pca.mean_,
        components=pca.components_, explained_variance_ratio=pca.explained_variance_ratio_)
    # Independent sequence API checks the original LSTMCell recurrence including
    # persistence across hit respawns. No h/c resets occur within these episodes.
    inputs = net._concat(embedding.flatten(0, 1), code.flatten(0, 1), goal.flatten(0, 1), pa.flatten(), pr.flatten(), None).reshape(steps, n, -1)
    seq = torch.nn.LSTM(inputs.shape[-1], net.nh).eval().requires_grad_(False)
    seq.load_state_dict({f'{k}_l0': v for k, v in net.lstm.state_dict().items()})
    reference, _ = seq(inputs)
    max_error = float(np.max(np.abs(reference.flatten(0, 1)[replay['sample_index']].numpy() - data['h'])))
    assert max_error < 1e-4, f'LSTM sequence validation: {max_error}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint_hash
    ratios = pca.explained_variance_ratio_
    meta = dict(condition=cfg['cond'], seed=cfg['seed'], frames=ckpt['frames'],
        checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=checkpoint_hash,
        samples=len(data['h']), pc3_variance=float(ratios[:3].sum()),
        pc10_variance=float(ratios[:10].sum()),
        dimensions_90pct=int(np.searchsorted(ratios.cumsum(), .9) + 1),
        sequence_cell_max_error=max_error,
        goal_snapshot_changes=int(hit[:-1].sum()))
    return data, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_variable_1B')
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--replay-seed', type=int, default=271828)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.replay_seed)
    start = time.monotonic()
    _, cfg = load_config(args.models, 'gc100', 1)
    for condition in CONDITIONS:
        for seed in args.seeds:
            _, other = load_config(args.models, condition, seed)
            assert all(cfg[k] == other[k] for k in ENV_KEYS)
    replay, rgb = make_replay(cfg, args.episodes, args.replay_seed, args.stride)
    np.savez_compressed(args.out / 'replay.npz', **replay)
    # This verifies the same exact images, actions, and rewards were replayed;
    # images are reproducible from the script/seed but deliberately not saved.
    rgb_hash = hashlib.sha256(rgb.numpy()).hexdigest()
    replay_hash = hashlib.sha256((args.out / 'replay.npz').read_bytes()).hexdigest()
    rows = []
    for seed in args.seeds:
        for condition in CONDITIONS:
            stem, model_cfg = load_config(args.models, condition, seed)
            data, meta = record_model(stem, model_cfg, replay, rgb, args.batch_size)
            np.savez_compressed(args.out / f'{condition}_s{seed}_activity.npz', **data)
            rows.append(meta)
            print(f'{condition} seed {seed}: PCA3 {meta["pc3_variance"]:.1%}; {time.monotonic() - start:.1f}s elapsed', flush=True)
    assert hashlib.sha256(rgb.numpy()).hexdigest() == rgb_hash
    sel = replay['sample_index']
    summary = dict(protocol=dict(episodes=args.episodes, episode_len=cfg['episode_len'],
        seeds=args.seeds, replay_seed=args.replay_seed, stride=args.stride,
        decisions=len(rgb), saved_states=len(sel), known_fraction=float(replay['goal_known'][sel].mean()),
        hits_per_episode=replay['hit'].reshape(-1, args.episodes).sum(0).tolist(),
        replay_sha256=replay_hash, rgb_sha256=rgb_hash,
        selection='Every fifth decision after step 100, speed > 0.05 m/s; all movement/facing directions; includes unknown and known goal states.',
        replay='Shared scripted waypoint replay, 16 separate standard-length episodes by default; original motor/image noise and per-episode layouts; physical goal fixed within each episode, distinct across episodes.',
        goal_memory='Zero before first hit; refreshed from actual pickup position, movement direction and velocity after every hit. h,c persist through respawns and start at zero for each episode.',
        pca='All 256 h coordinates, centered without neuron scaling; separate PCA per model. No UMAP.',
        limitation='Off-policy representation probe. Predicted policy actions do not control replay. Pooling direction/goal conditions need not yield a single 2D torus. Decodability does not establish causal use or explain reward advantage.',
        elapsed_seconds=time.monotonic() - start), models=rows)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Completed {len(rows)} frozen models in {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
