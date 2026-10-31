"""Frozen 1B LSTM hidden-state PCA on one shared, controlled arena sweep.

Run: python scripts/lstm_pca.py
Outputs include the replay, all 256-D hidden vectors, PCA bases, figures and
an interactive 3-D viewer. This is an off-policy representation probe, not a
performance evaluation or a test of toroidal topology / attractor dynamics.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Resolve this repository's `code` package before torch imports Python's
# similarly named standard-library module through its debugging utilities.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from code.common import CodeSource, build_env, build_policy, build_population

CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid only', 'pgc33': 'Grid + 33% PGC'}
ENV_KEYS = ('arena_half', 'action_repeat', 'v_max', 'rot_max', 'goal_radius',
            'vision', 'img_noise', 'randomize_textures')


def load_config(models, condition, seed):
    stem = models / condition / f'{condition}_lstm_s{seed}_1000M'
    return stem, json.loads(stem.with_suffix('.json').read_text())


@torch.inference_mode()
def make_replay(cfg, rows=32, replay_seed=424242):
    """Use real smoothed environment kinematics with a scripted controller.

    The room is fixed; facing stays east. Motor noise is disabled to make
    direction selection exact; original image noise is retained. The goal is
    above the sampled rectangle, so no hit, teleport, or goal-memory update
    occurs. The diagnostic session is extended beyond training episode length.
    """
    sweep_cfg = dict(cfg, episode_len=20000)
    env = build_env(sweep_cfg, 1, replay_seed, 'cpu', stagger=False)
    env.reset_all()
    env.noise_v = env.noise_r = 0.0
    env.pos[:] = torch.tensor([[-1.05, -1.05]])
    env.spawn[:] = env.pos
    env.hd.zero_()
    env.vf.zero_(); env.vs.zero_(); env.vr.zero_()
    env.mdir[:] = torch.tensor([[1.0, 0.0]])
    env.goal[:] = torch.tensor([[1.10, 1.10]])
    obs = env.observe()
    fields = {k: [] for k in ('pos', 'dir', 'vel', 'hd', 'prev_action',
                              'prev_reward', 'lane', 'leg_step', 'action')}
    images = []
    previous_action = -1
    previous_reward = 0.0

    def advance(action, lane, leg_step):
        nonlocal obs, previous_action, previous_reward
        for key in ('pos', 'dir', 'vel', 'hd'):
            fields[key].append(obs[key][0].numpy().copy())
        for key, value in (('prev_action', previous_action),
                           ('prev_reward', previous_reward), ('lane', lane),
                           ('leg_step', leg_step), ('action', action)):
            fields[key].append(value)
        images.append(obs['rgb'][0].numpy().copy())
        obs, reward, done, info = env.step(torch.tensor([action]))
        assert not done.item() and not info['hit'].item(), 'Replay hit/reset'
        previous_action, previous_reward = action, float(reward[0])

    for lane, target_y in enumerate(np.linspace(-1.05, 0.60, rows)):
        while float(env.pos[0, 1]) < target_y:
            advance(4, lane, -1)  # strafe north while facing east
        action, direction = (2, 1) if lane % 2 == 0 else (3, -1)
        leg_step = 0
        while direction * float(env.pos[0, 0]) < 1.01:
            advance(action, lane, leg_step)
            leg_step += 1
            assert leg_step < 300, 'Scripted leg did not finish'
    replay = {k: np.asarray(v) for k, v in fields.items()}
    speed = np.linalg.norm(replay['vel'], axis=1)
    motion_angle = np.arctan2(replay['dir'][:, 1], replay['dir'][:, 0])
    replay['selected'] = ((replay['action'] == 2) & (replay['prev_action'] == 2)
                          & (replay['leg_step'] >= 15)
                          & (np.arange(len(speed)) >= 100)
                          & (np.abs(motion_angle) <= np.deg2rad(5))
                          & (speed > 0.30) & (speed < 0.37))
    replay['goal_pos'] = env.goal[0].numpy().copy()
    replay['goal_dir'] = np.array([1.0, 0.0], dtype=np.float32)
    replay['goal_vel'] = replay['goal_dir'] * cfg['v_max']
    replay['landmark_pos'] = env.land_pos[0].numpy().copy()
    replay['stripe_phase'] = env.stripe_phase[0].numpy().copy()
    assert replay['selected'].sum() >= 500
    assert np.allclose(replay['hd'], 0)
    assert np.all(replay['prev_reward'] == 0)
    assert np.linalg.norm(replay['pos'] - replay['goal_pos'], axis=1).min() > cfg['goal_radius']
    return replay, torch.from_numpy(np.stack(images))


@torch.inference_mode()
def record_model(stem, cfg, replay, rgb, batch_size):
    ckpt = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    assert cfg['policy'] == 'lstm' and cfg['vision'] == 'raycast'
    net = build_policy(cfg, 'cpu').eval()
    net.load_state_dict(ckpt['policy'], strict=True)
    net.requires_grad_(False)
    population = build_population(cfg, 'cpu')
    source = CodeSource(cfg, population, 'cpu')
    pos, direction, velocity = (torch.as_tensor(replay[k], dtype=torch.float32)
                                for k in ('pos', 'dir', 'vel'))
    code = source(pos, direction, velocity)
    goal = source.goal(*(torch.as_tensor(replay[k][None], dtype=torch.float32)
                         for k in ('goal_pos', 'goal_dir', 'goal_vel')))
    embedding = torch.cat([net.embed(rgb[i:i + batch_size])
                           for i in range(0, len(rgb), batch_size)])
    previous_action = torch.as_tensor(replay['prev_action'], dtype=torch.long)
    previous_reward = torch.as_tensor(replay['prev_reward'], dtype=torch.float32)
    inputs = net._concat(embedding, code, goal.expand(len(rgb), -1),
                         previous_action, previous_reward, None)
    # nn.LSTM evaluates exactly the same weights as LSTMCell, efficiently over
    # a full sequence. Check against the original cell before using its output.
    sequence = torch.nn.LSTM(inputs.shape[1], net.nh).eval()
    sequence.load_state_dict({f'{k}_l0': v for k, v in net.lstm.state_dict().items()})
    sequence.requires_grad_(False)
    hidden, _ = sequence(inputs[:, None, :])
    hidden = hidden[:, 0]
    state = net.zero_state(1, 'cpu')
    errors = []
    for t in range(min(64, len(inputs))):
        state = net.lstm(inputs[t:t + 1], state)
        errors.append(float((state[0][0] - hidden[t]).abs().max()))
    assert max(errors) < 2e-5, f'LSTM sequence/cell mismatch: {max(errors)}'
    h = hidden.numpy().copy()
    assert np.isfinite(h).all() and h.shape == (len(rgb), 256)
    selected = h[replay['selected']]
    pca = PCA(svd_solver='full').fit(selected)  # center, no unit standardization
    scores = pca.transform(selected)[:, :3]
    ratios = pca.explained_variance_ratio_
    metadata = dict(seed=cfg['seed'], condition=cfg['cond'], frames=ckpt['frames'],
                    checkpoint=str(stem.with_suffix('.pt').relative_to(ROOT)),
                    checkpoint_sha256=hashlib.sha256(stem.with_suffix('.pt').read_bytes()).hexdigest(),
                    samples=len(selected), pc3_variance=float(ratios[:3].sum()),
                    pc10_variance=float(ratios[:10].sum()),
                    dimensions_80pct=int(np.searchsorted(np.cumsum(ratios), 0.8) + 1),
                    dimensions_90pct=int(np.searchsorted(np.cumsum(ratios), 0.9) + 1),
                    sequence_cell_max_error=max(errors))
    return dict(h=h, scores=scores, pca_mean=pca.mean_, components=pca.components_,
                explained_variance_ratio=ratios, goal_code=goal.numpy()[0]), metadata


def draw_cloud(ax, data, color, label, norm):
    xyz = data['scores']
    ax.scatter(*xyz.T, c=color, cmap='viridis', norm=norm, s=3, alpha=0.65,
               linewidths=0, depthshade=False, rasterized=True)
    variance = 100 * data['explained_variance_ratio'][:3].sum()
    ax.set_title(f'{label}\nPC1–3: {variance:.1f}% of variance', fontsize=10, pad=7)
    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), ('PC1', 'PC2', 'PC3')):
        axis.set_label_text(name, fontsize=8)
        axis.labelpad = 0
        axis.set_tick_params(labelsize=6, pad=0)
        axis.set_major_locator(plt.MaxNLocator(3))
    ax.set_box_aspect((1, 1, 1), zoom=0.83)
    ax.view_init(elev=23, azim=-58)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_figures(results, replay, out, seeds):
    selected_pos = replay['pos'][replay['selected']]
    norm = plt.Normalize(-1.17, 1.17)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    for seed in seeds:
        fig = plt.figure(figsize=(10, 9), layout='constrained')
        axes = []
        for row, coordinate in enumerate(('x', 'y')):
            row_axes = []
            for col, condition in enumerate(CONDITIONS):
                ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection='3d')
                draw_cloud(ax, results[condition, seed], selected_pos[:, row],
                           LABELS[condition], norm)
                row_axes.append(ax)
            fig.colorbar(sm, ax=row_axes, shrink=0.65, pad=0.06,
                         label=f'Arena {coordinate} position (m)')
            axes.extend(row_axes)
        fig.suptitle(f'Seed {seed} · LSTM hidden activity at 1B frames\n'
                     'Same eastward samples; top: x color, bottom: y color', fontsize=15)
        fig.supxlabel('PCA fitted separately to each model; a 3-D projection is not a topology test.', fontsize=9)
        save_figure(fig, out / f'seed_{seed:02d}_pca')

    nrows = (len(seeds) + 1) // 2
    fig = plt.figure(figsize=(16, 3.8 * nrows + 1.0), layout='constrained')
    axes = []
    for i, seed in enumerate(seeds):
        for j, condition in enumerate(CONDITIONS):
            ax = fig.add_subplot(nrows, 4, (i // 2) * 4 + (i % 2) * 2 + j + 1, projection='3d')
            draw_cloud(ax, results[condition, seed], selected_pos[:, 0],
                       f'Seed {seed} · {LABELS[condition]}', norm)
            axes.append(ax)
    fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.035, label='Arena x position (m)')
    fig.suptitle('Frozen LSTMs at 1B frames · all seed pairs\n'
                 'Matched trajectory, fixed goal memory, eastward motion', fontsize=17)
    fig.supxlabel('Independent, centered PCA per model. PC axes are not aligned across models. No torus inference from these projections alone.', fontsize=10)
    save_figure(fig, out / 'all_seeds_pca')

    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.plot(*replay['pos'].T, color='#d7dce1', lw=0.8, label='Full replay')
    ax.scatter(*selected_pos.T, c=selected_pos[:, 0], cmap='viridis', norm=norm, s=5,
               label='Samples used for PCA', zorder=3)
    ax.scatter(*replay['goal_pos'], marker='*', s=150, color='#dc5535', label='Fixed goal', zorder=5)
    ax.add_patch(plt.Circle(replay['goal_pos'], 0.3, fill=False, color='#dc5535', alpha=0.5))
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect='equal', xlabel='x (m)', ylabel='y (m)',
           title=f'Shared replay · {len(replay["pos"]):,} decisions\n'
                 f'{len(selected_pos):,} eastward samples used for every model')
    ax.legend(loc='upper left', fontsize=8)
    save_figure(fig, out / 'replay_coverage')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    colors = {'gc100': '#3478ac', 'pgc33': '#d66b37'}
    for condition in CONDITIONS:
        ratios = np.stack([results[condition, seed]['explained_variance_ratio'] for seed in seeds])
        cumulative = np.cumsum(ratios, axis=1)[:, :30] * 100
        for curve in cumulative:
            axes[0].plot(np.arange(1, 31), curve, color=colors[condition], alpha=0.17)
        axes[0].plot(np.arange(1, 31), cumulative.mean(0), color=colors[condition], label=LABELS[condition], lw=2)
    label_values = []
    for seed in seeds:
        values = [100 * results[c, seed]['explained_variance_ratio'][:3].sum() for c in CONDITIONS]
        axes[1].plot([0, 1], values, 'o-', color='#8b949e', alpha=0.8)
        label_values.append((values[1], seed))
    last_label = -np.inf
    for value, seed in sorted(label_values):
        label_y = max(value, last_label + 0.45)
        axes[1].annotate(str(seed), (1, value), xytext=(1.07, label_y),
                         textcoords='data', fontsize=8, va='center',
                         arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.6))
        last_label = label_y
    axes[0].axvline(3, color='#888888', ls=':', lw=1)
    axes[0].set(xlabel='Number of principal components', ylabel='Explained variance (%)', title='How much activity does PCA retain?', ylim=(0, 100))
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=[0, 1], xticklabels=[LABELS[c] for c in CONDITIONS], xlim=(-0.3, 1.3),
                ylabel='Variance in PC1–3 (%)', title='Each line connects one training seed')
    save_figure(fig, out / 'pca_variance')


def make_interactive(results, replay, out, seeds):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=[LABELS[c] for c in CONDITIONS])
    x = replay['pos'][replay['selected'], 0]
    y = replay['pos'][replay['selected'], 1]
    for i, seed in enumerate(seeds):
        for col, condition in enumerate(CONDITIONS, 1):
            xyz = results[condition, seed]['scores']
            fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                visible=(i == 0), name=f'Seed {seed} {LABELS[condition]}',
                customdata=np.column_stack([x, y]),
                hovertemplate='x=%{customdata[0]:.2f} m<br>y=%{customdata[1]:.2f} m<extra></extra>',
                marker=dict(size=2, color=x, colorscale='Viridis', cmin=-1.17, cmax=1.17,
                            opacity=0.7, showscale=(col == 2), colorbar=dict(title='x (m)', len=0.7))), row=1, col=col)
    buttons = []
    for i, seed in enumerate(seeds):
        variance = [100 * results[c, seed]['explained_variance_ratio'][:3].sum() for c in CONDITIONS]
        title = f'Seed {seed} | Grid: {variance[0]:.1f}% · PGC+grid: {variance[1]:.1f}% variance in PC1–3'
        buttons.append(dict(label=f'Seed {seed}', method='update',
                            args=[{'visible': [j // 2 == i for j in range(2 * len(seeds))]}, {'title.text': title}]))
    fig.update_layout(title=buttons[0]['args'][1]['title.text'], height=720, showlegend=False,
                      updatemenus=[dict(buttons=buttons, x=0.0, y=1.12)],
                      margin=dict(l=10, r=20, b=65, t=110))
    for scene in ('scene', 'scene2'):
        fig.update_layout(**{scene: dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3', aspectmode='cube')})
    fig.add_annotation(text='Drag each cloud to rotate · Same eastward replay samples · Independent PCA per model · This is not a topology test',
                       x=0.5, y=-0.06, xref='paper', yref='paper', showarrow=False)
    fig.write_html(out / 'interactive_pca.html', include_plotlyjs=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', type=Path, default=ROOT / 'models')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/lstm_pca_1B')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 9)))
    ap.add_argument('--rows', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--figures-only', action='store_true', help='Redraw saved activity without model inference')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        with np.load(args.out / 'replay.npz') as archive:
            replay = dict(archive)
        results = {}
        for seed in args.seeds:
            for condition in CONDITIONS:
                with np.load(args.out / f'{condition}_s{seed}_activity.npz') as archive:
                    results[condition, seed] = dict(archive)
        make_figures(results, replay, args.out, args.seeds)
        make_interactive(results, replay, args.out, args.seeds)
        print(f'Figures redrawn: {args.out}', flush=True)
        return
    _, cfg = load_config(args.models, 'gc100', args.seeds[0])
    start = time.time()
    replay, rgb = make_replay(cfg, args.rows)
    np.savez_compressed(args.out / 'replay.npz', **replay)
    rgb_sha = hashlib.sha256(rgb.numpy().tobytes()).hexdigest()
    print(f'Replay: {len(rgb)} steps; {replay["selected"].sum()} selected; {time.time()-start:.1f}s', flush=True)
    results, summaries = {}, []
    for seed in args.seeds:
        for condition in CONDITIONS:
            stem, model_cfg = load_config(args.models, condition, seed)
            assert all(model_cfg[k] == cfg[k] for k in ENV_KEYS), 'Environment configs differ'
            data, summary = record_model(stem, model_cfg, replay, rgb, args.batch_size)
            np.savez_compressed(args.out / f'{condition}_s{seed}_activity.npz', **data)
            results[condition, seed] = data
            summaries.append(summary)
            print(f'{condition} seed {seed}: PC1–3 {summary["pc3_variance"]:.1%}, '
                  f'D90 {summary["dimensions_90pct"]}; elapsed {time.time()-start:.1f}s', flush=True)
    metadata = dict(protocol='scripted raster replay with real smoothed kinematics',
        replay_seed=424242, steps=len(rgb), samples=int(replay['selected'].sum()),
        rows=args.rows, rgb_sha256=rgb_sha, shared_inputs=['RGB', 'position', 'direction', 'velocity', 'previous action', 'previous reward'],
        goal_pos=replay['goal_pos'].tolist(), goal_snapshot_arrival_direction=[1.0, 0.0],
        goal_snapshot_arrival_speed=cfg['v_max'], goal_memory='preloaded, model-specific, fixed throughout',
        motor_noise=0, image_noise=cfg['img_noise'], heading_degrees=0,
        selection='eastward within 5 degrees, speed 0.30–0.37 m/s, >=15 steps into leg, >=100 steps overall',
        episode_resets=0, state_initialization='h=c=0 once at start',
        extended_diagnostic_episode=True, pca='centered h, no scaling, separate full SVD per model',
        limitations=['Off-policy controlled replay, not autonomous behavior.',
                     'One room / goal / heading / trajectory; selected samples omit the goal region.',
                     'h only; c is not included in PCA.',
                     'A 3-D PCA projection cannot establish or rule out a torus or recurrent attractor.'],
        models=summaries)
    (args.out / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    make_figures(results, replay, args.out, args.seeds)
    make_interactive(results, replay, args.out, args.seeds)
    print(f'Finished: {args.out}; elapsed {time.time()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
