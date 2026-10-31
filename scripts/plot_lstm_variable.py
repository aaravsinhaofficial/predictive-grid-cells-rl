"""Plot the matched, variable-direction and variable-goal LSTM replay.

Run after scripts/lstm_variable_replay.py. Uses saved activity only; no model
inference or fitting is performed here. The same PCA coordinates are colored
by movement direction and by remembered pickup position in separate figures.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid only', 'pgc33': 'Grid + 33% PGC'}


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def cloud(ax, data, color, known, title, cmap, norm):
    xyz = data['scores']
    assert xyz.shape == (len(color), 3)
    if (~known).any():
        ax.scatter(*xyz[~known].T, c='#b7bec5', s=2.4, alpha=.28,
                   linewidths=0, depthshade=False, rasterized=True)
    ax.scatter(*xyz[known].T, c=color[known], cmap=cmap, norm=norm,
               s=2.4, alpha=.57, linewidths=0, depthshade=False,
               rasterized=True)
    variance = 100 * data['explained_variance_ratio'][:3].sum()
    ax.set_title(f'{title}\nPC1–3: {variance:.1f}% of variance', fontsize=10, pad=8)
    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), ('PC1', 'PC2', 'PC3')):
        axis.set_label_text(name, fontsize=8)
        axis.labelpad = 0
        axis.set_tick_params(labelsize=6, pad=0)
        axis.set_major_locator(plt.MaxNLocator(3))
    ax.set_box_aspect((1, 1, 1), zoom=.82)
    ax.view_init(elev=23, azim=-58)


def all_seeds(data, replay, out, seeds, coloring):
    idx = replay['sample_index']
    if coloring == 'direction':
        d = replay['dir'][idx]
        color = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        known = np.ones(len(idx), dtype=bool)
        cmap, norm = 'twilight', plt.Normalize(-180, 180)
        color_label = 'Movement direction (degrees; 0° = east, 90° = north)'
        subtitle = 'Variable movement and facing; goals vary across episodes; snapshots refresh at hits'
    else:
        color = replay['memory_pos'][idx, 0]
        known = replay['goal_known'][idx].astype(bool)
        cmap, norm = 'viridis', plt.Normalize(-1.17, 1.17)
        color_label = 'Remembered pickup x coordinate (m)'
        unknown_count = int((~known).sum())
        sample_word = 'sample' if unknown_count == 1 else 'samples'
        subtitle = ('Same PCA coordinates, colored by remembered pickup location; '
                    f'{unknown_count:,} pre-discovery {sample_word} in gray')
    rows = (len(seeds) + 1) // 2
    fig = plt.figure(figsize=(16, rows * 3.8 + 1), layout='constrained')
    axes = []
    for i, seed in enumerate(seeds):
        for j, condition in enumerate(CONDITIONS):
            ax = fig.add_subplot(rows, 4, (i // 2) * 4 + (i % 2) * 2 + j + 1,
                                 projection='3d')
            cloud(ax, data[condition, seed], color, known,
                  f'Seed {seed} · {LABELS[condition]}', cmap, norm)
            axes.append(ax)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      shrink=.5, pad=.035, label=color_label)
    if coloring == 'direction':
        bar.set_ticks([-180, -90, 0, 90, 180])
    fig.suptitle(f'Frozen 1B LSTMs · matched replay across all eight seed pairs\n{subtitle}',
                 fontsize=15)
    fig.supxlabel('PCA fitted separately to each model; axes are not aligned. '
                  'Pooling goals and directions does not establish or exclude a torus.', fontsize=10)
    save_figure(fig, out / f'all_seeds_{coloring}_pca')


def coverage(replay, out):
    idx = replay['sample_index']
    episodes = np.unique(replay['episode'])
    direction = replay['dir'][idx]
    angles = np.degrees(np.arctan2(direction[:, 1], direction[:, 0]))
    norm = plt.Normalize(-180, 180)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    ax = axes[0, 0]
    for episode in episodes:
        ids = np.flatnonzero(replay['episode'] == episode)
        ids = ids[np.argsort(replay['step'][ids])]
        pos = replay['pos'][ids].copy()
        # Goal hits respawn the agent. Do not draw those teleports as motion.
        if len(ids) > 1:
            pos[1:][replay['hit'][ids[:-1]].astype(bool)] = np.nan
        ax.plot(*pos.T, color='#ccd0d5', lw=.5, alpha=.45, zorder=1)
    scatter = ax.scatter(*replay['pos'][idx].T, c=angles, cmap='twilight', norm=norm,
                         s=3, alpha=.6, linewidths=0, rasterized=True, zorder=2)
    goal_positions = np.stack([replay['goal_pos'][np.flatnonzero(replay['episode'] == e)[0]]
                               for e in episodes])
    ax.scatter(*goal_positions.T, marker='*', c='black', s=70, edgecolors='white',
               linewidths=.5, label='Episode goal centers', zorder=3)
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect='equal', xlabel='x (m)',
           ylabel='y (m)', title=f'A  |  Arena coverage · {len(idx):,} sampled states')
    ax.legend(fontsize=8, loc='upper left')
    cb = fig.colorbar(scatter, ax=ax, shrink=.8, pad=.025, label='Movement direction (°)')
    cb.set_ticks([-180, -90, 0, 90, 180])

    ax = axes[0, 1]
    edges = np.linspace(-180, 180, 25)
    ax.hist(angles, bins=edges, color='#407ca8', alpha=.75, label='Movement')
    ax.hist(np.degrees(replay['hd'][idx]), bins=edges, color='#cf6c39',
            histtype='step', lw=1.7, label='Facing')
    ax.set(xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180], xlabel='Angle (degrees)',
           ylabel='Sample count', title='B  |  Movement and facing both vary')
    ax.legend(fontsize=9)

    # Show one point per actual snapshot update, not one per repeated sample.
    updates = []
    for episode in episodes:
        ids = np.flatnonzero(replay['episode'] == episode)
        ids = ids[np.argsort(replay['step'][ids])]
        if len(ids) > 1:
            updates.extend(ids[1:][replay['hit'][ids[:-1]].astype(bool)].tolist())
    updates = np.asarray(updates, dtype=int)
    ax = axes[1, 0]
    ax.scatter(*goal_positions.T, marker='*', c='black', s=70, label='Goal center')
    if len(updates):
        p, d = replay['memory_pos'][updates], replay['memory_dir'][updates]
        angle = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        ax.quiver(*p.T, *d.T, angle, cmap='twilight', norm=norm, angles='xy',
                  scale_units='xy', scale=6, width=.005, zorder=3)
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect='equal', xlabel='x (m)',
           ylabel='y (m)', title=f'C  |  {len(updates)} remembered pickup updates')
    ax.legend(fontsize=8, loc='upper left')
    ax.text(.02, .02, 'Arrows: arrival direction at each remembered pickup\n'
            'Pickup positions lie within the goal radius.', transform=ax.transAxes,
            fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))

    ax = axes[1, 1]
    known = replay['goal_known'][idx].astype(bool)
    sampled_episode = replay['episode'][idx]
    pre = [np.sum((sampled_episode == e) & ~known) for e in episodes]
    post = [np.sum((sampled_episode == e) & known) for e in episodes]
    positions = np.arange(len(episodes))
    ax.bar(positions, pre, color='#b7bec5', label='Before discovery')
    ax.bar(positions, post, bottom=pre, color='#407ca8', label='Goal remembered')
    ax.set(xticks=positions, xticklabels=[str(int(e) + 1) for e in episodes],
           xlabel='Episode', ylabel='Sample count', title='D  |  Goal-memory availability')
    ax.legend(fontsize=8)
    fig.suptitle(f'Shared replay · {len(episodes)} episodes · '
                 'variable goals and movement directions', fontsize=16)
    fig.supxlabel('Scripted off-policy replay, identical across models. '
                  'Physical goals stay fixed within episodes; remembered snapshots update at hits.', fontsize=10)
    save_figure(fig, out / 'replay_coverage')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/lstm_variable_1B')
    args = ap.parse_args(argv)
    replay = dict(np.load(args.out / 'replay.npz'))
    idx = replay['sample_index']
    assert idx.ndim == 1 and len(idx) > 0 and np.all(np.diff(idx) > 0)
    data = {}
    missing = []
    for condition in CONDITIONS:
        for seed in range(1, 9):
            path = args.out / f'{condition}_s{seed}_activity.npz'
            if path.exists():
                with np.load(path) as f:
                    data[condition, seed] = {key: f[key] for key in
                                             ('scores', 'explained_variance_ratio')}
            else:
                missing.append(path.name)
    if missing:
        raise FileNotFoundError('All eight matched pairs are required. Missing: ' + ', '.join(missing))
    seeds = list(range(1, 9))
    coverage(replay, args.out)
    for coloring in ('direction', 'goal'):
        all_seeds(data, replay, args.out, seeds, coloring)
    print(f'Wrote three PNG/PDF figure pairs for {len(seeds)} seed pairs to {args.out}')


if __name__ == '__main__':
    main()
