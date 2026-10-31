"""All 16 frozen LSTMs: spatially colored PCA across fixed component blocks.

Run after scripts/lstm_pca_spectrum.py. Each plot contains the same matched
variable-direction/variable-goal samples; no components are picked for shape.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid only', 'pgc33': 'PGC + grid'}


def draw_panel(ax, data, block, color, norm, title, cmap):
    xyz = data['scores_first12'][:, block:block + 3]
    variance = 100 * data['explained_variance_ratio'][block:block + 3].sum()
    ax.scatter(*xyz.T, c=color, cmap=cmap, norm=norm, s=2.6, alpha=.62,
               linewidths=0, depthshade=False, rasterized=True)
    ax.set_title(f'{title}\nPC{block + 1}–{block + 3}: {variance:.1f}% of variance',
                 fontsize=10, pad=8)
    # Equal numerical spans preserve relative scale across the three axes.
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2
    radius = .54 * (hi - lo).max()
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    for axis, pc in zip((ax.xaxis, ax.yaxis, ax.zaxis), range(block + 1, block + 4)):
        axis.set_label_text(f'PC{pc}', fontsize=8)
        axis.labelpad = 0
        axis.set_tick_params(labelsize=6, pad=0)
        axis.set_major_locator(plt.MaxNLocator(3))
    ax.set_box_aspect((1, 1, 1), zoom=.83)
    ax.view_init(elev=23, azim=-58)


def make_clouds(data, replay, out, block=0, coloring='x'):
    sample = replay['sample_index']
    if coloring in ('x', 'y'):
        axis = int(coloring == 'y')
        color = replay['pos'][sample, axis]
        norm, cmap = plt.Normalize(-1.17, 1.17), 'viridis'
        color_label = f'Current arena {coloring} position (m)'
    elif coloring == 'direction':
        direction = replay['dir'][sample]
        color = np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0]))
        norm, cmap = plt.Normalize(-180, 180), 'twilight'
        color_label = 'Movement direction (degrees)'
    else:
        raise ValueError(coloring)
    fig = plt.figure(figsize=(16, 16.2), layout='constrained')
    axes = []
    for i, seed in enumerate(range(1, 9)):
        for col, condition in enumerate(CONDITIONS):
            panel = (i // 2) * 4 + (i % 2) * 2 + col + 1
            ax = fig.add_subplot(4, 4, panel, projection='3d')
            draw_panel(ax, data[condition, seed], block, color, norm,
                       f'Seed {seed} · {LABELS[condition]}', cmap)
            axes.append(ax)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                      shrink=.5, pad=.035, label=color_label)
    if coloring == 'direction':
        bar.set_ticks([-180, -90, 0, 90, 180])
    fig.suptitle(f'All eight seed pairs · frozen LSTMs at 1B frames · PCs {block + 1}–{block + 3}\n'
                 '3,393 matched states: varying movement, facing, and remembered goals', fontsize=15)
    fig.supxlabel('Independent centered PCA per model; no whitening or UMAP. '
                  'Equal scale within each 3-D panel; axes are not aligned across models. '
                  'Shape alone is not a gridness or torus test.', fontsize=9)
    name = 'all_seeds_pca' if block == 0 and coloring == 'x' else f'all_seeds_pc{block + 1}_{block + 3}_{coloring}'
    fig.savefig(out / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {name}.png/.pdf', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/lstm_variable_1B')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_variable_1B/pca_spectrum')
    args = parser.parse_args()
    with np.load(args.source / 'replay.npz') as f:
        replay = dict(f)
    data = {}
    for condition in CONDITIONS:
        for seed in range(1, 9):
            with np.load(args.out / f'{condition}_s{seed}_pca.npz') as f:
                data[condition, seed] = dict(f)
            assert data[condition, seed]['scores_first12'].shape == (len(replay['sample_index']), 12)
    for block in (0, 3, 6):
        make_clouds(data, replay, args.out, block=block)
    make_clouds(data, replay, args.out, coloring='y')
    make_clouds(data, replay, args.out, coloring='direction')


if __name__ == '__main__':
    main()
