"""Render a simple figure from the saved LSTM variance decomposition only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}
COLORS = {'gc100': '#3478ac', 'pgc33': '#d66b37'}


def make_figure(summary, out):
    by = {(r['condition'], r['seed']): r for r in summary['models']}
    assert all((c, s) in by for c in CONDITIONS for s in range(1, 9))
    def values(condition, key):
        x = 100 * np.array([by[condition, seed][key] for seed in range(1, 9)])
        assert np.isfinite(x).all()
        return x

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12,
                         'axes.labelsize': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1),
                             gridspec_kw={'width_ratios': [1.3, 1.]})
    offsets = np.linspace(-.045, .045, 8)
    handles = []
    ax = axes[0]
    for j, condition in enumerate(CONDITIONS):
        xs = np.array([0., 1.]) + (j - .5) * .24
        all_values = np.array([values(condition, key) for key in
                              ('top10_share', 'within_top10_share')])
        for seed in range(8):
            ax.plot(xs + offsets[seed], all_values[:, seed],
                    color=COLORS[condition], alpha=.17, linewidth=.8, zorder=1)
        for x, data in zip(xs, all_values):
            mean, sem = data.mean(), data.std(ddof=1) / np.sqrt(8)
            ax.scatter(x + offsets, data, color=COLORS[condition], s=21,
                       alpha=.32, edgecolor='none', zorder=2)
            line = ax.errorbar(x, mean, yerr=sem, marker='o', markersize=7,
                              color=COLORS[condition], linewidth=2, capsize=4,
                              zorder=3, label=LABELS[condition])
            ax.text(x, max(data.max(), mean + sem) + 2., f'{mean:.1f}%',
                    ha='center', va='bottom', color=COLORS[condition],
                    fontsize=11, fontweight='bold')
        handles.append(line)
    ax.set(title='A  |  Variance captured by the first 10 PCs',
           ylabel='Variance captured (%)', ylim=(0, 62), xlim=(-.4, 1.4),
           xticks=[0, 1], xticklabels=['Original activity', 'Episode means removed\nPCA refitted'])

    ax = axes[1]
    data = np.stack([values(c, 'between_episode_share') for c in CONDITIONS])
    for seed in range(8):
        ax.plot(np.array([0., 1.]) + offsets[seed], data[:, seed],
                color='#9aa4ad', alpha=.35, linewidth=1, zorder=1)
    for j, condition in enumerate(CONDITIONS):
        mean, sem = data[j].mean(), data[j].std(ddof=1) / np.sqrt(8)
        ax.scatter(j + offsets, data[j], color=COLORS[condition],
                   alpha=.4, s=27, edgecolor='none', zorder=2)
        ax.errorbar(j, mean, yerr=sem, marker='o', markersize=8,
                    color=COLORS[condition], linewidth=2, capsize=4, zorder=3)
        ax.text(j, data[j].max() + 2., f'{mean:.1f}%',
                ha='center', va='bottom', color=COLORS[condition],
                fontsize=11, fontweight='bold')
    ax.set(title='B  |  Variation between episode means',
           ylabel='Share of total activity variance (%)',
           ylim=(0, 42), xlim=(-.4, 1.4), xticks=[0, 1],
           xticklabels=[LABELS[c] for c in CONDITIONS])

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e7ebef', linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0, pad=8)
        ax.tick_params(labelsize=9)
    fig.suptitle('What contributes to the PCA variance difference?', fontsize=15, y=.975)
    fig.text(.5, .914, 'Frozen 1B LSTMs · mean ± SEM across 8 seed pairs · faint points show individual seeds',
             ha='center', fontsize=9.5, color='#53636f')
    fig.legend(handles, [LABELS[c] for c in CONDITIONS], frameon=False,
               loc='upper center', bbox_to_anchor=(.5, .885), ncol=2, fontsize=10)
    fig.text(.5, .064, 'The 16 episodes differ in goals, room layouts, and sampled histories; episode effects do not isolate goal information.',
             ha='center', fontsize=8.5, color='#53636f')
    fig.text(.5, .031, 'PCA measures activity covariance, not temporal dynamics. Removing episode means changes the variance denominator.',
             ha='center', fontsize=8.5, color='#53636f')
    fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.755, wspace=.29)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'variance_sources.{extension}', dpi=210, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_variance_sources_1B')
    args = parser.parse_args()
    make_figure(json.loads((args.out / 'summary.json').read_text()), args.out)
    print('Wrote variance_sources.png and variance_sources.pdf', flush=True)


if __name__ == '__main__':
    main()
