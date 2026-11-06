"""Small figures for the learned direction-drive intervention."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/ring_heading/direction_subspace/predictive_s1006'))
    args = p.parse_args()
    s = json.loads((args.out / 'summary.json').read_text())
    n = json.loads((args.out / 'navigation/summary.json').read_text())
    variants = ['native', 'ordinary', 'remove_direction', 'restore_direction']
    labels = ['Native', 'Ordinary', 'Remove\ndirection', 'Restore\ndirection', 'Random\nrestore']
    colors = ['#D87532', '#8E98A3', '#8C64B4', '#317BB5', '#D1D5DA']
    forced = s['forced']['scenarios']['new_places']
    nav = n['scenarios']['new_places']['durations']['40s']
    rows = s['anchoring']['rows']
    coverage = lambda v: 100 * np.mean([r['circle_coverage'] for r in rows if r['variant'] == v and r['step'] == 100 and r['context'] < 5])
    metrics = [lambda v: forced[v]['all_mae_deg'], lambda v: 100 * nav[v]['success_fraction'], coverage]
    titles = ['Heading at new places', 'Navigation at new places', 'Ring coverage']
    ylabels = ['Error (degrees)', 'Reaching the goal (%)', 'Circle covered (%)']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        sham = [metric(f'sham{j}_restore') for j in range(3)]
        values = [metric(v) for v in variants] + [np.mean(sham)]
        ax.bar(range(5), values, color=colors, width=.72)
        ax.scatter([4 - .13, 4, 4 + .13], sham, color='#5B6570', s=13, zorder=3)
        for i, value in enumerate(values[:4]):
            ax.text(i, value + 2, f'{value:.1f}', ha='center', fontsize=9)
        ax.set(title=title, ylabel=ylabel, xticks=range(5), xticklabels=labels)
        ax.set_ylim(0, 110 if '%' in ylabel else 95)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.supxlabel('One frozen trained model · Same rank-2 intervention in all tests · Dots: three matched-energy random controls', fontsize=9)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.out / f'direction_plane_mechanism.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
