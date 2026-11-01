#!/usr/bin/env python3
"""Plot existing trained feedforward controls; no new model evaluation."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'lstm_history_context_1B'


def main():
    source = json.loads((ROOT / 'results_ext' / 'extension.json').read_text())
    arms = [('ffb', 'Feedforward\ncode only'),
            ('ff', 'Feedforward\n+ vision'), ('lstm', 'LSTM\n+ vision')]
    colors = ['#5c879e', '#3b718d', '#b66c37']
    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    for i, ((arm, _), color) in enumerate(zip(arms, colors)):
        entry = source['gaps'][arm]['per_label']['600M']
        stat = entry['stats']
        differences = np.array([entry['pairs'][str(s)]['gap'] for s in range(1, 9)])
        assert len(differences) == stat['n_pairs'] == 8
        assert np.isclose(differences.mean(), stat['mean_diff'])
        ax.scatter(i + np.linspace(-0.14, 0.14, 8), differences,
                   s=23, color=color, alpha=0.6, zorder=3)
        low, high = stat['boot_ci95']
        ax.errorbar(i, stat['mean_diff'], yerr=[[stat['mean_diff'] - low],
                    [high - stat['mean_diff']]], fmt='D', ms=7,
                    color=color, ecolor=color, linewidth=2.4, capsize=5, zorder=4)
        ax.text(i, high + 2.2, f"+{stat['mean_diff']:.1f}", ha='center',
                fontsize=12, weight='bold', color=color)
    ax.axhline(0, color='#8b9299', lw=1)
    ax.set_xticks(range(3), [label for _, label in arms], fontsize=10)
    ax.set_ylabel('PGC − Grid score', fontsize=11)
    ax.set_ylim(-4, 41)
    ax.set_title('The benefit also appears without an RNN', fontsize=13, pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#b4b9be')
    ax.tick_params(axis='both', length=0, pad=7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#e9ecef', lw=0.7)
    fig.text(0.5, 0.015, '600M training frames · 8 seed pairs · mean and bootstrap 95% interval',
             ha='center', fontsize=8, color='#626970')
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'trained_feedforward_context.png', dpi=180)
    fig.savefig(OUT / 'trained_feedforward_context.svg', metadata={'Date': None})
    svg = OUT / 'trained_feedforward_context.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(OUT / 'trained_feedforward_context.png')


if __name__ == '__main__':
    main()
