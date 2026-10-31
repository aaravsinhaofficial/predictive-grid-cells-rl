"""Simple cumulative variance chart, limited to the first 100 PCs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_variable_1B/pca_spectrum'


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    pcs = np.arange(101)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    endpoints = []
    for condition, label, color in (
            ('gc100', 'Grid', '#3478ac'),
            ('pgc33', 'PGC + grid', '#d66b37')):
        spectra = np.array([r['explained_variance_ratio'] for r in summary['models']
                            if r['protocol'] == 'variable' and r['condition'] == condition])
        assert spectra.shape == (8, 256)
        cumulative = np.r_[0., spectra.cumsum(1).mean(0)[:100] * 100]
        endpoints.append(float(cumulative[-1]))
        ax.plot(pcs, cumulative, color=color, linewidth=2.3, label=label)
    upper = np.ceil(max(endpoints) * 10) / 10
    ax.set(xlabel='Number of principal components', ylabel='Variance explained (%)',
           xlim=(0, 100), ylim=(0, upper), yticks=[0, 20, 40, 60, 80, upper],
           xticks=[0, 20, 40, 60, 80, 100])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e8ec', linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    ax.set_title('Variance explained by the first 100 PCs', fontsize=15, pad=15)
    fig.text(.5, .025, f'Means across 8 seeds · At 100 PCs: Grid {endpoints[0]:.1f}% | PGC + grid {endpoints[1]:.1f}%',
             ha='center', fontsize=9, color='#57636d')
    fig.tight_layout(rect=(0, .055, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'pca_variance_first100.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
