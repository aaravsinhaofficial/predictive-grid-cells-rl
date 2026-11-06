"""Original varied-replay PCA: retain episode averages, show the first 100 PCs.

Source PCA uses overall mean-centering, with no episode-average subtraction
and no neuron-wise scaling. Curves are equal-weight means of eight model fits.
"""
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
    assert summary['protocol']['variable_samples'] == 3393
    pcs = np.arange(101)
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    endpoints = []
    for condition, label, color in (
            ('gc100', 'Grid', '#3478ac'),
            ('pgc33', 'PGC + grid', '#d66b37')):
        spectra = np.array([r['explained_variance_ratio'] for r in summary['models']
                            if r['protocol'] == 'variable' and r['condition'] == condition])
        assert spectra.shape == (8, 256)
        assert np.allclose(spectra.sum(1), 1.)
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
    fig.text(.5, .025, 'Varying movement, goals and rooms · Means across 8 seed pairs\n'
             'Original activity: episode averages retained; no neuron-wise scaling',
             ha='center', fontsize=9, color='#57636d')
    fig.tight_layout(rect=(0, .085, 1, 1))
    for suffix in ('png', 'pdf', 'svg'):
        metadata = {'Date': None} if suffix == 'svg' else None
        fig.savefig(OUT / f'pca_variance_first100.{suffix}', dpi=220, facecolor='white', metadata=metadata)
    svg = OUT / 'pca_variance_first100.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
