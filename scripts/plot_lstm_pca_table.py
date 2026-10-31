"""Render a compact table of PCA variance and component counts, all 8 seeds."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_variable_1B/pca_spectrum'


def one_decimal(x):
    return str(Decimal(str(float(x))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def main():
    source = json.loads((OUT / 'summary.json').read_text())
    spectra = {}
    for condition in ('gc100', 'pgc33'):
        rows = [r for r in source['models'] if r['protocol'] == 'variable'
                and r['condition'] == condition]
        assert sorted(r['seed'] for r in rows) == list(range(1, 9))
        spectra[condition] = np.array([r['explained_variance_ratio'] for r in rows])
    cells, numeric = [], []
    for k in (3, 10):
        values = [float(v[:, :k].sum(1).mean() * 100) for v in spectra.values()]
        label = f'Variance captured by first {k} PCs'
        cells.append([label, *[one_decimal(x) + '%' for x in values]])
        numeric.append(dict(measure=label, units='percent variance', grid=values[0], pgc=values[1]))
    for target in (50, 60, 70, 80, 90, 95, 99, 100):
        values = []
        for v in spectra.values():
            # 100% explicitly denotes the full basis, not a rank estimate:
            # roundoff can make cumulative fractions reach 1 prematurely.
            counts = np.full(8, v.shape[1]) if target == 100 else np.array([
                np.searchsorted(row, target / 100) + 1 for row in v.cumsum(1)])
            values.append(float(counts.mean()))
        label = f'PCs needed for {target}% variance' if target < 100 else 'Full PCA basis (100% variance)'
        cells.append([label, *[one_decimal(x) for x in values]])
        numeric.append(dict(measure=label, units='mean component count', grid=values[0], pgc=values[1]))

    fig = plt.figure(figsize=(10, 7.8), facecolor='white')
    ax = fig.add_axes([.05, .19, .9, .66])
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=['Measure', 'Grid', 'PGC + grid'],
                     colWidths=[.64, .18, .18], cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dce2e7')
        cell.set_linewidth(.65)
        cell.PAD = .035
        if r == 0:
            cell.set_facecolor('#243746')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f5f8' if r % 2 else 'white')
            cell.set_text_props(color='#1e2c36')
        if c == 0:
            cell.get_text().set_ha('left')
    fig.text(.05, .945, 'Variance explained by PCA', fontsize=20, weight='bold', color='#172b3a')
    fig.text(.05, .903, 'Frozen 1B LSTMs · means across 8 seeds · varying movement and goals',
             fontsize=11, color='#53636f')
    fig.text(.05, .135, 'PC = principal component. Percentages show variance captured; other values are component counts.',
             fontsize=9.3, color='#53636f')
    fig.text(.05, .106, 'Counts are averaged across seeds, so they can be fractional. 100% uses the full 256-PC basis.',
             fontsize=9.3, color='#53636f')
    fig.text(.05, .077, 'Explained variance measures variation in hidden activity, not task accuracy.',
             fontsize=9.3, color='#53636f')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'pca_variance_table.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)
    (OUT / 'pca_variance_table.json').write_text(json.dumps(dict(
        source='summary.json; variable replay only', rows=numeric,
        rounding='One decimal, half up. 100% denotes full basis, not estimated rank.'), indent=2) + '\n')
    print('Wrote pca_variance_table.png and pca_variance_table.pdf')


if __name__ == '__main__':
    main()
