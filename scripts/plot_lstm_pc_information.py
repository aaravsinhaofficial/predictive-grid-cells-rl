"""Simple held-out readout curves for leading LSTM principal components.

Run after the companion analysis has written results/lstm_pc_information_1B/summary.json.
This script only reads saved metrics and renders figures.
"""
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
PCS = np.array([3, 10, 20, 50, 100, 256])


def make_figure(summary, out):
    by = {(r['condition'], r['seed']): r for r in summary['models']}
    assert all((c, s) in by for c in CONDITIONS for s in range(1, 9))
    panels = [
        ('memory_position', 'rmse_m', 'baseline_rmse_m', 100.,
         'Remembered goal pickup position', 'Position error (cm)'),
        ('direction', 'mae_deg', 'baseline_mae_deg', 1.,
         'Movement direction', 'Angular error (degrees)'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    handles, legend_labels = [], []
    for panel_index, (ax, specification) in enumerate(zip(axes, panels)):
        target, metric, baseline_key, factor, title, ylabel = specification
        baseline_values = []
        top = 0.
        for condition in CONDITIONS:
            rows = []
            for seed in range(1, 9):
                metrics = {int(r['pcs']): r for r in by[condition, seed]['metrics']}
                assert set(PCS) <= set(metrics)
                rows.append([metrics[int(k)][target][metric] * factor for k in PCS])
                baseline_values.extend(metrics[int(k)][target][baseline_key] * factor for k in PCS)
            values = np.asarray(rows, dtype=float)
            assert values.shape == (8, 6) and np.isfinite(values).all()
            mean = values.mean(0)
            sem = values.std(0, ddof=1) / np.sqrt(len(values))
            line, = ax.plot(PCS, mean, color=COLORS[condition], marker='o',
                            markersize=4.5, linewidth=2, label=LABELS[condition])
            ax.fill_between(PCS, mean - sem, mean + sem,
                            color=COLORS[condition], alpha=.16, linewidth=0)
            if panel_index == 0:
                handles.append(line)
                legend_labels.append(LABELS[condition])
            top = max(top, float((mean + sem).max()))
        baseline_values = np.asarray(baseline_values)
        assert np.isfinite(baseline_values).all()
        assert np.allclose(baseline_values, baseline_values[0], rtol=1e-8, atol=1e-8), \
            'Training-mean baseline must be common to all models and PC counts.'
        baseline = float(baseline_values[0])
        line = ax.axhline(baseline, color='#858e96', linestyle='--', linewidth=1.2)
        if panel_index == 0:
            handles.append(line)
            legend_labels.append('Training-mean baseline')
        top = max(top, baseline)
        ax.set_xscale('log')
        ax.set(xlim=(2.7, 285), ylim=(0, top * 1.1), xticks=PCS,
               xticklabels=['3', '10', '20', '50', '100', 'All\n256'],
               xlabel='Principal components retained (log scale)', ylabel=ylabel,
               title=title)
        ax.minorticks_off()
        ax.grid(axis='y', color='#e7ebef', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    fig.suptitle('Can the leading PCs recover goal and movement information?',
                 fontsize=15, y=.98)
    fig.text(.5, .905, 'Readouts tested on held-out episodes · mean ± SEM across 8 seeds',
             ha='center', fontsize=10, color='#53636f')
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .885),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Lower is better. PCA and readouts use training episodes only. '
             'Decodability does not establish causal policy use.',
             ha='center', fontsize=9, color='#53636f')
    fig.subplots_adjust(left=.075, right=.985, bottom=.18, top=.75, wspace=.27)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'pc_information.{suffix}', dpi=210, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_pc_information_1B')
    args = parser.parse_args()
    summary = json.loads((args.out / 'summary.json').read_text())
    make_figure(summary, args.out)
    print('Wrote pc_information.png and pc_information.pdf', flush=True)


if __name__ == '__main__':
    main()
