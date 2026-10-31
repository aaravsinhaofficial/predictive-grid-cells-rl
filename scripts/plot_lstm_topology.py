"""Plot saved persistent-homology results for the frozen 1B LSTMs.

Run after scripts/lstm_topology.py:
    python scripts/plot_lstm_topology.py

The figures summarize a torus-compatible topology screen, not proof of an
attractor. Persistence is measured across distance scales, not across time.
"""
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('gc100', 'pgc33')
LABELS = {'gc100': 'Grid only', 'pgc33': 'Grid + 33% PGC'}
CONTROL_NAMES = ('control_grid_module', 'control_pgc_module',
                 'control_clifford_torus', 'control_position_sheet')
CONTROL_LABELS = {
    'control_grid_module': 'Analytic grid module · positive control',
    'control_pgc_module': 'Analytic PGC + grid module · positive control',
    'control_clifford_torus': 'Known synthetic torus · positive control',
    'control_position_sheet': 'Physical position sheet · negative control',
}
COLORS = {1: ('#bd5828', '#ecd5c9'), 2: ('#286b95', '#cbdfe9')}


def finite_bars(doc, dimension):
    """Finite bars, ranked by descending lifetime; shape is always (n, 2)."""
    values = np.asarray(doc['diagrams'][dimension], dtype=float).reshape(-1, 2)
    values = values[np.isfinite(values).all(axis=1)]
    order = np.argsort(-(values[:, 1] - values[:, 0]), kind='stable')
    return values[order]


def maximum_death(docs):
    deaths = [bar[1] for doc in docs for dim, count in ((1, 6), (2, 4))
              for bar in finite_bars(doc, dim)[:count]]
    return max(deaths, default=1.0) * 1.04


def draw_barcodes(ax, doc, title, x_max):
    for dim, count, y_start in ((1, 6, 10.5), (2, 4, 3.5)):
        bars = finite_bars(doc, dim)[:count]
        threshold = doc['thresholds'][f'H{dim}']
        for rank, (birth, death) in enumerate(bars):
            color = COLORS[dim][0 if death - birth > threshold else 1]
            ax.plot([birth, death], [y_start - rank] * 2,
                    color=color, linewidth=3.3, solid_capstyle='butt')
        if len(bars) == 0:
            ax.text(0.025, (y_start - 1) / 12.0, 'No finite bars',
                    transform=ax.transAxes, fontsize=8, color='#70777c')
    ax.axhline(4.5, color='#e6e8eb', linewidth=0.8)
    ax.set(yticks=[8.0, 2.0], yticklabels=['H1\nloops', 'H2\nsurfaces'],
           ylim=(-0.4, 11.5), xlim=(0, x_max))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis='both', labelsize=8)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.grid(axis='x', color='#f0f1f3', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#aab0b6')
    m = doc['metrics']
    core = 'yes' if m.get('core_torus_signature', m['torus_screen']) else 'no'
    strict = 'yes' if m['torus_screen'] else 'no'
    ax.set_title(f'{title}\nAbove null: {m["significant_h1"]} H1, '
                 f'{m["significant_h2"]} H2 · core: {core}; strict: {strict}',
                 fontsize=10, loc='left', pad=8)


def save_figure(fig, path):
    for extension in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(extension), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {path.name}.png and .pdf', flush=True)


def protocol_caption(docs):
    landmarks = sorted({doc['landmarks'] for doc in docs})
    number = str(landmarks[0]) if len(landmarks) == 1 else '/'.join(map(str, landmarks))
    return f'{number} landmarks · original activity distances · no UMAP'


def make_all_seed_barcodes(models, out):
    seeds = sorted({doc['seed'] for doc in models})
    lookup = {(doc['condition'], doc['seed']): doc for doc in models}
    fig, axes = plt.subplots(len(seeds), 2, figsize=(13, 2.65 * len(seeds) + 1.8), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.94,
                        hspace=0.52, wspace=0.20)
    x_max = maximum_death(models)
    for row, seed in enumerate(seeds):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            doc = lookup[condition, seed]
            draw_barcodes(ax, doc, f'Seed {seed} · {LABELS[condition]}', x_max)
            if row == len(seeds) - 1:
                ax.set_xlabel('Connection distance / median pairwise distance', fontsize=9)
    fig.suptitle('Persistence barcodes · all eight seed pairs\n' + protocol_caption(models),
                 fontsize=17, y=0.985)
    fig.text(0.075, 0.035,
             'Each bar spans its birth and death distances; longer bars survive more scales, not more time.\n'
             'Dark bars exceed the 95th percentile of shuffled maximum lifetimes in that dimension; pale bars do not.\n'
             'Top six H1 and four H2 bars shown. Core: strong torus features coexist; strict: no extra above-null bars. Neither proves an attractor.',
             fontsize=9, va='center', linespacing=1.55)
    save_figure(fig, out / 'all_seeds_barcodes')


def make_signature(models, out):
    seeds = sorted({doc['seed'] for doc in models})
    lookup = {(doc['condition'], doc['seed']): doc for doc in models}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.78, wspace=0.25)
    palette = plt.get_cmap('tab10')
    metrics = (('h1_second', 'H1', 'Second-longest H1 loop'),
               ('h2_first', 'H2', 'Longest H2 surface'))
    handles = []
    for col, (key, dimension, title) in enumerate(metrics):
        ax = axes[col]
        values_all = []
        for index, seed in enumerate(seeds):
            values = []
            for condition in CONDITIONS:
                doc = lookup[condition, seed]
                threshold = doc['thresholds'][dimension]
                values.append(doc['metrics'][key] / threshold if threshold > 0 else np.nan)
            values_all.extend(values)
            handle, = ax.plot([0, 1], values, 'o-', color=palette(index % 10),
                             linewidth=1.4, markersize=5, alpha=0.9, label=f'Seed {seed}')
            if col == 0:
                handles.append(handle)
        finite = np.asarray(values_all)[np.isfinite(values_all)]
        y_max = max(1.25, float(finite.max()) * 1.18) if len(finite) else 1.25
        ax.axhspan(0, 1, color='#f1f3f5', zorder=-2)
        ax.axhline(1, color='#4d535a', linestyle='--', linewidth=1.1)
        ax.text(0.02, 1.015, 'Shuffle threshold', transform=ax.get_yaxis_transform(),
                fontsize=8, color='#4d535a', va='bottom')
        ax.set(xticks=[0, 1], xticklabels=[LABELS[c] for c in CONDITIONS],
               xlim=(-0.15, 1.15), ylim=(0, y_max), title=title,
               ylabel='Observed lifetime / null threshold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    passed = sum(doc['metrics'].get('core_torus_signature', doc['metrics']['torus_screen']) for doc in models)
    fig.suptitle(f'Core torus signature · {passed}/{len(models)} models detected\n' +
                 protocol_caption(models), fontsize=16, y=0.97)
    fig.legend(handles=handles, loc='lower center', ncol=min(8, len(seeds)),
               bbox_to_anchor=(0.5, 0.09), fontsize=8, frameon=False)
    fig.text(0.5, 0.025,
             'A core signature requires two strong H1 loops and one strong H2 surface to coexist with Betti counts (1, 2, 1).\n'
             'Per-model exploratory thresholds from independent circular neuron shifts; persistence refers to distance scale, not elapsed time.',
             ha='center', fontsize=8.5, linespacing=1.4)
    save_figure(fig, out / 'torus_signature')


def make_controls(controls, out):
    lookup = {doc['name']: doc for doc in controls}
    ordered = [lookup[name] for name in CONTROL_NAMES if name in lookup]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.81,
                        hspace=0.50, wspace=0.22)
    x_max = maximum_death(ordered)
    for ax, doc in zip(axes.flat, ordered):
        title = CONTROL_LABELS.get(doc['name'], doc.get('label', doc['name']))
        title = '\n'.join(textwrap.wrap(title, width=51))
        draw_barcodes(ax, doc, title, x_max)
        ax.set_xlabel('Connection distance / median pairwise distance', fontsize=9)
    for ax in axes.flat[len(ordered):]:
        ax.set_visible(False)
    fig.suptitle('Reference shapes · can this analysis detect a torus?\n' + protocol_caption(ordered),
                 fontsize=16, y=0.97)
    fig.text(0.08, 0.065,
             'Grid-module controls use the same spatial samples as the LSTMs. A clear torus has two dominant H1 loops and one H2 surface.\n'
             'Dark bars exceed their dimension-specific shuffle threshold; pale bars do not. Top six H1 and four H2 bars shown.\n'
             'The synthetic torus has an extra short loop that ends before its torus interval; core passes, strict fails.\n'
             'These are distance-scale barcodes, not temporal trajectories; the physical position sheet is not expected to be toroidal.',
             fontsize=8.7, va='center', linespacing=1.5)
    save_figure(fig, out / 'controls_barcodes')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'results/lstm_topology_1B')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.input
    required = [f'{condition}_s{seed}' for seed in range(1, 9) for condition in CONDITIONS]
    required += list(CONTROL_NAMES)
    missing = [name for name in required if not (args.input / f'{name}.json').is_file()]
    if missing:
        raise SystemExit('Missing completed results: ' + ', '.join(missing))
    records = [json.loads((args.input / f'{name}.json').read_text()) for name in required]
    models = [doc for doc in records if doc['kind'] == 'model']
    controls = [doc for doc in records if doc['kind'] == 'control']
    assert len(models) == 16 and len(controls) == 4
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': 'normal',
                         'axes.labelsize': 9, 'axes.titlesize': 11, 'savefig.facecolor': 'white'})
    make_all_seed_barcodes(models, out)
    make_signature(models, out)
    make_controls(controls, out)


if __name__ == '__main__':
    main()
