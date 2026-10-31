"""Small, data-backed figures for mechanisms/README.md.

No simulations or fitting: read the published analysis summaries and plot
training-seed means with SEM. The spatial diagram is explicitly illustrative.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'mechanisms/figures'
GRID = '#3379B5'
PGC = '#D87532'
LIGHT_PGC = '#E8AD82'
GRAY = '#8B939B'
INK = '#253442'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'semibold',
    'axes.labelsize': 11, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': '#ABB2B8',
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'svg.fonttype': 'none',
    'svg.hashsalt': 'pgc-lstm-mechanisms',
})


def read(name):
    return json.loads((ROOT / 'results' / name / 'summary.json').read_text())


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        path = OUT / f'{stem}.{suffix}'
        options = {'metadata': {'Date': None}} if suffix == 'svg' else {}
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.14, **options)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def sem(values):
    return np.std(values, ddof=1) / np.sqrt(len(values))


def vertical_bar(ax, x, values, color, width=.3):
    mean, error = np.mean(values), sem(values)
    ax.bar(x, mean, width, color=color, edgecolor='white', linewidth=.6, zorder=3)
    ax.errorbar(x, mean, yerr=error, color=INK, capsize=3, lw=1, zorder=4)
    ax.text(x, mean + error + 1.5, f'{mean:.1f}%', ha='center', va='bottom', fontsize=11)


def pca_figure():
    rows = read('lstm_variance_sources_1B')['models']
    conditions = [('gc100', GRID), ('pgc33', PGC)]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), layout='constrained',
                             gridspec_kw={'width_ratios': [1.55, 1]})
    for j, field in enumerate(('top10_share', 'within_top10_share')):
        for i, (condition, color) in enumerate(conditions):
            values = [100 * r[field] for r in rows if r['condition'] == condition]
            assert len(values) == 8
            vertical_bar(axes[0], j + (i - .5) * .33, values, color)
    axes[0].set(xticks=[0, 1], xticklabels=['Original activity', 'Episode means removed'],
                ylim=(0, 63), ylabel='Variance in first 10 PCs (%)', title='PCA concentration')
    for i, (condition, color) in enumerate(conditions):
        values = [100 * r['between_episode_share'] for r in rows if r['condition'] == condition]
        vertical_bar(axes[1], i, values, color, width=.5)
    axes[1].set(xticks=[0, 1], xticklabels=['Grid', 'PGC + grid'], ylim=(0, 40),
                ylabel='Share of total variance (%)', title='Differences between episode means')
    for ax in axes:
        ax.yaxis.grid(True, color='#E8EBED', lw=.65, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=GRID, label='Grid'), Patch(color=PGC, label='PGC + grid')],
                   loc='upper center', bbox_to_anchor=(.5, 1.03), frameon=False, ncol=2, fontsize=10)
    fig.supxlabel('Mean ± SEM across 8 training seeds', fontsize=9, color='#687A86')
    save(fig, '01_episode_variance')


def geometry_figure():
    fig, ax = plt.subplots(figsize=(7.1, 4.2), layout='constrained')
    center = np.array([0., 0.])
    hit = np.array([-.29, 0.])
    direction = np.array([np.cos(.5), np.sin(.5)])
    forward = hit + .2 * direction
    mirror = hit + .2 * direction * np.array([1., -1.])
    reverse = hit - .2 * direction
    ax.add_patch(Circle(center, .3, color='#F0F4F5', ec='#A5B3BC', lw=1.5))
    ax.plot([hit[0], .3], [0, 0], color='#C5CDD2', ls=':', lw=1)
    for point in (forward, mirror):
        ax.plot([point[0], 0], [point[1], 0], color='#BAC3C9', ls=':', lw=1)
    for point, color, style in [(reverse, GRAY, '-'), (forward, PGC, '-'), (mirror, PGC, '--')]:
        ax.annotate('', xy=point, xytext=hit,
                    arrowprops=dict(arrowstyle='->', lw=2.4, color=color, linestyle=style))
        ax.scatter(*point, s=65, color=color, zorder=5)
    ax.scatter(*hit, s=75, color=GRID, zorder=6)
    ax.scatter(*center, s=30, color=INK, zorder=6)
    ax.annotate('Pickup', hit, xytext=(-5, 14), textcoords='offset points',
                ha='right', color=GRID, weight='semibold')
    ax.annotate('20 cm ahead', forward, xytext=(7, 5), textcoords='offset points', color=PGC)
    ax.annotate('Reflected', mirror, xytext=(7, -12), textcoords='offset points', color=PGC)
    ax.annotate('Backward', reverse, xytext=(-2, -18), textcoords='offset points', color=GRAY, ha='center')
    ax.annotate('Center', center, xytext=(8, -4), textcoords='offset points', color=INK)
    ax.text(0, .325, 'Goal area', ha='center', weight='semibold', fontsize=13)
    ax.text(0, -.335, '30 cm radius', ha='center', color='#687A86', fontsize=10)
    ax.text(-.515, -.335, 'Illustration', color='#7F8A93', fontsize=9)
    ax.set(xlim=(-.54, .35), ylim=(-.37, .37), aspect='equal')
    ax.axis('off')
    save(fig, '02_goal_message')


def horizontal_scores(ax, rows, color, xmax=125):
    for y, (label, record, shade) in enumerate(rows):
        mean, error = record['mean'], record['sem']
        assert record['n'] == 8
        ax.barh(y, mean, height=.5, color=shade or color, edgecolor='none')
        ax.errorbar(mean, y, xerr=error, fmt='none', ecolor=INK, capsize=3, lw=1)
        ax.text(mean + error + 2.0, y, f'{mean:.1f}', va='center', fontsize=11, weight='semibold')
    ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows],
           ylim=(len(rows) - .5, -.5), xlim=(0, xmax), xticks=[0, 40, 80, 120],
           xlabel='Score (mean ± SEM)')
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=9)
    ax.tick_params(axis='x', length=0)
    ax.xaxis.grid(True, color='#E8EBED', lw=.65)
    ax.set_axisbelow(True)


def goal_scores_figure():
    data = read('lstm_goal_geometry_1B')['estimates']['mean_score']
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.9), layout='constrained')
    horizontal_scores(axes[0], [
        ('Original', data['pgc33/intact'], None),
        ('Ordinary goal code', data['pgc33/plain_goal'], LIGHT_PGC),
        ('Reflected', data['pgc33/mirror_goal'], None),
        ('Backward', data['pgc33/reverse_goal'], LIGHT_PGC),
        ('True center', data['pgc33/center_pgc_goal'], LIGHT_PGC),
    ], PGC)
    horizontal_scores(axes[1], [
        ('Original', data['gc100/intact'], None),
        ('Goal shifted ahead', data['gc100/lookahead_goal'], '#83AFCF'),
        ('True center', data['gc100/center_goal'], '#83AFCF'),
    ], GRID)
    axes[0].set_title('PGC-trained RNNs', color=PGC, pad=14)
    axes[1].set_title('Grid-trained RNNs', color=GRID, pad=14)
    save(fig, '03_goal_behavior')


def rescue_figure():
    data = read('lstm_gate_rescue_1B')['estimates']['mean_score']
    fig, ax = plt.subplots(figsize=(8.7, 3.35), layout='constrained')
    horizontal_scores(ax, [
        ('Original predictive inputs', data['intact'], None),
        ('Replace both inputs with grid code', data['both_grid'], GRAY),
        ('Restore state content + output', data['restore_candidate_output'], None),
        ('Restore input + forget gates', data['restore_input_forget'], LIGHT_PGC),
    ], PGC)
    ax.set_title('Restoring predictive influence inside PGC-trained RNNs', pad=14)
    save(fig, '04_rnn_rescue')


if __name__ == '__main__':
    pca_figure()
    geometry_figure()
    goal_scores_figure()
    rescue_figure()
    print('Wrote four PNG/SVG figures to', OUT)
