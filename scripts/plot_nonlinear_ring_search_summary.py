"""Small figures for the exploratory native ring search; saved results only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ring_nonlinear_design'


def main():
    source = ROOT / 'results/native_movement_pulses/report/summary.json'
    report = json.loads(source.read_text())
    assert report['complete']
    routes = ('current_grid', 'current_candidate_output_rescue')
    values, counts = [], []
    for seed in (1, 2):
        model = report['models'][f'pgc33_s{seed}']
        values.append([100*model['unpulsed_mode_effects'][route]['action_difference_fraction']['mean']
                       for route in routes])
        counts.append(model['eligible'])
    values = np.asarray(values)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.6, 3.9), layout='constrained')
    colors = ('#e27635', '#389875')
    labels = ('Current prediction removed', 'Candidate + output restored')
    for i, (color, label) in enumerate(zip(colors, labels)):
        x = np.arange(2) + (i-.5)*.31
        bars = ax.bar(x, values[:, i], width=.28, color=color, label=label)
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values[:, i]], padding=4, fontsize=11)
    ax.set(xticks=[0, 1], xticklabels=['PGC model, seed 1', 'PGC model, seed 2'],
           ylabel='Actions different from native (%)', ylim=(0, 28),
           title='Predictive input changes steering')
    ax.legend(frameon=False, fontsize=10, loc='upper center')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'predictive_steering.{ext}', dpi=180)
    (OUT / 'predictive_steering_values.json').write_text(json.dumps(dict(
        source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metric='Mean within-case sampled-action difference fraction, during shared preterminal updates',
        original_seed_ids=[1, 2], selected_turn_cases=counts, input_routes=list(routes),
        percentages=values.tolist(),
        scope='Unpulsed branches; same frozen agent and initial state, native goal cue throughout, matched random tapes. '
              'Two discovery models. Restored action similarity is distinct from navigation improvement.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
