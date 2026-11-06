"""Descriptive display of the complete frozen learned-score observation test.

Reads completed numerical outputs only. Does not modify probes or criteria.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_actor_logit_observable'
COLORS = {'gc100': '#357eb0', 'pgc33': '#dc7830'}


def main():
    data = json.loads((OUT / 'summary.json').read_text())
    assert data['complete'] and data['checks'] == data['expected_checks'] == 152
    rows = data['rows']
    assert Counter(r['cohort'] for r in rows) == {'original1B': 56, 'forty': 32, 'ninety': 64}
    assert all(r['replay_max_abs_error'] == 0 for r in rows)
    quantities = np.array([[r['radial_over_tangent'], r['centered_departure_p90']] for r in rows])
    assert np.isfinite(quantities).all() and (quantities > 0).all()
    low = np.minimum(quantities.min(axis=0) * .65, [.01, .01])
    high = np.maximum(quantities.max(axis=0) * 1.4, [1.5, 1.5])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharex=True, sharey=True, layout='constrained')
    cohorts = [('original1B', 'Original 1B'), ('forty', '40M task'), ('ninety', '90M entropy')]
    for ax, (cohort, title) in zip(axes, cohorts):
        selected = [r for r in rows if r['cohort'] == cohort]
        ax.add_patch(Rectangle((low[0], low[1]), .5-low[0], .05-low[1],
                               color='#dce8da', alpha=.6, zorder=0))
        for condition, color in COLORS.items():
            for metric in ('raw_full_state_kicks', 'h_c_RMS_full_state_kicks'):
                points = [r for r in selected if r['condition'] == condition and r['metric'] == metric]
                raw = metric == 'raw_full_state_kicks'
                ax.scatter([r['radial_over_tangent'] for r in points],
                           [r['centered_departure_p90'] for r in points], s=29,
                           facecolors=color if raw else 'none', edgecolors=color,
                           linewidths=1, alpha=.75, zorder=2)
        ax.axvline(.5, color='.45', lw=.9, ls='--')
        ax.axhline(.05, color='.45', lw=.9, ls='--')
        ax.set(xscale='log', yscale='log', xlim=(low[0], high[0]), ylim=(low[1], high[1]),
               title=f'{title} · {len(selected)} checks', xlabel='Radial / angular change')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.1)
    axes[0].set_ylabel('Departure from earlier score curve\n(radii; common drift removed)')
    handles = [Line2D([], [], marker='o', ls='', color=color, label=label)
               for (condition, color), label in zip(COLORS.items(), ('Grid', 'PGC + grid'))]
    handles += [Line2D([], [], marker='o', ls='', markerfacecolor=fill, markeredgecolor='.35',
                       label=label) for fill, label in (('.35', 'Raw kicks'), ('none', 'h/c-scaled kicks'))]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Two checks of a stable circular action-score pattern', fontsize=13)
    fig.savefig(OUT / 'actor_score_recovery.png', dpi=190)
    plt.close(fig)

    lines = ['# Dynamics in the learned action scores', '',
             f"**{data['centered_candidates']}/152 drift-subtracted and {data['noncentered_candidates']}/152 original-curve checks meet all frozen criteria.**", '',
             '![Selective recovery and score-curve preservation](actor_score_recovery.png)', '',
             'We looked directly at the network’s six learned action scores, observed at four future times. This removes hidden activity that never reaches those future scores. No decoder or circular representation was fitted.', '',
             '**Leftward:** a radial disturbance shrinks faster than an angular disturbance. **Downward:** the later score patterns remain close to their earlier curve after common drift is removed. The shaded region meets these two criteria only; amplitude, curve geometry, phase correspondence and the other recovery guards must also pass. All contexts, cue branches and both perturbation metrics are plotted; exact shams overlap.', '',
             '| Cohort | Checks | Drift-subtracted passes | Original-curve passes |',
             '|---|---:|---:|---:|']
    for key, label in [('pilot', 'Original seeds 1–2'), ('heldout', 'Original seeds 3–8'),
                       ('fixed40M', 'Fixed 40M endpoints'), ('fixed90M', 'Fixed 90M endpoints')]:
        selected = [r for r in rows if r['subgroup'] == key]
        lines.append(f"| {label} | {len(selected)} | {sum(r['centered_candidate'] for r in selected)} | {sum(r['noncentered_candidate'] for r in selected)} |")
    lines += ['', 'The higher-entropy Grid model’s second context is the point inside the shaded region. Its native raw-kick radial/angular ratio is 0.019 and curve departure is 0.004 radii. It fails the separation of distant curve locations and the phase-coverage/correspondence guards, so those two favorable numbers alone do not identify a ring. Its exact cue shams and both metrics are retained.', '',
              'These are nested tests of 16 original networks and six continued endpoints from one of those seed pairs, not 152 independent networks. The earlier held-out label describes the original collection cohort; its other analyses were already known before this follow-up.', '',
              'Selective recovery can occur in a changing directional memory or a contracting plane as well as a ring. These tests ask for a stable circular factor in the exact learned scores under one common input. An evolving representation can retain direction while failing correspondence to the earlier curve. The results should therefore be read with the native navigation and cue/state-intervention results.', '',
              'Every saved trajectory replayed exactly before testing. The unchanged amplitude threshold is measured in centered-logit units here. The six action scores have their action-common component removed; subtracting the mean across the direction family is a separate operation.', '',
              '[Complete numerical results and all gates](summary.json) · [Frozen method and commands](README.md) · [Freeze and independent reviews](review.json) · [Display source](../../scripts/plot_actor_logit_observable.py)', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
