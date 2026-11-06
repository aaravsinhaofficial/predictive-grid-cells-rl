"""Simple descriptive report for the complete native movement census."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/native_movement_cycles'
NAMES = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}
COLORS = {'gc100': '#2877ad', 'pgc33': '#e27635'}
VIEWS = ('full_hc_raw', 'full_hc_block_rms', 'hidden_h', 'centered_actor_logits')


def main():
    records, hashes = [], {}
    for condition in NAMES:
        for seed in (1, 2):
            folder = OUT / f'{condition}_s{seed}'
            for name in ('summary.json', 'census.json'):
                p = folder / name
                hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
            summary = json.loads((folder / 'summary.json').read_text())
            census = json.loads((folder / 'census.json').read_text())
            assert summary['complete'] and summary['policy_unchanged']
            cycles = [x for x in census['cycles'] if x['known_goal']]
            views = {}
            for view in VIEWS:
                values = np.array([x[view]['closure_over_radius'] for x in cycles
                                   if x[view]['closure_over_radius'] is not None])
                views[view] = dict(n=len(values), median=float(np.median(values)),
                                   minimum=float(values.min()), q10=float(np.quantile(values, .1)),
                                   q90=float(np.quantile(values, .9)))
            records.append(dict(condition=condition, seed=seed, episodes=census['episodes'],
                return_segments=census['return_segments'],
                returns_with_fullturn=census['return_segments_with_heading_turns'],
                returns_with_repeated_turns=census['return_segments_with_repeated_turns'],
                all_heading_turns=census['total_heading_turns'], return_heading_turns=len(cycles),
                mean_native_score=summary['episode_summary']['mean_score'], closure=views,
                seconds=summary['seconds'], trace_sha256=summary['trace_sha256'],
                interpretation='Descriptive discovery census; turns and trips are nested within episodes and trained models.'))
    result = dict(models=4, original_seed_pairs=2, episodes=sum(x['episodes'] for x in records),
                  recurrent_updates=4*32*1350, records=records, source_hashes=hashes)
    (OUT / 'census_report.json').write_text(json.dumps(result, indent=2)+'\n')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.7), layout='constrained')
    for ci, condition in enumerate(NAMES):
        rows = [x for x in records if x['condition'] == condition]
        xx = ci + np.array([-.07, .07])
        ax[0].scatter(xx, [100*x['returns_with_fullturn']/x['return_segments'] for x in rows],
                      color=COLORS[condition], s=62)
        ax[1].scatter(xx, [x['closure']['hidden_h']['median'] for x in rows],
                      color=COLORS[condition], s=62)
        for x, row in zip(xx, rows):
            ax[0].annotate(f"s{row['seed']}", (x, 100*row['returns_with_fullturn']/row['return_segments']),
                           xytext=(6, 2), textcoords='offset points', fontsize=9)
            ax[1].annotate(f"s{row['seed']}", (x, row['closure']['hidden_h']['median']),
                           xytext=(6, 2), textcoords='offset points', fontsize=9)
    for panel in ax:
        panel.set_xticks([0, 1], NAMES.values())
        panel.set_xlim(-.4, 1.45)
    ax[0].set(title='Most return trips contain no full turn',
              ylabel='Return segments with a full turn (%)', ylim=(0, 25))
    ax[1].axhline(0, color='.65', ls='--', lw=1)
    ax[1].set(title='Activity usually ends somewhere else',
              ylabel='Endpoint gap / activity radius', ylim=(-.05, 2))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'native_turn_census.{ext}', dpi=180)
    lines = ['# Recurrent activity during native turns', '',
        '**We recorded 128 complete episodes from four original agents, with no training or input changes.** '
        'Full heading turns occurred in 12–19% of return segments. After a full turn, hidden activity usually '
        'ended roughly one to two activity radii from its starting pattern.', '',
        '![Native movement census](native_turn_census.png)', '',
        'A point at zero in the right panel would mean the hidden-activity trajectory closed. '
        'The plot uses all 256 hidden units; PCA and fitted angular decoders are not involved. '
        'Each point is one trained model, summarized over its native return turns.', '',
        '| Original agent | Return segments | With a full turn | With repeated turns | Median hidden-activity endpoint gap / radius |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append(f"| {NAMES[r['condition']]} seed {r['seed']} | {r['return_segments']} | "
                     f"{r['returns_with_fullturn']} | {r['returns_with_repeated_turns']} | "
                     f"{r['closure']['hidden_h']['median']:.2f} |")
    lines += ['', '**What this suggests.** A full body turn is usually part of a changing navigation computation, '
        'rather than one completed orbit through an unchanged neural activity pattern. The PGC models have '
        'smaller median hidden-activity gaps in these two discovery seed pairs. The causal perturbation experiment '
        'tests whether prediction helps recover useful movement-related activity.', '',
        '**What was measured.** A turn is a first passage through a new signed multiple of 2π within one uninterrupted '
        'search or return segment. Goal pickups and teleports are never stitched into cycles. Return segments begin '
        'with a saved goal and include both completed and episode-censored trips. Endpoints use the actual recorded '
        'discrete steps, so crossing overshoot, sensory noise and changing physical position contribute to the gap. '
        'This census alone neither tests attraction nor isolates an invariant subsystem.', '',
        'The complete data also retain full `(h,c)` geometry in raw and fixed block-RMS units, and all centered actor '
        'logits. Every model passed exact native-forward equivalence, and its weights remained unchanged. '
        'The ordinary-current-code substitution was separately checked against an independently instantiated grid population; '
        'it changes 169 current-code slots in PGC agents and is an exact sham in Grid agents. The census itself uses native inputs only.', '',
        'This is an exploratory result from two trained seed pairs. Turns are repeated observations within each model, '
        'not independent training replications. [All model summaries and source hashes](census_report.json).', '',
        'Reproduce this report: `python scripts/report_native_movement_cycles.py`.']
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_hashes')}))


if __name__ == '__main__':
    main()
