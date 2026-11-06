"""Episode-level summaries for the prospectively frozen reorientation pilot.

All scheduled endpoints are retained. Windows are not independent replicates:
uncertainty resamples paired environment slots, each containing two episodes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rl_blind_reorientation'
NAMES = {'gc100': 'ring-20261105-reorient-grid-01',
         'pgc33': 'ring-20261105-reorient-pgc-01'}
STAGE_FRAMES = 10_014_720
BOOTSTRAPS, SEED = 10_000, 148207
METRICS = ('free_hits', 'progress_m', 'path_m', 'free_updates', 'free_windows')


def episode_values(result, variant):
    """Retain every original episode, including no-discovery/no-window zeros."""
    original = result['variants'][variant]['rows']
    keys = [(r['slot'], r['episode']) for r in original]
    assert len(set(keys)) == len(keys) == result['episodes_per_variant']
    rows = {key: {metric: 0. for metric in METRICS} for key in keys}
    for event in result['free_window'][variant]['rows']:
        key = (event['slot'], event['episode'])
        row = rows[key]
        row['free_hits'] += int(event['goal_hit'])
        row['progress_m'] += event['progress_m']
        row['path_m'] += event['path_m']
        row['free_updates'] += event['free_updates']
        row['free_windows'] += 1
    ordered = sorted(rows)
    values = np.asarray([[rows[k][m] for m in METRICS] for k in ordered])
    totals = values.sum(0)
    source = result['free_window'][variant]
    for metric, field in [('free_hits', 'free_hits'), ('progress_m', 'free_progress_m'),
                          ('path_m', 'free_path_m'), ('free_updates', 'free_updates')]:
        assert np.isclose(totals[METRICS.index(metric)], source[field], atol=1e-4, rtol=1e-5)
    return ordered, values


def clustered_contrast(result, positive, negative, metric='free_hits'):
    """Positive-minus-negative within the same fixed network and eval slots."""
    keys, a = episode_values(result, positive)
    keys_b, b = episode_values(result, negative)
    assert keys == keys_b
    difference = a[:, METRICS.index(metric)] - b[:, METRICS.index(metric)]
    slots = sorted({k[0] for k in keys})
    clusters = [difference[[k[0] == s for k in keys]] for s in slots]
    assert len({len(c) for c in clusters}) == 1
    means = np.array([c.mean() for c in clusters])
    rng = np.random.default_rng(SEED)
    estimates = means[rng.integers(len(slots), size=(BOOTSTRAPS, len(slots)))].mean(1)
    return dict(mean=float(difference.mean()), ci95=np.quantile(estimates, [.025, .975]).tolist(),
                paired_slots=len(slots), episodes=len(keys), metric=metric,
                units='per original episode', contrast=f'{positive} minus {negative}')


def summarize(result):
    assert result['all_first_discoveries_identical'] and result['clean_no_noise']
    variants = {}
    for variant in result['variants']:
        keys, values = episode_values(result, variant)
        means = dict(zip(METRICS, values.mean(0).tolist()))
        totals = values.sum(0)
        steps = totals[METRICS.index('free_updates')]
        variants[variant] = dict(**means, episodes=len(keys),
            free_hits_per_100_updates=float(100*totals[0]/steps) if steps else None,
            progress_m_per_100_updates=float(100*totals[1]/steps) if steps else None,
            whole_episode_score=result['variants'][variant]['mean_score'],
            forced_hits_total=result['free_window'][variant]['forced_hits'])
    contrasts = {}
    for vision in ('blind', 'seen'):
        for intervention in ('ordinary', 'plus60', 'minus60', 'reset_after_glimpse', 'reset_every_blind'):
            for metric in ('free_hits', 'progress_m'):
                key = f'{vision}_native_minus_{intervention}_{metric}'
                contrasts[key] = clustered_contrast(result, f'{vision}_native',
                                                     f'{vision}_{intervention}', metric)
    return dict(variants=variants, contrasts=contrasts, schedule=result['schedule'],
                probe_audit=result['probe_audit'])


def collect(require_complete=False):
    data = {}
    for condition, name in NAMES.items():
        folder = ROOT / 'results/ring_cloud_artifacts' / name
        endpoints = []
        for stage in range(5):
            path = folder / ('baseline.json' if stage == 0 else f'stage{stage}_evaluation.json')
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            endpoints.append(dict(stage=stage, learning_frames=stage*STAGE_FRAMES,
                source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **summarize(result)))
        completed = (folder / 'summary.json').exists() and len(endpoints) == 5
        if completed:
            completed = bool(json.loads((folder / 'summary.json').read_text())['completed'])
        if require_complete:
            assert completed, f'Incomplete reorientation pilot: {condition}'
        data[condition] = dict(completed=completed, endpoints=endpoints)
    return data


def figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout='constrained')
    styles = [('native', '#2878a4', '-', 'Native cue'),
              ('ordinary', '#d27927', '--', 'Predictive pulse removed'),
              ('reset_after_glimpse', '#765596', ':', 'Memory reset after pulse')]
    for ax, (condition, item) in zip(axes, data.items()):
        x = [r['learning_frames']/1e6 for r in item['endpoints']]
        for suffix, color, line, label in styles:
            y = [r['variants'][f'blind_{suffix}']['free_hits'] for r in item['endpoints']]
            ax.plot(x, y, color=color, ls=line, marker='o', label=label)
        ax.set(title='Grid' if condition == 'gc100' else 'PGC + grid',
               xlabel='Additional training frames (millions)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Goals reached during free movement\nper original episode')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Blind reorientation' + ('' if all(v['completed'] for v in data.values()) else ' — in progress'))
    fig.supxlabel('One original seed pair · forced-movement rewards excluded · every fixed endpoint', fontsize=9)
    fig.savefig(OUT / 'free_navigation_learning.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    data = collect(args.require_complete)
    OUT.mkdir(parents=True, exist_ok=True)
    output = dict(conditions=data,
        inference='Within-checkpoint evaluation uncertainty only; one original training seed per population. Episodes with no event remain zero. No windows, phases, or checkpoints are treated as independent training seeds.',
        uncertainty=f'{BOOTSTRAPS} paired bootstrap resamples of environment slots, preserving the two episodes within each slot; seed{SEED}. Descriptive95% intervals, no multiplicity-adjusted significance claim.',
        exposure='Total free-window outcomes per original episode are primary. Per100free-update ratios are descriptive because intervention can alter window admission and reward times.',
        schedule='Baseline and all four prespecified stages; no favorable checkpoint selection.')
    (OUT / 'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    if any(v['endpoints'] for v in data.values()):
        figure(data)
    print(json.dumps({c:len(v['endpoints']) for c,v in data.items()}))


if __name__ == '__main__':
    main()
