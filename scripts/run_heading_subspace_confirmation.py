"""Run the frozen subspace follow-up on every locally collected final model.

Does not contact cloud services. With --watch, waits for the parent to collect
the 16 final checkpoints. No checkpoint is selected by behavioral outcome.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import stats
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2001, 2017))
VARIANTS = ('native', 'ordinary', 'remove_direction', 'restore_direction',
            'sham0_restore', 'sham1_restore', 'sham2_restore',
            'sham0_remove', 'sham1_remove', 'sham2_remove')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = x.mean()
    half = stats.t.ppf(.975, n - 1) * stats.sem(x) if n > 1 else np.nan
    signs = np.array(list(itertools.product((-1., 1.), repeat=n)))
    null = np.mean(signs * x[None], axis=1)
    exact = np.mean(np.abs(null) >= abs(mean) - 1e-12)
    return {'n_seeds': n, 'mean_difference': float(mean),
            'ci95': [float(mean - half), float(mean + half)],
            'exact_two_sided_p': float(exact), 'positive_seeds': int(np.sum(x > 0)),
            'negative_seeds': int(np.sum(x < 0)), 'paired_values': x.tolist()}


def correct(family):
    items = sorted(family, key=lambda k: family[k]['exact_two_sided_p'])
    previous = 0.
    for i, key in enumerate(items):
        previous = min(1., max(previous, (len(items) - i) * family[key]['exact_two_sided_p']))
        family[key]['holm_exact_p'] = previous


def summarize(out):
    rows = []
    for seed in SEEDS:
        p = out / f'predictive_s{seed}'
        if not (p / 'summary.json').exists() or not (p / 'navigation/summary.json').exists():
            continue
        a = json.loads((p / 'summary.json').read_text())
        b = json.loads((p / 'navigation/summary.json').read_text())
        f = a['fit']['first_harmonic']['new_places_test']
        row = {'seed': seed, 'projected_energy_fraction': f['projected_raw_energy_fraction'],
               'harmonic_r2': f['harmonic_regression_r2'],
               'within_position_harmonic_r2': f['within_position_harmonic_r2'],
               'direction_readout_error_deg': f['projected_direction_mae_deg']}
        for variant in VARIANTS:
            row[f'heading_{variant}'] = a['forced']['scenarios']['new_places'][variant]['all_mae_deg']
            row[f'navigation_{variant}'] = b['scenarios']['new_places']['durations']['40s'][variant]['success_fraction']
            row[f'holding_{variant}'] = b['scenarios']['new_places']['durations']['40s'][variant]['sustained_dwell_success_fraction']
            ar = [r for r in a['anchoring']['rows'] if r['variant'] == variant and r['step'] == 100 and r['context'] < 5]
            for metric in ('phase_local_gain_median', 'baseline_mae_deg', 'circle_coverage', 'state_radius_retained', 'full_state_error_retained'):
                row[f'anchor_{variant}_{metric}'] = float(np.mean([r[metric] for r in ar]))
        rows.append(row)
    if not rows:
        return
    with (out / 'paired_seeds.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    complete = len(rows) == len(SEEDS)
    summary = {'expected_seeds': list(SEEDS), 'completed_seeds': [r['seed'] for r in rows],
               'complete': complete, 'pilot_excluded': True, 'rows': rows,
               'means': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}}
    if complete:
        heading = {
            'native_benefit': [r['heading_ordinary'] - r['heading_native'] for r in rows],
            'removal_cost': [r['heading_remove_direction'] - r['heading_native'] for r in rows],
            'restoration_benefit': [r['heading_ordinary'] - r['heading_restore_direction'] for r in rows],
            'specificity_over_random': [np.mean([r[f'heading_sham{j}_restore'] for j in range(3)]) - r['heading_restore_direction'] for r in rows]}
        navigation = {
            'native_benefit': [r['navigation_native'] - r['navigation_ordinary'] for r in rows],
            'removal_cost': [r['navigation_native'] - r['navigation_remove_direction'] for r in rows],
            'restoration_benefit': [r['navigation_restore_direction'] - r['navigation_ordinary'] for r in rows],
            'specificity_over_random': [r['navigation_restore_direction'] - np.mean([r[f'navigation_sham{j}_restore'] for j in range(3)]) for r in rows]}
        summary['heading_tests'] = {k: paired(v) for k, v in heading.items()}
        summary['navigation_tests'] = {k: paired(v) for k, v in navigation.items()}
        correct(summary['heading_tests']); correct(summary['navigation_tests'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if complete:
        figure(summary, out)
    return summary


def figure(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['rows']
    variants = ['native', 'ordinary', 'remove_direction', 'restore_direction']
    labels = ['Native', 'Ordinary', 'Remove\ndirection', 'Restore\ndirection', 'Random\nrestore']
    colors = ['#D87532', '#8E98A3', '#8C64B4', '#317BB5', '#D1D5DA']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
    for ax, prefix, scale, title, ylabel in zip(axes,
            ['heading_', 'navigation_', 'coverage_'], [1, 100, 100],
            ['Heading at new places', 'Navigation at new places', 'Ring coverage'],
            ['Error (degrees)', 'Reaching the goal (%)', 'Circle covered (%)']):
        get = lambda r, v: r[f'anchor_{v}_circle_coverage'] if prefix == 'coverage_' else r[prefix + v]
        values = [np.array([get(r, v) for r in rows]) * scale for v in variants]
        values.append(np.array([np.mean([get(r, f'sham{j}_restore') for j in range(3)]) for r in rows]) * scale)
        means = [v.mean() for v in values]
        ci = [stats.sem(v) * stats.t.ppf(.975, len(v) - 1) for v in values]
        ax.bar(range(5), means, yerr=ci, color=colors, width=.72, capsize=3)
        for i, v in enumerate(values):
            ax.scatter(i + np.linspace(-.22, .22, len(v)), v, s=10, color='#44505C', alpha=.5, zorder=3)
        ax.set(title=title, ylabel=ylabel, xticks=range(5), xticklabels=labels)
        ax.tick_params(axis='x', labelsize=8); ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('16 frozen trained models · Dots: training seeds · Bars: mean with 95% confidence interval', fontsize=9)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'direction_plane_confirmation.{ext}', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, default=ROOT / 'results/ring_cloud_artifacts')
    p.add_argument('--out', type=Path, default=ROOT / 'results/ring_heading/subspace_confirmation')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=30.)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.out / 'frozen_manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        assert sha(ROOT / name) == digest, ('Frozen source changed', name)
    assert sha(args.out / 'PROTOCOL.md') == manifest['protocol_sha256']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    processed, last_missing = set(), None
    while True:
        for seed in SEEDS:
            if seed in processed:
                continue
            candidates = sorted(args.artifacts.glob(f'ring-20261105-confirm-*/predictive_s{seed}.pt'))
            if not candidates:
                continue
            if len(candidates) > 1:
                assert len({sha(p) for p in candidates}) == 1, ('Conflicting checkpoint copies', seed)
            checkpoint = candidates[0]
            try:
                info = torch.load(checkpoint, map_location='cpu', weights_only=False)
            except (EOFError, RuntimeError, OSError):
                if args.watch:
                    continue  # The parent may still be copying the artifact.
                raise
            assert info['condition'] == 'predictive' and info['seed'] == seed and info['steps_completed'] == 6000
            out = args.out / f'predictive_s{seed}'; out.mkdir(exist_ok=True)
            for script, extra, final, log in (
                ('heading_direction_subspace.py', [], out / 'summary.json', out / 'analysis.log'),
                ('heading_subspace_navigation.py', ['--subspaces', str(out / 'fitted_subspaces.npz'), '--validate-reference'],
                 out / 'navigation/summary.json', out / 'navigation.log')):
                if final.exists():
                    prior = json.loads(final.read_text())
                    assert prior['protocol']['checkpoint_sha256'] == sha(checkpoint)
                    continue
                target = out if script == 'heading_direction_subspace.py' else out / 'navigation'
                command = [sys.executable, str(ROOT / 'scripts' / script), '--checkpoint', str(checkpoint), '--out', str(target)] + extra
                print('RUN', seed, script, flush=True)
                with log.open('w') as stream:
                    subprocess.run(command, check=True, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            processed.add(seed)
            summary = summarize(args.out)
            print('COMPLETED', seed, 'total', len(summary['completed_seeds']), flush=True)
        remaining = [seed for seed in SEEDS if seed not in processed]
        if remaining != last_missing:
            print('WAITING FOR LOCAL CHECKPOINTS', remaining, flush=True)
            last_missing = remaining
        if not remaining or not args.watch:
            break
        time.sleep(args.poll_seconds)
    summarize(args.out)


if __name__ == '__main__':
    main()
