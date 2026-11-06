"""Run the separate tonic/variable-drive diagnostic on all 16 local models."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import stats

from run_heading_subspace_confirmation import paired, correct


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2001, 2017))
VARIANTS = ('native', 'ordinary', 'restore_direction', 'mean_only_restore', 'mean_preserving_removal')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(out):
    rows = []
    for seed in SEEDS:
        p = out / f'predictive_s{seed}/summary.json'
        if not p.exists(): continue
        s = json.loads(p.read_text())
        r = {'seed': seed, 'mean_projected_energy_fraction': s['calibration']['mean_fraction_of_projected_drive_energy']}
        for v in VARIANTS:
            r[f'heading_{v}'] = s['forced']['scenarios']['new_places'][v]['all_mae_deg']
            r[f'navigation_{v}'] = s['navigation']['new_places']['durations']['40s'][v]['success_fraction']
            a = [a for a in s['anchoring']['rows'] if a['variant'] == v and a['step'] == 100 and a['context'] < 5]
            for key in ('baseline_mae_deg', 'circle_coverage', 'phase_local_gain_median', 'full_state_error_retained'):
                r[f'anchor_{v}_{key}'] = float(np.mean([x[key] for x in a]))
        rows.append(r)
    if not rows: return None
    with (out / 'paired_seeds.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    s = {'complete': len(rows) == len(SEEDS), 'planned_seeds': list(SEEDS), 'completed_seeds': [r['seed'] for r in rows],
         'additional_exploratory_diagnostic': True, 'original_primary_test_families_unchanged': True,
         'rows': rows, 'means': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}}
    if s['complete']:
        for endpoint, beneficial_sign in (('heading', -1), ('navigation', 1)):
            contrasts = {
                'mean_only_benefit': [(r[f'{endpoint}_mean_only_restore'] - r[f'{endpoint}_ordinary']) * beneficial_sign for r in rows],
                'full_plane_over_mean_only': [(r[f'{endpoint}_restore_direction'] - r[f'{endpoint}_mean_only_restore']) * beneficial_sign for r in rows],
                'mean_preserving_removal_cost': [(r[f'{endpoint}_native'] - r[f'{endpoint}_mean_preserving_removal']) * beneficial_sign for r in rows]}
            family = {k: paired(v) for k, v in contrasts.items()}; correct(family)
            s[endpoint + '_exploratory_tests'] = family
        figure(s, out)
    (out / 'summary.json').write_text(json.dumps(s, indent=2, allow_nan=False) + '\n')
    return s


def figure(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Native', 'Ordinary', 'Full plane', 'Mean only', 'Remove\nvariation']
    colors = ['#D87532', '#8E98A3', '#317BB5', '#7FAD91', '#8C64B4']
    rows = summary['rows']; fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
    for ax, family, scale, title, ylabel in zip(axes, ['heading', 'navigation', 'coverage'], [1, 100, 100],
            ['Heading at new places', 'Navigation at new places', 'Ring coverage'],
            ['Error (degrees)', 'Reaching the goal (%)', 'Circle covered (%)']):
        values = [np.array([r[f'anchor_{v}_circle_coverage'] if family == 'coverage' else r[f'{family}_{v}'] for r in rows]) * scale for v in VARIANTS]
        ax.bar(range(5), [v.mean() for v in values], color=colors, capsize=3,
               yerr=[stats.sem(v) * stats.t.ppf(.975, 15) for v in values])
        for i, v in enumerate(values): ax.scatter(i + np.linspace(-.22, .22, len(v)), v, s=10, color='#44505C', alpha=.5)
        ax.set(title=title, ylabel=ylabel, xticks=range(5), xticklabels=labels, ylim=(0, 110))
        ax.tick_params(axis='x', labelsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('16 frozen trained models · Additional mean-versus-varying-drive diagnostic · 95% intervals over training seeds', fontsize=9)
    for ext in ('png', 'pdf', 'svg'): fig.savefig(out / f'tonic_direction_confirmation.{ext}', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=ROOT / 'results/ring_heading/subspace_confirmation')
    p.add_argument('--out', type=Path, default=ROOT / 'results/ring_heading/tonic_direction_confirmation')
    p.add_argument('--watch', action='store_true'); p.add_argument('--poll-seconds', type=float, default=30.)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'scripts/heading_tonic_direction_diagnostic.py'
    frozen = json.loads((args.original / 'frozen_manifest.json').read_text())
    hashes = {p: sha(ROOT / p) for p in frozen['sources']}
    assert hashes == frozen['sources']
    metadata = {'diagnostic_source_sha256': sha(source), 'runner_source_sha256': sha(__file__),
        'protocol_sha256': sha(args.out / 'PROTOCOL.md'), 'frozen_original_sources': hashes, 'seeds': list(SEEDS)}
    (args.out / 'source_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    done, last_remaining = set(), None
    while True:
        for seed in SEEDS:
            if seed in done: continue
            original = args.original / f'predictive_s{seed}'
            if not (original / 'summary.json').exists(): continue
            assert {p: sha(ROOT / p) for p in hashes} == hashes
            parent = json.loads((original / 'summary.json').read_text())
            checkpoint = Path(parent['protocol']['checkpoint']); out = args.out / f'predictive_s{seed}'
            out.mkdir(exist_ok=True)
            if not (out / 'summary.json').exists():
                print('RUN TONIC', seed, flush=True)
                with (out / 'diagnostic.log').open('w') as stream:
                    subprocess.run([sys.executable, str(source), '--checkpoint', str(checkpoint),
                        '--subspaces', str(original / 'fitted_subspaces.npz'), '--out', str(out)],
                        cwd=ROOT, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT)
            saved = json.loads((out / 'summary.json').read_text())
            assert saved['protocol']['checkpoint_sha256'] == parent['protocol']['checkpoint_sha256']
            done.add(seed); summary = aggregate(args.out)
            print('COMPLETED TONIC', seed, 'total', len(summary['completed_seeds']), flush=True)
        remaining = [seed for seed in SEEDS if seed not in done]
        if remaining != last_remaining:
            print('WAITING FOR CALIBRATION', remaining, flush=True); last_remaining = remaining
        if not remaining or not args.watch: break
        time.sleep(args.poll_seconds)
    aggregate(args.out)


if __name__ == '__main__': main()
