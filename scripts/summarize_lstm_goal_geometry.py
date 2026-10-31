"""Summarize paired frozen-model goal-snapshot geometry interventions.

Run only after all sixteen model files exist. Each training seed, rather than
each episode or pickup, is one statistical replicate. The mirror tolerance of
five score points is a coarse, pre-run diagnostic, not a claim that strategies
or hidden states are equivalent.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy import stats

from summarize_lstm_code_swap import estimate, paired, holm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/lstm_goal_geometry_1B'
SEEDS = tuple(range(1, 9))
VARIANTS = {
    'pgc33': ('intact', 'plain_goal', 'center_pgc_goal', 'mirror_goal', 'reverse_goal'),
    'gc100': ('intact', 'lookahead_goal', 'center_goal'),
}
LABELS = {
    'pgc33/intact': 'Original predictive goal',
    'pgc33/plain_goal': 'Plain goal at pickup',
    'pgc33/center_pgc_goal': 'Predictive slots at center (oracle)',
    'pgc33/mirror_goal': 'Reflected arrival direction (oracle)',
    'pgc33/reverse_goal': 'Reversed arrival direction',
    'gc100/intact': 'Original grid goal at pickup',
    'gc100/lookahead_goal': 'All goal slots shifted ahead',
    'gc100/center_goal': 'All goal slots at center (oracle)',
}
# Every difference is second minus first. Keep this same ten-contrast family
# for the primary score outcome and each separately labeled exploratory one.
CONTRASTS = (
    ('pgc_plain_minus_intact', 'pgc33/intact', 'pgc33/plain_goal', 'PGC: plain − original'),
    ('pgc_center_minus_intact', 'pgc33/intact', 'pgc33/center_pgc_goal', 'PGC: center − original'),
    ('pgc_mirror_minus_intact', 'pgc33/intact', 'pgc33/mirror_goal', 'PGC: mirror − original'),
    ('pgc_reverse_minus_intact', 'pgc33/intact', 'pgc33/reverse_goal', 'PGC: reverse − original'),
    ('pgc_mirror_minus_plain', 'pgc33/plain_goal', 'pgc33/mirror_goal', 'PGC: mirror − plain'),
    ('pgc_center_minus_plain', 'pgc33/plain_goal', 'pgc33/center_pgc_goal', 'PGC: center − plain'),
    ('grid_lookahead_minus_intact', 'gc100/intact', 'gc100/lookahead_goal', 'Grid: ahead − original'),
    ('grid_center_minus_intact', 'gc100/intact', 'gc100/center_goal', 'Grid: center − original'),
    ('native_pgc_minus_native_grid', 'gc100/intact', 'pgc33/intact', 'Original PGC − original grid'),
    ('grid_lookahead_minus_native_pgc', 'pgc33/intact', 'gc100/lookahead_goal', 'Grid ahead − original PGC'),
)
METRICS = ('mean_score', 'mean_path_eff', 'post_discovery_goal_rate',
           'mean_first_goal_step', 'frac_found_goal')
MIRROR_SCORE_TOLERANCE = 5.0
GEOMETRY_METRICS = ('hit_distance', 'forward_distance', 'mirror_distance',
                    'reverse_distance', 'reflection_angle_degrees',
                    'predictive_point_separation')


def holm_signflips(records):
    running = 0.0
    for i, key in enumerate(sorted(records, key=lambda k: records[k]['exact_signflip_p'])):
        running = max(running, min(1.0, (len(records) - i) * records[key]['exact_signflip_p']))
        records[key]['holm_signflip_p'] = running


def mirror_tolerance_test(differences):
    """Unadjusted paired TOST and corresponding 90% CI for ±5 score points."""
    d = np.asarray(differences, dtype=float)
    mean, sem = float(d.mean()), float(stats.sem(d))
    margin, df = MIRROR_SCORE_TOLERANCE, len(d) - 1
    if sem > 0:
        ci = stats.t.interval(.90, df, loc=mean, scale=sem)
        lower_p = float(stats.t.sf((mean + margin) / sem, df))
        upper_p = float(stats.t.cdf((mean - margin) / sem, df))
    else:
        ci = (mean, mean)
        lower_p = 0.0 if mean > -margin else (0.5 if mean == -margin else 1.0)
        upper_p = 0.0 if mean < margin else (0.5 if mean == margin else 1.0)
    within = bool(ci[0] > -margin and ci[1] < margin)
    return dict(difference_mirror_minus_intact=mean, paired_differences=d.tolist(),
                ci90=list(map(float, ci)), tolerance_points=margin,
                lower_one_sided_p=lower_p, upper_one_sided_p=upper_p,
                tost_p=max(lower_p, upper_p), ci90_within_tolerance=within,
                interpretation=('The 90% paired interval is entirely within ±5 score points.' if within
                                else 'The 90% paired interval is not entirely within ±5 score points.'),
                scope='Pre-run coarse score tolerance; unadjusted separate diagnostic. This does not establish identical mechanisms, directional information, or recurrent states.')


def verify_runs(runs, episode_len=1350):
    first_checks = 0
    max_geometry_error = 0.0
    frames = set()
    for (condition, seed), run in runs.items():
        assert (run['condition'], run['seed']) == (condition, seed)
        assert (run['episodes'], run['eval_seed'], run['n_envs']) == (128, 737373, 64)
        assert run['checkpoint_sha256'], f'Missing checkpoint hash: {condition}, {seed}'
        frames.add(run['checkpoint_frames'])
        assert set(run['variants']) == set(VARIANTS[condition])
        baseline = None
        for variant in VARIANTS[condition]:
            summary = run['variants'][variant]
            rows = summary['rows']
            assert summary['episodes'] == len(rows) == 128
            keyed = {(r['slot'], r['episode']): r for r in rows}
            assert len(keyed) == len(rows), f'Duplicate episode: {condition}, {seed}, {variant}'
            first = {key: row['first_goal_t'] for key, row in keyed.items()}
            if baseline is None:
                baseline = first
            else:
                assert first == baseline, f'Goal-only intervention altered first pickup: {condition}, {seed}, {variant}'
                first_checks += len(rows)
            assert np.isclose(summary['mean_score'], np.mean([r['ret'] for r in rows]), atol=1e-9)
            first_values = [r['first_goal_t'] if r['first_goal_t'] >= 0 else episode_len for r in rows]
            assert np.isclose(summary['mean_first_goal_step'], np.mean(first_values), atol=1e-9)
            assert np.isclose(summary['frac_found_goal'], np.mean([r['first_goal_t'] >= 0 for r in rows]), atol=1e-9)
            assert all(np.isfinite(summary[m]) for m in METRICS), f'Nonfinite outcome: {condition}, {seed}, {variant}'
            geometry = run['geometry'][variant]
            assert geometry['n'] > 0, f'No pickup geometry: {condition}, {seed}, {variant}'
            assert all(np.isfinite(geometry[m]) for m in GEOMETRY_METRICS)
            error = abs(geometry['forward_distance'] - geometry['mirror_distance'])
            max_geometry_error = max(max_geometry_error, error)
            assert error < 1e-5, f'Reflection changed goal-center distance: {condition}, {seed}, {variant}'
    assert len(frames) == 1, f'Checkpoint frame budgets differ: {frames}'
    return dict(first_goal_episode_checks=first_checks, first_goal_invariance_exact=True,
                max_mean_forward_mirror_distance_error_m=max_geometry_error,
                checkpoint_frames=next(iter(frames)), complete_models=len(runs),
                note='First-goal times and found-goal indicators match exactly across every goal-only variant within each model.')


def aggregate(runs, protocol):
    validation = verify_runs(runs, protocol.get('episode_len', 1350))
    estimates, comparisons = {}, {}
    for metric in METRICS:
        values = {f'{condition}/{variant}': [runs[condition, s]['variants'][variant][metric] for s in SEEDS]
                  for condition, variants in VARIANTS.items() for variant in variants}
        estimates[metric] = {name: estimate(v) for name, v in values.items()}
        family = {}
        for name, a, b, label in CONTRASTS:
            family[name] = paired(values[a], values[b])
            family[name].update(reference=a, intervention=b, label=label)
        holm(family)
        holm_signflips(family)
        comparisons[metric] = family
    geometry = {}
    for condition, variants in VARIANTS.items():
        for variant in variants:
            records = [runs[condition, s]['geometry'][variant] for s in SEEDS]
            counts = np.array([r['n'] for r in records])
            fields = set.intersection(*(set(r) for r in records)) - {'n'}
            numeric = [m for m in sorted(fields) if all(isinstance(r[m], (int, float)) for r in records)]
            geometry[f'{condition}/{variant}'] = dict(
                per_seed_hit_counts=counts.tolist(), total_hits=int(counts.sum()),
                seed_mean={m: estimate([r[m] for r in records]) for m in numeric},
                pooled_hit_mean={m: float(np.average([r[m] for r in records], weights=counts)) for m in numeric})
    tolerance = mirror_tolerance_test(comparisons['mean_score']['pgc_mirror_minus_intact']['differences'])
    return dict(protocol=protocol, validation=validation,
                checkpoints={f'{c}/s{s}': r['checkpoint_sha256'] for (c, s), r in runs.items()},
                estimates=estimates, comparisons=comparisons, geometry=geometry,
                mirror_score_tolerance=tolerance,
                statistics='Eight paired training-seed means are the replicates. Primary outcome: episode score; ten planned contrasts with paired 95% t intervals, paired t tests, and exact two-sided sign flips. Holm adjustments are supplied separately for the t and sign-flip p values within the ten score contrasts. Path efficiency and post-discovery goal rate are exploratory, each with the same separately corrected family. First-goal outcomes are validation/descriptive. Intervals themselves are unadjusted. Pickup-weighted geometry is descriptive and does not treat hits as independent statistical replicates.',
                limitations=[
                    'Reflection preserves distance to the true goal center and shift length, but only reverses the tangential component of arrival direction.',
                    'A small reflection angle is a weak directional perturbation; a null effect does not rule out directional information use.',
                    'Equal distance to the goal center does not match wall proximity, grid-map responses, or routes from later spawn locations.',
                    'Center substitutions and the reflection use the hidden true goal location and are oracle diagnostics.',
                    'Current population code and frozen network weights stay intact; only the stored goal snapshot changes after pickup.',
                    'Closed-loop variants develop their own trajectories and later pickup snapshots, so geometry differences across variants can be consequences as well as causes of behavior.',
                ])


def score_panel(ax, result, condition, title, color):
    variants = VARIANTS[condition]
    scores = result['estimates']['mean_score']
    for i, variant in enumerate(variants):
        key = f'{condition}/{variant}'
        record = scores[key]
        ax.scatter(record['values'], i + np.linspace(-.13, .13, len(SEEDS)), s=22, color=color, alpha=.5)
        ax.errorbar(record['mean'], i, xerr=record['sem'], color=color, marker='o', ms=7, capsize=4, lw=2)
    ax.axvline(scores[f'{condition}/intact']['mean'], color=color, ls=':', lw=1)
    ax.set(yticks=np.arange(len(variants)), yticklabels=[LABELS[f'{condition}/{v}'] for v in variants],
           ylim=(len(variants) - .5, -.5), xlabel='Score per episode (mean ± SEM)', title=title)


def geometry_panel(ax):
    center = np.zeros(2)
    hit = np.array([-.28, .06])
    direction = np.array([np.cos(.4), np.sin(.4)])
    radial = (center - hit) / np.linalg.norm(center - hit)
    reflected = 2 * np.dot(direction, radial) * radial - direction
    forward, mirror, reverse = (hit + .2 * direction, hit + .2 * reflected, hit - .2 * direction)
    ax.add_patch(Circle(center, .3, facecolor='#f0f5f1', edgecolor='#8caa95', lw=1.5))
    ax.plot([hit[0], center[0]], [hit[1], center[1]], color='#9ca3a9', ls='--', lw=1)
    for point in (forward, mirror):
        ax.plot([point[0], 0], [point[1], 0], color='#7d858a', ls=':', lw=1)
    for point, color in [(forward, '#d66b37'), (mirror, '#9564aa'), (reverse, '#92989c')]:
        ax.annotate('', xy=point, xytext=hit, arrowprops=dict(arrowstyle='->', lw=2, color=color))
    for point, label, offset, color in [
        (hit, 'Pickup', (-42, 7), '#263a45'),
        (center, 'True center\n(oracle)', (9, -8), '#31815c'),
        (forward, 'Original ahead', (6, 6), '#d66b37'),
        (mirror, 'Reflected ahead', (-15, -24), '#9564aa'),
        (reverse, 'Reversed', (-14, -20), '#737b80'),
    ]:
        ax.scatter(*point, color=color, s=45, zorder=5)
        ax.annotate(label, point, xytext=offset, textcoords='offset points', fontsize=9, color=color)
    ax.text(0, .335, 'Reward region (30 cm radius)', ha='center', fontsize=10, color='#466f51')
    ax.text(-.49, -.37, 'Original and reflected points:\nsame 20 cm shift; same distance to center.\nOnly the sideways arrival component flips.', fontsize=10, va='top')
    ax.set(xlim=(-.54, .39), ylim=(-.52, .41), aspect='equal', title='D  |  What the reflection controls')
    ax.axis('off')


def figure(result, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout='constrained',
                             gridspec_kw=dict(height_ratios=[1, 1.55]))
    score_panel(axes[0, 0], result, 'pgc33', 'A  |  PGC-trained networks', '#d66b37')
    score_panel(axes[0, 1], result, 'gc100', 'B  |  Grid-trained networks', '#3478ac')
    ax = axes[1, 0]
    rows = result['comparisons']['mean_score']
    for i, (name, _, _, _) in enumerate(CONTRASTS):
        r = rows[name]
        color = '#d66b37' if name.startswith('pgc_') else '#3478ac'
        err = np.array([[r['difference'] - r['ci95'][0]], [r['ci95'][1] - r['difference']]])
        ax.errorbar(r['difference'], i, xerr=np.maximum(err, 0), color=color, fmt='o', ms=6, capsize=3)
    ax.axvline(0, color='#7f868c', lw=1)
    ax.set(yticks=np.arange(len(CONTRASTS)), yticklabels=[c[3] for c in CONTRASTS],
           ylim=(len(CONTRASTS) - .5, -.5), xlabel='Paired score difference (95% t interval)',
           title='C  |  Ten planned score comparisons')
    geometry_panel(axes[1, 1])
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle('Does the predictive goal snapshot provide a better place to aim for?', fontsize=16)
    fig.supxlabel('Frozen 1B models; 8 paired training seeds × 128 episodes per variant. Dots in A/B: seed means.\nOnly goal snapshots change; current code stays intact. Center and reflection controls use privileged goal coordinates. Diagram is illustrative.', fontsize=10)
    fig.savefig(out / 'goal_geometry.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / 'goal_geometry.pdf', bbox_inches='tight')
    plt.close(fig)


def print_report(result):
    for metric, unit in [('mean_score', 'points'), ('mean_path_eff', 'efficiency'),
                         ('post_discovery_goal_rate', 'goals / 1,000 post-discovery steps')]:
        print(f'\n{metric} ({"PRIMARY" if metric == "mean_score" else "EXPLORATORY"}):')
        for key, r in result['estimates'][metric].items():
            print(f'  {key}: {r["mean"]:.4f} ± {r["sem"]:.4f} SEM {unit}')
        for key, r in result['comparisons'][metric].items():
            print(f'  {key}: Δ={r["difference"]:+.4f}, 95% CI [{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}]; '
                  f't p={r["paired_t_p"]:.6g}, Holm={r["holm_t_p"]:.6g}; '
                  f'exact sign-flip p={r["exact_signflip_p"]:.6g}, Holm={r["holm_signflip_p"]:.6g}')
    r = result['mirror_score_tolerance']
    print(f'\nMirror − original: {r["difference_mirror_minus_intact"]:+.4f} points, '
          f'90% paired CI [{r["ci90"][0]:+.4f}, {r["ci90"][1]:+.4f}], '
          f'pre-run tolerance ±{r["tolerance_points"]:.1f}, TOST p={r["tost_p"]:.6g}.')
    print(r['interpretation'])
    print(r['scope'])
    print('\nPickup geometry (mean of seed means; meters except angle):')
    for key, r in result['geometry'].items():
        print(f'  {key}: ' + ', '.join(f'{name}={r["seed_mean"][name]["mean"]:.4f}' for name in GEOMETRY_METRICS))
    print('\nValidation:', json.dumps(result['validation']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    paths = {(c, s): args.out / f'{c}_s{s}.json' for c in VARIANTS for s in SEEDS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        parser.error('All sixteen model files are required before aggregation. Missing: ' + ', '.join(missing))
    runs = {key: json.loads(path.read_text()) for key, path in paths.items()}
    protocol_path = args.out / 'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else dict(
        eval_seed=737373, episodes_per_seed_variant=128, n_envs=64, episode_len=1350,
        note='No protocol.json was found; these are the validated run settings.')
    result = aggregate(runs, protocol)
    # The unmodified and plain-goal branches repeat earlier conditions. Check
    # complete episode records, not just rounded group averages.
    reference_dir = ROOT / 'results/lstm_code_swap_1B'
    checked = 0
    if all((reference_dir / path.name).exists() for path in paths.values()):
        for key, run in runs.items():
            old = json.loads((reference_dir / paths[key].name).read_text())
            assert run['checkpoint_sha256'] == old['checkpoint_sha256']
            pairs = [('intact', 'intact')]
            if key[0] == 'pgc33':
                pairs.append(('plain_goal', 'goal_grid'))
            for new_name, old_name in pairs:
                assert run['variants'][new_name]['rows'] == old['variants'][old_name]['rows'], (key, new_name)
                checked += len(run['variants'][new_name]['rows'])
    result['validation']['prior_baseline_episode_records_checked'] = checked
    result['validation']['prior_baselines_reproduce_exactly'] = checked == 8 * 3 * 128
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    figure(result, args.out)
    print_report(result)


if __name__ == '__main__':
    main()
