"""Aggregate frozen-code-swap behavior and causal RNN interventions by seed."""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/lstm_code_swap_1B'
SEEDS = range(1, 9)
SWAPS = ('current_grid', 'goal_grid', 'both_grid')
LABELS = ['Current only', 'Goal only', 'Both']
COLORS = ['#3b82b1', '#9c6fa9', '#db7545']
HORIZONS = (0, 1, 5, 15, 30, 60)


def estimate(values):
    v = np.asarray(values, float)
    return dict(values=v.tolist(), n=len(v), mean=float(v.mean()), sem=float(stats.sem(v)))


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    permutations = np.array([np.mean(d * signs) for signs in itertools.product((-1, 1), repeat=len(d))])
    ci = stats.t.interval(.95, len(d)-1, loc=d.mean(), scale=stats.sem(d)) if np.std(d) > 0 else (d.mean(), d.mean())
    return dict(n=len(d), a=a.tolist(), b=b.tolist(), mean_a=float(a.mean()), mean_b=float(b.mean()),
                difference=float(d.mean()), differences=d.tolist(), ci95=list(map(float, ci)),
                paired_t_p=float(stats.ttest_rel(b, a).pvalue) if np.std(d) > 0 else (1. if d[0] == 0 else 0.),
                exact_signflip_p=float(np.mean(np.abs(permutations) >= abs(d.mean()) - 1e-12)))


def holm(records):
    running = 0.
    for i, key in enumerate(sorted(records, key=lambda k: records[k]['paired_t_p'])):
        running = max(running, min(1., (len(records)-i)*records[key]['paired_t_p']))
        records[key]['holm_t_p'] = running


def main():
    runs = {(c, s): json.loads((OUT / f'{c}_s{s}.json').read_text()) for c in ('pgc33', 'gc100') for s in SEEDS}
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for r in runs.values():
        assert r['episodes'] == protocol['episodes_per_seed_variant']
        assert r['eval_seed'] == protocol['eval_seed']
        assert all(v['episodes'] == r['episodes'] for v in r['variants'].values())
    metrics = ('mean_score', 'mean_path_eff', 'post_discovery_goal_rate', 'mean_first_goal_step', 'frac_found_goal')
    estimates, comparisons = {}, {}
    for metric in metrics:
        values = {(c, variant): [runs[c, s]['variants'][variant][metric] for s in SEEDS]
                  for c in ('pgc33', 'gc100') for variant in runs[c, 1]['variants']}
        estimates[metric] = {f'{c}/{v}': estimate(a) for (c, v), a in values.items()}
        family = {f'pgc_{v}_minus_intact': paired(values['pgc33', 'intact'], values['pgc33', v]) for v in SWAPS}
        family['grid_mixed_minus_intact'] = paired(values['gc100', 'intact'], values['gc100', 'both_mixed'])
        family['native_pgc_minus_native_grid'] = paired(values['gc100', 'intact'], values['pgc33', 'intact'])
        family['pgc_with_grid_minus_native_grid'] = paired(values['gc100', 'intact'], values['pgc33', 'both_grid'])
        family['goal_swap_minus_current_swap'] = paired(values['pgc33', 'current_grid'], values['pgc33', 'goal_grid'])
        # Positive interaction means the combined loss is smaller than the sum
        # of the two isolated losses (both use grid, removing a code mismatch).
        interaction = (np.array(values['pgc33', 'both_grid']) - values['pgc33', 'current_grid']
                       - np.array(values['pgc33', 'goal_grid']) + values['pgc33', 'intact'])
        family['current_goal_interaction'] = paired(np.zeros(8), interaction)
        holm(family)
        comparisons[metric] = family
    effects = {}
    for key in runs['pgc33', 1]['rnn_effects']:
        keys = set.intersection(*(set(runs['pgc33', s]['rnn_effects'][key]) for s in SEEDS))
        effects[key] = {m: estimate([runs['pgc33', s]['rnn_effects'][key][m] for s in SEEDS]) for m in keys if m != 'n'}
        effects[key]['sample_counts'] = [runs['pgc33', s]['rnn_effects'][key]['n'] for s in SEEDS]
    for variant in SWAPS:
        for gate in ('input', 'forget', 'candidate', 'output'):
            reductions = []
            for s in SEEDS:
                r = runs['pgc33', s]['rnn_effects'][f'immediate/{variant}']
                reductions.append(1-r[f'restore_{gate}_action_tv']/r['action_tv'])
            effects[f'immediate/{variant}'][f'restore_{gate}_tv_reduction_fraction'] = estimate(reductions)
        for horizon in HORIZONS:
            for metric in ('h_rms', 'c_rms', 'action_tv'):
                ratio = [runs['pgc33', s]['rnn_effects'][f'pulse/{variant}/{horizon}'][metric] /
                         runs['pgc33', s]['rnn_effects'][f'pulse/{variant}/0'][metric] for s in SEEDS]
                effects[f'pulse/{variant}/{horizon}'][f'{metric}_fraction_of_initial'] = estimate(ratio)
        for s in SEEDS:
            counts = [runs['pgc33', s]['rnn_effects'][f'pulse/{variant}/{h}']['n'] for h in HORIZONS]
            assert len(set(counts)) == 1, f'Incomplete pulse cohorts: seed {s}, {variant}, {counts}'
    result = dict(protocol=protocol, estimates=estimates, comparisons=comparisons, rnn_effects=effects,
                  gate_reconstruction_max_error=max(r['gate_reconstruction_max_error'] for r in runs.values()),
                  statistics='Each training seed is one replicate. Paired t, t interval, exact two-sided sign flips. Holm correction within each eight-comparison behavioral metric family. RNN effects summarize per-seed means; samples are not treated as independent replicates.')
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    figure(result)
    for metric in ('mean_score', 'mean_path_eff', 'post_discovery_goal_rate', 'mean_first_goal_step'):
        print(metric)
        for key, r in estimates[metric].items():
            print(f'  {key}: {r["mean"]:.4f} +/- {r["sem"]:.4f}')
    for variant in SWAPS:
        direct = effects[f'immediate/{variant}']
        sustained = effects[f'sustained/{variant}']
        print(variant, 'immediate/sustained TV:', direct['action_tv']['mean'], sustained['action_tv']['mean'])
        print('  Gate TV reductions:', {g:direct[f'restore_{g}_tv_reduction_fraction']['mean'] for g in ('input','forget','candidate','output')})
        print('  Pulse TV:', {h:effects[f'pulse/{variant}/{h}']['action_tv']['mean'] for h in HORIZONS})


def figure(result):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout='constrained')
    ax = axes[0, 0]
    scores = result['estimates']['mean_score']
    for c, color, label in [('gc100', '#3478ac', 'Grid-trained'), ('pgc33', '#d66b37', 'PGC-trained')]:
        names = ['gc100/intact', 'gc100/both_mixed'] if c == 'gc100' else ['pgc33/both_grid', 'pgc33/intact']
        vals = np.array([scores[name]['values'] for name in names])
        for seed in range(8):
            ax.plot([0, 1], vals[:, seed], color=color, alpha=.15, lw=1)
        ax.errorbar([0, 1], vals.mean(1), yerr=stats.sem(vals, axis=1), color=color, marker='o', lw=2.5, capsize=4, label=label)
    ax.set(xticks=[0, 1], xticklabels=['Grid code at evaluation', 'Mixed code at evaluation'],
           ylabel='Score / episode', title='A  |  Same weights, different population code')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    names = ['intact', *SWAPS]
    vals = np.array([scores[f'pgc33/{name}']['values'] for name in names])
    for seed in range(8):
        ax.plot(np.arange(4), vals[:, seed], color='#9aa2ab', alpha=.35, lw=1)
    ax.errorbar(np.arange(4), vals.mean(1), yerr=stats.sem(vals, axis=1), color='#d66b37', marker='o', lw=2.5, capsize=4)
    ax.set(xticks=np.arange(4), xticklabels=['Intact', 'Current\ncode only', 'Goal\ncode only', 'Both'],
           ylabel='Score / episode', title='B  |  Which PGC input is replaced by grid code?')
    effects = result['rnn_effects']
    ax = axes[1, 0]
    for i, (kind, color, label) in enumerate([('immediate', '#698da1', 'One step, same preceding state'),
                                             ('sustained', '#d28b5f', 'Sustained swap, same experience')]):
        rows = [effects[f'{kind}/{v}']['action_tv'] for v in SWAPS]
        ax.bar(np.arange(3)+(i-.5)*.36, [r['mean'] for r in rows], .34,
               yerr=[r['sem'] for r in rows], color=color, capsize=3, label=label)
    ax.set(xticks=np.arange(3), xticklabels=LABELS, ylabel='Action-distribution change (total variation)',
           title='C  |  Internal effect before behavioral feedback')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    gates = ('input', 'forget', 'candidate', 'output')
    for i, (variant, color, label) in enumerate(zip(SWAPS, COLORS, LABELS)):
        rows = [effects[f'immediate/{variant}'][f'restore_{g}_tv_reduction_fraction'] for g in gates]
        ax.bar(np.arange(4)+(i-1)*.25, [100*r['mean'] for r in rows], .24,
               yerr=[100*r['sem'] for r in rows], color=color, capsize=2, label=label)
    ax.axhline(0, color='#888888', lw=1)
    ax.set(xticks=np.arange(4), xticklabels=['Input gate', 'Forget gate', 'Candidate\ncell update', 'Output gate'],
           ylabel='Action change removed by restoring component (%)', title='D  |  Causal restoration of one LSTM component')
    ax.legend(fontsize=8)
    ax = axes[2, 0]
    for variant, color, label in zip(SWAPS, COLORS, LABELS):
        rows = [effects[f'pulse/{variant}/{h}']['action_tv'] for h in HORIZONS]
        mean, sem = np.array([r['mean'] for r in rows]), np.array([r['sem'] for r in rows])
        x = np.array(HORIZONS)/15
        ax.plot(x, mean, color=color, marker='o', label=label)
        ax.fill_between(x, mean-sem, mean+sem, color=color, alpha=.15)
    ax.set(xlabel='Seconds after a single altered-code step', ylabel='Action-distribution change (total variation)',
           title='E  |  Persistence after intact input is restored')
    ax.legend(fontsize=9)
    ax = axes[2, 1]
    vals = np.array([result['estimates']['mean_path_eff'][f'pgc33/{name}']['values'] for name in names])
    for seed in range(8):
        ax.plot(np.arange(4), vals[:, seed], color='#9aa2ab', alpha=.35, lw=1)
    ax.errorbar(np.arange(4), vals.mean(1), yerr=stats.sem(vals, axis=1), color='#d66b37', marker='o', lw=2.5, capsize=4)
    ax.set(xticks=np.arange(4), xticklabels=['Intact', 'Current\ncode only', 'Goal\ncode only', 'Both'],
           ylabel='Return-path efficiency', title='F  |  Effect on homing paths')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.title.set_fontsize(11)
    fig.suptitle('Frozen 1B LSTMs: population-code swaps and causal state interventions', fontsize=15)
    fig.supxlabel('Eight matched training seeds; 128 fresh episodes per condition per seed. Error bars / bands: SEM across seeds.\nRNN forks share intact experience; gate-restoration effects interact and need not add to 100%.', fontsize=9)
    fig.savefig(OUT / 'code_swap_and_dynamics.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'code_swap_and_dynamics.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
