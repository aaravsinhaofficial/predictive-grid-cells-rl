"""Simple figures and paired-seed report for exact recurrent-history limits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_lstm_code_swap import estimate, paired, holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/lstm_history_horizon_1B'
VARIANTS = ('full', 'k1', 'k15', 'k60', 'pickup_reset')
CONDITIONS = ('gc100', 'pgc33')


def main():
    runs = [json.loads((OUT/f's{s}.json').read_text()) for s in range(1, 9)]
    assert all(all(r['checks'].values()) for r in runs)
    metrics = ('mean_score', 'post_discovery_goal_rate', 'mean_path_eff', 'mean_first_goal_step')
    estimates, contrasts = {}, {}
    for metric in metrics:
        values = {f'{c}/{v}': np.array([r['variants'][f'{c}/{v}'][metric] for r in runs])
                  for c in CONDITIONS for v in VARIANTS}
        estimates[metric] = {key: estimate(val) for key, val in values.items()}
        gaps = {v: values[f'pgc33/{v}']-values[f'gc100/{v}'] for v in VARIANTS}
        family = {v: paired(gaps['full'], gaps[v]) for v in VARIANTS if v != 'full'}
        holm(family)
        running = 0.
        for j, key in enumerate(sorted(family, key=lambda k: family[k]['exact_signflip_p'])):
            running = max(running, min(1., (len(family)-j)*family[key]['exact_signflip_p']))
            family[key]['holm_exact_p'] = running
        contrasts[metric] = dict(gap_change_vs_full=family,
            population_gap={v: paired(values[f'gc100/{v}'], values[f'pgc33/{v}']) for v in VARIANTS},
            within_model_change={f'{c}/{v}': paired(values[f'{c}/full'], values[f'{c}/{v}'])
                                for c in CONDITIONS for v in VARIANTS if v != 'full'})
    result = dict(estimates=estimates, comparisons=contrasts,
        statistics='Eight paired training seeds. Primary family: four score gap-change contrasts versus full. Holm t and exact sign flips separately. Paired 95% t intervals; other contrasts/metrics exploratory.',
        checks=dict(both_full_baselines_exact=True, all_first_goal_times_identical=True))
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    figure(result)
    report(result)
    print(json.dumps({v: {c: estimates['mean_score'][f'{c}/{v}']['mean'] for c in CONDITIONS} for v in VARIANTS}, indent=2))
    print(json.dumps(contrasts['mean_score']['gap_change_vs_full'], indent=2))


def figure(result):
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3), layout='constrained', gridspec_kw={'width_ratios': [1.5, 1]})
    names = ('k1', 'k15', 'k60', 'full')
    for c, color, label in [('gc100', '#3478ac', 'Grid'), ('pgc33', '#d66b37', 'PGC + grid')]:
        rows = [result['estimates']['mean_score'][f'{c}/{v}'] for v in names]
        axes[0].errorbar(range(4), [r['mean'] for r in rows], yerr=[r['sem'] for r in rows],
                         color=color, marker='o', lw=2.5, capsize=4, label=label)
        rows = [result['estimates']['mean_score'][f'{c}/{v}'] for v in ('full', 'pickup_reset')]
        axes[1].errorbar(range(2), [r['mean'] for r in rows], yerr=[r['sem'] for r in rows],
                         color=color, marker='o', lw=2.5, capsize=4)
    for ax, order in zip(axes, (names, ('full', 'pickup_reset'))):
        ax.plot(range(len(order)), [result['estimates']['mean_score'][f'pgc33/{v}']['values'][6] for v in order],
                color='#956c58', lw=1.5, ls='--', marker='.', label='PGC seed 7')
    axes[0].set(xticks=range(4), xticklabels=['Current\ninput only', 'Last\n1 second', 'Last\n4 seconds', 'Full\nhistory'],
                ylabel='Navigation score', title='How much recent history is needed?')
    axes[1].set(xticks=range(2), xticklabels=['Full\nhistory', 'Reset after\neach pickup'],
                title='Must state survive between trips?')
    ymax = max(r['mean']+r['sem'] for r in result['estimates']['mean_score'].values())*1.12
    for ax in axes: ax.set_ylim(0, ymax)
    axes[0].legend(frameon=False, loc='upper left')
    fig.supxlabel('Native inputs and saved goal remain available. Same discovery path. Eight seed pairs; bars: SEM.', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(OUT/f'history_horizon.{ext}', dpi=180, metadata={'Date': None} if ext=='svg' else None)
    svg = OUT/'history_horizon.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def report(r):
    s = r['estimates']['mean_score']
    gaps = r['comparisons']['mean_score']['population_gap']
    text = ['# What depends on recurrent history?', '',
        '![Navigation with different amounts of carried history](history_horizon.png)', '',
        'We kept the weights, images, live spatial code, saved goal code, and previous action/reward. We changed only how much internal LSTM state could carry forward. Both hidden state **h** and cell state **c** were limited.', '',
        'Every branch followed the same grid controller until finding the first goal. After discovery, each model used its own controller and actual trajectory. This holds goal discovery fixed across both populations and all interventions.', '',
        '| Carried history | Grid | PGC + grid | PGC − grid |', '| --- | ---: | ---: | ---: |']
    labels = {'full':'Full', 'k1':'None: current input only', 'k15':'Last 1 second (15 inputs)',
              'k60':'Last 4 seconds (60 inputs)', 'pickup_reset':'Reset after every goal pickup'}
    for v in VARIANTS:
        text.append(f'| {labels[v]} | {s[f"gc100/{v}"]["mean"]:.2f} | {s[f"pgc33/{v}"]["mean"]:.2f} | {gaps[v]["difference"]:+.2f} |')
    text += ['', '**Models differ.** Most of the absolute navigation score returns with a few seconds of history, but the mean PGC advantage is only partly retained. PGC seed 7 is a substantial exception: its score falls from 85.16 to 65.86 with four seconds of history, and to 66.33 after pickup resets. It is included in every mean and test and shown as a dashed line. Across the other seven pairs, pickup resets change the population gap by between −3.28 and +2.27 points; seed 7 changes it by −19.22. This is a descriptive follow-up, not a replacement for the all-eight-seed tests.', '',
        '**What “current input only” means:** zero the preceding h/c at every decision, then process the current input once. The externally saved goal and previous action/reward remain. It is not an agent with all memory removed.', '',
        '**What the windows mean:** each decision uses exactly the latest 15 or 60 input vectors, replayed from zero. Their original goal messages, actions, rewards, and images are preserved. This is not a periodic reset with a varying history length.', '',
        '**What pickup reset means:** clear h/c after each goal hit and teleport, then allow normal recurrence during the next return. Keep the updated goal message and hit reward.', '',
        '## Does limiting history change the population gap?', '',
        'These are the four predeclared comparisons. A negative change means the PGC advantage shrinks relative to full history. Intervals are paired 95% t intervals across eight seeds; the two p-value columns correct the same four contrasts separately.', '',
        '| Intervention | Change in gap [95% interval] | Holm t p | Holm exact p |', '| --- | ---: | ---: | ---: |']
    for v, d in r['comparisons']['mean_score']['gap_change_vs_full'].items():
        text.append(f'| {labels[v]} | {d["difference"]:+.2f} [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}] | {d["holm_t_p"]:.4g} | {d["holm_exact_p"]:.4g} |')
    text += ['', '## What this can establish', '',
        'The interventions test dependence on carried state in these frozen models. A collapse after zeroing state can reflect loss of useful recent information, an unfamiliar operating state, or both. It does not by itself prove superior memory, special attractors, or a causal role for episode-average PCA offsets. Success after pickup resets would show that state need not persist across return trips under this intervention; the external goal still preserves episode information.', '',
        'The full PGC baseline is the PGC controller **after common grid-controlled discovery**, not native PGC exploration. Both full-history episode records reproduce the existing controller-hybrid baselines exactly, and all first-goal times match across all conditions. The ring-window implementation was checked against independent explicit replay, including asynchronous episode resets and windows longer than the available episode history. Shared-projection gates also matched native LSTMCell updates exactly at recorded inputs from all 16 checkpoints. Checkpoint hashes were unchanged.', '',
        '[Numerical results, including return rate](summary.json) · [Protocol](protocol.json)', '',
        '```sh', 'python scripts/lstm_history_horizon.py', 'python scripts/summarize_lstm_history_horizon.py', '```', '']
    (OUT/'README.md').write_text('\n'.join(text))


if __name__ == '__main__': main()
