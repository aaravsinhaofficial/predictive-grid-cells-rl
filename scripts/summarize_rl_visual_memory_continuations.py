"""Show every fixed endpoint for both continuations, preserving the common pilot."""
import argparse
import copy
import json
from pathlib import Path

import summarize_rl_visual_memory as pilot
from audit_rl_visual_memory_continuations import NAMES

ROOT = Path(__file__).resolve().parents[1]


def evaluation_summary(result, baseline, frames, block):
    assert result['episodes_per_variant'] == 64 and result['max_blind_gap'] == 60
    assert result['all_first_discoveries_identical']
    scores = {key: value['mean_score'] for key, value in result['variants'].items()}
    returns = {key: value['mean_returns_per_episode'] for key, value in result['variants'].items()}
    ratios = dict(blind_seen_return_ratio=returns['blind'] / max(returns['seen'], 1e-12),
                  blind_original_seen_return_ratio=returns['blind'] / max(baseline['returns']['seen'], 1e-12),
                  blind_seen_score_ratio=scores['blind'] / max(scores['seen'], 1e-12),
                  blind_original_seen_score_ratio=scores['blind'] / max(baseline['scores']['seen'], 1e-12),
                  seen_original_seen_score_ratio=scores['seen'] / max(baseline['scores']['seen'], 1e-12))
    return dict(block=block, learning_frames=frames, scores=scores, returns=returns,
                contrasts=result['descriptive_contrasts'],
                blind_exposure_fraction=result['blind_fraction_of_post_discovery_updates'],
                state_reset_counts=result['state_reset_counts'], **ratios)


def collect(arm, condition, original, require_complete):
    folder = ROOT / 'results/ring_cloud_artifacts' / NAMES[arm][condition]
    stages = copy.deepcopy(original['stages'])
    baseline = stages[0]
    blocks = []
    for block in range(1, 6):
        path = folder / f'block{block}_evaluation.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if condition == 'gc100':
            for a, b in [('blind', 'blind_current_grid'), ('seen', 'seen_current_grid')]:
                assert result['variants'][a]['rows'] == result['variants'][b]['rows']
        row = evaluation_summary(result, baseline, 25_024_000 + block * 10_009_600, block)
        row['phase'] = arm
        stages.append(row); blocks.append(block)
    complete = len(blocks) == 5 and (folder / 'summary.json').exists()
    if complete:
        complete = json.loads((folder / 'summary.json').read_text())['completed']
        if arm == 'noise': complete = complete and (folder / 'final_noisy_evaluation.json').exists()
    if require_complete: assert complete, f'Incomplete fixed continuation: {arm}/{condition}'
    incoming = folder / 'baseline.json'
    if incoming.exists():
        result = json.loads(incoming.read_text())
        assert {key: value['mean_score'] for key, value in result['variants'].items()} == original['stages'][-1]['scores']
    endpoints = {}
    for label, filename in [('baseline', 'baseline_noisy.json'), ('final', 'final_noisy_evaluation.json')]:
        path = folder / filename
        if path.exists():
            result = json.loads(path.read_text())
            endpoints[label] = evaluation_summary(result, baseline,
                                                  25_024_000 if label == 'baseline' else 75_072_000, label)
            endpoints[label]['recurrent_noise'] = result['recurrent_noise']
    return dict(arm=arm, condition=condition, completed=complete, stages=stages, noisy_endpoints=endpoints,
                inference='One preselected original training-seed pair; all fixed endpoints retained.')


def figures(data, out, partial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'blind': ('#d86b2a', '-', 'Blackouts'), 'blind_reset': ('#d86b2a', '--', 'Blackouts + reset'),
              'seen': ('#347bb5', '-', 'Vision'), 'seen_reset': ('#347bb5', '--', 'Vision + reset')}
    for arm, pairs in data.items():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True, layout='constrained')
        for ax, (condition, item) in zip(axes, pairs.items()):
            rows = item['stages']; x = [row['learning_frames'] / 1e6 for row in rows]
            for key, (color, line, label) in styles.items():
                ax.plot(x, [row['scores'][key] for row in rows], marker='o', color=color, ls=line, label=label)
            ax.axhline(rows[0]['scores']['seen'], color='#777777', ls=':', lw=1.2, label='Original vision score')
            ax.axvline(25.024, color='#999999', lw=1)
            ax.set(title='Grid' if condition == 'gc100' else 'PGC + grid', xlabel='Additional RL frames (millions)')
            ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15)
        axes[0].set_ylabel('Navigation score'); axes[1].legend(frameon=False, fontsize=8)
        title = 'Continuation without state noise' if arm == 'noiseless' else 'Continuation with state noise'
        fig.suptitle(title + (' (in progress)' if partial else ''))
        fig.supxlabel('Vertical line: fixed continuation begins · same clean evaluation throughout · one original seed pair', fontsize=9)
        fig.savefig(out / f'{arm}_continuation_learning.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout='constrained')
    for arm, line in [('noiseless', '-'), ('noise', '--')]:
        for condition, color in [('gc100', '#347bb5'), ('pgc33', '#d86b2a')]:
            rows = data[arm][condition]['stages'][5:]
            x = [row['learning_frames'] / 1e6 for row in rows]
            label = f"{'Grid' if condition == 'gc100' else 'PGC + grid'}, {'no noise' if arm == 'noiseless' else 'noise'}"
            axes[0].plot(x, [r['contrasts']['additional_reset_cost_in_darkness'] for r in rows],
                         color=color, ls=line, marker='o', label=label)
            axes[2].plot(x, [100 * r['blind_exposure_fraction']['blind'] for r in rows],
                         color=color, ls=line, marker='o', label=label)
        rows = data[arm]['pgc33']['stages'][5:]; x = [r['learning_frames'] / 1e6 for r in rows]
        for key, color, label in [('blind_current_prediction_benefit', '#d86b2a', 'Blackouts'),
                                   ('seen_current_prediction_benefit', '#347bb5', 'Vision')]:
            axes[1].plot(x, [r['contrasts'][key] for r in rows], color=color, ls=line, marker='o',
                         label=f"{label}, {'no noise' if arm == 'noiseless' else 'noise'}")
    axes[0].set(title='Additional memory benefit in darkness', ylabel='Score difference')
    axes[1].set(title='PGC current-input benefit', ylabel='Score difference')
    axes[2].set(title='Actual blackout exposure', ylabel='Post-discovery updates (%)')
    for ax in axes:
        ax.set_xlabel('Additional RL frames (millions)'); ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.legend(frameon=False, fontsize=8)
    for ax in axes[:2]: ax.axhline(0, color='#777777', lw=1)
    if partial: fig.suptitle('Continuation checks (in progress)')
    fig.supxlabel('Noise labels describe training; every curve uses the same clean evaluation', fontsize=9)
    fig.savefig(out / 'continuation_checks.png', dpi=180, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    originals = {condition: pilot.collect(condition, name, True) for condition, name in pilot.NAMES.items()}
    data = {arm: {condition: collect(arm, condition, originals[condition], not args.partial) for condition in pairs}
            for arm, pairs in NAMES.items()}
    out = ROOT / 'results/ring_visual_memory'
    (out / 'continuation_summary.json').write_text(json.dumps(data, indent=2) + '\n')
    figures(data, out, args.partial)
    print(json.dumps({arm: {condition: len(item['stages']) for condition, item in pairs.items()}
                      for arm, pairs in data.items()}))


if __name__ == '__main__':
    main()
