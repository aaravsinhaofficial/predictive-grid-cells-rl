"""Display every fixed 90M endpoint; never refit a representation or change gates.

This presentation layer runs only after the independent artifact audit and all
frozen primary, causal, observable and actor analyses are complete.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/lstm_reorientation_entropy_mechanism'
OUT = BASE / 'report'
ARMS = ('original', 'explore')
CONDITIONS = ('gc100', 'pgc33')
CUES = ('native', 'ordinary', 'plus60', 'minus60')


def main():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    audit = read(BASE / 'independent_artifact_audit.json')
    observable = read(ROOT / 'results/lstm_reorientation_entropy_observable/summary.json')
    actor = read(ROOT / 'results/lstm_reorientation_entropy_actor_readout/summary.json')
    behavior = read(ROOT / 'results/rl_blind_reorientation_entropy/summary.json')
    assert audit['passed'] and audit['complete']
    assert observable['all64_checks_complete'] and len(observable['rows']) == 64
    assert actor['complete'] and actor['all128rows_retained'] and len(actor['rows']) == 128
    rows = []
    final_behavior = []
    for arm in ARMS:
        for condition in CONDITIONS:
            model = behavior['conditions'][condition][arm]
            assert model['completed'] and len(model['endpoints']) == 6
            endpoint = model['endpoints'][-1]
            assert endpoint['block'] == 5 and endpoint['total_extra_frames'] == 90_132_480
            final_behavior.append(dict(arm=arm, condition=condition, endpoint=endpoint))
            for context in (0, 1):
                folder = BASE / arm / condition
                primary = read(folder / f'context{context}.json')
                causal_path = folder / 'causal' / f'context{context}' / 'result.json'
                causal = read(causal_path)
                model_audit = audit['models'][arm][condition]
                causal_audit = next(r for r in model_audit['causal'] if r['context'] == context)
                primary_audit = next(r for r in model_audit['primary'] if r['context'] == context)
                observed = read(ROOT / 'results/lstm_reorientation_entropy_observable' /
                    arm / condition / f'context{context}.json')
                assert primary['checkpoint_sha256'] == model_audit['checkpoint_sha256'] == observed['checkpoint_sha256']
                assert observed['archive_sha256'] == primary_audit['archive_sha256']
                assert sources[str(causal_path.relative_to(ROOT))] == causal_audit['result_sha256']
                record = dict(arm=arm, condition=condition, context=context,
                    checkpoint_sha256=primary['checkpoint_sha256'], branches={},
                    phase_repair=causal['phase_repair_contrasts'],
                    native_navigation=causal['branches']['raw__native']['navigation']['none'],
                    opportunity=causal_audit['opportunity'])
                for cue in CUES:
                    branch = primary['branches'][cue]
                    flow = branch['invariance']['30']['raw']['horizons']['30']
                    matches = lambda r: (r['arm'], r['condition'], r['context'], r['cue']) == (arm, condition, context, cue)
                    obs = [r for r in observable['rows'] if matches(r)]
                    readout = [r for r in actor['rows'] if matches(r) and r['time_updates'] == 60]
                    assert len(obs) == 2 and len(readout) == 1
                    assert readout[0]['checkpoint_sha256'] == primary['checkpoint_sha256']
                    assert readout[0]['archive_sha256'] == primary_audit['archive_sha256']
                    record['branches'][cue] = dict(
                        fullstate_candidate=branch['passes_both_metric_candidate_screen'],
                        hidden_radius_retention=branch['trajectory']['60']['hidden_radius_retention'],
                        raw_radial_over_tangent=flow['radial_over_tangent_gain']['median'],
                        raw_departure_p90=flow['residual_to_fixed_curve_over_radius']['p90'],
                        observable=obs, actor_at60=readout[0])
                rows.append(record)
    assert len(rows) == 8
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(rows=rows, final_behavior=final_behavior, source_sha256=sources,
        fullstate_candidates=sum(b['fullstate_candidate'] for r in rows for b in r['branches'].values()),
        fullstate_branch_count=32, observable_centered_candidates=observable['centered_candidates'],
        observable_noncentered_candidates=observable['noncentered_candidates'],
        observable_checks=64, actor_rows=128,
        scope='One original training seed pair; two matched entropy arms; all fixed final checkpoints and two native contexts each. Contexts, phases, cues and metrics are nested measurements.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), layout='constrained')
    for i, r in enumerate(rows):
        x = i // 2 + (-.11 if r['context'] == 0 else .11)
        color = '#357eb0' if r['condition'] == 'gc100' else '#d66b2c'
        for cue, shift, marker in [('native', -.025, 'o'), ('ordinary', .025, 's')]:
            b = r['branches'][cue]
            obs = next(q for q in b['observable'] if q['metric'] == 'raw_full_state_kicks')
            axes[0].scatter(x + shift, 100 * b['hidden_radius_retention'], color=color, marker=marker, s=36)
            axes[1].scatter(x + shift, obs['radial_over_tangent'], color=color, marker=marker, s=36)
            error = r['phase_repair']['raw']['60']['raw'][cue + '_error_deg']
            axes[2].scatter(x + shift, error, color=color, marker=marker, s=36)
    axes[0].set(title='Direction-related activity', ylabel='Activity radius retained at 4 s (%)')
    axes[1].set(title='Recovery from disturbances', ylabel='Radial / angular persistence\nin future hidden activity')
    axes[1].axhline(1, color='.6', ls='--', lw=1)
    axes[2].set(title='Repair by one predictive cue', ylabel='Full-state phase error\nafter a 30° shift (°)')
    axes[2].axhline(30, color='.6', ls='--', lw=1)
    labels = ['Grid\nOriginal entropy', 'PGC + grid\nOriginal entropy', 'Grid\nHigher entropy', 'PGC + grid\nHigher entropy']
    for ax in axes:
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.13)
        ax.set_xlim(-.4, 3.4)
    axes[0].scatter([], [], color='.3', marker='o', label='Native cue')
    axes[0].scatter([], [], color='.3', marker='s', label='Ordinary cue')
    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel('Fixed final models · two native contexts per model · one original seed pair', fontsize=9)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'memory_recovery_and_repair.{suffix}', dpi=180)
    plt.close(fig)

    lines = ['# Matched exploration: final neural and behavioral measurements', '',
        'All four fixed final models are included. Each continued from the same 40M parent within its population for another 50,073,600 learning frames. Only the exploration coefficient differs between arms.', '',
        '![Activity, recovery and cue repair](memory_recovery_and_repair.png)', '',
        'Each model contributes its first two physically eligible native histories. The circles and squares compare predictive and ordinary cue inputs within that frozen model. The middle panel measures recovery relative to the evolving activity family; the complete ring criteria also test its geometry and persistence. A low ratio alone is one component of the mechanism. Grid cue branches are identity controls.', '',
        '| Model | Exploration | Blind free goals/episode | Ordinary cue | Reset every blind update | Vision restored |',
        '|---|---|---:|---:|---:|---:|']
    for r in final_behavior:
        variants = r['endpoint']['variants']
        vals = [variants[k]['free_hits'] for k in ('blind_native', 'blind_ordinary', 'blind_reset_every_blind', 'seen_native')]
        lines.append(f"| {'Grid' if r['condition'] == 'gc100' else 'PGC + grid'} | {r['arm']} | " + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    lines += ['', 'These rates retain all 64 original evaluation episodes, including zero-event episodes. Full learning curves, all cue and reset variants, goalward progress and paired slot uncertainty remain in the [behavior summary](../../rl_blind_reorientation_entropy/summary.json).', '',
        '| Model / exploration / context | Hidden radius retained | Phase error: ordinary → predictive | Translation probability | Mean goalward progress |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        b = r['branches']['native']; phase = r['phase_repair']['raw']['60']['raw']
        probability = 100 * b['actor_at60']['probabilities']['total_translation_mean']
        progress = r['native_navigation']['total_goalward_progress_m']['mean']
        lines.append(f"| {'Grid' if r['condition'] == 'gc100' else 'PGC + grid'} / {r['arm']} / {r['context'] + 1} | {100*b['hidden_radius_retention']:.3g}% | {phase['ordinary_error_deg']:.3f}° → {phase['native_error_deg']:.3f}° | {probability:.4g}% | {progress:+.4f} m |")
    lines += ['',
        'Translation probability uses the exact learned action layer across all 768 controlled directions at free-choice onset. Navigation uses 192 directions, matched action and motor-noise draws, and 30 free actions. These controlled contexts differ from whole-episode behavioral evaluation. Goal opportunity bounds are retained for every context; zero hits from unreachable starts are not evidence about memory.', '',
        f"The unchanged full-state screen nominates **{summary['fullstate_candidates']} / 32 cue branches**. The all-hidden future-activity screen nominates **{summary['observable_centered_candidates']} / 64 centered comparisons** and **{summary['observable_noncentered_candidates']} / 64 uncentered comparisons**. These counts describe nested checks of four endpoints, not independent trained agents.", '',
        'All cue directions, both perturbation metrics, every geometry gate, all state edits and all 128 action-layer rows remain in the source outputs. This report adds no fits, thresholds or checkpoint selection.', '',
        '[Complete report data](summary.json) · [Independent audit](../independent_artifact_audit.json) · [Frozen method](../../../docs/research/reorientation_entropy_mechanism_reproduction.md) · [Reproduction inputs](../../rl_blind_reorientation_entropy/REPRODUCE.md)']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
