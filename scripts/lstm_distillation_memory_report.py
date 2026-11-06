"""Summarize every frozen distillation direction-memory assay, without selection."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lstm_heading_memory_probe import sha, save
from lstm_distillation_memory import OUT, CONDITIONS, ENDPOINTS, WINDOWS

MAPS = ('full_state', 'all_hidden', 'centered_actor_logits')
LABELS = {'full_state': 'Full memory state', 'all_hidden': 'Hidden activity',
          'centered_actor_logits': 'Action scores'}
COLORS = {'gc100': '#2776b4', 'pgc33': '#e76a32'}
NAMES = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}


def collect(out):
    protocol = sha(out / 'protocol.json'); rows = []; contexts = []
    frozen = json.loads((out / 'protocol.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for condition in CONDITIONS:
        for endpoint in ENDPOINTS:
            folder = out / condition / endpoint
            summary = json.loads((folder / 'summary.json').read_text())
            assert summary['complete'] and summary['protocol_sha256'] == protocol
            regimes = ('own_policy',) if endpoint == 'initial' else ('own_policy', 'matched_initial_contexts')
            assert summary['contexts'] == (4 if endpoint == 'initial' else 8)
            for regime in regimes:
                for family in ('heading', 'travel'):
                    base = folder / family if regime == 'own_policy' else folder / regime / family
                    for index in range(2):
                        path = base / f'context{index}_adapted.json'
                        record = json.loads(path.read_text())
                        assert record['protocol_sha256'] == protocol
                        assert record['checkpoint_sha256'] == summary['checkpoint_sha256']
                        assert record['source_sha256'] == sha(ROOT / 'scripts/lstm_distillation_memory.py')
                        original_path = base / f'context{index}.json' if family == 'heading' else base / f'{condition}_s1/context{index}.json'
                        original = json.loads(original_path.read_text())
                        archive_path = original_path.with_name(f'context{index}_states.npz')
                        assert sha(archive_path) == record['archive_sha256']
                        source_path = ((base / 'contexts') if regime == 'own_policy' else
                                       (out / condition / 'initial' / family / 'contexts')) / f'context{index}.npz'
                        assert sha(source_path) == record['context_source_sha256']
                        if regime == 'matched_initial_contexts':
                            assert record['paired_comparison']['inputs_and_angles_byte_identical']
                            assert record['paired_comparison']['all15_preparation_inputs_byte_identical']
                            assert record['paired_comparison']['initial_hc_never_transferred']
                        branches = original['branches'] if family == 'heading' else {'ordinary_current_release': original['ordinary_release']}
                        common = dict(condition=condition, endpoint=endpoint, regime=regime,
                                      family=family, context=index, path=str(path.relative_to(ROOT)))
                        contexts.append(dict(**common, archive_sha256=record['archive_sha256'],
                            source_sha256=record['context_source_sha256'], prefix_sha256=record['prefix_sha256'],
                            prefix_mask_sha256=record['prefix_mask_sha256']))
                        for branch, windows in record['full_state_candidate_gates'].items():
                            for window, metrics in windows.items():
                                reference, horizon = (15, 15) if window == 'within_trained_2s' else (30, 30)
                                for metric, gates in metrics.items():
                                    original_branch = branches[branch]
                                    invariance = original_branch['invariance'][str(reference)][metric]
                                    result = invariance['horizons'][str(horizon)]
                                    rows.append(dict(**common, branch=branch, window=window,
                                        metric=metric, observed='full_state', centered=False,
                                        candidate=all(gates.values()), gates=gates,
                                        departure_p90=result['residual_to_fixed_curve_over_radius']['p90'],
                                        radial_gain=result['radial_normal_gain']['median'],
                                        radial_over_phase=result['radial_over_tangent_gain']['median'],
                                        radius_retention=original_branch['trajectory'][str(reference)][metric]['radius_retention'],
                                        phase_coverage=result['phase_transport']['circular_coverage'], actor_probability_rms=None))
                        for branch, results in record['fixed_observable'].items():
                            for window, outputs in results['windows'].items():
                                for observed, metrics in outputs.items():
                                    for metric, result in metrics.items():
                                        for centering in ('centered', 'noncentered'):
                                            invariance = result['invariance'][centering]
                                            gates = result['candidate_gates'][centering]
                                            rows.append(dict(**common, branch=branch, window=window,
                                                metric=metric, observed=observed, centered=centering == 'centered',
                                                candidate=all(gates.values()), gates=gates,
                                                departure_p90=invariance['residual_to_reference_curve_over_radius']['p90'],
                                                radial_gain=result['radial_observable_normal_gain']['median'],
                                                radial_over_phase=result['radial_over_tangent_observable_gain']['median'],
                                                radius_retention=invariance['evolved_radius_retention'],
                                                phase_coverage=invariance['phase_transport']['circular_coverage'],
                                                actor_probability_rms=result['actor']['evolved']['family_probability_RMS']))
    assert len(contexts) == 24 and len(rows) == 720
    assert len({tuple(r[k] for k in ('condition', 'endpoint', 'regime', 'family', 'context'))
                for r in contexts}) == 24
    assert len({tuple(r[k] for k in ('condition', 'endpoint', 'regime', 'family', 'context',
                                    'branch', 'window', 'metric', 'observed', 'centered'))
                for r in rows}) == 720
    return rows, contexts


def paired(rows, *, condition, endpoint, observed, window='within_trained_2s', metric=None):
    regime = 'own_policy' if endpoint == 'initial' else 'matched_initial_contexts'
    return [r for r in rows if r['condition'] == condition and r['endpoint'] == endpoint
            and r['regime'] == regime and r['observed'] == observed and r['window'] == window
            and r['centered'] == (observed != 'full_state')
            and (metric is None or r['metric'] == metric)]


def both_metric_count(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('condition', 'endpoint', 'regime', 'family', 'context',
                                     'branch', 'window', 'observed', 'centered'))
        groups.setdefault(key, {})[row['metric']] = row['candidate']
    assert all(set(values) == {'raw', 'relative_rms'} for values in groups.values())
    return sum(all(values.values()) for values in groups.values()), len(groups)


def plot_counts(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.1), sharey=True)
    for ax, observed in zip(axes, MAPS):
        for ci, condition in enumerate(CONDITIONS):
            counts = [both_metric_count(paired(rows, condition=condition, endpoint=e, observed=observed))[0]
                      for e in ENDPOINTS]
            x = np.arange(2) + (-.17 if ci == 0 else .17)
            ax.bar(x, counts, width=.3, color=COLORS[condition], label=NAMES[condition])
            for xx, n in zip(x, counts):
                ax.text(xx, n + .13, str(n), ha='center', va='bottom', fontsize=10)
        ax.set_title(LABELS[observed], fontsize=12)
        ax.set_xticks((0, 1), ('Initial', 'Final'))
        ax.set_ylim(0, 6.8); ax.set_yticks((0, 2, 4, 6))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Tests meeting every criterion (of 6)')
    axes[-1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('Same histories · 2-second test', fontsize=14)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'distillation_paired_candidates.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_diagnostics(rows, out):
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 7.0), sharex=True)
    measures = [('departure_p90', 'Departure from the curve (%)', 5.),
                ('radial_over_phase', 'Radial error / phase signal (%)', 50.)]
    for i, observed in enumerate(MAPS):
        for j, (key, title, cutoff) in enumerate(measures):
            ax = axes[i, j]
            for condition in CONDITIONS:
                grouped = {}
                for endpoint in ENDPOINTS:
                    selected = paired(rows, condition=condition, endpoint=endpoint,
                                      observed=observed, metric='relative_rms')
                    grouped[endpoint] = {tuple(r[k] for k in ('family', 'context', 'branch')): r for r in selected}
                assert grouped['initial'].keys() == grouped['final'].keys()
                values = []
                for case in grouped['initial']:
                    y = [max(1e-8, 100 * grouped[e][case][key]) for e in ENDPOINTS]
                    values.append(y)
                    ax.plot((0, 1), y, color=COLORS[condition], alpha=.22, linewidth=.7, marker='o', markersize=2)
                median = np.median(values, axis=0)
                ax.plot((0, 1), median, color=COLORS[condition], marker='o', linewidth=2,
                        label=NAMES[condition])
            ax.axhline(cutoff, color='0.4', linestyle='--', linewidth=.8)
            ax.set_yscale('log'); ax.set_xticks((0, 1), ('Initial', 'Final'))
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0: ax.set_title(title, fontsize=11)
            if j == 0: ax.set_ylabel(LABELS[observed], fontsize=11)
    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.suptitle('What changed at the same histories?', fontsize=14)
    fig.text(.5, .01, '2 seconds · fixed h/c RMS metric · thin lines: fixed tests · thick lines: medians',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96))
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'distillation_paired_diagnostics.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def report(rows, out):
    table = []; counts = {}
    for condition in CONDITIONS:
        for observed in MAPS:
            values = {endpoint: both_metric_count(paired(rows, condition=condition,
                       endpoint=endpoint, observed=observed)) for endpoint in ENDPOINTS}
            counts[f'{condition}/{observed}'] = values
            table.append(f'| {NAMES[condition]} | {LABELS[observed]} | {values["initial"][0]}/6 | {values["final"][0]}/6 |')
    nominations = [r for r in rows if r['candidate']]
    nomination_counts = {}
    for row in rows:
        group = '/'.join(str(row[key]) for key in ('condition', 'endpoint', 'regime', 'window',
                                                  'observed', 'metric', 'centered'))
        value = nomination_counts.setdefault(group, dict(positive=0, total=0))
        value['positive'] += int(row['candidate']); value['total'] += 1
    failure_counts = {}
    for row in rows:
        for key, passed in row['gates'].items():
            if not passed:
                failure_counts[key] = failure_counts.get(key, 0) + 1
    own_final = [r for r in rows if r['endpoint'] == 'final' and r['regime'] == 'own_policy'
                 and r['window'] == 'within_trained_2s' and r['centered'] == (r['observed'] != 'full_state')]
    long_paired = [r for r in rows if r['window'] == 'legacy_4s_extrapolation'
                   and r['regime'] == ('own_policy' if r['endpoint'] == 'initial' else 'matched_initial_contexts')
                   and r['centered'] == (r['observed'] != 'full_state')]
    own_n, own_d = both_metric_count(own_final); long_n, long_d = both_metric_count(long_paired)
    paired_totals = {e: both_metric_count([r for r in rows if r['endpoint'] == e
        and r['regime'] == ('own_policy' if e == 'initial' else 'matched_initial_contexts')
        and r['window'] == 'within_trained_2s' and r['centered'] == (r['observed'] != 'full_state')])
        for e in ENDPOINTS}
    initial_n, initial_d = paired_totals['initial']; final_n, final_d = paired_totals['final']
    descriptive = {}
    for condition in CONDITIONS:
        for endpoint in ENDPOINTS:
            for observed in MAPS:
                selected = paired(rows, condition=condition, endpoint=endpoint,
                                  observed=observed, metric='relative_rms')
                descriptive[f'{condition}/{endpoint}/{observed}'] = {
                    key: float(np.median([r[key] for r in selected]))
                    for key in ('departure_p90', 'radial_gain', 'radial_over_phase', 'radius_retention', 'phase_coverage')}
    contraction_table = []
    for condition in CONDITIONS:
        a = descriptive[f'{condition}/initial/centered_actor_logits']
        b = descriptive[f'{condition}/final/centered_actor_logits']
        contraction_table.append(f'| {NAMES[condition]} | {100*a["radial_gain"]:.0f}% → {100*b["radial_gain"]:.0f}% | {100*a["radius_retention"]:.0f}% → {100*b["radius_retention"]:.0f}% |')
    finding = (f'The matched two-second screen found **{final_n}/{final_d}** jointly qualifying final tests, '
               f'compared with **{initial_n}/{initial_d}** initially. ')
    if final_n == 0:
        finding += 'The tested direction families did not acquire the combination of persistent geometry and selective radial recovery required by this screen.'
    else:
        maps = [LABELS[m] for m in MAPS if sum(counts[f'{c}/{m}']['final'][0] for c in CONDITIONS)]
        finding += 'The qualifying observation maps are: ' + ', '.join(maps) + '. These candidates need the broader perturbation check described below.'
    text = (
        '# Direction memory after distillation\n\n' + finding + '\n\n'
        'We tested both original agents and both fixed final distillation agents. We replayed the same recorded history through each network, letting each build its own memory state. We then gave them identical heading or movement experiences and identical future inputs. This isolates the weight change at those histories.\n\n'
        'Both populations use seed 1. Each final model received 1,036,800 extra physics frames of policy distillation, with only the four LSTM tensors allowed to change. During the memory test, images are black. Heading tests compare native and ordinary current codes. Travel-direction tests replace the predictive current entries with ordinary values, giving every remembered direction the same future input. The native saved goal code remains available.\n\n'
        'A ring candidate must keep different directions distinguishable, stay close to the same closed family, and correct the tested radial error faster than it loses the direction signal. We examined the complete hidden-and-cell state, all hidden outputs, and all learned action scores. No fitted plane or angle decoder defines these tests.\n\n'
        '![Matched two-second candidates](distillation_paired_candidates.png)\n\n'
        '| Population | What we observe | Initial | Final |\n| --- | --- | ---: | ---: |\n' + '\n'.join(table) + '\n\n'
        'Each denominator is six fixed context/input tests. A count requires every criterion in both raw and h/c RMS coordinates; hidden-output and action-score counts use the prespecified common-mean correction. All noncentered results remain in the data. The duplicated native/ordinary Grid inputs are identity controls, not independent replications.\n\n'
        '![Matched two-second diagnostics](distillation_paired_diagnostics.png)\n\n'
        'The left panels show how far the family moves away from its earlier shape. The right panels compare the remaining radial error with the surviving phase signal. Lower values are favorable, but neither measurement alone is sufficient: all geometry, visibility and recovery checks must pass. Dashed lines mark the existing thresholds.\n\n'
        '**What changed:** output responses contracted more strongly after training. The tested action-score error became smaller, and the direction-dependent pattern also shrank.\n\n'
        '| Population | Radial action-score error left, initial → final | Direction-pattern size left, initial → final |\n| --- | ---: | ---: |\n' + '\n'.join(contraction_table) + '\n\n'
        'These are descriptive medians of six fixed tests in the h/c RMS metric, measured over one second of evolution; the complete primary observation window ends two seconds after blackout onset. Every matched test still failed the curve-stability criterion. In the complete hidden-and-cell state, radial error and phase signal decayed at roughly the same rate. This suggests stronger contraction of short-term output activity, with direction patterns that continue to deform, rather than selective correction toward a stable direction ring.\n\n'
        f'The final agents’ own-policy contexts produce **{own_n}/{own_d}** jointly qualifying primary tests. The separate four-second matched-context extrapolation produces **{long_n}/{long_d}** qualifying tests across both endpoints. These counts combine the three observation maps; they do not count independent trained agents.\n\n'
        'All 24 context evaluations are retained: eight initial contexts, eight final own-policy contexts and eight final tests matched to the initial contexts. The two-second window matches the largest training blackout. Four seconds is an extrapolation.\n\n'
        'These screens test one specific inward/outward radial perturbation. A qualifying observable is a candidate for broader normal-direction and predictive-cue repair tests. The actor-probability amplitudes and every failed gate are saved in aggregate.json, so a small score-space pattern is not silently treated as a useful navigation mechanism.\n')
    (out / 'README.md').write_text(text)
    return dict(primary_paired_both_metric_counts=counts, individual_metric_nominations=len(nominations),
                exact_nomination_counts=nomination_counts,
                primary_paired_both_metric_totals=paired_totals,
                primary_paired_relative_rms_descriptive_medians=descriptive,
                nominations=nominations, failed_gate_counts=failure_counts,
                own_final_primary_both_metric_count=[own_n, own_d],
                paired_four_second_both_metric_count=[long_n, long_d])


def main():
    out = OUT
    rows, contexts = collect(out)
    plot_counts(rows, out); plot_diagnostics(rows, out)
    summary = report(rows, out)
    save(out / 'aggregate.json', dict(complete=True, protocol_sha256=sha(out / 'protocol.json'),
        source_sha256=sha(__file__), contexts=contexts, rows=rows, **summary))
    print(json.dumps({key: value for key, value in summary.items() if key not in ('nominations', 'failed_gate_counts', 'exact_nomination_counts')}, indent=2))


if __name__ == '__main__':
    main()
