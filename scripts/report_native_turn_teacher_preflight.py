"""Describe both completed teacher preflights using saved JSON only."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_teacher_preflight'
PROTOCOL = ROOT / 'docs/research/native_turn_teacher_preflight_protocol.json'
CONDITIONS = ('gc100', 'pgc33')
KINDS = ('turn_only', 'probe_glimpse')
NAMES = {'gc100': 'Grid', 'pgc33': 'PGC + grid'}
MODES = {'turn_only': 'Turn only', 'probe_glimpse': 'Probe + cue'}
COLORS = {'gc100': '#2776b4', 'pgc33': '#e76a32'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = list(values)
    assert values and all(math.isfinite(value) for value in values)
    return sum(values) / len(values)


def context_information(information, kind):
    """Allocate each exact-input group's KL to its cases' source contexts.

    This remains well defined if a byte-identical input group spans contexts.
    No group is split and refitted to improve the displayed information value.
    """
    contributions = defaultdict(list)
    seen = set()
    for group in information['records']:
        assert len(group['cases']) == len(group['per_case_kl_to_common_nats']) == group['count']
        for case, value in zip(group['cases'], group['per_case_kl_to_common_nats']):
            assert case['kind'] == kind
            assert case['pair'] == case['case'] // 2 and case['member'] == case['case'] % 2
            key = (case['context'], case['case'])
            assert key not in seen
            seen.add(key)
            contributions[case['context']].append(value)
    assert seen == {(context, case) for context in range(16) for case in range(16)}
    assert information['cases'] == 256 and information['groups'] == len(information['records'])
    result = {context: mean(contributions[context]) for context in range(16)}
    assert math.isclose(mean(result.values()), information['case_weighted_minimum_forward_kl_nats'], abs_tol=1e-12)
    return result


def collect():
    protocol = read(PROTOCOL)
    assert protocol['no_optimization'] and protocol['qualification'] is None
    assert protocol['contexts'] == 16 and protocol['pairs_per_context'] == 8
    assert protocol['teacher_continuation_updates'] == 30 and set(protocol['kinds']) == set(KINDS)
    for field in ('source_sha256', 'asset_sha256'):
        for name, expected in protocol[field].items():
            assert sha(ROOT / name) == expected, name
    # Inspect both terminal summaries before making any report output.
    summaries = {condition: read(BASE / condition / 'summary.json') for condition in CONDITIONS}
    assert all(value['complete'] for value in summaries.values())
    contexts = []
    groups = []
    provenance = {}
    for condition, summary in summaries.items():
        folder = BASE / condition
        assert summary['condition'] == condition and not (folder / 'incomplete.json').exists()
        assert summary['protocol_sha256'] == sha(PROTOCOL)
        assert summary['qualification'] is None and summary['qualification_status'] == 'descriptive_only_no_scientific_gate'
        assert summary['no_optimization'] and summary['no_cloud'] and summary['global_cpu_rng_preserved']
        checkpoint = ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.pt'
        assert sha(checkpoint) == summary['checkpoint_sha256']
        config = read(checkpoint.with_suffix('.json'))
        assert config['action_repeat'] * 30 / 60 == 2
        for name, entry in summary['artifacts'].items():
            path = folder / name
            assert path.is_file() and not path.is_symlink() and path.parent == folder
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
        inventory = read(folder / 'inventory.json')
        assert inventory['bank_sha256'] == protocol['banks'][condition]['sha256']
        assert len(inventory['inventory']) == 32
        selected = [row for row in inventory['inventory'] if row['selected']]
        assert len(selected) == len({row['slot'] for row in selected}) == 16
        assert {row['context_id'] for row in selected} == set(range(16))
        prefixes = read(folder / 'native_prefix_verification.json')
        assert prefixes['complete'] and prefixes['all_incoming_hc_exact']
        pairs = summary['pairs']
        assert len(pairs) == 256
        assert {(p['kind'], p['context'], p['pair']) for p in pairs} == {
            (kind, context, pair) for kind in KINDS for context in range(16) for pair in range(8)}
        continuations = {(row['kind'], row['context']): row for row in summary['teacher_continuations']}
        assert len(continuations) == len(summary['teacher_continuations']) == 32
        assert set(continuations) == {(kind, context) for kind in KINDS for context in range(16)}
        for kind in KINDS:
            assert summary['pair_counts'][kind] == 128
            information = summary['exact_input_group_information'][kind]
            context_js = context_information(information, kind)
            mode_contexts = []
            for context in range(16):
                selected_pairs = [p for p in pairs if p['kind'] == kind and p['context'] == context]
                continuation = continuations[kind, context]
                assert continuation['steps'] == 30 and continuation['first_hit_censored']
                assert continuation['preteleport_endpoint_accounting'] and continuation['shared_uniforms_within_pair']
                cases = continuation['rows']
                assert len(cases) == 16 and {r['case'] for r in cases} == set(range(16))
                assert all(1 <= row['active_updates'] <= 30 for row in cases)
                assert all(row['hit'] == (row['first_hit_step'] >= 0) for row in cases)
                row = dict(condition=condition, kind=kind, context=context,
                    source_slot=selected_pairs[0]['source_slot'], source_step=selected_pairs[0]['source_step'],
                    pairs=8, continuation_cases=16, generalized_js_nats=context_js[context],
                    mean_pair_tv=mean(p['teacher_total_variation'] for p in selected_pairs),
                    mean_pair_js_nats=mean(p['teacher_jensen_shannon'] for p in selected_pairs),
                    mean_first_step_teacher_progress_m=mean(p['teacher_mean_expected_progress_m'] for p in selected_pairs),
                    mean_first_step_common_ceiling_m=mean(p['best_common_action_ceiling_m'] for p in selected_pairs),
                    mean_first_step_teacher_advantage_m=mean(p['teacher_advantage_over_common_m'] for p in selected_pairs),
                    mean_continuation_goal_progress_m=mean(r['goal_distance_progress_m'] for r in cases),
                    continuation_hits=sum(r['hit'] for r in cases),
                    continuation_hit_fraction=mean(float(r['hit']) for r in cases),
                    continuation_episode_ends=sum(r['first_done_step'] >= 0 for r in cases),
                    mean_continuation_active_updates=mean(r['active_updates'] for r in cases))
                assert all(p['source_slot'] == row['source_slot'] and p['source_step'] == row['source_step'] for p in selected_pairs)
                contexts.append(row)
                mode_contexts.append(row)
            group = dict(condition=condition, kind=kind, contexts=16, pairs=128, continuation_cases=256,
                exact_input_groups=information['groups'], generalized_js_nats=information['case_weighted_minimum_forward_kl_nats'],
                continuation_hits=sum(r['continuation_hits'] for r in mode_contexts),
                continuation_episode_ends=sum(r['continuation_episode_ends'] for r in mode_contexts))
            for key in ('mean_pair_tv', 'mean_pair_js_nats', 'mean_first_step_teacher_progress_m',
                        'mean_first_step_common_ceiling_m', 'mean_first_step_teacher_advantage_m',
                        'mean_continuation_goal_progress_m', 'continuation_hit_fraction',
                        'mean_continuation_active_updates'):
                group[key] = mean(row[key] for row in mode_contexts)
            groups.append(group)
        all_modes = summary['exact_input_group_information']['all_modes']
        assert all_modes['cases'] == 512
        provenance[condition] = dict(summary_sha256=sha(folder / 'summary.json'),
            checkpoint_sha256=summary['checkpoint_sha256'], verified_artifacts=len(summary['artifacts']),
            all_modes_exact_input_information=all_modes, inventory=inventory['inventory'],
            native_prefix_verification_sha256=sha(folder / 'native_prefix_verification.json'))
    assert len(contexts) == 64 and len(groups) == 4
    return dict(complete=True,source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
        contexts=contexts, groups=groups, provenance=provenance,
        information_scope='Mode-specific exact current-input equivalence classes. Each context point averages its cases\' KL to their class-optimal common action distribution; if a class spans contexts it stays intact. Context means average to the case-weighted generalized JS.',
        scope='One original training seed per population;16 distinct source episodes, 128 antipodal pairs and 256 continuation cases per mode/model. Descriptive only: no p-values or scientific qualification gate.')


def render(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    panels = (('generalized_js_nats', 1., 'Teacher information', 'Minimum teacher KL (nats)'),
              ('mean_continuation_goal_progress_m', 100., 'Teacher navigation · up to 2 s', 'Goal distance reduced (cm)'))
    positions = {(condition, kind): ci * 3 + ki for ci, condition in enumerate(CONDITIONS) for ki, kind in enumerate(KINDS)}
    for ax, (measure, scale, title, label) in zip(axes, panels):
        for condition in CONDITIONS:
            for context in range(16):
                records = [next(r for r in result['contexts'] if r['condition'] == condition and r['kind'] == kind and r['context'] == context) for kind in KINDS]
                offset = (context - 7.5) * .012
                x = [positions[condition, kind] + offset for kind in KINDS]
                ax.plot(x, [scale * r[measure] for r in records], color=COLORS[condition], alpha=.35,
                        marker='o', markersize=3, linewidth=.6)
            for kind in KINDS:
                group = next(r for r in result['groups'] if r['condition'] == condition and r['kind'] == kind)
                x = positions[condition, kind]
                ax.plot([x - .2, x + .2], [scale * group[measure]] * 2, color=COLORS[condition], linewidth=3)
        ax.axhline(0, color='.5', linewidth=.7)
        ax.set_xticks(list(positions.values()), ['Grid\nTurn', 'Grid\nProbe', 'PGC + grid\nTurn', 'PGC + grid\nProbe'], fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'teacher_preflight.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_text(result, out):
    lines = ['# Native turn teacher: information and navigation', '',
        'This descriptive preflight asks whether the original full-vision teacher supplies different action targets when the student receives the same current input, and how the teacher then navigates. Every context and both preparations are retained.', '',
        '![Teacher information and navigation](teacher_preflight.png)', '',
        'Each dot is one of 16 source contexts; thin lines join the same context across preparations, and thick marks show means. Left: generalized Jensen–Shannon information in nats, grouped by exact complete student-current-input equality. Right: teacher goal-distance reduction during up to 30 native decisions (two seconds), scored through the first hit or episode end using positions before teleportation.', '',
        '| Population | Preparation | Pair TV | Pair JS (nats) | Current-input JS (nats) | First-step progress / common ceiling (mm) | Two-second progress (cm) | Hits /256 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in result['groups']:
        lines.append(f'| {NAMES[row["condition"]]} | {MODES[row["kind"]]} | {row["mean_pair_tv"]:.3f} | {row["mean_pair_js_nats"]:.3f} | {row["generalized_js_nats"]:.3f} | {1000*row["mean_first_step_teacher_progress_m"]:.3f} / {1000*row["mean_first_step_common_ceiling_m"]:.3f} | {100*row["mean_continuation_goal_progress_m"]:.2f} | {row["continuation_hits"]}/256 ({100*row["continuation_hit_fraction"]:.1f}%) |')
    lines += ['',
        'Each row contains 16 distinct source episodes, 128 antipodal heading pairs and 256 continuation cases. These are controlled copies of one original seed per population, not independent trained-model replications. The two preparations share the source contexts. No p-values, ranking-based case selection or scientific pass/fail threshold is applied.', '',
        'Pair TV and JS compare each antipodal pair. Generalized JS includes every case with the same full 1,287-entry current input; it is the smallest average forward-KL mismatch achievable by one common action distribution. Context values allocate that group-level KL to their member cases, preserving groups even if they span contexts. The combined-mode grouping is also retained in summary.json.', '',
        '**What the information measure means:** positive JS identifies teacher information absent from the student’s current input. The teacher itself receives full RGB and native current codes, which can differ between those cases. This measures a demand that student history could help satisfy. Learning that history and developing a ring are subsequent empirical questions.', '',
        'First-step progress is the signed movement projected toward the goal. Its ceiling is the best common action across the two members of each pair. A useful turn can have zero immediate translation, so the actual two-second full-vision continuation is reported alongside it; that continuation is not a later-horizon oracle bound. Image and motor noise are disabled for this controlled preflight. Native saved-goal information remains available.', '',
        'All 64 context summaries, denominators, combined-mode information groups, source-episode inventory and provenance: [summary.json](summary.json). Raw pair probabilities and every continuation case remain in the two model summary files: [Grid](../gc100/summary.json) and [PGC + grid](../pgc33/summary.json).']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    result = collect()
    out = BASE / 'report'
    out.mkdir(parents=True, exist_ok=True)
    render(result, out)
    write_text(result, out)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'contexts': len(result['contexts']), 'groups': result['groups']}, indent=2))


if __name__ == '__main__':
    main()
