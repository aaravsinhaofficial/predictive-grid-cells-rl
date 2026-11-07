"""Plan or package the complete native-turn study; never execute scientific models.

The default is an inventory with explicit missing/stale completion records.
Build requires every prespecified endpoint and independent audit, irrespective
of scientific outcome. The existing verified deterministic archive writer is
reused; parts stay below 1.5 GB and every member is checked by size and SHA256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import package_nonlinear_ring_outputs as archive

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'results/native_turn_release'
CONDITIONS = ('gc100', 'pgc33')
ARMS = ('clean', 'noise')
TREES = tuple('results/' + n for n in (
    'native_turn_teacher_preflight', 'native_turn_original_memory',
    'native_turn_distillation', 'native_turn_geometry',
    'native_turn_imitation_decomposition',
    'native_turn_imitation_decomposition_design',
    'native_turn_imitation_prefix', 'native_turn_imitation_prefix_design',
    'native_turn_distillation_design', 'native_turn_original_memory_design',
    'native_turn_geometry_design'))
archive.TREES = TREES
archive.MANIFEST_NAME = f'{RELEASE}/manifest.json'
PROTOCOLS = {n: f'docs/research/native_turn_{n}_protocol.json' for n in (
    'teacher_preflight', 'original_memory', 'distillation', 'geometry')}
DOCS = (
    'README.md', 'mechanisms/learned-ring-in-navigation.md',
    'requirements-ring-analysis.txt', 'docs/research/ring_local_environment.json',
    *PROTOCOLS.values(),
    *('docs/research/' + n for n in (
        'native_turn_teacher_preflight_review.json',
        'native_turn_original_memory_review.json',
        'native_turn_distillation_review.json', 'native_turn_geometry_review.json',
        'native_turn_geometry_plan.md', 'native_turn_distillation_acquisition_spec.md',
        'native_turn_distillation_minimal_plan.md', 'native_turn_distillation_rationale.md',
        'native_rl_ring_literature_followup.md',
        'native_forward_backward_memory_next_design.md',
        'native_input_observability_ring_design.md')),
    f'{RELEASE}/REPRODUCE.md')
EXTRA_SOURCE = tuple('scripts/' + n + '.py' for n in (
    'package_native_turn_outputs', 'report_native_turn_teacher_preflight',
    'audit_native_turn_teacher_preflight_outputs', 'report_native_turn_original_memory',
    'audit_native_turn_original_memory_outputs', 'audit_native_turn_distillation_baselines',
    'audit_native_turn_distillation_outputs',
    'report_native_turn_distillation', 'report_native_turn_geometry',
    'report_native_turn_geometry_failures', 'audit_native_turn_geometry_report',
    'run_native_turn_remaining', 'inspect_native_turn_clean_alignment',
    'decompose_native_turn_imitation', 'audit_native_turn_imitation_decomposition',
    'decompose_native_turn_prefix', 'audit_native_turn_imitation_prefix')) + tuple(
        'tests/' + n + '.py' for n in (
            'test_report_native_turn_distillation', 'test_report_native_turn_geometry',
            'test_audit_native_turn_distillation_outputs', 'test_audit_native_turn_geometry_report',
            'test_decompose_native_turn_imitation', 'test_decompose_native_turn_prefix'))


def verify_presentation_addendum(current, addendum):
    """Prove the current report differs only by its recorded text replacement."""
    assert addendum['complete'] and addendum['file'] == 'README.md'
    assert hashlib.sha256(current).hexdigest() == addendum['updated_sha256']
    previous = addendum['previous_text'].encode('utf-8')
    updated = addendum['updated_text'].encode('utf-8')
    assert previous != updated and current.count(updated) == 1 and previous not in current
    original = current.replace(updated, previous)
    assert hashlib.sha256(original).hexdigest() == addendum['previous_sha256']
    return original


def inventory(root, commit=None):
    root = Path(root).resolve()
    sources = set(filter(None, subprocess.check_output(
        ['git', 'ls-files', '-z', 'code', 'scripts', 'tests'], cwd=root).decode().split('\0')))
    sources.update(EXTRA_SOURCE)
    names = sources | set(DOCS)
    issues, checks = [], []

    def check(label, action):
        try:
            action()
            checks.append(label)
        except (AssertionError, OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(check=label, reason=str(error) or type(error).__name__))

    def read(name):
        names.add(name)
        return archive.read(archive.checked(root, name))

    def prove(name, expected):
        names.add(name)
        return archive.prove(root, name, dict(sha256=expected) if isinstance(expected, str) else expected)

    def artifacts(base, rows):
        assert rows, base + ': empty artifact inventory'
        for name, record in rows.items():
            prove(f'{base}/{name}', record)

    def protocol_binding(value, kind, key='protocol_sha256'):
        prove(PROTOCOLS[kind], value[key])

    for kind, path in PROTOCOLS.items():
        def frozen(path=path, kind=kind):
            value = read(path)
            for field in ('source_sha256', 'asset_sha256'):
                assert value[field], path
                for name, expected in value[field].items():
                    prove(name, expected)
                    if field == 'source_sha256':
                        sources.add(name)
            if kind == 'teacher_preflight':
                assert value['contexts'] == 16 and set(value['banks']) == set(CONDITIONS)
                for bank in value['banks'].values():
                    prove(bank['path'], bank)
            elif kind == 'distillation':
                assert value['updates'] == 180 and value['extra_frames'] == 1036800
                assert value['acquisition_seconds'] == 1800 and value['evaluation_seconds'] == 600
            elif kind == 'geometry':
                assert value['phases'] == 128 and value['contexts_per_population'] == 16
        check('frozen protocol and inputs: ' + kind, frozen)

    for condition in CONDITIONS:
        for kind in ('teacher_preflight', 'original_memory'):
            def diagnostic(condition=condition, kind=kind):
                base = f'results/native_turn_{kind}/{condition}'
                value = read(f'{base}/summary.json')
                assert value['complete'] and value['condition'] == condition
                protocol_binding(value, kind)
                assert len(value['pairs']) == (256 if kind == 'teacher_preflight' else 2048)
                if kind == 'teacher_preflight':
                    assert value['pair_counts'] == {'turn_only': 128, 'probe_glimpse': 128}
                else:
                    assert len(value['rows']) == 4096
                artifacts(base, value['artifacts'])
                execution = read(f'results/native_turn_{kind}/{condition}_execution.json')
                assert execution['exit_code'] == 0 and execution['no_cloud']
            check(f'complete {kind}: {condition}', diagnostic)

        def baseline(condition=condition):
            base = f'results/native_turn_distillation/{condition}/original_baseline'
            value = read(f'{base}/summary.json')
            assert value['complete']
            protocol_binding(value, 'distillation')
            prove(f'{base}/live.json', value['live_sha256'])
            execution = read(f'results/native_turn_distillation/{condition}_baseline_execution.json')
            assert execution['complete'] and all(x['exit_code'] == 0 for x in execution['stages'])
            audit = read('results/native_turn_distillation/original_baseline_independent_validation.json')
            assert audit['complete'] and audit['models'][condition]['episodes'] == 96
            row = audit['models'][condition]
            prove(f'{base}/summary.json', row['summary_sha256'])
            prove(f'{base}/live.json', row['live_sha256'])
            prove(f'results/native_turn_distillation/{condition}_baseline_execution.json', row['execution_sha256'])
            prove('scripts/audit_native_turn_distillation_baselines.py', audit['audit_source_sha256'])
        check(f'complete shared baseline and audit: {condition}', baseline)

        for arm in ARMS:
            def training(condition=condition, arm=arm):
                base = f'results/native_turn_distillation/{condition}/{arm}'
                value = read(f'{base}/summary.json')
                assert value['complete'] and (value['condition'], value['arm']) == (condition, arm)
                assert (value['updates'], value['extra_frames'], value['policy_decisions']) == (180, 1036800, 259200)
                assert value['fixed_tensors_unchanged'] and value['teacher_unchanged']
                assert value['acquisition_monotonic_seconds'] <= 1800 and value['evaluation_monotonic_seconds'] <= 600
                protocol_binding(value, 'distillation')
                artifacts(base, value['artifacts'])
                assert set(('initial.pt', 'stage1.pt', 'stage2.pt', 'stage3.pt', 'final.pt',
                            'metrics.jsonl', 'final_live.json', 'final_paired/summary.json')) <= set(value['artifacts'])
                execution = read(f'results/native_turn_distillation/{condition}_{arm}_execution.json')
                assert execution['complete'] and execution['exit_code'] == 0 and execution['no_cloud']
                assert (execution['condition'], execution['arm']) == (condition, arm)
                protocol_binding(execution, 'distillation')
            check(f'complete acquisition and evaluation: {condition}/{arm}', training)

        for endpoint in ('original', *ARMS):
            def geometry(condition=condition, endpoint=endpoint):
                base = f'results/native_turn_geometry/{condition}/{endpoint}'
                value = read(f'{base}/summary.json')
                assert value['complete'] and (value['condition'], value['endpoint']) == (condition, endpoint)
                assert (value['contexts'], value['physical_families'], value['branches'], len(value['rows'])) == (16, 64, 4, 33280)
                protocol_binding(value, 'geometry')
                prove(f'results/native_turn_geometry/inputs/{condition}/summary.json', value['input_summary_sha256'])
                checkpoint = (f'models/{condition}/{condition}_lstm_s1_1000M.pt' if endpoint == 'original' else
                              f'results/native_turn_distillation/{condition}/{endpoint}/final.pt')
                prove(checkpoint, value['checkpoint_sha256'])
                artifacts(base, value['artifacts'])
                if endpoint == 'original':
                    execution = read(f'results/native_turn_geometry/{condition}_geometry_pipeline_execution.json')
                    assert execution['complete'] and [s['stage'] for s in execution['stages']] == ['prepare', 'original_geometry']
                    assert all(s['exit_code'] == 0 for s in execution['stages'])
                else:
                    execution = read(f'results/native_turn_geometry/{condition}_{endpoint}_geometry_execution.json')
                    assert execution['complete'] and execution['exit_code'] == 0
                    assert execution['condition'] == condition and execution['endpoint'] == endpoint
            check(f'complete dense geometry: {condition}/{endpoint}', geometry)

    def preliminary_reports():
        for kind, producer, auditor in (
            ('teacher_preflight', 'report_native_turn_teacher_preflight', 'audit_native_turn_teacher_preflight_outputs'),
            ('original_memory', 'report_native_turn_original_memory', 'audit_native_turn_original_memory_outputs')):
            base = f'results/native_turn_{kind}/report'
            report, audit = read(f'{base}/summary.json'), read(f'{base}/independent_validation.json')
            assert report['complete'] and audit['complete'] and set(audit['models']) == set(CONDITIONS)
            protocol_binding(report, kind); protocol_binding(audit, kind)
            prove(f'scripts/{producer}.py', report['source_sha256' if kind == 'teacher_preflight' else 'producer_sha256'])
            prove(f'scripts/{auditor}.py', audit['auditor_source_sha256' if kind == 'teacher_preflight' else 'audit_source_sha256'])
            for c, row in audit['models'].items():
                assert row['complete']
                prove(f'results/native_turn_{kind}/{c}/summary.json', row['summary_sha256'])
            if kind == 'original_memory':
                prove(f'{base}/summary.json', audit['report_summary_sha256'])
            archive.checked(root, f'{base}/README.md')
    check('complete preflight and original-memory reports and independent audits', preliminary_reports)

    def training_report():
        base = 'results/native_turn_distillation/report'
        report, audit = read(f'{base}/summary.json'), read(f'{base}/independent_validation.json')
        assert report['complete'] and audit['complete'] and set(report['models']) == set(CONDITIONS)
        protocol_binding(report, 'distillation'); protocol_binding(audit, 'distillation')
        prove('scripts/report_native_turn_distillation.py', report['producer_sha256'])
        prove('scripts/report_native_turn_distillation.py', audit['producer_sha256'])
        prove('scripts/audit_native_turn_distillation_outputs.py', audit['audit_source_sha256'])
        for name, expected in audit['independent_helper_source_sha256'].items():
            prove(name, expected)
        prove(f'{base}/summary.json', audit['report_summary_sha256'])
        for name, expected in report['terminal_summary_sha256'].items():
            prove(name, expected)
        expected_keys = {f'{c}/{a}' for c in CONDITIONS for a in ARMS}
        assert set(audit['endpoint_summary_sha256']) == set(audit['execution_sha256']) == expected_keys
        for key in expected_keys:
            c, a = key.split('/')
            assert set(report['models'][c]) == {'original', 'clean', 'noise'}
            prove(f'results/native_turn_distillation/{key}/summary.json', audit['endpoint_summary_sha256'][key])
            prove(f'results/native_turn_distillation/{c}_{a}_execution.json', audit['execution_sha256'][key])
        assert audit['no_new_rollout']
        assert audit['figures_visual_review']['complete']
        assert (audit['paired_cases'], audit['paired_pairs'], audit['live_episode_rows']) == (24576, 12288, 576)
        assert len(report['figures']) == 8
        artifacts(base, {n: dict(sha256=s) for n, s in report['figures'].items()})
        artifacts(base, audit['report_artifact_sha256'])
    check('complete four-arm report and independent audit', training_report)

    def imitation_decomposition():
        base = 'results/native_turn_imitation_decomposition'
        value = read(f'{base}/summary.json')
        assert value['complete'] and value['no_new_rollout'] and value['no_optimization']
        assert value['exploratory_descriptive_reanalysis']
        assert (len(value['aggregates']), len(value['groups']), len(value['improvements'])) == (144, 2304, 96)
        assert value['unique_saved_case_rows'] == 24576 and value['choice_reset_zero_gain_groups'] == 576
        prove('scripts/decompose_native_turn_imitation.py', value['producer_sha256'])
        prove('tests/test_decompose_native_turn_imitation.py', value['test_sha256'])
        for name, record in value['input_sha256'].items():
            prove(name, record)
        audit = read(f'{base}/independent_validation.json')
        assert audit['complete'] and audit['no_new_rollout'] and audit['no_model_or_environment_execution']
        assert (audit['groups'], audit['aggregates'], audit['improvements'],
                audit['choice_reset_zero_gain_groups'], audit['unique_raw_context_mode_files']) == (2304, 144, 96, 576, 192)
        prove('scripts/audit_native_turn_imitation_decomposition.py', audit['auditor_source_sha256'])
        prove('scripts/decompose_native_turn_imitation.py', audit['producer_sha256'])
        prove(f'{base}/summary.json', audit['summary_sha256'])
        prove('results/native_turn_distillation/report/independent_validation.json',
              audit['training_independent_audit_sha256'])
        artifacts(base, audit['report_artifact_sha256'])
        artifacts(base, read(f'{base}/artifacts.json'))
        design = 'results/native_turn_imitation_decomposition_design'
        revision = read(f'{design}/presentation_revision.json')
        assert revision['no_model_execution'] and revision['all_arithmetic_and_raw_source_hashes_unchanged']
        initial = revision['initial_render_preserved']
        assert initial == f'{design}/initial_render'
        prove(f'{initial}/summary.json', revision['previous_summary_sha256'])
        initial_summary = read(f'{initial}/summary.json')
        source = revision['preserved_initial_producer']
        prove(source['path'], source['sha256'])
        assert source['sha256'] == initial_summary['producer_sha256']
        assert revision['final_producer_sha256'] == value['producer_sha256']
        artifacts(initial, read(f'{initial}/artifacts.json'))
    check('complete saved-probability decomposition with exact audited input bindings', imitation_decomposition)

    def imitation_prefix():
        base = 'results/native_turn_imitation_prefix'
        value, audit = read(f'{base}/summary.json'), read(f'{base}/independent_validation.json')
        assert value['complete'] and audit['complete'] and value['no_new_rollout'] and value['no_optimization']
        assert value['exploratory_descriptive_reanalysis']
        assert (len(value['histories']), len(value['groups']), len(value['aggregates'])) == (64, 2304, 144)
        assert (audit['histories'], audit['groups'], audit['aggregates'],
                audit['choice_reset_zero_gain_groups'], audit['unique_raw_context_mode_files']) == (64, 2304, 144, 576, 192)
        assert audit['no_new_rollout'] and audit['no_model_or_environment_execution']
        prove('scripts/decompose_native_turn_prefix.py', value['producer_sha256'])
        prove('scripts/decompose_native_turn_prefix.py', audit['producer_sha256'])
        prove('scripts/audit_native_turn_imitation_prefix.py', audit['auditor_source_sha256'])
        prove('tests/test_decompose_native_turn_prefix.py', value['test_sha256'])
        prove(f'{base}/summary.json', audit['summary_sha256'])
        prove('results/native_turn_imitation_decomposition/summary.json', value['main_summary_sha256'])
        prove('results/native_turn_imitation_decomposition/independent_validation.json',
              audit['main_independent_audit_sha256'])
        for name, record in value['input_sha256'].items():
            prove(name, record)
        artifacts(base, audit['report_artifact_sha256'])
        artifacts(base, read(f'{base}/artifacts.json'))
        failure = read('results/native_turn_imitation_prefix_design/first_render_failure.json')
        assert failure['complete'] is False and failure['no_model_or_environment_run']
        prove(failure['source_snapshot'], failure['source_sha256'])
        assert failure['partial_artifacts_preserved'] == 'results/native_turn_imitation_prefix_design/first_render_failure'
        for extension in ('png', 'pdf'):
            archive.checked(root, f"{failure['partial_artifacts_preserved']}/prefix_comparison.{extension}")
    check('complete separately labeled prefix comparison and independent raw-array audit', imitation_prefix)

    for report_name, endpoints in (('report_original', ('original',)), ('report_all', ('original', *ARMS))):
        def geometry_report(report_name=report_name, endpoints=endpoints):
            base = f'results/native_turn_geometry/{report_name}'
            report, validation = read(f'{base}/summary.json'), read(f'{base}/validation.json')
            failure, audit = read(f'{base}/failure_patterns.json'), read(f'{base}/independent_validation.json')
            count = 2 * len(endpoints)
            assert all(d['complete'] for d in (report, validation, failure, audit))
            assert report['scope'] == list(endpoints) and report['models'] == count
            assert report['raw_rows'] == failure['raw_rows'] == 33280 * count
            assert len(validation['models']) == count and len(validation['inputs']) == 2
            prove(PROTOCOLS['geometry'], report['geometry_protocol_sha256'])
            prove('scripts/report_native_turn_geometry.py', report['source_sha256'])
            prove('scripts/report_native_turn_geometry.py', validation['source_sha256'])
            prove('scripts/report_native_turn_geometry_failures.py', failure['source_sha256'])
            prove(f'{base}/summary.json', validation['summary_sha256'])
            prove(f'{base}/validation.json', failure['validation_sha256'])
            for row in validation['models']:
                assert row['complete'] and row['rows'] == row['unique_keys'] == 33280
                prove(f"results/native_turn_geometry/{row['condition']}/{row['endpoint']}/summary.json", row['summary_sha256'])
            assert audit['raw_rows'] == 33280 * count
            assert audit['figures_visually_reviewed']
            assert audit['independently_recomputed_native_plot_points'] == 192 * count
            assert {(r['condition'], r['endpoint']) for r in audit['models']} == {
                (c, e) for c in CONDITIONS for e in endpoints}
            prove(f'{base}/summary.json', audit['summary_sha256'])
            prove(f'{base}/validation.json', audit['validation_sha256'])
            prove(f'{base}/failure_patterns.json', audit['failure_patterns_sha256'])
            prove('scripts/audit_native_turn_geometry_report.py', audit['audit_source_sha256'])
            artifacts(base, read(f'{base}/artifacts.json'))
            if report_name == 'report_all':
                addendum = read(f'{base}/presentation_addendum.json')
                verify_presentation_addendum(archive.checked(root, f'{base}/README.md').read_bytes(), addendum)
                reviewed = read(f'{base}/presentation_addendum_independent_validation.json')
                assert reviewed['complete'] and reviewed['presentation_only'] and reviewed['no_model_or_environment_execution']
                prove(f'{base}/README.md', reviewed['README_sha256'])
                prove(f'{base}/presentation_addendum.json', reviewed['presentation_addendum_sha256'])
                prove(f'{base}/independent_validation.json', reviewed['numerical_independent_audit_sha256'])
        check('complete saved-array report and independent audit: ' + report_name, geometry_report)

    def scheduler():
        value = read('results/native_turn_distillation/remaining_scheduler_execution.json')
        prove(value['source_path'], value['source_sha256'])
        assert value['status'] == 'complete', 'Operational scheduler has not completed.'
    check('terminal operational record with exact scheduler source', scheduler)
    for tree in TREES:
        check('result tree: ' + tree, lambda tree=tree: names.update(archive.tree_files(root, tree)))
    entries = []
    for name in sorted(names):
        def member(name=name):
            path = archive.check_name_and_content(root, name)
            entries.append(dict(path=name, bytes=path.stat().st_size, sha256=archive.sha(path)))
        check('member: ' + name, member)
    if commit:
        assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'Full commit SHA required'
        for name in sorted(sources | {n for n in DOCS if not n.startswith('results/')}):
            def committed(name=name):
                assert archive.sha(archive.checked(root, name)) == archive.commit_sha(root, commit, name), name
            check('source commit: ' + name, committed)
    else:
        issues.append(dict(check='source_commit', reason='Final reproduction commit required before build.'))
    return dict(version='native_turn_outputs_v1', source_commit=commit,
        completion=dict(complete=not issues, issues=issues, passed_checks=checks),
        scientific_success_required_for_packaging=False,
        scope='Preflight, original-memory controls, four fixed acquisitions, six 128-angle geometry endpoints, saved-probability decomposition and prefix comparison; prospective higher-resolution study excluded.',
        result_trees=list(TREES), source_files=sorted(sources), file_count=len(entries),
        uncompressed_bytes=sum(e['bytes'] for e in entries), files=entries)


def build(root, manifest, out, **limits):
    out = Path(out)
    result = archive.build(root, manifest, out, **limits)
    for part in result['parts']:
        old = part['path']
        assert old.startswith('nonlinear-ring-outputs-')
        new = old.replace('nonlinear-ring-outputs-', 'native-turn-outputs-', 1)
        (out / old).rename(out / new)
        part['path'] = new
    (out / 'index.json').write_bytes(archive.json_bytes(result))
    archive.verify_release(out)
    return result


def self_test():
    result = archive.self_test()
    before, after = b'Beginning\nPrevious wording\nEnd\n', b'Beginning\nCorrected wording\nEnd\n'
    addendum = dict(complete=True, file='README.md', previous_text='Previous wording',
                    updated_text='Corrected wording', previous_sha256=hashlib.sha256(before).hexdigest(),
                    updated_sha256=hashlib.sha256(after).hexdigest())
    assert verify_presentation_addendum(after, addendum) == before
    for content, record in ((after + b'extra', addendum),
                            (after, dict(addendum, previous_sha256='0' * 64))):
        try:
            verify_presentation_addendum(content, record)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unrecorded presentation mutation accepted')
    result['checks'].append('exact presentation replacement and both hashes; unrelated mutations rejected')
    with tempfile.TemporaryDirectory(prefix='native-turn-package-test-') as directory:
        root = Path(directory); rows = []
        for name in ('a.txt', 'b.txt'):
            path = root / name; path.write_bytes(b'x')
            rows.append(dict(path=name, bytes=1, sha256=archive.sha(path)))
        manifest = dict(version='synthetic', source_commit='a' * 40, completion=dict(complete=True), files=rows)
        index = build(root, manifest, root / 'parts', payload_limit=1, asset_limit=10000)
        assert len(index['parts']) == 2 and all(p['path'].startswith('native-turn-outputs-') for p in index['parts'])
        assert archive.verify_release(root / 'parts')['files'] == 2
        assert not list((root / 'parts').glob('nonlinear-ring-outputs-*'))
    result['checks'].append('native-turn part names, updated index and independently verified members')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    for name in ('plan', 'build', 'verify-release', 'self-test'):
        action.add_argument('--' + name, action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--out', type=Path, default=ROOT / RELEASE / 'archives')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.verify_release:
        result = archive.verify_release(args.out)
    else:
        result = inventory(ROOT, args.source_commit)
        if args.build:
            assert result['completion']['complete'], json.dumps(result['completion']['issues'], indent=2)
            result = build(ROOT, result, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
