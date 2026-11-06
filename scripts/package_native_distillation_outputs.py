"""Package the complete fixed distillation pilot and prospective memory assays.

Uses the already verified deterministic multipart writer. No model is executed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import package_nonlinear_ring_outputs as archive

ROOT = Path(__file__).resolve().parents[1]
RELEASE = 'results/native_distillation_release'
TREES = ('results/native_policy_distillation', 'results/lstm_distillation_memory')
archive.TREES = TREES
archive.MANIFEST_NAME = f'{RELEASE}/manifest.json'
DOCS = (
    'README.md', 'mechanisms/learned-ring-in-navigation.md',
    'docs/research/native_policy_distillation_pilot_design.md',
    'docs/research/native_policy_distillation_protocol.json',
    'docs/research/native_policy_distillation_memory_commands.md',
    'docs/research/native_distillation_next_design_audit.md',
    'docs/research/ring_local_environment.json',
    'requirements-ring-analysis.txt',
    f'{RELEASE}/REPRODUCE.md')
EXTRA_SOURCE = (
    'scripts/package_native_distillation_outputs.py',
    'scripts/report_native_policy_distillation.py',
    'scripts/lstm_distillation_memory_report.py')


def inventory(root, commit):
    root = Path(root).resolve()
    names = set(filter(None, subprocess.check_output(
        ['git', 'ls-files', '-z', 'code', 'scripts', 'tests'], cwd=root).decode().split('\0')))
    sources = names | set(EXTRA_SOURCE)
    names = sources | set(DOCS)
    issues = []
    checks = []

    def check(label, action):
        try:
            action()
            checks.append(label)
        except (AssertionError, OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(check=label, reason=str(error) or type(error).__name__))

    def read(name):
        return archive.read(archive.checked(root, name))

    def training_protocol():
        p = read('docs/research/native_policy_distillation_protocol.json')
        for key in ('source_sha256', 'asset_sha256'):
            for name, expected in p[key].items():
                names.add(name)
                archive.prove(root, name, dict(sha256=expected))
        assert p['total_updates'] == 180 and p['physics_frames_per_model'] == 1036800
    check('frozen training sources and original inputs', training_protocol)

    def geometry_protocol():
        p = read('results/lstm_distillation_memory/protocol.json')
        for key in ('source_sha256', 'original_asset_sha256'):
            for name, expected in p[key].items():
                names.add(name)
                archive.prove(root, name, dict(sha256=expected))
        archive.prove(root, 'docs/research/native_policy_distillation_protocol.json',
                      dict(sha256=p['training_protocol_sha256']))
    check('frozen geometry sources and inputs', geometry_protocol)

    for condition in ('gc100', 'pgc33'):
        def training(condition=condition):
            base = f'results/native_policy_distillation/{condition}_s1'
            s = read(f'{base}/summary.json')
            assert s['complete'] and s['condition'] == condition and s['updates'] == 180
            assert s['learning_physics_frames'] == 1036800
            assert s['frozen_tensors_unchanged'] and s['teacher_unchanged']
            for name, row in s['artifacts'].items():
                archive.prove(root, f'{base}/{name}', row)
            for name in ('initial.pt', 'stage1.pt', 'stage2.pt', 'stage3.pt', 'final.pt',
                         'baseline_behavior.json', 'final_behavior.json',
                         'baseline_replay.json', 'final_replay.json'):
                assert name in s['artifacts'], name
            archive.prove(root, 'docs/research/native_policy_distillation_protocol.json',
                          dict(sha256=s['protocol_sha256']))
        check('complete training ' + condition, training)
        for endpoint, count in (('initial', 4), ('final', 8)):
            def geometry(condition=condition, endpoint=endpoint, count=count):
                base = f'results/lstm_distillation_memory/{condition}/{endpoint}'
                s = read(f'{base}/summary.json')
                assert s['complete'] and s['condition'] == condition and s['endpoint'] == endpoint
                assert s['contexts'] == count and len(s['results']) == count
                archive.prove(root, 'results/lstm_distillation_memory/protocol.json',
                              dict(sha256=s['protocol_sha256']))
                archive.prove(root, f'results/native_policy_distillation/{condition}_s1/{endpoint}.pt',
                              dict(sha256=s['checkpoint_sha256']))
            check('complete geometry ' + condition + ' ' + endpoint, geometry)

    for name in ('results/native_policy_distillation/report/README.md',
                 'results/native_policy_distillation/report/summary.json',
                 'results/lstm_distillation_memory/README.md',
                 'results/native_policy_distillation/runtime_clock_audit.json'):
        check(name, lambda name=name: archive.checked(root, name))
    def training_report():
        base = 'results/native_policy_distillation'
        value = read(f'{base}/report/summary.json')
        assert value['complete'] and set(value['models']) == {'gc100', 'pgc33'}
        archive.prove(root, 'docs/research/native_policy_distillation_protocol.json',
                      dict(sha256=value['protocol_sha256']))
        archive.prove(root, 'scripts/report_native_policy_distillation.py',
                      dict(sha256=value['source_sha256']))
        for condition, model in value['models'].items():
            archive.prove(root, f'{base}/{condition}_s1/summary.json',
                          dict(sha256=model['summary_sha256']))
        execution = read(f'{base}/execution.json')
        assert execution['complete'] and set(execution['models']) == {'gc100', 'pgc33'}
        assert all(model['exit_code'] == 0 for model in execution['models'].values())
        archive.prove(root, f'{base}/execution.json', dict(sha256=value['execution_sha256']))
        archive.prove(root, f'{base}/runtime_clock_audit.json',
                      dict(sha256=value['runtime_clock_audit_sha256']))
        audit = read(f'{base}/report/independent_audit.json')
        assert audit['complete'] and set(audit['endpoint_summary_sha256']) == {'gc100', 'pgc33'}
        archive.prove(root, f'{base}/execution.json', dict(sha256=audit['execution_sha256']))
        archive.prove(root, f'{base}/report/summary.json',
                      dict(sha256=audit['report_summary_sha256']))
        archive.prove(root, 'docs/research/native_policy_distillation_protocol.json',
                      dict(sha256=audit['training_protocol_sha256']))
        for condition, expected in audit['endpoint_summary_sha256'].items():
            archive.prove(root, f'{base}/{condition}_s1/summary.json', dict(sha256=expected))
        assert audit['figures_visual_review']['complete']
        assert len(audit['report_artifact_sha256']) == 8
        for name, record in audit['report_artifact_sha256'].items():
            archive.prove(root, f'{base}/report/{name}', record)
    check('completed training report, CLI exits and independent audit', training_report)
    def geometry_report():
        value = read('results/lstm_distillation_memory/aggregate.json')
        assert value['complete'] and len(value['contexts']) == 24 and len(value['rows']) == 720
        archive.prove(root, 'results/lstm_distillation_memory/protocol.json',
                      dict(sha256=value['protocol_sha256']))
        archive.prove(root, 'scripts/lstm_distillation_memory_report.py',
                      dict(sha256=value['source_sha256']))
        execution = read('results/lstm_distillation_memory/execution.json')
        assert execution['complete']
        archive.prove(root, 'results/lstm_distillation_memory/aggregate.json',
                      dict(sha256=execution['aggregate_sha256']))
        archive.prove(root, 'results/lstm_distillation_memory/protocol.json',
                      dict(sha256=execution['protocol_sha256']))
        audit = read('results/lstm_distillation_memory/independent_audit.json')
        assert audit['complete'] and audit['contexts'] == 24 and audit['unique_rows'] == 720
        archive.prove(root, 'results/lstm_distillation_memory/aggregate.json',
                      dict(sha256=audit['aggregate_sha256']))
        archive.prove(root, 'results/lstm_distillation_memory/execution.json',
                      dict(sha256=audit['execution_sha256']))
    check('complete geometry report without selection', geometry_report)
    for tree in TREES:
        check(tree, lambda tree=tree: names.update(archive.tree_files(root, tree)))

    entries = []
    for name in sorted(names):
        def member(name=name):
            path = archive.check_name_and_content(root, name)
            entries.append(dict(path=name, bytes=path.stat().st_size, sha256=archive.sha(path)))
        check('member:' + name, member)
    if commit:
        assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'Full commit SHA required'
        for name in sorted(sources | {n for n in DOCS if not n.startswith('results/')}):
            def committed(name=name):
                assert archive.sha(archive.checked(root, name)) == archive.commit_sha(root, commit, name), name
            check('source_commit:' + name, committed)
    else:
        issues.append(dict(check='source_commit', reason='Final source commit is required to build.'))
    return dict(version='native_distillation_outputs_v1', source_commit=commit,
        completion=dict(complete=not issues, issues=issues, passed_checks=checks),
        scientific_success_required_for_packaging=False, result_trees=list(TREES),
        source_files=sorted(sources), file_count=len(entries),
        uncompressed_bytes=sum(e['bytes'] for e in entries), files=entries)


def build(root, manifest, out, **limits):
    out = Path(out)
    result = archive.build(root, manifest, out, **limits)
    for part in result['parts']:
        old = part['path']
        assert old.startswith('nonlinear-ring-outputs-')
        new = old.replace('nonlinear-ring-outputs-', 'native-distillation-outputs-', 1)
        (out/old).rename(out/new)
        part['path'] = new
    (out/'index.json').write_bytes(archive.json_bytes(result))
    archive.verify_release(out)
    return result


def self_test():
    result = archive.self_test()
    with tempfile.TemporaryDirectory(prefix='distillation-package-test-') as directory:
        root = Path(directory)
        rows = []
        for name in ('a.txt', 'b.txt'):
            path = root/name
            path.write_bytes(b'x')
            rows.append(dict(path=name, bytes=1, sha256=archive.sha(path)))
        manifest = dict(version='synthetic', source_commit='a'*40,
                        completion=dict(complete=True), files=rows)
        index = build(root, manifest, root/'parts', payload_limit=1, asset_limit=10000)
        assert len(index['parts']) == 2
        assert all(p['path'].startswith('native-distillation-outputs-') for p in index['parts'])
        assert not list((root/'parts').glob('nonlinear-ring-outputs-*'))
        assert archive.verify_release(root/'parts')['files'] == 2
    result['checks'].append('distinct multipart names and updated index verify together')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan', action='store_true')
    action.add_argument('--build', action='store_true')
    action.add_argument('--verify-release', action='store_true')
    action.add_argument('--self-test', action='store_true')
    parser.add_argument('--source-commit')
    parser.add_argument('--out', type=Path, default=ROOT/RELEASE/'archives')
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
