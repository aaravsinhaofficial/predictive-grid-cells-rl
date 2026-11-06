"""Synthetic metadata and byte archives only; no study model or bank is loaded."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import package_goal_cue_memory_outputs as pack


def put(root, name, value):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pack.json_bytes(value) if isinstance(value, dict) else value)
    return path


def proof(path):
    return dict(sha256=pack.sha(path), bytes=path.stat().st_size)


def complete_fixture(root, monkeypatch):
    sources = {}
    for name in ('scripts/audit_goal_cue_memory.py', 'scripts/audit_goal_cue_memory_original_diagnostic.py', *pack.TOOLS):
        sources[name] = pack.sha(put(root, name, b'# synthetic source only\n'))
    banks = {name:dict(path=f'{pack.DESIGN}/banks/{name}.pt', **proof(put(root,
        f'{pack.DESIGN}/banks/{name}.pt', b'not a neural bank'))) for name in ('qualification', 'memory')}
    parent_name = 'docs/research/goal_cue_memory_training_protocol.json'
    parent_sha = pack.sha(put(root, parent_name, dict(evaluation=dict(banks=banks))))
    variants = ['continuous', *[f'branch{i}' for i in range(10)]]
    design = dict(frozen_utc='fixed synthetic time', conditions=list(pack.CONDITIONS),
        stopped_parent_protocol=dict(path=parent_name, sha256=parent_sha,
            paired_training_qualification_passed=False, training_remains_prohibited=True),
        evaluation=dict(variants=variants), source_sha256=sources,
        input_assets={parent_name:parent_sha, **{b['path']:b['sha256'] for b in banks.values()}})
    protocol_sha = pack.sha(put(root, pack.PROTOCOL, design))
    monkeypatch.setattr(pack, 'PROTOCOL_SHA', protocol_sha)
    monkeypatch.setattr(pack, 'commit_sha', lambda root, commit, name:pack.sha(root/name))
    for phase in ('qualification', 'original_diagnostic'):
        proto = parent_sha if phase == 'qualification' else protocol_sha
        names = ['continuous'] if phase == 'qualification' else variants
        runs = []
        for condition in pack.CONDITIONS:
            folder = f'{pack.BASE}/{phase}/{condition}'
            sub = 'qualification_evaluation' if phase == 'qualification' else 'evaluation'
            for name in ('incoming.pt', 'qualification_after.pt' if phase == 'qualification' else 'after.pt',
                         'qualification_evaluation_integrity.json' if phase == 'qualification' else 'evaluation_integrity.json'):
                put(root, f'{folder}/{name}', b'synthetic immutable snapshot evidence')
            contexts = []
            for c in range(16):
                prefix = f'{folder}/{sub}/context{c}'
                put(root, f'{prefix}/input_plan.pt.gz', b'synthetic plan')
                for variant in names:
                    for name, data in (('input.json', {}), ('result.json', {'successes':0}), ('raw.pt.gz', b'synthetic raw bytes')):
                        put(root, f'{prefix}/{variant}/{name}', data)
                    contexts.append(dict(context=c, variant=variant))
            summary = dict(phase=phase, condition=condition, complete=True, no_optimization=True,
                protocol_sha256=proto, totals=dict(trials=256, successes=0), qualified=False,
                variant_order=variants, variants={name:dict(trials=256, successes=0) for name in variants},
                no_optimizer_created=True, parent_training_gate_still_failed=True,
                evaluator_descriptive_screens={'useful_memory':False})
            summary_path = put(root, f'{folder}/summary.json', summary)
            artifacts = {p.relative_to(root/folder).as_posix():proof(p) for p in (root/folder).rglob('*') if p.is_file()}
            auditor = 'scripts/audit_goal_cue_memory' + ('_original_diagnostic' if phase == 'original_diagnostic' else '') + '.py'
            put(root, f'{pack.BASE}/{phase}/audits/{condition}.json', dict(
                complete=True, passed=True, status='complete', failures=[], pending=[],
                phase=phase, condition=condition, protocol_sha256=proto, auditor_sha256=sources[auditor],
                frozen_numerical_auditor_sha256=sources['scripts/audit_goal_cue_memory.py'],
                parent_training_gate_still_failed=True, no_automatic_neural_nomination=True,
                checks=dict(evaluation=dict(complete=True, contexts=contexts)), artifacts=artifacts))
            log = put(root, f'{pack.DESIGN}/{phase}_execution_logs/{condition}.log', b'synthetic completed writer')
            runs.append(dict(condition=condition, writer_complete=True, exit_code=0, no_optimization=True,
                log_path=log.relative_to(root).as_posix(), log_sha256=pack.sha(log),
                summary_path=summary_path.relative_to(root).as_posix(), summary_sha256=pack.sha(summary_path)))
        put(root, f'{pack.DESIGN}/{phase}_execution.json', dict(complete=True, protocol_sha256=proto, runs=runs))
        for name, content in (('README.md', b'Synthetic report'), ('summary.json', {}), ('figure.png', b'not a rendered figure')):
            put(root, f'{pack.BASE}/{phase}/report/{name}', content)
    for name in pack.EXTRA_DOCS:
        put(root, name, b'Synthetic prospective note')
    put(root, 'results/goal_cue_memory_release/REPRODUCE.md', b'Synthetic guide')
    return design


def test_completed_negative_scientific_results_are_archived_without_side_effects(tmp_path, monkeypatch):
    complete_fixture(tmp_path, monkeypatch)
    before = {p.relative_to(tmp_path).as_posix():proof(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = pack.inventory(tmp_path, 'a'*40)
    assert result['parent_training_qualification_passed'] is False
    assert result['scientific_success_required_for_packaging'] is False
    assert len(result['audit_records']) == 4
    assert set(before) == {e['path'] for e in result['files']}
    assert before == {p.relative_to(tmp_path).as_posix():proof(p) for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure', ['missing_audit', 'pending_audit', 'raw_corruption', 'source_corruption', 'omitted_raw',
                                   'crossed_writer_summary', 'duplicated_writer_log'])
def test_incomplete_or_changed_artifacts_are_rejected(tmp_path, monkeypatch, failure):
    complete_fixture(tmp_path, monkeypatch)
    audit = tmp_path/f'{pack.BASE}/original_diagnostic/audits/pgc33.json'
    raw = tmp_path/f'{pack.BASE}/original_diagnostic/pgc33/evaluation/context0/continuous/raw.pt.gz'
    if failure == 'missing_audit':
        audit.unlink()
    elif failure == 'pending_audit':
        value = pack.read(audit); value.update(complete=False, status='pending', pending=['after.pt'])
        audit.write_bytes(pack.json_bytes(value))
    elif failure == 'raw_corruption':
        raw.write_bytes(b'changed bytes')
    elif failure == 'source_corruption':
        (tmp_path/'scripts/audit_goal_cue_memory.py').write_bytes(b'changed source')
    elif failure in ('crossed_writer_summary', 'duplicated_writer_log'):
        path = tmp_path/f'{pack.DESIGN}/original_diagnostic_execution.json'
        record = pack.read(path); key = 'summary' if failure == 'crossed_writer_summary' else 'log'
        for suffix in ('_path', '_sha256'):
            record['runs'][1][key+suffix] = record['runs'][0][key+suffix]
        path.write_bytes(pack.json_bytes(record))
    else:
        raw.unlink()
    with pytest.raises(AssertionError):
        pack.inventory(tmp_path, 'a'*40)


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'x/../escape', 'x//y', 'x\\y'])
def test_unsafe_relative_paths_are_rejected(tmp_path, name):
    with pytest.raises(AssertionError):
        pack.checked(tmp_path, name)


def test_symlink_file_and_ancestor_are_rejected(tmp_path):
    put(tmp_path, 'real/file', b'data')
    (tmp_path/'link').symlink_to(tmp_path/'real', target_is_directory=True)
    (tmp_path/'filelink').symlink_to(tmp_path/'real/file')
    for name in ('link/file', 'filelink'):
        with pytest.raises(AssertionError, match='Symlink'):
            pack.checked(tmp_path, name)
    with pytest.raises(AssertionError, match='Symlink'):
        pack.tree_files(tmp_path, 'link')


def small_manifest(root):
    entries = []
    for name, data in [('z.txt', b'ccc'), ('a.txt', b'aaa'), ('middle/b.txt', b'bbb')]:
        entries.append(dict(path=name, **proof(put(root, name, data))))
    return dict(version='synthetic', source_commit='a'*40, files=entries)


def test_parts_are_sorted_deterministic_and_every_member_verifies(tmp_path):
    manifest = small_manifest(tmp_path)
    first = pack.build(tmp_path, manifest, tmp_path/'one', payload_limit=4, asset_limit=10000)
    for entry in manifest['files']:
        os.utime(tmp_path/entry['path'], (111, 222)); os.chmod(tmp_path/entry['path'], 0o600)
    second = pack.build(tmp_path, manifest, tmp_path/'two', payload_limit=4, asset_limit=10000)
    assert first == second and len(first['parts']) == 3
    assert [p['members'] for p in first['parts']] == [['a.txt'], ['middle/b.txt'], ['z.txt']]
    for part in first['parts']:
        a, b = tmp_path/'one'/part['path'], tmp_path/'two'/part['path']
        assert a.read_bytes() == b.read_bytes() and a.stat().st_size <= 10000
        assert a.read_bytes()[4:8] == b'\0'*4  # gzip mtime
        entries = [e for e in manifest['files'] if e['path'] in part['members']]
        pack.verify_archive(a, entries, pack.json_bytes(manifest))


def test_payload_and_actual_compressed_size_limits_are_enforced(tmp_path):
    with pytest.raises(AssertionError, match='Single member'):
        pack.partition([dict(path='large', bytes=5)], limit=4)
    with pytest.raises(AssertionError, match='Compressed asset'):
        pack.build(tmp_path, small_manifest(tmp_path), tmp_path/'small', asset_limit=1)


def test_archive_corruption_and_source_changes_after_inventory_are_rejected(tmp_path):
    manifest = small_manifest(tmp_path)
    (tmp_path/'a.txt').write_bytes(b'changed')
    with pytest.raises(AssertionError, match='Hash differs'):
        pack.build(tmp_path, manifest, tmp_path/'changed')
    (tmp_path/'a.txt').write_bytes(b'aaa')
    body = pack.json_bytes(manifest)
    corrupt = tmp_path/'corrupt.tar.gz'
    with tarfile.open(corrupt, 'w:gz') as archive:
        for name, data in sorted([(e['path'], b'bad') for e in manifest['files']] + [(pack.MANIFEST_NAME, body)]):
            info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    with pytest.raises(AssertionError, match='Archived bytes differ'):
        pack.verify_archive(corrupt, manifest['files'], body)
