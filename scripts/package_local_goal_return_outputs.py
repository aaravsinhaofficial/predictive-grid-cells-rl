"""Prepare the complete local-return output asset, only after final cleanup.

No cloud access, learning, policy rollout, result selection or publication.
Every experiment file is retained, including explicit resumed/failed evidence.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd4b672a1cf510574420de1a512bb38e024b8919e'
PROTOCOL = 'docs/research/local_goal_return_training_protocol.json'
PROTOCOL_SHA = '0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
ROUTING = 'docs/research/local_goal_return_training_launch.json'
CLEANUP = 'results/local_goal_return_design/cloud_audit/cleanup_verification.json'
OUT_FOLDER = 'results/ring_local_goal_return'
CONDITIONS = ('gc100', 'pgc33')
CHECKPOINTS = ('incoming.pt', 'latest.pt', *(name for i in range(1, 5)
    for name in (f'stage{i}_before_evaluation.pt', f'stage{i}.pt')), 'final.pt')
TOOL_SOURCES = ('scripts/audit_local_goal_return_training.py',
    'scripts/audit_local_goal_return_baseline.py', 'scripts/plot_local_goal_return_training.py',
    'scripts/monitor_local_goal_return_training.py')
REPORT_FILES = ('summary.json', 'README.md', 'local_return.png', 'local_return.pdf', 'training.png', 'training.pdf')
SECRET_NAME = re.compile(r'(^\.env($|\.)|credential|client_secret|service.account|id_rsa|\.pem$|\.key$)', re.I)
SECRET_TEXT = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
    r'"(?:private_key|client_secret|refresh_token|access_token)"\s*:\s*"[^"\s]+', re.I)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_relative(value):
    path = Path(value)
    assert str(value) and path.parts and not path.is_absolute() and '..' not in path.parts
    return path


def checked_path(root, relative):
    root = Path(root).resolve(); relative = safe_relative(relative)
    path = root
    for part in relative.parts:
        path /= part
        assert not path.is_symlink(), f'No symlinks in package paths: {path}'
    assert path.resolve().is_relative_to(root), path
    return path


def inspect_file(path):
    assert path.is_file() and not path.is_symlink()
    assert not SECRET_NAME.search(path.name), f'Credential-like file is not a study output: {path}'
    if path.suffix in ('.json', '.jsonl', '.log', '.txt', '.md'):
        assert not SECRET_TEXT.search(path.read_text()), f'Potential credential material: {path}'


def experiment_files(root, relative):
    """Full tree or explicit failure: never silently drop unexpected evidence."""
    folder = checked_path(root, relative); assert folder.is_dir()
    paths = []
    for current, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs:
            assert not (Path(current)/name).is_symlink(), 'Symlinked evidence directory'
        for name in files:
            path = Path(current)/name
            assert path.suffix in ('.pt', '.json', '.jsonl', '.log', '.txt'), f'Unexpected experiment artifact: {path}'
            inspect_file(path); paths.append(path)
    assert paths, f'Empty experiment tree: {folder}'
    return sorted(paths)


def routed_attempts(routing):
    assert routing['protocol_sha256'] == PROTOCOL_SHA
    final = routing['workers']; prior = routing.get('prior_attempts', [])
    assert len(final) == 2 and {r['condition'] for r in final} == set(CONDITIONS)
    rows = final+prior
    assert len({r['name'] for r in rows}) == len(rows), 'Duplicate attempt routing'
    for row in rows:
        assert row['condition'] in CONDITIONS and row['name'].startswith('ring-20261105-local-return-')
        assert re.fullmatch(r'[a-z0-9-]+', row['name']) and isinstance(row['zone'], str)
        assert safe_relative(row['local_artifact_root']) == Path('results/ring_cloud_artifacts')/row['name']
        safe_relative(row['artifact_subdir'])
    return final, prior


def verify_cleanup(root, routing, cleanup):
    final, prior = routed_attempts(routing); expected = {r['name']:r for r in final+prior}
    assert cleanup['protocol_sha256'] == PROTOCOL_SHA and cleanup['complete'] is True
    assert cleanup['verified_utc']
    rows = cleanup['workers']; assert len(rows) == len(expected)
    assert {r['name'] for r in rows} == set(expected), 'Cleanup must cover every explicitly routed attempt'
    paths = []
    for row in rows:
        route = expected[row['name']]
        assert (row['condition'], row['zone']) == (route['condition'], route['zone'])
        assert row['instance_absent'] is True and row['disk_absent'] is True
        evidence = row['evidence']; kinds = set()
        for item in evidence:
            relative = safe_relative(item['path'])
            assert relative.parts[:3] == ('results', 'local_goal_return_design', 'cloud_audit')
            path = checked_path(root, relative); inspect_file(path)
            assert sha(path) == item['sha256']
            content = path.read_text()
            assert row['name'] in content and ('was not found' in content or 'notFound' in content)
            for kind in ('instances', 'disks'):
                if path.name == f"{row['name']}_{kind}_absence.log":
                    assert kind in content; kinds.add(kind)
            paths.append(path)
        assert kinds == {'instances', 'disks'}, 'Both explicit provider absence checks are required'
    return paths


def verify_final_audit(root, folder, condition, report):
    assert report['passed'] and report['complete'] and report['status'] == 'complete'
    assert not report['failures'] and not report['pending']
    assert report['condition'] == condition and report['protocol_sha256'] == PROTOCOL_SHA
    relative = folder.relative_to(root)
    assert Path(report['folder']).parts[-len(relative.parts):] == relative.parts
    assert report['auditor_sha256'] == sha(root/'scripts/audit_local_goal_return_training.py')
    for name, digest in report['audit_dependency_sha256'].items():
        assert sha(checked_path(root, name)) == digest, name
    checks = report['checks']
    assert checks['final_exact_stage4'] is True and checks['final_summary'] is True
    for stage in range(1, 5):
        assert checks[f'stage{stage}_evaluation_preserved_runtime'] is True
        assert f'stage{stage}_raw_evaluation' in checks
        assert f'stage{stage}_before_evaluation.pt' in checks and f'stage{stage}.pt' in checks
    required = set(CHECKPOINTS)|{'protocol.json', 'training_config.json', 'initialization.json', 'metrics.jsonl', 'summary.json'}
    required.update(f'stage{i}_evaluation/summary.json' for i in range(1, 5))
    required.update(f'stage{i}_evaluation_integrity.json' for i in range(1, 5))
    for stage in range(1, 5):
        evaluated = checks[f'stage{stage}_raw_evaluation']
        assert len(evaluated['contexts']) == 16 and len(evaluated['variants']) == 12
        assert sorted(r['context'] for r in evaluated['contexts']) == list(range(16))
        for context in evaluated['contexts']:
            prefix = f"stage{stage}_evaluation/context{context['context']}/"
            required.update(prefix+name for name in ('input.json', 'result.json'))
            required.add(prefix+('raw.pt' if context['valid'] else 'failed_plan.pt'))
    assert required <= set(report['artifacts']), 'Final audit is missing fixed endpoint evidence'
    for name, item in report['artifacts'].items():
        path = checked_path(folder, name); inspect_file(path)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    summary = read(folder/'summary.json')
    assert summary['completed'] and summary['protocol_sha256'] == PROTOCOL_SHA
    counters = summary['counters']
    assert (counters['updates'], counters['trials'], counters['completed_stages']) == (4096, 131072, 4)
    assert summary['final_checkpoint_sha256'] == sha(folder/'final.pt')


def inventory(root=ROOT, source_commit=COMMIT):
    root = Path(root).resolve()
    protocol_path = checked_path(root, PROTOCOL); assert sha(protocol_path) == PROTOCOL_SHA
    protocol = read(protocol_path); routing = read(root/ROUTING)
    final, prior = routed_attempts(routing)
    # This owner-written proof must exist before any output asset can be made.
    cleanup_path = checked_path(root, CLEANUP)
    paths = verify_cleanup(root, routing, read(cleanup_path))
    paths += [protocol_path, root/ROUTING, cleanup_path, root/OUT_FOLDER/'REPRODUCE.md']
    for name, digest in {**protocol['source_sha256'], **protocol['original_assets'], **protocol['passed_baseline_assets']}.items():
        assert sha(checked_path(root, name)) == digest, name
    sources = sorted(set(protocol['source_sha256'])|set(TOOL_SOURCES))
    for name in sources:
        path = checked_path(root, name)
        committed = subprocess.check_output(['git', 'show', f'{source_commit}:{name}'], cwd=root)
        assert hashlib.sha256(committed).hexdigest() == sha(path), f'Choose a reporting commit containing exact source: {name}'
        paths.append(path)
    paths.append(root/'scripts/package_local_goal_return_outputs.py')
    paths.append(root/'tests/test_package_local_goal_return_outputs.py')
    historical_audits = []
    for condition in CONDITIONS:
        for stage in range(1, 5):
            path = root/f'results/local_goal_return_design/cloud_audit/stage{stage}_{condition}.json'
            if path.exists():
                old = read(path)
                assert old['condition'] == condition
                if 'protocol_sha256' in old: assert old['protocol_sha256'] == PROTOCOL_SHA
                paths.append(path); historical_audits.append(str(path.relative_to(root)))
    implementation = root/'results/local_goal_return_design/cloud_audit/implementation_validation.json'
    if implementation.exists(): paths.append(implementation)
    models = []
    for row in final:
        folder = checked_path(root, safe_relative(row['local_artifact_root'])/safe_relative(row['artifact_subdir']))
        audit_path = root/f"results/local_goal_return_design/cloud_audit/{row['condition']}.json"
        report = read(audit_path); verify_final_audit(root, folder, row['condition'], report)
        paths.extend(experiment_files(root, folder.relative_to(root)))
        worker_root = checked_path(root, row['local_artifact_root'])
        worker_result = worker_root/'WORKER_RESULT.json'
        assert read(worker_result)['exit_code'] == 0
        paths += [audit_path, worker_result]
        environment = root/f"results/local_goal_return_design/cloud_audit/{row['name']}_environment.json"
        if environment.exists(): paths.append(environment)
        if (worker_root/'worker.log').exists(): paths.append(worker_root/'worker.log')
        models.append(dict(condition=row['condition'], folder=str(folder.relative_to(root)),
            final_sha256=sha(folder/'final.pt'), final_audit_sha256=sha(audit_path),
            fixed_checkpoints=list(CHECKPOINTS)))
    for row in prior:
        # A routing record may document a failed launch with no output directory.
        folder = checked_path(root, safe_relative(row['local_artifact_root'])/safe_relative(row['artifact_subdir']))
        if folder.exists(): paths.extend(experiment_files(root, folder.relative_to(root)))
        else: assert row.get('status') == 'launch_failed' and row.get('no_training_started') is True
        result = checked_path(root, safe_relative(row['local_artifact_root'])/'WORKER_RESULT.json')
        if result.exists(): paths.append(result)
        log = checked_path(root, safe_relative(row['local_artifact_root'])/'worker.log')
        if log.exists(): paths.append(log)
    report_folder = root/OUT_FOLDER/'report'; report = read(report_folder/'summary.json')
    assert report['complete'] and report['protocol_sha256'] == PROTOCOL_SHA
    assert report['fixed_training_updates_per_model'] == 4096 and report['fixed_training_trials_per_model'] == 131072
    assert set(report['models']) == set(CONDITIONS)
    assert report['reporting_source_sha256'] == sha(root/'scripts/plot_local_goal_return_training.py')
    for name, digest in report['input_sha256'].items(): assert sha(checked_path(root, name)) == digest, name
    paths += [report_folder/name for name in REPORT_FILES]
    input_assets = []
    for name in ('local_return_bank_archive.json', 'local_return_inputs_archive.json'):
        path = root/'docs/research'/name; record = read(path); paths.append(path)
        input_assets.append(dict(name=Path(record['archive_path']).name, bytes=record['archive_bytes'], sha256=record['archive_sha256']))
    paths = sorted(set(paths))
    for path in paths:
        checked_path(root, path.relative_to(root)); inspect_file(path)
    entries = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    return dict(version='local_return_complete_output_manifest_v1', source_commit=source_commit,
        training_protocol_sha256=PROTOCOL_SHA, models=models, prior_attempts=prior,
        cleanup_sha256=sha(cleanup_path), required_input_assets=input_assets,
        source_sha256={name:sha(root/name) for name in sources}, files=entries,
        file_count=len(entries), uncompressed_bytes=sum(r['bytes'] for r in entries),
        full_experiment_trees=True, failed_and_resume_evidence_retained=True,
        historical_stage_audits=historical_audits,
        historical_audit_scope='Earlier audits retain their original upload-time hashes; their historical latest.pt/metrics hashes are not compared to subsequently completed files. Final audits remain authoritative for final artifacts.',
        local_extraction_reproduction='Pending until this actual output asset is generated and separately tested; no preemptive completion claim')


def write_archive(root, archive_path, entries):
    """Deterministic archive; a caller-supplied synthetic inventory is testable."""
    rows = sorted(entries, key=lambda r:r['path'])
    assert len({r['path'] for r in rows}) == len(rows)
    with Path(archive_path).open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for row in rows:
                path = checked_path(root, row['path']); inspect_file(path)
                assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
                info = archive.gettarinfo(str(path), arcname=row['path'])
                info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
                with path.open('rb') as file: archive.addfile(info, file)
    with tarfile.open(archive_path, 'r:gz') as archive:
        assert archive.getnames() == [r['path'] for r in rows]
        for row in rows:
            member = archive.getmember(row['path']); assert member.isfile() and member.size == row['bytes']
            h = hashlib.sha256()
            with archive.extractfile(member) as file:
                for block in iter(lambda:file.read(1024*1024), b''): h.update(block)
            assert h.hexdigest() == row['sha256'], row['path']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', default=COMMIT)
    parser.add_argument('--out', type=Path, default=ROOT/'results/ring_public_assets/local-return-final-outputs-v1.tar.gz')
    args = parser.parse_args()
    assert not args.out.exists(), 'Preserve any existing published output asset'
    manifest = inventory(source_commit=args.source_commit)
    folder = ROOT/OUT_FOLDER; folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder/'final_outputs_manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest, 'Preserve an existing inventory with different inputs'
    else: manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    entries = manifest['files']+[dict(path=str(manifest_path.relative_to(ROOT)),
        bytes=manifest_path.stat().st_size, sha256=sha(manifest_path))]
    args.out.parent.mkdir(parents=True, exist_ok=True); temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    write_archive(ROOT, temporary, entries)
    # No partial/concurrent writer may change the scientific or cleanup inputs.
    assert inventory(source_commit=args.source_commit) == manifest
    assert all(sha(ROOT/r['path']) == r['sha256'] for r in entries)
    temporary.replace(args.out)
    record = dict(version='local_return_final_output_archive_v1', source_commit=args.source_commit,
        archive_path=str(args.out), archive_name=args.out.name, archive_bytes=args.out.stat().st_size,
        archive_sha256=sha(args.out), manifest_path=str(manifest_path.relative_to(ROOT)),
        manifest_sha256=sha(manifest_path), archive_members=len(entries), full_archived_bytes_verified=True,
        both_strict_final_audits_passed=True, all_four_stages_retained=True,
        owner_verified_provider_cleanup=True, cleanup_sha256=manifest['cleanup_sha256'],
        required_input_assets=manifest['required_input_assets'], generator_sha256=sha(__file__),
        isolated_extraction_reproduction_verified=False, publication='Not performed by packager')
    (ROOT/'docs/research/local_return_outputs_archive.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
