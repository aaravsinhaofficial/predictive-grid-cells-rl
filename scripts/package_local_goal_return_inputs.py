"""Package immutable local-return starting policies and complete baseline evidence.

This performs hash/metadata checks only. It does not run a policy, environment,
optimizer, cloud command, or publication operation. Banks are a separate asset.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '6f316f4e4f5df23f40b177c6ed88ae11bbf2e795'
TRAINING = ROOT/'docs/research/local_goal_return_training_protocol.json'
TRAINING_SHA = '0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'
BEHAVIOR_SHA = '4c7ba90dd79c8c0b5f8ac13ba89e4a7d6ed5bf0287556982ade07f911fc16e48'
BASE = ROOT/'results/local_goal_return_baseline'
DESIGN = ROOT/'results/local_goal_return_design'
MANIFEST = DESIGN/'training_inputs_manifest.json'
OUT = ROOT/'results/ring_public_assets/local-return-training-inputs-v1.tar.gz'
RECORD = ROOT/'docs/research/local_return_inputs_archive.json'
VALIDATION_NAMES = (
    'native_resume_validation.json', 'completed_baseline_audit.json',
    'completed_baseline_audit_validation.json', 'independent_bank_review.json',
    'independent_implementation_review.json', 'certificate_review.json',
    'isolated_package_preflight.json', 'bank_archive_manifest.json',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def committed_sha(name):
    return hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{COMMIT}:{name}'], cwd=ROOT)).hexdigest()


def required_inputs():
    assert sha(TRAINING) == TRAINING_SHA, 'Frozen training protocol changed'
    protocol = read(TRAINING)
    assert protocol['evaluation']['behavior_protocol_sha256'] == BEHAVIOR_SHA
    assert sha(DESIGN/'behavior_protocol.json') == BEHAVIOR_SHA
    assert committed_sha(str(TRAINING.relative_to(ROOT))) == TRAINING_SHA
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT/name) == digest == committed_sha(name), name
    for name, digest in protocol['original_assets'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in protocol['passed_baseline_assets'].items():
        assert sha(ROOT/name) == digest, name

    # Exact completed inventory: both models, both banks, every context/control.
    paths = [ROOT/name for name in protocol['original_assets']]
    for condition in ('gc100', 'pgc33'):
        folder = BASE/condition
        expected = {folder/name for name in (
            'summary.json', 'execution.json', 'initialization.json', 'initialized_policy.pt')}
        for bank in ('near_baseline', 'final_hard'):
            expected.add(folder/bank/'summary.json')
            for context in range(16):
                expected.update(folder/bank/f'context{context}'/name for name in (
                    'input.json', 'result.json', 'raw.pt', 'uniform_raw.pt', 'planner_raw.pt'))
        actual = {p for p in folder.rglob('*') if p.is_file()}
        assert expected == actual and len(actual) == 166, condition
        assert read(folder/'summary.json')['complete']
        paths.extend(sorted(actual))

    audit = read(DESIGN/'completed_baseline_audit.json')
    assert audit['complete'] and audit['passed'] and audit['artifact_integrity_passed']
    assert audit['source_sha256'] == sha(ROOT/'scripts/audit_local_goal_return_baseline.py')
    assert audit['source_sha256'] == committed_sha('scripts/audit_local_goal_return_baseline.py')
    assert audit['native_validation_sha256'] == sha(DESIGN/'native_resume_validation.json')
    assert audit['behavior_protocol_sha256'] == BEHAVIOR_SHA
    validation = read(DESIGN/'completed_baseline_audit_validation.json')
    assert validation['passed'] and validation['audit_output_sha256'] == sha(DESIGN/'completed_baseline_audit.json')
    bank_record_path = ROOT/'docs/research/local_return_bank_archive.json'
    bank_record = read(bank_record_path)
    bank_asset = ROOT/bank_record['archive_path']
    assert bank_record['contents_verified'] and sha(bank_asset) == bank_record['archive_sha256']
    assert bank_asset.stat().st_size == bank_record['archive_bytes']
    for name, row in bank_record['files'].items():
        assert sha(ROOT/name) == row['sha256'] and (ROOT/name).stat().st_size == row['bytes'], name
    paths.extend([TRAINING, bank_record_path, DESIGN/'behavior_protocol.json', DESIGN/'REPRODUCE.md'])
    paths.extend(DESIGN/name for name in VALIDATION_NAMES)
    paths = sorted(set(paths))
    for path in paths:
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        assert '/banks/' not in str(path), 'Sealed bank binaries belong to their separate archive'
    return protocol, bank_record, paths


def verify_archive(archive_path, entries):
    """Stream every archived member, including the manifest, without extraction."""
    expected = {row['path']: row for row in entries}
    with tarfile.open(archive_path, 'r:gz') as archive:
        assert archive.getnames() == sorted(expected)
        for member in archive:
            assert member.isfile() and not Path(member.name).is_absolute()
            assert '..' not in Path(member.name).parts
            row = expected[member.name]
            assert member.size == row['bytes']
            h = hashlib.sha256()
            with archive.extractfile(member) as file:
                for block in iter(lambda: file.read(1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == row['sha256'], member.name


def main():
    protocol, bank_record, paths = required_inputs()
    entries = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    manifest = dict(version='local_return_training_inputs_v1', source_commit=COMMIT,
        training_protocol_sha256=TRAINING_SHA, behavior_protocol_sha256=BEHAVIOR_SHA,
        source_sha256=protocol['source_sha256'], files=entries, file_count=len(entries),
        uncompressed_bytes=sum(row['bytes'] for row in entries),
        baseline_files=332, original_checkpoints=2, initialized_policies=2,
        separate_required_asset=dict(name=Path(bank_record['archive_path']).name,
            bytes=bank_record['archive_bytes'], sha256=bank_record['archive_sha256']),
        scope='Exact reset-head inputs and all baseline evidence; no trained local-task endpoint, bank binary, cloud routing or reporting duplicate. Source code comes from the pinned repository commit.',
        validation_records='Historical implementation/preflight records retain their original timestamps and source snapshots. The frozen training protocol is authoritative for runtime source hashes.')
    MANIFEST.write_text(json.dumps(manifest, indent=2)+'\n')
    all_entries = entries+[dict(path=str(MANIFEST.relative_to(ROOT)),
                               bytes=MANIFEST.stat().st_size, sha256=sha(MANIFEST))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix+'.tmp')
    with temporary.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for row in sorted(all_entries, key=lambda item:item['path']):
                path = ROOT/row['path']; info = archive.gettarinfo(str(path), arcname=row['path'])
                info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
                with path.open('rb') as file:
                    archive.addfile(info, file)
    verify_archive(temporary, all_entries)
    # Catch any concurrent edit to a packaged input or frozen source.
    required_inputs()
    assert all(sha(ROOT/r['path']) == r['sha256'] for r in all_entries)
    temporary.replace(OUT)
    record = dict(version='local_return_training_input_archive_v1', source_commit=COMMIT,
        archive_path=str(OUT.relative_to(ROOT)), archive_bytes=OUT.stat().st_size, archive_sha256=sha(OUT),
        manifest_path=str(MANIFEST.relative_to(ROOT)), manifest_sha256=sha(MANIFEST),
        archive_members=len(all_entries), file_count=len(entries), baseline_files=332,
        uncompressed_member_bytes=sum(row['bytes'] for row in all_entries),
        training_protocol_sha256=TRAINING_SHA, behavior_protocol_sha256=BEHAVIOR_SHA,
        generator_sha256=sha(__file__), all_archived_bytes_verified=True,
        frozen_sources_and_bank_asset_unchanged=True, published=False,
        required_separate_asset=manifest['separate_required_asset'],
        guide=str((DESIGN/'REPRODUCE.md').relative_to(ROOT)),
        extraction='Extract both assets at the root of a fresh checkout of the pinned source commit. Final trained checkpoints will be a separate later asset.',
        experiments_run='None: packaging uses file hashes and metadata only.')
    RECORD.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
