"""Hash-only inventory and deterministic multipart archives for the two studies.

No policy, environment, bank regeneration, training or cloud operations are run.
Both studies must have complete artifact audits; negative scientific results are
expected and never exclude files. --inventory-only writes nothing.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'docs/research/goal_cue_memory_original_diagnostic_protocol.json'
PROTOCOL_SHA = '2d9386fad85f9774d80ced37d7d6220d6bb75c682879684511a0da4d4bb04125'
CONDITIONS = ('gc100', 'pgc33')
BASE = 'results/goal_cue_memory'
DESIGN = 'results/goal_cue_memory_design'
TREES = (f'{BASE}/qualification', f'{BASE}/original_diagnostic', DESIGN)
REPORT = f'{BASE}/original_diagnostic/report'
INTERPRETATION = 'docs/research/original_goal_memory_diagnostic_interpretation.md'
EXTRA_DOCS = (INTERPRETATION, 'docs/research/next_native_memory_design_review.md',
              'docs/research/next_native_memory_independent_review.md')
TOOLS = ('scripts/package_goal_cue_memory_outputs.py',
         'tests/test_package_goal_cue_memory_outputs.py',
         'scripts/report_goal_cue_qualification.py',
         'scripts/report_goal_original_diagnostic.py',
         'scripts/plot_goal_cue_memory_design.py',
         'scripts/check_goal_cue_donor_equivalence.py')
MANIFEST_NAME = 'results/goal_cue_memory_release/manifest.json'
MAX_ASSET_BYTES = 1_800_000_000
PAYLOAD_BYTES = 1_750_000_000


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def read(path):
    return json.loads(Path(path).read_text())


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def checked(root, name, *, directory=False):
    """Every path component must be literal, repository-relative and nonlinked."""
    rel = PurePosixPath(name)
    assert isinstance(name, str) and name and '\\' not in name and '\0' not in name
    assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name
    path = Path(root)
    for component in rel.parts:
        path /= component
        assert not path.is_symlink(), f'Symlink: {name}'
    assert path.is_dir() if directory else path.is_file(), f'Missing or nonregular: {name}'
    return path


def tree_files(root, name):
    folder = checked(root, name, directory=True)
    result = set()
    for path in folder.rglob('*'):
        relative = path.relative_to(root).as_posix()
        assert not path.is_symlink(), f'Symlink: {relative}'
        if path.is_dir():
            continue
        checked(root, relative)
        assert not path.name.endswith('.tmp'), f'Unfinished writer: {relative}'
        result.add(relative)
    return result


def prove(root, name, record):
    path = checked(root, name)
    assert sha(path) == record['sha256'], f'Hash differs: {name}'
    if 'bytes' in record:
        assert path.stat().st_size == record['bytes'], f'Size differs: {name}'
    return path


def commit_sha(root, commit, name):
    return hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{commit}:{name}'], cwd=root)).hexdigest()


def audit_complete(root, phase, condition, protocol_sha, sources, variants):
    name = f'{BASE}/{phase}/audits/{condition}.json'
    audit = read(checked(root, name))
    assert audit['complete'] and audit['passed'] and audit['status'] == 'complete', name
    assert not audit['pending'] and not audit['failures'], name
    assert (audit['phase'], audit['condition'], audit['protocol_sha256']) == (phase, condition, protocol_sha)
    auditor = 'scripts/audit_goal_cue_memory' + ('_original_diagnostic' if phase == 'original_diagnostic' else '') + '.py'
    assert audit['auditor_sha256'] == sources[auditor]
    if phase == 'original_diagnostic':
        assert audit['frozen_numerical_auditor_sha256'] == sources['scripts/audit_goal_cue_memory.py']
        assert audit['parent_training_gate_still_failed'] and audit['no_automatic_neural_nomination']
    folder = f'{BASE}/{phase}/{condition}'
    files = tree_files(root, folder)
    expected = {f'{folder}/{relative}' for relative in audit['artifacts']}
    assert files == expected, f'Audit inventory is stale or incomplete: {folder}'
    for relative, proof in audit['artifacts'].items():
        prove(root, f'{folder}/{relative}', proof)
    evaluation = audit['checks']['evaluation']
    assert evaluation['complete']
    pairs = [(row['context'], row['variant']) for row in evaluation['contexts']]
    assert len(pairs) == 16 * len(variants)
    assert set(pairs) == {(c, variant) for c in range(16) for variant in variants}
    sub = 'evaluation' if phase == 'original_diagnostic' else 'qualification_evaluation'
    for c in range(16):
        prefix = f'{folder}/{sub}/context{c}'
        checked(root, f'{prefix}/input_plan.pt.gz')
        for variant in variants:
            branch = f'{prefix}/{variant}'
            checked(root, f'{branch}/input.json'); checked(root, f'{branch}/result.json')
            assert any(f'{branch}/{raw}' in files for raw in ('raw.pt.gz', 'failed_plan.pt.gz')), branch
    summary = read(checked(root, f'{folder}/summary.json'))
    assert summary['complete'] and summary['no_optimization']
    assert summary['phase'] == phase and summary['condition'] == condition and summary['protocol_sha256'] == protocol_sha
    if phase == 'original_diagnostic':
        assert summary['variant_order'] == list(variants)
        assert set(summary['variants']) == set(variants)
        assert all(row['trials'] == 256 for row in summary['variants'].values())
        assert summary['no_optimizer_created'] and summary['parent_training_gate_still_failed']
        for required in ('incoming.pt', 'after.pt', 'evaluation_integrity.json'):
            checked(root, f'{folder}/{required}')
    else:
        assert summary['totals']['trials'] == 256
        for required in ('incoming.pt', 'qualification_after.pt', 'qualification_evaluation_integrity.json'):
            checked(root, f'{folder}/{required}')
    return name


def execution_complete(root, phase, expected_protocol):
    path = f'{DESIGN}/{phase}_execution.json'
    record = read(checked(root, path))
    assert record['protocol_sha256'] == expected_protocol
    if phase == 'original_diagnostic':
        assert record['complete']
    runs = record['runs']
    assert len(runs) == 2 and {r['condition'] for r in runs} == set(CONDITIONS)
    assert len({r['log_path'] for r in runs}) == 2, 'Writer logs must be distinct'
    for row in runs:
        assert row['writer_complete'] and row['exit_code'] == 0 and row['no_optimization']
        assert row['summary_path'] == f'{BASE}/{phase}/{row["condition"]}/summary.json'
        prove(root, row['log_path'], {'sha256': row['log_sha256']})
        prove(root, row['summary_path'], {'sha256': row['summary_sha256']})
    return path


def inventory(root, source_commit, report=REPORT):
    """Validate completed metadata and bytes only; never import study modules."""
    root = Path(root).resolve()
    assert re.fullmatch(r'[0-9a-f]{40}', source_commit), 'Use a full reproduction commit SHA'
    design = read(prove(root, PROTOCOL, {'sha256': PROTOCOL_SHA}))
    assert design['frozen_utc'] and design['conditions'] == list(CONDITIONS)
    stopped = design['stopped_parent_protocol']
    assert not stopped['paired_training_qualification_passed'] and stopped['training_remains_prohibited']
    parent = read(prove(root, stopped['path'], {'sha256': stopped['sha256']}))
    variants = design['evaluation']['variants']
    assert len(variants) == len(set(variants)) == 11
    sources = design['source_sha256']
    # Check all four terminal audit statuses before reading any large artifact tree.
    for phase in ('qualification', 'original_diagnostic'):
        for condition in CONDITIONS:
            a = read(checked(root, f'{BASE}/{phase}/audits/{condition}.json'))
            assert a['passed'] and a['complete'] and a['status'] == 'complete' and not a['pending'] and not a['failures']
    files = {PROTOCOL, *EXTRA_DOCS, *TOOLS}
    for mapping in (sources, design['input_assets']):
        for name, expected in mapping.items():
            prove(root, name, {'sha256': expected}); files.add(name)
    for name in set(sources) | set(TOOLS):
        assert sha(checked(root, name)) == commit_sha(root, source_commit, name), f'Reproduction commit differs: {name}'
    assert commit_sha(root, source_commit, PROTOCOL) == PROTOCOL_SHA
    audits = []
    for phase, protocol_sha, branch_names in (
        ('qualification', stopped['sha256'], ['continuous']),
        ('original_diagnostic', PROTOCOL_SHA, variants)):
        audits.extend(audit_complete(root, phase, c, protocol_sha, sources, branch_names) for c in CONDITIONS)
        files.add(execution_complete(root, phase, protocol_sha))
    for bank in parent['evaluation']['banks'].values():
        prove(root, bank['path'], bank); files.add(bank['path'])
    for name in TREES:
        files.update(tree_files(root, name))
    for name in (f'{BASE}/qualification/report', report):
        members = tree_files(root, name)
        assert f'{name}/README.md' in members and f'{name}/summary.json' in members
        assert any(m.endswith('.png') for m in members), f'Missing figure: {name}'
        files.update(members)
    docs = checked(root, 'docs/research', directory=True)
    files.update(p.relative_to(root).as_posix() for p in docs.glob('goal_cue_memory_*') if p.suffix in ('.md', '.json'))
    # The publisher supplies a guide before final packaging; it is not generated here.
    files.add('results/goal_cue_memory_release/REPRODUCE.md')
    entries = [dict(path=name, bytes=checked(root, name).stat().st_size, sha256=sha(root/name)) for name in sorted(files)]
    assert MANIFEST_NAME not in files
    return dict(version='goal_cue_memory_outputs_v1', source_commit=source_commit,
        protocol_path=PROTOCOL, protocol_sha256=PROTOCOL_SHA,
        parent_protocol_sha256=stopped['sha256'], parent_training_qualification_passed=False,
        scientific_success_required_for_packaging=False, numerical_audits_rerun_by_packager=False,
        audit_records={name: sha(root/name) for name in audits},
        max_asset_bytes=MAX_ASSET_BYTES, payload_limit_bytes=PAYLOAD_BYTES,
        file_count=len(entries), uncompressed_bytes=sum(e['bytes'] for e in entries), files=entries)


def partition(entries, limit=PAYLOAD_BYTES):
    assert limit > 0
    parts, current, size = [], [], 0
    for entry in sorted(entries, key=lambda e: e['path']):
        assert 0 <= entry['bytes'] <= limit, f'Single member exceeds payload bound: {entry["path"]}'
        if current and size + entry['bytes'] > limit:
            parts.append(current); current, size = [], 0
        current.append(entry); size += entry['bytes']
    if current:
        parts.append(current)
    return parts


def verify_archive(path, entries, manifest_bytes):
    expected = {e['path']: e for e in entries}
    assert len(expected) == len(entries) and MANIFEST_NAME not in expected
    expected[MANIFEST_NAME] = dict(bytes=len(manifest_bytes), sha256=hashlib.sha256(manifest_bytes).hexdigest())
    seen = []
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = member.name; rel = PurePosixPath(name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert name in expected and name not in seen and rel.as_posix() == name
            assert (member.uid, member.gid, member.mtime, member.mode) == (0, 0, 0, 0o644)
            assert not member.uname and not member.gname
            row = expected[name]; assert member.size == row['bytes']
            with archive.extractfile(member) as stream:
                assert digest(stream) == row['sha256'], f'Archived bytes differ: {name}'
            seen.append(name)
    assert seen == sorted(expected), 'Archive member set/order differs'


def build(root, manifest, out, *, payload_limit=PAYLOAD_BYTES, asset_limit=MAX_ASSET_BYTES):
    root, out = Path(root).resolve(), Path(out).resolve()
    for name in TREES:
        assert not out.is_relative_to(root/name), 'Archive destination overlaps input tree'
    assert not out.exists() or not any(out.iterdir()), 'Use a fresh archive directory'
    entries = manifest['files']
    assert len({e['path'] for e in entries}) == len(entries)
    for entry in entries:
        prove(root, entry['path'], entry)
    parts = partition(entries, payload_limit); assert parts
    body = json_bytes(manifest); out.mkdir(parents=True, exist_ok=True)
    index = dict(version=manifest['version'], source_commit=manifest['source_commit'],
        manifest_sha256=hashlib.sha256(body).hexdigest(), max_asset_bytes=asset_limit, parts=[])
    for number, part in enumerate(parts, 1):
        path = out/f'goal-cue-memory-outputs-part{number:03d}-of{len(parts):03d}.tar.gz'
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                contents = {e['path']: e for e in part}
                for name in sorted([*contents, MANIFEST_NAME]):
                    entry = contents.get(name); data = body if entry is None else None
                    info = tarfile.TarInfo(name); info.size = len(body) if entry is None else entry['bytes']
                    info.mode = 0o644; info.mtime = info.uid = info.gid = 0
                    if data is not None:
                        archive.addfile(info, io.BytesIO(data))
                    else:
                        with checked(root, name).open('rb') as stream:
                            archive.addfile(info, stream)
        assert temporary.stat().st_size <= asset_limit, 'Compressed asset exceeds limit'
        verify_archive(temporary, part, body)
        temporary.replace(path)
        index['parts'].append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path),
            members=[e['path'] for e in part], all_archived_bytes_verified=True))
    (out/'manifest.json').write_bytes(body)
    (out/'index.json').write_bytes(json_bytes(index))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--report-dir', default=REPORT)
    parser.add_argument('--out', type=Path, default=ROOT/'results/goal_cue_memory_release/archives')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args(); manifest = inventory(ROOT, args.source_commit, args.report_dir)
    if args.inventory_only:
        print(json_bytes(manifest).decode(), end=''); return
    print(json.dumps(build(ROOT, manifest, args.out), indent=2))


if __name__ == '__main__':
    main()
