"""Preserve exact neural arrays for the frozen learned-action-score assay.

NPZ files are already compressed. Separate uncompressed tar assets keep each
cohort under GitHub's per-asset limit and preserve every locked byte, including
historical checkpoint-path metadata. Checkpoints have separate release assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import lstm_actor_logit_observable as assay
import lstm_reorientation_entropy_mechanism as producer


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('existing', 'ninety'), required=True)
    args = parser.parse_args()
    design_path = assay.OUT / 'protocol.json'
    design = json.loads(design_path.read_text())
    assert design == assay.protocol()
    cases = [case for case in assay.cases()
             if (case['cohort'] == 'ninety') == (args.cohort == 'ninety')]
    assert len(cases) == (8 if args.cohort == 'ninety' else 32)
    paths = {design_path, assay.OUT / 'REPRODUCE.md'}
    for case in cases:
        path = ROOT / case['archive']
        result = json.loads((assay.OUT / case['output']).read_text())
        assert result['case'] == case
        assert result['source_sha256'] == sha(Path(assay.__file__))
        assert result['protocol_sha256'] == sha(design_path)
        assert result['archive_sha256'] == sha(path)
        if args.cohort == 'existing':
            assert sha(path) == design['existing_input_sha256'][case['archive']]
            assert result['checkpoint_sha256'] == design['existing_input_sha256'][case['checkpoint']]
        else:
            checkpoint = producer.resolve_checkpoint(case['condition'], case['arm'])
            producer.preflight(case['condition'], case['arm'], checkpoint, producer.ROUTING)
            assert result['checkpoint_sha256'] == sha(checkpoint)
            producer.validate_producer(path.parent, sha(checkpoint), case['condition'], case['arm'])
            paths.update(path.parent / name for name in ('extension_adapter.json',
                'implementation_protocol.json', 'collection.json', 'native_phase_coverage.json'))
        paths.add(path)
        paths.add(path.with_name(f"context{case['context']}.json"))
    entries = []
    for path in sorted(paths):
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path)))
    stem = 'actor-score-existing-inputs-v1' if args.cohort == 'existing' else 'reorientation90-neural-arrays-v1'
    manifest = assay.OUT / f'{stem}-manifest.json'
    manifest.write_text(json.dumps(dict(cohort=args.cohort, state_archives=len(cases),
        assay_protocol_sha256=sha(design_path), files=entries), indent=2) + '\n')
    names = sorted([entry['path'] for entry in entries] + [str(manifest.relative_to(ROOT))])
    archive_path = ROOT / 'results/ring_public_assets' / f'{stem}.tar'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'w') as archive:
        for name in names:
            path = ROOT / name
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    assert archive_path.stat().st_size < 2_000_000_000
    with tarfile.open(archive_path) as archive:
        assert archive.getnames() == names
        for entry in entries:
            stream = archive.extractfile(entry['path']); digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            assert digest.hexdigest() == entry['sha256']
    record = dict(archive_name=archive_path.name, archive_bytes=archive_path.stat().st_size,
        archive_sha256=sha(archive_path), cohort=args.cohort, state_archives=len(cases),
        archive_members=len(names), every_member_input_hash_verified=True,
        manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest),
        generator_sha256=sha(__file__), assay_protocol_sha256=sha(design_path),
        extraction='Extract at the repository root. Exact historical path strings are retained as metadata; analyses resolve checkpoints from the current repository.',
        checkpoint_dependency='Original 1B and40M models are supplied in earlier releases; the genuine90M models are supplied in the matched-entropy checkpoint archive.',
        release_url='https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/matched-entropy-continuations-2026-11-06')
    output = ROOT / 'docs/research' / f'{stem}-archive.json'
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
