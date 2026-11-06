"""Package the audited compact inputs, preserving repository-relative paths."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'results/lstm_endogenous_ring/reproduction_inputs_manifest.json'
OUTPUT=ROOT/'results/ring_public_assets/endogenous-ring-inputs-v1.tar.gz'
RELEASE='https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/endogenous-ring-tests-2026-11-06'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    manifest=json.loads(MANIFEST.read_text())
    entries={row['path']:row for row in manifest['files']}
    assert len(entries)==manifest['file_count']
    for name,row in entries.items():
        path=ROOT/name
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],name
    extra=['results/lstm_endogenous_ring/reproduction_inputs_manifest.json',
           'results/lstm_endogenous_ring/REPRODUCE.md']
    names=sorted([*entries,*extra]);assert len(names)==len(set(names))
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in names:
                path=ROOT/name;info=archive.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as file:archive.addfile(info,file)
    with tarfile.open(OUTPUT,'r:gz') as archive:
        assert archive.getnames()==names
        for name,row in entries.items():
            h=hashlib.sha256()
            member=archive.extractfile(name)
            for block in iter(lambda:member.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==row['sha256'],name
    record=dict(archive_name=OUTPUT.name,archive_bytes=OUTPUT.stat().st_size,archive_sha256=digest(OUTPUT),
        input_manifest=str(MANIFEST.relative_to(ROOT)),input_manifest_sha256=digest(MANIFEST),
        metadata_files_sha256={name:digest(ROOT/name) for name in extra},
        input_files=len(entries),archive_files=len(names),release_url=RELEASE,
        extract_at='Repository root; original relative paths are preserved',
        contents='Recorded native context inputs and metadata, two fixed25.024M final checkpoints and reference evaluations. Regenerable multi-gigabyte trajectories are omitted.',
        original_models='Existing rl-map-checkpoints.tar.gz asset on rl-map-memory-2026-11-06 release',
        generator_sha256=digest(Path(__file__)),verified_every_archived_input_hash=True)
    destination=ROOT/'docs/research/endogenous_ring_inputs_archive.json'
    destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
