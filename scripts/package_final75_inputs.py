"""Package exact final checkpoints and compact native inputs for all four75M models."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/ring_public_assets/final75-memory-inputs-v1.tar.gz'
MODELS={'v2':{'gc100':'ring-20261105-vision-long-grid-01','pgc33':'ring-20261105-vision-long-pgc-01'},
        'v3':{'gc100':'ring-20261105-vision-noise-grid-01','pgc33':'ring-20261105-vision-noise-pgc-01'}}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    paths=[];models=[]
    for arm,pair in MODELS.items():
        for condition,worker in pair.items():
            model=ROOT/'results/ring_cloud_artifacts'/worker
            inputs=ROOT/'results/lstm_final75_memory'/arm/condition
            audit=json.loads((inputs/'checkpoint_audit.json').read_text())
            checkpoint=model/'final.pt'
            assert digest(checkpoint)==audit['checkpoint_sha256']
            assert json.loads((model/'summary.json').read_text())['completed']
            paths.extend([checkpoint,model/'final_evaluation.json',model/'protocol.json',model/'summary.json'])
            if arm=='v3':paths.append(model/'final_noisy_evaluation.json')
            for kind in ('heading','travel'):
                paths.append(inputs/kind/'collection.json')
                paths.extend(inputs/kind/f'contexts/context{context}.npz' for context in (0,1))
            paths.extend([inputs/'checkpoint_audit.json',inputs/'protocol.json'])
            models.append(dict(arm=arm,condition=condition,checkpoint=str(checkpoint.relative_to(ROOT)),
                               checkpoint_sha256=audit['checkpoint_sha256']))
    # Keep exact episode rows, completed diagnostic references and the retained
    # first reset-import failure under their actual worker names.
    for worker in ('ring-20261105-vision-noisy-eval-01','ring-20261105-vision-reset-eval-01'):
        diagnostic=ROOT/'results/ring_cloud_artifacts'/worker
        assert (diagnostic/'WORKER_RESULT.json').exists()
        paths.extend(sorted(diagnostic.glob('*.json')))
    entries=[]
    for path in sorted(paths):
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
    assert len({r['path'] for r in entries})==len(entries)
    manifest=ROOT/'results/lstm_final75_memory/reproduction_inputs_manifest.json'
    manifest.write_text(json.dumps(dict(models=models,file_count=len(entries),
        uncompressed_bytes=sum(r['bytes'] for r in entries),files=entries),indent=2)+'\n')
    extras=[str(manifest.relative_to(ROOT)),'results/lstm_final75_memory/REPRODUCE.md']
    names=sorted([r['path'] for r in entries]+extras)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in names:
                path=ROOT/name;info=archive.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as file:archive.addfile(info,file)
    with tarfile.open(OUTPUT,'r:gz') as archive:
        assert archive.getnames()==names
        for row in entries:
            h=hashlib.sha256();member=archive.extractfile(row['path'])
            for block in iter(lambda:member.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==row['sha256']
    record=dict(archive_name=OUTPUT.name,archive_bytes=OUTPUT.stat().st_size,archive_sha256=digest(OUTPUT),
        manifest_sha256=digest(manifest),files=len(entries),archive_members=len(names),
        generator_sha256=digest(__file__),every_archived_input_hash_verified=True,
        release_url='https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/visual-memory-continuations-2026-11-06',
        contents='Four exact75Mfinal checkpoints and reference evaluations;16native context NPZ inputs with collection metadata; matched-noise and every-step-reset raw episode rows, runtime and success/failure records. All intermediate stages remain in audited cloud/local archives; no selected-stage replacement.',
        extraction='Repository root; original relative paths preserved')
    (ROOT/'docs/research/final75_inputs_archive.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
