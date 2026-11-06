"""Package all prescribed90M checkpoints and exact native contexts after completion."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import lstm_reorientation_entropy_mechanism as producer

OUT=ROOT/'results/ring_public_assets/reorientation90-inputs-v1.tar.gz'
CHECKPOINTS=('incoming.pt',*(f'block{i}.pt' for i in range(1,6)),'final.pt')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    paths=[];models=[]
    for arm in producer.ARMS:
        for condition in producer.CONDITIONS:
            checkpoint=producer.resolve_checkpoint(condition,arm)
            producer.preflight(condition,arm,checkpoint,producer.ROUTING)
            folder=checkpoint.parent;worker=folder.parent
            audit_path=ROOT/f'results/rl_blind_reorientation_entropy/cloud_audit/{condition}_{arm}.json'
            audit=json.loads(audit_path.read_text())
            assert audit['complete'] and audit['passed'] and audit['required_complete'] and not audit['failures']
            assert (audit['condition'],audit['arm'])==(condition,arm)
            assert audit['auditor_sha256']==sha(ROOT/'scripts/audit_rl_blind_reorientation_entropy.py')
            assert Path(audit['folder']).resolve()==folder.resolve()
            for name,record in audit['artifacts'].items():
                path=folder/name
                assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
            for name in CHECKPOINTS:
                path=folder/name
                assert sha(path)==audit['artifacts'][name]['sha256'],name
                paths.append(path)
            paths.extend(sorted(folder.glob('*.json')))
            paths.extend([folder/'metrics.jsonl',worker/'WORKER_RESULT.json',audit_path])
            assert json.loads((worker/'WORKER_RESULT.json').read_text())['exit_code']==0
            inputs=producer.OUT/arm/condition
            producer.validate_producer(inputs,sha(checkpoint),condition,arm)
            collection=json.loads((inputs/'collection.json').read_text())
            assert collection['checkpoint_sha256']==sha(checkpoint)
            paths.extend(inputs/name for name in ('collection.json','implementation_protocol.json',
                'extension_adapter.json','native_phase_coverage.json'))
            for i in (0,1):
                path=inputs/'contexts'/f'context{i}.npz'
                with np.load(path) as native:
                    assert str(native['checkpoint_sha256'])==sha(checkpoint)
                paths.append(path)
            models.append(dict(condition=condition,arm=arm,folder=str(folder.relative_to(ROOT)),
                final_checkpoint_sha256=sha(checkpoint),checkpoints=list(CHECKPOINTS)))
    paths.append(producer.ROUTING)
    entries=[]
    for path in sorted(set(paths)):
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=ROOT/'results/rl_blind_reorientation_entropy/reproduction_inputs_manifest.json'
    manifest.write_text(json.dumps(dict(models=models,files=entries,file_count=len(entries),
        checkpoint_count=28,uncompressed_bytes=sum(r['bytes'] for r in entries)),indent=2)+'\n')
    names=sorted([r['path'] for r in entries]+[str(manifest.relative_to(ROOT)),
        'results/rl_blind_reorientation_entropy/REPRODUCE.md'])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in names:
                path=ROOT/name;info=archive.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as file:archive.addfile(info,file)
    with tarfile.open(OUT,'r:gz') as archive:
        assert archive.getnames()==names
        for row in entries:
            member=archive.extractfile(row['path']);h=hashlib.sha256()
            for block in iter(lambda:member.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==row['sha256']
    record=dict(archive_name=OUT.name,archive_bytes=OUT.stat().st_size,archive_sha256=sha(OUT),
        manifest_sha256=sha(manifest),files=len(entries),archive_members=len(names),checkpoints=28,
        every_archived_input_hash_verified=True,generator_sha256=sha(__file__),
        release_url='https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/matched-entropy-continuations-2026-11-06',
        contents='Four fixed90M models: incoming,all5fixedblocks,final; raw evaluations/configs/provenance/metrics and complete audits; eight native context inputs and producer metadata; exact final runtime/RNG/RMSProp. Actual worker/experiment paths preserved.',
        extraction='Repository root. Original1B and40Mparent releases are required for unchanged provenance validation.')
    (ROOT/'docs/research/reorientation90_inputs_archive.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
