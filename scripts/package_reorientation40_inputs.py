"""Publish every prescribed40M checkpoint, raw evaluation and native input."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ring_public_assets/reorientation40-inputs-v1.tar.gz'
WORKERS={'gc100':'ring-20261105-reorient-grid-01','pgc33':'ring-20261105-reorient-pgc-02'}
CHECKPOINTS=('incoming.pt','after_warmup.pt','stage1_delay4.pt','stage2_delay15.pt',
             'stage3_delay30.pt','stage4_delay60.pt','final.pt')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    paths=[];models=[]
    for condition,worker in WORKERS.items():
        folder=ROOT/'results/ring_cloud_artifacts'/worker
        audit_path=ROOT/f'results/rl_blind_reorientation_review/cloud_audit/{condition}.json'
        audit=json.loads(audit_path.read_text())
        assert audit['complete'] and audit['passed'] and not audit['failures']
        assert audit['required_complete']
        for name in CHECKPOINTS:
            path=folder/name
            assert sha(path)==audit['artifacts'][name]['sha256'],name
            paths.append(path)
        paths.extend(sorted(folder.glob('*.json')))
        paths.extend([folder/'metrics.jsonl',audit_path])
        inputs=ROOT/'results/lstm_reorientation_mechanism/final'/condition
        collection=json.loads((inputs/'collection.json').read_text())
        assert collection['checkpoint_sha256']==sha(folder/'final.pt')
        paths.extend([inputs/'collection.json',inputs/'implementation_protocol.json',inputs/'native_phase_coverage.json'])
        paths.extend(inputs/'contexts'/f'context{i}.npz' for i in (0,1))
        models.append(dict(condition=condition,worker=worker,
            final_checkpoint_sha256=sha(folder/'final.pt'),checkpoints=list(CHECKPOINTS)))
    paths=sorted(set(paths));entries=[]
    for path in paths:
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=ROOT/'results/rl_blind_reorientation/reproduction_inputs_manifest.json'
    manifest.write_text(json.dumps(dict(models=models,files=entries,file_count=len(entries),
        checkpoint_count=14,uncompressed_bytes=sum(r['bytes'] for r in entries)),indent=2)+'\n')
    names=sorted([r['path'] for r in entries]+[str(manifest.relative_to(ROOT)),
        'results/rl_blind_reorientation/REPRODUCE.md'])
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
        manifest_sha256=sha(manifest),files=len(entries),archive_members=len(names),
        checkpoints=14,every_archived_input_hash_verified=True,generator_sha256=sha(__file__),
        release_url='https://github.com/aaravsinhaofficial/predictive-grid-cells-rl/releases/tag/blind-reorientation-2026-11-06',
        contents='Both40Mmodels: incoming,afterwarmup,all4fixedstages,final; allworkerJSON raw evaluations/configs/provenance/completion,metrics; complete audits; four native context inputs and coverage. No latest/checkpoint selection. Exact final runtime/RNG/RMSProp included.',
        extraction='Repository root, original actual-worker paths; no PGC01 symlink required for model inputs')
    (ROOT/'docs/research/reorientation40_inputs_archive.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
