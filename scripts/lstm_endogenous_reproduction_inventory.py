"""Inventory compact native inputs and archive full fixed-point numerical detail."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from lstm_endogenous_curve_recovery import sha,save


def main():
    out=ROOT/'results/lstm_endogenous_ring';files={}
    def include(path,role):
        assert path.is_file(),path
        relative=str(path.relative_to(ROOT))
        files[relative]=dict(path=relative,role=role,bytes=path.stat().st_size,sha256=sha(path))
    patterns=[
        ('results/original_goal_bearing_manifold/pilot/*_s*/native_contexts.npz','original goal/heading/travel development native histories',4),
        ('results/lstm_endogenous_ring/saturation_audit/*_s*/native_4s_contexts.npz','native4s saturation/quotient histories',4),
        ('results/lstm_heading_memory/continued/*/contexts/context*.npz','final25M native heading histories',4),
        ('results/lstm_travel_memory/continued/*/contexts/context*.npz','final25M physically eligible travel histories',4),
        ('results/lstm_travel_memory/heldout_original/*_s*/contexts/context*.npz','held-out original seeds3–8 travel histories',24)]
    for pattern,role,count in patterns:
        selected=sorted(ROOT.glob(pattern));assert len(selected)==count,(pattern,len(selected))
        for path in selected:
            include(path,role)
            parent=path.parent.parent if path.parent.name=='contexts' else path.parent
            include(parent/'collection.json','native-context collection provenance')
    for condition in ('grid','pgc'):
        folder=ROOT/f'results/ring_cloud_artifacts/ring-20261105-vision-{condition}-01'
        include(folder/'final.pt','new fixed25.024M checkpoint, complete policy/optimizer/RNG')
        for name in ('protocol.json','baseline.json','final_evaluation.json','summary.json'):
            include(folder/name,'new checkpoint provenance/reference behavior')
    previous=json.loads((ROOT/'docs/research/rl_map_checkpoint_archive.json').read_text())
    originals=[r for r in previous['files']if r['path'].startswith('models/')]
    assert len(originals)==32
    for item in originals:assert sha(ROOT/item['path'])==item['sha256']
    inventory=dict(files=list(files.values()),file_count=len(files),total_bytes=sum(f['bytes']for f in files.values()),
        original_checkpoints=dict(previous_release='rl-map-memory-2026-11-06',asset='rl-map-checkpoints.tar.gz',archive_sha256=previous['archive_sha256'],
            manifest='docs/research/rl_map_checkpoint_archive.json',files=originals,repackage=False),
        excluded='Regenerated heading/travel/goal trajectory arrays. The compact actual collector inputs above reproduce those arrays through unchanged analysis helpers.',
        numerical_note='Exact recorded native histories are retained. Rendering/BLAS backend differences can affect regenerated synthetic-state floating-point details; the published checkpoint and input hashes identify the analyzed originals.')
    save(out/'reproduction_inputs_manifest.json',inventory)
    (out/'reproduction_input_paths.txt').write_text(''.join(path+'\n'for path in sorted(files)))
    detailed=[]
    for folder in ('fixed_points','quotient_points','equilibrium_elimination'):
        detailed.extend(sorted((out/folder).glob('*/*.json')));detailed.append(out/folder/'protocol.json')
    archive=out/'endogenous-fixedpoint-detail.tar.gz'
    with archive.open('wb')as sink,gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0,compresslevel=6)as zipped,tarfile.open(fileobj=zipped,mode='w')as tar:
        for path in sorted(detailed):
            data=path.read_bytes();info=tarfile.TarInfo(str(path.relative_to(ROOT)));info.size=len(data);info.mode=0o644;info.mtime=0
            tar.addfile(info,io.BytesIO(data))
    manifest=dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),
        role='All full per-initializer fixed-point, validated-quotient and exact-elimination numerical JSON records plus their frozen protocols; no selected solution.',
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(detailed)])
    save(out/'fixedpoint_detail_archive_manifest.json',manifest)
    print(json.dumps(dict(compact_input_files=len(files),compact_input_bytes=inventory['total_bytes'],detail_archive_bytes=manifest['bytes'],detail_archive_sha256=manifest['sha256']),indent=2))


if __name__=='__main__':main()
