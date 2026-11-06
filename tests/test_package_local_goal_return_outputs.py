"""Synthetic manifests and bytes only: no study checkpoint or cloud access."""
from pathlib import Path
import copy
import json
import sys
import tarfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import package_local_goal_return_outputs as package


def routing():
    return dict(protocol_sha256=package.PROTOCOL_SHA, workers=[dict(
        condition=condition, name=f'ring-20261105-local-return-{label}-01', zone='test-zone',
        local_artifact_root=f'results/ring_cloud_artifacts/ring-20261105-local-return-{label}-01',
        artifact_subdir='experiment') for condition, label in [('gc100','grid'), ('pgc33','pgc')]])


def cleanup_fixture(root, route):
    rows=[]
    for worker in route['workers']+route.get('prior_attempts', []):
        row={key:worker[key] for key in ('condition','name','zone')}
        row.update(instance_absent=True, disk_absent=True, evidence=[])
        for kind in ('instances','disks'):
            relative=f"results/local_goal_return_design/cloud_audit/{row['name']}_{kind}_absence.log"
            path=root/relative;path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Resource projects/test/zones/test-zone/{kind}/{row['name']} was not found\n")
            row['evidence'].append(dict(path=relative, sha256=package.sha(path)))
        rows.append(row)
    return dict(protocol_sha256=package.PROTOCOL_SHA, complete=True, verified_utc='synthetic', workers=rows)


@pytest.mark.parametrize('value', ['', '.', '/tmp/outside', '../outside', 'a/../../outside'])
def test_unsafe_relative_paths_rejected(value):
    with pytest.raises(AssertionError): package.safe_relative(value)


def test_routing_rejects_duplicate_population_and_other_experiments():
    good=routing();package.routed_attempts(good)
    bad=copy.deepcopy(good);bad['workers'][1]['condition']='gc100'
    with pytest.raises(AssertionError):package.routed_attempts(bad)
    bad=copy.deepcopy(good);bad['workers'][0]['name']='unrelated-worker'
    with pytest.raises(AssertionError):package.routed_attempts(bad)


def test_cleanup_requires_both_resources_and_all_attempts(tmp_path):
    route=routing();proof=cleanup_fixture(tmp_path,route)
    assert len(package.verify_cleanup(tmp_path,route,proof))==4
    bad=copy.deepcopy(proof);bad['workers'][0]['disk_absent']=False
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,bad)
    bad=copy.deepcopy(proof);bad['workers'][0]['evidence'].pop()
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,bad)
    prior=copy.deepcopy(route['workers'][0]);prior['name']=prior['name'].replace('-01','-00')
    prior['local_artifact_root']=prior['local_artifact_root'].replace('-01','-00')
    route['prior_attempts']=[prior]
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,proof)
    expanded=cleanup_fixture(tmp_path,route)
    assert len(package.verify_cleanup(tmp_path,route,expanded))==6


def test_cleanup_rejects_changed_evidence(tmp_path):
    route=routing();proof=cleanup_fixture(tmp_path,route)
    (tmp_path/proof['workers'][0]['evidence'][0]['path']).write_text('different log')
    with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,proof)


def test_full_tree_retains_failed_and_ancestor_evidence(tmp_path):
    folder=tmp_path/'experiment';folder.mkdir()
    names=['final.pt','failed_plan.pt','incomplete.json',
           'resume_parent_evidence/ancestor_evidence/metrics.jsonl']
    for name in names:
        path=folder/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b'{}')
    found=package.experiment_files(tmp_path,'experiment')
    assert {str(p.relative_to(folder)) for p in found}==set(names)
    (folder/'credentials.json').write_text('{}')
    with pytest.raises(AssertionError):package.experiment_files(tmp_path,'experiment')


def test_symlink_and_private_material_rejected(tmp_path):
    outside=tmp_path/'actual';outside.mkdir();(outside/'data.json').write_text('{}')
    (tmp_path/'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(AssertionError):package.checked_path(tmp_path,'link/data.json')
    p=tmp_path/'result.json';p.write_text(json.dumps(dict(access_token='synthetic-private-value')))
    with pytest.raises(AssertionError):package.inspect_file(p)


def test_partial_audit_refused_before_any_file_reads(tmp_path):
    for report in (dict(passed=True,complete=False,status='pending'),
                   dict(passed=False,complete=True,status='failed')):
        with pytest.raises(AssertionError):package.verify_final_audit(tmp_path,tmp_path/'missing','gc100',report)


def test_complete_label_cannot_hide_a_missing_stage(tmp_path):
    source=tmp_path/'scripts/audit_local_goal_return_training.py'
    source.parent.mkdir();source.write_text('# synthetic audit source\n')
    folder=tmp_path/'results/experiment';folder.mkdir(parents=True)
    report=dict(passed=True,complete=True,status='complete',failures=[],pending=[],condition='gc100',
        protocol_sha256=package.PROTOCOL_SHA,folder=str(folder),auditor_sha256=package.sha(source),
        audit_dependency_sha256={},checks=dict(final_exact_stage4=True,final_summary=True),artifacts={})
    for stage in range(1,4):
        report['checks'].update({f'stage{stage}_evaluation_preserved_runtime':True,
            f'stage{stage}_raw_evaluation':{},f'stage{stage}_before_evaluation.pt':{},f'stage{stage}.pt':{}})
    with pytest.raises(KeyError,match='stage4'):
        package.verify_final_audit(tmp_path,folder,'gc100',report)


def test_archive_is_deterministic_and_checks_every_byte(tmp_path):
    (tmp_path/'a.json').write_text('{"synthetic":true}\n')
    (tmp_path/'b.pt').write_bytes(b'fake checkpoint bytes for archive testing only')
    entries=[dict(path=name,bytes=(tmp_path/name).stat().st_size,sha256=package.sha(tmp_path/name))
             for name in ('b.pt','a.json')]
    package.write_archive(tmp_path,tmp_path/'one.tar.gz',entries)
    package.write_archive(tmp_path,tmp_path/'two.tar.gz',entries)
    assert package.sha(tmp_path/'one.tar.gz')==package.sha(tmp_path/'two.tar.gz')
    with tarfile.open(tmp_path/'one.tar.gz') as archive:
        assert archive.getnames()==['a.json','b.pt']
        assert all(m.uid==0 and m.gid==0 and m.mtime==0 and m.isfile() for m in archive)
    (tmp_path/'a.json').write_text('changed')
    with pytest.raises(AssertionError):package.write_archive(tmp_path,tmp_path/'bad.tar.gz',entries)


def stream_fixture(root, historical_stage1=True):
    """Small saved tensors and synthetic stream-only checkpoints; no model/RNG runs."""
    import torch
    from compare_local_goal_return_private_streams import compare_snapshots
    folder=root/package.PRIVATE_FOLDER;folder.mkdir(parents=True,exist_ok=True)
    source=root/package.PRIVATE_SOURCE;source.parent.mkdir(parents=True,exist_ok=True)
    source.write_bytes((ROOT/package.PRIVATE_SOURCE).read_bytes())
    route=routing();files={}
    for row in route['workers']:
        condition=row['condition'];artifacts={};files[condition]={}
        for name in ('stage1.pt','stage2.pt','stage3.pt','stage4.pt','final.pt'):
            path=root/row['local_artifact_root']/row['artifact_subdir']/name
            path.parent.mkdir(parents=True,exist_ok=True)
            updates=4096 if name=='final.pt' else 1024*int(name[5])
            torch.save(dict(streams=dict(seed=294801,batches=updates,contexts=32*updates,
                generators={k:torch.arange(16,dtype=torch.uint8) for k in ('plan','motor','action','state')})),path)
            artifacts[name]=dict(sha256=package.sha(path),bytes=path.stat().st_size)
            files[condition][name]=path
        for stage in (None,1,2,3):
            audit=folder/(f'{condition}.json' if stage is None else f'stage{stage}_{condition}.json')
            audit.write_text(json.dumps(dict(passed=True,failures=[],pending=[],complete=stage is None,
                status='complete' if stage is None else 'partial',condition=condition,
                protocol_sha256=package.PROTOCOL_SHA,artifacts=artifacts)))
    for stage in (None,1,2):
        names={f'stage{i}.pt':1024*i for i in range(1,(4 if stage is None else stage)+1)}
        if stage is None:names['final.pt']=4096
        snapshots={};inputs={};comparisons={}
        for name,updates in names.items():
            snapshot=dict(seed=294801,batches=updates,contexts=32*updates,
                generators={k:torch.arange(16,dtype=torch.uint8) for k in ('plan','motor','action','state')})
            snapshots[name]={c:copy.deepcopy(snapshot) for c in package.CONDITIONS}
            comparisons[name]=compare_snapshots(snapshots[name]['gc100'],snapshots[name]['pgc33'],updates)
            inputs[name]={}
            for condition in package.CONDITIONS:
                audit_name=f'{condition}.json' if stage is None else f'stage{int(name[5])}_{condition}.json'
                audit=folder/audit_name;path=files[condition][name]
                inputs[name][condition]=dict(path=str(path.relative_to(root)),sha256=package.sha(path),
                    audit_path=str(audit.relative_to(root)),audit_sha256=package.sha(audit))
        raw_name='paired_private_streams.pt' if stage is None else f'paired_private_streams_stage{stage}.pt'
        json_name='paired_final_training_validation.json' if stage is None else f'paired_training_streams_stage{stage}.json'
        raw=folder/raw_name;record=folder/json_name
        torch.save(dict(protocol_sha256=package.PROTOCOL_SHA,inputs=inputs,snapshots=snapshots,
            complete=stage is None,global_worker_rng_included=False),raw)
        producer=source
        if stage==1 and historical_stage1:
            producer=folder/'private_streams_stage1_producer.py';producer.write_text('# synthetic historical producer\n')
        record.write_text(json.dumps(dict(complete=stage is None,passed=True,partial_through_stage=stage,
            protocol_sha256=package.PROTOCOL_SHA,global_worker_rng_compared=False,
            reporting_source_sha256=package.sha(producer),inputs=inputs,comparisons=comparisons,
            raw_private_snapshots=dict(file=raw_name,sha256=package.sha(raw),bytes=raw.stat().st_size))))
        if stage==1 and historical_stage1:
            (folder/'private_streams_stage1_producer.json').write_text(json.dumps(dict(
                original_source_path=package.PRIVATE_SOURCE,historical_record=str(record.relative_to(root)),
                record_sha256=package.sha(record),snapshot_path=str(producer.relative_to(root)),
                historical_source_sha256=package.sha(producer),snapshot_sha256=package.sha(producer),
                current_source_sha256=package.PRIVATE_SOURCE_SHA)))
    return route,folder


def test_private_stream_final_and_all_partial_history_preserved(tmp_path):
    route,folder=stream_fixture(tmp_path)
    paths,records=package.private_stream_inventory(tmp_path,route)
    assert [r['partial_through_stage'] for r in records]==[None,1,2]
    assert records[0]['complete'] and records[0]['passed']
    assert all(r['saved_comparisons_recomputed'] for r in records)
    assert all(r['snapshots_equal_original_checkpoint_streams'] for r in records)
    assert folder/'private_streams_stage1_producer.py' in paths
    assert folder/'private_streams_stage1_producer.json' in paths
    assert {p.name for p in paths if p.name.startswith('paired_private_streams')}=={
        'paired_private_streams.pt','paired_private_streams_stage1.pt','paired_private_streams_stage2.pt'}


def test_private_stream_raw_byte_corruption_rejected(tmp_path):
    route,folder=stream_fixture(tmp_path)
    raw=folder/'paired_private_streams.pt';raw.write_bytes(raw.read_bytes()+b'corrupt')
    with pytest.raises(AssertionError):package.private_stream_inventory(tmp_path,route)


def test_private_stream_changed_saved_counter_cannot_hide_behind_updated_hash(tmp_path):
    import torch
    route,folder=stream_fixture(tmp_path)
    raw=folder/'paired_private_streams.pt';saved=torch.load(raw,weights_only=True)
    saved['snapshots']['final.pt']['pgc33']['contexts']-=1;torch.save(saved,raw)
    record=folder/'paired_final_training_validation.json';report=json.loads(record.read_text())
    report['raw_private_snapshots'].update(sha256=package.sha(raw),bytes=raw.stat().st_size)
    record.write_text(json.dumps(report))
    with pytest.raises(AssertionError,match='Changed saved private comparison'):
        package.private_stream_inventory(tmp_path,route)


def test_both_raw_streams_and_recomputed_reports_must_match_original_checkpoints(tmp_path):
    import torch
    from compare_local_goal_return_private_streams import compare_snapshots
    route,folder=stream_fixture(tmp_path)
    raw=folder/'paired_private_streams.pt';saved=torch.load(raw,weights_only=True)
    pair=saved['snapshots']['final.pt']
    for condition in package.CONDITIONS:pair[condition]['generators']['motor'][3]+=1
    torch.save(saved,raw)
    record=folder/'paired_final_training_validation.json';report=json.loads(record.read_text())
    report['comparisons']['final.pt']=compare_snapshots(pair['gc100'],pair['pgc33'],4096)
    assert report['comparisons']['final.pt']['passed']
    report['raw_private_snapshots'].update(sha256=package.sha(raw),bytes=raw.stat().st_size)
    record.write_text(json.dumps(report))
    with pytest.raises(AssertionError,match='Private snapshot differs from original checkpoint'):
        package.private_stream_inventory(tmp_path,route)


@pytest.mark.parametrize('change',['incomplete_final','failed_final','historical_source'])
def test_private_stream_final_and_producer_guards(tmp_path,change):
    route,folder=stream_fixture(tmp_path)
    if change=='historical_source':
        (folder/'private_streams_stage1_producer.py').write_text('# changed historical bytes')
    else:
        path=folder/'paired_final_training_validation.json';record=json.loads(path.read_text())
        record['complete' if change=='incomplete_final' else 'passed']=False
        path.write_text(json.dumps(record))
    with pytest.raises(AssertionError):package.private_stream_inventory(tmp_path,route)
