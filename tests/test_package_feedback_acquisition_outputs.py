"""Synthetic metadata/archives only; no study policy, rollout, training or cloud."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import package_feedback_acquisition_outputs as package


def put(root, name, value):
    path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value,str) else value)
    return path


def routing():
    rows=[]
    for condition,label in [('gc100','grid'),('pgc33','pgc')]:
        for mode,short in [('retained_actor','retain'),('reset_actor','reset')]:
            name=f'ring-20261105-feedback-{label}-{short}-02'
            rows.append(dict(condition=condition,actor_mode=mode,name=name,zone='test-zone',
                local_artifact_root='results/ring_cloud_artifacts/'+name,artifact_subdir='experiment',
                audit_path=package.AUDIT+f'/{condition}_{mode}.json',status=package.COMPLETE))
    return dict(protocol_sha256=package.PROTOCOL_SHA,workers=rows)


def cleanup(root, route):
    rows=[]
    for worker in route['workers']:
        row={k:worker[k] for k in ('name','condition','actor_mode','zone')}
        row.update(instance_absent=True,disk_absent=True,evidence=[])
        for kind in ('instances','disks'):
            path=put(root,f"{package.AUDIT}/{row['name']}_{kind}_absence.log",
                     f"Resource projects/test/zones/test-zone/{kind}/{row['name']} was not found\n")
            row['evidence'].append(package.file_record(root,path))
        rows.append(row)
    result=dict(complete=True,protocol_sha256=package.PROTOCOL_SHA,verified_utc='synthetic',workers=rows)
    put(root,package.AUDIT+'/cleanup_verification.json',result)
    return result


@pytest.mark.parametrize('value',['','.','/tmp/outside','../outside','a/../../outside'])
def test_path_traversal_rejected(tmp_path,value):
    with pytest.raises(AssertionError):package.checked_path(tmp_path,value)


def test_symlinks_secrets_and_unexpected_files_rejected(tmp_path):
    p=put(tmp_path,'real/data.json',{})
    (tmp_path/'linked').symlink_to(p.parent,target_is_directory=True)
    with pytest.raises(AssertionError):package.checked_path(tmp_path,'linked/data.json')
    bad=put(tmp_path,'real/credentials.json',{})
    with pytest.raises(AssertionError):package.tree(tmp_path,'real')
    bad.unlink();bad=put(tmp_path,'real/result.json',dict(access_token='synthetic-private-value'))
    with pytest.raises(AssertionError):package.inspect_file(bad)
    bad.unlink();bad=put(tmp_path,'real/pending.tmp','')
    with pytest.raises(AssertionError):package.tree(tmp_path,'real')


def test_routing_requires_four_distinct_terminal_arms():
    route=routing();package.routes(route,package.PROTOCOL_SHA)
    bad=copy.deepcopy(route);bad['workers'][0]['status']='RUNNING'
    with pytest.raises(AssertionError,match='terminal'):package.routes(bad,package.PROTOCOL_SHA)
    bad=copy.deepcopy(route);bad['workers'][0]['actor_mode']='reset_actor'
    with pytest.raises(AssertionError):package.routes(bad,package.PROTOCOL_SHA)
    bad=copy.deepcopy(route);bad['workers'][0]['local_artifact_root']='results/unrelated'
    with pytest.raises(AssertionError):package.routes(bad,package.PROTOCOL_SHA)


def test_cleanup_requires_every_arm_both_resources_and_exact_evidence(tmp_path):
    route=routing();proof=cleanup(tmp_path,route);name=package.AUDIT+'/cleanup_verification.json'
    package.verify_cleanup(tmp_path,route,name,package.PROTOCOL_SHA)
    for alteration in ('missing_arm','missing_disk','changed_hash'):
        bad=copy.deepcopy(proof)
        if alteration=='missing_arm':bad['workers'].pop()
        if alteration=='missing_disk':bad['workers'][0]['evidence'].pop()
        if alteration=='changed_hash':bad['workers'][0]['evidence'][0]['sha256']='0'*64
        put(tmp_path,name,bad)
        with pytest.raises(AssertionError):package.verify_cleanup(tmp_path,route,name,package.PROTOCOL_SHA)


def audit_fixture(root):
    worker=routing()['workers'][0];folder=root/worker['local_artifact_root']/'experiment'
    source=put(root,'scripts/audit_local_return_feedback.py','# synthetic auditor\n')
    evaluations={};names=set(package.CHECKPOINTS)|{'protocol.json','training_config.json','initialization.json','metrics.jsonl','summary.json'}
    for endpoint in ('baseline','final'):
        contexts=[dict(context=i,valid=True) for i in range(16)]
        evaluations[endpoint]=dict(complete=True,totals=dict(trials=256),contexts=contexts)
        names.update({f'{endpoint}_evaluation/summary.json',f'{endpoint}_evaluation_integrity.json'})
        for i in range(16): names.update(f'{endpoint}_evaluation/context{i}/{s}' for s in ('input.json','result.json','raw.pt.gz'))
    for name in names:put(folder,name,'synthetic')
    worker['final_checkpoint_sha256']=package.sha(folder/'final.pt')
    put(folder,'summary.json',dict(completed=True,protocol_sha256=package.PROTOCOL_SHA,
        counters=dict(updates=1024,trials=32768,evaluation_records=dict(baseline={},final={})),
        final_checkpoint_sha256=worker['final_checkpoint_sha256']))
    report=dict(protocol_sha256=package.PROTOCOL_SHA,condition=worker['condition'],actor_mode=worker['actor_mode'],
        auditor_sha256=package.sha(source),folder=str(folder),passed=True,complete=True,status='complete',failures=[],pending=[],
        checks=dict(evaluations=evaluations),artifacts={n:{k:v for k,v in package.file_record(folder,folder/n).items() if k!='path'} for n in names})
    path=put(root,worker['audit_path'],report);worker['audit_sha256']=package.sha(path)
    return worker,report,folder


def test_complete_audit_requires_every_raw_context_and_checkpoint(tmp_path):
    worker,report,folder=audit_fixture(tmp_path)
    package.verify_audit(tmp_path,worker)
    bad=copy.deepcopy(report);bad['artifacts'].pop('final_evaluation/context15/raw.pt.gz')
    path=put(tmp_path,worker['audit_path'],bad);worker['audit_sha256']=package.sha(path)
    with pytest.raises(AssertionError,match='Missing fixed'):package.verify_audit(tmp_path,worker)
    path=put(tmp_path,worker['audit_path'],report);worker['audit_sha256']=package.sha(path)
    (folder/'final.pt').write_text('changed')
    with pytest.raises(AssertionError):package.verify_audit(tmp_path,worker)


def test_terminal_failure_is_explicit_and_never_completed(tmp_path):
    worker,report,folder=audit_fixture(tmp_path)
    worker['status']='failed_collected_deleted'
    report.update(complete=False,passed=False,status='failed',failures=['synthetic failure'],pending=['final'])
    put(tmp_path,worker['audit_path'],report)
    with pytest.raises(AssertionError,match='incomplete arm'):package.verify_audit(tmp_path,worker)
    result=package.verify_audit(tmp_path,worker,terminal_only=True)
    assert result['status']=='failed' and not result['complete']
    put(folder,'failed_plan.pt','saved failed plan')
    put(folder,'resume_parent_evidence/metrics.jsonl','rolled-back attempt')
    names={str(p.relative_to(folder)) for p in package.tree(tmp_path,folder.relative_to(tmp_path))}
    assert {'failed_plan.pt','resume_parent_evidence/metrics.jsonl'}<=names


def test_archive_is_deterministic_has_manifest_and_exact_members(tmp_path):
    p=put(tmp_path,'study/final.json',dict(synthetic=True))
    manifest=dict(group='grid',synthetic=True,files=[package.file_record(tmp_path,p)])
    a=package.write_archive(tmp_path,tmp_path/'a.tar.gz',manifest)
    b=package.write_archive(tmp_path,tmp_path/'b.tar.gz',manifest)
    assert a['sha256']==b['sha256'] and a['members']==2
    with tarfile.open(tmp_path/'a.tar.gz','r:gz') as tar:
        assert all(m.isfile() and m.mtime==0 and m.mode==0o644 for m in tar.getmembers())
        saved=json.load(tar.extractfile(a['manifest']['path']));assert saved==manifest
    with pytest.raises(AssertionError,match='overwrite'):package.write_archive(tmp_path,tmp_path/'a.tar.gz',manifest)


def test_archive_rejects_changed_bytes_and_size_limit_without_publishing(tmp_path):
    p=put(tmp_path,'synthetic.json',dict(synthetic=True));record=package.file_record(tmp_path,p)
    manifest=dict(group='common',synthetic=True,files=[record])
    p.write_text('changed')
    with pytest.raises(AssertionError):package.write_archive(tmp_path,tmp_path/'bad.tar.gz',manifest)
    assert not (tmp_path/'bad.tar.gz').exists() and not (tmp_path/'bad.tar.gz.tmp').exists()
    manifest['files']=[package.file_record(tmp_path,p)]
    with pytest.raises(AssertionError,match='size limit'):package.write_archive(tmp_path,tmp_path/'large.tar.gz',manifest,max_bytes=1)
    assert not (tmp_path/'large.tar.gz').exists() and not (tmp_path/'large.tar.gz.tmp').exists()


def test_synthetic_preflight_reads_no_study_inputs():
    assert package.synthetic_preflight()==dict(synthetic=True,passed=True,deterministic_archive=True,member_bytes_verified=True,no_study_inputs_read=True)


def stream_fixture(root):
    import torch
    route=routing();put(root,package.ROUTING,route)
    producer=put(root,'scripts/compare_feedback_training_streams.py','# synthetic source\n')
    snapshots={};inputs=[]
    for worker in route['workers']:
        state=dict(seed=494801,batches=1024,contexts=32768,
                   generators={k:torch.tensor([1,2,3],dtype=torch.uint8) for k in ('plan','motor','action','state')})
        snapshots[worker['name']]=copy.deepcopy(state)
        checkpoint=root/worker['local_artifact_root']/'experiment/final.pt';checkpoint.parent.mkdir(parents=True)
        counters=dict(updates=1024,trials=32768)
        torch.save(dict(streams=state,counters=counters,active_physics_frames=123),checkpoint)
        audit=put(root,worker['audit_path'],dict(synthetic=True))
        inputs.append(dict(name=worker['name'],checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=package.sha(checkpoint),
            audit=worker['audit_path'],audit_sha256=package.sha(audit),counters=counters,active_physics_frames=123))
    raw=root/package.AUDIT/'paired_final_private_streams.pt';torch.save(snapshots,raw)
    report=dict(complete=True,passed=True,protocol_sha256=package.PROTOCOL_SHA,routing_sha256=package.sha(root/package.ROUTING),
        producer_sha256=package.sha(producer),checks={k:True for k in ('plan','motor','action','state')},
        raw=package.file_record(root,raw),inputs=inputs,scope='synthetic')
    path=put(root,package.AUDIT+'/paired_final_training_validation.json',report)
    return route,snapshots,raw,report,path


def test_private_stream_recomputed_hash_cannot_hide_checkpoint_mismatch(tmp_path):
    import torch
    route,snapshots,raw,report,path=stream_fixture(tmp_path)
    package.verify_private_streams(tmp_path,route)
    for state in snapshots.values():state['generators']['motor'][0]=99
    torch.save(snapshots,raw);report['raw']=package.file_record(tmp_path,raw);path.write_text(json.dumps(report))
    with pytest.raises(AssertionError,match='differs from checkpoint'):package.verify_private_streams(tmp_path,route)
