"""Private saved-tensor/log fixtures only. No model, environment, or cloud runs."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import audit_simple_memory_training as a


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def guard(fallback=False):
    result=dict(guard_id=a.producer.GUARD_ID,accepted=True,fallback=fallback,fallback_count=int(fallback),
                mode='bounded_cnn_batch_roundoff' if fallback else 'original',tolerance=dict(rtol=1e-4,atol=1e-5))
    if fallback:
        result.update({k:True for k in a.INVARIANTS})
        result.update(learning_tensors_replaced=False,max_encoder_abs=2e-6,max_raw_logit_abs=3e-5)
    return result


def metrics(start,end,version,attempt=0,fallback_at=()):
    rows=[]
    for u in range(start+1,end+1):
        row=dict(updates=u,frames=u*12800,decisions=u*3200,attempt=attempt,rollout_steps=100,
                 **{k:0.1 for k in a.TECHNICAL})
        row.update(seconds=float(u-start),fps=12800.)
        if version==2:row['replay_guard']=guard(u in fallback_at)
        # Deliberately nonnumeric: auditor must not parse or emit outcome values.
        row.update(avg_return_100='OUTCOME_NOT_READ',goals='OUTCOME_NOT_READ')
        rows.append(row)
    return rows


def saved(cfg, proof, update):
    policy={f'tensor_{i}':torch.tensor([float(i),float(update)]) for i in range(20)}
    optimizer=dict(state={i:dict(square_avg=torch.tensor([.2,.3]),step=torch.tensor(float(update)))
                          for i in range(20)} if update else {},
        param_groups=[dict(params=list(range(20)),lr=.0003,alpha=.99,eps=1e-5,momentum=0,weight_decay=0,centered=False)])
    return dict(schema=1,policy=policy,optimizer=optimizer,
        runtime={'counters':dict(frames=update*12800,updates=update,decisions=update*3200),
                 'env':{'pos':torch.zeros(2,2)},'rng':torch.tensor([1,2,3],dtype=torch.uint8)},
        config=copy.deepcopy(cfg),provenance=copy.deepcopy(proof),frames=update*12800,updates=update,
        policy_training=True,policy_sha256=a.policy_hash(policy))


@pytest.fixture(autouse=True)
def small_endpoint(monkeypatch):
    monkeypatch.setattr(a,'FINAL_UPDATES',6)
    monkeypatch.setattr(a,'FINAL_FRAMES',76800)
    monkeypatch.setattr(a,'FINAL_DECISIONS',19200)
    monkeypatch.setattr(a.recovery.cloud,'run',lambda *x,**y:pytest.fail('Cloud access forbidden'))
    torch.set_num_threads(1)


@pytest.fixture
def make_run(tmp_path):
    def make(kind, label=None):
        run=tmp_path/(label or kind);run.mkdir()
        mode={'v1':'retained_v1','migration':'resume_v2','fresh':'fresh_v2','host':'fresh_v2'}[kind]
        row=dict(mode=mode,condition='gc100',seed=4 if kind=='host' else 2,task_max_gap=0,
                 start_frames=25600 if kind in ('migration','host') else 0)
        cfg={'private':True};base={'source_sha256':{},'protocol_sha256':'private'}
        proof=copy.deepcopy(base) if kind=='v1' else dict(base,provenance_version=2,base_provenance=base)
        initial=saved(cfg,base if kind=='migration' else proof,0)
        if kind!='v1':
            lineage=dict(initial_payload_sha256=a.producer.payload_digest(initial))
            if kind=='migration':
                parent=run/'parent_evidence';parent.mkdir()
                torch.save(initial,parent/'initial.pt');torch.save(saved(cfg,base,2),parent/'resume.pt')
                dump(parent/'config.json',cfg);dump(parent/'provenance.json',base)
                dump(parent/'attempt_000.json',dict(attempt=0,restored_frames=0,restored_updates=0,resume_sha256=None))
                write_rows(parent/'metrics_000.jsonl',metrics(0,3,1));(parent/'episodes_000.jsonl').write_text('opaque outcome bytes')
                pins={p.name:a.sha(p) for p in parent.iterdir()}
                parent_payload=a.producer.payload_digest(saved(cfg,base,2))
                lineage.update(parent_checkpoint_sha256=pins['resume.pt'],parent_evidence_sha256=pins,
                    parent_payload_sha256=parent_payload,migrated_payload_sha256=parent_payload,
                    last_logged_parent_update=3,rolled_back_logged_updates=1,accepted_parent_update_cutoff=2)
            dump(run/'lineage.json',lineage);proof['lineage_sha256']=a.sha(run/'lineage.json')
            if kind!='migration':initial['provenance']=copy.deepcopy(proof)
        torch.save(initial,run/'initial.pt')
        dump(run/'config.json',dict(cfg,population={'private':True}));dump(run/'provenance.json',proof)
        final=saved(cfg,proof,6);torch.save(final,run/'final.pt');torch.save(final,run/'resume.pt')
        endpoint=dict(producer_version=1 if kind=='v1' else 2,frames=76800,updates=6,decisions=19200,
            checkpoint_sha256=a.sha(run/'final.pt'),policy_sha256=final['policy_sha256'])
        if kind=='migration':torch.save(saved(cfg,proof,2),run/'migration_boundary.pt')
        for number in ([0,1] if kind=='host' else [0]):
            start=2 if kind=='migration' or number==1 else 0
            end=2 if kind=='host' and number==0 else 6
            attempt=dict(attempt=number,restored_frames=start*12800,restored_updates=start,
                final_frames=end*12800,final_updates=end,seconds=float(end-start)+.1,
                complete=not(kind=='host' and number==0),exit_code=143 if kind=='host' and number==0 else 0)
            if kind!='v1':
                incoming=saved(cfg,proof,start);torch.save(incoming,run/f'resume_input_{number:03d}.pt')
                attempt.update(producer_version=2,logical_run_id=a.logical_id(row),lineage_sha256=proof['lineage_sha256'],
                    resume_input=f'resume_input_{number:03d}.pt',resume_sha256=a.sha(run/f'resume_input_{number:03d}.pt'))
            else:attempt['resume_sha256']=None
            attempt['resume_sha256_after']=a.sha(run/'resume.pt')
            dump(run/f'attempt_{number:03d}.json',attempt)
            write_rows(run/f'metrics_{number:03d}.jsonl',metrics(start,end,endpoint['producer_version'],number,(4,) if kind=='migration' else ()))
            (run/f'episodes_{number:03d}.jsonl').write_text('opaque outcome bytes')
        if kind=='host':
            attempt=a.read(run/'attempt_000.json');attempt['resume_sha256_after']=a.sha(run/'resume_input_001.pt')
            dump(run/'attempt_000.json',attempt);dump(run/'interrupted_000.json',attempt)
            names=['config.json','provenance.json','initial.pt','lineage.json','attempt_000.json','interrupted_000.json',
                   'resume_input_000.pt','metrics_000.jsonl','episodes_000.jsonl']
            pins={n:a.sha(run/n) for n in names};pins['resume.pt']=a.sha(run/'resume_input_001.pt')
            row['continuation']=dict(start_updates=2,resume_sha256=pins['resume.pt'],
                resume_payload_sha256=a.producer.payload_digest(saved(cfg,proof,2)),source_files_sha256=pins)
        terminal=dict(complete=True,frames=76800,updates=6,decisions=19200,provenance=proof,
            final_sha256=endpoint['checkpoint_sha256'],final_policy_sha256=endpoint['policy_sha256'],
            initial_sha256=a.sha(run/'initial.pt'),attempt_records=['attempt_000.json','attempt_001.json'] if kind=='host' else ['attempt_000.json'])
        if kind!='v1':terminal['lineage_sha256']=proof['lineage_sha256']
        dump(run/'summary.json',terminal)
        return row,run,endpoint
    return make


@pytest.mark.parametrize('kind',['v1','migration','fresh','host'])
def test_four_complete_lineage_shapes(make_run,kind):
    row,run,endpoint=make_run(kind);result=a.audit_run(row,run,endpoint)
    assert result['accepted_updates']==6 and result['exact_final_resume_payload']
    assert result['initial']['policy_tensor_count']==20
    assert result['initial']['optimizer_participation']['initial_optimizer_state_empty']
    assert result['final']['optimizer_participation']['state_entries']==20
    assert result['final']['optimizer_participation']['all_parameter_steps_equal_checkpoint_updates']
    assert result['excluded_parent_suffix_rows']==(1 if kind=='migration' else 0)
    assert result['raw_metric_rows']==(7 if kind=='migration' else 6)
    assert 'OUTCOME_NOT_READ' not in json.dumps(result)
    if kind=='migration':assert result['fallback_updates']==[4]
    if kind=='host':assert [(x['start_update'],x['end_update']) for x in result['attempt_chains']]==[(0,2),(2,6)]


@pytest.mark.parametrize('corruption',['missing','duplicate','fps','nonfinite','bound','invariance','wrong_attempt','partial_line'])
def test_streamed_metrics_reject_corruption(tmp_path,corruption):
    path=tmp_path/'metrics.jsonl';rows=metrics(2,6,2,0,(4,))
    if corruption=='missing':rows.pop()
    if corruption=='duplicate':rows[1]=copy.deepcopy(rows[0])
    if corruption=='fps':rows[0]['fps']=rows[0]['frames']/rows[0]['seconds']
    if corruption=='nonfinite':rows[0]['grad_norm']=float('nan')
    if corruption=='bound':rows[1]['replay_guard']['max_encoder_abs']=.1
    if corruption=='invariance':rows[1]['replay_guard']['rng_unmodified']=False
    if corruption=='wrong_attempt':rows[0]['attempt']=1
    write_rows(path,rows)
    if corruption=='partial_line':path.write_bytes(path.read_bytes().rstrip(b'\n'))
    with pytest.raises(ValueError):a.metric_log(path,attempt=0,start=2,end=6,version=2)


@pytest.mark.parametrize('corruption',['optimizer','nonfinite','initial','input','migration','extra_attempt','rejected','prefix','host_gap'])
def test_saved_chain_corruptions(make_run,corruption):
    kind='host' if corruption in ('prefix','host_gap') else 'migration'
    row,run,endpoint=make_run(kind)
    if corruption in ('optimizer','nonfinite'):
        p=run/'final.pt';s=torch.load(p,weights_only=False)
        s['optimizer']['state'][0]['square_avg'][0]=float('inf') if corruption=='nonfinite' else 99
        torch.save(s,p);terminal=a.read(run/'summary.json');terminal['final_sha256']=a.sha(p)
        endpoint['checkpoint_sha256']=a.sha(p);dump(run/'summary.json',terminal)
    elif corruption=='initial':
        p=run/'initial.pt';s=torch.load(p,weights_only=False);s['runtime']['rng'][0]=9;torch.save(s,p)
        terminal=a.read(run/'summary.json');terminal['initial_sha256']=a.sha(p);dump(run/'summary.json',terminal)
    elif corruption=='input':
        p=run/'resume_input_000.pt';s=torch.load(p,weights_only=False);s['runtime']['rng'][0]=9;torch.save(s,p)
        attempt=a.read(run/'attempt_000.json');attempt['resume_sha256']=a.sha(p);dump(run/'attempt_000.json',attempt)
    elif corruption=='migration':
        p=run/'migration_boundary.pt';s=torch.load(p,weights_only=False);s['optimizer']['state'][0]['square_avg'][0]=99;torch.save(s,p)
    elif corruption=='extra_attempt':dump(run/'attempt_002.json',{})
    elif corruption=='rejected':dump(run/'rejected_update_000.json',{})
    elif corruption=='prefix':(run/'episodes_000.jsonl').write_text('changed opaque prefix')
    elif corruption=='host_gap':
        p=run/'metrics_001.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows.pop(0);write_rows(p,rows)
    with pytest.raises((ValueError,AssertionError)):a.audit_run(row,run,endpoint)


def test_same_values_different_tensor_dtype_changes_typed_digest():
    x={'x':torch.tensor([1.],dtype=torch.float32)};y={'x':torch.tensor([1.],dtype=torch.float64)}
    assert a.producer.tree_digest(x)!=a.producer.tree_digest(y)


def test_cli_partial_is_explicit_and_outputs_are_immutable(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'audit_cohort',lambda *x:dict(complete=False,cohort_complete=False,validation_passed=True,
        counts={'complete':6,'pending':10,'invalid':0}))
    p=tmp_path/'strict';assert a.main(['--out',str(p)])==1
    p=tmp_path/'partial';assert a.main(['--out',str(p),'--allow-partial'])==0
    assert a.read(p/'summary.json')['cohort_complete'] is False
    with pytest.raises(FileExistsError):a.main(['--out',str(p),'--allow-partial'])
    monkeypatch.setattr(a,'audit_cohort',lambda *x:dict(complete=False,cohort_complete=False,validation_passed=False))
    assert a.main(['--out',str(tmp_path/'invalid'),'--allow-partial'])==1


def test_cohort_pending_complete_and_initial_equality(tmp_path,monkeypatch,make_run):
    # Real audit_run on tiny saved arrays; only already-tested external auth is stubbed.
    rows=[];state={};ledger=[];original={}
    for seed in range(1,5):
        for cond in ('gc100','pgc33'):
            for gap in (0,60):
                name=f'{cond}-{gap}-{seed}';row=dict(logical_name=name,condition=cond,seed=seed,task_max_gap=gap,mode='fresh_v2',name=name)
                rows.append(row);original[name]={};state[name]={}
    selected=[]
    for index,row in enumerate(rows[:2]):
        _,oldrun,endpoint=make_run('fresh', 'fresh'+str(index))
        import shutil
        run=tmp_path/'results/ring_cloud_artifacts'/row['name']/f"seed{row['seed']}"
        run.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(oldrun,run)
        attempt=a.read(run/'attempt_000.json');attempt['logical_run_id']=a.logical_id(row);dump(run/'attempt_000.json',attempt)
        selected.append((row,run,endpoint));state[row['logical_name']]=dict(complete=True,run_dir=str(run.relative_to(tmp_path)),endpoint=endpoint)
        ledger.append(dict(name=row['name']))
    plan=tmp_path/'plan.json';status=tmp_path/'status.json';led=tmp_path/'ledger.json';script=tmp_path/'scripts/run_simple_memory_recovery.py'
    script.parent.mkdir();script.write_text('private controller')
    for name in ('scripts/simple_memory_endpoint.py','tests/test_audit_simple_memory_training.py'):
        p=tmp_path/name;p.parent.mkdir(exist_ok=True);p.write_text('private source')
    dump(plan,dict(models=rows));dump(status,dict(binding=dict(plan_sha256=a.sha(plan),supervisor_sha256=a.sha(script)),models=state));dump(led,dict(runs=ledger))
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'registry',lambda *x:original)
    monkeypatch.setattr(a.recovery,'authenticate',lambda row,run,plan:state[row['logical_name']]['endpoint'])
    monkeypatch.setattr(a,'worker_evidence',lambda *x:{'private':True})
    out=tmp_path/'out';out.mkdir();result=a.audit_cohort(plan,status,led,out)
    assert result['counts']==dict(complete=2,pending=14,invalid=0)
    assert result['initial_policy_equality_by_seed']['1']['equal_among_available']
    actual=a.audit_run
    def different_initial(row,run,endpoint):
        value=actual(row,run,endpoint)
        if row is rows[1] or row['logical_name']==rows[1]['logical_name']:
            value['initial']['policy_sha256']='different verified initial policy'
        return value
    monkeypatch.setattr(a,'audit_run',different_initial)
    different=tmp_path/'different';different.mkdir();result=a.audit_cohort(plan,status,led,different)
    assert result['counts']['complete']==2 and not result['validation_passed']
    assert not result['initial_policy_equality_by_seed']['1']['equal_among_available']
    monkeypatch.setattr(a,'audit_run',actual)
    # A completed row whose evidence disappears is invalid, never demoted to pending.
    selected[1][1].joinpath('resume.pt').unlink()
    out2=tmp_path/'out2';out2.mkdir();result=a.audit_cohort(plan,status,led,out2)
    assert result['counts']==dict(complete=1,pending=14,invalid=1) and not result['validation_passed']


def registry_fixture(tmp_path):
    rows=[];old=[];bindings={}
    resumes={('pgc33',1,0),('pgc33',1,60),('gc100',2,0),('pgc33',2,0)}
    for seed in range(1,5):
        for cond in ('gc100','pgc33'):
            for gap in (0,60):
                task='native' if gap==0 else 'blackout';name=f'ring-20261105-simplemem-{cond}-{task}-s{seed}'
                mode='resume_v2' if (cond,seed,gap) in resumes else 'fresh_v2' if seed>=3 else 'retained_v1'
                row=dict(logical_name=name,condition=cond,seed=seed,task_max_gap=gap,mode=mode)
                rows.append(row);old.append(dict(row,name=name))
                bindings[a.logical_id(row)]=dict(logical_job_name=name,kind={'retained_v1':'v1_completed','resume_v2':'v1_resume','fresh_v2':'v2_fresh'}[mode])
    host=next(r for r in rows if (r['condition'],r['seed'],r['task_max_gap'])==('gc100',4,0))
    host.update(name=host['logical_name']+'-r1',start_frames=9996800,
        trainer_output_dir='results/ring_simple_memory_v2/gc100/native/seed4',worker_output_dir='results/ring_simple_memory_v2/gc100/native')
    host['continuation']=dict(kind='explicit_v2_host_recovery',source_worker_name=host['logical_name'],
        source_instance_id='2749035052516857068',source_run=f"results/ring_cloud_artifacts/{host['logical_name']}/seed4",
        source_attempt=0,attempt=1,start_frames=9996800,start_updates=781,
        source_files_sha256={'resume.pt':'a'*64},resume_sha256='a'*64,resume_payload_sha256='b'*64)
    dump(tmp_path/'original.json',dict(jobs=old));dump(tmp_path/'amendment.json',dict(run_bindings=bindings))
    (tmp_path/'source.py').write_text('private source');(tmp_path/'validation.json').write_text('{}')
    return dict(status='frozen',models=rows,
        operational_source_sha256={'source.py':a.sha(tmp_path/'source.py')},
        validation_artifacts_sha256={'validation.json':a.sha(tmp_path/'validation.json')},
        bindings={k:dict(path=n,sha256=a.sha(tmp_path/n)) for k,n in [('original_plan','original.json'),('amendment','amendment.json')]})


@pytest.mark.parametrize('fault',[None,'duplicate','source','mode','host','registered_identity'])
def test_registry_has_exact_sixteen_bound_roles(tmp_path,fault):
    plan=registry_fixture(tmp_path)
    if fault=='duplicate':plan['models'][1]=plan['models'][0]
    elif fault=='source':(tmp_path/'source.py').write_text('changed')
    elif fault=='mode':plan['models'][0]['mode']='fresh_v2'
    elif fault=='host':next(r for r in plan['models'] if 'continuation' in r).pop('continuation')
    elif fault=='registered_identity':plan['models'][0]['logical_name']='unregistered'
    if fault:
        with pytest.raises((ValueError,AssertionError)):a.registry(plan,tmp_path)
    else:assert len(a.registry(plan,tmp_path))==16


@pytest.mark.parametrize('fault',[None,'package','command','interpreter','worker_tail','exit','missing_collection','file','instance'])
def test_retained_worker_package_terminal_authentication(tmp_path,monkeypatch,make_run,fault):
    row,run,_=make_run('v1');monkeypatch.setattr(a,'ROOT',tmp_path)
    row.update(logical_name='original',artifacts_sha256={str((run/'initial.pt').relative_to(tmp_path)):a.sha(run/'initial.pt')})
    argv=['python','-m','private_never_executed'];manifest=[dict(path='protocol.json',sha256='private')]
    record=dict(name='original',instance_id='123',status='DELETED',collected_at='private',
        local_artifacts=str(run.parent.relative_to(tmp_path)),command=argv,output_dir='private',
        package_manifest=manifest,package_manifest_sha256=a.json_hash(manifest),gcs_prefix='gs://private')
    done=dict(exit_code=0,command=['/usr/bin/python3',*argv[1:]],package_sha256=record['package_manifest_sha256'])
    record['worker_result']=copy.deepcopy(done);dump(run.parent/'WORKER_RESULT.json',done)
    original=dict(train_argv=argv,worker_output_dir='private');plan={'bindings':{'protocol':{'path':'protocol.json'}}}
    collection=tmp_path/a.COLLECTION_SUMMARY
    receipt=collection.parent/'private.json'
    dump(receipt,dict(complete=True,name='original',instance_id='123',gcs_prefix='gs://private',
        condition=row['condition'],training_seed=row['seed'],training_task_max_gap=0,final={'sha256':a.sha(run/'final.pt')}))
    dump(collection,dict(jobs=[dict(name='original',complete=True,receipt='private.json',receipt_sha256=a.sha(receipt))]))
    plan['validation_artifacts_sha256']={a.COLLECTION_SUMMARY:a.sha(collection)}
    if fault=='package':record['package_manifest'][0]['sha256']='corrupt'
    elif fault=='command':record['command']=['different']
    elif fault in ('interpreter','worker_tail'):
        done['command'][0 if fault=='interpreter' else -1]='unexpected'
        record['worker_result']=copy.deepcopy(done);dump(run.parent/'WORKER_RESULT.json',done)
    elif fault=='exit':done['exit_code']=1;record['worker_result']=done;dump(run.parent/'WORKER_RESULT.json',done)
    elif fault=='missing_collection':record.pop('collected_at')
    elif fault=='file':(run/'initial.pt').write_bytes(b'changed')
    elif fault=='instance':record['instance_id']='999'
    if fault:
        with pytest.raises((ValueError,AssertionError)):a.worker_evidence(row,{},run,record,original,plan)
    else:assert a.worker_evidence(row,{},run,record,original,plan)['exit_code']==0


@pytest.mark.parametrize('fault',['optimizer','other_counter','initial_mode','final_mode'])
def test_fresh_start_and_training_mode_are_explicit(tmp_path,fault):
    cfg={'private':True};proof={'private':True};update=6 if fault=='final_mode' else 0
    s=saved(cfg,proof,update)
    if fault=='optimizer':s['optimizer']['state']={0:{'step':torch.tensor(1)}}
    elif fault=='other_counter':s['runtime']['counters']['episodes']=1
    else:s['policy_training']=False
    p=tmp_path/'checkpoint.pt';torch.save(s,p)
    with pytest.raises(ValueError):a.checkpoint(p,cfg,proof,update)


@pytest.mark.parametrize('updates',[0,2,6])
def test_all_policy_optimizer_coverage_at_initial_boundary_final(tmp_path,updates):
    cfg={'private':True};proof={'private':True};s=saved(cfg,proof,updates)
    path=tmp_path/'snapshot.pt';torch.save(s,path)
    result=a.checkpoint(path,cfg,proof,updates)['optimizer_participation']
    assert result['parameter_groups']==1 and result['unique_parameter_ids']==list(range(20))
    assert result['state_entries']==(20 if updates else 0)
    assert result['checkpoint_updates']==updates
    assert result['all_parameter_steps_equal_checkpoint_updates'] is (True if updates else None)


@pytest.mark.parametrize('fault',['missing_param','duplicate_param','missing_state','extra_state','lagging_step',
    'changed_lr','changed_alpha','changed_eps','momentum','weight_decay','centered','extra_group','state_field','accumulator_shape','relabeled_ids'])
def test_optimizer_participation_corruption_rejected(tmp_path,fault):
    cfg={'private':True};proof={'private':True};s=saved(cfg,proof,6)
    optimizer=s['optimizer'];group=optimizer['param_groups'][0]
    if fault=='missing_param':group['params'].pop()
    elif fault=='duplicate_param':group['params'][-1]=0
    elif fault=='missing_state':optimizer['state'].pop(19)
    elif fault=='extra_state':optimizer['state'][20]=copy.deepcopy(optimizer['state'][0])
    elif fault=='lagging_step':optimizer['state'][19]['step']=torch.tensor(5.)
    elif fault=='changed_lr':group['lr']=.001
    elif fault=='changed_alpha':group['alpha']=.9
    elif fault=='changed_eps':group['eps']=1e-8
    elif fault=='momentum':group['momentum']=.1
    elif fault=='weight_decay':group['weight_decay']=.01
    elif fault=='centered':group['centered']=True
    elif fault=='extra_group':optimizer['param_groups'].append(copy.deepcopy(group))
    elif fault=='state_field':optimizer['state'][0]['momentum_buffer']=torch.zeros(2)
    elif fault=='accumulator_shape':optimizer['state'][0]['square_avg']=torch.zeros(3)
    elif fault=='relabeled_ids':
        group['params']=list(range(20,40))
        optimizer['state']={k+20:v for k,v in optimizer['state'].items()}
    path=tmp_path/'snapshot.pt';torch.save(s,path)
    with pytest.raises(ValueError):a.checkpoint(path,cfg,proof,6)
