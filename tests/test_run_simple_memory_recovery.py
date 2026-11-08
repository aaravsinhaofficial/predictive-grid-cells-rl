"""Private operational fixtures only: no scientific checkpoints or cloud requests."""
import argparse
import contextlib
import copy
import datetime as dt
import json
import hashlib
from pathlib import Path
import shlex
import shutil
from unittest import mock
import sys
import types

import pytest

from scripts import run_simple_memory_recovery as s


@pytest.fixture(autouse=True)
def no_cloud(monkeypatch):
    def fail(*a, **k): raise AssertionError('Unmocked cloud call')
    monkeypatch.setattr(s.cloud, 'run', fail)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    monkeypatch.setattr(s.cloud, 'ROOT', tmp_path)
    monkeypatch.setattr(s.cloud, 'LEDGER', tmp_path/'ledger.json')
    source = ['scripts/run_simple_memory_recovery.py', 'scripts/ring_cloud.py',
        'scripts/run_simple_memory_cohort.py', 'scripts/simple_memory_endpoint.py',
        'scripts/run_simple_memory_analysis.py',
        'tests/test_run_simple_memory_recovery.py', 'code/train_memory.py', 'code/train_memory_v2.py']
    for name in source:
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('private fixture '+name)
    protocol = 'docs/research/simple_memory_training_protocol.json'
    amendment = 'docs/research/simple_memory_guard_amendment.json'
    oldpath = 'docs/research/simple_memory_cloud_plan.json'
    envelope = 'results/private/budget.json'
    dump(tmp_path/protocol, dict(status='frozen', source_sha256={'code/train_memory.py':s.sha(tmp_path/'code/train_memory.py')}))
    dump(tmp_path/amendment, dict(status='frozen', source_sha256={'code/train_memory_v2.py':s.sha(tmp_path/'code/train_memory_v2.py')}))
    resumes = {('pgc33', 1, 60):(345600000,17.5), ('pgc33',1,0):(570880000,11.5),
        ('gc100',2,0):(855040000,4.25), ('pgc33',2,0):(880640000,3.75)}
    old=[]; models=[]; cont=[]
    for seed in range(1,5):
        for cond in ('gc100','pgc33'):
            for gap in (0,60):
                task='native' if gap==0 else 'blackout'; logical=f'ring-20261105-simplemem-{cond}-{task}-s{seed}'
                out=f'results/ring_simple_memory/{cond}/{task}/seed{seed}'
                original=['python','-m','code.train_memory','--cond',cond,'--seed',str(seed),'--code_seed',str(seed),
                    '--task_max_gap',str(gap),'--frames','1000000000','--out',out,'--protocol',protocol,'--tf32','0']
                old.append(dict(name=logical, condition=cond, seed=seed, task_max_gap=gap, train_argv=original))
                mode='resume_v2' if (cond,seed,gap) in resumes else ('fresh_v2' if seed>=3 else 'retained_v1')
                row=dict(logical_name=logical, mode=mode, condition=cond, seed=seed, task_max_gap=gap)
                if mode=='retained_v1':
                    names=('summary.json','final.pt','initial.pt','config.json','provenance.json')
                    for n in names:dump(tmp_path/out/n, {'private_fixture':True})
                    row.update(run_dir=out,start_frames=s.TARGET,artifacts_sha256={out+'/'+n:s.sha(tmp_path/out/n) for n in names})
                else:
                    frames,hours=resumes.get((cond,seed,gap),(0,25.5))
                    name=logical+'-r1' if mode=='resume_v2' else logical
                    worker=f'results/ring_simple_memory_v2/{name}'; target=worker+f'/seed{seed}'
                    argv=list(original); argv[2]='code.train_memory_v2'; argv[argv.index('--out')+1]=target
                    argv+=['--amendment',amendment]
                    files=['code/train_memory.py','code/train_memory_v2.py',protocol,amendment]
                    if mode=='resume_v2':
                        parent=f'results/private_parents/{logical}'
                        for n in ('resume.pt','initial.pt','config.json','provenance.json'):
                            p=tmp_path/parent/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('immutable '+n)
                        pins={parent+'/'+n:s.sha(tmp_path/parent/n) for n in ('resume.pt','initial.pt','config.json','provenance.json')}
                        row.update(parent_run=parent,parent_artifacts_sha256=pins)
                        files+=list(pins);argv+=['--parent-run',parent]
                        cont.append(dict(logical_name=logical,saved_frames=frames,provider_hours=hours))
                    manifest={f:s.sha(tmp_path/f) for f in files}
                    planargv=['python','scripts/ring_cloud.py','plan','--name',name,'--machine','n1-standard-8',
                        '--zone','us-central1-a','--hours',str(hours),'--output-dir',worker,'--command',shlex.join(argv),'--include-only']
                    for f in manifest:planargv+=['--include',f]
                    row.update(name=name,start_frames=frames,provider_hours=hours,reserved_usd=(hours+.25)*1.25,
                        worker_output_dir=worker,trainer_output_dir=target,train_argv=argv,plan_argv=planargv,package_manifest=manifest)
                models.append(row)
    dump(tmp_path/oldpath,dict(status='frozen',jobs=old));dump(tmp_path/envelope,dict(continuations=cont))
    amended=json.loads((tmp_path/amendment).read_text())
    amended.update(parent_protocol_sha256=s.sha(tmp_path/protocol),guard={'id':'private_fixture_guard','status':'frozen'},
                   numerical_runtime={'torch_version':'private-fixture'},run_bindings={})
    for r in models:
        key=f'{r["condition"]}/{"native" if r["task_max_gap"]==0 else "blackout"}/seed{r["seed"]}'
        binding={'kind':{'retained_v1':'v1_completed','resume_v2':'v1_resume','fresh_v2':'v2_fresh'}[r['mode']],
            'logical_job_name':r['logical_name']}
        if r['mode']=='retained_v1':binding.update(original_run=r['run_dir'],final_checkpoint_sha256=r['artifacts_sha256'][r['run_dir']+'/final.pt'])
        if r['mode']=='resume_v2':
            prefix=r['parent_run']+'/'
            binding.update(original_run=r['parent_run'],parent_evidence_sha256={k[len(prefix):]:v for k,v in r['parent_artifacts_sha256'].items()},parent_checkpoint_sha256=r['parent_artifacts_sha256'][prefix+'resume.pt'])
        amended['run_bindings'][key]=binding
    dump(tmp_path/amendment,amended)
    for r in models:
        if r['mode']!='retained_v1':r['package_manifest'][amendment]=s.sha(tmp_path/amendment)
    registry='results/private/analysis_registry.json';vp='results/private/gpu_protocol.json';vc='results/private/gpu_cloud.json'
    jobs=[dict(name=f'ring-20261105-simplemem-analysis-{"native" if gap==0 else "blackout"}-s{seed}',
        seed=seed,task_max_gap=gap) for seed in range(1,5) for gap in (0,60)]
    dump(tmp_path/registry,dict(status='frozen',jobs=jobs))
    counts={('pgc33',1,60):188,('pgc33',1,0):84,('gc100',2,0):45,('pgc33',2,0):58}
    cases=[]
    for r in models:
        if r['mode']=='resume_v2':
            key=f'{r["condition"]}/{"native" if r["task_max_gap"]==0 else "blackout"}/seed{r["seed"]}'
            start=r['start_frames']//12800;count=counts[(r['condition'],r['seed'],r['task_max_gap'])]
            cases.append(dict(logical_run_id=key,start_update=start,last_successful_update=start+count))
    vpdata=dict(status='frozen',amendment_sha256=s.sha(tmp_path/amendment),source_sha256=amended['source_sha256'],
        required_total_accepted_updates=375,required_corruption_controls=sorted(s.CORRUPTIONS),cases=cases)
    dump(tmp_path/vp,vpdata)
    expected=dict(name=s.VALIDATION_NAME,zone='us-central1-a',machine='n1-standard-8',command=['python','private_gpu_fixture.py'],
        output_dir='results/private/gpu',package_manifest_sha256='gpu-package')
    dump(tmp_path/vc,dict(status='frozen',prepared_cloud_plan=expected))
    report=dict(complete=True,sources_unchanged=True,all_input_bytes_preserved=True,
        manifest_sha256=s.sha(tmp_path/vp),amendment_sha256=s.sha(tmp_path/amendment),source_sha256=vpdata['source_sha256'],cases=[])
    for c in cases:
        r=dict(logical_run_id=c['logical_run_id'],complete=True,accepted_reconstructed=c['last_successful_update']-c['start_update'],
            failed_update=c['last_successful_update']+1,original_assertion_failure='private expected failure',original_update_metrics_exact=True,
            full_state_and_rng_exact=True,gradients_exact=True,fallback={'fallback':True,'accepted':True},reference_payload_sha256='payload',
            guarded_payload_sha256='payload',reference_gradients_sha256='gradients',guarded_gradients_sha256='gradients')
        if c['logical_run_id']=='pgc33/blackout/seed1':r.update(all_188_updates_full_boundary_exact=True,corruption_controls={k:{'rejected':True} for k in s.CORRUPTIONS})
        report['cases'].append(r)
    report_path='results/private/gpu/summary.json';done_path='results/private/gpu/DONE.json'
    dump(tmp_path/report_path,report);dump(tmp_path/done_path,dict(exit_code=0,package_sha256='gpu-package'))
    proof='results/private/validation.json';dump(tmp_path/proof,dict(complete=True))
    plan=dict(status='frozen',max_active_workers=8,cumulative_ceiling_usd=600,readiness={'tests_complete':True},
        operational_source_sha256={f:s.sha(tmp_path/f) for f in source},validation_artifacts_sha256={proof:s.sha(tmp_path/proof)},
        bindings={k:dict(path=f,sha256=s.sha(tmp_path/f)) for k,f in [('protocol',protocol),('amendment',amendment),('original_plan',oldpath),('budget_envelope',envelope),
            ('analysis_registry',registry),('validation_protocol',vp),('validation_cloud_plan',vc)]},
        gpu_release=dict(status='complete',worker_name=s.VALIDATION_NAME,instance_id='987',summary_path=report_path,summary_sha256=s.sha(tmp_path/report_path),
            done_path=done_path,done_sha256=s.sha(tmp_path/done_path)),
        models=models,other_reservations=[dict(r,purpose='analysis',reserved_usd=2.8125) for r in jobs]+
            [dict(name=s.VALIDATION_NAME,purpose='validation',reserved_usd=.9375,expected_worker=expected,instance_id='987')])
    path=tmp_path/'docs/research/simple_memory_recovery_plan.json';dump(path,plan)
    def cloud_plan(args):
        return dict(package_manifest=[dict(path=f,sha256=s.sha(tmp_path/f)) for f in sorted(args.include)], reserved_usd=round((args.hours+.25)*1.25,4))
    monkeypatch.setattr(s.cloud,'plan',cloud_plan)
    dump(s.cloud.LEDGER, dict(ancillary_reserve_usd=25,runs=[dict(name='old',reserved_usd=239.127,actual_upper_bound_usd=239.127)]))
    return plan,path


def record(row):
    return dict(name=row['name'],machine='n1-standard-8',zone='us-central1-a',instance_id='123',
        command=row['train_argv'],output_dir=row['worker_output_dir'],duration_seconds=row['provider_hours']*3600,
        reserved_usd=row['reserved_usd'],gcs_prefix=f'gs://{s.cloud.BUCKET}/{s.cloud.PREFIX}/{row["name"]}',
        package_manifest=[dict(path=k,sha256=v) for k,v in row['package_manifest'].items()],package_manifest_sha256='package',
        provider_deadline_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=row['provider_hours'])).isoformat())


def test_complete_registry_and_full_reservations(fixture):
    plan,path=fixture
    assert s.validate(path)['models']==plan['models']
    bound=s.ensure_budget(plan,json.loads(s.cloud.LEDGER.read_text()))
    assert bound['not_yet_launched_reserved_usd']==328.4375
    assert bound['full_scope_upper_usd']==pytest.approx(592.5645)


def test_old_one_hour_reservation_cannot_fund_two_hour_analysis(fixture):
    plan,path=fixture
    plan['other_reservations'][0]['reserved_usd']=1.5625
    dump(path,plan)
    with pytest.raises(AssertionError):s.validate(path)


@pytest.mark.parametrize('fault',['partial','sources','inputs','three_cases','wrong_count','missing_fault',
    'different_gradient','fallback_false','missing_boundary','corruption_accepted','wrong_manifest'])
def test_rehashed_failed_gpu_report_cannot_release_cohort(fixture,fault):
    plan,path=fixture;p=s.ROOT/plan['gpu_release']['summary_path'];report=json.loads(p.read_text())
    known=next(r for r in report['cases'] if r['logical_run_id']=='pgc33/blackout/seed1')
    if fault=='partial':report['complete']=False
    elif fault=='sources':report['sources_unchanged']=False
    elif fault=='inputs':report['all_input_bytes_preserved']=False
    elif fault=='three_cases':report['cases'].pop()
    elif fault=='wrong_count':report['cases'][0]['accepted_reconstructed']-=1
    elif fault=='missing_fault':report['cases'][0]['original_assertion_failure']=''
    elif fault=='different_gradient':report['cases'][0]['guarded_gradients_sha256']='different'
    elif fault=='fallback_false':report['cases'][0]['fallback']['fallback']=False
    elif fault=='missing_boundary':known['all_188_updates_full_boundary_exact']=False
    elif fault=='corruption_accepted':known['corruption_controls']['reset']['rejected']=False
    elif fault=='wrong_manifest':report['manifest_sha256']='unrelated'
    dump(p,report);plan['gpu_release']['summary_sha256']=s.sha(p);dump(path,plan)
    assert all(plan['readiness'].values())
    with pytest.raises(AssertionError):s.validate(path)


def test_auxiliary_names_cannot_be_replaced_by_completed_unrelated_jobs(fixture):
    plan,path=fixture
    plan['other_reservations'][0]['name']='unrelated-already-complete'
    dump(path,plan)
    with pytest.raises(AssertionError):s.validate(path)


def test_validation_same_name_wrong_physical_worker_cannot_release_reservation(fixture):
    plan,_=fixture;ledger=json.loads(s.cloud.LEDGER.read_text());r=plan['other_reservations'][-1]
    ledger['runs'].append(dict(r['expected_worker'],instance_id='999',reserved_usd=.9375,
        actual_upper_bound_usd=.1,worker_result={'exit_code':0}))
    with pytest.raises(AssertionError,match='ID changed'):s.ensure_budget(plan,ledger)


@pytest.mark.parametrize('change',['seventeenth','role','source','parent','scientific_flag','amendment','validation'])
def test_frozen_registry_and_provenance_tampering_rejected(fixture,change):
    plan,path=fixture; row=next(r for r in plan['models'] if r['mode']=='resume_v2')
    if change=='seventeenth':plan['models'].append(copy.deepcopy(row))
    elif change=='role':row['mode']='fresh_v2'
    elif change=='source':(s.ROOT/'code/train_memory_v2.py').write_text('changed')
    elif change=='parent':(s.ROOT/row['parent_run']/'resume.pt').write_bytes(b'changed')
    elif change=='validation':(s.ROOT/'results/private/validation.json').write_text('{}')
    else:
        flag='--tf32' if change=='scientific_flag' else '--amendment'
        row['train_argv'][row['train_argv'].index(flag)+1]='1' if change=='scientific_flag' else 'docs/wrong.json'
        row['plan_argv'][row['plan_argv'].index('--command')+1]=shlex.join(row['train_argv'])
    dump(path,plan)
    with pytest.raises(AssertionError):s.validate(path)


def test_full_budget_not_one_worker_and_no_double_count(fixture):
    plan,_=fixture; ledger=json.loads(s.cloud.LEDGER.read_text())
    first=next(r for r in plan['models'] if r['mode']=='resume_v2')
    ledger['runs'][0]['actual_upper_bound_usd']=270
    assert s.cloud.committed(ledger)+first['reserved_usd']<600
    with pytest.raises(AssertionError,match='All remaining'):s.ensure_budget(plan,ledger)
    ledger=json.loads(s.cloud.LEDGER.read_text());before=s.ensure_budget(plan,ledger)['full_scope_upper_usd']
    validation=plan['other_reservations'][-1]
    ledger['runs'].append(record(first));ledger['runs'].append(dict(validation['expected_worker'],instance_id=validation['instance_id'],reserved_usd=.9375))
    assert s.ensure_budget(plan,ledger)['full_scope_upper_usd']==pytest.approx(before)


def test_failed_auxiliary_reservation_does_not_disappear(fixture):
    plan,_=fixture;ledger=json.loads(s.cloud.LEDGER.read_text())
    validation=plan['other_reservations'][-1]
    ledger['runs'].append(dict(validation['expected_worker'],instance_id=validation['instance_id'],reserved_usd=.9375,
        actual_upper_bound_usd=.1,worker_result={'exit_code':1}))
    with pytest.raises(AssertionError,match='auxiliary'):s.ensure_budget(plan,ledger)


def test_completed_attempt_missing_from_ledger_never_relaunches(fixture,monkeypatch):
    plan,_=fixture;state={'models':{r['logical_name']:{'complete':True} for r in plan['models']}}
    with mock.patch.object(s.cloud,'launch') as launch:
        with pytest.raises(AssertionError,match='missing from ledger'):s.cycle(plan,state)
        launch.assert_not_called()


def metric(row,delta=2_009_600,fps=11200):
    frames=row['start_frames']+delta
    return dict(frames=frames,updates=frames//12800,fps=fps,seconds=delta/fps,attempt=0,loss=1.,grad_norm=2.,
        replay_logit_max_error=.00004,replay_value_max_error=.00003,avg_return_100='must never be read')


def test_resume_warmup_and_attempt_local_fps_deadline(fixture):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2');rec=record(row)
    near_end=dt.datetime.fromisoformat(rec['provider_deadline_utc']).timestamp()-1000
    early=metric(row,delta=12800)
    got=s.technical(json.dumps(early)+'\n',row,rec,now=near_end)
    assert not got['projection_enforced'] and 'avg_return_100' not in got
    with pytest.raises(AssertionError,match='600s'):s.technical(json.dumps(metric(row))+'\n',row,rec,now=near_end)
    wrong=metric(row);wrong['fps']*=2
    with pytest.raises(AssertionError,match='physical attempt'):s.technical(json.dumps(wrong)+'\n',row,rec)
    wrong=metric(row);wrong['loss']=float('nan')
    with pytest.raises(AssertionError,match='Nonfinite'):s.technical(json.dumps(wrong)+'\n',row,rec)


def test_identity_and_gcs_are_bound(fixture):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2')
    for field,value in [('instance_id',None),('gcs_prefix','gs://other/old'),('duration_seconds',1)]:
        rec=record(row);rec[field]=value
        with pytest.raises(AssertionError):s.matching_record(row,rec,{},plan)
    with pytest.raises(AssertionError,match='ID changed'):s.matching_record(row,record(row),{'instance_id':'456'},plan)


def test_no_duplicate_launch_and_eight_active_cap(fixture,monkeypatch):
    plan,_=fixture;state={'models':{}};launched=[]
    monkeypatch.setattr(s,'authenticate',lambda *a,**k:{'fixture':True})
    monkeypatch.setattr(s,'cat',lambda *a,**k:None)
    def launch(args):
        row=next(r for r in plan['models'] if r.get('name')==args.name)
        ledger=json.loads(s.cloud.LEDGER.read_text());ledger['runs'].append(record(row));dump(s.cloud.LEDGER,ledger)
        launched.append(args.name)
    monkeypatch.setattr(s.cloud,'launch',launch)
    s.cycle(plan,state);assert len(launched)==8
    s.cycle(plan,state);assert len(launched)==8
    assert sum(x.get('complete',False) for x in state['models'].values())==4
    assert json.loads(s.cloud.LEDGER.read_text())['runs'][0]['name']=='old'


def test_done_after_provider_delete_precedes_live_or_metrics(fixture,monkeypatch):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2')
    dump(s.cloud.LEDGER,dict(ancillary_reserve_usd=25,runs=[record(row)]))
    state={'models':{r['logical_name']:dict(complete=True) for r in plan['models'] if r is not row}}
    def cat(uri,tail=False):
        assert not tail,'Do not read metrics for a DONE worker'
        return '{"exit_code":0}'
    monkeypatch.setattr(s,'cat',cat)
    monkeypatch.setattr(s,'finish',lambda *a:dict(complete=True,endpoint={'fixture':True}))
    with mock.patch.object(s.cloud,'describe') as describe,mock.patch.object(s.cloud,'launch') as launch:
        # Other completed rows normally have ledger entries; retain placeholders
        # so this isolated terminal test cannot launch them.
        d=json.loads(s.cloud.LEDGER.read_text())
        d['runs'] += [dict(record(r),actual_upper_bound_usd=1) for r in plan['models'] if r['mode']!='retained_v1' and r is not row]
        dump(s.cloud.LEDGER,d);s.cycle(plan,state)
        describe.assert_not_called();launch.assert_not_called()
    assert state['models'][row['logical_name']]['complete']


def test_failed_attempt_does_not_trigger_new_launch(fixture,monkeypatch):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2')
    dump(s.cloud.LEDGER,dict(ancillary_reserve_usd=25,runs=[record(row)]))
    monkeypatch.setattr(s,'authenticate',lambda *a,**k:{})
    monkeypatch.setattr(s,'cat',lambda *a,**k:'{"exit_code":1}')
    def failure(*a):raise RuntimeError('Worker failed; no retry')
    monkeypatch.setattr(s,'finish',failure)
    with mock.patch.object(s.cloud,'launch') as launch:
        with pytest.raises(RuntimeError,match='no retry'):s.cycle(plan,{'models':{}},)
        launch.assert_not_called()


@pytest.mark.parametrize('exit_code,auth_fails',[(0,False),(1,False),(0,True)])
def test_terminal_cleanup_preserves_attempt_even_on_invalid_endpoint(fixture,monkeypatch,exit_code,auth_fails):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2');rec=record(row)
    ledger={'runs':[rec]};calls=[]
    @contextlib.contextmanager
    def lock():yield ledger
    monkeypatch.setattr(s.cloud,'locked_ledger',lock)
    monkeypatch.setattr(s.cloud,'describe',lambda *a:dict(id='123',status='RUNNING',labels={'study':s.cloud.PREFIX}))
    def collect(r):r['local_artifacts']='results/private_collected';calls.append('collect')
    monkeypatch.setattr(s.cloud,'collect',collect)
    def authenticate(*a):
        calls.append('authenticate')
        if auth_fails:raise AssertionError('Genuine endpoint proof mismatch')
        return {'producer_version':2,'frames':s.TARGET}
    monkeypatch.setattr(s,'authenticate',authenticate)
    def run(cmd,**kwargs):
        assert cmd[:4]==['gcloud','compute','instances','delete'] and cmd[4]==row['name']
        calls.append('delete');return argparse.Namespace(returncode=0)
    monkeypatch.setattr(s.cloud,'run',run)
    def refresh(r):calls.append('refresh');r['actual_upper_bound_usd']=1
    monkeypatch.setattr(s.cloud,'refresh',refresh)
    done={'exit_code':exit_code,'package_sha256':'package'}
    if exit_code or auth_fails:
        with pytest.raises(RuntimeError):s.finish(row,done,plan)
    else:assert s.finish(row,done,plan)['complete']
    assert calls[0]=='collect' and calls[-2:]==['delete','refresh']
    assert ('authenticate' in calls)==(exit_code==0)
    assert rec['local_artifacts']=='results/private_collected' and rec['actual_upper_bound_usd']==1


def test_final_must_be_genuine_v2_at_original_cutoff(fixture,monkeypatch):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2')
    endpoint=dict(condition=row['condition'],training_seed=row['seed'],training_task_max_gap=row['task_max_gap'],
        frames=s.TARGET,updates=78125,decisions=250000000,producer_version=1)
    module=types.ModuleType('scripts.simple_memory_endpoint')
    module.authenticate_run=lambda *a,**k:dict(endpoint)
    monkeypatch.setitem(sys.modules,'scripts.simple_memory_endpoint',module)
    with pytest.raises(AssertionError,match='producer'):s.authenticate(row,s.ROOT/'results/private',plan)
    endpoint['producer_version']=2
    assert s.authenticate(row,s.ROOT/'results/private',plan)==endpoint
    endpoint['frames']+=12800
    with pytest.raises(AssertionError):s.authenticate(row,s.ROOT/'results/private',plan)


@pytest.mark.parametrize('vm',[None,{'id':'456','status':'RUNNING','labels':{'study':s.cloud.PREFIX}},
    {'id':'123','status':'TERMINATED','labels':{'study':s.cloud.PREFIX}}])
def test_metric_only_promotion_requires_live_exact_vm(fixture,monkeypatch,vm):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='resume_v2')
    dump(s.cloud.LEDGER,dict(ancillary_reserve_usd=25,runs=[record(row)]))
    monkeypatch.setattr(s,'authenticate',lambda *a,**k:{})
    monkeypatch.setattr(s,'cat',lambda uri,tail=False:json.dumps(metric(row))+'\n' if tail else None)
    monkeypatch.setattr(s.cloud,'describe',lambda *a:vm)
    with mock.patch.object(s.cloud,'launch') as launch:
        with pytest.raises((AssertionError,RuntimeError)):s.cycle(plan,{'models':{}})
        launch.assert_not_called()


def host_fixture(plan):
    """Tiny saved tensors only; no model, optimizer step or environment."""
    import torch
    from code import train_memory_v2 as producer
    row=next(r for r in plan['models'] if r['logical_name']==s.HOST_LOGICAL)
    row['name']=row['logical_name']+'-r1'
    row['worker_output_dir']='results/ring_simple_memory_v2/gc100/native'
    row['trainer_output_dir']=row['worker_output_dir']+'/seed4'
    row['start_frames']=25600
    source=s.ROOT/f'results/ring_cloud_artifacts/{s.HOST_LOGICAL}/seed4';source.mkdir(parents=True)
    cfg=dict(cond='gc100',seed=4,code_seed=4,task_max_gap=0)
    amendment_path=s.ROOT/plan['bindings']['amendment']['path'];amendment=json.loads(amendment_path.read_text())
    amendment['numerical_runtime']={'torch_version':'private-fixture'};dump(amendment_path,amendment)
    plan['bindings']['amendment']['sha256']=s.sha(amendment_path)
    for model in plan['models']:
        if model['mode']!='retained_v1': model['package_manifest'][plan['bindings']['amendment']['path']]=s.sha(amendment_path)
    proof=dict(provenance_version=2,producer_module='code.train_memory_v2',source_sha256=amendment['source_sha256'],
        amendment_sha256=s.sha(amendment_path),protocol_sha256=plan['bindings']['protocol']['sha256'],
        logical_run_id='gc100/native/seed4',torch_version='private-fixture',
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
    initial=dict(schema=1,policy={'weight':torch.tensor([.2,.4])},optimizer={'state':{}},
        runtime={'counters':dict(frames=0,updates=0,decisions=0),'rng':torch.tensor([1,2,3],dtype=torch.uint8)},
        config=cfg,frames=0,updates=0,policy_training=True,policy_sha256='private-policy',provenance=proof)
    lineage=dict(kind='v2_fresh',logical_run_id='gc100/native/seed4',restored_frames=0,restored_updates=0,
        initial_payload_sha256=producer.payload_digest(initial))
    dump(source/'lineage.json',lineage);proof['lineage_sha256']=s.sha(source/'lineage.json')
    dump(source/'config.json',cfg);dump(source/'provenance.json',proof)
    torch.save(initial,source/'initial.pt');shutil.copyfile(source/'initial.pt',source/'resume_input_000.pt')
    saved=copy.deepcopy(initial);saved.update(frames=25600,updates=2)
    saved['runtime']['counters']=dict(frames=25600,updates=2,decisions=6400)
    saved['optimizer']['state']={0:{'step':torch.tensor(2.),'square_avg':torch.tensor([.1,.2])}}
    torch.save(saved,source/'resume.pt')
    attempt=dict(attempt=0,complete=False,exit_code=143,restored_frames=0,restored_updates=0,
        final_frames=25600,final_updates=2,resume_sha256_after=s.sha(source/'resume.pt'),
        resume_sha256=s.sha(source/'resume_input_000.pt'),lineage_sha256=proof['lineage_sha256'])
    dump(source/'attempt_000.json',attempt);dump(source/'interrupted_000.json',attempt)
    (source/'metrics_000.jsonl').write_text(''.join(json.dumps(dict(attempt=0,updates=i,frames=i*12800,decisions=i*3200,replay_guard={'accepted':True}))+'\n' for i in (1,2)))
    (source/'episodes_000.jsonl').write_text('')
    result=source.parent/'WORKER_RESULT.json';dump(result,dict(exit_code=143,package_sha256='old-package'))
    files={p.name:s.sha(p) for p in source.iterdir()}
    row['continuation']=dict(kind='explicit_v2_host_recovery',source_worker_name=s.HOST_LOGICAL,
        source_instance_id=s.HOST_INSTANCE_ID,source_run=str(source.relative_to(s.ROOT)),source_files_sha256=files,
        source_worker_result=dict(path=str(result.relative_to(s.ROOT)),sha256=s.sha(result)),
        resume_sha256=files['resume.pt'],resume_payload_sha256=producer.payload_digest(saved),
        source_attempt=0,attempt=1,start_frames=25600,start_updates=2)
    target=s.ROOT/row['trainer_output_dir'];shutil.copytree(source,target)
    row['package_manifest'].update({row['trainer_output_dir']+'/'+n:h for n,h in files.items()})
    argv=row['train_argv'];argv[argv.index('--out')+1]=row['trainer_output_dir'];argv.append('--resume')
    row['plan_argv']=['python','scripts/ring_cloud.py','plan','--name',row['name'],'--machine','n1-standard-8',
        '--zone','us-central1-a','--hours','25.5','--output-dir',row['worker_output_dir'],
        '--command',shlex.join(argv),'--include-only']
    for name in row['package_manifest']:row['plan_argv']+=['--include',name]
    return row,source,target


def test_registered_host_resume_preserves_inputs_and_scientific_argv(fixture):
    plan,_=fixture;row,source,target=host_fixture(plan)
    s.validate_continuation(row,plan)
    args=s.launch_args(row,plan)
    assert args.name==s.HOST_LOGICAL+'-r1' and args.hours==25.5
    assert row['mode']=='fresh_v2' and row['train_argv'][-1]=='--resume'
    assert row['start_frames']==25600 and s.expected_attempt(row)==1
    assert s.sha(source/'resume.pt')==s.sha(target/'resume.pt')
    (target/'resume.pt').write_bytes(b'changed copied checkpoint')
    with pytest.raises(AssertionError):s.validate_continuation(row,plan)


def test_complete_host_plan_keeps_fifteen_other_models_and_roles(fixture):
    plan,path=fixture
    original={r['logical_name']:copy.deepcopy(r) for r in plan['models'] if r['logical_name']!=s.HOST_LOGICAL}
    row,_,_=host_fixture(plan);dump(path,plan)
    assert s.validate(path)['models']==plan['models']
    assert {r['logical_name']:r for r in plan['models'] if r['logical_name']!=s.HOST_LOGICAL}==original
    assert {m:sum(r['mode']==m for r in plan['models']) for m in s.MODES}==s.MODES


@pytest.mark.parametrize('fault',['identity','worker_id','attempt','cutoff','payload','source_checkpoint','log_cutoff','rejected'])
def test_host_resume_requires_exact_bound_single_attempt(fixture,fault):
    plan,_=fixture;row,source,target=host_fixture(plan);c=row['continuation']
    if fault=='identity':row['logical_name']='another-model'
    elif fault=='worker_id':c['source_instance_id']='999'
    elif fault=='attempt':c['attempt']=2
    elif fault=='cutoff':c['start_updates']=3
    elif fault=='payload':c['resume_payload_sha256']='0'*64
    elif fault=='source_checkpoint':(source/'resume.pt').write_bytes(b'changed source checkpoint')
    elif fault=='log_cutoff':
        (source/'metrics_000.jsonl').write_text('{}\n');shutil.copyfile(source/'metrics_000.jsonl',target/'metrics_000.jsonl')
    elif fault=='rejected':dump(source/'rejected_update_000.json',{'accepted':False})
    with pytest.raises(AssertionError):s.validate_continuation(row,plan)


def test_resume_flag_without_registered_host_metadata_is_rejected(fixture):
    plan,_=fixture;row=next(r for r in plan['models'] if r['mode']=='fresh_v2')
    row['train_argv'].append('--resume');row['plan_argv'][row['plan_argv'].index('--command')+1]=shlex.join(row['train_argv'])
    with pytest.raises(AssertionError,match='explicit host'):s.launch_args(row,plan)


def test_host_budget_requires_closed_original_identity_and_retains_cost(fixture):
    plan,_=fixture;row,source,_=host_fixture(plan);ledger=json.loads(s.cloud.LEDGER.read_text())
    old=dict(name=s.HOST_LOGICAL,instance_id=s.HOST_INSTANCE_ID,status='DELETED',collected_at='private',
        actual_upper_bound_usd=.5,reserved_usd=32.1875,local_artifacts=str(source.parent.relative_to(s.ROOT)),
        worker_result=json.loads((source.parent/'WORKER_RESULT.json').read_text()))
    with pytest.raises(AssertionError,match='missing'):s.ensure_budget(plan,ledger)
    ledger['runs'].append(old);assert s.ensure_budget(plan,ledger)['full_scope_upper_usd']==pytest.approx(593.0645)
    for key,value in [('status','RUNNING'),('instance_id','999'),('actual_upper_bound_usd',1.01),('collected_at',None)]:
        changed=copy.deepcopy(ledger);changed['runs'][-1][key]=value
        with pytest.raises(AssertionError):s.ensure_budget(plan,changed)


def final_host_fixture(row,target):
    c=row['continuation'];shutil.copyfile(target/'resume.pt',target/'resume_input_001.pt')
    # The canonical final resume role may change; the input checkpoint may not.
    (target/'resume.pt').write_bytes(b'legitimate final mutable checkpoint')
    proof=json.loads((target/'provenance.json').read_text())
    dump(target/'attempt_001.json',dict(attempt=1,producer_version=2,logical_run_id='gc100/native/seed4',
        complete=True,exit_code=0,lineage_sha256=proof['lineage_sha256'],restored_frames=c['start_frames'],
        restored_updates=c['start_updates'],final_frames=s.TARGET,final_updates=78125,
        resume_input='resume_input_001.pt',resume_sha256=c['resume_sha256']))
    dump(target/'summary.json',dict(attempt_records=['attempt_000.json','attempt_001.json']))


def test_final_continuation_manifest_is_portable_and_pins_supplied_checkpoint(fixture,tmp_path):
    plan,_=fixture;row,source,target=host_fixture(plan);final_host_fixture(row,target)
    manifest=s.continuation_inputs(row,target)
    assert 'resume.pt' not in manifest and {'resume_input_001.pt','attempt_001.json','interrupted_000.json','metrics_000.jsonl'}<=set(manifest)
    relocated=tmp_path/'relocated';relocated.mkdir()
    for name in manifest:shutil.copyfile(target/name,relocated/name)
    shutil.rmtree(source)
    assert s.continuation_inputs(row,relocated)==manifest
    (relocated/'resume_input_001.pt').write_bytes(b'changed input')
    with pytest.raises(AssertionError):s.continuation_inputs(row,relocated)


def test_host_metric_clock_requires_attempt1_and_uses_only_new_frames(fixture):
    plan,_=fixture;row,_,_=host_fixture(plan);rec=record(row)
    m=metric(row);m['attempt']=1
    got=s.technical(json.dumps(m)+'\n',row,rec)
    assert got['new_frames']==2_009_600 and got['attempt']==1
    m['attempt']=0
    with pytest.raises(AssertionError):s.technical(json.dumps(m)+'\n',row,rec)


def test_host_cycle_reads_attempt1_log_without_retry(fixture,monkeypatch):
    plan,_=fixture;row,_,_=host_fixture(plan)
    dump(s.cloud.LEDGER,dict(ancillary_reserve_usd=25,runs=[record(row)]))
    monkeypatch.setattr(s,'authenticate',lambda *a,**k:{})
    seen=[]
    def cat(uri,tail=False):
        seen.append(uri)
        if tail:
            assert uri.endswith('/seed4/metrics_001.jsonl')
            m=metric(row);m['attempt']=1;return json.dumps(m)+'\n'
    monkeypatch.setattr(s,'cat',cat)
    monkeypatch.setattr(s.cloud,'describe',lambda *a:None)
    with mock.patch.object(s.cloud,'launch') as launch:
        with pytest.raises(AssertionError,match='Absent VM'):s.cycle(plan,{'models':{}})
        launch.assert_not_called()
    assert any(n.endswith('metrics_001.jsonl') for n in seen)
