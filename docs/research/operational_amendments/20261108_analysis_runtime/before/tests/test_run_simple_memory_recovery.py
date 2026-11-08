"""Private operational fixtures only: no scientific checkpoints or cloud requests."""
import argparse
import contextlib
import copy
import datetime as dt
import json
from pathlib import Path
import shlex
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
    amended.update(parent_protocol_sha256=s.sha(tmp_path/protocol),guard={'id':'private_fixture_guard','status':'frozen'},run_bindings={})
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
        models=models,other_reservations=[dict(r,purpose='analysis',reserved_usd=1.5625) for r in jobs]+
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
    assert bound['not_yet_launched_reserved_usd']==318.4375
    assert bound['full_scope_upper_usd']==pytest.approx(582.5645)


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
