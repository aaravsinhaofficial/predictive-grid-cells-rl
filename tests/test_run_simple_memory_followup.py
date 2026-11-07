"""Private operational guards; never creates or observes real cloud resources."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import run_simple_memory_followup as study


def plan():
    jobs=[]
    for seed in (1,2,3,4):
        for task in ('native','blackout'):
            jobs.append(dict(name=f'ring-20261105-simplemem-analysis-{task}-s{seed}',seed=seed,
                task=task,task_max_gap=60 if task=='blackout' else 0,machine='n1-standard-8',zone='us-central1-a',
                output_dir=f'results/ring_simple_memory_analysis/seed{seed}_{task}',
                training_jobs={c:f'ring-20261105-simplemem-{c}-{task}-s{seed}' for c in ('gc100','pgc33')}))
    return dict(jobs=jobs,per_worker_hours=1,runner_max_seconds=2900,max_active_workers=8,
                training_plan_sha256='private-plan',training_protocol_sha256='private-protocol',training_supervisor_sha256='private-supervisor',
                worker_sources=['scripts/run_simple_memory_analysis.py'])


def status(p,complete=False):
    return dict(complete=complete,updated_at='private-fixture',binding=dict(plan_sha256=p['training_plan_sha256'],
        protocol_sha256=p['training_protocol_sha256'],supervisor_sha256=p['training_supervisor_sha256']),
        jobs={name:{} for j in p['jobs'] for name in j['training_jobs'].values()})


def test_waiting_does_not_launch_or_read_nonfinal_models(tmp_path,monkeypatch):
    p=plan();state=dict(jobs={});path=tmp_path/'training.json'
    path.write_text(json.dumps(status(p)))
    monkeypatch.setattr(study,'TRAINING_STATUS',path)
    monkeypatch.setattr(study,'live_training_supervisor',lambda:{'alive':True,'pid':123})
    monkeypatch.setattr(study.cloud,'launch',lambda _:pytest.fail('No cloud creation before all16 finals'))
    study.cycle(p,state)
    assert state['phase']=='waiting_for_all16_training_finals'
    assert state['training_complete_count']==0
    assert state['training_supervisor']['alive']


def test_alive_requires_actual_command_not_a_status_file(tmp_path,monkeypatch):
    path=tmp_path/'process.json';path.write_text(json.dumps({'pid':123}))
    monkeypatch.setattr(study.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='unrelated process'))
    assert not study.live_training_supervisor(path)['alive']
    monkeypatch.setattr(study.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    assert not study.live_training_supervisor(path)['alive']
    monkeypatch.setattr(study.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='python scripts/run_simple_memory_cohort.py'))
    assert study.live_training_supervisor(path)['alive']


def test_entire_future_analysis_budget_is_reserved_conceptually():
    p=plan()
    ledger=dict(ancillary_reserve_usd=25,runs=[dict(name='previous',actual_upper_bound_usd=561.699,reserved_usd=561.699)])
    assert study.whole_analysis_budget(p,ledger)==pytest.approx(599.199)
    ledger['runs'][0]['actual_upper_bound_usd']+=.802
    with pytest.raises(RuntimeError,match='600'):study.whole_analysis_budget(p,ledger)
    ledger=dict(ancillary_reserve_usd=25,runs=[dict(name=j['name'],reserved_usd=1.5625) for j in p['jobs'][:3]])
    assert study.whole_analysis_budget(p,ledger)==37.5  # Do not reserve existing workers twice.


def test_changed_or_partial_training_cohort_cannot_be_promoted():
    p=plan();s=status(p)
    assert study.training_endpoints(p,s) is None
    s['jobs'].pop(next(iter(s['jobs'])))
    with pytest.raises(AssertionError,match='cohort'):study.training_endpoints(p,s)
    s=status(p,True)
    with pytest.raises(KeyError):study.training_endpoints(p,s)
    s['halted']=True
    with pytest.raises(AssertionError,match='halted'):study.training_endpoints(p,s)


def test_finished_training_requires_actual_endpoint_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    p=plan();s=status(p,True)
    for name,row in s['jobs'].items():
        folder=tmp_path/'results/ring_cloud_artifacts'/name/'seed1';folder.mkdir(parents=True)
        (folder/'summary.json').write_text('{}');(folder/'final.pt').write_bytes(b'private hash fixture')
        row.update(complete=True,frames=1_000_000_000,summary_path=str(folder/'summary.json'),
            summary_sha256=study.sha(folder/'summary.json'),final_sha256=study.sha(folder/'final.pt'))
    result=study.training_endpoints(p,s);assert len(result)==16
    first=next(iter(result.values()));(first/'final.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError):study.training_endpoints(p,s)


def test_worker_command_is_fixed_and_portable(monkeypatch):
    p=plan();job=p['jobs'][0]
    monkeypatch.setattr(study,'static_files',lambda _:['code/nets.py','scripts/run_simple_memory_analysis.py'])
    args=study.launch_args(p,job)
    assert args.hours==1 and args.include_only and args.machine=='n1-standard-8'
    argv=study.shlex.split(args.command)
    assert argv[:3]==['python','scripts/run_simple_memory_analysis.py','--runs']
    assert argv[-2:]==['--max-seconds','2900']
    assert len(args.include)==10
    assert not any(Path(x).is_absolute() for x in args.include)
    assert set(Path(x).name for x in args.include[2:])==set(study.INPUTS)


def test_record_matching_rejects_changed_identity_and_package(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    file=tmp_path/'fixture.py';file.write_text('pass\n')
    args=argparse.Namespace(name='private',machine='n1-standard-8',zone='us-central1-a',
        output_dir='results/ring_private',command='python fixture.py',include=['fixture.py'])
    monkeypatch.setattr(study,'launch_args',lambda *a:args)
    record=dict(instance_id='123',name=args.name,machine=args.machine,zone=args.zone,output_dir=args.output_dir,
        gcs_prefix=f'gs://{study.cloud.BUCKET}/{study.cloud.PREFIX}/{args.name}',duration_seconds=3600,reserved_usd=1.5625,
        command=['python','fixture.py'],package_manifest=[dict(path='fixture.py',sha256=study.sha(file))])
    study.match_record({}, {},record,'123')
    with pytest.raises(AssertionError):study.match_record({}, {},record,'124')
    broken={**record,'instance_id':'not numeric'}
    with pytest.raises(AssertionError):study.match_record({}, {},broken)
    broken={**record,'gcs_prefix':'gs://other/outputs'}
    with pytest.raises(AssertionError):study.match_record({}, {},broken)
    file.write_text('changed\n')
    with pytest.raises(AssertionError):study.match_record({}, {},record,'123')


def test_status_binding_is_checked_even_while_waiting():
    p=plan();s=status(p);s['binding']['protocol_sha256']='changed'
    with pytest.raises(AssertionError,match='binding'):study.training_endpoints(p,s)


def test_full_endpoint_authentication_checks_all16_identities(tmp_path,monkeypatch):
    from scripts import run_simple_memory_analysis as runner
    p=plan();endpoints={};expected={};calls=[]
    for job in p['jobs']:
        for c,name in job['training_jobs'].items():
            path=tmp_path/name;endpoints[name]=path
            expected[str(path)]=dict(condition=c,training_seed=job['seed'],training_task_max_gap=job['task_max_gap'])
    def auth(path):calls.append(str(path));return expected[str(path)]
    monkeypatch.setattr(runner,'authenticate_run',auth)
    assert len(study.authenticate_endpoints(p,endpoints))==len(calls)==16
    expected[str(next(iter(endpoints.values())))]['training_seed']=999
    with pytest.raises(AssertionError,match='identity'):study.authenticate_endpoints(p,endpoints)


def test_pair_completion_rejects_partial_or_different_original_inputs(tmp_path,monkeypatch):
    from scripts import run_simple_memory_analysis as runner
    job=plan()['jobs'][0]
    expected={c:dict(condition=c,training_seed=1,training_task_max_gap=0,checkpoint_sha256=c) for c in ('gc100','pgc33')}
    data=dict(runs=[dict(endpoint=v) for v in expected.values()]);path=tmp_path/'pipeline.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner,'verify_pipeline',lambda _:dict(models=2,stages=10))
    assert study.verify_pair(tmp_path,job,expected)['models']==2
    data['runs'][1]['endpoint']['checkpoint_sha256']='different'
    path.write_text(json.dumps(data))
    original={c:dict(condition=c,training_seed=1,training_task_max_gap=0,checkpoint_sha256=c) for c in ('gc100','pgc33')}
    with pytest.raises(AssertionError,match='original'):study.verify_pair(tmp_path,job,original)
    monkeypatch.setattr(runner,'verify_pipeline',lambda _:dict(models=1,stages=5))
    with pytest.raises(AssertionError,match='incomplete'):study.verify_pair(tmp_path,job,original)


def test_complete_cache_requires_collected_bytes_on_restart(tmp_path,monkeypatch):
    p=plan();job=p['jobs'][0]
    monkeypatch.setattr(study,'ROOT',tmp_path)
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'runs':[{'name':job['name']}]}))
    monkeypatch.setattr(study.cloud,'LEDGER',ledger)
    monkeypatch.setattr(study,'match_record',lambda *a:None)
    folder=tmp_path/'results/ring_cloud_artifacts'/job['name'];folder.mkdir(parents=True)
    file=folder/'pipeline.json';file.write_text('{}')
    monkeypatch.setattr(study,'verify_pair',lambda *a:dict(models=2))
    state=dict(jobs={job['name']:dict(complete=True,instance_id='123',local_artifacts=str(folder.relative_to(tmp_path)),
        pipeline_sha256=study.sha(file),inputs={},verification=dict(models=2))})
    study.verify_completed_cache(p,state)
    file.unlink()
    with pytest.raises(FileNotFoundError):study.verify_completed_cache(p,state)


def frozen_plan_fixture(tmp_path,monkeypatch):
    """Independent temporary manifest closure; no repository/cloud state reads."""
    monkeypatch.setattr(study,'ROOT',tmp_path)
    operations=('scripts/run_simple_memory_followup.py','scripts/run_simple_memory_analysis.py',
                'scripts/ring_cloud.py','scripts/run_simple_memory_cohort.py')
    scientific='code/private_scientific.py'
    validation='results/private_validation.json'
    for name in (*operations,scientific,validation):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# private bytes '+name)
    training_protocol=tmp_path/'docs/research/simple_memory_training_protocol.json'
    training_protocol.parent.mkdir(parents=True,exist_ok=True)
    training_protocol.write_text(json.dumps({'status':'frozen'}))
    training_plan=tmp_path/'docs/research/simple_memory_cloud_plan.json'
    training_plan.write_text(json.dumps({'status':'frozen','private_fixture':True}))
    analysis_protocol=tmp_path/'docs/research/simple_memory_analysis_protocol.json'
    analysis_protocol.write_text(json.dumps(dict(status='frozen',
        training_protocol=training_protocol.relative_to(tmp_path).as_posix(),
        source_sha256={scientific:study.sha(tmp_path/scientific)},
        validation_artifacts_sha256={validation:study.sha(tmp_path/validation)})))
    monkeypatch.setattr(study,'ANALYSIS_PROTOCOL',analysis_protocol)
    monkeypatch.setattr(study,'TRAINING_PLAN',training_plan)
    value=plan()
    value.update(status='frozen',cumulative_ceiling_usd=600,total_analysis_reservation_usd=12.5,
        analysis_protocol_sha256=study.sha(analysis_protocol),training_plan_sha256=study.sha(training_plan),
        training_protocol_sha256=study.sha(training_protocol),
        training_supervisor_sha256=study.sha(tmp_path/'scripts/run_simple_memory_cohort.py'),
        operational_source_sha256={name:study.sha(tmp_path/name) for name in operations})
    path=tmp_path/'followup.json';path.write_text(json.dumps(value))
    assert study.validated_plan(path)==value
    return path,value,scientific,validation


def test_frozen_plan_rejects_conflicting_scientific_operational_overlap(tmp_path,monkeypatch):
    path,value,scientific,_=frozen_plan_fixture(tmp_path,monkeypatch)
    # Even a real current byte hash cannot override the frozen scientific hash.
    (tmp_path/scientific).write_text('changed scientific producer')
    value['operational_source_sha256'][scientific]=study.sha(tmp_path/scientific)
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError,match='Conflicting source manifests'):
        study.validated_plan(path)


def test_frozen_plan_rejects_changed_validation_evidence(tmp_path,monkeypatch):
    path,_,_,validation=frozen_plan_fixture(tmp_path,monkeypatch)
    (tmp_path/validation).write_text('changed private validation evidence')
    with pytest.raises(AssertionError,match='private_validation'):
        study.validated_plan(path)


@pytest.mark.parametrize('missing',[
    'scripts/run_simple_memory_followup.py','scripts/run_simple_memory_analysis.py',
    'scripts/ring_cloud.py','scripts/run_simple_memory_cohort.py'])
def test_frozen_plan_requires_every_operational_producer(tmp_path,monkeypatch,missing):
    path,value,_,_=frozen_plan_fixture(tmp_path,monkeypatch)
    value['operational_source_sha256'].pop(missing)
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):study.validated_plan(path)
