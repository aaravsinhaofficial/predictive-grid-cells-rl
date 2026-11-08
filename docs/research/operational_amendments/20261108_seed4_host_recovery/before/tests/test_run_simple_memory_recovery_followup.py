"""Private operational fixtures: never inspect or mutate real cloud resources."""
import copy
import json
from pathlib import Path

import pytest
from scripts import run_simple_memory_recovery_followup as task


def plan():
    jobs=[]
    for seed in range(1,5):
        for kind in ('native','blackout'):
            jobs.append(dict(name=f'ring-20261105-simplemem-analysis-{kind}-s{seed}',seed=seed,task=kind,
                task_max_gap=60 if kind=='blackout' else 0,machine='n1-standard-8',zone='us-central1-a',
                output_dir=f'results/ring_simple_memory_analysis/seed{seed}_{kind}',
                training_jobs={c:f'ring-20261105-simplemem-{c}-{kind}-s{seed}' for c in ('gc100','pgc33')}))
    return dict(jobs=jobs,recovery_plan_sha256='plan',recovery_supervisor_sha256='supervisor',
        per_worker_hours=2,runner_max_seconds=6600,per_worker_reservation_usd=2.8125,
        max_active_workers=8,total_analysis_reservation_usd=22.5)


def recovery_plan(p):
    models=[]
    retained={'gc100-native-s1','gc100-blackout-s1','gc100-blackout-s2','pgc33-blackout-s2'}
    for job in p['jobs']:
        for condition,name in job['training_jobs'].items():
            short=f'{condition}-{job["task"]}-s{job["seed"]}'
            mode='retained_v1' if short in retained else 'fresh_v2' if job['seed']>=3 else 'resume_v2'
            physical=name+('-r1' if mode=='resume_v2' else '')
            models.append(dict(logical_name=name,name=physical,mode=mode,condition=condition,seed=job['seed'],
                task_max_gap=job['task_max_gap'],run_dir=f'results/ring_cloud_artifacts/{name}/seed{job["seed"]}',
                trainer_output_dir=f'results/new/{physical}/seed{job["seed"]}',reserved_usd=10.))
    return dict(models=models,other_reservations=[dict(name=j['name'],purpose='analysis',reserved_usd=2.8125) for j in p['jobs']]+[
        dict(name='private-validation',purpose='validation',reserved_usd=.9375)])


def status(p,r):
    return dict(binding=dict(plan_sha256=p['recovery_plan_sha256'],supervisor_sha256=p['recovery_supervisor_sha256']),
        complete=False,models={x['logical_name']:{} for x in r['models']})


def completed(tmp_path,p,r):
    state=status(p,r);state['complete']=True
    for model in r['models']:
        relative=(model['run_dir'] if model['mode']=='retained_v1' else
                  f'results/ring_cloud_artifacts/{model["name"]}/seed{model["seed"]}')
        folder=tmp_path/relative;folder.mkdir(parents=True)
        for name in task.INPUTS:(folder/name).write_text('private '+name)
        endpoint=dict(condition=model['condition'],training_seed=model['seed'],training_task_max_gap=model['task_max_gap'],
            frames=1_000_000_000,updates=78125,decisions=250_000_000,
            producer_version=1 if model['mode']=='retained_v1' else 2)
        for name,key in [('final.pt','checkpoint_sha256'),('config.json','config_sha256'),
                         ('summary.json','terminal_summary_sha256'),('provenance.json','training_provenance_sha256')]:
            endpoint[key]=task.sha(folder/name)
        state['models'][model['logical_name']]=dict(complete=True,run_dir=relative,endpoint=endpoint)
    return state


def test_waits_for_all16_without_ledger_auth_or_cloud_calls(tmp_path,monkeypatch):
    p=plan();r=recovery_plan(p);s=status(p,r);f=tmp_path/'status.json';f.write_text(json.dumps(s))
    monkeypatch.setattr(task,'TRAINING_STATUS',f)
    monkeypatch.setattr(task,'authenticate_endpoints',lambda *a:pytest.fail('Partial endpoints must not run'))
    monkeypatch.setattr(task.cloud,'launch',lambda *a:pytest.fail('No launch'))
    state=dict(jobs={});task.cycle(p,r,state)
    assert state['training_complete_count']==0 and state['phase']=='waiting_for_all16_authenticated_finals'
    f.unlink();task.cycle(p,r,state)
    assert state['phase']=='waiting_for_recovery_supervisor'
    s['halted']=True;f.write_text(json.dumps(s))
    with pytest.raises(RuntimeError,match='halted'):task.cycle(p,r,state)


def test_completion_requires_exact16_identity_version_counters_and_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(task,'ROOT',tmp_path)
    p=plan();r=recovery_plan(p);s=completed(tmp_path,p,r)
    assert len(task.training_endpoints(p,r,s))==16
    for change in ('binding','missing','version','counter','path'):
        b=copy.deepcopy(s);first=next(iter(b['models'].values()))
        if change=='binding':b['binding']['plan_sha256']='wrong'
        if change=='missing':b['models'].pop(next(iter(b['models'])))
        if change=='version':first['endpoint']['producer_version']=7
        if change=='counter':first['endpoint']['updates']=78124
        if change=='path':first['run_dir']='results/elsewhere'
        with pytest.raises(AssertionError):task.training_endpoints(p,r,b)
    first=next(iter(s['models'].values()));(tmp_path/first['run_dir']/'final.pt').write_text('changed')
    with pytest.raises(AssertionError):task.training_endpoints(p,r,s)


def input_fixture(folder,version,kind=None):
    folder.mkdir(parents=True)
    for name in task.INPUTS:(folder/name).write_text('private '+name)
    endpoint=dict(producer_version=version)
    if version==2:
        (folder/'initial.pt').write_bytes(b'private original initialization')
        lineage=dict(kind=kind)
        if kind=='v1_resume':
            (folder/'migration_boundary.pt').write_bytes(b'private full payload')
            parent=folder/'parent_evidence';parent.mkdir()
            for name in ('resume.pt','initial.pt','config.json','provenance.json','metrics_000.jsonl','episodes_000.jsonl','attempt_000.json'):
                (parent/name).write_text('private parent '+name)
            lineage['parent_evidence_sha256']={p.name:task.sha(p) for p in parent.iterdir()}
        (folder/'lineage.json').write_text(json.dumps(lineage))
        endpoint.update(lineage_sha256=task.sha(folder/'lineage.json'),producer_assignment=kind)
    return endpoint


def test_portable_allowlist_keeps_every_parent_file_and_excludes_unrelated(tmp_path):
    for version,kind in ((1,None),(2,'v2_fresh'),(2,'v1_resume')):
        folder=tmp_path/(kind or 'old');ep=input_fixture(folder,version,kind)
        (folder/'unrelated.txt').write_text('not a model input')
        got=task.required_inputs(folder,ep)
        assert set(task.INPUTS)<=set(got) and 'unrelated.txt' not in got
        if kind=='v1_resume':
            assert {'initial.pt','lineage.json','migration_boundary.pt','parent_evidence/metrics_000.jsonl',
                    'parent_evidence/episodes_000.jsonl','parent_evidence/resume.pt'}<=set(got)
            (folder/'parent_evidence/metrics_000.jsonl').write_text('changed')
            with pytest.raises(AssertionError):task.required_inputs(folder,ep)
        elif kind=='v2_fresh': assert set(got)==set(task.INPUTS)|{'initial.pt','lineage.json'}
        else:assert set(got)==set(task.INPUTS)


def test_parent_evidence_traversal_is_rejected(tmp_path):
    ep=input_fixture(tmp_path/'run',2,'v1_resume');folder=tmp_path/'run'
    lineage=json.loads((folder/'lineage.json').read_text());lineage['parent_evidence_sha256']['../escape']='0'*64
    (folder/'lineage.json').write_text(json.dumps(lineage));ep['lineage_sha256']=task.sha(folder/'lineage.json')
    with pytest.raises(ValueError,match='Nonlocal'):task.required_inputs(folder,ep)


def test_stage_authenticates_both_source_and_relocated_complete_lineage(tmp_path,monkeypatch):
    monkeypatch.setattr(task,'ROOT',tmp_path)
    p=plan();r=recovery_plan(p);job=p['jobs'][0];endpoints={};expected={};calls=[]
    for c,name in job['training_jobs'].items():
        endpoints[name]=tmp_path/'source'/c
        expected[name]=input_fixture(endpoints[name],2,'v1_resume')
    def auth(model,path,recovered):
        calls.append(Path(path));return expected[model['logical_name']]
    monkeypatch.setattr(task.recovery,'authenticate',auth)
    manifests=task.stage_inputs(p,r,job,endpoints,expected)
    assert len(calls)==4
    for c,name in job['training_jobs'].items():
        target=tmp_path/job['output_dir']/c
        assert task.required_inputs(target,expected[name])==manifests[c]
        assert (target/'parent_evidence/metrics_000.jsonl').read_bytes()==(endpoints[name]/'parent_evidence/metrics_000.jsonl').read_bytes()
    first=tmp_path/job['output_dir']/'gc100/lineage.json';first.write_text('changed')
    with pytest.raises(AssertionError,match='Staged evidence'):task.stage_inputs(p,r,job,endpoints,expected)


def test_fixed_worker_command_has_all_v2_inputs_and_no_extra_cases(tmp_path,monkeypatch):
    monkeypatch.setattr(task,'ROOT',tmp_path);p=plan();r=recovery_plan(p);job=p['jobs'][0];expected={}
    monkeypatch.setattr(task,'static_files',lambda *a:['scripts/run_simple_memory_analysis.py','code/nets.py'])
    for c,name in job['training_jobs'].items():expected[name]=input_fixture(tmp_path/job['output_dir']/c,2,'v1_resume')
    args=task.launch_args(p,r,job,expected);argv=task.shlex.split(args.command)
    assert argv[:3]==['python','scripts/run_simple_memory_analysis.py','--runs']
    assert argv[-2:]==['--max-seconds','6600'] and args.hours==2 and args.include_only
    assert len(args.include)==len(set(args.include))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in args.include)
    for c in ('gc100','pgc33'):
        assert str(Path(job['output_dir'])/c/'parent_evidence/resume.pt') in args.include
        assert str(Path(job['output_dir'])/c/'migration_boundary.pt') in args.include


def test_immutable_worker_matching_requires_new_duration_reserve_and_command(tmp_path,monkeypatch):
    monkeypatch.setattr(task,'ROOT',tmp_path)
    p=plan();r=recovery_plan(p);job=p['jobs'][0];expected={}
    source=tmp_path/'scripts/run_simple_memory_analysis.py';source.parent.mkdir();source.write_text('private runner')
    monkeypatch.setattr(task,'static_files',lambda *a:['scripts/run_simple_memory_analysis.py'])
    for c,name in job['training_jobs'].items():
        expected[name]=input_fixture(tmp_path/job['output_dir']/c,2,'v1_resume')
    args=task.launch_args(p,r,job,expected)
    record={k:getattr(args,k) for k in ('name','machine','zone','output_dir')}
    record.update(instance_id='12345',command=task.shlex.split(args.command),duration_seconds=7200,reserved_usd=2.8125,
        gcs_prefix=f'gs://{task.cloud.BUCKET}/{task.cloud.PREFIX}/{args.name}',
        package_manifest=[dict(path=n,sha256=task.sha(tmp_path/n)) for n in args.include])
    task.match_record(p,r,job,expected,record,'12345')
    for field,value in [('duration_seconds',3600),('duration_seconds',7201),('reserved_usd',1.5625),
                        ('reserved_usd',2.8126),('instance_id','54321')]:
        changed=copy.deepcopy(record);changed[field]=value
        with pytest.raises(AssertionError):task.match_record(p,r,job,expected,changed,'12345')
    changed=copy.deepcopy(record);changed['command'][-1]='2900'
    with pytest.raises(AssertionError):task.match_record(p,r,job,expected,changed,'12345')


@pytest.mark.parametrize('prepared',[
    dict(duration_seconds=3600,reserved_usd=1.5625),
    dict(duration_seconds=7200,reserved_usd=1.5625),
    dict(duration_seconds=7201,reserved_usd=2.8125),
])
def test_prepared_runtime_or_reserve_mismatch_blocks_cloud_launch(tmp_path,monkeypatch,prepared):
    p=plan();r=recovery_plan(p)
    expected={n:{} for j in p['jobs'] for n in j['training_jobs'].values()}
    f=tmp_path/'status.json';f.write_text(json.dumps(dict(models={n:dict(endpoint={}) for n in expected})))
    monkeypatch.setattr(task,'TRAINING_STATUS',f)
    monkeypatch.setattr(task,'training_endpoints',lambda *a:{})
    monkeypatch.setattr(task,'authenticate_endpoints',lambda *a:expected)
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(dict(runs=[])))
    monkeypatch.setattr(task.cloud,'LEDGER',ledger)
    monkeypatch.setattr(task.recovery,'ensure_budget',lambda *a:{'full_scope_upper_usd':599})
    monkeypatch.setattr(task,'stage_inputs',lambda *a:{})
    monkeypatch.setattr(task,'launch_args',lambda *a:object())
    monkeypatch.setattr(task.cloud,'plan',lambda *a:prepared)
    monkeypatch.setattr(task.cloud,'launch',lambda *a:pytest.fail('Mismatched prepared plan must never launch'))
    with pytest.raises(AssertionError):task.cycle(p,r,dict(jobs={}))


def test_budget_helper_accounts_for_unlaunched_training_and_validation_too():
    p=plan();r=recovery_plan(p)
    ledger=dict(ancillary_reserve_usd=0,runs=[dict(name='past',actual_upper_bound_usd=470,reserved_usd=470)])
    # 12 future training*10 + 8 analyses*2.8125 + validation.9375 exceeds600.
    with pytest.raises(AssertionError,match='remaining training'):
        task.recovery.ensure_budget(r,ledger)
    ledger['runs'][0]['actual_upper_bound_usd']=450
    assert task.recovery.ensure_budget(r,ledger)['full_scope_upper_usd']==pytest.approx(593.4375)
    ledger['runs'].append(dict(name=p['jobs'][0]['name'],reserved_usd=2.8125))
    assert task.recovery.ensure_budget(r,ledger)['full_scope_upper_usd']==pytest.approx(593.4375)


def test_terminal_failed_analysis_never_launches_a_replacement(tmp_path,monkeypatch):
    p=plan();r=recovery_plan(p);job=p['jobs'][0]
    f=tmp_path/'status.json';f.write_text('{}');monkeypatch.setattr(task,'TRAINING_STATUS',f)
    monkeypatch.setattr(task,'training_endpoints',lambda *a:{})
    expected={n:{} for j in p['jobs'] for n in j['training_jobs'].values()}
    training=dict(models={n:dict(endpoint={}) for n in expected});f.write_text(json.dumps(training))
    monkeypatch.setattr(task,'authenticate_endpoints',lambda *a:expected)
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(dict(runs=[dict(name=job['name'],instance_id='123',gcs_prefix='gs://private')])) )
    monkeypatch.setattr(task.cloud,'LEDGER',ledger);monkeypatch.setattr(task,'match_record',lambda *a:None)
    monkeypatch.setattr(task,'cat',lambda *a:json.dumps({'exit_code':1}))
    def fail(*a):raise RuntimeError('worker exit1; outputs retained')
    monkeypatch.setattr(task,'finish',fail)
    monkeypatch.setattr(task.cloud,'launch',lambda *a:pytest.fail('No automatic retry'))
    with pytest.raises(RuntimeError,match='exit1'):task.cycle(p,r,dict(jobs={}))


def test_existing_terminal_helpers_are_reused_without_global_rebinding():
    assert task.finish is task.original.finish
    assert task.verify_pair is task.original.verify_pair


def frozen_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(task,'ROOT',tmp_path)
    p=plan();r=recovery_plan(p)
    ops={}
    for name in task.OPERATIONS:
        f=tmp_path/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# '+name)
        ops[name]=task.sha(f)
    for name in ('code/base.py','code/guard.py','scripts/science.py','results/validation.json'):
        f=tmp_path/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('private '+name)
    docs=tmp_path/'docs/research';docs.mkdir(parents=True)
    training=docs/'training.json';training.write_text(json.dumps(dict(status='frozen',source_sha256={'code/base.py':task.sha(tmp_path/'code/base.py')})))
    amendment=docs/'amendment.json';amendment.write_text(json.dumps(dict(status='frozen',source_sha256={
        'code/base.py':task.sha(tmp_path/'code/base.py'),'code/guard.py':task.sha(tmp_path/'code/guard.py')})))
    analysis=docs/'analysis.json';analysis.write_text(json.dumps(dict(status='frozen',training_protocol='docs/research/training.json',
        source_sha256={'scripts/science.py':task.sha(tmp_path/'scripts/science.py')},
        validation_artifacts_sha256={'results/validation.json':task.sha(tmp_path/'results/validation.json')})))
    original=docs/'original.json';original.write_text(json.dumps(dict(status='frozen',jobs=p['jobs'])))
    rp=docs/'recovery.json';rp.write_text('{}')
    r['bindings']=dict(amendment=dict(path='docs/research/amendment.json',sha256=task.sha(amendment)),
                       protocol=dict(path='docs/research/training.json',sha256=task.sha(training)))
    monkeypatch.setattr(task,'ANALYSIS_PROTOCOL',analysis);monkeypatch.setattr(task,'ORIGINAL_PLAN',original)
    monkeypatch.setattr(task.recovery,'validate',lambda path:r)
    p.update(status='frozen',readiness={'review':True},cumulative_ceiling_usd=600,
        recovery_plan_path='docs/research/recovery.json',recovery_plan_sha256=task.sha(rp),
        recovery_supervisor_sha256=ops['scripts/run_simple_memory_recovery.py'],
        analysis_protocol_sha256=task.sha(analysis),training_protocol_sha256=task.sha(training),
        amendment_sha256=task.sha(amendment),original_followup_plan_sha256=task.sha(original),
        operational_source_sha256=ops,worker_sources=['scripts/run_simple_memory_analysis.py','scripts/simple_memory_endpoint.py'])
    path=docs/'followup.json';path.write_text(json.dumps(p))
    assert task.validated_plan(path)==(p,r)
    return path,p,r,analysis


def test_frozen_plan_binds_new_analysis_and_recovery_versions(tmp_path,monkeypatch):
    path,p,r,analysis=frozen_fixture(tmp_path,monkeypatch)
    p['analysis_protocol_sha256']='old-analysis-protocol'
    path.write_text(json.dumps(p))
    with pytest.raises(AssertionError):task.validated_plan(path)
    p['analysis_protocol_sha256']=task.sha(analysis);p['operational_source_sha256'].pop('scripts/simple_memory_endpoint.py')
    path.write_text(json.dumps(p))
    with pytest.raises(AssertionError):task.validated_plan(path)


def test_runtime_draft_preserves_exact_eight_jobs_with_prospective_limits(tmp_path,monkeypatch):
    _,p,r,_=frozen_fixture(tmp_path,monkeypatch)
    recovery_path=tmp_path/'docs/research/recovery.json';recovery_path.write_text(json.dumps(r))
    draft=task.draft_plan(recovery_path)
    assert draft['status']=='draft' and not any(draft['readiness'].values())
    assert draft['jobs']==p['jobs'] and len(draft['jobs'])==8
    assert draft['per_worker_hours']==2 and draft['runner_max_seconds']==6600
    assert draft['per_worker_reservation_usd']==2.8125 and draft['total_analysis_reservation_usd']==22.5
    assert draft['max_active_workers']==8 and draft['cumulative_ceiling_usd']==600


def test_fixed_pair_or_budget_cannot_be_changed_by_new_plan(tmp_path,monkeypatch):
    path,p,r,_=frozen_fixture(tmp_path,monkeypatch)
    for key,value in [('runner_max_seconds',2900),('runner_max_seconds',6601),('per_worker_hours',1),
                      ('per_worker_reservation_usd',1.5625),('max_active_workers',9),('total_analysis_reservation_usd',12.5)]:
        changed=copy.deepcopy(p);changed[key]=value;path.write_text(json.dumps(changed))
        with pytest.raises(AssertionError):task.validated_plan(path)
    changed=copy.deepcopy(p);changed['jobs'][0]['seed']=99;path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError,match='Original eight'):task.validated_plan(path)


def test_manifest_conflict_cannot_override_frozen_analysis_source(tmp_path,monkeypatch):
    path,p,r,analysis=frozen_fixture(tmp_path,monkeypatch)
    science=tmp_path/'scripts/science.py';science.write_text('changed')
    p['operational_source_sha256']['scripts/science.py']=task.sha(science);path.write_text(json.dumps(p))
    with pytest.raises(AssertionError):task.validated_plan(path)


def test_static_package_contains_original_amended_and_analysis_closures(tmp_path,monkeypatch):
    path,p,r,_=frozen_fixture(tmp_path,monkeypatch)
    files=task.static_files(p,r)
    assert {'code/base.py','code/guard.py','scripts/science.py','results/validation.json',
            'scripts/run_simple_memory_analysis.py','scripts/simple_memory_endpoint.py',
            'docs/research/amendment.json','docs/research/training.json'}<=set(files)
    assert len(files)==len(set(files))


def test_lightweight_poll_checks_plan_sources_and_manifest_but_never_parent_bytes(tmp_path,monkeypatch):
    path,p,r,_=frozen_fixture(tmp_path,monkeypatch)
    # All source and small immutable JSON bindings are checked, while a large
    # parent/package path deliberately does not exist in this private fixture.
    r['operational_source_sha256']=p['operational_source_sha256']
    r['validation_artifacts_sha256']={'results/validation.json':task.sha(tmp_path/'results/validation.json')}
    r['models'][0]['package_manifest']={'missing_large_parent_checkpoint.pt':'never-read'}
    r['gpu_release']=dict(summary_path='results/validation.json',summary_sha256=task.sha(tmp_path/'results/validation.json'),
                          done_path='results/validation.json',done_sha256=task.sha(tmp_path/'results/validation.json'))
    binding=dict(plan_sha256=task.sha(path),controller_sha256=task.sha(task.__file__),
                 recovery_plan_sha256=p['recovery_plan_sha256'],analysis_protocol_sha256=p['analysis_protocol_sha256'])
    monkeypatch.setattr(task.recovery,'validate',lambda *a:pytest.fail('Do not revalidate large recovery packages while waiting'))
    task.check_poll_bindings(path,p,r,binding)
    original=path.read_bytes();path.write_bytes(original+b'\n')
    with pytest.raises(AssertionError,match='plan changed'):task.check_poll_bindings(path,p,r,binding)
    path.write_bytes(original)
    source=tmp_path/'scripts/run_simple_memory_recovery.py';source.write_text('changed operational source')
    with pytest.raises(AssertionError,match='run_simple_memory_recovery.py'):
        task.check_poll_bindings(path,p,r,binding)


def test_main_runs_full_validation_only_once_then_lightweight_polls(tmp_path,monkeypatch):
    path,p,r,_=frozen_fixture(tmp_path,monkeypatch)
    calls=[]
    def full(*a):calls.append('full');return p,r
    def poll(*a):calls.append('poll')
    def cycle(plan,recovered,state):
        calls.append('cycle');state['complete']=True
    monkeypatch.setattr(task,'validated_plan',full)
    monkeypatch.setattr(task,'check_poll_bindings',poll)
    monkeypatch.setattr(task,'verify_completed_cache',lambda *a:None)
    monkeypatch.setattr(task,'cycle',cycle)
    assert task.main(['--plan',str(path),'--out',str(tmp_path/'status')])==0
    assert calls==['full','poll','cycle']
