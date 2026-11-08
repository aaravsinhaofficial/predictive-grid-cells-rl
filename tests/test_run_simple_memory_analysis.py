"""Private operational fixtures only: no trained policy/environment execution."""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_simple_memory_analysis as runner


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))
    return runner.sha(path)


def endpoint_fixture(tmp_path):
    root=tmp_path/'repo'; run=tmp_path/'outputs'/'gc100'; run.mkdir(parents=True)
    files=list(runner.PRODUCERS.values())+['code/train_memory.py']
    for name in files:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# private fixture '+name)
    sources={name:runner.sha(root/name) for name in files}
    training=dict(status='frozen',source_sha256={'code/train_memory.py':sources['code/train_memory.py']},
        training_config=dict(frames=1_000_000_000,tf32=0),
        allowed_cohort=dict(conditions=['gc100','pgc33'],seeds=[1,2,3,4],task_max_gaps=[0,60],
          pgc_fractions={'gc100':0.,'pgc33':.33},code_seed_equals_seed=True,clock_seed_multiplier=1000,clock_seed_offset=2))
    training_sha=put(root/'docs/research/simple_memory_training_protocol.json',training)
    ap=root/'docs/research/simple_memory_analysis_protocol.json'
    ah=put(ap,dict(status='frozen',source_sha256=sources,training_protocol='docs/research/simple_memory_training_protocol.json'))
    cfg=dict(frames=1_000_000_000,tf32=0,cond='gc100',seed=1,code_seed=1,clock_seed=1002,task_max_gap=60,pgc_frac=0.)
    proof=dict(source_sha256=training['source_sha256'],protocol_sha256=training_sha,
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        deterministic_algorithms=True,cudnn_benchmark=False,cudnn_tf32=False,matmul_tf32=False,
        torch_version='private',cuda_version='fixture',cudnn_version=1,
        cuda_device_name='Tesla T4',cuda_capability=[7,5])
    weights={'private.weight':torch.arange(6,dtype=torch.float32).reshape(2,3)}
    h=hashlib.sha256()
    for key,x in sorted(weights.items()):h.update(key.encode());h.update(x.numpy().tobytes())
    saved=dict(schema=1,frames=1_000_000_000,updates=78125,runtime={'counters':{'frames':1_000_000_000,'updates':78125,'decisions':250_000_000}},
               config=cfg,provenance=proof,policy=weights,policy_sha256=h.hexdigest())
    torch.save(saved,run/'final.pt')
    put(run/'config.json',dict(cfg,population={'private_fixture':True}));put(run/'provenance.json',proof)
    put(run/'summary.json',dict(complete=True,frames=1_000_000_000,updates=78125,decisions=250_000_000,provenance=proof,
                               final_sha256=runner.sha(run/'final.pt'),final_policy_sha256=h.hexdigest()))
    auth=functools.partial(runner.authenticate_run,analysis_protocol=ap,expected_analysis_sha256=ah)
    return root,run,auth


def test_endpoint_authentication_exact_config_policy_and_sources(tmp_path):
    root,run,auth=endpoint_fixture(tmp_path)
    state=torch.get_rng_state().clone()
    proof=auth(run,root=root)
    assert proof['frames']==1_000_000_000 and torch.equal(state,torch.get_rng_state())
    cfg=runner.read(run/'config.json');cfg['pgc_frac']=.33;put(run/'config.json',cfg)
    with pytest.raises(ValueError,match='Population'):auth(run,root=root)
    cfg['pgc_frac']=0.;put(run/'config.json',cfg)
    p=root/runner.PRODUCERS['dynamics'];p.write_text('changed')
    with pytest.raises(ValueError,match='source'):auth(run,root=root)


def test_endpoint_rejects_missing_config_key_and_changed_weights(tmp_path):
    root,run,auth=endpoint_fixture(tmp_path)
    saved=torch.load(run/'final.pt',weights_only=False)
    saved['config'].pop('clock_seed')
    torch.save(saved,run/'final.pt');s=runner.read(run/'summary.json');s['final_sha256']=runner.sha(run/'final.pt');put(run/'summary.json',s)
    with pytest.raises(ValueError,match='config'):auth(run,root=root)
    saved['config']['clock_seed']=1002;saved['policy']['private.weight'][0,0]=99
    torch.save(saved,run/'final.pt');s['final_sha256']=runner.sha(run/'final.pt');put(run/'summary.json',s)
    with pytest.raises(ValueError,match='Policy tensor'):auth(run,root=root)


def test_commands_are_fixed_and_activity_has_outer_timeout_only(tmp_path):
    for stage in runner.STAGES:
        cmd=runner.command(stage,tmp_path,123.456)
        assert cmd[:2]==[sys.executable,str(ROOT/runner.PRODUCERS[stage])]
        assert '--device' in cmd if stage!='activity' else '--device' not in cmd
        if stage!='activity':assert cmd[cmd.index('--max-seconds')+1]=='123.456000'
        if stage!='behavior':assert cmd[cmd.index('--threads')+1]=='2'
        if stage=='activity':assert '--figures' in cmd and '--max-seconds' not in cmd
    with pytest.raises(ValueError):runner.command('states',tmp_path,0)


def test_pipeline_order_shared_deadline_valid_skip_and_portable_paths(tmp_path):
    run=tmp_path/'gc100';run.mkdir()
    (run/runner.OUTPUTS['behavior']).write_text('existing valid fixture')
    ticks=[0.];calls=[]
    def clock():return ticks[0]
    def auth(run,**kwargs):return {'checkpoint_sha256':'fixture'}
    def validate(stage,run,endpoint,**kwargs):return {'complete':True,'fixture_stage':stage}
    def execute(argv,log,timeout,**kwargs):
        calls.append((argv,timeout));log.write_text('private child');ticks[0]+=3
        return dict(exit_code=0,timed_out=False,seconds=3)
    out=tmp_path/'pipeline.json'
    p=runner.run_pipeline([run],out,30,authenticate=auth,validate=validate,execute=execute,clock=clock,runtime_probe=lambda e:{})
    assert p['complete'] and p['runs'][0]['run']=='gc100'
    assert p['runs'][0]['stages']['behavior']['status']=='skipped_valid'
    assert [Path(a[1]).name for a,t in calls]==[Path(runner.PRODUCERS[s]).name for s in runner.STAGES[1:]]
    assert [t for a,t in calls]==[30,27,24,21]
    assert p['seconds']==12


def test_auth_all_runs_before_execution_and_preserve_partial_without_retry(tmp_path):
    runs=[tmp_path/'gc100',tmp_path/'pgc33']
    for run in runs:run.mkdir()
    calls=[]
    def auth(run,**kwargs):
        if run.name=='pgc33':raise ValueError('bad endpoint')
        return {}
    def execute(*args,**kwargs):calls.append(args);raise AssertionError('must not execute')
    p=runner.run_pipeline(runs,tmp_path/'pipeline.json',authenticate=auth,execute=execute)
    assert not p['complete'] and not calls
    partial=runs[0]/'memory_eval_gap60.json';partial.write_text('partial evidence')
    def invalid(*args,**kwargs):raise ValueError('partial producer')
    p=runner.run_pipeline(runs[:1],tmp_path/'attempt2.json',authenticate=lambda *a,**k:{},validate=invalid,execute=execute,runtime_probe=lambda e:{})
    assert not p['complete'] and partial.read_text()=='partial evidence' and not calls


def test_timeout_records_failure_and_does_not_start_next_stage(tmp_path):
    run=tmp_path/'gc100';run.mkdir();calls=[]
    def execute(argv,log,timeout,**kwargs):
        calls.append(argv);log.write_text('before timeout')
        return dict(exit_code=-9,timed_out=True,seconds=1)
    p=runner.run_pipeline([run],tmp_path/'pipeline.json',authenticate=lambda *a,**k:{},execute=execute,runtime_probe=lambda e:{})
    assert not p['complete'] and len(calls)==1
    assert p['runs'][0]['stages']['behavior']['timed_out']
    assert (tmp_path/'pipeline_logs/0_behavior.log').read_text()=='before timeout'


@pytest.mark.parametrize('limit',[0,-1,6600.001,float('inf'),float('nan')])
def test_global_limit_rejects_invalid_budget_before_authentication(tmp_path,limit):
    def forbidden(*args,**kwargs):pytest.fail('Invalid budget must not authenticate or execute')
    out=tmp_path/'pipeline.json'
    with pytest.raises(ValueError,match='Global budget'):
        runner.run_pipeline([tmp_path/'gc100'],out,limit,authenticate=forbidden,execute=forbidden)
    assert not out.exists()


@pytest.mark.parametrize('final_overrun',[0.,.001])
def test_two_models_share_6600_second_limit_including_final_checks(tmp_path,final_overrun):
    runs=[tmp_path/'gc100',tmp_path/'pgc33']
    for run in runs:run.mkdir()
    ticks=[0.];timeouts=[]
    def execute(argv,log,timeout,**kwargs):
        timeouts.append(timeout);log.write_text('private fixed stage')
        ticks[0]+=660+ (final_overrun if len(timeouts)==10 else 0)
        return dict(exit_code=0,timed_out=False,seconds=660)
    p=runner.run_pipeline(runs,tmp_path/'pipeline.json',6600,
        authenticate=lambda *a,**k:{},validate=lambda *a,**k:{'complete':True},
        execute=execute,clock=lambda:ticks[0],runtime_probe=lambda e:{})
    assert timeouts==[6600-660*i for i in range(10)]
    assert p['max_seconds']==6600 and p['complete']==(final_overrun==0)
    assert all(set(record['stages'])==set(runner.STAGES) for record in p['runs'])
    if final_overrun:assert 'final authentication' in p['error']['message']


def test_subprocess_timeout_kills_child_and_retains_output(tmp_path):
    result=runner.run_child([sys.executable,'-c',"import time; print('retained',flush=True); time.sleep(10)"],tmp_path/'child.log',.3)
    assert result['timed_out'] and result['exit_code']!=0
    assert (tmp_path/'child.log').read_text().strip()=='retained'


def test_behavior_gate_retains_all_eight_variants_and_actual_frames(tmp_path):
    root,run,auth=endpoint_fixture(tmp_path);e=auth(run,root=root)
    rows=[dict(episode=i//32,slot=i%32) for i in range(128)]
    s=dict(complete=True,condition='gc100',training_seed=1,training_task_max_gap=60,
        checkpoint_sha256=e['checkpoint_sha256'],config_sha256=e['config_sha256'],checkpoint_frames=e['frames'],
        source_sha256={runner.PRODUCERS['behavior']:runner.sha(root/runner.PRODUCERS['behavior'])},
        seed=741021,episodes_per_variant=128,n_envs=32,max_gap=60,min_gap=30,visible_steps=15,frames=5_529_600,
        seconds=1.,all_first_discoveries_identical=True,grid_swap_is_exact_sham=True,
        evaluation_numerics=dict(device='cuda:0',deterministic_algorithms=True,cudnn_tf32=False,matmul_tf32=False),
        variants={v:dict(rows=rows,scheduled_action_counts=[0]*6,totals={}) for v in runner.VARIANTS})
    put(run/'memory_eval_gap60.json',s)
    assert runner.validate_stage('behavior',run,e,root=root)['frames']==5_529_600
    s['variants'].pop('seen');put(run/'memory_eval_gap60.json',s)
    with pytest.raises(ValueError,match='variants'):runner.validate_stage('behavior',run,e,root=root)


def test_zero_context_dynamics_is_valid_completion_not_positive_selection(tmp_path):
    root,run,auth=endpoint_fixture(tmp_path);e=auth(run,root=root)
    put(run/'memory_states/summary.json',dict(onset_contexts={'count':0}))
    s=dict(complete=True,condition='gc100',training_seed=1,training_task_max_gap=60,
        checkpoint_sha256=e['checkpoint_sha256'],config_sha256=e['config_sha256'],checkpoint_frames=e['frames'],
        analysis_protocol_sha256=e['analysis_protocol_sha256'],policy_unchanged=True,
        source_sha256={runner.PRODUCERS['dynamics']:runner.sha(root/runner.PRODUCERS['dynamics'])},
        contexts=[],state_summary_sha256=runner.sha(run/'memory_states/summary.json'),
        numerical_settings={'family_device':'cuda:0','threads':2})
    put(run/'dynamics_analysis/summary.json',s)
    assert runner.validate_stage('dynamics',run,e,root=root)['complete']
    s['contexts']=[{'index':0}];put(run/'dynamics_analysis/summary.json',s)
    with pytest.raises(ValueError,match='dropped contexts'):runner.validate_stage('dynamics',run,e,root=root)


def test_runtime_mismatch_fails_before_first_scientific_subprocess(tmp_path):
    root,run,auth=endpoint_fixture(tmp_path);e=auth(run,root=root)
    actual=dict(e['training_runtime'],cuda_available=True)
    runner.validate_runtime_versions(actual,[e])
    actual['cudnn_version']+=1
    with pytest.raises(ValueError,match='match'):runner.validate_runtime_versions(actual,[e])
    actual=dict(e['training_runtime'],cuda_available=False)
    with pytest.raises(ValueError,match='CUDA'):runner.validate_runtime_versions(actual,[e])
    def missing(endpoints):raise ImportError('private missing pyplot fixture')
    def forbidden(*a,**k):raise AssertionError('Must not start a stage')
    p=runner.run_pipeline([run],tmp_path/'outputs/pipeline.json',root=root,authenticate=auth,runtime_probe=missing,execute=forbidden)
    assert not p['complete'] and p['error']['type']=='ImportError'
    assert not p['runs'][0]['stages']


def test_unqualified_noop_is_retained_and_cannot_be_promoted(tmp_path):
    root,run,auth=endpoint_fixture(tmp_path);e=auth(run,root=root)
    put(run/'memory_states/summary.json',dict(onset_contexts={'count':1,'rows':[{'episode_id':3,'global_step':44}]}))
    (run/'memory_states/states.npz').write_bytes(b'private hash fixture')
    put(run/'activity_analysis/summary.json',dict(complete=True))
    proof=dict(final_checkpoint_sha256=e['checkpoint_sha256'],final_policy_sha256=e['policy_sha256'],
        analysis_protocol_sha256=e['analysis_protocol_sha256'],training_protocol_sha256=e['training_protocol_sha256'],
        collection_summary_sha256=runner.sha(run/'memory_states/summary.json'),states_sha256=runner.sha(run/'memory_states/states.npz'),
        producer_sha256=runner.sha(root/runner.PRODUCERS['activity']))
    base=run/'intervention_analysis';base.mkdir()
    for name in ('cue.npz','behavior.pt'):(base/name).write_bytes(b'private archived artifact')
    row=dict(index=0,episode_id=3,global_step=44,family={'qualified':True},qualified=False,cue_repair={'complete':True},
        behavior=dict(complete=True,validation_qualified=False,validations={'no_op_native_continuation':{'qualified':False}},
                      tapes=4,horizon=150,records=[{} for _ in range(100)]),
        artifacts={n:{'sha256':runner.sha(base/n),'bytes':(base/n).stat().st_size} for n in ('cue.npz','behavior.pt')})
    put(base/'context0.json',row)
    s=dict(complete=True,condition='gc100',training_seed=1,training_task_max_gap=60,provenance=proof,
        analysis_protocol_sha256=e['analysis_protocol_sha256'],policy_unchanged=True,contexts=[row],
        source_sha256={runner.PRODUCERS['interventions']:runner.sha(root/runner.PRODUCERS['interventions'])},
        activity_summary_sha256=runner.sha(run/'activity_analysis/summary.json'))
    put(base/'summary.json',s)
    assert runner.validate_stage('interventions',run,e,root=root)['complete']
    row['qualified']=True;put(base/'context0.json',row);put(base/'summary.json',s)
    with pytest.raises(ValueError,match='No-op failure'):runner.validate_stage('interventions',run,e,root=root)


def test_relocated_pipeline_reauthenticates_relative_paths(tmp_path,monkeypatch):
    import shutil
    root,run,auth=endpoint_fixture(tmp_path)
    def validate(stage,run,endpoint,**kwargs):
        p=run/(stage+'.fixture')
        assert p.read_text()=='retained'
        return dict(complete=True,artifacts={p.name:{'sha256':runner.sha(p),'bytes':p.stat().st_size}})
    def execute(argv,log,timeout,**kwargs):
        stage=next(s for s in runner.STAGES if str(root/runner.PRODUCERS[s])==argv[1])
        (run/(stage+'.fixture')).write_text('retained');log.write_text('complete')
        return dict(exit_code=0,timed_out=False,seconds=.1)
    def probe(endpoints):return dict(endpoints[0]['training_runtime'],cuda_available=True)
    p=runner.run_pipeline([run],run.parent/'pipeline.json',root=root,authenticate=auth,validate=validate,execute=execute,runtime_probe=probe)
    assert p['complete']
    relocated=tmp_path/'downloaded';shutil.copytree(run.parent,relocated)
    monkeypatch.setattr(runner,'authenticate_run',auth);monkeypatch.setattr(runner,'validate_stage',validate)
    assert runner.verify_pipeline(relocated,root=root)['stages']==5
    # Relocated verification uses the same amended ceiling and rejects both an
    # oversized requested envelope and a result exceeding its accepted envelope.
    record=runner.read(relocated/'pipeline.json')
    record.update(max_seconds=6600,seconds=6600)
    put(relocated/'pipeline.json',record)
    assert runner.verify_pipeline(relocated,root=root)['stages']==5
    for change in ({'max_seconds':6600.001},{'seconds':6600.001}):
        put(relocated/'pipeline.json',dict(record,**change))
        with pytest.raises(ValueError,match='deadline record'):
            runner.verify_pipeline(relocated,root=root)
    put(relocated/'pipeline.json',record)
    (relocated/'gc100/states.fixture').write_text('changed')
    with pytest.raises(AssertionError):runner.verify_pipeline(relocated,root=root)
