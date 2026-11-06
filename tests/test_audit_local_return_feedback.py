"""Synthetic corruption checks only; no study policy or sealed bank is loaded."""
import copy
import json
import math
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import audit_local_return_feedback as audit
import local_return_feedback as feedback
import local_return_feedback_evaluate as evaluator
import local_return_feedback_run as runner
from test_local_return_feedback import fixture


@pytest.fixture(scope='module')
def sample():
    net,cfg,source,_=fixture(741971)
    plan=feedback.TrialStreams(741972).draw(1,feedback.STAGE,cfg,headings=16)
    roll,rows,checks,first=feedback.collect(net,cfg,source,plan)
    raw={key:torch.stack(value) for key,value in roll.items() if isinstance(value,list) and value}
    raw.update(goal=roll['goal'],bootstrap=roll['bootstrap'],goal_world=plan['goal'],first_free=first)
    return net,cfg,plan,raw,rows,checks


def test_saved_native_inputs_physics_and_actor_reconcile(sample):
    net,cfg,plan,raw,rows,_=sample
    audit.base.verify_trajectory(plan,cfg,raw,rows,raw['first_free'],['feedback_native'])
    audit.verify_observation_history(plan,cfg,raw)
    report=audit.verify_inputs(plan,cfg,raw)
    assert report['pixels']>0 and report['nonboundary_max_abs_error']==0
    audit.verify_actor(plan,cfg,raw,rows,net.state_dict())


def test_stale_current_code_and_post_brake_velocity_are_detected(sample):
    _,cfg,plan,raw,_,_=sample
    broken=dict(raw,code=raw['code'].clone());t=int(feedback.schedule(plan,'cpu')['glimpse'][0])
    broken['code'][t]=raw['code'][t-1]
    with pytest.raises(AssertionError,match='Current code'):audit.verify_inputs(plan,cfg,broken)
    broken=dict(raw,observation_velocity=raw['observation_velocity'].clone())
    broken['observation_velocity'][t].zero_()
    with pytest.raises(AssertionError,match='velocity'):audit.verify_observation_history(plan,cfg,broken)


def test_black_stale_and_unverified_rgb_frames_are_rejected(sample):
    _,cfg,plan,raw,_,_=sample
    t=20;expected,boundary=audit.render_saved(raw['observation_position'][t],raw['observation_heading'][t],
        plan['land_pos'],plan['stripe_phase'],cfg['arena_half'])
    with pytest.raises(AssertionError,match='RGB'):audit.verify_rgb(torch.zeros_like(expected),expected,boundary)
    with pytest.raises(AssertionError,match='RGB'):audit.verify_rgb(raw['rgb'][t-5],expected,boundary)
    with pytest.raises(AssertionError,match='Insufficient'):audit.verify_rgb(expected,expected,torch.ones_like(boundary))


def test_explicit_stripe_boundary_is_counted_separately():
    # Set one ray's stripe fraction exactly to 0.15 in float64 geometry.
    pos=torch.zeros(1,2);hd=torch.tensor([.2]);land=torch.full((1,4),-.9);stripe=torch.zeros(1,4)
    col=42;q=(torch.arange(84,dtype=torch.float32)+.5)/84;angle=float(hd[0])+float(math.pi/4-math.pi/2*q[col])
    along=1.25*math.tan(angle);stripe[0,0]=(.15-((along+1.25)/2.5)*6)%1
    image,boundary=audit.render_saved(pos,hd,land,stripe,1.25)
    assert bool(boundary[0,:,col].all()) and float((~boundary).float().mean())>.5
    changed=image.clone();changed[:,:,20,col]=1-changed[:,:,20,col]
    report=audit.verify_rgb(changed,image,boundary)
    assert report['excluded_boundary_pixels']>0 and report['boundary_mismatches_above_atol']==1
    other=(~boundary[0]).nonzero()[0];changed[:,:,other[0],other[1]]+=.01
    with pytest.raises(AssertionError,match='RGB'):audit.verify_rgb(changed,image,boundary)


def test_exact_raw_tensor_and_compressed_byte_corruption(tmp_path):
    value={'rgb':torch.arange(18,dtype=torch.float32).reshape(2,3,3),'metadata':{'unchanged':True}}
    record=evaluator.save_raw(tmp_path/'raw.pt.gz',value)
    assert audit.equal(audit.load_raw(tmp_path,record),value)
    bad=copy.deepcopy(record);bad['tensor_schema']['/rgb']['sha256']='0'*64
    with pytest.raises(AssertionError,match='Uncompressed'):audit.load_raw(tmp_path,bad)
    path=tmp_path/'raw.pt.gz';path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(AssertionError):audit.load_raw(tmp_path,record)


def test_first_free_actor_input_and_cdf_corruption(sample):
    net,cfg,plan,raw,rows,_=sample
    first={i:dict(v) for i,v in raw['first_free'].items()};first[0]['logits']=first[0]['logits']+.1
    with pytest.raises(AssertionError):audit.verify_actor(plan,cfg,dict(raw,first_free=first),rows,net.state_dict())
    first={i:dict(v) for i,v in raw['first_free'].items()};first[0]['input']=first[0]['input'].clone();first[0]['input'][256]+=.1
    with pytest.raises(AssertionError):audit.verify_actor(plan,cfg,dict(raw,first_free=first),rows,net.state_dict())
    broken=dict(raw,action=raw['action'].clone());start=int(feedback.schedule(plan,'cpu')['free'][0])
    broken['action'][start,0]=(broken['action'][start,0]+1)%6
    with pytest.raises(AssertionError):audit.verify_actor(plan,cfg,broken,rows,net.state_dict())


def valid_metric(update=1):
    free=960;forced=1100;active=free+forced;ent=1.5;entropy_sum=ent*free/32
    return dict(update=update,completed_trials=32,setup_pickups=32,setup_raw_reward=320,
        setup_learning_reward=0,forced_hits=0,terminal_bootstrap_zero=True,controller='policy',
        fresh_feedback_on_every_active_input=True,state_noise_sigma=0,midtrial_resets=0,unique_endpoint_cue=False,
        allocated_slots=5120,active_physics_updates=active,forced_updates=forced,free_updates=free,
        padded_slots=5120-active,free_hits=0,clearance_min=.2,wall_clearance_min=.3,snap_max_m=1e-7,
        replay_error=0,value_replay_error=0,grad_norm=1,entropy_per_free_decision=ent,
        entropy_sum_per_trial=entropy_sum,policy_loss=.1,value_loss=.2,
        loss=(100/160)*(.1+.5*.2-8e-5*entropy_sum),action_counts=[160]*6,
        active_physics_frames=active*update*4)


@pytest.mark.parametrize('key,value',[('loss',2.),('value_replay_error',.001),('forced_hits',1),
    ('state_noise_sigma',.02),('midtrial_resets',1),('action_counts',[0]*6)])
def test_learning_metric_corruptions_rejected(key,value):
    row=valid_metric();audit.audit_metric(row);row[key]=value
    with pytest.raises(AssertionError):audit.audit_metric(row)


def test_metric_resume_counts_only_accepted_prefix(tmp_path):
    parent=tmp_path/'resume_parent_evidence';parent.mkdir()
    (parent/'metrics.jsonl').write_text(''.join(json.dumps(valid_metric(u))+'\n' for u in (1,2,3)))
    (tmp_path/'resume_lineage.json').write_text(json.dumps({'restored_updates':2}))
    (tmp_path/'metrics.jsonl').write_text(json.dumps(valid_metric(3))+'\n')
    rows,meta=audit.accepted_metrics(tmp_path)
    assert len(rows)==3 and meta['rolled_back_rows']==1
    (tmp_path/'metrics.jsonl').write_text(json.dumps(valid_metric(4))+'\n')
    with pytest.raises(AssertionError):audit.accepted_metrics(tmp_path)


def test_checkpoint_all_parameter_optimizer_rng_and_boundary_corruptions():
    net,cfg,_,_=fixture(741991);opt=feedback.optimizer(net,cfg);streams=feedback.TrialStreams(494801)
    initial={'policy':copy.deepcopy(net.state_dict()),'config':cfg,'initialization':{'synthetic':True}}
    provenance={'device':'cpu'}
    saved=runner.snapshot(net,opt,streams,cfg,initial['initialization'],runner.fresh_counters(),provenance)
    audit.audit_checkpoint(saved,initial,provenance)
    for mutate in (
        lambda d:d['optimizer']['param_groups'][0]['params'].pop(),
        lambda d:d['streams']['generators'].pop('state'),
        lambda d:d['global_rng'].pop('mps'),
        lambda d:d['policy']['pi.bias'].add_(.1),
        lambda d:d.update(completed=True)):
        bad=copy.deepcopy(saved);mutate(bad)
        with pytest.raises(AssertionError):audit.audit_checkpoint(bad,initial,provenance)
    after=copy.deepcopy(saved);after['counters']['evaluation_records']['baseline']={'record':True}
    audit.compare_boundary(saved,after,'baseline')
    after['streams']['generators']['action'][20]^=1
    with pytest.raises(AssertionError,match='streams'):audit.compare_boundary(saved,after,'baseline')


def test_summary_invalid_failures_keep_full_denominator():
    rows=[dict(variant='feedback_native',context=c,heading_index=h,hit=False,invalid=True) for c in range(16) for h in range(16)]
    expected=evaluator.summarize(rows,16,16);actual=audit.summarize_rows(rows)
    assert actual==expected and actual['trials']==256 and actual['invalid_trials']==256
    rows[0]['hit']=True
    with pytest.raises(AssertionError):audit.summarize_rows(rows)


@pytest.mark.parametrize('controller',['uniform','planner'])
def test_reference_rules_use_saved_noise_and_no_source_policy(controller):
    _,cfg,source,_=fixture(742001)
    plan=feedback.TrialStreams(742002).draw(1,feedback.STAGE,cfg,headings=16)
    net=evaluator.ReferencePolicy(cfg['nh'])
    roll,rows,_,first=feedback.collect(net,cfg,source,plan,controller=controller,retain_inputs=False)
    raw={key:torch.stack(value) for key,value in roll.items() if isinstance(value,list) and value}
    raw.update(goal=roll['goal'],bootstrap=roll['bootstrap'],goal_world=plan['goal'],first_free=first)
    audit.base.verify_trajectory(plan,cfg,raw,rows,first,['feedback_native'])
    audit.verify_observation_history(plan,cfg,raw)
    audit.verify_reference(plan,cfg,raw,rows,controller)
    broken=dict(raw,action=raw['action'].clone());start=int(feedback.schedule(plan,'cpu')['free'][0])
    broken['action'][start,0]=(broken['action'][start,0]+3)%6
    with pytest.raises(AssertionError):audit.verify_reference(plan,cfg,broken,rows,controller)


def test_initialization_composition_rejects_nonheads_and_wrong_actor(tmp_path):
    net,cfg,_,_=fixture(742011)
    original=copy.deepcopy(net.state_dict());archived=copy.deepcopy(original)
    for key in feedback.HEADS:archived[key]=archived[key]+.05
    config=json.loads((ROOT/'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    cfg=dict(config,n_envs=32,rollout=160,entropy=8e-5)
    path=tmp_path/'original.pt';archive=tmp_path/'archive.pt'
    torch.save({'frames':1_000_000_000,'policy':original},path)
    path.with_suffix('.json').write_text(json.dumps(config))
    torch.save({'no_optimization':True,'policy':archived},archive)
    for mode in feedback.ACTOR_MODES:
        composed={key:(archived[key] if key.startswith('v.') or (mode=='reset_actor' and key.startswith('pi.')) else value).clone() for key,value in original.items()}
        init=dict(condition='pgc33',actor_mode=mode,parent_checkpoint_path=path.name,parent_checkpoint_sha256=audit.sha(path),
            parent_config_sha256=audit.sha(path.with_suffix('.json')),archived_heads_path=archive.name,archived_heads_sha256=audit.sha(archive),
            parent_protocol_sha256=feedback.PARENT_PROTOCOL_SHA256,full_initial_policy_hash=audit.base.policy_hash(composed),
            incoming_head_hashes={k:feedback.task.tensor_hash(composed[k]) for k in feedback.HEADS},
            original_nonhead_hashes={k:feedback.task.tensor_hash(v) for k,v in original.items() if k not in feedback.HEADS},
            all_parameters_trainable=True,critic_from_exact_untrained_archive=True,no_optimization=True)
        initial=dict(policy=composed,config=cfg,initialization=init,no_forward_pass=True,no_optimization=True,
            collector_source_sha256=audit.sha(feedback.__file__))
        audit.audit_initial(initial,'pgc33',mode,tmp_path)
        for key in ('encoder.weight','pi.weight','v.weight'):
            bad=copy.deepcopy(initial);bad['policy'][key][0,0]+=.1
            with pytest.raises(AssertionError,match='composition'):audit.audit_initial(bad,'pgc33',mode,tmp_path)
