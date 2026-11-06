"""Adversarial saved-evidence fixtures; no study policies or sealed banks."""
import copy
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import pytest
import torch
from code.nets import PolicyNet
import goal_cue_memory as task
import goal_cue_memory_evaluate as evaluator
import audit_goal_cue_memory as audit
from test_goal_cue_memory import fixture


def collected(seed=793901,delay=45,name='brief_native',angles=2,**kwargs):
    net,cfg,source,plan=fixture(seed,delay=delay,angles=angles)
    roll,rows,checks,evidence=task.collect(net,cfg,source,plan,**evaluator.variant_spec(name),**kwargs)
    raw={key:torch.stack(value)if isinstance(value,list)and value else value for key,value in roll.items()}
    raw['evidence']=evidence
    return net,cfg,plan,raw,rows


def test_complete_stationary_prefix_from_zero_checks_erasure_and_single_free_input():
    for name in ('brief_native','brief_reset','continuous_reset','no_cue'):
        net,cfg,plan,raw,rows=collected(name=name)
        report=audit.verify_stationary_prefix(plan,cfg,raw,net.state_dict(),embedding_function=net.embed)
        assert report['stationary_updates']==60 and report['zero_initial_state_reconstructed']
        assert max(report['maximum_errors'].values())<1e-5
        assert audit.verify_masks(plan,raw,name)['wait_updates']==120
        damaged=copy.deepcopy(raw);damaged['evidence']['cue_end_state'][0,0]+=.01
        with pytest.raises(AssertionError,match='Cue-end'):audit.verify_stationary_prefix(plan,cfg,damaged,net.state_dict(),embedding_function=net.embed)
        damaged=copy.deepcopy(raw);damaged['evidence']['first_free'][0]['state_before_input'][0]+=.01
        with pytest.raises(AssertionError,match='Incoming'):audit.verify_stationary_prefix(plan,cfg,damaged,net.state_dict(),embedding_function=net.embed)
        if name=='brief_reset':
            damaged=copy.deepcopy(raw);damaged['reset_before'][15]=False
            with pytest.raises(AssertionError):audit.verify_masks(plan,damaged,name)
            with pytest.raises(AssertionError):audit.verify_stationary_prefix(plan,cfg,damaged,net.state_dict(),embedding_function=net.embed)


def test_native_shape_functional_encoder_cell_and_cross_dtype_guard():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(793902)
        net=PolicyNet(nh=256,conv_arch='si',encoder_norm='layernorm')
        rgb=torch.rand(4,3,84,84)
        x=torch.randn(4,1287)*.1
    before=torch.get_rng_state().clone();policy=net.state_dict()
    with torch.no_grad():
        embedded=audit.embed_saved(rgb,policy,{'conv_arch':'si','encoder_norm':'layernorm'})
        assert torch.equal(embedded,net.embed(rgb))
        state=torch.zeros(4,512);native=net.zero_state(4,'cpu');double=state.double()
        policy64={key:value.double()for key,value in policy.items()}
        for t in range(61):
            if t==15:
                state[:2]=0;double[:2]=0
                native=tuple(torch.cat((torch.zeros_like(value[:2]),value[2:]))for value in native)
            state=audit.cell_saved(x,state,policy);native=net.lstm(x,native)
            double=audit.cell_saved(x.double(),double,policy64)
            audit.near(state,torch.cat(native,1),'Native-shape LSTM implementation')
            audit.near(state,double,'Prospective float32/float64 accumulation guard')
        assert torch.equal(before,torch.get_rng_state())
        damaged=double.clone();damaged[0,0]+=.01
        with pytest.raises(AssertionError):audit.near(state,damaged,'Material corruption')


def test_native_physics_reconstruction_rejects_wrong_tape_clock_and_wait_motion():
    net,cfg,plan,raw,rows=collected(delay=3)
    report=audit.verify_physics(plan,cfg,raw,rows)
    assert max(report['maximum_errors'].values())<audit.PHYSICS_ATOL
    audit.verify_action_statistics(plan,raw,rows)
    damaged=copy.deepcopy(plan);damaged['motor']=damaged['motor'].roll(1,0)
    with pytest.raises(AssertionError):audit.verify_physics(damaged,cfg,raw,rows)
    damaged=copy.deepcopy(raw);damaged['position'][3,0,0]=.001
    with pytest.raises(AssertionError):audit.verify_physics(plan,cfg,damaged,rows)
    damaged=copy.deepcopy(raw);damaged['action'][18,0]=(damaged['action'][18,0]+1)%6
    with pytest.raises(AssertionError):audit.verify_action_statistics(plan,damaged,rows)
    damaged=copy.deepcopy(raw);damaged['terminal'][18,0]=True
    with pytest.raises(AssertionError):audit.verify_masks(plan,damaged,'brief_native')


def test_current_images_and_goal_inputs_cannot_leak_or_be_stale():
    net,cfg,plan,raw,rows=collected(delay=3)
    report=audit.verify_inputs(plan,cfg,raw,'brief_native')
    assert report['exact_full_goal_withdrawal'] and report['pixels']>0
    damaged=copy.deepcopy(raw);damaged['goal_input'][15]=damaged['saved_goal']
    with pytest.raises(AssertionError,match='leaked'):audit.verify_inputs(plan,cfg,damaged,'brief_native')
    damaged=copy.deepcopy(raw);damaged['rgb'][30]=0
    with pytest.raises(AssertionError,match='Fresh RGB'):audit.verify_inputs(plan,cfg,damaged,'brief_native')
    damaged=copy.deepcopy(raw);damaged['code'][30]=damaged['code'][0]
    with pytest.raises(AssertionError,match='Fresh current'):audit.verify_inputs(plan,cfg,damaged,'brief_native')


def test_genuine_donor_before_input_replay_and_wrong_source_rejection():
    net,cfg,source,plan=fixture(793903,delay=3,angles=4)
    roll,rows,checks,evidence=task.collect(net,cfg,source,plan,cue_mode='brief')
    native={key:torch.stack(value)if isinstance(value,list)and value else value for key,value in roll.items()};native['evidence']=evidence
    ids=evaluator.donor_indices('donor_plus90',4)
    override=torch.stack([evidence['first_free'][int(i)]['state_before_input']for i in ids])
    roll,rows,checks,evidence=task.collect(net,cfg,source,plan,cue_mode='brief',first_free_state_override=override)
    raw={key:torch.stack(value)if isinstance(value,list)and value else value for key,value in roll.items()};raw['evidence']=evidence
    audit.verify_inputs(plan,cfg,raw,'donor_plus90',native)
    audit.verify_masks(plan,raw,'donor_plus90')
    audit.verify_stationary_prefix(plan,cfg,raw,net.state_dict(),embedding_function=net.embed)
    damaged=copy.deepcopy(raw);damaged['first_free_state_override']=torch.stack([native['evidence']['first_free'][int(i)]['state']for i in ids])
    with pytest.raises(AssertionError,match='genuine'):audit.verify_inputs(plan,cfg,damaged,'donor_plus90',native)
    damaged=copy.deepcopy(raw);damaged['state_override_mask'][19]=True
    with pytest.raises(AssertionError,match='Donor'):audit.verify_masks(plan,damaged,'donor_plus90')


def test_literal_denominator_and_fixed_gate_arithmetic():
    rows=[dict(context=c,angle_index=a,hit=False)for c in range(16)for a in range(16)]
    with pytest.raises(AssertionError):audit.summarize_rows(rows[:-1],16,16)
    with pytest.raises(AssertionError):audit.summarize_rows(rows[:-1]+[rows[0]],16,16)
    branches={}
    for name,hits in [('continuous',192),('brief_native',167),('brief_reset',128)]:
        values=copy.deepcopy(rows)
        for item in values[:hits]:item['hit']=True
        branches[name]=dict(totals=audit.summarize_rows(values,16,16),all_physical_checks=True)
    assert audit.expected_gates(branches)['useful_memory']['passed']
    branches['brief_reset']['totals']['successes']=129
    assert not audit.expected_gates(branches)['useful_memory']['passed']
    branches['continuous']['totals']['valid_trials']=255
    assert not audit.expected_gates(branches)['qualification']['passed']


def test_complete_evaluator_artifact_reconciliation_and_source_corruption(tmp_path):
    net,cfg,_,plan=fixture(793904,delay=0,angles=4)
    folder=tmp_path/'evaluation';reference={'synthetic':True}
    report=evaluator.evaluate_plan(net,cfg,plan,reference,folder,variants=('brief_native','brief_reset','donor_plus90','self_swap'))
    checked=audit.audit_evaluation(folder,{'policy':net.state_dict(),'config':cfg},plan,reference,report['variant_order'],embedding_function=net.embed)
    assert checked['complete'] and checked['valid'] and len(checked['contexts'])==4
    saved=audit.read(folder/'summary.json');saved['source_sha256']='0'*64
    (folder/'summary.json').write_text(__import__('json').dumps(saved))
    with pytest.raises(AssertionError):audit.audit_evaluation(folder,{'policy':net.state_dict(),'config':cfg},plan,reference,report['variant_order'],embedding_function=net.embed)


def checkpoint_fixture():
    import goal_cue_memory_run as runner
    net,cfg,_,_=fixture(793905,delay=0)
    opt=torch.optim.RMSprop(net.parameters(),lr=3e-4,alpha=.99,eps=1e-5)
    provenance={'device':'cpu'};initialization={'synthetic':True}
    counters=runner.fresh_counters();streams=task.TrialStreams(694801)
    data=runner.snapshot(net,opt,streams,cfg,initialization,counters,provenance)
    initial={'policy':copy.deepcopy(net.state_dict()),'config':cfg,'initialization':initialization}
    return data,initial,provenance


def test_checkpoint_rejects_hidden_runtime_optimizer_and_clock_changes():
    saved,initial,provenance=checkpoint_fixture()
    assert audit.audit_checkpoint(saved,initial,provenance)['updates']==0
    corruptions=[lambda x:x['streams'].__setitem__('delay_draws',1),
        lambda x:x['optimizer']['param_groups'][0].__setitem__('lr',.01),
        lambda x:x['counters'].__setitem__('wait_updates',1),
        lambda x:x['counters']['delay_histograms']['block1'].__setitem__(1,1),
        lambda x:x.__setitem__('completed',True),
        lambda x:x['policy']['pi.bias'].add_(.01)]
    for change in corruptions:
        damaged=copy.deepcopy(saved);change(damaged)
        with pytest.raises(AssertionError):audit.audit_checkpoint(damaged,initial,provenance)
    after=copy.deepcopy(saved);after['counters']['evaluation_records']['baseline']={'synthetic':True}
    audit.compare_boundary(saved,after,'baseline')
    after['global_rng']['cpu'][10]^=1
    with pytest.raises(AssertionError,match='global_rng'):audit.compare_boundary(saved,after,'baseline')


def metric_fixture():
    entropy=math.log(6);free=2880
    return dict(update=1,block=1,delay_cap=0,delay_steps=0,completed_trials=32,setup_pickups=32,
        setup_learning_reward=0.,forced_rewards=0.,forced_actions=0,terminal_bootstrap_zero=True,
        controller='policy',cue_mode='brief',goal_kind='native',img_noise=0.,state_noise_sigma=0.,erased_states=0,
        goal_override_used=False,first_free_state_override_used=False,native_environment_rng_unchanged=True,
        full_goal_vector_withdrawn=True,task_owned_stationary_wait=True,free_motor_noise_v=.03,free_motor_noise_r=.15,
        allocated_slots=5120,wait_updates=480,active_recurrent_updates=3360,free_updates=free,active_physics_updates=free,
        padded_slots=1760,free_hits=0,replay_error=0.,value_replay_error=0.,grad_norm=1.,
        entropy_per_free_decision=entropy,entropy_sum_per_trial=entropy*free/32,
        policy_loss=.2,value_loss=.3,loss=(100/160)*(.2+.5*.3-8e-5*entropy*free/32))


def test_training_metric_and_delay_draw_accounting_reject_credit_corruption(tmp_path):
    row=metric_fixture();audit.audit_metric(row)
    path=tmp_path/'metrics.jsonl';path.write_text(__import__('json').dumps(row)+'\n')
    rows,report=audit.accepted_metrics(tmp_path)
    assert rows==[row] and report['active_physics_frames']==11520
    for field,value in [('wait_updates',481),('policy_loss',.3),('delay_steps',1),('free_hits',33),('setup_learning_reward',10.)]:
        damaged=dict(row,**{field:value})
        with pytest.raises(AssertionError):audit.audit_metric(damaged)
    path.write_text(__import__('json').dumps(row)+'\n'+__import__('json').dumps(row)+'\n')
    with pytest.raises(AssertionError):audit.accepted_metrics(tmp_path)


def test_checkpoint_delay_swap_is_rejected_with_same_histogram_draw_count_and_rng():
    generator=torch.Generator().manual_seed(694806);rows=[]
    counts={f'block{block}':[0]*46 for block in (1,2,3)}
    for index in range(1040):
        block=index//1024+1;cap=(0,15,45)[block-1]
        value=int(torch.randint(0,cap+1,(1,),generator=generator))
        rows.append(dict(update=index+1,block=block,delay_cap=cap,delay_steps=value))
        counts[f'block{block}'][value]+=1
    saved=dict(counters=dict(updates=len(rows),delay_histograms=counts),
        streams=dict(delay_draws=len(rows),generators={'delay':generator.get_state().clone()}))
    audit.verify_checkpoint_delays(rows,saved)
    damaged=copy.deepcopy(rows);left=1024
    right=next(i for i in range(left+1,len(rows))if rows[i]['delay_steps']!=rows[left]['delay_steps'])
    damaged[left]['delay_steps'],damaged[right]['delay_steps']=damaged[right]['delay_steps'],damaged[left]['delay_steps']
    changed_counts={f'block{block}':[0]*46 for block in (1,2,3)}
    same_rng=torch.Generator().manual_seed(694806)
    for row in damaged:
        changed_counts[f'block{row["block"]}'][row['delay_steps']]+=1
        torch.randint(0,row['delay_cap']+1,(1,),generator=same_rng)
    assert changed_counts==counts and len(damaged)==saved['streams']['delay_draws']
    assert torch.equal(same_rng.get_state(),saved['streams']['generators']['delay'])
    with pytest.raises(AssertionError,match='Checkpoint private delay draw'):
        audit.verify_checkpoint_delays(damaged,saved)


def test_completed_negative_qualification_is_integrity_pass_not_training_permission(tmp_path,monkeypatch):
    import goal_cue_memory_run as runner
    net,cfg,_,plan=fixture(793906,delay=0,angles=4)
    root=tmp_path/'qualification';root.mkdir();protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    design={'synthetic':True};init={'synthetic':True}
    provenance=dict(phase='qualification',condition=cfg['cond'],original_seed=1,protocol_sha256=audit.sha(protocol),design=design,qualification_gate=None)
    for name,value in [('protocol.json',provenance),('initialization.json',init),('training_config.json',cfg)]:audit.atomic_json(root/name,value)
    reference={'synthetic':True}
    monkeypatch.setattr(runner.evaluator,'evaluate_bank',lambda net,cfg,out,**kwargs:evaluator.evaluate_plan(net,cfg,plan,reference,out,variants=('continuous',)))
    runner.run_qualification(net,cfg,init,provenance,root,device='cpu',manifest=tmp_path/'unused')
    monkeypatch.setattr(audit,'validate_protocol',lambda p:design)
    monkeypatch.setattr(audit,'audit_initial',lambda incoming,condition:({'policy':net.state_dict(),'config':cfg,'initialization':init},{'synthetic':True}))
    monkeypatch.setattr(audit,'bank_reference',lambda design,name:(plan,reference))
    monkeypatch.setattr(audit,'embed_saved',lambda rgb,policy,cfg:net.embed(rgb))
    result=audit.audit(root,cfg['cond'],'qualification',protocol)
    assert result['passed'] and result['complete'] and not result['checks']['qualified'],result['failures']
    after=torch.load(root/'qualification_after.pt',weights_only=True);after['policy']['pi.bias']+=.01
    torch.save(after,root/'qualification_after.pt')
    result=audit.audit(root,cfg['cond'],'qualification',protocol)
    assert not result['passed'] and 'Qualification changed'in result['failures'][0]
