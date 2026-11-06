"""Corruption tests for the independent fixed-budget training auditor."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import pytest
import torch
import audit_local_goal_return_training as audit


def metric(update):
    forced=32*33;free=960;active=forced+free
    return dict(update=update,stage=(update-1)//1024+1,completed_trials=32,setup_pickups=32,
        setup_raw_reward=320,setup_learning_reward=0,forced_hits=0,terminal_bootstrap_zero=True,
        controller='policy',allocated_slots=5120,active_physics_updates=active,forced_updates=forced,
        free_updates=free,padded_slots=5120-active,free_hits=0,clearance_min=.02,wall_clearance_min=.5,
        snap_max_m=1e-7,replay_error=1e-6,grad_norm=.7,entropy_per_free_decision=.8,
        entropy_sum_per_trial=24.,policy_loss=2.,value_loss=3.,loss=.625*(2+1.5-.24),
        active_physics_frames=active*4*update)


def checkpoint(updates=0,stages=0):
    policy={'pi.weight':torch.zeros(6,256),'pi.bias':torch.zeros(6),
        'v.weight':torch.zeros(1,256),'v.bias':torch.zeros(1)}
    baseline=dict(policy=policy,config={'action_repeat':4},initialization={'test':True})
    provenance=dict(device='cuda');c=dict(updates=updates,trials=updates*32,completed_stages=stages,
        evaluation_records=[dict(stage=i) for i in range(1,stages+1)])
    for key in audit.COUNTERS:c[key]=metric(1)[key]*updates
    stream=audit.task.TrialStreams(294801).snapshot();stream.update(batches=updates,contexts=updates*32)
    group=dict(lr=3e-4,alpha=.99,eps=1e-5,weight_decay=0,momentum=0,centered=False,
        capturable=False,maximize=False,differentiable=False,foreach=None,params=list(range(4)))
    optimizer=dict(param_groups=[group],state={i:dict(step=torch.tensor(float(updates)),square_avg=torch.ones_like(v))
        for i,v in enumerate(policy.values())} if updates else {})
    data=dict(version='local_return_exact_boundary_resume_v1',no_inflight_trial=True,
        boundary_state='Everytrialterminated; nexttrialh/czero,pa−1/pr0',config=baseline['config'],
        initialization=baseline['initialization'],provenance=provenance,counters=c,
        active_physics_frames=c['active_physics_updates']*4,completed=updates==4096 and stages==4,
        policy=copy.deepcopy(policy),optimizer=optimizer,streams=stream,
        global_rng=dict(cpu=torch.Generator().manual_seed(5).get_state(),cuda=[torch.zeros(16,dtype=torch.uint8)]))
    return data,baseline,provenance


def test_metric_rejects_forced_reward_and_wrong_noise_replay():
    audit.audit_metric(metric(1))
    for key,value in [('forced_hits',1),('setup_learning_reward',1),('replay_error',.001),('loss',2.5)]:
        row=metric(1);row[key]=value
        with pytest.raises(AssertionError):audit.audit_metric(row)


def test_checkpoint_rejects_incoming_weight_or_private_rng_changes():
    data,baseline,provenance=checkpoint();audit.audit_checkpoint(data,baseline,provenance,0,0)
    data['policy']['pi.weight'][0,0]=1
    with pytest.raises(AssertionError):audit.audit_checkpoint(data,baseline,provenance,0,0)
    data,baseline,provenance=checkpoint();data['streams']['generators']['action']=torch.Generator().manual_seed(9).get_state()
    with pytest.raises(AssertionError):audit.audit_checkpoint(data,baseline,provenance,0,0)


def test_checkpoint_rejects_optimizer_restart_and_padding_miscount():
    data,baseline,provenance=checkpoint(1024,1);audit.audit_checkpoint(data,baseline,provenance,1024,1)
    data['optimizer']['state'][0]['step']=torch.tensor(1.)
    with pytest.raises(AssertionError):audit.audit_checkpoint(data,baseline,provenance,1024,1)
    data,baseline,provenance=checkpoint(1024,1);data['counters']['padded_slots']+=1
    with pytest.raises(AssertionError):audit.audit_checkpoint(data,baseline,provenance,1024,1)


def test_evaluation_must_preserve_policy_optimizer_and_rng():
    before,_,_=checkpoint(1024,0);after=copy.deepcopy(before)
    after['counters']['completed_stages']=1;after['counters']['evaluation_records']=[dict(stage=1)]
    assert audit.compare_evaluation_boundary(before,after,1)
    for field in ('policy','optimizer','streams','global_rng'):
        bad=copy.deepcopy(after)
        if field=='policy':bad[field]['pi.bias'][0]=.1
        elif field=='optimizer':bad[field]['state'][0]['square_avg'][0,0]=2
        elif field=='streams':bad[field]['generators']['plan'][20]^=1
        else:bad[field]['cuda'][0][0]=1
        with pytest.raises(AssertionError):audit.compare_evaluation_boundary(before,bad,1)


def write_rows(folder,rows):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_resume_excludes_rolled_back_attempt_and_rejects_missing_accepted_rows(tmp_path):
    parent=tmp_path/'resume_parent_evidence';write_rows(parent,[metric(i) for i in range(1,5)])
    write_rows(tmp_path,[metric(i) for i in range(3,6)])
    (tmp_path/'resume_lineage.json').write_text(json.dumps(dict(restored_updates=2)))
    rows,meta=audit.accepted_metrics(tmp_path)
    assert len(rows)==5 and meta['rolled_back_rows']==2
    write_rows(parent,[metric(1)])
    with pytest.raises(AssertionError):audit.accepted_metrics(tmp_path)


def test_metrics_reject_duplicate_update_and_nonfinite_loss(tmp_path):
    write_rows(tmp_path,[metric(1),metric(1)])
    with pytest.raises(AssertionError):audit.accepted_metrics(tmp_path)
    row=metric(1);row['loss']=float('nan');write_rows(tmp_path,[row])
    with pytest.raises(AssertionError):audit.accepted_metrics(tmp_path)


def test_saved_first_free_actor_is_independently_reconstructed():
    folder=ROOT/'results/local_goal_return_baseline/gc100'
    if not (folder/'summary.json').exists():pytest.skip('Published baseline raw inputs required')
    raw=torch.load(folder/'final_hard/context0/raw.pt',map_location='cpu',weights_only=True)
    saved=torch.load(folder/'initialized_policy.pt',map_location='cpu',weights_only=True)
    audit.verify_actor(raw,saved['policy'],list(audit.task.VARIANTS))
    raw['first_free'][0]['logits'][0]+=.1
    with pytest.raises(AssertionError):audit.verify_actor(raw,saved['policy'],list(audit.task.VARIANTS))
