"""Adversarial checks of the independent artifact audit (not training code)."""
import copy
import json
import pytest
import torch
from code.common import build_policy
from scripts import audit_rl_blind_reorientation as audit
from scripts import rl_blind_reorientation as task
from tests.test_rl_visual_memory_noise import HookablePolicy
from tests.test_rl_goal_memory import config


@pytest.fixture(scope='module')
def evaluation():
    parent=task.MatchedArena
    class FirstHitArena(parent):
        def reset_all(self):
            super().reset_all();self.pos[:]=self.goal
            return self.observe()
    task.MatchedArena=FirstHitArena
    try:
        cfg=config();cfg.update(cond='gc100',pgc_frac=0.,episode_len=180,rollout=160)
        data=task.evaluate(HookablePolicy(),cfg,device='cpu',episodes=2,n_envs=2,max_delay=60)
    finally:task.MatchedArena=parent
    return data,cfg


def test_real_small_evaluation_passes_all_independent_event_checks(evaluation):
    data,cfg=evaluation
    result=audit.audit_evaluation(data,cfg,'gc100',episodes=2,n_envs=2)
    assert len(result)==12 and all(r['episodes']==2 for r in result.values())
    assert sum(r['admitted_events'] for r in result.values())>0


@pytest.mark.parametrize('corruption', ['first_discovery','free_hits','brake','missing_window','reset_count'])
def test_corrupt_saved_evaluation_is_rejected(evaluation,corruption):
    original,cfg=evaluation;data=copy.deepcopy(original)
    if corruption=='first_discovery':data['variants']['blind_ordinary']['rows'][0]['first_goal_step']+=1
    if corruption=='free_hits':data['free_window']['blind_native']['free_hits']+=1
    if corruption=='brake':data['window_rows']['blind_native'][0]['post_brake_max_abs'][2]=.01
    if corruption=='missing_window':data['window_rows']['blind_native'].pop()
    if corruption=='reset_count':data['free_window']['blind_reset_after_glimpse']['resets']+=1
    with pytest.raises(AssertionError):audit.audit_evaluation(data,cfg,'gc100',episodes=2,n_envs=2)


@pytest.fixture(scope='module')
def incoming():
    cfgpath=audit.ROOT/'models/gc100/gc100_lstm_s1_1000M.json'
    cfg=json.loads(cfgpath.read_text());cfg['rollout']=160
    original=torch.load(cfgpath.with_suffix('.pt'),map_location='cpu',weights_only=True)
    net=build_policy(cfg,'cpu');net.load_state_dict(original['policy'])
    runtime=task.fresh_runtime(net,cfg,'cpu')
    design=json.loads(audit.PROTOCOL.read_text());design.pop('frozen_at')
    opt=torch.optim.RMSprop(net.parameters(),lr=0.,alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    data=dict(policy=net.state_dict(),optimizer=opt.state_dict(),config=cfg,provenance=design,
        completed=False,original_frames=1_000_000_000,extra_frames=0,total_updates=0,warmup_updates=200,
        runtime_snapshot=task.capture_runtime(runtime,'cpu'),stage_evaluations_completed=0,recent_scores=[])
    return data,cfg,design,original


def test_real_incoming_checkpoint_has_original_weights_new_optimizer_and_complete_runtime(incoming):
    data,cfg,design,original=incoming
    result=audit.audit_checkpoint(data,cfg,design,original,0,0,0,False,True)
    assert result['runtime']['noise_replays']==0 and result['runtime']['admissions']==0
    assert 'optimizer' not in original


@pytest.mark.parametrize('corruption',['parent_weights','goal_survives_reset','coverage','missing_rng','missing_motor','training_reset','restored_optimizer'])
def test_corrupt_saved_checkpoint_is_rejected(incoming,corruption):
    original_data,cfg,design,original=incoming;data=copy.deepcopy(original_data)
    if corruption=='parent_weights':next(iter(data['policy'].values())).flatten()[0]+=.1
    if corruption=='goal_survives_reset':data['runtime_snapshot']['goal'][0,0]=1.
    if corruption=='coverage':data['runtime_snapshot']['clock_stats']['coverage_violations'][0]=1
    if corruption=='missing_rng':del data['runtime_snapshot']['noise_rng']
    if corruption=='missing_motor':del data['runtime_snapshot']['environment']['vf']
    if corruption=='training_reset':data['runtime_snapshot']['input_tensors']['reset_every'][0]=True
    if corruption=='restored_optimizer':data['optimizer']['state'][0]={'step':torch.tensor(200.)}
    with pytest.raises(AssertionError):audit.audit_checkpoint(data,cfg,design,original,0,0,0,False,True)


def test_missing_uploads_are_pending_and_audit_is_read_only(tmp_path):
    result=audit.run(tmp_path,'gc100')
    assert result['status']=='pending' and result['passed'] and not result['complete']
    assert not list(tmp_path.iterdir())
    strict=audit.run(tmp_path,'gc100',require_complete=True)
    assert strict['status']=='pending' and not strict['passed']


def test_nonfinite_nested_losses_or_states_are_rejected():
    with pytest.raises(AssertionError):audit.finite_tree({'loss':float('nan')})
    with pytest.raises(AssertionError):audit.finite_tree({'state':(torch.tensor([float('inf')]),)})
