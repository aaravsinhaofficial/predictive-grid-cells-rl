"""Continuation-specific invariants; original task tests remain authoritative."""
import copy
from types import SimpleNamespace
import pytest
import torch
from code.common import CodeSource,build_population
from scripts import rl_blind_reorientation_entropy_extend as ext
from scripts import rl_blind_reorientation as task
from scripts.rl_visual_memory_extend import tree_equal
from tests.test_rl_visual_memory_noise import HookablePolicy
from tests.test_rl_blind_reorientation import cfg_for_test


def trained_tiny_parent():
    torch.set_num_threads(2);torch.manual_seed(119)
    cfg=cfg_for_test();net=HookablePolicy()
    opt=torch.optim.RMSprop(net.parameters(),lr=cfg['lr'],alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    runtime=task.fresh_runtime(net,cfg,'cpu',stagger=False)
    runtime['clock'].known[:]=True
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    task.update(net,cfg,opt,source,runtime,60)
    parent=dict(policy=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        runtime_snapshot=task.capture_runtime(runtime,'cpu'),config=cfg,recent_scores=[10.,20.])
    return parent,net,opt,runtime,source


def test_frame_budget_and_only_entropy_configuration_change():
    assert ext.PARENT_UPDATES==2156 and ext.BLOCK_FRAMES==10_014_720
    assert ext.EXTENSION_FRAMES==50_073_600 and ext.TOTAL_EXTRA_FRAMES==90_132_480
    assert ext.PARENT_UPDATES+ext.BLOCKS*ext.BLOCK_UPDATES==4601
    cfg=cfg_for_test()
    assert ext.continuation_config(cfg,'original')==cfg
    changed=ext.continuation_config(cfg,'explore')
    assert changed['entropy']==.01
    assert {k for k in cfg if cfg[k]!=changed[k]}=={'entropy'}
    assert cfg['entropy']==8e-5


def test_exact_optimizer_runtime_rng_resume_matches_uninterrupted_next_update():
    parent,net,opt,runtime,source=trained_tiny_parent();cfg=parent['config']
    expected=task.update(net,cfg,opt,source,runtime,60)
    resumed=HookablePolicy();resumed_opt,resumed_runtime=ext.restore_training(resumed,cfg,parent,'cpu')
    actual=task.update(resumed,cfg,resumed_opt,source,resumed_runtime,60)
    assert actual==expected
    assert tree_equal(net.state_dict(),resumed.state_dict())
    assert tree_equal(opt.state_dict(),resumed_opt.state_dict())
    assert tree_equal(task.capture_runtime(runtime,'cpu'),task.capture_runtime(resumed_runtime,'cpu'))
    assert parent['runtime_snapshot']['noise_replays']==1  # original snapshot not mutated


def test_entropy_arms_start_identical_and_only_differ_after_loss_update():
    parent,_,_,_,source=trained_tiny_parent();rollouts=[];trained=[]
    for arm in ext.ARMS:
        cfg=ext.continuation_config(parent['config'],arm);net=HookablePolicy()
        opt,runtime=ext.restore_training(net,cfg,parent,'cpu')
        roll,state0,_,_=task.collect(net,cfg,source,runtime,60)
        rollouts.append(roll)
        logits,values=task.replay(net,roll,state0)
        # Controlled same target isolates entropy; forced rows remain masked.
        loss,*_=task.losses(logits,values,values.detach()+1,torch.stack(roll['a']),torch.stack(roll['policy_mask']),cfg)
        opt.zero_grad();loss.backward();opt.step();trained.append(copy.deepcopy(net.state_dict()))
    assert tree_equal(rollouts[0],rollouts[1])
    assert not tree_equal(trained[0],trained[1])


def test_high_entropy_preserves_forced_action_gradient_mask():
    logits=torch.randn(5,2,6,requires_grad=True);values=torch.zeros(5,2,requires_grad=True)
    mask=torch.tensor([[0.,0.],[1.,1.],[0.,1.],[0.,0.],[1.,1.]])
    cfg=ext.continuation_config(cfg_for_test(),'explore')
    total,*_=task.losses(logits,values,torch.ones_like(values),torch.zeros(5,2,dtype=torch.long),mask,cfg)
    total.backward()
    assert not logits.grad[mask==0].any() and logits.grad[mask==1].abs().sum()>0
    assert values.grad[mask==0].abs().sum()>0


def test_checked_evaluation_preserves_runtime_optimizer_and_rejects_corruption(monkeypatch):
    parent,net,opt,runtime,_=trained_tiny_parent()
    monkeypatch.setattr(ext.task,'evaluate',lambda *a,**kw:dict(clean=True))
    assert ext.checked_evaluate(net,parent['config'],opt,runtime,'cpu',float('inf'))==dict(clean=True)
    assert tree_equal(task.capture_runtime(runtime,'cpu'),parent['runtime_snapshot'])
    def corrupt(*args,**kwargs):
        runtime['state'][0].add_(1);return {}
    monkeypatch.setattr(ext.task,'evaluate',corrupt)
    with pytest.raises(AssertionError,match='runtime'):
        ext.checked_evaluate(net,parent['config'],opt,runtime,'cpu',float('inf'))


def test_baseline_requires_all_twelve_episode_event_window_rows():
    base=dict(variants={v:dict(rows=[dict(slot=0,score=0.)]) for v in task.VARIANTS},
        free_window={v:dict(rows=[dict(free_updates=30)]) for v in task.VARIANTS},
        window_rows={v:[dict(end_phase='free_blind')] for v in task.VARIANTS},schedule={'admitted':[1]*12})
    ext.assert_baseline_equal(copy.deepcopy(base),base)
    for key,mutate in [('variants',lambda r:r.update(score=1.)),
                       ('free_window',lambda r:r.update(free_updates=29)),
                       ('window_rows',lambda r:r.update(end_phase='left_delay'))]:
        bad=copy.deepcopy(base);v=task.VARIANTS[-1]
        row=bad[key][v][0] if key=='window_rows' else bad[key][v]['rows'][0]
        mutate(row)
        with pytest.raises(AssertionError):ext.assert_baseline_equal(bad,base)
    bad=copy.deepcopy(base);bad['schedule']['admitted'][0]=2
    with pytest.raises(AssertionError):ext.assert_baseline_equal(bad,base)


def test_parent_validation_rejects_nonfinal_wrong_frames_and_optimizer(monkeypatch):
    cfg=ext.parent_design()
    import json
    config=json.loads((ext.ROOT/'models/gc100/gc100_lstm_s1_1000M.json').read_text());config['rollout']=160
    parent=dict(completed=True,original_frames=1_000_000_000,extra_frames=ext.PARENT_FRAMES,
        total_updates=ext.PARENT_UPDATES,warmup_updates=200,stage_evaluations_completed=4,
        provenance=cfg,config=config,optimizer={'state':{0:{'step':torch.tensor(2156.)}}},
        runtime_snapshot={'noise_replays':2156,'noise_max_replay_error':1e-6,'clock':{'phase':torch.zeros(32,dtype=torch.long)}})
    ext.validate_parent(parent,'gc100')
    for key,value in [('completed',False),('extra_frames',ext.PARENT_FRAMES-1),('total_updates',2155),('stage_evaluations_completed',3)]:
        bad=copy.deepcopy(parent);bad[key]=value
        with pytest.raises(AssertionError):ext.validate_parent(bad,'gc100')
    bad=copy.deepcopy(parent);bad['optimizer']['state'][0]['step']=torch.tensor(0.)
    with pytest.raises(AssertionError):ext.validate_parent(bad,'gc100')
