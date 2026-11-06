import copy
import math
import torch
from scripts import lstm_reorientation_entropy_diagnostic as diagnostic
from scripts import rl_blind_reorientation as task
from tests.test_rl_visual_memory_noise import HookablePolicy
from tests.test_rl_goal_memory import config


def test_private_rng_substitution_preserves_every_other_saved_state_and_input():
    cfg=config();cfg['rollout']=160
    runtime=task.fresh_runtime(HookablePolicy(),cfg,'cpu')
    saved=task.capture_runtime(runtime,'cpu')
    for k in diagnostic.SEEDS:saved[k]=torch.arange(16,dtype=torch.uint8)
    before=diagnostic.tree_digest(saved);private=diagnostic.cpu_snapshot(saved)
    assert diagnostic.tree_digest(saved)==before
    assert all(private[k].numel()==5056 for k in diagnostic.SEEDS)
    restored=task.fresh_runtime(HookablePolicy(),cfg,'cpu')
    task.restore_runtime(restored,private,'cpu')
    assert diagnostic.tree_digest(task.capture_runtime(restored,'cpu'))==diagnostic.tree_digest(private)
    restored['env'].pos.add_(1)
    assert diagnostic.tree_digest(saved)==before


def test_free_entropy_and_counts_exclude_other_phases_and_keep_zero_exposure():
    logits=torch.zeros(2,1,6);logits[0,0,0]=100
    actions=torch.tensor([[0],[2]])
    free=torch.tensor([[True],[False]])
    result=diagnostic.distribution_summary(logits,actions,free)
    assert result['entropy_nats']['mean']<1e-30 and result['turn_probability']>.99999
    assert result['actual_action_counts']==[1,0,0,0,0,0]
    visible=diagnostic.distribution_summary(logits,actions,~free)
    assert abs(visible['entropy_nats']['mean']-math.log(6))<1e-12
    absent=diagnostic.distribution_summary(logits,actions,torch.zeros_like(free))
    assert absent['updates']==0 and absent['entropy_nats'] is None and absent['retained_zero_exposure']
