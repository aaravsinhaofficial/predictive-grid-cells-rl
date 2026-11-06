import copy
import numpy as np
import torch
from code.common import CodeSource,build_population
from scripts.rl_visual_memory_noise import StateNoise,extension
from tests.test_rl_goal_memory import TinyPolicy,config


class HookablePolicy(TinyPolicy):
    @property
    def lstm(self):return self.cell


def test_noisy_snapshot_reproduces_training_and_noise_exactly():
    torch.set_num_threads(2);torch.manual_seed(813);np.random.seed(813)
    cfg=config();cfg.update(vision='raycast',img_noise=.02,episode_len=45)
    net=HookablePolicy();opt=torch.optim.RMSprop(net.parameters(),lr=cfg['lr'],alpha=cfg['rms_alpha'],eps=cfg['rms_eps'])
    runtime=extension.fresh_runtime(net,cfg,'cpu');runtime['env'].t.zero_();runtime['env'].partial.zero_()
    runtime['env'].pos[:]=runtime['env'].goal;runtime['obs']=runtime['env'].observe()
    source=CodeSource(cfg,build_population(cfg,'cpu'),'cpu')
    noise=StateNoise('cpu')
    with noise.installed():
        for _ in range(4):extension.update(net,cfg,opt,source,runtime)
        assert runtime['clock'].mask().any()
        checkpoint=dict(policy=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                        snapshot=extension.capture_runtime(runtime,'cpu'))
        expected=[extension.update(net,cfg,opt,source,runtime) for _ in range(5)]
        expected_runtime=extension.capture_runtime(runtime,'cpu')
    restored=HookablePolicy();restored.load_state_dict(checkpoint['policy'])
    restored_opt=extension.restored_optimizer(restored,cfg,checkpoint)
    restored_runtime=extension.fresh_runtime(restored,cfg,'cpu')
    resumed_noise=StateNoise('cpu')
    with resumed_noise.installed():
        extension.restore_runtime(restored_runtime,checkpoint['snapshot'],'cpu')
        actual=[extension.update(restored,cfg,restored_opt,source,restored_runtime) for _ in range(5)]
        assert actual==expected
        assert extension.tree_equal(extension.capture_runtime(restored_runtime,'cpu'),expected_runtime)
    assert extension.tree_equal(restored.state_dict(),net.state_dict())
    assert extension.tree_equal(restored_opt.state_dict(),opt.state_dict())
    assert noise.max_replay_error<1e-6


def test_noise_is_absent_when_visible_and_matched_across_evaluation_branches():
    n=3;groups=6;state=(torch.zeros(n*groups,5),torch.zeros(n*groups,5))
    mask=torch.tensor([True,False,True]).repeat(groups)
    noise=StateNoise('cpu');before=torch.get_rng_state().clone()
    eps=noise.draw(state,mask,groups=groups)
    assert torch.equal(torch.get_rng_state(),before),'Dedicated noise must not steal action RNG draws'
    for e in eps:
        assert torch.count_nonzero(e[~mask])==0
        assert e[mask].abs().sum()>0
        for g in range(1,groups):assert torch.equal(e[:n],e[g*n:(g+1)*n])
