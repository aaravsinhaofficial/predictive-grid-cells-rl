import numpy as np
import pytest
import torch
from code.env import ArenaEnv
from scripts.lstm_travel_action_context import BODY_OFFSET
from scripts.lstm_travel_memory_probe import forward_distance


@pytest.mark.parametrize('action',[2,3,4,5])
def test_translation_actions_share_world_path_and_final_movement_direction(action):
    cfg=dict(action_repeat=4,v_max=.35,arena_half=1.25,goal_radius=.3)
    n=96;theta=torch.arange(n)*2*np.pi/n
    endpoint=torch.tensor([.1,-.1]);unit=torch.stack((theta.cos(),theta.sin()),1)
    paths=[]
    for action_id in (2,action):
        env=ArenaEnv(n,vision='none',img_noise=0,noise_v=0,noise_r=0,stagger=False,device='cpu')
        env.reset_all();env.hd[:]=theta+BODY_OFFSET[action_id]
        env.vf.zero_();env.vs.zero_();env.vr.zero_()
        env.pos[:]=endpoint-forward_distance(cfg)*unit;env.goal[:]=torch.tensor([.9,.9])
        history=[]
        for _ in range(14):
            _,reward,done,info=env.step(torch.full((n,),action_id,dtype=torch.long))
            assert not bool(done.any() or info['hit'].any() or info['hit_wall'].any() or reward.any())
            history.append(env.pos.clone())
        assert float((env.pos-endpoint).norm(dim=1).max())<1e-5
        actual=torch.atan2(env.mdir[:,1],env.mdir[:,0])
        delta=torch.atan2((actual-theta).sin(),(actual-theta).cos())
        assert float(delta.abs().max())<1e-5
        paths.append(torch.stack(history))
    assert float((paths[0]-paths[1]).abs().max())<1e-5
