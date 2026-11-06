import numpy as np
import torch
from code.env import ArenaEnv
from scripts.lstm_travel_memory_probe import forward_distance,path_clearance


def test_forward_starts_reach_one_endpoint_with_native_physics():
    cfg=dict(action_repeat=4,v_max=.35,arena_half=1.25,goal_radius=.3)
    n=96;env=ArenaEnv(n,vision='none',img_noise=0,noise_v=0,noise_r=0,stagger=False,device='cpu')
    env.reset_all();theta=torch.arange(n)*2*np.pi/n
    env.hd[:]=theta;env.vf.zero_();env.vs.zero_();env.vr.zero_()
    endpoint=torch.tensor([.1,-.1])
    env.pos[:]=endpoint-forward_distance(cfg)*torch.stack((theta.cos(),theta.sin()),1)
    env.goal[:]=torch.tensor([.9,.9])
    for _ in range(14):
        _,reward,done,info=env.step(torch.full((n,),2,dtype=torch.long))
        assert not bool(done.any() or info['hit'].any() or info['hit_wall'].any() or reward.any())
    assert float((env.pos-endpoint).norm(dim=1).max())<1e-5
    actual=torch.atan2(env.mdir[:,1],env.mdir[:,0])
    delta=torch.atan2((actual-theta).sin(),(actual-theta).cos())
    assert float(delta.abs().max())<1e-5
    assert float(env.vf.min())>.3  # a turn command alone would not stop it


def test_all_angle_clearance_rejects_wall_and_goal_intersections():
    cfg=dict(action_repeat=4,v_max=.35,arena_half=1.25,goal_radius=.3)
    assert path_clearance(np.array([0.,0.]),np.array([.9,.9]),cfg)['eligible']
    assert not path_clearance(np.array([1.1,0.]),np.array([-.9,-.9]),cfg)['eligible']
    assert not path_clearance(np.array([0.,0.]),np.array([.4,0.]),cfg)['eligible']
