import copy
import torch
from code.env import ArenaEnv
from scripts.preflight_local_goal_return import predicted_next_distance


def test_analysis_planner_predictor_matches_actual_native_step_for_every_action():
    base=ArenaEnv(4,device='cpu',vision='none',img=1,noise_v=0,noise_r=0,stagger=False,episode_len=1000)
    base.reset_all();base.goal[:]=torch.tensor([.9,.9]);base.pos[:]=torch.tensor([-.4,-.4])
    base.hd[:]=torch.tensor([0.,1.,2.,3.]);base.vf[:]=.03;base.vs[:]=-.02;base.vr[:]=1.
    expected=predicted_next_distance(base)
    for action in range(6):
        env=copy.deepcopy(base)
        _,_,_,info=env.step(torch.full((4,),action))
        actual=(info['hit_pos']-env.goal).norm(dim=1)
        torch.testing.assert_close(actual,expected[:,action],atol=2e-7,rtol=1e-6)
