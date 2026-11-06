import numpy as np
import torch
from code.env import ArenaEnv
from scripts.lstm_blind_turn_necessity import ceilings,identical_bytes,brake,require_clean_step


def test_any_common_action_mixture_is_bounded_and_phase_aware_can_exceed_it():
    p=np.array([[0,0,1,-1,0,0],[0,0,-1,1,0,0]],dtype=float)
    c=ceilings(p);assert c['common_action_progress_m']==0 and c['phase_aware_progress_m']==1
    assert c['best_phase_actions']==[2,3]
    rng=np.random.default_rng(61)
    for _ in range(100):
        pi=rng.dirichlet(np.ones(6));assert (p*pi).sum(1).mean()<=c['common_action_progress_m']+1e-12


def test_byte_equality_catches_signed_zero_that_numeric_equality_misses():
    x=np.zeros((2,4),dtype=np.float32);assert identical_bytes(x)
    x[1,0]=-0.;assert np.array_equal(x[0],x[1]) and not identical_bytes(x)


def test_common_suffix_matches_motors_after_different_native_turns():
    env=ArenaEnv(2,vision='none',img_noise=0,noise_v=0,noise_r=0,stagger=False,device='cpu')
    env.reset_all();env.pos[:]=torch.tensor([0.,0.]);env.goal[:]=torch.tensor([.9,.9]);env.hd.zero_();brake(env)
    for _ in range(14):require_clean_step(env,[0,1])
    assert not torch.equal(env.vr[:1],env.vr[1:])
    brake(env)
    for action in [0,1,0]:require_clean_step(env,[action,action])
    assert identical_bytes(torch.stack((env.vf,env.vs,env.vr),1).numpy())
    assert identical_bytes(env.pos.numpy())
    separation=torch.atan2((env.hd[0]-env.hd[1]).sin(),(env.hd[0]-env.hd[1]).cos()).abs()
    assert float(separation)>2.8
