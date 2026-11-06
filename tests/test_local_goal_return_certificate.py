import itertools
import math
import numpy as np
import pytest
import torch
from code.env import ArenaEnv
from scripts.certify_local_goal_return import all_action_path_bound,certify


def fixture():
    n=4;x=np.zeros((n,1287),dtype=np.float32);x[:,1280]=1
    a=dict(first_free_input=x,goal_snapshot=np.zeros((n,512),dtype=np.float32),
        erased_state=np.zeros((n,512),dtype=np.float32),terminal_pos=np.zeros((n,2),dtype=np.float32),
        terminal_hd=np.arange(n)*2*np.pi/n,terminal_motors=np.tile([0.,0.,2.2],(n,1)),
        motor=np.zeros((30,3,n)),uniform=np.full((30,n),.4))
    cfg=dict(action_repeat=4,v_max=.35,arena_half=1.25,goal_radius=.3)
    return a,cfg,np.array([.60,0.])


def test_certificate_passes_reachable_distant_pair_but_not_short_gap():
    a,cfg,goal=fixture();r=certify(a,cfg,goal)
    assert r['certified_all_pairs'] and r['common_information_success_upper_bound']==.5
    assert all(abs(p['all_action_path_upper_bound_m']-.6236743746370297)<1e-14 for p in r['pairs'])
    short=certify(a,cfg,np.array([.45,0.]))
    assert not short['certified_all_pairs']
    assert all('path_shorter_than_both_goal_visits' in p['failures'] for p in short['pairs'])


@pytest.mark.parametrize('leak',('goal','position','motor','tape','random','state','heading','wall'))
def test_detects_input_and_symmetry_failures(leak):
    a,cfg,goal=fixture()
    if leak=='goal':a['first_free_input'][2,768]=1
    elif leak=='position':a['terminal_pos'][2,0]=.01
    elif leak=='motor':a['terminal_motors'][2,0]=.01
    elif leak=='tape':a['motor'][0,0,2]=1
    elif leak=='random':a['uniform'][0,2]=.9
    elif leak=='state':a['erased_state'][2,0]=.1
    elif leak=='heading':a['terminal_hd'][2]+=.2
    else:a['terminal_pos'][:,0]=.8;goal=goal+np.array([.8,0.])
    assert not certify(a,cfg,goal)['certified_all_pairs']


def test_nonzero_motor_bound_covers_all_short_native_action_sequences():
    tape=np.random.default_rng(19).normal(size=(4,3,1));v0=np.array([[.11,-.08]])
    bound=all_action_path_bound(tape,.35,.03,4/60,.25,v0)[0]
    targets=np.array([[0,0],[0,0],[.35,0],[-.35,0],[0,.35],[0,-.35]])
    alpha=math.exp(-(4/60)/.25)
    for sequence in itertools.product(range(6),repeat=4):
        v=v0[0].copy();length=0.
        for t,action in enumerate(sequence):
            v=alpha*v+(1-alpha)*(targets[action]+.03*tape[t,:2,0]);length+=(4/60)*np.linalg.norm(v)
        assert length<=bound+1e-14


def test_native_body_noise_coupling_preserves_mirrored_paths_before_hit():
    env=ArenaEnv(2,half=3.,goal_radius=.1,vision='none',img=1,stagger=False,seed=19,device='cpu')
    env.reset_all();env.pos.zero_();env.hd[:]=torch.tensor([.2,.2+math.pi]);env.goal[:]=torch.tensor([2.,2.])
    env.vf[:]=.11;env.vs[:]=-.08;env.vr[:]=1.1
    tape=torch.tensor(np.random.default_rng(20).normal(size=(30,3)),dtype=torch.float32);index=0
    def draw(*shape):
        nonlocal index
        value=tape[index//3,index%3].repeat(2);index+=1;return value
    env._randn=draw;length=0.;maximum=0.
    for t in range(30):
        before=env.pos.clone();_,_,_,info=env.step(torch.full((2,),t%6))
        assert not info['hit'].any() and not info['hit_wall'].any()
        assert torch.equal(env.vf[:1],env.vf[1:]) and torch.equal(env.vs[:1],env.vs[1:])
        maximum=max(maximum,float((env.pos[0]+env.pos[1]).norm()))
        length+=float((env.pos[0]-before[0]).norm())
    assert maximum<1e-5
    bound=all_action_path_bound(tape.numpy()[:,:,None],env.v_max,env.noise_v,env.dt,env.tau,np.array([[.11,-.08]]))[0]
    assert length<=bound+1e-6
