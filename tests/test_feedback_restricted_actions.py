"""Analytic bound checked against exhaustive synthetic native-equation controls."""
import itertools,math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from analyze_feedback_restricted_actions import bound
CFG=dict(action_repeat=4,v_max=.35,rot_max=2.2,goal_radius=.03,arena_half=10.)


def trajectory(actions,noise,heading=.4,motors=(0.,0.,1.)):
    dt=4/60;a=math.exp(-dt/.25);m=np.array(motors,dtype=float);p=np.zeros(2);result=[]
    for action,draw in zip(actions,noise):
        target=np.array([0.,0.,-2.2]) if action==1 else np.array([0.,-.35,0.])
        m=a*m+(1-a)*(target+draw*np.array([.03,.03,.15]))
        heading+=m[2]*dt
        p+=dt*np.array([m[0]*math.cos(heading)-m[1]*math.sin(heading),m[0]*math.sin(heading)+m[1]*math.cos(heading)])
        result.append(p.copy())
    return np.array(result)


def test_exhaustive_switches_and_every_preterminal_hit_are_never_excluded():
    n=9;noise=np.stack([np.sin(np.arange(n)),np.cos(np.arange(n)*.7),np.sin(np.arange(n)*1.3)],1)*1.7
    for actions in itertools.product((1,5),repeat=n):
        positions=trajectory(actions,noise)
        for goal in positions[2::2]:
            assert bound([0.,0.],.4,[0.,0.,1.],goal,noise,CFG)['possible']


def test_initial_rotation_sign_and_walls_retain_unknown_cases():
    tape=np.zeros((30,3))
    r=bound([0,0],0,[0,0,-1],[.6,0],tape,CFG)
    assert r['possible'] and 'premise failed' in r['reason']
    r=bound([.9,0],0,[0,0,1],[.3,0],tape,dict(CFG,arena_half=1.25))
    assert r['possible'] and 'wall-free' in r['reason']


def test_known_opposite_sector_is_excluded_but_rightward_target_retained():
    cfg=dict(CFG,goal_radius=.3)
    toward=bound([0,0],0,[0,0,1],[0,-.6],np.zeros((30,3)),cfg)
    opposite=bound([0,0],0,[0,0,1],[0,.6],np.zeros((30,3)),cfg)
    assert toward['possible'] and not opposite['possible']
    assert opposite['angle_exclusion_margin_rad']>0


def test_large_noise_keeps_possible_when_premises_do_not_constrain():
    result=bound([0,0],0,[0,0,1],[.6,0],np.ones((30,3))*100,CFG)
    assert result['possible'] and result['reason']!='Certified native-equation angle sector'
