"""Focused synthetic physical/input checks; no original-policy outcome."""
from pathlib import Path
import sys
import math
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import native_movement_cycles as cycle


def test_preteleport_observer_does_not_change_native_physics_or_rng():
    cfg=dict(arena_half=1.25,episode_len=8,action_repeat=4,v_max=.35,rot_max=2.2,goal_radius=.3,
        vision='raycast',img_noise=.02,randomize_textures=1)
    native=cycle.MatchedArena(cfg,2,1,11223,'cpu');trace=cycle.TraceArena(cfg,2,1,11223,'cpu')
    a=native.reset_all();b=trace.reset_all()
    for e in (native,trace):e.goal[0]=e.pos[0]
    for step in range(8):
        trace.trace_first=None;action=torch.tensor([step%6,(step+2)%6])
        a=native.step(action);b=trace.step(action)
        for key in a[0]:assert torch.equal(a[0][key],b[0][key])
        assert torch.equal(a[1],b[1])and torch.equal(a[2],b[2])
        for key in cycle.ENV_KEYS:assert torch.equal(getattr(native,key),getattr(trace,key)),key
        assert torch.equal(native.gen.get_state(),trace.gen.get_state())
        assert torch.equal(trace.trace_first['pos'],b[3]['hit_pos'])
        if step==0:assert bool(b[3]['hit'][0])and not torch.equal(trace.trace_first['pos'][0],trace.pos[0])


def test_cycle_census_respects_hit_boundaries_and_missing_turns():
    T,n=100,2;theta=torch.arange(T+1)*(.1*math.pi)
    prehd=cycle.wrap_pi(theta[:-1])[:,None].expand(T,n).clone()
    posthd=cycle.wrap_pi(theta[1:])[:,None].expand(T,n).clone()
    # Slot1 is stationary: it must remain in every episode denominator.
    prehd[:,1]=posthd[:,1]=0
    pos=torch.zeros(T,n,2);goal=pos.clone();goal[:,:,0]=1
    direction=torch.zeros(T,n,2);direction[:,:,0]=1
    pre=dict(pos=pos,goal=goal,hd=prehd,mdir=direction,n_goals=torch.ones(T,n,dtype=torch.long))
    post=dict(pos=pos.clone(),hd=posthd,mdir=direction)
    hit=torch.zeros(T,n,dtype=torch.bool);hit[49,0]=True;hit[-1]=True
    done=torch.zeros_like(hit);done[-1]=True
    state=torch.zeros(T+1,n,512);state[:,:,0]=theta.cos()[:,None];state[:,:,1]=theta.sin()[:,None]
    data=dict(pre=pre,post=post,state_before=state[:-1],state_after=state[1:],hit=hit,done=done,
        actor_weight=torch.zeros(6,256),actor_bias=torch.zeros(6),config={},dt=1/15)
    result=cycle.census(data)
    assert result['episodes']==2 and result['episodes_with_heading_turns']==1
    assert result['total_heading_turns']==4
    assert len(result['segments'])==3 and result['return_segments_with_heading_turns']==2
    assert result['return_segments_with_repeated_turns']==2
    assert not any(c['start_step']<50<=c['end_step']for c in result['cycles'])
    assert result['physical_usefulness_descriptors']['without_fullturn']['segments']==1


def test_grid_rescue_is_exact_identity_and_goal_slots_stay_literal():
    class Net:
        def __init__(self):self.lstm=torch.nn.LSTMCell(1287,256)
    net=Net();x=torch.zeros(2,1287);state=(torch.zeros(2,256),torch.zeros(2,256))
    expected=net.lstm(x,state)
    for mode in ('native','current_grid','current_candidate_output_rescue'):
        actual=cycle.intervention_step(net,x,x.clone(),state,mode)
        assert all(torch.equal(a,b)for a,b in zip(actual,expected))
