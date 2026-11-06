"""Synthetic first-action equations and saved-data guards; no learned outcomes."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_local_return_action_use as action

CFG=dict(action_repeat=4,v_max=.35,rot_max=2.2,arena_half=1.25,goal_radius=.3,vision='none')


def test_six_counterfactuals_match_native_physics_with_noise_inertia_and_walls():
    position=np.array([[0.,0.],[1.169,-.8],[-1.169,1.169],[.3,-.4]])
    heading=np.array([0.,3.13,-3.13,1.5]);motors=np.array([[0.,0.,2.2],[.3,.2,-.8],[-.3,.3,1.],[.1,-.2,.5]])
    noise=np.array([[.3,-.4,.2],[-.2,.5,-.7],[.8,.1,.4],[0.,-.8,-.2]])
    endpoint,angle,next_motors=action.first_action_physics(position,heading,motors,noise,CFG)
    env=action.task.TrialArena(CFG,24,'cpu',vision='none');env.no_respawn=True
    env.pos[:]=torch.tensor(np.repeat(position,6,axis=0),dtype=torch.float32)
    env.hd[:]=torch.tensor(np.repeat(heading,6),dtype=torch.float32)
    actual_motors=torch.tensor(np.repeat(motors,6,axis=0),dtype=torch.float32)
    for i,key in enumerate(('vf','vs','vr')):getattr(env,key)[:]=actual_motors[:,i]
    env.goal.fill_(10.);env.set_motor(torch.tensor(np.repeat(noise,6,axis=0).T,dtype=torch.float32))
    env.step(torch.arange(6).repeat(4))
    np.testing.assert_allclose(endpoint.reshape(-1,2),env.pos.numpy(),atol=action.POSITION_ATOL,rtol=0)
    np.testing.assert_allclose(next_motors.reshape(-1,3),torch.stack((env.vf,env.vs,env.vr),1).numpy(),atol=2e-7,rtol=0)
    np.testing.assert_allclose(angle.ravel(),env.hd.numpy(),atol=5e-7,rtol=0)


def test_probabilities_are_literal_and_six_action_bounds_are_preserved():
    position=np.zeros((2,2));goal=np.tile([1.,0.],(2,1))
    endpoints=np.tile(np.array([[.01,0.],[-.02,0.],[0.,0.],[.03,0.],[.01,.02],[-.01,.02]]),(2,1,1))
    p=np.array([[0,0,0,1,0,0],[1/6]*6],dtype=float)
    values=action.action_metrics(position,goal,endpoints,p)
    assert values['expected_goalward_progress_m'][0]==values['best_six_same_noise_progress_m'][0]
    assert values['expected_goalward_progress_m'][1]==pytest.approx(values['uniform_goalward_progress_m'][1])
    p[1]*=(1+1e-7)
    literal=action.action_metrics(position,goal,endpoints,p)
    assert literal['probability_sum'][1]>1
    assert literal['expected_goalward_progress_m'][1]==pytest.approx(values['uniform_goalward_progress_m'][1]*(1+1e-7),abs=1e-15)
    p[1,0]=-.1
    with pytest.raises(AssertionError):action.action_metrics(position,goal,endpoints,p)


def fixture():
    n=16;variants=action.VARIANTS;groups=len(variants)
    plan=dict(heading=torch.arange(n)*2*torch.pi/n,context=torch.full((n,),3),
        duration_a=torch.full((n,),5),duration_b=torch.full((n,),7),delay=torch.full((n,),60),
        turn_a=torch.zeros(n,dtype=torch.long),turn_b=torch.ones(n,dtype=torch.long),
        goal=torch.tensor([[.6,0.]]).repeat(n,1),motor=torch.zeros(160,3,n))
    start=int(action.task.schedule(plan,'cpu')['free'][0]);plan['motor'][start]=torch.tensor([.1,-.2,.3])[:,None]
    heading=plan['heading'].numpy();position=np.zeros((n,2));motors=np.tile([0.,0.,2.2],(n,1))
    endpoints=action.first_action_physics(position,heading,motors,plan['motor'][start].T.numpy(),CFG)[0]
    rows=[];first={};raw=dict(policy_mask=torch.zeros(160,n*groups,dtype=torch.bool),
        active=torch.ones(160,n*groups,dtype=torch.bool),action=torch.zeros(160,n*groups,dtype=torch.long),
        position=torch.zeros(160,n*groups,2))
    raw['policy_mask'][start:start+30]=True
    for j in range(n*groups):
        h=j%n;chosen=h%6;p=torch.nn.functional.one_hot(torch.tensor(chosen),6).float()
        first[j]=dict(position=torch.tensor(position[h],dtype=torch.float32),heading=plan['heading'][h],
            motors=torch.tensor(motors[h],dtype=torch.float32),probabilities=p)
        raw['action'][start,j]=chosen;raw['position'][start,j]=torch.tensor(endpoints[h,chosen],dtype=torch.float32)
        rows.append(dict(context=3,heading_index=h,variant=variants[j//n],first_free_probabilities=p.tolist(),
            hit=False,free_steps=30,progress_m=.01,free_path_m=.2))
    raw['first_free']=first
    return raw,rows,plan,start


def test_every_saved_chosen_action_is_reconciled_at_actual_free_clock():
    raw,rows,plan,start=fixture()
    result=action.analyze_context(raw,rows,plan,CFG)
    assert len(result)==192 and all(r['first_free_global_step']==start for r in result)
    assert max(r['chosen_position_max_abs_error_m'] for r in result)<action.POSITION_ATOL
    assert result[0]['saved_motor_noise']==pytest.approx([.1,-.2,.3])
    assert all(r['recorded_30_action_progress_m']==.01 for r in result)
    raw['position'][start,191,0]+=.001
    with pytest.raises(AssertionError,match='native-physics disagreement'):
        action.analyze_context(raw,rows,plan,CFG)


def test_controller_override_is_used_and_physical_cue_mismatch_rejected():
    raw,rows,plan,start=fixture()
    override=[dict(r,first_free_probabilities=[1/6]*6) for r in rows[:16]]
    small={k:(v[:,:16] if isinstance(v,torch.Tensor) else v) for k,v in raw.items() if k!='first_free'}
    first={i:raw['first_free'][i] for i in range(16)}
    result=action.analyze_context(small,override,plan,CFG,('blind_native',),first,'uniform')
    assert all(r['controller']=='uniform' for r in result)
    assert all(abs(r['expected_minus_uniform_progress_m'])<1e-15 for r in result)
    bad=deepcopy(raw);bad['first_free'][17]['motors'][2]+=.01
    with pytest.raises(AssertionError):action.analyze_context(bad,rows,plan,CFG)


def test_failed_context_rows_remain_undefined_and_duplicates_fail():
    raw,rows,plan,_=fixture();good=action.analyze_context(raw,rows,plan,CFG)[:16]
    values=action.summarize(good,[3]);assert values['valid_trials']==16
    failed=action.invalid_rows([dict(r,hit=False) for r in rows[:16]],'policy','synthetic failure')
    undefined=action.summarize(failed,[3])
    assert undefined['trials']==undefined['undefined_trials']==16 and undefined['valid_trials']==0
    assert all(v is None for v in undefined['means_over_valid_trials'].values())
    duplicate=deepcopy(good);duplicate[-1]=deepcopy(duplicate[0])
    with pytest.raises(AssertionError):action.summarize(duplicate,[3])


def test_production_requires_both_complete_final_audits():
    reports={c:dict(condition=c,passed=True,complete=True,status='complete',failures=[],pending=[],
        checks=dict(final_exact_stage4=True,final_summary=True)) for c in action.CONDITIONS}
    action.require_complete_pair(reports)
    bad=deepcopy(reports);bad['pgc33']['complete']=False
    with pytest.raises(AssertionError):action.require_complete_pair(bad)
    with pytest.raises(AssertionError):action.require_complete_pair({'gc100':reports['gc100']})
