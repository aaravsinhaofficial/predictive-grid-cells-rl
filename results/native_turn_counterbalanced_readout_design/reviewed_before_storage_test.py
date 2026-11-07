from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import native_turn_counterbalanced_readout as study


def test_static_actor_uses_post_input_hidden_only():
    gen=np.random.default_rng(317);state=gen.normal(0,.1,(128,512))
    weight=gen.normal(0,.1,(6,256));bias=gen.normal(0,.1,6)
    logits,q=study.static_actor(state,weight,bias)
    expected=torch.nn.functional.linear(torch.from_numpy(state[:,:256]),torch.from_numpy(weight),torch.from_numpy(bias))
    np.testing.assert_allclose(logits,expected.numpy(),atol=1e-14)
    np.testing.assert_allclose(q,expected.softmax(1).numpy(),atol=1e-14)
    changed=state.copy();changed[:,256:]+=1e6
    assert np.array_equal(study.static_actor(changed,weight,bias)[1],q)


def test_shared_phase_match_uses_true_prefix_and_angles_not_row_parity():
    target=.31+np.arange(128)*2*np.pi/128
    indices=np.r_[np.arange(0,64,8),np.arange(64,128,8)]
    labels=np.r_[np.zeros(8,dtype=int),np.ones(8,dtype=int)]
    order=np.random.default_rng(4).permutation(16)
    cases,selected=study.shared_indices(target,target[indices][order],labels[order],1)
    assert np.array_equal(selected,indices[order][cases])
    bad=target[indices][order].copy();bad[cases[0]]+=.002
    with pytest.raises(AssertionError):study.shared_indices(target,bad,labels[order],1)


def terminal_fixture():
    n=128;endpoint=torch.tensor([.12,.25]);heading=torch.arange(n)*2*torch.pi/n
    physical=dict(pos=endpoint.expand(2,n,2).clone(),mdir=torch.tensor([.3,.9539392]).expand(2,n,2).clone(),
        t=torch.full((2,n),52,dtype=torch.long),vf=torch.ones(2,n),vs=torch.ones(2,n),vr=torch.ones(2,n))
    row=dict(updates=8,after_choice_brake=True,physical_position_error_m=0.,episode_t=[60]*n,heading=heading.tolist())
    family=dict(choice_heading=heading,target_choice_heading=heading,native_x=torch.zeros(2,n,1287),physical=physical,
        audit=dict(kind='turn_only',labels=['visible_0','turn_13'],reference_input_index=1,left_physical=[row]))
    return family,dict(endpoint=endpoint,goal_center=torch.tensor([.8,.3]))


def test_choice_terminal_uses_new_heading_clock_and_zero_brake_not_old_motors():
    family,context=terminal_fixture();terminal=study.terminal_from_family(family,context)
    assert torch.equal(terminal.pos,family['physical']['pos'][-1])
    assert torch.equal(terminal.mdir,family['physical']['mdir'][-1])
    assert torch.equal(terminal.hd,family['choice_heading'])and bool((terminal.t==60).all())
    assert torch.count_nonzero(terminal.vf)+torch.count_nonzero(terminal.vs)+torch.count_nonzero(terminal.vr)==0
    family['audit']['left_physical'][0]['episode_t'][0]=59
    with pytest.raises(AssertionError):study.terminal_from_family(family,context)


def test_full_circle_prefix_constant_policy_has_no_progress_and_zero_pair_tv():
    theta=np.arange(128)*2*np.pi/128
    physical=dict(progress_m=np.c_[np.zeros((128,2)),np.cos(theta),-np.cos(theta),np.sin(theta),-np.sin(theta)]*.005,
        body_goal_angle_error_reduction_rad=np.zeros((128,6)))
    q=np.repeat([[.1,.1,.3,.1,.2,.2]],128,axis=0)
    r=study.family_scores(q,physical)
    assert abs(r['progress_mm'])<1e-12 and r['mean_antipodal_policy_tv']==0
    assert abs(r['progress_above_average_policy_mm'])<1e-12
    altered=q.copy();altered[:,2]+=.05*np.cos(theta);altered[:,3]-=.05*np.cos(theta)
    assert study.family_scores(altered,physical)['progress_mm']>0


def test_numeric_provenance_rejects_probability_error_without_adaptive_tolerance():
    reference=np.repeat([[.1,.1,.3,.1,.2,.2]],8,axis=0);changed=reference.copy();changed[0,0]+=.001
    with pytest.raises(AssertionError):study.compare('probabilities',changed,reference,study.TOL['probability'])
    result=study.compare('float arithmetic',reference+1e-8,reference,study.TOL['probability'])
    assert result['max_abs']>0


def test_complete_cohort_aggregation_refuses_missing_or_duplicated_contexts():
    rows=[]
    fields=('progress_mm','average_policy_progress_mm','progress_above_average_policy_mm','best_common_action_progress_mm',
        'mean_antipodal_policy_tv','body_goal_alignment_rad','turn_weighted_body_goal_alignment_rad')
    for c in study.CONDITIONS:
        for e in study.ENDPOINTS:
            for k in study.KINDS:
                for b in study.BRANCHES:
                    for i in range(16):
                        for p in (0,1):rows.append(dict(condition=c,endpoint=e,kind=k,branch=b,context=i,prefix=p,shared_angle_checks_passed=True,**{f:float(p)for f in fields}))
    totals=study.aggregate_rows(rows);assert len(totals)==144
    assert all(r['progress_mm']==.5 and r['cases']==4096 for r in totals if r['prefix']=='both')
    with pytest.raises(AssertionError):study.aggregate_rows(rows[:-1])
    with pytest.raises(AssertionError):study.aggregate_rows(rows+[rows[0]])
