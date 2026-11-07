from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_native_turn_counterbalanced_readout as audit
import report_native_turn_counterbalanced_readout as report


def fixture():
    theta=np.arange(128)*2*np.pi/128
    physics=np.c_[np.zeros((128,2)),np.cos(theta),-np.cos(theta),np.sin(theta),-np.sin(theta)]*.005
    q=np.repeat([[.1,.1,.3,.1,.2,.2]],128,axis=0)
    return q,physics,np.zeros((128,6))


def test_raw_arithmetic_and_zero_history_policy():
    q,p,a=fixture();metrics,arrays=audit.recompute(q,p,a)
    assert abs(metrics['progress_mm'])<1e-12 and metrics['mean_antipodal_policy_tv']==0
    assert np.max(np.abs(arrays['case_progress_mm']-1000*(q*p).sum(1)))<1e-14
    bad=q.copy();bad[0,0]+=.01
    with pytest.raises(AssertionError):audit.recompute(bad,p,a)


def test_controlled_first_action_physics_has_zero_turn_translation_and_antipodal_symmetry():
    hd=np.arange(128)*2*np.pi/128
    terminal=dict(pos=np.zeros((128,2)),goal=np.repeat([[.8,0]],128,axis=0),hd=hd,
        vf=np.zeros(128),vs=np.zeros(128),vr=np.zeros(128),t=np.full(128,100))
    cfg=dict(action_repeat=4,v_max=.35,rot_max=2.2,arena_half=1.25,goal_radius=.3,episode_len=1350)
    p=audit.physical_expected(terminal,cfg)
    assert np.max(np.abs(p['progress_m'][:,:2]))==0
    assert np.max(np.abs(p['progress_m'][:64]+p['progress_m'][64:]))<1e-15
    terminal['vf'][0]=.01
    with pytest.raises(AssertionError):audit.physical_expected(terminal,cfg)


def test_failed_equivalence_cannot_be_promoted_or_omit_failure():
    failures=[dict(condition='gc100',endpoint=e,context=11,prefix=1,kind='probe_glimpse',branch=b,reason='prefix embedding')for e in audit.ENDPOINTS for b in audit.BRANCHES]
    assert len(audit.rejected_equivalence_cohort(dict(complete=False,failures=failures)))==12
    with pytest.raises(AssertionError):audit.rejected_equivalence_cohort(dict(complete=True,failures=failures))
    with pytest.raises(AssertionError):audit.rejected_equivalence_cohort(dict(complete=False,failures=failures[:-1]))


def test_complete_cohort_does_not_select_only_compatible_cases():
    rows=[dict(condition=c,endpoint=e,context=i,prefix=p,kind=k,branch=b)for c in audit.CONDITIONS for e in audit.ENDPOINTS for i in range(16)for p in (0,1)for k in audit.KINDS for b in audit.BRANCHES]
    assert len(audit.coverage(rows))==1536
    with pytest.raises(AssertionError):audit.coverage(rows[:-1])
    with pytest.raises(AssertionError):audit.coverage(rows+[rows[0]])


def test_old_assignment_can_look_positive_while_each_full_circle_mean_is_zero():
    phase=np.arange(128)*2*np.pi/128
    result=report.assignment_comparison(np.sin(phase),-np.sin(phase))
    assert result['old_assignment_dense_mm']>.6 and abs(result['balanced_mm'])<1e-15
    assert abs(result['assignment_minus_balanced_mm']-result['old_assignment_dense_mm'])<1e-15


def test_assignment_keeps_sign_and_refuses_missing_angles():
    values=np.linspace(-2,1,128);result=report.assignment_comparison(values,values)
    assert result['balanced_mm']<0 and abs(result['assignment_minus_balanced_mm'])<1e-15
    with pytest.raises(AssertionError):report.assignment_comparison(values[:-1],values)


def test_old_assignment_is_verified_from_full_phase_order_and_actual_prefix_labels():
    target=.31+np.arange(128)*2*np.pi/128;ix=np.r_[np.arange(0,64,8),np.arange(64,128,8)]
    labels=np.r_[np.zeros(8,dtype=int),np.ones(8,dtype=int)];order=np.random.default_rng(42).permutation(16)
    cases,matched=audit.verified_old_assignment_indices(target,.31,target[ix][order],labels[order],1)
    assert np.array_equal(matched,ix[order][cases])
    with pytest.raises(AssertionError):audit.verified_old_assignment_indices(np.roll(target,1),.31,target[ix],labels,0)
    with pytest.raises(AssertionError):audit.verified_old_assignment_indices(target,.31,target[ix],1-labels,0)
