"""Arithmetic and saved-input grouping tests; no trained model execution."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import decompose_native_turn_imitation as study


def distributions():
    return np.array([[.75,.05,.05,.05,.05,.05],[.05,.75,.05,.05,.05,.05]])


def test_exact_chain_rule_and_positive_zero_negative_history_gain():
    p=distributions()
    perfect=study.decomposition(p,p)
    assert perfect['observed_kl']==0 and abs(perfect['history_gain']-perfect['teacher_floor'])<1e-14
    common=np.broadcast_to(np.array([.2,.4,.1,.1,.1,.1]),p.shape)
    flat=study.decomposition(p,common)
    assert flat['student_identical_within_group']and abs(flat['history_gain'])<1e-14
    reversed=study.decomposition(p,p[::-1])
    assert reversed['history_gain']<0 and reversed['conditional_case_mismatch']>reversed['teacher_floor']


def test_teacher_zeros_are_exact_and_student_zeros_are_not_silently_clamped():
    p=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]],float)
    r=study.decomposition(p,np.full((2,6),1/6))
    assert abs(r['teacher_floor']-np.log(2))<1e-14
    with pytest.raises(AssertionError):study.decomposition(p,p)


def test_groups_are_exact_inputs_and_totals_case_weighted():
    inputs=np.array([[1.,2.],[1.,2.],[1.,2.000001]])
    p=np.vstack((distributions(),np.full(6,1/6)));q=np.full((3,6),1/6)
    refs=[{'case':i}for i in range(3)]
    rows,mean=study.group_cases(inputs,p,q,refs)
    assert len(rows)==2 and sorted(r['cases']for r in rows)==[1,2]
    assert abs(mean['teacher_floor']-2/3*study.decomposition(p[:2],q[:2])['teacher_floor'])<1e-14
    assert abs(mean['observed_kl']-study.kl(p,q).mean())<1e-14


def test_original_to_final_improvement_is_exact_with_signed_contributions():
    p=distributions();a=study.decomposition(p,np.array([[.1,.5,.1,.1,.1,.1],[.5,.1,.1,.1,.1,.1]]))
    b=study.decomposition(p,np.broadcast_to(p.mean(0),p.shape))
    a['groups']=b['groups']=1
    change=study.improvement(a,b)
    assert abs(change['kl_improvement']-change['mean_policy_improvement']-change['history_gain_change'])<1e-14
    assert change['history_gain_change']>0
    bad=dict(b,teacher_floor=b['teacher_floor']+.01)
    with pytest.raises(AssertionError):study.improvement(a,bad)


def test_probability_row_scale_does_not_change_decomposition():
    p=distributions();q=np.array([[.4,.2,.1,.1,.1,.1],[.2,.4,.1,.1,.1,.1]])
    a=study.decomposition(p,q);b=study.decomposition(p*np.array([[3],[7]]),q*np.array([[2],[9]]))
    for key in study.FIELDS:assert abs(a[key]-b[key])<1e-14


def test_shared_target_signature_keeps_teacher_and_membership_not_student_outcome():
    a={'probe_glimpse':{'cases':2,'records':[{'input_sha256':'fixed','cases':[0,1],
        'optimal_common_action_distribution':[.5,.5],'variants':{'native':{'kl':2.}}}]}}
    b={'probe_glimpse':{'cases':2,'records':[dict(a['probe_glimpse']['records'][0],variants={'native':{'kl':1.}})]}}
    assert study.teacher_groups(a)==study.teacher_groups(b)
    b['probe_glimpse']['records'][0]['cases']=[1,0]
    assert study.teacher_groups(a)!=study.teacher_groups(b)
