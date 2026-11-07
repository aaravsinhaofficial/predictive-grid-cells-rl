from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import decompose_native_turn_prefix as study


def test_prefix_labels_use_actual_actions_not_case_parity():
    labels=np.array([0]*8+[1]*8);pa=np.repeat(labels[None],14,axis=0)
    x=torch.zeros(14,16,1287);x[:,:,1280:1286]=torch.from_numpy(np.eye(6,dtype=np.float32)[pa])
    raw=dict(labels=[f'turn_{i}'for i in range(14)],previous_actions=torch.from_numpy(pa),teacher_x=x,student_native_x=x.clone(),student_ordinary_x=x.clone())
    actual,record=study.prefix_labels(raw);assert np.array_equal(actual,labels)
    order=np.array([3,12,5,1,15,9,2,14,6,0,10,7,4,13,11,8])
    shuffled={k:(v[:,order]if isinstance(v,torch.Tensor)else v)for k,v in raw.items()}
    assert np.array_equal(study.prefix_labels(shuffled)[0],labels[order])
    raw['student_native_x'][2,1,1280:1286]=0
    with pytest.raises(AssertionError):study.prefix_labels(raw)


def fixture():
    labels=np.array([0,0,1,1]);p=np.array([[.6,.1,.1,.1,.05,.05],[.1,.6,.1,.1,.05,.05],[.1,.1,.6,.1,.05,.05],[.1,.1,.1,.6,.05,.05]])
    physics=np.array([[1,2,-1,-2,0,0],[-1,-2,1,2,0,0],[2,-1,-2,1,0,0],[-2,1,2,-1,0,0]])*.001
    return p,labels,physics


def test_prefix_only_policy_has_zero_within_prefix_effect():
    p,labels,physics=fixture();q=p.copy()
    for k in (0,1):q[labels==k]=p[labels==k].mean(0)
    r=study.prefix_decomposition(p,q,labels,physics)
    assert abs(r['within_prefix_gain'])<1e-14 and abs(r['within_prefix_progress_mm'])<1e-14


def test_decompositions_are_exact_and_do_not_drop_negative_parts():
    p,labels,physics=fixture();r=study.prefix_decomposition(p,p[::-1],labels,physics)
    assert abs(r['total_history_gain']-r['prefix_associated_gain']-r['within_prefix_gain'])<1e-14
    assert abs(r['total_progress_advantage_mm']-r['prefix_associated_progress_mm']-r['within_prefix_progress_mm'])<1e-14
    assert r['total_history_gain']<0


def test_identical_choice_policy_has_zero_gain_at_both_averaging_levels():
    p,labels,physics=fixture();q=np.repeat(np.array([[.1,.2,.3,.2,.1,.1]]),4,axis=0)
    r=study.prefix_decomposition(p,q,labels,physics)
    for field in ('total_history_gain','prefix_associated_gain','within_prefix_gain','total_progress_advantage_mm','prefix_associated_progress_mm','within_prefix_progress_mm'):
        assert abs(r[field])<1e-14
