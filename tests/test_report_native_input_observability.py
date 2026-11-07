"""Saved-array corruption controls; no trained query or trajectory."""
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import native_input_observability as study
import report_native_input_observability as report
from test_native_input_observability_review import fixture
from test_native_input_observability import SyntheticQuery
from test_native_turn_geometry_controls import two_section_system


def test_probability_units_and_independent_geometry_gates():
    first,_,_=fixture();first=first[:128]
    for last in (first,first+.01,first.mean(0)+(first-first.mean(0))*.01):
        # Offset is paired between output coordinates to preserve probabilities.
        if np.all(last==first+.01):
            last=first.copy();last[:,::6]+=.01;last[:,1::6]-=.01
        assert report.independent_probability_gates(first,last)==study.geometry(first,last)['gates']
    bad=first.copy();bad[0,0]+=.01
    with pytest.raises(AssertionError):report.probability_audit(bad)


def test_alias_includes_periodic_section_and_horizon():
    canonical=dict(family='base',context=0,prefix=0,kind='turn_only',branch='native',exact_sham_of=None,
        regime='fb_B',retention_updates=2,physical_valid=True,meaningful='B')
    alias=dict(canonical,family='alias',branch='ordinary',exact_sham_of='base')
    report.audit_alias(alias,canonical)
    with pytest.raises(AssertionError):report.audit_alias(alias,dict(canonical,regime='fb_F'))
    with pytest.raises(AssertionError):report.audit_alias(dict(alias,meaningful='F'),canonical)


def test_status_preserves_visibility_geometry_and_inconclusive_fibers():
    gates={report.INITIAL:True,report.EVOLVED:True,'shape':True}
    row=dict(physical_valid=True,primary_geometry={'gates':gates})
    probe=dict(physical_construction_conclusive=True,initial_observation_conclusive=True,all_normal_directions_pass=True)
    closure=dict(passes=True,checks={'a':dict(unresolved_interpolation_cells=0)})
    metric=dict(nominal=probe,bounded_followups=[],heldout_closure=closure)
    conditional=dict(metrics={'raw':copy.deepcopy(metric),'h_c_RMS':copy.deepcopy(metric)},heldout_physics={'0':{'valid':True}},broad_candidate=True)
    assert report.status(row,conditional)=='observed_candidate'
    gates['shape']=False;assert report.status(row)=='geometry_failed'
    gates['shape']=True;gates[report.EVOLVED]=False;assert report.status(row)=='evolved_probability_below_floor'
    gates[report.EVOLVED]=True;assert report.status(row)=='conditional_pending'
    conditional['metrics']['raw']['nominal']['initial_observation_conclusive']=False
    assert report.status(row,conditional)=='transverse_observation_inconclusive'
    assert report.status(None)=='screen_pending'
    assert report.status(dict(physical_valid=False))=='physics_unavailable'


def test_heldout_independent_label_and_error_arithmetic():
    first,psi,metadata=fixture(invisible=True)
    changed=psi.copy();changed[128,3,0]+=.001;changed[128,3,1]-=.001
    for changed in (psi,changed):
        expected,ea=study.heldout_closure(first,first,psi,changed,metadata)
        actual,aa=report.closure_audit(first,first,psi,changed,metadata)
        report.compare(actual,expected)
        assert set(aa)==set(ea)
        for name in aa:np.testing.assert_allclose(aa[name],ea[name],atol=2e-10,rtol=2e-8)


def test_saved_probe_audit_catches_direction_and_gain_corruption(tmp_path):
    curve,_,forward,backward,_,_=two_section_system();curve=curve.numpy();obs=SyntheticQuery()
    transition=lambda z,t:(forward if t%2==0 else backward)(z)
    for method in (None,'shared','clip_equalize'):
        result,arrays,_=study.finite_probe(transition,curve,np.ones(4),obs,2,method)
        name=f'{method}.npz';np.savez_compressed(tmp_path/name,**arrays)
        report.audit_probe(tmp_path,result,name,'raw',arrays['first'][:128],arrays['last'][:128],curve,arrays['final_states'][:128])
    bad={k:v.copy()for k,v in arrays.items()};bad['normal_gain'][0,0,0]+=.1
    np.savez_compressed(tmp_path/'bad.npz',**bad)
    with pytest.raises(AssertionError):report.audit_probe(tmp_path,result,'bad.npz','raw',arrays['first'][:128],arrays['last'][:128],curve,arrays['final_states'][:128])


def test_physical_query_hash_requires_the_complete_truthful_action_bank():
    bank=study.query_bank(torch.zeros(1287));sequences=[[a]for a in range(6)]
    audit=dict(valid=True,issues=[],phases=128,measurement_updates=1,queries=sequences,
        input_sha256=[study.tensor_sha(bank[torch.arange(6).repeat(128)])])
    report.physical_tape_metadata(audit,bank,128)
    changed=bank.clone();changed[2,1282]=0
    with pytest.raises(AssertionError):report.physical_tape_metadata(audit,changed,128)
    corrupt=copy.deepcopy(audit);corrupt['issues']=[{'reason':'goal_hit'}]
    with pytest.raises(AssertionError):report.physical_tape_metadata(corrupt,bank,128)
    missing=copy.deepcopy(audit);missing['input_sha256']=[]
    with pytest.raises(AssertionError):report.physical_tape_metadata(missing,bank,128)
    relabeled=copy.deepcopy(audit);relabeled['queries'][2]=[1]
    with pytest.raises(AssertionError):report.physical_tape_metadata(relabeled,bank,128)


def test_static_saved_logit_probability_consistency_rejects_corruption():
    first,_,_=fixture();p=first[:128];logit=np.log(p.reshape(-1,6,6))
    logit-=logit.mean(-1,keepdims=True);logit=logit.reshape(-1,36)
    report.static_softmax_audit(p,logit)
    wrong=p.copy();wrong[17,0]+=.001;wrong[17,1]-=.001
    with pytest.raises(AssertionError):report.static_softmax_audit(wrong,logit)
