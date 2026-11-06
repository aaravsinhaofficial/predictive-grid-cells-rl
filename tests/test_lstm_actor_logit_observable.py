import math
import numpy as np
import pytest
import torch
from scripts import lstm_actor_logit_observable as adapter
from tests.test_lstm_reorientation_entropy_mechanism import metadata


def actor():
    w=torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,0.],[0.,0.,0.]],dtype=torch.float64)
    return w,torch.zeros(6,dtype=torch.float64)


def family():
    t=np.arange(192)*2*np.pi/192
    return np.column_stack((np.cos(t),np.sin(t),.5*np.cos(2*t)))


def test_exact_learned_logits_remove_only_action_common_score():
    rng=torch.Generator().manual_seed(12)
    z=torch.randn(20,8,generator=rng,dtype=torch.float64)
    w=torch.randn(6,4,generator=rng,dtype=torch.float64);b=torch.randn(6,generator=rng,dtype=torch.float64)
    logits=z[:,:4]@w.T+b;q=torch.eye(6,dtype=torch.float64)-torch.ones(6,6,dtype=torch.float64)/6
    out=adapter.centered_actor_logits(z,w,b)
    torch.testing.assert_close(out,logits@q,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(out.softmax(1),logits.softmax(1),atol=1e-14,rtol=1e-14)
    same=adapter.centered_actor_logits(z,w+torch.arange(4,dtype=torch.float64)[None,:],b+17)
    torch.testing.assert_close(out,same,atol=1e-13,rtol=1e-13)
    assert out.shape==(20,6)


def test_actor_null_phase_dependent_spectator_removed_without_fitted_map():
    def step(z):
        r=z[:,:2].norm(dim=1,keepdim=True);xy=z[:,:2]*(.8+.2/r)
        x=xy[:,0]*math.cos(.04)-xy[:,1]*math.sin(.04)
        y=xy[:,0]*math.sin(.04)+xy[:,1]*math.cos(.04)
        return torch.stack((x,y,1.02*z[:,2]),1)
    w,b=actor();out=lambda z:adapter.centered_actor_logits(z,w,b)
    score=adapter.observable_probe(step,family(),np.ones(3),out,count=192)
    hidden=adapter.observable_probe(step,family(),np.ones(3),lambda z:z,count=192)
    assert score['observable_dimensions']==24 and score['centered_conditional_factor_candidate']
    assert not hidden['centered_conditional_factor_candidate']
    assert not hidden['candidate_gates']['centered']['p90_invariance_residual_below5percent']


def test_contracting_cartesian_memory_and_vanishing_scores_do_not_pass():
    w,b=actor();curve=family();curve[:,2]=0
    point=adapter.observable_probe(lambda z:.98*z,curve,np.ones(3),lambda z:adapter.centered_actor_logits(z,w,b),count=192)
    assert not point['centered_conditional_factor_candidate']
    assert .9<point['radial_over_tangent_observable_gain']['median']<1.1
    zero=adapter.observable_probe(lambda z:z,curve,np.ones(3),lambda z:adapter.centered_actor_logits(z,w*0,b),count=192)
    assert not zero['centered_conditional_factor_candidate']
    assert not zero['candidate_gates']['centered']['observable_radial_kick_nontrivial_in80percent']
    assert not zero['candidate_gates']['centered']['observable_radius_RMS_at_least1e_minus4']


def test_exact_existing_cohort_counts_and_no_missing_pilot_context_invention():
    cases=adapter.cases();counts={cohort:sum(2 if row['cohort']=='original1B' else 8 for row in cases if row['cohort']==cohort) for cohort in ('original1B','forty','ninety')}
    assert counts=={'original1B':56,'forty':32,'ninety':64} and sum(counts.values())==152
    assert len({(r['condition'],r['seed']) for r in cases if r['cohort']=='original1B'})==16
    assert all(r['context']==0 for r in cases if r['subgroup']=='pilot')


def test_genuine90m_metadata_not_relabelled_as40m_or_wrong_entropy():
    data=metadata('pgc33','explore');cfg=data['config']
    case=dict(cohort='ninety',condition='pgc33',arm='explore')
    adapter.validate_metadata(data,cfg,case)
    with pytest.raises(AssertionError):adapter.validate_metadata(data,cfg,dict(case,arm='original'))
    data['extra_frames']=40_058_880
    with pytest.raises(AssertionError):adapter.validate_metadata(data,cfg,case)
    assert data['extra_frames']==40_058_880
