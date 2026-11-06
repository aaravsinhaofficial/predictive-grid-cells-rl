import numpy as np
from scripts.lstm_reorientation_actor_readout import decompose,softmax


def fixture():
    theta=np.arange(96)*2*np.pi/96
    h=np.zeros((96,8));h[:,0]=np.cos(theta);h[:,1]=np.sin(theta)
    w=np.zeros((6,8));w[0,0]=1;w[1,0]=-1;w[2,1]=1;w[3,1]=-1
    return h,w,np.zeros(6),theta


def test_exact_actor_reconstruction_and_rowspace_decomposition():
    h,w,b,theta=fixture();h[:,7]=.4*np.cos(2*theta)
    r=decompose(h,w,b,theta)
    assert r['reconstruction_max_abs_error']<1e-12
    np.testing.assert_allclose(r['actor_rowspace_hidden_variance_fraction'],1/1.08)
    np.testing.assert_allclose(r['direction_first_harmonic']['fraction_centered_logit_variance'],1)
    assert r['actor_rank']==2


def test_action_common_weight_and_bias_shifts_leave_actor_relevant_results_unchanged():
    h,w,b,theta=fixture();r=decompose(h,w,b,theta)
    shared=np.arange(8)*.3
    shifted=decompose(h,w+shared,b+7,theta)
    for key in ('actor_rowspace_hidden_variance_fraction','centered_logit_family_rms','hidden_to_centered_logit_gain'):
        np.testing.assert_allclose(r[key],shifted[key],atol=1e-12)
    np.testing.assert_allclose(r['probabilities']['mean_across_phases'],shifted['probabilities']['mean_across_phases'],atol=1e-14)
    np.testing.assert_allclose(softmax(h@w.T+b),softmax(h@(w+shared).T+b+7),atol=1e-14)


def test_exact_nullspace_variation_produces_no_directional_logits():
    h,w,b,theta=fixture();h[:,:2]=0;h[:,7]=np.cos(theta)
    r=decompose(h,w,b,theta)
    assert r['actor_rowspace_hidden_variance_fraction']==0
    assert r['centered_logit_family_rms']==0
    assert r['probabilities']['family_rms']<1e-15


def test_saturation_reduces_probability_variation_without_removing_logit_signal():
    h,w,b,theta=fixture();normal=decompose(h,w,b,theta)
    b[1]=20;saturated=decompose(h,w,b,theta)
    assert saturated['centered_logit_family_rms']==normal['centered_logit_family_rms']
    assert saturated['actor_rowspace_hidden_variance_fraction']==normal['actor_rowspace_hidden_variance_fraction']
    assert saturated['probabilities']['family_rms']<normal['probabilities']['family_rms']*1e-6
    assert saturated['softmax']['actual_observed_signal_gain']<1e-7
    assert saturated['preferred_mean_logit_action']=='turn_right'


def test_near_zero_margin_ratios_are_null_and_exact_action_changes_reported():
    h,w,b,theta=fixture();r=decompose(h,w,b,theta)
    assert all(q['variation_over_mean_margin'] is None for q in r['pairwise_margins'])
    assert r['probabilities']['fraction_phases_changing_mean_preferred_action']>0
