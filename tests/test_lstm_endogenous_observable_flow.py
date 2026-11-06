import json
import numpy as np
import torch
from scripts.lstm_endogenous_observable_flow import observable_probe


def circular_family():
    theta=np.arange(192)*2*np.pi/192
    return np.column_stack((np.cos(theta),np.sin(theta),np.zeros_like(theta)))


def test_known_ring_times_shared_drift_passes_centered_observable_control():
    def step(z):
        r=z[:,:2].norm(dim=1,keepdim=True)
        xy=z[:,:2]*(.8+.2/r)
        angle=.04
        x=xy[:,0]*np.cos(angle)-xy[:,1]*np.sin(angle)
        y=xy[:,0]*np.sin(angle)+xy[:,1]*np.cos(angle)
        return torch.stack((x,y,z[:,2]+.1),1)
    result=observable_probe(step,circular_family(),np.ones(3),lambda z:z,count=192)
    json.dumps(result,allow_nan=False)
    assert result['centered_conditional_factor_candidate']
    assert not result['noncentered_observable_candidate']
    assert result['radial_over_tangent_observable_gain']['median']<.05


def test_shrinking_plane_sampled_on_a_circle_is_not_a_ring_factor():
    def step(z):
        return torch.cat((z[:,:2]*.98,z[:,2:]+.1),1)
    result=observable_probe(step,circular_family(),np.ones(3),lambda z:z,count=192)
    assert not result['centered_conditional_factor_candidate']
    assert .9<result['radial_over_tangent_observable_gain']['median']<1.1


def test_output_silent_circle_fails_observable_kick_guard():
    result=observable_probe(lambda z:z,circular_family(),np.ones(3),lambda z:z[:,2:],count=192)
    assert not result['centered_conditional_factor_candidate']
    assert not result['candidate_gates']['centered']['observable_radial_kick_nontrivial_in80percent']
