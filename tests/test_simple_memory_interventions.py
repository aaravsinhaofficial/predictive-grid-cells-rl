"""Synthetic controls for intervention semantics, never trained outcomes."""
import numpy as np
import pytest
import torch
from scripts import simple_memory_interventions as study


class LinearCell(torch.nn.Module):
    def forward(self, inputs, state):
        h,c = state
        return .5*h+.5*inputs[:,:2], .5*c+.5*inputs[:,2:]


class LinearPolicy(torch.nn.Module):
    nh = 2
    def __init__(self):
        super().__init__()
        self.lstm = LinearCell()
        self.pi = torch.nn.Linear(2,6,bias=False)
        with torch.no_grad(): self.pi.weight.copy_(torch.arange(12).reshape(6,2)/12)


def circle(n=24):
    theta = np.arange(n)*2*np.pi/n
    z = .2*np.tile(np.c_[np.cos(theta),np.sin(theta)],(1,2))
    states = torch.tensor(z,dtype=torch.float32)
    return dict(states=states,theta=torch.tensor(theta,dtype=torch.float64),
        scale=torch.ones(4),dark_inputs=torch.zeros_like(states),pulse_inputs=dict(
            ordinary=torch.zeros_like(states),correct=states,
            wrong_plus=torch.roll(states,-n//6,0),wrong_minus=torch.roll(states,n//6,0)))


def test_state_donors_and_random_control_have_correct_norms():
    family = circle()
    edits, valid, audit = study.make_edits(family)
    z = family['states'].numpy()
    assert np.array_equal(edits['plus30'],np.roll(z,-2,0))
    assert np.array_equal(edits['minus30'],np.roll(z,2,0))
    plusnorm = np.linalg.norm(edits['plus30']-z,axis=1)
    minusnorm = np.linalg.norm(edits['minus30']-z,axis=1)
    assert np.allclose(np.linalg.norm(edits['normal_plus']-z,axis=1),(plusnorm+minusnorm)/2)
    assert np.allclose(edits['normal_plus']+edits['normal_minus'],2*z)
    assert audit['maximum_normal_tangent_dot']<1e-12
    assert all(v.all() for v in valid.values())


def test_invalid_random_pair_is_not_clipped_or_replaced():
    family = circle()
    family['states'][:,:2] += .81
    edits,valid,audit = study.make_edits(family)
    assert not valid['normal_plus'].all()
    assert np.array_equal(valid['normal_plus'],valid['normal_minus'])
    assert np.max(np.abs(edits['normal_plus'][:,:2]))>1
    assert max(audit['hidden_bound_excess']['normal_plus'])>0
    edits2,valid2,_ = study.make_edits(family)
    assert all(np.array_equal(edits[k],edits2[k]) for k in edits)
    assert all(np.array_equal(valid[k],valid2[k]) for k in valid)


def test_correct_pulse_repairs_phase_in_known_linear_fixture_only():
    family,policy = circle(),LinearPolicy()
    before = torch.get_rng_state().clone()
    result,raw = study.cue_repair(policy,family,times=(0,1,5,15),stride=1)
    assert torch.equal(before,torch.get_rng_state())
    # This fixture has an explicitly defined input-driven angular correction,
    # not a learned ring: all trajectories collapse with repeated dark inputs.
    contrast = result['contrasts']['15']['ordinary_minus_correct']
    assert contrast['donor_mean']>10
    assert contrast['donor_minus_no_edit']>10
    assert np.allclose(raw['t5_correct_plus30'],raw['t1_correct_plus30']/16)
    assert result['rows']['15']['correct']['none']['correct_reference']['radius']<1e-4
    assert 'ring' not in result or result.get('ring') is not True


def test_grid_pulse_sham_and_no_pulse_are_distinct():
    family,policy = circle(),LinearPolicy()
    same = family['pulse_inputs']['correct'].clone()
    family['pulse_inputs']={key:same.clone() for key in family['pulse_inputs']}
    result,raw = study.cue_repair(policy,family,times=(0,1,5),stride=1)
    for name in study.EDITS:
        for cue in ('ordinary','wrong_plus','wrong_minus'):
            assert np.array_equal(raw[f't5_{cue}_{name}'],raw[f't5_correct_{name}'])
    assert not np.array_equal(raw['t5_none_none'],raw['t5_correct_none'])
    assert result['contrasts']['5']['ordinary_minus_correct']['donor_mean']==0


def trace_fixture():
    return [dict(input_hashes={'rgb':'same'},action=2,reward=10.,hit=True,done=False,
        post_physics_hashes={'pos':'same'},h_before=np.zeros(2),c_before=np.zeros(2),
        h_after=np.ones(2)*.2,c_after=np.ones(2)*.4,hit_pos=np.zeros(2))]


def test_independent_trace_check_rejects_actions_inputs_physics_and_missing_rows():
    expected = trace_fixture()
    assert study.compare_native_trace(trace_fixture(),expected)['qualified']
    for key,value in [('action',3),('reward',0.),('input_hashes',{'rgb':'changed'}),
                      ('post_physics_hashes',{'pos':'changed'})]:
        actual=trace_fixture();actual[0][key]=value
        assert not study.compare_native_trace(actual,expected)['qualified']
    actual=trace_fixture();actual[0]['c_after']+=.01
    assert not study.compare_native_trace(actual,expected)['qualified']
    assert not study.compare_native_trace([],expected)['qualified']
    actual=trace_fixture();actual[0]['h_after']+=1e-6
    assert study.compare_native_trace(actual,expected)['qualified']
    assert not study.compare_native_trace(actual,expected,exact_states=True)['qualified']


def test_zero_radius_has_no_phase_error_claim():
    points=np.zeros((24,4))
    result=study.phase_error(points,points,np.arange(24)*2*np.pi/24,
                             np.arange(24)*2*np.pi/24,np.ones(4))
    assert not result['defined'] and 'phase_mae_deg' not in result
