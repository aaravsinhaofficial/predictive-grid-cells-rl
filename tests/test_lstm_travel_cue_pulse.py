import numpy as np
import torch
from scripts.lstm_travel_cue_pulse import semantic_nearest,run_dynamics


def circle(n=192):
    theta=np.arange(n)*2*np.pi/n
    return np.column_stack((np.cos(theta),np.sin(theta))),theta


def test_semantic_phase_recovers_signed_substitution_across_wrap():
    state,theta=circle()
    distance,phase=semantic_nearest(np.roll(state,-16,axis=0),state,theta)
    error=np.angle(np.exp(1j*(phase-theta)))
    np.testing.assert_allclose(error,np.pi/6,atol=1e-12)
    assert distance.max()<1e-12


def test_single_cue_correction_persists_only_in_recurrent_memory():
    state,theta=circle(24);ordinary=torch.zeros(1,2,dtype=torch.float64)
    native=torch.tensor(state,dtype=torch.float64)
    def memory(z,x):
        # Known ring: a cue overwrites direction; no cue retains direction.
        active=x.norm(dim=1,keepdim=True)>0
        return torch.where(active,x,z/z.norm(dim=1,keepdim=True))
    result=run_dynamics(memory,state,ordinary,native,np.ones(2),horizons=(0,1,5))
    np.testing.assert_allclose(result[5]['correct_pulse'][1],state,atol=1e-12)
    np.testing.assert_allclose(result[5]['ordinary'][1],np.roll(state,-2,0),atol=1e-12)
    np.testing.assert_allclose(result[5]['wrong_plus60_pulse'][1],np.roll(state,-4,0),atol=1e-12)
    def reactive(z,x):return x
    no_memory=run_dynamics(reactive,state,ordinary,native,np.ones(2),horizons=(0,1,5))
    assert np.linalg.norm(no_memory[1]['correct_pulse'])>0
    np.testing.assert_array_equal(no_memory[5]['correct_pulse'],0)
    assert np.linalg.norm(no_memory[5]['correct_sustained'])>0
