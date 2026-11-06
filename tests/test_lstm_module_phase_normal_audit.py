import numpy as np
from scripts.lstm_module_phase_normal_audit import make_starts,synthetic_controls


def test_direction_blocks_orthogonality_and_equal_amplitudes():
    phase=np.arange(128)*2*np.pi/128
    curve=np.zeros((128,6));curve[:,0]=np.cos(phase);curve[:,1]=np.sin(phase)
    starts,meta=make_starts(curve,np.ones(6),count=128)
    perturb=starts[128:].reshape(20,2,8,6)-starts[np.asarray(meta['anchor_indices'])][None,None]
    np.testing.assert_allclose(np.linalg.norm(perturb,axis=-1),meta['amplitude_scaled_norm'],atol=1e-12)
    np.testing.assert_allclose(perturb[:,0],-perturb[:,1],atol=1e-12)
    np.testing.assert_allclose(meta['normal_dot_slice_tangent'],0,atol=1e-12)
    np.testing.assert_allclose(perturb[16,:,:,3:],0,atol=1e-12)
    np.testing.assert_allclose(perturb[17,:,:,:3],0,atol=1e-12)


def test_attracting_ring_with_clock_passes_and_shrinking_plane_fails():
    controls=synthetic_controls()
    assert controls['analytic_ring_with_unobserved_clock']['horizons']['30']['all_19_normal_directions_pass']
    assert not controls['shrinking_plane']['horizons']['30']['all_19_normal_directions_pass']
