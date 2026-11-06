import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('module_observable',Path(__file__).resolve().parents[1]/'scripts/lstm_module_phase_observable.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_shared_drift_is_removed_but_radial_decay_is_retained():
    phase=np.arange(384)*2*np.pi/384
    ring=np.c_[np.cos(phase),np.sin(phase),np.zeros_like(phase)]
    drift=np.array([0.,0.,7.])
    result=module.necessary_gates(ring,ring+drift)
    assert result['centering']['centered']['passes_necessary_gates']
    assert not result['centering']['noncentered']['passes_necessary_gates']
    contracting=module.necessary_gates(ring,.5*ring+drift)
    assert not contracting['centering']['centered']['passes_necessary_gates']
    assert not contracting['centering']['centered']['gates']['p90_invariance_residual_below5percent']


def test_numerically_visible_tiny_ring_fails_fixed_amplitude_guard():
    phase=np.arange(384)*2*np.pi/384
    ring=1e-8*np.c_[np.cos(phase),np.sin(phase)]
    result=module.necessary_gates(ring,ring)
    assert not result['centering']['centered']['gates']['observable_radius_RMS_at_least1e_minus4']
    assert not result['centering']['centered']['passes_necessary_gates']
