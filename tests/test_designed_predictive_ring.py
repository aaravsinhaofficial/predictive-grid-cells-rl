"""Independent exact-map invariants for the explicitly designed ring."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from designed_predictive_ring import DesignedRing, analytic_spectrum
from summarize_designed_predictive_ring import statistics


def ring(gain=4.):
    m = DesignedRing(3999, gain=gain).double()
    m.basis.copy_(torch.linalg.qr(m.basis).Q)
    return m


def angle(model, h):
    y = model.readout(h)
    return torch.atan2(y[..., 1], y[..., 0])


def test_every_circle_state_fixed_with_no_motion_or_cue():
    m = ring()
    theta = torch.linspace(-3., 3., 137, dtype=torch.float64)
    h = m.initialize(torch.stack((theta.cos(), theta.sin()), -1))
    actual = m.step_cue(h, torch.zeros_like(theta), torch.zeros(len(theta), 2, dtype=theta.dtype))
    torch.testing.assert_close(actual, h, atol=1e-14, rtol=1e-14)


def test_all_63_transverse_modes_contract_and_phase_remains_neutral():
    m = ring()
    h = m.initialize(torch.tensor([[1., 0.]], dtype=torch.float64))[0]
    jac = torch.autograd.functional.jacobian(lambda x: m.step_cue(x[None], torch.zeros(1, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64))[0], h)
    actual = torch.linalg.eigvals(jac).numpy()
    np.testing.assert_allclose(np.sort(actual.real), np.sort(analytic_spectrum(4.)), atol=1e-12)
    assert np.max(np.abs(actual.imag)) < 1e-12


def test_current_cue_follows_completed_motion_without_double_counting():
    m = ring()
    h = m.initialize(torch.tensor([[1., 0.]], dtype=torch.float64))
    omega = torch.tensor([1.], dtype=torch.float64)
    target = torch.tensor([[np.cos(.05), np.sin(.05)]], dtype=torch.float64)
    result = m.step_cue(h, omega, target)
    torch.testing.assert_close(angle(m, result), torch.tensor([.05], dtype=torch.float64), atol=1e-13, rtol=0)


def test_sensory_anchor_contracts_phase_and_reverse_destabilizes_correct_phase():
    m = ring()
    eps = 1e-5
    h = m.initialize(torch.tensor([[np.cos(eps), np.sin(eps)]], dtype=torch.float64))
    zero = torch.zeros(1, dtype=torch.float64)
    forward = m.step_cue(h, zero, torch.tensor([[1., 0.]], dtype=torch.float64))
    reverse = m.step_cue(h, zero, torch.tensor([[-1., 0.]], dtype=torch.float64))
    np.testing.assert_allclose(float(angle(m, forward))/eps, .8, atol=1e-9)
    np.testing.assert_allclose(float(angle(m, reverse))/eps, 1.2, atol=1e-9)


def test_tangent_displacement_persists_but_radius_and_hidden_residual_decay():
    m = ring()
    theta = torch.tensor([.3], dtype=torch.float64)
    circle = m.initialize(torch.stack((theta.cos(), theta.sin()), -1))
    noise = torch.arange(64, dtype=torch.float64)[None]/1000
    normal = noise-(noise@m.basis)@m.basis.T
    h = 1.1*circle+normal
    result = m.step_cue(h, torch.zeros(1, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64))
    torch.testing.assert_close(result-circle, .9*(h-circle), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(angle(m, result), theta, atol=1e-13, rtol=0)


def test_near_zero_cue_has_near_zero_phase_effect():
    m = ring()
    h = m.initialize(torch.tensor([[1., 0.]], dtype=torch.float64))
    result = m.step_cue(h, torch.zeros(1, dtype=torch.float64), torch.tensor([[0., 1e-9]], dtype=torch.float64))
    assert abs(float(angle(m, result))) < 1e-8


def test_exact_eight_seed_comparison_rejects_partial_cohort_and_keeps_zeros():
    assert statistics(np.ones(8))["exact_p_two_sided"] == 2/256
    assert statistics(np.zeros(8))["exact_p_two_sided"] == 1
    assert statistics(np.r_[np.ones(7), 0])["exact_p_two_sided"] == 4/256
    import pytest
    with pytest.raises(ValueError):
        statistics(np.ones(7))
