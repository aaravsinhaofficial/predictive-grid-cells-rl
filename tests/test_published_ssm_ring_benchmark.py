"""Calibration checks that protect the full-state geometry and derivative test."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("published_ssm", Path(__file__).resolve().parents[1]/"scripts/published_ssm_ring_benchmark.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_state_distance_does_not_accept_an_inward_collapsing_circle():
    phase = np.arange(4096)*2*np.pi/4096
    circle = np.c_[np.cos(phase),np.sin(phase),np.zeros_like(phase)]
    points = circle[::256]
    assert module.distance_to_curve(points,circle).max() < 1e-12
    distance = module.distance_to_curve(points*np.exp(-1),circle)
    assert np.max(np.abs(distance-(1-np.exp(-1)))) < 1e-6
    # A spectator coordinate cannot be silently projected away.
    displaced = points.copy(); displaced[:,2] = .2
    assert np.max(np.abs(module.distance_to_curve(displaced,circle)-.2)) < 1e-12


def test_lift_derivative_at_zero_and_nonlinear_state():
    powers = np.array([[1,0],[0,1],[2,0],[1,1],[0,2],[3,0]])
    rng = np.random.default_rng(24)
    coefficients = rng.normal(size=(5,6))
    for q in [np.zeros(2),np.array([.7,-.3])]:
        analytic = module.polynomial_jacobian(q,coefficients,powers)
        fd = np.stack([(module.polynomial(q+1e-6*e,coefficients,powers)-module.polynomial(q-1e-6*e,coefficients,powers))/2e-6 for e in np.eye(2)],axis=-1)
        np.testing.assert_allclose(analytic,fd,rtol=1e-7,atol=1e-8)


def test_open_branch_sampling_never_invents_an_endpoint_chord():
    angles = np.linspace(0,np.pi,1025)
    branch = np.c_[np.cos(angles),np.sin(angles)]
    samples = module.arc_resample(branch,20,closed=False)
    assert np.min(np.linalg.norm(samples,axis=1)) > .99999
    assert np.min(samples[:,1]) > 0
