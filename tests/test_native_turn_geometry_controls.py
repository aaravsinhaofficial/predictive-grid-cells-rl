"""Independent controls for two-section geometry; no original model execution."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_turn_geometry import bound_perturbations, finite_probe, make_perturbations


DTYPE = torch.float64


def two_section_system():
    # Both h-like/c-like blocks carry a nondegenerate part of the phase plane.
    basis = torch.tensor([[1., 0.], [0., 1.], [.6, .2], [-.2, .6]], dtype=DTYPE)
    basis, _ = torch.linalg.qr(basis)
    matrix = torch.tensor([[1.1, .25, .08, 0.], [0., .9, 0., .05],
                           [.1, 0., 1.05, .15], [0., .08, 0., .95]], dtype=DTYPE)
    inverse = torch.linalg.inv(matrix)
    shift = torch.tensor([.11, -.07, .08, .04], dtype=DTYPE)
    radius = .6

    def radial(z, decay):
        coordinates = z @ basis
        r = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
        assert torch.all(r > 1e-8)
        plane = coordinates @ basis.T
        return (coordinates * (radius + decay * (r-radius))/r) @ basis.T + decay*(z-plane)

    def transform(z): return z @ matrix.T + shift
    def untransform(z): return (z-shift) @ inverse.T
    def forward(z): return transform(radial(z, .3))
    def backward(z): return radial(untransform(z), .4)
    angles = torch.arange(128, dtype=DTYPE) * (2*torch.pi/128)
    circle = radius*torch.stack((angles.cos(), angles.sin()), -1) @ basis.T
    return circle, transform(circle), forward, backward, transform, untransform


def test_affine_two_section_ring_survives_and_restores_broad_normals():
    circle, other, forward, backward, _, _ = two_section_system()
    for curve, operations in ((circle, (forward, backward)), (other, (backward, forward))):
        def transition(z, step): return operations[step % 2](z)
        for offsets in ((0,), (0, 1)):
            result, arrays = finite_probe(transition, curve.numpy(), np.ones(4), lambda z: z, offsets, 2)
            assert result['geometry']['False']['necessary_geometry_pass']
            assert result['geometry']['True']['necessary_geometry_pass']
            assert result['all_normal_directions_pass']
            assert result['leakage_count'] == 0
            assert result['tangent_total_gain']['median'] > .9
            assert arrays['normal_gain'][:-1].max() < .3
    # The one-step sections have different geometry: pooling them is inappropriate.
    assert not torch.allclose(circle, other)


def test_action_locked_period_two_point_fails_phase_geometry():
    circle, other, _, _, transform, _ = two_section_system()
    anchor = circle[:1]
    for curve, phase in ((circle, 0), (other, 1)):
        def transition(z, step):
            value = transform(anchor) if (step+phase) % 2 == 0 else anchor
            return value.expand_as(z).clone()
        result, _ = finite_probe(transition, curve.numpy(), np.ones(4), lambda z: z, (0, 1), 2)
        assert not result['geometry']['False']['necessary_geometry_pass']
        assert not result['geometry']['True']['necessary_geometry_pass']
        assert not result['all_normal_directions_pass']


def test_affine_shrinking_plane_fails_despite_circular_cross_sections():
    circle, other, _, _, transform, untransform = two_section_system()
    forward = lambda z: transform(.9*z)
    backward = lambda z: .9*untransform(z)
    for curve, operations in ((circle, (forward, backward)), (other, (backward, forward))):
        def transition(z, step): return operations[step % 2](z)
        result, _ = finite_probe(transition, curve.numpy(), np.ones(4), lambda z: z, (0, 1), 2)
        assert not result['geometry']['False']['necessary_geometry_pass']
        assert not result['all_normal_directions_pass']
        radial = next(r for r in result['directions'] if r['name'] == 'full_radial')
        assert radial['normal_over_tangent_gain']['median'] > .9


def test_poincare_chain_uses_correct_section_and_evaluation_points():
    _, _, forward, backward, _, _ = two_section_system()
    point = torch.tensor([.52, -.23, .17, .11], dtype=DTYPE)
    for first, second in ((forward, backward), (backward, forward)):
        jfirst = torch.func.jacrev(first)(point)
        jsecond = torch.func.jacrev(second)(first(point))
        chain = jsecond @ jfirst
        eps = 1e-6; directions = torch.eye(4, dtype=DTYPE)
        fd = torch.stack([(second(first(point+eps*d))-second(first(point-eps*d)))/(2*eps)
                          for d in directions], -1)
        torch.testing.assert_close(chain, fd, atol=2e-9, rtol=2e-8)
        assert float((jfirst @ jsecond-fd).abs().max()) > 1e-3
        wrong_point = torch.func.jacrev(second)(point) @ jfirst
        assert float((wrong_point-fd).abs().max()) > 1e-3


def test_shared_bound_uses_one_positive_amplitude_and_preserves_directions():
    circle, _, _, _, _, _ = two_section_system()
    curve = circle.numpy().copy()
    curve[:, 0] += .9999-curve[:, 0].max()
    scale = np.array([.7, .7, 1.3, 1.3])
    starts, metadata = make_perturbations(curve, scale)
    assert np.any(metadata['physical_hidden_bound_excess'] > 0)
    bounded, info = bound_perturbations(starts, metadata, scale, 'shared')
    assert 0 < info['shared_amplitude_multiplier'] < 1
    assert info['physical_construction_conclusive']
    assert np.abs(bounded[128:, :2]).max() <= 1
    np.testing.assert_allclose(info['actual_scaled_norms'], info['bounded_target_norms'][0], rtol=1e-10)
    np.testing.assert_allclose(info['actual_intended_cosine'], 1, atol=1e-12)
    assert info['block_purity_exact'] and info['actual_geometry_guard_all']


def test_clipped_equalized_fallback_preserves_blocks_and_reports_zero_kicks():
    circle, _, _, _, _, _ = two_section_system()
    curve = circle.numpy().copy()
    curve[:, 0] += 1-curve[:, 0].max()
    scale = np.array([.7, .7, 1.3, 1.3])
    starts, metadata = make_perturbations(curve, scale)
    shared, shared_info = bound_perturbations(starts, metadata, scale, 'shared')
    assert shared_info['shared_amplitude_multiplier'] == 0
    assert not shared_info['physical_construction_conclusive']
    bounded, info = bound_perturbations(starts, metadata, scale, 'clip_equalize')
    assert np.abs(bounded[128:, :2]).max() <= 1 and info['block_purity_exact']
    expected = np.broadcast_to(np.asarray(info['bounded_target_norms']), info['actual_scaled_norms'].shape)
    np.testing.assert_allclose(info['actual_scaled_norms'], expected, atol=2e-16)
    assert not info['actual_nonzero_all']
    assert not info['actual_geometry_guard_all']
    assert not info['physical_construction_conclusive']
    # No clipping/reprojection is concealed: actual vectors reconstruct starts.
    base = curve[np.asarray(metadata['anchor_indices'])]
    np.testing.assert_array_equal(bounded[128:].reshape(21, 2, 8, 4),
                                  base[None, None]+info['actual_physical_kicks'])


def test_clipped_nonzero_kicks_do_not_hide_changed_normal_direction():
    # At this anchor the limiting h coordinate has a tangent component. Clipping
    # must therefore be checked geometrically, not merely for matching lengths.
    base = np.array([[.9999, 0., .1, .1]])
    root = np.sqrt(.5)
    directions = np.array([[root, 0., -root, 0.], [0., 1., 0., 0.],
                           [0., 0., 0., 1.], [0., 0., 0., 1.], [root, 0., root, 0.]])[:, None]
    kicks = .02*directions[:, None]*np.array([1., -1.])[None, :, None, None]
    starts = np.concatenate((base, (base[None, None]+kicks).reshape(-1, 4)))
    metadata = dict(count=1, anchor_indices=[0], names=['mixed_normal', 'hidden_only', 'cell_only', 'cell_radial', 'tangent'],
                    scaled_directions=directions, amplitude=.02, full_radius=1.)
    _, info = bound_perturbations(starts, metadata, np.ones(4), 'clip_equalize')
    assert info['actual_nonzero_all'] and info['exact_hidden_bounds'] and info['block_purity_exact']
    assert not info['actual_geometry_guard_all']
    assert not info['physical_construction_conclusive']
    assert np.max(np.abs(info['actual_tangent_fraction'][:-1])) > .05
