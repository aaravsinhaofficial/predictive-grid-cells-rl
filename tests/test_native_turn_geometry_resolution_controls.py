"""Independent prospective controls for nested physical-history resolution."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import native_turn_geometry_resolution as assay


def curve(theta):
    return np.column_stack((.3*np.cos(theta), .4*np.sin(theta), .08*np.cos(2*theta),
                            np.zeros_like(theta), 2+.6*np.cos(theta),
                            3+.7*np.sin(theta), .2*np.sin(2*theta), np.zeros_like(theta)))


@pytest.mark.parametrize('center', [False, True])
def test_midpoint_wiggle_is_not_hidden_by_exact_shared_samples(center):
    theta = 2*np.pi*np.arange(1024)/1024
    fine = curve(theta); coarse = fine[::2]
    good = assay.convergence(coarse, coarse, fine, fine, center)
    assert good['passes']
    # Every shared 512-point state is unchanged. Only genuinely simulated
    # midpoints leave the coarse polygon, so vertex matching alone would miss it.
    wiggle = fine.copy()
    wiggle[1::2, 3] += .03
    assert np.array_equal(wiggle[::2], coarse)
    for first, last in ((wiggle, fine), (fine, wiggle)):
        result = assay.convergence(coarse, coarse, first, last, center)
        assert not result['passes']
        bad_label = 'initial' if first is wiggle else 'evolved'
        assert result['comparisons'][bad_label]['max_distance_over_initial_radius'] > assay.CONVERGENCE


@pytest.mark.parametrize('metric', ['raw', 'h_c_RMS'])
def test_both_densities_keep_all_eight_physical_anchors_and_tangent_type(metric):
    theta = 2*np.pi*np.arange(1024)/1024
    reference = curve(theta)
    agreement = assay.tangent_agreement(reference, metric)
    direct = []
    for density in (512, 1024):
        selected = reference[assay.nested_indices(1024, density)]
        scale = np.ones(8) if metric == 'raw' else assay.old.state_scales(selected)
        starts, metadata = assay.old.make_perturbations(selected, scale, anchors=8)
        ids = np.asarray(metadata['anchor_indices'])
        assert np.array_equal(ids, np.arange(8)*density//8)
        assert np.array_equal(starts[:density][ids], reference[np.arange(8)*128])
        assert len(set(ids)) == 8 and metadata['names'][-1] == 'tangent'
        physical_angles = theta[assay.nested_indices(1024, density)[ids]]
        assert np.allclose(physical_angles, np.arange(8)*np.pi/4, atol=0, rtol=0)
        directions = metadata['scaled_directions'][-1]
        nearby = (np.roll(selected/scale, -1, axis=0)-np.roll(selected/scale, 1, axis=0))[ids]
        nearby /= np.linalg.norm(nearby, axis=1, keepdims=True)
        assert np.allclose(directions, nearby, atol=2e-15, rtol=2e-15)
        direct.append(directions)
    cosines = np.sum(direct[0]*direct[1], axis=1)
    assert np.allclose(agreement['cosines'], cosines, atol=2e-15, rtol=2e-15)
    assert agreement['minimum_cosine'] > .9999
