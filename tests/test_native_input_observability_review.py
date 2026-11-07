"""Independent synthetic controls; no original policies or environments."""
import sys
from pathlib import Path
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'scripts')]
import native_input_observability as assay


def probabilities(theta):
    theta = np.asarray(theta)
    q = np.full(theta.shape + (6,), 1 / 6)
    q[..., 0] += .025 * np.cos(theta)
    q[..., 1] -= .025 * np.cos(theta)
    q[..., 2] += .025 * np.sin(theta)
    q[..., 3] -= .025 * np.sin(theta)
    return q


def fixture(invisible=False):
    n = 128
    ids = np.arange(8) * 16
    meta = dict(count=n, anchor_indices=ids, names=['normal', 'tangent'])
    labels = np.broadcast_to(ids, (2, 2, 8)).copy()
    labels[:, 0] = (labels[:, 0] + 2) % n
    labels[:, 1] = (labels[:, 1] - 2) % n
    if invisible:
        labels[0] = ids
    theta = np.arange(n) * 2 * np.pi / n
    discovery = np.tile(probabilities(theta), (1, 6))
    first = np.concatenate([discovery, discovery[labels.reshape(-1)]])
    held = np.stack([probabilities(theta + j * .17) for j in range(6)], 1)
    psi = np.concatenate([held, held[labels.reshape(-1)]])
    return first, psi, meta


def test_phase_only_equivalence_uses_one_initial_label_in_every_query():
    first, psi, meta = fixture()
    result, arrays = assay.heldout_closure(first, first.copy(), psi, psi.copy(), meta)
    assert result['passes'] and result['same_initial_phase_label']
    assert result['class_counts']['phase_only'] == 16
    assert result['class_counts']['tangent_control'] == 16
    assert np.allclose(arrays['initial_heldout_expected'], psi[128:], atol=1e-15)
    changed = psi.copy()
    changed[128] = psi[7]
    failed, _ = assay.heldout_closure(first, first, psi, changed, meta)
    assert not failed['passes']
    assert failed['checks']['initial']['passes']
    assert not failed['checks']['evolved']['passes']


def test_invisible_response_cannot_hide_later_query_leakage():
    first, psi, meta = fixture(invisible=True)
    result, arrays = assay.heldout_closure(first, first, psi, psi, meta)
    assert result['passes'] and result['class_counts']['invisible'] == 16
    assert np.array_equal(arrays['phase_labels'][0], np.broadcast_to(meta['anchor_indices'], (2, 8)))
    leaked = psi.copy()
    leaked[128, 3, 0] += .01
    leaked[128, 3, 1] -= .01
    failed, _ = assay.heldout_closure(first, first, psi, leaked, meta)
    assert not failed['passes']
    assert failed['checks']['evolved']['failed_equivalence_cells'] == 1


def test_constant_heldout_queries_do_not_excuse_wrong_retained_phase():
    first, psi, meta = fixture()
    constant = np.full_like(psi, 1 / 6)
    last = first.copy()
    last[128] = first[8]
    result, _ = assay.heldout_closure(first, last, constant, constant, meta)
    assert not result['passes']
    assert result['checks']['initial']['passes'] and result['checks']['evolved']['passes']
    assert not result['checks']['discovery_evolved']['passes']


def test_action_clock_is_not_a_phase_circle_and_erasure_fails():
    first, _, _ = fixture()
    # Six query-specific answers, identical across physical headings.
    clock = np.concatenate([probabilities(j * .7) for j in range(6)])
    constant = np.broadcast_to(clock, first[:128].shape).copy()
    stationary = assay.geometry(constant, constant)
    erased = assay.geometry(first[:128], constant)
    assert not stationary['necessary_geometry_pass']
    assert not erased['necessary_geometry_pass']
    assert not stationary['gates']['initial_probability_RMS_at_least_1e_minus4']
    assert not erased['gates']['evolved_probability_RMS_at_least_1e_minus4']
