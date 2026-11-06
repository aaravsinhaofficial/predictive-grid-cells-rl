"""Analytical invariants for the published-cohort recurrent-output interventions."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_public_internal_pgc import intervention_arrays, initial_normal_perturbation
from evaluate_internal_pgc_moving import movement_metrics
from summarize_public_internal_pgc import paired_summary, holm, select_value


def test_mean_replacement_preserves_calibration_mean_but_removes_variation():
    mean = np.array([1., 2., 3., 4.], np.float32)
    specs, gates, constants = intervention_arrays(np.array([0, 2]), np.array([[1, 3]]), mean, [0., .25])
    mean_index = next(i for i, s in enumerate(specs) if s["mode"] == "mean_replacement" and s["group"] == "pgc")
    np.testing.assert_allclose(mean*gates[mean_index]+constants[mean_index], mean)
    h = mean+np.array([.2, -.3, .4, -.5])
    actual = h*gates[mean_index]+constants[mean_index]
    np.testing.assert_allclose(actual-mean, (h-mean)*np.array([.75, 1., .75, 1.]))


def test_recurrent_output_gate_scales_columns_only():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(8, 4))
    weight = rng.normal(size=(4, 4))
    gate = np.array([1., .75, 1., .75])
    np.testing.assert_allclose((h*gate)@weight.T, h@(weight*gate[None]).T)
    assert not np.allclose((h*gate)@weight.T, h@(weight*gate[:, None]).T)


def test_normal_perturbation_orthogonal_and_positive_at_actual_precision():
    rng = np.random.default_rng(3)
    h = rng.uniform(.2, 2., size=(5, 128)).astype(np.float32)
    h[:, :20] = 0
    tangents = rng.normal(size=(5, 128, 2))
    delta, report = initial_normal_perturbation(h, tangents, 14)
    assert np.min(h+delta) >= 0
    assert np.max(np.abs(delta[:, :20])) == 0
    assert report["actual_float32_max_tangent_cosine"] < 1e-4
    assert report["actual_float32_max_radial_cosine"] < 1e-4
    np.testing.assert_allclose(np.linalg.norm(delta, axis=-1)/np.linalg.norm(h, axis=-1), .01, atol=1e-7)


def test_stationary_position_score_does_not_reward_following_a_moving_target():
    actual = np.zeros((21, 3, 2))
    wrong = np.zeros_like(actual)
    wrong[:, :, 0] = np.arange(21)[:, None]/20
    stat, _ = movement_metrics(wrong, actual)
    np.testing.assert_allclose(stat["position_error_m"], np.mean(np.arange(1, 21)/20))
    assert stat["increment_alignment"] == 0
    assert stat["increment_gain"] == 0
    assert np.isfinite(list(stat.values())).all()


def test_motion_increment_gain_retains_zero_predictions_and_negative_alignment():
    actual = np.zeros((21, 4, 2))
    actual[:, :, 0] = np.arange(21)[:, None]*.01
    correct, _ = movement_metrics(actual, actual)
    wrong, _ = movement_metrics(-actual, actual)
    frozen, _ = movement_metrics(np.zeros_like(actual), actual)
    np.testing.assert_allclose([correct["increment_gain"], wrong["increment_gain"], frozen["increment_gain"]], [1, -1, 0])
    assert frozen["zero_prediction_fraction"] == 1
    assert frozen["increment_alignment"] == 0


def test_mean_replacement_is_affine_and_does_not_impose_normalization():
    # Native homogeneity implies a neutral scale mode at any nonzero fixed point.
    # Replacing a variable source by a mean can break homogeneity, but is not
    # state normalization and must not be interpreted as preserving actual norm.
    h = torch.tensor([[1., 2.]])
    weight = torch.eye(2)
    gate = torch.tensor([.5, 1.])
    constant = torch.tensor([.2, 0.])
    native = lambda x: (x@weight.T).relu()
    intervention = lambda x: ((x*gate+constant)@weight.T).relu()
    torch.testing.assert_close(native(2*h), 2*native(h))
    assert not torch.allclose(intervention(2*h), 2*intervention(h))
    assert not torch.allclose(intervention(h).norm(), h.norm())


def test_exact_ten_seed_statistics_keep_zero_contrasts_and_adjust_full_family():
    assert paired_summary(np.ones(10))["exact_p_two_sided"] == 2/1024
    assert paired_summary(np.zeros(10))["exact_p_two_sided"] == 1
    zero_included = paired_summary(np.array([1.]*9+[0.]))
    assert zero_included["exact_p_two_sided"] == 4/1024
    np.testing.assert_allclose(holm([.001, .01, .03, .04, .8]), [.005, .04, .09, .09, .8])
    import pytest
    with pytest.raises(ValueError):
        paired_summary(np.ones(9))


def test_controls_averaged_within_seed_and_missing_control_is_error():
    data = {"rows": [{"context": "moving", "mode": "mean", "dose": .25,
                      "steps": 20, "group": f"matched_nonpgc_{i:02d}", "metric": i}
                     for i in range(20)]}
    assert select_value(data, "moving", "mean", .25, 20, "metric", "controls") == 9.5
    data["rows"].pop()
    import pytest
    with pytest.raises(ValueError):
        select_value(data, "moving", "mean", .25, 20, "metric", "controls")


def test_complete_projective_contraction_can_destroy_all_position_patterns():
    # A positive rank-one recurrent map is an explicit counterexample to
    # interpreting greater contraction alone as a better memory attractor.
    h = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    delta = torch.tensor([[.01, -.02, .01], [-.01, .02, -.01]])
    weight = torch.ones((3, 3))/3
    unit = lambda x: x/x.norm(dim=-1, keepdim=True)
    next_h = (h@weight.T).relu()
    next_p = ((h+delta)@weight.T).relu()
    assert (unit(h)[0]-unit(h)[1]).norm() > .5
    torch.testing.assert_close(unit(next_h)[0], unit(next_h)[1])
    torch.testing.assert_close(unit(next_h), unit(next_p))
