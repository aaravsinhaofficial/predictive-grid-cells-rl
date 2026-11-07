"""Private arrays only: variance algebra and genuinely episode-held-out readouts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/analyze_simple_memory_activity.py'
spec = importlib.util.spec_from_file_location('simple_activity', SOURCE)
activity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activity)


def synthetic(n_episodes=25, rows=40):
    rng = np.random.default_rng(5)
    episode = np.repeat(np.arange(n_episodes), rows)
    theta = rng.uniform(-np.pi, np.pi, len(episode))
    target = np.stack([np.column_stack((np.cos(theta+shift), np.sin(theta+shift)))
                       for shift in (0., .5, -1.)], axis=1)
    x = np.column_stack((np.cos(theta), np.sin(theta), rng.normal(size=len(theta)), np.ones(len(theta))))
    visible = np.tile(np.arange(rows) < rows//2, n_episodes)
    return x, target, episode, visible, np.ones(len(theta), bool)


def test_weighted_covariance_identity_with_unequal_episode_counts():
    rng = np.random.default_rng(7)
    episode = np.repeat([0, 1, 2], [5, 13, 2])
    x = rng.normal(size=(20, 5))+episode[:, None]*np.arange(5)[None, :]
    cov, counts = activity.covariances(x, episode, block=7)
    grand = x.mean(0)
    expected_total = (x-grand).T @ (x-grand)/len(x)
    expected_between = sum(np.sum(episode == e)*np.outer(x[episode == e].mean(0)-grand,
        x[episode == e].mean(0)-grand) for e in np.unique(episode))/len(x)
    expected_within = sum((x[episode == e]-x[episode == e].mean(0)).T @
        (x[episode == e]-x[episode == e].mean(0)) for e in np.unique(episode))/len(x)
    np.testing.assert_allclose(cov['total'], expected_total, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cov['between'], expected_between, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cov['within'], expected_within, rtol=1e-13, atol=1e-13)
    assert counts['episode_counts'] == [5, 13, 2] and counts['identity_error'] < 1e-13
    wrong_unweighted = np.cov(np.stack([x[episode == e].mean(0) for e in np.unique(episode)]), rowvar=False, bias=True)
    assert not np.allclose(cov['between'], wrong_unweighted)


def test_zero_variance_and_actor_softmax_common_direction():
    x = np.full((15, 3), .1)
    result, _ = activity.covariance_analysis(x, np.repeat(np.arange(3), 5), np.ones((6, 3)))
    assert result['between_fraction'] is None
    assert result['spectra']['total']['numerical_rank'] == 0
    assert result['spectra']['total']['top10_share'] is None
    assert result['actor_alignment']['between_fraction'] is None
    assert result['subspaces']['total_vs_within']['mean_squared_overlap'] is None


def test_episode_heldout_angle_recovered_with_one_factorization_per_fold():
    x, target, episode, visible, known = synthetic()
    pred, baseline, folds, artifact = activity.fit_episode_readouts(x, target, episode, visible, known, replicates=3)
    assert sum(fold['factorization_count'] for fold in folds) == 5
    for fold in folds:
        assert set(fold['train_episode_ids']).isdisjoint(fold['test_episode_ids'])
        assert fold['constant_coordinates'] == 1
    scores = activity.angular_metrics(pred[~visible, 0, 0], target[~visible, 0], baseline[~visible, 0], episode[~visible])
    assert scores['ridge']['circular_mae_deg']['pooled'] < .1
    assert scores['unit_angle_comparison']['pooled']['r2_vs_constant'] > .99
    null_errors = [activity.angular_metrics(pred[~visible, i, 0], target[~visible, 0],
        baseline[~visible, 0], episode[~visible])['ridge']['circular_mae_deg']['pooled'] for i in (1, 2, 3)]
    assert min(null_errors) > 30
    assert artifact['null_episode_rotation_radians'].shape == (3, 25)


def test_test_episode_labels_and_features_never_enter_fold_fit_or_scaling():
    x, target, episode, visible, known = synthetic()
    first = activity.fit_episode_readouts(x, target, episode, visible, known, replicates=2)
    changed_x, changed_target = x.copy(), target.copy()
    test_fold = episode % 5 == 0
    changed_x[test_fold] *= 10000
    changed_target[test_fold] = changed_target[test_fold, :, ::-1]
    second = activity.fit_episode_readouts(changed_x, changed_target, episode, visible, known, replicates=2)
    for key in ('fold0_mean', 'fold0_scale', 'fold0_coefficients', 'fold0_intercept'):
        np.testing.assert_array_equal(first[3][key], second[3][key])
    np.testing.assert_array_equal(first[1][test_fold], second[1][test_fold])
    # Changing only blind training-episode observations must not affect any fit.
    changed_x = x.copy(); changed_x[~visible] += 12345
    third = activity.fit_episode_readouts(changed_x, target, episode, visible, known, replicates=2)
    for key in first[3]: np.testing.assert_array_equal(first[3][key], third[3][key])


def test_constant_baseline_not_refit_on_blind_labels_and_null_table_shared():
    x, target, episode, visible, known = synthetic()
    pred, baseline, folds, artifacts = activity.fit_episode_readouts(x, target, episode, visible, known, replicates=2)
    for fold in range(5):
        tr = visible & (episode % 5 != fold)
        te = episode % 5 == fold
        np.testing.assert_allclose(baseline[te], np.broadcast_to(target[tr].mean(0), baseline[te].shape))
    changed_target = target.copy(); changed_target[~visible] *= -1
    other = activity.fit_episode_readouts(x, changed_target, episode, visible, known, replicates=2)
    np.testing.assert_array_equal(pred, other[0])
    np.testing.assert_array_equal(baseline, other[1])
    np.testing.assert_array_equal(artifacts['null_episode_rotation_radians'],
        activity.episode_rotation_offsets(np.arange(25)[::-1], replicates=2))


def test_undefined_angles_remain_in_raw_sincos_check_and_no_constant_information_gain():
    target = np.tile([1., 0.], (8, 1)); episode = np.repeat([0, 1], 4)
    zero = np.zeros_like(target)
    result = activity.angular_metrics(zero, target, zero, episode)
    assert result['undefined_prediction_rows'] == result['undefined_baseline_rows'] == 8
    assert result['paired_angle_rows'] == 0 and result['unit_angle_comparison']['pooled']['r2_vs_constant'] is None
    assert result['sincos_regression_comparison']['pooled']['r2_vs_constant'] == 0
    equal = activity.angular_metrics(target, target, target, episode)
    assert equal['ridge']['circular_mae_deg']['pooled'] == 0
    assert equal['unit_angle_comparison']['pooled']['r2_vs_constant'] is None
    assert equal['sincos_regression_comparison']['pooled']['mse_improvement'] == 0


def test_missing_visible_training_is_reported_without_test_fallback():
    x, target, episode, visible, known = synthetic(n_episodes=5, rows=10)
    visible[:] = False
    pred, baseline, folds, artifacts = activity.fit_episode_readouts(x, target, episode, visible, known, replicates=2)
    assert np.isnan(pred).all() and np.isnan(baseline).all()
    assert all(row['status'] == 'no_known_visible_training_rows' for row in folds)
    assert all(row['factorization_count'] == 0 for row in folds)


def test_episode_equal_weights_differ_from_row_pooling():
    values = np.array([0., 2., 2., 2.]); episode = np.array([0, 1, 1, 1])
    assert activity.averaged(values, episode) == {'pooled': 1.5, 'episode_equal': 1.}


def test_saved_array_integration_retains_empty_bins_and_all_nulls():
    x, target, episode, visible, known = synthetic(n_episodes=10, rows=20)
    theta = np.arctan2(target[:, 0, 1], target[:, 0, 0])
    arrays = dict(h=x.astype(np.float32), c=(x*.7).astype(np.float32), episode_id=episode,
        known=known, blind=~visible, phase_age=np.tile(np.arange(1, 11), 20),
        body_heading=theta, travel_dir=target[:, 1], pos=np.zeros((len(x), 2)),
        goal_pos=np.column_stack((np.cos(theta*.5), np.sin(theta*.5))))
    result, artifacts = activity.analyze_arrays(arrays, np.arange(24).reshape(6, 4))
    assert set(result['covariance']) == {'h', 'c', 'hc'}
    assert set(result['angular_readouts']) == {'h', 'hc'}
    for feature in ('h', 'hc'):
        readout = result['angular_readouts'][feature]
        assert readout['factorization_count'] == 5
        for name in activity.TARGETS:
            assert len(readout['targets'][name]['all_known']['null_replicates']) == 20
            assert readout['targets'][name]['blind_age_46_60']['actual']['rows'] == 0
    json.dumps(result, allow_nan=False)
    assert all(np.isfinite(value).all() for value in artifacts.values())


@pytest.mark.parametrize('manifest', [None, {}, {'unrelated.py': '0'*64}])
def test_authentication_requires_nonempty_manifest_binding_own_source(tmp_path, manifest):
    protocol = dict(status='frozen')
    if manifest is not None: protocol['source_sha256'] = manifest
    path = tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    with pytest.raises(AssertionError, match='source manifest'):
        activity.authenticate(tmp_path/'no_run', tmp_path/'no_states', path)
