"""Known saved coefficients and synthetic traces; no trained model or fitting."""
import copy
import json

import numpy as np
import pytest

from scripts import simple_memory_behavior_readout as study


def fixture(episode=7, count=20):
    angle = np.linspace(-1.1, 1.1, count)
    records = []
    for i, theta in enumerate(angle):
        target = np.array([[np.cos(theta), np.sin(theta)],
            [np.cos(theta+.3), np.sin(theta+.3)], [np.cos(-theta), np.sin(-theta)]])
        h = target.ravel()
        records.append(dict(step=i, known=True, blind=i < 10, done=i == count-1,
            pos=np.zeros(2), body_heading=theta, travel_dir=target[1],
            h_after=h, c_after=h*2, h_before=np.full(6, 999.), c_before=np.full(6, 999.),
            hit_pos=np.array([100., 100.]), post_physics_hashes={'pos': 'new episode, never used'}))
    context = dict(episode_id=episode, selected_slot=1,
        runtime=dict(environment_tensors=dict(goal=np.array([[-3., 5.], [1., 0.]]))))
    artifacts = {}
    for fold in range(5):
        for feature in ('h', 'hc'):
            width = 6 if feature == 'h' else 12
            beta = np.zeros((width, 21, 3, 2))
            intercept = np.zeros((21, 3, 2))
            intercept[0, :, 0] = .2  # Fixed train constant; prediction removes this via centering.
            beta[:6, 0] = np.eye(6).reshape(6, 3, 2)
            mean = np.tile([.2, 0.], 3) if feature == 'h' else np.r_[np.tile([.2, 0.], 3), np.zeros(6)]
            if fold != episode % 5:
                beta[:] = 999.  # Wrong folds cannot accidentally recover the angle.
            beta[:, 1:] = -777.  # Nor may a null fit be used.
            prefix = f'{feature}_readout_fold{fold}_'
            artifacts.update({prefix+'mean': mean, prefix+'scale': np.ones(width),
                prefix+'coefficients': beta, prefix+'intercept': intercept})
    record = dict(tape=0, edit='none', cue='none', qualified=True, decisions=count, trace=records)
    return [record], context, artifacts


def test_known_coefficients_recover_targets_using_held_out_fold_and_after_input_state(monkeypatch):
    records, context, artifacts = fixture()
    import analyze_simple_memory_activity as activity
    monkeypatch.setattr(activity, 'fit_episode_readouts', lambda *a, **k: pytest.fail('No fitting allowed'))
    result = study.analyze_behavior_readouts(records, context, artifacts)
    assert result['decoder_fold'] == 2 and result['fitted_here'] is False
    branch = result['branch_records'][0]
    for feature in ('h', 'hc'):
        output = branch['readouts'][feature]
        assert output['decoder']['fold'] == 2 and output['decoder']['null_index'] == 0
        for target in study.TARGETS:
            metrics = output['targets'][target]['all']
            assert metrics['ridge']['circular_mae_deg']['pooled'] < 1e-6
            assert metrics['unit_angle_comparison']['pooled']['r2_vs_constant'] > .99999
            assert metrics['rows'] == 20 and metrics['paired_angle_rows'] == 20
    assert branch['readouts']['h']['targets']['body_heading']['blind']['rows'] == 10
    assert branch['readouts']['h']['targets']['body_heading']['seen']['rows'] == 10
    assert branch['readouts']['h']['targets']['body_heading']['first15_steps']['rows'] == 15
    json.dumps(result, allow_nan=False)


def test_terminal_pre_action_goal_and_pose_not_post_hit_or_reset_fields():
    records, context, artifacts = fixture(count=1)
    first = study.analyze_behavior_readouts(records, context, artifacts)
    altered = copy.deepcopy(records)
    altered[0]['trace'][0]['hit_pos'] = np.array([-1000., -1000.])
    altered[0]['trace'][0]['post_physics_hashes'] = {'pos': 'another episode, also unused'}
    second = study.analyze_behavior_readouts(altered, context, artifacts)
    assert first == second
    assert first['branch_records'][0]['terminal_row_retained']
    assert first['branch_records'][0]['readouts']['h']['targets']['egocentric_goal_bearing']['all']['rows'] == 1
    crossing = copy.deepcopy(records); crossing[0]['trace'].append(dict(crossing[0]['trace'][0], step=1))
    crossing[0]['decisions'] = 2
    with pytest.raises(ValueError, match='episode boundary'):
        study.analyze_behavior_readouts(crossing, context, artifacts)


def test_missing_fold_no_trace_and_undefined_targets_remain_explicit():
    records, context, artifacts = fixture()
    records.append(dict(tape=0, edit='normal_plus', cue='none', qualified=False, reason='bounds'))
    artifacts = {key: value for key, value in artifacts.items() if not key.startswith('h_readout_fold2_')}
    records[0]['trace'][0]['travel_dir'] = np.zeros(2)
    result = study.analyze_behavior_readouts(records, context, artifacts)
    assert result['branch_count'] == 2 and result['branches_without_trace'] == 1
    first = result['branch_records'][0]
    assert first['readouts']['h']['decoder']['status'] == 'missing_training_fold'
    assert first['readouts']['h']['targets']['body_heading']['all']['missing_fit_rows'] == 20
    travel = first['readouts']['hc']['targets']['last_travel_direction']['all']
    assert travel['invalid_target_rows'] == 1 and travel['rows'] == 19 and travel['requested_rows'] == 20
    assert result['branch_records'][1]['status'] == 'no_saved_trace'
    json.dumps(result, allow_nan=False)


def test_training_baseline_stays_fixed_across_bins_and_branches():
    records, context, artifacts = fixture()
    changed = copy.deepcopy(records[0]); changed.update(edit='plus30', cue='correct')
    for row in changed['trace']: row['h_after'] *= -1; row['c_after'] *= -1
    result = study.analyze_behavior_readouts(records+[changed], context, artifacts)
    base = [branch['readouts']['h']['decoder']['constant_baseline'] for branch in result['branch_records']]
    assert base[0] == base[1] == [[.2, 0.]]*3
    assert result['branch_records'][1]['readouts']['h']['targets']['body_heading']['all']['unit_angle_comparison']['pooled']['r2_vs_constant'] < 0
