"""Apply fixed episode-held-out activity decoders to saved causal fork traces.

No fitting, policy/environment execution, or causal/attractor qualification.
The caller authenticates the activity artifact, checkpoint and source manifest.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from analyze_simple_memory_activity import TARGETS, angular_metrics


def array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def decoder(artifacts, feature, episode_id):
    """Load one prespecified fold and actual (not rotated-null) coefficients."""
    fold = int(episode_id) % 5
    prefix = f'{feature}_readout_fold{fold}_'
    names = ('mean', 'scale', 'coefficients', 'intercept')
    missing = [prefix+name for name in names if prefix+name not in artifacts]
    if missing:
        return None, dict(status='missing_training_fold', fold=fold, missing_artifact_keys=missing)
    mean, scale, coefficient, intercept = [array(artifacts[prefix+name]).astype(np.float64)
                                           for name in names]
    if (mean.ndim != 1 or scale.shape != mean.shape or coefficient.ndim != 4 or
        coefficient.shape[0] != len(mean) or coefficient.shape[1:] != (21, 3, 2) or
        intercept.shape != (21, 3, 2)):
        raise ValueError('Activity fold artifact shape does not match the frozen 20-null schema')
    if not all(np.isfinite(v).all() for v in (mean, scale, coefficient, intercept)) or np.any(scale <= 0):
        return None, dict(status='invalid_training_fold', fold=fold, missing_artifact_keys=[])
    return (mean, scale, coefficient[:, 0], intercept[0]), dict(
        status='fixed_held_out_fold', fold=fold, null_index=0,
        excluded_episode_rule='episode_id % 5 equals this fold; known-visible fitting used the other folds',
        constant_baseline=intercept[0].tolist(), missing_artifact_keys=[])


def trace_targets(trace, physical_goal):
    """Pre-action pose labels; terminal post-reset and hit coordinates are unused."""
    n = len(trace)
    heading = np.asarray([row['body_heading'] for row in trace], float)
    position = np.asarray([row['pos'] for row in trace], float).reshape(n, 2)
    travel = np.asarray([row['travel_dir'] for row in trace], float).reshape(n, 2)
    displacement = np.asarray(physical_goal, float)[None]-position
    angles = np.column_stack((heading, np.arctan2(travel[:, 1], travel[:, 0]),
                             np.arctan2(displacement[:, 1], displacement[:, 0])-heading))
    valid = np.isfinite(angles)
    valid[:, 1] &= np.isfinite(travel).all(1) & (np.linalg.norm(travel, axis=1) > 0)
    valid[:, 2] &= np.isfinite(displacement).all(1) & (np.linalg.norm(displacement, axis=1) > 0)
    return np.stack((np.cos(angles), np.sin(angles)), -1), valid


def analyze_behavior_readouts(records, context, activity_artifacts):
    """Return one small result per original branch, including unavailable rows.

    `records` are full behavioral-fork records before trace extraction. Artifacts
    are the named arrays from analyze_simple_memory_activity.analyze_arrays or
    its activity_arrays.npz. No data from another decoder fold is used.
    """
    episode = int(context['episode_id'])
    if episode < 0: raise ValueError('Nonnegative native episode ID required')
    slot = int(context['selected_slot'])
    goal = array(context['runtime']['environment_tensors']['goal'])[slot].astype(float)
    if goal.shape != (2,) or not np.isfinite(goal).all(): raise ValueError('Invalid archived physical goal')
    fixed = {feature: decoder(activity_artifacts, feature, episode) for feature in ('h', 'hc')}
    branches = []
    keys = set()
    for record in records:
        identity = {key: record[key] for key in ('tape', 'edit', 'cue')}
        key = tuple(identity.values())
        if key in keys: raise ValueError('Duplicate behavioral branch')
        keys.add(key)
        row = dict(**identity, episode_id=episode, branch_qualified=bool(record.get('qualified', False)),
                   readouts={}, requested_decisions=record.get('decisions'))
        if 'trace' not in record:
            row.update(status='no_saved_trace', reason=record.get('reason'), trace_rows=0)
            branches.append(row)
            continue
        trace = record['trace']
        if record.get('decisions', len(trace)) != len(trace):
            raise ValueError('Behavior decision count does not match saved trace')
        if [item['step'] for item in trace] != list(range(len(trace))):
            raise ValueError('Expected the complete consecutive onset trace')
        if any(item['done'] for item in trace[:-1]):
            raise ValueError('Trace crosses an episode boundary; archived goal no longer applies')
        row.update(status='evaluated_saved_trace', trace_rows=len(trace),
                   terminal_row_retained=bool(trace and trace[-1]['done']))
        targets, valid_targets = trace_targets(trace, goal)
        known = np.asarray([item['known'] for item in trace], bool)
        blind = np.asarray([item['blind'] for item in trace], bool)
        if np.any(blind & ~known): raise ValueError('Native blind input requires a known goal')
        early = np.arange(len(trace)) < 15
        bins = dict(all=np.ones(len(trace), bool), blind=known & blind, seen=known & ~blind,
                    first15_steps=early, blind_first15_steps=known & blind & early)
        for feature, (fit, metadata) in fixed.items():
            result = dict(decoder=metadata, targets={})
            if fit is None:
                prediction = np.full((len(trace), 3, 2), np.nan)
                baseline = np.full_like(prediction, np.nan)
                feature_finite = np.zeros(len(trace), bool)
            else:
                mean, scale, beta, intercept = fit
                width = len(mean) if feature == 'h' else len(mean)//2
                h = np.asarray([array(item['h_after']) for item in trace], float).reshape(len(trace), width)
                x = h
                if feature == 'hc':
                    c = np.asarray([array(item['c_after']) for item in trace], float).reshape(len(trace), width)
                    x = np.concatenate((h, c), axis=1)
                feature_finite = np.isfinite(x).all(1)
                prediction = np.einsum('nd,dtk->ntk', (x-mean)/scale, beta)+intercept
                baseline = np.broadcast_to(intercept, prediction.shape)
            for target_index, target in enumerate(TARGETS):
                result['targets'][target] = {}
                for label, requested in bins.items():
                    mask = requested & valid_targets[:, target_index]
                    metrics = angular_metrics(prediction[mask, target_index], targets[mask, target_index],
                        baseline[mask, target_index], np.full(int(mask.sum()), episode, dtype=np.int64))
                    metrics.update(requested_rows=int(requested.sum()),
                        invalid_target_rows=int((requested & ~valid_targets[:, target_index]).sum()),
                        nonfinite_feature_rows=int((requested & ~feature_finite).sum()) if fit is not None else None,
                        weighting='One native context episode per branch; tapes are not independent training seeds')
                    result['targets'][target][label] = metrics
            row['readouts'][feature] = result
        branches.append(row)
    return dict(complete=True, episode_id=episode, decoder_fold=episode % 5, fitted_here=False,
        physical_goal=goal.tolist(), branch_count=len(branches),
        branches_without_trace=sum(row['status']=='no_saved_trace' for row in branches),
        branch_records=branches,
        timing='h_after/c_after follow the current input and precede the action; labels use the same pre-action pose. The final terminal input is retained; post-reset/hit positions never label it.',
        baseline='Actual-fit training-fold sin/cos intercept, unchanged for every fork and bin.',
        scope='Fixed episode-held-out descriptive angular readout; no refitting, new decoder, ring detection or causal qualification.')
