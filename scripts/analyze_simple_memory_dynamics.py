"""Fixed full-state dynamics of the prospective native memory study.

Pure geometry helpers accept private synthetic transitions. The CLI requires a
frozen analysis protocol and authenticated final models; it never trains or
searches for a favorable context, phase, input law, or horizon.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

import lstm_heading_curve_flow as flow
import native_turn_geometry as normals
from lstm_continued_heading_memory import geometry_gates
from lstm_endogenous_curve_recovery import advance, decomposition, describe
from lstm_local_dynamics_spectra import local_jacobian, autograd_check
from lstm_travel_cue_pulse import semantic_nearest

PROTOCOL = ROOT / 'docs/research/simple_memory_analysis_protocol.json'
PHASES, ARC_PROBES, ANCHORS = 768, 384, 8
NORMAL_SEED = 741024
REGIMES = ('constant', 'forward_backward')
SOURCES = ('scripts/analyze_simple_memory_dynamics.py',
           'scripts/simple_memory_family.py', 'scripts/collect_simple_memory_states.py',
           'scripts/lstm_heading_curve_flow.py', 'scripts/native_turn_geometry.py',
           'scripts/lstm_continued_heading_memory.py', 'scripts/lstm_endogenous_curve_recovery.py',
           'scripts/lstm_local_dynamics_spectra.py', 'scripts/lstm_travel_cue_pulse.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()


def json_value(value):
    if isinstance(value, dict): return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wrap(angle):
    return np.angle(np.exp(1j * angle))


def coverage(angle):
    angle = np.sort(np.mod(np.asarray(angle,dtype=np.float64), 2 * np.pi))
    return float(np.clip(1 - np.diff(np.r_[angle, angle[0] + 2*np.pi]).max() / (2*np.pi),0,1)) if len(angle) else None


class FrozenTransition:
    """One private LSTM copy; all inputs explicit, no global RNG or policy edit."""
    def __init__(self, cell, inputs, *, native_float32=False):
        self.native_float32 = native_float32
        device = next(cell.parameters()).device if native_float32 else torch.device('cpu')
        dtype = torch.float32 if native_float32 else torch.float64
        self.cell = copy.deepcopy(cell).to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.inputs = torch.as_tensor(array(inputs), dtype=dtype, device=device)
        if self.inputs.ndim == 1: self.inputs = self.inputs[None]
        if not torch.isfinite(self.inputs).all(): raise ValueError('Nonfinite frozen input')
        self.drives = F.linear(self.inputs, self.cell.weight_ih, self.cell.bias_ih)

    @torch.no_grad()
    def __call__(self, state, step=0):
        z = state.to(device=self.inputs.device, dtype=self.inputs.dtype)
        index = step % len(self.inputs)
        if self.native_float32:
            h, c = self.cell(self.inputs[index:index+1].expand(len(z), -1), z.chunk(2, 1))
            result = torch.cat((h, c), 1)
        else:
            result = advance(self.cell, self.drives[index:index+1], z)
        return result.cpu().double()


def input_laws(dark_inputs, n_code=512, n_actions=6):
    dark = array(dark_inputs).astype(np.float64)
    if dark.ndim != 2 or not np.array_equal(dark, np.repeat(dark[:1], len(dark), 0)):
        raise ValueError('All complete dark inputs must be exactly identical')
    action_start = dark.shape[1] - n_actions - 1
    if action_start != 256 + 2*n_code: raise ValueError('Unexpected native input layout')
    if np.any(dark[:, 256:256+n_code]) or np.any(dark[:, -1]):
        raise ValueError('Current code and reward must be zero in the fixed blind law')
    action = dark[0, action_start:action_start+n_actions]
    if not (np.count_nonzero(action) == 1 and action.max() == 1 and action.min() == 0):
        raise ValueError('Actual onset PA must be one native action, never episode-start PA=-1')
    cycle = np.repeat(dark[:1], 2, 0)
    cycle[:, action_start:action_start+n_actions] = 0
    cycle[0, action_start+2] = cycle[1, action_start+3] = 1
    return dict(constant=dark[:1], forward_backward=cycle)


@torch.no_grad()
def evolve(transition, start, steps):
    state = torch.as_tensor(np.asarray(start), dtype=torch.float64)
    result = {0: state.numpy().copy()}
    for step in range(steps):
        state = transition(state, step)
        if state.shape != result[0].shape or not torch.isfinite(state).all():
            raise ValueError('Nonfinite or wrong-shaped recurrent transition')
        result[step+1] = state.cpu().double().numpy().copy()
    return result


def macro_transition(transition, regime):
    if regime == 'constant': return lambda z: transition(z, 0)
    if regime == 'forward_backward': return lambda z: transition(transition(z, 0), 1)
    raise ValueError(regime)


def broad_probe(transition, reference, scale, horizon, *, anchors=ANCHORS):
    """Reuse tested finite normal distances and shared bound, with three randoms."""
    with patch.object(normals, 'NORMAL_COUNT', 3):
        starts, metadata = normals.make_perturbations(reference, scale, anchors=anchors, seed=NORMAL_SEED)
    starts, metadata = normals.bound_perturbations(starts, metadata, scale, 'shared')
    final = evolve(lambda z, _: transition(z), starts, horizon)[horizon]
    first, last = starts / scale, final / scale
    result, archive = normals.finite_normal_metrics(first, last, metadata)
    result['normal_directions'] = len(metadata['names']) - 1  # Old generic helper's count is historical.
    result['perturbation_design'] = {key: value for key, value in metadata.items()
                                     if not isinstance(value, np.ndarray)}
    result['physical_conclusive'] = bool(metadata['physical_construction_conclusive']
                                         and normals.initially_observable(result))
    result['physical_status'] = ('pass' if result['all_normal_directions_pass'] else 'failed_recovery') \
        if result['physical_conclusive'] else 'unqualified'
    result['hidden_bound_excess_count'] = int((metadata['physical_hidden_bound_excess'] > 0).sum())
    archive.update(first=first, last=last, starts=starts,
                   **{k: v for k, v in metadata.items() if isinstance(v, np.ndarray)})
    return result, archive, metadata


def radial_construction(reference, scale, count):
    """Audit the exact unbounded radial starts used by the existing core."""
    points = flow.uniform_arc_samples(reference/scale, count)
    tangent = np.roll(points,-1,0)-np.roll(points,1,0)
    unit = tangent/np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-30)
    radial = points-points.mean(0)
    radial -= np.sum(radial*unit,axis=1,keepdims=True)*unit
    lengths = np.linalg.norm(radial,axis=1)
    radius = flow.curve_geometry(reference/scale)['radius']
    radial /= np.maximum(lengths[:,None],1e-30)
    starts = np.concatenate((points+.02*radius*radial,points-.02*radius*radial))*scale
    excess = np.maximum(np.abs(starts[:,:reference.shape[1]//2])-1,0).max(1)
    return dict(hidden_bound_excess_count=int((excess>0).sum()),
        hidden_bound_max_excess=float(excess.max()),
        degenerate_normal_count=int((lengths<=1e-12*max(radius,1e-30)).sum()),
        nominal_metric_kick=.02*radius,proposed_signed_kicks=2*count)


def analyze_regime(transition, start, scale, regime, *, arc_probes=ARC_PROBES, anchors=ANCHORS):
    """One fixed input law, 30 burn-in +30 future neural updates in either case."""
    start, scale = np.asarray(start, float), np.asarray(scale, float)
    if start.ndim != 2 or start.shape[1] % 2 or scale.shape != (start.shape[1],):
        raise ValueError('Expected full h/c family and fixed positive coordinate scale')
    if not np.isfinite(start).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid family/scale')
    states = evolve(transition, start, 60)
    macro = macro_transition(transition, regime)
    horizon, probes = (30, (1, 15, 30)) if regime == 'constant' else (15, (1, 7, 15))
    times = (0, 1, 5, 15, 30, 45, 60) if regime == 'constant' else (0, 2, 14, 30, 44, 60)
    scales = dict(raw=np.ones(start.shape[1]), relative_rms=scale)
    trajectory, invariant, legacy_invariant, broad, archives = {}, {}, {}, {}, {}
    for step in times:
        trajectory[str(step)] = {}
        for name, s in scales.items():
            geometry = flow.curve_geometry(states[step] / s)
            radius0 = flow.curve_geometry(start / s)['radius']
            geometry['radius_retention'] = geometry['radius'] / max(radius0, 1e-30)
            trajectory[str(step)][name] = geometry
    for name, s in scales.items():
        with patch.object(flow, 'FLOW', probes):
            record = flow.invariance_probe(None, None, states[30], s, arc_probes, macro)
        record['future_neural_updates'] = [k if regime == 'constant' else 2*k for k in probes]
        record['radial_construction'] = radial_construction(states[30],s,arc_probes)
        invariant[name] = record
        # The legacy gate uses key30; this explicit remapping is physical time,
        # not30 F/B cycles (which would accidentally double the horizon).
        legacy_invariant[name] = dict(record, horizons={'30': record['horizons'][str(horizon)]})
        broad[name], saved, _ = broad_probe(macro, states[30], s, horizon, anchors=anchors)
        archives.update({f'broad_{name}_{key}': value for key, value in saved.items()})
    gates = geometry_gates({'30': legacy_invariant}, trajectory)
    for name, s in scales.items():
        gates[name]['reference_radius_RMS_at_least_1e_minus4'] = \
            invariant[name]['geometry']['radius'] / np.sqrt(start.shape[1]) >= 1e-4
        construction=invariant[name]['radial_construction']
        gates[name]['radial_starts_within_hidden_bounds']=construction['hidden_bound_excess_count']==0
        gates[name]['radial_directions_nondegenerate']=construction['degenerate_normal_count']==0
    candidate = all(all(g.values()) for g in gates.values())
    result = dict(regime=regime, reference_neural_updates=30, future_neural_updates=30,
        reference_map_steps=30 if regime == 'constant' else 15, future_map_steps=horizon,
        input_law='one complete constant input' if regime == 'constant' else 'forward then backward PA; fixed stroboscopic section',
        trajectory=trajectory, invariance=invariant, gates=gates,
        conditional_geometry_candidate=bool(candidate), broad_normal=broad,
        tested_normal_recovery=all(x['physical_conclusive'] and x['all_normal_directions_pass'] for x in broad.values()),
        qualification='Float64 screen only until native-float32 confirmation; no global attractor claim.')
    archives.update({f'z_{t}': states[t] for t in times})
    return result, states, archives


@torch.no_grad()
def jvp_check(transition, reference, scale, regime, *, anchors=ANCHORS):
    """Exact derivative directions along the evolving family; one finite check."""
    with patch.object(normals, 'NORMAL_COUNT', 3):
        _, metadata = normals.make_perturbations(reference, scale, anchors=anchors, seed=NORMAL_SEED)
    indices = np.asarray(metadata['anchor_indices'])
    # Tangent first is the coordinate for decomposition; seven named normals follow.
    order = [len(metadata['names'])-1, *range(len(metadata['names'])-1)]
    vectors = metadata['scaled_directions'][order].transpose(1, 2, 0) * np.asarray(scale)[None, :, None]
    directions = torch.tensor(vectors, dtype=torch.float64)
    state = torch.tensor(reference[indices], dtype=torch.float64)
    initial_state, initial_directions = state.clone(), directions.clone()
    for k in range(30):
        drive = transition.drives[k % len(transition.inputs):k % len(transition.inputs)+1]
        state, directions = advance(transition.cell, drive, state, directions)
    scaled = directions / torch.as_tensor(scale)[None, :, None]
    tangent = scaled[:, :, 0]
    rows = []
    for j, source in enumerate(order[1:], 1):
        phase, residual = decomposition(scaled[:, :, j], tangent)
        rows.append(dict(name=metadata['names'][source], transverse_gain=describe(residual.numpy()),
                         induced_phase_per_unit_tangent=describe(phase.abs().numpy()),
                         total_gain=describe(scaled[:, :, j].norm(dim=1).numpy())))
    epsilon = 1e-4
    initial = initial_directions[0].T
    plus, minus = initial_state[:1]+epsilon*initial, initial_state[:1]-epsilon*initial
    for k in range(30):
        plus, minus = transition(plus, k), transition(minus, k)
    fd = (plus-minus)/(2*epsilon)
    exact = directions[0].T
    absolute = float((fd-exact).abs().max())
    relative = float((fd-exact).norm()/exact.norm().clamp_min(1e-30))
    return dict(regime=regime, neural_updates=30, directions=rows,
        tangent_gain=describe(tangent.norm(dim=1).numpy()), finite_difference_epsilon=epsilon,
        finite_difference_max_absolute_error=absolute, finite_difference_relative_error=relative,
        finite_difference_qualified=bool(relative < .005),
        scope='Seven tested normal directions, not all511 transverse directions; strong collapse can make a relative finite-difference check uninformative.')


def tangent_overlap(vectors, tangent):
    vectors = np.asarray(vectors)
    real = np.column_stack((vectors.real, vectors.imag)) if np.iscomplexobj(vectors) else vectors
    u, singular, _ = linalg.svd(real, full_matrices=False, check_finite=False)
    keep = singular > (singular[0]*1e-10 if len(singular) else 0)
    length = np.linalg.norm(tangent)
    return None if not keep.any() or length < 1e-12 else float(np.sum((u[:, keep].T @ (tangent/length))**2))


def matrix_spectra(matrix, tangent, scale):
    """Local eigenpairs and singular vectors; raw/scaled similarity is explicit."""
    a = np.asarray(matrix, float)
    values, left, right = linalg.eig(a, left=True, right=True, check_finite=False)
    order = np.argsort(-np.abs(values), kind='stable')
    selected = list(order[:min(10, len(order))])
    for j in list(selected):
        if abs(values[j].imag) > 1e-10:
            partner = int(np.argmin(np.abs(values-values[j].conjugate())))
            if abs(values[partner]-values[j].conjugate()) > 1e-8: raise AssertionError('Missing conjugate eigenpair')
            if partner not in selected: selected.append(partner)
    result, archive = {}, dict(eigenvalues=values)
    for name, s in (('raw', np.ones(len(a))), ('relative_rms', np.asarray(scale))):
        scaled = a * s[None, :] / s[:, None]
        rv, lv = right/s[:, None], left*s[:, None]
        rv /= np.maximum(np.linalg.norm(rv, axis=0), 1e-30)
        lv /= np.maximum(np.linalg.norm(lv, axis=0), 1e-30)
        residual = np.linalg.norm(scaled@rv-rv*values[None], axis=0)/max(linalg.norm(scaled), 1e-30)
        if residual.max() > 1e-8: raise AssertionError('Eigenpair residual too large')
        condition = 1/np.maximum(np.abs(np.sum(lv.conj()*rv, axis=0)), 1e-30)
        _, singular, vh = linalg.svd(scaled, full_matrices=False, check_finite=False)
        # Directly recompute scaled eigenvalues to audit similarity, using nearest
        # spectral-set distance so arbitrary repeated-eigenvalue order is harmless.
        scaled_values = linalg.eigvals(scaled, check_finite=False)
        mismatch = float(np.max(np.min(np.abs(scaled_values[:, None]-values[None]), axis=1)))
        result[name] = dict(spectral_radius=float(np.abs(values).max()),
            fraction_eigenvalues_above_one=float(np.mean(np.abs(values)>1)),
            fraction_eigenvalues_above_1_01=float(np.mean(np.abs(values)>1.01)),
            fraction_complex=float(np.mean(np.abs(values.imag)>1e-8)),
            ordered_eigenvalue_modulus=np.sort(np.abs(values))[::-1].tolist(),
            singular_values=singular.tolist(), largest_singular_value=float(singular[0]),
            eigenpair_relative_residual_max=float(residual.max()), eigenpair_condition_max=float(condition.max()),
            coordinate_similarity_eigenvalue_set_error=mismatch,
            top_eigenvector_indices=selected,
            top_eigenvector_tangent_overlap=tangent_overlap(rv[:, selected], np.asarray(tangent)/s),
            top_singular_vector_tangent_overlap=tangent_overlap(vh[:10].T, np.asarray(tangent)/s),
            cutoff_modulus_tie=bool(len(order)>10 and abs(abs(values[order[9]])-abs(values[order[10]]))<1e-8),
            caveat='Local eigenvectors of a possibly non-normal derivative; nearly degenerate modes do not have individually stable meanings.')
        archive.update({f'{name}_top_right_eigenvectors': rv[:, selected],
                        f'{name}_top_right_singular_vectors': vh[:10].T,
                        f'{name}_singular_values': singular,
                        f'{name}_eigenpair_conditions': condition})
    return result, archive


def local_spectra(transition, states, scale, regime):
    net = SimpleNamespace(lstm=transition.cell)
    records, archive = {}, {}
    for t in ((0, 30) if regime == 'constant' else (30,)):
        curve = states[t]
        h, c = torch.as_tensor(curve[0], dtype=torch.float64).chunk(2)
        x = transition.inputs[0]
        first = local_jacobian(net, x, h, c)
        check = autograd_check(net, x, h, c, first)
        if regime == 'forward_backward':
            next_state = transition(torch.cat((h,c))[None], 0)[0]
            second = local_jacobian(net, transition.inputs[1], *next_state.chunk(2))
            check = max(check, autograd_check(net, transition.inputs[1], *next_state.chunk(2), second))
            matrix = second @ first
        else:
            matrix = first
        tangent = (curve[1]-curve[-1])/(4*np.pi/len(curve))
        records[str(t)], saved = matrix_spectra(matrix.numpy(), tangent, scale)
        records[str(t)]['autograd_max_absolute_error'] = check
        records[str(t)]['map_neural_updates'] = 1 if regime == 'constant' else 2
        archive.update({f't{t}_{key}': value for key, value in saved.items()})
    return records, archive


def native_occupancy(data, context, states, scale, theta):
    """Strict passive support; no fitting, recentering or context replacement."""
    slot, rt = context['selected_slot'], context['runtime']
    pos = array(rt['obs']['pos'][slot]); velocity = array(rt['obs']['vel'][slot])
    travel = array(rt['obs']['dir'][slot]); body = float(rt['obs']['hd'][slot])
    goal_pos = array(rt['environment_tensors']['goal'][slot])
    episode, onset = context['episode_id'], context['global_step']
    prior = np.flatnonzero((data['episode_id']==episode) & data['known'] & (data['global_step']<onset))
    if not len(prior):
        return dict(qualified=False, broad_native_support=False, reason='No archived known pre-onset row identifies the cached goal pickup'), {}
    reference = prior[np.argmax(data['global_step'][prior])]
    same_pickup = np.ones(len(data['h']), bool)
    for key in ('goal_pickup_pos', 'goal_pickup_dir', 'goal_pickup_vel'):
        same_pickup &= np.all(data[key] == data[key][reference], axis=1)
    angle = np.arctan2(data['travel_dir'][:,1], data['travel_dir'][:,0])
    native_angle = np.arctan2(travel[1], travel[0])
    offset = wrap(body-native_angle)
    checks = dict(same_episode=data['episode_id']==episode,
        later_blackout=data['global_step']>onset+60, known_blind=data['known'] & data['blind'],
        same_goal_pickup=same_pickup, same_goal_position=np.all(data['goal_pos']==goal_pos, axis=1),
        same_PA=data['prev_action']==int(rt['pa'][slot]), zero_PR=data['prev_reward']==0,
        nearby_position=np.linalg.norm(data['pos']-pos, axis=1)<=.05,
        matched_speed=np.abs(np.linalg.norm(data['velocity'],axis=1)-np.linalg.norm(velocity))<=.05,
        matched_body_travel_offset=np.abs(wrap(data['body_heading']-angle-offset))<=np.deg2rad(5),
        valid_age=(data['phase_age']>=1)&(data['phase_age']<=60))
    mask = np.logical_and.reduce(list(checks.values()))
    selected = np.flatnonzero(mask)
    intervals = np.unique(data['global_step'][selected]-data['phase_age'][selected]+1)
    physical_phase = wrap(angle[selected]-native_angle)
    angular_coverage = coverage(physical_phase)
    adequate = len(selected)>=32 and len(intervals)>=2 and angular_coverage is not None and angular_coverage>=.8
    result = dict(qualified=bool(adequate), compatible_rows=len(selected), later_blackouts=len(intervals),
        physical_phase_coverage=angular_coverage, candidate_goal_reference_row=int(reference),
        match_rule='Exact archived native pickup pos/dir/vel tuple: identical deterministic cached message provenance, avoiding cross-backend last-bit code differences.',
        individual_filter_counts={k:int(v.sum()) for k,v in checks.items()}, metrics={},
        scope='Broad passive support only; insufficient coverage does not invalidate selective causal effects at genuine native anchors.')
    saved = dict(row_indices=selected, physical_relative_phase=physical_phase)
    for name, s in (('raw',np.ones(len(scale))),('relative_rms',np.asarray(scale))):
        residuals, errors, phases = [np.empty(len(selected)) for _ in range(3)]
        for age in np.unique(data['phase_age'][selected]):
            positions=np.flatnonzero(data['phase_age'][selected]==age);rows=selected[positions]
            curve=states[int(age)]/s
            z=np.column_stack((data['h'][rows],data['c'][rows]))/s
            distance,phase=semantic_nearest(z,curve,theta)
            radius=float(np.sqrt(np.mean(np.sum((curve-curve.mean(0))**2,axis=1))))
            residuals[positions]=distance/max(radius,1e-30)
            errors[positions]=np.abs(wrap(phase-physical_phase[positions]));phases[positions]=phase
        metric = dict(residual_over_radius=describe(residuals) if len(residuals) else None,
                      semantic_phase_mae_degrees=float(np.rad2deg(errors).mean()) if len(errors) else None)
        metric['passes'] = bool(adequate and metric['residual_over_radius']['p90']<=.05
                                and metric['semantic_phase_mae_degrees']<=15)
        result['metrics'][name] = metric
        saved.update({f'{name}_residual_over_radius':np.asarray(residuals),
                      f'{name}_phase':np.asarray(phases),f'{name}_phase_error':np.asarray(errors)})
    result['broad_native_support'] = all(m['passes'] for m in result['metrics'].values())
    if not adequate: result['reason'] = 'Insufficient prespecified row/interval/phase coverage; no relaxed matching'
    return result, saved


def analyze_family(policy, family, *, native_data=None, arc_probes=ARC_PROBES, anchors=ANCHORS, spectra=True):
    """Bounded within-model assay; no checkpoint reads in this pure entry point."""
    if not family['metadata']['qualified']:
        return dict(qualified=False, reason='Family adapter numerical qualification failed', family=family['metadata']), {}
    start, scale, theta = array(family['states']), array(family['scale']), array(family['theta'])
    if start.shape[1] != 2*policy.lstm.hidden_size: raise ValueError('Family state dimension does not match LSTM')
    laws = input_laws(family['dark_inputs'], getattr(policy, 'n_code', 512), getattr(policy, 'n_actions', 6))
    records, archive, occupancy = {}, {}, None
    for regime in REGIMES:
        transition = FrozenTransition(policy.lstm, laws[regime])
        record, states, saved = analyze_regime(transition, start, scale, regime, arc_probes=arc_probes, anchors=anchors)
        record['jvp'] = {name:jvp_check(transition,states[30],s,regime,anchors=anchors)
                         for name,s in [('raw',np.ones(len(scale))),('relative_rms',scale)]}
        native = FrozenTransition(policy.lstm, laws[regime], native_float32=True)
        native_states = evolve(native, start, 60)
        error = {}
        for name,s in [('raw',np.ones(len(scale))),('relative_rms',scale)]:
            maximum = max(float(np.linalg.norm((native_states[t]-states[t])/s,axis=1).max()) for t in range(61))
            radius = flow.curve_geometry(states[30]/s)['radius']
            multiplier = record['broad_normal'][name]['perturbation_design']['shared_amplitude_multiplier']
            kick = .02*radius*multiplier
            error[name] = dict(maximum_full_state_distance=maximum, reference_radius=radius, actual_broad_kick=kick,
                relative_to_radius=maximum/max(radius,1e-30), relative_to_kick=maximum/max(kick,1e-30),
                qualified=bool(radius>0 and kick>0 and maximum<=.01*radius and maximum<=.01*kick))
        record['native_float32_comparison'] = error
        if record['conditional_geometry_candidate']:
            check, _, _ = analyze_regime(native,start,scale,regime,arc_probes=arc_probes,anchors=anchors)
            record['native_float32_confirmation'] = dict(gates=check['gates'],
                conditional_geometry_candidate=check['conditional_geometry_candidate'],
                tested_normal_recovery=check['tested_normal_recovery'])
        else:
            record['native_float32_confirmation'] = dict(run=False, reason='Float64 geometry screen failed; native trajectory mismatch still reported')
        record['conditional_ring_candidate'] = bool(record['conditional_geometry_candidate']
            and all(x['qualified'] for x in error.values())
            and record['native_float32_confirmation'].get('conditional_geometry_candidate',False)
            and all(x['finite_difference_qualified'] for x in record['jvp'].values()))
        record['physical_recovery_construction_qualified'] = all(
            (record['invariance'][name]['radial_construction']['hidden_bound_excess_count']==0
             and record['invariance'][name]['radial_construction']['degenerate_normal_count']==0)
            for name in ('raw','relative_rms'))
        record['conditional_ring_candidate'] &= record['physical_recovery_construction_qualified']
        record['tested_normal_recovery_confirmed'] = bool(record['conditional_ring_candidate']
            and record['tested_normal_recovery']
            and record['native_float32_confirmation'].get('tested_normal_recovery',False))
        if spectra:
            record['local_spectra'], spectrum_arrays = local_spectra(transition, states, scale, regime)
            saved.update({f'spectra_{k}':v for k,v in spectrum_arrays.items()})
        if regime=='constant' and native_data is not None:
            occupancy, occupancy_arrays = native_occupancy(native_data, family['original_context'],states,scale,theta)
            saved.update({f'occupancy_{k}':v for k,v in occupancy_arrays.items()})
        archive.update({f'{regime}_{k}':v for k,v in saved.items()})
        records[regime] = record
    return dict(qualified=True, family=family['metadata'], regimes=records, native_occupancy=occupancy,
                scope='Finite-horizon conditional dynamics; broad passive occupancy and native causal behavior remain distinct.'), archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True, help='Authenticated memory_states directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=3600)
    args = parser.parse_args(argv)
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status']!='frozen': raise RuntimeError('Analysis protocol must be frozen before model execution')
    hashes = protocol.get('source_sha256',{})
    if not hashes or not all(name in hashes for name in SOURCES):
        raise RuntimeError('Analysis protocol must freeze all required source files')
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest: raise RuntimeError(f'Analysis source changed: {name}')
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError('Analysis output must be empty')
    from collect_simple_memory_states import load_final
    from simple_memory_family import build_family
    from code import train_memory as training
    policy,cfg,proof = load_final(args.run,args.device,args.threads)
    if cfg['seed'] not in (1,2,3,4) or cfg['cond'] not in ('gc100','pgc33') or cfg['task_max_gap'] not in (0,60):
        raise ValueError('Model outside frozen16-model cohort')
    summary = json.loads((args.states/'summary.json').read_text())
    if summary['checkpoint_sha256']!=proof['checkpoint_sha256']: raise AssertionError('State/checkpoint provenance mismatch')
    expected=dict(seed=741022,episodes=128,n_envs=32,max_gap=60,stride=3,samples=57600)
    if not summary.get('complete') or any(summary.get(k)!=v for k,v in expected.items()):
        raise AssertionError('Collection does not match frozen native128-episode sampling')
    for name,digest in summary['source_sha256'].items():
        if sha(ROOT/name)!=digest: raise AssertionError(f'Collection source changed: {name}')
    for key in ('array_file','contexts_file'):
        if sha(args.states/summary[key]['path'])!=summary[key]['sha256']: raise AssertionError('Changed state artifact')
    payload = torch.load(args.states/'onset_contexts.pt',map_location='cpu',weights_only=False)
    if payload['checkpoint_sha256']!=proof['checkpoint_sha256']: raise AssertionError('Context/checkpoint provenance mismatch')
    contexts = payload['contexts']
    if len(contexts)>4: raise ValueError('Too many selected contexts')
    with np.load(args.states/'states.npz',allow_pickle=False) as archive:
        native_data = {key:archive[key] for key in archive.files}
    before = training.policy_hash(policy)
    args.out.mkdir(parents=True,exist_ok=True)
    deadline = time.monotonic()+args.max_seconds
    rows = []
    for index, context in enumerate(contexts):
        if time.monotonic()>=deadline: raise TimeoutError('Dynamics deadline; partial context artifacts retained without complete summary')
        family = build_family(policy,cfg,context,args.device,phases=PHASES)
        with threadpool_limits(limits=args.threads):
            record, arrays = analyze_family(policy,family,native_data=native_data)
        target = args.out/f'context{index}.npz'
        with target.with_suffix('.npz.tmp').open('wb') as handle: np.savez_compressed(handle,**arrays)
        target.with_suffix('.npz.tmp').replace(target)
        record.update(index=index,episode_id=context['episode_id'],global_step=context['global_step'],
                      arrays=dict(path=target.name,sha256=sha(target),bytes=target.stat().st_size))
        save_json(args.out/f'context{index}.json',record)
        rows.append(record)
        print(json.dumps(dict(context=index,qualified=record['qualified'])),flush=True)
    if training.policy_hash(policy)!=before: raise AssertionError('Analysis mutated policy')
    save_json(args.out/'summary.json',dict(complete=True,condition=cfg['cond'],training_seed=cfg['seed'],
        training_task_max_gap=cfg['task_max_gap'],contexts=rows,**proof,
        analysis_protocol_sha256=sha(PROTOCOL),source_sha256={name:sha(ROOT/name) for name in SOURCES},
        state_summary_sha256=sha(args.states/'summary.json'),policy_unchanged=True,
        numerical_settings=dict(family_device=args.device,geometry='float64',native_confirmation='float32 original policy device',threads=args.threads)))


if __name__=='__main__': main()
