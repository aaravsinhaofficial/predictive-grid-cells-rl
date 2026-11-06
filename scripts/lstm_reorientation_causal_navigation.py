"""Pre-glimpse state/cue interventions and native free navigation.

Separate prospective wrapper: no change to the frozen task or geometry assay.
Only the original actor controls the thirty free actions. All angle estimates
are offline measurements and never enter the policy or its input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from code.common import build_policy, build_population
from code.env import ArenaEnv
from lstm_heading_memory_probe import sha, save, wrap
from lstm_heading_curve_flow import curve_geometry
from lstm_travel_cue_pulse import semantic_nearest
from lstm_endogenous_curve_recovery import advance, describe

PHASES = 192
TIMES = (0, 1, 5, 15, 30, 45, 60)
CUES = ('native', 'ordinary', 'plus60', 'minus60')
EDITS = ('none', 'plus30', 'minus30', 'radial_plus', 'radial_minus',
         'isotropic_plus', 'isotropic_minus')
RADIAL_EDITS = ('none', 'radial_plus', 'radial_minus', 'isotropic_plus', 'isotropic_minus')
RANDOM_SEED = 188201
SOURCES = ('scripts/lstm_reorientation_causal_navigation.py',
    'scripts/lstm_reorientation_mechanism.py', 'scripts/lstm_travel_cue_pulse.py',
    'scripts/lstm_heading_curve_flow.py', 'scripts/lstm_endogenous_curve_recovery.py',
    'code/common.py', 'code/nets.py', 'code/env.py')


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def state_edits(curve, scale, subset, include_phase=True):
    """Full-state edits, with radial direction explicitly normal to the curve."""
    x = np.asarray(curve, dtype=np.float64)/scale
    centered = x-x.mean(0)
    radius = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    tangent = np.roll(x, -1, 0)-np.roll(x, 1, 0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(tangent_norm, 1e-30)
    radial = centered-np.sum(centered*tangent, axis=1, keepdims=True)*tangent
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    result = {'none':np.asarray(curve)[subset].copy()}
    audit = dict(radius=float(radius), kick_norm=float(.02*radius), invalid_edits={},
        perturbation_metric='Euclidean after specified full-state scaling; isotropic controls are not tangent-projected')
    requested = EDITS if include_phase else RADIAL_EDITS
    if radius < 1e-12:
        audit['invalid_edits'] = {name:'Collapsed source family; no nonzero state intervention defined' for name in requested if name != 'none'}
        return result, audit
    radial /= np.maximum(radial_norm, 1e-30)
    isotropic = np.random.default_rng(RANDOM_SEED).standard_normal(x.shape)
    isotropic /= np.linalg.norm(isotropic, axis=1, keepdims=True)
    kick = .02*radius
    if include_phase:
        assert len(curve)%12 == 0
        result.update(plus30=np.roll(curve, -len(curve)//12, 0)[subset],
                      minus30=np.roll(curve, len(curve)//12, 0)[subset])
    for name, vector in [('radial', radial), ('isotropic', isotropic)]:
        if name == 'radial' and (np.any(radial_norm[subset] < 1e-12) or np.any(tangent_norm[subset] < 1e-12)):
            for sign in ('plus','minus'): audit['invalid_edits'][name+'_'+sign] = 'At least one prespecified phase has no nonzero transverse radial direction'
            continue
        result[name+'_plus'] = (x[subset]+kick*vector[subset])*scale
        result[name+'_minus'] = (x[subset]-kick*vector[subset])*scale
    audit['maximum_radial_tangent_dot'] = float(np.abs(np.sum(radial[subset]*tangent[subset], 1)).max())
    return result, audit


def step(net, state, inp):
    drive = torch.nn.functional.linear(inp, net.lstm.weight_ih, net.lstm.bias_ih)
    return advance(net.lstm, drive, state)


@torch.no_grad()
def reference_families(net, archive):
    pre = torch.tensor(archive['pre_cue_state'], dtype=torch.float64)
    dark = torch.tensor(archive['ordinary_input'], dtype=torch.float64)
    refs = {}
    for cue in CUES:
        state = step(net, pre, torch.tensor(archive[cue+'_glimpse_inputs'], dtype=torch.float64))
        refs[cue] = {0:state.numpy().copy()}
        for t in range(1, 61):
            state = step(net, state, dark)
            if t in TIMES: refs[cue][t] = state.numpy().copy()
    state = torch.zeros_like(pre)
    refs['native_reset'] = {0:state.numpy().copy()}
    for t in range(1, 61):
        state = step(net, state, dark)
        if t in TIMES: refs['native_reset'][t] = state.numpy().copy()
    return refs


def phase_metrics(points, reference, own_reference, theta, expected, scale, radius):
    distance, phase = semantic_nearest(points/scale, reference/scale, theta)
    own_distance, own_phase = semantic_nearest(points/scale, own_reference/scale, theta)
    error = wrap(phase-expected)
    ordered = np.sort(np.mod(phase, 2*np.pi))
    coverage = 1-np.diff(np.r_[ordered, ordered[0]+2*np.pi]).max()/(2*np.pi)
    own_centered = (own_reference-own_reference.mean(0))/scale
    own_radius = float(np.sqrt(np.mean(np.sum(own_centered**2, axis=1))))
    valid = radius >= 1e-12
    own_valid = own_radius >= 1e-12
    return dict(phase_estimate_valid=valid,
        phase_mae_deg=float(np.rad2deg(np.abs(error)).mean()) if valid else None,
        signed_phase_bias_deg=float(np.rad2deg(np.angle(np.exp(1j*error).mean()))) if valid else None,
        phase_error_concentration=float(abs(np.exp(1j*error).mean())) if valid else None,
        phase_coverage=float(coverage) if valid else None,
        off_correct_curve_over_radius=describe(distance/radius) if valid else None,
        own_branch_phase_mae_deg=float(np.rad2deg(np.abs(wrap(own_phase-expected))).mean()) if own_valid else None,
        own_reference_radius=own_radius,
        own_reference_collapsed=own_radius < 1e-12,
        off_own_curve=describe(own_distance),
        off_own_curve_over_radius=describe(own_distance/own_radius) if own_valid else None)


def geometry_report(net, states, refs, cue, theta, subset, scales, geometries):
    report = {}
    for t, edits in states.items():
        rows = {}
        native = refs['native'][t]; own = refs[cue][t]
        p0 = net.pi(torch.tensor(own[subset, :256])).softmax(-1).detach().numpy()
        for edit, z in edits.items():
            metrics = {}
            for metric, scale in scales.items():
                metrics[metric] = phase_metrics(z, native, own, theta, theta[subset], scale,
                    geometries[(t, metric)]['radius'])
            prob = net.pi(torch.tensor(z[:, :256])).softmax(-1).detach().numpy()
            rows[edit] = dict(metrics=metrics,
                actor_total_variation_from_own_unperturbed=describe(.5*np.abs(prob-p0).sum(1)))
        report[str(t)] = rows
    return report


def navigation_draws(n, seed=RANDOM_SEED):
    motor_generator = torch.Generator().manual_seed(seed)
    motor = torch.stack([torch.stack([torch.randn(n, generator=motor_generator) for _ in range(3)]) for _ in range(30)])
    uniform = torch.rand(30, n, generator=torch.Generator().manual_seed(seed+1))
    return motor, uniform


def physical_free_start(cfg, theta, endpoint, goal):
    """Actual native sixty-turn physics after the task's endpoint brake."""
    n = len(theta)
    env = ArenaEnv(n, half=cfg['arena_half'], episode_len=10000,
        action_repeat=cfg['action_repeat'], v_max=cfg['v_max'], rot_max=cfg['rot_max'],
        goal_radius=cfg['goal_radius'], vision='none', img=1, img_noise=0.,
        noise_v=0., noise_r=0., stagger=False, paired=True, pair_seed=RANDOM_SEED,
        seed=RANDOM_SEED, device='cpu')
    env.reset_all(); env.pos[:] = torch.tensor(endpoint); env.goal[:] = torch.tensor(goal)
    env.hd[:] = torch.tensor(theta, dtype=torch.float32)
    env.mdir[:] = torch.stack((env.hd.cos(), env.hd.sin()), 1)
    env.spawn[:] = env.pos; env.vf.zero_(); env.vs.zero_(); env.vr.zero_()
    for _ in range(60):
        _, reward, done, info = env.step(torch.zeros(n, dtype=torch.long))
        assert not bool(reward.any() or done.any() or info['hit_wall'].any())
    assert torch.equal(env.pos, torch.tensor(endpoint).expand_as(env.pos))
    env.noise_v = .03; env.noise_r = .15
    return env


@torch.no_grad()
def free_navigation(net, start, dark, cfg, theta, endpoint, goal, motor, uniform):
    """Use pi(h) first; after each move, process the next input exactly once."""
    env = physical_free_start(cfg, theta, endpoint, goal)
    n = len(theta); state = torch.tensor(start, dtype=torch.float64)
    active = torch.ones(n, dtype=torch.bool); first_hit = torch.full((n,), -1, dtype=torch.long)
    distance0 = (env.pos-env.goal).norm(dim=1); distance = distance0.clone()
    path_length = torch.zeros(n); first_progress = torch.zeros(n)
    actions = torch.full((30, n), -1, dtype=torch.long)
    path = torch.empty(31, n, 2); path[0] = env.pos
    motor_index = 0
    def fixed_motor(*shape):
        nonlocal motor_index
        assert shape == (n,) and motor_index < 90
        value = motor[motor_index//3, motor_index%3]; motor_index += 1
        return value
    env._randn = fixed_motor  # Only supply draws; unmodified native step equations.
    first_probs = net.pi(state[:, :256]).softmax(-1).numpy().copy()
    processed_free_inputs = 0
    for t in range(30):
        probs = net.pi(state[:, :256]).softmax(-1)
        action = (uniform[t, :, None] > probs.cumsum(1)).sum(1).clamp_max(5)
        actions[t] = torch.where(active, action, torch.full_like(action, -1))
        before = env.pos.clone()
        _, reward, done, info = env.step(action)
        assert not bool(done.any())
        # hit_pos is the pre-respawn position; stopped windows never include respawn.
        after = info['hit_pos']; next_distance = (after-env.goal).norm(dim=1)
        if t == 0: first_progress = distance-next_distance
        path_length += active*(after-before).norm(dim=1)
        distance = torch.where(active, next_distance, distance)
        path[t+1] = torch.where(active[:, None], after, path[t])
        hit = active & info['hit']; first_hit[hit] = t+1; active &= ~hit
        if t < 29:
            inp = dark.expand(n, -1).clone()
            inp[:, 1280:1286] = torch.nn.functional.one_hot(action, 6).double()
            inp[:, -1] = 0.  # Reward terminates a window before its next input.
            new = step(net, state, inp); state = torch.where(active[:, None], new, state)
            processed_free_inputs += 1
    assert motor_index == 90 and processed_free_inputs == 29
    return dict(hit_fraction=float((first_hit > 0).float().mean()),
        goals=int((first_hit > 0).sum()), windows=n, zero_hit_windows=int((first_hit < 0).sum()),
        first_action_goalward_progress_m=describe(first_progress.numpy()),
        total_goalward_progress_m=describe((distance0-distance).numpy()),
        path_length_m=describe(path_length.numpy()),
        processed_inputs_after_first_free_action=processed_free_inputs), dict(
            actions=actions.numpy(), path=path.numpy(), first_hit_step=first_hit.numpy(),
            first_action_progress_m=first_progress.numpy(), total_progress_m=(distance0-distance).numpy(),
            path_length_m=path_length.numpy(), first_action_probabilities=first_probs)


@torch.no_grad()
def follow(net, start, dark, cue_input=None, reset=False, anchor=0):
    state = torch.tensor(start, dtype=torch.float64)
    if cue_input is not None: state = step(net, state, cue_input)
    if reset: state.zero_()
    states = {anchor:state.numpy().copy()}
    for t in range(anchor+1, 61):
        state = step(net, state, dark)
        if t in TIMES: states[t] = state.numpy().copy()
    return states


def analyze(net, cfg, archive, out):
    theta = archive['theta']; n = len(theta); assert n >= PHASES and n%PHASES == 0
    subset = np.arange(0, n, n//PHASES)
    assert np.max(np.abs(wrap(np.roll(theta, -1)-theta)-2*np.pi/n)) < 1e-4
    scales = dict(raw=np.ones(512), relative_rms=archive['scale'])
    dark = torch.tensor(archive['ordinary_input'], dtype=torch.float64)
    pop = build_population(cfg, 'cpu'); allowed = np.zeros(1287, dtype=bool)
    allowed[256+np.asarray(pop.pgc_idx, dtype=int)] = True
    native_cue = archive['native_glimpse_inputs']
    for cue in CUES:
        assert np.array_equal(archive[cue+'_glimpse_inputs'][:, ~allowed], native_cue[:, ~allowed])
    refs = reference_families(net, archive)
    geometries = {(t,m):curve_geometry(refs['native'][t]/s) for t in TIMES for m,s in scales.items()}
    motor, uniform = navigation_draws(PHASES)
    report = {}; arrays = {}; pre = archive['pre_cue_state']
    for construction, scale in scales.items():
        edits, audit = state_edits(pre, scale, subset)
        for cue in CUES+('native_reset',):
            source_cue = cue.replace('_reset', '')
            inp = torch.tensor(archive[source_cue+'_glimpse_inputs'][subset], dtype=torch.float64)
            records = {t:{} for t in TIMES}; navigation = {}
            for edit, start in edits.items():
                flow = follow(net, start, dark, inp, reset=cue.endswith('_reset'))
                for t,z in flow.items(): records[t][edit] = z
                stats, values = free_navigation(net, flow[60], dark, cfg, theta[subset],
                    archive['endpoint'], archive['goal_center'], motor, uniform)
                navigation[edit] = stats
                for key,value in values.items(): arrays[f'{construction}__{cue}__{edit}__{key}'] = value
                arrays[f'{construction}__{cue}__{edit}__final_state'] = flow[60]
            report[construction+'__'+cue] = dict(anchor='pre_glimpse', perturbations=audit,
                geometry=geometry_report(net, records, refs, cue, theta, subset, scales, geometries),
                navigation=navigation)
            if cue == 'native_reset':
                for t in TIMES:
                    assert all(np.array_equal(z, records[t]['none']) for z in records[t].values())
                for edit in navigation:
                    for key in ('actions','path','first_hit_step','first_action_probabilities'):
                        assert np.array_equal(arrays[f'{construction}__native_reset__{edit}__{key}'],
                                              arrays[f'{construction}__native_reset__none__{key}'])
        # Explicit later radial sensitivity; cue is already withdrawn for one second.
        for cue in CUES:
            edits, audit = state_edits(refs[cue][15], scale, subset, include_phase=False)
            records = {t:{} for t in TIMES if t >= 15}; navigation = {}
            for edit,start in edits.items():
                flow = follow(net, start, dark, anchor=15)
                for t,z in flow.items(): records[t][edit] = z
                stats,values = free_navigation(net, flow[60], dark, cfg, theta[subset],
                    archive['endpoint'], archive['goal_center'], motor, uniform)
                navigation[edit] = stats
                for key,value in values.items(): arrays[f'{construction}__{cue}_radial15__{edit}__{key}'] = value
            report[construction+'__'+cue+'_radial15'] = dict(anchor='15_updates_after_glimpse', perturbations=audit,
                geometry=geometry_report(net, records, refs, cue, theta, subset, scales, geometries), navigation=navigation)
    audit = dict(phases=PHASES, dense_reference_phases=n, predictive_current_slots=int(allowed.sum()),
        other_cue_inputs_byte_identical=True, actual_travel_theta=True,
        action_uniform_sha256=array_hash(uniform.numpy()), motor_draw_sha256=array_hash(motor.numpy()),
        state_noise_during_assay=False, forced_delay_noise=False, free_motor_noise_v=.03, free_motor_noise_r=.15,
        physical_start='Same native noiseless60-LEFT evolution from endpoint brake for every branch; native noisy free physics thereafter',
        first_free_action='pi(h) from t60 state; exactly29 further recurrent updates for30 actual free actions',
        reward_stop='Stop every window at first actual hit; keep zero-hit windows and first-hit position before respawn',
        post_glimpse_reset_erases_all_state_edit_differences_exactly=True,
        grid_sham=bool(np.array_equal(native_cue, archive['ordinary_glimpse_inputs'])),
        native_cue_float32_to_float64_replay_max=float(np.abs(refs['native'][0]-archive['native_states'][0]).max()))
    if audit['grid_sham']:
        for cue in CUES: assert all(np.array_equal(refs[cue][t], refs['ordinary'][t]) for t in TIMES)
        for construction in scales:
            for anchor in ('','_radial15'):
                for edit in report[construction+'__ordinary'+anchor]['navigation']:
                    for key in ('actions','path','first_hit_step','first_action_probabilities'):
                        reference = arrays[f'{construction}__ordinary{anchor}__{edit}__{key}']
                        for cue in CUES:
                            assert np.array_equal(reference, arrays[f'{construction}__{cue}{anchor}__{edit}__{key}'])
        audit['grid_behavior_sham_exact'] = True
    contrasts = {}
    for construction in scales:
        rows = {}
        for t in TIMES:
            rows[str(t)] = {}
            for measurement in scales:
                def phase_error(cue):
                    records = report[construction+'__'+cue]['geometry'][str(t)]
                    values = [records[edit]['metrics'][measurement]['phase_mae_deg'] if edit in records else None for edit in ('plus30','minus30')]
                    return float(np.mean(values)) if all(value is not None for value in values) else None
                ordinary_error, native_error = phase_error('ordinary'), phase_error('native')
                rows[str(t)][measurement] = dict(
                    ordinary_error_deg=ordinary_error, native_error_deg=native_error,
                    correct_cue_improvement_deg=ordinary_error-native_error if ordinary_error is not None and native_error is not None else None,
                    plus60_error_deg=phase_error('plus60'), minus60_error_deg=phase_error('minus60'))
        contrasts[construction] = rows
    save(out/'result.json', dict(audit=audit, reference_geometry={f'{t}__{m}':v for (t,m),v in geometries.items()},
        phase_repair_contrasts=contrasts, branches=report))
    np.savez_compressed(out/'navigation_paths.npz', theta=theta[subset], motor_draws=motor.numpy(),
        action_uniforms=uniform.numpy(), **arrays)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path); parser.add_argument('--states', type=Path)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    protocol = dict(version='task_native_pre_glimpse_rescue_navigation_v1',
        design_sha256=sha(ROOT/'docs/research/reorientation_mechanism_protocol.json'),
        review_addendum_sha256=sha(ROOT/'docs/research/reorientation_mechanism_review_addendum.md'),
        phases=PHASES, times=list(TIMES), edits=list(EDITS), cues=list(CUES)+['native_reset'],
        radial_anchor_updates=[0,15], radial_fraction=.02, free_actions=30, random_seed=RANDOM_SEED,
        construction_metrics=['raw','relative_rms'], measurement_metrics=['raw','relative_rms'],
        references='Dense768-point unperturbed correct-native family and each own-cue family; full512 closest segments; no decoder enters actor',
        invalid_interventions='Retain unperturbed and every defined edit; record missing radial/phase edits per construction/anchor/cue without discarding other branches. Collapsed reference curves have null phase estimates.',
        precision='Recompute all unperturbed cue references in float64 from saved pre-cue states; retain discrepancy from original float32 cue update',
        navigation='Noiseless native60-left delay from exactendpoint/brakedmotors to align saved neural states, then native free motor noise from shared predrawn tapes; original actor with shared inverseCDF uniforms; stop firstreward',
        fixed_cohort='Both final40,058,880-frame new-task checkpoints, both physically selected contexts each; no selection on candidate or reward',
        source_sha256={p:sha(ROOT/p) for p in SOURCES})
    manifest = args.out/'protocol.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == protocol
    else: save(manifest, protocol)
    if args.freeze_only: return
    assert args.checkpoint and args.checkpoint.name == 'final.pt' and args.states
    data = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert data['completed'] and data['original_frames'] == 1_000_000_000 and data['extra_frames'] == 40_058_880
    assert data['warmup_updates'] == 200 and data['total_updates'] == 2156
    expected = json.loads((ROOT/'docs/research/rl_blind_reorientation_protocol.json').read_text()); expected.pop('frozen_at')
    assert data['provenance'] == expected
    archive = np.load(args.states)
    assert str(archive['checkpoint_sha256']) == sha(args.checkpoint)
    assert len(archive['theta']) == 768 and np.array_equal(archive['times'], TIMES)
    producer = json.loads(args.states.with_name(args.states.name.replace('_states.npz', '.json')).read_text())
    assert producer['checkpoint_sha256'] == sha(args.checkpoint)
    assert producer['source_sha256'] == sha(ROOT/'scripts/lstm_reorientation_mechanism.py')
    assert data['config']['rollout'] == 160 and data['config']['seed'] == 1
    net = build_policy(data['config'], 'cpu').eval().requires_grad_(False)
    net.load_state_dict(data['policy']); net.double()
    save(args.out/'inputs.json', dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        state_archive=str(args.states), state_archive_sha256=sha(args.states)))
    try: audit = analyze(net, data['config'], archive, args.out)
    except ValueError as exc:
        save(args.out/'invalid_intervention.json', dict(reason=str(exc), preserve_context=True)); raise
    print(json.dumps(audit), flush=True)


if __name__ == '__main__': main()
