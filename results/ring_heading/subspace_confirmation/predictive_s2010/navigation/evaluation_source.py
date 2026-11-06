"""Closed-loop navigation with a fixed learned direction-drive projector.

The intervention receives observed encoder drives only. Physical heading and
position enter only the unchanged observation generator, simulator, and scoring.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from heading_direction_subspace import DriveInterventions
from ring_heading_model import load_checkpoint
from ring_task_model import wrap_angle
import evaluate_ring_heading_navigation as navigation


VARIANTS = ('native', 'ordinary', 'remove_direction', 'restore_direction',
            'sham0_restore', 'sham1_restore', 'sham2_restore',
            'sham0_remove', 'sham1_remove', 'sham2_remove')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def simulate(model, task, checkpoint, intervention, noise, position_offset=0.,
             capture_radius=.15):
    c, device = task.config, task.device
    length, episodes = noise['visible'].shape
    starts, goals = noise['starts'].clone(), noise['goals'].clone()
    starts[:, 0] += position_offset
    goals[:, 0] += position_offset
    zeros, ones = torch.zeros(episodes, device=device), torch.ones(episodes, device=device)
    base_heading = noise['initial_heading']
    h = torch.zeros(episodes, model.hidden_size, device=device)
    for t in range(c.initial_heading_steps):
        x, _ = navigation.observation_inputs(task, starts, base_heading, zeros,
            zeros, ones, ones, checkpoint['condition'], noise, t)
        h = model.step(x, h)
    nv = len(VARIANTS)
    state = h[None].expand(nv, -1, -1).clone()
    position = starts[None].expand(nv, -1, -1).clone()
    estimated = position.clone()
    heading = base_heading[None].expand(nv, -1).clone()
    angular, speed = torch.zeros_like(heading), torch.zeros_like(heading)
    first_arrival = torch.full_like(heading, length + 1, dtype=torch.long)
    traveled = torch.zeros_like(heading)
    holding = navigation.HoldingScores(heading, c.dt, length, capture_radius)
    heading_error = torch.zeros_like(heading)
    max_native_identity, max_ordinary_identity, max_blank_dose = 0., 0., 0.
    max_dose_relative = [0., 0., 0.]
    snapshots, archive = {}, {'starts': starts.cpu().numpy(), 'goals': goals.cpu().numpy()}
    trace = {name: [] for name in ('position', 'estimated_position', 'heading', 'decoded_heading')}
    reports = [round(sec / c.dt) for sec in (20, 40) if round(sec / c.dt) <= length]
    for t in range(length):
        xn, xg, measured = [], [], []
        for v in range(nv):
            args = (task, position[v], heading[v], angular[v], speed[v], noise['visible'][t], zeros)
            a, s = navigation.observation_inputs(*args, 'predictive', noise, t + c.initial_heading_steps)
            b, s2 = navigation.observation_inputs(*args, 'grid', noise, t + c.initial_heading_steps)
            assert torch.equal(s, s2)
            assert torch.equal(a[:, c.n_code:], b[:, c.n_code:])
            xn.append(a); xg.append(b); measured.append(s)
        xn, xg = torch.stack(xn), torch.stack(xg)
        drives, dose = intervention.paired_drives(xn, xg)
        drive = torch.stack([drives[name][v] for v, name in enumerate(VARIANTS)])
        if t == 0:
            a = model.step(xn[0], state[0])
            b = model.step(xg[1], state[1])
        state = intervention.step_from_drive(drive, state)
        if t == 0:
            max_native_identity = float((state[0] - a).abs().max())
            max_ordinary_identity = float((state[1] - b).abs().max())
        blank = noise['visible'][t] == 0
        if blank.any():
            max_blank_dose = max(max_blank_dose, float(dose['direction_norm'][:, blank].max()))
        for j in range(3):
            max_dose_relative[j] = max(max_dose_relative[j], float(dose[f'sham{j}_relative_dose_error'].max()))
        output = model.readout(state)
        decoded = torch.atan2(output[..., 1], output[..., 0])
        heading_error += wrap_angle(decoded - heading).abs() * 180 / math.pi
        if t:
            estimated = navigation.integrate_position(estimated, decoded, torch.stack(measured), c.dt)
        angular_command, forward_command = navigation.controller(estimated, decoded, goals[None])
        previous_position, previous_heading = position.clone(), heading.clone()
        stop = forward_command == 0
        angular = (.8 * angular + .2 * angular_command + .03 * noise['motor_angular_noise'][t][None]).masked_fill(stop, 0)
        speed = (.8 * speed + .2 * forward_command + .005 * noise['motor_speed_noise'][t][None]).clamp(.001, .35).masked_fill(stop, 0)
        assert not torch.any((speed == 0) & (angular != 0))
        heading = wrap_angle(heading + c.dt * angular)
        direction = torch.stack((heading.cos(), heading.sin()), -1)
        position = position + c.dt * speed[..., None] * direction
        traveled += speed * c.dt
        distance = (position - goals[None]).norm(dim=-1)
        first_arrival[(distance <= capture_radius) & (first_arrival > length)] = t + 1
        holding.update(distance, speed, first_arrival <= t + 1)
        if t + 1 in reports:
            seconds, rows = round((t + 1) * c.dt), {}
            for v, name in enumerate(VARIANTS):
                success = first_arrival[v] <= t + 1
                capped = first_arrival[v].clamp_max(t + 1).float() * c.dt
                self_error = (estimated[v] - previous_position[v]).norm(dim=-1)
                rows[name] = {'success_fraction': float(success.float().mean()),
                    'mean_capped_arrival_seconds': float(capped.mean()),
                    'mean_final_goal_distance_m': float(distance[v].mean()),
                    'mean_self_position_error_m': float(self_error.mean()),
                    'mean_heading_error_deg': float((heading_error[v] / (t + 1)).mean()),
                    'mean_path_length_m': float(traveled[v].mean())}
                hold, hold_arrays = holding.summarize(v, success)
                rows[name].update(hold)
                for key, value in hold_arrays.items():
                    archive[f'{seconds}s_{name}_{key}'] = value.cpu().numpy()
                for key, value in (('success', success), ('capped_arrival', capped),
                                   ('final_goal_distance', distance[v]), ('self_position_error', self_error)):
                    archive[f'{seconds}s_{name}_{key}'] = value.cpu().numpy()
            snapshots[f'{seconds}s'] = rows
        if t % 2 == 0:
            for name, value in (('position', previous_position), ('estimated_position', estimated),
                                ('heading', previous_heading), ('decoded_heading', decoded)):
                trace[name].append(value[:, :32].clone())
    for name, values in trace.items():
        archive[f'trace_{name}'] = torch.stack(values).cpu().numpy()
    assert max(max_native_identity, max_ordinary_identity) < 1e-5
    assert max_blank_dose == 0.
    assert max(max_dose_relative) < 1e-6
    return {'durations': snapshots, 'validation': {
        'native_step_identity_max_error': max_native_identity,
        'ordinary_step_identity_max_error': max_ordinary_identity,
        'direction_dose_on_blank_max': max_blank_dose,
        'sham_relative_dose_max': max_dose_relative}}, archive


def reference_comparison(model, task, checkpoint, noise, results, archives):
    reference, arrays = navigation.simulate(model, task, checkpoint, noise)
    checks = {}
    for duration in results['durations']:
        for variant in ('native', 'ordinary'):
            prefix = f'{duration}_{variant}'
            for metric in ('success', 'capped_arrival', 'final_goal_distance', 'self_position_error'):
                a, b = archives[f'{prefix}_{metric}'], arrays[f'{prefix}_{metric}']
                if metric == 'success':
                    checks[f'{prefix}_{metric}_mismatch_count'] = int(np.sum(a != b))
                    assert np.array_equal(a, b)
                else:
                    checks[f'{prefix}_{metric}_max_error'] = float(np.max(np.abs(a - b)))
                    tolerance = 1e-6 if metric == 'capped_arrival' else 1e-4
                    assert checks[f'{prefix}_{metric}_max_error'] <= tolerance, (prefix, metric, checks[f'{prefix}_{metric}_max_error'])
            assert abs(results['durations'][duration][variant]['success_fraction'] - reference['durations'][duration][variant]['success_fraction']) == 0
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('results/ring_heading/encoder_pilot/predictive_s1006.pt'))
    p.add_argument('--subspaces', type=Path, default=Path('results/ring_heading/direction_subspace/predictive_s1006/fitted_subspaces.npz'))
    p.add_argument('--out', type=Path, default=Path('results/ring_heading/direction_subspace/predictive_s1006/navigation'))
    p.add_argument('--episodes', type=int, default=128)
    p.add_argument('--seed', type=int, default=836219)
    p.add_argument('--seconds', type=int, choices=(20, 40), default=40)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--validate-reference', action='store_true')
    args = p.parse_args(); torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    model, task, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.requires_grad_(False)
    assert checkpoint['condition'] == 'predictive'
    fitted = np.load(args.subspaces)
    assert fitted['checkpoint_sha256'].item() == sha(args.checkpoint), 'Projector belongs to another checkpoint'
    assert int(fitted['model_seed']) == checkpoint['seed']
    intervention = DriveInterventions(model, fitted['basis'], fitted['sham_bases'])
    protocol = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
        'subspaces': str(args.subspaces), 'subspaces_sha256': sha(args.subspaces),
        'source_sha256': sha(__file__), 'navigation_helper_sha256': sha(inspect.getfile(navigation)),
        'intervention_source_sha256': sha(inspect.getfile(DriveInterventions)),
        'seed': args.seed, 'episodes': args.episodes, 'seconds': args.seconds,
        'variants': list(VARIANTS), 'condition': checkpoint['condition'], 'model_seed': checkpoint['seed'],
        'same_native_warmup': True, 'recurrent_weights_frozen': True,
        'intervention_inputs': ['observed_native_encoder_drive', 'observed_ordinary_encoder_drive'],
        'controller_inputs': ['estimated_position', 'decoded_heading', 'known_goal'],
        'independent_reference_simulation': args.validate_reference}
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.out / 'evaluation_source.py').write_text(Path(__file__).read_text())
    noise = navigation.realization(task, args.seed, round(args.seconds / task.config.dt), args.episodes)
    results, arrays = {}, {}
    for name, offset in (('baseline', 0.), ('new_places', 10.)):
        results[name], a = simulate(model, task, checkpoint, intervention, noise, offset)
        arrays.update({f'{name}_{k}': v for k, v in a.items()})
        if name == 'baseline' and args.validate_reference:
            results[name]['independent_reference_validation'] = reference_comparison(model, task, checkpoint, noise, results[name], a)
        print(name, {v: round(results[name]['durations'][f'{args.seconds}s'][v]['success_fraction'], 4) for v in VARIANTS}, flush=True)
    assert sha(args.checkpoint) == protocol['checkpoint_sha256']
    summary = {'protocol': protocol, 'scenarios': results, 'elapsed_seconds': time.monotonic() - started}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.out / 'episodes.npz', **arrays)
    print('DONE', summary['elapsed_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
