"""Separate fixed mean and varying components of the fitted direction drive.

This is an additional diagnostic. It does not modify the frozen rank-2 study.
The calibration mean is read from that study's independent fitting data. During
evaluation its only gate is the observed spatial-availability input flag.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from heading_direction_subspace import DriveInterventions, anchoring, forced_trajectories
import heading_subspace_navigation as nav
from ring_heading_model import load_checkpoint


VARIANTS = ('native', 'ordinary', 'restore_direction', 'mean_only_restore', 'mean_preserving_removal')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MeanDirectionInterventions(DriveInterventions):
    def __init__(self, model, basis, shams, calibration_mean):
        super().__init__(model, basis, shams)
        self.mean = torch.as_tensor(calibration_mean, dtype=self.basis.dtype, device=self.basis.device)
        self.projected_mean = (self.mean @ self.basis) @ self.basis.T
        self.maximum_blackout_effect = 0.
        self.maximum_decomposition_error = 0.

    def paired_drives(self, native_input, ordinary_input):
        original, dose = super().paired_drives(native_input, ordinary_input)
        drives = {k: original[k] for k in ('native', 'ordinary', 'remove_direction', 'restore_direction', 'residual_only')}
        visible_index = self.model.config.n_code + 2
        available = native_input[..., visible_index:visible_index + 1]
        assert torch.equal(available, ordinary_input[..., visible_index:visible_index + 1])
        assert torch.all((available == 0) | (available == 1))
        delta = drives['native'] - drives['ordinary']
        projected = (delta @ self.basis) @ self.basis.T
        tonic = available * self.projected_mean
        centered = projected - tonic
        drives['mean_only_restore'] = drives['ordinary'] + tonic
        drives['mean_preserving_removal'] = drives['native'] - centered
        self.maximum_decomposition_error = max(self.maximum_decomposition_error,
            float((projected - (tonic + centered)).abs().max()))
        blank = available[..., 0] == 0
        if blank.any():
            self.maximum_blackout_effect = max(self.maximum_blackout_effect,
                max(float((drives[k][blank] - drives['ordinary'][blank]).abs().max())
                    for k in ('mean_only_restore', 'mean_preserving_removal')))
        dose.update(tonic_norm=tonic.norm(dim=-1), centered_direction_norm=centered.norm(dim=-1))
        return drives, dose


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Native', 'Ordinary', 'Full plane', 'Mean only', 'Remove\nvariation']
    colors = ['#D87532', '#8E98A3', '#317BB5', '#7FAD91', '#8C64B4']
    forced = result['forced']['scenarios']['new_places']
    navdata = result['navigation']['new_places']['durations']['40s']
    metric = [lambda v: forced[v]['all_mae_deg'],
              lambda v: 100 * navdata[v]['success_fraction'],
              lambda v: 100 * np.mean([r['circle_coverage'] for r in result['anchoring']['rows']
                                       if r['variant'] == v and r['step'] == 100 and r['context'] < 5])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
    for ax, fn, title, label in zip(axes, metric,
            ['Heading at new places', 'Navigation at new places', 'Ring coverage'],
            ['Error (degrees)', 'Reaching the goal (%)', 'Circle covered (%)']):
        values = [fn(v) for v in VARIANTS]
        ax.bar(range(5), values, color=colors)
        for i, value in enumerate(values): ax.text(i, value + 2, f'{value:.1f}', ha='center', fontsize=9)
        ax.set(title=title, ylabel=label, xticks=range(5), xticklabels=labels, ylim=(0, 110))
        ax.tick_params(axis='x', labelsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Fixed calibration mean · No direction label enters the intervention · One frozen model', fontsize=9)
    for extension in ('png', 'pdf'): fig.savefig(out / f'tonic_direction.{extension}', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('results/ring_heading/encoder_pilot/predictive_s1006.pt'))
    p.add_argument('--subspaces', type=Path, default=Path('results/ring_heading/direction_subspace/predictive_s1006/fitted_subspaces.npz'))
    p.add_argument('--out', type=Path, default=Path('results/ring_heading/tonic_direction/predictive_s1006'))
    p.add_argument('--device', default='cpu'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--n-phase', type=int, default=72); p.add_argument('--anchor-steps', type=int, default=200)
    p.add_argument('--length', type=int, default=256); p.add_argument('--trajectories', type=int, default=256)
    p.add_argument('--evaluation-seed', type=int, default=931731)
    p.add_argument('--navigation-seed', type=int, default=836219); p.add_argument('--navigation-episodes', type=int, default=128)
    args = p.parse_args(); torch.set_num_threads(args.threads); started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    model, task, ckpt = load_checkpoint(args.checkpoint, args.device); model.requires_grad_(False)
    fitted = np.load(args.subspaces)
    assert fitted['checkpoint_sha256'].item() == sha(args.checkpoint)
    assert int(fitted['model_seed']) == ckpt['seed'] and ckpt['condition'] == 'predictive'
    calibration = fitted['train_delta'].astype(np.float64)
    mean = calibration.mean(0); basis = fitted['basis']; projected_mean = (mean @ basis) @ basis.T
    projected = (calibration @ basis) @ basis.T
    mean_energy_fraction = float(np.sum(projected_mean**2) / np.mean(np.sum(projected**2, axis=1)))
    protocol = {'role': 'additional mechanistic diagnostic, separate from frozen rank2 primary families',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
        'subspaces': str(args.subspaces), 'subspaces_sha256': sha(args.subspaces), 'source_sha256': sha(__file__),
        'seed': ckpt['seed'], 'calibration_mean_source': 'all original fixed calibration train_delta rows, averaged in float64',
        'calibration_samples': len(calibration), 'fixed_projector': True,
        'tonic_drive': 'observed spatial availability times P times calibration mean',
        'mean_only_restore': 'ordinary drive plus tonic drive',
        'mean_preserving_removal': 'native drive minus (projected native-ordinary difference minus tonic drive)',
        'inference_inputs': ['observed paired encoder drives', 'observed spatial-availability flag', 'fixed calibration mean'],
        'no_heading_or_position_label_at_inference': True,
        'forced_seed': args.evaluation_seed, 'forced_episodes': args.trajectories, 'forced_steps': args.length,
        'forced_exclude_first_steps': 8, 'common_native_initialization_steps': 4,
        'anchor_phases': args.n_phase, 'anchor_steps': args.anchor_steps,
        'navigation_seed': args.navigation_seed, 'navigation_episodes': args.navigation_episodes, 'navigation_seconds': 40,
        'frozen_dependencies': {name: sha(inspect.getfile(module)) for name, module in
            [('DriveInterventions', DriveInterventions), ('heading_subspace_navigation', nav), ('navigation_helpers', nav.navigation)]}}
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.out / 'diagnostic_source.py').write_text(Path(__file__).read_text())
    np.savez_compressed(args.out / 'mean_decomposition.npz', mean=mean, projected_mean=projected_mean, basis=basis,
                        checkpoint_sha256=protocol['checkpoint_sha256'])
    intervention = MeanDirectionInterventions(model, basis, fitted['sham_bases'], mean)
    result = {'protocol': protocol, 'calibration': {'projected_mean_norm': float(np.linalg.norm(projected_mean)),
        'projected_drive_rms_norm': float(np.sqrt(np.mean(np.sum(projected**2, axis=1)))),
        'mean_fraction_of_projected_drive_energy': mean_energy_fraction}}
    result['anchoring'] = anchoring(model, task, intervention, args)
    result['forced'] = forced_trajectories(model, task, intervention, args)
    # The imported simulator supports the named drive variants. This process-
    # local configuration does not change its source or any concurrent study.
    original_variants = nav.VARIANTS
    nav.VARIANTS = VARIANTS
    try:
        noise = nav.navigation.realization(task, args.navigation_seed, round(40 / task.config.dt), args.navigation_episodes)
        result['navigation'], arrays = {}, {}
        for name, offset in (('baseline', 0.), ('new_places', 10.)):
            report, a = nav.simulate(model, task, ckpt, intervention, noise, offset)
            if name == 'baseline':
                report['independent_reference_validation'] = nav.reference_comparison(model, task, ckpt, noise, report, a)
            result['navigation'][name] = report
            arrays.update({f'{name}_{k}': v for k, v in a.items()})
            print('navigation', name, {v: report['durations']['40s'][v]['success_fraction'] for v in VARIANTS}, flush=True)
        np.savez_compressed(args.out / 'navigation_episodes.npz', **arrays)
    finally:
        nav.VARIANTS = original_variants
    assert intervention.maximum_blackout_effect == 0
    assert intervention.maximum_decomposition_error < 1e-5
    assert sha(args.checkpoint) == protocol['checkpoint_sha256'] and sha(args.subspaces) == protocol['subspaces_sha256']
    result['validation'] = {'maximum_blackout_effect': intervention.maximum_blackout_effect,
        'maximum_decomposition_error': intervention.maximum_decomposition_error}
    result['elapsed_seconds'] = time.monotonic() - started
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot(result, args.out)
    print('DONE', result['elapsed_seconds'], 'seconds; tonic energy fraction', mean_energy_fraction, flush=True)


if __name__ == '__main__':
    main()
