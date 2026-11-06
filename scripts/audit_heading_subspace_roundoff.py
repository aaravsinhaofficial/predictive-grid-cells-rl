"""Independent numerical audit of the two algebraically identical aliases.

No evaluator or checkpoint is edited. The original float32 trajectory is
compared with the same counterfactual arithmetic and recurrent map in float64.
Only the unused residual-only validation alias is at issue.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from heading_direction_subspace import DriveInterventions
from evaluate_ring_heading_transfer import alter_measurements
from ring_heading_model import load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=2012)
    p.add_argument('--out', type=Path, default=Path('results/ring_heading/subspace_confirmation/numerical_alias_audit_s2012.json'))
    args = p.parse_args(); torch.set_num_threads(2)
    checkpoint = sorted(Path('results/ring_cloud_artifacts').glob(f'ring-20261105-confirm-*/predictive_s{args.seed}.pt'))[0]
    model, task, _ = load_checkpoint(checkpoint, 'cpu'); model.requires_grad_(False)
    double_model = copy.deepcopy(model).double()
    fitted = np.load(Path('results/ring_heading/subspace_confirmation') / f'predictive_s{args.seed}/fitted_subspaces.npz')
    iv = DriveInterventions(model, fitted['basis'], fitted['sham_bases'])
    iv64 = DriveInterventions(double_model, fitted['basis'], fitted['sham_bases'])
    generator = torch.Generator().manual_seed(931731)
    base = task.realization(256, 256, generator)
    results = {}
    for name, offset in [('same_region', 0.), ('new_places', 10.)]:
        data = alter_measurements(base, offset, 1.)
        xp, _, _ = task.batch(256, 256, generator, 'predictive', data)
        xg, _, _ = task.batch(256, 256, generator, 'grid', data)
        bn, bo = model.input(xp), model.input(xg)
        modes = {}
        for precision, intervention, native, ordinary, readout in (
            ('float32', iv, bn, bo, model.readout),
            ('float64_same_encoder_drives', iv64, bn.double(), bo.double(), double_model.readout)):
            drives, _ = intervention.transform_drives(native, ordinary)
            for k in drives: drives[k][:4] = drives['native'][:4]
            names = list(drives)
            state = native.new_zeros(len(names) * 256, model.hidden_size)
            max_state = max_angle = 0.
            for t in range(256):
                state = intervention.step_from_drive(torch.cat([drives[k][t] for k in names]), state)
                a, b = state.reshape(len(names), 256, -1)[[names.index('remove_direction'), names.index('residual_only')]]
                max_state = max(max_state, float((a-b).abs().max()))
                ra, rb = readout(a), readout(b)
                angle = torch.atan2(ra[:, 1], ra[:, 0]) - torch.atan2(rb[:, 1], rb[:, 0])
                max_angle = max(max_angle, float(torch.atan2(angle.sin(), angle.cos()).abs().max()) * 180 / np.pi)
            modes[precision] = {'maximum_drive_difference': float((drives['remove_direction'] - drives['residual_only']).abs().max()),
                'maximum_trajectory_hidden_coordinate_difference': max_state,
                'maximum_decoded_heading_difference_deg': max_angle}
        assert modes['float64_same_encoder_drives']['maximum_trajectory_hidden_coordinate_difference'] < 1e-10
        assert modes['float32']['maximum_trajectory_hidden_coordinate_difference'] < 1e-4
        assert modes['float32']['maximum_decoded_heading_difference_deg'] < .01
        results[name] = modes
    result = {'seed': args.seed, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'note': 'Both aliases implement bN-P(bN-bO). Different float32 arithmetic order produces a tiny roundoff error, amplified by finite-time dynamics. Neither primary contrast nor navigation uses residual_only.',
        'scenarios': results}
    args.out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
