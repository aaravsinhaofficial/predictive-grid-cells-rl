"""Independently validate saved cue spectra on every pilot phase and context."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_heading_cue_spectra import phase_gradient
from evaluate_ring_dynamics import jacobian
from ring_dynamics import tangents
from ring_heading_model import load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=Path('results/ring_heading/encoder_pilot/cue_spectra_predictive_s1006_final.json'))
    p.add_argument('--out', type=Path, default=Path('results/ring_heading/encoder_pilot/cue_spectra_independent_audit.json'))
    args = p.parse_args(); torch.set_num_threads(2)
    report = json.loads(args.report.read_text())
    archive = np.load(args.report.with_suffix('.npz'))
    model, task, _ = load_checkpoint(report['checkpoint'], 'cpu')
    model = model.double().requires_grad_(False)
    assert hashlib.sha256(Path(report['checkpoint']).read_bytes()).hexdigest() == report['checkpoint_sha256']
    n = report['protocol']['n_phase']; phase = torch.arange(n) * 2 * np.pi / n
    direction = torch.stack((phase.cos(), phase.sin()), -1)
    zero, one = torch.zeros(n), torch.ones(n)
    rng = torch.Generator().manual_seed(927391)
    rows = []
    def delta_phase(a, b):
        ya, yb = model.readout(a), model.readout(b)
        return torch.atan2(ya[:, 0] * yb[:, 1] - ya[:, 1] * yb[:, 0], (ya * yb).sum(-1))
    for saved in report['rows']:
        context, reference, variant = saved['context'], saved['reference_state'], saved['input']
        h = torch.as_tensor(archive[f'context{context}_{reference}_states'])
        position = torch.tensor(saved['position']).expand(n, 2)
        if variant == 'blank':
            x = torch.zeros(n, model.input_size, dtype=torch.float64)
        else:
            x = task.inputs(position, direction, zero, zero, one, zero,
                            report['condition'] if variant == 'native' else 'grid').double()
        tangent = torch.as_tensor(tangents(h.numpy()))
        h_grad = h.clone().requires_grad_(True)
        y = model.readout(h_grad)
        automatic = torch.autograd.grad(torch.atan2(y[:, 1], y[:, 0]).sum(), h_grad)[0]
        grad = phase_gradient(model, h)
        gradient_error = float((automatic - grad).abs().max())
        matrix = jacobian(model, h, x)
        step = model.step(x, h)
        before = (grad * tangent).sum(-1)
        mapped = torch.einsum('bij,bj->bi', matrix, tangent)
        analytic = (phase_gradient(model, step) * mapped).sum(-1) / before
        eps = 1e-5
        a, b = h - eps * tangent, h + eps * tangent
        finite = delta_phase(model.step(x, a), model.step(x, b)) / delta_phase(a, b)
        valid = before.abs() > 1e-5
        finite_error = float((analytic[valid] - finite[valid]).abs().max())
        recorded = np.array([r['phase_gain'] for r in saved['per_phase']])
        saved_error = float(np.max(np.abs(analytic[valid].numpy() - recorded[valid.numpy()])))
        random = torch.randn(n, model.hidden_size, generator=rng, dtype=torch.float64)
        random /= random.norm(dim=-1, keepdim=True)
        fd_vector = (model.step(x, h + eps * random) - model.step(x, h - eps * random)) / (2 * eps)
        predicted_vector = torch.einsum('bij,bj->bi', matrix, random)
        jvp_error = float((fd_vector - predicted_vector).abs().max())
        amplitude_before = model.readout(h).norm(dim=-1)
        amplitude_after = model.readout(step).norm(dim=-1)
        frozen_covector = (grad * mapped).sum(-1) / before
        pre = model.input(x) + model.recurrent(h)
        row = {'context': context, 'reference': reference, 'input': variant,
            'tested_phases': n, 'valid_fraction': float(valid.float().mean()),
            'autograd_phase_gradient_max_error': gradient_error,
            'central_finite_phase_gain_max_error': finite_error,
            'saved_phase_gain_max_error': saved_error,
            'central_finite_state_jvp_max_error': jvp_error,
            'readout_amplitude_before_min': float(amplitude_before.min()),
            'readout_amplitude_after_min': float(amplitude_after.min()),
            'phase_gain_median': float(analytic[valid].quantile(.5)),
            'frozen_before_phase_covector_gain_median': float(frozen_covector[valid].quantile(.5)),
            'tangent_state_gain_median': float(mapped.norm(dim=-1).quantile(.5)),
            'mean_tanh_derivative': float((1 - pre.tanh().square()).mean())}
        assert gradient_error < 1e-10 and finite_error < 1e-5 and saved_error < 1e-10 and jvp_error < 1e-9, row
        rows.append(row)
    result = {'report': str(args.report), 'report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checkpoint_sha256': report['checkpoint_sha256'], 'rows': rows,
        'summary': {'tested_context_reference_input_combinations': len(rows),
                    'tested_phase_conditions': sum(r['tested_phases'] for r in rows)}}
    for key in ('autograd_phase_gradient_max_error', 'central_finite_phase_gain_max_error',
                'saved_phase_gain_max_error', 'central_finite_state_jvp_max_error'):
        result['summary'][key] = max(r[key] for r in rows)
    for key in ('readout_amplitude_before_min', 'readout_amplitude_after_min', 'valid_fraction'):
        result['summary'][key] = min(r[key] for r in rows)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))
    for reference in ('cue_free', 'native_anchored'):
        for variant in ('blank', 'native', 'ordinary'):
            selected = [r for r in rows if r['context'] < 5 and r['reference'] == reference and r['input'] == variant]
            print(reference, variant, {k: round(float(np.mean([r[k] for r in selected])), 5) for k in (
                'phase_gain_median', 'frozen_before_phase_covector_gain_median', 'tangent_state_gain_median', 'mean_tanh_derivative')})


if __name__ == '__main__':
    main()
