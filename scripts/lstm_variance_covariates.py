"""Descriptive current-state sensitivity check for episode-associated offsets.

A single modest quadratic basis of current position, movement/facing direction,
speed and previous action is fitted to each saved h cloud. This is an in-sample
variance partition, not a decoder accuracy, causal adjustment or complete
control for nonlinear spatial/visual tuning.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/lstm_variable_1B'
OUT = ROOT / 'results/lstm_variance_sources_1B'


def between_energy(x, episodes):
    return sum(np.sum(episodes == e) * np.sum(x[episodes == e].mean(0)**2)
               for e in np.unique(episodes))


def main():
    hashes = {}
    def record(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record(SOURCE / 'replay.npz')
    with np.load(SOURCE / 'replay.npz') as r:
        idx = r['sample_index']
        episodes = r['episode'][idx]
        hd = r['hd'][idx].astype(float)
        direction = r['dir'][idx].astype(float)
        angle = np.arctan2(direction[:, 1], direction[:, 0])
        base = np.column_stack([r['pos'][idx], np.cos(angle), np.sin(angle), np.cos(hd), np.sin(hd),
                                np.linalg.norm(r['vel'][idx].astype(float), axis=1)])
        action = np.eye(6)[r['prev_action'][idx]]
    terms = [base[:, i] for i in range(base.shape[1])]
    terms += [base[:, i] * base[:, j] for i in range(base.shape[1]) for j in range(i, base.shape[1])]
    features = np.column_stack(terms + [action[:, i] for i in range(6)])
    features -= features.mean(0)
    sd = features.std(0)
    features = features[:, sd > 1e-10] / sd[sd > 1e-10]
    u, singular, _ = np.linalg.svd(features, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    basis = u[:, :rank]
    rows = []
    for seed in range(1, 9):
        for condition in ('gc100', 'pgc33'):
            path = SOURCE / f'{condition}_s{seed}_activity.npz'
            record(path)
            with np.load(path) as f:
                h = f['h'].astype(float)
            h -= h.mean(0)
            fitted = basis @ (basis.T @ h)
            residual = h - fitted
            total = np.sum(h*h)
            between = between_energy(h, episodes)
            residual_between = between_energy(residual, episodes)
            assert np.allclose(basis.T @ residual, 0, atol=1e-10)
            assert np.isclose(total, np.sum(fitted*fitted) + np.sum(residual*residual), atol=1e-8)
            rows.append(dict(condition=condition, seed=seed,
                fitted_current_state_share=float(np.sum(fitted*fitted) / total),
                residual_between_share_of_original_total=float(residual_between / total),
                residual_episode_offset_fraction_retained=float(residual_between / between)))
    summary = {key: {condition: float(np.mean([r[key] for r in rows if r['condition'] == condition]))
                    for condition in ('gc100', 'pgc33')} for key in rows[0] if key not in ('condition', 'seed')}
    result = dict(protocol=dict(samples=len(idx), basis_rank=rank,
        covariates='x,y, movement cos/sin, facing cos/sin, speed; all first- and second-order terms; previous-action indicators.',
        fitting='One common centered orthonormal covariate basis for all models. Ordinary least-squares projection of centered h; all samples used. No goals, room IDs, episode IDs, actor outputs or future variables in basis.',
        status='Post hoc descriptive sensitivity check, not part of the five primary comparisons. No extra significance screening. No held-out prediction or causal claim.',
        limitation='A modest quadratic basis cannot remove arbitrary periodic spatial tuning or image features. Covariates correlate with goals/history, so projection can also absorb those signals. Residual episode offsets are not uniquely goal or room information.'),
        models=rows, means=summary, source_sha256=hashes)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    (OUT / 'current_state_sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(basis_rank=rank, means=summary), indent=2))


if __name__ == '__main__':
    main()
