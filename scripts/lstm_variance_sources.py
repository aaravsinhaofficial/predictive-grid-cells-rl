"""Decompose the PCA variance difference using existing frozen-LSTM recordings.

No policy training or new rollouts. Separates absolute variance, episode means,
within-episode variation, and sensitivity to individual neuron amplitude.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/lstm_variable_1B'
OUT = ROOT / 'results/lstm_variance_sources_1B'
CONDITIONS = ('gc100', 'pgc33')
K = 10


def spectrum(x):
    covariance = x.T @ x / len(x)
    values, vectors = np.linalg.eigh(covariance)
    return values[::-1], vectors[:, ::-1], covariance


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    t = stats.ttest_rel(b, a)
    interval = t.confidence_interval(.95)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    return dict(grid=a.tolist(), pgc=b.tolist(), grid_mean=float(a.mean()),
                pgc_mean=float(b.mean()), difference=float(d.mean()),
                paired_sem=float(stats.sem(d)), ci95=[float(interval.low), float(interval.high)],
                pgc_greater_seeds=int((d > 0).sum()), t_p=float(t.pvalue),
                exact_p=float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def record(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record(SOURCE / 'replay.npz')
    with np.load(SOURCE / 'replay.npz') as r:
        indices = r['sample_index']
        episodes = r['episode'][indices]
    rows = []
    for seed in range(1, 9):
        for condition in CONDITIONS:
            path = SOURCE / f'{condition}_s{seed}_activity.npz'
            record(path)
            with np.load(path) as f:
                h = f['h'].astype(np.float64)
            x = h - h.mean(0)
            between = np.zeros_like(x)
            for ep in np.unique(episodes):
                mask = episodes == ep
                between[mask] = x[mask].mean(0)
            within = x - between
            eigenvalues, vectors, covariance = spectrum(x)
            wvalues, _, wcov = spectrum(within)
            bvalues, _, bcov = spectrum(between)
            sd = np.sqrt(np.diag(covariance))
            active = sd > 1e-10
            standardized = x[:, active] / sd[active]
            zvalues, _, _ = spectrum(standardized)
            wsd = np.sqrt(np.diag(wcov))
            wactive = wsd > 1e-10
            wzvalues, _, _ = spectrum(within[:, wactive] / wsd[wactive])
            top = vectors[:, :K]
            total = float(eigenvalues.sum())
            top_variance = float(eigenvalues[:K].sum())
            btop = float(np.trace(top.T @ bcov @ top))
            wtop = float(np.trace(top.T @ wcov @ top))
            relative_error = float(np.linalg.norm(covariance - bcov - wcov) / np.linalg.norm(covariance))
            assert relative_error < 1e-12
            assert np.isclose(btop + wtop, top_variance, atol=1e-10)
            permutation = np.random.default_rng(4521).permutation(len(x))
            shuffled_values, _, _ = spectrum(x[permutation])
            assert np.allclose(eigenvalues, shuffled_values, atol=1e-12)
            row = dict(condition=condition, seed=seed, samples=len(h),
                total_variance=total, top10_absolute=top_variance,
                tail_absolute=total - top_variance,
                top10_share=top_variance / total,
                between_episode_share=float(bvalues.sum() / total),
                within_top10_share=float(wvalues[:K].sum() / wvalues.sum()),
                standardized_top10_share=float(zvalues[:K].sum() / zvalues.sum()),
                within_standardized_top10_share=float(wzvalues[:K].sum() / wzvalues.sum()),
                top10_between_fraction=btop / top_variance,
                top10_between_share_of_total=btop / total,
                top10_within_share_of_total=wtop / total,
                between_absolute=float(bvalues.sum()), within_absolute=float(wvalues.sum()),
                top10_between_absolute=btop, top10_within_absolute=wtop,
                minimum_unit_std=float(sd.min()), standardized_units=int(active.sum()),
                within_standardized_units=int(wactive.sum()),
                covariance_decomposition_relative_error=relative_error,
                time_shuffle_eigenvalue_max_difference=float(np.max(np.abs(eigenvalues - shuffled_values))))
            rows.append(row)
            print(f'{condition} s{seed}: total={total:.2f}, leading={top_variance:.2f}, tail={total-top_variance:.2f}, '
                  f'raw={row["top10_share"]:.3f}, within={row["within_top10_share"]:.3f}, '
                  f'std={row["standardized_top10_share"]:.3f}, between={row["between_episode_share"]:.3f}', flush=True)
    by = {(r['condition'], r['seed']): r for r in rows}
    keys = [k for k in rows[0] if k not in ('condition', 'seed', 'samples')]
    comparisons = {k: compare(*[[by[c, s][k] for s in range(1, 9)] for c in CONDITIONS])
                   for k in keys if k not in ('standardized_units', 'within_standardized_units',
                       'covariance_decomposition_relative_error', 'time_shuffle_eigenvalue_max_difference')}
    primary = ['top10_absolute', 'tail_absolute', 'between_episode_share',
               'within_top10_share', 'standardized_top10_share']
    for field in ('t_p', 'exact_p'):
        running = 0.
        for rank, key in enumerate(sorted(primary, key=lambda k: comparisons[k][field])):
            running = max(running, min(1., (len(primary) - rank) * comparisons[key][field]))
            comparisons[key][field + '_holm5'] = running
    result = dict(protocol=dict(
        samples=len(indices), episodes=len(np.unique(episodes)), seeds=8, pcs=K,
        method='All 3393 original samples. Center h globally. Covariance uses divisor n. '
               'Between-episode term repeats each episode mean at its sampled rows; within term subtracts that mean. '
               'This gives exact sample-weighted total = between + within covariance. '
               'Project both terms onto the ORIGINAL leading 10 PCs for additive attribution. '
               'Separately refit PCA after episode demeaning and/or individual-neuron standardization.',
        standardization='Exclude only units with standard deviation <=1e-10; report count. '
                        'Standardization is a sensitivity analysis and changes the activity metric.',
        primary_comparisons=primary,
        statistics='Exploratory paired training-seed tests, two-sided t and exhaustive sign-flip, '
                   'Holm within these five preselected contrasts. Other quantities descriptive. '
                   'Eight seeds, not samples/episodes, are statistical replicates.',
        limitations='Episode identity jointly includes goal, visual room layout, and sampled history. '
                    'Demeaning cannot isolate goal information. Covariance is insensitive to time order '
                    'and does not measure recurrent dynamics, stability, causal use, or topology.'),
        models=rows, comparisons=comparisons, source_sha256=hashes)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for key in primary + ['top10_share', 'top10_between_share_of_total', 'top10_within_share_of_total', 'within_standardized_top10_share']:
        print(key, comparisons[key], flush=True)


if __name__ == '__main__':
    main()
