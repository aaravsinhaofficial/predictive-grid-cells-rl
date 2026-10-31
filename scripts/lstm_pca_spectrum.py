"""Refit PCA and quantify hidden-activity variance across frozen LSTM seeds.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_pca_spectrum.py
Uses saved states only. Neither policies nor recurrent weights are trained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('gc100', 'pgc33')
COLORS = {'gc100': '#3478ac', 'pgc33': '#d66b37'}
LABELS = {'gc100': 'Grid only', 'pgc33': 'Grid + 33% PGC'}
LEVELS = (1, 3, 6, 10, 20)
THRESHOLDS = (50, 80, 90, 95, 99)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), sem=float(x.std(ddof=1) / np.sqrt(len(x))),
                min=float(x.min()), max=float(x.max()), values=x.tolist())


def paired_metrics(rows, protocol):
    by = {(r['condition'], r['seed']): r for r in rows if r['protocol'] == protocol}
    metrics = {}
    for name in [f'pc{k}' for k in LEVELS] + [f'p{q}' for q in THRESHOLDS] + ['participation_ratio']:
        def value(c, seed):
            row = by[c, seed]
            if name == 'participation_ratio':
                return row[name]
            return row['cumulative_variance' if name.startswith('pc') else 'dimensions'][name]
        a, b = (np.array([value(c, seed) for seed in range(1, 9)]) for c in CONDITIONS)
        metrics[name] = dict(grid=mean_sem(a), pgc=mean_sem(b),
            pgc_minus_grid=mean_sem(b - a), pgc_greater_seeds=int((b > a).sum()),
            pgc_lower_seeds=int((b < a).sum()), ties=int((a == b).sum()))
    return metrics


def curve_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set(xlim=(1, 256), ylim=(0, 101), xticks=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    for q in (80, 90, 95):
        ax.axhline(q, color='#ced3d7', lw=.7, ls='--', zorder=0)
    ax.grid(axis='x', alpha=.12)


def make_figures(summary, out):
    rows = {(r['protocol'], r['condition'], r['seed']): r for r in summary['models']}
    spectra = {c: np.array([rows['variable', c, s]['explained_variance_ratio'] for s in range(1, 9)])
               for c in CONDITIONS}
    pcs = np.arange(1, 257)
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, sharey=True, layout='constrained')
    for seed, ax in enumerate(axes.flat, 1):
        for c in CONDITIONS:
            row = rows['variable', c, seed]
            y = 100 * np.cumsum(row['explained_variance_ratio'])
            ax.plot(pcs, y, color=COLORS[c], lw=2, label=LABELS[c])
            q = row['dimensions']['p90']
            ax.scatter(q, y[q - 1], color=COLORS[c], s=25, zorder=3)
        g, p = rows['variable', 'gc100', seed], rows['variable', 'pgc33', seed]
        ax.set_title(f'Seed {seed} · PCs for 90%: grid {g["dimensions"]["p90"]}, PGC {p["dimensions"]["p90"]}', fontsize=11)
        curve_axis(ax)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=9, loc='lower right')
    fig.suptitle('All eight seed pairs · cumulative hidden-state variance\n'
                 'Variable movement and goals; full 256-dimensional PCA refitted per model', fontsize=15)
    fig.supxlabel('Number of principal components (logarithmic spacing)', fontsize=11)
    fig.supylabel('Cumulative variance explained (%)', fontsize=11)
    save(fig, out / 'all_seeds_cumulative_variance')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    ax = axes[0, 0]
    for c in CONDITIONS:
        data = spectra[c][:, :30] * 100
        mu, sem = data.mean(0), data.std(0, ddof=1) / np.sqrt(8)
        ax.plot(pcs[:30], mu, color=COLORS[c], label=LABELS[c], lw=2)
        ax.fill_between(pcs[:30], mu - sem, mu + sem, color=COLORS[c], alpha=.18)
    ax.set(xlabel='Principal component', ylabel='Variance explained by that PC (%)',
           title='A  |  Individual components · mean ± SEM', xlim=(1, 30), ylim=(0, None))
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for c in CONDITIONS:
        data = spectra[c].cumsum(1) * 100
        mu, sem = data.mean(0), data.std(0, ddof=1) / np.sqrt(8)
        ax.plot(pcs, mu, color=COLORS[c], lw=2)
        ax.fill_between(pcs, mu - sem, mu + sem, color=COLORS[c], alpha=.18)
    curve_axis(ax)
    ax.set(xlabel='Number of PCs (logarithmic spacing)', ylabel='Cumulative variance explained (%)',
           title='B  |  Cumulative variance · mean ± SEM')
    ax = axes[1, 0]
    for j, c in enumerate(CONDITIONS):
        vals = np.array([[rows['variable', c, seed]['dimensions'][f'p{q}'] for q in THRESHOLDS]
                         for seed in range(1, 9)])
        x = np.arange(len(THRESHOLDS)) + (j - .5) * .18
        for seed in range(8):
            ax.scatter(x, vals[seed], color=COLORS[c], s=17, alpha=.4)
        ax.errorbar(x, vals.mean(0), yerr=vals.std(0, ddof=1)/np.sqrt(8),
                    color=COLORS[c], lw=1.5, marker='o', capsize=3, label=LABELS[c])
    ax.set(xticks=np.arange(len(THRESHOLDS)), xticklabels=[f'{q}%' for q in THRESHOLDS],
           ylabel='Number of PCs', xlabel='Target cumulative variance',
           title='C  |  Components needed · individual seeds + mean ± SEM', ylim=(0, None))
    ax = axes[1, 1]
    for j, c in enumerate(CONDITIONS):
        vals = np.array([[100 * rows[protocol, c, seed]['cumulative_variance']['pc3']
                          for protocol in ('fixed', 'variable')] for seed in range(1, 9)])
        x = np.array([0, 1]) + (j - .5) * .12
        for pair in vals:
            ax.plot(x, pair, color=COLORS[c], alpha=.25, lw=1)
        ax.errorbar(x, vals.mean(0), yerr=vals.std(0, ddof=1)/np.sqrt(8),
                    marker='o', capsize=3, color=COLORS[c], lw=2)
    ax.set(xticks=[0, 1], xticklabels=['Earlier fixed probe', 'Variable probe'],
           ylabel='Variance captured by PC1–3 (%)', ylim=(0, 75),
           title='D  |  Broader replay spreads variance across more PCs')
    fig.suptitle('Frozen 1B LSTMs · PCA variance spectrum across eight matched seeds', fontsize=15)
    fig.supxlabel('Centered h, no neuron scaling. PCA dimension measures linear variance, not gridness or torus dimension.\n'
                  'Fixed/variable probes also differ in trajectory, rooms, sampling, and episode structure.', fontsize=9)
    save(fig, out / 'pca_spectrum_summary')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout='constrained')
    mats = [np.array([[rows['variable', c, seed]['dimensions'][f'p{q}'] for q in THRESHOLDS]
                     for seed in range(1, 9)]) for c in CONDITIONS]
    vmax = max(a.max() for a in mats)
    for ax, c, arr in zip(axes, CONDITIONS, mats):
        im = ax.imshow(arr, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        for (i, j), val in np.ndenumerate(arr):
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11,
                    color='white' if val > .55 * vmax else '#172b3b')
        ax.set(xticks=np.arange(len(THRESHOLDS)), xticklabels=[f'{q}%' for q in THRESHOLDS],
               yticks=np.arange(8), yticklabels=[f'Seed {s}' for s in range(1, 9)],
               xlabel='Cumulative variance target', title=LABELS[c])
    fig.colorbar(im, ax=axes, shrink=.8, label='Number of principal components')
    fig.suptitle('How many PCs explain the activity? · exact counts for every model', fontsize=15)
    fig.supxlabel('Variable-direction, variable-goal replay. Each number is the minimum count reaching that variance target.', fontsize=9)
    save(fig, out / 'components_by_seed')


def write_readme(summary, out):
    comparison = summary['comparisons']['variable']
    lines = ['# PCA spectrum of all eight seed pairs', '',
        'PCA was refitted from the saved 256-dimensional hidden vectors `h`, using float64 full SVD, mean centering, and no neuron-wise scaling. The variable replay uses 3,393 matched moving states per model; the earlier fixed replay uses its original 1,185 selected states. Policies were not rerun or retrained. No UMAP is used.', '',
        '[Per-seed cumulative curves](all_seeds_cumulative_variance.png) · [Spectrum summary](pca_spectrum_summary.png) · [Exact component counts](components_by_seed.png)', '',
        'The companion projection script writes [all 16 PC1–3 clouds colored by arena x](all_seeds_pca.png), [arena y](all_seeds_pc1_3_y.png), and [movement direction](all_seeds_pc1_3_direction.png), plus [PC4–6](all_seeds_pc4_6_x.png) and [PC7–9](all_seeds_pc7_9_x.png). These are fixed consecutive component blocks, not projections chosen to resemble a particular shape.', '',
        '## Variable replay: summary across eight seeds', '',
        '| Quantity | Grid only, mean ± SEM | PGC + grid, mean ± SEM | PGC − grid, paired mean ± SEM |',
        '|---|---:|---:|---:|']
    for name, record in comparison.items():
        factor = 100 if name.startswith('pc') else 1
        label = (f'Variance captured by PC1 (%)' if name == 'pc1' else f'Variance captured by first {name[2:]} PCs (%)') if name.startswith('pc') else (
            f'PCs for {name[1:]}% variance' if name.startswith('p') and name != 'participation_ratio' else 'Participation ratio')
        cells = [f'{factor * record[k]["mean"]:.2f} ± {factor * record[k]["sem"]:.2f}'
                 for k in ('grid', 'pgc', 'pgc_minus_grid')]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'SEM is across the eight independent training seeds, not across time samples. Comparisons are descriptive, without p-value screening.', '',
        '## Every model', '',
        '| Seed | Condition | PC1–3 (%) | PC1–6 (%) | PC1–10 (%) | PCs: 50% | 80% | 90% | 95% | 99% |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['models']:
        if row['protocol'] != 'variable':
            continue
        cells = [str(row['seed']), LABELS[row['condition']]]
        cells += [f'{100 * row["cumulative_variance"][f"pc{k}"]:.2f}' for k in (3, 6, 10)]
        cells += [str(row['dimensions'][f'p{q}']) for q in THRESHOLDS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Interpretation and limitations', '',
        'The inspected 3D projections do not show a consistently clear toroidal surface or regular lattice across the seed pairs. PGC models concentrate more variance in the leading PCs: all eight need fewer PCs to reach 50% and 80% variance. This difference weakens at higher thresholds; at 99%, the group means are nearly identical and the seed-wise direction is mixed. Thus the supported result is greater concentration of linear variance, not greater gridness.', '',
        'A principal component is a linear axis of variation, ordered from largest variance to smallest. Cumulative variance is the fraction of total squared centered activity retained by the first k axes. For example, 90% variance means that projecting onto those axes discards 10% of the sample variance; it does not mean 90% task accuracy or 90% biological explanation.', '',
        'Participation ratio is 1 / sum(p_i²), where p_i is each component’s variance fraction. It is a spectrum-weighted measure of linear dimensionality, not an estimate of a manifold’s intrinsic topological dimension.', '',
        'Greater variance in early PCs means a more concentrated linear spectrum. It does not establish grid-cell tuning, a torus, a recurrent attractor, stability, or better task performance. A curved manifold can require many linear PCs. Mixing directions, goals, vision, and histories can also broaden the spectrum.', '',
        'The fixed-to-variable contrast is descriptive: the probes also differ in sampled trajectory, room layouts, sample count, episode duration/reset structure, and direction/goal coverage. It cannot attribute a change to one factor alone. These fits summarize the sampled activity, not held-out prediction performance.', '',
        'The variable replay is scripted and effectively post-discovery. Movement and facing are correlated, goal snapshots are externally supplied, and multiple goals and rooms are pooled. These conditions limit mechanistic conclusions.', '',
        '## Reproduce and audit', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_pca_spectrum.py', '```', '',
        '`summary.json` stores all 256 variance fractions, per-model counts and metrics, paired comparisons, input hashes, and numerical checks. Each variable-model NPZ stores the first 12 scores and the full fitted PCA basis/mean/spectrum. Row order matches `../replay.npz` → `sample_index`.', '',
        'Validation checks finite values, sorted/nonnegative eigenvalues, variance conservation, basis orthonormality, full-rank reconstruction, and agreement with the earlier stored variance spectra. Source files are read-only and hashed again after the run.', '']
    (out / 'README.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variable', type=Path, default=ROOT / 'results/lstm_variable_1B')
    ap.add_argument('--fixed', type=Path, default=ROOT / 'results/lstm_pca_1B')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/lstm_variable_1B/pca_spectrum')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.fixed / 'replay.npz') as f:
        fixed_mask = f['selected'].astype(bool)
    hashes, rows = {}, []
    for protocol, source in [('variable', args.variable), ('fixed', args.fixed)]:
        replay_path = source / 'replay.npz'
        hashes[str(replay_path.relative_to(ROOT))] = fingerprint(replay_path)
        for seed in range(1, 9):
            for condition in CONDITIONS:
                path = source / f'{condition}_s{seed}_activity.npz'
                hashes[str(path.relative_to(ROOT))] = fingerprint(path)
                with np.load(path) as data:
                    h = data['h'].astype(np.float64)
                    if protocol == 'fixed':
                        h = h[fixed_mask]
                    previous_ratio = data['explained_variance_ratio'].astype(np.float64)
                assert h.ndim == 2 and h.shape[1] == 256 and np.isfinite(h).all()
                pca = PCA(svd_solver='full').fit(h)
                scores = pca.transform(h)
                ratio = pca.explained_variance_ratio_
                cum = ratio.cumsum()
                reconstruction_error = float(np.linalg.norm(scores @ pca.components_ + pca.mean_ - h) / np.linalg.norm(h - pca.mean_))
                orthogonality_error = float(np.max(np.abs(pca.components_ @ pca.components_.T - np.eye(256))))
                variance_error = float(abs(pca.explained_variance_.sum() / h.var(axis=0, ddof=1).sum() - 1))
                previous_spectrum_error = float(np.max(np.abs(ratio - previous_ratio)))
                assert np.all(ratio >= 0) and np.all(np.diff(ratio) <= 0)
                assert abs(cum[-1] - 1) < 1e-12
                assert reconstruction_error < 1e-12 and orthogonality_error < 1e-12
                assert variance_error < 1e-12 and previous_spectrum_error < 1e-6
                row = dict(protocol=protocol, condition=condition, seed=seed, samples=len(h),
                    cumulative_variance={f'pc{k}': float(cum[k-1]) for k in LEVELS},
                    dimensions={f'p{q}': int(np.searchsorted(cum, q/100) + 1) for q in THRESHOLDS},
                    participation_ratio=float(1 / (ratio**2).sum()),
                    explained_variance_ratio=ratio.tolist(),
                    validation=dict(relative_reconstruction_error=reconstruction_error,
                        orthogonality_max_error=orthogonality_error, variance_relative_error=variance_error,
                        previous_variance_fraction_max_error=previous_spectrum_error))
                rows.append(row)
                if protocol == 'variable':
                    np.savez_compressed(args.out / f'{condition}_s{seed}_pca.npz',
                        scores_first12=scores[:, :12], components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=ratio)
                print(f'{protocol} {condition} s{seed}: PC3 {100*cum[2]:.2f}%; '
                      f'PCs50/80/90/95/99 {list(row["dimensions"].values())}', flush=True)
    assert all(fingerprint(ROOT / path) == value for path, value in hashes.items())
    summary = dict(protocol=dict(
        method='Full float64 SVD PCA on mean-centered hidden h; no neuron-wise scaling; each model fit independently.',
        variables='All 256 hidden units; c not analyzed.', levels=list(LEVELS), thresholds=list(THRESHOLDS),
        variable_samples=3393, fixed_samples=int(fixed_mask.sum()),
        statistics='Means/SEM across eight paired training seeds; differences descriptive, no p-value screening.',
        limitation='Linear variance is not gridness, topology, intrinsic manifold dimension, dynamical stability, or behavior. Fixed/variable replay conditions differ in several factors.'),
        models=rows, comparisons={p: paired_metrics(rows, p) for p in ('variable', 'fixed')},
        source_sha256=hashes)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_readme(summary, args.out)
    make_figures(summary, args.out)
    print('Wrote all PCA refits, spectra, tables, and figures.', flush=True)


if __name__ == '__main__':
    main()
