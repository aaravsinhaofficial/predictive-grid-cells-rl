"""Compare 1B LSTM code use, spatial readouts, and population geometry.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_mechanism_analysis.py

This is an exploratory analysis of saved results and frozen weights. It does
not train policies or treat samples/episodes as independent training seeds.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from code.common import build_population

OUT = ROOT / 'results/lstm_mechanisms_1B'
SOURCE = ROOT / 'results/lstm_pca_1B'
CONDITIONS = ('gc100', 'pgc33')
SEEDS = range(1, 9)
ALPHAS = np.array([0.01, 0.1, 1., 10., 100., 1000.])


def paired(a, b):
    """Report b - a over independent matched training seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ci = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    permutations = np.array([np.mean(d * s) for s in itertools.product((-1, 1), repeat=len(d))])
    return dict(n=len(d), a=a.tolist(), b=b.tolist(), mean_a=float(a.mean()),
                mean_b=float(b.mean()), mean_difference=float(d.mean()), differences=d.tolist(),
                ci95=list(map(float, ci)), paired_t_p=float(stats.ttest_rel(b, a).pvalue),
                exact_signflip_p=float(np.mean(np.abs(permutations) >= abs(d.mean()) - 1e-12)))


def holm(records):
    names = sorted(records, key=lambda k: records[k]['paired_t_p'])
    running = 0.
    for i, name in enumerate(names):
        running = max(running, min(1., (len(names) - i) * records[name]['paired_t_p']))
        records[name]['holm_p_within_family'] = running


def ridge_predictions(x, y, test, alphas=ALPHAS, pcs=None):
    """Fit centering, scalar feature scale, and optional PCA on training only.

    One scalar standardizes mean per-feature variance; individual neurons are
    not rescaled. SVD gives all ridge penalties without repeated decompositions.
    """
    xm, ym = x.mean(0), y.mean(0)
    scale = max(float(np.sqrt(np.mean((x - xm) ** 2))), 1e-12)
    xc, xt = (x - xm) / scale, (test - xm) / scale
    u, sv, vt = np.linalg.svd(xc, full_matrices=False)
    if pcs is not None:
        u, sv, vt = u[:, :pcs], sv[:pcs], vt[:pcs]
    uy = u.T @ (y - ym)
    xtv = xt @ vt.T
    return np.stack([xtv @ ((sv / (sv**2 + alpha))[:, None] * uy) + ym for alpha in alphas])


def spatial_splits(groups, n=4, buffer=True):
    """Hold out contiguous bands of sampled lanes, optionally guard adjacent lanes."""
    unique = np.unique(groups)
    for test_groups in np.array_split(unique, n):
        test = np.isin(groups, test_groups)
        forbidden = list(test_groups)
        if buffer:
            lo, hi = np.searchsorted(unique, test_groups[[0, -1]])
            if lo > 0:
                forbidden.append(unique[lo - 1])
            if hi < len(unique) - 1:
                forbidden.append(unique[hi + 1])
        train = ~np.isin(groups, forbidden)
        assert train.any() and test.any() and not np.any(train & test)
        yield train, test


def decode(x, y, groups, pcs=None):
    prediction = np.full_like(y, np.nan, dtype=float)
    folds = []
    for train, test in spatial_splits(groups):
        # Inner validation is also grouped by contiguous spatial bands. Alpha
        # and optional PCA never see the outer test or its adjacent guard lanes.
        tx, ty, tg = x[train], y[train], groups[train]
        losses = []
        for inner_train, inner_test in spatial_splits(tg, n=3, buffer=False):
            preds = ridge_predictions(tx[inner_train], ty[inner_train], tx[inner_test], pcs=pcs)
            losses.append(np.mean((preds - ty[inner_test][None]) ** 2, axis=(1, 2)))
        alpha = ALPHAS[np.argmin(np.mean(losses, axis=0))]
        prediction[test] = ridge_predictions(tx, ty, x[test], [alpha], pcs=pcs)[0]
        folds.append(dict(alpha=float(alpha), train_n=int(train.sum()), test_n=int(test.sum()),
                          test_lanes=np.unique(groups[test]).tolist()))
    assert np.isfinite(prediction).all()
    squared = (prediction - y) ** 2
    return dict(rmse_m=float(np.sqrt(squared.sum(1).mean())),
                r2_xy=(1 - squared.sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist(),
                folds=folds), prediction


def analysis():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(314159)
    extension = json.loads((ROOT / 'results_ext/extension.json').read_text())['runs']
    score = {c: {} for c in CONDITIONS}
    probes = ('intact', 'no_code', 'no_current', 'no_goal', 'no_pgc', 'pgc_as_gc', 'shuffled_current')
    for c in CONDITIONS:
        for probe in probes:
            score[c][probe] = np.array([extension[f'{c}_lstm_s{s}']['milestones']['1B'][probe] for s in SEEDS])
        score[c]['sham_mean'] = np.array([
            np.mean([extension[f'{c}_lstm_s{s}']['milestones']['1B'][f'sham{k}'] for k in range(5)]) for s in SEEDS])
    # The intact gap is the project's original endpoint. The remaining tests
    # below form a new exploratory family; they are not prespecified outcomes.
    behavior = {'intact_gap': paired(score['gc100']['intact'], score['pgc33']['intact'])}
    behavior['gap_without_goal'] = paired(score['gc100']['no_goal'], score['pgc33']['no_goal'])
    behavior['gap_without_current'] = paired(score['gc100']['no_current'], score['pgc33']['no_current'])
    for probe in ('pgc_as_gc', 'no_pgc', 'sham_mean', 'no_goal'):
        behavior[f'pgc_cost_{probe}'] = paired(score['pgc33'][probe], score['pgc33']['intact'])
    behavior['pgc_specific_lesion_cost'] = paired(score['pgc33']['no_pgc'], score['pgc33']['sham_mean'])
    behavior['extra_goal_dependence'] = paired(
        score['gc100']['intact'] - score['gc100']['no_goal'],
        score['pgc33']['intact'] - score['pgc33']['no_goal'])
    behavior['extra_current_dependence'] = paired(
        score['gc100']['intact'] - score['gc100']['no_current'],
        score['pgc33']['intact'] - score['pgc33']['no_current'])
    holm({k: v for k, v in behavior.items() if k != 'intact_gap'})

    replay = dict(np.load(SOURCE / 'replay.npz'))
    mask = replay['selected']
    pos = replay['pos'][mask].astype(float)
    lanes = replay['lane'][mask]
    rows, predictions, fingerprints = [], {}, {}
    rng = np.random.default_rng(314159)
    sample_pos = torch.tensor(rng.uniform(-1.1, 1.1, (128, 2)), dtype=torch.float32)
    theta = torch.tensor(rng.uniform(-np.pi, np.pi, 128), dtype=torch.float32)
    direction = torch.stack((theta.cos(), theta.sin()), 1)
    for seed in SEEDS:
        mixed_cfg = json.loads((ROOT / f'models/pgc33/pgc33_lstm_s{seed}_1000M.json').read_text())
        mixed = build_population(mixed_cfg, 'cpu')
        labels = mixed.is_pgc.numpy()
        for c in CONDITIONS:
            cfg_path = ROOT / f'models/{c}/{c}_lstm_s{seed}_1000M.json'
            ckpt_path = cfg_path.with_suffix('.pt')
            cfg = json.loads(cfg_path.read_text())
            fingerprints[str(ckpt_path.relative_to(ROOT))] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            assert weights['frames'] == 1_000_000_000
            w = weights['policy']['lstm.weight_ih'].numpy().astype(float)
            data = dict(np.load(SOURCE / f'{c}_s{seed}_activity.npz'))
            h = data['h'][mask].astype(float)
            spectrum = data['explained_variance_ratio'].astype(float)
            row = dict(condition=c, seed=seed, pc3_variance=float(spectrum[:3].sum()),
                       participation_ratio=float(spectrum.sum()**2 / np.sum(spectrum**2)))
            for name, pcs in [('full', None), ('pc3', 3)]:
                result, pred = decode(h, pos, lanes, pcs=pcs)
                row[f'position_decode_{name}'] = result
                predictions[f'{c}_s{seed}_{name}'] = pred
            for block, offset in [('current', 256), ('goal', 768)]:
                norms = np.linalg.norm(w[:, offset:offset + 512], axis=0)
                row[f'{block}_pgc_to_gc_weight_ratio'] = float(norms[labels].mean() / norms[~labels].mean())
            population = build_population(cfg, 'cpu')
            with torch.no_grad():
                same = population.activity(sample_pos, direction)
                opposite = population.activity(sample_pos, -direction)
                row['opposite_direction_code_rms'] = float(((same - opposite)**2).mean().sqrt())
                # Local differential rank of the joint position/direction code.
                eps = 1e-3
                derivatives = []
                for axis in range(2):
                    delta = torch.zeros_like(sample_pos)
                    delta[:, axis] = eps
                    derivatives.append((population.activity(sample_pos + delta, direction) -
                                        population.activity(sample_pos - delta, direction)) / (2 * eps))
                dplus = torch.stack(((theta + eps).cos(), (theta + eps).sin()), 1)
                dminus = torch.stack(((theta - eps).cos(), (theta - eps).sin()), 1)
                derivatives.append((population.activity(sample_pos, dplus) -
                                    population.activity(sample_pos, dminus)) / (2 * eps))
                j = torch.stack(derivatives, -1).numpy().astype(float)
                sv = np.linalg.svd(j, compute_uv=False)
                row['local_jacobian_rank_median'] = float(np.median((sv > sv[:, :1] * 1e-4).sum(1)))
                # Portion of direction derivative not explained by translating x/y.
                residuals = []
                for jj in j:
                    dtheta = jj[:, 2]
                    residual = dtheta - jj[:, :2] @ np.linalg.lstsq(jj[:, :2], dtheta, rcond=None)[0]
                    residuals.append(np.linalg.norm(residual) / max(np.linalg.norm(dtheta), 1e-12))
                row['direction_derivative_independent_fraction'] = float(np.mean(residuals))
            rows.append(row)
            print(f'{c} seed {seed}: position RMSE {row["position_decode_full"]["rmse_m"]:.3f} m; '
                  f'current/goal weight ratios {row["current_pgc_to_gc_weight_ratio"]:.2f}/'
                  f'{row["goal_pgc_to_gc_weight_ratio"]:.2f}', flush=True)

    by = {(r['condition'], r['seed']): r for r in rows}
    representation = {}
    for metric in ('pc3_variance', 'participation_ratio', 'current_pgc_to_gc_weight_ratio', 'goal_pgc_to_gc_weight_ratio'):
        representation[metric] = paired([by['gc100', s][metric] for s in SEEDS], [by['pgc33', s][metric] for s in SEEDS])
    for name in ('full', 'pc3'):
        representation[f'position_rmse_{name}'] = paired(
            [by['gc100', s][f'position_decode_{name}']['rmse_m'] for s in SEEDS],
            [by['pgc33', s][f'position_decode_{name}']['rmse_m'] for s in SEEDS])
    holm(representation)
    correlations = {}
    gap = score['pgc33']['intact'] - score['gc100']['intact']
    for metric, record in representation.items():
        r, p = stats.pearsonr(record['differences'], gap)
        correlations[metric] = dict(pearson_r=float(r), p=float(p), n=8)
    summary = dict(
        protocol=dict(checkpoint_frames=1_000_000_000, seeds=list(SEEDS),
                      behavior_source='results_ext/extension.json',
                      representation_source='results/lstm_pca_1B/*_activity.npz',
                      decoder='Nested ridge; four contiguous held-out y bands, one adjacent sampled-lane guard on each side; three inner band folds without guards; training-only centering/scalar scaling/PCA',
                      alphas=ALPHAS.tolist(), decoder_samples=len(pos),
                      limitation='One fixed room/goal/heading and eastward samples. Position readout cannot distinguish a future-position code from current position plus a constant shift; it cannot isolate goal coding.',
                      statistics='Paired t and exact two-sided sign-flip over 8 training seeds; Holm t-p correction separately within exploratory behavior and representation families; correlations descriptive, uncorrected.'),
        behavior=behavior, representation=representation, performance_correlations=correlations,
        per_model=rows, checkpoint_sha256=fingerprints,
        score_by_probe={c: {k: v.tolist() for k, v in x.items()} for c, x in score.items()})
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(OUT / 'position_predictions.npz', pos=pos, lanes=lanes, **predictions)
    make_figure(summary)
    return summary


def make_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    colors = ['#3478ac', '#d66b37']
    labels = ['Grid only', 'Grid + 33% PGC']
    ax = axes[0, 0]
    names = ['intact', 'no_goal', 'pgc_as_gc', 'no_pgc', 'sham_mean']
    ticks = ['Intact', 'Goal memory\nremoved', 'PGCs\nde-shifted', '169 PGCs\nsilenced', '169 ordinary\ncells silenced']
    for i, c in enumerate(CONDITIONS):
        vals = np.array([summary['score_by_probe'][c][k] for k in names])
        ax.bar(np.arange(len(names)) + (i - .5) * .36, vals.mean(1), .34,
               yerr=stats.sem(vals, axis=1), color=colors[i], label=labels[i], capsize=2)
    ax.set(xticks=np.arange(len(names)), xticklabels=ticks, ylabel='Score / episode (mean ± SEM)', title='A  |  Existing 1B behavioral interventions')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(fontsize=8)
    ax.text(.02, .03, 'Grid-only has no PGCs: those probes leave it intact.', transform=ax.transAxes, fontsize=8)

    def paired_panel(ax, a, b, title, ylabel):
        for aa, bb in zip(a, b):
            ax.plot([0, 1], [aa, bb], color='#a3a9ae', alpha=.7, lw=1, zorder=1)
        for i, vals in enumerate([a, b]):
            ax.scatter(np.full(8, i), vals, color=colors[i], s=26, zorder=2)
            ax.scatter(i, np.mean(vals), color='black', marker='_', s=250, zorder=3)
        ax.set(xticks=[0, 1], xticklabels=labels, title=title, ylabel=ylabel, xlim=(-.3, 1.3))

    d = summary['representation']['position_rmse_full']
    paired_panel(axes[0, 1], np.array(d['a']) * 100, np.array(d['b']) * 100,
                 'B  |  Position decoded from all 256 hidden units', 'Held-out position RMSE (cm; lower is better)')
    axes[0, 1].set_title('B  |  Position decoded from all 256 hidden units\n'
                         f'Spatially blocked nested CV; paired p = {d["paired_t_p"]:.3g}', fontsize=11)
    d = summary['representation']['goal_pgc_to_gc_weight_ratio']
    paired_panel(axes[1, 0], d['a'], d['b'], 'C  |  Learned goal-memory input weights', 'Mean per-cell norm: PGC-designated / ordinary slots')
    axes[1, 0].axhline(1, color='#999999', ls=':', lw=1)
    axes[1, 0].text(.02, .96, 'Same designated cell indices in each seed pair.\nWeight magnitude is not a causal importance measure.', va='top', transform=axes[1, 0].transAxes, fontsize=8)
    ax = axes[1, 1]
    d = summary['representation']['pc3_variance']
    gap = summary['behavior']['intact_gap']['differences']
    ax.scatter(np.array(d['differences']) * 100, gap, color='#606b78', s=40)
    for i, (x, y) in enumerate(zip(np.array(d['differences']) * 100, gap), 1):
        ax.annotate(str(i), (x, y), xytext=(4, 4), textcoords='offset points', fontsize=9)
    corr = summary['performance_correlations']['pc3_variance']
    ax.axhline(0, color='#bbbbbb', lw=1)
    ax.axvline(0, color='#bbbbbb', lw=1)
    ax.set(xlabel='PGC − grid: PC1–3 variance (percentage points)', ylabel='PGC − grid: intact score',
           title=f'D  |  PCA compression vs performance (r={corr["pearson_r"]:.2f}, p={corr["p"]:.2f})')
    fig.suptitle('What distinguishes the 1B-frame LSTM populations?  ·  Eight matched seeds', fontsize=14)
    fig.supxlabel('Exploratory analysis. Behavior: existing 200-episode evaluations. Hidden states: one controlled eastward replay.', fontsize=9)
    fig.savefig(OUT / 'mechanism_comparison.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'mechanism_comparison.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    result = analysis()
    for family in ('behavior', 'representation'):
        for name, r in result[family].items():
            print(f'{family}/{name}: {r["mean_a"]:.4g} -> {r["mean_b"]:.4g}, '
                  f'p={r["paired_t_p"]:.4g}, Holm={r.get("holm_p_within_family", float("nan")):.4g}')
