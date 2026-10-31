"""Test whether leading PCs retain goal and movement information in frozen LSTMs.

Uses saved activity only; fits diagnostic readouts, never the trained policies.
PCA and ridge regularization are fitted inside whole-episode validation folds.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
from scipy import stats
from decode_lstm_variable import splits, angular_errors

SOURCE = ROOT / 'results/lstm_variable_1B'
OUT = ROOT / 'results/lstm_pc_information_1B'
COUNTS = (3, 10, 20, 50, 100, 256)
ALPHAS = 10. ** np.arange(-2, 6)
TARGETS = ('memory_position', 'direction')
CONDITIONS = ('gc100', 'pgc33')


def prefix_predictions(train_x, train_y, test_x):
    """One training-only SVD serves every PCA prefix and ridge parameter.

    A common scalar sets mean feature variance to one, without changing the
    PCA axes or separately rescaling neurons/PCs. Predictions use the original
    singular-value scales (no whitening). At k=256 this equals full ridge.
    """
    xm, ym = train_x.mean(0), train_y.mean(0)
    scale = max(float(np.sqrt(np.mean((train_x - xm) ** 2))), 1e-12)
    x, test = (train_x - xm) / scale, (test_x - xm) / scale
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    uy, test_v = u.T @ (train_y - ym), test @ vt.T
    predictions = []
    for count in COUNTS:
        s = singular[:count]
        predictions.append(np.stack([
            test_v[:, :count] @ ((s / (s*s + alpha))[:, None] * uy[:count]) + ym
            for alpha in ALPHAS]))
    return np.stack(predictions)


def fit_readouts(h, y, episode):
    predictions = np.full((len(COUNTS), len(y), 4), np.nan)
    baseline = np.full_like(y, np.nan)
    visits = np.zeros(len(y), int)
    folds = []
    for train, test in splits(episode, 4):
        losses, inner_folds = [], []
        train_h, train_y, train_ep = h[train], y[train], episode[train]
        for inner_train, inner_test in splits(train_ep, 3):
            pred = prefix_predictions(train_h[inner_train], train_y[inner_train], train_h[inner_test])
            # [PC count, alpha, sample, target, xy] -> independent alpha per target.
            loss = ((pred - train_y[inner_test][None, None]) ** 2).reshape(
                len(COUNTS), len(ALPHAS), -1, 2, 2).mean(axis=(2, 4))
            losses.append(loss)
            inner_folds.append(dict(train_episodes=np.unique(train_ep[inner_train]).tolist(),
                                    test_episodes=np.unique(train_ep[inner_test]).tolist()))
        choices = np.argmin(np.mean(losses, axis=0), axis=1)  # [PC count, target]
        candidates = prefix_predictions(train_h, train_y, h[test])
        for k in range(len(COUNTS)):
            for target in range(2):
                sl = slice(2 * target, 2 * target + 2)
                predictions[k, test, sl] = candidates[k, choices[k, target], :, sl]
        baseline[test] = train_y.mean(0)
        visits[test] += 1
        folds.append(dict(train_episodes=np.unique(episode[train]).tolist(),
            test_episodes=np.unique(episode[test]).tolist(), inner_folds=inner_folds,
            alpha_by_pc_and_target=ALPHAS[choices].tolist(),
            train_samples=int(train.sum()), test_samples=int(test.sum())))
    assert (visits == 1).all() and np.isfinite(predictions).all() and np.isfinite(baseline).all()
    return predictions, baseline, folds


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = b - a
    test = stats.ttest_rel(b, a)
    ci = test.confidence_interval(.95)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    p_exact = float(np.mean(np.abs((signs*d).mean(1)) >= abs(d.mean()) - 1e-12))
    return dict(grid_mean=float(a.mean()), pgc_mean=float(b.mean()),
        grid=a.tolist(), pgc=b.tolist(), pgc_minus_grid=d.tolist(),
        mean_difference=float(d.mean()), sem_difference=float(stats.sem(d)),
        marginal_t_ci95=[float(ci.low), float(ci.high)],
        pgc_lower_seeds=int((b < a).sum()), paired_t_p=float(test.pvalue), exact_signflip_p=p_exact)


def holm(records, field):
    order = sorted(records, key=lambda k: records[k][field])
    running = 0.
    for i, key in enumerate(order):
        running = max(running, min(1., records[key][field] * (len(order) - i)))
        records[key][field + '_holm4'] = running


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    replay_path = SOURCE / 'replay.npz'
    hashes = {str(replay_path.relative_to(ROOT)): hashlib.sha256(replay_path.read_bytes()).hexdigest()}
    with np.load(replay_path) as r:
        saved_index = r['sample_index']
        known = r['goal_known'][saved_index].astype(bool)
        index = saved_index[known]
        episode = r['episode'][index]
        memory = r['memory_pos'][index].astype(float)
        direction = r['dir'][index].astype(float)
    assert len(np.unique(episode)) == 16
    assert np.allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-5)
    y = np.column_stack([memory, direction])
    saved = dict(sample_index=index, episode=episode, memory_position=memory, direction=direction,
                 pc_counts=np.array(COUNTS))
    rows = []
    for seed in range(1, 9):
        for condition in CONDITIONS:
            path = SOURCE / f'{condition}_s{seed}_activity.npz'
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as f:
                h = f['h'][known].astype(float)
            assert h.shape == (len(y), 256) and np.isfinite(h).all()
            prediction, baseline, folds = fit_readouts(h, y, episode)
            metrics = []
            for i, count in enumerate(COUNTS):
                goal_rmse = np.sqrt(np.mean(np.sum((prediction[i, :, :2] - memory)**2, axis=1)))
                base_rmse = np.sqrt(np.mean(np.sum((baseline[:, :2] - memory)**2, axis=1)))
                direction_mae = angular_errors(prediction[i, :, 2:], direction).mean()
                base_mae = angular_errors(baseline[:, 2:], direction).mean()
                direction_mse = np.mean(np.sum((prediction[i, :, 2:] - direction)**2, axis=1))
                metrics.append(dict(pcs=count,
                    memory_position=dict(rmse_m=float(goal_rmse), baseline_rmse_m=float(base_rmse)),
                    direction=dict(mae_deg=float(direction_mae), baseline_mae_deg=float(base_mae),
                                   vector_mse=float(direction_mse))))
            rows.append(dict(condition=condition, seed=seed, metrics=metrics, folds=folds))
            saved[f'{condition}_s{seed}_prediction'] = prediction
            saved[f'{condition}_s{seed}_baseline'] = baseline
            print(f'{condition} s{seed}: goal cm(10/full)={100*metrics[1]["memory_position"]["rmse_m"]:.1f}/'
                  f'{100*metrics[-1]["memory_position"]["rmse_m"]:.1f}; direction deg(10/full)='
                  f'{metrics[1]["direction"]["mae_deg"]:.1f}/{metrics[-1]["direction"]["mae_deg"]:.1f}', flush=True)
    by = {(r['condition'], r['seed']): {m['pcs']: m for m in r['metrics']} for r in rows}
    comparisons, all_counts = {}, {}
    for target, metric in [('memory_position', 'rmse_m'), ('direction', 'mae_deg')]:
        values = {c: np.array([[by[c, s][k][target][metric] for k in COUNTS] for s in range(1, 9)]) for c in CONDITIONS}
        comparisons[target + '_10pc_error'] = paired(*(values[c][:, 1] for c in CONDITIONS))
        comparisons[target + '_compression_penalty'] = paired(*(values[c][:, 1] - values[c][:, -1] for c in CONDITIONS))
        all_counts[target] = {str(k): paired(*(values[c][:, j] for c in CONDITIONS)) for j, k in enumerate(COUNTS)}
    for field in ('paired_t_p', 'exact_signflip_p'):
        holm(comparisons, field)
    summary = dict(protocol=dict(
        hypothesis='Do the first 10 PCs of PGC hidden activity support better goal/movement readout, and lose less readout accuracy relative to all 256 PCs?',
        primary_pc_count=10, pc_counts=list(COUNTS), samples=len(y), episodes=16, training_seeds=list(range(1,9)),
        targets=['Remembered pickup x,y (not physical goal center)', 'Movement direction cosine,sine'],
        selection='Same 3392 known-goal states for both targets and all models; the one unknown-goal state is excluded.',
        method='Linear ridge readouts of PCA prefixes. 4 whole-episode outer folds and 3 whole-episode inner folds. PCA, centering, and a scalar feature scale are fitted on each training fold only. No neuron/PC rescaling or whitening.',
        alphas=ALPHAS.tolist(), alpha_selection='Inner validation mean squared coordinate error, separately for each PC count and target.',
        full_anchor='Fresh 256-PC readout on identical samples/alpha grid. Equivalent to ridge on all h; do not substitute the earlier full readout with its smaller alpha grid.',
        compression_penalty='Error using 10 PCs minus error using 256 PCs. Lower means less loss when restricted to leading PCs. Negative values are allowed: finite-sample PCA denoising can improve generalization.',
        statistics='Four primary contrasts: two targets times 10-PC error/compression penalty. Two-sided paired t and exhaustive sign-flip across 8 seeds, Holm corrected within these 4 contrasts separately per method. Pointwise t CIs are not simultaneous. Other PC counts are descriptive.',
        limitations='Exploratory linear decodability on one scripted replay; goal code is externally supplied, movement/facing/vision are correlated. A result cannot establish causal use, recurrent attractors, gridness, or why reward improves. No new policy rollouts or model training.'),
        models=rows, primary_comparisons=comparisons, descriptive_by_pc=all_counts,
        source_sha256=hashes)
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(OUT / 'predictions.npz', **saved)
    print('Finished all 16 models; source activity unchanged.', flush=True)


if __name__ == '__main__':
    main()
