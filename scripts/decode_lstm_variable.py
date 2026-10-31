"""Linear readouts of frozen LSTMs on the shared variable-goal/direction replay.

Run after scripts/lstm_variable_replay.py:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/decode_lstm_variable.py

Only diagnostic ridge readouts are fitted; saved policies are never trained.
All test splits hold out whole episodes, including their goals and rooms.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
CONDITIONS = ('gc100', 'pgc33')
LABELS = ('Grid only', 'Grid + 33% PGC')


def splits(episode, n):
    """Each episode appears in one test fold and never in its training set."""
    unique = np.unique(episode)
    assert len(unique) >= n
    for test_ids in np.array_split(unique, n):
        test = np.isin(episode, test_ids)
        train = ~test
        assert train.any() and test.any()
        assert not set(episode[train]) & set(episode[test])
        yield train, test


def ridge_grid(train_x, train_y, test_x, alphas=ALPHAS):
    """One SVD serves every target and ridge penalty; preprocessing is train-only."""
    mean_x, mean_y = train_x.mean(0), train_y.mean(0)
    scale = max(float(np.sqrt(np.mean((train_x - mean_x) ** 2))), 1e-12)
    x, test = (train_x - mean_x) / scale, (test_x - mean_x) / scale
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    uy, test_v = u.T @ (train_y - mean_y), test @ vt.T
    return np.stack([test_v @ ((singular / (singular**2 + a))[:, None] * uy) + mean_y
                     for a in alphas])


def nested_readouts(x, targets, episode):
    """Share decompositions across two-coordinate targets, tune each separately."""
    names = list(targets)
    y = np.concatenate([targets[name] for name in names], axis=1)
    prediction, baseline = np.full_like(y, np.nan), np.full_like(y, np.nan)
    count = np.zeros(len(y), dtype=int)
    folds = {name: [] for name in names}
    for train, test in splits(episode, 4):
        losses = []
        for inner_train, inner_test in splits(episode[train], 3):
            p = ridge_grid(x[train][inner_train], y[train][inner_train], x[train][inner_test])
            error = (p - y[train][inner_test][None])**2
            # Equal weight per coordinate pair; independent alpha per target.
            losses.append(error.reshape(len(ALPHAS), -1, len(names), 2).mean(axis=(1, 3)))
        choices = np.argmin(np.mean(losses, axis=0), axis=0)
        final = ridge_grid(x[train], y[train], x[test])
        for j, name in enumerate(names):
            coords = slice(2*j, 2*j + 2)
            prediction[test, coords] = final[choices[j], :, coords]
            folds[name].append(dict(
                alpha=float(ALPHAS[choices[j]]),
                train_episodes=np.unique(episode[train]).tolist(),
                test_episodes=np.unique(episode[test]).tolist(),
                train_n=int(train.sum()), test_n=int(test.sum())))
        baseline[test] = y[train].mean(0)
        count[test] += 1
    assert np.all(count == 1) and np.isfinite(prediction).all() and np.isfinite(baseline).all()
    return {name: (prediction[:, 2*j:2*j+2], baseline[:, 2*j:2*j+2], folds[name])
            for j, name in enumerate(names)}


def angular_errors(prediction, target):
    pred_angle = np.arctan2(prediction[:, 1], prediction[:, 0])
    true_angle = np.arctan2(target[:, 1], target[:, 0])
    return np.rad2deg(np.abs((pred_angle - true_angle + np.pi) % (2*np.pi) - np.pi))


def metrics(prediction, baseline, target, direction=False):
    result = dict(samples=len(target),
                  euclidean_rmse=float(np.sqrt(np.mean(np.sum((prediction - target)**2, axis=1)))),
                  baseline_euclidean_rmse=float(np.sqrt(np.mean(np.sum((baseline - target)**2, axis=1)))))
    if direction:
        result.update(angular_mae_deg=float(angular_errors(prediction, target).mean()),
                      baseline_angular_mae_deg=float(angular_errors(baseline, target).mean()))
    else:
        result['units'] = 'meters'
    return result


def make_figure(summary, out):
    rows = {(r['condition'], r['seed']): r for r in summary['models']}
    panels = [('position', 'euclidean_rmse', 100, 'Current position', 'Held-out RMSE (cm)'),
              ('memory_position', 'euclidean_rmse', 100, 'Remembered pickup position', 'Held-out RMSE (cm)'),
              ('direction', 'angular_mae_deg', 1, 'Movement direction', 'Held-out angular error (degrees)')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout='constrained')
    for ax, (name, metric, factor, title, ylabel) in zip(axes, panels):
        values = np.array([[rows[c, seed][name][metric]*factor for seed in range(1, 9)]
                           for c in CONDITIONS])
        for seed, pair in enumerate(values.T, start=1):
            ax.plot([0, 1], pair, color='#aeb6bd', lw=1, zorder=1)
        for i, color in enumerate(('#3478ac', '#d66b37')):
            ax.scatter(np.full(8, i), values[i], color=color, s=28, zorder=2)
            ax.scatter(i, values[i].mean(), color='black', marker='_', s=250, zorder=3)
        baseline = rows['gc100', 1][name]['baseline_' + metric]*factor
        ax.axhline(baseline, color='#747b82', ls='--', lw=1, label='Training-mean baseline')
        ax.set(xticks=[0, 1], xticklabels=LABELS, xlim=(-.3, 1.3), title=title, ylabel=ylabel)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Frozen 1B LSTMs · varying movement and goal memory\n'
                 'Linear readouts tested on entirely held-out episodes; lines join training seeds', fontsize=13)
    fig.supxlabel('Lower is better. Goal readout uses known-goal samples. Decodability does not establish causal use or explain the reward advantage.', fontsize=9)
    fig.savefig(out / 'heldout_decoding.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / 'heldout_decoding.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/lstm_variable_1B')
    args = ap.parse_args()
    out = args.out
    replay_path = out / 'replay.npz'
    with np.load(replay_path) as data:
        index = data['sample_index']
        episode = data['episode'][index]
        pos, direction = data['pos'][index].astype(float), data['dir'][index].astype(float)
        known = data['goal_known'][index].astype(bool)
        memory = data['memory_pos'][index].astype(float)
    assert len(np.unique(episode)) >= 8
    assert len(np.unique(episode[known])) == len(np.unique(episode))
    assert np.allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-4)
    rows, predictions, hashes = [], dict(sample_index=index, episode=episode, pos=pos,
                                        direction=direction, goal_known=known, memory_pos=memory), {}
    for seed in range(1, 9):
        for condition in CONDITIONS:
            path = out / f'{condition}_s{seed}_activity.npz'
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                h = data['h'].astype(float)
            assert h.shape == (len(index), 256) and np.isfinite(h).all()
            fits = nested_readouts(h, dict(position=pos, direction=direction), episode)
            fits.update(nested_readouts(h[known], dict(memory_position=memory[known]), episode[known]))
            row = dict(condition=condition, seed=seed)
            for name, (pred, base, folds) in fits.items():
                target = {'position': pos, 'direction': direction, 'memory_position': memory[known]}[name]
                row[name] = metrics(pred, base, target, direction=(name == 'direction'))
                row[name]['folds'] = folds
                predictions[f'{condition}_s{seed}_{name}'] = pred
                predictions[f'{condition}_s{seed}_{name}_baseline'] = base
            rows.append(row)
            print(f'{condition} seed {seed}: position {row["position"]["euclidean_rmse"]*100:.1f} cm; '
                  f'goal snapshot {row["memory_position"]["euclidean_rmse"]*100:.1f} cm; '
                  f'direction {row["direction"]["angular_mae_deg"]:.1f} deg', flush=True)
    comparisons = {}
    by = {(r['condition'], r['seed']): r for r in rows}
    for name, metric in [('position', 'euclidean_rmse'), ('memory_position', 'euclidean_rmse'),
                         ('direction', 'angular_mae_deg')]:
        a, b = [np.array([by[c, seed][name][metric] for seed in range(1, 9)]) for c in CONDITIONS]
        comparisons[name] = dict(metric=metric, grid_mean=float(a.mean()), pgc_mean=float(b.mean()),
                                 pgc_minus_grid=(b-a).tolist(), mean_difference=float((b-a).mean()),
                                 difference_sem=float((b-a).std(ddof=1)/np.sqrt(8)),
                                 pgc_lower_error_seeds=int(np.sum(b < a)),
                                 baseline=float(rows[0][name]['baseline_' + metric]))
    summary = dict(protocol=dict(
        source='Shared variable-direction/variable-goal off-policy replay; all 256 hidden units h.',
        decoder='Nested ridge: 4 whole-episode outer folds, 3 whole-episode inner folds. Training-only centering and one scalar feature scale. No unit-wise rescaling or PCA.',
        alphas=ALPHAS.tolist(), tuning='Mean squared coordinate error; direction predicts cos/sin and is scored by wrapped angular error.',
        baseline='Outer-training target mean; evaluated on the same held-out points.',
        goal_target='Actual remembered pickup position, not the physical goal center. Unknown-goal samples excluded only for this target.',
        limitation='Goal code is supplied externally to the LSTM. Goal decodability does not demonstrate internal memory storage, causal use, or behavioral benefit. Whole-episode holdout tests generalization to new sampled goals and rooms; it does not isolate direction from every correlated input.',
        aggregation='Point-pooled errors per model; model comparisons are descriptive across eight matched training seeds.'),
        models=rows, comparisons=comparisons, replay_sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        activity_sha256=hashes)
    (out / 'decoding_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(out / 'decoding_predictions.npz', **predictions)
    make_figure(summary, out)


if __name__ == '__main__':
    main()
