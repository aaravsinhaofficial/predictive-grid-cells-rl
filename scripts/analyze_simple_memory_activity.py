"""Fixed descriptive covariance and episode-held-out angular readouts.

Reads authenticated saved states and actor weights only. It never executes a
policy/environment, trains a network, selects neurons, or detects a ring.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
PROTOCOL = ROOT / 'docs/research/simple_memory_analysis_protocol.json'
TARGETS = ('body_heading', 'last_travel_direction', 'egocentric_goal_bearing')
ALPHA, NULL_SEED, NULLS, FOLDS = 10., 741023, 20, 5
ANGLE_FLOOR = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariances(x, episodes, block=4096):
    """Independent, sample-weighted total/within/between accumulation / N."""
    x, episodes = np.asarray(x), np.asarray(episodes)
    assert x.ndim == 2 and len(x) == len(episodes) and np.isfinite(x).all()
    n, d = x.shape
    ids, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    if not n:
        return {k: np.zeros((d, d)) for k in ('total', 'within', 'between')}, dict(
            rows=0, episodes=0, episode_ids=[], episode_counts=[], identity_error=None)
    origin = np.asarray(x[0], dtype=np.float64)
    means = np.asarray([origin+(np.asarray(x[inverse == i], np.float64)-origin).mean(0)
                        for i in range(len(ids))])
    mean = origin+np.average(means-origin, axis=0, weights=counts)
    total, within = np.zeros((d, d)), np.zeros((d, d))
    for start in range(0, n, block):
        z = np.asarray(x[start:start+block], dtype=np.float64)
        a, b = z-mean, z-means[inverse[start:start+block]]
        total += a.T @ a
        within += b.T @ b
    offsets = (means-mean) * np.sqrt(counts[:, None])
    between = offsets.T @ offsets
    denominator = np.linalg.norm(total)
    error = float(np.linalg.norm(total-within-between)/denominator) if denominator else None
    assert error is None or error < 1e-10, error
    return dict(total=total/n, within=within/n, between=between/n), dict(
        rows=n, episodes=len(ids), episode_ids=ids.tolist(), episode_counts=counts.tolist(),
        identity_error=error, zero_total_variance=bool(denominator == 0))


def spectrum(covariance):
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    assert values.min(initial=0) >= -1e-10 * max(values.max(initial=0), 1.)
    values = values.clip(0)
    total = float(values.sum())
    rank = int(np.sum(values > values[0]*1e-10)) if total else 0
    proportions = values/total if total else None
    gaps = ((values[:-1]-values[1:])/np.maximum(values[:-1], np.finfo(float).tiny)).tolist()
    result = dict(eigenvalues=values.tolist(), total_variance=total, numerical_rank=rank,
        proportions=None if proportions is None else proportions.tolist(),
        cumulative=None if proportions is None else proportions.cumsum().tolist(),
        top10_share=None if not total else float(proportions[:10].sum()),
        participation_ratio=None if not total else float(1/np.sum(proportions**2)),
        entropy_effective_rank=None if not total else float(np.exp(-np.sum(
            proportions[proportions > 0]*np.log(proportions[proportions > 0])))),
        dimensions_90pct=None if not total else int(np.searchsorted(proportions.cumsum(), .9)+1),
        relative_eigenvalue_gaps=gaps,
        zero_variance=not bool(total))
    return result, values, vectors


def subspace(first, second):
    a, b = first.shape[1], second.shape[1]
    cosines = np.linalg.svd(first.T @ second, compute_uv=False).clip(0, 1)
    return dict(first_dimensions=a, second_dimensions=b,
        principal_angles_degrees=np.degrees(np.arccos(cosines)).tolist(),
        mean_squared_overlap=float(np.mean(cosines**2)) if min(a, b) else None,
        overlap_fraction_of_first=float(np.sum(cosines**2)/a) if min(a, b) else None)


def covariance_analysis(x, episodes, actor=None):
    covs, counts = covariances(x, episodes)
    decomposed = {name: spectrum(value) for name, value in covs.items()}
    spectra = {name: value[0] for name, value in decomposed.items()}
    bases = {name: value[2][:, :min(10, value[0]['numerical_rank'])]
             for name, value in decomposed.items()}
    total_trace = spectra['total']['total_variance']
    result = dict(counts=counts, spectra=spectra,
        between_fraction=spectra['between']['total_variance']/total_trace if total_trace else None,
        subspaces={a+'_vs_'+b: subspace(bases[a], bases[b])
                   for a, b in itertools.combinations(covs, 2)})
    split_bases, split_counts = [], []
    for parity in (0, 1):
        mask = np.asarray(episodes) % 2 == parity
        scovs, scounts = covariances(x[mask], np.asarray(episodes)[mask])
        split_counts.append(scounts)
        split = {}
        for name, cov in scovs.items():
            spec, _, vec = spectrum(cov)
            split[name] = vec[:, :min(10, spec['numerical_rank'])]
        split_bases.append(split)
    result['even_odd_episode_stability'] = dict(counts=split_counts,
        subspaces={name: subspace(split_bases[0][name], split_bases[1][name]) for name in covs})
    result['actor_alignment'] = None
    if actor is not None:
        actor = np.asarray(actor, np.float64)
        assert actor.shape[1] == x.shape[1]
        actor = actor-actor.mean(0)
        actor_variance = {name: float(np.trace(actor @ cov @ actor.T)) for name, cov in covs.items()}
        denominator = actor_variance['total']
        mode_results = {}
        for name, (_, values, vectors) in decomposed.items():
            contribution = values*np.sum((actor @ vectors)**2, axis=0)
            trace = actor_variance[name]
            mode_results[name] = dict(absolute_contribution=contribution.tolist(),
                fraction_of_own_actor_variance=(contribution/trace).tolist() if trace > 0 else None,
                top10_share_of_own_actor_variance=float(contribution[:10].sum()/trace) if trace > 0 else None)
        result['actor_alignment'] = dict(centered_logit_variance=actor_variance,
            between_fraction=actor_variance['between']/denominator if denominator > 0 else None,
            modes=mode_results, zero_denominator=denominator <= 0)
    arrays = {name+'_covariance': value for name, value in covs.items()}
    arrays.update({name+'_eigenvectors': value[2] for name, value in decomposed.items()})
    return result, arrays


def angular_targets(arrays):
    heading = np.asarray(arrays['body_heading'], np.float64)
    travel = np.asarray(arrays['travel_dir'], np.float64)
    delta = np.asarray(arrays['goal_pos'], np.float64)-np.asarray(arrays['pos'], np.float64)
    angles = np.column_stack((heading, np.arctan2(travel[:, 1], travel[:, 0]),
        np.arctan2(delta[:, 1], delta[:, 0])-heading))
    assert np.isfinite(angles).all()
    assert np.all(np.linalg.norm(travel, axis=1) > 0), 'Last travel direction is undefined'
    assert np.all(np.linalg.norm(delta, axis=1) > 0), 'Goal bearing is undefined at goal center'
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def episode_rotation_offsets(episodes, replicates=NULLS, seed=NULL_SEED):
    """Index by numeric episode ID, so all models/targets share one table."""
    episodes = np.asarray(episodes)
    assert np.issubdtype(episodes.dtype, np.integer) and np.all(episodes >= 0)
    return np.random.default_rng(seed).uniform(0, 2*np.pi, (replicates, int(episodes.max())+1))


def fit_episode_readouts(x, targets, episodes, visible, known, *, alpha=ALPHA,
                         replicates=NULLS, seed=NULL_SEED):
    """One Cholesky factorization per fold; all target/null RHS solved together."""
    x, targets, episodes = np.asarray(x), np.asarray(targets), np.asarray(episodes)
    visible, known = np.asarray(visible, bool), np.asarray(known, bool)
    assert x.ndim == 2 and targets.shape == (len(x), 3, 2)
    assert np.isfinite(x).all() and np.isfinite(targets).all()
    offsets = episode_rotation_offsets(episodes, replicates, seed)
    prediction = np.full((len(x), replicates+1, 3, 2), np.nan)
    baseline = np.full((len(x), 3, 2), np.nan)
    folds, artifacts = [], {'null_episode_rotation_radians': offsets}
    for fold in range(FOLDS):
        train = known & visible & (episodes % FOLDS != fold)
        test = known & (episodes % FOLDS == fold)
        record = dict(fold=fold, train_rows=int(train.sum()), test_rows=int(test.sum()),
            train_episode_ids=np.unique(episodes[train]).tolist(),
            test_episode_ids=np.unique(episodes[test]).tolist(), factorization_count=0)
        assert not set(record['train_episode_ids']) & set(record['test_episode_ids'])
        if not train.any():
            record['status'] = 'no_known_visible_training_rows'
            folds.append(record)
            continue
        xt = np.asarray(x[train], np.float64)
        mean, scale = xt.mean(0), xt.std(0)
        constant = xt.max(0) == xt.min(0)
        scale[constant] = 1.
        xt = (xt-mean)/scale
        y = targets[train]
        rotated = np.empty((len(xt), replicates+1, 3, 2))
        rotated[:, 0] = y
        rotation = offsets[:, episodes[train]].T
        co, si = np.cos(rotation)[:, :, None], np.sin(rotation)[:, :, None]
        rotated[:, 1:, :, 0] = co*y[:, None, :, 0]-si*y[:, None, :, 1]
        rotated[:, 1:, :, 1] = si*y[:, None, :, 0]+co*y[:, None, :, 1]
        response = rotated.reshape(len(xt), -1)
        intercept = response.mean(0)
        matrix = xt.T @ xt
        matrix.flat[::len(matrix)+1] += alpha
        factor = cho_factor(matrix, lower=True, check_finite=False)
        beta = cho_solve(factor, xt.T @ (response-intercept), check_finite=False)
        # The physical-label training constant is unchanged across null fits.
        baseline[test] = y.mean(0)
        for start in range(0, int(test.sum()), 4096):
            ids = np.flatnonzero(test)[start:start+4096]
            pred = ((np.asarray(x[ids], np.float64)-mean)/scale) @ beta+intercept
            prediction[ids] = pred.reshape(len(ids), replicates+1, 3, 2)
        record.update(status='fitted', factorization_count=1, constant_coordinates=int(constant.sum()),
            training_baseline_vector=y.mean(0).tolist())
        folds.append(record)
        artifacts.update({f'fold{fold}_mean': mean, f'fold{fold}_scale': scale,
            f'fold{fold}_coefficients': beta.reshape(x.shape[1], replicates+1, 3, 2),
            f'fold{fold}_intercept': intercept.reshape(replicates+1, 3, 2)})
    return prediction, baseline, folds, artifacts


def averaged(values, episodes):
    values = np.asarray(values, float)
    if not len(values): return dict(pooled=None, episode_equal=None)
    episodes = np.asarray(episodes, dtype=np.int64)
    counts = np.bincount(episodes)
    sums = np.bincount(episodes, weights=values, minlength=len(counts))
    return dict(pooled=float(values.mean()),
                episode_equal=float(np.mean(sums[counts > 0]/counts[counts > 0])))


def comparison(pred_error, base_error, episodes):
    p, b = averaged(pred_error, episodes), averaged(base_error, episodes)
    return {weight: dict(ridge_mse=p[weight], constant_mse=b[weight],
        mse_improvement=None if p[weight] is None else b[weight]-p[weight],
        r2_vs_constant=None if b[weight] is None or b[weight] <= 1e-12 else 1-p[weight]/b[weight])
        for weight in ('pooled', 'episode_equal')}


def angular_metrics(prediction, target, baseline, episodes):
    prediction, target, baseline, episodes = map(np.asarray, (prediction, target, baseline, episodes))
    finite = np.isfinite(prediction).all(1) & np.isfinite(baseline).all(1)
    norms, baseline_norms = np.linalg.norm(prediction, axis=1), np.linalg.norm(baseline, axis=1)
    defined = finite & (norms >= ANGLE_FLOOR)
    base_defined = finite & (baseline_norms >= ANGLE_FLOOR)
    paired = defined & base_defined
    n = len(target)
    result = dict(rows=n, episodes=len(np.unique(episodes)), finite_prediction_rows=int(finite.sum()),
        finite_prediction_episodes=len(np.unique(episodes[finite])),
        missing_fit_rows=int((~finite).sum()), undefined_prediction_rows=int((finite & ~defined).sum()),
        undefined_baseline_rows=int((finite & ~base_defined).sum()),
        undefined_prediction_episodes=len(np.unique(episodes[finite & ~defined])),
        undefined_baseline_episodes=len(np.unique(episodes[finite & ~base_defined])),
        defined_angle_rows=int(defined.sum()), defined_angle_episodes=len(np.unique(episodes[defined])),
        defined_baseline_rows=int(base_defined.sum()), defined_baseline_episodes=len(np.unique(episodes[base_defined])),
        paired_angle_rows=int(paired.sum()), paired_angle_episodes=len(np.unique(episodes[paired])),
        paired_retained_fraction=float(paired.sum()/n) if n else None,
        true_label_resultant=float(np.linalg.norm(target.mean(0))) if n else None,
        prediction_norm_mean=float(norms[finite].mean()) if finite.any() else None,
        prediction_norm_quantiles=np.quantile(norms[finite], [.1, .5, .9]).tolist() if finite.any() else None)
    for name, pred, mask in [('ridge', prediction, defined), ('constant', baseline, base_defined)]:
        unit = pred[mask]/np.linalg.norm(pred[mask], axis=1, keepdims=True)
        cosine = np.sum(unit*target[mask], axis=1).clip(-1, 1)
        result[name] = dict(circular_mae_deg=averaged(np.degrees(np.arccos(cosine)), episodes[mask]),
                            cosine_accuracy=averaged(cosine, episodes[mask]))
    unit = prediction[paired]/norms[paired, None]
    base_unit = baseline[paired]/baseline_norms[paired, None]
    result['unit_angle_comparison'] = comparison(np.sum((unit-target[paired])**2, axis=1),
        np.sum((base_unit-target[paired])**2, axis=1), episodes[paired])
    result['sincos_regression_comparison'] = comparison(np.sum((prediction[finite]-target[finite])**2, axis=1),
        np.sum((baseline[finite]-target[finite])**2, axis=1), episodes[finite])
    return result


def readout_analysis(x, arrays):
    episodes = arrays['episode_id']
    known, blind = np.asarray(arrays['known'], bool), np.asarray(arrays['blind'], bool)
    visible = known & ~blind
    target = angular_targets(arrays)
    prediction, baseline, folds, artifacts = fit_episode_readouts(x, target, episodes, visible, known)
    bins = dict(all_known=known, known_visible=visible)
    age = arrays['phase_age']
    for lo, hi in ((1, 15), (16, 30), (31, 45), (46, 60)):
        bins[f'blind_age_{lo}_{hi}'] = known & blind & (age >= lo) & (age <= hi)
    result = dict(folds=folds, ridge_alpha=ALPHA, null_seed=NULL_SEED, null_replicates=NULLS,
        null_baseline='Unchanged original physical-label training-fold constant; only null training labels rotate.',
        factorization_count=sum(row['factorization_count'] for row in folds), targets={})
    for t, name in enumerate(TARGETS):
        result['targets'][name] = {}
        for label, mask in bins.items():
            measured = [angular_metrics(prediction[mask, replicate, t], target[mask, t], baseline[mask, t], episodes[mask])
                        for replicate in range(NULLS+1)]
            null_mae = [r['ridge']['circular_mae_deg']['pooled'] for r in measured[1:]]
            defined_mae = [v for v in null_mae if v is not None]
            result['targets'][name][label] = dict(actual=measured[0], null_replicates=measured[1:],
                null_pooled_mae_range=[min(defined_mae), max(defined_mae)] if defined_mae else None)
    return result, artifacts


def analyze_arrays(arrays, actor_weight):
    """Pure saved-array API: returns JSON summary and named numeric artifacts."""
    h, c = np.asarray(arrays['h']), np.asarray(arrays['c'])
    assert h.shape == c.shape and len(h) == len(arrays['episode_id'])
    actor_weight = np.asarray(actor_weight)
    assert actor_weight.shape == (6, h.shape[1])
    known, blind = np.asarray(arrays['known'], bool), np.asarray(arrays['blind'], bool)
    assert not np.any(blind & ~known)
    masks = dict(all=np.ones(len(h), bool), known_visible=known & ~blind, known_blind=known & blind)
    result, artifacts = dict(covariance={}, angular_readouts={}), {}
    for feature in ('h', 'c', 'hc'):
        x = h if feature == 'h' else c if feature == 'c' else np.concatenate((h, c), axis=1)
        actor = actor_weight if feature == 'h' else None
        if feature == 'hc': actor = np.concatenate((actor_weight, np.zeros_like(actor_weight)), axis=1)
        result['covariance'][feature] = {}
        for label, mask in masks.items():
            summary, numeric = covariance_analysis(x[mask], arrays['episode_id'][mask], actor)
            result['covariance'][feature][label] = summary
            artifacts.update({f'{feature}_{label}_{key}': value for key, value in numeric.items()})
        if feature != 'c':
            summary, numeric = readout_analysis(x, arrays)
            result['angular_readouts'][feature] = summary
            artifacts.update({f'{feature}_readout_{key}': value for key, value in numeric.items()})
    return result, artifacts


def authenticate(run, states, protocol_path):
    """No network construction: actor and hash checks use saved tensors only."""
    import torch
    protocol = json.loads(protocol_path.read_text())
    assert protocol['status'] == 'frozen', 'Analysis protocol is not frozen'
    manifest = protocol.get('source_sha256')
    own_source = Path(__file__).resolve().relative_to(ROOT).as_posix()
    assert isinstance(manifest, dict) and manifest and own_source in manifest, \
        'Frozen analysis source manifest must be nonempty and bind this producer'
    spec = protocol['native_angular_readouts']
    assert spec['ridge_alpha'] == ALPHA and spec['null']['seed'] == NULL_SEED and spec['null']['replicates'] == NULLS
    for path, digest in manifest.items(): assert sha(ROOT/path) == digest, path
    cfg = json.loads((run/'config.json').read_text())
    terminal = json.loads((run/'summary.json').read_text())
    proof = json.loads((run/'provenance.json').read_text())
    producer_evidence = dict(producer_version=1)
    if proof.get('provenance_version') == 2:
        from scripts.simple_memory_endpoint import validate_v2_provenance
        producer_evidence = validate_v2_provenance(run, cfg, proof, root=ROOT)
    else:
        assert 'provenance_version' not in proof, 'Unknown training producer version'
    saved = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
    assert terminal['complete'] and terminal['frames'] == saved['frames'] == cfg['frames'] == 1_000_000_000
    assert saved['updates'] == terminal['updates'] == 78125
    assert terminal['final_sha256'] == sha(run/'final.pt')
    assert saved['provenance'] == terminal['provenance'] == proof
    assert proof['protocol_sha256'] == sha(ROOT/protocol['training_protocol'])
    assert saved['config'] == {key: cfg[key] for key in saved['config']}
    for path, digest in proof['source_sha256'].items(): assert sha(ROOT/path) == digest, path
    hasher = hashlib.sha256()
    for name, value in sorted(saved['policy'].items()):
        hasher.update(name.encode()); hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert hasher.hexdigest() == saved['policy_sha256'] == terminal['final_policy_sha256']
    collection = json.loads((states/'summary.json').read_text())
    assert collection['complete'] and collection['checkpoint_sha256'] == terminal['final_sha256']
    assert collection['array_file']['sha256'] == sha(states/'states.npz')
    for key, value in dict(episodes=128, n_envs=32, seed=741022, stride=3, max_gap=60, samples=57600).items():
        assert collection[key] == value, key
    for path, digest in collection['source_sha256'].items(): assert sha(ROOT/path) == digest, path
    with np.load(states/'states.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ('h', 'c', 'episode_id', 'known', 'blind', 'phase_age',
            'body_heading', 'travel_dir', 'pos', 'goal_pos')}
    for name, array in arrays.items():
        item = collection['array_schema'][name]
        assert list(array.shape) == item['shape'] and str(array.dtype) == item['dtype']
        assert hashlib.sha256(array.tobytes()).hexdigest() == item['sha256']
    assert arrays['h'].shape == arrays['c'].shape == (57600, 256)
    ids, counts = np.unique(arrays['episode_id'], return_counts=True)
    assert np.array_equal(ids, np.arange(128)) and np.all(counts == 450)
    assert cfg['cond'] in ('gc100', 'pgc33') and cfg['seed'] in (1, 2, 3, 4) and cfg['task_max_gap'] in (0, 60)
    return arrays, saved['policy']['pi.weight'].numpy(), cfg, dict(
        final_checkpoint_sha256=terminal['final_sha256'], final_policy_sha256=saved['policy_sha256'],
        analysis_protocol_sha256=sha(protocol_path), training_protocol_sha256=proof['protocol_sha256'],
        collection_summary_sha256=sha(states/'summary.json'), states_sha256=sha(states/'states.npz'),
        producer_sha256=sha(__file__), **producer_evidence)


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    primary = summary['covariance']['h']['all']
    for label, name in [('All activity', 'total'), ('Episode means removed', 'within')]:
        curve = primary['spectra'][name]['cumulative']
        if curve is not None: axes[0].plot(np.arange(1, len(curve)+1), 100*np.asarray(curve), label=label)
    axes[0].set(xlim=(1, 100), ylim=(0, 100), xlabel='Principal components', ylabel='Variance explained (%)')
    axes[0].legend(frameon=False, fontsize=8)
    for target in TARGETS:
        rows = summary['angular_readouts']['h']['targets'][target]
        labels = ['known_visible']+[f'blind_age_{lo}_{lo+14}' for lo in (1, 16, 31, 46)]
        values = [rows[label]['actual']['unit_angle_comparison']['pooled']['r2_vs_constant'] for label in labels]
        axes[1].plot(range(5), [np.nan if v is None else v for v in values], marker='o', label=target.replace('_', ' '))
    axes[1].axhline(0, color='0.7', lw=1)
    axes[1].set(xticks=range(5), xticklabels=['Visible', '1–15', '16–30', '31–45', '46–60'],
                xlabel='Blind age (updates)', ylabel='Held-out improvement over constant (R²)')
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(out/'activity_summary.png', dpi=180)
    fig.savefig(out/'activity_summary.pdf')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--states', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args(argv)
    states, out = args.states or args.run/'memory_states', args.out or args.run/'activity_analysis'
    assert 1 <= args.threads <= 8
    if out.exists() and any(out.iterdir()): raise FileExistsError('Analysis output must be empty')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        arrays, actor, cfg, provenance = authenticate(args.run, states, args.protocol)
        result, artifacts = analyze_arrays(arrays, actor)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'activity_arrays.npz', **artifacts)
    result.update(complete=True, condition=cfg['cond'], training_seed=cfg['seed'],
        training_task_max_gap=cfg['task_max_gap'], created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        provenance=provenance, numeric_artifact_sha256=sha(out/'activity_arrays.npz'),
        scope='Descriptive covariance and episode-held-out angular information; no ring or causal performance test.',
        numerics=dict(accumulation='float64', threads=args.threads, numpy=np.__version__))
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.figures: plot(result, out)
    print(json.dumps(dict(complete=True, out=str(out), summary_sha256=sha(out/'summary.json'))))


if __name__ == '__main__':
    main()
