"""Persistent-homology screen of saved 1B LSTM activity; no UMAP or training.

python scripts/lstm_topology.py
python scripts/plot_lstm_topology.py

Uses Euclidean distances in the full hidden vector, 256 farthest-point
landmarks, Z47 coefficients, and 99 independent circular-shift nulls per
dataset. The 512-landmark run is an observed-data sensitivity check only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import ripser as ripser_package
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits


def persistence(x, landmarks, coeff=47):
    """Scalar normalization changes units, not the cloud's topology."""
    condensed = pdist(x, metric='euclidean')
    scale = float(np.median(condensed))
    if not scale > 0:
        raise ValueError('Degenerate cloud with zero median distance')
    distances = squareform(condensed / scale)
    result = ripser(distances, distance_matrix=True, maxdim=2, coeff=coeff,
                    n_perm=min(landmarks, len(x)))
    for diagram in result['dgms'][1:]:
        if not np.isfinite(diagram).all():
            raise ValueError('Unexpected infinite H1/H2 bar in full filtration')
    return result['dgms'], dict(distance_scale=scale,
        covering_radius=float(result['r_cover']),
        landmark_indices=result['idx_perm'].tolist())


def ranked(diagram):
    finite = diagram[np.isfinite(diagram).all(axis=1)]
    return finite[np.argsort(finite[:, 1] - finite[:, 0])[::-1]]


def longest(diagram, rank=1):
    bars = ranked(diagram)
    return float(bars[rank - 1, 1] - bars[rank - 1, 0]) if len(bars) >= rank else 0.0


def topology_intervals(diagrams):
    """Coexistence of the top two H1 bars and top H2, then exact Betti counts.

    Even exact Betti numbers (1,2,1) do not uniquely identify a torus.
    """
    loops, surfaces = ranked(diagrams[1]), ranked(diagrams[2])
    if len(loops) < 2 or len(surfaces) < 1:
        return None, None
    h0 = ranked(diagrams[0])
    connected = float(h0[:, 1].max()) if len(h0) else 0.0
    starts = [connected, loops[0, 0], loops[1, 0], surfaces[0, 0]]
    ends = [loops[0, 1], loops[1, 1], surfaces[0, 1]]
    lo, hi = float(max(starts)), float(min(ends))
    if lo >= hi:
        return None, None
    critical = [lo, hi]
    for diagram in diagrams[1:]:
        critical.extend(diagram[(diagram > lo) & (diagram < hi)].tolist())
    critical = sorted(set(critical))
    best = None
    current = None
    for left, right in zip(critical[:-1], critical[1:]):
        mid = (left + right) / 2
        betti = [int(((d[:, 0] <= mid) & (mid < d[:, 1])).sum()) for d in diagrams]
        if betti == [1, 2, 1]:
            current = [current[0], right] if current is not None else [left, right]
            if best is None or current[1] - current[0] > best[1] - best[0]:
                best = list(current)
        else:
            current = None
    return [lo, hi], best


def empirical_p(value, null):
    return float((1 + np.count_nonzero(null >= value)) / (len(null) + 1))


def summarize(diagrams, null):
    thresholds = np.quantile(null, 0.95, axis=0, method='higher')
    h1first = longest(diagrams[1])
    h1second = longest(diagrams[1], 2)
    h2first = longest(diagrams[2])
    significant = [int(np.sum((d[:, 1] - d[:, 0]) > threshold))
                   for d, threshold in zip(diagrams[1:], thresholds)]
    common, betti = topology_intervals(diagrams)
    return dict(H1=float(thresholds[0]), H2=float(thresholds[1])), dict(
        h1_first=h1first, h1_second=h1second, h2_first=h2first,
        significant_h1=significant[0], significant_h2=significant[1],
        p_h1_second=empirical_p(h1second, null[:, 0]),
        p_h2_first=empirical_p(h2first, null[:, 1]),
        common_interval=common, torus_betti_interval=betti,
        betti_interval_width=float(betti[1] - betti[0]) if betti is not None else 0.0,
        core_torus_signature=bool(h1second > thresholds[0] and h2first > thresholds[1]
                                  and betti is not None),
        torus_screen=bool(significant == [2, 1] and betti is not None))


def serial_diagrams(diagrams):
    return [[[float(v) if np.isfinite(v) else None for v in row] for row in d]
            for d in diagrams]


def compute_dataset(job):
    description, x, opts = job
    name = description['name']
    start = time.time()
    with threadpool_limits(limits=1):
        diagrams, info = persistence(x, opts['landmarks'])
        dense, dense_info = persistence(x, opts['sensitivity_landmarks'])
        rng = np.random.default_rng(np.random.SeedSequence([opts['random_seed'], description['dataset_id']]))
        null = np.zeros((opts['shuffles'], 2))
        null_cover = []
        shifts_saved = []
        time_index = np.arange(len(x))[:, None]
        for i in range(opts['shuffles']):
            shifts = rng.integers(0, len(x), size=x.shape[1])
            shuffled = np.take_along_axis(x, (time_index - shifts[None, :]) % len(x), axis=0)
            null_diagrams, null_info = persistence(shuffled, opts['landmarks'])
            null[i] = [longest(null_diagrams[d]) for d in (1, 2)]
            null_cover.append(null_info['covering_radius'])
            shifts_saved.append(shifts)
            if (i + 1) % 33 == 0:
                print(f'{name}: {i+1}/{opts["shuffles"]} shuffles', flush=True)
        thresholds, metrics = summarize(diagrams, null)
        _, dense_betti = topology_intervals(dense)
        result = dict(description, dimensions=x.shape[1], samples=len(x),
            landmarks=opts['landmarks'], coeff=47, diagrams=serial_diagrams(diagrams),
            sampling=info, null_max_lifetimes=null.tolist(),
            thresholds=thresholds, metrics=metrics,
            null_covering_radius_range=[float(min(null_cover)), float(max(null_cover))],
            null_shifts= np.asarray(shifts_saved).tolist(),
            sensitivity512=dict(landmarks=opts['sensitivity_landmarks'],
                diagrams=serial_diagrams(dense), sampling=dense_info,
                h1_second=longest(dense[1], 2), h2_first=longest(dense[2]),
                torus_betti_interval=dense_betti,
                note='Observed-data geometry check only; no null test at this sample size.'),
            seconds=time.time() - start)
        path = Path(opts['out']) / f'{name}.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(f'{name}: DONE; significant H1/H2 = {metrics["significant_h1"]}/{metrics["significant_h2"]}; '
              f'torus screen = {metrics["torus_screen"]}; {time.time()-start:.1f}s', flush=True)
        return result


def datasets(source, models):
    with np.load(source / 'replay.npz') as archive:
        replay = dict(archive)
    selected = replay['selected']
    output = []
    for seed in range(1, 9):
        for condition, label in [('gc100', 'Grid only'), ('pgc33', 'Grid + 33% PGC')]:
            path = source / f'{condition}_s{seed}_activity.npz'
            with np.load(path) as archive:
                x = archive['h'][selected].astype(np.float64)
            assert x.shape == (1185, 256) and np.isfinite(x).all()
            output.append((dict(name=f'{condition}_s{seed}', label=f'Seed {seed} · {label}',
                seed=seed, condition=condition, kind='model',
                source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), x))
    # Controls are processed by exactly the same distance, landmark, and null rules.
    import torch
    from code.common import build_population
    torch.set_num_threads(1)
    for condition, name, label in [('gc100', 'control_grid_module', 'Analytic grid module'),
                                   ('pgc33', 'control_pgc_module', 'Analytic PGC + grid module')]:
        cfg_path = models / condition / f'{condition}_lstm_s1_1000M.json'
        cfg = json.loads(cfg_path.read_text())
        pop = build_population(cfg, 'cpu')
        with torch.no_grad():
            activity = pop.activity(torch.tensor(replay['pos'][selected]),
                dir=torch.tensor(replay['dir'][selected]), vel=torch.tensor(replay['vel'][selected])).numpy()
        x = activity[:, pop.module == 1].astype(np.float64)
        output.append((dict(name=name, label=label, seed=1, condition=condition,
            kind='control', expected='torus', module=1, spacing_m=float(pop.spacings[1]),
            note='128 cells of one analytic module on the identical sampled trajectory.'), x))
    rng = np.random.default_rng(20261031)
    theta = rng.uniform(0, 2*np.pi, (selected.sum(), 2))
    torus = np.column_stack([np.cos(theta[:, 0]), np.sin(theta[:, 0]),
                            np.cos(theta[:, 1]), np.sin(theta[:, 1])])
    output.append((dict(name='control_clifford_torus', label='Known synthetic torus', seed=None,
                       condition='reference', kind='control', expected='torus'), torus))
    output.append((dict(name='control_position_sheet', label='Physical position sheet', seed=None,
                       condition='reference', kind='control', expected='no torus'),
                   replay['pos'][selected].astype(np.float64)))
    for i, (description, _) in enumerate(output):
        description['dataset_id'] = i
    return output


def write_summary(results, opts):
    models = [r for r in results if r['kind'] == 'model']
    controls = [r for r in results if r['kind'] == 'control']
    compact = lambda r: dict(name=r['name'], label=r['label'], kind=r['kind'],
                            **r['metrics'], thresholds=r['thresholds'])
    summary = dict(protocol=opts, model_count=len(models),
        torus_screen_count=sum(r['metrics']['torus_screen'] for r in models),
        core_signature_count=sum(r['metrics']['core_torus_signature'] for r in models),
        sensitivity512_betti_interval_count=sum(r['sensitivity512']['torus_betti_interval'] is not None for r in models),
        results=[compact(r) for r in results],
        interpretation='Exploratory torus-compatible signature screen; not a torus proof, an absence proof, or an attractor test.')
    out = Path(opts['out'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Persistent homology of 1B LSTM activity', '',
        f'Analyzed all {len(models)} frozen LSTMs (8 matched seed pairs), using the same saved eastward activity as the PCA analysis. No UMAP, neuron rescaling, PCA projection, retraining, or new model rollouts were used.', '',
        f'**Result: {summary["core_signature_count"]} of {len(models)} LSTM activity clouds show the coexisting torus-compatible signature.** The stricter global-count screen also finds {summary["torus_screen_count"]}/{len(models)}. This result is conditional on the sampled states and analysis below; it does not establish that a model has no toroidal representation.', '',
        f'At 512 landmarks, {summary["sensitivity512_betti_interval_count"]}/{len(models)} LSTM clouds have a common interval for the dominant features with raw Betti counts (1,2,1). This is an observed-data sensitivity check, not a second null-calibrated test.', '',
        '[All-seed barcodes](all_seeds_barcodes.png) · [Signature summary](torus_signature.png) · [Controls](controls_barcodes.png)', '',
        '| Dataset | Significant H1 / H2 bars | p: second H1 | p: strongest H2 | Coexisting signature | Strict global count |',
        '|---|---:|---:|---:|---|---|']
    for r in results:
        m=r['metrics']
        lines.append(f'| {r["label"]} | {m["significant_h1"]} / {m["significant_h2"]} | {m["p_h1_second"]:.2f} | {m["p_h2_first"]:.2f} | {"Detected" if m["core_torus_signature"] else "Not detected"} | {"Pass" if m["torus_screen"] else "Does not pass"} |')
    lines += ['', '## What the test measures', '',
        'Each population vector is a point. A Vietoris–Rips complex connects points closer than an increasing distance threshold and fills all corresponding cliques with triangles, tetrahedra, and higher simplices. H0 counts connected components, H1 counts independent loops, and H2 counts two-dimensional cycles (closed surfaces). A barcode records each feature’s birth and death as the distance threshold changes. Persistence is a range of distance scales, not a duration in time.', '',
        'A torus has Betti numbers (1, 2, 1): one connected component, two independent loops, and one closed surface. We look for two strong H1 bars and one strong H2 bar existing together. These counts are compatible with a torus but do not uniquely characterize one; a positive result would need further validation such as circular-coordinate decoding.', '',
        '## Fixed analysis protocol', '',
        '- Data: the same 1,185 selected states, retaining every one of the 256 hidden-vector coordinates. Cell state c is not analyzed.',
        '- Metric: Euclidean distance, divided by the median pairwise distance of each cloud. This scalar normalization makes lifetimes relative to the cloud size; it does not change topology.',
        f'- Sampling: {opts["landmarks"]} farthest-point landmarks, always starting at row 0, selected afresh with the same rule for observed and shuffled clouds. This approximates the original sample; landmark indices and covering radii are saved.',
        '- Ripser computes H0, H1, and H2 over the full distance filtration with coefficient field Z47. The odd prime avoids the particular orientability ambiguity of characteristic 2, but does not make the torus signature unique.',
        f'- Null: {opts["shuffles"]} independent circular time shifts per neuron, applied to the ordered selected-sample traces. This preserves each neuron’s marginal values and cyclic serial structure while disturbing alignment between neurons. The selected series contains gaps between sweep lanes; the shifts preserve selected-index structure, not exact physical-time lag structure. A common shuffle of whole population vectors would leave topology unchanged and is not used.',
        '- The per-dimension reference is the 95th percentile (higher order statistic) of each shuffle’s longest bar. Every observed bar is compared to the same maximum-bar threshold for its dimension. These are exploratory per-dataset thresholds, without correction across 16 models.',
        '- Empirical p = (1 + number of null maxima at least as large as the observed lifetime) / (1 + number of shuffles). With 99 shuffles the smallest value is 0.01. These are null-tail statistics for bar lifetimes, not probabilities that a cloud is a torus.',
        '- Original strict global-count screen (`torus_screen`): exactly two H1 and one H2 bar exceed their null thresholds across the entire filtration, and the top two H1 and top H2 overlap at a distance interval with raw Betti counts (1, 2, 1). This flag is retained unchanged.',
        '- Coexisting signature (`core_torus_signature`): the second-longest H1 bar and longest H2 bar exceed their null thresholds, and the top two H1 bars and top H2 share an interval with raw Betti counts (1, 2, 1). Extra bars outside this interval are recorded rather than used as an automatic veto.',
        '- Transparent refinement: the known synthetic torus produced one additional above-threshold loop that died before its main surface appeared. The original global-count rule therefore rejected a known torus, despite a broad clean (1,2,1) interval. The scale-local signature was added after that control check and applied uniformly to every dataset; distances, sampling, nulls, thresholds, and the negative LSTM conclusions did not change. Both outcomes are reported.',
        f'- Sensitivity check: repeat the observed-cloud computation at {opts["sensitivity_landmarks"]} landmarks. Those results have no matching shuffled test and are not compared to the smaller-sample thresholds.', '',
        '## Controls and limits', '',
        'Positive controls are a known Clifford torus and the ordinary / mixed predictive grid populations from one module (128 cells, spacing 0.568 m, seed 1), evaluated at the identical positions and directions. The physical position sheet is a negative reference. Controls use the identical analysis and shuffle rules. Detecting their topology establishes basic sensitivity here, but cannot guarantee sensitivity to a noisy or partially sampled torus in the LSTM.', '',
        'The recordings use one fixed goal snapshot, one room, one facing direction, and a forced raster path. Only a rectangle away from the goal is sampled. The diagnostic sequence exceeds the training episode duration and has no resets. The full policy mixes multiple grid modules, visual information, and history; it is not necessarily one grid module or one two-dimensional torus. Negative findings apply only to this representation, sampling, metric, and scale.', '',
        '**A recurrent attractor is a different claim:** states must be dynamically drawn toward an invariant set. Persistent homology examines an activity cloud and discards time ordering; it does not measure attraction, stability, eigenvalues, or recurrent maintenance.', '',
        'The biological grid-cell study used persistent cohomology with six-PC and neighborhood-distance preprocessing, independent circular-shift controls, and additional circular-coordinate decoding. This analysis uses a simpler direct Euclidean pipeline and is not an exact replication. [Gardner et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8810387/).', '',
        '## Reproduce', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/lstm_topology.py',
        'python scripts/plot_lstm_topology.py', '```', '',
        'Individual JSON files retain observed and sensitivity diagrams, landmark indices, all null maxima and circular offsets, input hashes, thresholds, and metrics. `summary.json` contains the compact results.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/lstm_pca_1B')
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_topology_1B')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shuffles', type=int, default=99)
    parser.add_argument('--landmarks', type=int, default=256)
    parser.add_argument('--sensitivity-landmarks', type=int, default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    opts = dict(out=str(args.out), landmarks=args.landmarks,
                sensitivity_landmarks=args.sensitivity_landmarks, shuffles=args.shuffles,
                random_seed=20261031, coeff=47, ripser_version=ripser_package.__version__,
                distance='Euclidean / median full-cloud pairwise distance',
                reduction='none; all hidden coordinates retained',
                shuffle='independent circular shifts of selected neuron traces',
                threshold='95th percentile of per-shuffle longest bar, separately H1/H2',
                across_model_correction=False)
    (args.out / 'protocol.json').write_text(json.dumps(opts, indent=2) + '\n')
    data = datasets(args.source, args.models)
    jobs = [(description, x, opts) for description, x in data]
    results = []
    print(f'Computing {len(jobs)} datasets with {args.shuffles} nulls each; {args.workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compute_dataset, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r['dataset_id'])
    write_summary(results, opts)
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    main()
