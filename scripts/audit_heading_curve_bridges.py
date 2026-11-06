"""Audit sparse full-state curve gaps and interpolation sensitivity.

This does not infer topology. It asks how much recovery measurements depend on
long straight segments inserted between sparsely sampled conditional states.
No frozen evaluator, checkpoint, or existing result is modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def projection_sensitivity(points, curve, long_segment):
    segment = np.roll(curve, -1, axis=0) - curve
    length2 = np.sum(segment**2, axis=1)
    delta = points[:, None] - curve[None]
    dot = np.sum(delta * segment[None], axis=-1)
    squared_point = np.sum(delta**2, axis=-1)
    fraction = np.clip(dot / np.maximum(length2, 1e-24), 0, 1)
    squared_segment = np.maximum(0, squared_point - 2 * fraction * dot + fraction**2 * length2)
    nearest = np.argmin(squared_segment, axis=1)
    all_distance = np.sqrt(np.min(squared_segment, axis=1))
    if (~long_segment).any():
        short_distance = np.sqrt(np.minimum(np.min(squared_segment[:, ~long_segment], axis=1), np.min(squared_point, axis=1)))
    else:
        short_distance = np.sqrt(np.min(squared_point, axis=1))
    return all_distance, short_distance, long_segment[nearest]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, default=Path('results/ring_heading/confirmation_evaluation'))
    p.add_argument('--out', type=Path, default=Path('results/ring_heading/geometry_audit/curve_bridge_sensitivity.json'))
    p.add_argument('--seeds', type=int, nargs='+', default=[2001])
    args = p.parse_args(); rows = []
    for seed in args.seeds:
        for condition in ('grid', 'predictive', 'grid_heading'):
            path = args.folder / f'memory_{condition}_s{seed}.json'
            if not path.exists(): continue
            report = json.loads(path.read_text()); archive = np.load(path.with_suffix('.npz'))
            times = archive['sample_times']
            for context, source in enumerate(report['models']['trained']):
                base = archive[f'trained_context{context}_base'].astype(np.float64)
                cue = archive[f'trained_context{context}_cue_end'].astype(np.float64)
                curves = [('cue_end', -1, cue)] + [(str(int(t)), i, base[i]) for i, t in enumerate(times) if t in (0, 100, 300, 600)]
                for label, index, curve in curves:
                    jump = np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1)
                    radius = np.sqrt(np.mean(np.sum((curve - curve.mean(0))**2, axis=1)))
                    median = np.median(jump); long = jump > 5 * median
                    row = {'seed': seed, 'condition': condition, 'context': context, 'position': source['position'],
                        'continuation_step': label, 'n_phase': len(curve), 'state_radius': float(radius),
                        'adjacent_jump_median': float(median), 'adjacent_jump_max': float(jump.max()),
                        'largest_jump_segment_start_index': int(np.argmax(jump)),
                        'largest_jump_divided_by_radius': float(jump.max()/max(radius,1e-12)),
                        'largest_to_median_jump_ratio': float(jump.max()/max(median,1e-12)),
                        'long_segment_fraction': float(long.mean()),
                        'arc_length_fraction_in_long_segments': float(jump[long].sum()/max(jump.sum(),1e-12)),
                        'long_segment_indices': np.flatnonzero(long).tolist()}
                    if label == '0':
                        spectral = source['spectra'][0]
                        if 'per_phase' in spectral:
                            align = np.array([x['slow_mode_tangent_alignment'] for x in spectral['per_phase']])
                            affected = long | np.roll(long, 1)
                            row['slow_alignment_min_q10_median'] = [float(x) for x in np.quantile(align,[0,.1,.5])]
                            row['slow_alignment_median_near_long_segments'] = float(np.median(align[affected])) if affected.any() else None
                            row['slow_alignment_min_near_long_segments'] = float(align[affected].min()) if affected.any() else None
                    if index >= 0:
                        row['normal_projection_sensitivity'] = {}
                        for size in (.1, .3):
                            key = f'trained_context{context}_normal_{size}'
                            if key not in archive: continue
                            points = archive[key][index].astype(np.float64)
                            original, restricted, assigned_long = projection_sensitivity(points, curve, long)
                            row['normal_projection_sensitivity'][str(size)] = {
                                'mean_distance_all_segments': float(original.mean()),
                                'mean_distance_short_segments_and_all_vertices': float(restricted.mean()),
                                'relative_increase': float((restricted.mean()-original.mean())/max(original.mean(),1e-12)),
                                'fraction_projected_to_long_segments': float(assigned_long.mean()),
                                'fraction_changed_more_than_1e_minus6_radius': float(np.mean(restricted-original>1e-6*max(radius,1e-12)))}
                    rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {'role': 'exploratory numerical sensitivity audit; no topology classification or seed exclusions',
        'long_segment_definition': 'length greater than five times within-curve median adjacent length',
        'restricted_projection': 'all measured vertices retained; only long interpolated segment interiors omitted',
        'seeds_requested': args.seeds, 'rows': rows}
    result['model_summaries'] = {}
    for condition in ('grid','predictive','grid_heading'):
        models=[]
        for seed in args.seeds:
            initial=[r for r in rows if r['condition']==condition and r['seed']==seed and r['continuation_step']=='0']
            if not initial: continue
            item={'seed':seed,'contexts':len(initial),
                'maximum_adjacent_jump_over_radius_across_contexts':max(r['largest_jump_divided_by_radius'] for r in initial),
                'maximum_long_segment_arc_fraction_across_contexts':max(r['arc_length_fraction_in_long_segments'] for r in initial),
                'minimum_slow_alignment_across_contexts':min(r['slow_alignment_min_q10_median'][0] for r in initial)}
            for time in ('100','300','600'):
                selected=[r for r in rows if r['condition']==condition and r['seed']==seed and r['continuation_step']==time]
                if not selected: continue
                for mode,key in [('all','mean_distance_all_segments'),('without_long_interiors','mean_distance_short_segments_and_all_vertices')]:
                    item[f'normal_10pct_distance_over_current_radius_step{time}_{mode}']=float(np.mean([
                        r['normal_projection_sensitivity']['0.1'][key]/max(r['state_radius'],1e-12) for r in selected]))
            models.append(item)
        result['model_summaries'][condition]={'models':models,'n_models':len(models),
            'seeds':[r['seed'] for r in models],
            'complete_all_16':len(models)==16 and [r['seed'] for r in models]==list(range(2001,2017))}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for condition in ('grid','predictive','grid_heading'):
        selected=[r for r in rows if r['condition']==condition and r['context']==0 and r['continuation_step']=='0']
        if selected:
            print(json.dumps({'condition':condition,'n_models':len(selected),
                'seeds':[r['seed'] for r in selected],
                'median_largest_jump_divided_by_radius':float(np.median([r['largest_jump_divided_by_radius'] for r in selected])),
                'maximum_largest_jump_divided_by_radius':max(r['largest_jump_divided_by_radius'] for r in selected)}))


if __name__=='__main__': main()
