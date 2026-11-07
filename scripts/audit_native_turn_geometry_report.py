"""Independently verify native-turn report counts and plotted raw full-state data.

Reads only archived JSON/NPZ. No Torch/model/environment import or execution.
The full producer audit remains separately bound; this checks every aggregate
and independently reconstructs all native primary full-state plot points.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_geometry'
KEYS = ('context', 'prefix', 'kind', 'branch', 'regime', 'window_updates',
        'evolution_updates', 'observable', 'metric', 'centered')
SPACING = 'max_edge_below_10percent_radius'
DEPARTURE = 'p90_invariance_residual_below_5percent'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def same(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b: same(a[k], b[k])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    elif isinstance(b, float): assert np.isclose(a, b, rtol=2e-8, atol=2e-10), (a, b)
    else: assert a == b, (a, b)


def primary(row):
    return row['window_updates'] == (2 if row['regime'] != 'left' and row['observable'] == 'full_state' else 3)


def expected_row_keys():
    keys = set()
    for context in range(16):
        for prefix in (0, 1):
            for kind in ('turn_only', 'probe_glimpse'):
                for branch in ('native', 'ordinary', 'pre_cue_reset_native', 'pre_cue_reset_ordinary'):
                    for regime in ('left', 'fb_B', 'fb_F'):
                        for observable in ('full_state', 'all_hidden', 'all_actor'):
                            full = observable == 'full_state'
                            for value in ((3, 8, 15, 30, 60) if regime == 'left' else (2, 14, 30, 60)):
                                window = value if regime == 'left' or full else value+1
                                evolution = window if full else window-1
                                for metric in ('raw', 'h_c_RMS'):
                                    for centered in ((False,) if full else (False, True)):
                                        keys.add((context, prefix, kind, branch, regime, window, evolution,
                                                  observable, metric, centered))
    assert len(keys) == 33280
    return keys


def check_cohort(summary, validation, failure, report_directory):
    endpoints = ['original'] if report_directory == 'report_original' else ['original', 'clean', 'noise']
    expected = {(c, e) for c in ('gc100', 'pgc33') for e in endpoints}
    actual = [(r['condition'], r['endpoint']) for r in validation['models']]
    assert summary['scope'] == endpoints and len(actual) == len(set(actual)) == len(expected)
    assert set(actual) == expected and summary['models'] == len(expected)
    assert summary['raw_rows'] == failure['raw_rows'] == 33280*len(expected)
    assert len(summary['primary']) == 9*len(expected) and len(failure['groups']) == 130*len(expected)
    assert len(validation['inputs']) == 2 and {r['condition'] for r in validation['inputs']} == {'gc100', 'pgc33'}


def counts(rows):
    sets = Counter(tuple(sorted(k for k, v in r['gates'].items() if not v)) for r in rows)
    return dict(cases=len(rows), pass_all=sum(not f for f in (tuple(k for k, v in r['gates'].items() if not v) for r in rows)),
        pass_except_spacing=sum(set(k for k, v in r['gates'].items() if not v) <= {SPACING} for r in rows),
        pass_except_spacing_and_departure=sum(set(k for k, v in r['gates'].items() if not v) <= {SPACING, DEPARTURE} for r in rows),
        individual_gate_failures=dict(Counter(k for r in rows for k, v in r['gates'].items() if not v)),
        exact_failure_sets=[dict(gates=list(k), count=v) for k, v in sorted(sets.items())])


def raw_curve_values(first, last):
    """Direct closest-point segment projection, centered solely for numerical stability."""
    r0 = float(np.sqrt(np.mean(np.sum((first - first.mean(0)) ** 2, axis=1))))
    r1 = float(np.sqrt(np.mean(np.sum((last - last.mean(0)) ** 2, axis=1))))
    origin = first.mean(0); a = first - origin; p = last - origin
    edge = np.roll(a, -1, axis=0) - a
    edge2 = np.sum(edge * edge, axis=1)
    numer = p @ edge.T - np.sum(a * edge, axis=1)
    frac = np.clip(numer / np.maximum(edge2, 1e-40), 0, 1)
    d2 = np.sum(p*p, axis=1)[:, None] + np.sum(a*a, axis=1)[None, :] - 2*p@a.T
    d2 = np.maximum(d2 - 2*frac*numer + frac*frac*edge2, 0)
    index = d2.argmin(axis=1)
    closest = a[index] + frac[np.arange(len(p)), index, None] * edge[index]
    distances = np.linalg.norm(p-closest, axis=1)
    return r1/max(r0, 1e-30), float(np.quantile(distances, .9)/max(r0, 1e-30))


def audit(report_directory, visual_review=False):
    out = BASE / report_directory
    summary, validation, failure = [read(out / n) for n in ('summary.json', 'validation.json', 'failure_patterns.json')]
    assert all(x['complete'] for x in (summary, validation, failure))
    assert validation['summary_sha256'] == sha(out/'summary.json')
    assert failure['validation_sha256'] == sha(out/'validation.json')
    assert summary['source_sha256'] == validation['source_sha256'] == sha(ROOT/'scripts/report_native_turn_geometry.py')
    assert failure['source_sha256'] == sha(ROOT/'scripts/report_native_turn_geometry_failures.py')
    assert summary['geometry_protocol_sha256'] == sha(ROOT/'docs/research/native_turn_geometry_protocol.json')
    check_cohort(summary, validation, failure, report_directory)
    expected_keys = expected_row_keys()
    manifest = read(out/'artifacts.json')
    for name, record in manifest.items():
        assert name == Path(name).name and not (out/name).is_symlink()
        assert (out/name).stat().st_size == record['bytes'] and sha(out/name) == record['sha256']
    groups = {(r['condition'], r['endpoint'], r['regime'], r['observable'], r['centered'], r['window_updates'], r['metric']): r for r in failure['groups']}
    assert len(groups) == len(failure['groups'])
    total = normals = necessary = artifacts = native_points = 0
    largest_error = dict(radius_retention=0., p90_departure=0.)
    model_records = []; plot_points = []
    for record in validation['models']:
        c, endpoint = record['condition'], record['endpoint']; folder = BASE/c/endpoint
        assert record['complete'] and sha(folder/'summary.json') == record['summary_sha256']
        data = read(folder/'summary.json'); rows = data['rows']
        assert data['complete'] and (data['condition'], data['endpoint']) == (c, endpoint)
        assert len(rows) == len({tuple(r[k] for k in KEYS) for r in rows}) == record['rows'] == record['unique_keys'] == 33280
        assert {tuple(r[k] for k in KEYS) for r in rows} == expected_keys
        total += len(rows); normals += sum(r['normal_test_run'] for r in rows)
        necessary += sum(all(r['gates'].values()) for r in rows)
        for name, entry in data['artifacts'].items():
            path = folder/name
            assert not path.is_symlink() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
            artifacts += 1
        assert sum(r['exact_sham_of'] is not None for r in rows) == record['exact_sham_rows']
        assert len(data['natural_history_comparison']) == record['independently_recomputed_natural_history_rows']
        lookup = {tuple(r[k] for k in KEYS): r for r in rows}
        grouped = defaultdict(list)
        for r in rows:
            assert r['necessary_geometry_pass'] == all(r['gates'].values()) == r['normal_test_run']
            other = dict(r, metric='h_c_RMS' if r['metric'] == 'raw' else 'raw')
            assert r['both_metrics_candidate'] == bool(r['broad_candidate'] and lookup[tuple(other[k] for k in KEYS)]['broad_candidate'])
            grouped[(r['regime'], r['observable'], r['centered'], r['window_updates'], r['metric'])].append(r)
        for key, selected in grouped.items():
            group = groups[(c, endpoint, *key)]
            native = [r for r in selected if r['branch'] == 'native']
            assert len(selected) == 256 and len(native) == 64
            same(group['all'], counts(selected)); same(group['native'], counts(native))
            for key2, values in (
                ('native_median_radius_retention', [r['radius_retention'] for r in native]),
                ('native_median_p90_departure', [r['residual_to_reference_curve_over_radius']['p90'] for r in native]),
                ('native_median_max_edge_over_radius', [r['geometry']['max_edge_over_radius'] for r in native])):
                same(group[key2], float(np.median(values)))
        for r in rows:
            if not (primary(r) and r['branch'] == 'native' and r['metric'] == 'raw' and r['observable'] == 'full_state'): continue
            assert not r['centered'] and r['observation_offsets'] == [0]
            path = folder/(r['exact_sham_of'] or r['family'])/r['regime']/'baseline_states.npz'
            with np.load(path, allow_pickle=False) as states:
                retention, departure = raw_curve_values(states['t0'], states[f"t{r['evolution_updates']}"])
            for k, a, b in (('radius_retention', retention, r['radius_retention']),
                            ('p90_departure', departure, r['residual_to_reference_curve_over_radius']['p90'])):
                largest_error[k] = max(largest_error[k], abs(a-b)); same(a, b)
            native_points += 1
            plot_points.append(dict(condition=c, endpoint=endpoint, context=r['context'], prefix=r['prefix'], kind=r['kind'],
                regime=r['regime'], radius_retention=retention, p90_departure=departure, baseline_sha256=sha(path)))
        for group in [g for g in summary['primary'] if (g['condition'], g['endpoint']) == (c, endpoint)]:
            selected = [r for r in rows if r['metric'] == 'raw' and primary(r) and r['regime'] == group['regime'] and r['observable'] == group['observable'] and r['centered'] == group['centered']]
            assert len(selected) == group['cases'] == 256
            assert sum(r['both_metrics_candidate'] for r in selected) == group['physical_candidates']
            assert sum(r['normal_test_run'] for r in selected) == group['normal_tests_raw']
        model_records.append(dict(condition=c, endpoint=endpoint, summary_sha256=sha(folder/'summary.json'), unique_rows=len(rows)))
    assert total == summary['raw_rows'] == failure['raw_rows']
    assert normals == failure['normal_test_rows'] and necessary == failure['necessary_geometry_pass_rows']
    assert native_points == len(validation['models']) * 192
    result = dict(complete=True, audit_source_sha256=sha(__file__), summary_sha256=sha(out/'summary.json'),
        validation_sha256=sha(out/'validation.json'), failure_patterns_sha256=sha(out/'failure_patterns.json'),
        raw_rows=total, models=model_records, artifact_hashes_checked=artifacts,
        independently_recomputed_native_plot_points=native_points, maximum_numeric_errors=largest_error,
        plot_points=plot_points, normal_test_rows=normals, necessary_geometry_pass_rows=necessary,
        figures_visually_reviewed=visual_review,
        report_artifact_sha256={n: r for n, r in manifest.items() if n != 'independent_validation.json'},
        no_model_or_environment_execution=True,
        scope='All report group counts and raw/RMS candidate conjunction; all native primary full-state radii and point-to-segment departures independently reconstructed from NPZ. Full saved-array gate/normal audit is separately bound by validation.json.')
    (out/'independent_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'artifacts.json').write_text(json.dumps({p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifacts.json'}, indent=2)+'\n')
    return {k: v for k, v in result.items() if k not in ('plot_points', 'report_artifact_sha256')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-directory', choices=('report_original', 'report_all'), default='report_original')
    p.add_argument('--visual-review-complete', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(audit(a.report_directory, a.visual_review_complete), indent=2))
