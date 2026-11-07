"""Light independent count/claim review; full array arithmetic stays separately bound."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_geometry_resolution'
OUT = BASE / 'report_original'
KEYS = ('context', 'prefix', 'kind', 'branch', 'regime', 'window_updates', 'evolution_updates', 'observable', 'metric', 'centered')
STATUSES = ('sampling_unresolved', 'geometry_failed', 'physical_test_inconclusive', 'recovery_failed', 'screening_candidate')
SPACING = 'max_edge_below_10percent_radius'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def key(row):
    return tuple(row[k] for k in KEYS)


def main():
    protocol_path = ROOT / 'docs/research/native_turn_geometry_resolution_protocol.json'
    protocol = read(protocol_path)
    report, validation = read(OUT / 'summary.json'), read(OUT / 'validation.json')
    assert report['complete'] and validation['complete'] and report['endpoints'] == ['original']
    assert report['protocol_sha256'] == sha(protocol_path)
    assert validation['summary_sha256'] == sha(OUT / 'summary.json')
    assert report['source_sha256'] == sha(ROOT / 'scripts/report_native_turn_geometry_resolution.py')
    for name, digest in validation['audit_source_sha256'].items():
        assert sha(ROOT / name) == digest
    for name, entry in read(OUT / 'artifacts.json').items():
        path = OUT / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    models = {m['condition']: m for m in report['models']}
    assert len(models) == len(report['models']) == 2 and set(models) == {'gc100', 'pgc33'}
    reviews = []
    for condition in ('gc100', 'pgc33'):
        source = BASE / condition / 'original/summary.json'
        result, model = read(source), models[condition]
        array_audit = next(a for a in validation['models'] if a['condition'] == condition)
        assert result['complete'] and array_audit['complete'] and array_audit['summary_sha256'] == sha(source)
        frozen = protocol['original_cohorts'][condition]['cohort']
        assert result['cohort'] == frozen == array_audit['cohort']
        rows = {key(r): r for r in result['rows']}
        expected = {tuple(k) for k in frozen['source_keys']}
        assert set(rows) == expected and len(rows) == len(result['rows']) == frozen['selected_rows']
        statuses = Counter(); density_counts = Counter(); pairs = {}; physical = []; direction_failures = Counter()
        for row_key, row in rows.items():
            assert row['source_key'] == list(row_key)
            assert row['normal_test_run'] == row['resolved_geometry']
            if row['normal_test_run']:
                assert set(row['recovery']) == {'512', '1024'}
                for density, recovery in row['recovery'].items():
                    chosen = recovery['bounded_followups'][-1]['result'] if recovery['bounded_followups'] else recovery['nominal']
                    assert recovery['physical_status'] == chosen['physical_status']
                    assert recovery['passes'] == bool(chosen['physical_conclusive'] and chosen['all_normal_directions_pass'])
                    assert len(chosen['directions']) == 21 and chosen['directions'][-1]['name'] == 'tangent'
                    for direction in chosen['directions']:
                        g, relative = direction['normal_gain'], direction['normal_over_tangent_gain']
                        passed = bool(direction['informative_pair_fraction'] >= .8 and direction['factor_leakage_count'] == 0 and
                                      g is not None and relative is not None and g['median'] <= .5 and relative['median'] <= .5)
                        assert direction['passes'] == passed
                        if direction['name'] != 'tangent' and chosen['physical_conclusive'] and not passed:
                            direction_failures[direction['name']] += 1
                    assert chosen['all_normal_directions_pass'] == all(d['passes'] for d in chosen['directions'][:-1])
                    physical.append(dict(case=list(row_key), density=density, physical_conclusive=chosen['physical_conclusive'],
                                         status=chosen['physical_status'], cell_only_gain=next(d['normal_gain'] for d in chosen['directions'] if d['name'] == 'cell_only'),
                                         initially_visible_normals=all(d['normal_informative_fraction'] >= .8 for d in chosen['directions'][:-1])))
            else:
                assert 'recovery' not in row and not row['broad_candidate']
            if row['metric'] != 'raw':
                continue
            partner = dict(row, metric='h_c_RMS'); pair = (row, rows[key(partner)])
            if any(not r['convergence']['passes'] or any(not r['densities'][n]['gates'][SPACING] for n in ('512', '1024')) for r in pair):
                status = 'sampling_unresolved'
            elif any(not r['resolved_geometry'] for r in pair):
                status = 'geometry_failed'
            elif any(r['physical_status'] == 'inconclusive' for r in pair):
                status = 'physical_test_inconclusive'
            elif any(not r['broad_candidate'] for r in pair):
                status = 'recovery_failed'
            else:
                status = 'screening_candidate'
            statuses[status] += 1
            pkey = tuple(row[k] for k in KEYS if k != 'metric')
            pairs[pkey] = status
            for density in ('256', '512', '1024'):
                density_counts[density] += int(all(all(r['densities'][density]['gates'].values()) for r in pair))
        assert dict(model['statuses']) == {k: statuses[k] for k in STATUSES}
        assert model['necessary_geometry_by_density'] == dict(density_counts)
        assert model['selected_metric_pairs'] == len(pairs) == len(rows) // 2
        assert len(model['pairs']) == len(pairs)
        for pair in model['pairs']:
            assert pair['status'] == pairs[tuple(pair[k] for k in KEYS if k != 'metric')]
        assert model['exact_alias_pairs'] == sum(bool(r['source_exact_sham_of']) for r in rows.values() if r['metric'] == 'raw')
        assert array_audit['independently_audited_numerical_rows'] + array_audit['exact_alias_rows'] == len(rows)
        reviews.append(dict(condition=condition, source_summary_sha256=sha(source), selected_rows=len(rows), selected_pairs=len(pairs),
                            exact_alias_pairs=model['exact_alias_pairs'], statuses={k: statuses[k] for k in STATUSES},
                            geometry_pairs_by_density=dict(density_counts), physical_probes=physical,
                            conclusive_failed_direction_counts=dict(direction_failures)))
    assert report['source_rows'] == 66560 and report['selected_rows'] == 536 and report['selected_pairs'] == 268
    record = dict(complete=True, review_source_sha256=sha(__file__), summary_sha256=sha(OUT / 'summary.json'),
                  validation_sha256=sha(OUT / 'validation.json'), protocol_sha256=sha(protocol_path), models=reviews,
                  figures_visually_reviewed=True, no_model_environment_or_recurrent_execution=True,
                  scope='Independently verify exact prospectively frozen selected keys, metric-pair statuses, density plot counts, aliases and direction-level qualification arithmetic from saved JSON. Full input/state/polygon/perturbation-array reconstruction is supplied by the separately bound frozen saved-array validation; this lightweight review does not duplicate that heavy work.',
                  interpretation='Recovery gains refer to actor-output distance from the corresponding evolved curve after physical h/c kicks, not to hidden/cell state norm recovery. Sampling-first labels retain additional failed gates in the full records.',
                  report_artifact_sha256={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(OUT.iterdir())
                                         if p.is_file() and p.name not in ('artifacts.json', 'independent_review.json')})
    (OUT / 'independent_review.json').write_text(json.dumps(record, indent=2) + '\n')
    (OUT / 'artifacts.json').write_text(json.dumps({p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                                                  for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifacts.json'}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('models', 'report_artifact_sha256')}, indent=2))
    for model in reviews:
        print(model['condition'], model['statuses'], 'physical probes:', len(model['physical_probes']))


if __name__ == '__main__':
    main()
