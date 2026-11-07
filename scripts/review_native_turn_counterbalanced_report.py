"""Review the descriptive report against every saved phase array, without models."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_counterbalanced_readout'
OUT = BASE / 'report'
CONDITIONS = ('gc100', 'pgc33')
ENDPOINTS = ('original', 'clean', 'noise')
KINDS = ('turn_only', 'probe_glimpse')
BRANCHES = ('native', 'ordinary', 'pre_cue_reset_native', 'pre_cue_reset_ordinary')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    result = read(OUT / 'summary.json')
    audit = read(OUT / 'independent_validation.json')
    initial = ROOT / 'results/native_turn_counterbalanced_readout_design/initial_render'
    previous = read(initial / 'summary.json')
    source_review = read(ROOT / 'docs/research/native_turn_counterbalanced_report_review.json')
    assert result['report_complete'] and result['scientific_run_complete'] is False
    assert result['old16_input_equivalence'] == 'rejected' and result['failed_equivalence_comparisons'] == 12
    assert audit['arithmetic_audit_complete'] and audit['scientific_run_complete'] is False
    assert result['audit_sha256'] == sha(OUT / 'independent_validation.json')
    assert result['scientific_summary_sha256'] == sha(BASE / 'summary.json')
    assert result['producer_sha256'] == sha(ROOT / 'scripts/report_native_turn_counterbalanced_readout.py')
    assert result['auditor_sha256'] == audit['source_sha256'] == sha(ROOT / 'scripts/audit_native_turn_counterbalanced_readout.py')
    assert previous['producer_sha256'] == source_review['source_sha256']['scripts/report_native_turn_counterbalanced_readout.py']
    assert previous['producer_sha256'] == sha(initial.parent / 'initial_render_reporter.py')
    assert previous['contexts'] == result['contexts'] and previous['aggregates'] == result['aggregates']
    for name in ('all_contexts.csv', 'all_conditions.csv', 'independent_validation.json'):
        assert (OUT / name).read_bytes() == (initial / name).read_bytes()
    for name, entry in read(OUT / 'artifacts.json').items():
        path = OUT / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    expected = {(c, e, i, k, b) for c in CONDITIONS for e in ENDPOINTS for i in range(16) for k in KINDS for b in BRANCHES}
    rows = {tuple(r[k] for k in ('condition', 'endpoint', 'context', 'kind', 'branch')): r for r in result['contexts']}
    assert len(rows) == len(result['contexts']) == 768 and set(rows) == expected
    shared = {tuple(r[k] for k in ('condition', 'endpoint', 'context', 'prefix', 'kind', 'branch')): r for r in audit['shared_case_precision']}
    assert len(shared) == 1536
    paired, groups = {}, defaultdict(list)
    maximum_error = 0.
    arrays = 0

    def compare(actual, expected_value):
        nonlocal maximum_error
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected_value)), initial=0))
        maximum_error = max(maximum_error, error)
        assert error < 2e-12, error

    for tag, row in rows.items():
        c, e, i, kind, branch = tag
        if (c, e) not in paired:
            folder = (ROOT / 'results/native_turn_original_memory' / c if e == 'original' else
                      ROOT / 'results/native_turn_distillation' / c / e / 'final_paired')
            assert result['original_paired_summary_sha256'][str((folder / 'summary.json').relative_to(ROOT))] == sha(folder / 'summary.json')
            paired[c, e] = read(folder / 'summary.json')['rows']
        values = []
        for prefix in (0, 1):
            path = BASE / c / e / f'context{i:02d}_prefix{prefix}_{kind}_{branch}.npz'
            bound = audit['verified_artifacts'][str(path.relative_to(ROOT))]
            assert path.stat().st_size == bound['bytes'] and sha(path) == bound['sha256']
            with np.load(path, allow_pickle=False) as z:
                q = z['probabilities']; physical = z['action_progress_m'].astype(float)
                assert q.shape == physical.shape == (128, 6)
                compare(q.sum(1), np.ones(128))
                progress = 1000 * (q * physical).sum(1)
                compare(progress, z['case_progress_mm'])
                values.append(progress)
            arrays += 1
        left, right = values
        old = [r for r in paired[c, e] if (r['context'], r['kind'], r['variant']) == (i, kind, branch)]
        assert len(old) == 16 and len({r['case'] for r in old}) == 16
        measured = dict(left_full_circle_mm=left.mean(), right_full_circle_mm=right.mean(),
                        balanced_mm=(left.mean() + right.mean()) / 2,
                        old_assignment_dense_mm=np.concatenate((left[:64], right[64:])).mean(),
                        original_sparse_mm=1000 * np.mean([r['expected_progress_m'] for r in old]))
        measured['assignment_minus_balanced_mm'] = measured['old_assignment_dense_mm'] - measured['balanced_mm']
        measured['dense_minus_sparse_mm'] = measured['old_assignment_dense_mm'] - measured['original_sparse_mm']
        comparison = [shared[c, e, i, p, kind, branch] for p in (0, 1)]
        measured.update(max_shared_score_error_mm=max(r['max_abs_shared_score_error_mm'] for r in comparison),
                        max_shared_component_bound_mm=max(r['max_component_bound_mm'] for r in comparison),
                        rejected_old_input_comparisons=sum(r['old_input_equivalence_rejected'] for r in comparison))
        for field, value in measured.items():
            compare(row[field], value)
        groups[c, e, kind, branch].append(measured)
    assert arrays == 1536 and len(groups) == 48 and len(result['aggregates']) == 48
    seen = set()
    for row in result['aggregates']:
        tag = tuple(row[k] for k in ('condition', 'endpoint', 'kind', 'branch'))
        assert tag not in seen and row['contexts'] == 16
        seen.add(tag); local = groups[tag]
        assert len(local) == 16
        for field in local[0]:
            values = [r[field] for r in local]
            value = max(values) if field.startswith('max_') else sum(values) if field == 'rejected_old_input_comparisons' else np.mean(values)
            compare(row[field], value)
    assert seen == set(groups)
    presentation = read(OUT / 'presentation_validation.json')
    assert presentation['summary_sha256'] == sha(OUT / 'summary.json')
    for name, digest in presentation['figure_sha256'].items():
        assert sha(OUT / name) == digest
    for key, count in audit['downstream_failure_counts'].items():
        failed = [r for r in shared.values() if not r['downstream_checks'][key]]
        assert len(failed) == count
    assert audit['downstream_failure_counts']['cell'] == 6
    assert all(r['branch'] in ('native', 'ordinary') for r in audit['downstream_cell_failure_rows'])
    record = dict(complete=True, review_source_sha256=sha(__file__), report_summary_sha256=sha(OUT / 'summary.json'),
                  arithmetic_audit_sha256=sha(OUT / 'independent_validation.json'),
                  current_reporter_sha256=result['producer_sha256'], original_source_review_sha256=sha(ROOT / 'docs/research/native_turn_counterbalanced_report_review.json'),
                  presentation_revision_sha256=sha(OUT / 'presentation_validation.json'),
                  every_saved_phase_array_checked=arrays, phase_rows=arrays * 128, context_rows=768, aggregates=48,
                  maximum_numeric_discrepancy=maximum_error, figures_visually_reviewed=3,
                  initial_and_final_numerical_tables_and_audit_identical=True,
                  scientific_run_complete=False, old16_input_equivalence='rejected', failed_equivalence_comparisons=12,
                  downstream_cell_failures_retained=6, no_model_or_environment_execution=True,
                  scope='Independently reconstruct every plotted/context mean from saved probabilities and physical outcomes. Review the cached-render-only source change, all figures and explicit failed-run/precision claims. This does not promote the scientific equivalence gate or reexecute its models.',
                  report_artifact_sha256={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(OUT.iterdir())
                                         if p.is_file() and p.name not in ('artifacts.json', 'final_review.json')})
    (OUT / 'final_review.json').write_text(json.dumps(record, indent=2) + '\n')
    (OUT / 'artifacts.json').write_text(json.dumps({p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                                                  for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifacts.json'}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'report_artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
