"""Independent saved-array audit of the prefix-conditioned descriptive report."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_imitation_prefix'
KL = ('full_kl', 'prefix_average_kl', 'group_average_kl', 'total_history_gain',
      'prefix_associated_gain', 'within_prefix_gain')
PROGRESS = ('full_progress_mm', 'prefix_average_progress_mm', 'group_average_progress_mm',
            'total_progress_advantage_mm', 'prefix_associated_progress_mm', 'within_prefix_progress_mm')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(1)
    result = read(BASE / 'summary.json')
    assert result['complete'] and result['no_new_rollout'] and result['no_optimization']
    assert result['producer_sha256'] == sha(ROOT / 'scripts/decompose_native_turn_prefix.py')
    assert result['test_sha256'] == sha(ROOT / 'tests/test_decompose_native_turn_prefix.py')
    for name, entry in result['input_sha256'].items():
        path = ROOT / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    for name, entry in read(BASE / 'artifacts.json').items():
        path = BASE / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    prior = ROOT / 'results/native_turn_imitation_decomposition'
    assert result['main_summary_sha256'] == sha(prior / 'summary.json')
    prior_audit = read(prior / 'independent_validation.json')
    assert prior_audit['complete'] and prior_audit['summary_sha256'] == result['main_summary_sha256']
    assert len(result['histories']) == 64 and len(result['groups']) == 2304 and len(result['aggregates']) == 144
    histories = {}
    for history in result['histories']:
        key = tuple(history[k] for k in ('condition', 'context', 'kind'))
        assert key not in histories
        path = ROOT / history['path']
        assert sha(path) == history['sha256']
        raw = torch.load(path, map_location='cpu', weights_only=True)
        indices = [raw['labels'].index(f'turn_{i}') for i in range(14)]
        pa = raw['previous_actions'][indices].numpy()
        assert pa.shape == (14, 16) and np.array_equal(pa, np.repeat(pa[:1], 14, axis=0))
        labels = pa[0]
        assert set(labels) == {0, 1} and np.count_nonzero(labels == 0) == 8
        for field in ('teacher_x', 'student_native_x', 'student_ordinary_x'):
            assert np.array_equal(raw[field][indices, :, 1280:1286].numpy(), np.eye(6)[pa])
        assert history['turn_input_indices'] == indices
        assert history['prefix_action_by_case'] == labels.tolist()
        histories[key] = (labels, raw['first_update_physics']['progress_m'].numpy())
    expected_histories = {(c, i, k) for c in ('gc100', 'pgc33') for i in range(16)
                          for k in ('turn_only', 'probe_glimpse')}
    assert set(histories) == expected_histories
    cache, values, coverage = {}, defaultdict(list), defaultdict(set)
    max_error, zero_controls = 0., 0

    def compare(a, b):
        nonlocal max_error
        error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        max_error = max(max_error, error)
        assert error < 5e-12, error

    for group in result['groups']:
        c, e, scope, variant = (group[k] for k in ('condition', 'endpoint', 'scope', 'variant'))
        assert e in ('original', 'clean', 'noise')
        ps, qs, xs, labels, physics, refs = [], [], [], [], [], []
        for ref in group['cases_ref']:
            context, kind, i = (ref[k] for k in ('context', 'kind', 'case'))
            assert scope == 'all_modes' or scope == kind
            key = (c, e, context, kind)
            if key not in cache:
                folder = (ROOT / 'results/native_turn_original_memory' / c if e == 'original' else
                          ROOT / 'results/native_turn_distillation' / c / e / 'final_paired')
                cache[key] = torch.load(folder / f'context{context:02d}_{kind}.pt', map_location='cpu', weights_only=True)
            raw = cache[key]
            prefix, native_physics = histories[c, context, kind]
            assert ref['prefix'] == int(prefix[i])
            assert np.array_equal(raw['first_update_physics']['progress_m'].numpy(), native_physics)
            ps.append(raw['teacher_first_choice_probabilities'][i].numpy())
            qs.append(raw['variants'][variant]['first_choice_probabilities'][i].numpy())
            xs.append(raw['variants'][variant]['first_choice_input'][i].numpy())
            labels.append(prefix[i]); physics.append(native_physics[i]); refs.append((context, kind, i))
        assert len(refs) == len(set(refs)) == group['cases']
        assert all(np.array_equal(x, xs[0]) for x in xs)
        assert sha_bytes(xs[0]) == group['input_sha256']
        p, q = np.asarray(ps, dtype=float), np.asarray(qs, dtype=float)
        p /= p.sum(1, keepdims=True); q /= q.sum(1, keepdims=True)
        assert (q > 0).all()
        labels, physics = np.asarray(labels), np.asarray(physics)
        averaged = np.repeat(q.mean(0, keepdims=True), len(q), axis=0)
        within = np.empty_like(q)
        for label in (0, 1):
            within[labels == label] = q[labels == label].mean(0)
            compare(group['prefix_average_distributions'][str(label)], within[labels == label][0])
            assert group['prefix_counts'][str(label)] == int(np.count_nonzero(labels == label))
        compare(group['group_average_distribution'], averaged[0])
        computed = {}
        for name, policy in (('full', q), ('prefix_average', within), ('group_average', averaged)):
            terms = np.zeros_like(p); active = p > 0
            terms[active] = p[active] * np.log(p[active] / policy[active])
            kl = terms.sum(1); progress = 1000 * (policy * physics).sum(1)
            compare(kl, group['case_metrics'][name]['teacher_kl'])
            compare(progress, group['case_metrics'][name]['signed_progress_mm'])
            computed[name + '_kl'] = kl.mean()
            computed[name + '_progress_mm'] = progress.mean()
        computed.update(total_history_gain=computed['group_average_kl'] - computed['full_kl'],
                        prefix_associated_gain=computed['group_average_kl'] - computed['prefix_average_kl'],
                        within_prefix_gain=computed['prefix_average_kl'] - computed['full_kl'],
                        total_progress_advantage_mm=computed['full_progress_mm'] - computed['group_average_progress_mm'],
                        prefix_associated_progress_mm=computed['prefix_average_progress_mm'] - computed['group_average_progress_mm'],
                        within_prefix_progress_mm=computed['full_progress_mm'] - computed['prefix_average_progress_mm'])
        for field in KL + PROGRESS:
            compare(computed[field], group[field])
        if variant.startswith('choice_reset'):
            assert np.array_equal(q, np.repeat(q[:1], len(q), axis=0))
            for field in KL[3:] + PROGRESS[3:]:
                compare(computed[field], 0.)
            zero_controls += 1
        tag = c, e, scope, variant
        assert not coverage[tag].intersection(refs)
        coverage[tag].update(refs); values[tag].append((len(refs), computed))
    assert len(cache) == 192 and zero_controls == 576 and len(values) == 144
    aggregate_tags = set()
    for row in result['aggregates']:
        tag = tuple(row[k] for k in ('condition', 'endpoint', 'scope', 'variant'))
        assert tag not in aggregate_tags
        aggregate_tags.add(tag)
        expected = {(i, k, j) for i in range(16) for k in ('turn_only', 'probe_glimpse')
                    for j in range(16) if tag[2] == 'all_modes' or k == tag[2]}
        assert coverage[tag] == expected and row['cases'] == len(expected) and row['groups'] == len(values[tag])
        for field in KL + PROGRESS:
            compare(row[field], sum(n * x[field] for n, x in values[tag]) / len(expected))
    assert aggregate_tags == set(values)
    record = dict(complete=True, auditor_source_sha256=sha(__file__),
                  producer_sha256=result['producer_sha256'], summary_sha256=sha(BASE / 'summary.json'),
                  main_independent_audit_sha256=sha(prior / 'independent_validation.json'),
                  histories=64, groups=2304, aggregates=144, choice_reset_zero_gain_groups=576,
                  unique_raw_context_mode_files=192, maximum_absolute_error=max_error,
                  no_model_or_environment_execution=True, no_new_rollout=True,
                  procedure='Reconstruct actual PA prefix labels and all saved policy averages from raw arrays; verify every per-case KL and physical progress, every signed difference, input identity, complete case coverage and case-weighted aggregate.',
                  report_artifact_sha256={p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                                         for p in sorted(BASE.iterdir()) if p.is_file() and p.name not in ('artifacts.json', 'independent_validation.json')})
    (BASE / 'independent_validation.json').write_text(json.dumps(record, indent=2) + '\n')
    (BASE / 'artifacts.json').write_text(json.dumps({p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                                                   for p in sorted(BASE.iterdir()) if p.is_file() and p.name != 'artifacts.json'}, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'report_artifact_sha256'}, indent=2))


def sha_bytes(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


if __name__ == '__main__':
    main()
