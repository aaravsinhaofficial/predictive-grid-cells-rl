"""Independent saved-array audit; never instantiates a policy or environment."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_teacher_preflight'
PROTOCOL = ROOT / 'docs/research/native_turn_teacher_preflight_protocol.json'
EXPECTED_PROTOCOL = '19197d05383c006d574ed882fce48242220859c3bbd1ed9bf96544653df0f2f7'
ERRORS = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def arr(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(arr(value)).tobytes()).hexdigest()


def close(actual, expected, label, atol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    ERRORS[label] = max(ERRORS.get(label, 0.), error)
    assert np.isfinite(a).all() and np.isfinite(b).all() and error <= atol, (label, error)


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    y = np.exp(x - x.max(axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def information(p):
    p = np.asarray(p, dtype=np.float64); p = p / p.sum(1, keepdims=True)
    mean = p.mean(0); positive = p > 0
    terms = np.zeros_like(p)
    terms[positive] = (p * (np.log(np.maximum(p, np.finfo(float).tiny)) -
                           np.log(np.maximum(mean, np.finfo(float).tiny))))[positive]
    return p, mean, terms.sum(1)


def audit_groups(saved, inputs, targets, refs):
    groups = {}
    for x, p, reference in zip(inputs, targets, refs):
        digest = tensor_hash(x)
        group = groups.setdefault(digest, dict(x=x, probabilities=[], cases=[]))
        assert np.array_equal(group['x'], x)
        group['probabilities'].append(p); group['cases'].append(reference)
    assert saved['cases'] == len(inputs) and saved['groups'] == len(groups)
    assert len(saved['records']) == len(groups)
    weighted = 0.
    for record in saved['records']:
        group = groups[record['input_sha256']]
        assert record['cases'] == group['cases'] and record['count'] == len(group['cases'])
        _, mean, kl = information(group['probabilities'])
        close(record['optimal_common_action_distribution'], mean, 'group_mixture')
        close(record['per_case_kl_to_common_nats'], kl, 'group_case_kl')
        close(record['generalized_js_nats'], kl.mean(), 'group_js')
        close(record['minimum_mean_forward_kl_nats'], kl.mean(), 'group_minimum_kl')
        weighted += len(kl) * kl.mean()
    close(saved['case_weighted_minimum_forward_kl_nats'], weighted / len(inputs), 'weighted_group_js')


def audit_continuation(raw, saved, context, cfg):
    q = raw['continuation']; n = 16
    assert q['x'].shape == (30, n, 1287)
    assert torch.equal(q['x'][0], raw['teacher_x'][-1])
    assert torch.equal(q['logits'][0], raw['teacher_first_choice_logits'])
    assert torch.equal(q['state_after_input'][0], raw['teacher_states_after'][-1])
    assert torch.equal(q['uniform'], context['continuation_uniform'].repeat_interleave(2, 1))
    actions = (q['uniform'][:, :, None] > q['logits'].softmax(-1).cumsum(-1)).sum(-1).clamp_max(5)
    assert torch.equal(actions, q['action'])
    active = np.ones(n, dtype=bool); endpoint = arr(q['start_position']).copy()
    initial = endpoint.copy(); goal = arr(q['original_goal_center']); path = np.zeros(n)
    hit_time = np.full(n, -1); done_time = np.full(n, -1); active_count = np.zeros(n, dtype=int)
    before = initial.copy()
    for t in range(30):
        assert np.array_equal(active, arr(q['active_before'][t]))
        hit, done = arr(q['hit'][t]), arr(q['done'][t])
        assert not np.any((hit | done) & ~active)
        physical = arr(q['position_preteleport'][t])
        path += np.linalg.norm(physical.astype(float) - before, axis=1) * active
        endpoint[active] = physical[active]; active_count += active
        assert np.array_equal(endpoint, arr(q['scoring_position'][t]))
        distance = np.linalg.norm(physical.astype(float) - goal, axis=1)
        assert np.all(distance[hit] <= cfg['goal_radius'] + 1e-6)
        assert np.all(distance[active & ~hit] >= cfg['goal_radius'] - 1e-6)
        hit_time[hit] = t; done_time[done] = t
        if t:
            selected = torch.tensor(active)
            expected_action = torch.nn.functional.one_hot(q['action'][t-1], 6).float()
            assert torch.equal(q['x'][t, selected, 1280:1286], expected_action[selected])
            assert torch.equal(q['x'][t, selected, -1], q['reward'][t-1, selected])
            assert torch.equal(q['x'][t, selected, 768:1280], raw['goal'][selected])
        active &= ~(hit | done); before = physical
    progress = np.linalg.norm(initial.astype(float) - goal, axis=1) - np.linalg.norm(endpoint.astype(float) - goal, axis=1)
    assert len(saved['rows']) == n
    for i, row in enumerate(saved['rows']):
        assert (row['case'], row['pair'], row['member']) == (i, i // 2, i % 2)
        assert (row['first_hit_step'], row['first_done_step']) == (int(hit_time[i]), int(done_time[i]))
        assert row['active_updates'] == int(active_count[i]) and row['hit'] == bool(hit_time[i] >= 0)
        close(row['path_m'], path[i], 'continuation_path_m', 2e-6)
        close(row['goal_distance_progress_m'], progress[i], 'continuation_progress_m', 2e-6)
        close(row['endpoint_before_teleport'], endpoint[i], 'continuation_endpoint', 0.)
    return int((hit_time >= 0).sum())


def audit_model(condition, design):
    folder = BASE / condition; summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text()); assert summary['complete']
    assert not (folder / 'incomplete.json').exists()
    assert summary['protocol_sha256'] == EXPECTED_PROTOCOL
    assert summary['pair_counts'] == {'turn_only': 128, 'probe_glimpse': 128}
    assert summary['qualification'] is None and summary['no_optimization'] and summary['no_cloud']
    for name, expected in summary['artifacts'].items():
        path = folder / name
        assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes']
    bank_meta = design['banks'][condition]; bank_path = ROOT / bank_meta['path']
    assert sha(bank_path) == bank_meta['sha256']
    bank = torch.load(bank_path, weights_only=True, map_location='cpu'); contexts = bank['contexts']; cfg = bank['config']
    assert len(contexts) == 16 and len({c['slot'] for c in contexts}) == 16
    assert [c['context_id'] for c in contexts] == list(range(16)) and len(bank['inventory']) == 32
    prefixes = json.loads((folder / 'native_prefix_verification.json').read_text())
    assert prefixes['complete'] and prefixes['all_incoming_hc_exact'] and prefixes['original_batch_shape'] == 32
    assert len(prefixes['contexts']) == 16 and all(r['incoming_hc_max_error'] == 0 for r in prefixes['contexts'])
    source = torch.load(ROOT / f'results/native_movement_cycles/{condition}_s1/native_trace.pt', weights_only=True, map_location='cpu')
    for c in contexts:
        t, slot = c['step'], c['slot']
        assert torch.equal(c['initial_hc'], source['state_before'][t, slot])
        assert torch.equal(c['preceding_native_x'], source['x'][:t, slot])
        assert torch.equal(c['preceding_done'], source['done'][:t, slot])
        assert torch.equal(c['native_x'], source['x'][t, slot])
        assert cfg['episode_len'] - int(c['native_bookkeeping']['t']) >= 70
    del source
    inputs = []; targets = []; refs = []; rows = []; hit_total = 0; alias_classes = []
    expected_stems = {f'context{i:02d}_{kind}' for i in range(16) for kind in ('turn_only', 'probe_glimpse')}
    assert {p.stem for p in folder.glob('context*.pt')} == expected_stems
    assert {p.stem for p in folder.glob('context*.json')} == expected_stems
    for context in contexts:
        for kind in ('turn_only', 'probe_glimpse'):
            stem = folder / f'context{context["context_id"]:02d}_{kind}'
            result = json.loads(stem.with_suffix('.json').read_text()); assert result['complete']
            raw = torch.load(stem.with_suffix('.pt'), weights_only=True, map_location='cpu')
            sx, ox, tx = (raw[k] for k in ('student_native_x', 'student_ordinary_x', 'teacher_x'))
            length = 37 if kind == 'turn_only' else 41
            assert sx.shape == ox.shape == tx.shape == (length, 16, 1287)
            assert torch.equal(sx[-3:, ::2], sx[-3:, 1::2])
            assert torch.equal(sx[:, :, 768:], tx[:, :, 768:])
            assert torch.equal(sx[:15], tx[:15]) and not torch.equal(sx[15:, :, :256], tx[15:, :, :256])
            assert torch.equal(sx[:, :, 768:1280], raw['goal'][None].expand(length, -1, -1))
            assert torch.equal(raw['initial_hc'], context['initial_hc'].expand(16, -1))
            assert all(torch.count_nonzero(raw['physical'][k][-1]) == 0 for k in ('vf', 'vs', 'vr'))
            assert torch.equal(raw['physical']['pos'][-1], context['endpoint'].expand(16, -1))
            assert torch.equal(raw['physical']['t'], (int(context['native_bookkeeping']['t']) + torch.arange(length))[:, None].expand(-1, 16))
            angle = arr(raw['physical']['hd'][-1]); difference = np.angle(np.exp(1j * (angle[1::2] - angle[::2])))
            assert np.max(np.abs(np.abs(difference) - np.pi)) < 2e-5
            cue = result['audit']['cue_index']
            if cue is None or condition == 'gc100':
                assert torch.equal(sx, ox)
            else:
                assert length - 1 - cue == 8
                selector = torch.ones(length, dtype=torch.bool); selector[cue] = False
                assert torch.equal(sx[selector], ox[selector])
                assert torch.equal(sx[cue, :, :256], ox[cue, :, :256]) and torch.equal(sx[cue, :, 768:], ox[cue, :, 768:])
            for i, x in enumerate(sx[-1]):
                assert tensor_hash(x) == result['audit']['first_choice_input_hashes'][i]
            alias_classes.append(len({tensor_hash(x) for x in sx[-1]}))
            logits = arr(raw['teacher_first_choice_logits']); p = arr(raw['teacher_first_choice_probabilities'])
            close(probabilities(logits), p, 'teacher_softmax', 2e-7)
            q = arr(raw['first_update_physics']['progress_m']).astype(float)
            assert np.max(np.abs(q[:, :2])) == 0.
            goal = arr(context['goal_center']); start = arr(context['endpoint']); unit = (goal-start).astype(float); unit /= np.linalg.norm(unit)
            physical_q = ((arr(raw['first_update_physics']['endpoint']).astype(float) - start) * unit).sum(-1)
            close(physical_q, q, 'first_update_displacement', 1e-7)
            for i, row in enumerate(result['pairs']):
                pp, mean, kl = information(p[2*i:2*i+2]); qq = q[2*i:2*i+2]
                assert (row['context'], row['kind'], row['pair']) == (context['context_id'], kind, i)
                close(row['teacher_probabilities'], pp, 'pair_probabilities')
                close(row['teacher_jensen_shannon'], kl.mean(), 'pair_js')
                close(row['teacher_total_variation'], .5*np.abs(pp[0]-pp[1]).sum(), 'pair_tv')
                close(row['teacher_expected_progress_m'], (pp*qq).sum(1), 'pair_expected_progress')
                common = qq.mean(0).max(); oracle = qq.max(1).mean()
                close(row['best_common_action_ceiling_m'], common, 'common_action_ceiling')
                close(row['phase_specific_action_ceiling_m'], oracle, 'phase_action_ceiling')
                close(row['teacher_advantage_over_common_m'], (pp*qq).sum(1).mean()-common, 'teacher_advantage')
                rows.append(row)
            hit_total += audit_continuation(raw, result['teacher_continuation'], context, cfg)
            inputs.extend(arr(sx[-1])); targets.extend(p)
            refs.extend(dict(context=context['context_id'], kind=kind, case=i, pair=i//2, member=i%2) for i in range(16))
    assert rows == summary['pairs'] and len(rows) == 256
    audit_groups(summary['exact_input_group_information']['all_modes'], inputs, targets, refs)
    for kind in ('turn_only', 'probe_glimpse'):
        selected = [i for i, r in enumerate(refs) if r['kind'] == kind]
        audit_groups(summary['exact_input_group_information'][kind], [inputs[i] for i in selected],
                     [targets[i] for i in selected], [refs[i] for i in selected])
    return dict(complete=True, summary_sha256=sha(summary_path), contexts=16, context_mode_files=32,
        pairs=256, continuation_cases=512, continuation_first_hits=hit_total,
        within_context_choice_input_hash_classes=alias_classes, all_source_prefixes_bound=True,
        all_hashes_aliases_and_counts_verified=True, all_information_values_recomputed=True,
        all_first_hit_progress_and_paths_recomputed=True)


def main():
    torch.set_num_threads(2)
    assert sha(PROTOCOL) == EXPECTED_PROTOCOL
    design = json.loads(PROTOCOL.read_text())
    for group in ('source_sha256', 'asset_sha256'):
        for name, expected in design[group].items():
            assert sha(ROOT / name) == expected, name
    result = dict(complete=True, protocol_sha256=EXPECTED_PROTOCOL,
        auditor_source_sha256=sha(__file__), models={c: audit_model(c, design) for c in ('gc100', 'pgc33')},
        maximum_numeric_recomputation_errors=ERRORS, no_policy_or_environment_execution=True,
        no_source_or_model_changes=True)
    path = BASE / 'report/independent_validation.json'; path.parent.mkdir(exist_ok=True)
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, path=str(path.relative_to(ROOT)), models=result['models']), indent=2))


if __name__ == '__main__':
    main()
