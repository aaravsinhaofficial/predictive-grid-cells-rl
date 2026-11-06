"""Independent NumPy audit of saved memory interventions; no policy/env execution."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_original_memory'
PARENT = ROOT / 'results/native_turn_teacher_preflight'
PROTOCOL = ROOT / 'docs/research/native_turn_original_memory_protocol.json'
EXPECTED = '52e6b8c145a87dedf4889ebe864fc7376c8447e4e68ac4c4774234545b93a8fd'
VARIANTS = ('native', 'ordinary', 'cue_plus60', 'cue_minus60',
            'pre_cue_reset_native', 'pre_cue_reset_ordinary',
            'choice_reset_native', 'choice_reset_ordinary')
KINDS = ('turn_only', 'probe_glimpse')
ERRORS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def tensor_sha(value):
    return hashlib.sha256(np.ascontiguousarray(array(value)).tobytes()).hexdigest()


def close(actual, expected, key, atol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (key, a.shape, b.shape)
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    assert np.isfinite(a).all() and np.isfinite(b).all() and error <= atol, (key, error)
    ERRORS[key] = max(ERRORS.get(key, 0.), error)


def norm(p):
    p = np.asarray(p, dtype=np.float64)
    assert np.isfinite(p).all() and (p >= 0).all()
    return p / p.sum(-1, keepdims=True)


def kl(p, q):
    p, q = norm(p), norm(q)
    tiny = np.finfo(float).tiny
    return (p * (np.log(np.maximum(p, tiny))-np.log(np.maximum(q, tiny)))).sum(-1)


def population(cfg):
    """Independent analytic initialization, restricted to the frozen native design."""
    assert cfg['pgc_shift_mode'] == 'space' and cfg['pgc_shift_jitter'] == cfg['code_noise'] == 0
    assert cfg.get('pgc_layout', 'uniform') == 'uniform' and cfg.get('pgc_shift_rule', 'const') == 'const'
    assert cfg.get('cell_type', 'grid') == 'grid'
    n, m = cfg['n_cells'], cfg['n_modules']; rng = np.random.default_rng(cfg['code_seed'])
    module = np.arange(n) % m; theta = rng.uniform(0, np.pi/3, m)[module]
    spacing = cfg['spacing_min'] * cfg['spacing_ratio'] ** module
    angles = theta[:, None] + np.arange(3)[None, :]*np.pi/3
    wave = (4*np.pi/(np.sqrt(3)*spacing))[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), -1)
    a1 = spacing[:, None]*np.stack((np.cos(theta+np.pi/6), np.sin(theta+np.pi/6)), -1)
    a2 = spacing[:, None]*np.stack((np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)), -1)
    uv = rng.uniform(0, 1, (n, 2)); phase = uv[:, :1]*a1 + uv[:, 1:]*a2
    count = round(cfg['pgc_frac']*n); predictive = np.zeros(n, bool)
    for i in range(m):
        predictive[rng.choice(np.flatnonzero(module == i), count//m + (i < count % m), replace=False)] = True
    return wave.astype(np.float32), phase.astype(np.float32), predictive


def audit_groups(saved, inputs, targets, refs, predictions):
    groups = {}
    for i, x in enumerate(inputs):
        groups.setdefault(tensor_sha(x), []).append(i)
    assert saved['cases'] == len(inputs) and saved['groups'] == len(groups) == len(saved['records'])
    weighted = 0.
    for record in saved['records']:
        ix = groups[record['input_sha256']]; p = norm(targets[ix]); mixture = p.mean(0)
        assert record['cases'] == [refs[i] for i in ix] and record['count'] == len(ix)
        values = kl(p, np.broadcast_to(mixture, p.shape)); floor = values.mean()
        close(record['optimal_common_action_distribution'], mixture, 'group_mixture')
        close(record['per_case_kl_to_common_nats'], values, 'group_case_kl')
        close(record['generalized_js_nats'], floor, 'group_floor')
        close(record['minimum_mean_forward_kl_nats'], floor, 'group_minimum')
        assert set(record['variants']) == set(VARIANTS)
        for variant in VARIANTS:
            q = predictions[variant][ix]; actual = kl(p, q).mean()
            close(record['variants'][variant]['mean_teacher_forward_kl_nats'], actual, 'group_variant_kl')
            close(record['variants'][variant]['memoryless_floor_minus_observed_kl_nats'], floor-actual, 'group_improvement')
            if variant.startswith('choice_reset'):
                assert np.array_equal(q, np.broadcast_to(q[:1], q.shape)) and actual >= floor-1e-12
        weighted += len(ix)*floor
    close(saved['case_weighted_minimum_forward_kl_nats'], weighted/len(inputs), 'weighted_floor')


def audit_model(condition, design, old):
    folder = BASE/condition; summary_path = folder/'summary.json'; summary = json.loads(summary_path.read_text())
    assert summary['complete'] and summary['protocol_sha256'] == EXPECTED and not (folder/'incomplete.json').exists()
    assert all(summary[k] for k in ('source_unchanged', 'policy_unchanged', 'global_cpu_rng_preserved', 'no_optimization', 'no_cloud'))
    assert summary['qualification'] is None
    checkpoint = ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
    assert sha(checkpoint) == summary['checkpoint_sha256']
    for name, record in summary['artifacts'].items():
        path = folder/name; assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    bank_path = ROOT/old['banks'][condition]['path']; assert sha(bank_path) == old['banks'][condition]['sha256']
    bank = torch.load(bank_path, weights_only=True, map_location='cpu'); cfg = bank['config']
    assert len(bank['contexts']) == 16 and len({c['slot'] for c in bank['contexts']}) == 16
    prefix = json.loads((folder/'prefix_replay.json').read_text())
    assert prefix['complete'] and prefix['original_batch_shape'] == 32 and not prefix['cached_original_state_used_as_input']
    assert prefix['require_original_cache'] and len(prefix['contexts']) == 16
    assert all(r['original_cache_max_error'] == 0 for r in prefix['contexts'])
    wave, phase, predictive = population(cfg)
    outside = np.ones(1287, bool); outside[256+np.flatnonzero(predictive)] = False
    rows, pairs, refs, inputs, targets = [], [], [], [], []; predictions = {v: [] for v in VARIANTS}
    for context in bank['contexts']:
        c = context['context_id']
        for kind in KINDS:
            name = f'context{c:02d}_{kind}'
            saved = torch.load(folder/f'{name}.pt', weights_only=True, map_location='cpu')
            result = json.loads((folder/f'{name}.json').read_text())
            raw_path = PARENT/condition/f'{name}.pt'; ref_path = raw_path.with_suffix('.json')
            assert result['source_raw_sha256'] == sha(raw_path) and result['source_reference_sha256'] == sha(ref_path)
            source = torch.load(raw_path, weights_only=True, map_location='cpu'); reference = json.loads(ref_path.read_text())
            assert result['complete'] and result['context'] == c and result['kind'] == kind
            assert set(saved['variants']) == set(result['operations']) == set(VARIANTS)
            assert torch.equal(saved['own_prefix_state'], context['initial_hc'])
            assert tensor_sha(saved['own_prefix_state']) == next(r for r in prefix['contexts'] if r['context'] == c)['own_incoming_state_sha256']
            assert torch.equal(saved['teacher_first_choice_probabilities'], source['teacher_first_choice_probabilities'])
            for key in saved['first_update_physics']:
                assert torch.equal(saved['first_update_physics'][key], source['first_update_physics'][key])
            physics = saved['first_update_physics']; q = array(physics['progress_m']).astype(float)
            goal_vector = array(context['goal_center']).astype(float)-array(context['endpoint'])
            unit = goal_vector/np.linalg.norm(goal_vector)
            physical_q = ((array(physics['endpoint']).astype(float)-array(context['endpoint']))*unit).sum(-1)
            close(q, physical_q, 'physical_displacement', 1e-7)
            target = array(saved['teacher_first_choice_probabilities']); cue = reference['audit']['cue_index']
            for variant in VARIANTS:
                v = saved['variants'][variant]; op = result['operations'][variant]
                assert torch.equal(v['first_choice_input'], source['student_native_x'][-1])
                x = source['student_ordinary_x' if variant.endswith('ordinary') else 'student_native_x'].clone()
                reset = cue if cue is not None else source['labels'].index('suffix_0')
                expected_reset = reset if variant.startswith('pre_cue_reset') else len(x)-1 if variant.startswith('choice_reset') else None
                assert op['reset_before_input_index'] == expected_reset and op['cue_index'] == cue
                if cue is not None:
                    actual = array(v['actual_cue_input'])
                    assert np.array_equal(actual[:, outside], array(source['student_native_x'][cue])[:, outside])
                    if variant in ('cue_plus60', 'cue_minus60'):
                        direction = array(source['physical']['mdir'][cue]); angle = np.pi/3*(1 if variant == 'cue_plus60' else -1)
                        rotated = np.stack((np.cos(angle)*direction[:, 0]-np.sin(angle)*direction[:, 1],
                                            np.sin(angle)*direction[:, 0]+np.cos(angle)*direction[:, 1]), -1)
                        position = array(source['physical']['pos'][cue])[:, None, :] + (predictive*cfg['pgc_shift']).astype(np.float32)[None, :, None]*rotated[:, None, :]
                        projection = np.einsum('bnd,nid->bni', position-phase[None], wave)
                        code = np.clip((np.cos(projection).sum(-1)+1.5)/4.5, 0, 1)**cfg['sharpness']
                        close(actual[:, 256:768][:, predictive], code[:, predictive], 'rotated_cue_analytic', 2e-5)
                        x[cue] = v['actual_cue_input']
                    else:
                        assert torch.equal(v['actual_cue_input'], x[cue])
                assert tensor_sha(x) == op['altered_input_sha256']
                if expected_reset is not None:
                    assert torch.count_nonzero(v['reset_state_after']) == 0
                    if variant.startswith('choice_reset'):
                        assert torch.count_nonzero(v['first_choice_state_before_input']) == 0
                        assert torch.equal(v['reset_state_before'], v['first_choice_state_before_intervention'])
                logits = array(v['first_choice_logits']).astype(float); logits -= logits.max(1, keepdims=True)
                soft = np.exp(logits); soft /= soft.sum(1, keepdims=True)
                p = norm(array(v['first_choice_probabilities'])); close(soft, p, 'student_softmax', 2e-7)
                value = kl(target, p); progress = (p*q).sum(1)
                align = (p[:, :2]*array(physics['body_goal_angle_error_reduction_rad'])[:, :2]).sum(1)
                rr = [r for r in result['rows'] if r['variant'] == variant]
                pp = [r for r in result['pairs'] if r['variant'] == variant]
                assert len(rr) == 16 and len(pp) == 8
                for i, row in enumerate(rr):
                    assert (row['context'], row['kind'], row['case'], row['pair'], row['member']) == (c, kind, i, i//2, i%2)
                    close(row['teacher_forward_kl_nats'], value[i], 'case_kl')
                    close(row['expected_progress_m'], progress[i], 'case_progress')
                    close(row['turn_weighted_body_goal_alignment_rad'], align[i], 'case_alignment')
                    assert row['argmax_action'] == int(p[i].argmax())
                for i, row in enumerate(pp):
                    ix = slice(2*i, 2*i+2); ceiling = q[ix].mean(0).max()
                    assert (row['context'], row['kind'], row['pair']) == (c, kind, i)
                    close(row['teacher_forward_kl_nats'], value[ix].mean(), 'pair_kl')
                    close(row['expected_progress_m'], progress[ix].mean(), 'pair_progress')
                    close(row['best_common_action_ceiling_m'], ceiling, 'pair_ceiling')
                    close(row['progress_above_common_ceiling_m'], progress[ix].mean()-ceiling, 'pair_advantage')
                    close(row['student_pair_total_variation'], .5*np.abs(p[2*i]-p[2*i+1]).sum(), 'pair_tv')
                    tp = norm(target[ix]); close(row['teacher_pair_jensen_shannon_nats'], kl(tp, np.broadcast_to(tp.mean(0), tp.shape)).mean(), 'pair_teacher_js')
                    close(row['teacher_pair_total_variation'], .5*np.abs(tp[0]-tp[1]).sum(), 'pair_teacher_tv')
                predictions[variant].extend(array(v['first_choice_probabilities']))
            def equal(names):
                for key in ('first_choice_state_before_input', 'first_choice_state_after_input', 'first_choice_logits', 'first_choice_probabilities'):
                    assert all(torch.equal(saved['variants'][names[0]][key], saved['variants'][v][key]) for v in names[1:])
            equal(('choice_reset_native', 'choice_reset_ordinary'))
            if kind == 'turn_only' or condition == 'gc100':
                equal(('native', 'ordinary', 'cue_plus60', 'cue_minus60'))
                equal(('pre_cue_reset_native', 'pre_cue_reset_ordinary'))
            for variant in ('choice_reset_native', 'choice_reset_ordinary', 'pre_cue_reset_ordinary'):
                for key in ('first_choice_state_before_input', 'first_choice_state_after_input', 'first_choice_logits', 'first_choice_probabilities'):
                    v = saved['variants'][variant][key]; assert torch.equal(v, v[:1].expand_as(v))
            rows.extend(result['rows']); pairs.extend(result['pairs'])
            inputs.extend(array(source['student_native_x'][-1])); targets.extend(target)
            refs.extend(dict(context=c, kind=kind, case=i, pair=i//2, member=i%2) for i in range(16))
    assert rows == summary['rows'] and pairs == summary['pairs'] and len(rows) == 4096 and len(pairs) == 2048
    inputs, targets = np.asarray(inputs), np.asarray(targets); predictions = {v: np.asarray(p) for v, p in predictions.items()}
    for mode in ('all_modes', *KINDS):
        ix = [i for i, r in enumerate(refs) if mode == 'all_modes' or r['kind'] == mode]
        audit_groups(summary['exact_input_groups'][mode], inputs[ix], targets[ix], [refs[i] for i in ix], {v: p[ix] for v, p in predictions.items()})
    for kind in KINDS:
        for variant in VARIANTS:
            s = summary['summary'][kind][variant]; rr = [r for r in rows if r['kind'] == kind and r['variant'] == variant]
            pp = [r for r in pairs if r['kind'] == kind and r['variant'] == variant]
            assert s['cases'] == len(rr) == 256 and s['pairs'] == len(pp) == 128
            for key in ('teacher_forward_kl_nats', 'expected_progress_m', 'turn_weighted_body_goal_alignment_rad'):
                close(s[key], np.mean([r[key] for r in rr]), 'summary_'+key)
            for key in ('best_common_action_ceiling_m', 'progress_above_common_ceiling_m'):
                close(s[key], np.mean([r[key] for r in pp]), 'summary_'+key)
            close(s['memoryless_teacher_kl_floor_nats'], summary['exact_input_groups'][kind]['case_weighted_minimum_forward_kl_nats'], 'summary_floor')
    execution = json.loads((BASE/f'{condition}_execution.json').read_text()); assert execution['exit_code'] == 0
    return dict(complete=True, summary_sha256=sha(summary_path), artifact_count=len(summary['artifacts']),
                contexts=16, cases=4096, pairs=2048, exact_shams=True, ordinary_pre_cue_reset_all_phase_identity=True,
                summaries=summary['summary'])


def main():
    torch.set_num_threads(2); assert sha(PROTOCOL) == EXPECTED
    design = json.loads(PROTOCOL.read_text())
    for category in ('source_sha256', 'asset_sha256'):
        for name, expected in design[category].items(): assert sha(ROOT/name) == expected, name
    old = json.loads((ROOT/'docs/research/native_turn_teacher_preflight_protocol.json').read_text())
    assert design['parent_protocol_sha256'] == sha(ROOT/'docs/research/native_turn_teacher_preflight_protocol.json')
    models = {condition: audit_model(condition, design, old) for condition in ('gc100', 'pgc33')}
    report_path = BASE/'report/summary.json'; report = json.loads(report_path.read_text())
    assert report['complete'] and report['protocol_sha256'] == EXPECTED
    assert report['all_fixed_variants'] == list(VARIANTS)
    for condition, record in models.items():
        assert report['input_summary_sha256'][f'results/native_turn_original_memory/{condition}/summary.json'] == record['summary_sha256']
        assert report['summaries'][condition] == record.pop('summaries')
    assert report['producer_sha256'] == sha(ROOT/'scripts/report_native_turn_original_memory.py')
    for name, digest in report['figures'].items(): assert sha(BASE/'report'/name) == digest
    result = dict(complete=True, protocol_sha256=EXPECTED, audit_source_sha256=sha(__file__),
                  report_summary_sha256=sha(report_path), models=models,
                  frozen_sources_verified=len(design['source_sha256']), frozen_assets_verified=len(design['asset_sha256']),
                  max_absolute_errors=ERRORS,
                  procedure=['Only load saved numeric tensors/JSON with weights_only=True; no policy/environment execution.',
                             'Check every frozen source/asset and every raw outcome member against recorded SHA256/size.',
                             'Recompute six-action projected movement, teacher forward KL, paired common-action ceilings, TV/JS, exact-input-group mixture floors and all aggregate means in NumPy.',
                             'Reconstruct altered full input tapes using saved cue tensors; check declared hashes, cue-only PGC slots, independent analytic rotated grid code, all reset arrays and exact cue/reset shams.',
                             'Check original prefix-state bindings and exact ordinary pre-cue/choice-erasure identity across all sixteen phases.',
                             'Bind the unchanged report to both terminal summaries, reporter source and figure hashes.'],
                  limitations=['Saved-array audit does not replay recurrent computation; zero native-cache errors are checked against archived prefix reports and states.',
                               'Single original seed per population; descriptive controlled first choice, no significance or ring claim.'])
    output = BASE/'report/independent_validation.json'; output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, output=str(output), sha256=sha(output), cases=8192, pairs=4096)))


if __name__ == '__main__':
    main()
