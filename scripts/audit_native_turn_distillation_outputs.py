"""Independent audit of all completed native-turn distillation data and reports.

Only saved numeric tensors/JSON are read. No policy/environment instantiation,
recurrent replay, optimization or new outcome. Refuses partial cohorts.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import audit_native_turn_original_memory_outputs as memory
import audit_native_turn_distillation_baselines as baseline

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/native_turn_distillation'
PARENT = ROOT / 'results/native_turn_teacher_preflight'
PROTOCOL = ROOT / 'docs/research/native_turn_distillation_protocol.json'
EXPECTED = baseline.EXPECTED
CONDITIONS = ('gc100', 'pgc33')
ARMS = ('clean', 'noise')
VARIANTS = memory.VARIANTS
KINDS = memory.KINDS
LIVE_VARIANTS = baseline.VARIANTS
sha, array, tensor_sha = memory.sha, memory.array, memory.tensor_sha
close, norm, kl = memory.close, memory.norm, memory.kl
population, audit_groups = memory.population, memory.audit_groups
ERRORS = memory.ERRORS


def read(path): return json.loads(Path(path).read_text())


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def verify_artifacts(folder, records):
    for name, entry in records.items():
        path = folder/name
        assert not path.is_symlink() and path.is_file() and path.stat().st_size == entry['bytes']
        assert sha(path) == entry['sha256'], path


def audit_paired(condition, endpoint, folder, expected_policy, old):
    summary_path = folder/'summary.json'; summary = read(summary_path)
    assert summary['complete'] and summary['condition'] == condition
    assert summary['policy_unchanged'] and summary['global_cpu_rng_preserved']
    assert summary['policy_hash'] == expected_policy
    assert len(summary['rows']) == 4096 and len(summary['pairs']) == 2048
    if endpoint == 'original':
        assert summary['protocol_sha256'] == memory.EXPECTED
    else:
        assert summary['original_memory_protocol_sha256'] == memory.EXPECTED and summary['own_prefix_recomputed']
    for name, record in summary['artifacts'].items():
        path = folder/name; assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
    bank_path = ROOT/old['banks'][condition]['path']; assert sha(bank_path) == old['banks'][condition]['sha256']
    bank = torch.load(bank_path, weights_only=True, map_location='cpu'); cfg = bank['config']
    assert len(bank['contexts']) == 16 and len({c['slot'] for c in bank['contexts']}) == 16
    prefix = json.loads((folder/'prefix_replay.json').read_text())
    assert prefix['complete'] and prefix['original_batch_shape'] == 32 and not prefix['cached_original_state_used_as_input']
    assert prefix['require_original_cache'] == (endpoint == 'original') and len(prefix['contexts']) == 16
    if endpoint == 'original': assert all(r['original_cache_max_error'] == 0 for r in prefix['contexts'])
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
            if endpoint == 'original': assert torch.equal(saved['own_prefix_state'], context['initial_hc'])
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
    values = {}
    for kind in KINDS:
        values[kind] = {}
        floor = summary['exact_input_groups'][kind]['case_weighted_minimum_forward_kl_nats']
        for variant in VARIANTS:
            rr = [r for r in rows if r['kind'] == kind and r['variant'] == variant]
            pp = [r for r in pairs if r['kind'] == kind and r['variant'] == variant]
            assert len(rr) == len({(r['context'], r['case']) for r in rr}) == 256
            assert len(pp) == len({(r['context'], r['pair']) for r in pp}) == 128
            values[kind][variant] = dict(cases=256, pairs=128,
                teacher_kl=float(np.mean([r['teacher_forward_kl_nats'] for r in rr])),
                progress_mm=1000*float(np.mean([r['expected_progress_m'] for r in rr])),
                common_ceiling_mm=1000*float(np.mean([r['best_common_action_ceiling_m'] for r in pp])),
                teacher_memoryless_kl_floor=floor,
                pair_total_variation=float(np.mean([r['student_pair_total_variation'] for r in pp])))
    return values, dict(summary_sha256=sha(summary_path), cases=4096, pairs=2048, context_mode_files=32,
        policy_hash=expected_policy, all_cue_reset_shams=True, raw_targets_and_physics_equal_original=True,
        identical_choice_inputs=True, exact_input_group_floors_recomputed=True,
        prefix_report_sha256=sha(folder/'prefix_replay.json'))


def audit_live(live, condition):
    assert live['complete'] and live['episodes_per_variant'] == 16 and live['seed'] == 984901
    assert tuple(live['variants']) == LIVE_VARIANTS and live['evaluation_state_noise'] == 0
    assert live['grid_native_ordinary_sham_exact'] == (condition == 'gc100')
    discovery = None; variants = {}
    for i, variant in enumerate(LIVE_VARIANTS):
        result = live['variants'][variant]; rows = result['rows']
        assert len(rows) == 16 and {r['slot'] for r in rows} == set(range(16))
        ordered = sorted(rows, key=lambda r: r['slot']); first = [r['first_goal_step'] for r in ordered]
        if discovery is None: discovery = first
        assert first == discovery
        found = [r for r in rows if r['first_goal_step'] >= 0]
        efficiencies = [r['path_eff'] for r in rows if r['path_eff_n'] > 0]
        post_time = sum(1350-r['first_goal_step'] for r in found)
        assert result['episodes'] == 16
        expected = dict(mean_score=statistics.mean(r['score'] for r in rows),
            mean_goals=statistics.mean(r['n_goals'] for r in rows), fraction_discovered=len(found)/16,
            fraction_at_least_two_goals=sum(r['n_goals'] >= 2 for r in rows)/16,
            mean_returns_per_episode=statistics.mean(max(0, r['n_goals']-1) for r in rows),
            post_discovery_returns_per_1000_steps=1000*sum(max(0, r['n_goals']-1) for r in found)/post_time if post_time else None,
            conditional_mean_path_efficiency=statistics.mean(efficiencies) if efficiencies else None)
        for k, v in expected.items(): baseline.close(result[k], v)
        assert all(r['score'] == 10*r['n_goals'] and r['path_eff_n'] == max(0, r['n_goals']-1) for r in rows)
        exposure = live['exposure'][variant]; phase = live['event_clock']['phase_update_counts'][i]
        assert exposure == dict(choice=phase[6], other_control=phase[0]+phase[1], forced=sum(phase[2:6]))
        assert sum(phase) == sum(exposure.values()) == 16*1350
        for kind, count in exposure.items():
            baseline.close(live['mean_teacher_kl'][variant][kind], live['teacher_kl_sums'][variant][kind]/count if count else None)
            assert math.isfinite(live['teacher_kl_sums'][variant][kind])
        assert live['choice_erasure_counts'][variant] == (exposure['choice'] if 'reset' in variant else 0)
        rewards = live['reward_by_phase'][variant]
        baseline.close(sum(rewards), sum(r['score'] for r in rows))
        clock = live['event_clock']; audit = live['input_audit']
        assert clock['coverage_violations'][i] == 0 and clock['interrupted_done'][i] == 0
        assert clock['admitted'][i] == clock['completed'][i]+clock['interrupted_reward'][i]+clock['interrupted_done'][i]
        assert 3*clock['completed'][i] <= exposure['choice'] <= 3*clock['admitted'][i]
        baseline.close(audit['scheduled_blind_updates'][i], sum(phase[2:]))
        baseline.close(audit['forced_goal_hits'][i], sum(rewards[2:6])/10)
        baseline.close(audit['free_goal_hits'][i], rewards[6]/10)
        assert clock['completed'][i] <= audit['pulses'][i] <= clock['admitted'][i]
        variants[variant] = dict(mean_score=result['mean_score'], exposure=exposure,
            choice_erasure_count=live['choice_erasure_counts'][variant], reward_total=sum(rewards),
            teacher_kl_sum=live['teacher_kl_sums'][variant], cue_pulses=audit['pulses'][i])
    if condition == 'gc100':
        for a, b in (('blind_native', 'blind_ordinary'), ('blind_reset_every_choice', 'blind_ordinary_reset_every_choice')):
            for key in ('variants', 'exposure', 'teacher_kl_sums', 'mean_teacher_kl', 'choice_erasure_counts', 'reward_by_phase'):
                assert live[key][a] == live[key][b]
            ia, ib = LIVE_VARIANTS.index(a), LIVE_VARIANTS.index(b)
            for category in ('event_clock', 'input_audit'):
                assert all(values[ia] == values[ib] for values in live[category].values())
    scores = {v: variants[v]['mean_score'] for v in LIVE_VARIANTS}
    seen = scores['seen_native']-scores['seen_reset_every_choice']
    native = scores['blind_native']-scores['blind_reset_every_choice']
    ordinary = scores['blind_ordinary']-scores['blind_ordinary_reset_every_choice']
    values = dict(scores=scores, visible_erasure_cost=seen, native_blind_erasure_cost=native,
        ordinary_blind_erasure_cost=ordinary, additional_native_blind_erasure_cost=native-seen,
        additional_ordinary_blind_erasure_cost=ordinary-seen, exposure=live['exposure'],
        choice_erasure_counts=live['choice_erasure_counts'], mean_teacher_kl=live['mean_teacher_kl'],
        variants=live['variants'], reward_by_phase=live['reward_by_phase'])
    return values, dict(episodes=96, branches=6, updates_per_branch=21600,
        first_discovery_steps_common_to_all_branches=discovery, exact_grid_cue_shams=condition == 'gc100',
        exposure=live['exposure'], choice_erasures=live['choice_erasure_counts'])


def audit_update(row, condition, arm, update):
    assert (row['condition'], row['arm'], row['update'], row['stage']) == (condition, arm, update, 1+(update-1)//60)
    assert row['extra_frames'] == update*16*90*4
    assert all(math.isfinite(row[k]) for k in ('loss', 'grad_norm_before_clip', 'parameter_update_l2'))
    assert row['replay_max_error'] == row['teacher_replay_max_error'] == 0
    assert row['choice_count']+row['control_count']+row['forced_count'] == 1440
    assert sum(row['phase_counts']) == 1440
    assert row['phase_counts'][6] == row['choice_count']
    assert sum(row['phase_counts'][2:6]) == row['forced_count']
    assert sum(row['phase_counts'][:2]) == row['control_count']
    target = .5*(row['choice_kl']+row['control_kl']) if row['choice_count'] else row['control_kl']
    assert math.isclose(row['loss'], target, rel_tol=2e-6, abs_tol=2e-6)
    assert row['no_choice_fallback'] == (row['choice_count'] == 0)
    assert row['counts']['noise_applied'] == (row['counts']['noise_scheduled'] if arm == 'noise' else 0)
    if arm == 'clean':
        assert row['clipped_hidden_coordinates'] == 0
        assert row['noise_realized_rms'] == row['noise_realized_mean'] == [0., 0.]


def audit_probe(folder, metrics, arm, sigma):
    probe = torch.load(folder/'first_choice_unroll.pt', weights_only=True, map_location='cpu')
    row = metrics[probe['update']-1]
    assert row['choice_count'] > 0 and all(r['choice_count'] == 0 for r in metrics[:probe['update']-1])
    assert tuple(probe['student_x'].shape) == tuple(probe['teacher_x'].shape) == (90, 16, 1287)
    assert torch.equal(probe['student_x'][:,:,768:], probe['teacher_x'][:,:,768:])
    assert torch.equal(probe['student_x'][~probe['blind']], probe['teacher_x'][~probe['blind']])
    choices, forced = array(probe['choice']), array(probe['forced'])
    assert not (choices & forced).any()
    assert np.array_equal(choices, array(probe['phase']) == 6)
    assert np.array_equal(forced, np.isin(array(probe['phase']), (2, 3, 4, 5)))
    for k in ('blind','glimpse','ordinary_cue','noise_scheduled','noise_applied','teacher_action','hit','wall_hit'):
        assert int(probe[k].sum()) == row['counts'][k]
    assert int(probe['h_clip'].sum()) == row['clipped_hidden_coordinates']
    assert np.array_equal(array(probe['noise_applied']), array(probe['noise_scheduled']) & (arm == 'noise'))
    glimpse = array(probe['glimpse']); phase = array(probe['phase'])
    post = np.zeros_like(glimpse)
    post[2:] = glimpse[:-2] & (phase[1:-1] == 5) & (phase[2:] == 5)
    assert np.array_equal(array(probe['noise_scheduled']), glimpse | post)
    assert np.max(np.abs(array(probe['state0'][0]))) <= 1
    realized = array(probe['realized_noise']); attempted = array(probe['attempted_noise']); applied = array(probe['noise_applied'])
    assert np.count_nonzero(realized[~applied]) == 0
    assert np.max(np.abs(attempted)/np.asarray(sigma)[None,None,:,None]) <= np.sqrt(3)+1e-6
    if applied.any():
        # An LSTM cell adds at most one per update; f is in [0,1]. Add the
        # worst possible declared impulse each update to bound float32
        # cancellation when the archived realized kick subtracts two cells.
        bound = float(probe['state0'][1].abs().max()) + 90*(1+sigma[1]*np.sqrt(3))
        cancellation_bound = 4*np.finfo(np.float32).eps*max(1., bound)
        close(realized[:,:,1][applied], attempted[:,:,1][applied], 'probe_cell_noise', cancellation_bound)
        assert np.max(np.abs(realized[:,:,0][applied])) <= sigma[0]*np.sqrt(3)+8*np.finfo(np.float32).eps
        for i in (0, 1):
            a = realized[:,:,i][applied].astype(float)
            close(row['noise_realized_rms'][i], np.sqrt(np.mean(a*a)), 'probe_noise_rms', 2e-7)
            close(row['noise_realized_mean'][i], np.mean(a), 'probe_noise_mean', 2e-7)
    def softmax(x):
        x = array(x).astype(float); x -= x.max(-1, keepdims=True); e = np.exp(x)
        return e/e.sum(-1, keepdims=True)
    losses = kl(softmax(probe['teacher_logits']), softmax(probe['student_logits']))
    control = ~(choices | forced)
    close(row['choice_kl'], losses[choices].mean(), 'probe_choice_kl', 2e-6)
    close(row['control_kl'], losses[control].mean(), 'probe_control_kl', 2e-6)
    close(row['loss'], .5*(losses[choices].mean()+losses[control].mean()), 'probe_weighted_kl', 2e-6)
    assert [int((probe['phase'] == i).sum()) for i in range(7)] == row['phase_counts']
    close(row['reward_by_phase'], [float(probe['reward'][probe['phase'] == i].sum()) for i in range(7)], 'probe_reward_phases')
    return dict(update=probe['update'], sha256=sha(folder/'first_choice_unroll.pt'),
        weighted_choice_control_KL_recomputed=True, preserved_goal_action_reward_inputs=True,
        private_noise_counts_and_cell_impulses_checked=True)


def audit_weights(folder, summary, original, cfg, arm):
    original_hash = baseline.policy_hash(original)
    checkpoints = {}
    for name, update in (('initial.pt', 0), ('stage1.pt', 60), ('stage2.pt', 120), ('stage3.pt', 180), ('final.pt', 180)):
        cp = torch.load(folder/name, weights_only=True, map_location='cpu')
        assert cp['updates'] == update and cp['extra_frames'] == update*16*90*4
        assert cp['config'] == cfg and cp['original_frames'] == 1_000_000_000
        assert cp['protocol_sha256'] == EXPECTED and cp['arm'] == arm
        assert cp['teacher_policy_hash'] == original_hash and cp['policy'].keys() == original.keys()
        for k, v in original.items():
            assert torch.isfinite(cp['policy'][k]).all()
            if not k.startswith('lstm.'): assert torch.equal(v, cp['policy'][k])
        if update == 0: assert equal(cp['policy'], original)
        checkpoints[name] = cp
    assert equal(checkpoints['stage3.pt'], checkpoints['final.pt'])
    final = checkpoints['final.pt']
    assert baseline.policy_hash(final['policy']) == summary['final_policy_hash']
    assert summary['initial_policy_hash'] == original_hash
    assert len(final['optimizer']['state']) == 4
    assert all(float(v['step']) == 180 for v in final['optimizer']['state'].values())
    deltas = {k: float((v.double()-final['policy'][k].double()).norm()) for k, v in original.items() if k.startswith('lstm.')}
    assert len(deltas) == 4 and all(v > 0 for v in deltas.values())
    return final, deltas


def ensure_ready():
    # Check the entire terminal cohort before touching outcome arrays.
    for c in CONDITIONS:
        for arm in ARMS:
            summary = read(BASE/c/arm/'summary.json')
            execution = read(BASE/f'{c}_{arm}_execution.json')
            assert summary['complete'] and summary['updates'] == 180
            assert summary['condition'] == execution['condition'] == c
            assert summary['arm'] == execution['arm'] == arm
            assert execution['complete'] and execution['exit_code'] == 0
            assert summary['protocol_sha256'] == execution['protocol_sha256'] == EXPECTED
    report = read(BASE/'report/summary.json')
    assert report['complete'] and report['protocol_sha256'] == EXPECTED
    return report


def main(visual_complete=False):
    torch.set_num_threads(1)
    report = ensure_ready()
    assert sha(PROTOCOL) == EXPECTED
    design = read(PROTOCOL); parent = read(ROOT/'docs/research/native_turn_teacher_preflight_protocol.json')
    for category in ('source_sha256', 'asset_sha256'):
        for name, expected in design[category].items(): assert sha(ROOT/name) == expected, name
    assert set(report['models']) == set(CONDITIONS) and report['no_new_rollout'] and report['no_optimization']
    assert report['producer_sha256'] == sha(ROOT/'scripts/report_native_turn_distillation.py')
    models = {}; terminal_hashes = {}; executions = {}; streams = {}
    for condition in CONDITIONS:
        original_path = ROOT/f'models/{condition}/{condition}_lstm_s1_1000M.pt'
        original = torch.load(original_path, weights_only=True, map_location='cpu')['policy']
        cfg = read(original_path.with_suffix('.json')); original_hash = baseline.policy_hash(original)
        values, paired = audit_paired(condition, 'original', memory.BASE/condition, original_hash, parent)
        assert values == report['models'][condition]['original']['paired']
        base_path = BASE/condition/'original_baseline'
        base = baseline.audit_model(condition)
        live_values, live_audit = audit_live(read(base_path/'live.json'), condition)
        assert live_values == report['models'][condition]['original']['live']
        models[condition] = {'original': dict(paired=paired, live=live_audit, baseline_bindings=base)}
        for arm in ARMS:
            key = f'{condition}/{arm}'; folder = BASE/condition/arm; summary = read(folder/'summary.json')
            assert summary['extra_frames'] == 1036800 and summary['policy_decisions'] == 259200
            assert summary['fixed_tensors_unchanged'] and summary['teacher_unchanged']
            assert summary['no_cloud'] and summary['cost_usd'] == 0
            assert summary['original_checkpoint_sha256'] == sha(original_path)
            assert summary['original_config_sha256'] == sha(original_path.with_suffix('.json'))
            assert summary['acquisition_cap_seconds'] == 1800 and 0 < summary['acquisition_monotonic_seconds'] <= 1800
            assert summary['evaluation_cap_seconds'] == 600 and 0 < summary['evaluation_monotonic_seconds'] <= 600
            verify_artifacts(folder, summary['artifacts'])
            final, deltas = audit_weights(folder, summary, original, cfg, arm)
            streams[key] = {k: final[k] for k in ('action_rng', 'mixture_rng', 'event_tapes')}
            metrics = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics) == 180
            for update, row in enumerate(metrics, 1): audit_update(row, condition, arm, update)
            assert sum(len(r['completed_episode_scores']) for r in metrics) == 192
            sigma = array(final['runtime']['sigma']).tolist()
            calibration = read(ROOT/'results/native_turn_distillation_design/original_delta_rms.json')
            expected_sigma = [.1*calibration['models'][condition]['blocks'][b]['one_update_delta_rms'] for b in ('h', 'c')]
            assert sigma == expected_sigma
            probe = audit_probe(folder, metrics, arm, sigma)
            values, paired = audit_paired(condition, arm, folder/'final_paired', summary['final_policy_hash'], parent)
            live_values, live_audit = audit_live(read(folder/'final_live.json'), condition)
            shown = report['models'][condition][arm]
            assert values == shown['paired'] and live_values == shown['live'] and metrics == shown['metrics']
            assert deltas == shown['changed_lstm_l2']
            calendar = (dt.datetime.fromisoformat(summary['end_utc'])-dt.datetime.fromisoformat(summary['start_utc'])).total_seconds()
            assert shown['calendar_seconds'] == calendar
            for k in ('acquisition_monotonic_seconds', 'evaluation_monotonic_seconds'):
                assert shown[k] == summary[k]
            terminal_hashes[key] = sha(folder/'summary.json')
            executions[key] = sha(BASE/f'{condition}_{arm}_execution.json')
            assert shown['summary_sha256'] == terminal_hashes[key]
            assert shown['final_checkpoint_sha256'] == sha(folder/'final.pt')
            assert shown['execution']['sha256'] == executions[key]
            assert report['terminal_summary_sha256'][str((folder/'summary.json').relative_to(ROOT))] == terminal_hashes[key]
            models[condition][arm] = dict(paired=paired, live=live_audit, first_choice_training_probe=probe,
                recurrent_tensor_changes_l2=deltas, updates=180, episode_completions=192,
                choice_exposures=sum(r['choice_count'] for r in metrics),
                control_exposures=sum(r['control_count'] for r in metrics),
                no_choice_batches=sum(r['no_choice_fallback'] for r in metrics),
                calendar_seconds=calendar, acquisition_monotonic_seconds=summary['acquisition_monotonic_seconds'],
                evaluation_monotonic_seconds=summary['evaluation_monotonic_seconds'])
            print(json.dumps(dict(audited=key, paired_cases=4096, live_episodes=96)), flush=True)
    first = next(iter(streams.values()))
    common = {k: all(equal(first[k], v[k]) for v in streams.values()) for k in first}
    assert common == report['saved_final_private_stream_equality']
    for name, expected in report['terminal_summary_sha256'].items(): assert sha(ROOT/name) == expected
    assert len(report['figures']) == 8
    for name, expected in report['figures'].items(): assert sha(BASE/'report'/name) == expected
    result = dict(complete=True, protocol_sha256=EXPECTED, producer_sha256=report['producer_sha256'],
        audit_source_sha256=sha(__file__), independent_helper_source_sha256={
            'scripts/audit_native_turn_original_memory_outputs.py': sha(memory.__file__),
            'scripts/audit_native_turn_distillation_baselines.py': sha(baseline.__file__)},
        report_summary_sha256=sha(BASE/'report/summary.json'), endpoint_summary_sha256=terminal_hashes,
        execution_sha256=executions, models=models, paired_cases=24576, paired_pairs=12288, live_episode_rows=576,
        maximum_numeric_errors=ERRORS, final_private_stream_equality=common,
        figures_visual_review=dict(complete=visual_complete),
        report_artifact_sha256={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted((BASE/'report').iterdir())
            if p.is_file() and p.name != 'independent_validation.json'},
        no_new_rollout=True, no_model_or_environment_execution=True, no_optimization=True,
        procedure=['Recompute every saved paired case and pair from raw probabilities, true six-action displacement, teacher targets and exact altered-input tapes.',
            'Recompute all exact-input-group information floors, cue rotations, reset arrays, population/turn-only shams, aggregate means and teacher mismatches.',
            'Independently recompute live episode statistics and reconcile phase/exposure/reward/erasure clocks for every branch; first discovery is matched within each endpoint.',
            'Check all stage/final weights, exact frozen tensors, optimizer step counts, loss accounting, private-noise masks and the archived first-choice training batch.',
            'Bind every producer/source/input/terminal/report/figure hash; no scientific rollout or recurrent replay.'],
        limitations=['Live policies have saved episode/clock/KL totals rather than full per-update logits; live KL is reconciled from saved sums, not independently replayed.',
            'Only the archived first-choice training batch retains all inputs and logits; other updates are checked from complete scalar accounting and frozen source.',
            'One original seed pair. Cross-population differences combine initial weights and teachers; no population-level significance or ring claim.'])
    out = BASE/'report/independent_validation.json'; out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, path=str(out), sha256=sha(out), paired_cases=24576, live_episode_rows=576)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-review-complete', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1): main(args.visual_review_complete)
