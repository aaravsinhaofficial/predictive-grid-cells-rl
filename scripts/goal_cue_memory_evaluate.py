"""Goal-cue memory evaluation with lossless context/branch evidence.

No study bank, trained policy or environment runs on import. Production enters
through the strict saved-bank loader; evaluate_plan also supports tiny synthetic
fixtures. The complete prescribed denominator survives failed collections.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch

import goal_cue_memory as task
import goal_cue_memory_bank as banks
from local_return_feedback_evaluate import (
    global_rng_snapshot, load_raw, preserved_rng, save_raw, tensor_schema,
)
from rl_goal_memory import atomic_json, state_digest

BEHAVIOR = banks.BEHAVIOR
load_bank = banks.load_bank
context_plan = banks.context_plan
QUALIFICATION_VARIANTS = ('continuous',)
INTERIM_VARIANTS = ('brief_native', 'no_cue', 'brief_reset')
FULL_VARIANTS = (
    'brief_native', 'continuous', 'brief_ordinary', 'no_cue', 'brief_reset',
    'donor_plus90', 'donor_minus90', 'continuous_reset', 'wrong_plus90',
    'wrong_minus90', 'self_swap',
)
QUALIFICATION_SUCCESSES, MEMORY_SUCCESSES, RESET_COST = 192, 167, 39
DONOR_VARIANTS = ('donor_plus90', 'donor_minus90', 'self_swap')
WRONG_VARIANTS = ('wrong_plus90', 'wrong_minus90')
DEPENDENT_VARIANTS = DONOR_VARIANTS + WRONG_VARIANTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list): return [cpu_tree(item) for item in value]
    return value


def variant_spec(name):
    assert name in FULL_VARIANTS, f'Unknown prescribed variant: {name}'
    return dict(cue_mode='continuous' if name.startswith('continuous') else
                'no_cue' if name == 'no_cue' else 'brief',
                goal_kind='ordinary' if name == 'brief_ordinary' else 'native',
                erase_after_cue=name in ('brief_reset', 'continuous_reset'))


def ordered_variants(variants):
    names = tuple(variants)
    assert names and len(names) == len(set(names))
    assert set(names) <= set(FULL_VARIANTS)
    if set(names) & set(DEPENDENT_VARIANTS):
        assert 'brief_native' in names, 'Genuine donor controls require their same-context native source'
    # Collection order is fixed, independent of observed outcomes.
    return tuple(name for name in FULL_VARIANTS if name in names)


def donor_indices(name, count):
    assert name in DEPENDENT_VARIANTS and count >= 4 and count % 4 == 0
    shift = 0 if name == 'self_swap' else count // 4 * (1 if 'plus' in name else -1)
    return (torch.arange(count) + shift) % count


def first_free_transition_check(net, evidence, device):
    """Check exactly one saved incoming-state/input transition and its heads.

This is not a repeated full-history/CNN rollout. The latter has focused
synthetic tests; the production report states this distinction explicitly.
"""
    entries = [evidence['first_free'][i] for i in range(len(evidence['first_free']))]
    x = torch.stack([item['input'] for item in entries]).to(device)
    incoming = torch.stack([item['state_before_input'] for item in entries]).to(device)
    actual = torch.stack([item['state'] for item in entries]).to(device)
    logits = torch.stack([item['logits'] for item in entries]).to(device)
    values = torch.stack([item['value'] for item in entries]).to(device)
    h, c = net.lstm(x, (incoming[:, :net.nh], incoming[:, net.nh:]))
    state = torch.cat((h, c), 1)
    errors = dict(state_max_abs=float((state-actual).abs().max()),
                  logits_max_abs=float((net.pi(h)-logits).abs().max()),
                  value_max_abs=float((net.v(h).squeeze(-1)-values).abs().max()))
    assert all(value <= 3e-4 for value in errors.values()), errors
    return dict(**errors, tolerance=3e-4, full_history_replay_performed=False,
                scope='One first-FREE native LSTM transition and exact learned pi/v heads; input embedding is archived')


def physical_checks(audit, roll, evidence, expected_trials):
    active = torch.stack(roll['active']); free = torch.stack(roll['physics_active'])
    wait = active & ~free; action = torch.stack(roll['action']); pa = torch.stack(roll['pa'])
    terminals = torch.stack(roll['terminal'])
    initial, final = evidence['initial_physical_state'], evidence['final_physical_state']
    checks = dict(
        all_trials= audit['completed_trials'] == expected_trials,
        stationary_wait=bool(audit['task_owned_stationary_wait']),
        no_wait_actions=bool((action[wait] == -1).all() and (pa[wait] == -1).all()),
        no_wait_or_setup_reward=bool(not torch.stack(roll['raw_reward'])[~free].any()
            and audit['setup_learning_reward'] == 0 and audit['forced_rewards'] == 0),
        no_previous_reward=bool(not torch.stack(roll['pr']).any()),
        exact_policy_mask=torch.equal(free, torch.stack(roll['policy_mask'])),
        one_terminal_each=bool((terminals.sum(0) == 1).all()),
        zero_bootstrap=bool(not roll['bootstrap'].any()),
        initial_common_start=bool((initial['pos'] == 0).all() and (initial['hd'] == 0).all()
            and (initial['vf'] == 0).all() and (initial['vs'] == 0).all() and (initial['vr'] == 0).all()),
        clocks_count_physics=torch.equal(final['t'].cpu(), free.sum(0).cpu()),
        clean_image_and_state= audit['img_noise'] == 0 and audit['state_noise_sigma'] == 0,
        native_free_motor_noise=audit['free_motor_noise_v'] == .03 and audit['free_motor_noise_r'] == .15,
        native_rng_preserved=bool(audit['native_environment_rng_unchanged']),
    )
    assert all(checks.values()), checks
    return checks


def donor_behavior(raw, plan, indices):
    """Analysis-only virtual donor targets; physical reward targets stay native.

The fixed early horizon is 15 FREE updates. If the actual recipient hit ends a
trial earlier, its terminal position is held for that summary. Full-path donor
entry means entry before actual stopping, not an uncensored alternative return.
"""
    goal = plan['goal'].cpu(); donor = goal[indices]
    position = raw['position']; free = raw['physics_active']
    result = []
    for index in range(len(indices)):
        path = position[free[:, index], index]
        assert len(path) > 0
        donor_distance = (path-donor[index]).norm(dim=1)
        recipient_distance = (path-goal[index]).norm(dim=1)
        donor_hits = (donor_distance < .3).nonzero().flatten()
        recipient_hits = (recipient_distance < .3).nonzero().flatten()
        dh = int(donor_hits[0])+1 if len(donor_hits) else None
        rh = int(recipient_hits[0])+1 if len(recipient_hits) else None
        first = 'neither' if dh is None and rh is None else (
            'donor' if rh is None or (dh is not None and dh < rh) else
            'recipient' if dh is None or rh < dh else 'same_step')
        initial = plan['start'][index]
        early = path[min(14, len(path)-1)]
        result.append(dict(donor_index=int(indices[index]),
            donor_goal_world=donor[index].tolist(), recipient_goal_world=goal[index].tolist(),
            progress_to_donor_m=float((initial-donor[index]).norm()-donor_distance[-1]),
            progress_to_recipient_m=float((initial-goal[index]).norm()-recipient_distance[-1]),
            early_15_free_progress_to_donor_m=float((initial-donor[index]).norm()-(early-donor[index]).norm()),
            early_15_free_progress_to_recipient_m=float((initial-goal[index]).norm()-(early-goal[index]).norm()),
            early_horizon_free_updates=15, early_observed_free_updates=min(15, len(path)),
            early_terminal_position_held=len(path) < 15,
            first_donor_disk_entry_step=dh, first_recipient_disk_entry_step=rh,
            first_target_disk=first,
            full_path_recipient_hit_censored=rh is not None,
            full_path_censoring='Donor entry is observed only until the first actual recipient hit or 90 FREE updates; absence before a recipient hit does not establish absence on an extended path.',
            scope='Virtual donor disk is measured only; recipient goal/reward and first actual-hit stop are unchanged'))
    return result


def summarize(rows, n_contexts, angles):
    expected = {(c, a) for c in range(n_contexts) for a in range(angles)}
    assert len(rows) == n_contexts*angles and {(r['context'], r['angle_index']) for r in rows} == expected
    assert all(isinstance(row['hit'], bool) for row in rows)
    valid = [row for row in rows if not row.get('invalid', False)]
    assert all(not row['hit'] for row in rows if row.get('invalid', False))
    mean = lambda key: sum(row.get(key, 0.) for row in rows)/len(rows)
    result = dict(trials=len(rows), valid_trials=len(valid), invalid_trials=len(rows)-len(valid),
        successes=sum(row['hit'] for row in rows), success_fraction=sum(row['hit'] for row in rows)/len(rows),
        mean_path_m=mean('free_path_m'), mean_progress_m=mean('progress_m'),
        mean_free_entropy=mean('mean_free_entropy'), free_updates=sum(row.get('free_steps', 0) for row in rows),
        action_counts=[sum(row.get('action_counts', [0]*6)[a] for row in rows) for a in range(6)],
        descriptive_mean_convention='All prescribed trials remain in every denominator; invalid trials are failures and contribute zero to these descriptive means.')
    directional = [row['donor_behavior'] for row in valid if 'donor_behavior' in row]
    if directional:
        result['donor_behavior'] = dict(valid_directional_trials=len(directional), prescribed_trials=len(rows),
            mean_progress_to_donor_m=sum(r['progress_to_donor_m'] for r in directional)/len(rows),
            mean_progress_to_recipient_m=sum(r['progress_to_recipient_m'] for r in directional)/len(rows),
            mean_early_15_free_progress_to_donor_m=sum(r['early_15_free_progress_to_donor_m'] for r in directional)/len(rows),
            mean_early_15_free_progress_to_recipient_m=sum(r['early_15_free_progress_to_recipient_m'] for r in directional)/len(rows),
            early_terminal_position_held_trials=sum(r['early_terminal_position_held'] for r in directional),
            full_path_recipient_hit_censored_trials=sum(r['full_path_recipient_hit_censored'] for r in directional),
            early_horizon_convention='Position after 15 FREE updates; earlier terminal positions are held, not excluded.',
            full_path_censoring='Actual recipient hits end paths; donor-entry counts describe observation before that stop.',
            first_target_counts={key: sum(r['first_target_disk'] == key for r in directional)
                                 for key in ('donor', 'recipient', 'same_step', 'neither')})
    return result


def gates(variants):
    def eligible(name):
        return name in variants and variants[name]['totals']['trials'] == 256 and \
            variants[name]['totals']['valid_trials'] == 256 and variants[name]['all_physical_checks']
    control = eligible('continuous')
    memory = eligible('brief_native') and eligible('brief_reset')
    successes = variants['brief_native']['totals']['successes'] if 'brief_native' in variants else None
    cost = successes-variants['brief_reset']['totals']['successes'] if memory else None
    return dict(qualification=dict(eligible=control, success_threshold=QUALIFICATION_SUCCESSES,
            passed=control and variants['continuous']['totals']['successes'] >= QUALIFICATION_SUCCESSES),
        useful_memory=dict(eligible=memory, native_success_threshold=MEMORY_SUCCESSES,
            reset_cost_threshold=RESET_COST, native_successes=successes, reset_cost=cost,
            passed=memory and successes >= MEMORY_SUCCESSES and cost >= RESET_COST,
            inference='Descriptive nomination screen; goal-specific donor behavior is reported separately'),
        common_information_certificate_applies=False)


@torch.inference_mode()
def evaluate_plan(net, cfg, plan, reference, out, *, device='cpu', variants=FULL_VARIANTS):
    """Evaluate every requested context/branch without model selection or updates."""
    names = ordered_variants(variants); out = Path(out)
    out.mkdir(parents=True, exist_ok=True); assert not any(out.iterdir()), 'Use a fresh output directory'
    before, was_training = state_digest(net), net.training
    n_contexts, angles = plan['n_contexts'], plan['angles_per_context']
    if set(names) & set(DEPENDENT_VARIANTS): assert angles >= 4 and angles % 4 == 0
    all_rows = {name: [] for name in names}; contexts = []; started = time.monotonic()
    with preserved_rng():
        source = task.make_source(cfg, device); net.eval()
        try:
            for context in range(n_contexts):
                chunk = context_plan(plan, context); folder = out/f'context{context}'; folder.mkdir()
                plan_record = save_raw(folder/'input_plan.pt.gz', cpu_tree(chunk))
                base_reference = dict(reference, context_index=context, context_ids=chunk['context'].tolist(),
                    input_plan_sha256=plan_record['sha256'])
                results = {}; native = None
                for name in names:
                    branch = folder/name; branch.mkdir(); spec = variant_spec(name)
                    args = dict(spec); provenance = dict(spec, variant=name)
                    indices = None
                    if name in DEPENDENT_VARIANTS:
                        indices = donor_indices(name, angles)
                        provenance.update(donor_indices=indices.tolist(), source_variant='brief_native',
                            donor_angle_degrees=0 if name == 'self_swap' else 90 if 'plus' in name else -90,
                            recipient_physical_goal_unchanged=True)
                        if native is not None:
                            provenance.update(source_native_result_sha256=sha(folder/'brief_native'/'result.json'),
                                source_native_raw_sha256=results['brief_native']['raw']['sha256'])
                            if name in DONOR_VARIANTS:
                                args['first_free_state_override'] = torch.stack([
                                    native['evidence']['first_free'][int(i)]['state_before_input'] for i in indices])
                            else: args['goal_override'] = native['saved_goal'][indices]
                    input_reference = dict(base_reference, variant=provenance)
                    atomic_json(branch/'input.json', input_reference)
                    try:
                        assert name not in DEPENDENT_VARIANTS or native is not None, 'Native donor source failed; no replacement source selected'
                        roll, rows, audit, evidence = task.collect(net, cfg, source, chunk, device=device, **args)
                        checks = physical_checks(audit, roll, evidence, angles)
                        transition = first_free_transition_check(net, evidence, device)
                        raw = {key: torch.stack(value).cpu() for key, value in roll.items() if isinstance(value, list) and value}
                        raw.update({key: cpu_tree(value) for key, value in roll.items() if not isinstance(value, list)})
                        raw.update(evidence=cpu_tree(evidence), plan_reference=input_reference,
                            variant_provenance=provenance)
                        if indices is not None:
                            directional = donor_behavior(raw, chunk, indices)
                            for row, extra in zip(rows, directional): row['donor_behavior'] = extra
                        if name == 'self_swap':
                            assert native is not None
                            for key in ('position', 'action', 'reward', 'logits', 'values', 'goal_input'):
                                assert torch.equal(raw[key], native[key]), f'Self-swap sham differs: {key}'
                        record = save_raw(branch/'raw.pt.gz', raw)
                        result = dict(context=context, variant=name, valid=True, configuration=provenance,
                            rows=rows, audit=audit, physical_checks=checks, first_free_transition=transition, raw=record,
                            self_swap_exact=True if name == 'self_swap' else None)
                        if name == 'brief_native': native = raw
                    except Exception as error:
                        failed = save_raw(branch/'failed_plan.pt.gz', cpu_tree(chunk))
                        rows = [dict(context=context, angle_index=a, hit=False, invalid=True) for a in range(angles)]
                        result = dict(context=context, variant=name, valid=False, configuration=provenance, rows=rows,
                            failure_category='collection_or_validation', error_type=type(error).__name__,
                            error=str(error), failed_plan=failed)
                    all_rows[name].extend(rows); results[name] = result
                    atomic_json(branch/'result.json', result)
                contexts.append(dict(context=context, input_plan=plan_record, variants=results))
        finally: net.train(was_training)
    assert state_digest(net) == before, 'Evaluation changed model tensors'
    by_variant = {name: dict(rows=all_rows[name], totals=summarize(all_rows[name], n_contexts, angles),
        all_physical_checks=all(context['variants'][name]['valid'] and
            all(context['variants'][name]['physical_checks'].values()) for context in contexts)) for name in names}
    report = dict(version='native_goal_cue_memory_evaluation_v1', complete=True, condition=cfg['cond'],
        checkpoint_policy_hash=before, source_sha256=sha(__file__), collector_sha256=sha(task.__file__),
        bank_helper_sha256=sha(banks.__file__), reference=reference, variant_order=list(names),
        n_contexts=n_contexts, angles_per_context=angles, variants=by_variant, contexts=contexts,
        gates=gates(by_variant), all_variants_valid=all(v['totals']['invalid_trials'] == 0 for v in by_variant.values()),
        no_optimization=True, full_history_replay_performed=False,
        state_noise_sigma=0., img_noise=0., seconds=time.monotonic()-started,
        environment=dict(device=str(device), torch_version=str(torch.__version__)))
    atomic_json(out/'summary.json', report)
    return report


def evaluate_bank(net, cfg, out, *, bank_name='memory', manifest=BEHAVIOR,
                  device='cpu', variants=FULL_VARIANTS):
    plan, record, _ = load_bank(manifest, bank_name)
    reference = dict(manifest_path=str(Path(manifest).resolve().relative_to(ROOT)),
        manifest_sha256=sha(manifest), bank_name=bank_name,
        bank_path=record['path'], bank_sha256=record['sha256'])
    return evaluate_plan(net, cfg, plan, reference, out, device=device, variants=variants)
