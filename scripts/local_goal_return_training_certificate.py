"""Sealed-bank adapter to the unchanged opposite-heading path certificate.

Consumes actual first-FREE snapshots from the state-erased branch. It performs
no policy, recurrent, environment, optimizer or trial-selection operation.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
import local_goal_return_training as task
import certify_local_goal_return as inherited

BEHAVIOR = ROOT / 'results/local_goal_return_design/behavior_protocol.json'
BEHAVIOR_HASH = '4c7ba90dd79c8c0b5f8ac13ba89e4a7d6ed5bf0287556982ade07f911fc16e48'
HELPER_HASH = '83b6ccef847fb8296319f08de1649d3b8613b43e2e70b1e11ea767352809d53c'
RESET_VARIANT = 'blind_reset_after_glimpse'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def design():
    assert sha(BEHAVIOR) == BEHAVIOR_HASH, 'Behavior bank/gates changed'
    d = json.loads(BEHAVIOR.read_text())
    assert sha(inherited.__file__) == HELPER_HASH == d['certificate']['source_sha256']
    assert hashlib.sha256(inspect.getsource(task.TrialStreams).encode()).hexdigest() == d['generator_class_sha256']
    assert hashlib.sha256(inspect.getsource(task.Stage).encode()).hexdigest() == d['stage_class_sha256']
    assert d['physical_defaults'] == dict(tau=.25, noise_v=.03, noise_r=.15, wall_margin=.08)
    return d


@lru_cache(maxsize=2)
def _bank(name, expected_hash):
    d = design(); item = d['banks'][name]; assert item['sha256'] == expected_hash
    path = ROOT / item['path']
    assert path.stat().st_size == item['bytes'] and sha(path) == expected_hash
    value = torch.load(path, map_location='cpu', weights_only=True)
    for key, spec in item['tensors'].items():
        assert list(value[key].shape) == spec['shape'] and str(value[key].dtype) == spec['dtype']
        assert task.tensor_hash(value[key]) == spec['sha256'], key
    return value


def _validate_plan(plan, cfg):
    d = design()
    names = [name for name, item in d['banks'].items() if plan['stage'] == item['stage']]
    assert len(names) == 1, 'A certificate requires an exact sealed-bank stage'
    name = names[0]; item = d['banks'][name]
    for key, value in d['native_physics'].items(): assert cfg[key] == value, key
    assert cfg['n_cells'] == 512 and cfg['extra_input'] == 'none' and cfg['code_noise'] == 0
    assert plan['n_contexts'] == 1 and plan['headings'] == 16
    ids = plan['context'].detach().cpu()
    assert ids.shape == (16,) and bool((ids == ids[0]).all())
    context = int(ids[0]); assert 0 <= context < item['contexts']
    bank = _bank(name, item['sha256']); assert set(plan) == set(bank)
    section = slice(context*16, (context+1)*16)
    for key, value in plan.items():
        if isinstance(value, torch.Tensor):
            if key in ('motor', 'state'): expected = bank[key][:, :, section]
            elif key == 'action': expected = bank[key][:, section]
            else: expected = bank[key][section]
            assert torch.equal(value.detach().cpu(), expected), f'Sealed plan changed: {key}'
        else:
            assert value == (1 if key == 'n_contexts' else bank[key]), key
    assert torch.equal(plan['heading_index'].cpu(), torch.arange(16))
    return name, item, context


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _failed_context(context, reason):
    pairs = [dict(headings=[i, i+8], certified=False,
                  conditions={'complete_valid_artifacts': False},
                  common_policy_mean_success_upper_bound=1., failures=[reason]) for i in range(8)]
    return dict(context=context, certified_all_pairs=False, artifact_schema_valid=False,
        pairs=pairs, heading_trials=16, equally_weighted_pairs=8,
        common_information_success_upper_bound=1., failures=[reason])


def certify_context(plan, cfg, roll, first_free, variants):
    """Check one real context without replacing invalid observations.

    Plan/config/source provenance failures raise. Missing or invalid recorded
    snapshots remain explicitly uncertified, with the conservative bound one.
    """
    bank_name, bank, context = _validate_plan(plan, cfg)
    assert list(variants).count(RESET_VARIANT) == 1, 'Need the actual post-cue-erasure branch'
    branch = list(variants).index(RESET_VARIANT); section = slice(16*branch, 16*(branch+1))
    schedule = task.schedule(plan, 'cpu'); starts = schedule['free'].cpu()
    assert bool((starts == starts[0]).all())
    start = int(starts[0]); stop = start + 30
    assert stop <= task.MAX_STEPS and start > int(schedule['glimpse'][0])
    try:
        snapshots = [first_free[16*branch+i] for i in range(16)]
        arrays = {new: np.stack([_numpy(row[old]) for row in snapshots])
            for new, old in [('first_free_input', 'input'), ('erased_state', 'state'),
                             ('terminal_pos', 'position'), ('terminal_hd', 'heading'),
                             ('terminal_motors', 'motors')]}
        arrays['goal_snapshot'] = _numpy(roll['goal'][section])
        arrays['motor'] = _numpy(plan['motor'][start:stop])
        arrays['uniform'] = _numpy(plan['action'][start:stop])
        assert arrays['motor'].shape == (30, 3, 16) and arrays['uniform'].shape == (30, 16)
        assert np.array_equal(arrays['terminal_pos'], _numpy(plan['endpoint']))
        assert len(roll['phase']) == len(roll['active']) == len(roll['pa']) == len(roll['pr']) == len(roll['action']) == task.MAX_STEPS
        actual_phase = torch.stack(roll['phase'])[:, section].cpu()
        assert torch.equal(actual_phase, schedule['phase']), 'Recorded clock differs from sealed schedule'
        active = torch.stack(roll['active'])[:, section].cpu().bool()
        previous = torch.stack(roll['pa'])[:, section].cpu()
        reward_input = torch.stack(roll['pr'])[:, section].cpu()
        actions = torch.stack(roll['action'])[:, section].cpu()
        assert bool(active[start].all()), 'Missing a first-FREE observation'
        assert bool((reward_input[start:stop][active[start:stop]] == 0).all()), 'Pre-hit reward feedback'
        for t in range(start, stop):
            assert torch.equal(previous[t][active[t]], actions[t-1][active[t]]), 'Previous action is not the actual action'
        report = inherited.certify(arrays, cfg, _numpy(plan['goal'][0]))
        # This is an extra empirical check of the recorded erased controller.
        # The theorem itself covers arbitrary common policies under the stated
        # held-input rule, not just this recorded action sequence.
        for pair in report['pairs']:
            i, j = pair['headings']
            common_active = active[start:stop, i] & active[start:stop, j]
            equal_actions = torch.equal(actions[start:stop, i][common_active], actions[start:stop, j][common_active])
            pair['conditions']['erased_actions_equal_before_either_hit'] = equal_actions
            if not equal_actions: pair['failures'].append('erased_actions_equal_before_either_hit')
            pair['certified'] = all(pair['conditions'].values())
            pair['common_policy_mean_success_upper_bound'] = .5 if pair['certified'] else 1.
        report['certified_all_pairs'] = all(p['certified'] for p in report['pairs'])
        report['common_information_success_upper_bound'] = float(np.mean([p['common_policy_mean_success_upper_bound'] for p in report['pairs']]))
        report.update(context=context, artifact_schema_valid=True)
    except (KeyError, AssertionError, ValueError, IndexError, TypeError) as error:
        report = _failed_context(context, f'{type(error).__name__}: {error}')
    report.update(bank=bank_name, bank_sha256=bank['sha256'], behavior_protocol_sha256=BEHAVIOR_HASH,
        certificate_source_sha256=HELPER_HASH, adapter_source_sha256=sha(__file__),
        core_source_sha256=sha(task.__file__), environment_source_sha256=sha(ROOT/'code/env.py'),
        free_start_update=start, free_stop_exclusive=stop,
        physical_tape_rule='Use the exact30body-noise/action draws at this context FREE onset, not tape indices0:30.',
        observation_rule='Frozen collector: blackRGB, held ordinary endpoint snapshot and saved goal; actual previous action; zero previous reward before first-hit termination. No unrecorded heading input or heading-informed private controller state.',
        source_scope='Read-only saved-input adapter. No policy/environment execution, no new bound, no outcome-dependent exclusion.')
    return report


def aggregate_certificates(certificates, expected_contexts=16):
    assert expected_contexts == 16
    indexed = {r['context']: r for r in certificates}
    assert len(indexed) == len(certificates), 'Duplicate physical context'
    assert all(0 <= k < expected_contexts for k in indexed)
    if certificates:
        assert len({(r['bank'], r['bank_sha256']) for r in certificates}) == 1
    rows = []; certified = 0
    for context in range(expected_contexts):
        row = indexed.get(context) or _failed_context(context, 'Context collection/certificate missing; retained as uncertified')
        assert len(row['pairs']) == 8 and row['heading_trials'] == 16
        assert [p['headings'] for p in row['pairs']] == [[i, i+8] for i in range(8)]
        for pair in row['pairs']:
            assert pair['certified'] == all(pair['conditions'].values())
            assert pair['common_policy_mean_success_upper_bound'] == (.5 if pair['certified'] else 1.)
        certified += sum(pair['certified'] for pair in row['pairs'])
        rows.append(row)
    pairs = expected_contexts*8
    return dict(expected_contexts=expected_contexts, received_contexts=len(certificates),
        heading_trials=expected_contexts*16, equally_weighted_pairs=pairs,
        certified_pairs=certified, uncertified_pairs=pairs-certified,
        all_pairs_certified=certified == pairs,
        common_information_success_upper_bound=(.5*certified + pairs-certified)/pairs,
        contexts=rows, adapter_source_sha256=sha(__file__), certificate_source_sha256=HELPER_HASH,
        interpretation='Equal-weight conditional bound across all prescribedpairs. Every invalid/missingpair contributes1; certifiedpairs contribute.5. Context/cue/population replicas are not extra independentphysicalpairs. NativeGaussian noise has no unconditional50percent ceiling.')
