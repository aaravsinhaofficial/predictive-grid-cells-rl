"""Private synthetic controls only; no archived study checkpoint is executed."""
import ast
import copy
import inspect
import json
from pathlib import Path
import signal

import numpy as np
import pytest
import torch

from code import train_memory as base
from code import train_memory_v2 as v2


def cfg(tmp_path, condition='pgc33', frames=40):
    return base.parse_args(['--out', str(tmp_path/'new'), '--cond', condition,
        '--device', 'cpu', '--threads', '1', '--frames', str(frames), '--n_envs', '2', '--rollout', '2'])


def manifest(tmp_path, c):
    original = tmp_path/'original.json'
    original.write_text(json.dumps(dict(status='frozen', source_sha256={p: base.sha(base.ROOT/p) for p in base.SOURCES})))
    c['protocol'] = str(original)
    bindings = {}
    completed = {'gc100/native/seed1', 'gc100/blackout/seed1', 'gc100/blackout/seed2', 'pgc33/blackout/seed2'}
    resumed = {'pgc33/native/seed1', 'pgc33/blackout/seed1', 'gc100/native/seed2', 'pgc33/native/seed2'}
    for cond in ('gc100', 'pgc33'):
        for task in ('native', 'blackout'):
            for seed in (1, 2, 3, 4):
                key = f'{cond}/{task}/seed{seed}'
                bindings[key] = dict(kind='v1_completed' if key in completed else 'v1_resume' if key in resumed else 'v2_fresh')
    record = dict(schema=1, status='frozen', kind='simple_memory_replay_guard_amendment',
        parent_protocol_sha256=base.sha(original), source_sha256={p: base.sha(v2.ROOT/p) for p in v2.SOURCES},
        guard=dict(id='private_original_guard', status='frozen'), run_bindings=bindings,
        numerical_runtime={k: base.provenance(c)[k] for k in v2.RUNTIME_KEYS})
    path = tmp_path/'amendment.json'; path.write_text(json.dumps(record))
    return path, record


def old_guard(policy, roll, state0, logits, values, online, online_v):
    torch.testing.assert_close(logits.detach(), online, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(values.detach(), online_v, rtol=1e-4, atol=1e-5)
    return {'private_fixture': True}


def cli(c, amendment):
    args = []
    for k in ('out', 'cond', 'seed', 'code_seed', 'frames', 'n_envs', 'rollout', 'device', 'threads', 'protocol'):
        args += ['--'+k, str(c[k])]
    return args + ['--amendment', str(amendment)]


def test_learning_update_ast_identical_except_declared_guard_and_metadata():
    a = ast.parse(inspect.getsource(base.train_update)).body[0]
    b = ast.parse(inspect.getsource(v2.train_update)).body[0]
    old_checks = []
    def remove_old(n):
        yes = isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'torch.testing.assert_close'
        if yes: old_checks.append(ast.unparse(n))
        return yes
    a.body = [n for n in a.body if not remove_old(n)]
    new_checks = []
    def remove_new(n):
        text = ast.unparse(n)
        yes = text.startswith('replay_guard(') or text.startswith('guard_record = ') or text.startswith("metrics['replay_guard'] = ")
        if yes: new_checks.append(text)
        return yes
    b.body = [n for n in b.body if not remove_new(n)]
    assert len(old_checks) == 2 and 1 <= len(new_checks) <= 2
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def test_operational_cli_flags_never_change_scientific_configuration(tmp_path):
    c = cfg(tmp_path); amendment, _ = manifest(tmp_path, c)
    parsed, options = v2.parse_args(cli(c, amendment)+['--parent-run', '/unused/parent'])
    assert base.scientific_config(parsed) == base.scientific_config(c)
    assert options.parent_run == '/unused/parent'
    with pytest.raises(SystemExit): v2.parse_args(cli(c, amendment)+['--parent-run', '/unused', '--resume'])


def test_disabled_guard_refuses_before_any_environment_or_output(tmp_path, monkeypatch):
    c = cfg(tmp_path); amendment, _ = manifest(tmp_path, c)
    monkeypatch.setattr(v2, 'GUARD_ID', None)
    monkeypatch.setattr(base, 'fresh_training', lambda *a: pytest.fail('Environment must not be initialized'))
    before = (torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()))
    with pytest.raises(NotImplementedError): v2.main(cli(c, amendment)+['--parent-run', '/unused'])
    assert not Path(c['out']).exists()
    assert torch.equal(before[0], torch.get_rng_state())
    assert v2.tree_digest(before[1]) == v2.tree_digest(np.random.get_state())


def test_full_payload_migration_and_next_update_are_exact(tmp_path):
    c = cfg(tmp_path); manifest(tmp_path, c)
    net, opt, source, rt = base.fresh_training(c); proof = base.provenance(c)
    base.train_update(net, opt, source, rt, c)
    saved = base.snapshot(net, opt, rt, c, proof); before = v2.payload_digest(saved)
    expected_metrics, _ = base.train_update(net, opt, source, rt, c)
    expected = base.snapshot(net, opt, rt, c, proof)
    net2, opt2, source2, rt2 = base.fresh_training(c)
    assert v2.restore_parent(saved, net2, opt2, rt2, c, proof) == before
    actual_metrics, _ = base.train_update(net2, opt2, source2, rt2, c)
    assert actual_metrics == expected_metrics
    assert v2.payload_digest(base.snapshot(net2, opt2, rt2, c, proof)) == v2.payload_digest(expected)
    assert v2.payload_digest(saved) == before
    with pytest.raises(AssertionError): v2.restore_parent(saved, net2, opt2, rt2, c, dict(proof, torch_version='other'))


def test_migration_digest_covers_optimizer_environment_rng_and_container_types(tmp_path):
    c = cfg(tmp_path); net, opt, source, rt = base.fresh_training(c)
    base.train_update(net, opt, source, rt, c)
    saved = base.snapshot(net, opt, rt, c, {'private': True}); expected = v2.payload_digest(saved)
    for mutate in (
        lambda s: s['optimizer']['state'][0]['square_avg'].add_(1),
        lambda s: s['runtime']['environment_rng'].__setitem__(0, s['runtime']['environment_rng'][0]^1),
        lambda s: s['runtime']['state'][1].add_(.01),
        lambda s: s['runtime']['obs']['rgb'].zero_(),
    ):
        changed = copy.deepcopy(saved); mutate(changed); assert v2.payload_digest(changed) != expected
    assert v2.tree_digest([1, 2]) != v2.tree_digest((1, 2))


def test_accepted_update_policy_optimizer_runtime_rng_exact(tmp_path, monkeypatch):
    c = cfg(tmp_path); net, opt, source, rt = base.fresh_training(c)
    proof = {'private_fixture': True}; saved = base.snapshot(net, opt, rt, c, proof)
    expected_metrics, expected_ep = base.train_update(net, opt, source, rt, c)
    expected = base.snapshot(net, opt, rt, c, proof)
    net2, opt2, source2, rt2 = base.fresh_training(c)
    base.restore(saved, net2, opt2, rt2, c, proof)
    metrics, ep = v2.train_update(net2, opt2, source2, rt2, c)
    guard = metrics.pop('replay_guard')
    assert guard['accepted'] and guard['mode'] == 'original' and guard['fallback_count'] == 0
    assert metrics == expected_metrics and ep == expected_ep
    assert v2.payload_digest(base.snapshot(net2, opt2, rt2, c, proof)) == v2.payload_digest(expected)


def parent_fixture(tmp_path, c, record, amendment):
    parent = tmp_path/'parent'; parent.mkdir()
    net, opt, source, rt = base.fresh_training(c); proof = base.provenance(c)
    base.atomic_json(parent/'config.json', dict(base.scientific_config(c), population=source.pop.describe()))
    base.atomic_json(parent/'provenance.json', proof)
    base.atomic_torch(parent/'initial.pt', base.snapshot(net, opt, rt, c, proof))
    m1, _ = base.train_update(net, opt, source, rt, c)
    base.atomic_torch(parent/'resume.pt', base.snapshot(net, opt, rt, c, proof))
    m2, _ = base.train_update(net, opt, source, rt, c)
    (parent/'metrics_000.jsonl').write_text(json.dumps(m1)+'\n'+json.dumps(m2)+'\n')
    (parent/'episodes_000.jsonl').write_text('')
    base.atomic_json(parent/'attempt_000.json', dict(restored_updates=0, restored_frames=0))
    binding = record['run_bindings'][v2.logical_run_id(c)]
    binding.update(parent_checkpoint_sha256=base.sha(parent/'resume.pt'), last_logged_update=2,
                   parent_evidence_sha256={p.name: base.sha(p) for p in parent.iterdir()})
    amendment.write_text(json.dumps(record))
    return parent, binding, proof


def test_parent_auth_rejects_missing_changed_raw_log_and_wrong_lineage(tmp_path):
    c = cfg(tmp_path); amendment, record = manifest(tmp_path, c)
    parent, binding, proof = parent_fixture(tmp_path, c, record, amendment)
    saved, lineage = v2.authenticate_parent(parent, binding, c, proof)
    assert lineage['accepted_parent_update_cutoff'] == 1 and lineage['rolled_back_logged_updates'] == 1
    bad = copy.deepcopy(binding); bad['parent_evidence_sha256'].pop('episodes_000.jsonl')
    with pytest.raises(AssertionError): v2.authenticate_parent(parent, bad, c, proof)
    bad = copy.deepcopy(binding); bad['parent_checkpoint_sha256'] = '0'*64
    with pytest.raises(AssertionError): v2.authenticate_parent(parent, bad, c, proof)
    (parent/'metrics_000.jsonl').write_text('changed')
    with pytest.raises(AssertionError): v2.authenticate_parent(parent, binding, c, proof)


def test_parent_main_preserves_all_raw_bytes_cutoff_and_true_v2_payload(tmp_path, monkeypatch):
    c = cfg(tmp_path); amendment, record = manifest(tmp_path, c)
    parent, binding, _ = parent_fixture(tmp_path, c, record, amendment)
    before = {p.name: base.sha(p) for p in parent.iterdir()}
    monkeypatch.setattr(v2, 'GUARD_ID', 'private_original_guard')
    monkeypatch.setattr(v2, 'replay_guard', old_guard)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert v2.main(cli(c, amendment)+['--parent-run', str(parent)]) == 0
    finally:
        for s, h in handlers.items(): signal.signal(s, h)
    out = Path(c['out']); lineage = json.loads((out/'lineage.json').read_text())
    final = torch.load(out/'final.pt', weights_only=False)
    boundary = torch.load(out/'migration_boundary.pt', weights_only=False)
    old = torch.load(parent/'resume.pt', weights_only=False)
    assert v2.payload_digest(boundary) == v2.payload_digest(old)
    assert set(final) == set(old) and final['provenance']['provenance_version'] == 2
    assert final['provenance']['base_provenance'] == old['provenance']
    assert final['provenance']['lineage_sha256'] == base.sha(out/'lineage.json')
    assert final['frames'] == 40 and final['updates'] == 3
    assert lineage['accepted_parent_update_cutoff'] == 1 and lineage['last_logged_parent_update'] == 2
    assert [json.loads(s)['updates'] for s in (out/'metrics_000.jsonl').read_text().splitlines()] == [2, 3]
    assert {p.name: base.sha(p) for p in parent.iterdir()} == before
    assert {p.name: base.sha(p) for p in (out/'parent_evidence').iterdir()} == before
    assert base.sha(out/'initial.pt') == base.sha(parent/'initial.pt')


def test_manifest_rejects_completed_model_missing_union_or_changed_parent(tmp_path):
    c = cfg(tmp_path); amendment, record = manifest(tmp_path, c)
    assert v2.load_amendment(amendment, c)[1]['kind'] == 'v1_resume'
    for mutate in (lambda r: r['source_sha256'].pop('code/train_memory.py'),
                   lambda r: r.update(parent_protocol_sha256='0'*64),
                   lambda r: r['run_bindings'].pop('gc100/native/seed4')):
        changed = copy.deepcopy(record); mutate(changed); amendment.write_text(json.dumps(changed))
        with pytest.raises(AssertionError): v2.load_amendment(amendment, c)
    amendment.write_text(json.dumps(record))
    with pytest.raises(AssertionError): v2.load_amendment(amendment, dict(c, cond='gc100', task_max_gap=0))


def test_fresh_v2_sigterm_and_resume_keep_previous_attempt_and_checkpoint(tmp_path, monkeypatch):
    import os
    c = cfg(tmp_path); c.update(seed=3, code_seed=3, clock_seed=3002)
    amendment, record = manifest(tmp_path, c)
    monkeypatch.setattr(v2, 'GUARD_ID', 'private_original_guard')
    monkeypatch.setattr(v2, 'replay_guard', old_guard)
    original_update = v2.train_update
    def stop_after_update(*a, **kw):
        result = original_update(*a, **kw)
        os.kill(os.getpid(), signal.SIGTERM)
        return result
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        monkeypatch.setattr(v2, 'train_update', stop_after_update)
        assert v2.main(cli(c, amendment)) == 128+signal.SIGTERM
        out = Path(c['out']); old_checkpoint = (out/'resume.pt').read_bytes()
        old_log = (out/'metrics_000.jsonl').read_bytes()
        initial_hash = base.sha(out/'initial.pt'); lineage_hash = base.sha(out/'lineage.json')
        monkeypatch.setattr(v2, 'train_update', original_update)
        assert v2.main(cli(c, amendment)+['--resume']) == 0
        assert (out/'resume_input_001.pt').read_bytes() == old_checkpoint
        assert (out/'metrics_000.jsonl').read_bytes() == old_log
        assert base.sha(out/'initial.pt') == initial_hash and base.sha(out/'lineage.json') == lineage_hash
        summary = json.loads((out/'summary.json').read_text())
        assert summary['complete'] and summary['frames'] == 40 and summary['updates'] == 3
        assert summary['attempt_records'] == ['attempt_000.json', 'attempt_001.json']
        final = torch.load(out/'final.pt', weights_only=False)
        assert final['provenance']['provenance_version'] == 2
        assert final['config'] == base.scientific_config(c)
        with pytest.raises(FileExistsError): v2.main(cli(c, amendment)+['--resume'])
    finally:
        for s, h in handlers.items(): signal.signal(s, h)


def test_parent_migration_detects_runtime_corruption_by_restore(tmp_path, monkeypatch):
    c = cfg(tmp_path); net, opt, _, rt = base.fresh_training(c)
    saved = base.snapshot(net, opt, rt, c, {'private': True})
    original = base.restore
    def corrupt(*args, **kwargs):
        original(*args, **kwargs)
        rt['pa'][0] = 2
    monkeypatch.setattr(base, 'restore', corrupt)
    with pytest.raises(AssertionError, match='Migration changed'):
        v2.restore_parent(saved, net, opt, rt, c, {'private': True})


def test_v2_proof_binds_full_union_original_proof_runtime_and_lineage(tmp_path, monkeypatch):
    c = cfg(tmp_path); amendment, record = manifest(tmp_path, c)
    monkeypatch.setattr(v2, 'GUARD_ID', 'private_original_guard')
    proof = v2.provenance(c, amendment, 'a'*64)
    original = base.provenance(c)
    assert proof['base_provenance'] == original
    assert proof['config_sha256'] == original['config_sha256']
    assert proof['protocol_sha256'] == original['protocol_sha256']
    assert proof['amendment_sha256'] == base.sha(amendment)
    assert set(proof['source_sha256']) == set(v2.SOURCES)
    assert proof['lineage_sha256'] == 'a'*64
    record['numerical_runtime']['torch_version'] = 'other'
    amendment.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match='Numerical runtime'):
        v2.provenance(c, amendment, 'a'*64)


def endpoint_fixture(tmp_path, monkeypatch, resumed):
    """Use only private CPU initialization/updates, then authenticate saved bytes."""
    import shutil
    c = cfg(tmp_path)
    if not resumed: c.update(seed=3, code_seed=3, clock_seed=3002)
    amendment, record = manifest(tmp_path, c)
    parent = None
    if resumed: parent, _, _ = parent_fixture(tmp_path, c, record, amendment)
    monkeypatch.setattr(v2, 'GUARD_ID', 'private_original_guard')
    monkeypatch.setattr(v2, 'replay_guard', old_guard)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert v2.main(cli(c, amendment)+(['--parent-run', str(parent)] if resumed else [])) == 0
    finally:
        for s, h in handlers.items(): signal.signal(s, h)
    root = tmp_path/'isolated_auth_root'
    for name in v2.SOURCES:
        dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v2.ROOT/name, dest)
    protocol = root/'docs/research/simple_memory_training_protocol.json'
    protocol.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(c['protocol'], protocol)
    out = Path(c['out'])
    return root, out, amendment, json.loads((out/'config.json').read_text()), json.loads((out/'provenance.json').read_text())


@pytest.mark.parametrize('resumed', [False, True])
def test_endpoint_auth_genuine_v2_and_provenance_tampering(tmp_path, monkeypatch, resumed):
    from scripts.simple_memory_endpoint import validate_v2_provenance
    root, out, amendment, c, proof = endpoint_fixture(tmp_path, monkeypatch, resumed)
    def check(p=proof, config=c):
        return validate_v2_provenance(out, config, p, root=root, amendment_path=amendment)
    result = check()
    assert result['producer_assignment'] == ('v1_resume' if resumed else 'v2_fresh')
    assert result['restored_frames'] == (16 if resumed else 0)
    for key, value in [('producer_module', 'code.unknown'), ('provenance_version', 7),
                       ('logical_run_id', 'gc100/native/seed4'), ('lineage_sha256', '0'*64)]:
        with pytest.raises((AssertionError, ValueError)): check(dict(proof, **{key: value}))
    with pytest.raises((AssertionError, ValueError)): check(config=dict(c, seed=4))
    # File edits are confined to the temporary authentication root/output.
    for path in (root/'code/train_memory.py', amendment, out/'initial.pt'):
        before = path.read_bytes(); path.write_bytes(before+b'\n')
        try:
            with pytest.raises((AssertionError, ValueError)): check()
        finally:
            path.write_bytes(before)
    if resumed:
        log = out/'parent_evidence/metrics_000.jsonl'; before = log.read_bytes()
        log.write_bytes(before+b'\n')
        try:
            with pytest.raises((AssertionError, ValueError)): check()
        finally:
            log.write_bytes(before)
        boundary = out/'migration_boundary.pt'; before = boundary.read_bytes()
        changed = torch.load(boundary, weights_only=False)
        changed['runtime']['state'][1].add_(.01)
        torch.save(changed, boundary)
        try:
            with pytest.raises((AssertionError, ValueError), match='Migration changed'):
                check()
        finally:
            boundary.write_bytes(before)
    assert check() == result


def test_rejected_guard_record_preserved_without_partial_runtime_checkpoint(tmp_path, monkeypatch):
    from code.simple_memory_replay_guard import ReplayGuardError
    c = cfg(tmp_path); c.update(seed=3, code_seed=3, clock_seed=3002)
    amendment, _ = manifest(tmp_path, c)
    monkeypatch.setattr(v2, 'GUARD_ID', 'private_original_guard')
    def reject(*args):
        raise ReplayGuardError({'private': True}, 'private_corruption', 'Private negative control')
    monkeypatch.setattr(v2, 'replay_guard', reject)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(ReplayGuardError): v2.main(cli(c, amendment))
    finally:
        for s, h in handlers.items(): signal.signal(s, h)
    out = Path(c['out']); saved = torch.load(out/'resume.pt', weights_only=False)
    assert saved['frames'] == saved['updates'] == 0 and not (out/'final.pt').exists()
    rejected = json.loads((out/'rejected_update_000.json').read_text())
    assert rejected['guard']['rejection_stage'] == 'private_corruption'
    assert rejected['last_accepted_update'] == 0 and rejected['unaccepted_collected_frames'] == 16
    assert rejected['resume_sha256'] == base.sha(out/'resume.pt')
    assert not (out/'metrics_000.jsonl').read_text()
