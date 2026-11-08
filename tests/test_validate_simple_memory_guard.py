"""Private validation-harness controls; no archived model execution."""
import inspect
import json

import pytest
import torch

from code import train_memory as base
from code import train_memory_v2 as amended
from scripts import validate_simple_memory_guard as validation


def wrong_tolerance_update():
    torch.testing.assert_close(torch.ones(1), torch.zeros(1), rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(torch.ones(1), torch.zeros(1), rtol=1e-4, atol=1e-5)


def test_reference_preserves_original_accepted_update_and_all_state(tmp_path):
    cfg = base.parse_args(['--out', str(tmp_path/'unused'), '--device', 'cpu', '--threads', '1',
                          '--frames', '40', '--n_envs', '2', '--rollout', '2'])
    policy, optimizer, source, runtime = base.fresh_training(cfg)
    proof = {'private_fixture': True}
    saved = base.snapshot(policy, optimizer, runtime, cfg, proof)
    original_function = base.train_update
    original_source = inspect.getsource(base.train_update)
    expected_metrics, expected_episodes = original_function(policy, optimizer, source, runtime, cfg)
    expected = amended.payload_digest(base.snapshot(policy, optimizer, runtime, cfg, proof))
    gradients = validation.gradient_digest(policy)
    amended.restore_parent(saved, policy, optimizer, runtime, cfg, proof)
    reference, description = validation.reference_update()
    metrics, episodes = reference(policy, optimizer, source, runtime, cfg)
    assert metrics == expected_metrics and episodes == expected_episodes
    assert amended.payload_digest(base.snapshot(policy, optimizer, runtime, cfg, proof)) == expected
    assert validation.gradient_digest(policy) == gradients
    assert len(description['removed_statements']) == 2
    assert base.train_update is original_function and inspect.getsource(base.train_update) == original_source


def test_reference_refuses_unexpected_original_assertions(monkeypatch):
    monkeypatch.setattr(base, 'train_update', wrong_tolerance_update)
    with pytest.raises(RuntimeError, match='tolerances changed'):
        validation.reference_update()


def test_real_checkpoint_validation_refuses_cpu_before_read_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    out = tmp_path/'must_not_exist'
    with pytest.raises(RuntimeError, match='CUDA required'):
        validation.execute(tmp_path/'missing.json', out)
    assert not out.exists()


def test_reconstruction_requires_every_original_update_once(tmp_path):
    path = tmp_path/'metrics_000.jsonl'
    def write(rows):
        path.write_text(''.join(json.dumps({'updates': update})+'\n' for update in rows))
    write([10, 11, 12, 13])
    assert [row['updates'] for row in validation.accepted_rows(tmp_path, 10, 13)] == [11, 12, 13]
    for rows in ([10, 11, 13], [10, 11, 11, 12, 13], [10, 11, 12, 13, 14]):
        write(rows)
        with pytest.raises(RuntimeError):
            validation.accepted_rows(tmp_path, 10, 13)
