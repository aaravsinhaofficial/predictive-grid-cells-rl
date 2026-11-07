import copy
import numpy as np
import torch
from code import train_memory as train
from code.memory_task import gate_observation


def cfg(tmp_path, condition='gc100'):
    return train.parse_args(['--out', str(tmp_path/'run'), '--device', 'cpu', '--threads', '1',
        '--cond', condition, '--frames', '32', '--n_envs', '2', '--rollout', '2', '--task_ramp_frames', '1'])


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def make_blind_event(runtime):
    runtime['clock'].known.fill_(True); runtime['clock'].remaining.fill_(1)
    runtime['env'].t.zero_()
    # Keep the private fixture away from an accidental pickup/terminal.
    runtime['env'].goal[:] = -runtime['env'].pos.sign() * .9


def test_actual_online_replay_and_bootstrap_use_identical_gated_inputs(tmp_path):
    c = cfg(tmp_path); net, opt, source, rt = train.fresh_training(c); make_blind_event(rt)
    roll, state, boot, bi, _ = train.collect_rollout(net, source, rt, c, 2)
    assert not roll['blind'][0].any() and roll['blind'][1].all() and bi['blind'].all()
    assert roll['rgb'][1].count_nonzero() == roll['code'][1].count_nonzero() == 0
    assert bi['rgb'].count_nonzero() == bi['code'].count_nonzero() == 0
    assert rt['obs']['rgb'].count_nonzero() > 0  # rendering was not disabled
    ll, vv, _ = train.replay(net, roll, state)
    torch.testing.assert_close(ll, torch.stack(roll['actor_logits']), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(vv, torch.stack(roll['actor_values']), rtol=1e-5, atol=1e-6)
    rng = rt['clock'].generator.get_state().clone()
    with torch.no_grad():
        _, expected, _ = net.step(bi['rgb'], bi['code'], rt['goal'], rt['pa'], rt['pr'], rt['state'])
    assert torch.equal(boot, expected)
    assert torch.equal(rng, rt['clock'].generator.get_state()) and rt['clock'].draw_steps == 2


def test_terminal_masks_even_a_poisoned_bootstrap():
    rewards = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
    dones = [torch.tensor([1., 0.]), torch.tensor([1., 1.])]
    result = train.discounted_returns(rewards, dones, torch.full((2,), 1e20), .9)
    torch.testing.assert_close(result, torch.tensor([[1., 5.6], [3., 4.]]))


def test_complete_native_optimizer_resume_is_bitwise_exact(tmp_path):
    c = cfg(tmp_path); net, opt, source, rt = train.fresh_training(c); make_blind_event(rt)
    before = train.cpu_tree(net.state_dict())
    m1, _, _ = train.train_update(net, opt, source, rt, c, return_trace=True)
    assert np.isfinite(m1['grad_norm']) and m1['grad_norm'] > 0
    assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())
    proof = {'unit_test': 'native architecture, private synthetic fixture'}
    saved = train.snapshot(net, opt, rt, c, proof)
    path = tmp_path/'state.pt'; train.atomic_torch(path, saved)
    m2, ep2 = train.train_update(net, opt, source, rt, c)
    expected = train.snapshot(net, opt, rt, c, proof)
    net2, opt2, source2, rt2 = train.fresh_training(c)
    train.restore(torch.load(path, weights_only=False), net2, opt2, rt2, c, proof)
    actual_m, actual_ep = train.train_update(net2, opt2, source2, rt2, c)
    assert m2 == actual_m and ep2 == actual_ep
    assert equal(expected, train.snapshot(net2, opt2, rt2, c, proof))


def test_paired_initialization_is_exact_and_all_parameters_trainable(tmp_path):
    a, _, _, _ = train.fresh_training(cfg(tmp_path, 'gc100'))
    b, _, _, _ = train.fresh_training(cfg(tmp_path, 'pgc33'))
    assert train.policy_hash(a) == train.policy_hash(b)
    assert all(p.requires_grad for p in a.parameters())
    assert (a.nh, a.n_code, a.n_actions) == (256, 512, 6)


def test_exact_resume_rejects_changed_task_or_source(tmp_path):
    c = cfg(tmp_path); net, opt, _, rt = train.fresh_training(c)
    saved = train.snapshot(net, opt, rt, c, {'source': 'old'})
    import pytest
    with pytest.raises(AssertionError): train.restore(saved, net, opt, rt, c, {'source': 'changed'})
    c2 = dict(c, task_max_gap=0)
    with pytest.raises(AssertionError): train.restore(saved, net, opt, rt, c2, {'source': 'old'})


def test_sigterm_atomic_checkpoint_cli_resume_and_shortened_endpoint(tmp_path, monkeypatch):
    import json
    import os
    import signal
    out = tmp_path/'interrupted'
    args = ['--out', str(out), '--device', 'cpu', '--threads', '1', '--cond', 'gc100',
            '--frames', '40', '--n_envs', '2', '--rollout', '3']
    original_update = train.train_update
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def terminate_after_real_update(*a, **kw):
        result = original_update(*a, **kw)
        os.kill(os.getpid(), signal.SIGTERM)
        return result
    try:
        monkeypatch.setattr(train, 'train_update', terminate_after_real_update)
        assert train.main(args) == 128 + signal.SIGTERM
        first = torch.load(out/'resume.pt', weights_only=False)
        assert (first['frames'], first['updates']) == (24, 1)
        assert (out/'interrupted_000.json').is_file() and not (out/'summary.json').exists()
        first_bytes = (out/'resume.pt').read_bytes()
        monkeypatch.setattr(train, 'train_update', original_update)
        assert train.main(args + ['--resume']) == 0
        final = torch.load(out/'final.pt', weights_only=False)
        assert (final['frames'], final['updates']) == (40, 2)
        assert (out/'resume_input_001.pt').read_bytes() == first_bytes
        assert json.loads((out/'summary.json').read_text())['complete']
        second = json.loads((out/'metrics_001.jsonl').read_text())
        assert second['rollout_steps'] == 2 and second['frames'] == 40
        assert json.loads((out/'attempt_001.json').read_text())['restored_updates'] == 1
    finally:
        for s, handler in old_handlers.items(): signal.signal(s, handler)


def test_frozen_protocol_binds_population_map_clock_and_settings(tmp_path):
    import json
    import pytest
    c = cfg(tmp_path)
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(training_config={'n_envs': 2}, allowed_cohort=dict(
        conditions=['gc100', 'pgc33'], seeds=[1, 2, 3, 4], task_max_gaps=[0, 60],
        pgc_fractions={'gc100': 0., 'pgc33': .33}, code_seed_equals_seed=True,
        clock_seed_multiplier=1000, clock_seed_offset=2))))
    c['protocol'] = str(protocol)
    train.provenance(c)
    for key, bad in [('n_envs', 3), ('pgc_frac', .2), ('code_seed', 2), ('clock_seed', 999), ('task_max_gap', 30)]:
        with pytest.raises(AssertionError): train.provenance(dict(c, **{key: bad}))


def test_categorical_validation_elision_is_exact_for_native_shapes_and_preserves_rng():
    # Private synthetic actor/replay tensors; no policy or study initialization.
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator().manual_seed(813701)
        for shape in ((32, 6), (100, 32, 6)):
            source = torch.randn(shape, generator=generator) * 10
            actions = torch.randint(6, shape[:-1], generator=generator)
            results = []
            for kwargs in ({}, {'validate_args': False}):
                torch.random.default_generator.manual_seed(813702)
                logits = source.clone().requires_grad_()
                initial_rng = torch.get_rng_state().clone()
                dist = torch.distributions.Categorical(logits=logits, **kwargs)
                constructed_rng = torch.get_rng_state().clone()
                sampled = dist.sample()
                sampled_rng = torch.get_rng_state().clone()
                log_prob = dist.log_prob(actions); entropy = dist.entropy()
                (log_prob.sum() + 8e-5 * entropy.sum()).backward()
                results.append(tuple(x.detach().clone() for x in (
                    dist.logits, dist.probs, sampled, log_prob, entropy, logits.grad,
                    initial_rng, constructed_rng, sampled_rng, torch.get_rng_state())))
                assert torch.equal(initial_rng, constructed_rng)
                assert torch.equal(sampled_rng, torch.get_rng_state())
            assert all(torch.equal(a, b) for a, b in zip(*results))
    assert torch.equal(before, torch.get_rng_state())
