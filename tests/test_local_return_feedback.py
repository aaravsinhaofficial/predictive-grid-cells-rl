"""Synthetic policies/plans only; no study bank or trained checkpoint is loaded."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import pytest
import torch
from torch import nn

from code.nets import PolicyNet
import local_goal_return_training as base
import local_return_feedback as feedback


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__(); self.nh = 8; self.n_actions = 6
        self.encoder = nn.Linear(3, 256)
        self.lstm = nn.LSTMCell(1287, self.nh)
        self.pi = nn.Linear(self.nh, 6); self.v = nn.Linear(self.nh, 1)

    def embed(self, rgb): return self.encoder(rgb.mean((2, 3)))
    zero_state = PolicyNet.zero_state
    _concat = PolicyNet._concat
    extra_in = 0


def config():
    cfg = json.loads((ROOT / 'models/pgc33/pgc33_lstm_s1_1000M.json').read_text())
    return dict(cfg, nh=8, entropy=8e-5, n_envs=2, rollout=160)


def fixture(seed=827):
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); net = TinyPolicy()
    cfg = config(); source = base.make_source(cfg, 'cpu')
    plan = base.TrialStreams(seed + 1).draw(2, feedback.STAGE, cfg)
    return net, cfg, source, plan


def test_head_composition_keeps_original_features_and_matches_exact_archived_critic():
    original, archives, hashes = [], [], []
    for seed in (11, 12):
        net, _, _, _ = fixture(seed)
        original.append(copy.deepcopy(net.state_dict()))
        report = base.reset_native_heads(net, 149)
        archives.append(copy.deepcopy(net.state_dict())); hashes.append(report['new_head_hashes'])
    assert hashes[0] == hashes[1]
    composed = []
    for population in range(2):
        for mode in feedback.ACTOR_MODES:
            state = feedback.compose_initial_state(original[population], archives[population], mode, hashes[population])
            for key, value in state.items():
                expected = archives[population][key] if key.startswith('v.') or (
                    mode == 'reset_actor' and key.startswith('pi.')) else original[population][key]
                assert torch.equal(value, expected)
                assert value.data_ptr() != expected.data_ptr()
            composed.append(state)
    for key in ('v.weight', 'v.bias'):
        assert all(torch.equal(composed[0][key], state[key]) for state in composed)
    wrong = copy.deepcopy(archives[0]); wrong['lstm.weight_ih'][0, 0] += 1
    with pytest.raises(AssertionError, match='nonhead'):
        feedback.compose_initial_state(original[0], wrong, 'retained_actor', hashes[0])
    wrong = copy.deepcopy(archives[0]); wrong['v.weight'][0, 0] += 1
    with pytest.raises(AssertionError, match='v.weight'):
        feedback.compose_initial_state(original[0], wrong, 'retained_actor', hashes[0])


def test_disabled_branch_reproduces_frozen_clean_collector():
    net, cfg, source, plan = fixture()
    expected, rows, audit, first = base.collect(net, cfg, source, plan, sigma=0.)
    actual, actual_rows, actual_audit, actual_first = feedback._collect(
        net, cfg, source, plan, native_feedback=False)
    for key, value in expected.items():
        if isinstance(value, list):
            assert len(value) == len(actual[key])
            for old, new in zip(value, actual[key]): assert torch.equal(old, new), key
        else: assert torch.equal(value, actual[key]), key
    assert rows == actual_rows
    for key, value in audit.items(): assert actual_audit[key] == value, key
    for i in first:
        for key, value in first[i].items(): assert torch.equal(value, actual_first[i][key]), key


def test_all_active_inputs_have_fresh_native_rgb_and_spatial_code():
    net, cfg, source, plan = fixture(835)
    roll, _, audit, first = feedback.collect(net, cfg, source, plan)
    renderer = base.prepare_batch(cfg, source, plan)['env']
    for t, active in enumerate(roll['active']):
        expected = source(roll['observation_position'][t], roll['observation_direction'][t],
                          roll['observation_velocity'][t])
        assert torch.equal(roll['code'][t][active], expected[active])
        renderer.pos[:] = roll['observation_position'][t]
        renderer.hd[:] = roll['observation_heading'][t]
        assert torch.equal(roll['rgb'][t][active], renderer.render()[active])
        assert not bool(roll['rgb'][t][~active].any())
    sched = base.schedule(plan, 'cpu')
    for i in range(2):
        cue, free = int(sched['glimpse'][i]), int(sched['free'][i])
        assert float(roll['observation_velocity'][cue][i].norm()) > 0
        assert torch.equal(roll['motors_before_action'][cue][i], torch.zeros(3))
        assert torch.equal(roll['observation_position'][free][i], roll['position'][free-1][i])
        assert torch.equal(first[i]['position'], roll['observation_position'][free][i])
        assert torch.equal(first[i]['input'][256:768], roll['code'][free][i])
        # Both the preparation and every free update are genuinely observed.
        assert bool(roll['rgb'][base.VISIBLE_STEPS][i].any())
        assert bool(roll['rgb'][cue-1][i].any()) and bool(roll['rgb'][cue+1][i].any())
    assert audit['fresh_feedback_on_every_active_input']
    assert audit['midtrial_resets'] == 0 and audit['unique_endpoint_cue'] is False


def test_no_state_noise_or_reset_and_replay_uses_the_full_history():
    net, cfg, source, plan = fixture(846)
    roll, _, _, first = feedback.collect(net, cfg, source, plan)
    changed = copy.deepcopy(plan); changed['state'].fill_(1e9)
    other, _, _, other_first = feedback.collect(net, cfg, source, changed)
    for key in ('noise', 'reset_before', 'reset_after'):
        assert not bool(torch.stack(roll[key]).any())
    for key in ('action', 'logits', 'position'):
        assert torch.equal(torch.stack(roll[key]), torch.stack(other[key]))
    logits, values = feedback.replay(net, roll)
    assert torch.allclose(logits, torch.stack(roll['logits']), atol=1e-7, rtol=0.)
    assert torch.allclose(values, torch.stack(roll['values']), atol=1e-7, rtol=0.)
    for i in first: assert torch.equal(first[i]['state'], other_first[i]['state'])
    sched = base.schedule(plan, 'cpu'); cue = int(sched['glimpse'][0]); free = int(sched['free'][0])
    roll['code'][cue] = roll['code'][cue].clone().requires_grad_(True)
    logits, _ = feedback.replay(net, roll)
    gradient = torch.autograd.grad(logits[free, 0, 2], roll['code'][cue])[0]
    assert float(gradient[0].abs().sum()) > 0


def test_every_parameter_is_trainable_and_full_replay_gradients_reach_features():
    net, cfg, source, plan = fixture(851)
    for parameter in net.parameters(): parameter.requires_grad_(False)
    feedback.configure_trainable(net); opt = feedback.optimizer(net, cfg)
    assert {id(p) for group in opt.param_groups for p in group['params']} == {id(p) for p in net.parameters()}
    roll, _, _, _ = feedback.collect(net, cfg, source, plan)
    logits, values = feedback.replay(net, roll)
    logits.retain_grad(); values.retain_grad()
    targets = feedback.returns(roll, cfg['gamma'])
    loss, _ = feedback.losses(logits, values, targets, roll, cfg); loss.backward()
    for component in (net.encoder, net.lstm, net.pi, net.v):
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())
        assert sum(float(p.grad.abs().sum()) for p in component.parameters()) > 0
    mask, active = torch.stack(roll['policy_mask']), torch.stack(roll['active'])
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    assert torch.equal(values.grad[~active], torch.zeros_like(values.grad[~active]))
    heads_only = torch.optim.RMSprop(list(net.pi.parameters()) + list(net.v.parameters()), lr=cfg['lr'])
    with pytest.raises(AssertionError): feedback.update(net, cfg, heads_only, source, plan)


def test_first_hit_stops_native_physics_and_terminal_returns_have_no_future_value():
    net, cfg, source, plan = fixture(858)
    cfg['goal_radius'] = .50  # Synthetic easier geometry; never used by a study bank.
    roll, rows, audit, _ = feedback.collect(net, cfg, source, plan, retain_inputs=False, controller='planner')
    assert audit['free_hits'] == 2 and all(row['free_steps'] < 30 for row in rows)
    assert audit['forced_hits'] == 0 and audit['setup_learning_reward'] == 0
    assert audit['active_physics_updates'] + audit['padded_slots'] == 320
    assert audit['forced_updates'] + audit['free_updates'] == audit['active_physics_updates']
    for i in range(2):
        terminal = torch.stack(roll['terminal'])[:, i].nonzero().flatten()
        assert len(terminal) == 1
        end = int(terminal[0]); assert float(roll['raw_reward'][end][i]) == 10.
        assert not bool(torch.stack(roll['active'])[end+1:, i].any())
        assert torch.equal(torch.stack(roll['position'])[end:, i], roll['position'][end][i].expand(160-end, 2))
    roll['bootstrap'].fill_(1e9)
    targets = feedback.returns(roll, cfg['gamma'])
    assert float(targets.max()) <= 1.
    for reward, free in zip(roll['reward'], roll['policy_mask']): assert not bool(reward[~free].any())


def test_private_stream_restore_and_synthetic_full_network_optimizer_step():
    net, cfg, source, _ = fixture(862)
    feedback.configure_trainable(net); opt = feedback.optimizer(net, cfg)
    streams = feedback.TrialStreams(863)
    before = torch.random.get_rng_state().clone()
    plan = streams.draw(2, feedback.STAGE, cfg)
    saved = streams.snapshot(); peer = feedback.TrialStreams(863); peer.restore(saved)
    next_a, next_b = streams.draw(2, feedback.STAGE, cfg), peer.draw(2, feedback.STAGE, cfg)
    for key in next_a:
        if isinstance(next_a[key], torch.Tensor): assert torch.equal(next_a[key], next_b[key]), key
        else: assert next_a[key] == next_b[key]
    assert torch.equal(torch.random.get_rng_state(), before)
    # A synthetic optimizer step validates the preparation code, not a study run.
    old = copy.deepcopy(net.state_dict())
    metrics, _ = feedback.update(net, cfg, opt, source, plan)
    assert metrics['replay_error'] < 3e-4 and metrics['value_replay_error'] < 3e-4
    assert any(not torch.equal(old[key], value) for key, value in net.state_dict().items() if key.startswith('encoder.'))
    assert any(not torch.equal(old[key], value) for key, value in net.state_dict().items() if key.startswith('lstm.'))


def test_initializer_checks_archive_bytes_and_uses_original_or_exact_reset_actor(tmp_path, monkeypatch):
    """Every file here contains a TinyPolicy fixture; no real checkpoint is read."""
    net, cfg, _, _ = fixture(870); parent = copy.deepcopy(net.state_dict())
    reset = base.reset_native_heads(net, 871); archived = copy.deepcopy(net.state_dict())
    stem = 'models/pgc33/pgc33_lstm_s1_1000M'
    parent_path, config_path = tmp_path/(stem+'.pt'), tmp_path/(stem+'.json')
    parent_path.parent.mkdir(parents=True)
    torch.save(dict(frames=1_000_000_000, policy=parent), parent_path)
    config_path.write_text(json.dumps(cfg))
    archive_relative = 'results/local_goal_return_baseline/pgc33/initialized_policy.pt'
    archive_path = tmp_path/archive_relative; archive_path.parent.mkdir(parents=True)
    torch.save(dict(no_optimization=True, behavior_protocol_sha256='synthetic_behavior',
        config=dict(cfg, n_envs=32, rollout=160, entropy=.01), policy=archived,
        initialization=dict(new_head_hashes=reset['new_head_hashes'],
            parent_checkpoint_sha256=feedback.sha(parent_path), parent_config_sha256=feedback.sha(config_path))), archive_path)
    sources = {}
    for name in ('scripts/local_goal_return_training.py', 'code/nets.py', 'code/common.py', 'code/env.py', 'code/codes.py'):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic provenance fixture')
        sources[name] = feedback.sha(path)
    protocol = dict(source_sha256=sources,
        original_assets={stem+'.pt':feedback.sha(parent_path), stem+'.json':feedback.sha(config_path)},
        passed_baseline_assets={archive_relative:feedback.sha(archive_path)},
        evaluation=dict(behavior_protocol_sha256='synthetic_behavior'))
    protocol_path = tmp_path/feedback.PARENT_PROTOCOL; protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(feedback, 'PARENT_PROTOCOL_SHA256', feedback.sha(protocol_path))
    monkeypatch.setattr(feedback, 'build_policy', lambda config, device: TinyPolicy())
    rng = torch.random.get_rng_state().clone()
    retained, _, audit = feedback.initialize('pgc33', 'retained_actor', root=tmp_path)
    reset_net, _, _ = feedback.initialize('pgc33', 'reset_actor', root=tmp_path)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(retained.pi.weight, parent['pi.weight'])
    assert torch.equal(reset_net.pi.weight, archived['pi.weight'])
    assert torch.equal(retained.v.weight, reset_net.v.weight)
    assert all(p.requires_grad for p in retained.parameters()) and audit['no_optimization']
    with archive_path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(AssertionError): feedback.initialize('pgc33', 'retained_actor', root=tmp_path)
