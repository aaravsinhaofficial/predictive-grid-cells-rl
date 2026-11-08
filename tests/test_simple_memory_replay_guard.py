"""Private synthetic numerical controls; no study policy or GPU is loaded."""
import copy
import json
import math

import pytest
import torch
from torch import nn

from code import train_memory as original
from code import simple_memory_replay_guard as guard


class SmallPolicy(nn.Module):
    """Known batch-dependent tiny encoder offset, with explicit state/reset use."""
    def __init__(self, offset=3e-6, gain=16., log_probability_failure=False):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain))
        self.image_gain = nn.Parameter(torch.tensor(.01))
        self.offset = offset
        self.log_probability_failure = log_probability_failure

    def embed(self, rgb):
        result = rgb.flatten(1).mean(1, keepdim=True) * self.image_gain
        return result + (self.offset if len(rgb) > 2 else 0.)

    def step_from_embedding(self, e, code, goal, pa, pr, state):
        h = .5 * state[0] + .25 * state[1] + self.gain * e + code + goal + pr[:, None] + .01 * pa.clamp_min(0)[:, None]
        c = .5 * state[1] + e
        if self.log_probability_failure:
            logits = torch.cat((h + math.log(5), -h, -h, -h, -h, -h), dim=1)
        else:
            logits = torch.cat((h, -h, .5*h, -.5*h, h*0, h*0), dim=1)
        return logits, .01*h.squeeze(-1), (h, c)


def make_roll(policy):
    state0 = (torch.full((2, 1), .1), torch.full((2, 1), .2))
    state = tuple(v.clone() for v in state0)
    roll = {k: [] for k in ('rgb', 'code', 'goal', 'pa', 'pr', 'd', 'actor_logits', 'actor_values')}
    with torch.no_grad():
        for t in range(3):
            row = dict(rgb=torch.zeros(2, 1, 1, 1), code=torch.zeros(2, 1), goal=torch.zeros(2, 1),
                       pa=torch.full((2,), -1, dtype=torch.long), pr=torch.zeros(2), d=torch.zeros(2))
            ll, vv, state = policy.step_from_embedding(policy.embed(row['rgb']), row['code'], row['goal'], row['pa'], row['pr'], state)
            row.update(actor_logits=ll, actor_values=vv)
            for k, v in row.items(): roll[k].append(v)
            state = tuple(v * (1-row['d'])[:, None] for v in state)
    return roll, state0


def evaluate(policy, roll, state0):
    logits, values, _ = original.replay(policy, roll, state0)
    online, online_v = torch.stack(roll['actor_logits']), torch.stack(roll['actor_values'])
    return logits, values, online, online_v


def test_normal_path_retains_original_assertions_and_no_extra_forward(monkeypatch):
    policy = SmallPolicy(offset=0)
    roll, state0 = make_roll(policy)
    outputs = evaluate(policy, roll, state0)
    monkeypatch.setattr(guard, '_detached_replays', lambda *args: pytest.fail('Normal path must not replay again'))
    record = guard.validate_replay(policy, roll, state0, *outputs)
    assert record['accepted'] and not record['fallback'] and record['fallback_count'] == 0
    assert record['guard_id'] == guard.GUARD_ID and record['mode'] == 'original'
    json.dumps(record)
    ll, vv, online, online_v = outputs
    with pytest.raises(AssertionError):
        guard.validate_replay(policy, roll, state0, ll, vv+.001, online, online_v)


def test_bounded_fallback_preserves_graph_parameters_gradients_inputs_and_rng():
    policy = SmallPolicy()
    roll, state0 = make_roll(policy)
    outputs = evaluate(policy, roll, state0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(outputs[0], outputs[2], rtol=1e-4, atol=1e-5)
    for p in policy.parameters(): p.grad = torch.full_like(p, .125)
    before = copy.deepcopy((policy.state_dict(), [p.grad for p in policy.parameters()], roll, state0))
    rng = torch.get_rng_state().clone()
    record = guard.validate_replay(policy, roll, state0, *outputs)
    assert record['fallback'] and record['fallback_count'] == 1 and record['accepted']
    assert 1e-5 < record['max_raw_logit_abs'] < 1e-4
    assert record['max_encoder_abs'] < 1e-5
    assert record['actor_shaped_logits_and_values_exact'] and record['original_shaped_logits_and_values_exact']
    assert not record['learning_tensors_replaced'] and outputs[0].requires_grad
    assert torch.equal(rng, torch.get_rng_state())
    after = (policy.state_dict(), [p.grad for p in policy.parameters()], roll, state0)
    for a, b in zip(guard._tensor_leaves(before), guard._tensor_leaves(after)): assert torch.equal(a, b)
    json.dumps(record)


def test_guarded_update_is_exactly_the_original_learning_arithmetic_without_check():
    reference = SmallPolicy(); candidate = copy.deepcopy(reference)
    roll, state0 = make_roll(reference)
    results = []
    for policy, guarded in ((reference, False), (candidate, True)):
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=3e-4, alpha=.99, eps=1e-5)
        outputs = evaluate(policy, roll, state0)
        if guarded: guard.validate_replay(policy, roll, state0, *outputs)
        logits, values = outputs[:2]
        returns = torch.full_like(values, .3)
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        advantage = (returns-values).detach()
        policy_loss = -(dist.log_prob(torch.zeros(values.shape, dtype=torch.long))*advantage).sum(0).mean()
        value_loss = .5*(returns-values).square().sum(0).mean()
        entropy = dist.entropy().sum(0).mean()
        loss = policy_loss + .5*value_loss - 8e-5*entropy
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 40., error_if_nonfinite=True)
        gradients = [p.grad.clone() for p in policy.parameters()]
        optimizer.step()
        results.append((policy.state_dict(), optimizer.state_dict(), gradients, loss.detach()))
    for a, b in zip(guard._tensor_leaves(results[0]), guard._tensor_leaves(results[1])): assert torch.equal(a, b)


@pytest.mark.parametrize('corruption', ['rgb', 'code', 'goal', 'state', 'done'])
def test_corrupted_inputs_and_reset_are_rejected(corruption):
    policy = SmallPolicy(); roll, state0 = make_roll(policy)
    if corruption == 'state': state0[0][0, 0] += 1e-6
    elif corruption == 'done': roll['d'][0][0] = 1
    elif corruption == 'rgb': roll['rgb'][0][0, 0, 0, 0] += 1e-5
    else: roll[corruption][0][0, 0] += 1e-6
    outputs = evaluate(policy, roll, state0)
    with pytest.raises(guard.ReplayGuardError) as failure:
        guard.validate_replay(policy, roll, state0, *outputs)
    assert not failure.value.record['accepted']
    assert failure.value.record['rejection_stage'] in ('actor_shaped_replay_mismatch', 'raw_logit_ceiling', 'original_critic_check')


def test_corrupted_large_output_is_rejected_even_within_absolute_ceiling():
    policy = SmallPolicy(); roll, state0 = make_roll(policy)
    ll, vv, online, online_v = evaluate(policy, roll, state0)
    ll = ll.clone(); ll[0, 0, 4] += 2e-5
    with pytest.raises(guard.ReplayGuardError, match='original_shaped_replay_mismatch'):
        guard.validate_replay(policy, roll, state0, ll, vv, online, online_v)


def test_excessive_encoder_rounding_is_rejected_even_when_logit_ceiling_passes():
    policy = SmallPolicy(offset=2e-5, gain=2.); roll, state0 = make_roll(policy)
    outputs = evaluate(policy, roll, state0)
    assert float((outputs[0].detach()-outputs[2]).abs().max()) < 1e-4
    with pytest.raises(guard.ReplayGuardError, match='encoder_tolerance|encoder_absolute_ceiling'):
        guard.validate_replay(policy, roll, state0, *outputs)


def test_log_probability_retains_original_tolerance():
    policy = SmallPolicy(log_probability_failure=True); roll, state0 = make_roll(policy)
    outputs = evaluate(policy, roll, state0)
    assert float((outputs[0].detach()-outputs[2]).abs().max()) < 1e-4
    with pytest.raises(guard.ReplayGuardError, match='log_probability_tolerance'):
        guard.validate_replay(policy, roll, state0, *outputs)


def test_no_unbounded_or_critic_fallback():
    policy = SmallPolicy(); roll, state0 = make_roll(policy)
    ll, vv, online, online_v = evaluate(policy, roll, state0)
    with pytest.raises(guard.ReplayGuardError, match='raw_logit_ceiling'):
        guard.validate_replay(policy, roll, state0, ll+.001, vv, online, online_v)
    with pytest.raises(guard.ReplayGuardError, match='original_critic_check'):
        guard.validate_replay(policy, roll, state0, ll, vv+.001, online, online_v)


def test_rng_consumption_in_fallback_is_rejected(monkeypatch):
    policy = SmallPolicy(); roll, state0 = make_roll(policy)
    outputs = evaluate(policy, roll, state0)
    actual = guard._detached_replays
    def consume(*args):
        torch.rand(1)
        return actual(*args)
    monkeypatch.setattr(guard, '_detached_replays', consume)
    with pytest.raises(guard.ReplayGuardError, match='rng_mutation'):
        guard.validate_replay(policy, roll, state0, *outputs)


def test_accumulated_gradient_mutation_is_rejected(monkeypatch):
    policy = SmallPolicy(); roll, state0 = make_roll(policy)
    outputs = evaluate(policy, roll, state0)
    for p in policy.parameters(): p.grad = torch.full_like(p, .125)
    actual = guard._detached_replays
    def mutate(*args):
        checked = actual(*args)
        policy.gain.grad.add_(1.)
        return checked
    monkeypatch.setattr(guard, '_detached_replays', mutate)
    with pytest.raises(guard.ReplayGuardError, match='tensor_mutation'):
        guard.validate_replay(policy, roll, state0, *outputs)
