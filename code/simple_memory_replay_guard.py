"""Narrow numerical guard for the diagnosed FP32 CNN batch-shape discrepancy.

The original assertions are the normal path. A logit-only failure can be
accepted only after independent, detached replays establish the observed
batch-shape explanation within fixed ceilings. Learning tensors are never
replaced. This module performs no backward pass or optimizer update.
"""
from __future__ import annotations

import torch

GUARD_ID = 'cnn_batch_roundoff_v1'
RTOL, ATOL = 1e-4, 1e-5
ENCODER_ABS_MAX = 1e-5
RAW_LOGIT_ABS_MAX = 1e-4


class ReplayGuardError(AssertionError):
    """A rejected update, with a JSON-safe record for the attempt audit."""

    def __init__(self, record, stage, message):
        self.record = dict(record, accepted=False, rejection_stage=stage)
        super().__init__(f'{GUARD_ID}: {stage}: {message}')


def _require(condition, record, stage, message):
    if not condition:
        raise ReplayGuardError(record, stage, message)


def _close(actual, expected, record, stage):
    try:
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        raise ReplayGuardError(record, stage, str(exc)) from exc


def _max_abs(actual, expected):
    return float((actual.detach() - expected.detach()).abs().max())


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child)


def _snapshot_side_effects(policy, roll, state0, outputs):
    params = list(policy.parameters())
    buffers = list(policy.buffers())
    leaves = list(_tensor_leaves((roll, state0, outputs))) + params + buffers
    gradients = [p.grad for p in params]
    leaves += [g for g in gradients if g is not None]
    devices = sorted({t.device.index for t in leaves if t.device.type == 'cuda'})
    return dict(tensors=[(t, t._version) for t in leaves], parameters=params, buffers=buffers,
                gradients=gradients, requires_grad=[p.requires_grad for p in params],
                modes=[(module, module.training) for module in policy.modules()],
                cpu_rng=torch.get_rng_state().clone(),
                cuda_rng={device: torch.cuda.get_rng_state(device).clone() for device in devices})


def _check_side_effects(policy, saved, record):
    _require(all(t._version == version for t, version in saved['tensors']), record,
             'tensor_mutation', 'A guarded input, parameter, buffer or gradient changed in place')
    _require(all(p.grad is g for p, g in zip(saved['parameters'], saved['gradients'])), record,
             'gradient_mutation', 'A parameter gradient was replaced')
    current_params, current_buffers = list(policy.parameters()), list(policy.buffers())
    _require(len(current_params) == len(saved['parameters']) and
             all(a is b for a, b in zip(current_params, saved['parameters'])) and
             len(current_buffers) == len(saved['buffers']) and
             all(a is b for a, b in zip(current_buffers, saved['buffers'])), record,
             'parameter_or_buffer_replacement', 'A parameter or buffer was replaced')
    _require(all(p.requires_grad == flag for p, flag in zip(saved['parameters'], saved['requires_grad'])),
             record, 'gradient_mode_mutation', 'Parameter trainability changed')
    _require(all(module.training == mode for module, mode in saved['modes']), record,
             'mode_mutation', 'A policy module mode changed')
    _require(torch.equal(torch.get_rng_state(), saved['cpu_rng']), record,
             'rng_mutation', 'CPU RNG changed during detached checks')
    _require(all(torch.equal(torch.cuda.get_rng_state(d), state) for d, state in saved['cuda_rng'].items()),
             record, 'rng_mutation', 'CUDA RNG changed during detached checks')


def _detached_replays(policy, roll, state0):
    """Repeat the original batch shape, then the actor's per-step batch shape."""
    length, n = len(roll['rgb']), len(roll['pa'][0])
    large_e = policy.embed(torch.cat(roll['rgb'], 0)).view(length, n, -1)
    state = tuple(v.detach() for v in state0)
    large_logits, large_values = [], []
    for t in range(length):
        logits, values, state = policy.step_from_embedding(
            large_e[t], roll['code'][t], roll['goal'][t], roll['pa'][t], roll['pr'][t], state)
        large_logits.append(logits); large_values.append(values)
        state = tuple(v * (1 - roll['d'][t])[:, None] for v in state)
    state = tuple(v.detach() for v in state0)
    small_e, small_logits, small_values = [], [], []
    for t in range(length):
        embedding = policy.embed(roll['rgb'][t])
        logits, values, state = policy.step_from_embedding(
            embedding, roll['code'][t], roll['goal'][t], roll['pa'][t], roll['pr'][t], state)
        small_e.append(embedding); small_logits.append(logits); small_values.append(values)
        state = tuple(v * (1 - roll['d'][t])[:, None] for v in state)
    return dict(large_embeddings=large_e, small_embeddings=torch.stack(small_e),
                large_logits=torch.stack(large_logits), large_values=torch.stack(large_values),
                small_logits=torch.stack(small_logits), small_values=torch.stack(small_values))


def validate_replay(policy, roll, state0, logits, values, online, online_v):
    """Validate, return an audit record, and leave all learning tensors untouched.

    Fallback is deliberately limited to a logit comparison failure. The critic
    retains its original check. Equal actor-shaped outputs and equal repeated
    original outputs are required, followed by fixed embedding/output bounds.
    """
    record = dict(guard_id=GUARD_ID, accepted=True, fallback=False, mode='original',
                  fallback_count=0, tolerance=dict(rtol=RTOL, atol=ATOL))
    try:
        torch.testing.assert_close(logits.detach(), online, rtol=1e-4, atol=1e-5)
    except AssertionError as exc:
        record.update(fallback=True, fallback_count=1, mode='bounded_cnn_batch_roundoff',
                      original_logit_assertion=str(exc),
                      ceilings=dict(encoder_absolute=ENCODER_ABS_MAX, raw_logit_absolute=RAW_LOGIT_ABS_MAX))
    else:
        torch.testing.assert_close(values.detach(), online_v, rtol=1e-4, atol=1e-5)
        return record

    _close(values.detach(), online_v, record, 'original_critic_check')
    _require(all(t.dtype == torch.float32 for t in (logits, values, online, online_v)), record,
             'unsupported_dtype', 'Fallback is validated only for native FP32')
    _require(all(bool(torch.isfinite(t).all()) for t in (logits, values, online, online_v)), record,
             'nonfinite_output', 'Fallback requires finite learning and actor outputs')
    record['max_raw_logit_abs'] = _max_abs(logits, online)
    record['max_value_abs'] = _max_abs(values, online_v)
    _require(record['max_raw_logit_abs'] <= RAW_LOGIT_ABS_MAX, record,
             'raw_logit_ceiling', 'Original batched logit discrepancy exceeds the fixed ceiling')
    length = len(roll['rgb'])
    _require(length > 0 and all(len(roll[k]) == length for k in
             ('code', 'goal', 'pa', 'pr', 'd', 'actor_logits', 'actor_values')), record,
             'rollout_shape', 'Rollout fields have inconsistent lengths')
    _require(torch.equal(torch.stack(roll['actor_logits']), online) and
             torch.equal(torch.stack(roll['actor_values']), online_v), record,
             'actor_record_mismatch', 'Online references differ from the recorded actor outputs')
    _require(all(bool(((done == 0) | (done == 1)).all()) for done in roll['d']), record,
             'invalid_reset', 'Done masks must contain only zero or one')

    saved = _snapshot_side_effects(policy, roll, state0, (logits, values, online, online_v))
    try:
        with torch.no_grad():
            checked = _detached_replays(policy, roll, state0)
            _require(torch.equal(checked['small_logits'], online) and
                     torch.equal(checked['small_values'], online_v), record,
                     'actor_shaped_replay_mismatch', 'Same-batch replay is not exactly equal to the actor')
            _require(torch.equal(checked['large_logits'], logits.detach()) and
                     torch.equal(checked['large_values'], values.detach()), record,
                     'original_shaped_replay_mismatch', 'Repeated original replay differs from the learning tensors')
            _close(checked['large_embeddings'], checked['small_embeddings'], record, 'encoder_tolerance')
            record['max_encoder_abs'] = _max_abs(checked['large_embeddings'], checked['small_embeddings'])
            _require(record['max_encoder_abs'] <= ENCODER_ABS_MAX, record,
                     'encoder_absolute_ceiling', 'CNN batch discrepancy exceeds the fixed absolute ceiling')
            logp, online_logp = logits.detach().log_softmax(-1), online.detach().log_softmax(-1)
            _close(logp, online_logp, record, 'log_probability_tolerance')
            record['max_log_probability_abs'] = _max_abs(logp, online_logp)
            record.update(actor_shaped_logits_and_values_exact=True,
                          original_shaped_logits_and_values_exact=True,
                          encoder_original_tolerance_passed=True,
                          log_probability_original_tolerance_passed=True,
                          critic_original_tolerance_passed=True)
    finally:
        _check_side_effects(policy, saved, record)
    record.update(inputs_parameters_buffers_gradients_unmodified=True, rng_unmodified=True,
                  learning_tensors_replaced=False)
    return record
