import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts import lstm_distillation_memory as assay


def family():
    theta = np.arange(192) * 2 * np.pi / 192
    return np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))


def short_probe(transition, output):
    timing = assay.WINDOWS['within_trained_2s']
    with patch.object(assay.observable, 'OBSERVATIONS', timing['offsets']), \
         patch.object(assay.observable, 'EVOLUTION', timing['evolution']):
        return assay.observable.observable_probe(transition, family(), np.ones(3),
                                                 output, count=192)


def test_timings_fit_two_seconds_and_retain_exact_four_second_extrapolation():
    assert [(t['reference'] + t['evolution'] + max(t['offsets']))
            for t in assay.WINDOWS.values()] == [30, 60]
    assert assay.WINDOWS['legacy_4s_extrapolation'] == dict(
        reference=15, evolution=30, offsets=(0, 5, 10, 15))


def test_new_short_window_recognizes_ring_with_common_drift():
    def transition(z):
        r = z[:, :2].norm(dim=1, keepdim=True)
        xy = z[:, :2] * (.8 + .2 / r)
        x = xy[:, 0] * np.cos(.04) - xy[:, 1] * np.sin(.04)
        y = xy[:, 0] * np.sin(.04) + xy[:, 1] * np.cos(.04)
        return torch.stack((x, y, z[:, 2] + .1), 1)
    result = short_probe(transition, lambda z: z)
    assert result['centered_conditional_factor_candidate']
    assert not result['noncentered_observable_candidate']


def test_new_short_window_rejects_shrinking_plane_and_silent_output():
    shrinking = short_probe(lambda z: torch.cat((z[:, :2] * .98, z[:, 2:] + .1), 1), lambda z: z)
    assert not shrinking['centered_conditional_factor_candidate']
    assert .9 < shrinking['radial_over_tangent_observable_gain']['median'] < 1.1
    silent = short_probe(lambda z: z, lambda z: z[:, 2:])
    assert not silent['centered_conditional_factor_candidate']
    assert not silent['candidate_gates']['centered']['observable_radial_kick_nontrivial_in80percent']


def test_full_state_gate_adapter_reproduces_original_four_second_criteria():
    # Probe all criteria near both sides of their thresholds, without a neural fit.
    for factor in (.5, 1., 2.):
        g = dict(max_edge_over_radius=.1 * factor,
                 nonlocal_distance_over_radius=dict(p10=.05 * factor))
        r = dict(phase_transport=dict(circular_coverage=.8 * factor,
                     winding_degree=1. + .09 * factor, backward_edge_fraction=.05 * factor),
                 residual_to_fixed_curve_over_radius=dict(p90=.05 * factor),
                 radial_normal_gain=dict(median=.5 * factor),
                 radial_over_tangent_gain=dict(median=.5 * factor))
        invariance = {'30': {m: dict(geometry=g, horizons={'30': r})
                              for m in ('raw', 'relative_rms')}}
        trajectory = {'30': {m: dict(radius_retention=.05 * factor)
                             for m in ('raw', 'relative_rms')}}
        old = assay.heading.geometry_gates(invariance, trajectory)
        new = assay.full_state_gates(dict(invariance=invariance, trajectory=trajectory), 30, 30)
        assert all(list(old[m].values()) == list(new[m].values()) for m in old)


def test_actual_collection_clock_caps_at_thirty_and_clears_swaps_on_reset():
    clock = assay.CollectionClock(16, 'cpu', 98203)
    hit = torch.ones(16, dtype=torch.bool); clear = torch.zeros_like(hit)
    clock.advance(hit, clear, 60)
    for _ in range(15):
        entering = clock.advance(clear, clear, 60)
    assert bool(entering.all()) and bool(clock.mask().all())
    assert bool(((clock.remaining >= 15) & (clock.remaining <= 30)).all())
    assert bool(clock.swap.any()) and not bool(clock.swap.all())
    clock.advance(clear, hit, 60)
    assert not bool(clock.mask().any()) and not bool(clock.swap.any())


def test_collection_current_substitution_preserves_goal_code():
    holder, Clock, Source = assay.collection_adapters()
    class Population:
        def activity(self, pos, dir, vel, shift_scale):
            return torch.full((len(pos), 512), float(shift_scale))
    cfg = dict(n_cells=512, code_mode='grid', goal_code_shift='intact', arena_half=1.25)
    source = Source(cfg, Population(), 'cpu')
    clock = Clock(2, 'cpu', 98203)
    clock.known.fill_(True); clock.blind.fill_(True)
    clock.swap[:] = torch.tensor([True, False])
    x = torch.zeros(2, 2)
    assert torch.equal(source(x, x, x)[:, 0], torch.tensor([0., 1.]))
    assert bool((source.goal(x, x, x) == 1.).all())


def test_checkpoint_selection_rejects_partial_stage_and_changed_frozen_tensor():
    training = json.loads(assay.TRAINING_PROTOCOL.read_text())
    original = {'lstm.weight_ih': torch.ones(2), 'pi.weight': torch.ones(2)}
    summary = dict(complete=True, updates=180, frozen_tensors_unchanged=True,
                   teacher_unchanged=True, teacher_policy_hash='teacher')
    data = dict(config=dict(cond='gc100', seed=1), original_frames=1_000_000_000,
                protocol_sha256=assay.sha(assay.TRAINING_PROTOCOL), updates=180,
                extra_frames=1_036_800, teacher_policy_hash='teacher',
                source_sha256=training['source_sha256'], policy=copy.deepcopy(original))
    assay.validate_endpoint(data, 'gc100', 'final', summary, original)
    wrong = copy.deepcopy(data); wrong['updates'] = 60
    with pytest.raises(AssertionError):
        assay.validate_endpoint(wrong, 'gc100', 'final', summary, original)
    wrong = copy.deepcopy(data); wrong['policy']['pi.weight'][0] = 2.
    with pytest.raises(AssertionError):
        assay.validate_endpoint(wrong, 'gc100', 'final', summary, original)
    wrong = copy.deepcopy(summary); wrong['complete'] = False
    with pytest.raises(AssertionError):
        assay.validate_endpoint(data, 'gc100', 'final', wrong, original)


def test_complete_prefix_replay_resets_between_episodes():
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lstm = torch.nn.LSTMCell(3, 4)
        def zero_state(self, n, device):
            return torch.zeros(n, 4, device=device), torch.zeros(n, 4, device=device)
    torch.manual_seed(994101)
    net = Small().eval().requires_grad_(False)
    inputs = torch.randn(9, 3).numpy()
    reset = np.array([True, False, False, False, False, True, False, False, False])
    actual = assay.replay_prefix(net, inputs, reset, 'cpu')
    expected = net.zero_state(1, 'cpu')
    for x in inputs[5:]:
        expected = net.lstm(torch.tensor(x[None]), expected)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    no_reset = assay.replay_prefix(net, inputs, np.zeros(9, dtype=bool), 'cpu')
    assert any(not torch.equal(a, b) for a, b in zip(actual, no_reset))


def test_paired_context_replays_own_weights_before_visible_interval(tmp_path):
    class NativeSize(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lstm = torch.nn.LSTMCell(1287, 256)
        def zero_state(self, n, device):
            return torch.zeros(n, 256, device=device), torch.zeros(n, 256, device=device)
    torch.manual_seed(994201)
    initial = NativeSize().eval().requires_grad_(False)
    final = copy.deepcopy(initial)
    with torch.no_grad():
        final.lstm.bias_ih.add_(.2)
    prefix = torch.randn(4, 1287).numpy(); resets = np.array([True, False, False, False])
    history = torch.randn(15, 1287).numpy()
    source = dict(prefix_inputs=prefix, prefix_reset_before=resets,
                  prefix_sha256=np.array(assay.prefix_hash(prefix, resets)),
                  history_x=history, checkpoint_sha256=np.array('source'),
                  history_pos=np.ones((15, 2), dtype=np.float32))
    checkpoint = tmp_path / 'checkpoint.pt'; checkpoint.write_bytes(b'not actual weights')
    a = assay.replay_context(initial, source, checkpoint, 'cpu')
    b = assay.replay_context(final, source, checkpoint, 'cpu')
    state = assay.replay_prefix(final, prefix, resets, 'cpu')
    assert np.array_equal(b['history_h_before'][0], state[0][0].numpy())
    assert np.array_equal(b['history_c_before'][0], state[1][0].numpy())
    for t, x in enumerate(history):
        state = final.lstm(torch.tensor(x[None]), state)
        assert np.array_equal(b['history_h_after'][t], state[0][0].numpy())
    assert not np.array_equal(a['history_h_before'], b['history_h_before'])
    assert np.array_equal(a['history_x'], b['history_x'])
    assert np.array_equal(a['history_pos'], b['history_pos'])
    assert a['prefix_sha256'] == b['prefix_sha256']
