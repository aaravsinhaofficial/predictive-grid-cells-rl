import math
import numpy as np
import torch
from scripts import lstm_reorientation_causal_navigation as causal


CFG = dict(arena_half=1.25, action_repeat=4, v_max=.35, rot_max=2.2, goal_radius=.3)


def test_radial_kick_is_transverse_and_does_not_collapse_into_random_plane_test():
    theta = np.arange(24)*2*np.pi/24
    curve = np.zeros((24, 512)); curve[:, 0] = np.cos(theta); curve[:, 1] = np.sin(theta)
    edits, audit = causal.state_edits(curve, np.ones(512), np.arange(24))
    np.testing.assert_allclose(edits['plus30'], np.roll(curve, -2, axis=0))
    np.testing.assert_allclose(np.linalg.norm(edits['radial_plus']-curve, axis=1), .02)
    np.testing.assert_allclose(np.linalg.norm(edits['isotropic_plus']-curve, axis=1), .02)
    assert audit['maximum_radial_tangent_dot'] < 1e-12
    # A radial kick of this planar circle stays in its two-dimensional plane;
    # isotropic noise mostly misses that plane, the critical confound controlled.
    assert np.all(edits['radial_plus'][:, 2:] == 0)
    assert np.linalg.norm(edits['isotropic_plus'][:, 2:]) > .09


def test_collapsed_family_retains_unperturbed_cue_experiment_and_marks_phase_invalid():
    curve = np.ones((24, 512))
    edits, audit = causal.state_edits(curve, np.ones(512), np.arange(24))
    assert list(edits) == ['none'] and len(audit['invalid_edits']) == 6
    theta = np.arange(24)*2*np.pi/24
    metrics = causal.phase_metrics(curve, curve, curve, theta, theta, np.ones(512), 0.)
    assert metrics['phase_mae_deg'] is None and metrics['own_branch_phase_mae_deg'] is None


class Actor:
    def pi(self, h):
        logits = torch.full((len(h), 6), -100., dtype=h.dtype)
        action = torch.where(h[:, 0] > .5, 2, 3)
        logits[torch.arange(len(h)), action] = 100.
        return logits


def test_first_free_action_uses_saved_state_and_noise_is_exactly_matched(monkeypatch):
    calls = []
    def transition(net, state, inp):
        calls.append(inp.clone())
        result = state.clone(); result[:, 0] = 0
        return result
    monkeypatch.setattr(causal, 'step', transition)
    start = np.zeros((2, 512)); start[:, 0] = 1
    dark = torch.zeros(1, 1287, dtype=torch.float64)
    motor, uniform = causal.navigation_draws(2)
    args = (Actor(), start, dark, CFG, np.array([0., np.pi]), np.zeros(2), np.array([1., 0.]), motor, uniform)
    summary, paths = causal.free_navigation(*args)
    assert len(calls) == 29 and summary['processed_inputs_after_first_free_action'] == 29
    np.testing.assert_array_equal(paths['actions'][0], [2, 2])
    np.testing.assert_array_equal(paths['actions'][1], [3, 3])
    assert torch.equal(calls[0][:, 1280:1286].argmax(1), torch.tensor([2, 2]))
    _, repeated = causal.free_navigation(*args)
    for key in paths: np.testing.assert_array_equal(paths[key], repeated[key])


def test_first_hit_stops_scoring_before_native_respawn(monkeypatch):
    monkeypatch.setattr(causal, 'step', lambda net, state, inp:state)
    start = np.zeros((1, 512)); start[:, 0] = 1
    dark = torch.zeros(1, 1287, dtype=torch.float64)
    motor = torch.zeros(30, 3, 1); uniform = torch.full((30, 1), .5)
    env = causal.physical_free_start(CFG, np.array([0.]), np.zeros(2), np.array([1., 0.]))
    first_heading = float(env.hd[0]+env.vr[0]*math.exp(-env.dt/env.tau)*env.dt)
    goal = .301*np.array([math.cos(first_heading), math.sin(first_heading)])
    summary, paths = causal.free_navigation(Actor(), start, dark, CFG, np.array([0.]),
        np.zeros(2), goal, motor, uniform)
    assert summary['goals'] == 1 and paths['first_hit_step'][0] == 1
    assert np.all(paths['actions'][1:] == -1)
    np.testing.assert_array_equal(paths['path'][2:], np.repeat(paths['path'][1:2], 29, axis=0))
    assert paths['path_length_m'][0] < .01
