"""Independent coordinate and derivative checks for cue-conditioned spectra."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_heading_cue_spectra as spectra
from evaluate_ring_dynamics import jacobian


class SmallRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.input_size = 4, 3
        self.config = SimpleNamespace(leak=.25)
        self.input = torch.nn.Linear(3, 4)
        self.recurrent = torch.nn.Linear(4, 4, bias=False)
        self.readout = torch.nn.Linear(4, 2)

    def step(self, x, h):
        return .75 * h + .25 * torch.tanh(self.input(x) + self.recurrent(h))


class CueSpectraAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9103)
        self.model = SmallRNN().double().requires_grad_(False)
        self.state = torch.randn(12, 4, dtype=torch.float64)
        self.x = torch.randn(12, 3, dtype=torch.float64)
        tangent = np.random.default_rng(31).normal(size=(12, 4))
        self.tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)

    def test_affine_phase_gradient_includes_readout_bias(self):
        with torch.no_grad():
            self.model.readout.bias.copy_(torch.tensor([1.2, -.7]))
        state = self.state.clone().requires_grad_(True)
        y = self.model.readout(state)
        exact = torch.autograd.grad(torch.atan2(y[:, 1], y[:, 0]).sum(), state)[0]
        torch.testing.assert_close(spectra.phase_gradient(self.model, state), exact, rtol=1e-12, atol=1e-12)

    def test_jacobian_holds_input_fixed(self):
        analytic = jacobian(self.model, self.state, self.x)
        for h, x, j in zip(self.state, self.x, analytic):
            independent = torch.autograd.functional.jacobian(lambda s: self.model.step(x, s), h)
            torch.testing.assert_close(j, independent, rtol=1e-12, atol=1e-12)
        self.model.recurrent.weight.zero_()
        expected = .75 * torch.eye(4, dtype=torch.float64)
        torch.testing.assert_close(jacobian(self.model, self.state, self.x), expected.expand(12, -1, -1), rtol=0, atol=0)

    def test_phase_gain_matches_circular_finite_differences(self):
        report, _ = spectra.one_state_spectrum(self.model, self.state, self.x, self.tangent)
        tangent = torch.as_tensor(self.tangent)
        def phase_delta(a, b):
            ya, yb = self.model.readout(a), self.model.readout(b)
            return torch.atan2(ya[:, 0] * yb[:, 1] - ya[:, 1] * yb[:, 0], (ya * yb).sum(-1))
        eps = 1e-5
        a, b = self.state - eps * tangent, self.state + eps * tangent
        finite = phase_delta(self.model.step(self.x, a), self.model.step(self.x, b)) / phase_delta(a, b)
        np.testing.assert_allclose([r['phase_gain'] for r in report['per_phase']], finite, rtol=1e-6, atol=1e-6)

    def test_phase_gain_is_invariant_to_tangent_scale_and_sign(self):
        base, _ = spectra.one_state_spectrum(self.model, self.state, self.x, self.tangent)
        for scale in (2.7, -3.1):
            other, _ = spectra.one_state_spectrum(self.model, self.state, self.x, scale * self.tangent)
            np.testing.assert_allclose([r['phase_gain'] for r in base['per_phase']],
                                       [r['phase_gain'] for r in other['per_phase']], atol=1e-12)

    def test_phase_gradient_is_invariant_to_readout_rotation_and_scaling(self):
        transformed = deepcopy(self.model)
        angle, scale = .73, 3.8
        rotation = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=torch.float64)
        transformed.readout.weight.copy_(scale * rotation @ self.model.readout.weight)
        transformed.readout.bias.copy_(scale * rotation @ self.model.readout.bias)
        for h in (self.state, self.model.step(self.x, self.state)):
            torch.testing.assert_close(spectra.phase_gradient(self.model, h), spectra.phase_gradient(transformed, h), rtol=1e-12, atol=1e-12)

    def test_input_changes_jacobian_through_tanh_gain_only(self):
        other_x = self.x + .7
        delta = jacobian(self.model, self.state, other_x) - jacobian(self.model, self.state, self.x)
        pre1 = self.model.input(other_x) + self.model.recurrent(self.state)
        pre0 = self.model.input(self.x) + self.model.recurrent(self.state)
        predicted = .25 * (pre0.tanh().square() - pre1.tanh().square())[:, :, None] * self.model.recurrent.weight
        torch.testing.assert_close(delta, predicted, rtol=1e-12, atol=1e-12)

    def test_known_radial_and_phase_contraction_are_recovered(self):
        theta = torch.arange(16, dtype=torch.float64) * 2 * np.pi / 16
        state = torch.stack((theta.cos(), theta.sin()), -1)
        tangent = torch.stack((-theta.sin(), theta.cos()), -1)
        class Instrument(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.readout = torch.nn.Linear(2, 2).double()
                with torch.no_grad():
                    self.readout.weight.copy_(torch.eye(2)); self.readout.bias.zero_()
            def step(self, x, h):
                radial = h.norm(dim=-1)
                phase = torch.atan2(h[:, 1], h[:, 0])
                delta = torch.atan2(torch.sin(x[:, 0] - phase), torch.cos(x[:, 0] - phase))
                phase = phase + .2 * delta
                return (1 + .5 * (radial - 1))[:, None] * torch.stack((phase.cos(), phase.sin()), -1)
        instrument = Instrument().requires_grad_(False)
        known = .5 * state[:, :, None] * state[:, None, :] + .8 * tangent[:, :, None] * tangent[:, None, :]
        with patch.object(spectra, 'jacobian', return_value=known):
            report, _ = spectra.one_state_spectrum(instrument, state, theta[:, None], tangent.numpy())
        for row in report['per_phase']:
            self.assertAlmostEqual(row['phase_gain'], .8, places=12)
            self.assertAlmostEqual(row['spectral_radius'], .8, places=12)
            self.assertAlmostEqual(row['normal_largest_singular_value'], .5, places=12)


if __name__ == '__main__':
    unittest.main()
