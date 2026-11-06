"""Causality, information, and full-state stability tests of the engineered model."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from train_sensorimotor_ring import (Config, build_decoders, cue_readouts, correction,
    memory_step, simulate, experience, controller_candidates, local_loop_spectrum,
    delay_state_step, validate_full_jacobian, load_fitted_decoders, wrap)


class SensorimotorRingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(decoder_examples=1024)
        cls.ordinary, cls.forecast, cls.basis = build_decoders(cls.config, 491)

    def test_actual_predictive_codes_equal_compiled_encoder_decoder(self):
        rng = np.random.default_rng(92)
        phase, velocity = rng.uniform(-10, 10, 30), rng.uniform(-3, 3, 30)
        base = self.ordinary.encode(phase)
        mixed = base.copy()
        mask = self.basis["predictive_mask"]
        mixed[:, mask] = self.forecast.encode(phase+self.config.code_horizon*velocity)[:, mask]
        a = self.ordinary.from_code(mixed)
        b = self.forecast.from_code(mixed)
        ra, rb = cue_readouts(self.ordinary, self.forecast, phase, velocity, "predictive", .2)
        np.testing.assert_allclose(wrap(np.arctan2(a[:, 1], a[:, 0])-ra), 0, atol=1e-12)
        np.testing.assert_allclose(wrap(np.arctan2(b[:, 1], b[:, 0])-rb), 0, atol=1e-12)
        np.testing.assert_array_equal(mixed[:, ~mask], base[:, ~mask])

    def test_predictive_controller_has_no_direct_velocity_or_error_derivative(self):
        a, b = np.array([.3, .6]), np.array([.4, .7])
        first = correction(a, b, np.array([1., -2]), 0., 4., .5, "predictive", .2)
        changed = correction(a, b, np.array([100., 200]), 0., 4., .5, "predictive", .2)
        np.testing.assert_array_equal(first, changed)
        self.assertFalse(np.array_equal(first, correction(a, b, np.array([100., 200]),
                                                          0., 4., .5, "velocity", .2)))

    def test_memory_ring_has_neutral_phase_and_contracting_radius(self):
        phase = np.linspace(-3, 3, 16)
        initial = 1.1*np.column_stack((np.cos(phase), np.sin(phase)))
        state = initial.copy()
        for _ in range(100):
            state = memory_step(state, .02, 1.)
        np.testing.assert_allclose(wrap(np.arctan2(state[:, 1], state[:, 0])-phase), 0, atol=1e-14)
        self.assertLess(np.max(np.abs(np.linalg.norm(state, axis=1)-1)), .002)
        eps = 1e-6
        point = np.array([1., 0.])
        jac = np.column_stack([(memory_step(point+eps*d, .02, 1)-memory_step(point-eps*d, .02, 1))/(2*eps)
                               for d in np.eye(2)])
        np.testing.assert_allclose(np.linalg.eigvals(jac), [.96, 1.], atol=1e-10)

    def test_full_state_eigenvalues_match_delay_polynomial(self):
        for delay in (0, 3, 15):
            cfg = Config(**{**asdict(self.config), "delay_steps": delay})
            for condition in ("grid", "predictive", "velocity", "removed", "reversed"):
                check = validate_full_jacobian(cfg, self.ordinary, self.forecast, condition, 4.5, .625)
                self.assertLess(check["full_state_jacobian_spectrum_error"], 2e-7)
                self.assertLess(check["neutral_eigenvalue_error"], 2e-9)

    def test_independent_delay_map_matches_simulator_time_alignment(self):
        for delay in (0, 3):
            cfg = Config(**{**asdict(self.config), "delay_steps": delay, "duration": .3,
                            "phase_noise": 0., "velocity_noise": 0., "drift_noise": 0.})
            data = experience(cfg, 1, 82)
            data["goal"][:] = .1
            data["initial"][:] = .3
            goal = self.ordinary.angle(.1)
            state = np.r_[np.cos(goal), np.sin(goal), np.full(delay+2, .3)]
            for condition in ("grid", "predictive", "velocity"):
                result = simulate(cfg, self.ordinary, self.forecast, condition, [2.5], [.5], data, record=True)
                actual = state.copy()
                errors = []
                for _ in range(len(data["drift"])):
                    actual = delay_state_step(cfg, self.ordinary, self.forecast, condition, 2.5, .5, actual)
                    errors.append(wrap(actual[2]-.1))
                np.testing.assert_allclose(result["error_trace"][:, 0, 0], errors, atol=2e-8)

    def test_no_future_noise_leak_and_exact_restoration(self):
        cfg = Config(**{**asdict(self.config), "duration": 1.})
        data = experience(cfg, 8, 83)
        changed = {key: value.copy() for key, value in data.items()}
        changed["phase_noise"][23:] += .7
        changed["velocity_noise"][23:] -= 1.
        changed["drift"][23:] += .2
        for condition in ("grid", "predictive", "velocity"):
            a = simulate(cfg, self.ordinary, self.forecast, condition, [4.5], [.5], data, record=True)
            b = simulate(cfg, self.ordinary, self.forecast, condition, [4.5], [.5], changed, record=True)
            np.testing.assert_array_equal(a["error_trace"][:23], b["error_trace"][:23])
        native = simulate(cfg, self.ordinary, self.forecast, "predictive", [4.5], [.5], data, record=True)
        restored = simulate(cfg, self.ordinary, self.forecast, "restored", [4.5], [.5], data, record=True)
        np.testing.assert_array_equal(native["error_trace"], restored["error_trace"])

    def test_training_search_budgets_are_equal(self):
        for condition in ("grid", "predictive", "velocity"):
            gains, mixtures = controller_candidates(condition)
            self.assertEqual(len(gains), 108)
            self.assertEqual(len(mixtures), 108)
            self.assertAlmostEqual(gains.min(), .4)
            self.assertAlmostEqual(gains.max(), 8.)
        self.assertEqual(len(np.unique(controller_candidates("grid")[0])), 108)

    def test_saved_decoder_roundtrip_never_refits(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.npz"
            np.savez(path, **self.basis, ordinary_weights=self.ordinary.weights,
                     forecast_weights=self.forecast.weights, ordinary_cosine=self.ordinary.cosine,
                     ordinary_sine=self.ordinary.sine, forecast_cosine=self.forecast.cosine,
                     forecast_sine=self.forecast.sine)
            loaded = load_fitted_decoders(path)
        phase = np.linspace(-20, 20, 83)
        for original, restored in zip((self.ordinary, self.forecast), loaded):
            np.testing.assert_array_equal(original.compiled(phase), restored.compiled(phase))
            np.testing.assert_array_equal(original.from_code(original.encode(phase)),
                                          restored.from_code(restored.encode(phase)))


if __name__ == "__main__":
    unittest.main()
