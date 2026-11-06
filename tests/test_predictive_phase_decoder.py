"""Input-only and geometric validation of the original-code compass decoder."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from decode_predictive_phase_difference import PhaseDifferenceDecoder
from ring_heading_model import HeadingConfig, HeadingTask
from codes import GridPopulation


class PredictiveCompassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.population = GridPopulation(n_cells=512, pgc_frac=.33, seed=13)
        gen = np.random.default_rng(291)
        cls.positions = gen.uniform(-1.25, 1.25, (1024, 2)).astype(np.float32)
        angle = gen.uniform(-np.pi, np.pi, 1024)
        cls.directions = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
        rates = cls.population.activity(torch.from_numpy(cls.positions), torch.from_numpy(cls.directions)).numpy()
        cls.decoder = PhaseDifferenceDecoder(cls.population).fit(rates, cls.positions, cls.directions)

    def test_prediction_uses_only_rates_after_calibration(self):
        gen = np.random.default_rng(91)
        position = gen.uniform(8, 12, (128, 2)).astype(np.float32)
        angle = gen.uniform(-np.pi, np.pi, 128)
        direction = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
        rates = self.population.activity(torch.from_numpy(position), torch.from_numpy(direction)).numpy()
        expected = self.decoder.predict(rates)
        # Remove the population object entirely; inference cannot query physical
        # state, activity(), K, or calibration labels through that object.
        previous = self.decoder.pop
        self.decoder.pop = object()
        try:
            np.testing.assert_array_equal(self.decoder.predict(rates), expected)
            order = gen.permutation(len(rates))
            np.testing.assert_allclose(self.decoder.predict(rates[order]), expected[order], atol=1e-12)
        finally:
            self.decoder.pop = previous
        np.testing.assert_allclose(expected, direction, atol=3e-5, rtol=0)

    def test_shift_magnitude_and_direction_follow_code_without_heading_input(self):
        position = np.tile(np.array([[.37, -.81]], dtype=np.float32), (72, 1))
        angle = np.arange(72)*2*np.pi/72
        direction = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
        for scale in (1., -.75, .5, 0.):
            rates = self.population.activity(torch.from_numpy(position), torch.from_numpy(direction),
                                              shift_scale=scale).numpy()
            decoded = self.decoder.predict(rates)
            np.testing.assert_allclose(decoded, scale*direction, atol=8e-6, rtol=0)

    def test_aliasing_module_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "aliases"):
            PhaseDifferenceDecoder(self.population, modules=(0,))
        self.assertEqual(np.linalg.matrix_rank(self.decoder.geometry), 2)
        self.assertLess(np.max(np.linalg.norm(self.decoder.geometry, axis=1)), np.pi)

    def test_zero_shift_codes_are_direction_independent(self):
        position = torch.tensor([[.2, -.4]]).expand(32, 2)
        angle = torch.arange(32)*2*np.pi/32
        direction = torch.stack((angle.cos(), angle.sin()), -1)
        rates = self.population.activity(position, direction, shift_scale=0).numpy()
        np.testing.assert_array_equal(rates, np.repeat(rates[:1], 32, axis=0))
        self.assertLess(np.linalg.norm(self.decoder.predict(rates), axis=1).max(), 1e-5)

    def test_heading_anchor_blank_has_no_implicit_direction_channel(self):
        task = HeadingTask(HeadingConfig(n_code=128, code_seed=13))
        phase = torch.arange(36)*2*np.pi/36
        direction = torch.stack((phase.cos(), phase.sin()), -1)
        position = torch.ones(36, 2)*.4
        zero, one = torch.zeros(36), torch.ones(36)
        blank = task.inputs(position, direction, zero, zero, zero, zero, "predictive")
        self.assertTrue(torch.equal(blank, torch.zeros_like(blank)))
        ordinary = task.inputs(position, direction, zero, zero, one, zero, "grid")
        self.assertTrue(torch.equal(ordinary, ordinary[:1].expand_as(ordinary)))
        predictive = task.inputs(position, direction, zero, zero, one, zero, "predictive")
        self.assertFalse(torch.equal(predictive, predictive[:1].expand_as(predictive)))
        self.assertTrue(torch.equal(predictive[:, -3:], torch.zeros_like(predictive[:, -3:])))


if __name__ == "__main__":
    unittest.main()
