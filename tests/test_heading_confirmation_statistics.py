from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from summarize_heading_confirmation import paired_summary, holm


class ConfirmationStatisticsTests(unittest.TestCase):
    def test_exact_extreme_probability_and_relabeling(self):
        values = np.arange(1, 17, dtype=float)
        result = paired_summary(values)
        self.assertEqual(result["p_exact_sign_flip"], 2/65536)
        opposite = paired_summary(-values)
        self.assertEqual(opposite["p_exact_sign_flip"], result["p_exact_sign_flip"])
        self.assertEqual(opposite["ci95"], [-result["ci95"][1], -result["ci95"][0]])

    def test_zero_effect_and_incomplete_cohort(self):
        self.assertEqual(paired_summary(np.zeros(16))["p_exact_sign_flip"], 1.)
        self.assertEqual(paired_summary(np.tile([-1., 1.], 8))["p_exact_sign_flip"], 1.)
        with self.assertRaises(ValueError):
            paired_summary(np.ones(15))

    def test_holm_monotonicity_and_original_order(self):
        np.testing.assert_allclose(holm([.04, .001, .03, .2]), [.09, .004, .09, .2])


if __name__ == "__main__":
    unittest.main()
