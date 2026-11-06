"""Independent derivatives and nuisance-identifiability checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_predictive_fraction_information import (
    GridPopulation, rates_and_position_jacobian, information_from_blocks,
    information_from_jacobian, homogeneous_information, alias_diagnostic)


class FractionInformationTests(unittest.TestCase):
    def test_homogeneous_schur_formula_including_anisotropic_geometry(self):
        A = np.array([[17., 3.], [3., 5.]])
        perpendicular = np.array([-.8, .6])
        for fraction in np.linspace(0, 1, 11):
            raw, adjusted, _ = information_from_blocks((1-fraction)*A, fraction*A, perpendicular, .2)
            expected_raw, expected_adjusted = homogeneous_information(A, fraction, perpendicular, .2)
            np.testing.assert_allclose(raw, expected_raw, atol=1e-14)
            np.testing.assert_allclose(adjusted, expected_adjusted, atol=1e-14)

    def test_rate_jacobian_agrees_with_original_torch_autograd(self):
        population = GridPopulation(n_cells=64, pgc_frac=.33, seed=3)
        positions = torch.tensor([[.27, -.61], [-.39, .24]], requires_grad=True)
        angle = torch.tensor([.81, -1.17], requires_grad=True)
        direction = torch.stack((angle.cos(), angle.sin()), -1)
        rates = population.activity(positions, direction)
        analytic_rates, spatial = rates_and_position_jacobian(population, positions.detach().numpy(), direction.detach().numpy())
        np.testing.assert_allclose(analytic_rates, rates.detach().numpy(), atol=2e-6, rtol=0)
        perpendicular = torch.stack((-angle.sin(), angle.cos()), -1).detach().numpy()
        angular = (spatial*perpendicular[:, None, :]).sum(-1)*population.shift.numpy()
        spatial_auto, angular_auto = [], []
        for cell in range(population.n_cells):
            dx, dtheta = torch.autograd.grad(rates[:, cell].sum(), (positions, angle), retain_graph=True)
            spatial_auto.append(dx.numpy())
            angular_auto.append(dtheta.numpy())
        np.testing.assert_allclose(spatial, np.stack(spatial_auto, 1), atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose(angular, np.stack(angular_auto, 1), atol=6e-6, rtol=1e-5)

    def test_actual_schur_matches_independent_score_projection(self):
        rng = np.random.default_rng(31)
        position = rng.uniform(-1, 1, (9, 2))
        angle = rng.uniform(-np.pi, np.pi, 9)
        direction = np.column_stack((np.cos(angle), np.sin(angle)))
        perpendicular = np.column_stack((-np.sin(angle), np.cos(angle)))
        for fraction in (0., .05, .33, .5, .95, 1.):
            population = GridPopulation(n_cells=128, pgc_frac=fraction, seed=2)
            _, spatial = rates_and_position_jacobian(population, position, direction)
            mask = population.is_pgc.numpy()
            noise_sd = .13
            ordinary = np.einsum("bni,bnj->bij", spatial[:, ~mask], spatial[:, ~mask])/noise_sd**2
            predictive = np.einsum("bni,bnj->bij", spatial[:, mask], spatial[:, mask])/noise_sd**2
            d = float(np.float32(.2))
            angular = d*(spatial*perpendicular[:, None]).sum(-1)*mask
            raw, adjusted, _ = information_from_blocks(ordinary, predictive, perpendicular, d)
            score_raw, score_adjusted = information_from_jacobian(spatial, angular, noise_sd)
            np.testing.assert_allclose(raw, score_raw, atol=1e-10, rtol=1e-12)
            np.testing.assert_allclose(adjusted, score_adjusted, atol=1e-10, rtol=1e-12)
            if fraction in (0., 1.):
                self.assertLess(float(adjusted.max()), 1e-10)
            else:
                self.assertGreater(float(adjusted.min()), 0.)

    def test_all_predictive_has_exact_position_direction_gauge(self):
        population = GridPopulation(n_cells=512, pgc_frac=1., seed=4)
        position = torch.tensor([[.21, -.48]])
        u = torch.tensor([[.6, .8]])
        alternative_u = torch.tensor([[-1., 0.]])
        alternative_x = position + float(population.shift[0])*(u-alternative_u)
        rates0 = population.activity(position, u)
        rates1 = population.activity(alternative_x, alternative_u)
        torch.testing.assert_close(rates0, rates1, atol=3e-6, rtol=0)
        # Position knowledge breaks the same confounding: changing direction
        # with position fixed changes predictive rates substantially.
        fixed_x = population.activity(position, alternative_u)
        self.assertGreater(float((rates0-fixed_x).abs().max()), .5)

    def test_position_prior_can_restore_information_to_all_predictive(self):
        A = np.array([[20., 3.], [3., 10.]])
        perpendicular, d = np.array([0., 1.]), .2
        coupling, raw = d*A@perpendicular, d*d*perpendicular@A@perpendicular
        values = [raw-coupling@np.linalg.inv(A+p*np.eye(2))@coupling for p in (0., 1., 100., 1e8)]
        self.assertAlmostEqual(values[0], 0.)
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertAlmostEqual(values[-1], raw, places=6)

    def test_single_module_can_alias_opposite_directions_despite_local_information(self):
        diagnostic = alias_diagnostic(5)
        self.assertLess(diagnostic["finest_module_predictive_max_rate_difference"], 5e-6)
        self.assertGreater(diagnostic["coarser_modules_predictive_max_rate_difference"], .2)
        population = GridPopulation(n_cells=512, pgc_frac=.33, seed=5)
        theta = population.orientation[0]+np.pi/6
        direction = np.array([[np.cos(theta), np.sin(theta)]])
        perpendicular = np.array([[-np.sin(theta), np.cos(theta)]])
        _, gradient = rates_and_position_jacobian(population, [[.17, -.31]], direction)
        selected = population.module == 0
        angular = np.sum(gradient*perpendicular[:, None], -1)*population.shift.numpy()
        _, adjusted = information_from_jacobian(gradient[:, selected], angular[:, selected])
        self.assertGreater(float(adjusted[0]), 0.)


if __name__ == "__main__":
    unittest.main()
