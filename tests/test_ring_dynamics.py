"""Independent scientific checks for ring geometry, dynamics and information.

The analytic systems here include a ring, a cue-driven circle that collapses,
and a discrete attractor set. Passing these tests does not classify a trained
network; it validates the measurements and exposes important false positives.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ring_dynamics import (perturbations, project_to_ring, ring_radius, tangents,
                           trajectory_measurements, validate_metrics, wrap)
from ring_task_model import RingConfig, RingRNN, RingTask
from evaluate_ring_dynamics import jacobian
from ring_mechanism_theory import (cue_force, phase_lock, normal_form_spectrum,
                                   normal_form_field, feedback_delay_limit, characteristic_root)


class RingGeometryTests(unittest.TestCase):
    def test_closed_polyline_projection_has_known_distance_and_phase(self):
        square = np.array([[1., 0], [0, 1], [-1, 0], [0, -1]])
        points = np.array([[1., 1], [1., -1], [-1., -1]])
        distance, phase = project_to_ring(points, square)
        np.testing.assert_allclose(distance, np.sqrt(.5), atol=1e-14)
        np.testing.assert_allclose(phase, [np.pi/4, 7*np.pi/4, 5*np.pi/4], atol=1e-14)

    def test_empirical_normals_and_tangents_have_equal_norm(self):
        angles = np.arange(128) * 2*np.pi/128
        # Curve embedded into 11 dimensions under a generic orthogonal rotation.
        rotation = np.linalg.qr(np.random.default_rng(31).normal(size=(11, 11)))[0]
        curve = np.column_stack((2*np.cos(angles), 2*np.sin(angles),
                                 np.zeros((128, 9)))) @ rotation
        t, n = perturbations(curve, relative_size=.13, seed=71)
        unit = tangents(curve)
        np.testing.assert_allclose(np.sum(n*unit, axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(t, axis=1), .26, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), .26, atol=1e-14)
        np.testing.assert_allclose(ring_radius(curve), 2., atol=1e-14)

    def test_collapsed_initial_curve_is_rejected(self):
        curve = np.ones((64, 5))
        with self.assertRaises(ValueError):
            perturbations(curve)
        with self.assertRaises(ValueError):
            trajectory_measurements(curve[None], curve[None], curve[None])

    def test_shared_spontaneous_motion_is_not_perturbation_recovery(self):
        phase = np.arange(64)*2*np.pi/64
        curve = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(64)))
        paths = np.stack([curve + [0, 0, 2*t] for t in range(8)])
        rows = trajectory_measurements(paths, paths, paths)
        for row in rows:
            self.assertLess(row["normal_distance_radius"], 1e-12)
            self.assertLess(row["normal_phase_displacement"], 1e-12)
            self.assertAlmostEqual(row["radius_ratio"], 1)

    def test_ring_retains_tangent_memory_but_driven_circle_collapses(self):
        result = validate_metrics()
        ring = result["positive_ring"][-1]
        point = result["driven_circle_after_cue_removal"][-1]
        self.assertAlmostEqual(ring["radius_ratio"], 1)
        self.assertGreater(ring["tangent_phase_displacement"], .04)
        self.assertLess(ring["normal_pair_distance_radius"], .001)
        self.assertLess(point["radius_ratio"], .002)

    def test_four_point_attractor_set_passes_radius_but_is_not_a_ring(self):
        # A required negative control: nonzero radius alone cannot identify a
        # continuous attractor. The phase map here collapses to four plateaus.
        phase = (np.arange(128)+.173)*2*np.pi/128
        initial = np.column_stack((np.cos(phase), np.sin(phase)))
        final_phase = np.round(phase/(np.pi/2))*(np.pi/2)
        final = np.column_stack((np.cos(final_phase), np.sin(final_phase)))
        self.assertAlmostEqual(ring_radius(final)/ring_radius(initial), 1)
        self.assertEqual(len(np.unique(np.round(final, 10), axis=0)), 4)
        unique_phase = np.unique(np.round(np.mod(final_phase, 2*np.pi), 10))
        coverage = 1 - np.diff(np.r_[unique_phase, unique_phase[0]+2*np.pi]).max()/(2*np.pi)
        self.assertAlmostEqual(coverage, .75, places=9)


class RingJacobianTests(unittest.TestCase):
    def test_analytic_jacobian_matches_autograd_away_from_zero(self):
        torch.manual_seed(318)
        model = RingRNN(RingConfig(hidden_size=9, n_code=8, leak=.37)).double()
        states = torch.randn(4, 9, dtype=torch.float64)*.7
        inputs = torch.randn(4, model.input_size, dtype=torch.float64)*.8
        actual = jacobian(model, states, inputs)
        for i in range(4):
            expected = torch.autograd.functional.jacobian(lambda h: model.step(inputs[i], h), states[i])
            torch.testing.assert_close(actual[i], expected, atol=2e-15, rtol=2e-15)
        # Independent central differences catch shared mistakes in the formulas.
        direction = torch.randn(9, dtype=torch.float64)
        epsilon = 1e-6
        finite = (model.step(inputs[0], states[0]+epsilon*direction)
                  - model.step(inputs[0], states[0]-epsilon*direction))/(2*epsilon)
        torch.testing.assert_close(actual[0]@direction, finite, atol=1e-9, rtol=1e-9)

    def test_known_local_neutral_and_contracting_spectrum(self):
        model = RingRNN(RingConfig(hidden_size=3, n_code=4, leak=.25)).double()
        with torch.no_grad():
            model.input.weight.zero_()
            model.input.bias.zero_()
            model.recurrent.weight.copy_(torch.diag(torch.tensor([1., .6, -.2], dtype=torch.float64)))
        j = jacobian(model, torch.zeros(1, 3, dtype=torch.float64))[0].detach().numpy()
        np.testing.assert_allclose(np.linalg.eigvalsh(j), [.7, .9, 1.], atol=1e-15)

    def test_projected_normal_eigenvalue_alone_can_hide_instability(self):
        # Tangent/normal feedback is present. Compression to normal space is
        # stable even though the complete state dynamics are unstable.
        j = np.array([[1., 10.], [.2, .8]])
        q = np.array([[0.], [1.]])
        self.assertLess(np.abs(np.linalg.eigvals(q.T@j@q)).max(), 1)
        self.assertGreater(np.abs(np.linalg.eigvals(j)).max(), 2)


class RingInformationTests(unittest.TestCase):
    def test_all_conditions_share_realization_and_nonpredictive_channels(self):
        task = RingTask(RingConfig(code_seed=17))
        data = task.realization(80, 12, torch.Generator().manual_seed(41))
        original = {key: value.clone() for key, value in data.items()}
        inputs = {condition: task.batch(80, 12, None, condition, data)[0]
                  for condition in ("grid", "predictive", "reverse", "shuffled")}
        for condition, x in inputs.items():
            torch.testing.assert_close(x[..., -2:], inputs["grid"][..., -2:], rtol=0, atol=0)
            torch.testing.assert_close(x[..., :task.config.n_code][..., ~task.predictive_mask],
                inputs["grid"][..., :task.config.n_code][..., ~task.predictive_mask], rtol=0, atol=0)
        for key, value in original.items():
            torch.testing.assert_close(data[key], value, rtol=0, atol=0)
        torch.testing.assert_close(data["shuffled_velocity"].sort(dim=1).values,
                                   data["delayed_velocity"].sort(dim=1).values, rtol=0, atol=0)

    def test_forecast_exactly_compensates_matched_delay_for_constant_velocity(self):
        config = RingConfig(code_seed=18, dt=.05, horizon=.2, delay_steps=4)
        task = RingTask(config)
        length, velocity = 32, .7
        theta = torch.arange(length)*config.dt*velocity
        indices = (torch.arange(length)-config.delay_steps).clamp_min(0)
        v = torch.full_like(theta, velocity)
        forecast = task.encode(theta[indices], v, "predictive")
        now = task.encode(theta, v, "grid")
        old = task.encode(theta[indices], v, "grid")
        torch.testing.assert_close(forecast[4:, task.predictive_mask], now[4:, task.predictive_mask],
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(forecast[:, ~task.predictive_mask], old[:, ~task.predictive_mask],
                                   atol=0, rtol=0)

    def test_future_measurement_changes_do_not_change_past_input(self):
        task = RingTask(RingConfig(delay_steps=4))
        data = task.realization(64, 8, torch.Generator().manual_seed(83))
        changed = {key: value.clone() for key, value in data.items()}
        cutoff = 23
        changed["observed_phase"][cutoff:] += 1.3
        changed["observed_velocity"][cutoff:] -= .8
        indices = changed["observation_index"]
        changed["delayed_phase"] = changed["observed_phase"][indices]
        changed["delayed_velocity"] = changed["observed_velocity"][indices]
        changed["shuffled_velocity"] = changed["delayed_velocity"].roll(4, dims=1)
        for condition in ("grid", "predictive", "reverse", "shuffled"):
            a = task.batch(64, 8, None, condition, data)[0]
            b = task.batch(64, 8, None, condition, changed)[0]
            torch.testing.assert_close(a[:cutoff], b[:cutoff], atol=0, rtol=0)

    def test_completed_motion_timestep_alignment(self):
        config = RingConfig(dt=.05, velocity_noise=0, position_noise=0)
        task = RingTask(config)
        data = task.realization(64, 8, torch.Generator().manual_seed(83), stationary_tail=False)
        delta = torch.atan2((data["theta"][1:]-data["theta"][:-1]).sin(),
                            (data["theta"][1:]-data["theta"][:-1]).cos())
        torch.testing.assert_close(delta, config.dt*data["velocity"][1:], atol=5e-7, rtol=1e-5)
        torch.testing.assert_close(data["observed_velocity"], data["velocity"], atol=0, rtol=0)


class RingMechanismTheoryTests(unittest.TestCase):
    def test_single_cue_phase_lead_changes_bias_but_not_stability(self):
        ordinary = cue_force(1.5, .2, 0, predictive_weight=1, gain=2)
        predictive = cue_force(1.5, .2, .2, predictive_weight=1, gain=2)
        a, b = phase_lock(ordinary), phase_lock(predictive)
        self.assertAlmostEqual(float(a["error"]), -.3)
        self.assertAlmostEqual(float(b["error"]), 0)
        self.assertAlmostEqual(float(a["exponent"]), float(b["exponent"]))
        sa, sb = normal_form_spectrum(ordinary), normal_form_spectrum(predictive)
        self.assertAlmostEqual(sa["radial_exponent"], sb["radial_exponent"])
        self.assertAlmostEqual(sa["tangent_exponent"], sb["tangent_exponent"])

    def test_plain_cue_can_contract_more_and_track_worse(self):
        ordinary = cue_force(2., .2, 0, predictive_weight=.5, gain=2)
        predictive = cue_force(2., .2, .2, predictive_weight=.5, gain=2)
        reverse = cue_force(2., .2, -.2, predictive_weight=.5, gain=2)
        self.assertGreater(abs(ordinary), abs(predictive))
        self.assertLess(abs(np.angle(predictive)), abs(np.angle(ordinary)))
        self.assertAlmostEqual(abs(predictive), abs(reverse))
        self.assertGreater(abs(np.angle(reverse)), abs(np.angle(ordinary)))
        self.assertLess(normal_form_spectrum(ordinary)["radial_exponent"],
                        normal_form_spectrum(predictive)["radial_exponent"])

    def test_normal_form_spectrum_matches_numerical_jacobian(self):
        force = cue_force(1.5, .2, .2, predictive_weight=.5, gain=2)
        spectrum = normal_form_spectrum(force)
        angle, radius = spectrum["equilibrium_angle"], spectrum["radius"]
        state = radius*np.array([np.cos(angle), np.sin(angle)])
        np.testing.assert_allclose(normal_form_field(state, force), 0, atol=1e-12)
        eps = 1e-6
        j = np.column_stack([(normal_form_field(state+eps*d, force)
                              - normal_form_field(state-eps*d, force))/(2*eps) for d in np.eye(2)])
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(j)),
            sorted([spectrum["radial_exponent"], spectrum["tangent_exponent"]]), atol=2e-9)
        off = normal_form_spectrum(0j)
        self.assertEqual(off["tangent_exponent"], 0)
        self.assertEqual(off["radial_exponent"], -2)

    def test_delay_feedback_boundary_and_characteristic_root_agree(self):
        self.assertAlmostEqual(feedback_delay_limit(1, 0), np.pi/2)
        self.assertGreater(feedback_delay_limit(1, .4), 1.7)
        self.assertLess(feedback_delay_limit(1, 0), 1.7)
        self.assertGreater(characteristic_root(1, 0, 1.7)["real"], 0)
        self.assertLess(characteristic_root(1, .4, 1.7)["real"], 0)
        self.assertEqual(feedback_delay_limit(1, 1.1), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
