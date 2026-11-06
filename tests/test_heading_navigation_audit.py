"""Independent timing, sensory isolation, and aggregation checks for heading studies."""
from pathlib import Path
import inspect
import json
import math
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_heading_confirmation as confirmation
import evaluate_ring_heading_navigation as navigation
from ring_heading_model import HeadingConfig, HeadingTask
from ring_task_model import wrap_angle


class ExactMeasuredAngularIntegrator:
    """Two-state test instrument; reads only supplied angular and initial cues."""
    def __init__(self, config):
        self.config = config
        self.hidden_size = 2
        self.input_size = config.input_size
        self.calls = []

    def step(self, x, h):
        self.calls.append((x.clone(), h.clone()))
        delta = self.config.dt * x[:, self.config.n_code]
        rotated = torch.stack((h[:, 0] * delta.cos() - h[:, 1] * delta.sin(),
                               h[:, 0] * delta.sin() + h[:, 1] * delta.cos()), -1)
        return torch.where(x[:, -1:].bool(), x[:, -3:-1], rotated)

    @staticmethod
    def readout(h):
        return h


class HeadingNavigationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def task(self, **changes):
        values = dict(n_code=32, code_seed=17, angular_velocity_noise=0.,
                      angular_velocity_bias=0., speed_noise=0., direction_noise=0.,
                      position_noise=0., channel_noise=0., recurrent_noise=0.)
        values.update(changes)
        return HeadingTask(HeadingConfig(**values))

    def test_training_motion_is_completed_before_current_observation(self):
        task = self.task()
        data = task.realization(128, 32, torch.Generator().manual_seed(72))
        c = task.config
        torch.testing.assert_close(wrap_angle(data["heading"][1:] - data["heading"][:-1]),
                                   c.dt * data["angular_velocity"][1:], atol=1e-6, rtol=0)
        torch.testing.assert_close(data["position"][1:] - data["position"][:-1],
                                   c.dt * data["speed"][1:, :, None] * data["target"][1:],
                                   atol=3e-7, rtol=0)
        self.assertTrue(torch.all(data["speed"][0] == 0))
        self.assertTrue(torch.all(data["angular_velocity"][data["stopped"]] == 0))

    def test_input_matching_and_explicit_heading_use_same_measured_direction(self):
        task = self.task(direction_noise=.05, channel_noise=.03,
                         angular_velocity_noise=.30, angular_velocity_bias=.10)
        generator = torch.Generator().manual_seed(91)
        data = task.realization(128, 16, generator)
        inputs = {condition: task.batch(128, 16, generator, condition, data)[0]
                  for condition in ("grid", "predictive", "grid_heading")}
        torch.testing.assert_close(inputs["grid"][..., :task.config.n_code],
                                   inputs["grid_heading"][..., :task.config.n_code], atol=0, rtol=0)
        torch.testing.assert_close(inputs["grid"][..., task.config.n_code:],
                                   inputs["predictive"][..., task.config.n_code:], atol=0, rtol=0)
        available = torch.maximum(data["visible"], data["initial_cue"])
        torch.testing.assert_close(inputs["grid_heading"][..., -3:-1],
                                   data["measured_direction"] * available[..., None], atol=0, rtol=0)
        # Once initialized, dropping the spatial cue removes PGC and explicit
        # heading information while leaving the common idiothetic sensors.
        blank = (data["visible"] == 0) & (data["initial_cue"] == 0)
        self.assertTrue(blank.any())
        for condition in inputs:
            self.assertTrue(torch.all(inputs[condition][..., :task.config.n_code][blank] == 0))
            self.assertTrue(torch.all(inputs[condition][..., -3:][blank] == 0))
        torch.testing.assert_close(inputs["predictive"][blank], inputs["grid_heading"][blank], atol=0, rtol=0)

    def test_misleading_initial_cue_does_not_change_predictive_spatial_code(self):
        task = self.task(direction_noise=.05, channel_noise=.03)
        noise = navigation.realization(task, 83, 20, 16)
        z, one = torch.zeros(16), torch.ones(16)
        args = (task, noise["starts"], noise["initial_heading"], z, z, one,
                one, "predictive", noise, 0)
        ordinary, speed0 = navigation.observation_inputs(*args)
        misleading, speed1 = navigation.observation_inputs(*args, initial_error_degrees=20.)
        torch.testing.assert_close(ordinary[..., :-3], misleading[..., :-3], atol=0, rtol=0)
        torch.testing.assert_close(ordinary[..., -1], misleading[..., -1], atol=0, rtol=0)
        torch.testing.assert_close(speed0, speed1, atol=0, rtol=0)
        angle0 = torch.atan2(ordinary[..., -2], ordinary[..., -3])
        angle1 = torch.atan2(misleading[..., -2], misleading[..., -3])
        torch.testing.assert_close(wrap_angle(angle1-angle0),
                                   math.radians(20)*noise["initial_error_sign"], atol=4e-7, rtol=0)
        # With no initial cue, passing the same manipulation has no effect.
        args = (task, noise["starts"], noise["initial_heading"], z, z, one,
                z, "predictive", noise, 0)
        a, _ = navigation.observation_inputs(*args)
        b, _ = navigation.observation_inputs(*args, initial_error_degrees=20.)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_perfect_sensor_integrator_tracks_preaction_position_without_pose_input(self):
        task = self.task()
        model = ExactMeasuredAngularIntegrator(task.config)
        noise = navigation.realization(task, 97, 160, 16)
        _, archive = navigation.simulate(model, task, {"condition": "predictive"}, noise)
        # Nonzero, varying motor noise and turning remain. Equality would fail
        # with the preceding rather than current direction or one-step offset.
        np.testing.assert_allclose(archive["trace_estimated_position"], archive["trace_position"], atol=3e-6, rtol=0)
        angular_error = np.angle(np.exp(1j*(archive["trace_decoded_heading"]-archive["trace_heading"])))
        self.assertLess(np.max(np.abs(angular_error)), 4e-6)
        parameters = set(inspect.signature(navigation.controller).parameters)
        self.assertFalse(parameters & {"position", "heading", "true_position", "true_heading"})

    def test_warmup_is_shared_and_restoration_changes_input_without_reset(self):
        task = self.task()
        model = ExactMeasuredAngularIntegrator(task.config)
        noise = navigation.realization(task, 103, 60, 8)
        noise["visible"][:] = 1
        navigation.simulate(model, task, {"condition": "predictive"}, noise, restore_seconds=2.)
        warm = task.config.initial_heading_steps
        self.assertEqual(len(model.calls), warm+60)
        self.assertTrue(all(len(x) == 8 for x, _ in model.calls[:warm]))
        for t, (x, h) in enumerate(model.calls[warm:]):
            x, h = x.reshape(4, 8, -1), h.reshape(4, 8, -1)
            # This test instrument ignores grid inputs, so all physical and
            # recurrent trajectories stay exactly matched across interventions.
            torch.testing.assert_close(h, h[:1].expand_as(h), atol=0, rtol=0)
            torch.testing.assert_close(x[3], x[1 if t < 40 else 0], atol=0, rtol=0)
            self.assertGreater(float((x[0, :, :task.config.n_code]-x[1, :, :task.config.n_code]).abs().max()), .01)
        self.assertGreater(float(model.calls[warm+40][1].norm()), 1.)
        # Exact no-op branches for an ordinary-code trained model.
        model2 = ExactMeasuredAngularIntegrator(task.config)
        _, archive = navigation.simulate(model2, task, {"condition": "grid"}, noise)
        for name in ("position", "estimated_position", "heading", "decoded_heading"):
            values = archive[f"trace_{name}"]
            np.testing.assert_array_equal(values[:, 0], values[:, 1])
            np.testing.assert_array_equal(values[:, 0], values[:, 3])

    def test_confirmation_has_2048_episodes_and_excludes_exactly_first_four(self):
        class FakeModel:
            def requires_grad_(self, _):
                return self
            def __call__(self, x):
                return x

        class FakeTask:
            device = torch.device("cpu")
            config = SimpleNamespace(dt=.05, encoder_size=256, hidden_size=64)
            calls = []
            def realization(self, length, episodes, generator):
                self.calls.append((generator.initial_seed(), length, episodes))
                target = torch.zeros(length, episodes, 2)
                target[..., 0] = 1
                return {"position": torch.zeros_like(target), "measured_position": torch.zeros_like(target), "target": target}
            def batch(self, length, episodes, generator, condition, data):
                degrees = torch.full((length, episodes), 2. if condition == "predictive" else 10.)
                degrees[:4] = 90.
                degrees[4] += 38.  # Detect accidentally excluding 8 rather than 4.
                radians = degrees * math.pi / 180
                return torch.stack((radians.cos(), radians.sin()), -1), data["target"], data

        task = FakeTask()
        ckpt = {"seed": 2001, "steps_completed": 6000, "condition": "predictive",
                "config": HeadingConfig(encoder_size=256, code_seed=2001).to_dict(), "trainer_args": {"length": 128, "steps": 6000,
                    "batch_size": 64, "learning_rate": .001},
                "study_stage": "confirmatory", "exploratory_arm": False, "stopping_reason": "steps_completed"}
        locked = json.loads((Path(__file__).resolve().parents[1]/"docs/research/heading_confirmation_sources.json").read_text())
        ckpt["source_sha256"] = {"/work/ring_repo/"+p: v for p, v in locked["sources"].items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"checkpoint.pt"
            checkpoint.write_bytes(b"audit mock checkpoint")
            output = Path(directory)/"out"
            with patch.object(confirmation, "load_checkpoint", return_value=(FakeModel(), task, ckpt)), \
                 patch.object(confirmation, "verify_training_sources", return_value={"verified_training_sources": {"audit_mock": True}}):
                result = confirmation.evaluate(checkpoint, output)
            arrays = np.load(output.with_suffix(".npz"))
            self.assertEqual(arrays["same_region_native"].shape, (2048,))
            np.testing.assert_array_equal(arrays["evaluation_seed"], np.repeat([71001, 71002, 71003, 71004], 512))
        self.assertEqual(task.calls, [(seed, 256, 512) for seed in (71001, 71002, 71003, 71004)])
        self.assertAlmostEqual(result["metrics"]["same_region_native_mae_deg"], 2+38/252, places=5)
        self.assertAlmostEqual(result["metrics"]["ordinary_minus_native_mae_deg"], 8., places=5)
        self.assertEqual(result["protocol"]["independent_replication_unit"], "training seed")

    def test_confirmation_rejects_changed_training_source(self):
        locked = json.loads((Path(__file__).resolve().parents[1]/"docs/research/heading_confirmation_sources.json").read_text())
        checkpoint = {"source_sha256": dict(locked["sources"]), "study_stage": "confirmatory",
                      "exploratory_arm": False, "trainer_args": {"steps": 6000, "length": 128,
                      "batch_size": 64, "learning_rate": .001}, "seed": 2001, "steps_completed": 6000,
                      "stopping_reason": "steps_completed", "config": HeadingConfig(encoder_size=256, code_seed=2001).to_dict()}
        confirmation.verify_training_sources(checkpoint)
        checkpoint["source_sha256"]["scripts/train_ring_heading.py"] = "0"*64
        with self.assertRaisesRegex(ValueError, "does not match"):
            confirmation.verify_training_sources(checkpoint)

    def test_mean_circular_shift_can_hide_locally_nonuniform_dynamics(self):
        # A monotone, degree-one angular distortion retains mean cyclic shift
        # exactly despite sharply heterogeneous local expansion/contraction.
        angle = np.arange(72)*2*np.pi/72
        distorted = angle + .85*np.sin(angle)
        difference = np.angle(np.exp(1j*(np.roll(distorted, -2)-distorted)))
        np.testing.assert_allclose(np.mean(np.abs(difference)), np.deg2rad(10), atol=1e-14)
        gain = np.abs(difference)/np.deg2rad(10)
        self.assertLess(np.quantile(gain, .1), .25)
        self.assertGreater(np.quantile(gain, .9), 1.75)


if __name__ == "__main__":
    unittest.main()
