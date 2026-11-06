"""Independent checks of rank-two causal contrasts and seed-level inference."""
from pathlib import Path
import copy
import json
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from heading_direction_subspace import DriveInterventions, fit_harmonics, fit_metrics, make_shams
import heading_subspace_navigation as subspace_navigation
import evaluate_ring_heading_navigation as navigation
from ring_heading_model import HeadingConfig, HeadingRNN, HeadingTask
from summarize_heading_confirmation import paired_summary, holm, anchor_endpoint, validate_report
from evaluate_heading_confirmation import verify_training_sources


class HeadingMechanisticAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model_and_task(self):
        torch.manual_seed(172)
        config = HeadingConfig(hidden_size=8, encoder_size=16, n_code=32,
                               angular_velocity_noise=.1, angular_velocity_bias=.03,
                               direction_noise=.02, position_noise=.01)
        return HeadingRNN(config).eval(), HeadingTask(config)

    def test_projector_algebra_and_shams_are_matched_before_tanh(self):
        model, _ = self.model_and_task()
        basis = np.linalg.qr(np.random.default_rng(37).normal(size=(8, 2)))[0]
        shams = make_shams(basis, 38)
        probe = DriveInterventions(model, basis, shams)
        native, ordinary = torch.randn(13, 8), torch.randn(13, 8)
        drives, doses = probe.transform_drives(native, ordinary)
        component = drives["restore_direction"]-ordinary
        torch.testing.assert_close(drives["remove_direction"], drives["residual_only"])
        torch.testing.assert_close(native-drives["remove_direction"], component)
        torch.testing.assert_close((component@probe.basis)@probe.basis.T, component)
        for j in range(3):
            sham = drives[f"sham{j}_restore"]-ordinary
            torch.testing.assert_close(sham.norm(dim=-1), component.norm(dim=-1), atol=5e-7, rtol=1e-6)
            torch.testing.assert_close(sham@probe.basis, torch.zeros(13, 2), atol=5e-7, rtol=0)
        # Equal drive norms do NOT imply equal changes after a nonlinear,
        # coordinate-dependent tanh. This prevents a stronger interpretation
        # of the matched-dose control than the code actually supplies.
        context = torch.linspace(-3., 3., 8).expand_as(native)
        direction_change = (context+component).tanh()-context.tanh()
        sham = drives["sham0_restore"]-ordinary
        sham_change = (context+sham).tanh()-context.tanh()
        self.assertGreater(float((direction_change.norm(dim=-1)-sham_change.norm(dim=-1)).abs().max()), .01)
        blank_drives, blank_doses = probe.transform_drives(native, native)
        self.assertTrue(torch.all(blank_doses["direction_norm"] == 0))
        for drive in blank_drives.values():
            torch.testing.assert_close(drive, native, atol=0, rtol=0)

    def test_intervened_step_reproduces_unmodified_model_from_native_drive(self):
        model, task = self.model_and_task()
        basis = np.eye(8)[:, :2]
        probe = DriveInterventions(model, basis, make_shams(basis, 39))
        x = torch.randn(19, task.config.input_size)
        h = torch.randn(19, task.config.hidden_size)
        torch.testing.assert_close(probe.step_from_drive(model.input(x), h), model.step(x, h), atol=0, rtol=0)

    def test_harmonic_fit_recovers_direction_plane_from_independent_positions(self):
        rng = np.random.default_rng(43)
        directions = np.linalg.qr(rng.normal(size=(8, 8)))[0]
        truth = directions[:, :2]
        def data(n_position):
            phase = (np.arange(32)[None, :]*2*np.pi/32 + rng.uniform(-np.pi, np.pi, (n_position, 1))).reshape(-1)
            heading = np.column_stack((np.cos(phase), np.sin(phase)))
            position_offsets = np.repeat(rng.normal(size=(n_position, 6)), 32, axis=0)@directions[:, 2:].T
            # Spatial offsets contain much more energy than the rank-two
            # heading component but average out within each sampled position.
            delta = heading@truth.T + 5*position_offsets
            return {"phase": phase, "direction": heading, "delta": delta,
                    "positions": n_position, "angles": 32, "offset": 0.}
        train, heldout = data(40), data(17)
        fitted = fit_harmonics(train, 1)
        np.testing.assert_allclose(fitted["basis"]@fitted["basis"].T, truth@truth.T, atol=1e-12)
        self.assertLess(fit_metrics(heldout, fitted)["projected_direction_mae_deg"], .001)
        self.assertFalse(np.array_equal(train["delta"][:len(heldout["delta"])], heldout["delta"]))

    def test_closed_loop_native_and_ordinary_match_independent_evaluator(self):
        model, task = self.model_and_task()
        probe = DriveInterventions(model, np.eye(8)[:, :2], make_shams(np.eye(8)[:, :2], 41))
        noise = navigation.realization(task, 451, 400, 6)
        checkpoint = {"condition": "predictive"}
        _, reference = navigation.simulate(model, task, checkpoint, noise)
        _, actual = subspace_navigation.simulate(model, task, checkpoint, probe, noise)
        for variant in ("native", "ordinary"):
            for metric in ("success", "capped_arrival", "final_goal_distance", "self_position_error"):
                key = f"20s_{variant}_{metric}"
                np.testing.assert_allclose(actual[key], reference[key], atol=2e-5, rtol=0)
        for trace in ("position", "estimated_position", "heading", "decoded_heading"):
            np.testing.assert_allclose(actual[f"trace_{trace}"][:, :2], reference[f"trace_{trace}"][:, :2], atol=2e-5, rtol=0)

    def test_exact_sign_flip_counts_zero_seeds_without_dropping_them(self):
        unanimous = paired_summary(np.ones(16))
        self.assertEqual(unanimous["p_exact_sign_flip"], 2/2**16)
        self.assertEqual(unanimous["ci95"], [1., 1.])
        zero_half = paired_summary(np.r_[np.ones(8), np.zeros(8)])
        self.assertEqual(zero_half["p_exact_sign_flip"], 2/2**8)
        self.assertEqual(zero_half["n_training_seeds"], 16)
        centered = paired_summary(np.r_[np.arange(1, 9), -np.arange(1, 9)])
        self.assertEqual(centered["p_exact_sign_flip"], 1.)
        with self.assertRaises(ValueError):
            paired_summary(np.ones(15))
        with self.assertRaises(ValueError):
            paired_summary(np.r_[np.ones(15), np.nan])

    def test_holm_and_anchor_contexts(self):
        np.testing.assert_allclose(holm([.01, .04, .03, .2]), [.04, .09, .09, .2], atol=1e-15)
        rows = [{"context": context, "step": 100, "variant": variant,
                 "phase_local_gain_median": .1*context, "baseline_mae_deg": 2.,
                 "circle_coverage": .96, "full_state_phase_perturbation_gain_median": .2,
                 "state_radius_retained": 1.}
                for variant in ("native", "ordinary", "blank") for context in range(5)]
        protocol = {"phase_perturbation_deg": 10, "initial_heading_cue_steps": 4,
                    "blank_settling_steps": 40, "n_phase": 72, "continuation_steps": 200,
                    "physical_positions": [[-.8,-.8],[-.8,.8],[.8,-.8],[.8,.8],[0.,0.],[10.,10.]]}
        result = anchor_endpoint({"rows": rows, "protocol": protocol})
        self.assertAlmostEqual(result["native"]["phase_local_gain_median"], .2)
        with self.assertRaises(ValueError):
            anchor_endpoint({"rows": rows[:-1], "protocol": protocol})
        with self.assertRaises(ValueError):
            anchor_endpoint({"rows": rows+[rows[0]], "protocol": protocol})
        with self.assertRaises(ValueError):
            anchor_endpoint({"rows": rows, "protocol": {**protocol, "n_phase": 36}})

    def test_provenance_rejects_source_config_stage_and_training_budget_changes(self):
        root = Path(__file__).resolve().parents[1]
        locked = json.loads((root/"docs/research/heading_confirmation_sources.json").read_text())
        checkpoint = {"source_sha256": {"/opt/cloud/work/"+p: h for p,h in locked["sources"].items() if p != "scripts/run_ring_batch.py"},
                      "seed": 2001, "study_stage": "confirmatory", "exploratory_arm": False,
                      "steps_completed": 6000, "stopping_reason": "steps_completed",
                      "config": HeadingConfig(encoder_size=256, hidden_size=64, code_seed=2001).to_dict(),
                      "trainer_args": {"steps": 6000, "length": 128, "batch_size": 64, "learning_rate": .001}}
        proof = verify_training_sources(checkpoint)
        self.assertEqual(len(proof["verified_training_sources"]), 4)
        changed = copy.deepcopy(checkpoint); changed["config"]["displacement"] = .40
        with self.assertRaises(ValueError):
            verify_training_sources(changed)
        changed = copy.deepcopy(checkpoint); changed["study_stage"] = "exploratory"
        with self.assertRaises(ValueError):
            verify_training_sources(changed)
        changed = copy.deepcopy(checkpoint); changed["stopping_reason"] = "time_limit"
        with self.assertRaises(ValueError):
            verify_training_sources(changed)
        changed = copy.deepcopy(checkpoint); changed["source_sha256"]["/opt/cloud/work/code/codes.py"] = "0"*64
        with self.assertRaises(ValueError):
            verify_training_sources(changed)
        changed = copy.deepcopy(checkpoint); changed["source_sha256"]["code/codes.py"] = locked["sources"]["code/codes.py"]
        with self.assertRaises(ValueError):
            verify_training_sources(changed)

    def test_aggregation_rejects_changed_evaluation_sampling(self):
        root = Path(__file__).resolve().parents[1]
        locked = json.loads((root/"docs/research/heading_confirmation_sources.json").read_text())
        protocol = {"evaluation_seeds": [71001,71002,71003,71004], "episodes_per_seed": 512,
                    "total_episodes_per_network": 2048, "sequence_length": 256,
                    "exclude_first_steps": 4, "recurrent_noise": 0, "position_translation_m": [10.,0.],
                    "pilot_override": False, "native_and_ordinary_share_all_underlying_experience": True}
        report = {"training_seed": 2001, "condition": "predictive", "steps_completed": 6000,
                  "protocol": protocol, "provenance": {
                      "verified_training_sources": {p:h for p,h in locked["sources"].items() if p != "scripts/run_ring_batch.py"},
                      "protocol_sha256": locked["protocol_sha256"]}}
        validate_report(report, 2001, "predictive", locked)
        changed = copy.deepcopy(report); changed["protocol"]["exclude_first_steps"] = 8
        with self.assertRaises(ValueError):
            validate_report(changed, 2001, "predictive", locked)
        changed = copy.deepcopy(report); changed["protocol"]["episodes_per_seed"] = 128
        with self.assertRaises(ValueError):
            validate_report(changed, 2001, "predictive", locked)
        with self.assertRaises(ValueError):
            validate_report(report, 2002, "predictive", locked)


if __name__ == "__main__":
    unittest.main()
