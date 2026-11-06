"""Controller provenance and intervention checks for heading navigation."""
import json

import numpy as np
import torch

from evaluate_ring_heading_navigation import HoldingScores, controller, observation_inputs, realization, simulate
from ring_heading_model import HeadingConfig, HeadingRNN, HeadingTask


def main():
    torch.set_num_threads(2)
    scores = HoldingScores(torch.zeros(1, 3), dt=0.05, episode_steps=200, capture_radius=0.15)
    entered = torch.zeros(1, 3, dtype=torch.bool)
    for step in range(200):
        distance = torch.tensor([[0.1, 0.1 if 20 <= step < 30 else 0.5, 0.8]])
        entered |= distance <= 0.15
        scores.update(distance, torch.zeros_like(distance), entered)
    summary, arrays = scores.summarize(0, entered[0])
    assert summary["post_entry_n"] == 2
    assert torch.allclose(arrays["max_distance_after_entry"][:2], torch.tensor([0.1, 0.5]))
    assert torch.isnan(arrays["max_distance_after_entry"][2])
    assert torch.equal(arrays["terminal_goal_occupancy"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(arrays["sustained_dwell_success"], torch.tensor([True, False, False]))
    assert torch.equal(arrays["capped_sustained_dwell_seconds"], torch.tensor([2.0, 10.0, 10.0]))
    torch.manual_seed(121)
    config = HeadingConfig()
    task, model = HeadingTask(config), HeadingRNN(config)
    noise = realization(task, 19123, 400, 8)
    estimate = torch.randn(8, 2, dtype=torch.float64)
    decoded = torch.randn(8, dtype=torch.float64)
    goals = torch.randn(8, 2, dtype=torch.float64)
    original = controller(estimate, decoded, goals)
    shifted = controller(estimate + 10, decoded, goals + 10)
    assert all(torch.allclose(a, b, atol=1e-12, rtol=0) for a, b in zip(original, shifted))
    zeros, ones = torch.zeros(8), torch.ones(8)
    ordinary, _ = observation_inputs(task, noise["starts"], noise["initial_heading"], zeros, zeros,
        ones, ones, "predictive", noise, 0)
    misleading, _ = observation_inputs(task, noise["starts"], noise["initial_heading"], zeros, zeros,
        ones, ones, "predictive", noise, 0, initial_error_degrees=20)
    assert torch.equal(ordinary[..., :-3], misleading[..., :-3])
    assert torch.equal(ordinary[..., -1], misleading[..., -1])
    assert not torch.equal(ordinary[..., -3:-1], misleading[..., -3:-1])
    small, small_trace = simulate(model, task, {"condition": "grid"}, noise, capture_radius=0.001)
    large, large_trace = simulate(model, task, {"condition": "grid"}, noise, capture_radius=100.0)
    for name in ("trace_position", "trace_estimated_position", "trace_heading", "trace_decoded_heading"):
        assert np.array_equal(small_trace[name], large_trace[name]), name
    assert large["durations"]["20s"]["native"]["success_fraction"] == 1.0
    print(json.dumps({"passed": True, "checks": ["controller_translation_invariance",
        "misleading_heading_cue_preserves_truthful_pgc", "scoring_does_not_affect_behavior",
        "no_rotation_while_stopped", "grid_input_branches_identical", "holding_outcomes_known_trajectories"]}))


if __name__ == "__main__":
    main()
