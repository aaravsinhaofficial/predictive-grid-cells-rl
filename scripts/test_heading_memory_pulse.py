"""Check exact no-op injection and measured-state perturbation controls."""
import json

import numpy as np
import torch

from evaluate_heading_memory_pulse import StateReplacement, prepare
import evaluate_ring_heading_navigation as navigation
from ring_heading_model import HeadingConfig, HeadingRNN, HeadingTask


def main():
    torch.set_num_threads(2)
    torch.manual_seed(181)
    config = HeadingConfig()
    task, model = HeadingTask(config), HeadingRNN(config)
    noise = navigation.realization(task, 961001, 400, 8)
    zeros, ones = torch.zeros(8), torch.ones(8)
    original_state = torch.zeros(8, model.hidden_size)
    with torch.no_grad():
        for t in range(config.initial_heading_steps):
            x, _ = navigation.observation_inputs(task, noise["starts"], noise["initial_heading"],
                zeros, zeros, ones, ones, "predictive", noise, t)
            original_state = model.step(x, original_state)
        proxy = StateReplacement(model, original_state)
        reference, arrays = navigation.simulate(model, task, {"condition": "predictive"}, noise)
        injected, other = navigation.simulate(proxy, task, {"condition": "predictive"}, noise)
        assert reference == injected
        assert proxy.replacements == 1
        for key, value in arrays.items():
            assert np.array_equal(value, other[key], equal_nan=True), key
        pulses, prepared = prepare(model, task, {"condition": "predictive"}, noise, 961001)
    normal = pulses["normal_matched20"] - pulses["baseline"]
    tangent = torch.tensor(prepared["empirical_tangent"])
    selected = torch.where(torch.tensor(prepared["balanced_phase_sign"])[:, None] > 0,
                           pulses["phase_plus20"], pulses["phase_minus20"])
    assert torch.allclose(normal.norm(dim=-1), (selected - pulses["baseline"]).norm(dim=-1), atol=1e-6)
    assert float((normal * tangent).sum(-1).abs().max()) < 1e-6
    print(json.dumps({"passed": True, "checks": ["noop_injection_bitexact_all_trajectories_and_metrics",
        "one_state_replacement_only", "actual_conditional_states", "equal_norm_normal_control", "normal_tangent_orthogonality"]}))


if __name__ == "__main__":
    main()
