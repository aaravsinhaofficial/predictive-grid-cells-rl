"""Validate timing, original-code fidelity and information controls."""
import json
import math

import torch

from ring_heading_model import HeadingConfig, HeadingRNN, HeadingTask
from ring_task_model import wrap_angle


def main():
    torch.set_num_threads(2)
    config = HeadingConfig(code_seed=1005)
    task = HeadingTask(config)
    xg, target, data = task.batch(96, 32, torch.Generator().manual_seed(12345), "grid")
    xp, tp, dp = task.batch(96, 32, torch.Generator().manual_seed(12345), "predictive")
    xe, te, de = task.batch(96, 32, torch.Generator().manual_seed(12345), "grid_heading")
    assert all(torch.equal(data[k], dp[k]) and torch.equal(data[k], de[k]) for k in data)
    assert torch.equal(xg[..., :config.n_code], xe[..., :config.n_code])
    assert torch.equal(xg[..., config.n_code:], xp[..., config.n_code:])
    assert torch.equal(xg[..., :config.n_code][..., ~task.predictive_mask],
                       xp[..., :config.n_code][..., ~task.predictive_mask])
    assert torch.equal(target, tp) and torch.equal(target, te)
    explicit_availability = torch.maximum(data["visible"], data["initial_cue"])
    assert torch.equal(xe[..., -3:-1], data["measured_direction"] * explicit_availability[..., None])
    assert torch.equal(xe[..., -1], explicit_availability)
    assert torch.count_nonzero(xg[config.initial_heading_steps:, :, -3:]) == 0
    p = data["measured_position"].flatten(0, 1)
    u = data["measured_direction"].flatten(0, 1)
    errors = {}
    for name, scale in (("grid", 0), ("predictive", 1), ("reverse", -1)):
        expected = task.population.activity(p, dir=u, shift_scale=scale)
        observed = task.encode(p, u, name)
        errors[name] = float((observed - expected).abs().max())
        assert errors[name] < 1e-5, (name, errors[name])
    assert torch.allclose(wrap_angle(data["heading"][1:] - data["heading"][:-1]),
                          config.dt * data["angular_velocity"][1:], atol=1e-6)
    assert torch.allclose(data["position"][1:] - data["position"][:-1],
                          config.dt * data["speed"][1:, :, None] * target[1:], atol=1e-6)
    assert torch.count_nonzero(data["speed"][data["stopped"]]) == 0
    assert torch.count_nonzero(data["angular_velocity"][data["stopped"]]) == 0
    angles = torch.arange(32) * (2 * math.pi / 32)
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    position = torch.zeros(32, 2)
    ordinary = task.encode(position, direction, "grid")
    predictive = task.encode(position, direction, "predictive")
    assert torch.equal(ordinary, ordinary[0:1].expand_as(ordinary))
    assert float(predictive[:, task.predictive_mask].var(0).mean()) > 0.01
    # Grid receives no instantaneous heading after the common initial cue:
    # changing direction alone must leave its entire input unchanged.
    zero, one = torch.zeros(32), torch.ones(32)
    grid_a = task.inputs(position, direction, zero, zero, one, zero, "grid")
    grid_b = task.inputs(position, direction.roll(4, 0), zero, zero, one, zero, "grid")
    assert torch.equal(grid_a, grid_b)
    model = HeadingRNN(config)
    output = model(xp, noise=data["recurrent_noise"])
    (output - target).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    encoded = HeadingRNN(HeadingConfig(code_seed=1005, encoder_size=256))
    short_input, short_noise = xp[:12, :4], data["recurrent_noise"][:12, :4]
    _, encoded_states = encoded(short_input, noise=short_noise, return_states=True)
    recurrent_state = short_input.new_zeros(4, config.hidden_size)
    encoded_error = 0.0
    for t in range(len(short_input)):
        recurrent_state = encoded.step(short_input[t], recurrent_state, short_noise[t])
        encoded_error = max(encoded_error, float((recurrent_state - encoded_states[t]).detach().abs().max()))
    assert encoded_error < 2e-6, encoded_error
    print(json.dumps({"passed": True, "original_code_max_errors": errors,
          "encoded_forward_step_max_error": encoded_error,
          "checks": ["matched_randomness", "same_spatial_basis", "same_nonspatial_information_grid_vs_pgc",
             "explicit_heading_matches_pgc_direction", "no_heading_leak_in_grid", "initial_cue_removed",
             "position_heading_timing", "retained_direction_when_stopped", "finite_gradient",
             "encoded_forward_step_equivalence"]}))


if __name__ == "__main__":
    main()
