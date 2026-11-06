"""Mechanistic validity checks for the controlled ring task; run directly."""
import json
import math

import torch

from ring_task_model import RingConfig, RingRNN, RingTask, wrap_angle


def main():
    torch.set_num_threads(2)
    config = RingConfig(code_seed=173, delay_steps=4)
    task = RingTask(config)
    theta = torch.linspace(-math.pi, math.pi, 37)[:-1]
    velocity = torch.linspace(-2, 2, len(theta))
    ordinary = task.encode(theta, velocity, "grid")
    shifted = task.encode(theta, velocity, "predictive")
    assert torch.equal(ordinary[:, ~task.predictive_mask], shifted[:, ~task.predictive_mask])
    expected = task.encode(theta + config.horizon * velocity, velocity, "grid")
    assert torch.allclose(shifted[:, task.predictive_mask], expected[:, task.predictive_mask], atol=1e-6)
    assert torch.equal(task.encode(theta, velocity * 0, "predictive"), ordinary)
    periodic = task.encode(theta + 2 * math.pi, velocity, "predictive")
    assert torch.allclose(periodic, shifted, atol=1e-5)
    first = torch.Generator().manual_seed(123456)
    second = torch.Generator().manual_seed(123456)
    xg, yg, grid = task.batch(128, 32, first, "grid")
    xp, yp, pgc = task.batch(128, 32, second, "predictive")
    for key in grid:
        assert torch.equal(grid[key], pgc[key]), key
    assert torch.equal(xg[..., -2:], xp[..., -2:])
    assert torch.equal(xg[..., :64][..., ~task.predictive_mask], xp[..., :64][..., ~task.predictive_mask])
    assert torch.equal(yg, yp)
    indices = (torch.arange(128) - config.delay_steps).clamp_min(0)
    assert torch.equal(grid["delayed_phase"], grid["observed_phase"][indices])
    assert torch.equal(grid["delayed_velocity"], grid["observed_velocity"][indices])
    assert torch.allclose(wrap_angle(grid["theta"][1:] - grid["theta"][:-1]),
                          config.dt * grid["velocity"][1:], atol=1e-6)
    assert torch.equal(xg[..., :64][grid["visible"] == 0],
                       torch.zeros_like(xg[..., :64][grid["visible"] == 0]))
    # Change all physical measurements after a cutoff. Earlier inputs cannot
    # change, including the predictive forecast. This catches look-ahead leaks.
    changed = {key: value.clone() for key, value in grid.items()}
    cutoff = 43
    changed["observed_phase"][cutoff:] += 0.7
    changed["observed_velocity"][cutoff:] += 0.4
    changed["delayed_phase"] = changed["observed_phase"][indices]
    changed["delayed_velocity"] = changed["observed_velocity"][indices]
    changed["shuffled_velocity"] = changed["delayed_velocity"].roll(16, 1)
    future_x, _, _ = task.batch(128, 32, first, "predictive", changed)
    assert torch.equal(future_x[:cutoff], xp[:cutoff])
    model = RingRNN(config)
    noise = grid["recurrent_noise"]
    output, states = model(xg, noise=noise, return_states=True)
    h = torch.zeros(32, config.hidden_size)
    for t in range(len(xg)):
        h = model.step(xg[t], h, noise[t])
        assert torch.equal(h, states[t])
    assert torch.equal(model.readout(states), output)
    loss = (output - yg).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps({"passed": True, "checks": ["basis_preserved", "forecast_rule", "zero_velocity", "periodicity",
          "paired_randomness", "same_velocity_information", "delay_alignment", "no_future_leakage",
          "blank_inputs", "forward_step_equivalence", "finite_training_gradient"]}))


if __name__ == "__main__":
    main()
