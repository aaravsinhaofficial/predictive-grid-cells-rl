"""Exploratory learned heading memory using the original hexagonal grid code.

Distinct from ring_task_model.py: physical motion is two-dimensional and PGCs
use a fixed spatial displacement along retained travel direction. This tests
whether predictive spatial codes provide a spatially indexed compass to a
learned heading integrator. An explicit-heading control matches that input's
direction information. No ring geometry or connectivity is imposed.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
# Import the original source without colliding with Python's stdlib `code`
# module, which PyTorch may already have imported through its debugger tools.
from codes import GridPopulation
from ring_task_model import RingRNN, wrap_angle


CONDITIONS = ("grid", "predictive", "grid_heading", "reverse", "shuffled")


@dataclass
class HeadingConfig:
    hidden_size: int = 64
    encoder_size: int = 0
    n_code: int = 128
    dt: float = 0.05
    leak: float = 0.25
    predictive_fraction: float = 1 / 3
    displacement: float = 0.20
    angular_velocity_noise: float = 0.30
    angular_velocity_bias: float = 0.10
    speed_noise: float = 0.01
    direction_noise: float = 0.05
    position_noise: float = 0.01
    channel_noise: float = 0.03
    recurrent_noise: float = 0.01
    visibility: float = 0.50
    initial_heading_steps: int = 4
    code_seed: int = 1

    @property
    def input_size(self):
        # Spatial code; noisy angular velocity; noisy forward speed; spatial
        # availability; two heading-cue components; heading-cue availability.
        return self.n_code + 6

    def to_dict(self):
        return asdict(self)


class HeadingRNN(RingRNN):
    def __init__(self, config):
        super().__init__(config)
        if config.encoder_size:
            width = config.encoder_size
            self.input = nn.Sequential(nn.Linear(config.input_size, width), nn.SiLU(),
                                       nn.Linear(width, width), nn.SiLU(),
                                       nn.Linear(width, config.hidden_size))
            for layer in self.input:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, sequence, h=None, noise=None, return_states=False):
        if not self.config.encoder_size:
            # Keep the original pilot execution path unchanged.
            return super().forward(sequence, h, noise, return_states)
        if h is None:
            h = sequence.new_zeros(sequence.shape[1], self.hidden_size)
        # Evaluate the sensory encoder over all timesteps in one batched call.
        # Only the unconstrained recurrent computation remains sequential.
        drive = self.input(sequence)
        states = []
        for t in range(len(sequence)):
            pre = drive[t] + self.recurrent(h)
            if noise is not None:
                pre = pre + noise[t]
            h = (1 - self.config.leak) * h + self.config.leak * pre.tanh()
            states.append(h)
        states = torch.stack(states)
        output = self.readout(states)
        return (output, states) if return_states else output


class HeadingTask:
    def __init__(self, config, device="cpu"):
        self.config, self.device = config, torch.device(device)
        self.population = GridPopulation(n_cells=config.n_code, n_modules=4,
            pgc_frac=config.predictive_fraction, pgc_shift=config.displacement,
            pgc_shift_mode="space", sharpness=2.0, noise=0, seed=config.code_seed,
            device=device)
        self.predictive_mask = self.population.is_pgc
        self._phase_offset = (self.population.K * self.population.phi[:, None, :]).sum(-1)

    def encode(self, position, direction, condition="grid", shuffled_direction=None):
        """Original three-cosine hexagonal rate, vectorized over time and batch.

        Inputs are measured location and measured retained travel direction.
        Explicit-heading uses the same ordinary spatial code as Grid.
        """
        if condition not in CONDITIONS:
            raise ValueError(condition)
        base = torch.einsum("...d,nkd->...nk", position, self.population.K) - self._phase_offset
        if condition in ("predictive", "reverse", "shuffled"):
            chosen = direction
            if condition == "reverse":
                chosen = -direction
            if condition == "shuffled":
                if shuffled_direction is None:
                    raise ValueError("shuffled_direction is required")
                chosen = shuffled_direction
            shift = torch.einsum("...d,nkd->...nk", chosen, self.population.K)
            base = base + shift * self.population.shift[:, None]
        return ((base.cos().sum(-1) + 1.5) / 4.5).clamp(0, 1).square()

    def inputs(self, position, direction, angular_velocity, speed, visible,
               initial_cue, condition, channel_noise=None, shuffled_direction=None):
        code = self.encode(position, direction, condition, shuffled_direction)
        if channel_noise is not None:
            code = (code + channel_noise).clamp(0, 1)
        code = code * visible[..., None]
        heading_available = initial_cue
        if condition == "grid_heading":
            heading_available = torch.maximum(initial_cue, visible)
        heading_code = direction * heading_available[..., None]
        return torch.cat((code, angular_velocity[..., None], speed[..., None],
                          visible[..., None], heading_code, heading_available[..., None]), -1)

    def realization(self, length, batch_size, generator):
        c, device = self.config, self.device
        rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)
        normal = lambda *shape: torch.randn(*shape, device=device, generator=generator)
        initial_heading = rand(batch_size) * (2 * math.pi) - math.pi
        initial_position = (rand(batch_size, 2) * 2 - 1) * 1.25
        angular = torch.zeros(length, batch_size, device=device)
        speed = torch.zeros_like(angular)
        angular_innovation, speed_innovation = normal(length, batch_size), normal(length, batch_size)
        speed[0] = 0.22
        for t in range(1, length):
            angular[t] = (0.92 * angular[t - 1] + 0.30 * angular_innovation[t]).clamp(-2.0, 2.0)
            speed[t] = (0.98 * speed[t - 1] + 0.02 * 0.22 + 0.015 * speed_innovation[t]).clamp(0.10, 0.35)
        times = torch.arange(length, device=device)[:, None]
        stop_start = length - (rand(batch_size) * max(1, length // 3)).long() - max(4, length // 8)
        stops = (times >= stop_start) & (rand(batch_size) < 0.5)
        # In this arm heading equals travel direction. A stopped agent does
        # not turn in place; therefore retained travel direction equals heading.
        speed = speed.masked_fill(stops, 0)
        angular = angular.masked_fill(stops, 0)
        speed[0] = 0
        heading = wrap_angle(initial_heading[None, :] + c.dt * angular.cumsum(0))
        direction = torch.stack((heading.cos(), heading.sin()), -1)
        position = initial_position[None, :, :] + c.dt * (speed[..., None] * direction).cumsum(0)
        measured_heading = heading + c.direction_noise * normal(length, batch_size)
        measured_direction = torch.stack((measured_heading.cos(), measured_heading.sin()), -1)
        measured_position = position + c.position_noise * normal(length, batch_size, 2)
        sensor_bias = c.angular_velocity_bias * normal(batch_size)
        measured_angular = angular + sensor_bias[None, :] + c.angular_velocity_noise * normal(length, batch_size)
        measured_speed = (speed + c.speed_noise * normal(length, batch_size)).clamp_min(0)
        visible = rand(length, batch_size) < c.visibility
        blackout_start = length // 4 + (rand(batch_size) * max(1, length // 4)).long()
        blackout_duration = max(1, length // 8) + (rand(batch_size) * max(1, length // 4)).long()
        visible &= ~((times >= blackout_start) & (times < blackout_start + blackout_duration))
        # Some stationary tails have cues (reanchoring), others are blank
        # (autonomous memory); this is shared by every population.
        visible &= ~(stops & (rand(batch_size) < 0.5))
        visible[:c.initial_heading_steps] = True
        initial_cue = (times < c.initial_heading_steps).expand(length, batch_size).float()
        shuffled_direction = measured_direction.roll(max(1, batch_size // 2), 1)
        return {"heading": heading, "target": direction, "position": position,
                "angular_velocity": angular, "speed": speed, "stopped": stops,
                "measured_position": measured_position, "measured_direction": measured_direction,
                "measured_angular_velocity": measured_angular, "measured_speed": measured_speed,
                "angular_velocity_sensor_bias": sensor_bias, "visible": visible.float(),
                "initial_cue": initial_cue, "shuffled_direction": shuffled_direction,
                "channel_noise": c.channel_noise * normal(length, batch_size, c.n_code),
                "recurrent_noise": c.recurrent_noise * normal(length, batch_size, c.hidden_size)}

    def batch(self, length, batch_size, generator, condition, realization=None):
        data = self.realization(length, batch_size, generator) if realization is None else realization
        x = self.inputs(data["measured_position"], data["measured_direction"],
            data["measured_angular_velocity"], data["measured_speed"], data["visible"],
            data["initial_cue"], condition, data["channel_noise"], data["shuffled_direction"])
        return x, data["target"], data


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = HeadingConfig(**checkpoint["config"])
    model, task = HeadingRNN(config).to(device), HeadingTask(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, task, checkpoint
