"""Controlled circular navigation memory task (a 1-D grid-code analogue).

This is a new model, not a retraining of the 1B-frame 2-D navigation LSTMs.
No circular connectivity, hidden-state geometry target, or attractor loss is
imposed. A learned state estimator supplies a fixed goal-seeking controller.

At time t, theta[t] is position AFTER movement v[t]*dt. Input velocity[t]
measures that same completed movement. A location observation delivered at t
was acquired at max(0, t-delay_steps). Predictive channels use that past
observation plus horizon * the SAME noisy velocity sample acquired then.
All conditions receive the identical current noisy velocity stream.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


CONDITIONS = ("grid", "predictive", "reverse", "shuffled")


@dataclass
class RingConfig:
    hidden_size: int = 64
    n_code: int = 64
    dt: float = 0.05
    leak: float = 0.25
    horizon: float = 0.20
    delay_steps: int = 4
    predictive_fraction: float = 1 / 3
    velocity_noise: float = 0.05
    position_noise: float = 0.10
    channel_noise: float = 0.04
    recurrent_noise: float = 0.01
    visibility: float = 0.25
    max_speed: float = 2.0
    code_seed: int = 1

    @property
    def input_size(self):
        return self.n_code + 2  # noisy velocity plus observation-available flag

    def to_dict(self):
        return asdict(self)


def wrap_angle(x):
    return torch.atan2(x.sin(), x.cos())


class RingRNN(nn.Module):
    """Unconstrained leaky tanh RNN. Neutral input is exactly all zeros."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_size = config.input_size
        self.input = nn.Linear(self.input_size, self.hidden_size)
        self.recurrent = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.readout = nn.Linear(self.hidden_size, 2)
        nn.init.orthogonal_(self.recurrent.weight)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.readout.bias)

    def step(self, x, h, noise=None):
        pre = self.input(x) + self.recurrent(h)
        if noise is not None:
            pre = pre + noise
        return (1 - self.config.leak) * h + self.config.leak * pre.tanh()

    def forward(self, sequence, h=None, noise=None, return_states=False):
        """sequence [time,batch,input]; explicit noise [time,batch,hidden]."""
        if h is None:
            h = sequence.new_zeros(sequence.shape[1], self.hidden_size)
        states = []
        for t in range(len(sequence)):
            h = self.step(sequence[t], h, None if noise is None else noise[t])
            states.append(h)
        states = torch.stack(states)
        outputs = self.readout(states)
        return (outputs, states) if return_states else outputs


class RingTask:
    """Deterministic basis and condition-independent random task realizations."""

    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(config.code_seed)
        if config.n_code % 4:
            raise ValueError("n_code must be divisible by four")
        size = config.n_code // 4
        self.frequency = torch.tensor([2, 3, 5, 7], dtype=torch.float32).repeat_interleave(size).to(device)
        phases = torch.arange(size).float() * (2 * math.pi / size)
        offsets = torch.rand(4, generator=gen) * (2 * math.pi)
        self.phase = (phases[None, :] + offsets[:, None]).flatten().to(device)
        # Stratification keeps forecast cells in every spatial-frequency module.
        counts = [int(config.predictive_fraction * size)] * 4
        remainder = round(config.predictive_fraction * config.n_code) - sum(counts)
        for i in range(remainder):
            counts[i % 4] += 1
        indices = torch.cat([torch.randperm(size, generator=gen)[:n] + i * size
                             for i, n in enumerate(counts)])
        self.predictive_mask = torch.zeros(config.n_code, dtype=torch.bool, device=device)
        self.predictive_mask[indices.to(device)] = True

    def encode(self, phase, velocity=None, condition="grid", shuffled_velocity=None):
        """Noiseless periodic code, with leading dimensions preserved.

        phase is a *measured* phase, not necessarily the true/current phase.
        Caller owns sensory delays, noise, and missing-observation masks.
        """
        if condition not in CONDITIONS:
            raise ValueError(condition)
        phase = torch.as_tensor(phase, dtype=self.frequency.dtype, device=self.device)
        positions = phase[..., None].expand(*phase.shape, self.config.n_code)
        if condition != "grid":
            if velocity is None:
                raise ValueError("velocity required for shifted codes")
            shift_velocity = velocity
            if condition == "reverse":
                shift_velocity = -velocity
            if condition == "shuffled":
                if shuffled_velocity is None:
                    raise ValueError("explicit shuffled_velocity required")
                shift_velocity = shuffled_velocity
            positions = positions + (self.config.horizon * shift_velocity)[..., None] * self.predictive_mask
        # Same range, periodic basis and number of channels in every condition.
        return (self.frequency * positions + self.phase).cos()

    def inputs(self, measured_phase, measured_velocity, visible, condition,
               delayed_velocity=None, shuffled_velocity=None, channel_noise=None):
        """Build inputs when caller has already selected the delayed observation."""
        cue_velocity = measured_velocity if delayed_velocity is None else delayed_velocity
        code = self.encode(measured_phase, cue_velocity, condition, shuffled_velocity)
        if channel_noise is not None:
            code = code + channel_noise
        code = code * visible[..., None]
        return torch.cat((code, measured_velocity[..., None], visible[..., None]), -1)

    def realization(self, length, batch_size, generator, stationary_tail=True):
        """Create one underlying experience, reusable across all code conditions.

        Random occlusions and long, motionless blank tails make history useful.
        Noisy observed velocity, including its past samples, is shared by all
        conditions. The phase noise is shared across all channels in a cue;
        independent channel noise is also shared between matched conditions.
        """
        c, device = self.config, self.device
        rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)
        normal = lambda *shape: torch.randn(*shape, device=device, generator=generator)
        initial = rand(batch_size) * (2 * math.pi) - math.pi
        innovations = normal(length, batch_size)
        velocity = torch.zeros(length, batch_size, device=device)
        for t in range(1, length):
            velocity[t] = (0.95 * velocity[t - 1] + 0.22 * innovations[t]).clamp(-c.max_speed, c.max_speed)
        visible = rand(length, batch_size) < c.visibility
        visible[:min(4, length)] = True
        # All samples have a moving occlusion, with varying onset and duration.
        starts = (rand(batch_size) * max(1, length // 3)).long() + length // 4
        durations = (rand(batch_size) * max(1, length // 3)).long() + max(1, length // 8)
        times = torch.arange(length, device=device)[:, None]
        visible &= ~((times >= starts) & (times < starts + durations))
        stopped = torch.zeros(length, batch_size, dtype=torch.bool, device=device)
        if stationary_tail:
            has_tail = rand(batch_size) < 0.5
            tail_start = length - (rand(batch_size) * max(1, length // 3)).long() - max(4, length // 8)
            stopped = (times >= tail_start) & has_tail
            velocity = velocity.masked_fill(stopped, 0)
            visible &= ~stopped
        theta = wrap_angle(initial[None, :] + c.dt * velocity.cumsum(0))
        observed_velocity = velocity + c.velocity_noise * normal(length, batch_size)
        observed_phase = wrap_angle(theta + c.position_noise * normal(length, batch_size))
        indices = (torch.arange(length, device=device) - c.delay_steps).clamp_min(0)
        delayed_phase, delayed_velocity = observed_phase[indices], observed_velocity[indices]
        # A fixed nonzero cyclic batch shift preserves the entire marginal
        # distribution and temporal structure of velocity, but breaks its
        # correspondence to the location trajectory. It never uses future time.
        if batch_size < 2:
            shuffled_velocity = -delayed_velocity
        else:
            shuffled_velocity = delayed_velocity.roll(batch_size // 2, dims=1)
        return {
            "theta": theta, "target": torch.stack((theta.cos(), theta.sin()), -1),
            "velocity": velocity, "observed_velocity": observed_velocity,
            "observed_phase": observed_phase, "delayed_phase": delayed_phase,
            "delayed_velocity": delayed_velocity, "shuffled_velocity": shuffled_velocity,
            "visible": visible.float(), "stopped": stopped,
            "channel_noise": c.channel_noise * normal(length, batch_size, c.n_code),
            "recurrent_noise": c.recurrent_noise * normal(length, batch_size, c.hidden_size),
            "observation_index": indices,
        }

    def batch(self, length, batch_size, generator, condition="grid", realization=None):
        data = self.realization(length, batch_size, generator) if realization is None else realization
        x = self.inputs(data["delayed_phase"], data["observed_velocity"],
                        data["visible"], condition, data["delayed_velocity"],
                        data["shuffled_velocity"], data["channel_noise"])
        return x, data["target"], data


def phase_error(outputs, target):
    estimate = torch.atan2(outputs[..., 1], outputs[..., 0])
    actual = torch.atan2(target[..., 1], target[..., 0])
    return wrap_angle(estimate - actual)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = RingConfig(**checkpoint["config"])
    model = RingRNN(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, RingTask(config, device), checkpoint
