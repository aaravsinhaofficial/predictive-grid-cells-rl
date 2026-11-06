"""Explicit 64D continuous-attractor memory with learned spatial cue readouts.

The circle and transverse contraction are DESIGNED. Spatial-phase readouts and
cue gain are learned. This is separate from every unconstrained RNN cohort.
"""
import math
import numpy as np
import torch
from torch import nn

from decode_predictive_phase_difference import PhaseDifferenceDecoder
from ring_heading_model import HeadingConfig, HeadingTask


KINDS = ("grid", "predictive", "grid_heading")
GAIN_GRID = (0., .25, .5, 1., 2., 3., 4., 6., 8., 12., 16.)


class TorchPhaseDecoder(nn.Module):
    """The original fitted phase readout/cancellation, without pose inputs."""
    def __init__(self, decoder):
        super().__init__()
        self.modules_used = decoder.modules
        self.register_buffer("inverse", torch.as_tensor(decoder.inverse, dtype=torch.float32))
        for m in decoder.modules:
            for label in ("current", "predictive"):
                self.register_buffer(f"index_{m}_{label}", torch.as_tensor(decoder.indices[m, label]))
                self.register_buffer(f"weight_{m}_{label}", torch.as_tensor(decoder.readouts[m, label], dtype=torch.float32))

    def forward(self, rates):
        differences = []
        for m in self.modules_used:
            values = []
            for label in ("current", "predictive"):
                idx = getattr(self, f"index_{m}_{label}")
                w = getattr(self, f"weight_{m}_{label}")
                values.append(rates.index_select(-1, idx)@w[:-1]+w[-1])
            current, predictive = values
            real = predictive[..., :3]*current[..., :3]+predictive[..., 3:]*current[..., 3:]
            imag = predictive[..., 3:]*current[..., :3]-predictive[..., :3]*current[..., 3:]
            differences.append(torch.atan2(imag, real))
        return torch.cat(differences, -1)@self.inverse.T


class DesignedRing(nn.Module):
    def __init__(self, seed, decoder=None, kind="predictive", gain=0., rho=.90, dt=.05):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        rng = np.random.default_rng(seed+95301)
        basis, _ = np.linalg.qr(rng.normal(size=(64, 2)))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32))
        self.decoder, self.kind = decoder, kind
        self.gain, self.rho, self.dt = float(gain), float(rho), float(dt)

    def initialize(self, direction):
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return direction@self.basis.T

    def readout(self, state):
        return state@self.basis

    def decode_inputs(self, inputs):
        if self.kind == "grid_heading":
            q = inputs[..., -3:-1]*inputs[..., -1:]
        else:
            q = self.decoder(inputs[..., :512])*inputs[..., 514:515]
        # Keep near-zero ordinary-code vectors near zero; never manufacture a
        # unit direction by dividing by their tiny magnitude.
        magnitude = q.norm(dim=-1, keepdim=True)
        return q*torch.clamp(1.5/magnitude.clamp_min(1e-12), max=1.)

    def step_cue(self, state, measured_angular, cue, gain=None):
        """Predict completed motion, then assimilate its current sensory cue."""
        plane = self.readout(state)
        radius = plane.norm(dim=-1)
        if torch.any(radius < 1e-8):
            raise ValueError("The designed ring requires a nonzero initialized phase state")
        theta = torch.atan2(plane[..., 1], plane[..., 0])
        prior_theta = theta+self.dt*measured_angular
        prior = torch.stack((prior_theta.cos(), prior_theta.sin()), -1)
        cross = prior[..., 0]*cue[..., 1]-prior[..., 1]*cue[..., 0]
        theta_next = prior_theta+self.dt*(self.gain if gain is None else gain)*cross
        next_radius = 1+self.rho*(radius-1)
        circle = next_radius[..., None]*torch.stack((theta_next.cos(), theta_next.sin()), -1)
        orthogonal = state-plane@self.basis.T
        return circle@self.basis.T+self.rho*orthogonal

    def step(self, inputs, state):
        return self.step_cue(state, inputs[..., 512], self.decode_inputs(inputs))


@torch.no_grad()
def fit_decoders(seed, n_calibration=4096):
    config = HeadingConfig(n_code=512, code_seed=seed, predictive_fraction=.33,
                           recurrent_noise=.01, initial_heading_steps=4)
    task = HeadingTask(config)
    rng = np.random.default_rng(seed+95311)
    position = rng.uniform(-1.25, 1.25, (n_calibration, 2)).astype(np.float32)
    angle = rng.uniform(-np.pi, np.pi, n_calibration)
    direction = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
    decoders, validation = {}, {}
    for kind in ("grid", "predictive"):
        rates = task.encode(torch.from_numpy(position), torch.from_numpy(direction), kind).numpy()
        # The grid population really represents x in BOTH pools. Targets must
        # match that encoding, not an unavailable future location.
        calibration_displacement = direction if kind == "predictive" else np.zeros_like(direction)
        fitted = PhaseDifferenceDecoder(task.population).fit(rates, position, calibration_displacement)
        decoder = TorchPhaseDecoder(fitted)
        reference = fitted.predict(rates[:256])
        decoded = decoder(torch.from_numpy(rates[:256])).numpy()
        max_error = float(np.max(np.abs(decoded-reference)))
        if max_error > 3e-5:
            raise ValueError(f"Tensor decoder differs from original inference: {max_error}")
        decoders[kind] = decoder
        validation[kind+"_tensor_vs_original_max_error"] = max_error
    decoders["grid_heading"] = None
    return task, decoders, validation


@torch.no_grad()
def fit_gains(seed, task, decoders, length=256, episodes=128):
    """Same fixed candidate budget and task realizations for all three inputs."""
    generator = torch.Generator().manual_seed(seed+95321)
    data = task.realization(length, episodes, generator)
    cue_prefix = data["measured_direction"][:4].mean(0)
    gain = torch.tensor(GAIN_GRID)[:, None]
    models, rows = {}, []
    for kind in KINDS:
        model = DesignedRing(seed, decoders[kind], kind)
        inputs, _, _ = task.batch(length, episodes, generator, kind, realization=data)
        cue = model.decode_inputs(inputs)
        state = model.initialize(cue_prefix)[None].expand(len(GAIN_GRID), -1, -1).clone()
        losses = torch.zeros(len(GAIN_GRID))
        for t in range(4, length):
            state = model.step_cue(state, data["measured_angular_velocity"][t], cue[t], gain)
            state = state+data["recurrent_noise"][t][None]
            output = model.readout(state)
            unit = output/output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            losses += (1-(unit*data["target"][t]).sum(-1)).mean(-1)
        losses /= length-4
        best = int(losses.argmin())  # Sorted candidates choose lower gain on exact ties.
        model.gain = GAIN_GRID[best]
        models[kind] = model
        rows.append({"condition": kind, "selected_gain_per_second": model.gain,
                     "candidate_gain_grid": list(GAIN_GRID), "training_losses": losses.tolist()})
    return models, rows


def analytic_spectrum(gain, rho=.90, dt=.05, cue_magnitude=0.):
    return np.array([1-dt*gain*cue_magnitude]+[rho]*63)
