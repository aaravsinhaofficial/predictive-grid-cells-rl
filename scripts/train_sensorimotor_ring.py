"""Train a clearly engineered goal-indexed sensorimotor attractor.

DESIGNED: periodic code basis, quadratic decoder features, goal-memory ring,
sinusoidal feedback law. LEARNED: decoder weights and navigation-optimal feedback
gain/preview mixture. The neural memory ring itself is identical in every arm.

Prediction can change transverse stability of the full agent/environment loop.
It is not claimed to strengthen the autonomous neural memory ring.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ring_mechanism_theory import feedback_delay_limit


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass
class Config:
    dt: float = .02
    delay_steps: int = 15
    code_horizon: float = .20
    n_code: int = 64
    decoder_examples: int = 8192
    training_episodes: int = 64
    evaluation_episodes: int = 512
    duration: float = 8.
    phase_noise: float = .03
    velocity_noise: float = .05
    drift_noise: float = .05
    motor_limit: float = 8.
    goal_radius: float = .08
    effort_cost: float = .0005
    memory_radial_rate: float = 1.


class Decoder:
    """Learned quadratic readout of periodic spatial cells.

    Products between frequency-2 and frequency-3 modules make frequency 1
    available. Their weights are fitted from labeled localization examples.
    The fast evaluator is an exact algebraic compilation of encode -> readout,
    independently checked against actual code arrays before any experiment.
    """
    def __init__(self, frequency, phase, selected, examples, seed):
        self.frequency, self.phase = frequency.copy(), phase.copy()
        self.first = selected[frequency[selected] == 2]
        self.second = selected[frequency[selected] == 3]
        gen = np.random.default_rng(seed)
        physical = gen.uniform(-np.pi, np.pi, examples)
        measured = physical + .03*gen.normal(size=examples)
        target = np.column_stack((np.cos(physical), np.sin(physical)))
        features = self.features(self.encode(measured))
        self.weights, _, self.rank, _ = np.linalg.lstsq(features, target, rcond=1e-10)
        self.cosine, self.sine = np.zeros((8, 2)), np.zeros((8, 2))
        self.cosine[0] = self.weights[0]
        index = 1
        for group, frequency_value in ((self.first, 2), (self.second, 3)):
            for cell in group:
                self.cosine[frequency_value] += self.weights[index]*np.cos(phase[cell])
                self.sine[frequency_value] -= self.weights[index]*np.sin(phase[cell])
                index += 1
        for a in self.first:
            for b in self.second:
                for frequency_value, offset in ((1, phase[b]-phase[a]), (5, phase[a]+phase[b])):
                    self.cosine[frequency_value] += .5*self.weights[index]*np.cos(offset)
                    self.sine[frequency_value] -= .5*self.weights[index]*np.sin(offset)
                index += 1
        heldout = gen.uniform(-20, 20, 4096)
        self.compilation_error = float(np.abs(self.from_code(self.encode(heldout))-self.compiled(heldout)).max())
        if self.compilation_error > 1e-10:
            raise AssertionError(self.compilation_error)
        self.heldout_clean_mae_deg = float(np.abs(wrap(self.angle(heldout)-heldout)).mean()*180/np.pi)
        self.training_data_sha256 = hashlib.sha256(np.column_stack((physical, measured)).tobytes()).hexdigest()

    def encode(self, measured_phase):
        return np.cos(np.asarray(measured_phase)[..., None]*self.frequency+self.phase)

    def features(self, code):
        a, b = code[..., self.first], code[..., self.second]
        return np.concatenate((np.ones((*a.shape[:-1], 1)), a, b,
                               (a[..., :, None]*b[..., None, :]).reshape(*a.shape[:-1], -1)), axis=-1)

    def from_code(self, code):
        return self.features(code)@self.weights

    def compiled(self, measured_phase):
        phase = np.asarray(measured_phase)
        output = np.broadcast_to(self.cosine[0], (*phase.shape, 2)).copy()
        for frequency in (1, 2, 3, 5):
            output += np.cos(frequency*phase)[..., None]*self.cosine[frequency]
            output += np.sin(frequency*phase)[..., None]*self.sine[frequency]
        return output

    def angle(self, measured_phase):
        output = self.compiled(measured_phase)
        return np.arctan2(output[..., 1], output[..., 0])

    def summary(self):
        return dict(rank=int(self.rank), compilation_error=self.compilation_error,
                    heldout_clean_mae_deg=self.heldout_clean_mae_deg,
                    training_data_sha256=self.training_data_sha256)


def build_decoders(config, seed):
    if config.n_code % 4:
        raise ValueError("Code size must divide into four modules")
    gen = np.random.default_rng(seed)
    module_size = config.n_code//4
    frequency = np.repeat([2, 3, 5, 7], module_size)
    phase = np.concatenate([np.arange(module_size)*2*np.pi/module_size+gen.uniform(0, 2*np.pi)
                            for _ in range(4)])
    mask = np.zeros(config.n_code, dtype=bool)
    for module, count in enumerate((6, 5, 5, 5)):
        if module_size < count:
            raise ValueError("This controlled population requires at least 64 cells")
        mask[module*module_size+gen.permutation(module_size)[:count]] = True
    ordinary = Decoder(frequency, phase, np.flatnonzero(~mask), config.decoder_examples, seed+179)
    forecast = Decoder(frequency, phase, np.flatnonzero(mask), config.decoder_examples, seed+179)
    # Identical supervised samples and harmonic function span; decoder errors are
    # therefore shared rather than being condition-specific training advantages.
    grid = np.linspace(-np.pi, np.pi, 1024)
    agreement = float(np.max(np.abs(wrap(ordinary.angle(grid)-forecast.angle(grid)))))
    assert agreement < 1e-8, agreement
    return ordinary, forecast, dict(frequency=frequency, phase=phase, predictive_mask=mask,
                                   decoded_phase_agreement=agreement)


def load_fitted_decoders(path):
    """Load saved decoder weights for new tests without refitting anything."""
    with np.load(path) as data:
        decoders = []
        for name, selected in (("ordinary", ~data["predictive_mask"]),
                               ("forecast", data["predictive_mask"])):
            decoder = Decoder.__new__(Decoder)
            decoder.frequency, decoder.phase = data["frequency"].copy(), data["phase"].copy()
            indices = np.flatnonzero(selected)
            decoder.first = indices[decoder.frequency[indices] == 2]
            decoder.second = indices[decoder.frequency[indices] == 3]
            decoder.weights = data[name+"_weights"].copy()
            decoder.cosine, decoder.sine = data[name+"_cosine"].copy(), data[name+"_sine"].copy()
            decoders.append(decoder)
    return tuple(decoders)


def experience(config, episodes, seed):
    gen = np.random.default_rng(seed)
    steps = int(round(config.duration/config.dt))
    goal = gen.uniform(-np.pi, np.pi, episodes)
    initial_error = gen.uniform(.5, 2.5, episodes)*gen.choice([-1, 1], episodes)
    drift = np.zeros((steps, episodes))
    innovation = gen.normal(size=(steps, episodes))
    decay = np.exp(-config.dt/.5)
    for t in range(1, steps):
        drift[t] = decay*drift[t-1]+config.drift_noise*np.sqrt(1-decay**2)*innovation[t]
    return dict(goal=goal, initial=goal+initial_error,
                phase_noise=config.phase_noise*gen.normal(size=(steps, episodes)),
                velocity_noise=config.velocity_noise*gen.normal(size=(steps, episodes)), drift=drift)


def cue_readouts(ordinary, forecast, phase, velocity, condition, horizon):
    """Only the encoder sees physical observations; the controller sees readouts.

    phase and velocity are noisy measurements at the same acquisition time.
    No future state or internal memory-error derivative is used.
    """
    a = ordinary.angle(phase)
    if condition in ("predictive", "restored"):
        preview = phase+horizon*velocity
    elif condition == "reversed":
        preview = phase-horizon*velocity
    elif condition in ("grid", "velocity", "removed"):
        preview = phase
    else:
        raise ValueError(condition)
    return a, forecast.angle(preview)


def correction(a, b, measured_velocity, goal_memory, gain, mixture, condition, horizon):
    if condition == "velocity":
        estimate = a+mixture*horizon*measured_velocity
    else:
        estimate = a+mixture*wrap(b-a)
    return -gain*np.sin(estimate-goal_memory)


def memory_step(state, dt, radial_rate):
    """Designed autonomous ring: neutral phase, attracting unit radius."""
    return state+dt*radial_rate*(1-np.sum(state**2, axis=-1, keepdims=True))*state


def simulate(config, ordinary, forecast, condition, gains, mixtures, data, *, record=False):
    gains, mixtures = np.atleast_1d(gains).astype(float), np.atleast_1d(mixtures).astype(float)
    assert gains.shape == mixtures.shape
    candidates, episodes = len(gains), len(data["goal"])
    goal = data["goal"][None, :]
    # One encoded goal cue initializes the designed autonomous memory ring.
    # Its phase is then constant, identically in all conditions; radius=1.
    goal_memory = ordinary.angle(goal)
    memory_state = np.stack((np.cos(goal_memory), np.sin(goal_memory)), axis=-1)
    x = np.broadcast_to(data["initial"], (candidates, episodes)).copy()
    velocity = np.zeros_like(x)
    length = config.delay_steps+1
    a0, b0 = cue_readouts(ordinary, forecast, x+data["phase_noise"][0],
                         velocity+data["velocity_noise"][0], condition, config.code_horizon)
    a_buffer = np.repeat(a0[None], length, axis=0)
    b_buffer = np.repeat(b0[None], length, axis=0)
    v_buffer = np.repeat(np.broadcast_to(data["velocity_noise"][0], x.shape)[None], length, axis=0)
    total_error, total_effort = np.zeros_like(x), np.zeros_like(x)
    first_entry = np.full_like(x, config.duration)
    last_outside = np.zeros_like(x)
    ever_entered = np.zeros_like(x, dtype=bool)
    exited_after_entry = np.zeros_like(x, dtype=bool)
    overshoot = np.zeros_like(x)
    start_sign = np.sign(wrap(x-goal))
    tail_squared = np.zeros_like(x)
    steps = len(data["drift"])
    tail_start = int(.75*steps)
    traces = []
    for t in range(steps):
        memory_state = memory_step(memory_state, config.dt, config.memory_radial_rate)
        goal_memory = np.arctan2(memory_state[..., 1], memory_state[..., 0])
        measured_phase = x+data["phase_noise"][t]
        measured_velocity = velocity+data["velocity_noise"][t]
        a, b = cue_readouts(ordinary, forecast, measured_phase, measured_velocity,
                            condition, config.code_horizon)
        a_buffer[t%length], b_buffer[t%length], v_buffer[t%length] = a, b, measured_velocity
        index = (t-config.delay_steps)%length
        command = correction(a_buffer[index], b_buffer[index], v_buffer[index], goal_memory,
                             gains[:, None], mixtures[:, None], condition, config.code_horizon)
        command = np.clip(command, -config.motor_limit, config.motor_limit)
        velocity = command+data["drift"][t]
        x = x+config.dt*velocity
        error = wrap(x-goal)
        inside = np.abs(error) <= config.goal_radius
        last_outside[~inside] = (t+1)*config.dt
        newly = inside & ~ever_entered
        first_entry[newly] = (t+1)*config.dt
        exited_after_entry |= ever_entered & ~inside
        ever_entered |= inside
        overshoot = np.maximum(overshoot, np.maximum(-start_sign*error, 0))
        total_error += 1-np.cos(error)
        total_effort += command**2
        if t >= tail_start:
            tail_squared += error**2
        if record:
            traces.append(error.astype(np.float32))
    objective = (total_error+config.effort_cost*total_effort)/steps
    result = dict(objective=objective, mean_objective=objective.mean(1), first_entry=first_entry,
                  settling_time=np.minimum(config.duration, last_outside+config.dt),
                  entered=ever_entered, exited_after_entry=exited_after_entry, max_overshoot=overshoot,
                  tail_rms=np.sqrt(tail_squared/(steps-tail_start)), final_error=error)
    if record:
        result["error_trace"] = np.asarray(traces)
    return result


def controller_candidates(condition):
    # Every condition receives exactly 108 evaluations on the same experiences.
    # Ordinary-only can use all 108 to tune its gain finely, rather than being
    # forced to use the higher gain chosen for a forecast controller.
    if condition == "grid":
        return np.linspace(.4, 8., 108), np.zeros(108)
    gain, mixture = np.meshgrid(np.linspace(.4, 8., 12), np.linspace(0, 1, 9), indexing="ij")
    return gain.ravel(), mixture.ravel()


def local_loop_spectrum(config, ordinary, forecast, condition, gain, mixture, goal=0.):
    """Measured learned readout slopes -> exact discrete delay characteristic.

    Position is measured after the previous action. Hence v[t]=(x[t]-x[t-1])/dt.
    Lambda^(d+2)-Lambda^(d+1)+(dt*kx+kv)*Lambda-kv=0.
    This is the transverse physical-position mode. The goal-memory tangent
    eigenvalue remains exactly 1 in every condition.
    """
    memory = ordinary.angle(goal)
    def u(position, speed):
        a, b = cue_readouts(ordinary, forecast, np.asarray(position), np.asarray(speed),
                            condition, config.code_horizon)
        return float(correction(a, b, speed, memory, gain, mixture, condition, config.code_horizon))
    eps = 1e-5
    kx = -(u(goal+eps, 0)-u(goal-eps, 0))/(2*eps)
    kv = -(u(goal, eps)-u(goal, -eps))/(2*eps)
    coefficients = np.zeros(config.delay_steps+3)
    coefficients[0], coefficients[1] = 1., -1.
    coefficients[-2] += config.dt*kx+kv
    coefficients[-1] -= kv
    roots = np.roots(coefficients)
    radius = float(np.abs(roots).max())
    effective_horizon = kv/kx if kx > 0 else np.nan
    return dict(spectral_radius=radius, exponent_per_second=float(np.log(radius)/config.dt),
                local_position_gain=kx, local_velocity_gain=kv,
                effective_horizon=effective_horizon,
                continuous_delay_limit=feedback_delay_limit(kx, effective_horizon) if kx > 0 else 0.,
                neural_memory_tangent_eigenvalue=1.,
                neural_memory_radial_eigenvalue=1-2*config.memory_radial_rate*config.dt,
                roots_real=roots.real.tolist(), roots_imag=roots.imag.tolist())


def delay_state_step(config, ordinary, forecast, condition, gain, mixture, state):
    """Independent full deterministic state map, used for Jacobian validation.

    state = [goal-ring x/y, physical position now, position one step ago, ...].
    The measured velocity is completed physical displacement, never a derivative
    of internal neural error. The first two coordinates are actual neural state.
    """
    state = np.asarray(state)
    memory = memory_step(state[:2], config.dt, config.memory_radial_rate)
    physical = state[2:]
    phase = physical[config.delay_steps]
    velocity = (physical[config.delay_steps]-physical[config.delay_steps+1])/config.dt
    a, b = cue_readouts(ordinary, forecast, phase, velocity, condition, config.code_horizon)
    u = correction(a, b, velocity, np.arctan2(memory[1], memory[0]), gain, mixture,
                   condition, config.code_horizon)
    position = physical[0]+config.dt*np.clip(u, -config.motor_limit, config.motor_limit)
    return np.r_[memory, position, physical[:-1]]


def validate_full_jacobian(config, ordinary, forecast, condition, gain, mixture, goal=.713):
    memory_phase = ordinary.angle(goal)
    state = np.r_[np.cos(memory_phase), np.sin(memory_phase),
                  np.full(config.delay_steps+2, goal)]
    fixed_point_error = float(np.max(np.abs(delay_state_step(
        config, ordinary, forecast, condition, gain, mixture, state)-state)))
    eps = 1e-5
    jac = np.column_stack([(delay_state_step(config, ordinary, forecast, condition, gain, mixture, state+eps*d)
                           - delay_state_step(config, ordinary, forecast, condition, gain, mixture, state-eps*d))/(2*eps)
                          for d in np.eye(len(state))])
    spectrum = local_loop_spectrum(config, ordinary, forecast, condition, gain, mixture, goal)
    expected = np.r_[1., 1-2*config.memory_radial_rate*config.dt,
                      np.asarray(spectrum["roots_real"])+1j*np.asarray(spectrum["roots_imag"])]
    actual = np.linalg.eigvals(jac)
    # Symmetric nearest-spectrum distance avoids arbitrary complex-conjugate order.
    error = max(np.min(np.abs(expected[:, None]-actual[None, :]), axis=1).max(),
                np.min(np.abs(actual[:, None]-expected[None, :]), axis=1).max())
    if error > 2e-7 or fixed_point_error > 1e-9:
        raise AssertionError((error, fixed_point_error))
    return dict(full_state_jacobian_spectrum_error=float(error), fixed_point_error=fixed_point_error,
                neutral_eigenvalue_error=float(np.min(np.abs(actual-1))), state_dimension=len(state))


def metrics(result):
    return dict(objective=float(result["mean_objective"][0]),
                first_entry_seconds=float(result["first_entry"][0].mean()),
                settling_seconds=float(result["settling_time"][0].mean()),
                reached_fraction=float(result["entered"][0].mean()),
                exited_after_entry_fraction=float(result["exited_after_entry"][0].mean()),
                max_overshoot_deg=float(result["max_overshoot"][0].mean()*180/np.pi),
                tail_rms_deg=float(np.sqrt(np.mean(result["tail_rms"][0]**2))*180/np.pi))


def train_seed(config, seed, out):
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out/f"seed_{seed}.json"
    if target.exists():
        raise FileExistsError(target)
    ordinary, forecast, basis = build_decoders(config, seed)
    training = experience(config, config.training_episodes, seed+70103)
    evaluation = experience(config, config.evaluation_episodes, seed+907103)
    models, arrays = {}, {}
    for condition in ("grid", "predictive", "velocity"):
        gains, mixtures = controller_candidates(condition)
        candidates = simulate(config, ordinary, forecast, condition, gains, mixtures, training)
        selected = int(np.argmin(candidates["mean_objective"]))
        gain, mixture = float(gains[selected]), float(mixtures[selected])
        result = simulate(config, ordinary, forecast, condition, [gain], [mixture], evaluation, record=True)
        models[condition] = dict(gain=gain, mixture=mixture, horizon=mixture*config.code_horizon,
            selected_candidate=selected, training_objective=float(candidates["mean_objective"][selected]),
            metrics=metrics(result), spectrum=local_loop_spectrum(config, ordinary, forecast, condition, gain, mixture),
            candidate_gains=gains.tolist(), candidate_mixtures=mixtures.tolist(),
            candidate_objectives=candidates["mean_objective"].tolist())
        models[condition]["full_jacobian_validation"] = validate_full_jacobian(
            config, ordinary, forecast, condition, gain, mixture)
        arrays[f"{condition}_error_trace"] = result["error_trace"]
        for key in ("objective", "first_entry", "settling_time", "entered", "exited_after_entry", "max_overshoot", "tail_rms"):
            arrays[f"{condition}_{key}"] = result[key]
        print(json.dumps(dict(seed=seed, condition=condition, gain=gain, mixture=mixture,
                              metrics=models[condition]["metrics"])), flush=True)
    pgc = models["predictive"]
    interventions = {}
    for condition in ("removed", "restored", "reversed"):
        result = simulate(config, ordinary, forecast, condition, [pgc["gain"]], [pgc["mixture"]], evaluation)
        interventions[condition] = dict(metrics=metrics(result), spectrum=local_loop_spectrum(
            config, ordinary, forecast, condition, pgc["gain"], pgc["mixture"]))
        arrays[f"{condition}_objective"] = result["objective"]
        arrays[f"{condition}_first_entry"] = result["first_entry"]
        arrays[f"{condition}_settling_time"] = result["settling_time"]
        arrays[f"{condition}_tail_rms"] = result["tail_rms"]
    assert interventions["restored"]["metrics"] == models["predictive"]["metrics"]
    assert np.array_equal(arrays["restored_objective"], arrays["predictive_objective"])
    # Sweep delay only after training. No controller or decoder is retuned.
    sweep = []
    for delay_steps in range(0, 41, 2):
        changed = Config(**{**asdict(config), "delay_steps": delay_steps})
        for condition in ("grid", "predictive", "velocity", "removed"):
            fitted = models[condition] if condition in models else models["predictive"]
            spectrum = local_loop_spectrum(changed, ordinary, forecast, condition,
                                            fitted["gain"], fitted["mixture"])
            sweep.append(dict(condition=condition, delay=delay_steps*config.dt,
                              spectral_radius=spectrum["spectral_radius"],
                              exponent_per_second=spectrum["exponent_per_second"]))
    # Small held-out behavior titrations, with no retuning, test beyond the training
    # latency and preview horizon. All arms share each underlying realization.
    titration_data = experience(config, 128, seed+1907103)
    behavior_sweep = []
    for delay_steps in (0, 5, 10, 15, 20, 25, 30):
        changed = Config(**{**asdict(config), "delay_steps": delay_steps})
        for condition in ("grid", "predictive", "velocity", "removed"):
            fitted = models[condition] if condition in models else pgc
            nav = simulate(changed, ordinary, forecast, condition, [fitted["gain"]], [fitted["mixture"]], titration_data)
            spec = local_loop_spectrum(changed, ordinary, forecast, condition, fitted["gain"], fitted["mixture"])
            behavior_sweep.append(dict(condition=condition, delay=delay_steps*config.dt,
                                       metrics=metrics(nav), spectral_radius=spec["spectral_radius"]))
    horizon_sweep = []
    for factor in (-1., 0., .25, .5, .75, 1., 1.25, 1.5, 2., 3.):
        changed = Config(**{**asdict(config), "code_horizon": config.code_horizon*factor})
        nav = simulate(changed, ordinary, forecast, "predictive", [pgc["gain"]], [pgc["mixture"]], titration_data)
        spec = local_loop_spectrum(changed, ordinary, forecast, "predictive", pgc["gain"], pgc["mixture"])
        horizon_sweep.append(dict(factor=factor, code_horizon=changed.code_horizon,
                                  effective_horizon=spec["effective_horizon"],
                                  metrics=metrics(nav), spectral_radius=spec["spectral_radius"]))
    result = dict(seed=seed, config=asdict(config), models=models, interventions=interventions,
                  delay_sweep=sweep, decoder_ordinary=ordinary.summary(), decoder_forecast=forecast.summary(),
                  behavior_delay_sweep=behavior_sweep, horizon_sweep=horizon_sweep,
                  decoded_phase_agreement=basis["decoded_phase_agreement"],
                  learned=["quadratic spatial-code decoder weights", "feedback gain", "forecast mixture"],
                  designed=["periodic basis", "quadratic features", "goal-memory ring", "sinusoidal feedback law"],
                  scope="goal-indexed sensorimotor attractor; intrinsic neural memory stability is unchanged",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=time.monotonic()-started)
    arrays.update(basis)
    arrays.update(ordinary_weights=ordinary.weights, forecast_weights=forecast.weights,
                  ordinary_cosine=ordinary.cosine, ordinary_sine=ordinary.sine,
                  forecast_cosine=forecast.cosine, forecast_sine=forecast.sine)
    np.savez_compressed(out/f"seed_{seed}.npz", **arrays)
    target.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(seed=seed, seconds=result["seconds"], interventions=interventions)), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[2100])
    p.add_argument("--out", type=Path, default="results/ring_attractor/sensorimotor_pilot")
    p.add_argument("--training-episodes", type=int, default=64)
    p.add_argument("--evaluation-episodes", type=int, default=512)
    p.add_argument("--delay-steps", type=int, default=15)
    p.add_argument("--duration", type=float, default=8.)
    a = p.parse_args()
    cfg = Config(training_episodes=a.training_episodes, evaluation_episodes=a.evaluation_episodes,
                 delay_steps=a.delay_steps, duration=a.duration)
    for training_seed in a.seeds:
        train_seed(cfg, training_seed, a.out)
