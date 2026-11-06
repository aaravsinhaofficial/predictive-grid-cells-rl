"""Independent calibration on Marraffa et al.'s published, trained 256D RNN.

Reads numeric weights only. Does not import or execute downloaded source, fit an
angle decoder, or change the navigation models. The protocol is written before
any simulated outcome. The reported heteroclinic circle is traced from the two
unstable branches of a full-network saddle, not fitted to an angular data set.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.linalg import null_space
from scipy.optimize import least_squares, root
from scipy.special import expit
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "results/ring_nonlinear_design/vendor/marraffa2026/mtRNN"
OUT = ROOT / "results/ring_nonlinear_design/published_benchmark"


def polynomial(q, coefficients, powers):
    q = np.asarray(q)
    return np.prod(q[..., None, :] ** powers, axis=-1) @ coefficients.T


def polynomial_jacobian(q, coefficients, powers):
    result = []
    for j in range(q.shape[-1]):
        reduced = powers.copy().astype(int)
        factor = reduced[:, j].copy()
        reduced[:, j] = np.maximum(0, reduced[:, j] - 1)
        result.append(polynomial(q, coefficients * factor, reduced))
    return np.stack(result, axis=-1)


class Model:
    def __init__(self):
        with np.load(VENDOR / "model_params.npz", allow_pickle=False) as arrays:
            w = arrays["w_in"].T.astype(np.float64)
            self.recurrent = w[:, 20:]
            u = np.zeros(20)
            u[[0, 6]] = 1
            self.bias = w[:, :20] @ u + arrays["b_in"]
            self.output = arrays["w_out"].T.astype(np.float64)
        self.step_size = .02

    def velocity(self, x):
        return (-x + np.logaddexp(0., x @ self.recurrent.T + self.bias)) / .1

    def step(self, x):
        return x + self.step_size * self.velocity(x)

    def jacobian(self, x, discrete=False):
        jac = (-np.eye(256) + expit(self.recurrent @ x + self.bias)[:, None]
               * self.recurrent) / .1
        return np.eye(256) + self.step_size * jac if discrete else jac


def integrate(model, starts, duration, kind, sample_dt=.1):
    """Independent full dynamics; ODE uses adaptive high-accuracy integration."""
    starts = np.atleast_2d(starts)
    if kind == "ode":
        times = np.linspace(0, duration, round(duration / sample_dt) + 1)
        sol = solve_ivp(lambda _t, y: model.velocity(y.reshape(starts.shape)).ravel(),
                        (0, duration), starts.ravel(), t_eval=times,
                        method="DOP853", rtol=2e-9, atol=2e-11)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y.T.reshape(len(times), *starts.shape).transpose(1, 0, 2)
    steps = round(duration / model.step_size)
    stride = max(1, round(sample_dt / model.step_size))
    history, state = [starts.copy()], starts.copy()
    for i in range(steps):
        state = model.step(state)
        if (i + 1) % stride == 0:
            history.append(state.copy())
    return np.stack(history, axis=1)


def arc_resample(curve, count):
    following = np.roll(curve, -1, axis=0)
    edges = np.linalg.norm(following - curve, axis=1)
    valid = edges > 1e-12
    curve, following, edges = curve[valid], following[valid], edges[valid]
    boundaries = np.r_[0., np.cumsum(edges)]
    loc = (np.arange(count) + .5) / count * boundaries[-1]
    ids = np.minimum(np.searchsorted(boundaries, loc, side="right") - 1, len(edges) - 1)
    weight = (loc - boundaries[ids]) / edges[ids]
    return curve[ids] + weight[:, None] * (following[ids] - curve[ids])


def distance_to_curve(points, curve):
    """Nearest local polygon segment, with dense-curve KD-tree candidate lookup."""
    points = np.atleast_2d(points)
    _, ids = cKDTree(curve).query(points, k=min(8, len(curve)))
    ids = np.atleast_2d(ids)
    candidates = np.concatenate((ids, (ids - 1) % len(curve)), axis=1)
    a = curve[candidates]
    b = curve[(candidates + 1) % len(curve)]
    edge = b - a
    fraction = np.sum((points[:, None] - a) * edge, axis=-1) / np.maximum(
        np.sum(edge * edge, axis=-1), 1e-30)
    nearest = a + np.clip(fraction, 0., 1.)[..., None] * edge
    return np.min(np.linalg.norm(points[:, None] - nearest, axis=-1), axis=1)


def qualify_points(model, originals):
    records, refined = [], []
    for label, original in originals:
        result = root(model.velocity, original, jac=model.jacobian,
                      method="hybr", options={"xtol": 1e-11, "maxfev": 600})
        candidate = result.x if np.linalg.norm(model.velocity(result.x)) < np.linalg.norm(
            model.velocity(original)) else original.copy()
        eig = np.linalg.eigvals(model.jacobian(candidate))
        records.append({"label": label, "original_residual_rms": float(np.sqrt(np.mean(
            model.velocity(original) ** 2))), "residual_rms": float(np.sqrt(np.mean(
            model.velocity(candidate) ** 2))), "refinement_distance": float(np.linalg.norm(
            candidate - original)), "unstable_dimensions": int((eig.real > 1e-7).sum()),
            "leading_real_eigenvalues": np.sort(eig.real)[::-1][:8].tolist()})
        refined.append(candidate)
    return records, refined


def transverse_map_gain(model, starts, duration=1.):
    """Worst Euclidean normal gain after quotienting induced phase, exact Euler map.

    Every normal direction is included via a full variational matrix. The output
    tangent is advected by that same derivative, so phase shifts are not called
    failures to recover.
    """
    records = []
    for x0 in starts:
        tangent = model.velocity(x0)
        tangent /= np.linalg.norm(tangent)
        normals = null_space(tangent[None])
        derivative, x = np.eye(256), x0.copy()
        for _ in range(round(duration / model.step_size)):
            derivative = model.jacobian(x, discrete=True) @ derivative
            x = model.step(x)
        advanced = derivative @ tangent
        tangent_gain = np.linalg.norm(advanced)
        unit = advanced / tangent_gain
        normal_derivative = derivative @ normals
        normal_derivative -= unit[:, None] * (unit @ normal_derivative)[None]
        worst = np.linalg.svd(normal_derivative, compute_uv=False)[0]
        records.append({"tangent_gain": float(tangent_gain),
                        "worst_transverse_gain": float(worst),
                        "normal_to_tangent_ratio": float(worst / tangent_gain)})
    return records


def stored_rom_check(model, anchor, full_trajectories):
    im, rom = loadmat(VENDOR / "Data/IM22.mat"), loadmat(VENDOR / "Data/ROM22.mat")
    coefficients = np.concatenate((im["tangentspace"], im["mfld_coeff"]), axis=1)
    powers, rd, rp = im["exp"], rom["rdcoeff"], rom["rdexps"]
    lift = lambda q: anchor + polynomial(q, coefficients, powers)
    reduced = lambda q: polynomial(q, rd, rp)
    # Held-out full-state branch states; no time-specific or angle-specific decoder.
    starts = np.concatenate([arc_resample(branch, 12) for branch in full_trajectories])
    scale = np.sqrt(np.mean(np.sum((starts - starts.mean(0)) ** 2, axis=1)))
    records = []
    for x in starts:
        q0 = np.linalg.lstsq(im["tangentspace"], x - anchor, rcond=None)[0]
        fit = least_squares(lambda q: lift(q) - x, q0,
                            jac=lambda q: polynomial_jacobian(q, coefficients, powers),
                            max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        q = fit.x
        invariance = model.velocity(lift(q)) - polynomial_jacobian(q, coefficients, powers) @ reduced(q)
        ode = solve_ivp(lambda t, y: reduced(y), (0, 1.), q,
                        method="DOP853", rtol=1e-9, atol=1e-11)
        pred = lift(ode.y[:, -1])
        full_ode = integrate(model, x, 1., "ode")[0, -1]
        full_map = integrate(model, x, 1., "map")[0, -1]
        records.append({"fit_error_fraction": float(np.linalg.norm(lift(q) - x) / scale),
                        "invariance_velocity_fraction_per_second": float(np.linalg.norm(invariance) / scale),
                        "ode_1s_forecast_error_fraction": float(np.linalg.norm(pred - full_ode) / scale),
                        "map_1s_forecast_error_fraction": float(np.linalg.norm(pred - full_map) / scale)})
    return {"validation_states": len(records), "state_scale": float(scale),
            "median": {key: float(np.median([r[key] for r in records])) for key in records[0]},
            "maximum": {key: float(np.max([r[key] for r in records])) for key in records[0]},
            "records": records}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=400.)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"purpose": "Calibrate ring detection on published trained weights before native navigation LSTMs",
        "source_commit": "f6702734490c90001db73cd7b7900998fb9a87cf",
        "input": "u[0]=u[6]=1; remaining18 zero", "state": "all256 native coordinates",
        "fixed_point_solver": "hybr exact Jacobian maxfev600; retain originals",
        "branch_trace": {"initial_displacement": 1e-4, "duration_seconds": args.duration,
                         "sample_dt": .1, "integrators": ["DOP853", "original Euler map dt=.02"]},
        "validation": "24 held-out full-state branch samples, stored cubic lift/ODE, no angular labels",
        "recovery": "16 arc samples; radial and 3 random normal directions, both signs; .005 loop RMS kick;1second",
        "normal_spectrum": "8 arc samples, all255 normal directions, exact discrete variational map1second",
        "negative_control": "isotropically contracting plane: same apparent circle, no preferential normal recovery",
        "closure_requirement": "Both traced endpoints within.001 loop RMS of stable point; no inferred bridge counts as evidence",
        "interpretation": "A positive learned-network calibration is not a result about navigation LSTMs"}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = Model()
    fp = np.load(VENDOR / "Data/fpts.npy", allow_pickle=False)
    all_fp = np.load(VENDOR / "Data/allFpts.npy", allow_pickle=False)
    records, refined = qualify_points(model, [("interior", fp[0]), ("stable", fp[1]), ("saddle", all_fp[0])])
    anchor, stable, saddle = refined
    # Derivative tests are independent of the author implementation.
    rng = np.random.default_rng(127)
    check = saddle + rng.normal(size=256) * .01
    vector = rng.normal(size=256); vector /= np.linalg.norm(vector)
    fd = (model.velocity(check + 1e-6 * vector) - model.velocity(check - 1e-6 * vector)) / 2e-6
    analytic = model.jacobian(check) @ vector
    import torch
    torch.set_num_threads(1)
    w, b = torch.tensor(model.recurrent), torch.tensor(model.bias)
    jt = torch.autograd.functional.jacobian(lambda x: (-x + torch.nn.functional.softplus(w @ x + b)) / .1,
                                           torch.tensor(check))
    derivative_checks = {"finite_difference_relative_error": float(np.linalg.norm(fd-analytic)/np.linalg.norm(fd)),
                         "autograd_max_absolute_error": float(np.max(np.abs(jt.numpy()-model.jacobian(check))))}
    eigenvalues, eigenvectors = np.linalg.eig(model.jacobian(saddle))
    v = eigenvectors[:, np.argmax(eigenvalues.real)].real
    v /= np.linalg.norm(v)
    starts = saddle + np.array([-1, 1])[:, None] * 1e-4 * v
    traced, summaries = {}, {}
    for kind in ("ode", "map"):
        trajectories = integrate(model, starts, args.duration, kind)
        traced[kind] = trajectories
        curve = np.concatenate((saddle[None], trajectories[0], stable[None], trajectories[1, ::-1]))
        sample = arc_resample(curve, 16)
        scale = np.sqrt(np.mean(np.sum((sample-sample.mean(0))**2, axis=1)))
        tangent = model.velocity(sample)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        directions = rng.normal(size=(16, 4, 256))
        directions[:, 0] = sample - sample.mean(0)
        directions -= np.sum(directions * tangent[:, None], axis=-1)[..., None] * tangent[:, None]
        directions /= np.linalg.norm(directions, axis=-1)[..., None]
        eps = .005 * scale
        perturbed = sample[:, None, None] + eps * np.array([-1, 1])[None, :, None, None] * directions[:, None]
        combined = np.concatenate((sample, perturbed.reshape(-1, 256)))
        evolved = integrate(model, combined, 1., kind)[:, -1]
        initial = distance_to_curve(perturbed.reshape(-1, 256), curve)
        final = distance_to_curve(evolved[16:], curve)
        endpoints = np.linalg.norm(trajectories[:, -1] - stable, axis=1) / scale
        summaries[kind] = {"loop_rms": float(scale), "endpoint_distance_fraction": endpoints.tolist(),
            "closure_qualified": bool(np.max(endpoints) < .001),
            "max_polygon_edge_fraction": float(np.linalg.norm(np.roll(curve,-1,axis=0)-curve,axis=1).max()/scale),
            "unperturbed_1s_invariance_error_fraction_max": float(distance_to_curve(evolved[:16],curve).max()/scale),
            "normal_kick_initial_distance_fraction_min": float(initial.min()/eps),
            "normal_kick_return_gain_median": float(np.median(final/initial)),
            "normal_kick_return_gain_max": float(np.max(final/initial))}
        if kind == "map":
            summaries[kind]["all_direction_transverse_variational_test"] = transverse_map_gain(
                model, arc_resample(curve, 8))
        np.savez_compressed(OUT / f"{kind}_trajectories.npz", branches=trajectories,
                            loop=arc_resample(curve,512), probes=sample, probe_final=evolved[:16],
                            perturbed_initial=perturbed, perturbed_final=evolved[16:])
        print(kind, json.dumps({k:v for k,v in summaries[kind].items() if k != "all_direction_transverse_variational_test"}), flush=True)
    stored = stored_rom_check(model, anchor, traced["ode"])
    # Analytic negative: a circle of initial conditions under x -> exp(-t)x.
    # Both radial and tangent perturbations shrink identically; the set moves inward.
    negative = {"system": "dx/dt=-x in2D", "1s_tangent_gain": float(np.exp(-1)),
        "1s_normal_gain": float(np.exp(-1)), "normal_to_tangent_ratio": 1.,
        "circle_1s_invariance_error_fraction": float(1-np.exp(-1)),
        "fixed_points": 1, "saddles": 0, "heteroclinic_circle": False}
    output = {"protocol": protocol, "derivative_checks": derivative_checks,
              "fixed_points": records, "full_network": summaries,
              "stored_cubic_model": stored, "negative_control": negative}
    (OUT / "summary.json").write_text(json.dumps(output,indent=2)+"\n")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), VENDOR/"model_params.npz", VENDOR/"Data/fpts.npy",
                        VENDOR/"Data/allFpts.npy", VENDOR/"Data/IM22.mat", VENDOR/"Data/ROM22.mat"]}
    (OUT / "source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    (OUT / "executed_source.py").write_text(Path(__file__).read_text())
    print(json.dumps({"derivatives":derivative_checks,"stored_rom_median":stored["median"]}),flush=True)


if __name__ == "__main__":
    main()
