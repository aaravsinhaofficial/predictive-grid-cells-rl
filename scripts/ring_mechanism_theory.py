"""Unfitted analytical mechanisms: accurate anchoring versus ring stability.

Three distinct mathematical systems are kept separate:
1. A ring phase estimator driven by ordinary and forecast spatial cues.
2. A radial ring normal form, whose cue-off stability is independent of phase lead.
3. A delayed reciprocal phase-correction circuit, where prediction inside the
   feedback loop can stabilize a transverse mode while leaving common phase neutral.

These are proposed explanatory models and positive/negative controls. None is
fitted to, or claimed to be the circuit learned by, the trained RNNs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root


def cue_force(velocity, delay, horizon, predictive_weight=.5, gain=2.):
    """Combine two spatial cues in the frame of the true moving phase.

    theta_dot = omega_hat + Kg*sin(q(t-delay)-theta)
                         + Kp*sin(q(t-delay)+horizon*v-theta).
    predictive_weight is a *coupling fraction*, not a neuron-count fraction.
    """
    if not 0 <= predictive_weight <= 1 or gain < 0 or delay < 0:
        raise ValueError("Invalid coupling fraction, gain, or delay")
    v = np.asarray(velocity)
    return gain * np.exp(-1j * delay * v) * (
        (1-predictive_weight) + predictive_weight*np.exp(1j*horizon*v))


def phase_lock(force, velocity_bias=0.):
    """Exact stable branch of the phase-only model, when locking is possible.

    e_dot = velocity_bias + R*sin(alpha-e).
    e* = alpha + asin(velocity_bias/R), lambda = -sqrt(R^2-bias^2).
    Rotation of a single cue changes alpha, not its restoring exponent.
    """
    force = np.asarray(force)
    strength = np.abs(force)
    bias = np.asarray(velocity_bias)
    locked = strength > np.abs(bias)
    denominator = np.maximum(strength, 1e-300)
    error = np.angle(force) + np.arcsin(np.clip(bias/denominator, -1, 1))
    error = np.arctan2(np.sin(error), np.cos(error))
    exponent = -np.sqrt(np.maximum(strength**2-bias**2, 0))
    return dict(error=np.where(locked, error, np.nan),
                exponent=np.where(locked, exponent, np.nan), locked=locked, strength=strength)


def normal_form_radius(strength, mu=1., beta=1.):
    """Stable aligned radius of z_dot=(mu-beta*|z|^2)z + F.

    This is the co-moving frame with exact velocity integration. At zero input,
    every phase at radius sqrt(mu/beta) is an equilibrium: an imposed analytic ring.
    A fixed nonzero input pins one phase, so its equilibrium is a driven point.
    """
    if mu <= 0 or beta <= 0 or strength < 0:
        raise ValueError("Require mu,beta positive and strength nonnegative")
    r0 = np.sqrt(mu/beta)
    if strength == 0:
        return float(r0)
    upper = max(2*r0, 1.)
    while beta*upper**3-mu*upper-strength <= 0:
        upper *= 2
    return float(brentq(lambda radius: beta*radius**3-mu*radius-strength, r0, upper))


def normal_form_spectrum(force, mu=1., beta=1.):
    strength = float(abs(force))
    radius = normal_form_radius(strength, mu, beta)
    return dict(radius=radius, radial_exponent=mu-3*beta*radius**2,
                tangent_exponent=-strength/radius,
                equilibrium_angle=float(np.angle(force)),
                cue_present=bool(strength > 0))


def normal_form_field(state, force=0j, mu=1., beta=1.):
    state = np.asarray(state)
    return (mu-beta*np.sum(state**2, axis=-1, keepdims=True))*state + np.array([force.real, force.imag])


def stationary_phase_mse(error_bias, restoring_strength, velocity_diffusion=.005,
                         cue_diffusion=.002):
    """Small-error OU approximation, under explicitly independent noises.

    de=-K(e-b)dt+sqrt(2Dv)dWv+K*sqrt(2Dc)dWc gives
    E[e^2]=b^2+Dv/K+K*Dc. Forecast-noise correlation is NOT fitted here.
    """
    k = np.asarray(restoring_strength)
    if np.any(k <= 0):
        raise ValueError("Restoring strength must be positive")
    return np.asarray(error_bias)**2 + velocity_diffusion/k + cue_diffusion*k


def feedback_delay_limit(gain, horizon):
    """First instability boundary of e'=-K[e(t-d)+H*e'(t-d)].

    For |K*H|<1, omega=K/sqrt(1-(KH)^2) at a Hopf crossing and
    d_crit=sqrt(1-(KH)^2)*(pi/2+asin(KH))/K.
    The high-frequency real parts approach log(|KH|)/d: |KH|>=1
    cannot be treated as a robustly stable neutral-delay system.
    """
    if gain <= 0:
        raise ValueError("Positive feedback-correction gain required")
    a = gain*horizon
    if abs(a) >= 1:
        return 0.
    return float(np.sqrt(1-a*a)*(np.pi/2+np.arcsin(a))/gain)


def characteristic_root(gain, horizon, delay, guess=(0., 1.)):
    """Numerical root used to check the analytical boundary, not an all-root audit."""
    def residual(value):
        s = value[0] + 1j*value[1]
        f = s + gain*(1+horizon*s)*np.exp(-s*delay)
        return [f.real, f.imag]
    solution = root(residual, guess)
    if not solution.success or np.linalg.norm(residual(solution.x)) > 1e-8:
        raise RuntimeError("Characteristic-root numerical check failed")
    return dict(real=float(solution.x[0]), imag=float(solution.x[1]),
                residual=float(np.linalg.norm(residual(solution.x))))


def build(out, delay=.2, predictive_weight=.5, gain=2.):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    speeds = np.linspace(0, 2, 121)
    specs = [("Ordinary code", 0., "#2c7bb6"),
             ("Predictive code", delay, "#e76f36"),
             ("Reversed prediction", -delay, "#777777")]
    records = {}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for name, horizon, color in specs:
        force = cue_force(speeds, delay, horizon, predictive_weight, gain)
        lock = phase_lock(force)
        spectrum = [normal_form_spectrum(f) for f in force]
        remaining = np.exp([s["tangent_exponent"] for s in spectrum])
        axes[0].plot(speeds, np.abs(lock["error"])*180/np.pi, color=color, label=name, lw=2.5)
        axes[1].plot(speeds, remaining, color=color, lw=2.5,
                     linestyle="--" if horizon < 0 else "-")
        records[name] = dict(horizon=horizon, speed=speeds.tolist(),
                            bias_deg=(lock["error"]*180/np.pi).tolist(),
                            cue_strength=lock["strength"].tolist(), normal_form=spectrum)
    axes[0].set(xlabel="Movement speed (rad/s)", ylabel="Tracking lag (degrees)",
                title="Prediction changes where the cue pulls")
    axes[1].set(xlabel="Movement speed (rad/s)", ylabel="Disturbance left after 1 s",
                title="It need not pull harder", ylim=(0, .4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.supxlabel("Analytical model · illustrative parameters · not fitted to the trained networks", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out/f"phase_lead_and_stability.{suffix}", dpi=180)
    plt.close(fig)

    strengths = np.linspace(.1, 8, 160)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    bias_records = {}
    for name, horizon, color in specs:
        bias = float(np.angle(cue_force(1.5, delay, horizon, predictive_weight, gain)))
        rmse = np.sqrt(stationary_phase_mse(bias, strengths))*180/np.pi
        axes[0].plot(strengths, rmse, color=color, lw=2.5, label=name)
        bias_records[name] = bias
    axes[1].plot(strengths, 1/strengths, color="#4b5963", lw=2.5)
    axes[0].set(xlabel="Restoring strength (1/s)", ylabel="Position error (degrees RMS)",
                title="Less error at the same restoring strength")
    axes[1].set(xlabel="Restoring strength (1/s)", ylabel="Recovery time (seconds)",
                title="The recovery rule is shared", ylim=(0, 3))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.supxlabel("Independent-noise approximation · forecast noise and gain are not fitted", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out/f"gain_accuracy_tradeoff.{suffix}", dpi=180)
    plt.close(fig)

    horizons = np.linspace(-.9, .9, 181)
    limits = np.array([feedback_delay_limit(1., h) for h in horizons])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    ax.plot(horizons, limits, lw=2.5, color="#e76f36")
    ax.fill_between(horizons, 0, limits, alpha=.12, color="#e76f36")
    ax.axhline(1.7, color="#555555", ls="--", lw=1)
    ax.scatter([0, .4], [feedback_delay_limit(1, 0), feedback_delay_limit(1, .4)],
               color=["#2c7bb6", "#e76f36"], zorder=3)
    ax.set(xlabel="Prediction horizon × correction gain", ylabel="Critical delay × correction gain",
           title="Prediction inside feedback can stabilize a ring", ylim=(0, 2.05))
    ax.text(.5, .8, "Stable", ha="center", color="#7a4c3b")
    ax.text(.5, 1.96, "Unstable", ha="center", color="#555555")
    ax.grid(alpha=.15)
    fig.supxlabel("Different circuit hypothesis · reciprocal delayed correction · not fitted", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out/f"feedback_stability_boundary.{suffix}", dpi=180)
    plt.close(fig)
    feedback = [dict(gain=1., delay=1.7, horizon=h,
                     critical_delay=feedback_delay_limit(1., h),
                     numerical_root=characteristic_root(1., h, 1.7)) for h in (0., .4, -.4)]
    result = dict(status="analytical model; not fitted to trained data",
                  parameters=dict(delay=delay, predictive_coupling_fraction=predictive_weight, total_gain=gain),
                  cue_models=records, matched_strength_bias_radians=bias_records,
                  cue_off_spectrum=normal_form_spectrum(0j), delayed_feedback_example=feedback,
                  conclusions=["A single cue's phase lead changes tracking bias without changing its local stability.",
                    "For mixed cues, replacing prediction by ordinary code can increase total cue strength while increasing lag.",
                    "Removing all cues restores the same intrinsic ring spectrum in this normal-form model.",
                    "Prediction inside delayed corrective feedback can change transverse stability; this extra circuit must be demonstrated in data."])
    (out/"theory.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(output=str(out), cue_off=result["cue_off_spectrum"], feedback=feedback), indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default="results/ring_attractor/theory")
    parser.add_argument("--delay", type=float, default=.2)
    parser.add_argument("--predictive-weight", type=float, default=.5)
    parser.add_argument("--gain", type=float, default=2.)
    a = parser.parse_args()
    build(a.out, a.delay, a.predictive_weight, a.gain)
