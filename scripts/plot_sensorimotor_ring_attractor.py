"""Visualize the designed goal ring within the trained agent/environment system.

The diagnostic latency is selected from the analytical stability boundary,
not from searching navigation scores. Seed 2101 is the first registered seed.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from train_sensorimotor_ring import (Config, load_fitted_decoders, simulate, local_loop_spectrum, delay_state_step)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path("results/ring_attractor/sensorimotor_confirmatory")
    fitted = json.loads((folder/"seed_2101.json").read_text())
    ordinary, forecast = load_fitted_decoders(folder/"seed_2101.npz")
    model = fitted["models"]["predictive"]
    # The discrete characteristic predicts native stable / removal unstable at
    # 0.36 s, beyond the 0.30 s training delay. Both use the exact same trained gain.
    cfg = Config(**{**fitted["config"], "delay_steps": 18, "duration": 40.,
                    "phase_noise": 0., "velocity_noise": 0., "drift_noise": 0.})
    phases = np.linspace(-np.pi, np.pi, 20, endpoint=False)
    steps = int(round(cfg.duration/cfg.dt))
    data = dict(goal=phases, initial=phases+.04, phase_noise=np.zeros((steps, len(phases))),
                velocity_noise=np.zeros((steps, len(phases))), drift=np.zeros((steps, len(phases))))
    memory_phase = ordinary.angle(phases)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(10, 4.8), layout="constrained")
    result = {}
    for panel, condition, title, color in ((1, "predictive", "Forecast present", "#e76f36"),
                                           (2, "removed", "Forecast removed", "#8664a7")):
        spectrum = local_loop_spectrum(cfg, ordinary, forecast, condition, model["gain"], model["mixture"])
        nav = simulate(cfg, ordinary, forecast, condition, [model["gain"]], [model["mixture"]], data, record=True)
        trace = nav["error_trace"][:, 0]
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        angle = np.linspace(-np.pi, np.pi, 200)
        ax.plot(np.cos(angle), np.sin(angle), np.zeros(200), color="#333333", lw=2.5)
        for index, phase in enumerate(memory_phase):
            # Same neural goal state throughout a trajectory; the third axis is
            # the physical navigation error, a transverse sensorimotor coordinate.
            ax.plot(np.full(steps, np.cos(phase)), np.full(steps, np.sin(phase)),
                    trace[:, index], color=color, alpha=.7, lw=1.2)
        ax.scatter(np.cos(memory_phase), np.sin(memory_phase), trace[-1], color=color, s=22)
        ax.set(xlabel="Goal-memory state 1", ylabel="Goal-memory state 2", zlabel="Physical goal error (rad)",
               title=title, zlim=(-.8, .8), xticks=(-1, 0, 1), yticks=(-1, 0, 1), zticks=(-.8, 0, .8))
        ax.view_init(elev=23, azim=-61)
        ax.set_box_aspect((1, 1, .65))
        result[condition] = dict(spectrum=spectrum, initial_error=.04,
                                 final_mean_abs_error=float(np.abs(trace[-1]).mean()),
                                 final_second_rms=float(np.sqrt(np.mean(trace[-50:]**2))))
    fig.suptitle("A goal-indexed ring in the agent + environment", fontsize=15)
    fig.supxlabel("Designed neural memory ring · trained code readouts and controller · 0.36 s diagnostic delay · 40 s evolution", fontsize=8.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder/f"joint_ring_attractor.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    # A true online rescue, not a fresh trial restarted in the native condition.
    # The state and physical delay history continue unchanged at the switch.
    rescue_time, total_time, goal = 20., 80., .713
    memory_phase = ordinary.angle(goal)
    start = np.r_[np.cos(memory_phase), np.sin(memory_phase), np.full(cfg.delay_steps+2, goal+.04)]
    online = {}
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    for name, color in (("Forecast removed", "#8664a7"), ("Restored at 20 s", "#e76f36"),
                         ("Forecast always present", "#2c7bb6")):
        state, errors = start.copy(), []
        for t in range(int(total_time/cfg.dt)):
            condition = "predictive" if (name == "Forecast always present" or
                name == "Restored at 20 s" and t*cfg.dt >= rescue_time) else "removed"
            state = delay_state_step(cfg, ordinary, forecast, condition, model["gain"], model["mixture"], state)
            errors.append(state[2]-goal)
        errors = np.asarray(errors)
        # A one-second RMS window removes the oscillation sign, not its amplitude.
        rms = np.sqrt(np.convolve(errors**2, np.ones(50)/50, mode="valid"))
        times = (np.arange(len(rms))+50)*cfg.dt
        ax.plot(times, rms, color=color, lw=2.1, label=name)
        online[name] = dict(final_second_rms=float(np.sqrt(np.mean(errors[-50:]**2))),
                           max_abs_error=float(np.max(np.abs(errors))))
    ax.axvline(rescue_time, color="#888888", ls="--", lw=1)
    ax.set(xlabel="Time (seconds)", ylabel="Physical goal error (rad RMS)",
           title="Restoring the forecast stops the growing oscillation")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    fig.supxlabel("Same learned controller, neural state and physical history continue through the switch", fontsize=9)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder/f"online_forecast_rescue.{ext}", dpi=180)
    plt.close(fig)
    result.update(seed=2101, config=asdict(cfg), gain=model["gain"], mixture=model["mixture"],
                  online_rescue=online, online_switch_time_seconds=rescue_time,
                  reason_for_latency="Analytical boundary discrimination; not a baseline controller-performance comparison",
                  intrinsic_neural_ring="Identical and stable in both panels; prediction stabilizes the transverse physical feedback mode")
    (folder/"joint_ring_diagnostic.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("seed", "gain", "mixture", "predictive", "removed")}, indent=2))


if __name__ == "__main__":
    main()
