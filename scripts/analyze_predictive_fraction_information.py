"""Local direction information in one grid-population snapshot.

Position is an unknown nuisance parameter. Noise is an explicit idealized
independent additive Gaussian rate noise, not the full agent observation model.
No RNN is trained or selected. All original population seeds 1--8 are retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from codes import GridPopulation


def rates_and_position_jacobian(population, position, direction, shift_scale=1.):
    """Float64 analytic rates/derivatives for the original stored rate-map basis.

    The original tensors are float32; convert their fixed values to float64 so
    near-zero Schur complements do not suffer float32 cancellation. This does
    not replace the maps or change their phases, wavevectors or predictive mask.
    """
    if population.cell_type != "grid" or population.pgc_shift_mode != "space":
        raise ValueError("This analysis implements the original spatial hexagonal code")
    position, direction = np.asarray(position, float), np.asarray(direction, float)
    wavevectors = population.K.cpu().numpy().astype(float)
    phases = population.phi.cpu().numpy().astype(float)
    shifts = shift_scale * population.shift.cpu().numpy().astype(float)
    location = position[..., None, :] + shifts[:, None] * direction[..., None, :]
    argument = np.einsum("...nd,nkd->...nk", location - phases, wavevectors)
    base = (np.cos(argument).sum(-1) + 1.5) / 4.5
    clipped = np.clip(base, 0., 1.)
    rate = clipped ** population.sharpness
    gradient_base = -np.einsum("...nk,nkd->...nd", np.sin(argument), wavevectors) / 4.5
    derivative = population.sharpness * clipped ** (population.sharpness - 1)
    derivative *= (base > 0) & (base < 1)
    return rate, derivative[..., None] * gradient_base


def information_from_blocks(ordinary, predictive, perpendicular, displacement):
    """Schur complement with position unrestricted and no position prior.

    ordinary/predictive are 2x2 sums of spatial gradient outer products already
    divided by rate-noise variance. Arrays may have arbitrary leading axes.
    """
    total = ordinary + predictive
    coupling = displacement * np.einsum("...ij,...j->...i", predictive, perpendicular)
    raw = displacement**2 * np.einsum("...i,...ij,...j->...", perpendicular, predictive, perpendicular)
    inverse = np.linalg.pinv(total, rcond=1e-12, hermitian=True)
    confounded = np.einsum("...i,...ij,...j->...", coupling, inverse, coupling)
    adjusted = raw - confounded
    tolerance = 1e-10 * np.maximum(raw, 1.)
    if np.any(adjusted < -tolerance):
        raise ArithmeticError("Numerically invalid negative Schur complement")
    return raw, np.maximum(adjusted, 0.), total


def information_from_jacobian(spatial_gradient, angular_gradient, noise_sd=1.):
    """Independent efficient-score projection, useful for verification."""
    spatial_gradient = np.asarray(spatial_gradient, float) / noise_sd
    angular_gradient = np.asarray(angular_gradient, float) / noise_sd
    fitted = np.einsum("...ni,...i->...n", spatial_gradient,
                      np.einsum("...in,...n->...i", np.linalg.pinv(spatial_gradient), angular_gradient))
    return np.sum(angular_gradient**2, axis=-1), np.sum((angular_gradient-fitted)**2, axis=-1)


def homogeneous_information(total_precision, fraction, perpendicular, displacement):
    projection = displacement**2 * np.einsum("...i,...ij,...j->...", perpendicular, total_precision, perpendicular)
    return fraction * projection, fraction * (1-fraction) * projection


def alias_diagnostic(seed, displacement=.20):
    """Local information can coexist with a finite-angle periodic ambiguity."""
    population = GridPopulation(n_cells=512, pgc_frac=.33, pgc_shift=displacement, seed=seed)
    angle = population.orientation[0] + math.pi/6
    direction = np.array([math.cos(angle), math.sin(angle)], np.float32)
    directions = torch.from_numpy(np.stack((direction, -direction)))
    positions = torch.tensor([[.17, -.31], [.17, -.31]])
    rates = population.activity(positions, directions).numpy()
    difference = np.abs(rates[0]-rates[1])
    mask = population.is_pgc.numpy()
    return {"seed": seed, "direction_separation_degrees": 180,
            "finest_module_predictive_max_rate_difference": float(difference[(population.module == 0) & mask].max()),
            "coarser_modules_predictive_max_rate_difference": float(difference[(population.module > 0) & mask].max()),
            "d_times_wavevector_norm_by_module": (displacement*4*np.pi/(np.sqrt(3)*population.spacings)).tolist()}


def run(out, seeds=tuple(range(1, 9)), n_positions=64, n_directions=36, noise_sd=1.):
    if noise_sd <= 0:
        raise ValueError("Fisher information requires positive assumed noise")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fractions = np.unique(np.r_[np.linspace(0, 1, 21), .33])
    sampling_seed = 751039
    rng = np.random.default_rng(sampling_seed)
    positions = rng.uniform(-1.25, 1.25, (n_positions, 2))
    angles = np.arange(n_directions) * 2*np.pi/n_directions
    directions = np.stack((np.cos(angles), np.sin(angles)), -1)
    perpendicular = np.stack((-np.sin(angles), np.cos(angles)), -1)
    sample_positions = np.repeat(positions, n_directions, axis=0)
    sample_directions = np.tile(directions, (n_positions, 1))
    sample_perpendicular = np.tile(perpendicular, (n_positions, 1))
    raw_all, adjusted_all, reference_all, condition_all, actual_fractions = [], [], [], [], []
    seed_rows = []
    for seed in seeds:
        # Population construction fixes wavevectors/phases before drawing PGC
        # assignments. Maps therefore agree exactly across all fractions.
        base = GridPopulation(n_cells=512, pgc_frac=1., pgc_shift=.20, seed=seed)
        _, present = rates_and_position_jacobian(base, sample_positions, sample_directions, shift_scale=0.)
        _, future = rates_and_position_jacobian(base, sample_positions, sample_directions)
        total_ordinary = np.einsum("bni,bnj->bij", present, present) / noise_sd**2
        displacement = float(base.shift[0])
        reference = displacement**2 * np.einsum("bi,bij,bj->b", sample_perpendicular, total_ordinary, sample_perpendicular)
        raw_rows, adjusted_rows, condition_rows = [], [], []
        seed_fractions = []
        for fraction in fractions:
            pop = GridPopulation(n_cells=512, pgc_frac=float(fraction), pgc_shift=.20, seed=seed)
            assert torch.equal(base.K, pop.K) and torch.equal(base.phi, pop.phi)
            mask = pop.is_pgc.numpy()
            ordinary = np.einsum("bni,bnj->bij", present[:, ~mask], present[:, ~mask]) / noise_sd**2
            predictive = np.einsum("bni,bnj->bij", future[:, mask], future[:, mask]) / noise_sd**2
            raw, adjusted, total = information_from_blocks(ordinary, predictive, sample_perpendicular, displacement)
            eigenvalues = np.linalg.eigvalsh(total)
            condition = eigenvalues[:, 1] / eigenvalues[:, 0]
            raw_rows.append(raw)
            adjusted_rows.append(adjusted)
            condition_rows.append(condition)
            seed_fractions.append(pop.n_pgc/pop.n_cells)
        raw_rows, adjusted_rows, condition_rows = map(np.asarray, (raw_rows, adjusted_rows, condition_rows))
        fractions_actual = np.asarray(seed_fractions)
        if actual_fractions:
            np.testing.assert_array_equal(fractions_actual, actual_fractions[0])
        actual_fractions.append(fractions_actual)
        raw_all.append(raw_rows)
        adjusted_all.append(adjusted_rows)
        condition_all.append(condition_rows)
        reference_all.append(reference)
        means = adjusted_rows.mean(-1)
        seed_rows.append({"seed": seed, "best_sampled_fraction_for_mean_local_information": float(fractions_actual[means.argmax()]),
                          "mean_reference_precision": float(reference.mean()),
                          "all_predictive_max_adjusted_relative_to_raw": float(np.max(adjusted_rows[-1]/raw_rows[-1]))})
        print(json.dumps(seed_rows[-1]), flush=True)
    raw_all, adjusted_all, condition_all, reference_all = map(np.asarray, (raw_all, adjusted_all, condition_all, reference_all))
    fractions_actual = actual_fractions[0]
    normalization = reference_all.mean(-1)
    raw_mean = raw_all.mean(-1) / normalization[:, None]
    adjusted_mean = adjusted_all.mean(-1) / normalization[:, None]
    selected = int(np.argmin(abs(fractions_actual-.33)))
    homogeneous = fractions_actual*(1-fractions_actual)
    ratio = adjusted_mean[:, selected]/homogeneous[selected]
    relative_spread = adjusted_all[:, selected].std(-1)/adjusted_all[:, selected].mean(-1)
    result = {"scope": "one instantaneous population sample; local direction information with unrestricted unknown position",
              "noise_model": "independent additive Gaussian rate noise with fixed variance; unclipped likelihood",
              "noise_sd": noise_sd, "not_the_full_agent_observation_likelihood": True,
              "seeds": list(seeds), "n_cells": 512, "displacement_m": .20,
              "sampling_seed": sampling_seed, "n_positions": n_positions, "n_directions": n_directions,
              "position_bounds_m": [-1.25, 1.25], "fractions_requested": fractions.tolist(),
              "fractions_actual": fractions_actual.tolist(), "per_seed": seed_rows,
              "normalized_raw_information_mean": raw_mean.mean(0).tolist(),
              "normalized_adjusted_information_mean": adjusted_mean.mean(0).tolist(),
              "homogeneous_prediction": homogeneous.tolist(),
              "fraction_033": {"actual": float(fractions_actual[selected]),
                  "adjusted_information_relative_to_homogeneous_by_seed": ratio.tolist(),
                  "coefficient_of_variation_across_position_direction_by_seed": relative_spread.tolist(),
                  "position_precision_anisotropy_eigenvalue_ratio_quantiles": np.quantile(condition_all[:, selected], [.05, .5, .95]).tolist(),
                  "positive_adjusted_information_fraction": float(np.mean(adjusted_all[:, selected] > 1e-10))},
              "alias_checks": [alias_diagnostic(seed) for seed in seeds],
              "provenance": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (Path(__file__), ROOT/"code/codes.py")}}
    (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(out/"information.npz", seeds=np.asarray(seeds), fractions=fractions_actual,
                        positions=positions, angles=angles, raw=raw_all, adjusted=adjusted_all,
                        reference=reference_all, position_precision_condition_number=condition_all)
    plot(out, fractions_actual, raw_mean, adjusted_mean, adjusted_all, n_positions, n_directions, angles, selected)
    return result


def plot(out, fractions, raw_mean, adjusted_mean, adjusted, n_positions, n_directions, angles, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    dense = np.linspace(0, 1, 301)
    for ax, values, prediction, title, color in (
        (axes[0], raw_mean, dense, "Position known", "#3977a8"),
        (axes[1], adjusted_mean, dense*(1-dense), "Position unknown", "#dd713c")):
        for row in values:
            ax.plot(fractions, row, color=color, alpha=.19, lw=.8)
        ax.plot(fractions, values.mean(0), "o-", color=color, lw=2, markersize=3.5, label="Original grid code")
        ax.plot(dense, prediction, "--", color="0.25", lw=1.5, label="Homogeneous theory")
        ax.set(title=title, xlabel="Predictive cells (%)", ylabel="Normalized direction information", xlim=(0, 1), ylim=(0, None))
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
        ax.grid(axis="y", alpha=.15)
    axes[1].legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.5, .02))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out/f"information_vs_fraction.{suffix}", bbox_inches="tight")
    plt.close(fig)

    # Predetermined population 1 and first eight sampled positions. The broad
    # average law does not erase local anisotropy or uneven spatial sampling.
    curves = adjusted[0, selected].reshape(n_positions, n_directions)
    fig, ax = plt.subplots(figsize=(6.8, 3.5), layout="constrained")
    for row in curves[:8]:
        ax.plot(np.r_[angles, 2*np.pi]*180/np.pi, np.r_[row, row[0]]/curves.mean(), lw=1, alpha=.6)
    mean = curves.mean(0)/curves.mean()
    ax.plot(np.r_[angles, 2*np.pi]*180/np.pi, np.r_[mean, mean[0]], color="black", lw=2, label=f"Mean over {n_positions} positions")
    ax.set(xlabel="Movement direction (degrees)", ylabel="Information / overall mean", xlim=(0, 360),
           title="The same mixture has uneven local precision")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(frameon=False, fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out/f"local_anisotropy.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/ring_heading/fraction_information")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--directions", type=int, default=36)
    parser.add_argument("--noise-sd", type=float, default=1.)
    args = parser.parse_args()
    run(args.out, tuple(args.seeds), args.positions, args.directions, args.noise_sd)
