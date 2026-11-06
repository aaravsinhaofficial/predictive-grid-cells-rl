"""Geometry and dynamical checks for a measured periodic state curve.

All geometry uses full recurrent state. PCA is only a display transform.
A periodic set of states is an input to these measurements, not evidence that
the network possesses an attractor. Cue-off retention must be assessed too.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def ring_radius(curve):
    curve = np.asarray(curve)
    return float(np.sqrt(np.mean(np.sum((curve - curve.mean(0)) ** 2, axis=-1))))


def tangents(curve):
    """Central differences on an ordered, periodic full-state curve."""
    curve = np.asarray(curve)
    difference = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    length = np.linalg.norm(difference, axis=1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError('Collapsed curve has no identifiable local tangent')
    return difference / length


def project_to_ring(points, curve):
    """Exact nearest projection onto closed polyline; return distance and phase.

    Curve samples must be in increasing uniformly spaced parameter order.
    Returned phase is that parameter, not an independently discovered variable.
    Uses chunks to bound memory for high-dimensional LSTM states.
    """
    points, curve = np.asarray(points), np.asarray(curve)
    if points.shape[-1] != curve.shape[-1]:
        raise ValueError('State dimensions differ')
    a, segment = curve, np.roll(curve, -1, axis=0) - curve
    length2 = np.sum(segment * segment, axis=1)
    denom = np.maximum(length2, 1e-24)
    distances, phase = [], []
    for start in range(0, len(points), 64):
        delta = points[start:start + 64, None, :] - a[None, :, :]
        fraction = np.clip(np.sum(delta * segment, axis=-1) / denom, 0, 1)
        residual = delta - fraction[..., None] * segment
        squared = np.sum(residual * residual, axis=-1)
        index = squared.argmin(axis=1)
        row = np.arange(len(index))
        distances.append(np.sqrt(squared[row, index]))
        phase.append((index + fraction[row, index]) * (2 * np.pi / len(curve)))
    return np.concatenate(distances), np.concatenate(phase)


def perturbations(curve, relative_size=0.1, seed=913):
    """Equal-norm tangent and isotropic transverse state perturbations.

    Size is a fraction of full-state ring radius, not a per-neuron scale.
    Normal direction is random in the complement of the empirical tangent.
    This includes all dimensions, including those invisible to an output.
    """
    curve = np.asarray(curve)
    tangent = tangents(curve)
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=curve.shape)
    normal -= np.sum(normal * tangent, axis=1, keepdims=True) * tangent
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    scale = relative_size * ring_radius(curve)
    return tangent * scale, normal * scale


def trajectory_measurements(base, tangent_trajectory, normal_trajectory):
    """Measure perturbed paths against the corresponding unperturbed curve.

    base has shape [time, phase, state]. Using the time-matched base separates
    perturbation effects from spontaneous drift. Radius retention separately
    reveals a contracting cloud that would otherwise masquerade as recovery.
    """
    base = np.asarray(base)
    initial_radius = ring_radius(base[0])
    if initial_radius < 1e-12:
        raise ValueError('Initial state curve is collapsed')
    angle = np.arange(base.shape[1]) * 2 * np.pi / base.shape[1]
    rows = []
    for t, curve in enumerate(base):
        dn, pn = project_to_ring(normal_trajectory[t], curve)
        dt, pt = project_to_ring(tangent_trajectory[t], curve)
        rows.append(dict(step=t, radius_ratio=ring_radius(curve)/initial_radius,
            normal_distance_radius=float(np.mean(dn)/initial_radius),
            tangent_distance_radius=float(np.mean(dt)/initial_radius),
            normal_phase_displacement=float(np.mean(np.abs(wrap(pn-angle)))),
            tangent_phase_displacement=float(np.mean(np.abs(wrap(pt-angle)))),
            normal_pair_distance_radius=float(np.mean(np.linalg.norm(
                normal_trajectory[t]-curve, axis=1))/initial_radius),
            tangent_pair_distance_radius=float(np.mean(np.linalg.norm(
                tangent_trajectory[t]-curve, axis=1))/initial_radius)))
    return rows


def validate_metrics(out=None):
    """Positive ring and driven-circle/point-collapse negative controls."""
    n, steps, gain = 128, 30, 0.8
    phase = np.arange(n)*2*np.pi/n
    curve = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(n)))
    tangent, normal = perturbations(curve, 0.05, seed=31)

    def ring_update(x):
        radius = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
        xy = x[:, :2] / radius * (1 + gain * (radius-1))
        return np.concatenate((xy, gain*x[:, 2:]), axis=1)

    def run(update):
        paths = []
        for x in (curve, curve+tangent, curve+normal):
            trajectory = [x]
            for _ in range(steps):
                x = update(x)
                trajectory.append(x)
            paths.append(np.array(trajectory))
        return trajectory_measurements(*paths)

    positive = run(ring_update)
    collapse = run(lambda x: gain*x)
    ring_end, collapse_end = positive[-1], collapse[-1]
    assert abs(ring_end['radius_ratio']-1) < 1e-10
    assert ring_end['normal_pair_distance_radius'] < positive[0]['normal_pair_distance_radius']*0.01
    assert ring_end['tangent_phase_displacement'] > 0.04
    assert collapse_end['radius_ratio'] < 0.002
    # Nearest-polyline interpolation is independently checked at vertices.
    distance, decoded = project_to_ring(curve, curve)
    assert distance.max() < 1e-12
    assert np.abs(wrap(decoded-phase)).max() < 1e-12
    result = dict(status='passed', analytical_normal_gain=gain,
        positive_ring=positive, driven_circle_after_cue_removal=collapse,
        interpretation='Both systems contract disturbances; only the ring retains a finite memory manifold.')
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--out', default='results/ring_attractor/diagnostic_validation.json')
    args = parser.parse_args()
    if args.validate:
        result = validate_metrics(args.out)
        print(result['status'], result['interpretation'])
