"""Bounded exploratory pilot on an existing bias-free ReLU path integrator.

The sibling repository is read-only. This probes internally classified units,
not the supplied predictive input code of the new heading experiments.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
from scipy.spatial import cKDTree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_controls(predictive, pool, features, seed, draws=20):
    """Match count, mean activity and RMS; no held-out response is used."""
    if len(pool) < len(predictive):
        raise ValueError("Not enough non-PGC control cells for count matching")
    transformed = np.log1p(features)
    scale = np.maximum(transformed.std(0), 1e-8)
    standardized = (transformed-transformed.mean(0))/scale
    tree = cKDTree(standardized[pool])
    _, candidate = tree.query(standardized[predictive], k=min(512, len(pool)))
    rng = np.random.default_rng(seed)
    sets, balance = [], []
    for _ in range(draws):
        available = np.ones(len(pool), bool)
        chosen = []
        for j in rng.permutation(len(predictive)):
            near = candidate[j][available[candidate[j]]]
            if len(near):
                # Random order produces independent matched sets. Pick the
                # nearest available control, rather than accepting a weak match.
                selected = int(near[0])
            else:
                remaining = np.flatnonzero(available)
                distance = np.sum((standardized[pool[remaining]]-standardized[predictive[j]])**2, axis=1)
                selected = int(remaining[distance.argmin()])
            available[selected] = False
            chosen.append(pool[selected])
        ids = np.asarray(chosen)
        sets.append(ids)
        balance.append({"mean_activity_ratio": float(features[ids, 0].mean()/features[predictive, 0].mean()),
                        "rms_activity_ratio": float(features[ids, 1].mean()/features[predictive, 1].mean()),
                        "standardized_mean_difference": (standardized[ids].mean(0)-standardized[predictive].mean(0)).tolist()})
    return np.asarray(sets), balance


def projective_state(h):
    return h/h.norm(dim=-1, keepdim=True).clamp_min(1e-20)


@torch.no_grad()
def run(args):
    started = time.monotonic()
    deadline = started+args.max_seconds
    root, out = args.sibling, args.out
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = root/"Seed 0(trained advanced)/most_recent_model.pth"
    class_path = root/"analysis_outputs/Seed 0(trained advanced)/spatial_path_cmp/gridness_data.npz"
    sys.path.insert(0, str(root/"code"))
    from place_cells import PlaceCells
    from trajectory_generator import TrajectoryGenerator
    from model import RNN
    torch.set_num_threads(2)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    n_units, n_place = weights["encoder.weight"].shape
    config = SimpleNamespace(batch_size=128, sequence_length=40, Ng=n_units, Np=n_place,
        velocity_dim=2, place_cell_rf=.12, surround_scale=2., activation="relu",
        weight_decay=1e-4, DoG=True, periodic=False, box_width=2.2, box_height=2.2,
        learning_rate=1e-4, device="cpu", trajectory_dt=.02, trajectory_speed_scale=1.,
        trajectory_border_region=.03, trajectory_wall_slowdown=.25,
        trajectory_wall_turn_scale=1., trajectory_style="random_walk")
    place_cells = PlaceCells(config)
    trajectory = TrajectoryGenerator(config, place_cells)
    device = torch.device(args.device)
    E, W, V, D = (weights[name].to(device) for name in
        ("encoder.weight", "RNN.weight_hh_l0", "RNN.weight_ih_l0", "decoder.weight"))
    centers = place_cells.us.to(dtype=torch.float32, device=device)
    def step(h, velocity=None, gate=None):
        drive = (h if gate is None else h*gate)@W.T
        if velocity is not None:
            drive = drive+velocity@V.T
        return drive.relu()
    def decode(h, mask=None):
        logits = (h if mask is None else h*mask)@D.T
        return centers[torch.topk(logits, 3, dim=-1).indices].mean(-2)
    def trajectory_states(seed, batch):
        np.random.seed(seed)
        (velocities, initial), positions, _ = trajectory.get_test_batch(batch_size=batch)
        h = initial.to(device)@E.T
        states = []
        for v in velocities.to(device):
            h = step(h, v)
            states.append(h)
        return torch.stack(states), positions.to(dtype=torch.float32, device=device), (velocities, initial)

    calibration, _, _ = trajectory_states(42701, 128)
    features = torch.stack((calibration.mean((0,1)), calibration.square().mean((0,1)).sqrt()), -1).cpu().numpy()
    del calibration
    with np.load(class_path, allow_pickle=False) as archive:
        predictive = archive["classes_predictive"].astype(int)
        zero_grid = archive["classes_phase_locked"].astype(int)
        retrospective = archive["classes_phase_precession"].astype(int)
        lag = archive["best_cm"].copy()
    control_pool = np.setdiff1d(np.arange(n_units), predictive)
    controls, balance = matched_controls(predictive, control_pool, features, 42702, args.controls)
    h_path, true_path, original_inputs = trajectory_states(42703, args.episodes)
    state0, true_position = h_path[-1], true_path[-1]
    # Verify the standalone step against the original class with exact weights.
    original_model = RNN(config, place_cells).eval()
    original_model.load_state_dict(weights)
    reference = original_model.g(original_inputs)
    replay_error = float((h_path.cpu()-reference).abs().max())
    if replay_error > 2e-4:
        raise ValueError(f"Original trajectory replay mismatch: {replay_error}")
    del original_model, reference

    names = ["native", "pgc_output_gate25", "pgc_output_gate50", "pgc_output_gate100"]
    names += [f"matched_nonpgc_{i:02d}" for i in range(args.controls)]
    gates = np.ones((len(names), n_units), np.float32)
    for row, strength in ((1,.25), (2,.5), (3,1.)):
        gates[row, predictive] = 1-strength
    for i, ids in enumerate(controls):
        gates[4+i, ids] = 0
    gate = torch.from_numpy(gates).to(device)[:,None,None,:]
    rng = np.random.default_rng(42704)
    random = torch.from_numpy(rng.normal(size=state0.shape).astype(np.float32)).to(device)
    delta = state0*random
    delta -= state0*((delta*state0).sum(-1,keepdim=True)/state0.square().sum(-1,keepdim=True))
    delta *= .01*state0.norm(dim=-1,keepdim=True)/delta.norm(dim=-1,keepdim=True)
    if (state0+delta).min() < 0:
        raise ValueError("Multiplicative small perturbation left nonnegative rate space")
    initial_pair = torch.stack((state0, state0+delta))
    h = initial_pair[None].expand(len(names),-1,-1,-1).clone()
    norm0 = state0.norm(dim=-1)
    distance0 = delta.norm(dim=-1)
    projective0 = (projective_state(initial_pair[1])-projective_state(initial_pair[0])).norm(dim=-1)
    initial_unit = projective_state(state0)
    spread0 = ((initial_unit-initial_unit.mean(0)).square().sum(-1).mean()).sqrt()
    readout_mask = torch.ones(n_units, device=device); readout_mask[predictive] = 0
    check_state = step(state0)
    scaled = step(1.7*state0)
    homogeneous_error = float((scaled-1.7*check_state).norm()/check_state.norm())
    active = (state0@W.T)>0
    j_radial = (state0@W.T)*active
    radial_identity_error = float((j_radial-check_state).norm()/check_state.norm())
    records, archive = [], {"positions": true_position.cpu().numpy(), "initial_state": state0.cpu().numpy(),
        "perturbation": delta.cpu().numpy(), "predictive_ids": predictive, "control_ids": controls,
        "calibration_features": features, "gates": gates}
    report_steps = {0,1,2,5,10,20,40,60,100,150,args.steps}
    stop_reason, completed = "completed", 0
    for t in range(args.steps+1):
        if t in report_steps:
            norm = h[:,0].norm(dim=-1)
            unit = projective_state(h[:,0])
            phase_distance = (projective_state(h[:,1])-unit).norm(dim=-1)
            full_distance = (h[:,1]-h[:,0]).norm(dim=-1)
            spread = ((unit-unit.mean(1,keepdim=True)).square().sum(-1).mean(-1)).sqrt()
            position = decode(h[:,0])
            physical_error = (position-true_position).norm(dim=-1)
            memory_drift = (position-decode(state0)).norm(dim=-1)
            next_state = step(h[:,0], gate=gate[:,0])
            residual = (next_state-h[:,0]).norm(dim=-1)/norm.clamp_min(1e-20)
            for i,name in enumerate(names):
                records.append({"condition":name,"step":t,"seconds":t*.02,
                    "median_norm_ratio":float((norm[i]/norm0).median()),
                    "mean_projective_perturbation_gain":float((phase_distance[i]/projective0).mean()),
                    "mean_full_state_perturbation_gain":float((full_distance[i]/distance0).mean()),
                    "projective_position_spread_ratio":float(spread[i]/spread0),
                    "mean_fixed_readout_position_error_m":float(physical_error[i].mean()),
                    "mean_fixed_readout_memory_drift_m":float(memory_drift[i].mean()),
                    "median_one_step_state_residual":float(residual[i].median())})
            records.append({"condition":"pgc_readout_only","step":t,"seconds":t*.02,
                "mean_fixed_readout_position_error_m":float((decode(h[0,0],readout_mask)-true_position).norm(dim=-1).mean()),
                "mean_fixed_readout_memory_drift_m":float((decode(h[0,0],readout_mask)-decode(state0,readout_mask)).norm(dim=-1).mean()),
                "state_identical_to_native":True})
            archive[f"step{t}_position"] = position.cpu().numpy()
            archive[f"step{t}_projective_gain"] = (phase_distance/projective0).cpu().numpy()
            archive[f"step{t}_norm_ratio"] = (norm/norm0).cpu().numpy()
            print(json.dumps({"step":t,"elapsed":time.monotonic()-started,
                "native":records[-len(names)-1],"pgc":records[-len(names)+2]}),flush=True)
        if t==args.steps:
            completed=t;break
        if time.monotonic()>deadline:
            completed=t;stop_reason="bounded_runtime";break
        h = step(h, gate=gate)
    protocol = {"exploratory_single_checkpoint":True,"checkpoint":str(checkpoint),
        "checkpoint_sha256":sha(checkpoint),"classification_path":str(class_path),"classification_sha256":sha(class_path),
        "classification": "stored legacy spatial-path threshold labels; no new significance classification",
        "predictive_cells":len(predictive),"normal_grid_cells":len(zero_grid),"retrospective_cells":len(retrospective),
        "predictive_units_at_shift_window_edge":int(np.sum(np.abs(lag[predictive])>=20)),
        "control_pool":"all non-PGC units; same count matched on independently calibrated mean and RMS activity",
        "control_draws":args.controls,"perturbation":"same 1%-norm multiplicative rate disturbance, orthogonal to state scaling; not claimed to be an empirical manifold normal",
        "stability_reference":"perturbed and unperturbed trajectories under the identical gated recurrence",
        "gating":"selected source-unit output into recurrence only; input/encoder/decoder and initial states unchanged",
        "readout_only_control":"native recurrence; PGC contribution masked only when computing original decoder output",
        "motion_input_after_snapshot":0,"trajectory_steps_before_snapshot":40,"zero_input_steps_requested":args.steps,
        "episodes":args.episodes,"dt_seconds":.02,"device":str(device),"maximum_runtime_seconds":args.max_seconds,
        "seeds":{"calibration":42701,"matching":42702,"test_trajectory":42703,"perturbation":42704},
        "place_and_trajectory_configuration":vars(config),"source_sha256":sha(__file__),
        "sibling_sources":{name:sha(root/"code"/name) for name in ("model.py","place_cells.py","trajectory_generator.py")}}
    result = {"protocol":protocol,"matching_balance":balance,"rows":records,
        "validation":{"original_replay_max_abs_error":replay_error,"positive_homogeneity_relative_error":homogeneous_error,
                      "J_times_state_equals_next_state_relative_error":radial_identity_error},
        "completed_steps":completed,"stopping_reason":stop_reason,"elapsed_seconds":time.monotonic()-started}
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"pilot_arrays.npz",**archive)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sibling",type=Path,default=Path("/Users/aaravsinha/predictive-grid-cell-analysis"))
    parser.add_argument("--out",type=Path,default=Path("results/ring_attractor/internal_pgc_pilot"))
    parser.add_argument("--device",default="mps")
    parser.add_argument("--episodes",type=int,default=32)
    parser.add_argument("--controls",type=int,default=20)
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--max-seconds",type=float,default=300)
    run(parser.parse_args())
