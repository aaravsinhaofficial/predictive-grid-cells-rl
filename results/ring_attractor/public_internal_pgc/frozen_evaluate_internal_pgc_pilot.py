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
    """Match count/activity/outgoing norm; no held-out response is used."""
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
                        "outgoing_recurrent_column_norm_ratio": float(features[ids, 2].mean()/features[predictive, 2].mean()),
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
    def step(h, velocity=None, gate=None, constant=None):
        source = h if gate is None else h*gate
        if constant is not None:
            source = source+constant
        drive = source@W.T
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
    features = torch.stack((calibration.mean((0,1)), calibration.square().mean((0,1)).sqrt(), W.norm(dim=0)), -1).cpu().numpy()
    del calibration
    with np.load(class_path, allow_pickle=False) as archive:
        predictive = archive["classes_predictive"].astype(int)
        zero_grid = archive["classes_phase_locked"].astype(int)
        retrospective = archive["classes_phase_precession"].astype(int)
        lag = archive["best_cm"].copy()
    control_pool = np.setdiff1d(np.arange(n_units), predictive)
    controls, balance = matched_controls(predictive, control_pool, features, 42702, args.controls)
    evaluation_seed = 42713 if args.dose_sweep else 42703
    h_path, true_path, original_inputs = trajectory_states(evaluation_seed, args.episodes)
    state0, true_position = h_path[-1], true_path[-1]
    # Verify the standalone step against the original class with exact weights.
    original_model = RNN(config, place_cells).eval()
    original_model.load_state_dict(weights)
    reference = original_model.g(original_inputs)
    replay_error = float((h_path.cpu()-reference).abs().max())
    if replay_error > 2e-4:
        raise ValueError(f"Original trajectory replay mismatch: {replay_error}")
    del original_model, reference

    doses = [0., .1, .25, .5, .75, 1.]
    if args.dose_sweep:
        specifications = [{"name":"native","mode":"native","dose":0.,"group":"native","indices":np.array([],int)}]
        for mode in ("zero_output", "mean_replacement"):
            for dose in doses[1:]:
                for group, ids in [("pgc",predictive)]+[(f"matched_nonpgc_{i:02d}",ids) for i,ids in enumerate(controls)]:
                    specifications.append({"name":f"{mode}_{group}_dose{round(dose*100):03d}",
                        "mode":mode,"dose":dose,"group":group,"indices":ids})
    else:
        specifications = [{"name":"native","mode":"native","dose":0.,"group":"native","indices":np.array([],int)}]
        specifications += [{"name":f"pgc_output_gate{round(dose*100)}","mode":"zero_output","dose":dose,"group":"pgc","indices":predictive} for dose in (.25,.5,1.)]
        specifications += [{"name":f"matched_nonpgc_{i:02d}","mode":"zero_output","dose":1.,"group":f"matched_nonpgc_{i:02d}","indices":ids} for i,ids in enumerate(controls)]
    names = [s["name"] for s in specifications]
    gates = np.ones((len(names), n_units), np.float32)
    constants = np.zeros_like(gates)
    for row,spec in enumerate(specifications):
        gates[row,spec["indices"]] = 1-spec["dose"]
        if spec["mode"] == "mean_replacement":
            constants[row] = (1-gates[row])*features[:,0]
    gate = torch.from_numpy(gates).to(device)[:,None,None,:]
    constant = torch.from_numpy(constants).to(device)[:,None,None,:]
    # Empirical position tangents: shift the initial coordinate by +/-2 mm,
    # then replay exactly the same valid measured displacement sequence.
    # The physical trajectory translates; its velocity history stays fixed.
    velocity_cpu, initial_code = original_inputs
    initial_position = true_path[0].cpu()-velocity_cpu[0]
    eps = .002
    offsets = torch.tensor([[eps,0.],[-eps,0.],[0.,eps],[0.,-eps]])
    translated_positions = (initial_position[None]+offsets[:,None]).reshape(-1,1,2)
    translated = place_cells.get_activation(translated_positions).squeeze(1).to(device)@E.T
    for v in velocity_cpu.to(device):
        translated = step(translated, v.repeat(4,1))
    translated = translated.reshape(4,args.episodes,n_units).cpu().numpy()
    tangents = np.stack(((translated[0]-translated[1])/(2*eps),
                         (translated[2]-translated[3])/(2*eps)), -1)
    rng = np.random.default_rng(42704)
    h_np = state0.cpu().numpy().astype(float)
    eligible = h_np > .1*np.sqrt(np.mean(h_np*h_np,axis=-1,keepdims=True))
    geometry = np.concatenate((h_np[...,None],tangents),-1)*eligible[...,None]
    q,_ = np.linalg.qr(geometry)
    delta_np = h_np*rng.normal(size=h_np.shape)*eligible
    delta_np -= np.einsum("bni,bi->bn",q,np.einsum("bni,bn->bi",q,delta_np))
    # QR roundoff can produce tiny values in nominally zero rows. Explicitly
    # retain the intended support before enforcing positivity; otherwise a
    # tiny negative value on an inactive zero-rate cell forces the dose to zero.
    delta_np *= eligible
    delta_np *= .01*np.linalg.norm(h_np,axis=-1,keepdims=True)/np.linalg.norm(delta_np,axis=-1,keepdims=True)
    safe = np.min(np.where(delta_np<0, h_np/np.maximum(-delta_np,1e-300),np.inf),axis=-1,keepdims=True)
    delta_np *= np.minimum(1.,.5*safe)
    delta = torch.from_numpy(delta_np.astype(np.float32)).to(device)
    if not torch.all(delta.norm(dim=-1)>1e-7*state0.norm(dim=-1)):
        raise ValueError("Normal perturbation is too small for float32 evaluation")
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
    tangent_cosines = np.einsum("bn,bni->bi",delta_np,tangents)/(np.linalg.norm(delta_np,axis=-1)[:,None]*np.linalg.norm(tangents,axis=1))
    radial_cosines = np.sum(delta_np*h_np,axis=-1)/(np.linalg.norm(delta_np,axis=-1)*np.linalg.norm(h_np,axis=-1))
    records, archive = [], {"positions": true_position.cpu().numpy(), "initial_state": state0.cpu().numpy(),
        "perturbation": delta.cpu().numpy(), "predictive_ids": predictive, "control_ids": controls,
        "calibration_features": features, "gates": gates, "constants":constants,
        "empirical_position_tangents":tangents}
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
            next_state = step(h[:,0], gate=gate[:,0],constant=constant[:,0])
            residual = (next_state-h[:,0]).norm(dim=-1)/norm.clamp_min(1e-20)
            for i,name in enumerate(names):
                records.append({"condition":name,"step":t,"seconds":t*.02,
                    **{key:specifications[i][key] for key in ("mode","dose","group")},
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
            selected = [r for r in records[-len(names)-1:] if r["condition"] == "native" or (r.get("group")=="pgc" and r.get("dose")==.25)]
            print(json.dumps({"step":t,"elapsed":time.monotonic()-started,"selected":selected}),flush=True)
        if t==args.steps:
            completed=t;break
        if time.monotonic()>deadline:
            completed=t;stop_reason="bounded_runtime";break
        h = step(h, gate=gate,constant=constant)
    protocol = {"exploratory_single_checkpoint":True,"checkpoint":str(checkpoint),
        "checkpoint_sha256":sha(checkpoint),"classification_path":str(class_path),"classification_sha256":sha(class_path),
        "classification": "stored legacy spatial-path threshold labels; no new significance classification",
        "predictive_cells":len(predictive),"normal_grid_cells":len(zero_grid),"retrospective_cells":len(retrospective),
        "predictive_units_at_shift_window_edge":int(np.sum(np.abs(lag[predictive])>=20)),
        "control_pool":"all non-PGC units; same count matched on independently calibrated mean and RMS activity and outgoing recurrent-column norm",
        "control_draws":args.controls,"perturbation":"same initially normal small rate disturbance, orthogonal to empirical x/y translation tangents and state scaling; positivity preserved; subsequent total separation is measured",
        "normal_perturbation_fraction_of_initial_norm":(np.linalg.norm(delta_np,axis=-1)/np.linalg.norm(h_np,axis=-1)).tolist(),
        "translation_tangent_difference_m":eps,
        "dose_grid":doses if args.dose_sweep else [0.,.25,.5,1.],
        "zero_dose_shared_native":True,"mean_replacement_vector":"independent valid-trajectory calibration mean, constant across evaluation states/time",
        "stability_reference":"perturbed and unperturbed trajectories under the identical gated recurrence",
        "gating":"selected source-unit output into recurrence only; input/encoder/decoder and initial states unchanged",
        "readout_only_control":"native recurrence; PGC contribution masked only when computing original decoder output",
        "motion_input_after_snapshot":0,"trajectory_steps_before_snapshot":40,"zero_input_steps_requested":args.steps,
        "episodes":args.episodes,"dt_seconds":.02,"device":str(device),"maximum_runtime_seconds":args.max_seconds,
        "seeds":{"calibration":42701,"matching":42702,"test_trajectory":evaluation_seed,"perturbation":42704},
        "place_and_trajectory_configuration":vars(config),"source_sha256":sha(__file__),
        "sibling_sources":{name:sha(root/"code"/name) for name in ("model.py","place_cells.py","trajectory_generator.py")}}
    result = {"protocol":protocol,"matching_balance":balance,"rows":records,
        "validation":{"original_replay_max_abs_error":replay_error,"positive_homogeneity_relative_error":homogeneous_error,
                      "J_times_state_equals_next_state_relative_error":radial_identity_error,
                      "normal_perturbation_max_tangent_cosine":float(np.abs(tangent_cosines).max()),
                      "normal_perturbation_max_radial_cosine":float(np.abs(radial_cosines).max())},
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
    parser.add_argument("--dose-sweep",action="store_true")
    run(parser.parse_args())
