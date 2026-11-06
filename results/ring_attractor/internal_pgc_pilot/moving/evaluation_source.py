"""Frozen class/dose interventions during movement in one existing RNN.

Reuse the completed stationary pilot's exact control sets and calibration means.
No new cell selection, matching, decoder fitting or network training occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def movement_metrics(predicted, actual, lag=5):
    """Arrays [time,episode,2], including the common pre-intervention frame."""
    error = np.linalg.norm(predicted[1:]-actual[1:], axis=-1)
    p, a = predicted[lag:]-predicted[:-lag], actual[lag:]-actual[:-lag]
    dot = np.sum(p*a,axis=-1)
    pn, an = np.linalg.norm(p,axis=-1), np.linalg.norm(a,axis=-1)
    true_energy = np.sum(a*a,axis=(0,2))
    pred_energy = np.sum(p*p,axis=(0,2))
    signed_gain = dot.sum(0)/np.maximum(true_energy,1e-20)
    alignment = dot.sum(0)/np.maximum(np.sqrt(true_energy*pred_energy),1e-20)
    local_cosine = dot/np.maximum(pn*an,1e-20)
    amplitude_gain = pn.sum(0)/np.maximum(an.sum(0),1e-20)
    arrays = {"episode_position_error_m":error.mean(0),"episode_final_position_error_m":error[-1],
              "episode_increment_gain":signed_gain,"episode_increment_alignment":alignment,
              "episode_increment_magnitude_gain":amplitude_gain,
              "episode_mean_local_direction_cosine":local_cosine.mean(0),
              "episode_zero_prediction_fraction":(pn<=1e-12).mean(0)}
    return {name.removeprefix("episode_"):float(values.mean()) for name,values in arrays.items()},arrays


@torch.no_grad()
def run(args):
    start=time.monotonic()
    args.out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((args.stationary/"summary.json").read_text())
    data=np.load(args.stationary/"pilot_arrays.npz",allow_pickle=False)
    checkpoint=Path(prior["protocol"]["checkpoint"])
    if sha(checkpoint)!=prior["protocol"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed from frozen stationary pilot")
    ids, controls = data["predictive_ids"],data["control_ids"]
    mean=data["calibration_features"][:,0]
    sibling=checkpoint.parents[1]
    sys.path.insert(0,str(sibling/"code"))
    from place_cells import PlaceCells
    from trajectory_generator import TrajectoryGenerator
    from model import RNN
    torch.set_num_threads(2)
    values=dict(prior["protocol"]["place_and_trajectory_configuration"])
    values["sequence_length"]=240
    config=SimpleNamespace(**values)
    place=PlaceCells(config);generator=TrajectoryGenerator(config,place)
    np.random.seed(args.seed)
    inputs,true_path,_=generator.get_test_batch(batch_size=args.episodes)
    velocities,initial=inputs
    weights=torch.load(checkpoint,map_location="cpu",weights_only=True)
    device=torch.device(args.device)
    E,W,V,D=(weights[name].to(device) for name in
        ("encoder.weight","RNN.weight_hh_l0","RNN.weight_ih_l0","decoder.weight"))
    centers=place.us.to(device=device,dtype=torch.float32)
    def decode(h,mask=None):
        logits=(h if mask is None else h*mask)@D.T
        return centers[torch.topk(logits,3,dim=-1).indices].mean(-2)
    h=initial.to(device)@E.T
    for v in velocities[:40].to(device):
        h=(h@W.T+v@V.T).relu()
    h0=h.clone()
    specs=[{"condition":"native","mode":"native","dose":0.,"group":"native","indices":np.array([],int)}]
    doses=[0.,.1,.25,.5,1.]
    for mode in ("zero_output","mean_replacement"):
        for dose in doses[1:]:
            for group,units in [("pgc",ids)]+[(f"matched_nonpgc_{i:02d}",x) for i,x in enumerate(controls)]:
                specs.append({"condition":f"{mode}_{group}_dose{round(dose*100):03d}",
                    "mode":mode,"dose":dose,"group":group,"indices":units})
    gates=np.ones((len(specs),len(mean)),np.float32);constants=np.zeros_like(gates)
    for i,spec in enumerate(specs):
        gates[i,spec["indices"]]=1-spec["dose"]
        if spec["mode"]=="mean_replacement":constants[i]=(1-gates[i])*mean
    gate=torch.from_numpy(gates).to(device)[:,None]
    constant=torch.from_numpy(constants).to(device)[:,None]
    h=h[None].expand(len(specs),-1,-1).clone()
    mask=torch.ones(len(mean),device=device);mask[ids]=0
    predicted=[decode(h).cpu().numpy()]
    readout=[decode(h[0],mask).cpu().numpy()]
    norm=[h.norm(dim=-1).cpu().numpy()]
    protocol={"stage":"single-checkpoint exploratory follow-up, fresh moving trajectories",
        "checkpoint_sha256":sha(checkpoint),"source_sha256":sha(__file__),
        "frozen_stationary_summary_sha256":sha(args.stationary/"summary.json"),
        "frozen_matching_archive_sha256":sha(args.stationary/"pilot_arrays.npz"),
        "seed":args.seed,"episodes":args.episodes,"native_warmup_steps":40,
        "intervention_steps":200,"dt_seconds":.02,"dose_grid":doses,
        "readout":"original frozen linear decoder and original top-three-place-cell position readout",
        "increment_lag_steps":5,"increment_lag_seconds":.1,
        "zero_displacement_handling":"included with zero local alignment; zero-prediction fraction reported",
        "recurrent_controls":"same20 control sets as stationary study; no rematching",
        "all_conditions_receive_same_physical_motion":True,"no_fitted_parameters":True,
        "maximum_runtime_seconds":args.max_seconds,"device":args.device}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    for t,v in enumerate(velocities[40:].to(device),1):
        h=((h*gate+constant)@W.T+(v@V.T)[None]).relu()
        predicted.append(decode(h).cpu().numpy())
        readout.append(decode(h[0],mask).cpu().numpy())
        norm.append(h.norm(dim=-1).cpu().numpy())
        if t%50==0:print(json.dumps({"steps":t,"seconds_elapsed":time.monotonic()-start}),flush=True)
        if time.monotonic()-start>args.max_seconds:raise TimeoutError("Bounded moving pilot runtime exceeded")
    predicted=np.stack(predicted);readout=np.stack(readout);norm=np.stack(norm)
    actual=true_path[39:].cpu().numpy().astype(np.float32)
    rows,archive=[],{"true_positions":actual,"predicted_positions":predicted,
        "readout_only_positions":readout,"state_norm":norm,"initial_state":h0.cpu().numpy(),
        "predictive_ids":ids,"control_ids":controls}
    for steps in (20,50,100,200):
        for i,spec in enumerate(specs):
            metrics,arrays=movement_metrics(predicted[:steps+1,i],actual[:steps+1])
            rows.append({**{k:spec[k] for k in ("condition","mode","dose","group")},
                "steps":steps,"seconds":steps*.02,**metrics,
                "median_norm_ratio":float(np.median(norm[steps,i]/norm[0,i]))})
            archive.update({f"{steps}_{i}_{k}":v for k,v in arrays.items()})
        metrics,arrays=movement_metrics(readout[:steps+1],actual[:steps+1])
        rows.append({"condition":"pgc_readout_only","mode":"readout_only","group":"pgc","dose":1.,
            "steps":steps,"seconds":steps*.02,**metrics,"state_identical_to_native":True})
    # The unmodified RNN provides an independent full moving-trajectory check.
    reference=RNN(config,place).eval();reference.load_state_dict(weights)
    original_states=reference.g(inputs)
    original_position=place.get_nearest_cell_pos(reference.decoder(original_states)).cpu().numpy()[39:]
    replay_error=float(np.max(np.abs(predicted[:,0]-original_position)))
    if replay_error>1e-6:raise ValueError(f"Moving native readout mismatch: {replay_error}")
    result={"protocol":protocol,"rows":rows,"validation":{"native_original_position_max_error_m":replay_error},
        "elapsed_seconds":time.monotonic()-start}
    (args.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(args.out/"moving_arrays.npz",**archive)
    (args.out/"evaluation_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"complete":True,"elapsed_seconds":result["elapsed_seconds"]}),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stationary",type=Path,default=Path("results/ring_attractor/internal_pgc_pilot/dose_mean_replacement"))
    p.add_argument("--out",type=Path,default=Path("results/ring_attractor/internal_pgc_pilot/moving"))
    p.add_argument("--device",default="mps");p.add_argument("--episodes",type=int,default=32)
    p.add_argument("--seed",type=int,default=42723);p.add_argument("--max-seconds",type=float,default=300)
    run(p.parse_args())
