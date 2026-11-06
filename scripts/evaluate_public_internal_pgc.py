"""Prospective, fixed-protocol causal replication on ten published path integrators.

Only recurrent SOURCE outputs are changed. No checkpoint or classification is
trained, selected, relabelled, or rewritten. Public sources remain read-only.
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

from evaluate_internal_pgc_pilot import matched_controls, projective_state
from evaluate_internal_pgc_moving import movement_metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/research/public_internal_pgc_protocol.json"
ASSETS = ROOT / "docs/research/public_internal_pgc_sources.json"
OUT = ROOT / "results/ring_attractor/public_internal_pgc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out):
    """This must happen before the first public-cohort network is evaluated."""
    paths = [Path(__file__), PROTOCOL, ASSETS,
             Path(__file__).with_name("evaluate_internal_pgc_pilot.py"),
             Path(__file__).with_name("evaluate_internal_pgc_moving.py")]
    source = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    out.mkdir(parents=True, exist_ok=True)
    target = out / "frozen_manifest.json"
    if target.exists():
        if json.loads(target.read_text())["sources"] != source:
            raise ValueError("Frozen evaluator/protocol changed; require an explicit recorded amendment")
        return
    target.write_text(json.dumps({"sources": source, "frozen_before_public_evaluation": True}, indent=2)+"\n")
    for p in paths:
        (out / ("frozen_" + p.name)).write_bytes(p.read_bytes())


def verified_asset(manifest, suffix):
    rows = [r for r in manifest["files"] if r["path"].endswith(suffix)]
    if len(rows) != 1:
        raise ValueError(f"Ambiguous/missing pinned asset {suffix}")
    row = rows[0]
    p = ROOT / row["local_path"]
    if sha(p) != row["sha256"]:
        raise ValueError(f"Pinned asset hash mismatch: {row['path']}")
    return p


def intervention_arrays(predictive, controls, mean, doses):
    n = len(mean)
    specifications = [{"condition": "native", "mode": "native", "dose": 0., "group": "native"}]
    gates = [np.ones(n, np.float32)]
    constants = [np.zeros(n, np.float32)]
    for mode in ("zero_output", "mean_replacement"):
        for dose in doses:
            if dose == 0:
                continue
            for group, ids in [("pgc", predictive)] + [(f"matched_nonpgc_{i:02d}", x) for i, x in enumerate(controls)]:
                gate = np.ones(n, np.float32)
                gate[ids] = 1-dose
                constant = (1-gate)*mean if mode == "mean_replacement" else np.zeros(n, np.float32)
                specifications.append({"condition": f"{mode}_{group}_dose{round(dose*100):03d}",
                                       "mode": mode, "dose": dose, "group": group})
                gates.append(gate)
                constants.append(constant)
    return specifications, np.stack(gates), np.stack(constants)


def initial_normal_perturbation(h, tangents, seed):
    """Orthogonal initially to position derivatives and scale, with positive rates."""
    h = np.asarray(h, dtype=np.float64)
    eligible = h > .1*np.sqrt(np.mean(h*h, axis=-1, keepdims=True))
    geometry = np.concatenate((h[..., None], tangents), -1)*eligible[..., None]
    q, _ = np.linalg.qr(geometry)
    delta = h*np.random.default_rng(seed).normal(size=h.shape)*eligible
    delta -= np.einsum("bni,bi->bn", q, np.einsum("bni,bn->bi", q, delta))
    delta *= eligible
    delta *= .01*np.linalg.norm(h, axis=-1, keepdims=True)/np.linalg.norm(delta, axis=-1, keepdims=True)
    safe = np.min(np.where(delta < 0, h/np.maximum(-delta, 1e-300), np.inf), axis=-1, keepdims=True)
    delta *= np.minimum(1., .5*safe)
    rounded = (h.astype(np.float32)+delta.astype(np.float32))-h.astype(np.float32)
    if np.min(h+rounded) < 0 or np.any(np.linalg.norm(rounded, axis=-1) < 1e-7*np.linalg.norm(h, axis=-1)):
        raise ValueError("Invalid perturbation positivity/amplitude")
    radial = np.sum(rounded*h, -1)/(np.linalg.norm(rounded, axis=-1)*np.linalg.norm(h, axis=-1))
    tangent = np.einsum("bn,bni->bi", rounded, tangents)/(np.linalg.norm(rounded, axis=-1)[:, None]*np.linalg.norm(tangents, axis=1))
    validation = {"actual_float32_max_radial_cosine": float(np.max(np.abs(radial))),
                  "actual_float32_max_tangent_cosine": float(np.max(np.abs(tangent))),
                  "actual_float32_dose_range": [float(x) for x in (np.min(np.linalg.norm(rounded, axis=-1)/np.linalg.norm(h, axis=-1)), np.max(np.linalg.norm(rounded, axis=-1)/np.linalg.norm(h, axis=-1)))]}
    if validation["actual_float32_max_radial_cosine"] > 1e-4 or validation["actual_float32_max_tangent_cosine"] > 1e-4:
        raise ValueError("Initially normal perturbation failed float32 orthogonality")
    return rounded.astype(np.float32), validation


@torch.no_grad()
def evaluate(seed, out, device):
    start = time.monotonic()
    freeze(out)
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(ASSETS.read_text())
    if sha(ASSETS) != protocol["asset_manifest_sha256"] or manifest["commit"] != protocol["public_commit"] or not manifest["download_verified"]:
        raise ValueError("Public asset provenance mismatch")
    if seed not in protocol["model_seeds"]:
        raise ValueError("Seed outside frozen cohort")
    dest = out / f"seed{seed:02d}"
    if (dest / "summary.json").exists():
        raise FileExistsError(f"Completed seed already exists: {seed}; do not overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    # Validate ALL imported publication code, then verify import resolution.
    code_paths = [verified_asset(manifest, "/code/"+name) for name in ("model.py", "place_cells.py", "trajectory_generator.py")]
    sys.path.insert(0, str(code_paths[0].parent))
    import model
    import place_cells
    import trajectory_generator
    for module, p in zip((model, place_cells, trajectory_generator), code_paths):
        if Path(module.__file__).resolve() != p.resolve():
            raise ValueError("Unexpected public-source import resolution")
    prefix = f"RNNs/models/random_walk/Seed {seed} weight decay 1e-04/"
    candidates = [r for r in manifest["files"] if r["path"].startswith(prefix) and r["path"].endswith("/final_model.pth")]
    if len(candidates) != 1:
        raise ValueError("Checkpoint path is ambiguous")
    checkpoint = verified_asset(manifest, candidates[0]["path"])
    modelbase = candidates[0]["path"].removesuffix("final_model.pth")
    class_path = verified_asset(manifest, modelbase+"analysis_outputs/spatial shift/predictive_retrospective/predictive_ids_random_walk.npy")
    predictive = np.load(class_path, allow_pickle=False).astype(int)
    if predictive.ndim == 2 and predictive.shape[1] == 1:
        predictive = predictive[:, 0]  # Publication stores IDs as an N-by-1 column.
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(weights) != {"encoder.weight", "RNN.weight_hh_l0", "RNN.weight_ih_l0", "decoder.weight"}:
        raise ValueError("Unexpected architecture; no silent checkpoint adaptation")
    n, npcs = weights["encoder.weight"].shape
    if (n, npcs) != (4096, 512) or predictive.ndim != 1 or len(np.unique(predictive)) != len(predictive) or predictive.min() < 0 or predictive.max() >= n:
        raise ValueError("Invalid fixed architecture/classification")
    config = SimpleNamespace(batch_size=128, sequence_length=40, Ng=n, Np=npcs,
        velocity_dim=2, place_cell_rf=.12, surround_scale=2., activation="relu",
        weight_decay=1e-4, DoG=True, periodic=False, box_width=2.2, box_height=2.2,
        learning_rate=1e-4, device="cpu", trajectory_dt=.02, trajectory_speed_scale=1.,
        trajectory_border_region=.03, trajectory_wall_slowdown=.25,
        trajectory_wall_turn_scale=1., trajectory_style="random_walk")
    place = place_cells.PlaceCells(config)
    generator = trajectory_generator.TrajectoryGenerator(config, place)
    torch.set_num_threads(2)
    E, W, V, D = (weights[key].to(device) for key in ("encoder.weight", "RNN.weight_hh_l0", "RNN.weight_ih_l0", "decoder.weight"))
    centers = place.us.to(device=device, dtype=torch.float32)
    def decode(h, mask=None):
        return centers[torch.topk((h if mask is None else h*mask)@D.T, 3, dim=-1).indices].mean(-2)
    def step(h, v=None):
        return (h@W.T + (v@V.T if v is not None else 0)).relu()
    def generate(random_seed, episodes, length):
        config.sequence_length = length
        np.random.seed(random_seed)
        return generator.get_test_batch(batch_size=episodes)
    (cv, c0), _, _ = generate(52701+100*seed, 128, 40)
    ch = c0.to(device)@E.T
    total = torch.zeros(n, device=device)
    square = torch.zeros_like(total)
    for v in cv.to(device):
        ch = step(ch, v)
        total += ch.sum(0)
        square += ch.square().sum(0)
    count = 40*128
    features = torch.stack((total/count, (square/count).sqrt(), W.norm(dim=0)), -1).cpu().numpy()
    controls, balance = matched_controls(predictive, np.setdiff1d(np.arange(n), predictive), features, 52702+100*seed, 20)
    episodes = protocol["evaluation_episodes"]
    inputs, true, _ = generate(52703+100*seed, episodes, 240)
    velocities, initial = inputs
    h0 = initial.to(device)@E.T
    warmup = []
    for v in velocities[:40].to(device):
        h0 = step(h0, v)
        warmup.append(h0)
    original = model.RNN(config, place).eval()
    original.load_state_dict(weights)
    reference_h = original.g(inputs)
    reference_pos = place.get_nearest_cell_pos(original.decoder(reference_h)).numpy()[39:]
    replay_error = float((torch.stack(warmup).cpu()-reference_h[:40]).abs().max())
    if replay_error > 2e-4:
        raise ValueError(f"Original class state replay failed: {replay_error}")
    del original, reference_h, warmup
    # Position derivatives arise from physically translated initial coordinates,
    # replaying the SAME displacement history, rather than a decoder direction.
    eps = .002
    initial_position = true[0]-velocities[0]
    offsets = torch.tensor([[eps, 0.], [-eps, 0.], [0., eps], [0., -eps]])
    translated_position = (initial_position[None]+offsets[:, None]).reshape(-1, 1, 2)
    translated = place.get_activation(translated_position).squeeze(1).to(device)@E.T
    for v in velocities[:40].to(device):
        translated = step(translated, v.repeat(4, 1))
    translated = translated.reshape(4, episodes, n).cpu().numpy()
    tangents = np.stack(((translated[0]-translated[1])/(2*eps), (translated[2]-translated[3])/(2*eps)), -1)
    delta, perturbation_validation = initial_normal_perturbation(h0.cpu().numpy(), tangents, 52704+100*seed)
    delta = torch.from_numpy(delta).to(device)
    specs, gates, constants = intervention_arrays(predictive, controls, features[:, 0], protocol["doses"])
    gate = torch.from_numpy(gates).to(device)[:, None, None]
    constant = torch.from_numpy(constants).to(device)[:, None, None]
    # branch0=stationary native-initial-state, branch1=stationary perturbed,
    # branch2=moving native-initial-state. Every condition has its own reference.
    h = torch.stack((h0, h0+delta, h0))[None].expand(len(specs), -1, -1, -1).clone()
    hnorm = h0.norm(dim=-1)
    unit0 = projective_state(h0)
    spread0 = ((unit0-unit0.mean(0)).square().sum(-1).mean()).sqrt()
    pdist0 = (projective_state(h0+delta)-unit0).norm(dim=-1)
    distance0 = delta.norm(dim=-1)
    predicted = [decode(h[:, [0, 2]]).cpu().numpy()]
    mask = torch.ones(n, device=device)
    mask[predictive] = 0
    readout_only = [decode(h[0, [0, 2]], mask).cpu().numpy()]
    physical = true[39:].numpy().astype(np.float32)
    stat_actual = np.broadcast_to(physical[:1], physical.shape)
    metrics_by_step = {}
    archive = {"initial_state": h0.cpu().numpy(), "perturbation": delta.cpu().numpy(),
               "position_tangents": tangents, "calibration_features": features,
               "predictive_ids": predictive, "control_ids": controls,
               "gates": gates, "constants": constants, "moving_true_positions": physical}
    identity_error = float((step(1.7*h0)-1.7*step(h0)).norm()/step(h0).norm())
    for t, v in enumerate(velocities[40:].to(device), 1):
        drive = torch.zeros((3, episodes, n), device=device)
        drive[2] = v@V.T
        h = ((h*gate+constant)@W.T+drive[None]).relu()
        predicted.append(decode(h[:, [0, 2]]).cpu().numpy())
        readout_only.append(decode(h[0, [0, 2]], mask).cpu().numpy())
        if t in (20, 50, 100, 200):
            unit = projective_state(h[:, [0, 2]])
            spread = ((unit-unit.mean(2, keepdim=True)).square().sum(-1).mean(-1)).sqrt()/spread0
            norm = h[:, [0, 2]].norm(dim=-1)/hnorm
            pgain = (projective_state(h[:, 1])-unit[:, 0]).norm(dim=-1)/pdist0
            fgain = (h[:, 1]-h[:, 0]).norm(dim=-1)/distance0
            metrics_by_step[t] = {"spread": spread.cpu().numpy(), "norm": norm.cpu().numpy(),
                                  "pgain": pgain.cpu().numpy(), "fgain": fgain.cpu().numpy()}
            archive.update({f"step{t}_{k}": x for k, x in metrics_by_step[t].items()})
        if time.monotonic()-start > protocol["runtime_limit_per_seed_seconds"]:
            raise TimeoutError("Frozen per-seed runtime limit reached; no partial seed analysis")
    predicted = np.stack(predicted)
    readout_only = np.stack(readout_only)
    native_position_error = float(np.max(np.abs(predicted[:, 0, 1]-reference_pos)))
    if native_position_error > 1e-6:
        raise ValueError(f"Original full moving readout mismatch: {native_position_error}")
    rows = []
    for t in (20, 50, 100, 200):
        m = metrics_by_step[t]
        for context_index, (context, actual) in enumerate((("stationary_zero_input", stat_actual), ("moving_physical_displacements", physical))):
            for i, spec in enumerate(specs):
                metrics, episode_metrics = movement_metrics(predicted[:t+1, i, context_index], actual[:t+1])
                row = {**spec, "context": context, "steps": t, "seconds": t*.02, **metrics,
                       "median_norm_ratio": float(np.median(m["norm"][i, context_index])),
                       "projective_position_spread_ratio": float(m["spread"][i, context_index])}
                if context_index == 0:
                    row.update(mean_projective_perturbation_gain=float(m["pgain"][i].mean()),
                               mean_full_state_perturbation_gain=float(m["fgain"][i].mean()))
                rows.append(row)
                # Positions retain every episode; store endpoint arrays explicitly.
                archive[f"{context}_{t}_{i}_position_error"] = episode_metrics["episode_position_error_m"]
            metrics, _ = movement_metrics(readout_only[:t+1, context_index], actual[:t+1])
            rows.append({"condition": "pgc_readout_only", "mode": "readout_only", "dose": 1., "group": "pgc",
                         "context": context, "steps": t, "seconds": t*.02, **metrics, "state_identical_to_native": True})
    archive.update(predicted_positions=predicted, readout_only_positions=readout_only)
    validation = {"native_original_state_max_error": replay_error,
                  "native_original_position_max_error_m": native_position_error,
                  "positive_homogeneity_relative_error": identity_error, **perturbation_validation}
    result = {"seed": seed, "protocol_sha256": sha(PROTOCOL), "frozen_manifest_sha256": sha(out/"frozen_manifest.json"),
              "public_commit": manifest["commit"], "checkpoint_sha256": sha(checkpoint),
              "classification_sha256": sha(class_path), "predictive_count": len(predictive),
              "configuration": vars(config), "matching_balance": balance,
              "specifications": specs, "rows": rows, "validation": validation,
              "elapsed_seconds": time.monotonic()-start}
    np.savez_compressed(dest/"arrays.npz", **archive)
    (dest/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"seed": seed, "complete": True, "elapsed_seconds": result["elapsed_seconds"], "predictive_count": len(predictive), "validation": validation}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.freeze_only:
        freeze(args.out)
    elif args.seed is None:
        parser.error("--seed is required unless --freeze-only")
    else:
        evaluate(args.seed, args.out, torch.device(args.device))
