"""Map the effective aiming point of an ORIGINAL reward-trained navigation policy.

This is a clamped-input diagnostic, not an autonomous-attractor test. The real
square-arena renderer, 512-cell maps, saved pickup code, and six-action actor
are used unchanged. Only predictive goal slots are rotated or made ordinary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from code.common import build_env, build_policy, build_population

VARIANTS = {"ordinary": None, "native": 0., "left": math.pi/2,
            "reverse": math.pi, "right": -math.pi/2}
ACTION_OFFSET = {2: 0., 3: math.pi, 4: math.pi/2, 5: -math.pi/2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aiming_point(position, direction, weight, mask):
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    w = np.sqrt(np.maximum(weight[mask], 1e-9))
    return np.linalg.lstsq(normal[mask]*w[:, None],
                          (normal[mask]*position[mask]).sum(1)*w, rcond=None)[0]


def angular_error(position, direction, point):
    delta = point-position
    delta /= np.linalg.norm(delta, axis=1, keepdims=True).clip(1e-9)
    return np.degrees(np.arccos(np.clip((direction*delta).sum(1), -1, 1)))


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    cfgpath = ROOT/f"models/pgc33/pgc33_lstm_s{args.seed}_1000M.json"
    cfg = json.loads(cfgpath.read_text())
    ckpath = cfgpath.with_suffix(".pt")
    checkpoint = torch.load(ckpath, map_location="cpu", weights_only=True)
    assert checkpoint["frames"] == 1_000_000_000
    net = build_policy(cfg, args.device).eval().requires_grad_(False)
    net.load_state_dict(checkpoint["policy"])
    pop = build_population(cfg, args.device)
    old = ROOT/f"results/lstm_action_computation_1B/on_policy/pgc33_s{args.seed}.json"
    strategy = json.loads(old.read_text())["action_strategy"]
    go, turn = strategy["dominant_translation_index"], strategy["dominant_turn_index"]
    goal = np.array([0., 0.], dtype=np.float32)
    pickup = np.array([-.29, 0.], dtype=np.float32)
    # Checkerboard spatial split is fixed without examining network responses.
    coordinates = np.linspace(-.95, .95, args.side)
    positions, split = [], []
    for i, x in enumerate(coordinates):
        for j, y in enumerate(coordinates):
            if np.linalg.norm([x, y]) > .4:
                positions.append([x, y]); split.append((i+j) % 2 == 0)
    positions = np.asarray(positions, dtype=np.float32)
    train = np.asarray(split)
    headings = np.arange(args.headings)*2*np.pi/args.headings-math.pi
    goal_codes = {}
    pp = torch.tensor(pickup[None], device=args.device)
    for name, angle in VARIANTS.items():
        a = 0. if angle is None else angle
        direction = torch.tensor([[math.cos(a), math.sin(a)]], device=args.device)
        goal_codes[name] = pop.activity(pp, direction, direction*.25,
                                       shift_scale=0. if angle is None else 1.)
    for name in VARIANTS:
        assert torch.equal(goal_codes[name][:, ~pop.is_pgc],
                           goal_codes["native"][:, ~pop.is_pgc])
    no_noise_cfg = dict(cfg, img_noise=0.)
    template = build_env(no_noise_cfg, 1, args.scene_seed, args.device, stagger=False, paired=True)
    template.reset_all()
    probabilities = {step: {name: [] for name in VARIANTS} for step in args.steps}
    for start in range(0, len(positions), args.position_batch):
        places = positions[start:start+args.position_batch]
        count = len(places)*args.headings
        env = build_env(no_noise_cfg, count, args.scene_seed, args.device,
                        stagger=False, paired=True)
        env.pos[:] = torch.tensor(np.repeat(places, args.headings, axis=0), device=args.device)
        env.hd[:] = torch.tensor(np.tile(headings, len(places)), device=args.device)
        env.mdir[:] = torch.tensor([1., 0.], device=args.device)
        env.land_pos[:] = template.land_pos
        env.stripe_phase[:] = template.stripe_phase
        observation = env.observe()
        embedding = net.embed(observation["rgb"], chunk=256)
        current = pop.activity(observation["pos"], observation["dir"], observation["vel"])
        previous_action = torch.full((count,), turn, dtype=torch.long, device=args.device)
        reward = torch.zeros(count, device=args.device)
        inputs = [net._concat(embedding, current, goal_codes[name].expand(count, -1),
                              previous_action, reward, None) for name in VARIANTS]
        x = torch.cat(inputs)
        state = net.zero_state(len(x), args.device)
        for step in range(1, max(args.steps)+1):
            state = net.lstm(x, state)
            if step in probabilities:
                prob = net.pi(state[0]).softmax(-1).cpu().numpy()
                for index, name in enumerate(VARIANTS):
                    probabilities[step][name].append(prob[index*count:(index+1)*count].reshape(len(places), args.headings, 6))
        print(json.dumps({"seed": args.seed, "positions_complete": start+len(places),
                          "positions_total": len(positions)}), flush=True)
    arrays = {"positions": positions, "headings": headings, "training_positions": train,
              "goal": goal, "pickup": pickup}
    rows = {}
    movement_direction = np.exp(1j*(headings+ACTION_OFFSET[go]))
    for step, by_name in probabilities.items():
        rows[str(step)] = {}
        for name, values in by_name.items():
            prob = np.concatenate(values)
            go_prob = prob[:, :, go]
            moment = (go_prob*movement_direction).sum(1)/go_prob.sum(1).clip(1e-12)
            direction = np.column_stack((np.cos(np.angle(moment)), np.sin(np.angle(moment))))
            strength = np.abs(moment)
            weight = strength**2
            point = aiming_point(positions, direction, weight, train)
            errors = angular_error(positions, direction, point)
            weight_test = weight[~train]/weight[~train].sum().clip(1e-12)
            rows[str(step)][name] = {"fitted_aimpoint_xy": point.tolist(),
                "heldout_weighted_angle_error_deg": float(errors[~train]@weight_test),
                "mean_heading_tuning_strength": float(strength.mean()),
                "low_tuning_position_fraction": float((strength < .1).mean()),
                "heldout_forward_ray_fraction": float(np.mean(errors[~train] < 90)),
                "mean_go_probability": float(go_prob.mean()),
                "mean_dominant_pair_probability": float(prob[:, :, [go, turn]].sum(-1).mean()),
                "pickup_target_error_deg": float(angular_error(positions, direction, pickup)[~train]@weight_test),
                "true_center_target_error_deg": float(angular_error(positions, direction, goal)[~train]@weight_test)}
            arrays[f"{name}_step{step}_probability"] = prob
            arrays[f"{name}_step{step}_field"] = direction
            arrays[f"{name}_step{step}_strength"] = strength
    result = {"seed": args.seed, "original_RL_checkpoint_sha256": sha(ckpath),
        "frames_trained": checkpoint["frames"], "source_sha256": sha(__file__),
        "probe": "clamped observer positions/headings; identical physical sensors/current code; only 169 saved goal PGC slots changed",
        "no_attractor_claim": True, "initial_state": "zero; compare fixed 1/15/60-step histories explicitly",
        "previous_action_clamped": turn, "dominant_translation": go,
        "scene_seed": args.scene_seed, "grid_side": args.side, "heading_count": args.headings,
        "map_identity": pop.describe(), "results": rows, "elapsed_seconds": time.monotonic()-started}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/f"pgc33_s{args.seed}"
    path.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    print(json.dumps({"results": rows, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--side", type=int, default=7)
    parser.add_argument("--headings", type=int, default=64)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 15, 60])
    parser.add_argument("--position-batch", type=int, default=4)
    parser.add_argument("--scene-seed", type=int, default=989131)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT/"results/rl_map_mechanism/policy_map_pilot")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
