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
from scipy.optimize import least_squares
import torch
from code.common import build_env, build_policy, build_population

VARIANTS = {"ordinary": None, "native": 0., "left": math.pi/2,
            "reverse": math.pi, "right": -math.pi/2}
ACTION_OFFSET = {2: 0., 3: math.pi, 4: math.pi/2, 5: -math.pi/2}


def wrap(angle):
    return np.angle(np.exp(1j*angle))


def fit_field_models(position, direction, common_weight, train, point_bound=2.5):
    """Competing spatial-target versus global-angle explanations, fitted on train only.

    Same position weights apply to every variant. Both models have a free common
    angular reference. The target model varies the aiming point; its alternative
    varies only the angular offset. Heldout directions decide descriptive fit.
    """
    names = list(VARIANTS)
    angle = np.stack([np.arctan2(direction[n][:,1], direction[n][:,0]) for n in names])
    initial_points = np.stack([aiming_point(position, direction[n], common_weight, train) for n in names])
    initial_points = np.clip(initial_points, -point_bound+.01, point_bound-.01)
    initial_bearing = np.arctan2(initial_points[:,None,1]-position[None,:,1],
                                initial_points[:,None,0]-position[None,:,0])
    beta = np.angle(np.sum(common_weight[None,train]*np.exp(1j*(angle-initial_bearing)[:,train]),axis=1))
    training_weight = common_weight[train]/max(float(common_weight[train].mean()),1e-12)
    test_weight = common_weight[~train]/max(float(common_weight[~train].sum()),1e-12)
    output = {}
    for model in ('vary_target_shared_offset','shared_target_vary_offset'):
        if model == 'vary_target_shared_offset':
            initial = np.r_[initial_points.ravel(), np.angle(np.exp(1j*beta).sum())]
            lower = np.r_[np.full(10,-point_bound),-np.pi]
            upper = np.r_[np.full(10,point_bound),np.pi]
        else:
            initial = np.r_[initial_points.mean(0), beta]
            lower = np.r_[np.full(2,-point_bound),np.full(5,-np.pi)]
            upper = np.r_[np.full(2,point_bound),np.full(5,np.pi)]
        def unpack(v):
            if model == 'vary_target_shared_offset': return v[:10].reshape(5,2), np.repeat(v[-1],5)
            return np.repeat(v[:2][None],5,0), v[2:]
        def angular_residual(v):
            q,b = unpack(v)
            pred = np.arctan2(q[:,None,1]-position[None,:,1],q[:,None,0]-position[None,:,0])+b[:,None]
            return wrap(pred-angle)
        fitted = least_squares(lambda v:(angular_residual(v)[:,train]*np.sqrt(training_weight)[None]).ravel(),
                               initial, bounds=(lower,upper), max_nfev=1000, ftol=1e-10,xtol=1e-10,gtol=1e-10)
        points, offsets = unpack(fitted.x)
        error = np.degrees(np.abs(angular_residual(fitted.x)))
        test_error = error[:,~train]@test_weight
        output[model] = {'aimpoints_xy': {name: points[j].tolist() for j,name in enumerate(names)},
            'angular_offsets_deg': {name: float(np.degrees(offsets[j])) for j,name in enumerate(names)},
            'heldout_angle_error_deg': {name: float(test_error[j]) for j,name in enumerate(names)},
            'mean_heldout_angle_error_deg': float(test_error.mean()),
            'training_angle_rmse_deg': float(np.sqrt(np.mean(error[:,train]**2))),
            'optimizer_success': bool(fitted.success), 'optimizer_nfev': int(fitted.nfev),
            'point_bound_m':point_bound,'point_bound_reached':bool(np.max(np.abs(points))>point_bound-1e-4)}
    output['target_model_heldout_improvement_deg'] = (output['shared_target_vary_offset']['mean_heldout_angle_error_deg']-
                                                      output['vary_target_shared_offset']['mean_heldout_angle_error_deg'])
    return output


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
    goal = np.array([args.goal_x, args.goal_y], dtype=np.float32)
    arrival = np.array([math.cos(args.arrival_angle), math.sin(args.arrival_angle)], dtype=np.float32)
    pickup = goal-.29*arrival
    assert np.max(np.abs(goal)) < cfg['arena_half']-.3
    assert np.max(np.abs(pickup)) < cfg['arena_half']-.08
    # Checkerboard spatial split is fixed without examining network responses.
    coordinates = np.linspace(-.95, .95, args.side)
    positions, split = [], []
    for i, x in enumerate(coordinates):
        for j, y in enumerate(coordinates):
            if np.linalg.norm(np.array([x, y])-goal) > .4:
                positions.append([x, y]); split.append((i+j) % 2 == 0)
    positions = np.asarray(positions, dtype=np.float32)
    train = np.asarray(split)
    headings = np.arange(args.headings)*2*np.pi/args.headings-math.pi
    goal_codes = {}
    pp = torch.tensor(pickup[None], device=args.device)
    for name, angle in VARIANTS.items():
        a = args.arrival_angle+(0. if angle is None else angle)
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
        env.mdir[:] = torch.tensor(arrival, device=args.device)
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
              "goal": goal, "pickup": pickup, "arrival": arrival}
    rows, fitted_models = {}, {}
    movement_direction = np.exp(1j*(headings+ACTION_OFFSET[go]))
    for step, by_name in probabilities.items():
        rows[str(step)] = {}
        direction_all, strength_all, direction_go = {}, {}, {}
        for name, values in by_name.items():
            prob = np.concatenate(values)
            go_prob = prob[:, :, go]
            moment = (go_prob*movement_direction).sum(1)/go_prob.sum(1).clip(1e-12)
            direction = np.column_stack((np.cos(np.angle(moment)), np.sin(np.angle(moment))))
            strength = np.abs(moment)
            direction_go[name] = direction
            all_moment = (prob[:,:,2:6]*np.exp(1j*(headings[:,None]+np.array([ACTION_OFFSET[a] for a in range(2,6)])))).sum((1,2))
            all_moment /= prob[:,:,2:6].sum((1,2)).clip(1e-12)
            direction_all[name] = np.column_stack((np.cos(np.angle(all_moment)),np.sin(np.angle(all_moment))))
            strength_all[name] = np.abs(all_moment)
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
            arrays[f"{name}_step{step}_all_translation_field"] = direction_all[name]
            arrays[f"{name}_step{step}_all_translation_strength"] = strength_all[name]
        common_weight = np.maximum(strength_all['ordinary']**2,1e-9)
        fitted_models[str(step)] = {}
        for field_name,field,weight in (('all_translation',direction_all,common_weight),
                                       ('all_translation_unweighted',direction_all,np.ones(len(positions))),
                                       ('dominant_translation',direction_go,common_weight)):
            fitted = fit_field_models(positions,field,weight,train)
            points = fitted['vary_target_shared_offset']['aimpoints_xy']
            displacement = np.asarray(points['native'])-np.asarray(points['ordinary'])
            fitted['native_minus_ordinary_xy_m'] = displacement.tolist()
            fitted['native_minus_ordinary_along_arrival_m'] = float(displacement@arrival)
            fitted['native_minus_ordinary_across_arrival_m'] = float(displacement@np.array([-arrival[1],arrival[0]]))
            fitted_models[str(step)][field_name] = fitted
        arrays[f'common_step{step}_position_weight'] = common_weight
    result = {"seed": args.seed, "original_RL_checkpoint_sha256": sha(ckpath),
        "frames_trained": checkpoint["frames"], "source_sha256": sha(__file__),
        "probe": "clamped observer positions/headings; identical physical sensors/current code; only 169 saved goal PGC slots changed",
        "no_attractor_claim": True, "initial_state": "zero; compare the requested fixed-duration clamped histories explicitly",
        "previous_action_clamped": turn, "dominant_translation": go,
        "scene_seed": args.scene_seed, "grid_side": args.side, "heading_count": args.headings,
        "goal_center": goal.tolist(), "pickup": pickup.tolist(), "arrival_direction": arrival.tolist(),
        "horizons": args.steps,
        "all_translation_moment": "Sum p(action|heading)*exp(i*(heading+physical action offset)) across all four translations and uniformly sampled headings, divided by total translation probability",
        "fit_weights": "Same ordinary all-translation tuning strength squared for every variant; uniform-weight and dominant-action robustness retained",
        "fit_models": "Variant targets with one shared angular offset versus one shared target with variant angular offsets; fit training positions only and compare heldout angular errors",
        "legacy_results_note": "The results field retains the original dominant-action ray fit for comparison. Primary revised estimates are field_models/all_translation.",
        "map_identity": pop.describe(), "results": rows, "field_models":fitted_models,
        "dependencies_sha256":{p:sha(ROOT/p) for p in ('code/nets.py','code/env.py','code/codes.py','code/common.py')},
        "elapsed_seconds": time.monotonic()-started}
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
    parser.add_argument("--goal-x", type=float, default=0.)
    parser.add_argument("--goal-y", type=float, default=0.)
    parser.add_argument("--arrival-angle", type=float, default=0.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT/"results/rl_map_mechanism/policy_map_pilot")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)
