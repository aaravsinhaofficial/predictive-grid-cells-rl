"""Goal-bearing cell-state interventions in the original reward-trained agents.

Diagnostic readouts are fitted on separate native episodes. Policy weights,
the square arena, visual observations, six actions and goal-snapshot semantics
are unchanged. A decoder-directed edit is not a manifold tangent or ring claim.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena


INPUTS = ("native", "goal_ordinary", "current_ordinary")
PULSES = ("baseline", "goal_angle", "random", "decoder_silent")
ENV_FIELDS = ("pos", "goal", "mdir", "spawn", "hd", "vf", "vs", "vr", "t", "partial", "ep_idx",
              "draw_idx", "travelled", "ep_return", "n_goals", "first_goal_t", "path_eff_sum", "path_eff_n",
              "land_pos", "stripe_phase")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def labels(pos, heading, goal, half):
    delta = goal - pos
    bearing = torch.atan2(delta[:, 1], delta[:, 0]) - heading
    return torch.cat((torch.stack((bearing.cos(), bearing.sin()), 1), pos / half,
                      torch.stack((heading.cos(), heading.sin()), 1), delta.norm(dim=1, keepdim=True) / half), 1)


def angular_error(prediction, truth):
    return np.abs(wrap(np.arctan2(prediction[..., 1], prediction[..., 0])
                       - np.arctan2(truth[..., 1], truth[..., 0]))) * 180 / np.pi


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9)),
            "min": float(values.min()), "max": float(values.max())}


@torch.inference_mode()
def collect_native(net, pop, cfg, count, seed, device, horizon, calibration=False, stride=8):
    """Actual native policy episodes; c/h are AFTER the current input, before action."""
    env = MatchedArena(cfg, count, 1, seed, device)
    obs = env.reset_all()
    h, c = net.zero_state(count, device)
    goal_native = torch.zeros(count, cfg["n_cells"], device=device)
    goal_plain = torch.zeros_like(goal_native)
    memory = {key: torch.zeros(count, 2, device=device) for key in ("pos", "dir", "vel")}
    pa = torch.full((count,), -1, dtype=torch.long, device=device)
    pr = torch.zeros(count, device=device)
    age = torch.zeros(count, dtype=torch.long, device=device)
    action_rng = torch.Generator(device=device).manual_seed(seed + 1)
    capture_at = np.full(count, -1, dtype=int)
    samples, snapshots = [], {}
    input_size = net.lstm.input_size
    future = {name: np.zeros((horizon, count, input_size), np.float32) for name in INPUTS} if not calibration else {}
    future_labels = np.zeros((horizon, count, 7), np.float32) if not calibration else None
    future_hits = np.zeros((horizon, count), bool) if not calibration else None
    started, last = time.monotonic(), time.monotonic()
    for step in range(cfg["episode_len"]):
        code = pop.activity(obs["pos"], obs["dir"], obs["vel"])
        plain = pop.activity(obs["pos"], obs["dir"], obs["vel"], shift_scale=0)
        embedding = net.embed(obs["rgb"])
        x = net._concat(embedding, code, goal_native, pa, pr, None)
        y = labels(obs["pos"], obs["hd"], env.goal, cfg["arena_half"])
        distance = (env.goal - obs["pos"]).norm(dim=1)
        if not calibration:
            times = step - capture_at
            active = np.flatnonzero((capture_at >= 0) & (times >= 1) & (times <= horizon))
            if len(active):
                tensors = {"native": x,
                           "goal_ordinary": net._concat(embedding, code, goal_plain, pa, pr, None),
                           "current_ordinary": net._concat(embedding, plain, goal_native, pa, pr, None)}
                for name, tensor in tensors.items():
                    future[name][times[active] - 1, active] = tensor[active].cpu().numpy()
                future_labels[times[active] - 1, active] = y[active].cpu().numpy()
        h, c = net.lstm(x, (h, c))
        known = env.n_goals > 0
        if calibration and step % stride == 0:
            eligible = known & (age >= 5) & (distance >= .6)
            if eligible.any():
                samples.append({"c": c[eligible].cpu().numpy(), "h": h[eligible].cpu().numpy(),
                                "target": y[eligible].cpu().numpy(),
                                "episode": torch.arange(count, device=device)[eligible].cpu().numpy(),
                                "step": np.full(int(eligible.sum()), step, dtype=int)})
        if not calibration and step + horizon < cfg["episode_len"]:
            eligible = known & (age >= 15) & (distance >= .9) & (distance <= 1.8)
            new = np.flatnonzero(eligible.cpu().numpy() & (capture_at < 0))
            for slot in new:
                snapshots[int(slot)] = {**{key: getattr(env, key)[slot].cpu().numpy().copy() for key in ENV_FIELDS},
                    "h": h[slot].cpu().numpy().copy(), "c": c[slot].cpu().numpy().copy(),
                    "goal_native": goal_native[slot].cpu().numpy().copy(), "goal_plain": goal_plain[slot].cpu().numpy().copy(),
                    "target": y[slot].cpu().numpy().copy(), "slot": np.array(slot), "capture_step": np.array(step),
                    **{"memory_" + key: tensor[slot].cpu().numpy().copy() for key, tensor in memory.items()}}
                capture_at[slot] = step
        probabilities = net.pi(h).softmax(1)
        uniform = torch.rand(count, generator=action_rng, device=device)
        action = (uniform[:, None] > probabilities.cumsum(1)).sum(1).clamp_max(5)
        obs, reward, done, info = env.step(action)
        if not calibration:
            times = step - capture_at
            active = np.flatnonzero((capture_at >= 0) & (times >= 0) & (times < horizon))
            future_hits[times[active], active] = info["hit"][active].cpu().numpy()
        if info["hit"].any():
            native_hit = pop.activity(info["hit_pos"], info["hit_dir"], info["hit_vel"])
            plain_hit = pop.activity(info["hit_pos"], info["hit_dir"], info["hit_vel"], shift_scale=0)
            goal_native = torch.where(info["hit"][:, None], native_hit, goal_native)
            goal_plain = torch.where(info["hit"][:, None], plain_hit, goal_plain)
            for key in memory:
                memory[key] = torch.where(info["hit"][:, None], info["hit_" + key], memory[key])
        pa, pr = action, reward.clamp(-cfg["reward_clip"], cfg["reward_clip"])
        age = torch.where(info["hit"] | done, torch.zeros_like(age), age + 1)
        if done.any():
            assert step == cfg["episode_len"] - 1
        if time.monotonic() - last > 25:
            print(json.dumps({"collect": "calibration" if calibration else "evaluation", "step": step,
                              "captured": len(snapshots), "seconds": time.monotonic() - started}), flush=True)
            last = time.monotonic()
    if calibration:
        if not samples:
            raise RuntimeError("Native policy produced no eligible calibration return states")
        return {key: np.concatenate([s[key] for s in samples]) for key in samples[0]}
    slots = sorted(snapshots)
    if not slots:
        raise RuntimeError("Native policy produced no eligible evaluation return states")
    saved = {key: np.stack([snapshots[slot][key] for slot in slots]) for key in snapshots[slots[0]]}
    saved.update({"replay_" + key: value[:, slots] for key, value in future.items()})
    saved["replay_target"] = future_labels[:, slots]
    saved["replay_hit"] = future_hits[:, slots]
    saved["eligible_episode_count"] = np.array(len(slots))
    saved["attempted_episode_count"] = np.array(count)
    return saved


class NonlinearChecker(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.layers = nn.Sequential(nn.Linear(len(mean), 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, c):
        return self.layers((c - self.mean) / self.scale)


def fit_readouts(data, seed):
    """Disjoint episode banks fit the linear intervention and nonlinear checker."""
    episodes = np.unique(data["episode"])
    if len(episodes) < 16:
        raise ValueError("At least16 calibration episodes with return states are required")
    first, second = len(episodes) // 2, 3 * len(episodes) // 4
    linear_mask = np.isin(data["episode"], episodes[:first])
    nonlinear_mask = np.isin(data["episode"], episodes[first:second])
    validation_mask = np.isin(data["episode"], episodes[second:])
    c = data["c"].astype(np.float64)
    y = data["target"].astype(np.float64)
    mean = c[linear_mask].mean(0)
    scale = max(float(np.sqrt(np.mean((c[linear_mask] - mean) ** 2))), 1e-6)
    z = (c - mean) / scale
    target_mean = y[linear_mask].mean(0)
    u, singular, vt = np.linalg.svd(z[linear_mask], full_matrices=False)
    uy = u.T @ (y[linear_mask] - target_mean)
    candidates = []
    for alpha in (.01, .1, 1., 10., 100., 1000.):
        weight = vt.T @ ((singular / (singular ** 2 + alpha))[:, None] * uy)
        pred = z[validation_mask] @ weight + target_mean
        candidates.append((float(np.mean((pred[:, :2] - y[validation_mask, :2]) ** 2)), alpha, weight))
    _, alpha, standardized_weight = min(candidates, key=lambda row: row[0])
    weight = standardized_weight / scale
    bias = target_mean - mean @ weight
    torch.manual_seed(seed)
    checker = NonlinearChecker(mean, scale)
    optimizer = torch.optim.Adam(checker.parameters(), lr=.003)
    train_x = torch.tensor(c[nonlinear_mask], dtype=torch.float32)
    train_y = torch.tensor(y[nonlinear_mask, :2], dtype=torch.float32)
    valid_x = torch.tensor(c[validation_mask], dtype=torch.float32)
    valid_y = torch.tensor(y[validation_mask, :2], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 1)
    best, best_loss = None, np.inf
    for update in range(600):
        index = torch.randint(len(train_x), (min(512, len(train_x)),), generator=generator)
        loss = (checker(train_x[index]) - train_y[index]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (update + 1) % 20 == 0:
            with torch.no_grad():
                value = float((checker(valid_x) - valid_y).square().mean())
            if value < best_loss:
                best_loss, best = value, copy.deepcopy(checker.state_dict())
    checker.load_state_dict(best); checker.eval().requires_grad_(False)
    with torch.no_grad():
        nonlinear = checker(valid_x).numpy()
    linear = c[validation_mask] @ weight + bias
    radius = float(np.sqrt(np.mean(np.sum((c[linear_mask] - mean) ** 2, axis=1))))
    report = {"linear_training_episodes": episodes[:first].tolist(), "nonlinear_training_episodes": episodes[first:second].tolist(),
        "validation_episodes": episodes[second:].tolist(), "ridge_alpha": alpha, "cell_state_radius": radius,
        "linear_validation_goal_bearing_mae_deg": float(angular_error(linear, y[validation_mask]).mean()),
        "nonlinear_validation_goal_bearing_mae_deg": float(angular_error(nonlinear, y[validation_mask]).mean()),
        "validation_nuisance_coordinate_rmse": np.sqrt(np.mean((linear[:, 2:] - y[validation_mask, 2:]) ** 2, axis=0)).tolist(),
        "linear_rank": int(np.linalg.matrix_rank(weight)), "rows": int(len(c)),
        "targets": ["egocentric_goal_cos", "egocentric_goal_sin", "current_x/half", "current_y/half",
                    "heading_cos", "heading_sin", "goal_distance/half"],
        "policy_training_unchanged": True, "no_decoder_output_supplied_as_policy_input": True}
    return weight, bias, checker, report


def make_pulses(snapshot, calibration, weight, bias, checker, args):
    c = snapshot["c"].astype(float)
    n, width = c.shape
    prediction = c @ weight + bias
    rng = np.random.default_rng(args.eval_seed + 200)
    signs = np.where(np.arange(n) % 2, -1., 1.)
    rng.shuffle(signs)
    angle = np.deg2rad(args.angle_degrees) * signs
    desired = prediction.copy()
    desired[:, 0] = prediction[:, 0] * np.cos(angle) - prediction[:, 1] * np.sin(angle)
    desired[:, 1] = prediction[:, 0] * np.sin(angle) + prediction[:, 1] * np.cos(angle)
    delta = (desired - prediction) @ np.linalg.pinv(weight)
    radius = float(np.sqrt(np.mean(np.sum((calibration["c"] - calibration["c"].mean(0)) ** 2, axis=1))))
    limit = args.max_radius_fraction * radius
    scale = np.minimum(1., limit / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))
    delta *= scale[:, None]
    norms = np.linalg.norm(delta, axis=1)
    random = rng.normal(size=(n, width))
    random *= (norms / np.linalg.norm(random, axis=1))[:, None]
    q = np.linalg.qr(weight)[0]
    silent = rng.normal(size=(n, width)); silent -= (silent @ q) @ q.T
    silent *= (norms / np.linalg.norm(silent, axis=1))[:, None]
    changes = np.stack((np.zeros_like(delta), delta, random, silent))
    states = c[None] + changes
    linear = states @ weight + bias
    with torch.no_grad():
        nonlinear = checker(torch.tensor(states, dtype=torch.float32)).numpy()
    linear_shift = wrap(np.arctan2(linear[..., 1], linear[..., 0]) - np.arctan2(linear[0, :, 1], linear[0, :, 0])) * 180 / np.pi
    nonlinear_shift = wrap(np.arctan2(nonlinear[..., 1], nonlinear[..., 0]) - np.arctan2(nonlinear[0, :, 1], nonlinear[0, :, 0])) * 180 / np.pi
    tree = cKDTree(calibration["c"])
    before = tree.query(c)[0]
    after = tree.query(states.reshape(-1, width))[0].reshape(len(PULSES), n)
    # A c-only intervention deliberately holds the actor-facing h fixed. Report
    # any departure from the instantaneous LSTM relation h=o*tanh(c), o in [0,1].
    original_h = snapshot["h"][None]
    incompatible = ((np.abs(original_h) > np.abs(np.tanh(states)) + 1e-6)
                    | (original_h * states < -1e-6))
    diagnostics = {"nominal_angle_degrees": args.angle_degrees, "cap_fraction_of_cell_radius": args.max_radius_fraction,
        "cap_applied_fraction": float((scale < 1).mean()), "actual_cell_norm": describe(norms),
        "all_control_norms_match_max_error": float(np.max(np.abs(np.linalg.norm(changes[1:], axis=2) - norms[None]))),
        "target_nuisance_change_max": float(np.abs((changes[1] @ weight)[:, 2:]).max()),
        "silent_linear_readout_change_max": float(np.abs(changes[3] @ weight).max()),
        "linear_fresh_state_mae_deg": float(angular_error(linear[0], snapshot["target"]).mean()),
        "nonlinear_fresh_state_mae_deg": float(angular_error(nonlinear[0], snapshot["target"]).mean()),
        "targeted_sign_agreement_nonlinear": float((nonlinear_shift[1] * signs > 0).mean()),
        "targeted_nonlinear_signed_shift_deg": describe(nonlinear_shift[1] * signs),
        "targeted_linear_signed_shift_deg": describe(linear_shift[1] * signs),
        "nearest_native_state_distance_ratio": {name: describe(after[i] / np.maximum(before, 1e-12)) for i, name in enumerate(PULSES)},
        "h_c_incompatible_coordinate_fraction": {name: float(incompatible[i].mean()) for i, name in enumerate(PULSES)},
        "cell_state_radius_used_for_cap": radius,
        "interpretation": "Constrained decoder-directed c edit, not a manifold tangent; nonlinear agreement is a semantic validity check, not guaranteed."}
    assert diagnostics["all_control_norms_match_max_error"] < 1e-8
    assert diagnostics["target_nuisance_change_max"] < 1e-8
    assert diagnostics["silent_linear_readout_change_max"] < 1e-8
    return states.astype(np.float32), diagnostics, {"cell_changes": changes, "signs": signs, "linear_initial_shift_deg": linear_shift,
                                                  "nonlinear_initial_shift_deg": nonlinear_shift, "nearest_distance": after}


def decoded(state, weight, bias, checker):
    linear = state @ weight + bias
    nonlinear = checker(state)
    return torch.stack((torch.atan2(linear[..., 1], linear[..., 0]),
                        torch.atan2(nonlinear[..., 1], nonlinear[..., 0])), -1)


@torch.inference_mode()
def replay(net, snapshot, cell_states, weight, bias, checker, device):
    n, width = snapshot["c"].shape
    groups = len(INPUTS) * len(PULSES)
    h = torch.tensor(np.tile(snapshot["h"], (groups, 1)), device=device)
    c = torch.tensor(np.tile(cell_states, (len(INPUTS), 1, 1)).reshape(-1, width), device=device)
    initial = c.reshape(len(INPUTS), len(PULSES), n, width)
    norms = (initial[:, 1:] - initial[:, :1]).norm(dim=-1).clamp_min(1e-12)
    traces = {key: [] for key in ("decoded_angles", "cell_error_ratio", "hidden_error_rms", "action_tv")}
    horizon = len(snapshot["replay_native"])
    for step in range(horizon + 1):
        shaped_c = c.reshape(len(INPUTS), len(PULSES), n, width)
        shaped_h = h.reshape_as(shaped_c)
        probabilities = net.pi(h).softmax(-1).reshape(len(INPUTS), len(PULSES), n, 6)
        traces["decoded_angles"].append(decoded(shaped_c, weight, bias, checker).cpu().numpy())
        traces["cell_error_ratio"].append(((shaped_c[:, 1:] - shaped_c[:, :1]).norm(dim=-1) / norms).cpu().numpy())
        traces["hidden_error_rms"].append((shaped_h[:, 1:] - shaped_h[:, :1]).square().mean(-1).sqrt().cpu().numpy())
        traces["action_tv"].append((.5 * (probabilities[:, 1:] - probabilities[:, :1]).abs().sum(-1)).cpu().numpy())
        if step == horizon:
            break
        x = torch.tensor(np.stack([snapshot["replay_" + name][step] for name in INPUTS]), device=device)
        x = x[:, None].expand(-1, len(PULSES), -1, -1).reshape(groups * n, -1)
        h, c = net.lstm(x, (h, c))
    return {key: np.stack(value) for key, value in traces.items()}


@torch.inference_mode()
def closed_loop(net, pop, cfg, snapshot, cell_states, weight, bias, checker, args):
    n, width = snapshot["c"].shape
    groups = len(INPUTS) * len(PULSES)
    env = MatchedArena(cfg, n, groups, args.eval_seed + 300, args.device)
    for key in ENV_FIELDS:
        value = torch.as_tensor(snapshot[key], device=args.device)
        getattr(env, key).copy_(value.repeat(groups, *([1] * (value.ndim - 1))))
    h = torch.tensor(np.tile(snapshot["h"], (groups, 1)), device=args.device)
    c = torch.tensor(np.tile(cell_states, (len(INPUTS), 1, 1)).reshape(-1, width), device=args.device)
    gn = torch.tensor(np.tile(snapshot["goal_native"], (groups, 1)), device=args.device)
    gp = torch.tensor(np.tile(snapshot["goal_plain"], (groups, 1)), device=args.device)
    input_index = torch.arange(len(INPUTS), device=args.device).repeat_interleave(len(PULSES) * n)
    rng = torch.Generator(device=args.device).manual_seed(args.eval_seed + 301)
    first_hit = torch.full((groups * n,), args.horizon, dtype=torch.long, device=args.device)
    received = torch.zeros(groups * n, device=args.device)
    output = {key: [] for key in ("position", "distance", "decoded_angles", "true_bearing", "action", "action_probability", "hit")}
    for step in range(args.horizon + 1):
        delta = env.goal - env.pos
        true_bearing = torch.atan2(delta[:, 1], delta[:, 0]) - env.hd
        output["position"].append(env.pos.cpu().numpy().reshape(len(INPUTS), len(PULSES), n, 2))
        output["distance"].append(delta.norm(dim=1).cpu().numpy().reshape(len(INPUTS), len(PULSES), n))
        output["decoded_angles"].append(decoded(c, weight, bias, checker).cpu().numpy().reshape(len(INPUTS), len(PULSES), n, 2))
        output["true_bearing"].append(true_bearing.cpu().numpy().reshape(len(INPUTS), len(PULSES), n))
        if step == args.horizon:
            break
        probabilities = net.pi(h).softmax(-1)
        uniform = torch.rand(n, generator=rng, device=args.device).repeat(groups)
        action = (uniform[:, None] > probabilities.cumsum(1)).sum(1).clamp_max(5)
        if step == 0:
            assert all(torch.equal(action[:n], action[j*n:(j+1)*n]) for j in range(1, groups))
        obs, reward, done, info = env.step(action)
        first_hit = torch.where(info["hit"] & (first_hit == args.horizon), torch.full_like(first_hit, step), first_hit)
        received += reward
        output["action"].append(action.cpu().numpy().reshape(len(INPUTS), len(PULSES), n))
        output["action_probability"].append(probabilities.cpu().numpy().reshape(len(INPUTS), len(PULSES), n, 6))
        output["hit"].append(info["hit"].cpu().numpy().reshape(len(INPUTS), len(PULSES), n))
        if done.any():
            raise AssertionError("Selected return-trip window crossed its original episode boundary")
        if info["hit"].any():
            new_native = pop.activity(info["hit_pos"], info["hit_dir"], info["hit_vel"])
            new_plain = pop.activity(info["hit_pos"], info["hit_dir"], info["hit_vel"], shift_scale=0)
            gn = torch.where(info["hit"][:, None], new_native, gn)
            gp = torch.where(info["hit"][:, None], new_plain, gp)
        current = pop.activity(obs["pos"], obs["dir"], obs["vel"])
        plain = pop.activity(obs["pos"], obs["dir"], obs["vel"], shift_scale=0)
        current = torch.where((input_index == 2)[:, None], plain, current)
        goal = torch.where((input_index == 1)[:, None], gp, gn)
        x = net._concat(net.embed(obs["rgb"]), current, goal, action,
                        reward.clamp(-cfg["reward_clip"], cfg["reward_clip"]), None)
        h, c = net.lstm(x, (h, c))
    arrays = {key: np.stack(value) for key, value in output.items()}
    arrays["first_hit_step"] = first_hit.cpu().numpy().reshape(len(INPUTS), len(PULSES), n)
    arrays["reward"] = received.cpu().numpy().reshape(len(INPUTS), len(PULSES), n)
    return arrays


def summarize(replayed, behavior, dt):
    results = {}
    for i, input_name in enumerate(INPUTS):
        results[input_name] = {}
        for j, pulse_name in enumerate(PULSES):
            first = behavior["first_hit_step"][i, j]
            horizon = len(behavior["hit"])
            row = {"goal_hit_fraction": float((first < horizon).mean()),
                   "capped_first_goal_seconds": float(np.minimum(first + 1, horizon).mean() * dt),
                   "mean_reward": float(behavior["reward"][i, j].mean())}
            for step in (0, 5, 15, 30, 60):
                if step >= len(replayed["decoded_angles"]):
                    continue
                target = replayed["decoded_angles"][step, i, j]
                base = replayed["decoded_angles"][step, i, 0]
                row[f"replay_{step}_linear_displacement_deg"] = float(np.abs(wrap(target[:, 0] - base[:, 0])).mean() * 180 / np.pi)
                row[f"replay_{step}_nonlinear_displacement_deg"] = float(np.abs(wrap(target[:, 1] - base[:, 1])).mean() * 180 / np.pi)
                if j:
                    row[f"replay_{step}_cell_error_ratio"] = float(replayed["cell_error_ratio"][step, i, j - 1].mean())
                    row[f"replay_{step}_action_tv"] = float(replayed["action_tv"][step, i, j - 1].mean())
            limit = min(15, len(behavior["position"]) - 1)
            path = np.linalg.norm(behavior["position"][:limit + 1, i, j] - behavior["position"][:limit + 1, i, 0], axis=-1)
            row["first_second_mean_path_change_m"] = float(path.mean())
            row["first_second_distance_progress_m"] = float((behavior["distance"][0, i, j] - behavior["distance"][limit, i, j]).mean())
            row["first_second_pre_goal_fraction"] = float((first >= limit).mean())
            results[input_name][pulse_name] = row
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--condition", choices=("pgc33", "gc100"), default="pgc33")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--calibration-episodes", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=872401)
    parser.add_argument("--eval-seed", type=int, default=872501)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--angle-degrees", type=float, default=15.)
    parser.add_argument("--max-radius-fraction", type=float, default=.1)
    parser.add_argument("--out", type=Path, default=ROOT / "results/lstm_goal_state_pulse/pilot_s1")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = ROOT / f"models/{args.condition}/{args.condition}_lstm_s{args.seed}_1000M"
    cfg = json.loads(stem.with_suffix(".json").read_text())
    checkpoint_path = stem.with_suffix(".pt")
    checksum = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["frames"] == 1_000_000_000
    net = build_policy(cfg, args.device).eval().requires_grad_(False)
    net.load_state_dict(checkpoint["policy"])
    pop = build_population(cfg, args.device)
    started = time.monotonic()
    paths = {"calibration": args.out / "calibration.npz", "evaluation": args.out / "evaluation_snapshots.npz"}
    source_hashes = {path: sha(ROOT / path) for path in ("scripts/lstm_goal_state_pulse.py", "scripts/lstm_code_swap.py",
                     "code/nets.py", "code/common.py", "code/env.py", "code/codes.py")}
    protocol = {"stage": "exploratory_original_RL_seed1_pilot", "condition": args.condition, "seed": args.seed,
        "checkpoint_sha256": checksum, "source_hashes": source_hashes,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "inputs": INPUTS, "pulses": PULSES,
        "agent": "Original1B-frame reward-trained LSTM; frozen policy, original2D square arena and six actions",
        "state_timing": "after native input update, before action; pulse changes c only and keeps h unchanged",
        "calibration": "native independent episodes; first half fits linear edit, next quarter nonlinear checker, final quarter validation",
        "evaluation_selection": "first observed known-goal state per new episode with age>=15steps, distance0.9to1.8m and full continuation available",
        "pairing": "identical initial physical states; MatchedArena motor/image noise and action-sampling uniforms across branches",
        "interpretation": "functional goal-bearing cell-state intervention, not an autonomous ring or manifold tangent"}
    old_protocol = args.out / "protocol.json"
    if old_protocol.exists():
        old = json.loads(old_protocol.read_text())
        for key in ("condition", "seed", "checkpoint_sha256", "arguments", "source_hashes"):
            assert old[key] == json.loads(json.dumps(protocol[key])), (key, "cached run configuration differs")
    write_json(old_protocol, protocol)
    if not paths["calibration"].exists():
        data = collect_native(net, pop, cfg, args.calibration_episodes, args.calibration_seed, args.device, args.horizon, True)
        np.savez_compressed(paths["calibration"], **data)
    calibration = dict(np.load(paths["calibration"]))
    if not paths["evaluation"].exists():
        data = collect_native(net, pop, cfg, args.evaluation_episodes, args.eval_seed, args.device, args.horizon)
        np.savez_compressed(paths["evaluation"], **data)
    snapshot = dict(np.load(paths["evaluation"]))
    weight, bias, checker, fit = fit_readouts(calibration, args.eval_seed + 10)
    cell_states, validity, archive = make_pulses(snapshot, calibration, weight, bias, checker, args)
    print(json.dumps({"fit": fit, "intervention_validity": validity}), flush=True)
    np.savez_compressed(args.out / "readout_and_pulses.npz", weight=weight, bias=bias, **archive)
    torch.save(checker.state_dict(), args.out / "nonlinear_checker.pt")
    checker = checker.to(args.device)
    tw = torch.tensor(weight, dtype=torch.float32, device=args.device)
    tb = torch.tensor(bias, dtype=torch.float32, device=args.device)
    replayed = replay(net, snapshot, cell_states, tw, tb, checker, args.device)
    np.savez_compressed(args.out / "same_observation_replay.npz", **replayed)
    behavior = closed_loop(net, pop, cfg, snapshot, cell_states, tw, tb, checker, args)
    np.savez_compressed(args.out / "closed_loop.npz", **behavior)
    assert sha(checkpoint_path) == checksum
    result = {"protocol": protocol, "fit": fit, "intervention_validity": validity,
        "evaluation_episodes_attempted": int(snapshot["attempted_episode_count"]),
        "eligible_return_trials": int(snapshot["eligible_episode_count"]),
        "results": summarize(replayed, behavior, cfg["action_repeat"] / 60),
        "checkpoint_unchanged": True, "elapsed_seconds": time.monotonic() - started,
        "no_population_inference_from_single_seed": True}
    write_json(args.out / "summary.json", result)
    print(json.dumps({"complete": True, "trials": result["eligible_return_trials"], "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
