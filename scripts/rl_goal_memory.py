"""Exploratory reward-only goal-memory transfer in the original square arena.

Only task changes: uniformly circular goal centers (radius .65m), and a saved
actual-hit goal code shown for four policy updates following each reward.
No goal coordinate/angle, auxiliary target, or ring constraint reaches the agent.
"""
from __future__ import annotations

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
from code.common import CodeSource, build_policy, build_population, extra_from_obs
from code.env import ArenaEnv

CONDITIONS = ("gc100", "pgc33")
VARIANTS = ("flash", "flash_reset", "continuous", "continuous_reset", "no_goal")
SOURCE_FILES = ("scripts/rl_goal_memory.py", "code/__init__.py", "code/env.py",
                "code/common.py", "code/codes.py", "code/nets.py", "code/train.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def state_digest(net):
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GoalCircleArena(ArenaEnv):
    """Original dynamics/render/reward/respawn, with a circular goal prior.

    The reset implementation copies ArenaEnv._reset_envs, changing only the goal
    draw. Evaluation groups additionally couple layout, motor and image noise.
    """
    def __init__(self, cfg, n, seed, device, *, radius=.65, paired=False,
                 stagger=False, groups=1):
        self.circle_radius = float(radius)
        self.group_size, self.groups = int(n), int(groups)
        if not 0 < radius < cfg["arena_half"] - .08 - cfg["goal_radius"]:
            raise ValueError("Every goal disk must fit within the walkable square")
        super().__init__(n * groups, half=cfg["arena_half"],
                         episode_len=cfg["episode_len"], action_repeat=cfg["action_repeat"],
                         v_max=cfg["v_max"], rot_max=cfg["rot_max"],
                         goal_radius=cfg["goal_radius"], vision=cfg["vision"],
                         img_noise=cfg["img_noise"],
                         randomize_textures=bool(cfg.get("randomize_textures", 1)),
                         stagger=stagger, paired=paired, pair_seed=seed,
                         seed=seed, device=device)

    def _paired_rngs(self, idx):
        ids = idx.cpu().tolist()
        episodes = self.ep_idx[idx].cpu().tolist()
        draws = self.draw_idx[idx].cpu().tolist()
        rngs = [np.random.default_rng((self.pair_seed * 1_000_003 +
                                     (i % self.group_size) * 7_919 +
                                     ep * 104_729 + draw * 13) & 0x7FFFFFFF)
                for i, ep, draw in zip(ids, episodes, draws)]
        self.draw_idx[idx] += 1
        return rngs

    def _randn(self, *shape):
        if self.paired and self.groups > 1 and shape[0] == self.n:
            base = torch.randn(self.group_size, *shape[1:], generator=self.gen,
                               device=self.device)
            return base.repeat(self.groups, *([1] * (len(shape) - 1)))
        return super()._randn(*shape)

    def _reset_envs(self, mask):
        idx = mask.nonzero().squeeze(1)
        k = idx.numel()
        if not k:
            return
        self.ep_idx[idx] += 1
        self.draw_idx[idx] = 0
        if self.paired:
            rngs = self._paired_rngs(idx)
            angle = torch.tensor([r.uniform(-math.pi, math.pi) for r in rngs],
                                 dtype=torch.float32, device=self.device)
            self.goal[idx] = self.circle_radius * torch.stack((angle.cos(), angle.sin()), 1)
            if self.randomize_textures:
                tl = self.half - .2
                self.land_pos[idx] = torch.tensor(np.array([r.uniform(-tl, tl, 4) for r in rngs]),
                                                 dtype=torch.float32, device=self.device)
                self.stripe_phase[idx] = torch.tensor(np.array([r.uniform(0, 1, 4) for r in rngs]),
                                                     dtype=torch.float32, device=self.device)
        else:
            angle = (self._rand(k) * 2 - 1) * math.pi
            self.goal[idx] = self.circle_radius * torch.stack((angle.cos(), angle.sin()), 1)
        self.t[idx] = 0
        self.partial[idx] = False
        self.ep_return[idx] = 0
        self.n_goals[idx] = 0
        self.first_goal_t[idx] = -1
        self.path_eff_sum[idx] = 0
        self.path_eff_n[idx] = 0
        if self.randomize_textures and not self.paired:
            tl = self.half - .2
            self.land_pos[idx] = (self._rand(k, 4) * 2 - 1) * tl
            self.stripe_phase[idx] = self._rand(k, 4)
        self._respawn(mask)


class GoalFlash:
    """Countdown refers to policy updates still allowed to see the snapshot."""
    def __init__(self, n, n_code, device, steps=4):
        if steps < 1:
            raise ValueError("The goal flash must last at least one policy update")
        self.steps = int(steps)
        self.saved = torch.zeros(n, n_code, device=device)
        self.remaining = torch.zeros(n, dtype=torch.long, device=device)

    def visible(self):
        return self.saved * (self.remaining > 0).unsqueeze(1)

    def advance(self, hit, done, hit_code=None):
        # Called AFTER the current action and environment transition. A hit at
        # transition t makes its snapshot visible at updates t+1,...,t+4.
        expired = (self.remaining == 1) & ~hit & ~done
        self.remaining = (self.remaining - 1).clamp_min(0)
        if bool(hit.any()):
            if hit_code is None or hit_code.shape != self.saved.shape:
                raise ValueError("Actual-hit population code required on a reward")
            self.saved = torch.where(hit[:, None], hit_code, self.saved)
            self.remaining = torch.where(hit, torch.full_like(self.remaining, self.steps), self.remaining)
        self.saved = self.saved * (~done).unsqueeze(1)
        self.remaining = torch.where(done, torch.zeros_like(self.remaining), self.remaining)
        return expired


def evaluation_goal(flash, variants, n):
    visible = flash.visible()
    chunks = []
    for j, variant in enumerate(variants):
        sl = slice(j * n, (j + 1) * n)
        if variant.startswith("continuous"):
            chunks.append(flash.saved[sl])
        elif variant == "no_goal":
            chunks.append(torch.zeros_like(flash.saved[sl]))
        else:
            chunks.append(visible[sl])
    return torch.cat(chunks)


def reset_state(state, mask):
    keep = (~mask).unsqueeze(1)
    return state[0] * keep, state[1] * keep


def require_time(deadline, reserve=0):
    if time.monotonic() + reserve >= deadline:
        raise TimeoutError("The fixed pilot time limit was reached; partial files are retained")


def episode_summary(rows, episode_len):
    found = [r for r in rows if r["first_goal_step"] >= 0]
    efficiencies = [r["path_eff"] for r in rows if r["path_eff_n"] > 0]
    # The aggregate denominator retains all discovered episodes, including
    # those with zero returns. The unconditional return count retains all rows.
    post_time = sum(episode_len - r["first_goal_step"] for r in found)
    return {"episodes": len(rows), "mean_score": float(np.mean([r["score"] for r in rows])),
            "mean_goals": float(np.mean([r["n_goals"] for r in rows])),
            "fraction_discovered": len(found) / len(rows),
            "fraction_at_least_two_goals": float(np.mean([r["n_goals"] >= 2 for r in rows])),
            "mean_returns_per_episode": float(np.mean([max(0, r["n_goals"] - 1) for r in rows])),
            "post_discovery_returns_per_1000_steps":
                1000 * sum(max(0, r["n_goals"] - 1) for r in found) / post_time if post_time else None,
            "conditional_mean_path_efficiency": float(np.mean(efficiencies)) if efficiencies else None,
            "rows": rows}


@torch.inference_mode()
def evaluate(net, cfg, *, device, seed, episodes, n_envs, radius=.65,
             flash_steps=4, deadline=float("inf")):
    if episodes % n_envs:
        raise ValueError("Use complete, equal-sized evaluation episode batches")
    net.eval()
    n, groups = n_envs, len(VARIANTS)
    env = GoalCircleArena(cfg, n, seed, device, radius=radius, paired=True, groups=groups)
    pop = build_population(cfg, device)
    code_src = CodeSource(cfg, pop, device)
    obs = env.reset_all()
    for key in ("pos", "hd", "dir", "rgb"):
        assert all(torch.equal(obs[key][:n], obs[key][j*n:(j+1)*n]) for j in range(1, groups))
    state = net.zero_state(n * groups, device)
    flash = GoalFlash(n * groups, cfg["n_cells"], device, flash_steps)
    pa = torch.full((n * groups,), -1, dtype=torch.long, device=device)
    pr = torch.zeros(n * groups, device=device)
    action_rng = torch.Generator(device=device).manual_seed(seed + 1)
    reset_group = torch.tensor([v.endswith("_reset") for v in VARIANTS], device=device).repeat_interleave(n)
    rows = {v: [] for v in VARIANTS}
    first_hits = []
    reset_count = torch.zeros(groups, dtype=torch.long, device=device)
    for step in range(cfg["episode_len"] * (episodes // n)):
        if step % 100 == 0:
            require_time(deadline, 15)
        goal = evaluation_goal(flash, VARIANTS, n)
        logits, _, state = net.step(obs["rgb"], code_src(obs["pos"], obs["dir"], obs["vel"]),
                                    goal, pa, pr, state, extra_from_obs(cfg, obs))
        uniform = torch.rand(n, generator=action_rng, device=device).repeat(groups)
        action = (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        prior_goal, prior_known = env.goal.clone(), env.n_goals > 0
        obs, raw, done, info = env.step(action)
        hit = info["hit"]
        hit_code = code_src.goal(info["hit_pos"], info["hit_dir"], info["hit_vel"]) if bool(hit.any()) else None
        expired = flash.advance(hit, done, hit_code)
        reset = expired & reset_group
        reset_count += reset.view(groups, n).sum(1)
        state = reset_state(state, done | reset)
        reward = raw.clamp(-cfg["reward_clip"], cfg["reward_clip"]) if cfg["reward_clip"] > 0 else raw
        pa = torch.where(done, torch.full_like(action, -1), action)
        pr = reward * (~done)
        initial_hit = hit[:n] & ~prior_known[:n]
        if bool(initial_hit.any()):
            ids = initial_hit.nonzero().flatten()
            for i in ids.cpu().tolist():
                first_hits.append({"slot": i, "step": step,
                                   "goal": prior_goal[i].cpu().tolist(),
                                   "hit_pos": info["hit_pos"][i].cpu().tolist(),
                                   "hit_dir": info["hit_dir"][i].cpu().tolist()})
        stats = info["ep_stats"]
        if stats is not None:
            for k, index in enumerate(stats["env"].cpu().tolist()):
                group, slot = divmod(index, n)
                rows[VARIANTS[group]].append({"slot": slot, "episode": step // cfg["episode_len"],
                    "score": float(stats["ret"][k]), "n_goals": int(stats["n_goals"][k]),
                    "first_goal_step": int(stats["first_goal_t"][k]),
                    "path_eff": float(stats["path_eff"][k]), "path_eff_n": int(stats["path_eff_n"][k])})
    # All interventions begin after reward, so discovery must match exactly.
    reference = [r["first_goal_step"] for r in rows[VARIANTS[0]]]
    assert len(reference) == episodes
    assert all([r["first_goal_step"] for r in rows[v]] == reference for v in VARIANTS)
    return {"seed": seed, "episodes_per_variant": episodes, "n_envs_per_variant": n,
            "variants": {v: episode_summary(rows[v], cfg["episode_len"]) for v in VARIANTS},
            "all_first_discoveries_identical": True,
            "state_reset_counts": dict(zip(VARIANTS, reset_count.cpu().tolist())),
            "first_hit_geometry": first_hits,
            "interpretation": "Reset clears both h and c, retaining previous action/reward and physical state. Continuous+reset controls nonspecific recurrent disruption; no-goal removes the message entirely."}


def collect_rollout(net, env, code_src, cfg, obs, state, flash, pa, pr, length):
    """The exact actor-visible goal tensors are saved for the gradient replay."""
    state0 = tuple(x.detach() for x in state)
    roll = {k: [] for k in ("rgb", "code", "goal", "pa", "pr", "extra", "a", "r", "d", "actor_logits")}
    episodes = []
    with torch.no_grad():
        for _ in range(length):
            code, goal = code_src(obs["pos"], obs["dir"], obs["vel"]), flash.visible()
            extra = extra_from_obs(cfg, obs)
            logits, _, state = net.step(obs["rgb"], code, goal, pa, pr, state, extra)
            action = torch.distributions.Categorical(logits=logits).sample()
            for key, value in (("rgb", obs["rgb"]), ("code", code), ("goal", goal),
                               ("pa", pa), ("pr", pr), ("extra", extra), ("a", action),
                               ("actor_logits", logits)):
                roll[key].append(value)
            obs, raw, done, info = env.step(action)
            reward = raw.clamp(-cfg["reward_clip"], cfg["reward_clip"]) if cfg["reward_clip"] > 0 else raw
            roll["r"].append(reward)
            roll["d"].append(done.float())
            hit = info["hit"]
            hit_code = code_src.goal(info["hit_pos"], info["hit_dir"], info["hit_vel"]) if bool(hit.any()) else None
            flash.advance(hit, done, hit_code)
            state = reset_state(state, done)
            pa = torch.where(done, torch.full_like(action, -1), action)
            pr = reward * (~done)
            if info["ep_stats"] is not None:
                episodes.extend(info["ep_stats"]["ret"].cpu().tolist())
        bootstrap_goal = flash.visible()
        _, bootstrap, _ = net.step(obs["rgb"], code_src(obs["pos"], obs["dir"], obs["vel"]),
                                   bootstrap_goal, pa, pr, tuple(x.detach() for x in state), extra_from_obs(cfg, obs))
    return roll, state0, bootstrap, bootstrap_goal, obs, state, pa, pr, episodes


def replay(net, roll, state0):
    length, n = len(roll["r"]), roll["r"][0].numel()
    embeddings = net.embed(torch.cat(roll["rgb"], 0)).view(length, n, -1)
    state, logits, values = state0, [], []
    for t in range(length):
        logit, value, state = net.step_from_embedding(embeddings[t], roll["code"][t],
            roll["goal"][t], roll["pa"][t], roll["pr"][t], state, roll["extra"][t])
        logits.append(logit)
        values.append(value)
        state = reset_state(state, roll["d"][t].bool())
    return torch.stack(logits), torch.stack(values)


def train(net, cfg, *, out, device, extra_frames, warmup_updates, seed,
          radius=.65, flash_steps=4, deadline=float("inf"), provenance=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net.train()
    env = GoalCircleArena(cfg, cfg["n_envs"], seed * 1000 + 1, device,
                         radius=radius, stagger=True)
    pop = build_population(cfg, device)
    source = CodeSource(cfg, pop, device)
    opt = torch.optim.RMSprop(net.parameters(), lr=0 if warmup_updates else cfg["lr"],
                             alpha=cfg["rms_alpha"], eps=cfg["rms_eps"])
    n, length = cfg["n_envs"], cfg["rollout"]
    frame_increment = n * length * cfg["action_repeat"]
    updates = math.ceil(extra_frames / frame_increment)
    obs, state = env.reset_all(), net.zero_state(n, device)
    flash = GoalFlash(n, cfg["n_cells"], device, flash_steps)
    pa, pr = torch.full((n,), -1, dtype=torch.long, device=device), torch.zeros(n, device=device)
    initial_hash = state_digest(net)
    records, learning_frames, started = [], 0, time.monotonic()
    recent = []
    out.mkdir(parents=True, exist_ok=True)

    def save(name, completed=False):
        atomic_torch(out / name, {"policy": net.state_dict(), "optimizer": opt.state_dict(),
            "original_frames": 1_000_000_000, "extra_frames": learning_frames,
            "requested_extra_frames": extra_frames, "warmup_updates": warmup_updates,
            "completed": completed, "config": cfg, "provenance": provenance,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None})

    try:
        with (out / "metrics.jsonl").open("w") as log:
            for update in range(warmup_updates + updates):
                require_time(deadline, 30)
                warm = update < warmup_updates
                batch = collect_rollout(net, env, source, cfg, obs, state, flash, pa, pr, length)
                roll, state0, bootstrap, _, obs, state, pa, pr, episode_returns = batch
                target, future = [], bootstrap
                for t in reversed(range(length)):
                    future = roll["r"][t] + cfg["gamma"] * (1 - roll["d"][t]) * future
                    target.append(future)
                returns = torch.stack(list(reversed(target)))
                logits, values = replay(net, roll, state0)
                dist = torch.distributions.Categorical(logits=logits)
                advantage = (returns - values).detach()
                policy_loss = -(dist.log_prob(torch.stack(roll["a"])) * advantage).sum(0).mean()
                value_loss = .5 * (returns - values).square().sum(0).mean()
                entropy = dist.entropy().sum(0).mean()
                loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy"] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite reward-only A2C loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite policy gradient")
                opt.step()
                state = tuple(x.detach() for x in state)
                if warm and update + 1 == warmup_updates:
                    if state_digest(net) != initial_hash:
                        raise AssertionError("Zero-learning-rate optimizer warmup changed policy weights")
                    for group in opt.param_groups:
                        group["lr"] = cfg["lr"]
                    save("after_warmup.pt")
                if not warm:
                    learning_frames += frame_increment
                    recent = (recent + episode_returns)[-100:]
                if update == 0 or (update + 1) % 20 == 0 or update + 1 == warmup_updates + updates:
                    elapsed = time.monotonic() - started
                    record = {"total_updates": update + 1, "warmup": warm,
                              "extra_frames": learning_frames, "all_exposure_frames": (update + 1) * frame_increment,
                              "seconds": elapsed, "all_exposure_fps": (update + 1) * frame_increment / elapsed,
                              "loss": float(loss.detach()), "entropy": float(entropy.detach()) / length,
                              "grad_norm": float(grad_norm), "mean_recent_score": float(np.mean(recent)) if recent else None}
                    records.append(record)
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                if (update + 1) % 50 == 0:
                    save("latest.pt")
        save("final.pt", completed=True)
    except Exception:
        save("partial.pt")
        raise
    return {"requested_extra_frames": extra_frames, "extra_frames": learning_frames,
            "learning_updates": updates, "warmup_updates": warmup_updates,
            "warmup_exposure_frames": warmup_updates * frame_increment,
            "warmup_policy_hash_unchanged": True, "initial_policy_hash": initial_hash,
            "final_policy_hash": state_digest(net), "seconds": time.monotonic() - started,
            "optimizer": "Fresh RMSProp; original settings; zero LR warmup", "metrics": records}


def protocol(args):
    return {"stage": "exploratory_single_seed_feasibility", "original_model_seed": 1,
            "conditions": list(CONDITIONS), "circle_radius_m": .65, "flash_policy_updates": 4,
            "extra_frames_per_condition": args.frames, "warmup_updates": 200,
            "training_seed": args.train_seed, "evaluation_seed": args.eval_seed,
            "evaluation_episodes_per_variant": args.episodes, "evaluation_envs_per_variant": args.n_envs,
            "variants": list(VARIANTS), "no_auxiliary_loss": True, "ring_constraint": False,
            "original_checkpoint_files_modified": False,
            "task_changes": ["Goal centers uniformly on radius0.65m circle in the same square room",
                             "Actual-hit goal code available for four subsequent policy updates only"],
            "inference": "One preselected checkpoint pair; feasibility only, no population-level significance claim",
            "source_sha256": {p: sha256(ROOT / p) for p in SOURCE_FILES},
            "software": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/ring_rl_goal_memory/pilot_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=5_000_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=61001)
    parser.add_argument("--eval-seed", type=int, default=62001)
    parser.add_argument("--max-seconds", type=float, default=6600)
    parser.add_argument("--protocol", type=Path,
                        default=Path("docs/research/rl_goal_memory_pilot_protocol.json"))
    args = parser.parse_args()
    if args.frames < 1 or args.episodes < 1 or args.n_envs < 1 or args.episodes % args.n_envs:
        parser.error("Positive frame/episode budgets and complete evaluation batches are required")
    if not 0 < args.max_seconds <= 6600:
        parser.error("The pilot requires a finite deadline at most6600seconds")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU training, but CUDA is unavailable")
    # The bounded cloud wrapper creates worker.log before starting this command.
    if args.out.exists() and any(p.name != "worker.log" for p in args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory; prior pilot artifacts are immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    design = protocol(args)
    original_files = []
    for condition in CONDITIONS:
        base = ROOT / f"models/{condition}/{condition}_lstm_s1_1000M"
        original_files.extend((base.with_suffix(".json"), base.with_suffix(".pt")))
    design["original_asset_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in original_files}
    frozen = json.loads(args.protocol.read_text())
    for key in ("stage", "original_model_seed", "conditions", "circle_radius_m",
                "flash_policy_updates", "extra_frames_per_condition", "warmup_updates",
                "training_seed", "evaluation_seed", "evaluation_episodes_per_variant",
                "evaluation_envs_per_variant", "variants", "no_auxiliary_loss", "ring_constraint",
                "source_sha256", "original_asset_sha256"):
        if frozen[key] != design[key]:
            raise ValueError(f"Pilot differs from its frozen protocol in {key}")
    design["frozen_protocol_sha256"] = sha256(args.protocol)
    atomic_json(args.out / "protocol.json", design)
    deadline, completed = time.monotonic() + args.max_seconds, {}
    try:
        for condition in CONDITIONS:
            require_time(deadline, 60)
            base = ROOT / f"models/{condition}/{condition}_lstm_s1_1000M"
            cfg = json.loads(base.with_suffix(".json").read_text())
            checkpoint = torch.load(base.with_suffix(".pt"), map_location="cpu", weights_only=True)
            if checkpoint["frames"] != 1_000_000_000 or cfg["seed"] != 1 or cfg["cond"] != condition:
                raise ValueError("The pilot must start from the preselected original1B seed1pair")
            torch.manual_seed(args.train_seed)
            net = build_policy(cfg, args.device)
            net.load_state_dict(checkpoint["policy"], strict=True)
            folder = args.out / condition
            folder.mkdir()
            atomic_json(folder / "original_config.json", cfg)
            before = state_digest(net)
            evaluation_args = dict(device=args.device, seed=args.eval_seed, episodes=args.episodes,
                                   n_envs=args.n_envs, deadline=deadline)
            baseline = evaluate(net, cfg, **evaluation_args)
            if state_digest(net) != before:
                raise AssertionError("Baseline evaluation changed policy parameters")
            atomic_json(folder / "baseline.json", baseline)
            training = train(net, cfg, out=folder, device=args.device, extra_frames=args.frames,
                             warmup_updates=200, seed=args.train_seed, deadline=deadline, provenance=design)
            atomic_json(folder / "training.json", training)
            final = evaluate(net, cfg, **evaluation_args)
            atomic_json(folder / "final_evaluation.json", final)
            completed[condition] = {"baseline": baseline, "training": training, "final": final,
                                    "final_checkpoint_sha256": sha256(folder / "final.pt")}
            atomic_json(args.out / "progress.json", {"completed_conditions": list(completed)})
            del net, checkpoint
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        for path, expected in design["original_asset_sha256"].items():
            if sha256(ROOT / path) != expected:
                raise AssertionError("An original checkpoint/config changed during the pilot")
        atomic_json(args.out / "summary.json", {"protocol": design, "completed": True, "conditions": completed})
    except Exception as error:
        atomic_json(args.out / "incomplete.json", {"completed_conditions": list(completed),
                    "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
