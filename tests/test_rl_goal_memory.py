"""Independent invariants for the square-arena reward-only memory pilot."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from code.common import CodeSource, build_population
from code.env import ArenaEnv
from scripts import rl_goal_memory as study

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / "models/pgc33/pgc33_lstm_s1_1000M.json").read_text())
    cfg.update(vision="none", img_noise=0., n_envs=2, rollout=6, episode_len=2000)
    return cfg


class TinyPolicy(nn.Module):
    """Small ordinary recurrent actor for testing rollout/replay, not an agent arm."""
    def __init__(self):
        super().__init__()
        self.cell = nn.LSTMCell(2 * 512 + 3, 5)
        self.actor, self.critic = nn.Linear(5, 6), nn.Linear(5, 1)

    def zero_state(self, n, device):
        return torch.zeros(n, 5, device=device), torch.zeros(n, 5, device=device)

    def embed(self, image):
        return image.mean((2, 3))

    def step_from_embedding(self, embedded, code, goal, pa, pr, state, extra=None):
        state = self.cell(torch.cat((embedded, code, goal), 1), state)
        return self.actor(state[0]), self.critic(state[0]).squeeze(1), state

    def step(self, image, code, goal, pa, pr, state, extra=None):
        return self.step_from_embedding(self.embed(image), code, goal, pa, pr, state, extra)


class OneHitEnv:
    """A reward at the first transition exposes exact cue boundary timing."""
    def __init__(self, n=2):
        self.n, self.t = n, 0

    def observe(self):
        return {"rgb": torch.zeros(self.n, 3, 84, 84),
                "pos": torch.full((self.n, 2), -.4), "dir": torch.tensor([[1., 0.]]).repeat(self.n, 1),
                "vel": torch.zeros(self.n, 2)}

    def reset_all(self):
        return self.observe()

    def step(self, action):
        hit = torch.full((self.n,), self.t == 0)
        self.t += 1
        info = {"hit": hit, "hit_pos": torch.tensor([[.4, .1]]).repeat(self.n, 1),
                "hit_dir": torch.tensor([[1., 0.]]).repeat(self.n, 1),
                "hit_vel": torch.tensor([[.35, 0.]]).repeat(self.n, 1), "ep_stats": None}
        return self.observe(), hit.float() * 10, torch.zeros(self.n, dtype=torch.bool), info


def test_flash_has_four_updates_after_hit_then_expires():
    flash = study.GoalFlash(1, 3, "cpu", 4)
    yes, no = torch.tensor([True]), torch.tensor([False])
    snapshot = torch.tensor([[2., 3., 4.]])
    assert torch.equal(flash.visible(), torch.zeros_like(snapshot))
    assert not flash.advance(yes, no, snapshot).item()
    for remaining in (4, 3, 2, 1):
        assert flash.remaining.item() == remaining
        assert torch.equal(flash.visible(), snapshot)
        expired = flash.advance(no, no)
        assert expired.item() == (remaining == 1)
    assert flash.remaining.item() == 0
    assert not flash.visible().any()
    assert torch.equal(flash.saved, snapshot)  # only availability, not snapshot, is erased


def test_refresh_and_episode_end_precedence():
    flash = study.GoalFlash(2, 2, "cpu", 4)
    flash.remaining[:] = 1
    new = torch.tensor([[2., 3.], [4., 5.]])
    expired = flash.advance(torch.tensor([True, True]), torch.tensor([False, True]), new)
    assert not expired.any()  # a renewed reward is not a cue-expiry reset
    assert flash.remaining.tolist() == [4, 0]
    assert torch.equal(flash.saved[0], new[0])
    assert not flash.saved[1].any()  # a final-step reward cannot leak into next episode


def test_circle_changes_goals_but_keeps_square_spawns_and_mechanics():
    cfg = config()
    env = study.GoalCircleArena(cfg, 256, 5, "cpu")
    env.reset_all()
    assert torch.allclose(env.goal.norm(dim=1), torch.full((256,), .65), atol=1e-7)
    assert (env.pos.abs() <= 1.17).all()
    assert ((env.pos - env.goal).norm(dim=1) >= .4).all()
    assert env.pos.norm(dim=1).std() > .1  # spawns were not constrained to the circle
    assert study.GoalCircleArena.step is ArenaEnv.step
    assert study.GoalCircleArena.render is ArenaEnv.render
    assert study.GoalCircleArena._respawn is ArenaEnv._respawn
    assert env._targets.shape == (6, 3)
    with pytest.raises(ValueError):
        study.GoalCircleArena(cfg, 1, 0, "cpu", radius=.9)


def test_original_physics_reward_and_respawn_identical_at_matched_state():
    cfg = config()
    circle = study.GoalCircleArena(cfg, 2, 11, "cpu")
    original = ArenaEnv(2, episode_len=2000, vision="none", img_noise=0.,
                        stagger=False, seed=11, device="cpu")
    circle.reset_all()
    original.reset_all()
    circle.pos[:] = circle.goal  # force a valid immediate pickup in both envs
    for key, value in vars(circle).items():
        if isinstance(value, torch.Tensor) and hasattr(original, key):
            setattr(original, key, value.clone())
    original.gen.set_state(circle.gen.get_state())
    for t in range(20):
        action = torch.tensor([t % 6, (t + 2) % 6])
        a, b = circle.step(action), original.step(action)
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        for key in a[0]:
            assert torch.equal(a[0][key], b[0][key]), key
        for key in ("hit", "hit_pos", "hit_dir", "hit_vel", "hit_wall"):
            assert torch.equal(a[3][key], b[3][key]), key
        if t == 0:
            assert torch.equal(a[1], torch.full((2,), 10.))
            assert ((circle.pos - circle.goal).norm(dim=1) >= .4).all()
            assert not torch.equal(a[3]["hit_pos"], circle.pos)  # captured before teleport


def test_evaluation_groups_have_matched_layout_and_physical_noise():
    cfg = config()
    env = study.GoalCircleArena(cfg, 3, 12, "cpu", paired=True, groups=5)
    obs = env.reset_all()
    for t in range(15):
        for value in (env.goal, env.pos, env.hd, env.land_pos, obs["rgb"]):
            assert all(torch.equal(value[:3], value[j*3:(j+1)*3]) for j in range(1, 5))
        obs, _, _, _ = env.step(torch.tensor([t % 6, (t+1) % 6, (t+2) % 6]).repeat(5))


def test_actor_replay_and_bootstrap_use_exact_flash_mask():
    torch.manual_seed(31)
    cfg, env, net = config(), OneHitEnv(), TinyPolicy()
    source = CodeSource(cfg, build_population(cfg, "cpu"), "cpu")
    flash = study.GoalFlash(2, 512, "cpu", 4)
    batch = study.collect_rollout(net, env, source, cfg, env.reset_all(), net.zero_state(2, "cpu"),
                                 flash, torch.full((2,), -1), torch.zeros(2), 5)
    roll, state0, _, boot_goal = batch[:4]
    actual_hit = source.goal(torch.tensor([[.4, .1]]).repeat(2, 1),
                             torch.tensor([[1., 0.]]).repeat(2, 1),
                             torch.tensor([[.35, 0.]]).repeat(2, 1))
    assert not roll["goal"][0].any()
    for t in range(1, 5):
        assert torch.equal(roll["goal"][t], actual_hit)
    assert not boot_goal.any()  # bootstrap is the FIRST blank update
    replay_logits, _ = study.replay(net, roll, state0)
    torch.testing.assert_close(replay_logits, torch.stack(roll["actor_logits"]), atol=2e-7, rtol=2e-6)
    # Saved rollout tensors must survive future snapshot refreshes unchanged.
    flash.advance(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), actual_hit + 2)
    assert torch.equal(roll["goal"][1], actual_hit)


def test_state_reset_control_keeps_continuous_message_available():
    flash = study.GoalFlash(5, 2, "cpu", 4)
    flash.saved[:] = torch.tensor([2., 3.])
    flash.remaining[:] = 0
    result = study.evaluation_goal(flash, study.VARIANTS, 1)
    assert torch.equal(result[2], flash.saved[2])
    assert torch.equal(result[3], flash.saved[3])
    assert not result[[0, 1, 4]].any()
    h, c = torch.ones(5, 3), torch.ones(5, 3) * 2
    out = study.reset_state((h, c), torch.tensor([False, True, False, True, False]))
    assert not out[0][[1, 3]].any() and not out[1][[1, 3]].any()
    assert torch.equal(out[0][[0, 2, 4]], h[[0, 2, 4]])


def test_full_evaluation_first_discovery_and_expiry_reset_controls(monkeypatch):
    original = study.GoalCircleArena

    class FirstPickupArena(original):
        def reset_all(self):
            super().reset_all()
            self.pos[:] = self.goal  # deterministic reward to exercise all branches
            return self.observe()

    monkeypatch.setattr(study, "GoalCircleArena", FirstPickupArena)
    cfg = config()
    cfg["episode_len"] = 12
    result = study.evaluate(TinyPolicy(), cfg, device="cpu", seed=70, episodes=2, n_envs=2)
    assert result["all_first_discoveries_identical"]
    assert result["state_reset_counts"]["flash_reset"] >= 2
    assert result["state_reset_counts"]["continuous_reset"] >= 2
    assert result["state_reset_counts"]["flash"] == 0
    assert result["state_reset_counts"]["continuous"] == 0
    assert result["state_reset_counts"]["no_goal"] == 0
    assert all(v["rows"][0]["first_goal_step"] == 0 for v in result["variants"].values())


def test_zero_lr_warmup_preserves_policy_and_counts_extra_frames(tmp_path, monkeypatch):
    torch.manual_seed(50)
    cfg, net = config(), TinyPolicy()
    cfg.update(rollout=2)
    monkeypatch.setattr(study, "GoalCircleArena", lambda cfg, n, seed, device, **kw: OneHitEnv(n))
    report = study.train(net, cfg, out=tmp_path, device="cpu", extra_frames=17,
                         warmup_updates=2, seed=50)
    assert report["warmup_policy_hash_unchanged"]
    assert report["warmup_exposure_frames"] == 32
    assert report["extra_frames"] == 32
    assert report["initial_policy_hash"] != report["final_policy_hash"]
    warm = torch.load(tmp_path / "after_warmup.pt", weights_only=True)
    assert warm["extra_frames"] == 0
    assert any(v["square_avg"].abs().sum() > 0 for v in warm["optimizer"]["state"].values())
    final = torch.load(tmp_path / "final.pt", weights_only=True)
    assert final["completed"] and final["original_frames"] == 1_000_000_000


def test_failed_returns_remain_in_memory_endpoint():
    rows = [{"first_goal_step": 10, "score": 10., "n_goals": 1, "path_eff": 0., "path_eff_n": 0},
            {"first_goal_step": 10, "score": 20., "n_goals": 2, "path_eff": .8, "path_eff_n": 1},
            {"first_goal_step": -1, "score": 0., "n_goals": 0, "path_eff": 0., "path_eff_n": 0}]
    summary = study.episode_summary(rows, 100)
    assert summary["mean_returns_per_episode"] == pytest.approx(1/3)
    assert summary["post_discovery_returns_per_1000_steps"] == pytest.approx(1000/180)
    assert summary["fraction_at_least_two_goals"] == pytest.approx(1/3)
