"""Independent checks of the simple blackout task's information and timing.

These tests deliberately use native arenas/populations and separate reference
operations rather than the training implementation's own expected outputs.
"""
import copy
import random

import numpy as np
import pytest
import torch

from code.common import build_env, build_policy, build_population, resolve_cond
from code.memory_task import MemoryClock, gate_observation, max_gap_at
from code.train import parse_args


def _config(cond="pgc33", seed=19):
    cfg = resolve_cond(vars(parse_args([
        "--out", "/unused/independent-memory-test", "--cond", cond,
        "--seed", str(seed), "--device", "cpu", "--n_envs", "2",
        "--encoder_norm", "layernorm", "--rms_eps", "1e-5",
    ])))
    return cfg


def _numpy_state_equal(left, right):
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


@pytest.mark.parametrize("gap", [1, 2, 3, 7, 60])
def test_visible_window_and_whole_random_gap_have_exact_observation_counts(gap):
    clock = MemoryClock(1, "cpu", seed=511)
    no = torch.tensor([False])
    yes = ~no
    # An undiscovered goal cannot trigger a blackout, however much time passes.
    for _ in range(40):
        assert not clock.mask().item()
        assert not clock.advance(no, no, gap).item()
    assert not clock.advance(yes, no, gap).item()
    # The observation immediately after the hit is visible observation number 1.
    for index in range(15):
        assert not clock.mask().item(), index
        entering = clock.advance(no, no, gap)
        assert entering.item() == (index == 14)
    duration = int(clock.remaining.item())
    assert (gap + 1) // 2 <= duration <= gap
    for index in range(duration):
        assert clock.mask().item(), index
        assert not clock.advance(no, no, gap).item()
    assert not clock.mask().item()
    assert int(clock.remaining.item()) == 15


def test_native_hit_and_episode_end_override_clock_without_stale_information():
    clock = MemoryClock(3, "cpu", seed=512)
    clock.known[:] = True
    clock.blind[:] = True
    clock.remaining[:] = 1
    entering = clock.advance(torch.tensor([True, True, False]),
                             torch.tensor([False, True, True]), 60)
    assert not entering.any()
    assert not clock.mask().any()
    assert clock.known.tolist() == [True, False, False]
    assert clock.remaining.tolist() == [15, 15, 15]
    # No first-goal state can leak from a coincident terminal reward.
    for _ in range(16):
        clock.advance(torch.zeros(3, dtype=torch.bool),
                      torch.zeros(3, dtype=torch.bool), 60)
        assert not clock.mask()[1:].any()


def test_zero_gap_control_ramp_and_in_progress_gap():
    assert [max_gap_at(t) for t in (0, 3_333_333, 3_333_334, 100_000_000,
                                    199_999_999, 200_000_000, 1_000_000_000)] == [0, 0, 1, 30, 59, 60, 60]
    assert max_gap_at(1_000_000_000, max_gap=0) == 0
    clock = MemoryClock(1, "cpu", seed=513)
    no = torch.tensor([False])
    before = clock.generator.get_state().clone()
    clock.advance(~no, no, 0)
    for _ in range(100):
        assert not clock.mask().any()
        clock.advance(no, no, 0)
    assert clock.draw_steps == 0
    assert torch.equal(before, clock.generator.get_state())
    clock.known[:] = True
    clock.blind[:] = True
    clock.remaining[:] = 2
    clock.advance(no, no, 60)
    assert clock.mask().item() and clock.remaining.item() == 1
    clock.advance(no, no, 60)
    assert not clock.mask().item() and clock.remaining.item() == 15


def test_clock_is_independent_of_global_rng_and_couples_branch_duration_draws():
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    clock = MemoryClock(32, "cpu", seed=514, groups=3)
    no = torch.zeros(96, dtype=torch.bool)
    clock.advance(~no, no, 60)
    for _ in range(15):
        clock.advance(no, no, 60)
    assert clock.mask().all()
    assert 30 <= int(clock.remaining.min()) <= int(clock.remaining.max()) <= 60
    assert torch.equal(clock.remaining[:32], clock.remaining[32:64])
    assert torch.equal(clock.remaining[:32], clock.remaining[64:])
    assert torch.equal(torch_before, torch.get_rng_state())
    assert _numpy_state_equal(numpy_before, np.random.get_state())
    assert python_before == random.getstate()


def test_clock_restore_reproduces_long_stream_and_is_not_aliased():
    original = MemoryClock(4, "cpu", seed=515, groups=2)
    no = torch.zeros(8, dtype=torch.bool)
    original.advance(~no, no, 60)
    for _ in range(21):
        original.advance(no, no, 60)
    snapshot = original.state_dict()
    frozen = copy.deepcopy(snapshot)
    restored = MemoryClock(4, "cpu", seed=999, groups=2)
    restored.load_state_dict(snapshot)
    for step in range(150):
        hit, done = no.clone(), no.clone()
        hit[step % 8] = step % 17 == 0
        done[(step + 2) % 8] = step % 31 == 0
        left = original.advance(hit, done, 60)
        right = restored.advance(hit, done, 60)
        assert torch.equal(left, right)
        for key in ("known", "blind", "remaining"):
            assert torch.equal(getattr(original, key), getattr(restored, key))
        assert torch.equal(original.generator.get_state(), restored.generator.get_state())
    for key in ("known", "blind", "remaining", "rng"):
        assert torch.equal(snapshot[key], frozen[key])


def test_joint_gate_removes_all_current_channels_without_mutating_sources():
    rgb = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(3, 3, 4, 4) + 1
    current = torch.arange(3 * 512, dtype=torch.float32).reshape(3, 512) + 1
    image_copy, code_copy = rgb.clone(), current.clone()
    mask = torch.tensor([False, True, False])
    gated_rgb, gated_current = gate_observation(rgb, current, mask)
    assert torch.equal(gated_rgb[[0, 2]], rgb[[0, 2]])
    assert torch.equal(gated_current[[0, 2]], current[[0, 2]])
    assert torch.count_nonzero(gated_rgb[1]) == 0
    assert torch.count_nonzero(gated_current[1]) == 0
    assert torch.equal(rgb, image_copy) and torch.equal(current, code_copy)
    visible_rgb, visible_current = gate_observation(rgb, current, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(visible_rgb, rgb) and torch.equal(visible_current, current)


def test_gate_and_clock_do_not_change_native_physics_rendering_or_noise_stream():
    cfg = _config()
    cfg.update(episode_len=31, img_noise=.02)
    left = build_env(cfg, 2, 516, "cpu", stagger=False)
    right = build_env(cfg, 2, 516, "cpu", stagger=False)
    left_obs, right_obs = left.reset_all(), right.reset_all()
    # Include native reward/teleport, not just reward-free steps.
    left.pos[:] = left.goal
    right.pos[:] = right.goal
    clock = MemoryClock(2, "cpu", seed=517)
    for index in range(66):
        action = torch.tensor([index % 6, (index + 3) % 6])
        left_obs, left_reward, left_done, left_info = left.step(action)
        right_obs, right_reward, right_done, right_info = right.step(action)
        clock.advance(left_info["hit"], left_done, 60)
        before = {key: value.clone() for key, value in left_obs.items()}
        gate_observation(left_obs["rgb"], torch.ones(2, 512), clock.mask())
        for key in left_obs:
            assert torch.equal(left_obs[key], right_obs[key]), key
            assert torch.equal(left_obs[key], before[key]), key
        assert torch.equal(left_reward, right_reward)
        assert torch.equal(left_done, right_done)
        assert torch.equal(left_info["hit"], right_info["hit"])
        assert torch.equal(left.gen.get_state(), right.gen.get_state())


def test_population_pair_has_identical_ordinary_maps_and_policy_initialization():
    grid_cfg, pgc_cfg = _config("gc100"), _config("pgc33")
    grid, pgc = build_population(grid_cfg, "cpu"), build_population(pgc_cfg, "cpu")
    assert pgc.n_pgc == 169 and grid.n_pgc == 0
    assert torch.equal(grid.K, pgc.K) and torch.equal(grid.phi, pgc.phi)
    pos = torch.tensor([[.2, -.3], [-.7, .4]])
    direction = torch.tensor([[1., 0.], [0., -1.]])
    grid_code = grid.activity(pos, dir=direction)
    pgc_code = pgc.activity(pos, dir=direction)
    assert torch.equal(grid_code[:, pgc.gc_idx], pgc_code[:, pgc.gc_idx])
    assert torch.equal(grid_code, pgc.activity(pos, dir=direction, shift_scale=0.))
    with torch.random.fork_rng():
        torch.manual_seed(grid_cfg["seed"])
        grid_policy = build_policy(grid_cfg, "cpu")
        torch.manual_seed(pgc_cfg["seed"])
        pgc_policy = build_policy(pgc_cfg, "cpu")
    assert grid_policy.extra_in == pgc_policy.extra_in == 0
    for key, value in grid_policy.state_dict().items():
        assert torch.equal(value, pgc_policy.state_dict()[key]), key


def _training_config(gap=3):
    from code import train_memory as training
    return training.parse_args([
        "--out", "/unused/independent-memory-test", "--device", "cpu",
        "--cond", "pgc33", "--seed", "23", "--n_envs", "2",
        "--rollout", "17", "--episode_len", "23", "--frames", "800",
        "--conv_arch", "dqn", "--nh", "16", "--task_max_gap", str(gap),
        "--task_ramp_frames", "8", "--resume_every", "1",
    ])


def _assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_actual_rollout_replay_and_bootstrap_share_gate_and_preserve_goal():
    from code import train_memory as training
    from tests.test_rl_goal_memory import OneHitEnv

    class WhiteOneHitEnv(OneHitEnv):
        def observe(self):
            obs = super().observe()
            obs["rgb"].fill_(1.)
            return obs

    cfg = _training_config()
    policy, _, source, runtime = training.fresh_training(cfg)
    runtime["env"] = WhiteOneHitEnv(2)
    runtime["obs"] = runtime["env"].reset_all()
    roll, state0, bootstrap, boot_inputs, _ = training.collect_rollout(
        policy, source, runtime, cfg, 16)
    assert not any(mask.any() for mask in roll["blind"])
    assert all(torch.equal(rgb, torch.ones_like(rgb)) for rgb in roll["rgb"])
    assert all(code.any() for code in roll["code"])
    assert boot_inputs["blind"].all()
    assert not boot_inputs["rgb"].any() and not boot_inputs["code"].any()
    assert not roll["goal"][0].any()
    assert torch.equal(roll["pa"][1], roll["a"][0])
    assert torch.equal(roll["pr"][1], torch.ones(2))
    expected_goal = source.goal(torch.tensor([[.4, .1]]).repeat(2, 1),
                                torch.tensor([[1., 0.]]).repeat(2, 1),
                                torch.tensor([[.35, 0.]]).repeat(2, 1))
    for goal in roll["goal"][1:] + [boot_inputs["goal"], runtime["goal"]]:
        assert torch.equal(goal, expected_goal)
    before_clock = runtime["clock"].state_dict()
    logits, values, _ = training.replay(policy, roll, state0)
    torch.testing.assert_close(logits, torch.stack(roll["actor_logits"]), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(values, torch.stack(roll["actor_values"]), rtol=1e-4, atol=1e-5)
    _assert_tree_equal(before_clock, runtime["clock"].state_dict())
    with torch.no_grad():
        _, expected_boot, _ = policy.step(boot_inputs["rgb"], boot_inputs["code"],
            boot_inputs["goal"], boot_inputs["pa"], boot_inputs["pr"], runtime["state"])
    assert torch.equal(bootstrap, expected_boot)
    # Collection's bootstrap must not consume the first blind observation.
    next_roll, _, _, _, _ = training.collect_rollout(policy, source, runtime, cfg, 1)
    assert next_roll["blind"][0].all()
    assert not next_roll["rgb"][0].any() and not next_roll["code"][0].any()
    assert torch.equal(next_roll["goal"][0], expected_goal)


def test_terminal_pickup_clears_goal_action_reward_and_both_recurrent_states():
    from code import train_memory as training
    from tests.test_rl_goal_memory import OneHitEnv

    class TerminalPickupEnv(OneHitEnv):
        def step(self, action):
            obs, reward, _, info = super().step(action)
            return obs, reward, info["hit"].clone(), info

    cfg = _training_config()
    policy, _, source, runtime = training.fresh_training(cfg)
    runtime["env"] = TerminalPickupEnv(2)
    runtime["obs"] = runtime["env"].reset_all()
    roll, _, _, _, _ = training.collect_rollout(policy, source, runtime, cfg, 2)
    assert roll["hit"][0].all() and roll["d"][0].all()
    assert not roll["goal"][1].any() and not roll["pr"][1].any()
    assert (roll["pa"][1] == -1).all()
    assert torch.equal(roll["actor_logits"][0], roll["actor_logits"][1])
    assert torch.equal(roll["actor_values"][0], roll["actor_values"][1])
    assert not runtime["clock"].known.any()


def test_full_training_restart_reproduces_optimizer_actions_and_native_resets():
    from code import train_memory as training
    cfg = _training_config()
    policy, optimizer, source, runtime = training.fresh_training(cfg)
    runtime["env"].t.zero_()
    runtime["env"].partial.zero_()
    runtime["env"].pos[:] = runtime["env"].goal
    runtime["obs"] = runtime["env"].observe()
    training.train_update(policy, optimizer, source, runtime, cfg)
    assert runtime["clock"].mask().any()
    assert runtime["goal"].any()
    proof = {"independent_unit_test": True}
    saved = training.snapshot(policy, optimizer, runtime, cfg, proof)
    saved_copy = copy.deepcopy(saved)
    expected = [training.train_update(policy, optimizer, source, runtime, cfg,
                                      return_trace=True) for _ in range(3)]
    expected_final = training.snapshot(policy, optimizer, runtime, cfg, proof)
    restored_policy, restored_optimizer, restored_source, restored_runtime = training.fresh_training(cfg)
    training.restore(saved, restored_policy, restored_optimizer, restored_runtime, cfg, proof)
    actual = [training.train_update(restored_policy, restored_optimizer, restored_source,
                                    restored_runtime, cfg, return_trace=True) for _ in range(3)]
    _assert_tree_equal(expected, actual)
    _assert_tree_equal(expected_final, training.snapshot(restored_policy, restored_optimizer,
                                                       restored_runtime, cfg, proof))
    _assert_tree_equal(saved, saved_copy)
    assert restored_runtime["env"].episodes_done >= 4


def test_zero_gap_rollout_is_bitwise_native_without_mask_or_clock():
    """Independent direct native loop proves the factorial control's identity."""
    from code import train_memory as training
    cfg = _training_config(gap=0)
    policy, _, source, runtime = training.fresh_training(cfg)
    runtime["env"].t.zero_()
    runtime["env"].partial.zero_()
    runtime["env"].pos[:] = runtime["env"].goal
    runtime["obs"] = runtime["env"].observe()
    initial = training.capture_runtime(runtime)
    roll, _, bootstrap, _, _ = training.collect_rollout(policy, source, runtime, cfg, 30)
    expected_env_rng = runtime["env"].gen.get_state().clone()
    expected_action_rng = torch.get_rng_state().clone()
    training.restore_runtime(runtime, initial)
    with torch.no_grad():
        for index in range(30):
            obs = runtime["obs"]
            code = source(obs["pos"], obs["dir"], obs["vel"])
            logits, value, state = policy.step(obs["rgb"], code, runtime["goal"],
                                               runtime["pa"], runtime["pr"], runtime["state"])
            action = torch.distributions.Categorical(logits=logits).sample()
            for key, value_ in {"rgb": obs["rgb"], "code": code, "goal": runtime["goal"],
                                "pa": runtime["pa"], "pr": runtime["pr"], "a": action,
                                "actor_logits": logits, "actor_values": value}.items():
                assert torch.equal(value_, roll[key][index]), (index, key)
            obs, raw, done, info = runtime["env"].step(action)
            reward = raw.clamp(-cfg["reward_clip"], cfg["reward_clip"])
            goal = runtime["goal"]
            if bool(info["hit"].any()):
                snapshot = source.goal(info["hit_pos"], info["hit_dir"], info["hit_vel"])
                goal = torch.where(info["hit"][:, None], snapshot, goal)
            keep = (~done).float()
            runtime.update(obs=obs, state=tuple(s * keep[:, None] for s in state),
                           goal=goal * keep[:, None], pa=torch.where(done, torch.full_like(action, -1), action),
                           pr=reward * keep)
            assert torch.equal(reward, roll["r"][index])
            assert torch.equal(done.float(), roll["d"][index])
        obs = runtime["obs"]
        code = source(obs["pos"], obs["dir"], obs["vel"])
        _, native_bootstrap, _ = policy.step(obs["rgb"], code, runtime["goal"], runtime["pa"],
                                            runtime["pr"], runtime["state"])
    assert torch.equal(bootstrap, native_bootstrap)
    assert torch.equal(expected_env_rng, runtime["env"].gen.get_state())
    assert torch.equal(expected_action_rng, torch.get_rng_state())
    assert not any(mask.any() for mask in roll["blind"])


def test_behavioral_control_counts_and_grid_substitution_are_exact():
    from code import train_memory as training
    from scripts import evaluate_simple_memory as evaluation

    class InitialPickupArena(evaluation.MatchedArena):
        def reset_all(self):
            super().reset_all()
            self.pos[:] = self.goal
            return self.observe()

    cfg = _training_config()
    cfg.update(cond="gc100", pgc_frac=0., episode_len=40)
    policy, _, _, _ = training.fresh_training(cfg)
    before = training.policy_hash(policy)
    result = evaluation.evaluate(policy, cfg, device="cpu", seed=518,
        episodes=2, n_envs=2, max_gap=3, env_class=InitialPickupArena)
    assert result["complete"] and result["all_first_discoveries_identical"]
    assert result["grid_swap_is_exact_sham"]
    assert training.policy_hash(policy) == before
    variants = result["variants"]
    assert variants["blind"]["rows"] == variants["blind_current_grid"]["rows"]
    assert variants["seen"]["rows"] == variants["seen_current_grid"]["rows"]
    for name, summary in variants.items():
        for row in summary["rows"]:
            assert row["first_goal_t"] == 0
            assert row["scheduled_steps"] > 0
            assert row["masked_steps"] == (row["scheduled_steps"] if name.startswith("blind") else 0)
            if name.endswith("_reset_onset"):
                assert row["resets"] == row["onsets"] > 0
            elif name.endswith("_reset_each"):
                assert row["resets"] == row["scheduled_steps"] > 0
            else:
                assert row["resets"] == 0
