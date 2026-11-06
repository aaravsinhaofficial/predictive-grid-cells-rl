"""Independent semantic checks for the proposed standalone return trial."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from torch import nn
import json
import pytest
from code.env import ArenaEnv
import local_goal_return_training as task


class TinyHeads(nn.Module):
    def __init__(self, fill):
        super().__init__()
        self.nh, self.n_actions = 12, 6
        self.retained = nn.Parameter(torch.full((5, 7), fill))
        self.pi = nn.Linear(12, 6)
        self.v = nn.Linear(12, 1)


def test_head_reset_preserves_global_rng_and_original_features():
    first, second = TinyHeads(2.), TinyHeads(3.)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    mps_before = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    task.reset_native_heads(first)
    task.reset_native_heads(second)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    if cuda_before:
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_before))
    if mps_before is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_before)
    assert bool((first.retained == 2).all() and (second.retained == 3).all())
    for name in ('pi.weight', 'pi.bias', 'v.weight', 'v.bias'):
        assert torch.equal(first.state_dict()[name], second.state_dict()[name])
    assert torch.allclose(first.pi.weight @ first.pi.weight.T, torch.eye(6)*.0001, atol=1e-10)
    assert torch.allclose(first.v.weight.norm(), torch.tensor(1.))
    assert bool((first.pi.bias == 0).all() and (first.v.bias == 0).all())
    assert all(p.requires_grad for p in first.parameters())


def test_true_terminal_returns_ignore_poisoned_bootstrap_and_padding():
    roll = dict(bootstrap=torch.tensor([999., 999.]),
        reward=[torch.tensor([0., 0.]), torch.tensor([10., 0.]), torch.tensor([123., 0.])],
        terminal=[torch.tensor([False, False]), torch.tensor([True, False]), torch.tensor([False, True])],
        active=[torch.tensor([True, True]), torch.tensor([True, True]), torch.tensor([False, True])])
    expected = torch.tensor([[9., 0.], [10., 0.], [0., 0.]])
    assert torch.equal(task.returns(roll, .9), expected)


def test_forced_actor_and_all_padded_gradients_are_zero():
    logits = torch.zeros(3, 2, 6, requires_grad=True)
    values = torch.ones(3, 2, requires_grad=True)
    active = torch.tensor([[True, True], [True, True], [False, True]])
    free = torch.tensor([[False, False], [True, False], [False, True]])
    roll = dict(active=list(active), policy_mask=list(free),
                action=[torch.zeros(2, dtype=torch.long) for _ in range(3)])
    loss, _ = task.losses(logits, values, torch.zeros(3, 2), roll,
                          dict(value_coef=.5, entropy=.01))
    loss.backward()
    assert bool((logits.grad[~free] == 0).all())
    assert bool((logits.grad[free].abs().sum(-1) > 0).all())
    assert bool((values.grad[~active] == 0).all())
    assert bool((values.grad[active] != 0).all())  # critic credit includes forced updates


def test_terminal_no_respawn_keeps_native_pre_hit_physics_and_reward():
    cfg = dict(arena_half=1.25, action_repeat=4, v_max=.35, rot_max=2.2,
               goal_radius=.3, vision='none')
    local = task.TrialArena(cfg, 1, 'cpu'); local.reset_all(); local.no_respawn = True
    native = ArenaEnv(1, half=1.25, episode_len=10000, action_repeat=4,
        v_max=.35, rot_max=2.2, goal_radius=.3, vision='none', img_noise=0.,
        noise_v=0., noise_r=0., stagger=False, seed=0, device='cpu')
    native.reset_all()
    for env in (local, native):
        env.pos.zero_(); env.hd.zero_(); env.mdir[:] = torch.tensor([1., 0.])
        env.goal[:] = torch.tensor([.31, 0.]); env.vf.zero_(); env.vs.zero_(); env.vr.zero_()
        env.noise_v = env.noise_r = 0.
    hit = False
    for _ in range(10):
        _, reward_a, done_a, info_a = local.step(torch.tensor([2]))
        _, reward_b, done_b, info_b = native.step(torch.tensor([2]))
        assert torch.equal(reward_a, reward_b) and not bool(done_a.any() or done_b.any())
        for name in ('hit_pos', 'hit_dir', 'hit_vel', 'hit', 'hit_wall'):
            assert torch.equal(info_a[name], info_b[name])
        if bool(info_a['hit'][0]):
            assert float(reward_a[0]) == 10.
            assert torch.equal(local.pos, info_a['hit_pos'])
            assert not torch.equal(native.pos, info_b['hit_pos'])
            hit = True
            break
        assert torch.equal(local.pos, native.pos)
    assert hit


def test_resuming_a_boundary_requires_its_evaluation_before_more_learning():
    import local_goal_return_training_run as runner
    assert runner.pending_stage(dict(updates=0, completed_stages=0)) is None
    assert runner.pending_stage(dict(updates=1024, completed_stages=0)) == 1
    assert runner.pending_stage(dict(updates=4096, completed_stages=3)) == 4
    assert runner.pending_stage(dict(updates=4096, completed_stages=4)) is None
    with pytest.raises(AssertionError, match='skipping'):
        runner.pending_stage(dict(updates=1025, completed_stages=0))
    with pytest.raises(AssertionError, match='Multiple missing'):
        runner.pending_stage(dict(updates=2048, completed_stages=0))


def test_resume_preserves_rolled_back_metrics_and_partial_evaluation(tmp_path):
    import local_goal_return_training_run as runner
    parent=tmp_path/'parent'; parent.mkdir(); out=tmp_path/'resumed'; out.mkdir()
    checkpoint=parent/'latest.pt';checkpoint.write_bytes(b'exact-checkpoint-placeholder')
    metrics=b'{"update":1024}\n{"update":1025,"rolled_back_attempt":true}\n'
    (parent/'metrics.jsonl').write_bytes(metrics)
    partial=parent/'stage1_evaluation';partial.mkdir();(partial/'context0.json').write_text('{"partial":true}')
    ancestor=parent/'resume_parent_evidence';ancestor.mkdir();(ancestor/'metrics.jsonl').write_text('{"update":1}\n')
    counters=dict(updates=1024,completed_stages=0,evaluation_records=[])
    runner.inherit_resume_artifacts(checkpoint,out,counters)
    assert (out/'resume_parent_evidence/metrics.jsonl').read_bytes()==metrics
    assert (out/'resume_parent_evidence/checkpoint.pt').read_bytes()==checkpoint.read_bytes()
    assert (out/'resume_parent_evidence/ancestor_evidence/metrics.jsonl').read_bytes()==(ancestor/'metrics.jsonl').read_bytes()
    assert (out/'parent_incomplete_evaluation/context0.json').read_bytes()==(partial/'context0.json').read_bytes()
    lineage=json.loads((out/'resume_lineage.json').read_text())
    assert lineage['restored_updates']==1024 and lineage['pending_stage']==1
    assert 'rolled-back attempt evidence' in lineage['parent_metrics_scope']
    for record in lineage['copied_artifacts']:
        assert runner.sha(out/record['copied'])==record['sha256']
