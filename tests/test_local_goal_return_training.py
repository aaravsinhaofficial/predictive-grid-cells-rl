import copy
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import local_goal_return_training as task
from code.nets import PolicyNet


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__(); self.nh = 8; self.n_actions = 6
        self.encoder = nn.Linear(3, 256)
        self.lstm = nn.LSTMCell(1287, self.nh)
        self.pi = nn.Linear(self.nh, 6); self.v = nn.Linear(self.nh, 1)

    def embed(self, rgb): return self.encoder(rgb.mean((2, 3)))
    zero_state = PolicyNet.zero_state
    _concat = PolicyNet._concat
    extra_in = 0


def config(condition='pgc33'):
    cfg = json.loads((ROOT / f'models/{condition}/{condition}_lstm_s1_1000M.json').read_text())
    return dict(cfg, nh=8, entropy=.01, n_envs=2)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    return a == b


def test_private_heads_only_change_four_tensors_and_match_populations():
    first, second = TinyPolicy(), TinyPolicy()
    before = [{k: v.clone() for k, v in net.state_dict().items()} for net in (first, second)]
    rng = torch.random.get_rng_state().clone()
    audits = [task.reset_native_heads(net, 8271) for net in (first, second)]
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert audits[0]['new_head_hashes'] == audits[1]['new_head_hashes']
    for net, old in zip((first, second), before):
        for key, value in net.state_dict().items():
            if not key.startswith(('pi.', 'v.')): assert torch.equal(value, old[key])
        assert torch.equal(net.pi.bias, torch.zeros(6))
        assert torch.equal(net.v.bias, torch.zeros(1))


def test_private_stream_resume_reproduces_next_plan_and_all_tapes():
    cfg = config(); streams = task.TrialStreams(811)
    streams.draw(2, task.STAGES[0], cfg)
    saved = streams.snapshot(); global_rng = torch.random.get_rng_state().clone()
    expected = streams.draw(2, task.STAGES[3], cfg, headings=2)
    restored = task.TrialStreams(811); restored.restore(saved)
    actual = restored.draw(2, task.STAGES[3], cfg, headings=2)
    assert tree_equal(expected, actual)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.equal(actual['motor'][:, :, 0], actual['motor'][:, :, 1])
    assert torch.equal(actual['action'][:, 0], actual['action'][:, 1])


def test_terminal_targets_never_bootstrap_or_bridge_padding():
    roll = dict(bootstrap=torch.tensor([1000., 1000.]),
        reward=[torch.tensor([0., 0.]), torch.tensor([1., 0.]), torch.tensor([0., 0.])],
        terminal=[torch.tensor([False, False]), torch.tensor([True, False]), torch.tensor([False, True])],
        active=[torch.tensor([True, True]), torch.tensor([True, True]), torch.tensor([False, True])])
    answer = task.returns(roll, .9)
    assert torch.equal(answer[:, 0], torch.tensor([.9, 1., 0.]))
    assert torch.equal(answer[:, 1], torch.zeros(3))


def test_padding_has_zero_actor_entropy_and_value_gradient():
    logits = torch.randn(3, 2, 6, requires_grad=True); values = torch.randn(3, 2, requires_grad=True)
    roll = dict(policy_mask=[torch.tensor([False, True]), torch.tensor([True, True]), torch.tensor([False, False])],
        active=[torch.tensor([True, True]), torch.tensor([True, True]), torch.tensor([False, False])],
        action=[torch.zeros(2, dtype=torch.long) for _ in range(3)])
    loss, _ = task.losses(logits, values, torch.zeros(3, 2), roll, config()); loss.backward()
    assert torch.equal(logits.grad[2], torch.zeros_like(logits.grad[2]))
    assert torch.equal(values.grad[2], torch.zeros_like(values.grad[2]))
    assert torch.equal(logits.grad[0, 0], torch.zeros(6))


def test_native_setup_excluded_and_noisy_rollout_replays_exactly():
    torch.set_num_threads(2); cfg = config(); net = TinyPolicy()
    source = task.make_source(cfg, 'cpu'); plan = task.TrialStreams(917).draw(2, task.STAGES[0], cfg)
    roll, rows, audit, _ = task.collect(net, cfg, source, plan, sigma=.02)
    logits, values = task.replay(net, roll)
    assert torch.allclose(logits, torch.stack(roll['logits']), atol=1e-7, rtol=0.)
    assert audit['setup_pickups'] == 2 and audit['setup_raw_reward'] == 20
    assert audit['setup_learning_reward'] == 0 and audit['forced_hits'] == 0
    assert all(torch.equal(p, torch.zeros_like(p)) for p in roll['pr'])
    assert sum(int(x.sum()) for x in roll['terminal']) == 2
    assert all(not bool((r[~free] != 0).any()) for r, free in zip(roll['reward'], roll['policy_mask']))
    assert audit['active_physics_updates'] + audit['padded_slots'] == 320
    assert audit['forced_updates'] + audit['free_updates'] == audit['active_physics_updates']
    assert len(rows) == 2
    loss, _ = task.losses(logits, values, task.returns(roll, cfg['gamma']), roll, cfg)
    loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_antipodal_free_inputs_and_erased_states_are_identical():
    torch.set_num_threads(2); cfg = config(); net = TinyPolicy(); source = task.make_source(cfg, 'cpu')
    plan = task.TrialStreams(920).draw(1, task.STAGES[3], cfg, headings=2)
    _, _, audit, first = task.collect(net, cfg, source, plan,
        variants=('blind_reset_after_glimpse',), sigma=0., retain_inputs=False)
    assert audit['forced_hits'] == 0 and audit['clearance_min'] > .01
    for field in ('input', 'state', 'position', 'motors'):
        assert torch.equal(first[0][field], first[1][field]), field
    delta = first[0]['heading'] - first[1]['heading']
    assert abs(float(torch.atan2(delta.sin(), delta.cos()).abs()) - torch.pi) < 1e-5


def test_cue_gradient_crosses_whole_trial_and_reset_cuts_that_path():
    torch.set_num_threads(2); cfg = config(); net = TinyPolicy(); source = task.make_source(cfg, 'cpu')
    plan = task.TrialStreams(921).draw(1, task.STAGES[0], cfg)
    for variant in ('blind_native', 'blind_reset_after_glimpse'):
        roll, _, _, _ = task.collect(net, cfg, source, plan, variants=(variant,), sigma=0.)
        cue = int(task.schedule(plan, 'cpu')['glimpse'][0]); free = int(task.schedule(plan, 'cpu')['free'][0])
        roll['code'][cue] = roll['code'][cue].clone().requires_grad_(True)
        logits, _ = task.replay(net, roll)
        gradient = torch.autograd.grad(logits[free, 0, 2], roll['code'][cue])[0]
        if variant == 'blind_native': assert float(gradient.abs().sum()) > 0
        else: assert torch.equal(gradient, torch.zeros_like(gradient))
