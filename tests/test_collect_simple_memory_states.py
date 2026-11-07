"""Private CPU fixtures only: no trained policy or scientific rollout."""
import json
import time

import numpy as np
import pytest
import torch

from code import train_memory as training
from code.common import CodeSource, build_population
from code.memory_task import MemoryClock, gate_observation
from scripts import collect_simple_memory_states as study
from scripts import evaluate_simple_memory as evaluation


class TracePolicy(torch.nn.Module):
    recurrent, nh = True, 8

    def __init__(self):
        super().__init__()
        self.trace = []

    def zero_state(self, n, device):
        return torch.zeros(n, self.nh, device=device), torch.zeros(n, self.nh, device=device)

    def step(self, rgb, current, goal, pa, pr, state):
        h = torch.stack((state[0][:, 0]+1, pa.float()+1, pr, goal.mean(1),
                         rgb.mean((1, 2, 3)), current.mean(1), state[1][:, 0], goal[:, 0]), 1)
        c = state[1]*.5+h*.25
        logits = torch.stack((h[:, 3], h[:, 4], h[:, 5], -h[:, 3], h[:, 7], -h[:, 7]), 1)
        self.trace.append({key: value.detach().clone() for key, value in
            dict(h_before=state[0], c_before=state[1], h=h, c=c, goal=goal, current=current,
                 rgb_mean=rgb.mean((1, 2, 3)), pa=pa, pr=pr, logits=logits).items()})
        return logits, h[:, 0], (h, c)


class AuditedPickupArena(study.MatchedArena):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []
        self.instances.append(self)

    def reset_all(self):
        super().reset_all()
        self.goal[:] = torch.tensor([.8, 0.], device=self.device)
        self.pos[:] = self.goal  # private fixture's actual native first pickup
        return self.observe()

    def _respawn(self, mask):
        super()._respawn(mask)
        # Fixed safe private test respawn; production uses unmodified MatchedArena.
        selected = mask & (self.n_goals > 0)
        self.pos[selected] = 0.
        self.spawn[selected] = 0.
        self.hd[selected] = 0.
        self.mdir[selected] = torch.tensor([1., 0.], device=self.device)

    def step(self, action):
        pre = {key: getattr(self, key).clone() for key in ('pos', 'hd', 'mdir', 'goal', 't', 'ep_idx')}
        obs, raw, done, info = super().step(action)
        pre.update(action=action.clone(), hit=info['hit'].clone(), done=done.clone(),
                   reward=raw.clone(), hit_pos=info['hit_pos'].clone())
        self.trace.append(pre)
        return obs, raw, done, info


def config(tmp_path, condition='pgc33'):
    return training.parse_args(['--out', str(tmp_path/'unused'), '--device', 'cpu', '--threads', '1',
        '--cond', condition, '--episode_len', '40', '--n_envs', '4', '--frames', '16'])


def run_fixture(tmp_path, stride=3, condition='pgc33'):
    torch.set_num_threads(1)
    cfg = config(tmp_path, condition)
    net = TracePolicy()
    arrays, metadata, contexts = study.collect(net, cfg, episodes=8, n_envs=4,
        seed=93101, max_gap=3, stride=stride, env_class=AuditedPickupArena)
    return cfg, net, AuditedPickupArena.instances[-1], arrays, metadata, contexts


def test_every_state_and_physical_label_share_preaction_timing(tmp_path):
    _, net, env, arrays, metadata, contexts = run_fixture(tmp_path)
    assert metadata['samples'] == 8*14 and metadata['episodes'] == 8
    assert metadata['onset_contexts']['count'] == len(contexts) == 4
    assert net.training  # original mode restored
    for row in range(len(arrays['h'])):
        step, slot = int(arrays['global_step'][row]), int(arrays['slot'][row])
        physical, neural = env.trace[step], net.trace[step]
        for name in ('h', 'c'):
            assert np.array_equal(arrays[name][row], neural[name][slot].numpy())
        for name, original in [('pos', 'pos'), ('body_heading', 'hd'), ('travel_dir', 'mdir'),
                               ('goal_pos', 'goal'), ('episode_step', 't'), ('action', 'action'),
                               ('reward', 'reward'), ('hit_pos', 'hit_pos'), ('done', 'done')]:
            assert np.array_equal(arrays[name][row], physical[original][slot].numpy())
        assert arrays['episode_id'][row] == (step//40)*4+slot
        assert arrays['episode_step'][row] % 3 == 0
        if arrays['blind'][row]:
            assert not neural['current'][slot].any() and neural['rgb_mean'][slot] == 0
            assert arrays['known'][row] and arrays['phase_age'][row] >= 1
        if arrays['episode_step'][row] == 0:
            assert not neural['h_before'][slot].any() and not neural['c_before'][slot].any()
            assert neural['pa'][slot] == -1 and neural['pr'][slot] == 0
            assert not neural['goal'][slot].any() and not arrays['known'][row]
    assert arrays['h'].dtype == arrays['c'].dtype == np.float32
    assert np.unique(arrays['episode_id']).size == 8
    assert len(arrays['layout_episode_id']) == 8
    # Goal inputs are unchanged through blackout; exact pickup reconstruction.
    cfg = config(tmp_path)
    source = CodeSource(cfg, build_population(cfg, 'cpu'), 'cpu')
    for row in np.where(arrays['known'])[0]:
        goal = source.goal(*(torch.from_numpy(arrays['goal_pickup_'+key][row:row+1])
                             for key in ('pos', 'dir', 'vel')))[0]
        assert torch.equal(goal, net.trace[int(arrays['global_step'][row])]['goal'][int(arrays['slot'][row])])


def test_first_onset_contexts_restore_exact_preinput_state_action_and_physics(tmp_path):
    cfg, net, env, arrays, _, contexts = run_fixture(tmp_path)
    assert [(c['global_step'], c['selected_slot']) for c in contexts] == [(16, k) for k in range(4)]
    assert 16 not in arrays['global_step']  # no extra PCA rows for contexts
    for context in contexts:
        prefix = context['visible_prefix']
        assert prefix['rgb'].shape == (15, 3, 84, 84)
        assert prefix['global_step'].tolist() == list(range(1, 16))
        assert prefix['known'].all() and not prefix['blind'].any() and not prefix['done'].any()
        replay = study.verify_visible_prefix(TracePolicy(), context)
        assert replay['bitwise_equal'] and replay['h_max_absolute_error'] == replay['c_max_absolute_error'] == 0
        broken = {**context, 'visible_prefix': {k: v.clone() for k,v in prefix.items()}}
        broken['visible_prefix']['h_after'][-1, 0] += 1
        with pytest.raises(AssertionError): study.verify_visible_prefix(TracePolicy(), broken)
        validation = study.verify_visible_prefix(TracePolicy(), broken, strict=False)
        assert not validation['qualified'] and validation['h_max_absolute_error'] == 1.
        assert validation['c_max_absolute_error'] == 0. and validation['finite']
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        for context in contexts:
            restored_env = study.MatchedArena(cfg, 4, 1, 999, 'cpu')
            rt = {'env': restored_env, 'clock': MemoryClock(4, 'cpu', 999)}
            training.restore_runtime(rt, context['runtime'])
            assert rt['clock'].mask().all() and context['phase_ages'].tolist() == [1]*4
            source = CodeSource(cfg, build_population(cfg, 'cpu'), 'cpu')
            rgb, code = gate_observation(rt['obs']['rgb'], source(rt['obs']['pos'],rt['obs']['dir'],rt['obs']['vel']),rt['clock'].mask())
            probe = TracePolicy()
            logits, _, state = probe.step(rgb, code, rt['goal'], rt['pa'], rt['pr'], rt['state'])
            assert torch.equal(state[0], net.trace[16]['h']) and torch.equal(state[1], net.trace[16]['c'])
            sampler = torch.Generator().set_state(context['action_sampler_rng'].clone())
            uniform = torch.rand(4, generator=sampler)
            action = (uniform[:, None] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(5)
            assert torch.equal(action, env.trace[16]['action'])
            _, reward, done, info = restored_env.step(action)
            assert torch.equal(info['hit_pos'], env.trace[16]['hit_pos'])
            assert torch.equal(reward, env.trace[16]['reward']) and torch.equal(done,env.trace[16]['done'])
    assert torch.equal(before, torch.get_rng_state())


def test_matches_frozen_evaluation_blind_branch_and_grid_sham(tmp_path):
    cfg, _, _, _, metadata, _ = run_fixture(tmp_path, condition='gc100')
    result = evaluation.evaluate(TracePolicy(), cfg, episodes=8, n_envs=4, seed=93101,
        max_gap=3, env_class=AuditedPickupArena)
    assert result['grid_swap_is_exact_sham']
    expected = result['variants']['blind']['rows']
    actual = metadata['summary']['rows']
    for left, right in zip(actual, expected):
        for key in ('slot','ret','n_goals','first_goal_t','path_eff','path_eff_n',
                    'known_steps','blind_goals','seen_known_goals'):
            assert left[key] == right[key], (key,left,right)
        assert left['blind_steps'] == right['masked_steps'] == right['scheduled_steps']


def test_uniform_stride_and_lossless_storage(tmp_path):
    _, _, _, dense, _, _ = run_fixture(tmp_path, stride=1)
    _, _, _, sparse, metadata, contexts = run_fixture(tmp_path, stride=3)
    selected = dense['episode_step'] % 3 == 0
    for key in sparse:
        expected = dense[key] if key.startswith(('layout_', 'map_')) else dense[key][selected]
        assert np.array_equal(sparse[key], expected), key
    out = tmp_path/'saved'
    report = study.save_arrays(out, sparse, metadata, contexts)
    with np.load(out/'states.npz') as archive:
        assert archive.files == list(sparse)
        assert all(np.array_equal(archive[key], value) for key,value in sparse.items())
    saved = torch.load(out/'onset_contexts.pt', weights_only=False)
    assert len(saved['contexts']) == 4
    assert report['array_file']['sha256'] == study.sha(out/'states.npz')
    assert json.loads((out/'summary.json').read_text())['complete']
    assert report['uncompressed_array_bytes'] < 200_000_000
    with pytest.raises(FileExistsError): study.save_arrays(out, sparse, metadata, contexts)


def test_no_artificial_contexts_and_deadline_fails_closed(tmp_path):
    cfg = dict(config(tmp_path), episode_len=2)
    arrays, metadata, contexts = study.collect(TracePolicy(), cfg, episodes=4, n_envs=4,
        seed=93101, max_gap=60, stride=1, env_class=AuditedPickupArena)
    assert contexts == [] and metadata['onset_contexts']['count'] == 0
    assert not arrays['blind'].any()
    with pytest.raises(TimeoutError):
        study.collect(TracePolicy(), cfg, episodes=4,n_envs=4,deadline=time.monotonic()-1)


def test_native_untrained_cnn_lstm_visible_prefix_replay(tmp_path):
    # Actual architecture, random private test weights; never an original/final agent.
    cfg = config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(93109)
        net = study.build_policy(cfg, 'cpu')
        _, metadata, contexts = study.collect(net, cfg, episodes=4, n_envs=4,
            seed=93101, max_gap=3, stride=3, env_class=AuditedPickupArena)
        assert metadata['onset_contexts']['count'] == 4
        for context in contexts:
            replay = study.verify_visible_prefix(net, context)
            assert replay['h_max_absolute_error'] < 1e-5 and replay['c_max_absolute_error'] < 1e-5


def test_physical_selection_never_reads_neural_quality_and_logs_rejected_first_onsets(tmp_path, monkeypatch):
    cfg, _, _, _, _, contexts = run_fixture(tmp_path)
    context = contexts[0]; prefix = {key:value.clone() for key,value in context['visible_prefix'].items()}
    prefix['h_before'].fill_(float('nan')); prefix['c_after'].fill_(float('nan'))
    rt, slot = context['runtime'], context['selected_slot']
    args = (prefix, rt['obs']['pos'][slot], rt['environment_tensors']['goal'][slot],rt['goal'][slot])
    assert study.physical_eligibility(*args,half=cfg['arena_half'],wall_margin=.08,goal_radius=.3)['eligible']
    bad = study.physical_eligibility(prefix,torch.tensor([1.17,0.]),args[2],args[3],half=1.25,wall_margin=.08,goal_radius=.3)
    assert not bad['eligible'] and 'all_angle_wall_clearance' in bad['failed_rules']
    monkeypatch.setattr(study,'physical_eligibility',lambda *a,**kw:dict(eligible=False,failed_rules=['private_forced_physical_rejection']))
    _, metadata, contexts = study.collect(TracePolicy(),dict(cfg,episode_len=80),episodes=4,n_envs=4,
        seed=93101,max_gap=3,env_class=AuditedPickupArena)
    ledger = metadata['onset_contexts']['candidate_ledger']
    assert len(ledger) == 4 and contexts == []
    assert {row['global_step'] for row in ledger} == {16}  # Later onsets are never substitutes.
    assert all(row['status']=='skipped' for row in ledger)


def test_native_fork_matches_archived_continuation_without_second_reference_rollout(tmp_path):
    from scripts.simple_memory_interventions import native_fork
    cfg, _, _, _, _, contexts = run_fixture(tmp_path)
    context = contexts[0]
    result = native_fork(TracePolicy(),cfg,context,device='cpu',tape=0,horizon=150,max_gap=3)
    actual, expected = result['trace'], context['native_continuation']
    assert len(actual) == len(expected) == 24 and expected[-1]['done']
    for left,right in zip(actual,expected):
        for key in ('input_hashes','post_physics_hashes','action','reward','hit','done'):
            assert left[key] == right[key], key
        for key in ('h_before','c_before','h_after','c_after','probabilities','hit_pos'):
            np.testing.assert_array_equal(left[key],right[key].numpy())


def test_cli_freeze_guard_before_any_final_policy_load(tmp_path, monkeypatch):
    path = tmp_path/'analysis.json'
    path.write_text(json.dumps(dict(status='draft',source_sha256={})))
    monkeypatch.setattr(study,'load_final',lambda *args: pytest.fail('Policy load must not occur'))
    with pytest.raises(AssertionError):
        study.main(['--run',str(tmp_path/'run'),'--device','cpu','--analysis-protocol',str(path)])
    path.write_text(json.dumps(dict(status='frozen',source_sha256={})))
    with pytest.raises(AssertionError): study.authenticate_analysis_protocol(path)
    path.write_text(json.dumps(dict(status='frozen',source_sha256={'scripts/collect_simple_memory_states.py':study.sha(study.ROOT/'scripts/collect_simple_memory_states.py')})))
    assert study.authenticate_analysis_protocol(path)['analysis_protocol_sha256'] == study.sha(path)
