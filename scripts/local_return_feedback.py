"""Prepared acquisition control with native feedback on every active input.

This is a separate, currently unfrozen module. It does not generate study banks,
load policies, optimize, or launch anything on import. Physics, complete-trial
returns and recurrent replay are reused from the frozen local-return task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch

from code.common import build_policy
from rl_goal_memory import state_digest
import local_goal_return_training as task

Stage, TrialStreams = task.Stage, task.TrialStreams
schedule, prepare_batch = task.schedule, task.prepare_batch
make_source, returns, replay, losses = task.make_source, task.returns, task.replay, task.losses
STAGE = Stage(.58, .62, 4)
ACTOR_MODES = ('retained_actor', 'reset_actor')
HEADS = ('pi.weight', 'pi.bias', 'v.weight', 'v.bias')
PARENT_PROTOCOL = 'docs/research/local_goal_return_training_protocol.json'
PARENT_PROTOCOL_SHA256 = '0768cb770943c41a5dc44773abd450f9978f9f54ed6f7fedb2e504a71a125038'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def compose_initial_state(original, archived, actor_mode, archived_head_hashes):
    """Pure tensor composition; caller validates the archived file provenance."""
    assert actor_mode in ACTOR_MODES
    assert set(original) == set(archived)
    assert set(HEADS).issubset(original)
    assert set(archived_head_hashes) == set(HEADS)
    for key, value in original.items():
        other = archived[key]
        assert value.shape == other.shape and value.dtype == other.dtype, key
        if key in HEADS:
            assert task.tensor_hash(other) == archived_head_hashes[key], key
        else:
            assert torch.equal(value, other), f'Archived nonhead tensor changed: {key}'
    result = {key: value.detach().cpu().clone() for key, value in original.items()}
    for key in HEADS:
        if key.startswith('v.') or actor_mode == 'reset_actor':
            result[key] = archived[key].detach().cpu().clone()
    return result


def configure_trainable(net):
    """All original features and both heads adapt; no frozen-feature shortcut."""
    net.train()
    for parameter in net.parameters(): parameter.requires_grad_(True)
    assert all(parameter.requires_grad for parameter in net.parameters())
    return net


def initialize(condition, actor_mode, *, device='cpu', root=ROOT):
    """Read exact parent/archived-head assets, only when explicitly called."""
    assert condition in ('gc100', 'pgc33') and actor_mode in ACTOR_MODES
    root = Path(root); protocol_path = root / PARENT_PROTOCOL
    assert sha(protocol_path) == PARENT_PROTOCOL_SHA256
    protocol = json.loads(protocol_path.read_text())
    for relative in ('scripts/local_goal_return_training.py', 'code/nets.py',
                     'code/common.py', 'code/env.py', 'code/codes.py'):
        assert sha(root / relative) == protocol['source_sha256'][relative], relative
    stem = f'models/{condition}/{condition}_lstm_s1_1000M'
    original_path, config_path = root / (stem + '.pt'), root / (stem + '.json')
    archived_relative = f'results/local_goal_return_baseline/{condition}/initialized_policy.pt'
    archived_path = root / archived_relative
    for relative, path in ((stem + '.pt', original_path), (stem + '.json', config_path)):
        assert sha(path) == protocol['original_assets'][relative], relative
    assert sha(archived_path) == protocol['passed_baseline_assets'][archived_relative]
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    archived = torch.load(archived_path, map_location='cpu', weights_only=True)
    cfg_original = json.loads(config_path.read_text())
    assert original['frames'] == 1_000_000_000 and archived['no_optimization']
    assert archived['behavior_protocol_sha256'] == protocol['evaluation']['behavior_protocol_sha256']
    assert archived['config'] == dict(cfg_original, n_envs=32, rollout=160, entropy=.01)
    provenance = archived['initialization']
    assert provenance['parent_checkpoint_sha256'] == sha(original_path)
    assert provenance['parent_config_sha256'] == sha(config_path)
    state = compose_initial_state(original['policy'], archived['policy'], actor_mode,
                                  provenance['new_head_hashes'])
    assert cfg_original['entropy'] == 8e-5
    cfg = dict(cfg_original, n_envs=32, rollout=160, entropy=8e-5)
    with torch.random.fork_rng(devices=[]): net = build_policy(cfg, 'cpu')
    net.load_state_dict(state, strict=True); configure_trainable(net)
    audit = dict(condition=condition, actor_mode=actor_mode, original_seed=1,
        parent_protocol_sha256=PARENT_PROTOCOL_SHA256,
        parent_checkpoint_path=str(original_path), parent_checkpoint_sha256=sha(original_path),
        parent_config_sha256=sha(config_path), archived_heads_path=str(archived_path),
        archived_heads_sha256=sha(archived_path), full_initial_policy_hash=state_digest(net),
        incoming_head_hashes={key: task.tensor_hash(state[key]) for key in HEADS},
        original_nonhead_hashes={key: task.tensor_hash(value) for key, value in state.items() if key not in HEADS},
        all_parameters_trainable=True, critic_from_exact_untrained_archive=True,
        actor_changes_initial_exploration=True, no_optimization=True)
    return net.to(device), cfg, audit


def optimizer(net, cfg):
    assert all(parameter.requires_grad for parameter in net.parameters())
    return torch.optim.RMSprop(net.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])


@torch.no_grad()
def collect(net, cfg, source, plan, *, device='cpu', retain_inputs=True, controller='policy'):
    """Native feedback throughout; no special cue, state noise or midtrial reset."""
    return _collect(net, cfg, source, plan, device=device, retain_inputs=retain_inputs,
                    controller=controller, native_feedback=True)


@torch.no_grad()
def _collect(net, cfg, source, plan, *, device='cpu', retain_inputs=True,
             controller='policy', native_feedback=True):
    """The disabled branch exists solely for synthetic old-collector equivalence."""
    assert controller in ('policy', 'uniform', 'planner')
    assert controller == 'policy' or not retain_inputs, 'Physical-reference controllers never enter learning'
    variant = 'feedback_native' if native_feedback else 'blind_native'
    batch = prepare_batch(cfg, source, plan, (variant,), device)
    env, obs, goal, sched = (batch[key] for key in ('env', 'obs', 'goal', 'schedule'))
    n = env.n; state = net.zero_state(n, device)
    pa = torch.full((n,), -1, dtype=torch.long, device=device); pr = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    held = torch.zeros(n, cfg['n_cells'], device=device)
    motor, uniforms = plan['motor'].to(device), plan['action'].to(device)
    keys = ('rgb', 'code', 'pa', 'pr', 'action', 'reward', 'raw_reward', 'terminal', 'active',
            'policy_mask', 'noise', 'reset_before', 'reset_after', 'logits', 'values', 'phase',
            'position', 'observation_position', 'observation_heading', 'observation_direction',
            'observation_velocity', 'motors_before_action')
    roll = {key: [] for key in keys}; roll['goal'] = goal
    hits = torch.zeros(n, dtype=torch.bool, device=device); hit_step = torch.full((n,), -1, dtype=torch.long, device=device)
    free_steps = torch.zeros(n, dtype=torch.long, device=device); path = torch.zeros(n, device=device)
    start_distance = (env.pos-env.goal).norm(dim=1); free_distance = torch.zeros_like(start_distance)
    distance_end = start_distance.clone(); snap_max = 0.; forced_hits = 0; wall_hits = 0
    entropy_sum = torch.zeros(n, device=device); maxprob_sum = entropy_sum.clone()
    value_sum = entropy_sum.clone(); max_abs_logit = entropy_sum.clone()
    action_counts = torch.zeros(n, 6, dtype=torch.long, device=device); first_free = {}
    blank_embed = net.embed(torch.zeros(n, 3, 84, 84, device=device))
    for t in range(task.MAX_STEPS):
        phase = sched['phase'][t]; active = alive & (phase != task.PAD); free = active & (phase == task.FREE)
        blind = active & (phase >= task.TURN_A); endpoint = active & (sched['glimpse'] == t)
        if bool(endpoint.any()):
            error = (env.pos - batch['endpoint']).norm(dim=1)[endpoint]
            snap_max = max(snap_max, float(error.max())); assert snap_max < 1e-5
            env.pos[endpoint] = batch['endpoint'][endpoint]
            obs = env.observe(obs['vel'])  # Retain the actual preceding displacement before braking.
        native = source(obs['pos'], obs['dir'], obs['vel'])
        if native_feedback:
            code = native
            visible = active
        else:
            ordinary = source.pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0.)
            refresh = active & ((t == task.VISIBLE_STEPS) | endpoint)
            held = torch.where(refresh[:, None], ordinary, held)
            code = torch.where(blind[:, None], held, native)
            code = torch.where(endpoint[:, None], native, code)
            visible = active & ~blind
        # The input uses pre-brake obs['vel']; the next physical action uses braked motors.
        task.brake(env, active & ((t == task.VISIBLE_STEPS) | (sched['probe'] == t) | endpoint))
        rgb = obs['rgb'] * visible[:, None, None, None]
        embedding = blank_embed.clone()
        if bool(visible.any()): embedding[visible] = net.embed(rgb[visible])
        state = task.masked_state(state, active)
        x = net._concat(embedding, code, goal, pa, pr, None)
        state = net.lstm(x, state)
        eps = torch.zeros(2, n, cfg['nh'], device=device)
        logits, values = net.pi(state[0]), net.v(state[0]).squeeze(-1)
        probabilities = logits.softmax(1)
        if controller == 'uniform': probabilities = torch.full_like(probabilities, 1/6)
        if controller == 'planner':
            probabilities = torch.nn.functional.one_hot(task.expected_next_distance(env).argmin(1), 6).to(logits.dtype)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(1)
        entering = free & (t == sched['free'])
        if bool(entering.any()):
            for i in entering.nonzero().flatten():
                first_free[int(i)] = dict(input=x[i].cpu().clone(), state=torch.cat(state, 1)[i].cpu().clone(),
                    position=env.pos[i].cpu().clone(), heading=env.hd[i].cpu().clone(),
                    motors=torch.stack((env.vf, env.vs, env.vr), 1)[i].cpu().clone(),
                    probabilities=probabilities[i].cpu().clone(), logits=logits[i].cpu().clone(),
                    value=values[i].cpu().clone(), entropy=entropy[i].cpu().clone())
            free_distance[entering] = (env.pos-env.goal).norm(dim=1)[entering]
        sampled = (uniforms[t, :, None] > probabilities.cumsum(1)).sum(1).clamp_max(5)
        action = torch.where(free, sampled, sched['forced_action'][t])
        entropy_sum += free * entropy; maxprob_sum += free * probabilities.amax(1)
        value_sum += free * values; max_abs_logit = torch.maximum(max_abs_logit, free * logits.abs().amax(1))
        action_counts.scatter_add_(1, action[:, None], free.long()[:, None])
        observed = {key: obs[field].clone() for key, field in (
            ('observation_position', 'pos'), ('observation_heading', 'hd'),
            ('observation_direction', 'dir'), ('observation_velocity', 'vel'))}
        observed['motors_before_action'] = torch.stack((env.vf, env.vs, env.vr), 1).clone()
        old_pos = env.pos.clone()
        frozen = {key: value.clone() for key, value in vars(env).items()
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == n and key not in ('tape', '_targets')}
        env.noise_v = .03 * free.float(); env.noise_r = .15 * free.float(); env.set_motor(motor[t])
        obs, raw, native_done, info = env.step(action)
        assert env.tape_index == 3 and not bool(native_done.any())
        for key, value in frozen.items(): getattr(env, key)[~active] = value[~active]
        forced_bad = active & ~free & (info['hit'] | info['hit_wall'])
        forced_hits += int((active & ~free & info['hit']).sum())
        assert not bool(forced_bad.any()), 'Invalid preparation is retained as a failure, never rewarded/retried'
        terminal = active & (info['hit'] | (t == sched['end'] - 1))
        reward = torch.where(free, raw, torch.zeros_like(raw))
        learned_reward = reward.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else reward
        hits |= free & info['hit']; hit_step[free & info['hit']] = free_steps[free & info['hit']] + 1
        free_steps += free.long(); path += free * (info['hit_pos'] - old_pos).norm(dim=1)
        distance_end = torch.where(free, (info['hit_pos']-env.goal).norm(dim=1), distance_end)
        wall_hits += int((free & info['hit_wall']).sum())
        saved = dict(rgb=rgb, code=code, pa=pa, pr=pr, action=action, reward=learned_reward,
            raw_reward=reward, terminal=terminal, active=active, policy_mask=free, noise=eps,
            reset_before=torch.zeros_like(active), reset_after=torch.zeros_like(active), logits=logits,
            values=values, phase=phase, position=env.pos, **observed)
        for key, value in saved.items():
            if retain_inputs or key not in ('rgb', 'code', 'noise', 'logits'):
                roll[key].append(value.detach().clone())
        alive &= ~terminal; state = task.masked_state(state, alive)
        pa = torch.where(alive, action, torch.full_like(action, -1)); pr = torch.zeros_like(pr)
    assert not bool(alive.any())
    roll['bootstrap'] = torch.zeros(n, device=device)
    rows = [dict(variant=variant, context=int(plan['context'][i]), heading_index=int(plan['heading_index'][i]),
        hit=bool(hits[i]), hit_step=int(hit_step[i]), free_steps=int(free_steps[i]), free_path_m=float(path[i]),
        free_initial_distance_m=float(free_distance[i]), progress_m=float(free_distance[i]-distance_end[i]),
        mean_free_entropy=float(entropy_sum[i]/free_steps[i].clamp_min(1)),
        mean_max_action_probability=float(maxprob_sum[i]/free_steps[i].clamp_min(1)),
        mean_value=float(value_sum[i]/free_steps[i].clamp_min(1)), max_abs_logit=float(max_abs_logit[i]),
        first_free_entropy=float(first_free[i]['entropy']),
        first_free_probabilities=first_free[i]['probabilities'].tolist(),
        first_free_logits=first_free[i]['logits'].tolist(), first_free_value=float(first_free[i]['value']),
        action_counts=action_counts[i].tolist()) for i in range(n)]
    audit = dict(setup_pickups=batch['setup_pickups'], setup_raw_reward=batch['setup_raw_reward'],
        setup_learning_reward=0., forced_hits=forced_hits, free_hits=int(hits.sum()), free_wall_hits=wall_hits,
        active_physics_updates=int(torch.stack(roll['active']).sum()), free_updates=int(free_steps.sum()),
        padded_slots=task.MAX_STEPS*n-int(torch.stack(roll['active']).sum()), allocated_slots=task.MAX_STEPS*n,
        completed_trials=n, snap_max_m=snap_max, clearance_min=batch['clearance_min'],
        wall_clearance_min=batch['wall_clearance_min'], forced_path_radius_max=batch['forced_path_radius_max'],
        terminal_bootstrap_zero=True, controller=controller,
        fresh_feedback_on_every_active_input=native_feedback, state_noise_sigma=0., midtrial_resets=0,
        unique_endpoint_cue=not native_feedback)
    audit['forced_updates'] = audit['active_physics_updates'] - audit['free_updates']
    return roll, rows, audit, first_free


def update(net, cfg, opt, source, plan, *, device='cpu'):
    """One full-network update; only invoked by a future separately approved runner."""
    assert all(parameter.requires_grad for parameter in net.parameters())
    expected = {id(parameter) for parameter in net.parameters()}
    supplied = [parameter for group in opt.param_groups for parameter in group['params']]
    assert len(supplied) == len(expected) and {id(parameter) for parameter in supplied} == expected
    roll, rows, audit, _ = collect(net, cfg, source, plan, device=device)
    targets = returns(roll, cfg['gamma']); logits, values = replay(net, roll)
    logit_error = float((logits.detach() - torch.stack(roll['logits'])).abs().max())
    value_error = float((values.detach() - torch.stack(roll['values'])).abs().max())
    assert max(logit_error, value_error) < 3e-4
    loss, metrics = losses(logits, values, targets, roll, cfg)
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
    assert torch.isfinite(norm); opt.step()
    return dict(loss=float(loss.detach()), grad_norm=float(norm), replay_error=logit_error,
                value_replay_error=value_error, **metrics, **audit), rows
