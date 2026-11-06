"""Draft native goal-cue-memory primitives; no study execution on import.

This module is separate from the completed local-return studies. Goal snapshots
come from actual native boundary pickups. Setup never enters the learner.
Only synthetic policies and physical checks are permitted during draft review.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch

from local_goal_return_training import TrialArena, brake, make_source, masked_state, returns

CUE_STEPS, FREE_STEPS, MAX_STEPS = 15, 90, 160
GOAL_DISTANCE, PICKUP_OUTSIDE_GAP = .75, .002
CUE, DELAY, FREE, PAD = range(4)
STATE_FIELDS = ('pos', 'goal', 'spawn', 'hd', 'mdir', 'vf', 'vs', 'vr', 't', 'partial',
                'ep_idx', 'draw_idx', 'travelled', 'ep_return', 'n_goals', 'first_goal_t',
                'path_eff_sum', 'path_eff_n', 'land_pos', 'stripe_phase')


class TrialStreams:
    """Independent private CPU streams; complete draws precede all outcomes.

    Each conditional family shares room appearance, incoming offset RELATIVE to
    goal bearing, and tapes across goal bearings. A training family has one
    continuously sampled bearing. An independent uniform offset makes the
    incoming world angle marginally uniform and independent of that bearing.
The optional angular sweep is for physical/isolation tests and later explicitly
declared conditional analyses; it is not an implicit evaluation-bank choice.
"""
    NAMES = ('goal', 'pickup', 'layout', 'motor', 'action', 'delay')

    def __init__(self, seed):
        self.seed = int(seed); self.batches = 0; self.contexts = 0; self.delay_draws = 0
        self.generators = {name: torch.Generator().manual_seed(self.seed + i)
                           for i, name in enumerate(self.NAMES)}

    def snapshot(self):
        return dict(seed=self.seed, batches=self.batches, contexts=self.contexts, delay_draws=self.delay_draws,
                    generators={k: g.get_state().clone() for k, g in self.generators.items()})

    def restore(self, saved):
        assert saved['seed'] == self.seed and set(saved['generators']) == set(self.NAMES)
        self.batches, self.contexts = saved['batches'], saved['contexts']
        self.delay_draws = saved['delay_draws']
        assert 0 <= self.delay_draws <= self.batches
        for key, state in saved['generators'].items(): self.generators[key].set_state(state.cpu())

    def draw(self, n_contexts, cfg, *, delay_steps=0, angles_per_context=1):
        assert n_contexts > 0 and angles_per_context >= 1 and 0 <= delay_steps <= 45
        repeat = lambda value: value.repeat_interleave(angles_per_context, dim=0)
        angle = (torch.rand(n_contexts, generator=self.generators['goal']) * 2 - 1) * math.pi
        bearing = repeat(angle) + torch.arange(angles_per_context).repeat(n_contexts) * (2 * math.pi / angles_per_context)
        pickup_offset = (torch.rand(n_contexts, generator=self.generators['pickup']) * 2 - 1) * math.pi
        layout = torch.rand(n_contexts, 8, generator=self.generators['layout'])
        delay = torch.full((n_contexts,), int(delay_steps), dtype=torch.long)
        n = n_contexts * angles_per_context
        result = dict(goal_bearing=bearing, goal=GOAL_DISTANCE * torch.stack((bearing.cos(), bearing.sin()), 1),
            pickup_offset=repeat(pickup_offset), pickup_angle=bearing+repeat(pickup_offset),
            start=torch.zeros(n, 2), heading=torch.zeros(n),
            land_pos=repeat((layout[:, :4] * 2 - 1) * (cfg['arena_half'] - .2)),
            stripe_phase=repeat(layout[:, 4:]), delay=repeat(delay), delay_steps=int(delay_steps),
            delay_cap=None, delay_was_sampled=False,
            context=repeat(torch.arange(n_contexts) + self.contexts),
            angle_index=torch.arange(angles_per_context).repeat(n_contexts),
            n_contexts=n_contexts, angles_per_context=angles_per_context, batch=self.batches)
        motor = torch.randn(MAX_STEPS, 3, n_contexts, generator=self.generators['motor'])
        action = torch.rand(MAX_STEPS, n_contexts, generator=self.generators['action'])
        result['motor'] = motor.repeat_interleave(angles_per_context, dim=2)
        result['action'] = action.repeat_interleave(angles_per_context, dim=1)
        self.batches += 1; self.contexts += n_contexts
        return result

    def draw_training(self, n_contexts, cfg, *, delay_cap):
        """One shared private integer per batch, INCLUDING one call at cap0."""
        assert delay_cap in (0,15,45) and n_contexts > 0
        delay = int(torch.randint(0,delay_cap+1,(1,),generator=self.generators['delay']))
        plan = self.draw(n_contexts,cfg,delay_steps=delay)
        self.delay_draws += 1
        plan['delay_cap'] = int(delay_cap); plan['delay_was_sampled'] = True
        return plan


def validate_plan(cfg, plan):
    """Reject malformed geometry rather than retrying or silently fixing it."""
    assert cfg['action_repeat'] == 4 and cfg['goal_radius'] == .3
    assert cfg['pgc_shift'] == .2 and cfg['extra_input'] == 'none' and cfg['code_noise'] == 0
    assert cfg.get('goal_code_shift', 'intact') == 'intact'
    assert cfg['vision'] in ('raycast', 'heading_only', 'sighted', 'none')
    n = len(plan['goal_bearing'])
    assert n == plan['n_contexts'] * plan['angles_per_context']
    assert plan['goal'].shape == plan['start'].shape == (n, 2)
    assert torch.equal(plan['start'], torch.zeros_like(plan['start']))
    assert torch.equal(plan['heading'], torch.zeros_like(plan['heading']))
    expected = GOAL_DISTANCE * torch.stack((plan['goal_bearing'].cos(), plan['goal_bearing'].sin()), 1)
    # Geometric consistency only: CPU trig may differ by a few last bits across
    # platforms. A future sealed-bank loader must authenticate saved bytes/tensor
    # hashes exactly. Never regenerate, replace or "repair" these saved values.
    assert torch.allclose(plan['goal'], expected, atol=1e-6, rtol=0), 'Goal center must follow the declared bearing'
    assert torch.equal(plan['pickup_angle'],plan['goal_bearing']+plan['pickup_offset'])
    assert 0 <= plan['delay_steps'] <= 45
    assert torch.equal(plan['delay'], torch.full_like(plan['delay'], plan['delay_steps']))
    assert isinstance(plan['delay_was_sampled'],bool)
    if plan['delay_was_sampled']:
        assert plan['delay_cap'] in (0,15,45) and plan['delay_steps'] <= plan['delay_cap']
    else: assert plan['delay_cap'] is None
    assert plan['motor'].shape == (MAX_STEPS, 3, n)
    assert plan['action'].shape == (MAX_STEPS, n)
    for value in plan.values():
        if isinstance(value, torch.Tensor): assert bool(torch.isfinite(value).all())
    assert bool(((plan['action'] >= 0) & (plan['action'] < 1)).all())
    return n


def physical_snapshot(env):
    """All per-trial native state and its private draw clock, excluding its goal."""
    names = tuple(name for name in STATE_FIELDS if name != 'goal')
    return {**{name: getattr(env, name).detach().clone() for name in names},
            'environment_rng': env.gen.get_state().clone(), 'episodes_done': env.episodes_done}


@torch.no_grad()
def prepare_batch(cfg, source, plan, *, device='cpu'):
    """Generate a real boundary pickup, then construct a clean common start.

The setup and trial are separate native Arena objects. Neither calls random
respawn. No setup state, layout, draw clock or reward is transferred to the
trial; only the actual saved population code and physical reward target remain.
"""
    n = validate_plan(cfg, plan); goal = plan['goal'].to(device)
    assert source.probe == 'intact' and source.goal_shift_scale == 1 and not source.silence_pgc
    pickup = plan['pickup_angle'].to(device)
    direction = torch.stack((pickup.cos(), pickup.sin()), 1)
    setup = TrialArena(cfg, n, device, vision='none'); setup.no_respawn = True
    setup.noise_v = setup.noise_r = 0.
    setup.goal[:] = goal
    setup.pos[:] = goal - (cfg['goal_radius'] + PICKUP_OUTSIDE_GAP) * direction
    setup.spawn[:] = setup.pos; setup.hd[:] = pickup; setup.mdir[:] = direction
    brake(setup, torch.ones(n, dtype=torch.bool, device=device))
    before = setup.pos.clone(); distance_before = (before - goal).norm(dim=1)
    assert bool((distance_before > cfg['goal_radius']).all())
    assert bool((before.abs() < setup.half - setup.wall_margin).all())
    setup_rng = setup.gen.get_state().clone()
    setup.set_motor(torch.zeros(3, n, device=device))
    _, reward, done, info = setup.step(torch.full((n,), 2, dtype=torch.long, device=device))
    assert setup.tape_index == 3 and torch.equal(setup.gen.get_state(), setup_rng)
    assert bool((reward == 10).all() and info['hit'].all() and not done.any())
    assert not bool(info['hit_wall'].any())
    assert bool(((info['hit_pos'] - goal).norm(dim=1) < cfg['goal_radius']).all())
    assert torch.equal(setup.pos, info['hit_pos']), 'Setup pickup must not respawn'
    saved_goal = source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']).clone()
    ordinary_goal = source.pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0.).clone()
    # A fresh environment is an explicit reset of all native bookkeeping, not
    # a selective clearing of visible fields in the pickup environment.
    env = TrialArena(cfg, n, device); env.no_respawn = True
    env.noise_v = env.noise_r = 0.
    env.goal[:] = goal; env.pos[:] = plan['start'].to(device); env.spawn[:] = env.pos
    env.hd[:] = plan['heading'].to(device)
    env.mdir[:] = torch.stack((env.hd.cos(), env.hd.sin()), 1)
    env.land_pos[:] = plan['land_pos'].to(device); env.stripe_phase[:] = plan['stripe_phase'].to(device)
    brake(env, torch.ones(n, dtype=torch.bool, device=device))
    env.ep_idx.zero_(); env.partial.zero_(); env.draw_idx.zero_()
    # All other native counters are zero at construction; first_goal_t is -1.
    assert bool((env.t == 0).all() and (env.n_goals == 0).all() and (env.ep_return == 0).all())
    assert bool((env.first_goal_t == -1).all())
    observation = env.observe()
    return dict(env=env, obs=observation, saved_goal=saved_goal, ordinary_goal=ordinary_goal,
        initial_physical_state=physical_snapshot(env),
        setup=dict(position_before=before, distance_before=distance_before,
            hit_position=info['hit_pos'].clone(), hit_direction=info['hit_dir'].clone(),
            hit_velocity=info['hit_vel'].clone(), native_reward=reward.clone(),
            native_hit=info['hit'].clone(), native_wall=info['hit_wall'].clone(),
            heading=pickup.clone(), private_rng_unchanged=True, no_respawn=True,
            pickups=n, learning_reward=0., transferred_reward_history=False))


def schedule(plan, device='cpu'):
    """Wait inputs are recurrent updates, not physical or policy actions."""
    n = len(plan['goal_bearing']); t = torch.arange(MAX_STEPS, device=device)[:, None]
    free = CUE_STEPS + plan['delay'].to(device); end = free + FREE_STEPS
    phase = torch.full((MAX_STEPS, n), PAD, dtype=torch.long, device=device)
    phase = torch.where(t < CUE_STEPS, CUE, phase)
    phase = torch.where((t >= CUE_STEPS) & (t < free), DELAY, phase)
    phase = torch.where((t >= free) & (t < end), FREE, phase)
    assert int(end.max()) <= 150 < MAX_STEPS
    return dict(phase=phase, free=free, end=end)


@torch.no_grad()
def step_free(env, actions, active, motor):
    """Apply unmodified native physics ONLY to selected free-decision rows.

Rows that are waiting or terminal never enter ArenaEnv.step. Taking/merging
its explicit per-row state lets variable delays coexist in one tensor batch;
it avoids issuing a fictional no-op/seventh action or advancing wait clocks.
Native dynamics, reward, pre-hit positions and directions are unchanged.
"""
    n = env.n; indices = active.nonzero().flatten(); device = env.pos.device
    assert active.shape == actions.shape == (n,) and motor.shape == (3, n)
    assert env.no_respawn and env.img_noise == 0
    reward = torch.zeros(n, device=device); done = torch.zeros(n, dtype=torch.bool, device=device)
    velocity = torch.zeros(n, 2, device=device)
    info = dict(hit=done.clone(), hit_wall=done.clone(), hit_pos=env.pos.clone(),
                hit_dir=env.mdir.clone(), hit_vel=velocity.clone(), ep_stats=None)
    if indices.numel():
        assert bool(((actions[indices] >= 0) & (actions[indices] < 6)).all())
        full = {key: getattr(env, key) for key in STATE_FIELDS}
        try:
            for key, value in full.items(): setattr(env, key, value[indices].clone())
            env.n = len(indices); env.noise_v = .03; env.noise_r = .15
            env.set_motor(motor[:, indices])
            observed, native_reward, native_done, native_info = env.step(actions[indices])
            assert env.tape_index == 3 and not bool(native_done.any())
            assert native_info['ep_stats'] is None
            for key, value in full.items(): value[indices] = getattr(env, key)
            reward[indices] = native_reward; done[indices] = native_done
            velocity[indices] = observed['vel']
            for key in ('hit', 'hit_wall', 'hit_pos', 'hit_dir', 'hit_vel'):
                info[key][indices] = native_info[key]
        finally:
            env.n = n
            for key, value in full.items(): setattr(env, key, value)
    return env.observe(velocity), reward, done, info


def goal_input(saved_goal, active, clock, mode):
    """The ONLY route from the external goal cache into the RNN input."""
    assert mode in ('continuous', 'brief', 'no_cue')
    available = active & ((mode == 'continuous') | ((mode == 'brief') & (clock < CUE_STEPS)))
    return torch.where(available[:, None], saved_goal, torch.zeros_like(saved_goal))


@torch.no_grad()
def collect(net, cfg, source, plan, *, device='cpu', cue_mode='continuous',
            goal_kind='native', erase_after_cue=False, retain_inputs=True, controller='policy',
            goal_override=None, first_free_state_override=None):
    """Draft complete-trial collector. Does not load or optimize any model.

During cue/delay the task holds the native pose fixed, pa=-1 and pr=0. The
recurrent state updates and the critic remain active; actor/entropy masks are
zero. FREE uses six native actions, fresh rendered images/current code and
native motor noise. Image/recurrent noise are explicitly zero throughout.
"""
    assert cue_mode in ('continuous', 'brief', 'no_cue') and goal_kind in ('native', 'ordinary')
    assert controller in ('policy', 'uniform', 'planner')
    assert controller == 'policy' or not retain_inputs, 'Physical reference cannot enter learning'
    assert controller == 'policy' or (goal_override is None and first_free_state_override is None)
    batch = prepare_batch(cfg, source, plan, device=device); env, obs = batch['env'], batch['obs']
    sched = schedule(plan, device); n = env.n
    saved_goal = batch['saved_goal'] if goal_kind == 'native' else batch['ordinary_goal']
    if goal_override is not None:
        assert goal_override.shape == saved_goal.shape and goal_override.dtype == saved_goal.dtype
        assert bool(torch.isfinite(goal_override).all())
        saved_goal = goal_override.to(device).detach().clone()
    override = torch.zeros(n, 2*cfg['nh'], device=device)
    if first_free_state_override is not None:
        assert first_free_state_override.shape == override.shape and first_free_state_override.dtype == override.dtype
        assert bool(torch.isfinite(first_free_state_override).all())
        override = first_free_state_override.to(device).detach().clone()
    state = net.zero_state(n, device); alive = torch.ones(n, dtype=torch.bool, device=device)
    pa = torch.full((n,), -1, dtype=torch.long, device=device); pr = torch.zeros(n, device=device)
    motor, uniforms = plan['motor'].to(device), plan['action'].to(device)
    keys = ('rgb', 'code', 'goal_input', 'pa', 'pr', 'action', 'reward', 'raw_reward',
            'terminal', 'active', 'physics_active', 'policy_mask', 'noise', 'reset_before', 'state_override_mask',
            'reset_after', 'logits', 'values', 'phase', 'position', 'observation_position',
            'observation_heading', 'observation_direction', 'observation_velocity', 'motors_before_action')
    roll = {key: [] for key in keys}
    # Analysis metadata; replay deliberately never reads either external cache.
    roll['saved_goal'] = saved_goal.clone(); roll['physical_goal'] = env.goal.clone()
    roll['native_saved_goal'] = batch['saved_goal'].clone()
    roll['ordinary_saved_goal'] = batch['ordinary_goal'].clone()
    roll['first_free_state_override'] = override.clone()
    hits = torch.zeros(n, dtype=torch.bool, device=device)
    free_steps = torch.zeros(n, dtype=torch.long, device=device); hit_step = torch.full_like(free_steps, -1)
    path = torch.zeros(n, device=device); distance_end = (env.pos - env.goal).norm(dim=1)
    start_distance = distance_end.clone(); entropy_sum = torch.zeros(n, device=device)
    action_counts = torch.zeros(n, 6, dtype=torch.long, device=device)
    first_free = {}; cue_end = None; erase_count = 0; wall_hits = 0
    blank_embedding = net.embed(torch.zeros(n, 3, 84, 84, device=device))
    environment_rng = env.gen.get_state().clone()
    for t in range(MAX_STEPS):
        phase = sched['phase'][t]; active = alive & (phase != PAD)
        free = active & (phase == FREE); waiting = active & ~free
        reset = active & bool(erase_after_cue and t == CUE_STEPS)
        erase_count += int(reset.sum()); state = masked_state(state, active & ~reset)
        entering = free & (t == sched['free'])
        state_before_intervention = state
        override_mask = entering & (first_free_state_override is not None)
        state = tuple(torch.where(override_mask[:, None], replacement, value)
                      for value, replacement in zip(state, override.split(cfg['nh'], dim=1)))
        state_before_input = state
        assert bool((pa[waiting] == -1).all() and (pr == 0).all())
        assert bool((env.pos[waiting] == plan['start'].to(device)[waiting]).all())
        assert bool((env.hd[waiting] == plan['heading'].to(device)[waiting]).all())
        rgb = obs['rgb'] * active[:, None, None, None]
        native_code = source(obs['pos'], obs['dir'], obs['vel'])
        code = native_code * active[:, None]
        supplied_goal = goal_input(saved_goal, active, t, cue_mode)
        embedding = blank_embedding.clone()
        if bool(active.any()): embedding[active] = net.embed(rgb[active])
        x = net._concat(embedding, code, supplied_goal, pa, pr, None)
        state = net.lstm(x, state)
        logits, values = net.pi(state[0]), net.v(state[0]).squeeze(-1)
        probabilities = logits.softmax(1)
        if controller == 'uniform': probabilities = torch.full_like(probabilities, 1/6)
        elif controller == 'planner':
            from local_goal_return_training import expected_next_distance
            probabilities = torch.nn.functional.one_hot(expected_next_distance(env).argmin(1), 6).to(logits.dtype)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(1)
        for index in entering.nonzero().flatten().tolist():
            first_free[index] = dict(input=x[index].cpu().clone(), state=torch.cat(state, 1)[index].cpu().clone(),
                state_before_input=torch.cat(state_before_input, 1)[index].cpu().clone(),
                state_before_intervention=torch.cat(state_before_intervention, 1)[index].cpu().clone(),
                state_was_replaced=bool(override_mask[index]),
                state_timing='After processing the first FREE input; before sampling its action.',
                state_before_input_timing='After any scheduled erasure/donor substitution; immediately before processing the first FREE input.',
                state_before_intervention_timing='After any scheduled erasure; before an optional donor substitution at the first FREE input.',
                goal_input=supplied_goal[index].cpu().clone(), position=env.pos[index].cpu().clone(),
                heading=env.hd[index].cpu().clone(), motors=torch.stack((env.vf, env.vs, env.vr), 1)[index].cpu().clone(),
                probabilities=probabilities[index].cpu().clone(), logits=logits[index].cpu().clone(), value=values[index].cpu().clone())
        if t == CUE_STEPS - 1: cue_end = torch.cat(state, 1).cpu().clone()
        sampled = (uniforms[t, :, None] > probabilities.cumsum(1)).sum(1).clamp_max(5)
        action = torch.where(free, sampled, torch.full_like(sampled, -1))
        entropy_sum += free * entropy
        action_counts.scatter_add_(1, sampled[:, None], free.long()[:, None])
        observed = {key: obs[field].clone() for key, field in (
            ('observation_position', 'pos'), ('observation_heading', 'hd'),
            ('observation_direction', 'dir'), ('observation_velocity', 'vel'))}
        observed['motors_before_action'] = torch.stack((env.vf, env.vs, env.vr), 1).clone()
        before = env.pos.clone()
        obs, raw, done, info = step_free(env, action, free, motor[t])
        assert not bool(done.any()) and torch.equal(env.gen.get_state(), environment_rng)
        terminal = free & (info['hit'] | (t == sched['end'] - 1))
        reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
        assert not bool(raw[~free].any())
        hits |= free & info['hit']; hit_step[info['hit']] = free_steps[info['hit']] + 1
        free_steps += free.long(); path += free * (info['hit_pos'] - before).norm(dim=1)
        distance_end = torch.where(free, (info['hit_pos'] - env.goal).norm(dim=1), distance_end)
        wall_hits += int(info['hit_wall'].sum())
        saved = dict(rgb=rgb, code=code, goal_input=supplied_goal, pa=pa, pr=pr, action=action,
            reward=reward, raw_reward=raw, terminal=terminal, active=active, physics_active=free,
            policy_mask=free, noise=torch.zeros(2, n, cfg['nh'], device=device), reset_before=reset,
            state_override_mask=override_mask,
            reset_after=torch.zeros_like(active), logits=logits, values=values, phase=phase,
            position=env.pos, **observed)
        for key, value in saved.items():
            if retain_inputs or key not in ('rgb', 'code', 'goal_input', 'noise', 'logits'):
                roll[key].append(value.detach().clone())
        alive &= ~terminal; state = masked_state(state, alive & active)
        pa = torch.where(alive & free, action, torch.full_like(action, -1)); pr = torch.zeros_like(pr)
    assert not bool(alive.any()) and len(first_free) == n
    assert torch.equal(env.t, free_steps), 'Wait and terminal rows must not advance native clocks'
    roll['bootstrap'] = torch.zeros(n, device=device)
    rows = [dict(context=int(plan['context'][i]), angle_index=int(plan['angle_index'][i]),
        hit=bool(hits[i]), hit_step=int(hit_step[i]), free_steps=int(free_steps[i]),
        free_path_m=float(path[i]), initial_distance_m=float(start_distance[i]),
        final_distance_m=float(distance_end[i]), progress_m=float(start_distance[i]-distance_end[i]),
        mean_free_entropy=float(entropy_sum[i]/free_steps[i].clamp_min(1)),
        action_counts=action_counts[i].tolist()) for i in range(n)]
    recurrent_updates = int(torch.stack(roll['active']).sum()); physics_updates = int(free_steps.sum())
    audit = dict(completed_trials=n, setup_pickups=batch['setup']['pickups'], setup_learning_reward=0.,
        active_recurrent_updates=recurrent_updates, active_physics_updates=physics_updates,
        free_updates=physics_updates, wait_updates=recurrent_updates-physics_updates,
        allocated_slots=MAX_STEPS*n, padded_slots=MAX_STEPS*n-recurrent_updates,
        free_hits=int(hits.sum()), free_wall_hits=wall_hits, forced_actions=0, forced_rewards=0.,
        controller=controller, cue_mode=cue_mode, goal_kind=goal_kind, erased_states=erase_count,
        goal_override_used=goal_override is not None, first_free_state_override_used=first_free_state_override is not None,
        terminal_bootstrap_zero=True, task_owned_stationary_wait=True, img_noise=0.,
        free_motor_noise_v=.03, free_motor_noise_r=.15, state_noise_sigma=0.,
        full_goal_vector_withdrawn=cue_mode != 'continuous', saved_goal_used_only_by_explicit_input=True,
        native_environment_rng_unchanged=True)
    evidence = dict(setup=batch['setup'], initial_physical_state=batch['initial_physical_state'],
                    first_free=first_free, cue_end_state=cue_end, final_physical_state=physical_snapshot(env))
    return roll, rows, audit, evidence


def replay(net, roll):
    """Full-trial BPTT uses each recorded goal tensor, never a fixed cache."""
    n = roll['pa'][0].shape[0]; device = roll['pa'][0].device
    state = net.zero_state(n, device)
    blank = net.embed(torch.zeros(n, 3, 84, 84, device=device)); logits, values = [], []
    assert len(roll['goal_input']) == MAX_STEPS
    for t in range(MAX_STEPS):
        active = roll['active'][t]; state = masked_state(state, active & ~roll['reset_before'][t])
        state = tuple(torch.where(roll['state_override_mask'][t][:, None], replacement, value)
                      for value, replacement in zip(state, roll['first_free_state_override'].split(net.nh, dim=1)))
        embedding = blank.clone()
        if bool(active.any()): embedding[active] = net.embed(roll['rgb'][t][active])
        x = net._concat(embedding, roll['code'][t], roll['goal_input'][t], roll['pa'][t], roll['pr'][t], None)
        state = net.lstm(x, state)
        state = tuple(value + delta for value, delta in zip(state, roll['noise'][t]))
        logits.append(net.pi(state[0])); values.append(net.v(state[0]).squeeze(-1))
        state = masked_state(state, active & ~roll['reset_after'][t] & ~roll['terminal'][t])
    return torch.stack(logits), torch.stack(values)


def losses(logits, values, targets, roll, cfg):
    """Waits receive critic credit, with no fictitious policy-action loss."""
    distribution = torch.distributions.Categorical(logits=logits)
    actions = torch.stack(roll['action']); mask = torch.stack(roll['policy_mask'])
    active = torch.stack(roll['active'])
    assert bool((actions[~mask] == -1).all())
    assert bool(((actions[mask] >= 0) & (actions[mask] < 6)).all())
    # The clamped index is unused by the exact-zero actor/entropy mask.
    policy = -(mask * distribution.log_prob(actions.clamp_min(0)) * (targets-values).detach()).sum(0).mean()
    entropy = (mask * distribution.entropy()).sum(0).mean()
    value = .5 * (active * (targets-values).square()).sum(0).mean()
    loss = (100 / MAX_STEPS) * (policy + cfg['value_coef'] * value - cfg['entropy'] * entropy)
    return loss, dict(policy_loss=float(policy.detach()), value_loss=float(value.detach()),
        entropy_sum_per_trial=float(entropy.detach()),
        entropy_per_free_decision=float((mask*distribution.entropy()).sum().detach()/mask.sum().clamp_min(1)))
