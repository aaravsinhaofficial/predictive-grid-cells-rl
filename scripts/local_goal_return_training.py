"""Standalone native-map goal-return trials; prospective, not frozen or launched.

Every batch contains complete episodes. Setup generates an actual native pickup
snapshot but is outside the learner. A trial ends at its first FREE pickup or
timeout. No later visible reward, cross-trial state, or bootstrap is available.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from torch import nn
from code.env import ArenaEnv
from code.nets import _ortho
from code.common import CodeSource, build_population
from rl_visual_memory import NativeGoal

VISIBLE, TURN_A, TURN_B, PROBE, DELAY, FREE, PAD = range(7)
MAX_STEPS, FREE_STEPS, VISIBLE_STEPS = 160, 30, 15
VARIANTS = tuple(f'{view}_{cue}' for view in ('blind', 'seen') for cue in
    ('native', 'ordinary', 'plus60', 'minus60', 'reset_after_glimpse', 'reset_every_blind'))
HEAD_SEED = 291801


@dataclass(frozen=True)
class Stage:
    low: float
    high: float
    delay: int


STAGES = (Stage(.36, .40, 4), Stage(.44, .48, 15),
          Stage(.52, .56, 30), Stage(.58, .62, 60))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reset_native_heads(net, seed=HEAD_SEED):
    """Explicit native initialization, same CPU draws for both populations."""
    before = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        pi = _ortho(nn.Linear(net.nh, net.n_actions, device='cpu', dtype=torch.float32), .01)
        value = _ortho(nn.Linear(net.nh, 1, device='cpu', dtype=torch.float32), 1.)
    net.pi.load_state_dict(pi.state_dict()); net.v.load_state_dict(value.state_dict())
    heads = ('pi.weight', 'pi.bias', 'v.weight', 'v.bias')
    for key, item in net.state_dict().items():
        if key not in heads:
            assert torch.equal(item.cpu(), before[key]), key
    return dict(seed=seed, actor_gain=.01, critic_gain=1., biases_zero=True,
        retained_all_nonhead_parameters_exactly=True,
        old_head_hashes={k: tensor_hash(before[k]) for k in heads},
        new_head_hashes={k: tensor_hash(net.state_dict()[k]) for k in heads})


class TrialStreams:
    """Private CPU streams; all draws precede outcomes and are checkpointed."""
    def __init__(self, seed):
        self.seed = int(seed); self.batches = 0; self.contexts = 0
        self.generators = {name: torch.Generator().manual_seed(seed + delta)
            for name, delta in (('plan', 0), ('motor', 1), ('action', 2), ('state', 3))}

    def snapshot(self):
        return dict(seed=self.seed, batches=self.batches, contexts=self.contexts,
                    generators={k: g.get_state().clone() for k, g in self.generators.items()})

    def restore(self, saved):
        assert saved['seed'] == self.seed
        self.batches, self.contexts = saved['batches'], saved['contexts']
        for key, state in saved['generators'].items(): self.generators[key].set_state(state.cpu())

    def draw(self, n_contexts, stage, cfg, headings=1):
        assert n_contexts > 0 and headings >= 1 and 1 <= stage.delay <= 60
        u = torch.rand(n_contexts, 19, generator=self.generators['plan'])
        repeat = lambda value: value.repeat_interleave(headings, dim=0)
        endpoint = (u[:, :2] * 2 - 1) * .30
        distance = stage.low + u[:, 2] * (stage.high - stage.low)
        bearing = (u[:, 3] * 2 - 1) * math.pi
        goal = endpoint + distance[:, None] * torch.stack((bearing.cos(), bearing.sin()), 1)
        low_delay = (stage.delay + 1) // 2
        result = dict(endpoint=repeat(endpoint), goal=repeat(goal),
            radius=repeat(distance), heading=repeat((u[:, 4] * 2 - 1) * math.pi)
                + torch.arange(headings).repeat(n_contexts) * (2 * math.pi / headings),
            pickup_angle=repeat((u[:, 5] * 2 - 1) * math.pi),
            duration_a=repeat(5 + (6 * u[:, 6]).long()), duration_b=repeat(5 + (6 * u[:, 7]).long()),
            turn_a=repeat((u[:, 8] >= .5).long()), turn_b=repeat((u[:, 9] >= .5).long()),
            delay=repeat(low_delay + (u[:, 10] * (stage.delay - low_delay + 1)).long()),
            land_pos=repeat((u[:, 11:15] * 2 - 1) * (cfg['arena_half'] - .2)),
            stripe_phase=repeat(u[:, 15:19]),
            context=repeat(torch.arange(n_contexts) + self.contexts),
            heading_index=torch.arange(headings).repeat(n_contexts), stage=asdict(stage),
            n_contexts=n_contexts, headings=headings, batch=self.batches)
        for name, shape, normal in (('motor', (MAX_STEPS, 3, n_contexts), True),
                                    ('action', (MAX_STEPS, n_contexts), False),
                                    ('state', (MAX_STEPS, 2, n_contexts, cfg['nh']), True)):
            draw = (torch.randn if normal else torch.rand)(*shape, generator=self.generators[name])
            result[name] = draw.repeat_interleave(headings, dim=2 if normal else 1)
        self.batches += 1; self.contexts += n_contexts
        return result


def schedule(plan, device):
    n = len(plan['heading']); t = torch.arange(MAX_STEPS, device=device)[:, None]
    a = torch.full((n,), VISIBLE_STEPS, device=device)
    b = a + plan['duration_a'].to(device); probe = b + plan['duration_b'].to(device)
    glimpse = probe + 4; free = glimpse + plan['delay'].to(device); end = free + FREE_STEPS
    phase = torch.full((MAX_STEPS, n), PAD, dtype=torch.long, device=device)
    for value, start, stop in ((VISIBLE, 0, a), (TURN_A, a, b), (TURN_B, b, probe),
                              (PROBE, probe, glimpse), (DELAY, glimpse, free), (FREE, free, end)):
        phase = torch.where((t >= start) & (t < stop), value, phase)
    action = torch.zeros_like(phase)
    action = torch.where(phase == TURN_A, plan['turn_a'].to(device), action)
    action = torch.where(phase == TURN_B, plan['turn_b'].to(device), action)
    action = torch.where(phase == PROBE, 2, action)
    assert int(end.max()) <= 129 < MAX_STEPS
    return dict(phase=phase, forced_action=action, glimpse=glimpse, free=free, end=end, probe=probe)


class TrialArena(ArenaEnv):
    """Same native step/reward; terminal pickup has no subsequent respawn."""
    def __init__(self, cfg, n, device, vision=None):
        self.no_respawn = False; self.tape = None; self.tape_index = 0
        super().__init__(n, half=cfg['arena_half'], episode_len=10000,
            action_repeat=cfg['action_repeat'], v_max=cfg['v_max'], rot_max=cfg['rot_max'],
            goal_radius=cfg['goal_radius'], vision=vision or cfg['vision'], img_noise=0.,
            randomize_textures=True, stagger=False, seed=0, device=device)

    def _respawn(self, mask):
        if not self.no_respawn: super()._respawn(mask)

    def _randn(self, *shape):
        if self.tape is None: return super()._randn(*shape)
        assert shape == (self.n,) and self.tape_index < 3
        value = self.tape[self.tape_index]; self.tape_index += 1
        return value

    def set_motor(self, tape):
        assert tape.shape == (3, self.n)
        self.tape = tape; self.tape_index = 0


def rotate(vectors, angle):
    c, s = angle.cos(), angle.sin()
    return torch.stack((c * vectors[:, 0] - s * vectors[:, 1],
                        s * vectors[:, 0] + c * vectors[:, 1]), 1)


def brake(env, mask):
    for key in ('vf', 'vs', 'vr'): getattr(env, key)[mask] = 0.


@torch.no_grad()
def prepare_batch(cfg, source, plan, variants=('blind_native',), device='cpu'):
    """Physical preparation plan and actual setup pickup; no policy evaluated."""
    n = len(plan['heading']); base = schedule(plan, device)
    dummy = TrialArena(cfg, n, device, vision='none'); dummy.reset_all()
    dummy.no_respawn = True; dummy.pos.zero_(); dummy.goal.fill_(10.); dummy.hd.zero_()
    dummy.mdir[:] = torch.tensor([1., 0.], device=device); brake(dummy, torch.ones(n, dtype=torch.bool, device=device))
    dummy.noise_v = dummy.noise_r = 0.
    points = [dummy.pos.clone()]; end_pos = dummy.pos.clone(); offset = dummy.hd.clone()
    for t in range(int(base['glimpse'].max()) + 1):
        at_glimpse = base['glimpse'] == t
        end_pos[at_glimpse] = dummy.pos[at_glimpse]; offset[at_glimpse] = dummy.hd[at_glimpse]
        brake(dummy, (t == VISIBLE_STEPS) | (base['probe'] == t) | at_glimpse)
        dummy.step(base['forced_action'][t]); points.append(dummy.pos.clone())
    path_radius = (torch.stack(points) - end_pos).norm(dim=2).amax(0)
    endpoint, goal = plan['endpoint'].to(device), plan['goal'].to(device)
    clearance = (endpoint - goal).norm(dim=1) - cfg['goal_radius'] - path_radius
    wall = cfg['arena_half'] - .08 - endpoint.abs().amax(1) - path_radius
    assert bool((clearance > .01).all() and (wall > .01).all()), 'Physically invalid forced path; no outcome-based retry'
    initial_heading = plan['heading'].to(device) - offset
    initial_pos = endpoint - rotate(end_pos, initial_heading)
    groups = len(variants); expand = lambda x: x.repeat((groups,) + (1,) * (x.ndim - 1))
    env = TrialArena(cfg, n * groups, device); env.reset_all(); env.noise_v = env.noise_r = 0.
    env.goal[:] = expand(goal)
    pickup = expand(plan['pickup_angle'].to(device)); direction = torch.stack((pickup.cos(), pickup.sin()), 1)
    env.pos[:] = env.goal - .02 * direction; env.hd[:] = pickup; env.mdir[:] = direction
    brake(env, torch.ones(env.n, dtype=torch.bool, device=device))
    _, reward, done, info = env.step(torch.full((env.n,), 2, device=device, dtype=torch.long))
    assert bool((reward == 10.).all() and info['hit'].all() and not done.any())
    goal_memory = NativeGoal(env.n, cfg['n_cells'], device)
    goal_memory.advance(info['hit'], done, source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']))
    # This explicit new-trial initialization discards setup reward, pose and state.
    env.no_respawn = True; env.pos[:] = expand(initial_pos); env.spawn[:] = env.pos
    env.goal[:] = expand(goal); env.hd[:] = expand(initial_heading)
    env.mdir[:] = torch.stack((env.hd.cos(), env.hd.sin()), 1)
    env.land_pos[:] = expand(plan['land_pos'].to(device)); env.stripe_phase[:] = expand(plan['stripe_phase'].to(device))
    brake(env, torch.ones(env.n, dtype=torch.bool, device=device))
    for key in ('t', 'travelled', 'ep_return', 'n_goals', 'path_eff_sum', 'path_eff_n'): getattr(env, key).zero_()
    env.first_goal_t.fill_(-1)
    sched = {k: v.repeat(1, groups) if v.ndim == 2 else expand(v) for k, v in base.items()}
    return dict(env=env, obs=env.observe(), goal=goal_memory.saved.clone(), schedule=sched,
        endpoint=expand(endpoint), variants=variants, n_base=n, groups=groups,
        clearance_min=float(clearance.min()), wall_clearance_min=float(wall.min()),
        setup_pickups=env.n, setup_raw_reward=float(reward.sum()), forced_path_radius_max=float(path_radius.max()))


def masked_state(state, keep):
    return tuple(value * keep[:, None] for value in state)


def make_source(cfg, device):
    assert cfg['extra_input'] == 'none' and cfg['code_noise'] == 0
    return CodeSource(cfg, build_population(cfg, device), device)


@torch.no_grad()
def collect(net, cfg, source, plan, *, variants=('blind_native',), sigma=.02, device='cpu', retain_inputs=True):
    batch = prepare_batch(cfg, source, plan, variants, device)
    env, obs, goal, sched = (batch[k] for k in ('env', 'obs', 'goal', 'schedule'))
    n, groups = env.n, batch['groups']; state = net.zero_state(n, device)
    pa = torch.full((n,), -1, dtype=torch.long, device=device); pr = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device); held = torch.zeros(n, cfg['n_cells'], device=device)
    flags = lambda suffix: torch.tensor([v.endswith(suffix) for v in variants], device=device).repeat_interleave(batch['n_base'])
    seen = torch.tensor([v.startswith('seen_') for v in variants], device=device).repeat_interleave(batch['n_base'])
    ordinary, reset_after, reset_every = flags('_ordinary'), flags('_reset_after_glimpse'), flags('_reset_every_blind')
    angle = torch.tensor([math.pi/3 if v.endswith('_plus60') else -math.pi/3 if v.endswith('_minus60') else 0.
                          for v in variants], device=device).repeat_interleave(batch['n_base'])
    motor = plan['motor'].repeat(1, 1, groups).to(device)
    uniforms = plan['action'].repeat(1, groups).to(device)
    noise = plan['state'].repeat(1, 1, groups, 1).to(device) * sigma
    keys = ('rgb', 'code', 'pa', 'pr', 'action', 'reward', 'raw_reward', 'terminal',
            'active', 'policy_mask', 'noise', 'reset_before', 'reset_after', 'logits', 'phase')
    roll = {k: [] for k in keys}; roll['goal'] = goal
    hits = torch.zeros(n, dtype=torch.bool, device=device); hit_step = torch.full((n,), -1, dtype=torch.long, device=device)
    free_steps = torch.zeros(n, dtype=torch.long, device=device); path = torch.zeros(n, device=device)
    start_distance = (env.pos-env.goal).norm(dim=1); free_distance = torch.zeros_like(start_distance)
    distance_end = start_distance.clone(); snap_max = 0.; forced_hits = 0; wall_hits = 0
    first_free = {}; blank_embed = net.embed(torch.zeros(n, 3, 84, 84, device=device))
    for t in range(MAX_STEPS):
        phase = sched['phase'][t]; active = alive & (phase != PAD); free = active & (phase == FREE)
        blind = active & (phase >= TURN_A); glimpse = active & (sched['glimpse'] == t)
        if bool(glimpse.any()):
            error = (env.pos - batch['endpoint']).norm(dim=1)[glimpse]
            snap_max = max(snap_max, float(error.max())); assert snap_max < 1e-5
            env.pos[glimpse] = batch['endpoint'][glimpse]; obs = env.observe(obs['vel'])
        native = source(obs['pos'], obs['dir'], obs['vel'])
        ordinary_code = source.pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0.)
        update_held = active & ((t == VISIBLE_STEPS) | glimpse)
        held = torch.where(update_held[:, None], ordinary_code, held)
        code = torch.where(blind[:, None], held, native)
        pulse = source(obs['pos'], rotate(obs['dir'], angle), rotate(obs['vel'], angle))
        code = torch.where(glimpse[:, None], torch.where(ordinary[:, None], ordinary_code, pulse), code)
        brake(env, active & ((t == VISIBLE_STEPS) | (sched['probe'] == t) | glimpse))
        visible = active & (~blind | seen)
        rgb = obs['rgb'] * visible[:, None, None, None]
        embedding = blank_embed.clone()
        if bool(visible.any()): embedding[visible] = net.embed(rgb[visible])
        before_reset = blind & reset_every
        state = masked_state(state, active & ~before_reset)
        x = net._concat(embedding, code, goal, pa, pr, None)
        state = net.lstm(x, state)
        eps = noise[t] * (blind & ~seen)[None, :, None]
        state = tuple(value + delta for value, delta in zip(state, eps))
        logits, values = net.pi(state[0]), net.v(state[0]).squeeze(-1)
        after_reset = glimpse & reset_after
        state = masked_state(state, active & ~after_reset)
        entering = free & (t == sched['free'])
        if bool(entering.any()):
            indices = entering.nonzero().flatten()
            first_free.update({int(i): dict(input=x[i].cpu().clone(), state=torch.cat(state, 1)[i].cpu().clone(),
                position=env.pos[i].cpu().clone(), heading=env.hd[i].cpu().clone(),
                motors=torch.stack((env.vf, env.vs, env.vr), 1)[i].cpu().clone()) for i in indices})
            free_distance[entering] = (env.pos-env.goal).norm(dim=1)[entering]
        sampled = (uniforms[t, :, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        action = torch.where(free, sampled, sched['forced_action'][t])
        old_pos = env.pos.clone(); frozen = {k: v.clone() for k, v in vars(env).items()
            if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == n and k not in ('tape', '_targets')}
        env.noise_v = .03 * free.float(); env.noise_r = .15 * free.float(); env.set_motor(motor[t])
        obs, raw, native_done, info = env.step(action); assert env.tape_index == 3 and not bool(native_done.any())
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
        values_to_save = dict(rgb=rgb, code=code, pa=pa, pr=pr, action=action,
            reward=learned_reward, raw_reward=reward, terminal=terminal, active=active,
            policy_mask=free, noise=eps, reset_before=before_reset, reset_after=after_reset,
            logits=logits, phase=phase)
        for key, value in values_to_save.items():
            if retain_inputs or key not in ('rgb', 'code', 'noise', 'logits'): roll[key].append(value.detach().clone())
        alive &= ~terminal; state = masked_state(state, alive)
        pa = torch.where(alive, action, torch.full_like(action, -1)); pr = torch.zeros_like(pr)
    assert not bool(alive.any())
    roll['bootstrap'] = torch.zeros(n, device=device)
    rows = [dict(variant=variants[i//batch['n_base']], context=int(plan['context'][i%batch['n_base']]),
        heading_index=int(plan['heading_index'][i%batch['n_base']]), hit=bool(hits[i]),
        hit_step=int(hit_step[i]), free_steps=int(free_steps[i]), free_path_m=float(path[i]),
        free_initial_distance_m=float(free_distance[i]), progress_m=float(free_distance[i]-distance_end[i])) for i in range(n)]
    audit = dict(setup_pickups=batch['setup_pickups'], setup_raw_reward=batch['setup_raw_reward'],
        setup_learning_reward=0., forced_hits=forced_hits, free_hits=int(hits.sum()), free_wall_hits=wall_hits,
        active_physics_updates=int(torch.stack(roll['active']).sum()), free_updates=int(free_steps.sum()),
        padded_slots=MAX_STEPS*n-int(torch.stack(roll['active']).sum()),
        allocated_slots=MAX_STEPS*n, completed_trials=n, snap_max_m=snap_max,
        clearance_min=batch['clearance_min'], wall_clearance_min=batch['wall_clearance_min'],
        forced_path_radius_max=batch['forced_path_radius_max'], terminal_bootstrap_zero=True)
    audit['forced_updates'] = audit['active_physics_updates'] - audit['free_updates']
    return roll, rows, audit, first_free


def returns(roll, gamma):
    future = torch.zeros_like(roll['bootstrap']); result = []
    for reward, terminal, active in zip(reversed(roll['reward']), reversed(roll['terminal']), reversed(roll['active'])):
        future = (reward + gamma * (~terminal) * future) * active
        result.append(future)
    return torch.stack(result[::-1])


def replay(net, roll):
    n = roll['goal'].shape[0]; device = roll['goal'].device
    state = net.zero_state(n, device); blank = torch.zeros(n, 3, 84, 84, device=device)
    blank_embedding = net.embed(blank); logits, values = [], []
    for t in range(MAX_STEPS):
        state = masked_state(state, roll['active'][t] & ~roll['reset_before'][t])
        rgb = roll['rgb'][t]; visible = rgb.abs().flatten(1).sum(1) > 0
        embedding = blank_embedding.clone()
        if bool(visible.any()): embedding[visible] = net.embed(rgb[visible])
        x = net._concat(embedding, roll['code'][t], roll['goal'], roll['pa'][t], roll['pr'][t], None)
        state = net.lstm(x, state)
        state = tuple(value + delta for value, delta in zip(state, roll['noise'][t]))
        logits.append(net.pi(state[0])); values.append(net.v(state[0]).squeeze(-1))
        state = masked_state(state, roll['active'][t] & ~roll['reset_after'][t] & ~roll['terminal'][t])
    return torch.stack(logits), torch.stack(values)


def losses(logits, values, targets, roll, cfg):
    distribution = torch.distributions.Categorical(logits=logits)
    mask = torch.stack(roll['policy_mask']); active = torch.stack(roll['active'])
    policy = -(mask * distribution.log_prob(torch.stack(roll['action'])) * (targets-values).detach()).sum(0).mean()
    entropy = (mask * distribution.entropy()).sum(0).mean()
    value = .5 * (active * (targets-values).square()).sum(0).mean()
    # Fixed convention across stages; no success-dependent active-step denominator.
    total = (100 / MAX_STEPS) * (policy + cfg['value_coef'] * value - cfg['entropy'] * entropy)
    return total, dict(policy_loss=float(policy.detach()), value_loss=float(value.detach()),
                       entropy_sum_per_trial=float(entropy.detach()))


def update(net, cfg, opt, source, plan, device='cpu', sigma=.02):
    roll, rows, audit, _ = collect(net, cfg, source, plan, sigma=sigma, device=device)
    targets = returns(roll, cfg['gamma']); logits, values = replay(net, roll)
    error = float((logits.detach() - torch.stack(roll['logits'])).abs().max())
    assert error < 3e-4, f'Exact stored-noise replay failed: {error}'
    loss, metrics = losses(logits, values, targets, roll, cfg)
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
    assert torch.isfinite(norm); opt.step()
    return dict(loss=float(loss.detach()), grad_norm=float(norm), replay_error=error, **metrics, **audit), rows
