"""Reward-only blind reorientation in the original navigation LSTM.

Separate prospective experiment. No ring/angle target, controller, or decoder.
This module defines the event, exact noisy rollout/replay, and clean controls.
The training CLI and freeze manifest live in rl_blind_reorientation_train.py.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from code.common import CodeSource, build_env, build_population, extra_from_obs
from lstm_code_swap import MatchedArena
from rl_goal_memory import episode_summary, reset_state
from rl_visual_memory import NativeGoal, require_time
from rl_visual_memory_noise import StateNoise

READY, VISIBLE, TURN_A, TURN_B, PROBE, DELAY, FREE = range(7)
PHASE_NAMES = ('waiting', 'visible', 'turn_a', 'turn_b', 'forward_probe', 'left_delay', 'free_blind')
VISIBLE_STEPS, PROBE_STEPS, FREE_STEPS = 15, 4, 30
TURN_MIN, TURN_MAX, MAX_DELAY = 5, 10, 60
MAX_WINDOW = VISIBLE_STEPS + 2 * TURN_MAX + PROBE_STEPS + MAX_DELAY + FREE_STEPS
ROLLOUT, EVENT_CAP, SIGMA = 160, 8, .02
TRAIN_SEED, CLOCK_SEED, NOISE_SEED, EVAL_SEED = 146201, 146203, 146205, 148201
PULSES = ('native', 'ordinary', 'plus60', 'minus60', 'reset_after_glimpse', 'reset_every_blind')
VARIANTS = tuple(f'{vision}_{pulse}' for vision in ('blind', 'seen') for pulse in PULSES)


class ReorientationClock:
    """Per-slot, per-episode event banks; no global or policy RNG consumption.

Each nth admitted event has the same turn labels/durations and delay quantile
across matched branches, even if earlier free navigation changed its onset.
All branches retain their own actual reward-driven clock. At most one event is
admitted in an unroll, and the global worst case (129 updates) must fit in both
the unroll and episode. Waiting remains ordinary, visible native navigation.
"""
    def __init__(self, n, device, seed=CLOCK_SEED, groups=1):
        self.n, self.groups, self.seed = n, groups, int(seed)
        total = n * groups
        self.phase = torch.zeros(total, dtype=torch.long, device=device)
        self.remaining = torch.zeros_like(self.phase)
        self.age = torch.zeros_like(self.phase)
        self.known = torch.zeros(total, dtype=torch.bool, device=device)
        self.used_segment = torch.zeros_like(self.known)
        self.event_count = torch.zeros_like(self.phase)
        self.episode = torch.zeros_like(self.phase)
        self.turn_a = torch.zeros_like(self.phase); self.turn_b = torch.zeros_like(self.phase)
        self.duration_a = torch.zeros_like(self.phase)
        self.duration_b = torch.zeros_like(self.phase); self.duration_delay = torch.zeros_like(self.phase)
        self.elapsed = torch.zeros_like(self.phase)
        self.plans = torch.zeros(total, EVENT_CAP, 5, dtype=torch.float64, device='cpu')
        self.stats = {name: torch.zeros(total, dtype=torch.long, device=device) for name in
            ('admitted', 'completed', 'interrupted_reward', 'interrupted_done',
             'wait_segment', 'wait_cap', 'wait_episode', 'coverage_violations')}
        self.phase_counts = torch.zeros(total, len(PHASE_NAMES), dtype=torch.long, device=device)
        self._plans(torch.ones_like(self.known))

    def _plans(self, mask):
        for index in mask.nonzero().flatten().cpu().tolist():
            slot = index % self.n
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, slot, int(self.episode[index])]))
            self.plans[index] = torch.from_numpy(rng.random((EVENT_CAP, 5)))

    def begin_segment(self):
        active = self.phase != READY
        self.stats['coverage_violations'] += active.long()
        assert not bool(active.any()), 'An admitted event crossed a BPTT boundary'
        self.used_segment.zero_()

    def prepare(self, segment_remaining, episode_remaining, max_delay):
        assert 1 <= max_delay <= MAX_DELAY
        ready = self.known & (self.phase == READY)
        cap = self.event_count >= EVENT_CAP
        segment_bad = self.used_segment | (segment_remaining < MAX_WINDOW)
        episode_bad = episode_remaining < MAX_WINDOW
        self.stats['wait_cap'] += (ready & cap).long()
        self.stats['wait_segment'] += (ready & ~cap & segment_bad).long()
        self.stats['wait_episode'] += (ready & ~cap & ~segment_bad & episode_bad).long()
        admit = ready & ~cap & ~segment_bad & ~episode_bad
        if bool(admit.any()):
            idx = admit.nonzero().flatten()
            draws = self.plans[idx.cpu(), self.event_count[idx].cpu()].to(self.phase.device, dtype=torch.float32)
            duration_a = TURN_MIN + (draws[:, 1] * (TURN_MAX - TURN_MIN + 1)).long().clamp_max(TURN_MAX-TURN_MIN)
            self.turn_a[idx] = (draws[:, 0] >= .5).long()
            self.turn_b[idx] = (draws[:, 2] >= .5).long()
            self.duration_b[idx] = TURN_MIN + (draws[:, 3] * (TURN_MAX - TURN_MIN + 1)).long().clamp_max(TURN_MAX-TURN_MIN)
            low = (max_delay + 1) // 2
            self.duration_delay[idx] = low + (draws[:, 4] * (max_delay - low + 1)).long().clamp_max(max_delay-low)
            # Reuse remaining for visible; duration_a gets its own saved tensor.
            self.duration_a[idx] = duration_a
            self.phase[idx] = VISIBLE; self.remaining[idx] = VISIBLE_STEPS
            self.age[idx] = 0; self.elapsed[idx] = 0
            self.event_count[idx] += 1; self.stats['admitted'][idx] += 1
            self.used_segment[idx] = True
        return admit

    def blind(self):
        return self.phase >= TURN_A

    def forced(self):
        return (self.phase >= TURN_A) & (self.phase <= DELAY)

    def glimpse(self):
        return (self.phase == DELAY) & (self.age == 0)

    def action(self, sampled):
        answer = torch.where(self.phase == TURN_A, self.turn_a, sampled)
        answer = torch.where(self.phase == TURN_B, self.turn_b, answer)
        answer = torch.where(self.phase == PROBE, torch.full_like(answer, 2), answer)
        return torch.where(self.phase == DELAY, torch.zeros_like(answer), answer)

    def advance(self, hit, done):
        old = self.phase.clone(); active = old != READY
        self.phase_counts.scatter_add_(1, old[:, None], torch.ones_like(old[:, None]))
        self.age += active.long(); self.elapsed += active.long()
        self.remaining -= active.long()
        boundary = active & (self.remaining == 0) & ~hit & ~done
        for phase, next_phase, duration in (
                (VISIBLE, TURN_A, self.duration_a),
                (TURN_A, TURN_B, self.duration_b),
                (TURN_B, PROBE, torch.full_like(old, PROBE_STEPS)),
                (PROBE, DELAY, self.duration_delay),
                (DELAY, FREE, torch.full_like(old, FREE_STEPS)),
                (FREE, READY, torch.zeros_like(old))):
            change = boundary & (old == phase)
            self.phase = torch.where(change, torch.full_like(old, next_phase), self.phase)
            self.remaining = torch.where(change, duration, self.remaining)
            self.age = torch.where(change, torch.zeros_like(old), self.age)
        self.stats['completed'] += (boundary & (old == FREE)).long()
        self.stats['interrupted_reward'] += (active & hit & ~done).long()
        self.stats['interrupted_done'] += (active & done).long()
        interrupted = hit | done
        self.phase[interrupted] = READY; self.remaining[interrupted] = 0; self.age[interrupted] = 0
        self.known = (self.known | hit) & ~done
        self.event_count[done] = 0; self.episode += done.long()
        if bool(done.any()):
            self._plans(done)

    def summary(self):
        return {**{k: v.reshape(self.groups, self.n).sum(1).cpu().tolist() for k, v in self.stats.items()},
                'phase_update_counts': self.phase_counts.reshape(self.groups, self.n, -1).sum(1).cpu().tolist()}


def brake(env, mask):
    for name in ('vf', 'vs', 'vr'):
        value = getattr(env, name)
        value[mask] = 0.


def rotate(vectors, radians):
    c, s = radians.cos(), radians.sin()
    return torch.stack((c * vectors[:, 0] - s * vectors[:, 1],
                        s * vectors[:, 0] + c * vectors[:, 1]), 1)


class EventInputs:
    """Ordinary snapshot, endpoint pulse, RGB mask, and physical brakes.

The endpoint pulse uses the actual observation returned by the fourth forward
action, before the post-probe brake; the brake changes future motor dynamics
only. No synthetic pose, angle, velocity, or goal enters the network.
"""
    def __init__(self, n, n_code, device, variants=None, n_base=None):
        self.held = torch.zeros(n, n_code, device=device)
        self.probe_start = torch.zeros(n, 2, device=device)
        self.probe_walls = torch.zeros(n, dtype=torch.long, device=device)
        self.probe_displacement = torch.zeros(n, device=device)
        self.probe_speed = torch.zeros(n, device=device)
        self.probe_heading_error = torch.zeros(n, device=device)
        self.brake_before = torch.zeros(n, 3, 3, device=device)
        self.brake_after_max = torch.zeros(n, 3, device=device)
        self.probe_seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.audit = {name:torch.zeros(n, device=device) for name in
            ('pulses', 'probe_wall_updates', 'endpoint_speed_sum', 'absolute_heading_error_rad_sum',
             'probe_displacement_sum', 'forced_goal_hits', 'free_goal_hits', 'scheduled_blind_updates')}
        if variants is None:
            self.seen = torch.zeros(n, dtype=torch.bool, device=device)
            self.ordinary = torch.zeros_like(self.seen)
            self.reset_after = torch.zeros_like(self.seen); self.reset_every = torch.zeros_like(self.seen)
            self.rotation = torch.zeros(n, device=device)
        else:
            assert len(variants) * n_base == n
            self.seen = torch.tensor([v.startswith('seen_') for v in variants], device=device).repeat_interleave(n_base)
            self.ordinary = torch.tensor([v.endswith('_ordinary') for v in variants], device=device).repeat_interleave(n_base)
            self.reset_after = torch.tensor([v.endswith('_reset_after_glimpse') for v in variants], device=device).repeat_interleave(n_base)
            self.reset_every = torch.tensor([v.endswith('_reset_every_blind') for v in variants], device=device).repeat_interleave(n_base)
            self.rotation = torch.tensor([math.pi/3 if v.endswith('_plus60') else
                                          -math.pi/3 if v.endswith('_minus60') else 0.
                                          for v in variants], device=device).repeat_interleave(n_base)

    def prepare(self, env, obs, clock, source):
        blind = clock.blind(); glimpse = clock.glimpse()
        entry = (clock.phase == TURN_A) & (clock.age == 0)
        pre_probe = (clock.phase == PROBE) & (clock.age == 0)
        new = (clock.phase == VISIBLE) & (clock.age == 0)
        self.probe_seen[new] = False
        self.probe_walls[new] = 0
        self.brake_before[new] = 0.; self.brake_after_max[new] = 0.
        self.probe_start = torch.where(pre_probe[:, None], obs['pos'], self.probe_start)
        native = source(obs['pos'], obs['dir'], obs['vel'])
        ordinary = source.pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0.)
        self.held = torch.where((entry | glimpse)[:, None], ordinary, self.held)
        current = torch.where(blind[:, None], self.held, native)
        if bool(glimpse.any()):
            self.probe_seen |= glimpse
            self.probe_displacement = torch.where(glimpse, (obs['pos']-self.probe_start).norm(dim=1), self.probe_displacement)
            self.probe_speed = torch.where(glimpse, obs['vel'].norm(dim=1), self.probe_speed)
            angle = torch.atan2(obs['dir'][:,1],obs['dir'][:,0])-obs['hd']
            error = torch.atan2(angle.sin(),angle.cos())
            self.probe_heading_error = torch.where(glimpse,error,self.probe_heading_error)
            self.audit['pulses'] += glimpse.float()
            self.audit['endpoint_speed_sum'] += glimpse*self.probe_speed
            self.audit['absolute_heading_error_rad_sum'] += glimpse*self.probe_heading_error.abs()
            self.audit['probe_displacement_sum'] += glimpse*self.probe_displacement
            direction = rotate(obs['dir'], self.rotation)
            velocity = rotate(obs['vel'], self.rotation)
            shifted = source(obs['pos'], direction, velocity)
            pulse = torch.where(self.ordinary[:, None], ordinary, shifted)
            current = torch.where(glimpse[:, None], pulse, current)
        # Keep the native RGB rendering already drawn by the environment. Calling
        # observe() again here would consume an extra noise draw and break pairing.
        before = torch.stack((env.vf,env.vs,env.vr),1)
        for slot, mask in enumerate((entry,pre_probe,glimpse)):
            self.brake_before[:,slot] = torch.where(mask[:,None],before,self.brake_before[:,slot])
        brake(env, entry | pre_probe | glimpse)
        after = torch.stack((env.vf,env.vs,env.vr),1).abs().amax(1)
        for slot, mask in enumerate((entry,pre_probe,glimpse)):
            self.brake_after_max[:,slot] = torch.where(mask,after,self.brake_after_max[:,slot])
        rgb = obs['rgb'] * (~(blind & ~self.seen))[:, None, None, None]
        return rgb, current, blind, glimpse

    def transition(self, clock, info):
        wall = (clock.phase == PROBE) & info['hit_wall']
        self.probe_walls += wall.long(); self.audit['probe_wall_updates'] += wall.float()
        self.audit['forced_goal_hits'] += (clock.forced() & info['hit']).float()
        self.audit['free_goal_hits'] += ((clock.phase == FREE) & info['hit']).float()
        self.audit['scheduled_blind_updates'] += clock.blind().float()

    def event_audit(self, index):
        return dict(pulse_delivered=bool(self.probe_seen[index]), probe_wall_updates=int(self.probe_walls[index]),
            probe_displacement_m=float(self.probe_displacement[index]) if self.probe_seen[index] else None,
            endpoint_speed_m_s=float(self.probe_speed[index]) if self.probe_seen[index] else None,
            endpoint_travel_minus_heading_deg=float(torch.rad2deg(self.probe_heading_error[index])) if self.probe_seen[index] else None,
            pre_brake_vf_vs_vr=self.brake_before[index].cpu().tolist(),
            post_brake_max_abs=self.brake_after_max[index].cpu().tolist())

    def summary(self, groups=1):
        return {k:v.reshape(groups,-1).sum(1).cpu().tolist() for k,v in self.audit.items()}


@contextmanager
def noise_hook(net, eps):
    handle = net.lstm.register_forward_hook(lambda module, inputs, state:
                                            tuple(s + e for s, e in zip(state, eps)))
    try:
        yield
    finally:
        handle.remove()


def fresh_runtime(net, cfg, device, seed=TRAIN_SEED, stagger=True):
    n = cfg['n_envs']; env = build_env(cfg, n, seed=seed*1000+1, device=device, stagger=stagger)
    return dict(env=env, obs=env.reset_all(), state=net.zero_state(n, device),
                goal=NativeGoal(n, cfg['n_cells'], device), clock=ReorientationClock(n, device),
                inputs=EventInputs(n, cfg['n_cells'], device), noise=StateNoise(device, SIGMA, NOISE_SEED),
                pa=torch.full((n,), -1, dtype=torch.long, device=device), pr=torch.zeros(n, device=device),
                action_rng=torch.Generator(device=device).manual_seed(seed+1))


def sample_actions(logits, generator, groups=1):
    uniform = torch.rand(logits.shape[0]//groups, generator=generator, device=logits.device).repeat(groups)
    return (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)


@torch.no_grad()
def collect(net, cfg, source, runtime, max_delay):
    env, clock, inputs = runtime['env'], runtime['clock'], runtime['inputs']
    obs, state, pa, pr = (runtime[k] for k in ('obs', 'state', 'pa', 'pr'))
    goal, noise = runtime['goal'], runtime['noise']
    length = cfg['rollout']; assert length == ROLLOUT
    clock.begin_segment(); state0 = tuple(x.detach().clone() for x in state)
    keys = ('rgb', 'code', 'goal', 'pa', 'pr', 'extra', 'a', 'r', 'd', 'actor_logits',
            'policy_mask', 'phase', 'recurrent_noise', 'blind')
    roll = {k: [] for k in keys}; episodes = []
    for t in range(length):
        clock.prepare(length-t, cfg['episode_len']-env.t, max_delay)
        rgb, code, blind, _ = inputs.prepare(env, obs, clock, source)
        eps = noise.draw(state, blind)
        extra = extra_from_obs(cfg, obs)
        with noise_hook(net, eps):
            logits, _, state = net.step(rgb, code, goal.saved, pa, pr, state, extra)
        action = clock.action(sample_actions(logits, runtime['action_rng']))
        for key, value in (('rgb', rgb), ('code', code), ('goal', goal.saved), ('pa', pa), ('pr', pr),
                           ('extra', extra), ('a', action), ('actor_logits', logits),
                           ('policy_mask', (~clock.forced()).float()), ('phase', clock.phase.clone()),
                           ('recurrent_noise', eps), ('blind', blind.clone())):
            roll[key].append(value)
        obs, raw, done, info = env.step(action)
        inputs.transition(clock,info)
        reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
        roll['r'].append(reward); roll['d'].append(done.float())
        hit = info['hit']
        goal.advance(hit, done, source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']) if bool(hit.any()) else None)
        clock.advance(hit, done); state = reset_state(state, done)
        pa = torch.where(done, torch.full_like(action, -1), action); pr = reward*(~done)
        if info['ep_stats'] is not None:
            episodes.extend(info['ep_stats']['ret'].cpu().tolist())
    assert not bool((clock.phase != READY).any()), 'Full event must finish within the unroll'
    # The next input is visible (possibly the start of15visible), with no brake
    # or event admission performed by this bootstrap-only calculation.
    eps = noise.draw(state, clock.blind())
    with noise_hook(net, eps):
        _, bootstrap, _ = net.step(obs['rgb'], source(obs['pos'], obs['dir'], obs['vel']),
            goal.saved, pa, pr, state, extra_from_obs(cfg, obs))
    roll['bootstrap_recurrent_noise'] = eps
    runtime.update(obs=obs, state=tuple(x.detach() for x in state), pa=pa, pr=pr)
    return roll, state0, bootstrap, episodes


def replay(net, roll, state0):
    length, n = len(roll['rgb']), roll['rgb'][0].shape[0]
    images = torch.stack(roll['rgb'])
    embeddings = net.embed(images.reshape(-1, *images.shape[2:])).reshape(length, n, -1)
    state = state0; logits, values = [], []
    for t in range(length):
        with noise_hook(net, roll['recurrent_noise'][t]):
            ll, vv, state = net.step_from_embedding(embeddings[t], roll['code'][t], roll['goal'][t],
                roll['pa'][t], roll['pr'][t], state, roll['extra'][t])
        logits.append(ll); values.append(vv); state = reset_state(state, roll['d'][t].bool())
    return torch.stack(logits), torch.stack(values)


def losses(logits, values, targets, actions, policy_mask, cfg):
    dist = torch.distributions.Categorical(logits=logits)
    policy = -(policy_mask * dist.log_prob(actions) * (targets-values).detach()).sum(0).mean()
    entropy = (policy_mask * dist.entropy()).sum(0).mean()
    value = .5 * (targets-values).square().sum(0).mean()
    # Original code sums100updates. Keep its per-update gradient convention when
    # extending the credit window to160; do not renormalize by free-action count.
    total = (100 / ROLLOUT) * (policy + cfg['value_coef']*value - cfg['entropy']*entropy)
    return total, policy, value, entropy


def update(net, cfg, opt, source, runtime, max_delay):
    roll, state0, bootstrap, episodes = collect(net, cfg, source, runtime, max_delay)
    future = bootstrap; targets = []
    for t in reversed(range(cfg['rollout'])):
        future = roll['r'][t] + cfg['gamma']*(1-roll['d'][t])*future
        targets.append(future)
    targets = torch.stack(targets[::-1]); logits, values = replay(net, roll, state0)
    error = float((logits.detach()-torch.stack(roll['actor_logits'])).abs().max())
    assert error < 3e-4, f'Stored-noise replay mismatch: {error}'
    runtime['noise'].max_replay_error = max(error, runtime['noise'].max_replay_error)
    runtime['noise'].replays += 1
    loss, _, value, entropy = losses(logits, values, targets, torch.stack(roll['a']),
                                    torch.stack(roll['policy_mask']), cfg)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite blind-reorientation loss')
    opt.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite blind-reorientation gradient')
    opt.step()
    return dict(loss=float(loss.detach()), value_loss=float(value.detach()),
                entropy=float(entropy.detach()/ROLLOUT), grad_norm=float(norm), replay_error=error,
                free_policy_fraction=float(torch.stack(roll['policy_mask']).mean())), episodes


def capture_runtime(runtime, device):
    env, clock, noise = runtime['env'], runtime['clock'], runtime['noise']
    return dict(environment={k:v.detach().cpu().clone() for k,v in vars(env).items() if torch.is_tensor(v)},
        episodes_done=env.episodes_done, environment_rng=env.gen.get_state().cpu(),
        observation={k:v.detach().cpu().clone() for k,v in runtime['obs'].items()},
        state=tuple(v.detach().cpu().clone() for v in runtime['state']),
        goal=runtime['goal'].saved.detach().cpu().clone(), pa=runtime['pa'].cpu().clone(), pr=runtime['pr'].cpu().clone(),
        input_tensors={k:v.detach().cpu().clone() for k,v in vars(runtime['inputs']).items() if torch.is_tensor(v)},
        input_audit={k:v.cpu().clone() for k,v in runtime['inputs'].audit.items()},
        clock={k:v.detach().cpu().clone() for k,v in vars(clock).items() if torch.is_tensor(v)},
        clock_stats={k:v.cpu().clone() for k,v in clock.stats.items()},
        noise_rng=noise.generator.get_state().cpu(), noise_replays=noise.replays,
        noise_max_replay_error=noise.max_replay_error, action_rng=runtime['action_rng'].get_state().cpu())


def restore_runtime(runtime, snapshot, device):
    env, clock, noise = runtime['env'], runtime['clock'], runtime['noise']
    assert set(snapshot['environment']) == {k for k,v in vars(env).items() if torch.is_tensor(v)}
    for key, value in snapshot['environment'].items():
        setattr(env, key, value.clone().to(device))
    env.episodes_done = snapshot['episodes_done']; env.gen.set_state(snapshot['environment_rng'].cpu())
    runtime['obs'] = {k:v.clone().to(device) for k,v in snapshot['observation'].items()}
    runtime['state'] = tuple(v.clone().to(device) for v in snapshot['state'])
    runtime['goal'].saved = snapshot['goal'].clone().to(device)
    for key in ('pa', 'pr'):
        runtime[key] = snapshot[key].clone().to(device)
    for key,value in snapshot['input_tensors'].items():
        setattr(runtime['inputs'],key,value.clone().to(device))
    runtime['inputs'].audit = {k:v.clone().to(device) for k,v in snapshot['input_audit'].items()}
    for key, value in snapshot['clock'].items():
        setattr(clock, key, value.clone().to('cpu' if key == 'plans' else device))
    clock.stats = {k:v.clone().to(device) for k,v in snapshot['clock_stats'].items()}
    noise.generator.set_state(snapshot['noise_rng'].cpu()); noise.replays = snapshot['noise_replays']
    noise.max_replay_error = snapshot['noise_max_replay_error']
    runtime['action_rng'].set_state(snapshot['action_rng'].cpu())


@torch.inference_mode()
def evaluate(net, cfg, *, device, seed=EVAL_SEED, episodes=64, n_envs=32,
             max_delay=MAX_DELAY, deadline=float('inf')):
    """Clean12-branch evaluation; shared event banks, action draws and physics.

The seen branches restore only RGB; they retain the same held-code/pulse
schedule and forces. Cue/reset conditions can change later reward times and
window onset, so pairing is exact up to the first behavioral divergence and by
nth event plan thereafter, rather than claiming identical later trajectories.
"""
    assert episodes % n_envs == 0
    net.eval(); n, groups = n_envs, len(VARIANTS)
    env = MatchedArena(cfg, n, groups, seed, device); obs = env.reset_all()
    source = CodeSource(cfg, build_population(cfg, device), device)
    clock = ReorientationClock(n, device, seed+2, groups=groups)
    inputs = EventInputs(n*groups, cfg['n_cells'], device, VARIANTS, n)
    goal = NativeGoal(n*groups, cfg['n_cells'], device)
    state = net.zero_state(n*groups, device)
    pa = torch.full((n*groups,), -1, dtype=torch.long, device=device); pr = torch.zeros(n*groups, device=device)
    action_rng = torch.Generator(device=device).manual_seed(seed+1)
    rows = {v:[] for v in VARIANTS}; event_rows = {v:[] for v in VARIANTS}
    window_rows = {v:[] for v in VARIANTS}
    counters = {name:torch.zeros(n*groups, device=device) for name in
        ('free_updates', 'free_hits', 'free_progress_m', 'free_path_m', 'forced_hits', 'wall_hits', 'resets')}
    event_progress = torch.zeros(n*groups, device=device); event_path = event_progress.clone()
    event_start_distance = event_progress.clone(); event_steps = torch.zeros(n*groups, dtype=torch.long, device=device)
    for step in range(cfg['episode_len']*(episodes//n)):
        if step % ROLLOUT == 0:
            require_time(deadline); clock.begin_segment()
        clock.prepare(ROLLOUT-step%ROLLOUT, cfg['episode_len']-env.t, max_delay)
        rgb, code, blind, glimpse = inputs.prepare(env, obs, clock, source)
        reset_before = blind & inputs.reset_every
        state = reset_state(state, reset_before)
        logits, _, state = net.step(rgb, code, goal.saved, pa, pr, state, extra_from_obs(cfg, obs))
        reset_after = glimpse & inputs.reset_after
        state = reset_state(state, reset_after)
        counters['resets'] += (reset_before | reset_after).float()
        action = clock.action(sample_actions(logits, action_rng, groups=groups))
        free = clock.phase == FREE; forced = clock.forced(); phase = clock.phase.clone()
        entering_free = free & (clock.age == 0)
        distance_before = (env.pos-env.goal).norm(dim=1)
        event_start_distance[entering_free] = distance_before[entering_free]
        event_progress[entering_free] = 0.; event_path[entering_free] = 0.; event_steps[entering_free] = 0
        before_pos = env.pos.clone(); before_goal = env.goal.clone()
        obs, raw, done, info = env.step(action)
        inputs.transition(clock,info)
        hit = info['hit']
        # hit_pos is the actual pre-respawn endpoint, also on episode-ending steps.
        endpoint = info['hit_pos']
        progress = distance_before-(endpoint-before_goal).norm(dim=1)
        path = (endpoint-before_pos).norm(dim=1)
        counters['free_updates'] += free.float(); counters['free_hits'] += (free & hit).float()
        counters['free_progress_m'] += free*progress; counters['free_path_m'] += free*path
        counters['forced_hits'] += (forced & hit).float(); counters['wall_hits'] += (blind & info['hit_wall']).float()
        event_progress += free*progress; event_path += free*path; event_steps += free.long()
        finish_free = free & ((clock.remaining == 1) | hit | done)
        for index in finish_free.nonzero().flatten().cpu().tolist():
            group, slot = divmod(index, n)
            event_rows[VARIANTS[group]].append(dict(slot=slot, episode=step//cfg['episode_len'],
                event=int(clock.event_count[index]), delay=int(clock.duration_delay[index]),
                free_updates=int(event_steps[index]), goal_hit=bool(hit[index]),
                completed30=bool(clock.remaining[index] == 1), episode_end=bool(done[index]),
                progress_m=float(event_progress[index]), path_m=float(event_path[index]),
                initial_goal_distance_m=float(event_start_distance[index])))
        finish_window = (phase != READY) & (((phase == FREE) & (clock.remaining == 1)) | hit | done)
        for index in finish_window.nonzero().flatten().cpu().tolist():
            group,slot = divmod(index,n)
            window_rows[VARIANTS[group]].append(dict(slot=slot,episode=step//cfg['episode_len'],
                event=int(clock.event_count[index]),turn_a=int(clock.turn_a[index]),turn_b=int(clock.turn_b[index]),
                turn_a_updates=int(clock.duration_a[index]),turn_b_updates=int(clock.duration_b[index]),
                delay_updates=int(clock.duration_delay[index]),elapsed_updates=int(clock.elapsed[index])+1,
                end_phase=PHASE_NAMES[int(phase[index])],goal_hit=bool(hit[index]),episode_end=bool(done[index]),
                **inputs.event_audit(index)))
        goal.advance(hit, done, source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel']) if bool(hit.any()) else None)
        clock.advance(hit, done); state = reset_state(state, done)
        reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
        pa = torch.where(done, torch.full_like(action, -1), action); pr = reward*(~done)
        stats = info['ep_stats']
        if stats is not None:
            for k,index in enumerate(stats['env'].cpu().tolist()):
                group,slot = divmod(index,n)
                rows[VARIANTS[group]].append(dict(slot=slot, episode=step//cfg['episode_len'],
                    score=float(stats['ret'][k]), n_goals=int(stats['n_goals'][k]),
                    first_goal_step=int(stats['first_goal_t'][k]), path_eff=float(stats['path_eff'][k]),
                    path_eff_n=int(stats['path_eff_n'][k])))
    assert not bool((clock.phase != READY).any())
    reference = [r['first_goal_step'] for r in rows[VARIANTS[0]]]
    assert len(reference) == episodes
    assert all([r['first_goal_step'] for r in rr] == reference for rr in rows.values())
    if not bool(source.pop.is_pgc.any()):
        for vision in ('blind', 'seen'):
            for pulse in ('ordinary', 'plus60', 'minus60'):
                assert rows[f'{vision}_native'] == rows[f'{vision}_{pulse}']
                assert event_rows[f'{vision}_native'] == event_rows[f'{vision}_{pulse}']
    sums = {k:v.reshape(groups,n).sum(1).cpu().tolist() for k,v in counters.items()}
    return dict(seed=seed, episodes_per_variant=episodes, clean_no_noise=True, max_delay=max_delay,
        variants={v:episode_summary(rr,cfg['episode_len']) for v,rr in rows.items()},
        free_window={v:dict(rows=event_rows[v], **{k:values[j] for k,values in sums.items()}) for j,v in enumerate(VARIANTS)},
        schedule=clock.summary(), all_first_discoveries_identical=True,
        probe_audit=inputs.summary(groups),window_rows=window_rows,
        grid_pulse_controls_exact_noop=not bool(source.pop.is_pgc.any()),
        pairing='Shared per-slot episode/event plans, action uniforms, physics noise and layouts; post-intervention reward times can shift later admission times',
        reset_interpretation='Only h/c carryover removed; previous action/reward, saved goal, held spatial snapshot and physical history remain')
