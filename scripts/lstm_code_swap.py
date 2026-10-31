"""Causal code swaps in frozen 1B LSTMs: behavior and matched-state dynamics.

python scripts/lstm_code_swap.py --device mps --episodes 128 --n-envs 64

The PGC policy receives four current/goal code combinations. The grid policy
receives its own code and the reciprocal mixed-code substitution. All variants
share episode layouts, per-step environmental noise and action-sampling uniforms.
Within the intact PGC rollout, state forks isolate direct gate effects, sustained
recurrent effects with forced identical experience, and single-step pulse decay.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from code.common import CodeSource, build_policy, build_population
from code.env import ArenaEnv

PGC_VARIANTS = ('intact', 'current_grid', 'goal_grid', 'both_grid')
GRID_VARIANTS = ('intact', 'both_mixed')
HORIZONS = (0, 1, 5, 15, 30, 60)


class MatchedArena(ArenaEnv):
    """Identical random environments for different policies in the same batch."""
    def __init__(self, cfg, n_envs, groups, seed, device):
        self.group_size, self.groups = n_envs, groups
        super().__init__(n_envs * groups, half=cfg['arena_half'],
                         episode_len=cfg['episode_len'], action_repeat=cfg['action_repeat'],
                         v_max=cfg['v_max'], rot_max=cfg['rot_max'], goal_radius=cfg['goal_radius'],
                         vision=cfg['vision'], img_noise=cfg['img_noise'],
                         randomize_textures=bool(cfg['randomize_textures']),
                         stagger=False, paired=True, pair_seed=seed, seed=seed, device=device)

    def _paired_rngs(self, idx):
        # Preserve the canonical layout seed formula; pair by within-group slot.
        index = idx.cpu().tolist()
        episodes = self.ep_idx[idx].cpu().tolist()
        draws = self.draw_idx[idx].cpu().tolist()
        rngs = [np.random.default_rng((self.pair_seed * 1_000_003 +
                                     (i % self.group_size) * 7_919 + ep * 104_729 + draw * 13) & 0x7FFFFFFF)
                for i, ep, draw in zip(index, episodes, draws)]
        self.draw_idx[idx] += 1
        return rngs

    def _randn(self, *shape):
        if shape[0] == self.n:
            base = torch.randn(self.group_size, *shape[1:], generator=self.gen, device=self.device)
            return base.repeat(self.groups, *([1] * (len(shape) - 1)))
        return super()._randn(*shape)


def gates_and_state(net, x, state):
    h, c = state
    pre = F.linear(x, net.lstm.weight_ih, net.lstm.bias_ih) + F.linear(h, net.lstm.weight_hh, net.lstm.bias_hh)
    i, f, g, o = pre.chunk(4, dim=1)
    gates = (i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid())
    return gates, state_from_gates(gates, c)


def state_from_gates(gates, previous_c):
    i, f, g, o = gates
    c = f * previous_c + i * g
    return o * c.tanh(), c


def response_metrics(net, reference, altered):
    h, c = reference
    ah, ac = altered
    p = net.pi(h).softmax(1)
    q = net.pi(ah).softmax(1)
    return dict(h_rms=(ah - h).square().mean(1).sqrt(),
                c_rms=(ac - c).square().mean(1).sqrt(),
                action_tv=(p - q).abs().sum(1) / 2,
                preferred_action_changed=(p.argmax(1) != q.argmax(1)).float(),
                value_abs=(net.v(ah) - net.v(h)).abs().squeeze(1))


class Accumulator:
    def __init__(self):
        self.rows = {}

    def add(self, name, values, mask):
        if not bool(mask.any()):
            return
        data = {k: v[mask].detach().cpu().numpy() for k, v in values.items()}
        row = self.rows.setdefault(name, dict(n=0, sums={}))
        row['n'] += int(mask.sum())
        for k, value in data.items():
            row['sums'][k] = row['sums'].get(k, 0.) + float(value.astype(float).sum())

    def result(self):
        return {k: dict(n=v['n'], **{m: total / v['n'] for m, total in v['sums'].items()})
                for k, v in self.rows.items()}


def verify_codes(seed):
    """Compare to an independently constructed grid-only population, not a mask."""
    configs = {c: json.loads((ROOT / f'models/{c}/{c}_lstm_s{seed}_1000M.json').read_text())
               for c in ('gc100', 'pgc33')}
    pops = {c: build_population(configs[c], 'cpu') for c in configs}
    generator = torch.Generator().manual_seed(9000 + seed)
    pos = torch.rand(512, 2, generator=generator) * 2.34 - 1.17
    angle = torch.rand(512, generator=generator) * (2 * np.pi)
    direction = torch.stack((angle.cos(), angle.sin()), 1)
    velocity = direction * .35
    grid = pops['gc100'].activity(pos, direction, velocity)
    substituted = pops['pgc33'].activity(pos, direction, velocity, shift_scale=0)
    old_probe = CodeSource(configs['pgc33'], pops['pgc33'], 'cpu', probe='pgc_as_gc')
    assert torch.equal(grid, substituted)
    assert torch.equal(grid, old_probe(pos, direction, velocity))
    assert torch.equal(grid, old_probe.goal(pos, direction, velocity))
    assert torch.equal(pops['gc100'].phi, pops['pgc33'].phi)
    assert torch.equal(pops['gc100'].K, pops['pgc33'].K)
    return dict(samples=512, max_absolute_code_difference=0.,
                same_phases_and_wavevectors=True, existing_pgc_as_gc_is_exact_substitution=True)


def summarize_episodes(rows, episode_len):
    found = np.array([r['first_goal_t'] >= 0 for r in rows])
    first = np.array([r['first_goal_t'] if r['first_goal_t'] >= 0 else episode_len for r in rows])
    eff = [r['path_eff'] for r in rows if r['path_eff_n'] > 0]
    post = [(r['n_goals'] - 1) * 1000 / max(episode_len - r['first_goal_t'], 1)
            for r in rows if r['first_goal_t'] >= 0]
    return dict(episodes=len(rows), mean_score=float(np.mean([r['ret'] for r in rows])),
                mean_first_goal_step=float(first.mean()), frac_found_goal=float(found.mean()),
                mean_path_eff=float(np.mean(eff)) if eff else None,
                post_discovery_goal_rate=float(np.mean(post)) if post else None,
                frac_two_goals=float(np.mean([r['n_goals'] >= 2 for r in rows])),
                n_path_eff=len(eff), rows=rows)


@torch.inference_mode()
def run(seed, condition, args):
    start = time.perf_counter()
    cfg_path = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
    cfg = json.loads(cfg_path.read_text())
    checkpoint = cfg_path.with_suffix('.pt')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device == 'mps':
        torch.mps.manual_seed(args.eval_seed)
    net = build_policy(cfg, args.device).eval()
    net.load_state_dict(ckpt['policy'], strict=True)
    net.requires_grad_(False)
    mixed_cfg = dict(cfg, pgc_frac=.33, cond='pgc33')
    pop = build_population(mixed_cfg, args.device)
    variants = PGC_VARIANTS if condition == 'pgc33' else GRID_VARIANTS
    groups, n = len(variants), args.n_envs
    total = groups * n
    env = MatchedArena(cfg, n, groups, args.eval_seed, args.device)
    obs = env.reset_all()
    # All intervention branches start from exactly the same sensory conditions.
    for key in ('pos', 'hd', 'dir', 'rgb'):
        assert all(torch.equal(obs[key][:n], obs[key][j*n:(j+1)*n]) for j in range(1, groups))
    current_grid = torch.tensor([v in ('current_grid', 'both_grid') or (condition == 'gc100' and v == 'intact')
                                 for v in variants], device=args.device).repeat_interleave(n)
    goal_grid = torch.tensor([v in ('goal_grid', 'both_grid') or (condition == 'gc100' and v == 'intact')
                              for v in variants], device=args.device).repeat_interleave(n)
    state = net.zero_state(total, args.device)
    mixed_goal = torch.zeros(total, 512, device=args.device)
    grid_goal = torch.zeros_like(mixed_goal)
    previous_action = torch.full((total,), -1, dtype=torch.long, device=args.device)
    previous_reward = torch.zeros(total, device=args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.eval_seed + 1)
    rows = {v: [] for v in variants}
    accum = Accumulator()
    # Shadow states use intact observations/actions/rewards, but altered code.
    shadows = {v: net.zero_state(n, args.device) for v in PGC_VARIANTS}
    pulses = None
    pulse_age = None
    pulse_mask = None
    max_cell_error = 0.
    iteration = 0
    last_progress = time.perf_counter()
    while len(rows[variants[0]]) < args.episodes:
        mixed_code = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        grid_code = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0)
        code = torch.where(current_grid[:, None], grid_code, mixed_code)
        goal = torch.where(goal_grid[:, None], grid_goal, mixed_goal)
        embedding = net.embed(obs['rgb'])
        previous_state = state
        logits, values, state = net.step_from_embedding(embedding, code, goal, previous_action, previous_reward, state)
        if condition == 'pgc33':
            known = env.n_goals[:n] > 0
            episode_step = iteration % cfg['episode_len']
            reference = (state[0][:n], state[1][:n])
            old = (previous_state[0][:n], previous_state[1][:n])
            e, pa, pr = embedding[:n], previous_action[:n], previous_reward[:n]
            combos = [(mixed_code[:n], mixed_goal[:n]), (grid_code[:n], mixed_goal[:n]),
                      (mixed_code[:n], grid_goal[:n]), (grid_code[:n], grid_goal[:n])]
            xs = [net._concat(e, c, g, pa, pr, None) for c, g in combos]
            # Independent recurrent branches, always driven by intact experience.
            for j, name in enumerate(PGC_VARIANTS):
                shadows[name] = net.lstm(xs[j], shadows[name])
            if episode_step >= 40 and episode_step % 20 == 0:
                original_gates, reconstructed = gates_and_state(net, xs[0], old)
                error = max(float((a - b).abs().max()) for a, b in zip(reference, reconstructed))
                max_cell_error = max(max_cell_error, error)
                assert error < 2e-4, f'Gate reconstruction mismatch: {error}'
                accum.add('control/sustained_no_op', response_metrics(net, reference, shadows['intact']), known)
                for j, name in enumerate(PGC_VARIANTS[1:], 1):
                    changed_gates, one_step = gates_and_state(net, xs[j], old)
                    metrics = response_metrics(net, reconstructed, one_step)
                    for gate_i, gate_name in enumerate(('input', 'forget', 'candidate', 'output')):
                        metrics[f'{gate_name}_mean_abs_change'] = (changed_gates[gate_i] - original_gates[gate_i]).abs().mean(1)
                        restored = list(changed_gates)
                        restored[gate_i] = original_gates[gate_i]
                        restored_state = state_from_gates(restored, old[1])
                        restored_metrics = response_metrics(net, reconstructed, restored_state)
                        metrics[f'restore_{gate_name}_action_tv'] = restored_metrics['action_tv']
                        metrics[f'restore_{gate_name}_h_rms'] = restored_metrics['h_rms']
                    accum.add(f'immediate/{name}', metrics, known)
                    accum.add(f'sustained/{name}', response_metrics(net, shadows['intact'], shadows[name]), known)
                    if name == 'goal_grid' and bool((~known).any()):
                        assert float(metrics['action_tv'][~known].max()) < 1e-6
            # Inject a single altered-code step every 100 steps, then restore
            # intact input and track how long state/output changes remain.
            if (episode_step >= 100 and episode_step % 100 == 0
                    and episode_step + max(HORIZONS) < cfg['episode_len']):
                pulses = {name: net.lstm(xs[j], old) for j, name in enumerate(PGC_VARIANTS)}
                pulse_age, pulse_mask = 0, known.clone()
            elif pulses is not None:
                pulses = {name: net.lstm(xs[0], pstate) for name, pstate in pulses.items()}
                pulse_age += 1
            if pulses is not None and pulse_age in HORIZONS:
                for name, pstate in pulses.items():
                    if name == 'intact':
                        accum.add(f'control/pulse_no_op/{pulse_age}', response_metrics(net, reference, pstate), pulse_mask)
                    else:
                        accum.add(f'pulse/{name}/{pulse_age}', response_metrics(net, pulses['intact'], pstate), pulse_mask)
            if pulse_age is not None and pulse_age >= max(HORIZONS):
                pulses = None

        # Common random numbers: same uniforms for all branches. Marginally
        # each action is sampled from that branch's stochastic policy.
        uniforms = torch.rand(n, generator=generator, device=args.device).repeat(groups)
        actions = (uniforms[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        obs, raw_reward, done, info = env.step(actions)
        hit = info['hit']
        if bool(hit.any()):
            hit_mixed = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'])
            hit_grid = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0)
            mixed_goal = torch.where(hit[:, None], hit_mixed, mixed_goal)
            grid_goal = torch.where(hit[:, None], hit_grid, grid_goal)
        previous_action = actions
        previous_reward = raw_reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            keep = (~done).float()
            state = (state[0] * keep[:, None], state[1] * keep[:, None])
            mixed_goal *= keep[:, None]
            grid_goal *= keep[:, None]
            previous_action = torch.where(done, torch.full_like(actions, -1), actions)
            previous_reward *= keep
            if condition == 'pgc33':
                for name in shadows:
                    shadows[name] = tuple(z * keep[:n, None] for z in shadows[name])
                pulses, pulse_age = None, None
            es = {k: v.cpu().tolist() for k, v in info['ep_stats'].items()}
            episode_index = iteration // cfg['episode_len']
            for j, slot in enumerate(es['env']):
                rows[variants[slot // n]].append(dict(slot=slot % n, episode=episode_index,
                    **{k: es[k][j] for k in ('ret', 'n_goals', 'first_goal_t', 'path_eff', 'path_eff_n')}))
        iteration += 1
        if time.perf_counter() - last_progress > 25:
            print(f'{condition} s{seed}: {iteration}/{cfg["episode_len"] * (args.episodes // n)} steps; '
                  f'{time.perf_counter()-start:.0f}s', flush=True)
            last_progress = time.perf_counter()

    if condition == 'pgc33':
        # This branch changes nothing before discovering the first goal.
        first_intact = [r['first_goal_t'] for r in rows['intact']]
        first_goal_swap = [r['first_goal_t'] for r in rows['goal_grid']]
        assert first_intact == first_goal_swap, 'Goal-only swap altered pre-discovery behavior'
    behavioral = {name: summarize_episodes(values[:args.episodes], cfg['episode_len']) for name, values in rows.items()}
    result = dict(condition=condition, seed=seed, checkpoint_frames=ckpt['frames'],
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  eval_seed=args.eval_seed, episodes=args.episodes, n_envs=n, device=args.device,
                  variants=behavioral, rnn_effects=accum.result(),
                  gate_reconstruction_max_error=max_cell_error,
                  matched_no_op_control=True,
                  elapsed_seconds=time.perf_counter() - start)
    path = args.out / f'{condition}_s{seed}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'DONE {condition} s{seed}: ' + ', '.join(f'{v}={r["mean_score"]:.2f}' for v, r in behavioral.items()) +
          f'; {result["elapsed_seconds"]:.1f}s', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--n-envs', type=int, default=64)
    ap.add_argument('--eval-seed', type=int, default=737373)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 9)))
    ap.add_argument('--conditions', nargs='+', default=['pgc33', 'gc100'])
    ap.add_argument('--out', type=Path, default=ROOT / 'results/lstm_code_swap_1B')
    args = ap.parse_args()
    assert args.episodes % args.n_envs == 0
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    checks = {seed: verify_codes(seed) for seed in args.seeds}
    (args.out / 'protocol.json').write_text(json.dumps(dict(
        eval_seed=args.eval_seed, episodes_per_seed_variant=args.episodes, n_envs_per_variant=args.n_envs,
        device=args.device, seeds=args.seeds, code_identity_checks=checks,
        variants=dict(pgc33=list(PGC_VARIANTS), gc100=list(GRID_VARIANTS)),
        common_randomness='Layouts keyed by base slot and episode/draw; motor/image noise and action uniforms repeated across variants.',
        rnn_protocol='Intact PGC policy trajectories; same sensory input, previous action/reward, and goal-hit history. Immediate forks share exact preceding h/c; sustained forks retain their altered recurrent states; pulse forks receive one altered step then intact inputs. Sustained and pulse effects are compared against same-batch-size no-op forks; their tiny numerical differences from the original rollout are also reported.',
        rnn_sampling='Every 20 steps after step 40, after first goal. Pulses every 100 steps; horizons 0,1,5,15,30,60.',
        caveats=['Evaluation-only substitutions; input distribution changes relative to training.',
                 'RNN effects compare variants of the same model, not unaligned neurons across separately trained models.',
                 'Gate restorations are local causal interventions; their contributions need not sum because gates interact.',
                 'Off-policy shadow and pulse paths force intact experience; they isolate recurrence rather than predict independent behavior.']), indent=2) + '\n')
    for seed in args.seeds:
        for condition in args.conditions:
            path = args.out / f'{condition}_s{seed}.json'
            if path.exists():
                previous = json.loads(path.read_text())
                assert (previous['episodes'], previous['eval_seed'], previous['n_envs']) == (args.episodes, args.eval_seed, args.n_envs)
                if condition == 'gc100' or previous.get('matched_no_op_control'):
                    print(f'Using completed {condition} s{seed}', flush=True)
                    continue
            run(seed, condition, args)


if __name__ == '__main__':
    main()
