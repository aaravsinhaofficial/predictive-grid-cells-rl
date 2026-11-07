"""Matched behavioral controls for the native joint-blackout task.

This evaluates frozen policies; it does not train a decoder or infer a ring.
All branches share layouts, noise and action uniforms. Only post-discovery
inputs and explicitly specified recurrent-state resets differ.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from code import train_memory as training
import numpy as np
import torch
from code.common import CodeSource, build_policy, build_population
from code.memory_task import MemoryClock, gate_observation
from lstm_code_swap import MatchedArena, summarize_episodes

VARIANTS = ('blind', 'blind_reset_onset', 'blind_reset_each',
            'seen', 'seen_reset_onset', 'seen_reset_each',
            'blind_current_grid', 'seen_current_grid')
COUNTERS = ('scheduled_steps', 'masked_steps', 'known_steps', 'resets',
            'blind_goals', 'seen_known_goals', 'onsets')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervention_masks(scheduled, onset, known, n, variants=VARIANTS):
    """Masks are evaluated before the next policy observation, without RNG."""
    if len(scheduled) != n * len(variants):
        raise ValueError('Branch/slot count mismatch')
    dev = scheduled.device
    branch = lambda test: torch.tensor([test(v) for v in variants], device=dev).repeat_interleave(n)
    blind = scheduled & branch(lambda v: v.startswith('blind'))
    reset = ((onset & branch(lambda v: v.endswith('_reset_onset'))) |
             (scheduled & branch(lambda v: v.endswith('_reset_each'))))
    swap = known & branch(lambda v: v.endswith('_current_grid'))
    return blind, reset, swap


@torch.inference_mode()
def evaluate(policy, cfg, *, device='cpu', seed=741021, episodes=128,
             n_envs=32, max_gap=60, deadline=float('inf'), env_class=MatchedArena):
    if episodes < 1 or n_envs < 1 or episodes % n_envs:
        raise ValueError('Use a positive whole number of episode batches')
    if cfg.get('extra_input', 'none') != 'none' or cfg.get('code_noise', 0) != 0:
        raise ValueError('This fixed comparison requires no extra input or code noise')
    if cfg['policy'] != 'lstm' or not policy.recurrent:
        raise ValueError('This experiment evaluates LSTM policies')
    if max_gap < 1:
        raise ValueError('Use the same positive test blackout length for every training arm')
    policy.eval()
    n, groups = n_envs, len(VARIANTS)
    env = env_class(cfg, n, groups, seed, device)
    obs = env.reset_all()
    for key in ('pos', 'hd', 'dir', 'rgb'):
        if not all(torch.equal(obs[key][:n], obs[key][j*n:(j+1)*n]) for j in range(1, groups)):
            raise AssertionError('Initial branch observations differ: ' + key)
    pop = build_population(cfg, device)
    source = CodeSource(cfg, pop, device)
    clock = MemoryClock(n, device, seed + 2, groups=groups,
                        visible_steps=cfg.get('task_visible_steps', 15))
    sampler = torch.Generator(device=device).manual_seed(seed + 1)
    total = n * groups
    state = policy.zero_state(total, device)
    goal = torch.zeros(total, cfg['n_cells'], device=device)
    pa = torch.full((total,), -1, dtype=torch.long, device=device)
    pr = torch.zeros(total, device=device)
    onset = torch.zeros(total, dtype=torch.bool, device=device)
    counts = torch.zeros(total, len(COUNTERS), dtype=torch.long, device=device)
    action_counts = torch.zeros(total, 6, dtype=torch.long, device=device)
    rows = {v: [] for v in VARIANTS}
    steps = cfg['episode_len'] * (episodes // n)
    started = time.monotonic()
    for step in range(steps):
        if step % 100 == 0 and time.monotonic() >= deadline:
            raise TimeoutError('Evaluation deadline reached; no completion claim')
        scheduled, known = clock.mask(), clock.known.clone()
        blind, reset, swap = intervention_masks(scheduled, onset, known, n)
        current = source(obs['pos'], obs['dir'], obs['vel'])
        ordinary = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0.)
        current = torch.where(swap[:, None], ordinary, current)
        rgb, current = gate_observation(obs['rgb'], current, blind)
        keep = (~reset)[:, None]
        state = tuple(s * keep for s in state)
        logits, _, state = policy.step(rgb, current, goal, pa, pr, state)
        uniform = torch.rand(n, generator=sampler, device=device).repeat(groups)
        action = (uniform[:, None] > logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(5)
        obs, raw, done, info = env.step(action)
        hit = info['hit']
        increments = (scheduled, blind, known, reset, hit & blind & known,
                      hit & ~blind & known, onset)
        counts += torch.stack(increments, 1).long()
        action_counts.scatter_add_(1, action[:, None], (scheduled & known).long()[:, None])
        if bool(hit.any()):
            native_goal = source.goal(info['hit_pos'], info['hit_dir'], info['hit_vel'])
            goal = torch.where(hit[:, None], native_goal, goal)
        onset = clock.advance(hit, done, max_gap)
        reward = raw.clamp(-cfg['reward_clip'], cfg['reward_clip']) if cfg['reward_clip'] > 0 else raw
        pa = torch.where(done, torch.full_like(action, -1), action)
        pr = reward * (~done)
        goal = goal * (~done)[:, None]
        state = tuple(s * (~done)[:, None] for s in state)
        stats = info['ep_stats']
        if stats is not None:
            indexes = stats['env'].cpu().tolist()
            count_rows = counts[stats['env']].cpu().tolist()
            action_rows = action_counts[stats['env']].cpu().tolist()
            for j, index in enumerate(indexes):
                group, slot = divmod(index, n)
                row = dict(slot=slot, episode=step // cfg['episode_len'],
                           ret=float(stats['ret'][j]), n_goals=int(stats['n_goals'][j]),
                           first_goal_t=int(stats['first_goal_t'][j]),
                           path_eff=float(stats['path_eff'][j]), path_eff_n=int(stats['path_eff_n'][j]),
                           **dict(zip(COUNTERS, count_rows[j])),
                           scheduled_action_counts=action_rows[j])
                rows[VARIANTS[group]].append(row)
        counts *= (~done)[:, None]
        action_counts *= (~done)[:, None]
    assert all(len(rr) == episodes for rr in rows.values())
    first = [r['first_goal_t'] for r in rows['blind']]
    assert all([r['first_goal_t'] for r in rr] == first for rr in rows.values())
    if not pop.n_pgc:
        assert rows['blind'] == rows['blind_current_grid']
        assert rows['seen'] == rows['seen_current_grid']
    summaries = {}
    for name, rr in rows.items():
        summary = summarize_episodes(rr, cfg['episode_len'])
        summary['totals'] = {key: sum(row[key] for row in rr) for key in COUNTERS}
        summary['scheduled_action_counts'] = np.asarray([r['scheduled_action_counts'] for r in rr]).sum(0).tolist()
        summary['blind_goal_rate_per_1000_steps'] = (
            1000 * summary['totals']['blind_goals'] / summary['totals']['masked_steps']
            if summary['totals']['masked_steps'] else None)
        summaries[name] = summary
    score = {name: value['mean_score'] for name, value in summaries.items()}
    contrasts = {}
    for which in ('onset', 'each'):
        blind_cost = score['blind'] - score['blind_reset_' + which]
        seen_cost = score['seen'] - score['seen_reset_' + which]
        contrasts[which] = dict(blind_reset_cost=blind_cost, seen_reset_cost=seen_cost,
                               additional_reset_cost_in_blackout=blind_cost-seen_cost)
    return dict(complete=True, seed=seed, episodes_per_variant=episodes, n_envs=n,
                max_gap=max_gap, min_gap=(max_gap+1)//2, visible_steps=clock.visible_steps,
                seconds=time.monotonic()-started, frames=steps*total*cfg['action_repeat'],
                variants=summaries, reset_contrasts=contrasts,
                current_predictive_input_benefit=dict(
                    blind=score['blind']-score['blind_current_grid'],
                    seen=score['seen']-score['seen_current_grid']),
                all_first_discoveries_identical=True, grid_swap_is_exact_sham=not bool(pop.n_pgc),
                scope='Repeated episodes within one fixed model; analyze training seeds as independent units. '
                      'Erasure removes broad recurrent history. Reward-triggered visibility may diverge '
                      'between branches; actual exposure and reset counts are retained. '
                      'Goal snapshots remain native, including in current-code replacement controls.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='final.pt')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--n-envs', type=int, default=32)
    parser.add_argument('--max-gap', type=int, default=60)
    parser.add_argument('--seed', type=int, default=741021)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=float, default=3600)
    args = parser.parse_args(argv)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; refusing a silent CPU fallback')
    cfg_path = args.run / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    checkpoint = args.run / args.checkpoint
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    training.configure(dict(cfg, device=args.device, threads=2))
    policy = build_policy(cfg, args.device)
    policy.load_state_dict(saved['policy'], strict=True)
    result = evaluate(policy, cfg, device=args.device, seed=args.seed, episodes=args.episodes,
                      n_envs=args.n_envs, max_gap=args.max_gap,
                      deadline=time.monotonic()+args.max_seconds)
    result.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  checkpoint_sha256=sha(checkpoint), config_sha256=sha(cfg_path),
                  checkpoint_frames=saved.get('frames', saved.get('frames_done')),
                  condition=cfg['cond'], training_seed=cfg['seed'],
                  training_task_max_gap=cfg['task_max_gap'],
                  evaluation_numerics=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), cudnn_tf32=bool(cfg['tf32']),
                      matmul_tf32=False, deterministic_algorithms=True,
                      device=args.device,
                      gpu_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None),
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 [Path(__file__), ROOT/'code/memory_task.py', ROOT/'scripts/lstm_code_swap.py',
                                  ROOT/'code/common.py', ROOT/'code/env.py', ROOT/'code/nets.py', ROOT/'code/codes.py',
                                  ROOT/'code/train_memory.py']})
    out = args.out or args.run / f'memory_eval_gap{args.max_gap}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix+'.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(out)
    print(json.dumps({k:v for k,v in result.items() if k != 'variants'}), flush=True)


if __name__ == '__main__':
    main()
