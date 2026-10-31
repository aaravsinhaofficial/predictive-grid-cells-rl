"""Locate when predictive goal/current input helps frozen PGC navigation.

Only remove the predictive shift after first discovery, in a prespecified
spatial or post-respawn phase. True goal distance is an oracle diagnostic.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from code.common import build_population, build_policy
from lstm_code_swap import MatchedArena, summarize_episodes, verify_codes

VARIANTS = ('intact', 'goal_grid', 'goal_near', 'goal_far', 'goal_early',
            'current_grid_known', 'current_near', 'current_far')
PRIMARY = [('goal_near', 'goal_grid'), ('goal_far', 'goal_grid'),
           ('goal_near', 'goal_far'), ('current_near', 'current_far')]


@torch.inference_mode()
def run(seed, args):
    started = time.perf_counter()
    path = ROOT / f'models/pgc33/pgc33_lstm_s{seed}_1000M.json'
    cfg = json.loads(path.read_text())
    ckpath = path.with_suffix('.pt')
    ckhash = hashlib.sha256(ckpath.read_bytes()).hexdigest()
    ckpt = torch.load(ckpath, map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device == 'mps': torch.mps.manual_seed(args.eval_seed)
    net = build_policy(cfg, args.device).eval().requires_grad_(False)
    net.load_state_dict(ckpt['policy'])
    pop = build_population(cfg, args.device)
    n, groups = args.n_envs, len(VARIANTS)
    total = n * groups
    env = MatchedArena(cfg, n, groups, args.eval_seed, args.device)
    obs = env.reset_all()
    state = net.zero_state(total, args.device)
    mixed_goal = torch.zeros(total, 512, device=args.device)
    grid_goal = torch.zeros_like(mixed_goal)
    pa = torch.full((total,), -1, dtype=torch.long, device=args.device)
    pr = torch.zeros(total, device=args.device)
    age = torch.zeros(total, dtype=torch.long, device=args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.eval_seed + 1)
    rows = {v: [] for v in VARIANTS}
    counter_keys = ('known_steps', 'near_steps', 'goal_changed_steps',
                    'current_changed_steps', 'goal_mask_switches')
    count_tensor = torch.zeros(groups, len(counter_keys), dtype=torch.long, device=args.device)
    old_goal_mask = torch.zeros(total, dtype=torch.bool, device=args.device)
    step, last = 0, started
    while len(rows['intact']) < args.episodes:
        known = env.n_goals > 0
        near = (obs['pos'] - env.goal).norm(dim=1) <= args.near_radius
        early = age < args.early_steps
        goal_mask, current_mask = torch.zeros_like(known), torch.zeros_like(known)
        for j, variant in enumerate(VARIANTS):
            sl = slice(j*n, (j+1)*n)
            if variant.startswith('goal_'):
                phase = {'grid': torch.ones_like(known[sl]), 'near': near[sl],
                         'far': ~near[sl], 'early': early[sl]}[variant[5:]]
                goal_mask[sl] = known[sl] & phase
            elif variant.startswith('current_'):
                phase = {'grid_known': torch.ones_like(known[sl]),
                         'near': near[sl], 'far': ~near[sl]}[variant[8:]]
                current_mask[sl] = known[sl] & phase
        mixed = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        grid = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0)
        code = torch.where(current_mask[:, None], grid, mixed)
        goal = torch.where(goal_mask[:, None], grid_goal, mixed_goal)
        logits, _, state = net.step(obs['rgb'], code, goal, pa, pr, state)
        uniform = torch.rand(n, generator=gen, device=args.device).repeat(groups)
        action = (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        # Counters are descriptive exposure on each intervention's own path.
        count_tensor += torch.stack((known, known & near, goal_mask, current_mask,
                                     known & (goal_mask != old_goal_mask)),1).reshape(groups,n,-1).sum(1)
        old_goal_mask = goal_mask
        obs, reward, done, info = env.step(action)
        hit = info['hit']
        if bool(hit.any()):
            hm = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'])
            hg = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0)
            mixed_goal = torch.where(hit[:, None], hm, mixed_goal)
            grid_goal = torch.where(hit[:, None], hg, grid_goal)
        age = torch.where(hit | done, torch.zeros_like(age), age+1)
        pa, pr = action, reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            keep = (~done).float()
            state = tuple(s * keep[:, None] for s in state)
            mixed_goal *= keep[:, None]
            grid_goal *= keep[:, None]
            pa = torch.where(done, torch.full_like(pa, -1), pa)
            pr *= keep
            old_goal_mask &= ~done
            es = {k: v.cpu().tolist() for k, v in info['ep_stats'].items()}
            for i, slot in enumerate(es['env']):
                rows[VARIANTS[slot//n]].append(dict(slot=slot%n, episode=step//cfg['episode_len'],
                    **{k: es[k][i] for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')}))
        step += 1
        if time.perf_counter()-last > 25:
            print(f'phase s{seed}: {step}/{cfg["episode_len"]*(args.episodes//n)}; '
                  f'{time.perf_counter()-started:.0f}s', flush=True)
            last = time.perf_counter()
    first = [r['first_goal_t'] for r in rows['intact']]
    assert all([r['first_goal_t'] for r in records] == first for records in rows.values())
    checks = dict(first_goal_times_identical=True, native_baseline_exact=None, goal_grid_baseline_exact=None)
    if (args.episodes, args.n_envs, args.eval_seed)==(128,64,737373):
        old=json.loads((ROOT/f'results/lstm_code_swap_1B/pgc33_s{seed}.json').read_text())
        for v,k in [('intact','native_baseline_exact'),('goal_grid','goal_grid_baseline_exact')]:
            assert rows[v] == old['variants'][v]['rows'], (seed,v,'baseline mismatch')
            checks[k] = True
    assert hashlib.sha256(ckpath.read_bytes()).hexdigest() == ckhash
    counter_values = count_tensor.cpu().tolist()
    counters = {v:dict(zip(counter_keys,counter_values[j])) for j,v in enumerate(VARIANTS)}
    result=dict(seed=seed, condition='pgc33', checkpoint_sha256=ckhash,
        episodes=args.episodes,n_envs=n,eval_seed=args.eval_seed,near_radius=args.near_radius,
        early_steps=args.early_steps,device=args.device,checks=checks,exposure=counters,
        variants={v:summarize_episodes(records,cfg['episode_len']) for v,records in rows.items()},
        elapsed_seconds=time.perf_counter()-started)
    (args.out/f'pgc33_s{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'DONE phase s{seed}: '+', '.join(f'{v}={d["mean_score"]:.2f}' for v,d in result['variants'].items()),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='mps')
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--n-envs',type=int,default=64)
    p.add_argument('--eval-seed',type=int,default=737373)
    p.add_argument('--near-radius',type=float,default=.6)
    p.add_argument('--early-steps',type=int,default=15)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(1,9)))
    p.add_argument('--out',type=Path,default=ROOT/'results/lstm_phase_intervention_1B')
    a=p.parse_args()
    assert a.episodes%a.n_envs==0
    torch.set_num_threads(2)
    a.out.mkdir(exist_ok=True,parents=True)
    protocol=dict(seeds=a.seeds,episodes=a.episodes,n_envs=a.n_envs,eval_seed=a.eval_seed,
        near_radius=a.near_radius,early_steps=a.early_steps,variants=VARIANTS,
        primary_score_comparisons=[dict(reference=x,intervention=y) for x,y in PRIMARY],
        test_family='Four paired training-seed score contrasts, Holm separately for t and exhaustive sign flips. Other contrasts exploratory.',
        design='Frozen PGC networks; substitute ordinary same-phase grid activity only in specified slots/times. All interventions begin after first discovery. Near means true-center distance <=0.6m, twice reward radius. Early means first15decisions after each pickup/respawn. Goal and current deshifting separate. No state patching or retraining.',
        interpretation='Causal stage-conditional input intervention, not mediation percentage. Masks depend on each branch trajectory; recurrence carries history across boundaries. Distance mask uses privileged true center. Input switching can itself perturb computations.',
        code_identity={s:verify_codes(s) for s in a.seeds})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for s in a.seeds:
        dest=a.out/f'pgc33_s{s}.json'
        if dest.exists():
            old=json.loads(dest.read_text())
            assert (old['episodes'],old['n_envs'],old['eval_seed'],old['near_radius'],old['early_steps'])==(a.episodes,a.n_envs,a.eval_seed,a.near_radius,a.early_steps)
            print(f'Using completed phase s{s}',flush=True)
        else:run(s,a)


if __name__=='__main__':main()
