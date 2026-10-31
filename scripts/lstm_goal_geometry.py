"""Simple target-location versus arrival-direction interventions in goal memory.

Only the saved goal code changes. Live population input, vision, trained
weights, recurrence and environment are retained. Mirror preserves the 20cm
shift and predictive point's distance to the real goal while changing the
tangential arrival component. True-center conditions are explicitly oracle
diagnostics. The grid lookahead condition uses only observed hit position/dir.
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
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena, summarize_episodes, verify_codes

VARIANTS = dict(pgc33=('intact', 'plain_goal', 'center_pgc_goal', 'mirror_goal', 'reverse_goal'),
                gc100=('intact', 'lookahead_goal', 'center_goal'))


def reflection(hit, direction, center):
    offset = center - hit
    radius = offset.norm(dim=1, keepdim=True)
    radial = torch.where(radius > 1e-7, offset / radius.clamp_min(1e-7), direction)
    reflected = 2 * (direction * radial).sum(1, keepdim=True) * radial - direction
    return reflected, radial, radius.squeeze(1)


def goal_response(pop, name, hit, direction, center):
    if name == 'intact':
        return pop.activity(hit, direction)
    if name == 'plain_goal':
        return pop.activity(hit, direction, shift_scale=0)
    if name == 'center_pgc_goal':
        ordinary = pop.activity(hit, direction, shift_scale=0)
        centered = pop.activity(center, direction, shift_scale=0)
        return torch.where(pop.is_pgc[None], centered, ordinary)
    if name == 'mirror_goal':
        reflected, _, _ = reflection(hit, direction, center)
        return pop.activity(hit, reflected)
    if name == 'reverse_goal':
        return pop.activity(hit, -direction)
    if name == 'lookahead_goal':
        return pop.activity(hit + .2 * direction, direction, shift_scale=0)
    if name == 'center_goal':
        return pop.activity(center, direction, shift_scale=0)
    raise ValueError(name)


def synthetic_checks(cfg):
    pop = build_population(cfg, 'cpu')
    rng = np.random.default_rng(20261031)
    center = torch.tensor(rng.uniform(-.7, .7, (512, 2)), dtype=torch.float32)
    theta = torch.tensor(rng.uniform(-np.pi, np.pi, 512), dtype=torch.float32)
    radial = torch.stack((theta.cos(), theta.sin()), 1)
    hit = center - .29 * radial
    angle = theta + torch.tensor(rng.uniform(-1.2, 1.2, 512), dtype=torch.float32)
    direction = torch.stack((angle.cos(), angle.sin()), 1)
    reflected, axis, radius = reflection(hit, direction, center)
    twice, _, _ = reflection(hit, reflected, center)
    d1 = (hit + .2 * direction - center).norm(dim=1)
    d2 = (hit + .2 * reflected - center).norm(dim=1)
    assert torch.allclose(d1, d2, atol=1e-6, rtol=0)
    assert torch.allclose(reflected.norm(dim=1), direction.norm(dim=1), atol=1e-6, rtol=0)
    assert torch.allclose(twice, direction, atol=1e-6, rtol=0)
    assert torch.allclose((reflected*axis).sum(1), (direction*axis).sum(1), atol=1e-6, rtol=0)
    if cfg['cond'] == 'pgc33':
        assert int(pop.is_pgc.sum()) == 169
        original = goal_response(pop, 'intact', hit, direction, center)
        for name in ('plain_goal', 'center_pgc_goal', 'mirror_goal', 'reverse_goal'):
            altered = goal_response(pop, name, hit, direction, center)
            assert torch.equal(original[:, ~pop.is_pgc], altered[:, ~pop.is_pgc])
    return dict(samples=512, max_reflection_distance_error=float((d1-d2).abs().max()),
                unit_length_and_radial_component_preserved=True,
                ordinary_goal_slots_unchanged=cfg['cond'] == 'pgc33')


@torch.inference_mode()
def run(seed, condition, args):
    started = time.perf_counter()
    path = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
    cfg = json.loads(path.read_text())
    checks = synthetic_checks(cfg)
    checkpoint = path.with_suffix('.pt')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert ckpt['frames'] == 1_000_000_000
    torch.manual_seed(args.eval_seed)
    if args.device == 'mps':
        torch.mps.manual_seed(args.eval_seed)
    net = build_policy(cfg, args.device).eval()
    net.load_state_dict(ckpt['policy'], strict=True)
    net.requires_grad_(False)
    pop = build_population(cfg, args.device)
    variants = VARIANTS[condition]
    n, groups = args.n_envs, len(variants)
    total = n * groups
    env = MatchedArena(cfg, n, groups, args.eval_seed, args.device)
    obs = env.reset_all()
    for key in ('pos', 'hd', 'dir', 'rgb'):
        assert all(torch.equal(obs[key][:n], obs[key][j*n:(j+1)*n]) for j in range(1, groups))
    goal_code = torch.zeros(total, 512, device=args.device)
    pa = torch.full((total,), -1, dtype=torch.long, device=args.device)
    pr = torch.zeros(total, device=args.device)
    state = net.zero_state(total, args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.eval_seed + 1)
    rows = {name: [] for name in variants}
    hits = {name: [] for name in variants}
    max_distance_error = 0.
    max_norm_error = 0.
    near_center_fallbacks = 0
    step = 0
    last_progress = started
    while len(rows['intact']) < args.episodes:
        code = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        logits, _, state = net.step(obs['rgb'], code, goal_code, pa, pr, state)
        uniform = torch.rand(n, generator=generator, device=args.device).repeat(groups)
        action = (uniform[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        # The environment auto-resets finished episodes. Preserve the old
        # physical goal before stepping so a hit is never paired with a new goal.
        true_center = env.goal.clone()
        was_known = env.n_goals > 0
        obs, reward, done, info = env.step(action)
        hit = info['hit']
        if bool(hit.any()):
            for j, name in enumerate(variants):
                sl = slice(j*n, (j+1)*n)
                memory = goal_response(pop, name, info['hit_pos'][sl], info['hit_dir'][sl], true_center[sl])
                goal_code[sl] = torch.where(hit[sl, None], memory, goal_code[sl])
            # Store every hit's geometry, independent of whether it is a first
            # or return visit. Terminal hits are skipped because their memory
            # is immediately erased by the standard episode reset.
            record = hit & ~done
            if bool(record.any()):
                idx = record.nonzero().squeeze(1)
                hp, hd, gc = info['hit_pos'][idx], info['hit_dir'][idx], true_center[idx]
                mirrored, radial, radius = reflection(hp, hd, gc)
                fwd = (hp + .2*hd - gc).norm(dim=1)
                mirror_dist = (hp + .2*mirrored - gc).norm(dim=1)
                norm_error = (mirrored.norm(dim=1) - hd.norm(dim=1)).abs().max()
                distance_error = (fwd - mirror_dist).abs().max()
                max_distance_error = max(max_distance_error, float(distance_error))
                max_norm_error = max(max_norm_error, float(norm_error))
                near_center_fallbacks += int((radius <= 1e-7).sum())
                assert float(distance_error) < 2e-6 and float(norm_error) < 2e-6
                cosine = (hd*mirrored).sum(1) / (hd.norm(dim=1)*mirrored.norm(dim=1)).clamp_min(1e-8)
                data = torch.cat((hp, gc, hd, torch.stack((
                    radius, fwd, mirror_dist, (hp-.2*hd-gc).norm(dim=1),
                    cosine.clamp(-1,1).acos()*180/np.pi,
                    .2*(hd-mirrored).norm(dim=1),
                    (hd*radial).sum(1), was_known[idx].float(),
                    ((hp+.2*hd).abs().amax(1)>cfg['arena_half']-.08).float(),
                    ((hp+.2*mirrored).abs().amax(1)>cfg['arena_half']-.08).float()), 1)), 1).cpu().numpy()
                for i, global_slot in enumerate(idx.cpu().tolist()):
                    hits[variants[global_slot//n]].append(np.r_[global_slot % n, step//cfg['episode_len'], step%cfg['episode_len'], data[i]])
        pa = action
        pr = reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            keep = (~done).float()
            goal_code *= keep[:, None]
            state = tuple(z * keep[:, None] for z in state)
            pa = torch.where(done, torch.full_like(action, -1), action)
            pr *= keep
            es = {k:v.cpu().tolist() for k,v in info['ep_stats'].items()}
            for j, slot in enumerate(es['env']):
                rows[variants[slot//n]].append(dict(slot=slot%n, episode=step//cfg['episode_len'],
                    **{k:es[k][j] for k in ('ret','n_goals','first_goal_t','path_eff','path_eff_n')}))
        step += 1
        if time.perf_counter() - last_progress > 25:
            print(f'{condition} s{seed} goal geometry: {step}/{cfg["episode_len"]*(args.episodes//n)}; '
                  f'{time.perf_counter()-started:.0f}s', flush=True)
            last_progress = time.perf_counter()
    first = [r['first_goal_t'] for r in rows['intact']]
    assert all([r['first_goal_t'] for r in rlist] == first for rlist in rows.values())
    columns = ['slot','episode','step','hit_x','hit_y','goal_x','goal_y','dir_x','dir_y',
               'hit_distance','forward_distance','mirror_distance','reverse_distance',
               'reflection_angle_degrees','predictive_point_separation','arrival_radial_cosine',
               'was_known','forward_beyond_walkable_bounds','mirror_beyond_walkable_bounds']
    geometry = {}
    for name, values in hits.items():
        array = np.asarray(values, dtype=np.float64).reshape(-1,len(columns))
        hits[name] = array
        geometry[name] = dict(n=len(array), **{key:float(array[:,i].mean()) for i,key in enumerate(columns) if i>=9},
                              reflection_angle_gt30_fraction=float(np.mean(array[:,13] > 30)))
    np.savez_compressed(args.out/f'{condition}_s{seed}_hits.npz', columns=np.array(columns), **hits)
    result = dict(condition=condition, seed=seed, checkpoint_frames=ckpt['frames'],
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  eval_seed=args.eval_seed, episodes=args.episodes, n_envs=n, device=args.device,
                  variants={name:summarize_episodes(r[:args.episodes],cfg['episode_len']) for name,r in rows.items()},
                  geometry=geometry, checks=dict(**checks,
                      observed_max_reflection_distance_error=max_distance_error,
                      observed_max_reflection_norm_error=max_norm_error,
                      near_center_fallbacks=near_center_fallbacks,
                      all_first_goal_episode_records_identical=True),
                  elapsed_seconds=time.perf_counter()-started)
    (args.out/f'{condition}_s{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE '+condition+f' s{seed}: '+', '.join(f'{k}={v["mean_score"]:.2f}' for k,v in result['variants'].items()),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--n-envs', type=int, default=64)
    ap.add_argument('--eval-seed', type=int, default=737373)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1,9)))
    ap.add_argument('--conditions', nargs='+', choices=list(VARIANTS), default=['pgc33','gc100'])
    ap.add_argument('--out', type=Path, default=ROOT/'results/lstm_goal_geometry_1B')
    args = ap.parse_args()
    assert args.episodes % args.n_envs == 0
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'protocol.json').write_text(json.dumps(dict(
        seeds=args.seeds, episodes_per_seed_variant=args.episodes, n_envs_per_variant=args.n_envs,
        eval_seed=args.eval_seed, device=args.device, variants=VARIANTS,
        invariant='Only the supplied goal-memory vector changes; live code, network weights, vision, task and reward are retained.',
        oracle_conditions=['pgc33/center_pgc_goal','pgc33/mirror_goal','gc100/center_goal'],
        mirror_definition='Reflect arrival direction about the hit-to-true-center radial axis. Preserve 20cm shift and distance of predictive point to true goal; flip tangential component only.',
        equivalence_before_running=dict(comparison='pgc33 mirror_goal minus intact score', margin=5.,
            procedure='90% paired t confidence interval inside [-5,+5], equivalent to 5%-level TOST.',
            meaning='Exploratory coarse tolerance: half a goal per 90s episode; not absence of all arrival-direction use.'),
        caveats=['Center and mirror use privileged true goal center for diagnosis, not a realistic sensing improvement.',
                 'Mirror does not remove the radial arrival component and does not match wall proximity or all spatial relationships.',
                 'Evaluation-only changed goal messages, not retraining; input distribution effects remain possible.'],
        code_identity_checks={s:verify_codes(s) for s in args.seeds}),indent=2)+'\n')
    for s in args.seeds:
        for condition in args.conditions:
            path=args.out/f'{condition}_s{s}.json'
            if path.exists():
                old=json.loads(path.read_text())
                assert (old['episodes'],old['n_envs'],old['eval_seed'])==(args.episodes,args.n_envs,args.eval_seed)
                print(f'Using completed {condition} s{s}',flush=True)
                continue
            run(s,condition,args)


if __name__=='__main__':
    main()
