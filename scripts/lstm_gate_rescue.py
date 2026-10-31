"""Restore selected PGC-driven LSTM components during closed-loop evaluation.

For each actual preceding h/c and physical state, compute the four LSTM
components with the mixed code and with the grid substitution. Feed only selected
mixed-code components into the state update; all other components use grid code.
This tests whether an internal pathway can recover navigation performance.
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
import torch
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena, gates_and_state, state_from_gates, summarize_episodes, verify_codes

VARIANTS = ('intact', 'both_grid', 'restore_input', 'restore_forget',
            'restore_candidate', 'restore_output', 'restore_candidate_output')
RESTORE = ({0, 1, 2, 3}, set(), {0}, {1}, {2}, {3}, {2, 3})
RESTORE_BY_NAME = dict(zip(VARIANTS, RESTORE))
RESTORE_BY_NAME['restore_input_forget'] = {0, 1}


@torch.inference_mode()
def run(seed, args):
    start = time.perf_counter()
    path = ROOT / f'models/pgc33/pgc33_lstm_s{seed}_1000M.json'
    cfg = json.loads(path.read_text())
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
    variants = tuple(args.variants)
    restored_components = [RESTORE_BY_NAME[v] for v in variants]
    n, groups = args.n_envs, len(variants)
    total = n * groups
    env = MatchedArena(cfg, n, groups, args.eval_seed, args.device)
    obs = env.reset_all()
    for key in ('pos', 'hd', 'dir', 'rgb'):
        assert all(torch.equal(obs[key][:n], obs[key][j*n:(j+1)*n]) for j in range(1, groups))
    mixed_goal = torch.zeros(total, 512, device=args.device)
    grid_goal = torch.zeros_like(mixed_goal)
    pa = torch.full((total,), -1, dtype=torch.long, device=args.device)
    pr = torch.zeros(total, device=args.device)
    state = net.zero_state(total, args.device)
    masks = [torch.tensor([gate in restored for restored in restored_components], device=args.device)
             .repeat_interleave(n)[:, None] for gate in range(4)]
    generator = torch.Generator(device=args.device).manual_seed(args.eval_seed + 1)
    rows = {v: [] for v in variants}
    step, max_error = 0, 0.
    last_progress = start
    while len(rows[variants[0]]) < args.episodes:
        mixed = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        grid = pop.activity(obs['pos'], obs['dir'], obs['vel'], shift_scale=0)
        embedding = net.embed(obs['rgb'])
        xm = net._concat(embedding, mixed, mixed_goal, pa, pr, None)
        xg = net._concat(embedding, grid, grid_goal, pa, pr, None)
        mgates, mstate = gates_and_state(net, xm, state)
        ggates, gstate = gates_and_state(net, xg, state)
        if step % 200 == 0:
            for x, reconstructed in [(xm, mstate), (xg, gstate)]:
                reference = net.lstm(x, state)
                for a, b in zip(reference, reconstructed):
                    max_error = max(max_error, float((a-b).abs().max()))
                    assert torch.allclose(a, b, atol=2e-4, rtol=2e-5)
        gates = tuple(torch.where(mask, a, b) for mask, a, b in zip(masks, mgates, ggates))
        state = state_from_gates(gates, state[1])
        logits = net.pi(state[0])
        uniforms = torch.rand(n, generator=generator, device=args.device).repeat(groups)
        actions = (uniforms[:, None] > logits.softmax(1).cumsum(1)).sum(1).clamp_max(5)
        obs, reward, done, info = env.step(actions)
        if bool(info['hit'].any()):
            hm = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'])
            hg = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'], shift_scale=0)
            mixed_goal = torch.where(info['hit'][:, None], hm, mixed_goal)
            grid_goal = torch.where(info['hit'][:, None], hg, grid_goal)
        pa, pr = actions, reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            keep = (~done).float()
            mixed_goal *= keep[:, None]
            grid_goal *= keep[:, None]
            state = tuple(z * keep[:, None] for z in state)
            pa = torch.where(done, torch.full_like(actions, -1), actions)
            pr *= keep
            es = {k: v.cpu().tolist() for k, v in info['ep_stats'].items()}
            for j, slot in enumerate(es['env']):
                rows[variants[slot // n]].append(dict(slot=slot % n, episode=step // cfg['episode_len'],
                    **{k: es[k][j] for k in ('ret', 'n_goals', 'first_goal_t', 'path_eff', 'path_eff_n')}))
        step += 1
        if time.perf_counter()-last_progress > 25:
            print(f'gate rescue s{seed}: {step}/{cfg["episode_len"]*(args.episodes//n)}; '
                  f'{time.perf_counter()-start:.0f}s', flush=True)
            last_progress = time.perf_counter()
    result = dict(seed=seed, condition='pgc33', checkpoint_frames=ckpt['frames'],
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  eval_seed=args.eval_seed, episodes=args.episodes, n_envs=n, device=args.device,
                  variants={v:summarize_episodes(r[:args.episodes], cfg['episode_len']) for v,r in rows.items()},
                  gate_reconstruction_max_error=max_error, elapsed_seconds=time.perf_counter()-start)
    (args.out / f'pgc33_s{seed}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('DONE gate rescue s'+str(seed)+': '+', '.join(f'{k}={v["mean_score"]:.2f}'for k,v in result['variants'].items()), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--n-envs', type=int, default=64)
    ap.add_argument('--eval-seed', type=int, default=737373)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1,9)))
    ap.add_argument('--variants', nargs='+', choices=list(RESTORE_BY_NAME), default=list(VARIANTS))
    ap.add_argument('--out', type=Path, default=ROOT / 'results/lstm_gate_rescue_1B')
    args=ap.parse_args()
    assert args.episodes % args.n_envs == 0
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    checks={s:verify_codes(s)for s in args.seeds}
    (args.out/'protocol.json').write_text(json.dumps(dict(
        episodes_per_seed_variant=args.episodes, n_envs_per_variant=args.n_envs, eval_seed=args.eval_seed,
        seeds=args.seeds, device=args.device, variants=args.variants, code_identity_checks=checks,
        intervention='Both current and goal inputs use ordinary grid code. Selected LSTM components are replaced by their mixed-code values computed from the same actual preceding h/c. Recurrent state and behavior then evolve normally.',
        components={'input':0,'forget':1,'candidate':2,'output':3},
        controls='intact restores all four components; both_grid restores none. All branches use the same explicit LSTM update, checked against the original cell, and share environmental layouts/noise and action uniforms.',
        interpretation='Causal pathway intervention at evaluation, not retraining. Restored components have access to counterfactual predictive code; this is not a pure grid-input policy. Gate contributions interact.'),indent=2)+'\n')
    for s in args.seeds:
        path=args.out/f'pgc33_s{s}.json'
        if path.exists():
            old=json.loads(path.read_text())
            assert (old['episodes'],old['n_envs'],old['eval_seed'])==(args.episodes,args.n_envs,args.eval_seed)
            assert set(old['variants']) == set(args.variants)
            continue
        run(s,args)


if __name__=='__main__':
    main()
