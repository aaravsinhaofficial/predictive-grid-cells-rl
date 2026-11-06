"""Context-controlled goal-message families in the ORIGINAL frozen RL LSTMs.

This is an experimental replay, not a new task or newly trained model. Native
policy histories supply all visual/current/action/reward inputs. Only the saved
goal message is replaced. A periodic input sweep is not itself a ring attractor.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from code.common import build_policy, build_population
from lstm_code_swap import MatchedArena

OFFSETS = (0., np.pi / 2, -np.pi / 2)
HORIZONS = (0, 1, 5, 15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def angular_error(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def snapshot_geometry(observer, theta, offset, radius=.75, pickup_radius=.29, shift=.20):
    """All pickup points lie .29 m from the true goal, arriving radially inward."""
    theta = np.asarray(theta)
    u = np.stack((np.cos(theta), np.sin(theta)), -1)
    arrival = np.stack((np.cos(theta + offset), np.sin(theta + offset)), -1)
    center = np.asarray(observer) + radius * u
    pickup = center - pickup_radius * arrival
    ahead = pickup + shift * arrival
    return center, pickup, arrival, ahead


def family_geometry(z):
    z = np.asarray(z, dtype=np.float64)
    centered = z - z.mean(0)
    radius = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    adjacent = np.linalg.norm(np.roll(z, -1, axis=0) - z, axis=1)
    tangent = np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)
    turn = np.sum(tangent * np.roll(tangent, 1, axis=0), 1) / np.maximum(
        np.linalg.norm(tangent, axis=1) * np.linalg.norm(np.roll(tangent, 1, axis=0), axis=1), 1e-30)
    return dict(radius=float(radius), h_radius=float(np.sqrt(np.mean(np.sum(centered[:, :256] ** 2, 1)))),
                c_radius=float(np.sqrt(np.mean(np.sum(centered[:, 256:] ** 2, 1)))),
                adjacent_median=float(np.median(adjacent)), adjacent_max=float(adjacent.max()),
                adjacent_max_over_radius=float(adjacent.max() / max(radius, 1e-30)),
                adjacent_max_over_median=float(adjacent.max() / max(np.median(adjacent), 1e-30)),
                tangent_neighbor_cosine_median=float(np.median(turn)), tangent_neighbor_cosine_min=float(turn.min()))


def perturb_families(z, seed):
    """Exact neighbor-state substitutions plus equal-distance local-normal noise."""
    z = np.asarray(z, dtype=np.float64)
    plus, minus = np.roll(z, -1, 0), np.roll(z, 1, 0)
    tangent = plus - minus
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-30)
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=z.shape)
    normal -= np.sum(normal * tangent, 1, keepdims=True) * tangent
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    plus_norm = np.linalg.norm(plus - z, axis=1, keepdims=True)
    minus_norm = np.linalg.norm(minus - z, axis=1, keepdims=True)
    return np.stack((z, plus, minus, z + normal * plus_norm, z - normal * minus_norm)).astype(np.float32), tangent, normal


def fit_phase_readout(z, theta):
    """Analysis-only ridge readout; never supplies inputs or actions."""
    train = np.arange(len(z)) % 2 == 0
    center = z[train].mean(0)
    x = z[train] - center
    target = np.stack((np.cos(theta[train]), np.sin(theta[train])), 1)
    # Dual ridge keeps high-dimensional decoding regularized and inexpensive.
    gram = x @ x.T
    alpha = max(float(np.trace(gram) / len(gram)) * .01, 1e-7)
    weight = x.T @ np.linalg.solve(gram + alpha * np.eye(len(gram)), target)
    return center, weight, ~train


def phase_metrics(z, theta, decoder):
    center, weight, test = decoder
    y = (z - center) @ weight
    amplitude = np.linalg.norm(y, axis=1)
    valid = test & (amplitude >= .1)
    phase = np.arctan2(y[:, 1], y[:, 0])
    return dict(heldout_mae_deg=float(np.rad2deg(angular_error(phase[valid], theta[valid])).mean()) if valid.any() else None,
                heldout_valid_fraction=float(valid.sum() / test.sum()), heldout_amplitude_median=float(np.median(amplitude[test])))


def load_model(condition, seed, device):
    cfg_path = ROOT / f'models/{condition}/{condition}_lstm_s{seed}_1000M.json'
    cfg = json.loads(cfg_path.read_text())
    checkpoint = cfg_path.with_suffix('.pt')
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert data['frames'] == 1_000_000_000
    net = build_policy(cfg, device).eval().requires_grad_(False)
    net.load_state_dict(data['policy'], strict=True)
    return cfg, checkpoint, net, build_population(cfg, device)


@torch.inference_mode()
def collect_contexts(condition, seed, args, cfg, net, pop):
    """Replayable windows generated by the original actor, with no interventions."""
    device = args.device
    torch.manual_seed(args.eval_seed)
    if device == 'mps': torch.mps.manual_seed(args.eval_seed)
    env = MatchedArena(cfg, args.n_envs, 1, args.eval_seed, device)
    obs = env.reset_all()
    state = net.zero_state(args.n_envs, device)
    goal = torch.zeros(args.n_envs, cfg['n_cells'], device=device)
    pa = torch.full((args.n_envs,), -1, dtype=torch.long, device=device)
    pr = torch.zeros(args.n_envs, device=device)
    generator = torch.Generator(device=device).manual_seed(args.eval_seed + 1)
    length = args.prepare + args.future
    records = deque(maxlen=length)
    selected, used_slots = [], set()
    start = time.perf_counter()
    last = start
    for step in range(args.max_steps):
        current = pop.activity(obs['pos'], obs['dir'], obs['vel'])
        embedding = net.embed(obs['rgb'])
        x = net._concat(embedding, current, goal, pa, pr, None)
        state_before = state
        state = net.lstm(x, state)
        probabilities = net.pi(state[0]).softmax(1)
        uniform = torch.rand(args.n_envs, generator=generator, device=device)
        action = (uniform[:, None] > probabilities.cumsum(1)).sum(1).clamp_max(5)
        next_obs, reward, done, info = env.step(action)
        # Preserve actual model inputs and exact before/after state, not rerendered images.
        record = {k: v.detach().cpu().numpy().copy() for k, v in dict(
            x=x, h_before=state_before[0], c_before=state_before[1], h_after=state[0], c_after=state[1],
            pos=obs['pos'], hd=obs['hd'], direction=obs['dir'], velocity=obs['vel'],
            goal_center=env.goal, hit=info['hit'], done=done, action=action,
            known=(goal.abs().sum(1) > 0)).items()}
        record['step'] = np.full(args.n_envs, step, dtype=np.int64)
        records.append(record)
        if len(records) == length:
            batch = list(records)
            endpoint = batch[args.prepare - 1]['pos']
            for slot in range(args.n_envs):
                if slot in used_slots or not bool(batch[0]['known'][slot]): continue
                if np.max(np.abs(endpoint[slot])) > .30: continue
                if any(r['hit'][slot] or r['done'][slot] or r['x'][slot, -1] != 0 for r in batch): continue
                positions = np.stack([r['pos'][slot] for r in batch])
                # Sufficient, all-angle exclusion of every synthetic goal disk.
                if np.max(np.linalg.norm(positions - endpoint[slot], axis=1)) >= .75 - cfg['goal_radius'] - .01: continue
                window = {k: np.stack([r[k][slot] for r in batch]) for k in batch[0]}
                window['slot'] = np.array(slot)
                window['landmark_pos'] = env.land_pos[slot].detach().cpu().numpy().copy()
                window['stripe_phase'] = env.stripe_phase[slot].detach().cpu().numpy().copy()
                selected.append(window); used_slots.add(slot)
                print(f'{condition} s{seed}: context {len(selected)}/{args.contexts}, slot {slot}, endpoint step {int(window["step"][args.prepare-1])}', flush=True)
                if len(selected) >= args.contexts: break
        if len(selected) >= args.contexts: break
        if bool(info['hit'].any()):
            new_goal = pop.activity(info['hit_pos'], info['hit_dir'], info['hit_vel'])
            goal = torch.where(info['hit'][:, None], new_goal, goal)
        pa, pr = action, reward.clamp(-cfg['reward_clip'], cfg['reward_clip'])
        if bool(done.any()):
            keep = (~done).float()
            goal *= keep[:, None]; state = tuple(a * keep[:, None] for a in state)
            pa = torch.where(done, torch.full_like(pa, -1), pa); pr *= keep
        obs = next_obs
        if time.perf_counter() - last > 20:
            print(f'{condition} s{seed}: native collection {step+1} steps, {len(selected)} contexts, {time.perf_counter()-start:.1f}s', flush=True); last = time.perf_counter()
    if len(selected) != args.contexts:
        raise RuntimeError(f'Only {len(selected)}/{args.contexts} qualifying contexts after {step+1} decisions')
    return selected, dict(seconds=time.perf_counter()-start, decisions=step+1, slots=list(map(int, sorted(used_slots))))


def step_full(net, x, z):
    h, c = net.lstm(x, (z[:, :256], z[:, 256:]))
    return torch.cat((h, c), 1)


def condition_states(net, inputs, initial_state, goal_codes):
    """Elicit states with common recorded physical inputs and supplied goal codes.

    inputs: T x1287 actual original-agent input vectors; initial_state:512 or
    B x512 [h,c]; goal_codes:B x512. The original actor/critic remain unchanged.
    This helper supports an independent closed-loop false-goal-state assay.
    """
    z = initial_state
    if z.ndim == 1: z = z[None].expand(len(goal_codes), -1)
    for x in inputs:
        xx = x.repeat(len(goal_codes), 1)
        xx[:, 768:1280] = goal_codes
        z = step_full(net, xx, z)
    return z


@torch.inference_mode()
def analyze_context(net, pop, cfg, window, context_id, args, output):
    device = next(net.parameters()).device
    xseq = torch.tensor(window['x'], device=device)
    initial = np.concatenate((window['h_before'][0], window['c_before'][0]))
    z = torch.tensor(initial[None], device=device)
    reconstruction_error = 0.
    for k, x in enumerate(xseq):
        z = step_full(net, x[None], z)
        actual = np.concatenate((window['h_after'][k], window['c_after'][k]))
        reconstruction_error = max(reconstruction_error, float(np.max(np.abs(z.cpu().numpy()[0] - actual))))
    assert reconstruction_error < 3e-4, reconstruction_error
    observer = window['pos'][args.prepare - 1]
    # Fine grid is the real experiment; every other point is a coarse-grid diagnostic.
    theta = np.arange(args.angles) * (2 * np.pi / args.angles)
    result = dict(context=context_id, slot=int(window['slot']), endpoint_step=int(window['step'][args.prepare-1]),
                  observer=observer.tolist(), heading=float(window['hd'][args.prepare-1]),
                  native_replay_max_error=reconstruction_error, families=[])
    archive = {f'history_{k}': v for k,v in window.items()}
    for offset_index, offset in enumerate(OFFSETS):
        center, pickup, arrival, ahead = snapshot_geometry(observer, theta, offset)
        assert np.max(np.abs(center)) < 1.17 and np.max(np.abs(pickup)) < 1.17 and np.max(np.abs(ahead)) < 1.17
        assert np.min(np.linalg.norm(window['pos'][:, None] - center[None], axis=2)) > cfg['goal_radius']
        p = torch.tensor(pickup, dtype=torch.float32, device=device)
        direction = torch.tensor(arrival, dtype=torch.float32, device=device)
        native_goal = pop.activity(p, direction, direction * cfg['v_max'])
        ordinary_goal = pop.activity(p, direction, direction * cfg['v_max'], shift_scale=0)
        assert torch.equal(native_goal[:, ~pop.is_pgc], ordinary_goal[:, ~pop.is_pgc])
        for variant, goal in [('native', native_goal)] + ([('ordinary_goal', ordinary_goal)] if bool(pop.is_pgc.any()) else []):
            label = f'offset{offset_index}_{variant}'
            z = torch.tensor(initial, device=device).repeat(args.angles, 1)
            for x in xseq[:args.prepare]:
                xx = x.repeat(args.angles, 1); xx[:, 768:1280] = goal
                z = step_full(net, xx, z)
            family = z.cpu().numpy()
            decoder = fit_phase_readout(family.astype(np.float64), theta)
            perturb, tangent, normal = perturb_families(family, args.eval_seed + context_id * 10 + offset_index)
            metrics = dict(variant=variant, offset_deg=float(np.rad2deg(offset)), geometry=family_geometry(family),
                           coarse_geometry=family_geometry(family[::2]), phase_at_end=phase_metrics(family, theta, decoder), continuations={})
            archive[label + '_states'] = family
            archive[label + '_tangent'] = tangent.astype(np.float32)
            archive[label + '_geometry'] = np.stack((center, pickup, arrival, ahead))
            bearings = np.stack([np.arctan2((a-observer)[:,1], (a-observer)[:,0]) for a in (center,pickup,ahead)])
            archive[label + '_bearings_center_pickup_ahead'] = bearings
            initial_errors = np.linalg.norm(perturb[1:] - perturb[:1], axis=2)
            for mode in ('goal_held', 'goal_withheld'):
                branch = torch.tensor(perturb.reshape(-1,512), device=device)
                records = {}
                for t in range(args.future + 1):
                    states = branch.reshape(5,args.angles,512).cpu().numpy()
                    base = states[0]
                    geom = family_geometry(base)
                    errors = np.linalg.norm(states[1:] - base[None], axis=2)
                    if t in HORIZONS:
                        # Native moving sensory input is shared. Any retained angle is a state difference.
                        gains = errors / np.maximum(initial_errors, 1e-30)
                        action = net.pi(branch[:args.angles,:256]).softmax(1).cpu().numpy()
                        action_all = net.pi(branch[:, :256]).softmax(1).reshape(5,args.angles,6).cpu().numpy()
                        tv = .5 * np.abs(action_all[1:] - action[None]).sum(2)
                        records[str(t)] = dict(radius_retention=geom['radius'] / max(metrics['geometry']['radius'],1e-30),
                            geometry=geom, phase=phase_metrics(base,theta,decoder),
                            neighbor_state_error_gain=float(np.mean(gains[:2])), normal_error_gain=float(np.mean(gains[2:])),
                            neighbor_action_tv=float(np.mean(tv[:2])), normal_action_tv=float(np.mean(tv[2:])))
                        archive[label + '_' + mode + f'_t{t}'] = states
                    if t == args.future: break
                    xx = xseq[args.prepare+t].repeat(5 * args.angles, 1)
                    xx[:,768:1280] = goal.repeat(5,1) if mode=='goal_held' else 0
                    branch = step_full(net,xx,branch)
                metrics['continuations'][mode] = records
            result['families'].append(metrics)
    archive['theta'] = theta
    np.savez_compressed(output / f'context{context_id}_states.npz', **archive)
    return result


def plot_result(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
    styles = {'native':('#DC7738','Native goal'), 'ordinary_goal':('#5685A4','Ordinary goal')}
    for variant,(color,label) in styles.items():
        families=[f for c in result['contexts'] for f in c['families'] if f['variant']==variant]
        if not families: continue
        for mode,ls in [('goal_held','-'),('goal_withheld','--')]:
            radius=np.array([[f['continuations'][mode][str(t)]['radius_retention'] for t in HORIZONS] for f in families])
            axes[0].plot(np.array(HORIZONS)/15,radius.mean(0),ls,color=color,label=label+(' held' if mode=='goal_held' else ' withheld'))
            for ax,key in [(axes[1],'neighbor_state_error_gain'),(axes[2],'normal_error_gain')]:
                a=np.array([[f['continuations'][mode][str(t)][key] for t in HORIZONS] for f in families])
                ax.plot(np.array(HORIZONS)/15,a.mean(0),ls,color=color)
    axes[0].set(ylabel='Full-state family size / initial',title='Does goal-specific state persist?')
    axes[1].set(ylabel='Error / initial error',title='Neighbor goal-state swap')
    axes[2].set(ylabel='Error / initial error',title='Equal-size normal perturbation')
    for ax in axes: ax.set_xlabel('Native replay after intervention (s)'); ax.axhline(1,color='.8',lw=.8);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'pilot_dynamics.png',dpi=180);plt.close(fig)


def run(condition, seed, args):
    started=time.perf_counter()
    out=args.out/f'{condition}_s{seed}';out.mkdir(parents=True,exist_ok=True)
    cfg, checkpoint, net, pop=load_model(condition,seed,args.device)
    checkpoint_hash=sha(checkpoint)
    cache=out/'native_contexts.npz'
    if cache.exists():
        loaded=np.load(cache);contexts=[{k.split('__',1)[1]:loaded[k] for k in loaded.files if k.startswith(f'{i}__')} for i in range(args.contexts)]
        collection=json.loads((out/'collection.json').read_text())
    else:
        contexts,collection=collect_contexts(condition,seed,args,cfg,net,pop)
        np.savez_compressed(cache,**{f'{i}__{k}':v for i,c in enumerate(contexts) for k,v in c.items()})
        save_json(out/'collection.json',collection)
    # CPU recurrent analysis is fast and avoids MPS batch-shape accumulation differences.
    net=net.cpu();pop=build_population(cfg,'cpu')
    analyzed=[]
    for i,window in enumerate(contexts):
        analyzed.append(analyze_context(net,pop,cfg,window,i,args,out))
        print(f'{condition} s{seed}: analyzed context {i+1}/{len(contexts)}',flush=True)
    assert sha(checkpoint)==checkpoint_hash
    result=dict(condition=condition,seed=seed,checkpoint_sha256=checkpoint_hash,collection=collection,
                contexts=analyzed,elapsed_seconds=time.perf_counter()-started,
                interpretation='Input-defined goal families after native physical history. Cue withdrawal removes the saved goal input only; this is not the usual trained task. No global autonomous ring claim.')
    save_json(out/'summary.json',result);plot_result(result,out)
    print(f'DONE {condition} s{seed}: {result["elapsed_seconds"]:.1f}s',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--conditions',nargs='+',default=['gc100','pgc33']);ap.add_argument('--seeds',type=int,nargs='+',default=[1,2])
    ap.add_argument('--device',default='mps');ap.add_argument('--contexts',type=int,default=4)
    ap.add_argument('--n-envs',type=int,default=16);ap.add_argument('--max-steps',type=int,default=2700)
    ap.add_argument('--angles',type=int,default=192);ap.add_argument('--prepare',type=int,default=15);ap.add_argument('--future',type=int,default=15)
    ap.add_argument('--eval-seed',type=int,default=864201)
    ap.add_argument('--out',type=Path,default=ROOT/'results/original_goal_bearing_manifold/pilot')
    args=ap.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(vars(args));protocol['out']=str(args.out);protocol['offsets_deg']=[0,90,-90]
    protocol['sources']={p:sha(ROOT/p) for p in ['scripts/lstm_goal_bearing_manifold.py','code/env.py','code/codes.py','code/nets.py','code/common.py','scripts/lstm_code_swap.py']}
    protocol['role']='Exploratory original-agent feasibility pilot; no confirmatory seed-level test.'
    protocol['inference_inputs']='Exact native physical replay, synthetic saved-goal code only. No direction/position labels fed to the actor beyond original code maps.'
    save_json(args.out/'protocol.json',protocol)
    for seed in args.seeds:
        for condition in args.conditions: run(condition,seed,args)


if __name__=='__main__':main()
