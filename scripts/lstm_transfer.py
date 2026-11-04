"""Paired 1B LSTM transfer experiment. Default action only displays the plan."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = ('gc100', 'pgc33')
ARENAS = {'reference': 1.25, 'medium': 1.625, 'large': 2.0}  # half-width, metres
TEST_SEED = 730031
ADAPT_FRAMES = 1_280_000
FRAMES = 1_000_000_000


def spectrum(samples):
    """Centered covariance PCA without feature rescaling; null for flat activity."""
    x = np.array(samples, dtype=np.float64, copy=True)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('PCA needs at least two finite sample rows')
    x -= x.mean(axis=0)
    gram = x.T @ x if x.shape[1] <= len(x) else x @ x.T
    values = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0)
    total = float(values.sum())
    result = dict(samples=len(x), width=x.shape[1], total_variance=total / (len(x)-1),
                  pc3=None, pc10=None, d90=None, participation_ratio=None)
    if total and np.any(np.ptp(samples, axis=0)):
        ratios = values / total
        result.update(pc3=float(ratios[:3].sum()), pc10=float(ratios[:10].sum()),
                      d90=int(np.searchsorted(ratios.cumsum(), .9) + 1),
                      participation_ratio=float(1 / (ratios @ ratios)))
    return result


class Trajectory:
    """Record native actions or replay a partner's. Check matched physical samples."""
    def __init__(self, steps, stride, actions=None):
        self.steps, self.stride = steps, stride
        self.actions = [] if actions is None else actions
        self.replay = actions is not None
        self.t = 0
        self.physical = hashlib.sha256()
        self.inputs, self.hidden = [], []

    def cell_hook(self, module, args, output):
        if self.t < self.steps and self.t % self.stride == 0:
            self.inputs.append(args[0].detach().cpu().numpy())
            self.hidden.append(output[0].detach().cpu().numpy())

    def act(self, obs, sampled):
        action = sampled
        if self.t < self.steps:
            if self.replay:
                action = self.actions[self.t]
            else:
                self.actions.append(sampled)
            if self.t % self.stride == 0:
                for key in ('pos', 'vel', 'dir', 'hd'):
                    self.physical.update(obs[key].contiguous().cpu().numpy().tobytes())
        self.t += 1
        return action


def measure_pair(configs, policies, device, episodes, slots, stride):
    import torch
    from code.common import build_population
    from code.evaluate import run_eval

    def rollout(c, trace, count):
        cfg, net = configs[c], policies[c]
        # train.main changes cuDNN settings. Fix the evaluation precision in both modes.
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True,
                                        allow_tf32=False), net.lstm.register_forward_hook(trace.cell_hook):
            return run_eval(cfg, net, build_population(cfg, device), device,
                            episodes=count, n_envs=slots, seed=TEST_SEED,
                            paired=True, action_hook=trace.act)

    native, behavior = {}, {}
    for c in CONDITIONS:
        native[c] = Trajectory(configs[c]['episode_len'], stride)
        behavior[c] = rollout(c, native[c], episodes)
    results = {}
    for c, other in zip(CONDITIONS, reversed(CONDITIONS)):
        own, driver = native[c], native[other]
        replay = Trajectory(driver.steps, stride, driver.actions)
        check = rollout(c, replay, slots)
        if (replay.physical.digest() != driver.physical.digest()
                or check['scores'] != behavior[other]['scores'][:slots]
                or check['first_goal_steps'] != behavior[other]['first_goal_steps'][:slots]):
            raise RuntimeError('Partner replay did not reproduce the native trajectory')
        # Both networks now have exactly the same mix of trajectories, including
        # discoveries, goal updates, previous rewards/actions, and occupancy.
        x = np.concatenate(own.inputs + replay.inputs)
        h = np.concatenate(own.hidden + replay.hidden)
        n_cells = configs[c]['n_cells']
        # PolicyNet._concat layout: vision embedding, current code, goal code, ...
        start = policies[c].encoder.fc.out_features
        results[c] = dict(navigation=behavior[c], native_hidden=spectrum(np.concatenate(own.hidden)),
                          matched_hidden=spectrum(h), full_input=spectrum(x),
                          current_code=spectrum(x[:, start:start+n_cells]),
                          goal_code=spectrum(x[:, start+n_cells:start+2*n_cells]),
                          trajectory_sha256=own.physical.hexdigest())
    return results


def parents(models, seeds):
    """Use the published checkpoint identity; reject incomplete or mislabeled pairs."""
    manifest = json.loads((ROOT/'results/lstm_pca_1B/summary.json').read_text())['models']
    files, missing = {}, []
    for row in manifest:
        c, s = row['condition'], row['seed']
        if s not in seeds:
            continue
        path = models / c / Path(row['checkpoint']).name
        cfg = json.loads((ROOT / row['checkpoint']).with_suffix('.json').read_text())
        if (cfg['cond'], cfg['seed'], cfg['frames'], cfg['policy']) != (c, s, FRAMES, 'lstm'):
            raise ValueError(f'Wrong model config for {path}')
        if not path.is_file():
            missing.append(str(path))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash differs from original PCA manifest: {path}')
        files[s,c] = cfg, path, row['checkpoint_sha256']
    for s in seeds:
        grid, pgc = [files[s,c][0] for c in CONDITIONS]
        for key in grid.keys() | pgc.keys():
            if key not in ('cond', 'pgc_frac', 'out', 'population') and grid.get(key) != pgc.get(key):
                raise ValueError(f'Pair {s} differs in {key}')
    return files, missing


def adapt(cfg, parent, arena_half, out, device):
    from code import train
    # Pass the original model and A2C settings through the training parser.
    argv = ['--out', str(out), '--init_weights', str(parent), '--paired_layouts']
    for key in vars(train.parse_args(['--out', str(out)])):
        if key in cfg and key not in ('out', 'resume', 'init_weights', 'frames', 'arena_half',
                                       'seed', 'device', 'paired_layouts'):
            argv += ['--' + key, str(cfg[key])]
    train.main(argv + ['--arena_half', str(arena_half), '--frames', str(ADAPT_FRAMES),
                       '--seed', str(9000 + cfg['seed']), '--device', device])
    return out/'ckpt_final.pt'


def summarize(records):
    """Training seeds are the units. Compute contrasts before averaging."""
    from scipy.stats import spearmanr

    def contrast(b, a):
        return None if b is None or a is None else b - a

    def estimate(values):
        values = [v for v in values if v is not None]
        result = dict(n=len(values), mean=float(np.mean(values)) if values else None, ci95=None)
        if len(values) >= 2:
            draws = np.random.default_rng(0).choice(values, (10000, len(values)), replace=True)
            result['ci95'] = np.percentile(draws.mean(axis=1), [2.5, 97.5]).tolist()
        return result

    by_arena = {}
    for row in records:
        a, b = [row['models'][c] for c in CONDITIONS]
        deltas = dict(goals=b['navigation']['mean_goals'] - a['navigation']['mean_goals'])
        for k in ('pc3', 'pc10'):
            deltas[k] = contrast(b['matched_hidden'][k], a['matched_hidden'][k])
        for k in ('current_code', 'full_input', 'matched_hidden'):
            deltas[k + '_d90'] = contrast(b[k]['d90'], a[k]['d90'])
        by_arena.setdefault(row['arena'], {})[row['seed']] = deltas
    summary = {}
    for arena, pairs in by_arena.items():
        entry = dict(pgc_minus_grid=pairs,
                     paired_estimates={k: estimate(v[k] for v in pairs.values())
                                       for k in next(iter(pairs.values()))})
        for metric in ('pc3', 'pc10'):
            valid = [v for v in pairs.values() if v[metric] is not None]
            rho = None
            if len(valid) >= 3:
                variance, goals = ([v[k] for v in valid] for k in (metric, 'goals'))
                if np.ptp(variance) and np.ptp(goals):
                    rho = float(spearmanr(variance, goals).statistic)
            entry[f'spearman_delta_{metric}_vs_goals'] = rho
        if arena != 'reference' and 'reference' in by_arena:
            differences = {s: v['goals'] - by_arena['reference'][s]['goals']
                           for s, v in pairs.items()}
            entry['delta_goals_minus_reference_by_seed'] = differences
            entry['transfer_contrast'] = estimate(differences.values())
        summary[arena] = entry
    return summary


def plot_pc10(records, out):
    from matplotlib.figure import Figure

    arenas = list(dict.fromkeys(row['arena'] for row in records))
    fig = Figure(figsize=(4 * len(arenas), 3.7), layout='constrained')
    axes = fig.subplots(1, len(arenas), squeeze=False, sharey=True)[0]
    colors = ('#29658d', '#b74b1f')
    for ax, arena in zip(axes, arenas):
        rows = [row for row in records if row['arena'] == arena]
        for row in rows:
            models = [row['models'][c] for c in CONDITIONS]
            x = [m['matched_hidden']['pc10'] for m in models]
            x = [np.nan if v is None else 100 * v for v in x]
            y = [m['navigation']['mean_goals'] for m in models]
            ax.plot(x, y, color='0.8', lw=1, zorder=1)
            for j, condition in enumerate(CONDITIONS):
                ax.scatter(x[j], y[j], color=colors[j], s=24,
                           label=condition if row is rows[0] else None)
        ax.set(title=f"{rows[0]['arena_width_m']:g} m", xlabel='Hidden variance in first 10 PCs (%)')
    axes[0].set_ylabel('Autonomous goals per episode')
    axes[0].legend(frameon=False)
    fig.savefig(out/'pc10_vs_navigation.png', dpi=160)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k,v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'frozen', 'finetune'), nargs='?', default='plan')
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--out', type=Path, default=ROOT/'runs_transfer')
    parser.add_argument('--seeds', type=int, nargs='+', choices=range(1,9), default=list(range(1,9)))
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--slots', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or min(args.episodes,args.slots,args.stride) < 1
            or args.episodes % args.slots):
        parser.error('Choose distinct seeds, positive counts and complete slot batches')
    files, missing = parents(args.models, args.seeds)
    episode_steps = files[args.seeds[0], 'gc100'][0]['episode_len']
    if args.stride >= episode_steps:
        parser.error('PCA stride must be shorter than an episode')
    plan = dict(conditions=list(CONDITIONS), parent_frames=FRAMES, seeds=args.seeds, arena_widths_m={k:2*v for k,v in ARENAS.items()},
                episodes=args.episodes, slots=args.slots, pca_stride=args.stride,
                eval_seed=TEST_SEED, adapt_frames_per_model=ADAPT_FRAMES,
                adaptation_seed_offset=9000, episode_steps=episode_steps,
                frozen_autonomous_episodes=len(ARENAS)*2*len(args.seeds)*args.episodes,
                frozen_replay_episodes=len(ARENAS)*2*len(args.seeds)*args.slots,
                total_optional_training_frames=(len(ARENAS)-1)*2*len(args.seeds)*ADAPT_FRAMES,
                missing_checkpoints=missing)
    print(json.dumps(plan, indent=2), flush=True)
    if args.mode == 'plan':
        return
    if missing:
        parser.exit(2, 'Restore all requested .pt files before running.\n')

    import torch
    from code.common import build_policy
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    frozen = args.out/'frozen'
    if args.mode == 'finetune':
        if not (frozen/'summary.json').is_file():
            parser.error('Run the matching frozen experiment under --out before fine-tuning')
        baseline = json.loads((frozen/'summary.json').read_text())
        if baseline['plan'] != plan or baseline['device'] != args.device:
            parser.error('Fine-tuning requires the same frozen evaluation plan and device')
    out = args.out/args.mode
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for name, half in ARENAS.items():
        if args.mode == 'finetune' and name == 'reference':
            continue
        for seed in args.seeds:
            configs, policies, hashes = {}, {}, {}
            for c in CONDITIONS:
                cfg, parent, sha = files[seed,c]
                path = (adapt(cfg, parent, half, out/f'{name}_{c}_s{seed}', args.device)
                        if args.mode == 'finetune' else parent)
                ck = torch.load(path, map_location='cpu', weights_only=True)
                expected = ADAPT_FRAMES if args.mode == 'finetune' else FRAMES
                if ck['frames'] != expected:
                    raise ValueError(f'Unexpected frame count in {path}')
                configs[c] = dict(cfg, arena_half=half)
                net = build_policy(cfg, args.device).eval().requires_grad_(False)
                net.load_state_dict(ck['policy'], strict=True)
                policies[c] = net
                hashes[c] = dict(parent_sha256=sha, evaluated_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            record = dict(mode=args.mode, seed=seed, arena=name, arena_width_m=2*half,
                          checkpoints=hashes, models=measure_pair(configs, policies, args.device,
                                                                args.episodes, args.slots, args.stride))
            if args.mode == 'finetune':
                baseline_pair = json.loads((frozen/f'{name}_s{seed}.json').read_text())
                gain = {c: record['models'][c]['navigation']['mean_goals']
                        - baseline_pair['models'][c]['navigation']['mean_goals'] for c in CONDITIONS}
                record['adaptation_gain_goals'] = gain
                record['pgc_minus_grid_adaptation_gain'] = gain['pgc33'] - gain['gc100']
            save(out/f'{name}_s{seed}.json', record)
            records.append(record)
    plot_pc10(records, out)
    save(out/'summary.json', dict(mode=args.mode, device=args.device, plan=plan, arenas=summarize(records)))


if __name__ == '__main__':
    main()
