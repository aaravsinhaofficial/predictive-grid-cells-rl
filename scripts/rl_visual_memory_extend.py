"""Fixed final-gap RL continuation with restored RMSProp and full resume snapshots."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
import rl_visual_memory as pilot
from code.common import CodeSource, build_env, build_policy, build_population
from rl_goal_memory import atomic_json, atomic_torch, replay, state_digest

NAMES = {'gc100': 'ring-20261105-vision-grid-01', 'pgc33': 'ring-20261105-vision-pgc-01'}
BLOCK_UPDATES = 782
BLOCKS = 5
BLOCK_FRAMES = 10_009_600
TOTAL_FRAMES = BLOCKS * BLOCK_FRAMES
PARENT_FRAMES = 25_024_000
ENV_SEED = 106201
CLOCK_SEED = 106203


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def restored_optimizer(net, cfg, parent):
    opt = torch.optim.RMSprop(net.parameters(), lr=cfg['lr'], alpha=cfg['rms_alpha'], eps=cfg['rms_eps'])
    opt.load_state_dict(parent['optimizer'])
    assert tree_equal(opt.state_dict(), parent['optimizer'])
    assert all(group['lr'] == cfg['lr'] for group in opt.param_groups)
    return opt


def fresh_runtime(net, cfg, device):
    n = cfg['n_envs']
    env = build_env(cfg, n, seed=ENV_SEED * 1000 + 1, device=device, stagger=True)
    return dict(env=env, obs=env.reset_all(), state=net.zero_state(n, device),
                goal=pilot.NativeGoal(n, cfg['n_cells'], device),
                clock=pilot.VisualClock(n, device, CLOCK_SEED),
                pa=torch.full((n,), -1, dtype=torch.long, device=device), pr=torch.zeros(n, device=device))


def capture_runtime(runtime, device):
    """Capture all changing simulator, controller, clock, and sampling state."""
    if str(device).split(':')[0] not in ('cpu', 'cuda'):
        raise ValueError('Complete policy RNG snapshots are implemented for CPU and CUDA')
    env, clock = runtime['env'], runtime['clock']
    cpu = lambda x: x.detach().clone().cpu()
    numpy_state = np.random.get_state()
    return dict(
        environment_tensors={key: cpu(value) for key, value in vars(env).items() if isinstance(value, torch.Tensor)},
        environment_episodes_done=env.episodes_done, environment_rng=env.gen.get_state().cpu(),
        observation={key: cpu(value) for key, value in runtime['obs'].items()},
        recurrent_state=tuple(cpu(x) for x in runtime['state']), goal=cpu(runtime['goal'].saved),
        previous_action=cpu(runtime['pa']), previous_reward=cpu(runtime['pr']),
        clock={key: cpu(getattr(clock, key)) for key in ('known', 'blind', 'remaining')},
        clock_rng=clock.generator.get_state().cpu(), cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device).cpu() if str(device).startswith('cuda') else None,
        numpy_rng=dict(kind=numpy_state[0], keys=numpy_state[1].tolist(), position=int(numpy_state[2]),
                       has_gaussian=int(numpy_state[3]), cached_gaussian=float(numpy_state[4])))


def restore_runtime(runtime, snapshot, device):
    env, clock = runtime['env'], runtime['clock']
    expected = {key for key, value in vars(env).items() if isinstance(value, torch.Tensor)}
    assert expected == snapshot['environment_tensors'].keys()
    for key, value in snapshot['environment_tensors'].items():
        setattr(env, key, value.clone().to(device))
    env.episodes_done = snapshot['environment_episodes_done']
    env.gen.set_state(snapshot['environment_rng'].cpu())
    runtime['obs'] = {key: value.clone().to(device) for key, value in snapshot['observation'].items()}
    runtime['state'] = tuple(value.clone().to(device) for value in snapshot['recurrent_state'])
    runtime['goal'].saved = snapshot['goal'].clone().to(device)
    runtime['pa'] = snapshot['previous_action'].clone().to(device)
    runtime['pr'] = snapshot['previous_reward'].clone().to(device)
    for key, value in snapshot['clock'].items():
        setattr(clock, key, value.clone().to(device))
    clock.generator.set_state(snapshot['clock_rng'].cpu())
    torch.set_rng_state(snapshot['cpu_rng'].cpu())
    if str(device).startswith('cuda'):
        if snapshot['cuda_rng'] is None:
            raise ValueError('Exact CUDA resume requires a saved CUDA RNG state')
        torch.cuda.set_rng_state(snapshot['cuda_rng'].cpu(), device)
    nr = snapshot['numpy_rng']
    np.random.set_state((nr['kind'], np.asarray(nr['keys'], dtype=np.uint32), nr['position'],
                         nr['has_gaussian'], nr['cached_gaussian']))


def update(net, cfg, opt, source, runtime):
    roll, state0, bootstrap, _, obs, state, pa, pr, episodes = pilot.collect_rollout(
        net, runtime['env'], source, cfg, runtime['obs'], runtime['state'], runtime['goal'], runtime['clock'],
        runtime['pa'], runtime['pr'], cfg['rollout'], 60)
    future, targets = bootstrap, []
    for t in reversed(range(cfg['rollout'])):
        future = roll['r'][t] + cfg['gamma'] * (1 - roll['d'][t]) * future
        targets.append(future)
    targets = torch.stack(list(reversed(targets)))
    logits, values = replay(net, roll, state0)
    dist = torch.distributions.Categorical(logits=logits)
    policy_loss = -(dist.log_prob(torch.stack(roll['a'])) * (targets - values).detach()).sum(0).mean()
    value_loss = .5 * (targets - values).square().sum(0).mean()
    entropy = dist.entropy().sum(0).mean()
    loss = policy_loss + cfg['value_coef'] * value_loss - cfg['entropy'] * entropy
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite A2C loss')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite A2C gradient')
    opt.step()
    runtime.update(obs=obs, state=tuple(x.detach() for x in state), pa=pa, pr=pr)
    return dict(loss=float(loss.detach()), entropy=float(entropy.detach() / cfg['rollout']), grad_norm=float(norm)), episodes


def parent_folder(condition):
    return ROOT / 'results/ring_cloud_artifacts' / NAMES[condition]


def protocol():
    previous = json.loads((ROOT / 'docs/research/rl_visual_memory_pilot_protocol.json').read_text())
    for path, expected in previous['original_asset_sha256'].items():
        assert pilot.sha(ROOT / path) == expected, path
    paths = list(pilot.SOURCE_FILES) + ['scripts/rl_visual_memory_extend.py',
                                    'tests/test_rl_visual_memory_extend.py']
    parents = {}
    for condition in NAMES:
        folder = parent_folder(condition)
        parents[condition] = {name: pilot.sha(folder / name)
                              for name in ('final.pt', 'final_evaluation.json', 'baseline.json', 'summary.json')}
    return dict(version='visual_memory_extension_v2', conditions=list(NAMES), model_seed=1,
                parent_learning_frames=PARENT_FRAMES, added_learning_frames=TOTAL_FRAMES,
                total_learning_frames_beyond_1B=PARENT_FRAMES + TOTAL_FRAMES,
                updates_per_block=BLOCK_UPDATES, blocks=BLOCKS, frames_per_block=BLOCK_FRAMES,
                warmup_updates=0, fixed_max_gap=60, fixed_min_gap=30, visible_updates=15,
                environment_seed=ENV_SEED * 1000 + 1, clock_seed=CLOCK_SEED,
                policy_sampling_rng='Restore each fixed parent final checkpoint CPU/CUDA RNG after construction',
                boundary='Fresh native episodes and zero recurrent/goal states; parent v1 has no live runtime snapshot',
                optimizer='Restore complete parent RMSProp state and learning rate; no new warmup',
                objective='Unchanged original reward-only navigation A2C; all weights trainable',
                evaluation=dict(seed=98201, episodes=64, n_envs=32, variants=list(pilot.VARIANTS),
                                max_gap=60, before_learning=True, after_every_fixed_block=True),
                model_selection='Fixed final parent only; retain all five block endpoints; no adaptive selection',
                interpretation='One preselected training-seed pair; extension declared after pilot endpoint',
                original_asset_sha256=previous['original_asset_sha256'], parent_sha256=parents,
                source_sha256={path: pilot.sha(ROOT / path) for path in paths},
                budget='Two T4 workers, two-hour DELETE each; 5.625 USD reservation in original cumulative 500 USD')


def train(net, cfg, parent, args, design, deadline):
    assert (cfg['n_envs'], cfg['rollout'], cfg['action_repeat']) == (32, 100, 4)
    opt = restored_optimizer(net, cfg, parent)
    runtime = fresh_runtime(net, cfg, args.device)
    source = CodeSource(cfg, build_population(cfg, args.device), args.device)
    torch.set_rng_state(parent['cpu_rng'].cpu())
    if str(args.device).startswith('cuda'):
        torch.cuda.set_rng_state(parent['cuda_rng'].cpu(), args.device)
    np.random.seed(ENV_SEED)
    net.train()
    frames, recent, stages = 0, [], []
    started = time.monotonic()

    def save(name, completed=False):
        atomic_torch(args.out / name, dict(policy=net.state_dict(), optimizer=opt.state_dict(), config=cfg,
            original_frames=1_000_000_000, extra_frames=PARENT_FRAMES + frames, extension_frames=frames,
            completed=completed, provenance=design, runtime_snapshot=capture_runtime(runtime, args.device),
            recent_scores=recent, stage_evaluations_completed=len(stages)))

    save('incoming.pt')
    with (args.out / 'metrics.jsonl').open('w') as log:
        for index in range(BLOCKS * BLOCK_UPDATES):
            pilot.require_time(deadline, 60)
            metrics, episodes = update(net, cfg, opt, source, runtime)
            frames += cfg['n_envs'] * cfg['rollout'] * cfg['action_repeat']
            recent.extend(episodes); recent = recent[-256:]
            if (index + 1) % 20 == 0:
                item = dict(updates=index + 1, extension_frames=frames, extra_frames=PARENT_FRAMES + frames,
                            max_blind_gap=60, seconds=time.monotonic() - started,
                            mean_recent_score=float(np.mean(recent)) if recent else None, **metrics)
                log.write(json.dumps(item) + '\n'); log.flush(); print(json.dumps(item), flush=True)
            if (index + 1) % 100 == 0:
                save('latest.pt')
            if (index + 1) % BLOCK_UPDATES == 0:
                block = (index + 1) // BLOCK_UPDATES
                save(f'block{block}.pt', completed=block == BLOCKS)
                before = state_digest(net)
                result = pilot.evaluate(net, cfg, device=args.device, seed=98201, episodes=64,
                                        n_envs=32, max_gap=60, deadline=deadline)
                assert state_digest(net) == before
                atomic_json(args.out / f'block{block}_evaluation.json', result)
                stages.append(dict(block=block, extension_frames=frames, evaluation=result))
                atomic_json(args.out / 'progress.json', dict(stages=stages))
                net.train()
    save('final.pt', completed=True)
    atomic_json(args.out / 'final_evaluation.json', stages[-1]['evaluation'])
    return dict(completed=True, extension_frames=frames, extra_frames=PARENT_FRAMES + frames,
                seconds=time.monotonic() - started, stages=stages, optimizer_restored_exactly=True,
                warmup_updates=0, fresh_native_episodes_at_extension_boundary=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--condition', choices=tuple(NAMES)); ap.add_argument('--out', type=Path)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max-seconds', type=int, default=6600)
    ap.add_argument('--protocol', type=Path, default=ROOT / 'docs/research/rl_visual_memory_extension_protocol.json')
    ap.add_argument('--freeze-only', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(2)
    design = protocol()
    if args.freeze_only:
        if args.protocol.exists(): raise FileExistsError('Do not overwrite frozen extension protocol')
        design['frozen_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(args.protocol, design); print(pilot.sha(args.protocol)); return
    if not args.condition or args.out is None: ap.error('--condition and --out are required')
    frozen = json.loads(args.protocol.read_text()); frozen.pop('frozen_at')
    assert frozen == design, 'Frozen sources, assets, or extension protocol changed'
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'protocol.json').exists(): raise FileExistsError('Extension output already exists')
    design['runtime'] = dict(torch_version=str(torch.__version__), condition=args.condition, device=args.device,
                             protocol_sha256=pilot.sha(args.protocol))
    atomic_json(args.out / 'protocol.json', design)
    parent = torch.load(parent_folder(args.condition) / 'final.pt', map_location='cpu', weights_only=True)
    assert parent['completed'] and parent['extra_frames'] == PARENT_FRAMES
    assert {float(state['step']) for state in parent['optimizer']['state'].values()} == {2155.}
    cfg = parent['config']
    assert cfg['cond'] == args.condition
    assert cfg == json.loads((ROOT / f'models/{args.condition}/{args.condition}_lstm_s1_1000M.json').read_text())
    net = build_policy(cfg, args.device); net.load_state_dict(parent['policy'], strict=True)
    assert tree_equal(net.state_dict(), parent['policy'])
    deadline = time.monotonic() + args.max_seconds
    try:
        initial_policy_hash = state_digest(net)
        baseline = pilot.evaluate(net, cfg, device=args.device, seed=98201, episodes=64,
                                  n_envs=32, max_gap=60, deadline=deadline)
        assert state_digest(net) == initial_policy_hash
        expected = json.loads((parent_folder(args.condition) / 'final_evaluation.json').read_text())
        assert all(baseline['variants'][key]['rows'] == expected['variants'][key]['rows'] for key in pilot.VARIANTS)
        atomic_json(args.out / 'baseline.json', baseline)
        atomic_json(args.out / 'original_1B_baseline.json',
                    json.loads((parent_folder(args.condition) / 'baseline.json').read_text()))
        result = train(net, cfg, parent, args, design, deadline)
        atomic_json(args.out / 'training.json', result)
        atomic_json(args.out / 'summary.json', dict(completed=True, condition=args.condition, training=result,
                    baseline_matches_parent_behavior_exactly=True, final_sha256=pilot.sha(args.out / 'final.pt')))
    except Exception as error:
        atomic_json(args.out / 'incomplete.json', dict(error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
